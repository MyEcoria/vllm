// INT4 paged KV-cache attention for gfx1201 (RX 9070), head_dim 256, GQA.
//
// Cache content per (block, kv_head, slot): [K 128B | Ks 4B | V 128B | Vs 4B] (264 B).
//   K: standard order, byte j = dims (2j low nibble, 2j+1 high nibble), RHT-rotated.
//   V: TRANSPOSED per 16-slot tile. The tile's 16 V regions (2048 B) hold u32 words;
//      word w = (dl*2 + h)*16 + j  holds the 8 nibbles of dim 16j+dl for tokens h*8+[0,8),
//      token k at nibble (k>>1) + 4*(k&1). Word w lives in slot (w/32), bytes (w%32)*4.
//   Ks/Vs: fp32 scale whose low 4 mantissa bits hold the zero point; x = s*(code - zp).
// Everything is in the rotated basis x' = H D x / 16 (orthonormal RHT, H = Sylvester 256).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <cstdlib>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef unsigned v4u __attribute__((ext_vector_type(4)));

#define HD 256
#define KV_OFF_K 0
#define KV_OFF_KS 128
#define KV_OFF_V 132
#ifdef BONSAI_V3
// V: 3-bit codes, 96 B per slot. Per 16-slot tile, lane (dl, h) owns bytes [h*48, h*48+48) of
// slot dl's V area: plane A (8 u32, 2 low bits) then plane B (4 u32, high bit).
//   A word j>>1, bit 8*(j&1) + 2*i + 16*odd ; B word j>>2, bit 4*(j&3) + i + 16*odd
//   for dim 16j+dl and tile token h*8 + 2*i + odd.
#define KV_OFF_VS 228
#define V_LANE_BYTES 48
#else
#define KV_OFF_VS 260
#define V_LANE_BYTES 64
#endif

__device__ __forceinline__ float rht_sign(int i) {
  uint32_t x = (uint32_t)i * 0x9E3779B1u;
  x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12;
  return (x & 1u) ? -1.f : 1.f;
}

// In-wave FWHT of 256 values: lane l holds v[k] = x[l*8 + k]. Orthonormal (x 1/16).
__device__ __forceinline__ void fwht256(float v[8], int lane) {
#pragma unroll
  for (int b = 1; b < 8; b <<= 1)
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (!(k & b)) { float a = v[k], c = v[k | b]; v[k] = a + c; v[k | b] = a - c; }
#pragma unroll
  for (int b = 1; b < 32; b <<= 1) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float p = __shfl_xor(v[k], b, 32);
      v[k] = (lane & b) ? (p - v[k]) : (v[k] + p);
    }
  }
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] *= 0.0625f;
}

__device__ __forceinline__ float wave_max(float x) {
#pragma unroll
  for (int b = 16; b; b >>= 1) x = fmaxf(x, __shfl_xor(x, b, 32));
  return x;
}
__device__ __forceinline__ float wave_min(float x) {
#pragma unroll
  for (int b = 16; b; b >>= 1) x = fminf(x, __shfl_xor(x, b, 32));
  return x;
}
__device__ __forceinline__ int wave_sum_i(int x) {
#pragma unroll
  for (int b = 16; b; b >>= 1) x += __shfl_xor(x, b, 32);
  return x;
}

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_f<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __hip_bfloat16 from_f<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

// quantize 8 values per lane (256 per wave) to asym int4; returns codes and packed scale
__device__ __forceinline__ float quant4(const float v[8], int codes[8]) {
  float mn = v[0], mx = v[0];
#pragma unroll
  for (int k = 1; k < 8; ++k) { mn = fminf(mn, v[k]); mx = fmaxf(mx, v[k]); }
  mn = wave_min(mn); mx = wave_max(mx);
  float s = fmaxf((mx - mn) / 15.f, 1e-8f);
  float zp = fminf(fmaxf(rintf(-mn / s), 0.f), 15.f);
  float inv = 1.f / s;
#pragma unroll
  for (int k = 0; k < 8; ++k) codes[k] = (int)fminf(fmaxf(rintf(v[k] * inv + zp), 0.f), 15.f);
  uint32_t bits = (__float_as_uint(s) & ~0xFu) | (uint32_t)zp;
  return __uint_as_float(bits);
}

// 3-bit quantization with MSE clipping (mean +- 2.15 sigma within the observed range).
__device__ __forceinline__ float quant3(const float v[8], int codes[8]) {
  float mn = v[0], mx = v[0], sum = 0.f, sq = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) { mn = fminf(mn, v[k]); mx = fmaxf(mx, v[k]); sum += v[k]; sq += v[k] * v[k]; }
  mn = wave_min(mn); mx = wave_max(mx);
#pragma unroll
  for (int b = 16; b; b >>= 1) { sum += __shfl_xor(sum, b, 32); sq += __shfl_xor(sq, b, 32); }
  const float mu = sum / 256.f, sd = sqrtf(fmaxf(sq / 256.f - mu * mu, 0.f));
  const float half = 2.15f * sd * 8.f / 7.f;
  const float lo = fmaxf(mn, mu - half), hi = fminf(mx, mu + half);
  const float s = fmaxf((hi - lo) / 7.f, 1e-8f);
  const float zp = fminf(fmaxf(rintf(-lo / s), 0.f), 7.f);
#pragma unroll
  for (int k = 0; k < 8; ++k) codes[k] = (int)fminf(fmaxf(rintf(v[k] / s + zp), 0.f), 7.f);
  uint32_t bits = (__float_as_uint(s) & ~0xFu) | (uint32_t)zp;
  return __uint_as_float(bits);
}

// 3-bit (8 levels) quantization stored as even 4-bit codes, with MSE-oriented clipping
// (mean +- 2.15 sigma, clamped to the observed range). Same dequant as 4-bit: s*(code - zp).
__device__ __forceinline__ float quant3sim(const float v[8], int codes[8]) {
  float mn = v[0], mx = v[0], sum = 0.f, sq = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) { mn = fminf(mn, v[k]); mx = fmaxf(mx, v[k]); sum += v[k]; sq += v[k] * v[k]; }
  mn = wave_min(mn); mx = wave_max(mx);
#pragma unroll
  for (int b = 16; b; b >>= 1) { sum += __shfl_xor(sum, b, 32); sq += __shfl_xor(sq, b, 32); }
  const float mu = sum / 256.f, sd = sqrtf(fmaxf(sq / 256.f - mu * mu, 0.f));
  const float half = 2.15f * sd * 8.f / 7.f;
  const float lo = fmaxf(mn, mu - half), hi = fminf(mx, mu + half);
  const float s3 = fmaxf((hi - lo) / 7.f, 1e-8f);
  const float zp3 = fminf(fmaxf(rintf(-lo / s3), 0.f), 7.f);
#pragma unroll
  for (int k = 0; k < 8; ++k) codes[k] = 2 * (int)fminf(fmaxf(rintf(v[k] / s3 + zp3), 0.f), 7.f);
  const float s = s3 * 0.5f;  // code4 = 2*code3 -> s4 = s3/2, zp4 = 2*zp3
  uint32_t bits = (__float_as_uint(s) & ~0xFu) | (uint32_t)(2.f * zp3);
  return __uint_as_float(bits);
}

// ---------------------------------------------------------------------------------
// Cache write: one wave per (token, kv_head). key/value: [T, NKV, 256] (any token stride)
template <typename T>
__global__ void __launch_bounds__(32) kv_write_kernel(
    const T* __restrict__ key, const T* __restrict__ value, int64_t k_tok_stride, int64_t k_head_stride,
    int64_t v_tok_stride, int64_t v_head_stride, const int64_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size) {
  const int tok = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
  const int64_t slot = slot_mapping[tok];
  if (slot < 0) return;
  const int64_t blk = slot / block_size;
  const int p = (int)(slot % block_size);
  uint8_t* base = cache + blk * s_blk + head * s_head;
  float v[8];
  int codes[8];
  // ---- K
  {
    const T* src = key + tok * k_tok_stride + head * k_head_stride + lane * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = to_f<T>(src[k]) * rht_sign(lane * 8 + k);
    fwht256(v, lane);
#ifdef BONSAI_K3SIM
    float sp = quant3sim(v, codes);
#else
    float sp = quant4(v, codes);
#endif
    uint32_t w = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) w |= (uint32_t)(codes[2 * k] | (codes[2 * k + 1] << 4)) << (8 * k);
    *reinterpret_cast<uint32_t*>(base + p * s_slot + KV_OFF_K + lane * 4) = w;
    if (lane == 0) *reinterpret_cast<float*>(base + p * s_slot + KV_OFF_KS) = sp;
  }
  // ---- V (transposed tile layout)
  {
    const T* src = value + tok * v_tok_stride + head * v_head_stride + lane * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = to_f<T>(src[k]) * rht_sign(lane * 8 + k);
    fwht256(v, lane);
    const int p0 = p & ~15, pt = p & 15, h = pt >> 3, kk = pt & 7;
#ifdef BONSAI_V3
    float sp = quant3(v, codes);
    const int ii = kk >> 1, odd = kk & 1;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane * 8 + k, j = d >> 4, dl = d & 15;
      uint8_t* lb = base + (int64_t)(p0 + dl) * s_slot + KV_OFF_V + h * V_LANE_BYTES;
      uint32_t* wa = reinterpret_cast<uint32_t*>(lb + (j >> 1) * 4);
      const int ba = 8 * (j & 1) + 2 * ii + 16 * odd;
      atomicAnd(wa, ~(3u << ba));
      atomicOr(wa, (uint32_t)(codes[k] & 3) << ba);
      uint32_t* wb = reinterpret_cast<uint32_t*>(lb + 32 + (j >> 2) * 4);
      const int bb = 4 * (j & 3) + ii + 16 * odd;
      atomicAnd(wb, ~(1u << bb));
      atomicOr(wb, (uint32_t)(codes[k] >> 2) << bb);
    }
#else
#ifdef BONSAI_V3SIM
    float sp = quant3sim(v, codes);
#else
    float sp = quant4(v, codes);
#endif
    const int nib = (kk >> 1) + 4 * (kk & 1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane * 8 + k, j = d >> 4, dl = d & 15;
      const int w = (dl * 2 + h) * 16 + j;
      uint32_t* addr = reinterpret_cast<uint32_t*>(base + (p0 + (w >> 5)) * s_slot + KV_OFF_V + (w & 31) * 4);
      const uint32_t mask = 0xFu << (4 * nib);
      atomicAnd(addr, ~mask);
      atomicOr(addr, (uint32_t)codes[k] << (4 * nib));
    }
#endif
    if (lane == 0) *reinterpret_cast<float*>(base + p * s_slot + KV_OFF_VS) = sp;
  }
}

// ---------------------------------------------------------------------------------
// Q prep: one wave per (token, q_head). Writes q8 in the K-operand permuted order,
// qscale (includes softmax scale * log2e) and qsum.
// permuted position for (h, s, i): pos = h*128 + s*8 + i ; dims: i<4 -> 2*(h*64+s*4+i), else 2*(h*64+s*4+i-4)+1
template <typename T>
__global__ void __launch_bounds__(32) q_prep_kernel(const T* __restrict__ q, int64_t q_tok_stride, int64_t q_head_stride,
                                                    int8_t* __restrict__ q8, float* __restrict__ qs, int* __restrict__ qsum,
                                                    float softmax_scale_log2) {
  const int tok = blockIdx.x, head = blockIdx.y, nh = gridDim.y, lane = threadIdx.x;
  const T* src = q + tok * q_tok_stride + head * q_head_stride + lane * 8;
  float v[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = to_f<T>(src[k]) * rht_sign(lane * 8 + k);
  fwht256(v, lane);
  float am = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) am = fmaxf(am, fabsf(v[k]));
  am = wave_max(am);
  const float s = fmaxf(am / 127.f, 1e-12f), inv = 1.f / s;
  int sum = 0;
  int8_t* dst = q8 + ((int64_t)tok * nh + head) * HD;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = (int)rintf(v[k] * inv);
    sum += c;
    const int d = lane * 8 + k;           // rotated dim
    const int byte = d >> 1, odd = d & 1;  // K byte holding d
    const int h = byte >> 6, s_ = (byte & 63) >> 2, i = byte & 3;
    dst[h * 128 + s_ * 8 + i + 4 * odd] = (int8_t)c;
  }
  sum = wave_sum_i(sum);
  if (lane == 0) {
    qs[(int64_t)tok * nh + head] = s * softmax_scale_log2;
    qsum[(int64_t)tok * nh + head] = sum;
  }
}

// ---------------------------------------------------------------------------------
// Attention core over a range of 16-token key tiles for ONE q tile (16 rows) of one
// (seq, kv_head). Rows r -> (token qi = r / G, head g = r % G) ; rows >= nrows are padding.
// Returns per-lane online-softmax state; caller handles output.
struct QTile {
  const v2i* q;  // LDS: lane's 16 B-operand steps (8 bytes each)
  float qs;      // per row (lane l%16)
  float qsum;
  int qpos;      // absolute position of the row's token (for causal); -1 = padding row
};

__device__ __forceinline__ v8h unpack_v(uint32_t w) {
  v8h r;
  uint32_t x0 = (w & 0x000F000Fu) | 0x64006400u;
  uint32_t x1 = ((w >> 4) & 0x000F000Fu) | 0x64006400u;
  uint32_t x2 = ((w >> 8) & 0x000F000Fu) | 0x64006400u;
  uint32_t x3 = ((w >> 12) & 0x000F000Fu) | 0x64006400u;
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  const h2 off = {(_Float16)1024.f, (_Float16)1024.f};
  h2 a0 = __builtin_bit_cast(h2, x0) - off, a1 = __builtin_bit_cast(h2, x1) - off;
  h2 a2 = __builtin_bit_cast(h2, x2) - off, a3 = __builtin_bit_cast(h2, x3) - off;
  r[0] = a0[0]; r[1] = a0[1]; r[2] = a1[0]; r[3] = a1[1];
  r[4] = a2[0]; r[5] = a2[1]; r[6] = a3[0]; r[7] = a3[1];
  return r;
}

struct TileRegs {
  v4u kb[4];
  v4u vb[V_LANE_BYTES / 16];
  float ksp, vsp;
};

#ifdef BONSAI_V3
__device__ __forceinline__ v8h unpack_v3(uint32_t A, uint32_t B) {
  typedef _Float16 h2 __attribute__((ext_vector_type(2)));
  const h2 off = {(_Float16)1024.f, (_Float16)1024.f};
  v8h r;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t x = ((A >> (2 * i)) & 0x00030003u) | (((B >> i) & 0x00010001u) << 2) | 0x64006400u;
    const h2 hv = __builtin_bit_cast(h2, x) - off;
    r[2 * i] = hv[0];
    r[2 * i + 1] = hv[1];
  }
  return r;
}
#endif

__device__ __forceinline__ void load_tile(TileRegs& R, const uint8_t* __restrict__ cache_head,
                                          const int* __restrict__ btab, int block_size, int64_t s_blk,
                                          int64_t s_slot, int t, int lane) {
  const int tl = lane & 15, h = lane >> 4;
  const int tok0 = t * 16;
  const int blk = btab[tok0 / block_size];
  const int p0 = tok0 % block_size;
  const uint8_t* krow = cache_head + (int64_t)blk * s_blk + (int64_t)(p0 + tl) * s_slot;
#pragma unroll
  for (int i = 0; i < 4; ++i) R.kb[i] = *reinterpret_cast<const v4u*>(krow + KV_OFF_K + h * 64 + i * 16);
#pragma unroll
  for (int i = 0; i < V_LANE_BYTES / 16; ++i)
    R.vb[i] = *reinterpret_cast<const v4u*>(krow + KV_OFF_V + h * V_LANE_BYTES + i * 16);
  R.ksp = *reinterpret_cast<const float*>(krow + KV_OFF_KS);
  R.vsp = *reinterpret_cast<const float*>(krow + KV_OFF_VS);
}

// Online-softmax update of one 16-key tile from its integer scores sacc; returns p and the rescale factor alpha.
__device__ __forceinline__ void softmax_tile(float qsc, float qsum, int qpos, v8i sacc, float ksp, float vsp, int tok0,
                                             int seq_len, int lane, float& m, float& l, float& corr, v8h& p,
                                             float& alpha) {
  const int h = lane >> 4;
  float sc[8];
  float mx = -INFINITY;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int tt = h * 8 + i;
    const uint32_t kbits = __float_as_uint(__shfl(ksp, tt, 32));
    const float ks = __uint_as_float(kbits & ~0xFu);
    const float kz = (float)(kbits & 0xFu);
    float x = qsc * ks * ((float)sacc[i] - kz * qsum);
    const int kpos = tok0 + tt;
    if (kpos >= seq_len || kpos > qpos) x = -INFINITY;
    sc[i] = x;
    mx = fmaxf(mx, x);
  }
  mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
  const float m_new = fmaxf(m, mx);
  alpha = (m_new == -INFINITY) ? 1.f : exp2f(m - m_new);
  float psum = 0.f, pz = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float e = (m_new == -INFINITY) ? 0.f : exp2f(sc[i] - m_new);
    psum += e;
    const uint32_t vbits = __float_as_uint(__shfl(vsp, h * 8 + i, 32));
    const float ev = e * __uint_as_float(vbits & ~0xFu);
    pz += ev * (float)(vbits & 0xFu);
    p[i] = (_Float16)ev;
  }
  l = l * alpha + psum;
  corr = corr * alpha + pz;
  m = m_new;
}

// QK^T + online-softmax update for one 16-key tile; returns the probabilities p and the rescale factor alpha.
__device__ __forceinline__ void score_step(const QTile& Q, const TileRegs& R, int tok0, int seq_len, int lane,
                                           float& m, float& l, float& corr, v8h& p, float& alpha) {
  v8i sacc = {0, 0, 0, 0, 0, 0, 0, 0}, sacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int s = 0; s < 16; s += 2) {
    const uint32_t w = R.kb[s >> 2][s & 3], w2 = R.kb[(s + 1) >> 2][(s + 1) & 3];
    v2i a = {(int)(w & 0x0F0F0F0Fu), (int)((w >> 4) & 0x0F0F0F0Fu)};
    v2i a2 = {(int)(w2 & 0x0F0F0F0Fu), (int)((w2 >> 4) & 0x0F0F0F0Fu)};
    sacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, Q.q[s], sacc, false);
    sacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a2, true, Q.q[s + 1], sacc2, false);
  }
  sacc += sacc2;
  softmax_tile(Q.qs, Q.qsum, Q.qpos, sacc, R.ksp, R.vsp, tok0, seq_len, lane, m, l, corr, p, alpha);
}

__device__ __forceinline__ void compute_tile(const QTile& Q, const TileRegs& R, int tok0, int seq_len, int lane,
                                             v8f acc_o[16], float& m, float& l, float& corr) {
  v8h p;
  float alpha;
  score_step(Q, R, tok0, seq_len, lane, m, l, corr, p, alpha);
  // rescale only when some row's max moved (wave-uniform branch)
  if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i) acc_o[j][i] *= alpha;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j) {
#ifdef BONSAI_V3
    const uint32_t A = R.vb[(j >> 1) >> 2][(j >> 1) & 3] >> (8 * (j & 1));
    const uint32_t B = R.vb[2][j >> 2] >> (4 * (j & 3));
    acc_o[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(unpack_v3(A, B), p, acc_o[j]);
#else
    const uint32_t w = R.vb[j >> 2][j & 3];
    acc_o[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(unpack_v(w), p, acc_o[j]);
#endif
  }
}

// Process key tiles [t_begin, t_end) ; seq_len keys total. Double-buffered loads.
__device__ __forceinline__ void attn_tiles(const QTile& Q, const uint8_t* __restrict__ cache_head,
                                           const int* __restrict__ btab, int block_size, int64_t s_blk,
                                           int64_t s_slot, int t_begin, int t_end, int seq_len, int lane,
                                           v8f acc_o[16], float& m, float& l, float& corr) {
  TileRegs cur, nxt;
  load_tile(cur, cache_head, btab, block_size, s_blk, s_slot, t_begin, lane);
  for (int t = t_begin; t < t_end; ++t) {
    if (t + 1 < t_end) load_tile(nxt, cache_head, btab, block_size, s_blk, s_slot, t + 1, lane);
    compute_tile(Q, cur, t * 16, seq_len, lane, acc_o, m, l, corr);
    cur = nxt;
  }
}

// ---------------------------------------------------------------------------------
// Decode / short-query split-K kernel. grid = (num_splits, nkv * qtiles, num_seqs), 32 threads.
// Partial outputs: part_o [seq, nkv, qtile, split, 16 rows, 256] f32, part_ml [.., 16, 2]
// SPARSE (BONSAI_SPARSE_PREFILL, chunk path): the split iterates its share of the kept 128-key blocks of the
// q-tile's 64-row item (blist[((si * items + qt / 4) * nkv + kvh) * (nbmax + 1)], last entry = count).
template <bool SPARSE>
__global__ void __launch_bounds__(32) attn_decode_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ qs, const int* __restrict__ qsum, int nqh,
    const uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size,
    const int* __restrict__ block_table, int bt_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const int* __restrict__ seq_ids, int G, int qtiles,
    float* __restrict__ part_o, float* __restrict__ part_ml, int num_splits, const int* __restrict__ blist,
    int nbmax) {
  // grid = (nkv * qtiles, num_splits, nseq): the q tiles of one (head, split) are launched together so
  // their K/V reads are served by the L2 (a long query has many q tiles over the same keys)
  const int split = blockIdx.y, nkv = gridDim.x / qtiles;
  const int kvh = blockIdx.x / qtiles, qt = blockIdx.x % qtiles;
  const int si = blockIdx.z, seq = seq_ids[si];
  const int lane = threadIdx.x, r = lane & 15, h = lane >> 4;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0;
  const int seq_len = seq_lens[seq];
  const int row = qt * 16 + r;
  if (qt * 16 >= qlen * G) {  // tile without any valid row (shorter queries in a mixed launch)
    const int64_t pb = ((((int64_t)si * (gridDim.x / qtiles) + kvh) * qtiles + qt) * num_splits + split) * 16;
    if (lane < 16) { part_ml[(pb + lane) * 2] = -INFINITY; part_ml[(pb + lane) * 2 + 1] = 0.f; }
    return;
  }
  QTile Q;
  __shared__ v2i q_lds[32 * 16];
  const bool valid = row < qlen * G;
  const int qi = row / G, g = row % G;
  const int qtok = q0 + (valid ? qi : 0), qh = kvh * G + g;
  const int8_t* qrow = q8 + ((int64_t)qtok * nqh + qh) * HD + h * 128;
#pragma unroll
  for (int s = 0; s < 16; ++s) q_lds[lane * 16 + s] = valid ? *reinterpret_cast<const v2i*>(qrow + s * 8) : v2i{0, 0};
  Q.q = q_lds + lane * 16;
  Q.qs = valid ? qs[(int64_t)qtok * nqh + qh] : 0.f;
  Q.qsum = valid ? (float)qsum[(int64_t)qtok * nqh + qh] : 0.f;
  Q.qpos = valid ? (seq_len - qlen + qi) : -1;
  const int ntiles = (seq_len + 15) >> 4;
  const int per = (ntiles + num_splits - 1) / num_splits;
  const int tb = split * per, te = min(ntiles, tb + per);
  v8f acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY, l = 0.f, corr = 0.f;
  const uint8_t* cache_head = cache + (int64_t)kvh * s_head;
  const int* btab = block_table + (int64_t)seq * bt_stride;
  if constexpr (SPARSE) {
    const int items = (qtiles + 3) / 4;
    const int* bl = blist + (((int64_t)si * items + qt / 4) * nkv + kvh) * (nbmax + 1);
    const int n = bl[nbmax], sper = (n + num_splits - 1) / num_splits;
    const int i1 = min(n, split * sper + sper);
    for (int i = split * sper; i < i1;) {
      int e = i + 1;
      while (e < i1 && bl[e] == bl[e - 1] + 1) ++e;
      const int t0 = bl[i] * 8, t1 = min(bl[e - 1] * 8 + 8, ntiles);
      if (t0 < t1) attn_tiles(Q, cache_head, btab, block_size, s_blk, s_slot, t0, t1, seq_len, lane, acc, m, l, corr);
      i = e;
    }
  } else if (tb < te) {
    attn_tiles(Q, cache_head, btab, block_size, s_blk, s_slot, tb, te, seq_len, lane, acc, m, l, corr);
  }
  // combine the two half-lanes' l and corr (each lane holds 8 of the 16 tokens per tile)
  l += __shfl_xor(l, 16, 32);
  corr += __shfl_xor(corr, 16, 32);
  const int64_t pbase = ((((int64_t)si * nkv + kvh) * qtiles + qt) * num_splits + split) * 16;
  // acc[j][i] = O^T[dim 16j + h*8 + i][row r]
  float* po = part_o + (pbase + r) * HD;
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) po[16 * j + h * 8 + i] = acc[j][i] - corr;
  if (h == 0) {
    part_ml[(pbase + r) * 2 + 0] = m;
    part_ml[(pbase + r) * 2 + 1] = l;
  }
}

// Reduce splits + inverse RHT + store. grid = (num_seqs_dec, nkv*qtiles*16 rows), 32 threads
template <typename T>
__global__ void __launch_bounds__(32) attn_reduce_kernel(const float* __restrict__ part_o,
                                                         const float* __restrict__ part_ml, int num_splits, int nkv,
                                                         int qtiles, int G, const int* __restrict__ query_start_loc,
                                                         const int* __restrict__ seq_ids, T* __restrict__ out,
                                                         int64_t o_tok_stride, int64_t o_head_stride) {
  const int si = blockIdx.x, seq = seq_ids[si];
  const int idx = blockIdx.y;  // (kvh, qt, r)
  const int r = idx & 15, qt = (idx >> 4) % qtiles, kvh = (idx >> 4) / qtiles;
  const int lane = threadIdx.x;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0;
  const int row = qt * 16 + r;
  if (row >= qlen * G) return;
  const int qi = row / G, g = row % G;
  const int64_t pbase = (((int64_t)si * nkv + kvh) * qtiles + qt) * num_splits;
  float M = -INFINITY;
  for (int s = 0; s < num_splits; ++s) M = fmaxf(M, part_ml[((pbase + s) * 16 + r) * 2]);
  float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float L = 0.f;
  for (int s = 0; s < num_splits; ++s) {
    const float ms = part_ml[((pbase + s) * 16 + r) * 2], ls = part_ml[((pbase + s) * 16 + r) * 2 + 1];
    if (ms == -INFINITY) continue;
    const float w = exp2f(ms - M);
    L += ls * w;
    const float* po = part_o + ((pbase + s) * 16 + r) * HD + lane * 8;
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] += w * po[k];
  }
  const float inv = L > 0.f ? 1.f / L : 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] *= inv;
  // inverse RHT: x = D H x' / 16  (H symmetric)
  fwht256(v, lane);
  T* dst = out + (int64_t)(q0 + qi) * o_tok_stride + (int64_t)(kvh * G + g) * o_head_stride + lane * 8;
#pragma unroll
  for (int k = 0; k < 8; ++k) dst[k] = from_f<T>(v[k] * rht_sign(lane * 8 + k));
}

// ---------------------------------------------------------------------------------
// Prefill (long query) kernel. One wave = one 16-row q tile; WG = WAVES waves.
// work[i] = (seq, first_row) for WG i; rows of one kv head; grid = (num_work, nkv).
// Output is normalized, inverse-rotated and written directly.
// blist (BONSAI_SPARSE_PREFILL): iterates only over the 128-key blocks listed (ascending, -1 padded) in
// blist[(work_idx * nkv + kvh) * (nbmax + 1) ...]; nullptr = all tiles. Consecutive blocks are merged into one run, so
// a full list is bit-identical to the dense loop.
template <typename T, int WAVES>
__global__ void __launch_bounds__(32 * WAVES) attn_prefill_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ qs, const int* __restrict__ qsum, int nqh,
    const uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size,
    const int* __restrict__ block_table, int bt_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const int* __restrict__ work, int G, T* __restrict__ out,
    int64_t o_tok_stride, int64_t o_head_stride, const int* __restrict__ blist, int nbmax) {
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane & 15, h = lane >> 4;
  const int kvh = blockIdx.y;
  const int seq = work[blockIdx.x * 2], row0 = work[blockIdx.x * 2 + 1] + wave * 16;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0;
  const int seq_len = seq_lens[seq];
  const int row = row0 + r;
  __shared__ v2i q_lds[WAVES * 32 * 16];
  QTile Q;
  const bool valid = row < qlen * G;
  const int qi = row / G, g = row % G;
  const int qtok = q0 + (valid ? qi : 0), qh = kvh * G + g;
  const int8_t* qrow = q8 + ((int64_t)qtok * nqh + qh) * HD + h * 128;
  v2i* my = q_lds + (wave * 32 + lane) * 16;
#pragma unroll
  for (int s = 0; s < 16; ++s) my[s] = valid ? *reinterpret_cast<const v2i*>(qrow + s * 8) : v2i{0, 0};
  Q.q = my;
  Q.qs = valid ? qs[(int64_t)qtok * nqh + qh] : 0.f;
  Q.qsum = valid ? (float)qsum[(int64_t)qtok * nqh + qh] : 0.f;
  Q.qpos = valid ? (seq_len - qlen + qi) : -1;
  // key range for this wave: up to the last valid row's position
  const int last_row = min(row0 + 15, qlen * G - 1);
  if (last_row < row0) return;
  const int max_pos = seq_len - qlen + last_row / G;
  const int ntiles = (max_pos + 16) >> 4;
  v8f acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY, l = 0.f, corr = 0.f;
  const uint8_t* cache_head = cache + (int64_t)kvh * s_head;
  const int* btab = block_table + (int64_t)seq * bt_stride;
  // runs of consecutive kept blocks (blist == nullptr: one run over all tiles = dense)
  const int* bl = blist ? blist + ((int64_t)blockIdx.x * gridDim.y + kvh) * (nbmax + 1) : nullptr;
  for (int i = 0;;) {
    int t0 = 0, t1 = ntiles, e = nbmax;
    if (bl) {
      if (i >= nbmax || bl[i] < 0) break;
      e = i + 1;
      while (e < nbmax && bl[e] == bl[e - 1] + 1) ++e;
      t0 = bl[i] * 8;
      t1 = min(bl[e - 1] * 8 + 8, ntiles);
      if (t0 >= ntiles) break;
    }
    attn_tiles(Q, cache_head, btab, block_size, s_blk, s_slot, t0, t1, seq_len, lane, acc, m, l, corr);
    if (!bl) break;
    i = e;
  }
  l += __shfl_xor(l, 16, 32);
  corr += __shfl_xor(corr, 16, 32);
  const float inv = l > 0.f ? 1.f / l : 0.f;
  // x[j][i] = O'[16j + 8h + i]; inverse FWHT over d = 16j + 8h + i
  float x[16][8];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) x[j][i] = (acc[j][i] - corr) * inv;
#pragma unroll
  for (int b = 1; b < 8; b <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (!(i & b)) { float a = x[j][i], c = x[j][i | b]; x[j][i] = a + c; x[j][i | b] = a - c; }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float p = __shfl_xor(x[j][i], 16, 32);
      x[j][i] = h ? (p - x[j][i]) : (x[j][i] + p);
    }
#pragma unroll
  for (int b = 1; b < 16; b <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (!(j & b))
#pragma unroll
        for (int i = 0; i < 8; ++i) { float a = x[j][i], c = x[j | b][i]; x[j][i] = a + c; x[j | b][i] = a - c; }
  if (!valid) return;
  T* dst = out + (int64_t)qtok * o_tok_stride + (int64_t)qh * o_head_stride;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    T vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = 16 * j + 8 * h + i;
      vals[i] = from_f<T>(x[j][i] * 0.0625f * rht_sign(d));
    }
    *reinterpret_cast<uint4*>(dst + 16 * j + 8 * h) = *reinterpret_cast<uint4*>(vals);
  }
}

// ---------------------------------------------------------------------------------
// Block-sparse prefill (opt-in, lossy; BONSAI_SPARSE_PREFILL=tau). Blocks are SP_BLK = 128 keys.
// 1) sparse_kbox_kernel: per (128-key block, kv head, seq) the per-dim min/max of the dequantized K, stored as
//    center / radius, layout [seq][kvh][dim][nbmax] (blocks contiguous). One wave per block, K read once.
#define SP_BLK 128
#define SP_NBMAX 4096

__global__ void __launch_bounds__(32) sparse_kbox_kernel(
    const uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size,
    const int* __restrict__ block_table, int bt_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, int nkv, int nbmax, int min_ctx, int min_qlen,
    float* __restrict__ kc, float* __restrict__ kr) {
  const int b = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z, lane = threadIdx.x;
  const int qlen = query_start_loc[seq + 1] - query_start_loc[seq], seq_len = seq_lens[seq];
  if (qlen < min_qlen || seq_len - qlen < min_ctx || b * SP_BLK >= seq_len) return;
  const int* btab = block_table + (int64_t)seq * bt_stride;
  const uint8_t* head = cache + (int64_t)kvh * s_head;
  float mn[8], mx[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) { mn[k] = INFINITY; mx[k] = -INFINITY; }
  const int t_end = min(b * SP_BLK + SP_BLK, seq_len);
  for (int t = b * SP_BLK; t < t_end; ++t) {
    const uint8_t* row = head + (int64_t)btab[t / block_size] * s_blk + (int64_t)(t % block_size) * s_slot;
    const uint32_t w = *reinterpret_cast<const uint32_t*>(row + KV_OFF_K + lane * 4);
    const uint32_t kb = *reinterpret_cast<const uint32_t*>(row + KV_OFF_KS);
    const float ks = __uint_as_float(kb & ~0xFu), kz = (float)(kb & 0xFu);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float x = ks * ((float)((w >> (4 * k)) & 0xFu) - kz);
      mn[k] = fminf(mn[k], x);
      mx[k] = fmaxf(mx[k], x);
    }
  }
  const int64_t base = ((int64_t)seq * nkv + kvh) * HD;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    kc[(base + lane * 8 + k) * nbmax + b] = 0.5f * (mx[k] + mn[k]);
    kr[(base + lane * 8 + k) * nbmax + b] = 0.5f * (mx[k] - mn[k]);
  }
}

// 2) sparse_select_kernel: one wave per (work item, kv head). Block score = mean over the item's rows of the
//    per-row box upper bound of q.k  = mean(q).center + mean(|q|).radius (exact for the mean). Softmax over
//    the causally allowed blocks, then keep sink / local blocks plus the heaviest blocks until the softmax mass
//    reaches tau (threshold found by bisection). Writes the kept block ids ascending, -1 padded.
__device__ __forceinline__ void sparse_item_query(const int8_t* __restrict__ q8, const float* __restrict__ qs, int nqh,
                                                  int q0, int row0, int nrows, int G, int kvh, int lane,
                                                  float* qm, float* qa) {
  float sm[8], sa[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) { sm[k] = 0.f; sa[k] = 0.f; }
  for (int i = 0; i < 64; ++i) {
    const int row = row0 + i;
    if (i >= nrows) break;
    const int64_t qrow = (int64_t)(q0 + row / G) * nqh + kvh * G + row % G;
    const int8_t* src = q8 + qrow * HD + (lane >> 4) * 128 + (lane & 15) * 8;
    const float s = qs[qrow];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float v = (float)src[(k >> 1) + 4 * (k & 1)] * s;
      sm[k] += v;
      sa[k] += fabsf(v);
    }
  }
  const float inv = 1.f / (float)max(nrows, 1);
#pragma unroll
  for (int k = 0; k < 8; ++k) { qm[lane * 8 + k] = sm[k] * inv; qa[lane * 8 + k] = sa[k] * inv; }
}

__device__ __forceinline__ float wave_sum_f(float x) {
#pragma unroll
  for (int b = 16; b; b >>= 1) x += __shfl_xor(x, b, 32);
  return x;
}

__global__ void __launch_bounds__(32) sparse_select_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ qs, int nqh, const float* __restrict__ kc,
    const float* __restrict__ kr, const int* __restrict__ seq_lens, const int* __restrict__ query_start_loc,
    const int* __restrict__ work, const int* __restrict__ seq_ids, int G, int nbmax, float tau, int sink, int local,
    int min_ctx, int min_chunk, int* __restrict__ blist) {
  // work != nullptr: item = work[blockIdx.x] = (seq, first row); else item blockIdx.x of sequence seq_ids[blockIdx.z]
  // (split-K chunk path). List width is nbmax + 1: the last entry holds the number of kept blocks.
  const int lane = threadIdx.x, kvh = blockIdx.y, nkv = gridDim.y;
  const int seq = work ? work[blockIdx.x * 2] : seq_ids[blockIdx.z];
  const int row0 = work ? work[blockIdx.x * 2 + 1] : blockIdx.x * 64;
  const int item = work ? blockIdx.x : blockIdx.z * gridDim.x + blockIdx.x;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0, seq_len = seq_lens[seq];
  int* bl = blist + ((int64_t)item * nkv + kvh) * (nbmax + 1);
  if (row0 >= qlen * G) {  // empty item
    for (int i = lane; i <= nbmax; i += 32) bl[i] = i < nbmax ? -1 : 0;
    return;
  }
  const int last_row = min(row0 + 63, qlen * G - 1);
  const int max_pos = seq_len - qlen + last_row / G;
  const int nb = max_pos / SP_BLK + 1;
  __shared__ float qm[HD], qa[HD], sc[SP_NBMAX];
  const bool dense = tau >= 1.f || qlen < min_chunk || seq_len - qlen < min_ctx;
  if (!dense) {
    sparse_item_query(q8, qs, nqh, q0, row0, last_row - row0 + 1, G, kvh, lane, qm, qa);
    __syncthreads();
    const float* c = kc + (int64_t)(seq * nkv + kvh) * HD * nbmax;
    const float* r = kr + (int64_t)(seq * nkv + kvh) * HD * nbmax;
    for (int b = lane; b < nb; b += 32) {
      float a0 = 0.f;
      for (int d = 0; d < HD; ++d) a0 += qm[d] * c[(int64_t)d * nbmax + b] + qa[d] * r[(int64_t)d * nbmax + b];
      sc[b] = a0;
    }
    __syncthreads();
    float smax = -INFINITY;
    for (int b = lane; b < nb; b += 32) smax = fmaxf(smax, sc[b]);
    smax = wave_max(smax);
    float z = 0.f;
    for (int b = lane; b < nb; b += 32) { sc[b] = exp2f(sc[b] - smax); z += sc[b]; }
    z = wave_sum_f(z);
    float pmax = 0.f;
    for (int b = lane; b < nb; b += 32) { sc[b] /= z; pmax = fmaxf(pmax, sc[b]); }
    pmax = wave_max(pmax);
    float lo = 0.f, hi = pmax;
    for (int it = 0; it < 24; ++it) {
      const float mid = 0.5f * (lo + hi);
      float cov = 0.f;
      for (int b = lane; b < nb; b += 32) {
        const bool forced = b * SP_BLK < sink || b * SP_BLK + SP_BLK - 1 > max_pos - local;
        if (forced || sc[b] >= mid) cov += sc[b];
      }
      if (wave_sum_f(cov) >= tau) lo = mid; else hi = mid;
    }
    int off = 0;
    for (int base = 0; base < nb; base += 32) {
      const int b = base + lane;
      const bool forced = b * SP_BLK < sink || b * SP_BLK + SP_BLK - 1 > max_pos - local;
      const bool keep = b < nb && (forced || sc[b] >= lo);
      const uint32_t mask = __builtin_amdgcn_ballot_w32(keep);
      if (keep) bl[off + __builtin_popcount(mask & ((1u << lane) - 1u))] = b;
      off += __builtin_popcount(mask);
    }
    for (int i = off + lane; i < nbmax; i += 32) bl[i] = -1;
    if (lane == 0) bl[nbmax] = off;
  } else {
    for (int i = lane; i < nbmax; i += 32) bl[i] = i < nb ? i : -1;
    if (lane == 0) bl[nbmax] = nb;
  }
}

// ---------------------------------------------------------------------------------
// Prefill kernel with the K/V tile shared through LDS: the WG (WAVES waves = WAVES q tiles of
// the same kv head) loads each 16-token tile once, unpacks V once into an fp16 V^T layout
// ([dim][token]), and every wave reads its WMMA operands from LDS. Double-buffered.
template <typename T, int WAVES>
__global__ void __launch_bounds__(32 * WAVES) attn_prefill_lds_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ qs, const int* __restrict__ qsum, int nqh,
    const uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size,
    const int* __restrict__ block_table, int bt_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const int* __restrict__ work, int G, T* __restrict__ out,
    int64_t o_tok_stride, int64_t o_head_stride) {
  constexpr int NT = 32 * WAVES;
  __shared__ uint32_t k_lds[2][16 * 32];          // 16 tokens x 128 B
  __shared__ _Float16 vt_lds[2][256 * 16];        // V^T fp16: [dim][token]
  __shared__ float ks_lds[2][16], vs_lds[2][16];  // packed scale+zp
  __shared__ v2i q_lds[WAVES * 32 * 16];
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, r = lane & 15, h = lane >> 4;
  const int kvh = blockIdx.y;
  const int seq = work[blockIdx.x * 2], wg_row0 = work[blockIdx.x * 2 + 1];
  const int row0 = wg_row0 + wave * 16;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0;
  const int seq_len = seq_lens[seq];
  const int row = row0 + r;
  const bool valid = row < qlen * G;
  const int qi = row / G, g = row % G;
  const int qtok = q0 + (valid ? qi : 0), qh = kvh * G + g;
  {
    const int8_t* qrow = q8 + ((int64_t)qtok * nqh + qh) * HD + h * 128;
    v2i* my = q_lds + (wave * 32 + lane) * 16;
#pragma unroll
    for (int s = 0; s < 16; ++s) my[s] = valid ? *reinterpret_cast<const v2i*>(qrow + s * 8) : v2i{0, 0};
  }
  const float qsv = valid ? qs[(int64_t)qtok * nqh + qh] : 0.f;
  const float qsumv = valid ? (float)qsum[(int64_t)qtok * nqh + qh] : 0.f;
  const int qpos = valid ? (seq_len - qlen + qi) : -1;
  // WG key range: up to the last valid row of the WG
  const int wg_last = min(wg_row0 + 16 * WAVES - 1, qlen * G - 1);
  const int max_pos = seq_len - qlen + wg_last / G;
  const int ntiles = (max_pos + 16) >> 4;
  const int* btab = block_table + (int64_t)seq * bt_stride;
  const uint8_t* cache_head = cache + (int64_t)kvh * s_head;

  // Cooperative tile load (requires WAVES == 4: 128 threads).
  //   K: thread -> token tid>>3, 16 B piece tid&7.
  //   V: thread -> (dl = tid & 15, hh = (tid >> 4) & 1, jq = tid >> 5): 4 dim tiles 4*jq..4*jq+3.
  static_assert(WAVES == 4, "attn_prefill_lds_kernel expects 4 waves");
  uint4 kreg;
#ifdef BONSAI_V3
  uint2 va;
  uint32_t vbw;
#else
  uint4 vq;
#endif
  float sreg = 0.f;
  const int v_dl = tid & 15, v_hh = (tid >> 4) & 1, v_jq = tid >> 5;
  auto gload = [&](int t) {
    const int tok0 = t * 16;
    const int blk = btab[tok0 / block_size];
    const int p0 = tok0 % block_size;
    const uint8_t* tile = cache_head + (int64_t)blk * s_blk + (int64_t)p0 * s_slot;
    kreg = *reinterpret_cast<const uint4*>(tile + (int64_t)(tid >> 3) * s_slot + KV_OFF_K + (tid & 7) * 16);
    const uint8_t* vb = tile + (int64_t)v_dl * s_slot + KV_OFF_V + v_hh * V_LANE_BYTES;
#ifdef BONSAI_V3
    va = *reinterpret_cast<const uint2*>(vb + v_jq * 8);
    vbw = *reinterpret_cast<const uint32_t*>(vb + 32 + v_jq * 4);
#else
    vq = *reinterpret_cast<const uint4*>(vb + v_jq * 16);
#endif
    if (tid < 32) sreg = *reinterpret_cast<const float*>(tile + (int64_t)(tid & 15) * s_slot + ((tid < 16) ? KV_OFF_KS : KV_OFF_VS));
  };
  auto lstore = [&](int b) {
    *reinterpret_cast<uint4*>(&k_lds[b][tid * 4]) = kreg;
    if (tid < 16) ks_lds[b][tid] = sreg; else if (tid < 32) vs_lds[b][tid - 16] = sreg;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int j = 4 * v_jq + u;
#ifdef BONSAI_V3
      const uint32_t A = ((u >> 1) ? va.y : va.x) >> (8 * (u & 1));
      const v8h vv = unpack_v3(A, vbw >> (4 * u));
#else
      const uint32_t w = (u == 0) ? vq.x : (u == 1) ? vq.y : (u == 2) ? vq.z : vq.w;
      const v8h vv = unpack_v(w);
#endif
      *reinterpret_cast<v8h*>(&vt_lds[b][(16 * j + v_dl) * 16 + v_hh * 8]) = vv;
    }
  };
  v8f acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY, l = 0.f, corr = 0.f;
  const v2i* myq = q_lds + (wave * 32 + lane) * 16;
  if (ntiles > 0) { gload(0); }
  for (int t = 0; t < ntiles; ++t) {
    const int b = t & 1;
    lstore(b);
    __syncthreads();
    if (t + 1 < ntiles) gload(t + 1);
    // ---- S^T = K Q^T  (A = K tile from LDS: lane token r, bytes h*64..)
    v8i sacc = {0, 0, 0, 0, 0, 0, 0, 0}, sacc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t* krow = &k_lds[b][r * 32 + h * 16];
#pragma unroll
    for (int s = 0; s < 16; s += 2) {
      const uint32_t w = krow[s], w2 = krow[s + 1];
      v2i a = {(int)(w & 0x0F0F0F0Fu), (int)((w >> 4) & 0x0F0F0F0Fu)};
      v2i a2 = {(int)(w2 & 0x0F0F0F0Fu), (int)((w2 >> 4) & 0x0F0F0F0Fu)};
      sacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, myq[s], sacc, false);
      sacc2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a2, true, myq[s + 1], sacc2, false);
    }
    sacc += sacc2;
    const int tok0 = t * 16;
    float sc[8], mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int tt = h * 8 + i;
      const uint32_t kbits = __float_as_uint(ks_lds[b][tt]);
      float x = qsv * __uint_as_float(kbits & ~0xFu) * ((float)sacc[i] - (float)(kbits & 0xFu) * qsumv);
      const int kpos = tok0 + tt;
      if (kpos >= seq_len || kpos > qpos) x = -INFINITY;
      sc[i] = x;
      mx = fmaxf(mx, x);
    }
    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
    const float m_new = fmaxf(m, mx);
    const float alpha = (m_new == -INFINITY) ? 1.f : exp2f(m - m_new);
    v8h p;
    float psum = 0.f, pz = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float e = (m_new == -INFINITY) ? 0.f : exp2f(sc[i] - m_new);
      psum += e;
      const uint32_t vbits = __float_as_uint(vs_lds[b][h * 8 + i]);
      const float ev = e * __uint_as_float(vbits & ~0xFu);
      pz += ev * (float)(vbits & 0xFu);
      p[i] = (_Float16)ev;
    }
    l = l * alpha + psum;
    corr = corr * alpha + pz;
    m = m_new;
    if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
      for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[j][i] *= alpha;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const v8h a = *reinterpret_cast<const v8h*>(&vt_lds[b][(16 * j + r) * 16 + h * 8]);
      acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p, acc[j]);
    }
  }
  l += __shfl_xor(l, 16, 32);
  corr += __shfl_xor(corr, 16, 32);
  const float inv = l > 0.f ? 1.f / l : 0.f;
  float x[16][8];
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) x[j][i] = (acc[j][i] - corr) * inv;
#pragma unroll
  for (int bb = 1; bb < 8; bb <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (!(i & bb)) { float a = x[j][i], c = x[j][i | bb]; x[j][i] = a + c; x[j][i | bb] = a - c; }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float pp = __shfl_xor(x[j][i], 16, 32);
      x[j][i] = h ? (pp - x[j][i]) : (x[j][i] + pp);
    }
#pragma unroll
  for (int bb = 1; bb < 16; bb <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (!(j & bb))
#pragma unroll
        for (int i = 0; i < 8; ++i) { float a = x[j][i], c = x[j | bb][i]; x[j][i] = a + c; x[j | bb][i] = a - c; }
  if (!valid) return;
  T* dst = out + (int64_t)qtok * o_tok_stride + (int64_t)qh * o_head_stride;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    T vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) vals[i] = from_f<T>(x[j][i] * 0.0625f * rht_sign(16 * j + 8 * h + i));
    *reinterpret_cast<uint4*>(dst + 16 * j + 8 * h) = *reinterpret_cast<uint4*>(vals);
  }
}


// ---------------------------------------------------------------------------------
// Prefill variant 1 (BONSAI_PF_VARIANT=1, V3 only). Same per-wave math as attn_prefill_kernel (one 16-row q
// tile per wave, no cross-wave reduction -> bit-exact) but the WG (W waves, consecutive q tiles of one
// sequence) unpacks each 16-key tile ONCE into LDS in a compute-ready form: K as int8 pairs (iu8 WMMA
// A operand, row stride padded to 272 B: conflict-free), V as fp16 V^T. Double-buffered: tile t+1 is
// unpacked (from registers loaded one iteration earlier) while tile t is computed; one barrier per tile.
#ifdef BONSAI_V3
constexpr int PF3_KSTR = 34;  // K LDS row stride in v2i (32 words + 2 pad)

// In-place 16-point FWHT over (i, h) of one lane's 8 values and its xor-16 partner.
__device__ __forceinline__ void fwht16_lane(v8f& x, int h) {
#pragma unroll
  for (int b = 1; b < 8; b <<= 1)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      if (!(i & b)) { float a = x[i], c = x[i | b]; x[i] = a + c; x[i | b] = a - c; }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float p = __shfl_xor(x[i], 16, 32);
    x[i] = h ? (p - x[i]) : (x[i] + p);
  }
}

// Scale two outputs by 1/16 and the RHT signs of dims d, d+1, convert to T and pack into one word.
template <typename T>
__device__ __forceinline__ uint32_t pack_out(float a, float b, int d) {
  const uint16_t lo = __builtin_bit_cast(uint16_t, from_f<T>(a * 0.0625f * rht_sign(d)));
  const uint16_t hi = __builtin_bit_cast(uint16_t, from_f<T>(b * 0.0625f * rht_sign(d + 1)));
  return lo | ((uint32_t)hi << 16);
}

// Key tile count (exclusive) needed by rows [row0, row0 + n) of a sequence; 0 when no valid row.
__device__ __forceinline__ int pf3_ntiles(int row0, int n, int qlen, int G, int seq_len) {
  const int last_row = min(row0 + n - 1, qlen * G - 1);
  return last_row < row0 ? 0 : ((seq_len - qlen + last_row / G + 16) >> 4);
}

// Normalize + inverse FWHT in place (same operation order as attn_prefill_kernel's epilogue).
__device__ __forceinline__ void pf3_finish(v8f acc[16], float l, float corr, int h) {
  l += __shfl_xor(l, 16, 32);
  corr += __shfl_xor(corr, 16, 32);
  const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[j][i] = (acc[j][i] - corr) * inv;
    fwht16_lane(acc[j], h);
  }
#pragma unroll
  for (int b = 1; b < 16; b <<= 1)
#pragma unroll
    for (int j = 0; j < 16; ++j)
      if (!(j & b)) { v8f a = acc[j], c = acc[j | b]; acc[j] = a + c; acc[j | b] = a - c; }
}

template <typename T>
__device__ __forceinline__ void pf3_store(const v8f acc[16], T* dst, int h) {
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 8; i += 2) w[i >> 1] = pack_out<T>(acc[j][i], acc[j][i + 1], 16 * j + 8 * h + i);
    *reinterpret_cast<uint4*>(dst + 16 * j + 8 * h) = uint4{w[0], w[1], w[2], w[3]};
  }
}

template <typename T, int W>
__global__ void __launch_bounds__(32 * W, 8 / (W / 4)) attn_prefill3_kernel(
    const int8_t* __restrict__ q8, const float* __restrict__ qs, const int* __restrict__ qsum, int nqh,
    const uint8_t* __restrict__ cache, int64_t s_blk, int64_t s_head, int64_t s_slot, int block_size,
    const int* __restrict__ block_table, int bt_stride, const int* __restrict__ seq_lens,
    const int* __restrict__ query_start_loc, const int* __restrict__ work, int G, T* __restrict__ out,
    int64_t o_tok_stride, int64_t o_head_stride) {
  static_assert(W == 4 || W == 8, "attn_prefill3_kernel: W = 4 or 8");
  constexpr int UJ = 16 / W;  // V dim tiles unpacked per thread
  __shared__ __align__(16) v2i k_lds[2][16 * PF3_KSTR];
  __shared__ __align__(16) _Float16 vt_lds[2][256 * 16];  // V^T fp16: [dim][token]
  __shared__ float ks_lds[2][16], vs_lds[2][16];          // packed scale+zp
  __shared__ __align__(16) v2i q_lds[W * 16 * 32];        // [wave][step][lane]
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, r = lane & 15, h = lane >> 4;
  const int kvh = blockIdx.y;
  const int seq = work[blockIdx.x * 2], wg_row0 = work[blockIdx.x * 2 + 1], row0 = wg_row0 + wave * 16;
  const int q0 = query_start_loc[seq], qlen = query_start_loc[seq + 1] - q0;
  const int seq_len = seq_lens[seq];
  const int row = row0 + r, qi = row / G;
  const bool valid = row < qlen * G;
  const int qtok = q0 + (valid ? qi : 0), qh = kvh * G + row % G;
  {
    const int8_t* qrow = q8 + ((int64_t)qtok * nqh + qh) * HD + h * 128;
#pragma unroll
    for (int s = 0; s < 16; ++s)
      q_lds[(wave * 16 + s) * 32 + lane] = valid ? *reinterpret_cast<const v2i*>(qrow + s * 8) : v2i{0, 0};
  }
  const float qsc = valid ? qs[(int64_t)qtok * nqh + qh] : 0.f;
  const float qsm = valid ? (float)qsum[(int64_t)qtok * nqh + qh] : 0.f;
  const int qpos = valid ? (seq_len - qlen + qi) : -1;
  const int my_nt = pf3_ntiles(row0, 16, qlen, G, seq_len);
  const int blk_nt = pf3_ntiles(wg_row0, 16 * W, qlen, G, seq_len);
  const int* btab = block_table + (int64_t)seq * bt_stride;
  const uint8_t* cache_head = cache + (int64_t)kvh * s_head;

  // Cooperative tile load. K: thread (< 128) -> token tid>>3, 16 B piece tid&7.
  // V: thread -> (dl = tid & 15 = region, hh = (tid >> 4) & 1, jq = tid >> 5): dim tiles UJ*jq .. UJ*jq+UJ-1.
  uint4 kreg = {0, 0, 0, 0};
  uint2 va = {0, 0};
  uint32_t vbw = 0;
  float sreg = 0.f;
  const int v_dl = tid & 15, v_hh = (tid >> 4) & 1, v_jq = tid >> 5;
  auto gload = [&](int t) {
    const int tok0 = t * 16;
    const uint8_t* tile = cache_head + (int64_t)btab[tok0 / block_size] * s_blk + (int64_t)(tok0 % block_size) * s_slot;
    if (tid < 128) kreg = *reinterpret_cast<const uint4*>(tile + (int64_t)(tid >> 3) * s_slot + KV_OFF_K + (tid & 7) * 16);
    const uint8_t* vb = tile + (int64_t)v_dl * s_slot + KV_OFF_V + v_hh * V_LANE_BYTES;
    if (W == 4) {
      va = *reinterpret_cast<const uint2*>(vb + v_jq * 8);
      vbw = *reinterpret_cast<const uint32_t*>(vb + 32 + v_jq * 4);
    } else {
      va.x = *reinterpret_cast<const uint32_t*>(vb + v_jq * 4);
      vbw = *reinterpret_cast<const uint32_t*>(vb + 32 + (v_jq >> 1) * 4);
    }
    if (tid < 32) sreg = *reinterpret_cast<const float*>(tile + (int64_t)(tid & 15) * s_slot + ((tid < 16) ? KV_OFF_KS : KV_OFF_VS));
  };
  auto stage = [&](int b) {
    if (tid < 128) {
      const uint32_t wk[4] = {kreg.x, kreg.y, kreg.z, kreg.w};
      v2i u[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) u[k] = v2i{(int)(wk[k] & 0x0F0F0F0Fu), (int)((wk[k] >> 4) & 0x0F0F0F0Fu)};
      uint4* kd = reinterpret_cast<uint4*>(&k_lds[b][(tid >> 3) * PF3_KSTR + (tid & 7) * 4]);
      kd[0] = uint4{(uint32_t)u[0][0], (uint32_t)u[0][1], (uint32_t)u[1][0], (uint32_t)u[1][1]};
      kd[1] = uint4{(uint32_t)u[2][0], (uint32_t)u[2][1], (uint32_t)u[3][0], (uint32_t)u[3][1]};
    }
    if (tid < 16) ks_lds[b][tid] = sreg; else if (tid < 32) vs_lds[b][tid - 16] = sreg;
#pragma unroll
    for (int u = 0; u < UJ; ++u) {
      const int j = UJ * v_jq + u;
      const uint32_t A = (W == 4 ? ((u >> 1) ? va.y : va.x) >> (8 * (u & 1)) : va.x >> (8 * u));
      const uint32_t B = vbw >> (W == 4 ? 4 * u : 4 * (j & 3));
      *reinterpret_cast<v8h*>(&vt_lds[b][(16 * j + v_dl) * 16 + v_hh * 8]) = unpack_v3(A, B);
    }
  };
  v8f acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  float m = -INFINITY, l = 0.f, corr = 0.f;
  if (blk_nt > 0) { gload(0); stage(0); if (blk_nt > 1) gload(1); }
  __syncthreads();
  for (int t = 0; t < blk_nt; ++t) {
    const int b = t & 1;
    if (t + 1 < blk_nt) { stage(b ^ 1); if (t + 2 < blk_nt) gload(t + 2); }
    if (t < my_nt) {
      v8i s0 = {0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
      const v2i* kr = &k_lds[b][r * PF3_KSTR + h * 16];
#pragma unroll
      for (int s = 0; s < 16; s += 2) {
        s0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, kr[s], true, q_lds[(wave * 16 + s) * 32 + lane], s0, false);
        s1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, kr[s + 1], true, q_lds[(wave * 16 + s + 1) * 32 + lane], s1, false);
        if ((s & 3) == 2) __builtin_amdgcn_sched_barrier(0);  // bound the K/Q reads in flight (VGPRs)
      }
      s0 += s1;
      v8h p;
      float alpha;
      softmax_tile(qsc, qsm, qpos, s0, ks_lds[b][r], vs_lds[b][r], t * 16, seq_len, lane, m, l, corr, p, alpha);
      if (__builtin_amdgcn_ballot_w32(alpha != 1.f)) {
#pragma unroll
        for (int j = 0; j < 16; ++j)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[j][i] *= alpha;
      }
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const v8h a = *reinterpret_cast<const v8h*>(&vt_lds[b][(16 * j + r) * 16 + h * 8]);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, p, acc[j]);
      }
    }
    __syncthreads();
  }
  if (!valid) return;
  pf3_finish(acc, l, corr, h);
  pf3_store<T>(acc, out + (int64_t)qtok * o_tok_stride + (int64_t)qh * o_head_stride, h);
}
#endif

// ---------------------------------------------------------------------------------
// Small-M (M <= 16) ternary GEMM (byte-plane Q2 weights of ternary_fast.py): the WG splits K over
// SPLITK waves (LDS reduction), activations are loaded 16 B at a time and the activation
// scales are staged once in LDS. ~485 GB/s at M=16 vs ~340 for the plugin's wmma_small.
constexpr int T16_GROUP = 128, T16_CHUNK = 512;
constexpr unsigned T16_MASK = 0x03030303u;
__device__ __forceinline__ int t16_signed_codes(int v) { return ((v | 0x80808080) - 0x01010101) ^ 0x80808080; }
__device__ __forceinline__ v8i t16_wmma_iu8(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
template <typename scalar_t, int SPLITK, int TN, int MAXG>
__global__ void __launch_bounds__(32 * SPLITK) tern_m16_kernel(
    const uint8_t* __restrict__ q, const half* __restrict__ scales, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, scalar_t* __restrict__ y, const int M, const int N, const int K) {
  extern __shared__ float xs_lds[];                   // [g][m], groups * 16 floats (dynamic)
  __shared__ float red[SPLITK > 1 ? (SPLITK - 1) : 1][TN][8][32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int col = lane & 15, half_id = lane >> 4;
  const int groups = K / T16_GROUP, chunks = K / T16_CHUNK;
  const int n0 = blockIdx.x * 16 * TN;
  for (int i = threadIdx.x; i < groups * 16; i += 32 * SPLITK) {
    const int g = i >> 4, m = i & 15;
    xs_lds[i] = m < M ? xs[(int64_t)m * groups + g] : 0.f;
  }
  __syncthreads();
  const bool m_valid = col < M;
  const int8_t* xrow = xq + (int64_t)(m_valid ? col : 0) * K + half_id * 64;
  const uint8_t* qrow[TN];
  const half* srow[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const int n = min(n0 + t * 16 + col, N - 1);
    qrow[t] = q + (int64_t)n * (K / 4) + half_id * 64;
    srow[t] = scales + (int64_t)n * groups;
  }
  float acc[TN][8];
#pragma unroll
  for (int t = 0; t < TN; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;

  int4 wcur[TN][4], wnxt[TN][4];
  auto loadw = [&](int4 (&dst)[TN][4], int c) {
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u)
        dst[t][u] = c < chunks ? reinterpret_cast<const int4*>(qrow[t] + (int64_t)c * 128)[u] : make_int4(0, 0, 0, 0);
  };
  int c = wave;
  loadw(wcur, c);
  for (; c < chunks; c += SPLITK) {
    loadw(wnxt, c + SPLITK);
    // activations for the 4 groups of this chunk: 4 x 64 B per lane
    int4 xa[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int u = 0; u < 4; ++u)
        xa[i][u] = m_valid ? reinterpret_cast<const int4*>(xrow + (int64_t)c * T16_CHUNK + i * T16_GROUP)[u]
                           : make_int4(0, 0, 0, 0);
    // weight scales for the 4 groups (8 B per tile)
    float wsv[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t) {
      const uint2 sv = *reinterpret_cast<const uint2*>(srow[t] + c * 4);
      wsv[t][0] = __half2float(__ushort_as_half((unsigned short)(sv.x & 0xFFFF)));
      wsv[t][1] = __half2float(__ushort_as_half((unsigned short)(sv.x >> 16)));
      wsv[t][2] = __half2float(__ushort_as_half((unsigned short)(sv.y & 0xFFFF)));
      wsv[t][3] = __half2float(__ushort_as_half((unsigned short)(sv.y >> 16)));
    }
    v8i cc[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) cc[t][i] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int st = 0; st < 8; ++st) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int4 x4 = xa[i][st >> 1];
        const v2i av = (st & 1) ? v2i{x4.z, x4.w} : v2i{x4.x, x4.y};
#pragma unroll
        for (int t = 0; t < TN; ++t) {
          const int4 w4 = wcur[t][st >> 1];
          const int lo = (st & 1) ? w4.z : w4.x;
          const int hi = (st & 1) ? w4.w : w4.y;
          const int sh = 2 * i;
          const v2i bv = v2i{t16_signed_codes((lo >> sh) & T16_MASK), t16_signed_codes((hi >> sh) & T16_MASK)};
          cc[t][i] = t16_wmma_iu8(av, bv, cc[t][i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int g = c * 4 + i;
      const float4 xs0 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 + half_id * 8]);
      const float4 xs1 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 + half_id * 8 + 4]);
      const float xsv[8] = {xs0.x, xs0.y, xs0.z, xs0.w, xs1.x, xs1.y, xs1.z, xs1.w};
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] = fmaf(wsv[t][i] * xsv[r], (float)cc[t][i][r], acc[t][r]);
    }
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u) wcur[t][u] = wnxt[t][u];
  }
  if constexpr (SPLITK > 1) {
    if (wave > 0) {
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) red[wave - 1][t][r][lane] = acc[t][r];
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 0; w < SPLITK - 1; ++w)
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] += red[w][t][r][lane];
  }
  // accumulator: column col (n), rows half_id*8 + r (m)
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const int n = n0 + t * 16 + col;
    if (n >= N) continue;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int m = half_id * 8 + r;
      if (m < M) y[(int64_t)m * N + n] = from_f<scalar_t>(acc[t][r]);
    }
  }
}

// ---------------------------------------------------------------------------------
// Same scheme for int8 weights stored in the activation quantizer's permuted order (so the
// B operand is 8 weight bytes, no unpacking): used for the rotated, requantized MTP head.
// grid = (ceil(N / 16), ceil(M / 16)); each WG handles one 16-row M tile.
template <typename scalar_t, int SPLITK, int MAXG>
__global__ void __launch_bounds__(32 * SPLITK) w8_m16_kernel(
    const int8_t* __restrict__ w8, const half* __restrict__ scales, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, scalar_t* __restrict__ y, const int M, const int N, const int K) {
  extern __shared__ float xs_lds[];
  __shared__ float red[SPLITK > 1 ? (SPLITK - 1) : 1][8][32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int col = lane & 15, half_id = lane >> 4;
  const int groups = K / T16_GROUP, chunks = K / T16_CHUNK;
  const int n0 = blockIdx.x * 16, m0 = blockIdx.y * 16;
  for (int i = threadIdx.x; i < groups * 16; i += 32 * SPLITK) {
    const int g = i >> 4, m = m0 + (i & 15);
    xs_lds[i] = m < M ? xs[(int64_t)m * groups + g] : 0.f;
  }
  __syncthreads();
  const bool m_valid = m0 + col < M;
  const int8_t* xrow = xq + (int64_t)(m_valid ? m0 + col : 0) * K + half_id * 64;
  const int n = min(n0 + col, N - 1);
  const int8_t* wrow = w8 + (int64_t)n * K + half_id * 64;
  const half* srow = scales + (int64_t)n * groups;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int c = wave; c < chunks; c += SPLITK) {
    int4 wv[4][4], xa[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        wv[i][u] = reinterpret_cast<const int4*>(wrow + (int64_t)c * T16_CHUNK + i * T16_GROUP)[u];
        xa[i][u] = m_valid ? reinterpret_cast<const int4*>(xrow + (int64_t)c * T16_CHUNK + i * T16_GROUP)[u]
                           : make_int4(0, 0, 0, 0);
      }
    const uint2 sv = *reinterpret_cast<const uint2*>(srow + c * 4);
    const float wsv[4] = {__half2float(__ushort_as_half((unsigned short)(sv.x & 0xFFFF))),
                          __half2float(__ushort_as_half((unsigned short)(sv.x >> 16))),
                          __half2float(__ushort_as_half((unsigned short)(sv.y & 0xFFFF))),
                          __half2float(__ushort_as_half((unsigned short)(sv.y >> 16)))};
    v8i cc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) cc[i] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int st = 0; st < 8; ++st)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int4 x4 = xa[i][st >> 1], w4 = wv[i][st >> 1];
        const v2i av = (st & 1) ? v2i{x4.z, x4.w} : v2i{x4.x, x4.y};
        const v2i bv = (st & 1) ? v2i{w4.z, w4.w} : v2i{w4.x, w4.y};
        cc[i] = t16_wmma_iu8(av, bv, cc[i]);
      }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int g = c * 4 + i;
      const float4 xs0 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 + half_id * 8]);
      const float4 xs1 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 + half_id * 8 + 4]);
      const float xsv[8] = {xs0.x, xs0.y, xs0.z, xs0.w, xs1.x, xs1.y, xs1.z, xs1.w};
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[r] = fmaf(wsv[i] * xsv[r], (float)cc[i][r], acc[r]);
    }
  }
  if constexpr (SPLITK > 1) {
    if (wave > 0)
#pragma unroll
      for (int r = 0; r < 8; ++r) red[wave - 1][r][lane] = acc[r];
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 0; w < SPLITK - 1; ++w)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[r] += red[w][r][lane];
  }
  if (n0 + col >= N) return;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int m = m0 + half_id * 8 + r;
    if (m < M) y[(int64_t)m * N + n0 + col] = from_f<scalar_t>(acc[r]);
  }
}

// int8 weights x per-128 scales -> bf16 (large-M path of the MTP head)
__global__ void __launch_bounds__(256) w8_dequant_kernel(const int8_t* __restrict__ w8, const half* __restrict__ s,
                                                         __hip_bfloat16* __restrict__ out, int64_t total, int K) {
  const int64_t i = ((int64_t)blockIdx.x * 256 + threadIdx.x) * 8;
  if (i >= total) return;
  const float sc = __half2float(s[(i / K) * (K / 128) + (i % K) / 128]);
  const int2 v = *reinterpret_cast<const int2*>(w8 + i);
  const int8_t* b = reinterpret_cast<const int8_t*>(&v);
  __hip_bfloat16 o[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) o[k] = __float2bfloat16(sc * (float)b[k]);
  *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(o);
}

// ---------------------------------------------------------------------------------
// Ternary GEMM for 16 < M <= 64 (mixed decode + small prefill chunks): each wave keeps TM M-tiles,
// so the unpacked weight operand of a (step, group) is reused TM times. Split-K over the WG waves.
template <typename scalar_t, int SPLITK, int TM, int MAXG>
__global__ void __launch_bounds__(32 * SPLITK) tern_m64_kernel(
    const uint8_t* __restrict__ q, const half* __restrict__ scales, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, scalar_t* __restrict__ y, const int M, const int N, const int K) {
  extern __shared__ float xs_lds[];  // [g][m], groups * 16 * TM floats (dynamic)
  __shared__ float red[SPLITK > 1 ? (SPLITK - 1) : 1][TM][8][32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int col = lane & 15, half_id = lane >> 4;
  const int groups = K / T16_GROUP, chunks = K / T16_CHUNK;
  const int n0 = blockIdx.x * 16;
  const int m0 = blockIdx.y * 16 * TM;
  for (int i = threadIdx.x; i < groups * 16 * TM; i += 32 * SPLITK) {
    const int g = i / (16 * TM), m = m0 + i % (16 * TM);
    xs_lds[i] = m < M ? xs[(int64_t)m * groups + g] : 0.f;
  }
  __syncthreads();
  const int n = min(n0 + col, N - 1);
  const uint8_t* qrow = q + (int64_t)n * (K / 4) + half_id * 64;
  const half* srow = scales + (int64_t)n * groups;
  float acc[TM][8];
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;
  int4 wcur[4], wnxt[4];
  auto loadw = [&](int4 (&dst)[4], int c) {
#pragma unroll
    for (int u = 0; u < 4; ++u)
      dst[u] = c < chunks ? reinterpret_cast<const int4*>(qrow + (int64_t)c * 128)[u] : make_int4(0, 0, 0, 0);
  };
  int c = wave;
  loadw(wcur, c);
  for (; c < chunks; c += SPLITK) {
    loadw(wnxt, c + SPLITK);
    const uint2 sv = *reinterpret_cast<const uint2*>(srow + c * 4);
    const float wsv[4] = {__half2float(__ushort_as_half((unsigned short)(sv.x & 0xFFFF))),
                          __half2float(__ushort_as_half((unsigned short)(sv.x >> 16))),
                          __half2float(__ushort_as_half((unsigned short)(sv.y & 0xFFFF))),
                          __half2float(__ushort_as_half((unsigned short)(sv.y >> 16)))};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int4 xa[TM][4];
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        const int m = m0 + t * 16 + col;
        const int8_t* xr = xq + (int64_t)(m < M ? m : 0) * K + half_id * 64 + (int64_t)c * T16_CHUNK + i * T16_GROUP;
#pragma unroll
        for (int u = 0; u < 4; ++u) xa[t][u] = m < M ? reinterpret_cast<const int4*>(xr)[u] : make_int4(0, 0, 0, 0);
      }
      v8i cc[TM];
#pragma unroll
      for (int t = 0; t < TM; ++t) cc[t] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        const int4 w4 = wcur[st >> 1];
        const int lo = (st & 1) ? w4.z : w4.x, hi = (st & 1) ? w4.w : w4.y;
        const v2i bv = v2i{t16_signed_codes((lo >> (2 * i)) & T16_MASK), t16_signed_codes((hi >> (2 * i)) & T16_MASK)};
#pragma unroll
        for (int t = 0; t < TM; ++t) {
          const int4 x4 = xa[t][st >> 1];
          const v2i av = (st & 1) ? v2i{x4.z, x4.w} : v2i{x4.x, x4.y};
          cc[t] = t16_wmma_iu8(av, bv, cc[t]);
        }
      }
      const int g = c * 4 + i;
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        const float4 xs0 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 * TM + t * 16 + half_id * 8]);
        const float4 xs1 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 * TM + t * 16 + half_id * 8 + 4]);
        const float xsv[8] = {xs0.x, xs0.y, xs0.z, xs0.w, xs1.x, xs1.y, xs1.z, xs1.w};
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] = fmaf(wsv[i] * xsv[r], (float)cc[t][r], acc[t][r]);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) wcur[u] = wnxt[u];
  }
  if constexpr (SPLITK > 1) {
    if (wave > 0)
#pragma unroll
      for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) red[wave - 1][t][r][lane] = acc[t][r];
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 0; w < SPLITK - 1; ++w)
#pragma unroll
      for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] += red[w][t][r][lane];
  }
  if (n0 + col >= N) return;
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int m = m0 + t * 16 + half_id * 8 + r;
      if (m < M) y[(int64_t)m * N + n0 + col] = from_f<scalar_t>(acc[t][r]);
    }
}

// ---------------------------------------------------------------------------------
// The plugin's tiled ternary WMMA GEMM (copied to instantiate 64-row tiles for mixed steps)
template <typename scalar_t, int WAVES_M, int WAVES_N, int WM, int WN>
__global__ void __launch_bounds__(32 * WAVES_M* WAVES_N)
    pg_gemm_kernel(const uint8_t* __restrict__ q,
                     const half* __restrict__ scales,
                     const int8_t* __restrict__ xq,
                     const float* __restrict__ xs, scalar_t* __restrict__ y,
                     const int M, const int N, const int K) {
  constexpr int BM = WAVES_M * WM * 16;
  constexpr int BN = WAVES_N * WN * 16;
  constexpr int NT = 32 * WAVES_M * WAVES_N;
  constexpr int PAD = 136;
  constexpr int A_PER = (BM * 8 + NT - 1) / NT;
  constexpr int B_PER = (BN * 8 + NT - 1) / NT;
  constexpr int S_PER = (BN * 4 + NT - 1) / NT;
  __shared__ int8_t a_smem[BM * PAD];
  __shared__ uint8_t b_smem[BN * PAD];
  __shared__ float xs_smem[BM];
  __shared__ float ws_smem[BN * 4];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave / WAVES_N;
  const int wn = wave % WAVES_N;
  const int num_m = (M + BM - 1) / BM;
  const int m0 = (blockIdx.x % num_m) * BM;
  const int n0 = (blockIdx.x / num_m) * BN;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int groups = K / T16_GROUP;
  const int64_t row_bytes = K / 4;

  int4 a_reg[A_PER];
  int4 b_reg[B_PER];
  float ws_reg[S_PER];
  float xs_reg = 0.f;
  auto load_a = [&](int g) {
#pragma unroll
    for (int j = 0; j < A_PER; ++j) {
      const int idx = tid + j * NT;
      const int m = m0 + (idx >> 3);
      a_reg[j] = (idx < BM * 8 && m < M)
                     ? *reinterpret_cast<const int4*>(
                           xq + static_cast<int64_t>(m) * K + g * T16_GROUP +
                           (idx & 7) * 16)
                     : make_int4(0, 0, 0, 0);
    }
    if (tid < BM) {
      const int m = m0 + tid;
      xs_reg = m < M ? xs[static_cast<int64_t>(m) * groups + g] : 0.f;
    }
  };
  auto load_b = [&](int chunk) {
#pragma unroll
    for (int j = 0; j < B_PER; ++j) {
      const int idx = tid + j * NT;
      const int n = n0 + (idx >> 3);
      b_reg[j] = (idx < BN * 8 && n < N)
                     ? *reinterpret_cast<const int4*>(
                           q + n * row_bytes + chunk * 128 + (idx & 7) * 16)
                     : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < S_PER; ++j) {
      const int idx = tid + j * NT;
      const int n = n0 + (idx >> 2);
      ws_reg[j] =
          (idx < BN * 4 && n < N)
              ? __half2float(scales[static_cast<int64_t>(n) * groups +
                                    chunk * 4 + (idx & 3)])
              : 0.f;
    }
  };
  auto store_a = [&]() {
#pragma unroll
    for (int j = 0; j < A_PER; ++j) {
      const int idx = tid + j * NT;
      if (idx < BM * 8) {
        *reinterpret_cast<int4*>(a_smem + (idx >> 3) * PAD + (idx & 7) * 16) =
            a_reg[j];
      }
    }
    if (tid < BM) {
      xs_smem[tid] = xs_reg;
    }
  };
  auto store_b = [&]() {
#pragma unroll
    for (int j = 0; j < B_PER; ++j) {
      const int idx = tid + j * NT;
      if (idx < BN * 8) {
        *reinterpret_cast<int4*>(b_smem + (idx >> 3) * PAD + (idx & 7) * 16) =
            b_reg[j];
      }
    }
#pragma unroll
    for (int j = 0; j < S_PER; ++j) {
      const int idx = tid + j * NT;
      if (idx < BN * 4) {
        ws_smem[idx] = ws_reg[j];
      }
    }
  };

  float acc[WM][WN][8];
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[a][b][r] = 0.f;

  load_a(0);
  load_b(0);
  store_a();
  store_b();
  __syncthreads();
  for (int g = 0; g < groups; ++g) {
    const int i = g & 3;
    const bool more = g + 1 < groups;
    const bool new_chunk = ((g + 1) & 3) == 0;
    if (more) {
      load_a(g + 1);
      if (new_chunk) {
        load_b((g + 1) >> 2);
      }
    }
    v8i c[WM][WN];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
      for (int b = 0; b < WN; ++b) c[a][b] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    const int sh = 2 * i;
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      v2i av[WM];
      v2i bv[WN];
#pragma unroll
      for (int a = 0; a < WM; ++a) {
        av[a] = *reinterpret_cast<const v2i*>(
            a_smem + (wm * WM * 16 + a * 16 + col) * PAD + half_id * 64 +
            st * 8);
      }
#pragma unroll
      for (int b = 0; b < WN; ++b) {
        const v2i raw = *reinterpret_cast<const v2i*>(
            b_smem + (wn * WN * 16 + b * 16 + col) * PAD + half_id * 64 +
            st * 8);
        bv[b] = v2i{t16_signed_codes((raw.x >> sh) & T16_MASK),
                      t16_signed_codes((raw.y >> sh) & T16_MASK)};
      }
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WN; ++b) c[a][b] = t16_wmma_iu8(av[a], bv[b], c[a][b]);
    }
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const float wsv = ws_smem[(wn * WN * 16 + b * 16 + col) * 4 + i];
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float f =
              wsv * xs_smem[wm * WM * 16 + a * 16 + half_id * 8 + r];
          acc[a][b][r] = fmaf(f, static_cast<float>(c[a][b][r]), acc[a][b][r]);
        }
    }
    __syncthreads();
    if (more) {
      store_a();
      if (new_chunk) {
        store_b();
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const int n = n0 + wn * WN * 16 + b * 16 + col;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = m0 + wm * WM * 16 + a * 16 + half_id * 8 + r;
        if (m < M && n < N) {
          y[static_cast<int64_t>(m) * N + n] = from_f<scalar_t>(acc[a][b][r]);
        }
      }
    }
}

// Small-M GEMM: each wave owns TN 16x16 output tiles (sharing the activation
// loads) and reads activations and weight bytes straight from global memory,
// with the weights of the next PF chunks in flight.

// ---------------------------------------------------------------------------------
// Ternary GEMM for 16 < M <= 64 with the activations shared through LDS: the WG (8 waves) covers
// 128 output columns; per 128-wide group the 64 activation rows are staged once in LDS and each
// wave (one 16-column tile, TM 16-row tiles) reads its A operands there. K is split over gridDim.y
// (fp32 atomic accumulation) to give small-N shapes enough WGs.
template <int TM>
__global__ void __launch_bounds__(256) tern_lds_kernel(
    const uint8_t* __restrict__ q, const half* __restrict__ scales, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, float* __restrict__ yacc, const int M, const int N, const int K, const int ksplit) {
  __shared__ int8_t xa_lds[2][16 * TM][T16_GROUP + 16];  // padded rows
  __shared__ float xs_lds[2][16 * TM];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, col = lane & 15, half_id = lane >> 4;
  const int groups = K / T16_GROUP;
  const int n0 = blockIdx.x * 128 + wave * 16;
  const int gpk = (groups + ksplit - 1) / ksplit;
  const int g_begin = blockIdx.y * gpk, g_end = min(groups, g_begin + gpk);
  const int n = min(n0 + col, N - 1);
  const uint8_t* qrow = q + (int64_t)n * (K / 4) + half_id * 64;
  const half* srow = scales + (int64_t)n * groups;
  float acc[TM][8];
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;
  // activation group loader: 16*TM rows x 128 B = TM * 2 KB; 256 threads x 16 B per (TM/... ) pass
  auto load_x = [&](int b, int g) {
    for (int idx = tid; idx < 16 * TM * 8; idx += 256) {
      const int m = idx >> 3, piece = idx & 7;
      int4 v = make_int4(0, 0, 0, 0);
      if (m < M) v = *reinterpret_cast<const int4*>(xq + (int64_t)m * K + (int64_t)g * T16_GROUP + piece * 16);
      *reinterpret_cast<int4*>(&xa_lds[b][m][piece * 16]) = v;
    }
    if (tid < 16 * TM) xs_lds[b][tid] = tid < M ? xs[(int64_t)tid * groups + g] : 0.f;
  };
  if (g_begin < g_end) load_x(0, g_begin);
  __syncthreads();
  for (int g = g_begin; g < g_end; ++g) {
    const int b = (g - g_begin) & 1;
    if (g + 1 < g_end) load_x(b ^ 1, g + 1);
    // weights of group g: chunk c = g / 4, plane i = g % 4; lane bytes h*64..+64 of the chunk
    const int c = g >> 2, i = g & 3;
    int4 w4s[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) w4s[u] = reinterpret_cast<const int4*>(qrow + (int64_t)c * 128)[u];
    const float wsv = __half2float(srow[g]);
    v8i cc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) cc[t] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int4 w4 = w4s[st >> 1];
      const int lo = (st & 1) ? w4.z : w4.x, hi = (st & 1) ? w4.w : w4.y;
      const v2i bv = v2i{t16_signed_codes((lo >> (2 * i)) & T16_MASK), t16_signed_codes((hi >> (2 * i)) & T16_MASK)};
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        const v2i av = *reinterpret_cast<const v2i*>(&xa_lds[b][t * 16 + col][half_id * 64 + st * 8]);
        cc[t] = t16_wmma_iu8(av, bv, cc[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
#pragma unroll
      for (int r = 0; r < 8; ++r)
        acc[t][r] = fmaf(wsv * xs_lds[b][t * 16 + half_id * 8 + r], (float)cc[t][r], acc[t][r]);
    }
    __syncthreads();
  }
  if (n0 + col >= N) return;
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int m = t * 16 + half_id * 8 + r;
      if (m < M) {
        if (ksplit == 1) yacc[(int64_t)m * N + n0 + col] = acc[t][r];
        else atomicAdd(&yacc[(int64_t)m * N + n0 + col], acc[t][r]);
      }
    }
}

// ---------------------------------------------------------------------------------
// tern_m64 with the activation loads software-pipelined over the 128-wide groups (the next group's
// A operands are in flight while the current group's WMMAs run).
template <typename scalar_t, int SPLITK>
__global__ void __launch_bounds__(32 * SPLITK) tern_m64p_kernel(
    const uint8_t* __restrict__ q, const half* __restrict__ scales, const int8_t* __restrict__ xq,
    const float* __restrict__ xs, scalar_t* __restrict__ y, const int M, const int N, const int K) {
  constexpr int TM = 2;
  extern __shared__ float xs_lds[];
  __shared__ float red[SPLITK > 1 ? (SPLITK - 1) : 1][TM][8][32];
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int col = lane & 15, half_id = lane >> 4;
  const int groups = K / T16_GROUP, chunks = K / T16_CHUNK;
  const int n0 = blockIdx.x * 16, m0 = blockIdx.y * 16 * TM;
  for (int i = threadIdx.x; i < groups * 16 * TM; i += 32 * SPLITK) {
    const int g = i / (16 * TM), m = m0 + i % (16 * TM);
    xs_lds[i] = m < M ? xs[(int64_t)m * groups + g] : 0.f;
  }
  __syncthreads();
  const int n = min(n0 + col, N - 1);
  const uint8_t* qrow = q + (int64_t)n * (K / 4) + half_id * 64;
  const half* srow = scales + (int64_t)n * groups;
  const int8_t* xr[TM];
  bool mv[TM];
#pragma unroll
  for (int t = 0; t < TM; ++t) {
    const int m = m0 + t * 16 + col;
    mv[t] = m < M;
    xr[t] = xq + (int64_t)(mv[t] ? m : 0) * K + half_id * 64;
  }
  float acc[TM][8];
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;
  // the wave's groups: chunks c = wave, wave + SPLITK, ... ; 4 groups per chunk
  const int my_chunks = (chunks - wave + SPLITK - 1) / SPLITK;
  const int my_groups = my_chunks * 4;
  auto gidx = [&](int k) { return (wave + (k >> 2) * SPLITK) * 4 + (k & 3); };
  int4 xa0[TM][4], xa1[TM][4];
  int4 wv0[4], wv1[4];
  auto load = [&](int4 (&xa)[TM][4], int k) {
    const int g = gidx(k);
#pragma unroll
    for (int t = 0; t < TM; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u)
        xa[t][u] = mv[t] ? reinterpret_cast<const int4*>(xr[t] + (int64_t)g * T16_GROUP)[u] : make_int4(0, 0, 0, 0);
  };
  auto loadw = [&](int4 (&wv)[4], int k) {
    const int c = gidx(k) >> 2;
#pragma unroll
    for (int u = 0; u < 4; ++u) wv[u] = reinterpret_cast<const int4*>(qrow + (int64_t)c * 128)[u];
  };
  auto compute = [&](const int4 (&xa)[TM][4], const int4 (&wv)[4], int k) {
    const int g = gidx(k), i = g & 3;
    const float wsv = __half2float(srow[g]);
    v8i cc[TM];
#pragma unroll
    for (int t = 0; t < TM; ++t) cc[t] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      const int4 w4 = wv[st >> 1];
      const int lo = (st & 1) ? w4.z : w4.x, hi = (st & 1) ? w4.w : w4.y;
      const v2i bv = v2i{t16_signed_codes((lo >> (2 * i)) & T16_MASK), t16_signed_codes((hi >> (2 * i)) & T16_MASK)};
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        const int4 x4 = xa[t][st >> 1];
        cc[t] = t16_wmma_iu8((st & 1) ? v2i{x4.z, x4.w} : v2i{x4.x, x4.y}, bv, cc[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < TM; ++t) {
      const float4 xs0 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 * TM + t * 16 + half_id * 8]);
      const float4 xs1 = *reinterpret_cast<const float4*>(&xs_lds[g * 16 * TM + t * 16 + half_id * 8 + 4]);
      const float xsv[8] = {xs0.x, xs0.y, xs0.z, xs0.w, xs1.x, xs1.y, xs1.z, xs1.w};
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[t][r] = fmaf(wsv * xsv[r], (float)cc[t][r], acc[t][r]);
    }
  };
  // groups come in fours (one chunk): unroll so every buffer index is static
  if (my_groups > 0) { load(xa0, 0); loadw(wv0, 0); }
  for (int k = 0; k < my_groups; k += 4) {
    load(xa1, k + 1);
    compute(xa0, wv0, k);
    load(xa0, k + 2);
    compute(xa1, wv0, k + 1);
    load(xa1, k + 3);
    if (k + 4 < my_groups) loadw(wv1, k + 4);
    compute(xa0, wv0, k + 2);
    if (k + 4 < my_groups) load(xa0, k + 4);
    compute(xa1, wv0, k + 3);
#pragma unroll
    for (int u = 0; u < 4; ++u) wv0[u] = wv1[u];
  }
  if constexpr (SPLITK > 1) {
    if (wave > 0)
#pragma unroll
      for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) red[wave - 1][t][r][lane] = acc[t][r];
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int w = 0; w < SPLITK - 1; ++w)
#pragma unroll
      for (int t = 0; t < TM; ++t)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[t][r] += red[w][t][r][lane];
  }
  if (n0 + col >= N) return;
#pragma unroll
  for (int t = 0; t < TM; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int m = m0 + t * 16 + half_id * 8 + r;
      if (m < M) y[(int64_t)m * N + n0 + col] = from_f<scalar_t>(acc[t][r]);
    }
}

// ---------------------------------------------------------------------------------
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

void kv_write(torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping, torch::Tensor cache) {
  // cache: uint8 view [num_blocks, nkv, block_size, 264] with arbitrary strides (bytes)
  const int T_ = key.size(0), nkv = key.size(1);
  if (T_ == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  dim3 grid(T_, nkv);
  if (key.scalar_type() == at::kHalf)
   hipLaunchKernelGGL(( kv_write_kernel<__half>), dim3(grid), dim3(32), 0, stream, 
        (const __half*)key.data_ptr(), (const __half*)value.data_ptr(), key.stride(0), key.stride(1), value.stride(0),
        value.stride(1), slot_mapping.data_ptr<int64_t>(), cache.data_ptr<uint8_t>(), cache.stride(0),
        cache.stride(1), cache.stride(2), cache.size(2));
  else
   hipLaunchKernelGGL(( kv_write_kernel<__hip_bfloat16>), dim3(grid), dim3(32), 0, stream, 
        (const __hip_bfloat16*)key.data_ptr(), (const __hip_bfloat16*)value.data_ptr(), key.stride(0), key.stride(1),
        value.stride(0), value.stride(1), slot_mapping.data_ptr<int64_t>(), cache.data_ptr<uint8_t>(),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.size(2));
}

void q_prep(torch::Tensor q, torch::Tensor q8, torch::Tensor qs, torch::Tensor qsum, double softmax_scale) {
  const int T_ = q.size(0), nh = q.size(1);
  if (T_ == 0) return;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const float sl2 = (float)(softmax_scale * 1.4426950408889634);
  dim3 grid(T_, nh);
  if (q.scalar_type() == at::kHalf)
   hipLaunchKernelGGL(( q_prep_kernel<__half>), dim3(grid), dim3(32), 0, stream, (const __half*)q.data_ptr(), q.stride(0), q.stride(1),
                                                   q8.data_ptr<int8_t>(), qs.data_ptr<float>(), qsum.data_ptr<int>(), sl2);
  else
   hipLaunchKernelGGL(( q_prep_kernel<__hip_bfloat16>), dim3(grid), dim3(32), 0, stream, (const __hip_bfloat16*)q.data_ptr(), q.stride(0), q.stride(1),
                                                           q8.data_ptr<int8_t>(), qs.data_ptr<float>(),
                                                           qsum.data_ptr<int>(), sl2);
}

static float env_float(const char* name, float dflt) {
  const char* v = std::getenv(name);
  return v ? (float)std::atof(v) : dflt;
}

static torch::Tensor g_last_blist;  // debug: block list of the last sparse call (BONSAI_SPARSE_STATS=1)
static torch::Tensor sparse_last_list() { return g_last_blist; }

// Block lists of the opt-in block-sparse prefill: K box scan + selection. work != nullptr: one item per work entry
// (prefill kernel); else `items` 64-row items per sequence of seq_ids (split-K chunk path). Returns
// [items, nkv, nbmax + 1] (last entry = kept count). max_seq_len bounds the list width (no device sync).
static torch::Tensor sparse_lists(torch::Tensor q8, torch::Tensor qs, torch::Tensor cache, torch::Tensor block_table,
                                  torch::Tensor seq_lens, torch::Tensor query_start_loc, const torch::Tensor* work,
                                  torch::Tensor seq_ids, int items, int64_t max_seq_len, float tau, int& nbmax_out) {
  const int nqh = q8.size(1), nkv = cache.size(1), G = nqh / nkv, nseq = seq_lens.size(0);
  const int64_t cap = max_seq_len > 0 ? max_seq_len : (int64_t)block_table.size(1) * cache.size(2);
  const int nbmax = (int)((cap + SP_BLK - 1) / SP_BLK);
  TORCH_CHECK(nbmax <= SP_NBMAX, "BONSAI_SPARSE_PREFILL: context too long (> ", SP_NBMAX * SP_BLK, ")");
  const int sink = (int)env_float("BONSAI_SPARSE_SINK", 2048), local = (int)env_float("BONSAI_SPARSE_LOCAL", 4096);
  const int min_ctx = (int)env_float("BONSAI_SPARSE_MIN_CTX", 16384);
  const int min_chunk = (int)env_float("BONSAI_SPARSE_MIN_CHUNK", 128);
  auto fopt = q8.options().dtype(at::kFloat);
  auto kc = torch::empty({(int64_t)nseq * nkv * HD * nbmax}, fopt), kr = torch::empty_like(kc);
  const int nitems = work ? (int)work->size(0) : items * (int)seq_ids.size(0);
  auto blist = torch::empty({(int64_t)nitems * nkv * (nbmax + 1)}, q8.options().dtype(at::kInt));
  auto stream = c10::hip::getCurrentHIPStream().stream();
  dim3 kgrid(nbmax, nkv, nseq), grid(work ? nitems : items, nkv, work ? 1 : (int)seq_ids.size(0));
  const int kbox_ctx = tau >= 1.f ? 0x7fffffff : min_ctx;  // tau = 1 keeps everything: no box scan
 hipLaunchKernelGGL(( sparse_kbox_kernel), dim3(kgrid), dim3(32), 0, stream, cache.data_ptr<uint8_t>(), cache.stride(0), cache.stride(1),
                                               cache.stride(2), cache.size(2), block_table.data_ptr<int>(),
                                               block_table.stride(0), seq_lens.data_ptr<int>(),
                                               query_start_loc.data_ptr<int>(), nkv, nbmax, kbox_ctx, min_chunk,
                                               kc.data_ptr<float>(), kr.data_ptr<float>());
 hipLaunchKernelGGL(( sparse_select_kernel), dim3(grid), dim3(32), 0, stream, 
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), nqh, kc.data_ptr<float>(), kr.data_ptr<float>(),
      seq_lens.data_ptr<int>(), query_start_loc.data_ptr<int>(), work ? work->data_ptr<int>() : nullptr,
      work ? nullptr : seq_ids.data_ptr<int>(), G, nbmax, tau, sink, local, min_ctx, min_chunk,
      blist.data_ptr<int>());
  nbmax_out = nbmax;
  if (std::getenv("BONSAI_SPARSE_STATS")) { g_last_blist = blist.view({nitems, nkv, nbmax + 1}); }
  return blist;
}

// Opt-in block-sparse prefill (variant 0 only): score -> select -> sparse attention.
template <typename TT>
static void attn_prefill_sparse(torch::Tensor q8, torch::Tensor qs, torch::Tensor qsum, torch::Tensor cache,
                                torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor query_start_loc,
                                torch::Tensor work, torch::Tensor out, int64_t max_seq_len, float tau) {
  const int nwork = work.size(0), nqh = q8.size(1), nkv = cache.size(1), G = nqh / nkv;
  int nbmax = 0;
  auto blist = sparse_lists(q8, qs, cache, block_table, seq_lens, query_start_loc, &work, torch::Tensor(), 0,
                            max_seq_len, tau, nbmax);
  auto stream = c10::hip::getCurrentHIPStream().stream();
  dim3 grid(nwork, nkv);
 hipLaunchKernelGGL(( attn_prefill_kernel<TT, 4>), dim3(grid), dim3(128), 0, stream, 
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), qsum.data_ptr<int>(), nqh, cache.data_ptr<uint8_t>(),
      cache.stride(0), cache.stride(1), cache.stride(2), cache.size(2), block_table.data_ptr<int>(),
      block_table.stride(0), seq_lens.data_ptr<int>(), query_start_loc.data_ptr<int>(), work.data_ptr<int>(), G,
      (TT*)out.data_ptr(), out.stride(0), out.stride(1), blist.data_ptr<int>(), nbmax);
}

// max_seq_len > 0 (split-K chunk calls only) enables the opt-in block-sparse selection (BONSAI_SPARSE_PREFILL).
void attn_decode(torch::Tensor q8, torch::Tensor qs, torch::Tensor qsum, torch::Tensor cache, torch::Tensor block_table,
                 torch::Tensor seq_lens, torch::Tensor query_start_loc, torch::Tensor seq_ids, int64_t qtiles,
                 int64_t num_splits, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor out,
                 int64_t max_seq_len) {
  const int nseq = seq_ids.size(0);
  if (nseq == 0) return;
  const int nqh = q8.size(1), nkv = cache.size(1), G = nqh / nkv;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  dim3 grid(nkv * qtiles, num_splits, nseq);
  const float tau = max_seq_len > 0 ? env_float("BONSAI_SPARSE_PREFILL", 0.f) : 0.f;
#define DEC_ARGS                                                                                                   \
  q8.data_ptr<int8_t>(), qs.data_ptr<float>(), qsum.data_ptr<int>(), nqh, cache.data_ptr<uint8_t>(),               \
      cache.stride(0), cache.stride(1), cache.stride(2), cache.size(2), block_table.data_ptr<int>(),               \
      block_table.stride(0), seq_lens.data_ptr<int>(), query_start_loc.data_ptr<int>(), seq_ids.data_ptr<int>(),  \
      G, qtiles, part_o.data_ptr<float>(), part_ml.data_ptr<float>(), num_splits
  if (tau > 0.f && tau < 1.f) {  // tau = 1 keeps every block: the dense kernel is the exact result
    int nbmax = 0;
    const int items = (qtiles + 3) / 4;
    auto blist = sparse_lists(q8, qs, cache, block_table, seq_lens, query_start_loc, nullptr, seq_ids, items,
                              max_seq_len, tau, nbmax);
   hipLaunchKernelGGL(( attn_decode_kernel<true>), dim3(grid), dim3(32), 0, stream, DEC_ARGS, blist.data_ptr<int>(), nbmax);
  } else {
   hipLaunchKernelGGL(( attn_decode_kernel<false>), dim3(grid), dim3(32), 0, stream, DEC_ARGS, nullptr, 0);
  }
#undef DEC_ARGS
  dim3 rgrid(nseq, nkv * qtiles * 16);
  if (out.scalar_type() == at::kHalf)
   hipLaunchKernelGGL(( attn_reduce_kernel<__half>), dim3(rgrid), dim3(32), 0, stream, part_o.data_ptr<float>(), part_ml.data_ptr<float>(), num_splits,
                                                         nkv, qtiles, G, query_start_loc.data_ptr<int>(),
                                                         seq_ids.data_ptr<int>(), (__half*)out.data_ptr(),
                                                         out.stride(0), out.stride(1));
  else
   hipLaunchKernelGGL(( attn_reduce_kernel<__hip_bfloat16>), dim3(rgrid), dim3(32), 0, stream, 
        part_o.data_ptr<float>(), part_ml.data_ptr<float>(), num_splits, nkv, qtiles, G,
        query_start_loc.data_ptr<int>(), seq_ids.data_ptr<int>(), (__hip_bfloat16*)out.data_ptr(), out.stride(0),
        out.stride(1));
}

void attn_prefill(torch::Tensor q8, torch::Tensor qs, torch::Tensor qsum, torch::Tensor cache, torch::Tensor block_table,
                  torch::Tensor seq_lens, torch::Tensor query_start_loc, torch::Tensor work, torch::Tensor out,
                  int64_t max_seq_len) {
  const int nwork = work.size(0);
  if (nwork == 0) return;
  const int nqh = q8.size(1), nkv = cache.size(1), G = nqh / nkv;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  dim3 grid(nwork, nkv);
  const float tau = env_float("BONSAI_SPARSE_PREFILL", 0.f);  // read per call: tests toggle it in-process
  if (tau > 0.f) {
    TORCH_CHECK(std::getenv("BONSAI_PF_VARIANT") == nullptr || std::atoi(std::getenv("BONSAI_PF_VARIANT")) == 0,
                "BONSAI_SPARSE_PREFILL needs BONSAI_PF_VARIANT=0");
    if (out.scalar_type() == at::kHalf) {
      attn_prefill_sparse<__half>(q8, qs, qsum, cache, block_table, seq_lens, query_start_loc, work, out, max_seq_len, tau);
    } else {
      attn_prefill_sparse<__hip_bfloat16>(q8, qs, qsum, cache, block_table, seq_lens, query_start_loc, work, out,
                                          max_seq_len, tau);
    }
    return;
  }
  const int variant = [] { const char* v = std::getenv("BONSAI_PF_VARIANT"); return v ? std::atoi(v) : 0; }();  // read per call: tests toggle it in-process
  // WG width of variant 1; the work items must hold 16 * pf_w rows (bonsai_int4_attn.PF_ROWS)
  const int pf_w = [] { const char* v = std::getenv("BONSAI_PF_W"); return v ? std::atoi(v) : 4; }();
  static const bool use_lds = [] { const char* v = std::getenv("BONSAI_PREFILL_LDS"); return v && std::atoi(v) != 0; }();
#define PF_ARGS(TT) \
      q8.data_ptr<int8_t>(), qs.data_ptr<float>(), qsum.data_ptr<int>(), nqh, cache.data_ptr<uint8_t>(), \
      cache.stride(0), cache.stride(1), cache.stride(2), cache.size(2), block_table.data_ptr<int>(), \
      block_table.stride(0), seq_lens.data_ptr<int>(), query_start_loc.data_ptr<int>(), work.data_ptr<int>(), G, \
      (TT*)out.data_ptr(), out.stride(0), out.stride(1)
#define PF_ARGS_SP(TT) PF_ARGS(TT), nullptr, 0
#ifdef BONSAI_V3
#define PF3_LAUNCH(TT) \
  if (pf_w == 8) {hipLaunchKernelGGL(( attn_prefill3_kernel<TT, 8>), dim3(grid), dim3(256), 0, stream, PF_ARGS(TT)); } \
  else {hipLaunchKernelGGL(( attn_prefill3_kernel<TT, 4>), dim3(grid), dim3(128), 0, stream, PF_ARGS(TT)); }
#else
#define PF3_LAUNCH(TT) TORCH_CHECK(false, "BONSAI_PF_VARIANT=1 needs BONSAI_V3");
#endif
#define LAUNCH(TT) \
  if (variant == 1) { PF3_LAUNCH(TT) } \
  else if (use_lds) {hipLaunchKernelGGL(( attn_prefill_lds_kernel<TT, 4>), dim3(grid), dim3(128), 0, stream, PF_ARGS(TT)); } \
  else {hipLaunchKernelGGL(( attn_prefill_kernel<TT, 4>), dim3(grid), dim3(128), 0, stream, PF_ARGS_SP(TT)); }
  if (out.scalar_type() == at::kHalf) { LAUNCH(__half) } else { LAUNCH(__hip_bfloat16) }
#undef LAUNCH
#undef PF3_LAUNCH
#undef PF_ARGS_SP
#undef PF_ARGS
}

template <typename scalar_t, int SPLITK, int TN>
static void t16_launch(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, torch::Tensor Y, int M, int N,
                   int K, hipStream_t stream) {
  const dim3 grid((N + 16 * TN - 1) / (16 * TN));
 hipLaunchKernelGGL(( tern_m16_kernel<scalar_t, SPLITK, TN, 160>), dim3(grid), dim3(32 * SPLITK), (K / 128) * 16 * sizeof(float), stream, 
      Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),
      (scalar_t*)Y.data_ptr(), M, N, K);
}

torch::Tensor tern_m16(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t splitk, int64_t tn,
                       bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = Q.size(0);
  TORCH_CHECK(M <= 16 && K / 128 <= 160, "tern_m16: M <= 16, K <= 20480");
  auto Y = torch::empty({M, N}, xq.options().dtype(bf16 ? torch::kBFloat16 : torch::kHalf));
  auto stream = c10::hip::getCurrentHIPStream().stream();
#define L(ST, TT)                                                                     \
  if (splitk == ST && tn == TT) {                                                     \
    if (bf16) t16_launch<__hip_bfloat16, ST, TT>(Q, S, xq, xs, Y, M, N, K, stream);       \
    else t16_launch<__half, ST, TT>(Q, S, xq, xs, Y, M, N, K, stream);                    \
    return Y;                                                                         \
  }
  L(1, 1) L(2, 1) L(4, 1) L(8, 1) L(1, 2) L(2, 2) L(4, 2) L(4, 4) L(2, 4)
#undef L
  TORCH_CHECK(false, "unsupported splitk/tn");
}

torch::Tensor w8_gemm(torch::Tensor W, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t splitk, bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = W.size(0);
  TORCH_CHECK(K / 128 <= 160, "w8_gemm: K <= 20480");
  auto Y = torch::empty({M, N}, xq.options().dtype(bf16 ? torch::kBFloat16 : torch::kHalf));
  if (M == 0) return Y;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid((N + 15) / 16, (M + 15) / 16);
#define WL(ST)                                                                                                   \
  if (splitk == ST) {                                                                                            \
    if (bf16) {                                                                                                  \
     hipLaunchKernelGGL(( w8_m16_kernel<__hip_bfloat16, ST, 160>), dim3(grid), dim3(32 * ST), (K / 128) * 16 * sizeof(float), stream,                                       \
          W.data_ptr<int8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),           \
          (__hip_bfloat16*)Y.data_ptr(), M, N, K);                                                               \
    } else {                                                                                                     \
     hipLaunchKernelGGL(( w8_m16_kernel<__half, ST, 160>), dim3(grid), dim3(32 * ST), (K / 128) * 16 * sizeof(float), stream,                                               \
          W.data_ptr<int8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),           \
          (__half*)Y.data_ptr(), M, N, K);                                                                       \
    }                                                                                                            \
    return Y;                                                                                                    \
  }
  WL(1) WL(2) WL(4)
#undef WL
  TORCH_CHECK(false, "w8_gemm: unsupported splitk");
}

torch::Tensor w8_dequant(torch::Tensor W, torch::Tensor S) {
  const int64_t N = W.size(0), K = W.size(1);
  auto out = torch::empty({N, K}, W.options().dtype(torch::kBFloat16));
  const int64_t total = N * K;
  auto stream = c10::hip::getCurrentHIPStream().stream();
 hipLaunchKernelGGL(( w8_dequant_kernel), dim3((unsigned)((total / 8 + 255) / 256)), dim3(256), 0, stream, 
      W.data_ptr<int8_t>(), (const half*)S.data_ptr(), (__hip_bfloat16*)out.data_ptr(), total, (int)K);
  return out;
}

torch::Tensor tern_m64(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t splitk, int64_t tm,
                       bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = Q.size(0);
  TORCH_CHECK(K / 128 <= 160, "tern_m64: K <= 20480");
  auto Y = torch::empty({M, N}, xq.options().dtype(bf16 ? torch::kBFloat16 : torch::kHalf));
  if (M == 0) return Y;
  auto stream = c10::hip::getCurrentHIPStream().stream();
#define TL(ST, TMV)                                                                                             \
  if (splitk == ST && tm == TMV) {                                                                              \
    const dim3 grid((N + 15) / 16, (M + 16 * TMV - 1) / (16 * TMV));                                            \
    if (bf16) {                                                                                                 \
     hipLaunchKernelGGL(( tern_m64_kernel<__hip_bfloat16, ST, TMV, 160>), dim3(grid), dim3(32 * ST), (K / 128) * 16 * TMV * sizeof(float), stream,                              \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__hip_bfloat16*)Y.data_ptr(), M, N, K);                                                              \
    } else {                                                                                                    \
     hipLaunchKernelGGL(( tern_m64_kernel<__half, ST, TMV, 160>), dim3(grid), dim3(32 * ST), (K / 128) * 16 * TMV * sizeof(float), stream,                                      \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__half*)Y.data_ptr(), M, N, K);                                                                      \
    }                                                                                                           \
    return Y;                                                                                                   \
  }
  TL(1, 2) TL(2, 2) TL(4, 2) TL(1, 4) TL(2, 4) TL(4, 4)
#undef TL
  TORCH_CHECK(false, "tern_m64: unsupported config");
}

torch::Tensor tern_tile(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t cfg, bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = Q.size(0);
  auto Y = torch::empty({M, N}, xq.options().dtype(bf16 ? torch::kBFloat16 : torch::kHalf));
  if (M == 0) return Y;
  auto stream = c10::hip::getCurrentHIPStream().stream();
#define PG(C, WMV, WNV, TMV, TNV)                                                                               \
  if (cfg == C) {                                                                                               \
    constexpr int BM = WMV * TMV * 16, BN = WNV * TNV * 16;                                                     \
    const dim3 grid(((M + BM - 1) / BM) * ((N + BN - 1) / BN));                                                \
    if (bf16) {                                                                                                 \
     hipLaunchKernelGGL(( pg_gemm_kernel<__hip_bfloat16, WMV, WNV, TMV, TNV>), dim3(grid), dim3(32 * WMV * WNV), 0, stream,                   \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__hip_bfloat16*)Y.data_ptr(), M, N, K);                                                              \
    } else {                                                                                                    \
     hipLaunchKernelGGL(( pg_gemm_kernel<__half, WMV, WNV, TMV, TNV>), dim3(grid), dim3(32 * WMV * WNV), 0, stream,                           \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__half*)Y.data_ptr(), M, N, K);                                                                      \
    }                                                                                                           \
    return Y;                                                                                                   \
  }
  PG(0, 2, 4, 4, 2)  // plugin default: 128 x 128
  PG(1, 1, 8, 4, 1)  // 64 x 128
  PG(2, 1, 4, 4, 2)  // 64 x 128, 4 waves
  PG(3, 1, 4, 4, 1)  // 64 x 64
  PG(4, 2, 4, 2, 2)  // 64 x 128, 8 waves
  PG(5, 1, 8, 3, 1)  // 48 x 128
  PG(6, 1, 4, 3, 2)  // 48 x 128, 4 waves
#undef PG
  TORCH_CHECK(false, "tern_tile: bad cfg");
}

torch::Tensor tern_lds(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t ksplit, bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = Q.size(0);
  TORCH_CHECK(M <= 64, "tern_lds: M <= 64");
  auto acc = ksplit > 1 ? torch::zeros({M, N}, xq.options().dtype(torch::kFloat32))
                        : torch::empty({M, N}, xq.options().dtype(torch::kFloat32));
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid((N + 127) / 128, ksplit);
  if (M <= 32) {
   hipLaunchKernelGGL(( tern_lds_kernel<2>), dim3(grid), dim3(256), 0, stream, Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(),
                                                 xs.data_ptr<float>(), acc.data_ptr<float>(), M, N, K, (int)ksplit);
  } else {
   hipLaunchKernelGGL(( tern_lds_kernel<4>), dim3(grid), dim3(256), 0, stream, Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(),
                                                 xs.data_ptr<float>(), acc.data_ptr<float>(), M, N, K, (int)ksplit);
  }
  return acc.to(bf16 ? torch::kBFloat16 : torch::kHalf);
}

torch::Tensor tern_m64p(torch::Tensor Q, torch::Tensor S, torch::Tensor xq, torch::Tensor xs, int64_t splitk, bool bf16) {
  const int M = xq.size(0), K = xq.size(1), N = Q.size(0);
  auto Y = torch::empty({M, N}, xq.options().dtype(bf16 ? torch::kBFloat16 : torch::kHalf));
  if (M == 0) return Y;
  auto stream = c10::hip::getCurrentHIPStream().stream();
  const dim3 grid((N + 15) / 16, (M + 31) / 32);
  const size_t shm = (K / 128) * 32 * sizeof(float);
#define PL(ST)                                                                                                  \
  if (splitk == ST) {                                                                                           \
    if (bf16) {                                                                                                 \
     hipLaunchKernelGGL(( tern_m64p_kernel<__hip_bfloat16, ST>), dim3(grid), dim3(32 * ST), shm, stream,                                      \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__hip_bfloat16*)Y.data_ptr(), M, N, K);                                                              \
    } else {                                                                                                    \
     hipLaunchKernelGGL(( tern_m64p_kernel<__half, ST>), dim3(grid), dim3(32 * ST), shm, stream,                                              \
          Q.data_ptr<uint8_t>(), (const half*)S.data_ptr(), xq.data_ptr<int8_t>(), xs.data_ptr<float>(),         \
          (__half*)Y.data_ptr(), M, N, K);                                                                      \
    }                                                                                                           \
    return Y;                                                                                                   \
  }
  PL(1) PL(2) PL(4)
#undef PL
  TORCH_CHECK(false, "tern_m64p: bad splitk");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tern_m16", &tern_m16);
  m.def("tern_m64", &tern_m64);
  m.def("tern_tile", &tern_tile);
  m.def("tern_lds", &tern_lds);
  m.def("tern_m64p", &tern_m64p);
  m.def("w8_gemm", &w8_gemm);
  m.def("w8_dequant", &w8_dequant);
  m.def("kv_write", &kv_write);
  m.def("q_prep", &q_prep);
  m.def("attn_decode", &attn_decode, py::arg("q8"), py::arg("qs"), py::arg("qsum"), py::arg("cache"),
        py::arg("block_table"), py::arg("seq_lens"), py::arg("query_start_loc"), py::arg("seq_ids"), py::arg("qtiles"),
        py::arg("num_splits"), py::arg("part_o"), py::arg("part_ml"), py::arg("out"), py::arg("max_seq_len") = 0);
  m.def("sparse_last_list", &sparse_last_list);
  m.def("attn_prefill", &attn_prefill, py::arg("q8"), py::arg("qs"), py::arg("qsum"), py::arg("cache"),
        py::arg("block_table"), py::arg("seq_lens"), py::arg("query_start_loc"), py::arg("work"), py::arg("out"),
        py::arg("max_seq_len") = 0);
}
