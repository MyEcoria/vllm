// Ternary (prism.hadamard) decode kernels.
//
// Weights are repacked at load time ("byte-plane" layout): each row holds
// chunks of 512 weights in 128 bytes. Byte w * 4 + b of a chunk stores, at bits
// 2i, the code (value + 1) of weight i * 128 + b * 32 + w. Read as a
// little-endian int32 word w, (word >> 2i) & 0x03030303 therefore holds four
// codes of scale group i (128 weights, one fp16 scale) in its byte lanes.
//
// Activations are rotated (sign flip + normalized Sylvester-Hadamard) and
// quantized to int8 per 128-wide group, then stored in the same permuted order
// so that int32 i * 32 + w of a chunk pairs with word w of group i.

#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "../dispatch_utils.h"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

namespace {

inline cudaStream_t current_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return static_cast<cudaStream_t>(raw_stream);
}

constexpr int kGroup = 128;
constexpr int kGemvWaves = 4;
constexpr int kChunk = 512;
constexpr unsigned kCodeMask = 0x03030303u;

__device__ __forceinline__ float shfl_xor(float v, int offset) {
#ifdef USE_ROCM
  return __shfl_xor(v, offset, 32);
#else
  return __shfl_xor_sync(0xffffffffu, v, offset, 32);
#endif
}

__device__ __forceinline__ int shfl_xor(int v, int offset) {
#ifdef USE_ROCM
  return __shfl_xor(v, offset, 32);
#else
  return __shfl_xor_sync(0xffffffffu, v, offset, 32);
#endif
}

__device__ __forceinline__ int dot4_codes(int codes, int x, int acc) {
#ifdef USE_ROCM
  return __builtin_amdgcn_sudot4(false, codes, true, x, acc, false);
#else
  return __dp4a(codes, x, acc);
#endif
}

// One block of BLOCK / 16 threads per (row, Hadamard block); each thread
// holds 16 values. Element e of the block lives in thread (e >> 9) * 32 +
// ((e >> 7) & 3) * 8 + ((e >> 2) & 7), slot (e & 3) | ((e >> 5) & 3) << 2. The
// Hadamard butterflies over e bits 0, 1, 5, 6 are in-thread, bits 2-4 and 7-8
// are lane shuffles and bit 9 goes through shared memory. A 128-wide group is
// eight consecutive lanes, and each lane's 16 values are 16 consecutive bytes
// of the permuted int8 output.
template <typename scalar_t, int BLOCK>
__global__ void act_quant_kernel(const scalar_t* __restrict__ x,
                                 const float* __restrict__ signs,
                                 int8_t* __restrict__ xq, float* __restrict__ xs,
                                 float* __restrict__ xsx, const int K,
                                 const int64_t x_stride) {
  constexpr int kWaves = BLOCK / 512;
  __shared__ float exchange[kWaves > 1 ? BLOCK : 1];
  const int m = blockIdx.y;
  const int base = blockIdx.x * BLOCK;
  const int t = threadIdx.x;
  const int lane = t & 31;
  const int wave = t >> 5;
  const int e0 = (wave << 9) | ((lane >> 3) << 7) | ((lane & 7) << 2);
  const scalar_t* xr = x + m * x_stride + base;
  const float* sr = signs + base;
  float v[16];
  struct alignas(4 * sizeof(scalar_t)) Vec4 {
    scalar_t data[4];
  };
#pragma unroll
  for (int hi = 0; hi < 4; ++hi) {
    const int e = e0 + (hi << 5);
    const Vec4 xv = *reinterpret_cast<const Vec4*>(xr + e);
    const float4 sv = *reinterpret_cast<const float4*>(sr + e);
    v[hi * 4 + 0] = static_cast<float>(xv.data[0]) * sv.x;
    v[hi * 4 + 1] = static_cast<float>(xv.data[1]) * sv.y;
    v[hi * 4 + 2] = static_cast<float>(xv.data[2]) * sv.z;
    v[hi * 4 + 3] = static_cast<float>(xv.data[3]) * sv.w;
  }
  // In-thread butterflies (slot bits 0-3 = e bits 0, 1, 5, 6).
#pragma unroll
  for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      if (!(j & h)) {
        const float a = v[j];
        const float b = v[j | h];
        v[j] = a + b;
        v[j | h] = a - b;
      }
    }
  }
  // Lane butterflies (lane bits 0-4 = e bits 2-4, 7-8).
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const bool upper = lane & d;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float other = shfl_xor(v[j], d);
      v[j] = upper ? other - v[j] : v[j] + other;
    }
  }
  if constexpr (kWaves > 1) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      exchange[t * 16 + j] = v[j];
    }
    __syncthreads();
    const int partner = t ^ 32;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const float other = exchange[partner * 16 + j];
      v[j] = wave ? other - v[j] : v[j] + other;
    }
  }
  const float norm = rsqrtf(static_cast<float>(BLOCK));
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    v[j] *= norm;
    amax = fmaxf(amax, fabsf(v[j]));
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) {
    amax = fmaxf(amax, shfl_xor(amax, o));
  }
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int sum = 0;
  union {
    int4 vec;
    int8_t bytes[16];
  } out;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int q = __float2int_rn(v[j] * inv);
    sum += q;
    // Slot j = lo | hi << 2 holds w = (lane & 7) * 4 + lo and b = hi.
    out.bytes[(j & 3) * 4 + (j >> 2)] = static_cast<int8_t>(q);
  }
#pragma unroll
  for (int o = 1; o < 8; o <<= 1) {
    sum += shfl_xor(sum, o);
  }
  const int group = (wave << 2) | (lane >> 3);
  *reinterpret_cast<int4*>(xq + static_cast<int64_t>(m) * K + base +
                           group * kGroup + (lane & 7) * 16) = out.vec;
  if ((lane & 7) == 0) {
    const int gi = m * (K / kGroup) + base / kGroup + group;
    const float scale = amax / 127.f;
    xs[gi] = scale;
    xsx[gi] = scale * static_cast<float>(sum);
  }
}

// Each 32-lane wave computes R output rows. A lane reads 16 weight bytes (64
// weights spread over the four groups of a chunk) of every row per step, so
// the activation loads are shared by the R rows.
template <typename scalar_t, int MB, int R, int W>
__global__ void gemv_kernel(const int4* __restrict__ q,
                            const half* __restrict__ scales,
                            const int8_t* __restrict__ xq,
                            const float* __restrict__ xs,
                            const float* __restrict__ xsx,
                            scalar_t* __restrict__ y, const int M, const int N,
                            const int K) {
  // W waves of a block share each group of R rows, splitting K between them.
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int sub = wave % W;
  const int n0 = (blockIdx.x * (kGemvWaves / W) + wave / W) * R;
  if (W == 1 && n0 >= N) {
    return;
  }
  const int vecs_per_row = n0 < N ? K / 64 : 0;
  const int groups = K / kGroup;
  const int4* qrow[R];
  const half* srow[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int64_t n = n0 + r < N ? n0 + r : n0;
    qrow[r] = q + n * vecs_per_row;
    srow[r] = scales + n * groups;
  }
  float acc[R][MB];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      acc[r][m] = 0.f;
    }
  }
  for (int j = sub * 32 + lane; j < vecs_per_row; j += 32 * W) {
    const int chunk = j >> 3;
    const int w0 = (j & 7) * 4;
    const bool first = (j & 7) == 0;
    int4 codes[R][4];
    float ws[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int4 wv = qrow[r][j];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int sh = 2 * i;
        codes[r][i] = make_int4((wv.x >> sh) & kCodeMask, (wv.y >> sh) & kCodeMask,
                                (wv.z >> sh) & kCodeMask, (wv.w >> sh) & kCodeMask);
      }
      const uint2 sv = *reinterpret_cast<const uint2*>(srow[r] + chunk * 4);
      const half* sh = reinterpret_cast<const half*>(&sv);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        ws[r][i] = __half2float(sh[i]);
      }
    }
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      if (m < M) {
        const int8_t* xrow = xq + static_cast<int64_t>(m) * K + chunk * kChunk;
        const float4 xs4 =
            *reinterpret_cast<const float4*>(xs + m * groups + chunk * 4);
        float4 corr = make_float4(0.f, 0.f, 0.f, 0.f);
        if (first) {
          corr = *reinterpret_cast<const float4*>(xsx + m * groups + chunk * 4);
        }
        const float xsv[4] = {xs4.x, xs4.y, xs4.z, xs4.w};
        const float cv[4] = {corr.x, corr.y, corr.z, corr.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int4 xv =
              *reinterpret_cast<const int4*>(xrow + i * kGroup + w0 * 4);
#pragma unroll
          for (int r = 0; r < R; ++r) {
            int s = dot4_codes(codes[r][i].x, xv.x, 0);
            s = dot4_codes(codes[r][i].y, xv.y, s);
            s = dot4_codes(codes[r][i].z, xv.z, s);
            s = dot4_codes(codes[r][i].w, xv.w, s);
            acc[r][m] =
                fmaf(ws[r][i], fmaf(xsv[i], static_cast<float>(s), -cv[i]),
                     acc[r][m]);
          }
        }
      }
    }
  }
  __shared__ float partial[W > 1 ? kGemvWaves * R * MB : 1];
#pragma unroll
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int m = 0; m < MB; ++m) {
      float v = acc[r][m];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        v += shfl_xor(v, o);
      }
      if constexpr (W == 1) {
        if (lane == 0 && m < M && n0 + r < N) {
          y[static_cast<int64_t>(m) * N + n0 + r] = static_cast<scalar_t>(v);
        }
      } else if (lane == 0) {
        partial[(wave * R + r) * MB + m] = v;
      }
    }
  }
  if constexpr (W > 1) {
    __syncthreads();
    if (sub == 0 && lane < R * MB) {
      const int r = lane / MB;
      const int m = lane % MB;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < W; ++w) {
        v += partial[((wave + w) * R + r) * MB + m];
      }
      if (m < M && n0 + r < N) {
        y[static_cast<int64_t>(m) * N + n0 + r] = static_cast<scalar_t>(v);
      }
    }
  }
}


#if defined(USE_ROCM) && (defined(__gfx1200__) || defined(__gfx1201__))
  #define TERNARY_HAS_WMMA_GFX12 1
#endif

typedef int v2i_t __attribute__((ext_vector_type(2)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

// Codes {0, 1, 2} in byte lanes -> signed values {-1, 0, 1} without borrows.
__device__ __forceinline__ int signed_codes(int v) {
  return ((v | 0x80808080) - 0x01010101) ^ 0x80808080;
}

// gfx12 WMMA operands (wave32, 16x16x16 int8): lane l holds row/column l % 16
// and k = (l / 16) * 8 + [0, 8); the accumulator holds column l % 16 and rows
// (l / 16) * 8 + [0, 8). Within a 128-wide group the k order is free, so step
// st of lane half h covers group bytes h * 64 + st * 8 + [0, 8).
__device__ __forceinline__ v8i_t wmma_iu8(v2i_t a, v2i_t b, v8i_t c) {
#ifdef TERNARY_HAS_WMMA_GFX12
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c,
                                                          false);
#else
  __builtin_trap();
  return c;
#endif
}

// Prefill GEMM. A WAVES_M x WAVES_N grid of waves computes a
// (WAVES_M * WM * 16) x (WAVES_N * WN * 16) tile; activations of one group and
// the weight bytes of one chunk are staged in shared memory, and the next
// group is prefetched into registers while the current one is computed.
template <typename scalar_t, int WAVES_M, int WAVES_N, int WM, int WN>
__global__ void __launch_bounds__(32 * WAVES_M* WAVES_N)
    wmma_gemm_kernel(const uint8_t* __restrict__ q,
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
  const int groups = K / kGroup;
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
                           xq + static_cast<int64_t>(m) * K + g * kGroup +
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
    v8i_t c[WM][WN];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
      for (int b = 0; b < WN; ++b) c[a][b] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
    const int sh = 2 * i;
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      v2i_t av[WM];
      v2i_t bv[WN];
#pragma unroll
      for (int a = 0; a < WM; ++a) {
        av[a] = *reinterpret_cast<const v2i_t*>(
            a_smem + (wm * WM * 16 + a * 16 + col) * PAD + half_id * 64 +
            st * 8);
      }
#pragma unroll
      for (int b = 0; b < WN; ++b) {
        const v2i_t raw = *reinterpret_cast<const v2i_t*>(
            b_smem + (wn * WN * 16 + b * 16 + col) * PAD + half_id * 64 +
            st * 8);
        bv[b] = v2i_t{signed_codes((raw.x >> sh) & kCodeMask),
                      signed_codes((raw.y >> sh) & kCodeMask)};
      }
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WN; ++b) c[a][b] = wmma_iu8(av[a], bv[b], c[a][b]);
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
          y[static_cast<int64_t>(m) * N + n] = static_cast<scalar_t>(acc[a][b][r]);
        }
      }
    }
}

// Small-M GEMM: each wave owns TN 16x16 output tiles (sharing the activation
// loads) and reads activations and weight bytes straight from global memory,
// with the weights of the next PF chunks in flight.
template <typename scalar_t, int WAVES, int TN, int PF>
__global__ void __launch_bounds__(32 * WAVES)
    wmma_small_kernel(const uint8_t* __restrict__ q,
                      const half* __restrict__ scales,
                      const int8_t* __restrict__ xq,
                      const float* __restrict__ xs, scalar_t* __restrict__ y,
                      const int M, const int N, const int K) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int num_m = (M + 15) / 16;
  const int m0 = (blockIdx.x % num_m) * 16;
  const int n0 = ((blockIdx.x / num_m) * WAVES + wave) * 16 * TN;
  if (n0 >= N) {
    return;
  }
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int groups = K / kGroup;
  const int m = m0 + col;
  const bool m_valid = m < M;
  const int8_t* xrow = xq + static_cast<int64_t>(m_valid ? m : 0) * K;
  const uint8_t* qrow[TN];
  const half* srow[TN];
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const int n = n0 + t * 16 + col < N ? n0 + t * 16 + col : n0;
    qrow[t] = q + static_cast<int64_t>(n) * (K / 4) + half_id * 64;
    srow[t] = scales + static_cast<int64_t>(n) * groups;
  }
  float acc[TN][8];
#pragma unroll
  for (int t = 0; t < TN; ++t)
#pragma unroll
    for (int r = 0; r < 8; ++r) acc[t][r] = 0.f;
  const int chunks = K / kChunk;
  auto load = [&](int4 (&dst)[TN][4], int chunk) {
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u)
        dst[t][u] = chunk < chunks
                        ? reinterpret_cast<const int4*>(qrow[t] + chunk * 128)[u]
                        : make_int4(0, 0, 0, 0);
  };
  int4 n1[TN][4];
  int4 n2[TN][4];
  load(n1, 0);
  if constexpr (PF > 1) {
    load(n2, 1);
  }
  for (int chunk = 0; chunk < chunks; ++chunk) {
    int4 wb[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int u = 0; u < 4; ++u) wb[t][u] = n1[t][u];
    if constexpr (PF > 1) {
#pragma unroll
      for (int t = 0; t < TN; ++t)
#pragma unroll
        for (int u = 0; u < 4; ++u) n1[t][u] = n2[t][u];
      load(n2, chunk + 2);
    } else {
      load(n1, chunk + 1);
    }
    // The four groups are independent accumulation chains; interleaving them
    // hides the WMMA latency.
    v8i_t c[TN][4];
#pragma unroll
    for (int t = 0; t < TN; ++t)
#pragma unroll
      for (int i = 0; i < 4; ++i) c[t][i] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      v2i_t av[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        av[i] = m_valid ? *reinterpret_cast<const v2i_t*>(
                              xrow + chunk * kChunk + i * kGroup + half_id * 64 +
                              st * 8)
                        : v2i_t{0, 0};
      }
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const int4 w4 = wb[t][st >> 1];
        const int lo = (st & 1) ? w4.z : w4.x;
        const int hi = (st & 1) ? w4.w : w4.y;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int sh = 2 * i;
          const v2i_t bv = v2i_t{signed_codes((lo >> sh) & kCodeMask),
                                 signed_codes((hi >> sh) & kCodeMask)};
          c[t][i] = wmma_iu8(av[i], bv, c[t][i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int g = chunk * 4 + i;
      float xsv[8];
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int mr = m0 + half_id * 8 + r;
        xsv[r] = mr < M ? xs[static_cast<int64_t>(mr) * groups + g] : 0.f;
      }
#pragma unroll
      for (int t = 0; t < TN; ++t) {
        const float wsv = __half2float(srow[t][g]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          acc[t][r] = fmaf(wsv * xsv[r], static_cast<float>(c[t][i][r]), acc[t][r]);
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < TN; ++t) {
    const int n = n0 + t * 16 + col;
    if (n < N) {
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int mr = m0 + half_id * 8 + r;
        if (mr < M) {
          y[static_cast<int64_t>(mr) * N + n] = static_cast<scalar_t>(acc[t][r]);
        }
      }
    }
  }
}


// Small-M GEMM (M <= 16) with weight slabs staged in shared memory: a block of
// four waves owns 16 output columns; per iteration it streams four chunks
// (512 contiguous bytes) of each of the 16 weight rows, like the GEMV, and each
// wave computes one chunk with WMMA. Partial sums are reduced at the end.
template <typename scalar_t>
__global__ void __launch_bounds__(128)
    wmma_slab_kernel(const uint8_t* __restrict__ q,
                     const half* __restrict__ scales,
                     const int8_t* __restrict__ xq,
                     const float* __restrict__ xs, scalar_t* __restrict__ y,
                     const int M, const int N, const int K) {
  constexpr int kRowBytes = 4 * 128;
  constexpr int kPad = kRowBytes + 16;
  __shared__ uint8_t slab[16 * kPad];
  __shared__ float red[4][8][32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int n0 = blockIdx.x * 16;
  const int groups = K / kGroup;
  const int chunks = K / kChunk;
  const int64_t row_bytes = K / 4;
  const int m = col;
  const bool m_valid = m < M;
  const int8_t* xrow = xq + static_cast<int64_t>(m_valid ? m : 0) * K;
  const int n_lane = n0 + col < N ? n0 + col : n0;
  const half* srow = scales + static_cast<int64_t>(n_lane) * groups;
  float acc[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = 0.f;
  for (int c0 = 0; c0 < chunks; c0 += 4) {
    const int nch = min(4, chunks - c0);
    // 16 rows x 512 bytes: wave w loads rows w, w+4, w+8, w+12.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = wave + 4 * j;
      const int n = n0 + r < N ? n0 + r : n0;
      int4 v = make_int4(0, 0, 0, 0);
      if (lane * 16 < nch * 128) {
        v = *reinterpret_cast<const int4*>(q + n * row_bytes + c0 * 128 + lane * 16);
      }
      *reinterpret_cast<int4*>(slab + r * kPad + lane * 16) = v;
    }
    __syncthreads();
    if (wave < nch) {
      const int chunk = c0 + wave;
      v8i_t c[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
      const uint8_t* brow = slab + col * kPad + wave * 128 + half_id * 64;
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        const v2i_t raw = *reinterpret_cast<const v2i_t*>(brow + st * 8);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const v2i_t av =
              m_valid ? *reinterpret_cast<const v2i_t*>(
                            xrow + chunk * kChunk + i * kGroup + half_id * 64 + st * 8)
                      : v2i_t{0, 0};
          const int sh = 2 * i;
          const v2i_t bv = v2i_t{signed_codes((raw.x >> sh) & kCodeMask),
                                 signed_codes((raw.y >> sh) & kCodeMask)};
          c[i] = wmma_iu8(av, bv, c[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int g = chunk * 4 + i;
        const float wsv = __half2float(srow[g]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int mr = half_id * 8 + r;
          const float xsv = mr < M ? xs[static_cast<int64_t>(mr) * groups + g] : 0.f;
          acc[r] = fmaf(wsv * xsv, static_cast<float>(c[i][r]), acc[r]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) red[wave][r][lane] = acc[r];
  __syncthreads();
  if (wave == 0) {
    const int n = n0 + col;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const float v = red[0][r][lane] + red[1][r][lane] + red[2][r][lane] + red[3][r][lane];
      const int mr = half_id * 8 + r;
      if (mr < M && n < N) {
        y[static_cast<int64_t>(mr) * N + n] = static_cast<scalar_t>(v);
      }
    }
  }
}


// Split-K variant of the small-M GEMM for narrow outputs: `splits` blocks share
// each 16 x 64 output tile, each over a K range. They store fp32 partials, and
// the last block to finish (tracked by a self-resetting counter) reduces them.
template <typename scalar_t>
__global__ void __launch_bounds__(128)
    wmma_splitk_kernel(const uint8_t* __restrict__ q,
                       const half* __restrict__ scales,
                       const int8_t* __restrict__ xq,
                       const float* __restrict__ xs, scalar_t* __restrict__ y,
                       float* __restrict__ partial, int* __restrict__ counters,
                       const int M, const int N, const int K, const int splits) {
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int tile = blockIdx.x / splits;
  const int split = blockIdx.x % splits;
  const int num_m = (M + 15) / 16;
  const int m0 = (tile % num_m) * 16;
  const int n0 = ((tile / num_m) * 4 + wave) * 16;
  const int col = lane & 15;
  const int half_id = lane >> 4;
  const int groups = K / kGroup;
  const int chunks = K / kChunk;
  const int per = (chunks + splits - 1) / splits;
  const int c_begin = split * per;
  const int c_end = min(chunks, c_begin + per);
  const bool active = n0 < N;
  const int n = active && n0 + col < N ? n0 + col : 0;
  const int m = m0 + col;
  const bool m_valid = m < M;
  const uint8_t* qrow = q + static_cast<int64_t>(n) * (K / 4) + half_id * 64;
  const int8_t* xrow = xq + static_cast<int64_t>(m_valid ? m : 0) * K;
  const half* srow = scales + static_cast<int64_t>(n) * groups;
  float acc[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) acc[r] = 0.f;
  if (active) {
    int4 next[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      next[u] = c_begin < c_end
                    ? reinterpret_cast<const int4*>(qrow + c_begin * 128)[u]
                    : make_int4(0, 0, 0, 0);
    }
    for (int chunk = c_begin; chunk < c_end; ++chunk) {
      int4 wb[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) wb[u] = next[u];
      if (chunk + 1 < c_end) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          next[u] = reinterpret_cast<const int4*>(qrow + (chunk + 1) * 128)[u];
        }
      }
      v8i_t c[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) c[i] = v8i_t{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        const int4 w4 = wb[st >> 1];
        const int lo = (st & 1) ? w4.z : w4.x;
        const int hi = (st & 1) ? w4.w : w4.y;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const v2i_t av =
              m_valid ? *reinterpret_cast<const v2i_t*>(
                            xrow + chunk * kChunk + i * kGroup + half_id * 64 + st * 8)
                      : v2i_t{0, 0};
          const int sh = 2 * i;
          const v2i_t bv = v2i_t{signed_codes((lo >> sh) & kCodeMask),
                                 signed_codes((hi >> sh) & kCodeMask)};
          c[i] = wmma_iu8(av, bv, c[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int g = chunk * 4 + i;
        const float wsv = __half2float(srow[g]);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int mr = m0 + half_id * 8 + r;
          const float xsv = mr < M ? xs[static_cast<int64_t>(mr) * groups + g] : 0.f;
          acc[r] = fmaf(wsv * xsv, static_cast<float>(c[i][r]), acc[r]);
        }
      }
    }
    if (n0 + col < N) {
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int mr = m0 + half_id * 8 + r;
        if (mr < M) {
          partial[(static_cast<int64_t>(split) * M + mr) * N + n0 + col] = acc[r];
        }
      }
    }
  }
  __threadfence();
  __syncthreads();
  __shared__ int is_last;
  if (threadIdx.x == 0) {
    is_last = atomicAdd(&counters[tile], 1) == splits - 1;
  }
  __syncthreads();
  if (!is_last) {
    return;
  }
  __threadfence();
  if (active && n0 + col < N) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int mr = m0 + half_id * 8 + r;
      if (mr < M) {
        float v = 0.f;
        for (int sp = 0; sp < splits; ++sp) {
          v += __builtin_nontemporal_load(
              &partial[(static_cast<int64_t>(sp) * M + mr) * N + n0 + col]);
        }
        y[static_cast<int64_t>(mr) * N + n0 + col] = static_cast<scalar_t>(v);
      }
    }
  }
  if (threadIdx.x == 0) {
    counters[tile] = 0;
  }
}

template <typename scalar_t, int BLOCK>
void launch_act_quant(const dim3 grid, cudaStream_t stream, const Tensor& X,
                      const Tensor& signs, const Tensor& xq, const Tensor& xs,
                      const Tensor& xsx, const int K) {
  act_quant_kernel<scalar_t, BLOCK><<<grid, BLOCK / 16, 0, stream>>>(
      (const scalar_t*)X.data_ptr(), (const float*)signs.data_ptr(),
      (int8_t*)xq.data_ptr(), (float*)xs.data_ptr(), (float*)xsx.data_ptr(), K,
      X.strides()[0]);
}

template <typename scalar_t, int MB, int R, int W = 1>
void launch_gemv(cudaStream_t stream, const Tensor& Q, const Tensor& S,
                 const Tensor& xq, const Tensor& xs, const Tensor& xsx,
                 const Tensor& Y, const int M, const int N, const int K) {
  constexpr int kRowsPerBlock = kGemvWaves / W * R;
  const dim3 grid((N + kRowsPerBlock - 1) / kRowsPerBlock);
  const dim3 block(32 * kGemvWaves);
  gemv_kernel<scalar_t, MB, R, W><<<grid, block, 0, stream>>>(
      (const int4*)Q.data_ptr(), (const half*)S.data_ptr(),
      (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(),
      (const float*)xsx.data_ptr(), (scalar_t*)Y.data_ptr(), M, N, K);
}

template <typename scalar_t>
void dispatch_act_quant(const int64_t block_size, const dim3 grid,
                        cudaStream_t stream, const Tensor& X,
                        const Tensor& signs, const Tensor& xq, const Tensor& xs,
                        const Tensor& xsx, const int K) {
  if (block_size == 512) {
    launch_act_quant<scalar_t, 512>(grid, stream, X, signs, xq, xs, xsx, K);
  } else {
    STD_TORCH_CHECK(block_size == 1024,
                    "ternary_act_quant: unsupported block size");
    launch_act_quant<scalar_t, 1024>(grid, stream, X, signs, xq, xs, xsx, K);
  }
}

template <typename scalar_t, int MB, int R>
void dispatch_gemv_split(cudaStream_t stream, const Tensor& Q, const Tensor& S,
                         const Tensor& xq, const Tensor& xs, const Tensor& xsx,
                         const Tensor& Y, const int M, const int N, const int K) {
  static const int forced = [] {
    const char* v = std::getenv("TERNARY_GEMV_SPLIT");
    return v ? std::atoi(v) : 0;
  }();
  // Splitting K across waves did not pay off on gfx1201 (kept for tuning).
  int split = 1;
  if (forced) {
    split = forced;
  }
  if constexpr (R * MB <= 32) {
    if (split == 4) {
      launch_gemv<scalar_t, MB, R, 4>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
      return;
    }
    if (split == 2) {
      launch_gemv<scalar_t, MB, R, 2>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
      return;
    }
  }
  launch_gemv<scalar_t, MB, R, 1>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
}

template <typename scalar_t, int MB>
void dispatch_gemv_rows(cudaStream_t stream, const Tensor& Q, const Tensor& S,
                        const Tensor& xq, const Tensor& xs, const Tensor& xsx,
                        const Tensor& Y, const int M, const int N, const int K) {
  static const int forced = [] {
    const char* v = std::getenv("TERNARY_GEMV_ROWS");
    return v ? std::atoi(v) : 0;
  }();
  // Rows per wave: more rows share activation loads, fewer keep more waves
  // in flight (tuned on gfx1201).
  int rows = MB == 1 ? 1 : (MB == 2 ? 2 : (MB == 4 ? (N >= 65536 ? 4 : 2) : 4));
  if (forced) {
    rows = forced;
  }
  if (rows == 1) {
    dispatch_gemv_split<scalar_t, MB, 1>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else if (rows == 2) {
    dispatch_gemv_split<scalar_t, MB, 2>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else if (rows == 8) {
    dispatch_gemv_split<scalar_t, MB, 8>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else {
    dispatch_gemv_split<scalar_t, MB, 4>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  }
}

template <typename scalar_t>
void dispatch_gemv(cudaStream_t stream, const Tensor& Q, const Tensor& S,
                   const Tensor& xq, const Tensor& xs, const Tensor& xsx,
                   const Tensor& Y, const int M, const int N, const int K) {
  if (M <= 1) {
    dispatch_gemv_rows<scalar_t, 1>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else if (M <= 2) {
    dispatch_gemv_rows<scalar_t, 2>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else if (M <= 4) {
    dispatch_gemv_rows<scalar_t, 4>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  } else {
    dispatch_gemv_rows<scalar_t, 8>(stream, Q, S, xq, xs, xsx, Y, M, N, K);
  }
}


template <typename scalar_t>
void launch_wmma_gemm(cudaStream_t stream, const Tensor& Q, const Tensor& S,
                      const Tensor& xq, const Tensor& xs, const Tensor& Y,
                      const Tensor& counters, const int M, const int N,
                      const int K) {
  static const int forced_split = [] {
    const char* v = std::getenv("TERNARY_WMMA_SPLIT");
    return v ? std::atoi(v) : 0;
  }();
  // Split-K did not pay off on gfx1201 (kept for tuning via env).
  int splits = 1;
  if (forced_split) {
    splits = M <= 16 ? forced_split : 1;
  }
  const int tiles = ((M + 15) / 16) * ((N + 63) / 64);
  if (splits > 1 && tiles <= counters.numel()) {
    Tensor partial = torch::stable::new_empty(
        xq, {static_cast<int64_t>(splits) * M * N}, ScalarType::Float);
    wmma_splitk_kernel<scalar_t><<<tiles * splits, 128, 0, stream>>>(
        (const uint8_t*)Q.data_ptr(), (const half*)S.data_ptr(),
        (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(),
        (scalar_t*)Y.data_ptr(), (float*)partial.data_ptr(),
        (int*)counters.data_ptr(), M, N, K, splits);
    return;
  }
  if (M <= 64) {
    static const int variant = [] {
      const char* v = std::getenv("TERNARY_WMMA_SMALL");
      return v ? std::atoi(v) : 0;
    }();
    const auto run = [&](auto waves_c, auto tn_c, auto pf_c) {
      constexpr int W = decltype(waves_c)::value;
      constexpr int TN = decltype(tn_c)::value;
      constexpr int PF = decltype(pf_c)::value;
      const int cols = W * 16 * TN;
      const dim3 grid(((M + 15) / 16) * ((N + cols - 1) / cols));
      wmma_small_kernel<scalar_t, W, TN, PF><<<grid, 32 * W, 0, stream>>>(
          (const uint8_t*)Q.data_ptr(), (const half*)S.data_ptr(),
          (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(),
          (scalar_t*)Y.data_ptr(), M, N, K);
    };
    using I1 = std::integral_constant<int, 1>;
    using I2 = std::integral_constant<int, 2>;
    using I4 = std::integral_constant<int, 4>;
    using I8 = std::integral_constant<int, 8>;
    switch (variant) {
      case 1: run(I4{}, I1{}, I2{}); break;
      case 2: run(I4{}, I2{}, I1{}); break;
      case 3: run(I8{}, I1{}, I1{}); break;
      case 4: run(I4{}, I2{}, I2{}); break;
      case 5: run(I8{}, I1{}, I2{}); break;
      case 6: run(I2{}, I1{}, I2{}); break;
      case 7:
        if (M <= 16) {
          wmma_slab_kernel<scalar_t><<<(N + 15) / 16, 128, 0, stream>>>(
              (const uint8_t*)Q.data_ptr(), (const half*)S.data_ptr(),
              (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(),
              (scalar_t*)Y.data_ptr(), M, N, K);
        } else {
          run(I4{}, I1{}, I1{});
        }
        break;
      default: run(I4{}, I1{}, I1{}); break;
    }
  } else {
    constexpr int kWavesM = 2, kWavesN = 4, kTilesM = 4, kTilesN = 2;
    constexpr int kBM = kWavesM * kTilesM * 16, kBN = kWavesN * kTilesN * 16;
    const dim3 grid(((M + kBM - 1) / kBM) * ((N + kBN - 1) / kBN));
    wmma_gemm_kernel<scalar_t, kWavesM, kWavesN, kTilesM, kTilesN>
        <<<grid, 32 * kWavesM * kWavesN, 0, stream>>>(
            (const uint8_t*)Q.data_ptr(), (const half*)S.data_ptr(),
            (const int8_t*)xq.data_ptr(), (const float*)xs.data_ptr(),
            (scalar_t*)Y.data_ptr(), M, N, K);
  }
}

}  // namespace

void ternary_act_quant(Tensor X, Tensor signs, int64_t block_size, Tensor xq,
                       Tensor xs, Tensor xsx) {
  const int64_t M = X.sizes()[0];
  const int64_t K = X.sizes()[1];
  STD_TORCH_CHECK(X.strides()[1] == 1 && X.strides()[0] % 4 == 0,
                  "ternary_act_quant: X rows must be contiguous and 4-aligned");
  STD_TORCH_CHECK(K % kChunk == 0 && K % block_size == 0,
                  "ternary_act_quant: bad input width");
  STD_TORCH_CHECK(signs.scalar_type() == ScalarType::Float,
                  "ternary_act_quant: signs must be float32");
  if (M == 0) {
    return;
  }
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  cudaStream_t stream = current_stream(device_idx);
  const dim3 grid(K / block_size, M);
  const int k = static_cast<int>(K);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ternary_act_quant", [&] {
    dispatch_act_quant<scalar_t>(block_size, grid, stream, X, signs, xq, xs, xsx,
                                 k);
  });
}

Tensor ternary_gemv(Tensor Q, Tensor S, Tensor xq, Tensor xs, Tensor xsx,
                    ScalarType dtype) {
  const int64_t M = xq.sizes()[0];
  const int64_t K = xq.sizes()[1];
  const int64_t N = Q.sizes()[0];
  STD_TORCH_CHECK(Q.sizes()[1] * 4 == K, "ternary_gemv: weight width mismatch");
  STD_TORCH_CHECK(M <= 8, "ternary_gemv: at most 8 rows");
  const int32_t device_idx = xq.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_empty(xq, {M, N}, dtype);
  if (M == 0) {
    return Y;
  }
  cudaStream_t stream = current_stream(device_idx);
  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  const int k = static_cast<int>(K);
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ternary_gemv", [&] {
    dispatch_gemv<scalar_t>(stream, Q, S, xq, xs, xsx, Y, m, n, k);
  });
  return Y;
}

Tensor ternary_wmma_gemm(Tensor Q, Tensor S, Tensor xq, Tensor xs,
                         Tensor counters, ScalarType dtype) {
  const int64_t M = xq.sizes()[0];
  const int64_t K = xq.sizes()[1];
  const int64_t N = Q.sizes()[0];
  STD_TORCH_CHECK(Q.sizes()[1] * 4 == K, "ternary_wmma_gemm: width mismatch");
  const int32_t device_idx = xq.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_empty(xq, {M, N}, dtype);
  if (M == 0) {
    return Y;
  }
  cudaStream_t stream = current_stream(device_idx);
  const int m = static_cast<int>(M);
  const int n = static_cast<int>(N);
  const int k = static_cast<int>(K);
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "ternary_wmma_gemm", [&] {
    launch_wmma_gemm<scalar_t>(stream, Q, S, xq, xs, Y, counters, m, n, k);
  });
  return Y;
}
