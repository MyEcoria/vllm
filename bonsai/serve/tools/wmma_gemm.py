"""Prototype gfx12 WMMA int8 GEMM on byte-plane ternary weights.

Weights q uint8 [N, K/4] (byte w*4+b of a 128-byte chunk holds codes of group i
at bits 2i), scales s fp16 [N, K/128]. Activations xq int8 [M, K] in the same
permuted order, xs / xsx fp32 [M, K/128]. Within a 128-wide group the k order is
free, so WMMA step s (0..7) of lane half h covers group bytes 64h + 8s .. +8.
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

CPP = """
#include <torch/extension.h>
void wmma_gemm(torch::Tensor q, torch::Tensor s, torch::Tensor xq, torch::Tensor xs,
               torch::Tensor xsx, torch::Tensor y, int64_t cfg);
"""

HIP = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPStream.h>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

template <int WM, int WN, int WAVES_M, int WAVES_N>
__global__ void __launch_bounds__(32 * WAVES_M * WAVES_N)
wmma_kernel(const uint8_t* __restrict__ q, const __half* __restrict__ s,
            const int8_t* __restrict__ xq, const float* __restrict__ xs,
            const float* __restrict__ xsx, __hip_bfloat16* __restrict__ y,
            int M, int N, int K) {
  constexpr int BM = WAVES_M * WM * 16;
  constexpr int BN = WAVES_N * WN * 16;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  // grouped launch order for L2 reuse of weights
  const int num_m = (M + BM - 1) / BM;
  const int pid_m = blockIdx.x % num_m;
  const int pid_n = blockIdx.x / num_m;
  const int m_base = pid_m * BM + wm * WM * 16;
  const int n_base = pid_n * BN + wn * WN * 16;
  const int col = lane & 15;
  const int half = lane >> 4;
  const int G = K / 128;
  const int KB = K / 4;

  float acc[WM][WN][8];
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[a][b][r] = 0.f;

  const uint8_t* qrow[WN];
#pragma unroll
  for (int b = 0; b < WN; ++b) {
    int n = n_base + b * 16 + col;
    qrow[b] = q + (int64_t)(n < N ? n : 0) * KB;
  }
  const int8_t* xrow[WM];
  bool mvalid[WM];
#pragma unroll
  for (int a = 0; a < WM; ++a) {
    int m = m_base + a * 16 + col;
    mvalid[a] = m < M;
    xrow[a] = xq + (int64_t)(m < M ? m : 0) * K;
  }

  for (int c = 0; c < K / 512; ++c) {
    int4 wb[WN][4];
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const int4* p = reinterpret_cast<const int4*>(qrow[b] + c * 128 + half * 64);
#pragma unroll
      for (int t = 0; t < 4; ++t) wb[b][t] = p[t];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v8i C[WM][WN];
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WN; ++b) C[a][b] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
      const int sh = 2 * i;
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        v2i A[WM];
#pragma unroll
        for (int a = 0; a < WM; ++a) {
          v2i v = *reinterpret_cast<const v2i*>(xrow[a] + c * 512 + i * 128 + half * 64 + st * 8);
          A[a] = mvalid[a] ? v : v2i{0, 0};
        }
#pragma unroll
        for (int b = 0; b < WN; ++b) {
          const int4 w4 = wb[b][st >> 1];
          const int lo = (st & 1) ? w4.z : w4.x;
          const int hi = (st & 1) ? w4.w : w4.y;
          v2i B = v2i{(lo >> sh) & 0x03030303, (hi >> sh) & 0x03030303};
#pragma unroll
          for (int a = 0; a < WM; ++a)
            C[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, A[a], false, B, C[a][b], false);
        }
      }
      const int g = c * 4 + i;
#pragma unroll
      for (int b = 0; b < WN; ++b) {
        const int n = n_base + b * 16 + col;
        const float wsv = n < N ? __half2float(s[(int64_t)n * G + g]) : 0.f;
#pragma unroll
        for (int a = 0; a < WM; ++a) {
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const int m = m_base + a * 16 + half * 8 + r;
            const int mm = m < M ? m : 0;
            const float xsv = xs[(int64_t)mm * G + g];
            const float xcv = xsx[(int64_t)mm * G + g];
            acc[a][b][r] = fmaf(wsv, fmaf(xsv, (float)C[a][b][r], -xcv), acc[a][b][r]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const int n = n_base + b * 16 + col;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = m_base + a * 16 + half * 8 + r;
        if (m < M && n < N) y[(int64_t)m * N + n] = __float2bfloat16(acc[a][b][r]);
      }
    }
}


__device__ __forceinline__ int signed_codes(int v) {
  return ((v | 0x80808080) - 0x01010101) ^ 0x80808080;
}

template <int WAVES_M, int WAVES_N, int WM, int WN>
__global__ void __launch_bounds__(32 * WAVES_M * WAVES_N)
lds_kernel(const uint8_t* __restrict__ q, const __half* __restrict__ s,
           const int8_t* __restrict__ xq, const float* __restrict__ xs,
           __hip_bfloat16* __restrict__ y, int M, int N, int K) {
  constexpr int BM = WAVES_M * WM * 16;
  constexpr int BN = WAVES_N * WN * 16;
  constexpr int NT = 32 * WAVES_M * WAVES_N;
  constexpr int PAD = 136;
  __shared__ int8_t As[BM * PAD];
  __shared__ uint8_t Bs[BN * PAD];
  __shared__ float xs_s[BM * 4];
  __shared__ float ws_s[BN * 4];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  const int num_m = (M + BM - 1) / BM;
  const int pid_m = blockIdx.x % num_m;
  const int pid_n = blockIdx.x / num_m;
  const int m0 = pid_m * BM, n0 = pid_n * BN;
  const int col = lane & 15, half = lane >> 4;
  const int G = K / 128, KB = K / 4;

  float acc[WM][WN][8];
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b)
#pragma unroll
      for (int r = 0; r < 8; ++r) acc[a][b][r] = 0.f;

  for (int c = 0; c < K / 512; ++c) {
    // B chunk: BN rows x 128 bytes.
    for (int idx = tid; idx < BN * 8; idx += NT) {
      const int r = idx >> 3, part = idx & 7;
      const int n = n0 + r;
      int4 v = make_int4(0, 0, 0, 0);
      if (n < N) v = *reinterpret_cast<const int4*>(q + (int64_t)n * KB + c * 128 + part * 16);
      *reinterpret_cast<int4*>(Bs + r * PAD + part * 16) = v;
    }
    for (int idx = tid; idx < BN * 4; idx += NT) {
      const int r = idx >> 2, i = idx & 3, n = n0 + r;
      ws_s[idx] = n < N ? __half2float(s[(int64_t)n * G + c * 4 + i]) : 0.f;
    }
    for (int idx = tid; idx < BM * 4; idx += NT) {
      const int r = idx >> 2, i = idx & 3, m = m0 + r;
      xs_s[idx] = m < M ? xs[(int64_t)m * G + c * 4 + i] : 0.f;
    }
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
      for (int idx = tid; idx < BM * 8; idx += NT) {
        const int r = idx >> 3, part = idx & 7;
        const int m = m0 + r;
        int4 v = make_int4(0, 0, 0, 0);
        if (m < M) v = *reinterpret_cast<const int4*>(xq + (int64_t)m * K + c * 512 + i * 128 + part * 16);
        *reinterpret_cast<int4*>(As + r * PAD + part * 16) = v;
      }
      __syncthreads();
      v8i C[WM][WN];
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WN; ++b) C[a][b] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
      const int sh = 2 * i;
#pragma unroll
      for (int st = 0; st < 8; ++st) {
        v2i A[WM], B[WN];
#pragma unroll
        for (int a = 0; a < WM; ++a)
          A[a] = *reinterpret_cast<const v2i*>(As + (wm * WM * 16 + a * 16 + col) * PAD + half * 64 + st * 8);
#pragma unroll
        for (int b = 0; b < WN; ++b) {
          const v2i raw = *reinterpret_cast<const v2i*>(Bs + (wn * WN * 16 + b * 16 + col) * PAD + half * 64 + st * 8);
          B[b] = v2i{signed_codes((raw.x >> sh) & 0x03030303), signed_codes((raw.y >> sh) & 0x03030303)};
        }
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
          for (int b = 0; b < WN; ++b)
            C[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, A[a], true, B[b], C[a][b], false);
      }
#pragma unroll
      for (int b = 0; b < WN; ++b) {
        const float wsv = ws_s[(wn * WN * 16 + b * 16 + col) * 4 + i];
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
          for (int r = 0; r < 8; ++r) {
            const float f = wsv * xs_s[(wm * WM * 16 + a * 16 + half * 8 + r) * 4 + i];
            acc[a][b][r] = fmaf(f, (float)C[a][b][r], acc[a][b][r]);
          }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int a = 0; a < WM; ++a)
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const int n = n0 + wn * WN * 16 + b * 16 + col;
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = m0 + wm * WM * 16 + a * 16 + half * 8 + r;
        if (m < M && n < N) y[(int64_t)m * N + n] = __float2bfloat16(acc[a][b][r]);
      }
    }
}

template <int WAVES_M, int WAVES_N, int WM, int WN>
void launch_lds(torch::Tensor q, torch::Tensor s, torch::Tensor xq, torch::Tensor xs, torch::Tensor y) {
  int M = xq.size(0), K = xq.size(1), N = q.size(0);
  constexpr int BM = WAVES_M * WM * 16, BN = WAVES_N * WN * 16;
  dim3 grid(((M + BM - 1) / BM) * ((N + BN - 1) / BN));
  hipLaunchKernelGGL((lds_kernel<WAVES_M, WAVES_N, WM, WN>), grid, dim3(32 * WAVES_M * WAVES_N), 0,
                     c10::hip::getCurrentHIPStream().stream(),
                     q.data_ptr<uint8_t>(), (const __half*)s.data_ptr(), xq.data_ptr<int8_t>(),
                     xs.data_ptr<float>(), (__hip_bfloat16*)y.data_ptr(), M, N, K);
}


template <int WAVES_M, int WAVES_N, int WM, int WN>
__global__ void __launch_bounds__(32 * WAVES_M * WAVES_N)
pf_kernel(const uint8_t* __restrict__ q, const __half* __restrict__ s,
          const int8_t* __restrict__ xq, const float* __restrict__ xs,
          __hip_bfloat16* __restrict__ y, int M, int N, int K) {
  constexpr int BM = WAVES_M * WM * 16;
  constexpr int BN = WAVES_N * WN * 16;
  constexpr int NT = 32 * WAVES_M * WAVES_N;
  constexpr int PAD = 136;
  constexpr int A_PER = (BM * 8 + NT - 1) / NT;
  constexpr int B_PER = (BN * 8 + NT - 1) / NT;
  __shared__ int8_t As[BM * PAD];
  __shared__ uint8_t Bs[BN * PAD];
  __shared__ float xs_s[BM];
  __shared__ float ws_s[BN * 4];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  const int wm = wave / WAVES_N, wn = wave % WAVES_N;
  const int num_m = (M + BM - 1) / BM;
  const int pid_m = blockIdx.x % num_m;
  const int pid_n = blockIdx.x / num_m;
  const int m0 = pid_m * BM, n0 = pid_n * BN;
  const int col = lane & 15, half = lane >> 4;
  const int G = K / 128, KB = K / 4;

  int4 a_reg[A_PER], b_reg[B_PER];
  float xs_reg = 0.f;
  float ws_reg[(BN * 4 + NT - 1) / NT];
  auto load_a = [&](int g) {
#pragma unroll
    for (int j = 0; j < A_PER; ++j) {
      const int idx = tid + j * NT;
      const int r = idx >> 3, part = idx & 7, m = m0 + r;
      a_reg[j] = (idx < BM * 8 && m < M)
          ? *reinterpret_cast<const int4*>(xq + (int64_t)m * K + g * 128 + part * 16)
          : make_int4(0, 0, 0, 0);
    }
    if (tid < BM) {
      const int m = m0 + tid;
      xs_reg = m < M ? xs[(int64_t)m * G + g] : 0.f;
    }
  };
  auto load_b = [&](int c) {
#pragma unroll
    for (int j = 0; j < B_PER; ++j) {
      const int idx = tid + j * NT;
      const int r = idx >> 3, part = idx & 7, n = n0 + r;
      b_reg[j] = (idx < BN * 8 && n < N)
          ? *reinterpret_cast<const int4*>(q + (int64_t)n * KB + c * 128 + part * 16)
          : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < (BN * 4 + NT - 1) / NT; ++j) {
      const int idx = tid + j * NT;
      const int r = idx >> 2, i = idx & 3, n = n0 + r;
      ws_reg[j] = (idx < BN * 4 && n < N) ? __half2float(s[(int64_t)n * G + c * 4 + i]) : 0.f;
    }
  };
  auto store_a = [&]() {
#pragma unroll
    for (int j = 0; j < A_PER; ++j) {
      const int idx = tid + j * NT;
      if (idx < BM * 8) *reinterpret_cast<int4*>(As + (idx >> 3) * PAD + (idx & 7) * 16) = a_reg[j];
    }
    if (tid < BM) xs_s[tid] = xs_reg;
  };
  auto store_b = [&]() {
#pragma unroll
    for (int j = 0; j < B_PER; ++j) {
      const int idx = tid + j * NT;
      if (idx < BN * 8) *reinterpret_cast<int4*>(Bs + (idx >> 3) * PAD + (idx & 7) * 16) = b_reg[j];
    }
#pragma unroll
    for (int j = 0; j < (BN * 4 + NT - 1) / NT; ++j) {
      const int idx = tid + j * NT;
      if (idx < BN * 4) ws_s[idx] = ws_reg[j];
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
  for (int g = 0; g < G; ++g) {
    const int i = g & 3;
    if (g + 1 < G) {
      load_a(g + 1);
      if (((g + 1) & 3) == 0) load_b((g + 1) >> 2);
    }
    v8i C[WM][WN];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
      for (int b = 0; b < WN; ++b) C[a][b] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    const int sh = 2 * i;
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      v2i A[WM], B[WN];
#pragma unroll
      for (int a = 0; a < WM; ++a)
        A[a] = *reinterpret_cast<const v2i*>(As + (wm * WM * 16 + a * 16 + col) * PAD + half * 64 + st * 8);
#pragma unroll
      for (int b = 0; b < WN; ++b) {
        const v2i raw = *reinterpret_cast<const v2i*>(Bs + (wn * WN * 16 + b * 16 + col) * PAD + half * 64 + st * 8);
        B[b] = v2i{signed_codes((raw.x >> sh) & 0x03030303), signed_codes((raw.y >> sh) & 0x03030303)};
      }
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WN; ++b)
          C[a][b] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, A[a], true, B[b], C[a][b], false);
    }
#pragma unroll
    for (int b = 0; b < WN; ++b) {
      const float wsv = ws_s[(wn * WN * 16 + b * 16 + col) * 4 + i];
#pragma unroll
      for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const float f = wsv * xs_s[wm * WM * 16 + a * 16 + half * 8 + r];
          acc[a][b][r] = fmaf(f, (float)C[a][b][r], acc[a][b][r]);
        }
    }
    __syncthreads();
    if (g + 1 < G) {
      store_a();
      if (((g + 1) & 3) == 0) store_b();
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
        const int m = m0 + wm * WM * 16 + a * 16 + half * 8 + r;
        if (m < M && n < N) y[(int64_t)m * N + n] = __float2bfloat16(acc[a][b][r]);
      }
    }
}

template <int WAVES_M, int WAVES_N, int WM, int WN>
void launch_pf(torch::Tensor q, torch::Tensor s, torch::Tensor xq, torch::Tensor xs, torch::Tensor y) {
  int M = xq.size(0), K = xq.size(1), N = q.size(0);
  constexpr int BM = WAVES_M * WM * 16, BN = WAVES_N * WN * 16;
  dim3 grid(((M + BM - 1) / BM) * ((N + BN - 1) / BN));
  hipLaunchKernelGGL((pf_kernel<WAVES_M, WAVES_N, WM, WN>), grid, dim3(32 * WAVES_M * WAVES_N), 0,
                     c10::hip::getCurrentHIPStream().stream(),
                     q.data_ptr<uint8_t>(), (const __half*)s.data_ptr(), xq.data_ptr<int8_t>(),
                     xs.data_ptr<float>(), (__hip_bfloat16*)y.data_ptr(), M, N, K);
}

template <int WM, int WN, int WAVES_M, int WAVES_N>
void launch(torch::Tensor q, torch::Tensor s, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx, torch::Tensor y) {
  int M = xq.size(0), K = xq.size(1), N = q.size(0);
  constexpr int BM = WAVES_M * WM * 16, BN = WAVES_N * WN * 16;
  dim3 grid(((M + BM - 1) / BM) * ((N + BN - 1) / BN));
  hipLaunchKernelGGL((wmma_kernel<WM, WN, WAVES_M, WAVES_N>), grid, dim3(32 * WAVES_M * WAVES_N), 0,
                     c10::hip::getCurrentHIPStream().stream(),
                     q.data_ptr<uint8_t>(), (const __half*)s.data_ptr(), xq.data_ptr<int8_t>(),
                     xs.data_ptr<float>(), xsx.data_ptr<float>(), (__hip_bfloat16*)y.data_ptr(), M, N, K);
}

void wmma_gemm(torch::Tensor q, torch::Tensor s, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx,
               torch::Tensor y, int64_t cfg) {
  switch (cfg) {
    case 0: launch<2, 2, 2, 2>(q, s, xq, xs, xsx, y); break;
    case 1: launch<4, 2, 2, 4>(q, s, xq, xs, xsx, y); break;
    case 2: launch<2, 4, 4, 2>(q, s, xq, xs, xsx, y); break;
    case 3: launch<2, 2, 4, 4>(q, s, xq, xs, xsx, y); break;
    case 4: launch<4, 4, 2, 2>(q, s, xq, xs, xsx, y); break;
    case 5: launch<1, 4, 1, 4>(q, s, xq, xs, xsx, y); break;
    case 6: launch<1, 2, 1, 4>(q, s, xq, xs, xsx, y); break;
    case 7: launch<1, 1, 1, 4>(q, s, xq, xs, xsx, y); break;
    case 8: launch<2, 2, 2, 4>(q, s, xq, xs, xsx, y); break;
    case 10: launch_lds<2, 4, 4, 2>(q, s, xq, xs, y); break;
    case 11: launch_lds<2, 4, 2, 2>(q, s, xq, xs, y); break;
    case 12: launch_lds<4, 2, 2, 4>(q, s, xq, xs, y); break;
    case 13: launch_lds<2, 2, 4, 4>(q, s, xq, xs, y); break;
    case 14: launch_lds<4, 4, 2, 2>(q, s, xq, xs, y); break;
    case 15: launch_lds<2, 4, 2, 4>(q, s, xq, xs, y); break;
    case 20: launch_pf<4, 4, 2, 2>(q, s, xq, xs, y); break;
    case 21: launch_pf<2, 4, 4, 2>(q, s, xq, xs, y); break;
    case 22: launch_pf<2, 4, 2, 2>(q, s, xq, xs, y); break;
    case 23: launch_pf<4, 2, 2, 4>(q, s, xq, xs, y); break;
    case 24: launch_pf<2, 2, 4, 4>(q, s, xq, xs, y); break;
    case 25: launch_pf<4, 4, 2, 4>(q, s, xq, xs, y); break;
    case 26: launch_pf<4, 4, 4, 2>(q, s, xq, xs, y); break;
    default: TORCH_CHECK(false, "cfg");
  }
}
"""


def build():
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
    return load_inline(name="wmma_gemm_proto3", cpp_sources=CPP, cuda_sources=HIP,
                       functions=["wmma_gemm"], extra_cuda_cflags=["-O3", "--offload-arch=gfx1201"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import vllm_gguf_plugin.ops  # noqa: F401
    from vllm_gguf_plugin.triton.gemm.ternary_quant.packed import ternary_packed_gemm
    from tgemv import bench

    mod = build()
    dev = "cuda"
    torch.manual_seed(0)
    for M in [int(v) for v in os.environ.get('MS', '512,2048').split(',')]:
        for N, K in [(18432, 5120), (34816, 5120), (5120, 17408), (5120, 6144)]:
            q = torch.randint(0, 255, (N, K // 4), device=dev, dtype=torch.uint8)
            s = (torch.rand(N, K // 128, device=dev) * 0.02).half()
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xq = torch.empty(M, K, device=dev, dtype=torch.int8)
            xs = torch.empty(M, K // 128, device=dev)
            xsx = torch.empty_like(xs)
            torch.ops._C_gguf.ternary_act_quant(x, torch.ones(K, device=dev), 1024, xq, xs, xsx)
            # Triton reference uses (codes - 1) directly; same math.
            ref = ternary_packed_gemm(xq, xs, q, s, torch.bfloat16).float()
            flops = 2 * M * N * K
            tt = bench(lambda: ternary_packed_gemm(xq, xs, q, s, torch.bfloat16), 10)
            res = []
            for cfg in [int(c) for c in os.environ.get('CFGS', '6,7,10,11,12,13,14,15').split(',')]:
                y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                try:
                    mod.wmma_gemm(q, s, xq, xs, xsx, y, cfg)
                    err = ((y.float() - ref).norm() / ref.norm()).item()
                    t = bench(lambda: mod.wmma_gemm(q, s, xq, xs, xsx, y, cfg), 10)
                    res.append((flops / t / 1e9, cfg, err, t))
                except Exception as e:  # noqa: BLE001
                    print("fail", cfg, str(e)[:80])
            res.sort(reverse=True)
            nb = q.numel() + s.numel() * 2
            print(f"M={M} N={N} K={K}: triton {flops/tt/1e9:.0f} TOPS ({tt*1000:.0f}us) | wmma best "
                  f"{res[0][0]:.0f} TOPS cfg={res[0][1]} err={res[0][2]:.1e} t={res[0][3]*1000:.0f}us "
                  f"({nb/res[0][3]/1e6:.0f} GB/s) | others {[(round(r[0]), r[1]) for r in res[1:4]]}", flush=True)
