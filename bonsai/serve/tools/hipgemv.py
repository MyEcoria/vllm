"""Prototype HIP decode path: Hadamard+int8 activation quant and dot4 ternary GEMV.

Weight layout (dot4): codes int32 [N, K/16]; word j holds weights 16j..16j+15,
weight 16j + 4i + b at bits 8b + 2i (value = code - 1). Scales fp16 [N, K/128].
Activations: xq int8 [M, K]; per 64 group xs (scale) and xsx (= xs * sum(xq)).
"""

import os

import torch
from torch.utils.cpp_extension import load_inline

CPP = r"""
#include <torch/extension.h>
void hadamard_quant(torch::Tensor x, torch::Tensor signs, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx);
void ternary_gemv(torch::Tensor codes, torch::Tensor ws, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx, torch::Tensor y, int64_t rows_per_block);
"""

HIP = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <c10/hip/HIPStream.h>

__global__ void hadamard_quant_kernel(const __hip_bfloat16* __restrict__ x, const float* __restrict__ signs,
                                      int8_t* __restrict__ xq, float* __restrict__ xs, float* __restrict__ xsx,
                                      int K, long sx) {
  __shared__ float buf[1024];
  const int m = blockIdx.y, c = blockIdx.x, t = threadIdx.x;
  const __hip_bfloat16* xr = x + m * sx + c * 1024;
  #pragma unroll
  for (int j = t; j < 1024; j += 256) buf[j] = __bfloat162float(xr[j]) * signs[c * 1024 + j];
  __syncthreads();
  #pragma unroll
  for (int h = 1; h < 1024; h <<= 1) {
    #pragma unroll
    for (int b = t; b < 512; b += 256) {
      int i = (b / h) * 2 * h + (b % h);
      float a = buf[i], d = buf[i + h];
      buf[i] = a + d;
      buf[i + h] = a - d;
    }
    __syncthreads();
  }
  // 16 groups of 64; 16 threads per group, 4 values each.
  const int g = t >> 4, l = t & 15;
  float v[4];
  float amax = 0.f;
  #pragma unroll
  for (int e = 0; e < 4; ++e) { v[e] = buf[g * 64 + l * 4 + e] * (1.0f / 32.0f); amax = fmaxf(amax, fabsf(v[e])); }
  #pragma unroll
  for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 16));
  const float scale = amax / 127.f;
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  int q[4];
  int s = 0;
  #pragma unroll
  for (int e = 0; e < 4; ++e) { q[e] = __float2int_rn(v[e] * inv); s += q[e]; }
  #pragma unroll
  for (int o = 8; o > 0; o >>= 1) s += __shfl_xor(s, o, 16);
  char4 packed = make_char4(q[0], q[1], q[2], q[3]);
  *reinterpret_cast<char4*>(xq + (long)m * K + c * 1024 + g * 64 + l * 4) = packed;
  if (l == 0) {
    const int gi = m * (K / 64) + c * 16 + g;
    xs[gi] = scale;
    xsx[gi] = scale * (float)s;
  }
}

__device__ __forceinline__ int dot4(int a, int b, int c) {
  return __builtin_amdgcn_sudot4(false, a, true, b, c, false);
}

template <int MB>
__global__ void ternary_gemv_kernel(const int4* __restrict__ codes, const __half* __restrict__ ws,
                                    const int8_t* __restrict__ xq, const float* __restrict__ xs,
                                    const float* __restrict__ xsx, __hip_bfloat16* __restrict__ y,
                                    int M, int N, int K, int rows) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int n = blockIdx.x * rows + wid;
  if (n >= N) return;
  const int n64 = K >> 6;
  const int4* wrow = codes + (long)n * n64;
  const __half* wsr = ws + (long)n * (K >> 7);
  float acc[MB];
  #pragma unroll
  for (int m = 0; m < MB; ++m) acc[m] = 0.f;
  for (int g = lane; g < n64; g += 32) {
    const int4 w = wrow[g];
    const float wsc = __half2float(wsr[g >> 1]);
    #pragma unroll
    for (int m = 0; m < MB; ++m) {
      if (m < M) {
        const int4* xp = reinterpret_cast<const int4*>(xq + (long)m * K + g * 64);
        const int4 a = xp[0], b = xp[1], c = xp[2], d = xp[3];
        int s = 0;
        s = dot4(w.x & 0x03030303, a.x, s); s = dot4((w.x >> 2) & 0x03030303, a.y, s);
        s = dot4((w.x >> 4) & 0x03030303, a.z, s); s = dot4((w.x >> 6) & 0x03030303, a.w, s);
        s = dot4(w.y & 0x03030303, b.x, s); s = dot4((w.y >> 2) & 0x03030303, b.y, s);
        s = dot4((w.y >> 4) & 0x03030303, b.z, s); s = dot4((w.y >> 6) & 0x03030303, b.w, s);
        s = dot4(w.z & 0x03030303, c.x, s); s = dot4((w.z >> 2) & 0x03030303, c.y, s);
        s = dot4((w.z >> 4) & 0x03030303, c.z, s); s = dot4((w.z >> 6) & 0x03030303, c.w, s);
        s = dot4(w.w & 0x03030303, d.x, s); s = dot4((w.w >> 2) & 0x03030303, d.y, s);
        s = dot4((w.w >> 4) & 0x03030303, d.z, s); s = dot4((w.w >> 6) & 0x03030303, d.w, s);
        const int gi = m * n64 + g;
        acc[m] += wsc * (xs[gi] * (float)s - xsx[gi]);
      }
    }
  }
  #pragma unroll
  for (int m = 0; m < MB; ++m) {
    float v = acc[m];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if (lane == 0 && m < M) y[(long)m * N + n] = __float2bfloat16(v);
  }
}

void hadamard_quant(torch::Tensor x, torch::Tensor signs, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx) {
  const int M = x.size(0), K = x.size(1);
  dim3 grid(K / 1024, M);
  hipLaunchKernelGGL(hadamard_quant_kernel, grid, dim3(256), 0, c10::hip::getCurrentHIPStream().stream(),
                     (const __hip_bfloat16*)x.data_ptr(), signs.data_ptr<float>(), xq.data_ptr<int8_t>(),
                     xs.data_ptr<float>(), xsx.data_ptr<float>(), K, x.stride(0));
}

void ternary_gemv(torch::Tensor codes, torch::Tensor ws, torch::Tensor xq, torch::Tensor xs, torch::Tensor xsx,
                  torch::Tensor y, int64_t rows) {
  const int M = xq.size(0), K = xq.size(1), N = codes.size(0);
  dim3 grid((N + rows - 1) / rows);
  dim3 block(32 * rows);
  auto stream = c10::hip::getCurrentHIPStream().stream();
#define LAUNCH(MBV) hipLaunchKernelGGL(ternary_gemv_kernel<MBV>, grid, block, 0, stream, \
      (const int4*)codes.data_ptr(), (const __half*)ws.data_ptr(), xq.data_ptr<int8_t>(), \
      xs.data_ptr<float>(), xsx.data_ptr<float>(), (__hip_bfloat16*)y.data_ptr(), M, N, K, (int)rows)
  if (M <= 1) LAUNCH(1); else if (M <= 2) LAUNCH(2); else if (M <= 4) LAUNCH(4); else LAUNCH(8);
}
"""


def build():
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
    return load_inline(
        name="tern_hip_proto",
        cpp_sources=CPP,
        cuda_sources=HIP,
        functions=["hadamard_quant", "ternary_gemv"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1201"],
        verbose=False,
    )


def sylvester(n, dev):
    h = torch.ones(1, 1, device=dev)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


def pack_dot4(codes):
    """codes int64 [N, K] in {0,1,2} -> int32 [N, K/16] dot4 layout."""
    N, K = codes.shape
    c = codes.view(N, K // 16, 4, 4)  # [n, j, i, b] weight 16j+4i+b
    shifts = (8 * torch.arange(4, device=codes.device).view(1, 1, 1, 4)
              + 2 * torch.arange(4, device=codes.device).view(1, 1, 4, 1))
    w = (c << shifts).sum((2, 3))
    return torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)


if __name__ == "__main__":
    from tgemv import bench

    mod = build()
    dev = "cuda"
    torch.manual_seed(0)
    for N, K in [(18432, 5120), (5120, 6144), (34816, 5120), (5120, 17408), (248320, 5120)]:
        codes = torch.randint(0, 3, (N, K), device=dev, dtype=torch.int64)
        q = pack_dot4(codes)
        s = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.001).half()
        signs = (torch.randint(0, 2, (K,), device=dev).float() * 2 - 1)
        for M in (1, 4):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            H = sylvester(1024, dev)
            xr = ((x.float() * signs).view(M, K // 1024, 1024) @ H).view(M, K)
            ref = torch.empty(M, N, device=dev)
            for r0 in range(0, N, 32768):
                w = (codes[r0:r0 + 32768].float() - 1) * s[r0:r0 + 32768].float().repeat_interleave(128, 1)
                ref[:, r0:r0 + 32768] = xr @ w.T
            xq = torch.empty(M, K, device=dev, dtype=torch.int8)
            xs = torch.empty(M, K // 64, device=dev)
            xsx = torch.empty_like(xs)
            y = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            mod.hadamard_quant(x, signs, xq, xs, xsx)
            nbytes = q.numel() * 4 + s.numel() * 2
            res = []
            for rows in (1, 2, 4, 8, 16):
                mod.ternary_gemv(q, s, xq, xs, xsx, y, rows)
                err = ((y.float() - ref).norm() / ref.norm()).item()
                t = bench(lambda: mod.ternary_gemv(q, s, xq, xs, xsx, y, rows))
                res.append((nbytes / t / 1e6, rows, err, t))
            res.sort(reverse=True)
            tq = bench(lambda: mod.hadamard_quant(x, signs, xq, xs, xsx))
            b = res[0]
            print(f"N={N} K={K} M={M}: {b[0]:.0f} GB/s rows={b[1]} relerr={b[2]:.2e} t={b[3]*1000:.1f}us"
                  f" | quant {tq*1000:.1f}us | others {[(round(r[0]), r[1]) for r in res[1:3]]}", flush=True)
        del codes, q, s
        torch.cuda.empty_cache()
