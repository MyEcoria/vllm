"""Prototype ternary GEMV (decode) on repacked weights.

Layout: codes int32 [N, K/16] (16 x 2-bit codes, weight k = 16*w + i at bits
2i, value = code - 1), scales fp16 [N, K/128].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def tgemv_kernel(x_ptr, q_ptr, s_ptr, y_ptr, M, N, K, stride_xm, stride_ym,
                 MB: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    nmask = rn < N
    rm = tl.arange(0, MB)
    mmask = rm < M
    KW = K // 16
    G = K // 128
    rw = tl.arange(0, BK // 16)
    ri = tl.arange(0, 16)
    rk = tl.arange(0, BK)
    rg = tl.arange(0, BK // 128)
    acc = tl.zeros((MB, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        words = tl.load(q_ptr + rn[:, None] * KW + (k0 // 16 + rw)[None, :],
                        mask=nmask[:, None], other=0)
        codes = (words[:, :, None] >> (2 * ri)[None, None, :]) & 3
        w = tl.reshape(codes, (BN, BK)).to(tl.float32) - 1.0
        x = tl.load(x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :],
                    mask=mmask[:, None], other=0.0).to(tl.float32)
        prod = x[:, None, :] * w[None, :, :]
        part = tl.sum(tl.reshape(prod, (MB, BN, BK // 128, 128)), axis=3)
        s = tl.load(s_ptr + rn[:, None] * G + (k0 // 128 + rg)[None, :],
                    mask=nmask[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(part * s[None, :, :], axis=2)
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc,
             mask=mmask[:, None] & nmask[None, :])


def tgemv(x, q, s, BN=8, BK=1024, warps=4):
    M, K = x.shape
    N = q.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    MB = max(1, triton.next_power_of_2(M))
    tgemv_kernel[(triton.cdiv(N, BN),)](x, q, s, y, M, N, K, x.stride(0), y.stride(0),
                                        MB=MB, BN=BN, BK=BK, num_warps=warps)
    return y


def make(N, K, dev="cuda"):
    codes = torch.randint(0, 3, (N, K), device=dev, dtype=torch.int32)
    shifts = (2 * torch.arange(16, device=dev, dtype=torch.int32))
    q = (codes.view(N, K // 16, 16) << shifts).sum(-1, dtype=torch.int64)
    q = q.to(torch.int64).where(q < 2**31, q - 2**32).to(torch.int32)
    s = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.001).half()
    w = (codes.float() - 1) * s.float().repeat_interleave(128, 1)
    return q, s, w


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


if __name__ == "__main__":
    big = torch.empty(2 * 1024**3, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(big)
    t = bench(lambda: dst.copy_(big), 10)
    print(f"copy BW {2 * big.numel() / t / 1e6:.0f} GB/s")
    rd = big.view(torch.int32)
    t = bench(lambda: rd.sum(), 10)
    print(f"read(sum) BW {big.numel() / t / 1e6:.0f} GB/s")
    del big, dst
    shapes = [(10240 + 2 * 1024 + 6144, 5120), (5120, 6144), (34816, 5120), (5120, 17408)]
    for N, K in shapes:
        q, s, w = make(N, K)
        for M in (1, 4):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ w.T
            nbytes = q.numel() * 4 + s.numel() * 2
            best = None
            for BN, BK, wp in [(4, 1024, 4), (8, 1024, 4), (8, 512, 4), (16, 512, 4),
                               (16, 1024, 8), (8, 2048, 8), (4, 2048, 4), (2, 2048, 4),
                               (32, 256, 4)]:
                y = tgemv(x, q, s, BN, BK, wp)
                err = ((y.float() - ref).abs().max() / ref.abs().max()).item()
                tt = bench(lambda: tgemv(x, q, s, BN, BK, wp))
                bw = nbytes / tt / 1e6
                if best is None or bw > best[0]:
                    best = (bw, BN, BK, wp, err, tt)
            print(f"N={N} K={K} M={M}: best {best[0]:.0f} GB/s cfg BN={best[1]} BK={best[2]} "
                  f"w={best[3]} err={best[4]:.1e} t={best[5]*1000:.1f}us")
