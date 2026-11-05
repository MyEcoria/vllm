"""Measure achievable matmul throughput on this GPU (torch + Triton fp16/int8)."""

import torch
import triton
import triton.language as tl


@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
       BK: tl.constexpr, GROUP: tl.constexpr, INT8: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    width = GROUP * nn
    gid = pid // width
    first_m = gid * GROUP
    gsz = min(nm - first_m, GROUP)
    pm = first_m + (pid % width) % gsz
    pn = (pid % width) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * K + rk[None, :]
    b = b_ptr + rn[None, :] * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.int32 if INT8 else tl.float32)
    for _ in range(0, K, BK):
        acc += tl.dot(tl.load(a), tl.load(b))
        a += BK
        b += BK
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16))


def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


M, N, K = 512, 17408, 5120
flops = 2 * M * N * K
a16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
b16 = torch.randn(N, K, device="cuda", dtype=torch.float16)
print("torch fp16", flops / bench(lambda: a16 @ b16.T) / 1e9, "TFLOPS")
a8 = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
b8 = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
try:
    print("torch _int_mm", flops / bench(lambda: torch._int_mm(a8, b8.T)) / 1e9, "TOPS")
except Exception as e:  # noqa: BLE001
    print("torch _int_mm unavailable", type(e).__name__, str(e)[:100])
c = torch.empty(M, N, device="cuda", dtype=torch.float16)
for INT8, (a, b) in ((False, (a16, b16)), (True, (a8, b8))):
    best = None
    for BM, BN, BK, w, st in [(128, 128, 32, 4, 2), (128, 128, 64, 4, 2), (128, 128, 64, 8, 2),
                               (256, 128, 64, 8, 2), (128, 256, 64, 8, 2), (64, 128, 64, 4, 2),
                               (128, 128, 128, 8, 2), (256, 256, 64, 8, 2), (128, 64, 128, 4, 2)]:
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        try:
            t = bench(lambda: mm[grid](a, b, c, M, N, K, BM, BN, BK, 8, INT8,
                                       num_warps=w, num_stages=st))
        except Exception as e:  # noqa: BLE001
            print("fail", BM, BN, BK, w, type(e).__name__, str(e)[:80])
            continue
        r = flops / t / 1e9
        print("int8" if INT8 else "fp16", BM, BN, BK, w, f"{r:.1f}")
        best = max(best or 0, r)
    print("BEST", "int8" if INT8 else "fp16", best)
