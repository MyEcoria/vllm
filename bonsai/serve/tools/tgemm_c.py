import torch, triton, triton.language as tl
from tgemv import bench
from tgemm_b import pack_b, perm_x


@triton.jit
def tgemm_c_kernel(xq_ptr, xs_ptr, q_ptr, ws_ptr, y_ptr, M, N, K,
                   BM: tl.constexpr, BN: tl.constexpr, GROUP: tl.constexpr, ROWSCALE: tl.constexpr):
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
    rk = tl.arange(0, 128)
    mmask = rm < M
    nmask = rn < N
    G = K // 128
    a_ptrs = xq_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = q_ptr + rn[None, :] * (K // 4) + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(0, G):
        a = tl.load(a_ptrs + g * 128, mask=mmask[:, None], other=0)
        byts = tl.load(w_ptrs + (g // 4) * 128, mask=nmask[None, :], other=0)
        wt = (((byts.to(tl.int32) >> ((g % 4) * 2)) & 3) - 1).to(tl.int8)
        part = tl.dot(a, wt)
        ws = tl.load(ws_ptr + rn * G + g, mask=nmask, other=0.0).to(tl.float32)
        if ROWSCALE:
            acc += part.to(tl.float32) * ws[None, :]
        else:
            xs = tl.load(xs_ptr + rm * G + g, mask=mmask, other=0.0)
            acc += part.to(tl.float32) * xs[:, None] * ws[None, :]
    if ROWSCALE:
        acc = acc * tl.load(xs_ptr + rm * G, mask=mmask, other=0.0)[:, None]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


def tgemm(xq, xs, q, ws, BM=128, BN=128, warps=8, stages=1, group=8, rowscale=False):
    M, K = xq.shape
    N = q.shape[0]
    y = torch.empty(M, N, device=xq.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    tgemm_c_kernel[grid](xq, xs, q, ws, y, M, N, K, BM=BM, BN=BN, GROUP=group, ROWSCALE=rowscale,
                         num_warps=warps, num_stages=stages)
    return y


if __name__ == "__main__":
    dev = "cuda"
    torch.manual_seed(0)
    for M in (512, 2048):
        for N, K in [(18432, 5120), (34816, 5120), (5120, 17408), (5120, 6144)]:
            codes = torch.randint(0, 3, (N, K), device=dev, dtype=torch.int64)
            q = pack_b(codes)
            ws = (torch.rand(N, K // 128, device=dev) * 0.02 + 0.001).half()
            xq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
            xs = torch.rand(M, K // 128, device=dev) * 0.01
            wf = ((codes.float() - 1) * ws.float().repeat_interleave(128, 1))
            ref = (xq.float() * xs.repeat_interleave(128, 1)) @ wf.T
            xqp = perm_x(xq).contiguous()
            del codes, wf
            flops = 2 * M * N * K
            res = []
            for BM, BN, wp, st, gr, rs in [(128, 128, 8, 1, 8, False), (128, 128, 8, 2, 8, False),
                                           (256, 128, 8, 1, 8, False), (128, 256, 8, 1, 8, False),
                                           (128, 128, 4, 1, 8, False), (64, 128, 4, 1, 8, False),
                                           (128, 128, 8, 1, 4, False), (128, 128, 8, 1, 16, False),
                                           (128, 128, 8, 1, 8, True), (256, 128, 8, 1, 8, True)]:
                try:
                    y = tgemm(xqp, xs, q, ws, BM, BN, wp, st, gr, rs)
                    err = ((y.float() - ref).norm() / ref.norm()).item()
                    t = bench(lambda: tgemm(xqp, xs, q, ws, BM, BN, wp, st, gr, rs), 20)
                    res.append((flops / t / 1e9, BM, BN, wp, st, gr, rs, err))
                except Exception as e:
                    print("fail", BM, BN, wp, st, type(e).__name__, str(e)[:60])
            res.sort(reverse=True)
            nonrs = [r for r in res if not r[6]]
            print(f"M={M} N={N} K={K}: best {nonrs[0][0]:.0f} TOPS {nonrs[0][1:7]} err={nonrs[0][7]:.1e} | "
                  f"{[(round(r[0]), r[1:7]) for r in res[:5]]}", flush=True)
            del q, ws, ref
            torch.cuda.empty_cache()
