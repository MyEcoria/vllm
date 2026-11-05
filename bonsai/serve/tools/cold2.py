import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import vllm_gguf_plugin.ops  # noqa
from tgemv import bench
dev = "cuda"
for N, K in [(18432, 5120), (5120, 6144), (34816, 5120), (5120, 17408), (248320, 5120)]:
    nb = N * K // 4 + N * K // 64
    copies = max(2, int(400e6 // nb) + 1)
    qs = [torch.randint(0, 255, (N, K // 4), device=dev, dtype=torch.uint8) for _ in range(copies)]
    ss = [(torch.rand(N, K // 128, device=dev) * 0.02).half() for _ in range(copies)]
    out = []
    for M in (1, 2, 4, 8):
        xq = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
        xs = torch.rand(M, K // 128, device=dev); xsx = torch.rand_like(xs)
        it = [0]
        def g():
            i = it[0] = (it[0] + 1) % copies
            torch.ops._C_gguf.ternary_gemv(qs[i], ss[i], xq, xs, xsx, torch.bfloat16)
        def w():
            i = it[0] = (it[0] + 1) % copies
            torch.ops._C_gguf.ternary_wmma_gemm(qs[i], ss[i], xq, xs, torch.bfloat16)
        out.append(f"M={M}: dot4 {nb/bench(g,40)/1e6:.0f} wmma {nb/bench(w,40)/1e6:.0f}")
    print(f"N={N} K={K}: " + " | ".join(out), flush=True)
    del qs, ss; torch.cuda.empty_cache()
