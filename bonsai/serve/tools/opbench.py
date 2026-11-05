import torch, sys
sys.path.insert(0, "/tmp/claude-1000/-home-volta-Desktop-Perso-vllm/18551204-76a8-4fa3-9198-c3f30d4df18c/scratchpad/tools")
import vllm_gguf_plugin.ops  # noqa  loads _C_gguf
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
        def f():
            i = it[0] = (it[0] + 1) % copies
            torch.ops._C_gguf.ternary_gemv(qs[i], ss[i], xq, xs, xsx, torch.bfloat16)
        t = bench(f, 60)
        out.append(f"M={M}: {nb / t / 1e6:.0f} GB/s ({t*1000:.0f}us)")
    print(f"N={N} K={K}: " + " | ".join(out), flush=True)
    del qs, ss; torch.cuda.empty_cache()
