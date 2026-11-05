import torch, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import vllm_gguf_plugin.ops  # noqa
from tgemv import bench
dev = "cuda"
cnt = torch.zeros(1 << 16, dtype=torch.int32, device=dev)
shapes = [(18432, 5120), (5120, 6144), (34816, 5120), (5120, 17408), (248320, 5120)]
weights = {(16+6+48+16)*0: 0}
out = []; tot_b = 0; tot_t = 0
# per-layer weight mix approx: per 64 layers mixing shapes; report each + weighted
for N, K in shapes:
    nb = N * K // 4 + N * K // 64
    copies = max(2, int(400e6 // nb) + 1)
    qs = [torch.randint(0, 255, (N, K // 4), device=dev, dtype=torch.uint8) for _ in range(copies)]
    ss = [(torch.rand(N, K // 128, device=dev) * 0.02).half() for _ in range(copies)]
    xq = torch.randint(-127, 127, (8, K), device=dev, dtype=torch.int8)
    xs = torch.rand(8, K // 128, device=dev)
    it = [0]
    def w():
        i = it[0] = (it[0] + 1) % copies
        torch.ops._C_gguf.ternary_wmma_gemm(qs[i], ss[i], xq, xs, cnt, torch.bfloat16)
    t = bench(w, 40)
    out.append(f"{N}x{K}:{nb/t/1e6:.0f}")
    del qs, ss; torch.cuda.empty_cache()
print(os.environ.get("TERNARY_WMMA_SPLIT", "auto"), " ".join(out), flush=True)
