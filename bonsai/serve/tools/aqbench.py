import torch, sys
sys.path.insert(0, "/tmp/claude-1000/-home-volta-Desktop-Perso-vllm/18551204-76a8-4fa3-9198-c3f30d4df18c/scratchpad/tools")
import vllm_gguf_plugin.ops  # noqa
from tgemv import bench
for K in (5120, 6144, 17408):
    r = []
    for M in (1, 4, 512):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        sg = torch.ones(K, device="cuda")
        xq = torch.empty(M, K, device="cuda", dtype=torch.int8)
        xs = torch.empty(M, K // 128, device="cuda"); xsx = torch.empty_like(xs)
        t = bench(lambda: torch.ops._C_gguf.ternary_act_quant(x, sg, 1024, xq, xs, xsx), 200)
        r.append(f"M={M}: {t*1000:.1f}us")
    print(f"K={K}: " + " | ".join(r))
