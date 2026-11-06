# SPDX-License-Identifier: Apache-2.0
"""One-process micro-bench of the prefill hot ops of Ternary-Bonsai-2-27B.

Times, for T new tokens at KV depth D: int4 attention prefill (one full-attention layer),
the 4 ternary GEMM shapes, the act-quant op and the GDN chunked prefill (one layer), then
projects ms/token for the whole model (16 attn + 48 GDN layers, 64 layers of GEMMs).
"""
import argparse
import json
import math

import torch

NH, NKV, HD = 24, 4, 256
SHAPES = [(18432, 5120), (5120, 6144), (34816, 5120), (5120, 17408)]
ACT_K = 5120
N_ATTN, N_GDN, N_LAYERS = 16, 48, 64
DEFAULT_BLOCK = 1840
WEIGHT_BYTES = 400e6


def bench(fn, iters):
    """Mean ms per call after warm-up."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def build_cache(depth, block_size):
    """Fill a K4/V3 paged cache with random K/V (identity block table) and return it."""
    from vllm_gguf_plugin import bonsai_int4_attn as m

    e = m.ext()
    nblk = (depth + block_size - 1) // block_size
    cache = torch.zeros(nblk, NKV, block_size, m.CONTENT_BYTES, dtype=torch.uint8, device="cuda")
    slots = torch.arange(depth, dtype=torch.int64, device="cuda")
    for s in range(0, depth, 8192):
        k = torch.randn(min(8192, depth - s), NKV, HD, device="cuda", dtype=torch.float16)
        e.kv_write(k, torch.randn_like(k), slots[s:s + k.shape[0]], cache)
    bt = torch.arange(nblk, dtype=torch.int32, device="cuda")[None]
    return e, m, cache, bt


def bench_attn(t, depth, block_size, iters):
    """Time q_prep + attn_prefill for t query tokens ending at depth; returns ms, TFLOPS."""
    e, m, cache, bt = build_cache(depth, block_size)
    q = torch.randn(t, NH, HD, device="cuda", dtype=torch.float16)
    q8, qs, qsum = m._S.q_bufs(t, NH, q.device)
    out = torch.zeros(t, NH, HD, dtype=torch.float16, device="cuda")
    sl = torch.tensor([depth], dtype=torch.int32, device="cuda")
    qsl = torch.tensor([0, t], dtype=torch.int32, device="cuda")
    rows = [v for r0 in range(0, t * 6, m.PF_ROWS) for v in (0, r0)]
    work = torch.tensor(rows, dtype=torch.int32).view(-1, 2).cuda()

    def f():
        e.q_prep(q, q8, qs, qsum, 1.0 / math.sqrt(HD))
        e.attn_prefill(q8, qs, qsum, cache, bt, sl, qsl, work, out)

    ms = bench(f, iters)
    flops = 4 * NH * HD * (t * (depth - t) + t * (t + 1) / 2)
    return ms, flops / ms / 1e9


def bench_gemm(t, iters):
    """Time the 4 ternary GEMM shapes at M=t with rotating weight copies; returns dict."""
    res = {}
    for n, k in SHAPES:
        copies = max(2, int(WEIGHT_BYTES // (n * k // 4)) + 1)
        qs = [torch.randint(0, 255, (n, k // 4), device="cuda", dtype=torch.uint8) for _ in range(copies)]
        ss = [(torch.rand(n, k // 128, device="cuda") * 0.02).half() for _ in range(copies)]
        xq = torch.randint(-127, 127, (t, k), device="cuda", dtype=torch.int8)
        xs = torch.rand(t, k // 128, device="cuda")
        cnt = torch.zeros(4096, dtype=torch.int32, device="cuda")
        it = [0]

        def f():
            i = it[0] = (it[0] + 1) % copies
            torch.ops._C_gguf.ternary_wmma_gemm(qs[i], ss[i], xq, xs, cnt, torch.bfloat16)

        ms = bench(f, iters)
        res[f"{n}x{k}"] = {"ms": ms, "tops": 2 * t * n * k / ms / 1e9}
        del qs, ss
        torch.cuda.empty_cache()
    return res


def bench_act_quant(t, iters):
    """Time ternary_act_quant at M=t, K=ACT_K."""
    x = torch.randn(t, ACT_K, device="cuda", dtype=torch.bfloat16)
    signs = torch.ones(ACT_K, dtype=torch.float32, device="cuda")
    xq = torch.empty(t, ACT_K, dtype=torch.int8, device="cuda")
    xs = torch.empty(t, ACT_K // 128, dtype=torch.float32, device="cuda")
    xsx = torch.empty_like(xs)
    return bench(lambda: torch.ops._C_gguf.ternary_act_quant(x, signs, 1024, xq, xs, xsx), iters)


def bench_gdn(t, hk, hv, dk, dv, iters):
    """Time fla chunk_gated_delta_rule for one sequence of t tokens."""
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule

    bf = torch.bfloat16
    q = torch.randn(1, t, hk, dk, device="cuda", dtype=bf)
    k = torch.randn(1, t, hk, dk, device="cuda", dtype=bf)
    v = torch.randn(1, t, hv, dv, device="cuda", dtype=bf)
    g = -torch.rand(1, t, hv, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, t, hv, device="cuda", dtype=bf)
    h0 = torch.zeros(1, hv, dv, dk, device="cuda", dtype=bf)
    cu = torch.tensor([0, t], dtype=torch.int32, device="cuda")

    def f():
        chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
                               output_final_state=True, cu_seqlens=cu,
                               use_qk_l2norm_in_kernel=True)

    return bench(f, iters)


def project(attn_ms, gemm, act_ms, gdn_ms, t):
    """Whole-model prefill projection: total ms and ms/token."""
    gsum = sum(v["ms"] for v in gemm.values())
    total = N_ATTN * attn_ms + N_LAYERS * gsum + N_LAYERS * act_ms + N_GDN * gdn_ms
    return {"gemm_sum_ms": gsum, "total_ms": total, "ms_per_token": total / t}


def run_one(t, a):
    """Run all sections for one T and return the result dict."""
    attn_ms, tflops = bench_attn(t, a.depth, a.block_size, a.iters)
    gemm = bench_gemm(t, a.iters)
    act_ms = bench_act_quant(t, a.iters)
    gdn_ms = bench_gdn(t, a.gdn_k_heads, a.gdn_v_heads, a.gdn_k_dim, a.gdn_v_dim, a.iters)
    res = {"attn": {"ms": attn_ms, "tflops": tflops}, "gemm": gemm,
           "act_quant": {"ms": act_ms}, "gdn": {"ms": gdn_ms}}
    res["projected"] = project(attn_ms, gemm, act_ms, gdn_ms, t)
    return res


def run_all(a):
    """Run the bench for every T in a.tokens; returns {T: result}."""
    import vllm_gguf_plugin.ops  # noqa: F401

    return {t: run_one(t, a) for t in a.tokens}


def print_table(results, depth):
    """Print one block of rows per T."""
    for t, r in results.items():
        print(f"\n== T={t} depth={depth} ==")
        print(f"{'op':<28}{'ms':>10}{'TFLOPS/TOPS':>14}")
        print(f"{'attn_prefill (1 layer)':<28}{r['attn']['ms']:>10.3f}{r['attn']['tflops']:>14.1f}")
        for shp, v in r["gemm"].items():
            print(f"{'gemm ' + shp:<28}{v['ms']:>10.3f}{v['tops']:>14.1f}")
        print(f"{'act_quant':<28}{r['act_quant']['ms']:>10.3f}")
        print(f"{'gdn_chunk (1 layer)':<28}{r['gdn']['ms']:>10.3f}")
        p = r["projected"]
        print(f"{'model total ms':<28}{p['total_ms']:>10.1f}")
        print(f"{'model ms/token':<28}{p['ms_per_token']:>10.4f}")


def positive_int(text):
    """argparse type: an integer >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def block_size(text):
    """argparse type: a positive multiple of 16."""
    value = positive_int(text)
    if value % 16:
        raise argparse.ArgumentTypeError(f"must be a positive multiple of 16, got {value}")
    return value


def _check_depth(p, a):
    """Exit with a usage error when any T exceeds the KV depth."""
    too_big = [t for t in a.tokens if t > a.depth]
    if too_big:
        p.error(f"--tokens {too_big} must be <= --depth {a.depth}")


def parse_args(argv=None):
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=positive_int, nargs="+", default=[512, 2048],
                   help="new tokens T")
    p.add_argument("--depth", type=positive_int, default=50000,
                   help="KV depth D (incl. the T new tokens)")
    p.add_argument("--block-size", type=block_size, default=DEFAULT_BLOCK,
                   help="KV page size (positive multiple of 16)")
    p.add_argument("--iters", type=positive_int, default=20)
    p.add_argument("--gdn-k-heads", type=int, default=16)
    p.add_argument("--gdn-v-heads", type=int, default=48)
    p.add_argument("--gdn-k-dim", type=int, default=128)
    p.add_argument("--gdn-v-dim", type=int, default=128)
    p.add_argument("--json", metavar="PATH", help="write results as JSON")
    a = p.parse_args(argv)
    _check_depth(p, a)
    return a


def main(argv=None):
    """Entry point."""
    a = parse_args(argv)
    results = run_all(a)
    print_table(results, a.depth)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({str(t): r for t, r in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
