# SPDX-License-Identifier: Apache-2.0
"""Accuracy + microbench of the HIP INT4 (K4/V3) decode attention. Run as __main__ for the microbench."""
import math

import pytest
import torch

HD, NKV, G, BS = 256, 4, 6, 64
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _ext_and_mod():
    from vllm_gguf_plugin import bonsai_int4_attn as m
    return m.ext(), m


def build_case(seq_lens, qlens, seed=0):
    """Random K/V written through kv_write; returns the cache, block table and raw k/v for the reference."""
    e, m = _ext_and_mod()
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    nblk = [(s + BS - 1) // BS for s in seq_lens]
    cache = torch.zeros(sum(nblk) + 1, NKV, BS, m.CONTENT_BYTES, dtype=torch.uint8, device=dev)
    bt = torch.zeros(len(seq_lens), max(nblk), dtype=torch.int32, device=dev)
    nxt, ks, vs = 1, [], []
    for i, s in enumerate(seq_lens):
        bt[i, :nblk[i]] = torch.arange(nxt, nxt + nblk[i], dtype=torch.int32, device=dev)
        slots = (bt[i, :nblk[i]].long().repeat_interleave(BS) * BS + torch.arange(BS, device=dev).repeat(nblk[i]))[:s]
        k = torch.randn(s, NKV, HD, device=dev, dtype=torch.float16, generator=g)
        v = torch.randn(s, NKV, HD, device=dev, dtype=torch.float16, generator=g)
        e.kv_write(k, v, slots, cache)
        ks.append(k); vs.append(v)
        nxt += nblk[i]
    q = torch.randn(sum(qlens), NKV * G, HD, device=dev, dtype=torch.float16, generator=g)
    return cache, bt, ks, vs, q


def reference(q, ks, vs, seq_lens, qlens):
    """fp32 causal attention on the unquantized k/v."""
    outs, o = [], 0
    for k, v, s, ql in zip(ks, vs, seq_lens, qlens):
        qq = q[o:o + ql].float()
        kk = k.float().repeat_interleave(G, dim=1)
        vv = v.float().repeat_interleave(G, dim=1)
        sc = torch.einsum("qhd,khd->hqk", qq, kk) / math.sqrt(HD)
        pos = torch.arange(s, device=q.device)
        qpos = torch.arange(s - ql, s, device=q.device)
        sc = sc.masked_fill(pos[None, None, :] > qpos[None, :, None], -math.inf)
        outs.append(torch.einsum("hqk,khd->qhd", sc.softmax(-1), vv))
        o += ql
    return torch.cat(outs)


def run_decode(cache, bt, q, seq_lens, qlens):
    """Launch the decode attention exactly like the plugin dispatch; returns the output."""
    e, m = _ext_and_mod()
    dev = q.device
    T = q.shape[0]
    q8, qs, qsum = m._S.q_bufs(T, NKV * G, dev)
    e.q_prep(q, q8, qs, qsum, 1.0 / math.sqrt(HD))
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=dev)
    ids = m._S.arange(len(seq_lens), dev)
    qtiles = (max(qlens) * G + 15) // 16
    out = torch.zeros(T, NKV * G, HD, dtype=torch.float16, device=dev)
    m.decode_call(e, q8, qs, qsum, cache, bt, sl, qsl, ids, qtiles, out)
    return out


ERR_MAX = 1.0  # pure K4/V3 quantization noise (ctx 1 returns one 3-bit V row); kernel math is unchanged
CASES = [([1], [1]), ([255], [1]), ([4097], [4]), ([100000], [4]), ([1, 255, 4097], [1, 4, 3]),
         ([30000, 777, 100000, 5], [4, 4, 2, 1])]


@pytest.mark.parametrize("seq_lens,qlens", CASES)
def test_decode_accuracy(seq_lens, qlens):
    seq_lens = [max(s, q) for s, q in zip(seq_lens, qlens)]
    cache, bt, ks, vs, q = build_case(seq_lens, qlens)
    ref = reference(q, ks, vs, seq_lens, qlens)
    err = (run_decode(cache, bt, q, seq_lens, qlens).float() - ref).abs().max().item()
    print("max abs err", err)
    assert err <= ERR_MAX, err


def bench(iters=50, ctx=100000, nseq=4, nq=4):
    """ms/call and GB/s of K/V traffic for nseq sequences of ctx tokens."""
    seq_lens, qlens = [ctx] * nseq, [nq] * nseq
    cache, bt, _, _, q = build_case(seq_lens, qlens)
    for _ in range(5):
        run_decode(cache, bt, q, seq_lens, qlens)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        run_decode(cache, bt, q, seq_lens, qlens)
    b.record(); torch.cuda.synchronize()
    ms = a.elapsed_time(b) / iters
    gb = nseq * ctx * NKV * 232 / 1e9
    return ms, gb / (ms / 1e3)


if __name__ == "__main__":
    ms, gbs = bench()
    print(f"{ms:.3f} ms/call (incl. q_prep), {gbs:.0f} GB/s")
