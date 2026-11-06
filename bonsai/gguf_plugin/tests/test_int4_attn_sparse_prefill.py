# SPDX-License-Identifier: Apache-2.0
"""BONSAI_SPARSE_PREFILL (opt-in block-sparse prefill): tau=1.0 and unset must equal the dense kernel exactly;
tau=0.95 must keep a planted needle block and stay close to dense."""
import math

import pytest
import torch

try:
    from .test_int4_attn_decode import BS, G, HD, NKV, build_case
except ImportError:
    from test_int4_attn_decode import BS, G, HD, NKV, build_case

MIN_FREE = 2 << 30
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < MIN_FREE,
                       reason="needs >= 2 GiB of free GPU memory"),
]

CASES = [([6000], [512]), ([6000, 3000], [512, 300]), ([5000, 700], [517, 400])]
# deep contexts and ragged batches (default thresholds of the plugin: min ctx 16384 is lowered by run_prefill anyway)
DEEP = [([100000], [512]), ([100000], [2048]), ([50000], [512]), ([50000], [2048]),
        ([50000, 20000, 9000], [517, 2048, 300]), ([100000, 1000, 60000], [333, 700, 1025])]


def run_prefill(cache, bt, q, seq_lens, qlens, tau, monkeypatch):
    """Launch attn_prefill with BONSAI_SPARSE_PREFILL=tau (None = unset); returns the output."""
    from vllm_gguf_plugin import bonsai_int4_attn as m
    monkeypatch.delenv("BONSAI_PF_VARIANT", raising=False)
    monkeypatch.delenv("BONSAI_SPARSE_PREFILL", raising=False)
    if tau is not None:
        monkeypatch.setenv("BONSAI_SPARSE_PREFILL", str(tau))
    monkeypatch.setenv("BONSAI_SPARSE_MIN_CTX", "512")  # small data: sparse from 512 tokens of context
    monkeypatch.setenv("BONSAI_SPARSE_SINK", "256")
    monkeypatch.setenv("BONSAI_SPARSE_LOCAL", "512")
    e, dev = m.ext(), q.device
    q8, qs, qsum = m._S.q_bufs(q.shape[0], NKV * G, dev)
    e.q_prep(q, q8, qs, qsum, 1.0 / math.sqrt(HD))
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    work = [x for s, ql in enumerate(qlens) for r0 in range(0, ql * G, m.PF_ROWS) for x in (s, r0)]
    work = torch.tensor(work, dtype=torch.int32, device=dev).view(-1, 2)
    out = torch.full((q.shape[0], NKV * G, HD), 7.0, dtype=torch.float16, device=dev)
    e.attn_prefill(q8, qs, qsum, cache, bt, sl, qsl, work, out, max(seq_lens))
    torch.cuda.synchronize()
    return out.clone()


@pytest.mark.parametrize("seq_lens,qlens", CASES)
def test_tau_one_and_unset_equal_dense(seq_lens, qlens, monkeypatch):
    cache, bt, _, _, q = build_case(seq_lens, qlens, seed=3)
    dense = run_prefill(cache, bt, q, seq_lens, qlens, None, monkeypatch)
    assert torch.isfinite(dense).all()
    assert torch.equal(dense, run_prefill(cache, bt, q, seq_lens, qlens, 0, monkeypatch))
    one = run_prefill(cache, bt, q, seq_lens, qlens, 1.0, monkeypatch)
    assert torch.equal(dense, one), (dense.float() - one.float()).abs().max().item()


def test_needle_kept_at_tau_095(monkeypatch):
    """All queries share one direction per kv head; a key along it at position 3000 dominates attention."""
    seq_lens, qlens, pos = [6000], [512], 3000
    cache, bt, ks, vs, _ = build_case(seq_lens, qlens, seed=5)
    from vllm_gguf_plugin import bonsai_int4_attn as m
    base = torch.randn(NKV, HD, device="cuda", dtype=torch.float16, generator=torch.Generator("cuda").manual_seed(9))
    q = base.repeat_interleave(G, 0)[None].expand(qlens[0], -1, -1).contiguous()
    slot = torch.tensor([int(bt[0, pos // BS]) * BS + pos % BS], dtype=torch.int64, device="cuda")
    m.ext().kv_write(base[None], vs[0][pos:pos + 1], slot, cache)
    dense = run_prefill(cache, bt, q, seq_lens, qlens, None, monkeypatch)
    sparse = run_prefill(cache, bt, q, seq_lens, qlens, 0.95, monkeypatch)
    rel = (dense.float() - sparse.float()).abs().max().item() / dense.float().abs().max().item()
    assert rel < 0.05, rel


@pytest.mark.parametrize("seq_lens,qlens", DEEP)
def test_tau_one_equals_dense_deep(seq_lens, qlens, monkeypatch):
    cache, bt, _, _, q = build_case(seq_lens, qlens, seed=4)
    dense = run_prefill(cache, bt, q, seq_lens, qlens, None, monkeypatch)
    one = run_prefill(cache, bt, q, seq_lens, qlens, 1.0, monkeypatch)
    assert torch.equal(dense, one), (dense.float() - one.float()).abs().max().item()


# split-K chunk kernel (chunks <= BONSAI_PF_SPLITK_MAX tokens): block list consumed by the K-split loop
CHUNK_CASES = [([20000] * 4, [128] * 4), ([30000, 9000, 20000], [128, 200, 5]), ([100000], [128])]


def run_chunk(cache, bt, q, seq_lens, qlens, tau, monkeypatch):
    """Launch attn_decode on all sequences like the plugin's chunk path; tau None = unset."""
    from vllm_gguf_plugin import bonsai_int4_attn as m
    monkeypatch.delenv("BONSAI_SPARSE_PREFILL", raising=False)
    if tau is not None:
        monkeypatch.setenv("BONSAI_SPARSE_PREFILL", str(tau))
    monkeypatch.setenv("BONSAI_SPARSE_MIN_CTX", "512")
    monkeypatch.setenv("BONSAI_SPARSE_SINK", "256")
    monkeypatch.setenv("BONSAI_SPARSE_LOCAL", "512")
    monkeypatch.setenv("BONSAI_SPARSE_MIN_CHUNK", "100")
    e, dev = m.ext(), q.device
    q8, qs, qsum = m._S.q_bufs(q.shape[0], NKV * G, dev)
    e.q_prep(q, q8, qs, qsum, 1.0 / math.sqrt(HD))
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    ids = torch.arange(len(qlens), dtype=torch.int32, device=dev)
    qtiles = (max(qlens) * G + 15) // 16
    po, pml, splits = m._S.chunk_bufs(len(qlens), NKV, qtiles, dev)
    out = torch.full((q.shape[0], NKV * G, HD), 7.0, dtype=torch.float16, device=dev)
    e.attn_decode(q8, qs, qsum, cache, bt, sl, qsl, ids, qtiles, splits, po, pml, out, max(seq_lens))
    torch.cuda.synchronize()
    return out.clone()


@pytest.mark.parametrize("seq_lens,qlens", CHUNK_CASES)
def test_chunk_tau_one_equals_dense(seq_lens, qlens, monkeypatch):
    cache, bt, _, _, q = build_case(seq_lens, qlens, seed=6)
    dense = run_chunk(cache, bt, q, seq_lens, qlens, None, monkeypatch)
    assert torch.isfinite(dense).all()
    one = run_chunk(cache, bt, q, seq_lens, qlens, 1.0, monkeypatch)
    assert torch.equal(dense, one)


def test_chunk_needle_kept_at_tau_095(monkeypatch):
    seq_lens, qlens, pos = [20000] * 2, [128] * 2, 7000
    cache, bt, ks, vs, _ = build_case(seq_lens, qlens, seed=7)
    from vllm_gguf_plugin import bonsai_int4_attn as m
    base = torch.randn(NKV, HD, device="cuda", dtype=torch.float16, generator=torch.Generator("cuda").manual_seed(8))
    q = base.repeat_interleave(G, 0)[None].expand(sum(qlens), -1, -1).contiguous()
    for i in range(2):
        slot = torch.tensor([int(bt[i, pos // BS]) * BS + pos % BS], dtype=torch.int64, device="cuda")
        m.ext().kv_write(base[None], vs[i][pos:pos + 1], slot, cache)
    dense = run_chunk(cache, bt, q, seq_lens, qlens, None, monkeypatch)
    sparse = run_chunk(cache, bt, q, seq_lens, qlens, 0.95, monkeypatch)
    rel = (dense.float() - sparse.float()).abs().max().item() / dense.float().abs().max().item()
    assert rel < 0.05, rel
