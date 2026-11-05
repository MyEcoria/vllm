# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

from vllm_gguf_plugin.bonsai_kv_evict import gather_md, lag_scores, select_blocks

BS = 16
KW = dict(block_size=BS, ratio=4.0, sink=64, recent_min=128, min_ctx=512, segment=256)


def _run(seq, q, ratio=4.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = max(math.ceil(s / BS) for s in seq)
    sc = torch.rand(len(seq), m, generator=g)
    kw = dict(KW, ratio=ratio)
    return sc, select_blocks(sc, torch.tensor(seq), torch.tensor(q), **kw)


def test_count_sorted_floors():
    seq, q = [4000, 1000, 300], [1, 50, 1]
    sc, (idx, ln) = _run(seq, q)
    for r, s in enumerate(seq):
        nb = math.ceil(s / BS)
        kept = idx[r][idx[r] >= 0].tolist()
        assert kept == sorted(set(kept))
        if s < KW["min_ctx"]:
            assert kept == list(range(nb))
            continue
        assert len(kept) <= math.ceil(nb / 4.0) + 64 // BS + 128 // BS + 17
        assert set(range(64 // BS)) <= set(kept)
        tail = max((s - max(128, q[r] + 16)) // BS, 0)
        assert set(range(tail, nb)) <= set(kept)
        assert tail * BS <= s - q[r]
        assert ln[r] == (len(kept) - 1) * BS + s - (nb - 1) * BS
        for a in range(64 // BS, tail, 16):
            assert any(a <= b < min(a + 16, tail) for b in kept)


def test_identity():
    seq = [4000, 700]
    sc, (idx, ln) = _run(seq, [1, 1], ratio=1.0)
    assert ln.tolist() == seq
    assert idx[0, :250].tolist() == list(range(250))
    assert (idx[1, 44:] == -1).all()


def test_inf_kept_and_out_buffer():
    sc = torch.rand(1, 250)
    sc[0, 100] = float("inf")
    out = (torch.empty(1, 250, dtype=torch.int32), torch.empty(1, dtype=torch.int64))
    select_blocks(sc, torch.tensor([4000]), torch.tensor([1]), out=out, **KW)
    assert 100 in out[0][0].tolist()


def test_gather_md():
    seq = torch.tensor([4000])
    sc, (idx, ln) = _run([4000], [1])
    bt = torch.arange(1000, 1250, dtype=torch.int32)[None]
    nbt, nsl = gather_md(bt, idx, ln, seq, BS)
    n = int((idx >= 0).sum())
    assert nsl.item() == ln.item()
    assert nbt[0, :n].tolist() == (idx[0, :n] + 1000).tolist()
    assert nbt[0, n - 1] == 1249


def _lag_ref(k, v):
    n, scores = k.shape[0], []
    for i in range(n - 1):
        tot = 0
        for x in (k, v):
            lo, hi = x[i + 1].min(0).values, x[i + 1].max(0).values
            tot = tot + ((x[i] - lo) / (hi - lo)).std(0, unbiased=False).mean(-1)
        scores.append(tot.max())
    return torch.stack(scores)


def test_lag_scores():
    k, v = torch.randn(5, BS, 3, 8), torch.randn(5, BS, 3, 8)
    s = lag_scores(k, v, BS)
    assert s[-1] == float("inf")
    torch.testing.assert_close(s[:-1], _lag_ref(k, v), rtol=1e-4, atol=1e-5)


def test_evict_view_decode_only_and_scores_gather():
    from vllm_gguf_plugin.bonsai_kv_evict import evict_view, record_scores

    nb = 2000
    seq = torch.tensor([nb * BS - 3, 100 * BS, 50])
    ql = torch.tensor([1, 600, 1])
    perm = torch.randperm(4000)
    bt = torch.zeros(3, nb, dtype=torch.int32)
    bt[0] = perm[:nb].int()
    bt[1, :100] = perm[nb:nb + 100].int()
    bt[2, 0] = perm[-1].int()
    scores = torch.rand(4001).half()
    scores[bt[0, -1].long()] = float("inf")
    nbt, nsl = evict_view(bt, seq, ql, scores, BS, 5, 2.0)
    assert nbt.shape == bt.shape and nsl.dtype == seq.dtype
    assert nsl[0] < seq[0] and nsl[0] >= seq[0] // 2 - BS
    assert torch.equal(nbt[1], bt[1]) and nsl[1] == seq[1] and nsl[2] == seq[2]
    assert set(nbt[0, : (nsl[0] + BS - 1) // BS].tolist()) <= set(bt[0].tolist())
    assert nbt[0, 0] == bt[0, 0]


def test_record_scores_resets_touched_and_scores_partitions():
    from vllm_gguf_plugin import bonsai_kv_evict as ke

    scores = ke.score_buffer(10, torch.device("cpu"))
    scores[:] = 0.5
    slots = torch.arange(512) + 5 * 256
    k = torch.randn(512, 2, 8)
    ke.record_scores(scores, k, k.clone(), slots, 256, part=64)
    assert torch.isfinite(scores[5:7]).all() and scores[0] == 0.5
    ke.record_scores(scores, k[:1], k[:1], torch.tensor([7 * 256]), 256, part=64)
    assert torch.isinf(scores[7])


def test_record_scores_padded_slots_do_not_crash():
    """CUDA-graph padded batches (slots == -1) must not index with a negative block."""
    from vllm_gguf_plugin import bonsai_kv_evict as ke

    bs = 16
    scores = ke.score_buffer(64, torch.device("cpu"))
    slots = torch.cat([torch.arange(3 * bs, 3 * bs + 300), torch.full((212,), -1)])
    k = torch.randn(512, 2, 8)
    ke.record_scores(scores, k, k.clone(), slots, bs, part=64)
    assert torch.isfinite(scores[3:6]).any()
