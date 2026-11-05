# SPDX-License-Identifier: Apache-2.0
"""BONSAI_PF_VARIANT=1 (WG-shared unpacked K/V tiles, W = 4 or 8 waves) must be bit-exact vs variant 0."""
import math

import pytest
import torch

try:
    from .test_int4_attn_decode import G, HD, NKV, build_case
except ImportError:
    from test_int4_attn_decode import G, HD, NKV, build_case

MIN_FREE = 2 << 30
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < MIN_FREE,
                       reason="needs >= 2 GiB of free GPU memory"),
]

# (seq_lens, qlens): ragged batches, depth 1840..20000, chunk 128/512, tail tiles (qlen*6 % 16 != 0)
CASES = [
    ([1840], [128]), ([20000], [512]), ([20000], [517]), ([1840, 6000, 20000], [128, 512, 77]),
    ([300, 4100], [300, 33]), ([9999, 1850, 700], [512, 130, 9]), ([20000, 20000], [128, 512]),
]


def run_prefill(variant, w, monkeypatch, cache, bt, q, seq_lens, qlens):
    """Launch attn_prefill like the plugin dispatch for the given BONSAI_PF_VARIANT; returns the output."""
    from vllm_gguf_plugin import bonsai_int4_attn as m
    monkeypatch.setenv("BONSAI_PF_VARIANT", str(variant))
    monkeypatch.setenv("BONSAI_PF_W", str(w))
    rows = 16 * (w if variant == 1 else 4)  # rows per work item (bonsai_int4_attn.PF_ROWS)
    e, dev = m.ext(), q.device
    q8, qs, qsum = m._S.q_bufs(q.shape[0], NKV * G, dev)
    e.q_prep(q, q8, qs, qsum, 1.0 / math.sqrt(HD))
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    qsl = torch.tensor([0] + torch.tensor(qlens).cumsum(0).tolist(), dtype=torch.int32, device=dev)
    work = [x for s, ql in enumerate(qlens) for r0 in range(0, ql * G, rows) for x in (s, r0)]
    work = torch.tensor(work, dtype=torch.int32, device=dev).view(-1, 2)
    out = torch.full((q.shape[0], NKV * G, HD), 7.0, dtype=torch.float16, device=dev)
    e.attn_prefill(q8, qs, qsum, cache, bt, sl, qsl, work, out)
    torch.cuda.synchronize()
    return out.clone()


@pytest.mark.parametrize("w", [4, 8])
@pytest.mark.parametrize("seq_lens,qlens", CASES)
def test_variant1_bit_exact(seq_lens, qlens, w, monkeypatch):
    cache, bt, _, _, q = build_case(seq_lens, qlens, seed=1)
    ref = run_prefill(0, 4, monkeypatch, cache, bt, q, seq_lens, qlens)
    new = run_prefill(1, w, monkeypatch, cache, bt, q, seq_lens, qlens)
    assert torch.isfinite(ref).all()
    assert torch.equal(ref, new), (ref.float() - new.float()).abs().max().item()
