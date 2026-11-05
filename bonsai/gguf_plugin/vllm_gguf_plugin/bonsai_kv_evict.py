# SPDX-License-Identifier: Apache-2.0
"""Pure-tensor core of the opt-in "virtual" KV eviction (blocks stay allocated, attention reads a subset).

select_blocks picks the logical blocks to keep per request (sink + contiguous recent tail containing the
query chunk + per-segment floors + global top-k by score), gather_md rebuilds the block table / seq_lens
from them, lag_scores scores blocks with LagKV (arXiv 2504.04704). No Python loop over blocks and all
outputs have fixed shapes [num_requests, max_blocks], so everything is CUDA-graph safe.
"""
from __future__ import annotations

import torch


def _ceil_div(a: torch.Tensor, b) -> torch.Tensor:
    """Integer ceil(a / b) for tensors."""
    return -((-a) // b)


def _rank_desc(key: torch.Tensor) -> torch.Tensor:
    """Per row, rank (0 = largest) of each element; stable on ties."""
    order = torch.argsort(key, dim=1, descending=True, stable=True)
    return torch.argsort(order, dim=1, stable=True)


def _segment_keep(scores, seg, mid, floor_ratio):
    """Mask of the best ceil(count/floor_ratio) blocks inside every segment of the middle."""
    nseg = seg.shape[1] + 1
    seg = torch.where(mid, seg, torch.full_like(seg, nseg - 1))
    cnt = torch.zeros(seg.shape[0], nseg, dtype=torch.long, device=seg.device)
    cnt.scatter_add_(1, seg, mid.long())
    start = cnt.cumsum(1) - cnt
    o1 = torch.argsort(scores.masked_fill(~mid, float("-inf")), dim=1, descending=True, stable=True)
    o2 = torch.argsort(seg.gather(1, o1), dim=1, stable=True)
    order = o1.gather(1, o2)
    pos = torch.argsort(order, dim=1, stable=True)
    rank = pos - start.gather(1, seg)
    return mid & (rank < _ceil_div(cnt, floor_ratio).gather(1, seg))


def _must_keep(scores, nb, seq_lens, query_lens, bs, sink, recent_min):
    """Mask of sink blocks, contiguous recent tail (covers the query chunk) and +inf-scored blocks."""
    idx = torch.arange(scores.shape[1], device=scores.device)[None]
    recent = torch.clamp(query_lens + 16, min=recent_min)
    tail = ((seq_lens - recent).clamp_min(0) // bs)[:, None]
    return (idx < sink // bs) | (idx >= tail) | torch.isposinf(scores), tail


def _kept_tokens(nkept, seq_lens, bs):
    """Token count of the kept blocks, the last (tail) block being possibly partial."""
    last = seq_lens - (_ceil_div(seq_lens, bs) - 1).clamp_min(0) * bs
    return torch.where(nkept > 0, (nkept - 1) * bs + last, torch.zeros_like(seq_lens))


def _keep_mask(scores, seq_lens, query_lens, bs, ratio, sink, recent_min, min_ctx, segment):
    """Boolean [R, M] mask of the logical blocks to keep."""
    nb = _ceil_div(seq_lens, bs)
    idx = torch.arange(scores.shape[1], device=scores.device)[None]
    valid = idx < nb[:, None]
    must, tail = _must_keep(scores, nb, seq_lens, query_lens, bs, sink, recent_min)
    must &= valid
    mid = valid & ~must
    seg = ((idx - sink // bs) // max(segment // bs, 1)).clamp_min(0).expand_as(scores)
    keep = must | _segment_keep(scores, seg, mid, 2 * ratio)
    budget = _ceil_div(nb, 1) if ratio <= 1 else torch.ceil(nb / ratio).long()
    rest = valid & ~keep
    rank = _rank_desc(scores.masked_fill(~rest, float("-inf")))
    keep |= rest & (rank < (budget - keep.sum(1))[:, None])
    return torch.where((seq_lens < min_ctx)[:, None], valid, keep)


def select_blocks(scores, seq_lens, query_lens, block_size=16, ratio=2.0, sink=4096, recent_min=4096,
                  min_ctx=16384, segment=1024, out=None):
    """Return (kept_idx [R, M] ascending, -1 padded; kept_len [R] tokens); optionally written into `out`."""
    keep = _keep_mask(scores, seq_lens, query_lens, block_size, ratio, sink, recent_min, min_ctx, segment)
    idx = torch.arange(scores.shape[1], device=scores.device)[None].expand_as(scores)
    key = torch.where(keep, idx, idx + scores.shape[1])
    kept_idx = torch.argsort(key, dim=1, stable=True)
    nkept = keep.sum(1)
    kept_idx = torch.where(idx < nkept[:, None], kept_idx, torch.full_like(kept_idx, -1)).to(torch.int32)
    kept_len = _kept_tokens(nkept, seq_lens, block_size).to(seq_lens.dtype)
    if out is not None:
        out[0].copy_(kept_idx)
        out[1].copy_(kept_len)
        return out
    return kept_idx, kept_len


def gather_md(block_table, kept_idx, kept_len, seq_lens, block_size):
    """Return (new block table, new seq_lens) restricted to kept blocks; padding entries are 0."""
    ok = kept_idx >= 0
    bt = torch.gather(block_table, 1, kept_idx.clamp_min(0).to(torch.int64))
    bt = torch.where(ok, bt, torch.zeros_like(bt))
    return bt, _kept_tokens(ok.sum(1), seq_lens, block_size).to(seq_lens.dtype)


def _lag_std(x: torch.Tensor) -> torch.Tensor:
    """x [N, bs, H, D] -> [N, H]: block std of values normalised by the next block's min/max."""
    nxt = x[1:]
    lo, hi = nxt.amin(1, keepdim=True), nxt.amax(1, keepdim=True)
    norm = (x[:-1] - lo) / (hi - lo).clamp_min(1e-6)
    return norm.std(dim=1, unbiased=False).mean(-1)


def lag_scores(k, v, block_size):
    """LagKV block scores [N] from k, v [N*block_size or N, bs, H, D]; last block is +inf (must keep)."""
    n = k.shape[0]
    k = k.reshape(n, block_size, *k.shape[-2:]).float()
    v = v.reshape(n, block_size, *v.shape[-2:]).float()
    score = (_lag_std(k) + _lag_std(v)).amax(-1)
    return torch.cat([score, score.new_full((1,), float("inf"))])


# ---------------------------------------------------------------- wiring (mode "lag")
# Each full-attention layer is its own KV group (own block ids), so the selection is computed once per step
# as LOGICAL block indices from the scoring layer's group, and every layer's forward gathers its own table.
class _State:
    """Process-wide state: scores, per-step kept logical indices, layer order, kernel block size."""
    scores = None
    acc = None
    kept = None
    order = []
    bs = None
    scored = 0
    builds = 0


_S = _State()


def evict_idx(block_table, seq_lens, query_lens, scores, bs, dec_max, ratio):
    """Kept logical block indices [R, M] (-1 padded); identity for rows that are not decode/verify."""
    logical = scores[block_table.long()].float()
    kept, _ = select_blocks(logical, seq_lens, query_lens, bs, ratio, segment=max(1024, 8 * bs))
    idx = torch.arange(block_table.shape[1], device=block_table.device, dtype=torch.int32)[None]
    ident = torch.where(idx < _ceil_div(seq_lens, bs)[:, None], idx, torch.full_like(kept, -1))
    return torch.where((query_lens <= dec_max)[:, None], kept, ident)


def restrict(block_table, seq_lens, kept, bs):
    """Gathered (block table, seq_lens) for kept indices; padded rows (seq_len 0) stay 0."""
    bt, sl = gather_md(block_table, kept, None, seq_lens, bs)
    return bt, torch.where(seq_lens > 0, sl, seq_lens)


def evict_view(block_table, seq_lens, query_lens, scores, bs, dec_max, ratio):
    """(block_table, seq_lens) restricted to kept blocks for decode/verify rows; other rows unchanged."""
    kept = evict_idx(block_table, seq_lens, query_lens, scores, bs, dec_max, ratio)
    return restrict(block_table, seq_lens, kept, bs)


def score_buffer(num_blocks, device):
    """Lazily allocate fp16 scores (+1 dummy slot for slot -1; +inf = keep) and fp32 sum/count accumulators."""
    if _S.scores is None or _S.scores.shape[0] != num_blocks + 1:
        _S.scores = torch.full((num_blocks + 1,), float("inf"), dtype=torch.float16, device=device)
        _S.acc = torch.zeros(2, num_blocks + 1, dtype=torch.float32, device=device)
    return _S.scores


def _partition_scores(key, value, slots, part):
    """LagKV score of every `part`-token partition (vs the next one) and its block slot; ok-mask."""
    npart = slots.shape[0] // part
    ar = torch.arange(part, device=slots.device)
    idx = torch.arange(npart, device=slots.device)[:, None] * part + ar
    starts = idx[:-1, 0]
    sub = lag_scores(key[idx], value[idx], part)[:-1]
    ok = slots[starts + 2 * part - 1] == slots[starts] + 2 * part - 1
    return sub, ok, slots[starts]


def record_scores(scores, key, value, slots, bs, part=64):
    """Reset blocks starting in this update, then accumulate per-block LagKV partition scores.

    A block (e.g. 1840 tokens) rarely lies in one update, so it is scored by the mean of the scores of
    the `part`-token partitions seen across updates (fixed shapes: graph safe).
    """
    slots = slots.long()
    dummy = scores.shape[0] - 1
    reset = torch.where((slots >= 0) & (slots % bs == 0), slots // bs, torch.full_like(slots, dummy))
    scores.index_fill_(0, reset, float("inf"))
    _S.acc.index_fill_(1, reset, 0.0)
    if slots.shape[0] < 2 * part:
        return
    sub, ok, first = _partition_scores(key, value, slots, part)
    ok = ok & (first >= 0)
    blk, w = torch.where(ok, first // bs, torch.full_like(first, dummy)), ok.float()
    _S.acc[0].index_add_(0, blk, torch.nan_to_num(sub) * w)
    _S.acc[1].index_add_(0, blk, w)
    cnt = _S.acc[1][blk]
    mean = torch.where(cnt > 0, _S.acc[0][blk] / cnt.clamp_min(1), torch.full_like(cnt, float("inf")))
    scores.index_copy_(0, blk, mean.to(scores.dtype))


def _kept_buffer(rows, cols, device):
    """Persistent kept-index buffer (stable address for CUDA graphs), identity-initialised."""
    if _S.kept is None or _S.kept.shape[0] < rows or _S.kept.shape[1] != cols:
        ident = torch.arange(cols, device=device, dtype=torch.int32)
        _S.kept = ident.repeat(max(rows, 1024), 1)
    return _S.kept[:rows]


def compute_kept(md, common, dec_max, ratio):
    """Scoring group only: write this step's kept logical indices (identity until scores exist)."""
    bt, sl = md.block_table, md.seq_lens
    rows = sl.shape[0]
    kept = _kept_buffer(rows, bt.shape[1], bt.device)
    if _S.scores is None or _S.bs is None:
        return kept
    qsl = common.query_start_loc
    qlen = torch.nn.functional.pad(qsl[1: common.num_reqs + 1] - qsl[: common.num_reqs],
                                   (0, rows - common.num_reqs))
    kept.copy_(evict_idx(bt, sl, qlen, _S.scores, _S.bs, dec_max, ratio))
    return kept


def _debug(on, msg):
    """Print a debug line when BONSAI_KV_EVICT_DEBUG=1."""
    if on:
        print(f"[kv_evict] build#{_S.builds} {msg}", flush=True)


def _debug_rows(md, kept, on):
    """Debug: kept/total blocks of the first rows."""
    if on and _S.bs and md.max_query_len <= 5 and md.seq_lens.max().item() > 20000 and _S.builds % 20 == 0:
        n = min(4, kept.shape[0])
        tot = ((md.seq_lens[:n] + _S.bs - 1) // _S.bs).tolist()
        fin = torch.isfinite(_S.scores[md.block_table[0].long()[: tot[0]]]).sum().item() if _S.scores is not None else -1
        print(f"[kv_evict] kept/total {(kept[:n] >= 0).sum(1).tolist()}/{tot} bs={_S.bs} "
              f"scored={_S.scored} finite_row0={fin} qmax={md.max_query_len}", flush=True)


def patch():
    """Hook the Triton builder (selection), KV write (scoring layer) and forward (per-layer gather)."""
    import os
    import copy

    from vllm.v1.attention.backends import triton_attn as ta
    from vllm.v1.kv_cache_interface import KVQuantMode

    from .bonsai_int4_attn import DEC_MAX

    ratio = float(os.environ.get("BONSAI_KV_EVICT_RATIO", "2"))
    index = int(os.environ.get("BONSAI_KV_EVICT_LAYER", "7"))
    part = int(os.environ.get("BONSAI_KV_EVICT_PART", "64"))
    dbg = os.environ.get("BONSAI_KV_EVICT_DEBUG") == "1"
    builder, impl = ta.TritonAttentionMetadataBuilder, ta.TritonAttentionImpl
    orig_build, orig_update, orig_forward = builder.build, impl.do_kv_cache_update, impl.forward

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        name = self.layer_names[0]
        if not name.startswith("mtp") and name not in _S.order:
            _S.order.append(name)
            _debug(dbg, f"attention group {len(_S.order)}: {name} spec_bs={self.kv_cache_spec.block_size}")
        if len(_S.order) > index and name == _S.order[index]:
            _S.builds += 1
            kept = compute_kept(md, common_attn_metadata, DEC_MAX, ratio)
            _debug_rows(md, kept, dbg)
        return md

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        orig_update(self, layer, key, value, kv_cache, slot_mapping)
        if (self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD and kv_cache.numel()
                and len(_S.order) > index and getattr(layer, "layer_name", None) == _S.order[index]):
            n = slot_mapping.shape[0]
            c = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
            _S.bs = c.shape[2]
            record_scores(score_buffer(c.shape[0], key.device), key[:n], value[:n], slot_mapping, _S.bs, part)
            _S.scored += 1
            _debug(dbg and _S.scored == 1, f"first record_scores n={n} bs={_S.bs} num_blocks={c.shape[0]}")

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        name = getattr(layer, "layer_name", "mtp")
        if (attn_metadata is not None and _S.kept is not None and _S.bs and not name.startswith("mtp")
                and self._kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD
                and attn_metadata.block_table.shape[1] == _S.kept.shape[1]):
            attn_metadata = copy.copy(attn_metadata)
            rows = attn_metadata.seq_lens.shape[0]
            attn_metadata.block_table, attn_metadata.seq_lens = restrict(
                attn_metadata.block_table, attn_metadata.seq_lens, _S.kept[:rows], _S.bs)
        return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)

    builder.build, impl.do_kv_cache_update, impl.forward = build, do_kv_cache_update, forward
