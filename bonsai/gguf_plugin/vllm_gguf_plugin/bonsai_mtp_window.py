# SPDX-License-Identifier: Apache-2.0
"""Draft-only sliding window for the MTP head. BONSAI_MTP_WINDOW=N (tokens, 0 = off).

Only the MTP attention layer is declared sliding-window (SlidingWindowSpec), so vLLM's hybrid allocator
frees its old KV blocks; the 16 target attention layers stay full attention, so verification and the
output are unchanged (lossless). The int4 HIP kernel has no window argument: for such a layer the block
table is re-gathered to [last blocks covering the window + the query tokens] and seq_lens is shifted by
the dropped tokens (the kernel only uses relative positions). No attention sink: the allocator frees the
first blocks too. Prefill chunks see the window plus their own chunk length (approximation, draft only).
"""
from __future__ import annotations

import copy
import os

import torch


def _windowed(md, window: int, bs: int):
    """Copy of the metadata whose block_table / seq_lens only cover the trailing window."""
    bt, sl = md.block_table, md.seq_lens
    width = min(bt.shape[1], (window + md.max_query_len) // bs + 2)
    if width >= bt.shape[1]:
        return md
    drop = ((sl - md.max_query_len - window).clamp_min(0) // bs).to(torch.int32)
    cols = (drop[:, None] + torch.arange(width, device=bt.device, dtype=torch.int32)).clamp_(max=bt.shape[1] - 1)
    out = copy.copy(md)
    out.block_table = torch.gather(bt, 1, cols.to(torch.int64))
    out.seq_lens = sl - drop * bs
    return out


def patch():
    """Mark the MTP attention layers as sliding window and window the int4 kernel's view of them."""
    from vllm.model_executor.models import qwen3_5_mtp as m
    from vllm.v1.attention.backends import triton_attn as ta

    window = int(os.environ.get("BONSAI_MTP_WINDOW", "0"))
    if window <= 0:
        return
    pred = m.Qwen3_5MultiTokenPredictor
    orig_init = pred.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for layer in self.layers:
            layer.self_attn.attn.sliding_window = window
            layer.self_attn.attn.bonsai_win = window

    pred.__init__ = __init__
    impl = ta.TritonAttentionImpl
    orig_forward = impl.forward

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        w = getattr(layer, "bonsai_win", None)
        if w and attn_metadata is not None and kv_cache.numel():
            c = kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
            attn_metadata = _windowed(attn_metadata, w, c.shape[2])
        return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)

    impl.forward = forward
