# SPDX-License-Identifier: Apache-2.0
"""HIP INT4 KV-cache attention for gfx1201 (bonsai). Enabled with BONSAI_INT4_ATTN=1 and
--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN.

Replaces the Triton INT4 read/write paths of TritonAttentionImpl with the kernels in
csrc/int4attn/int4_attn.hip (own cache layout: K per token, V transposed per 16-slot tile,
orthonormal RHT). Decode / spec-verify sequences (query <= DEC_MAX tokens) use a split-K
kernel with a fixed grid (CUDA-graph safe); longer queries use the prefill kernel.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch

DEC_MAX = int(os.environ.get("BONSAI_INT4_DEC_MAX", "5"))
# K 4-bit + V 3-bit (MSE-clipped): 232 B per (token, kv head) instead of 264
V3 = os.environ.get("BONSAI_KV_V3", "1") == "1"
CONTENT_BYTES = 232 if V3 else 264
NUM_SPLITS = int(os.environ.get("BONSAI_INT4_SPLITS", "32"))
PF_ROWS = 64  # rows per prefill work item (4 waves x 16)
SPLIT_CAP = int(os.environ.get("BONSAI_INT4_SPLIT_CAP", "1024"))
CHUNK_SPLITK_MAX = int(os.environ.get("BONSAI_CHUNK_SPLITK_MAX", "256"))
OVERLAP = os.environ.get("BONSAI_ATTN_OVERLAP", "1") == "1"

_ext = None
_DUMP = os.environ.get("BONSAI_DUMP")
_dump_count = {}


def _dump(impl, q, k, v, md):
    """Debug: save the q/k/v of every attention call of the first long prefill (<= 40 chunks)."""
    key = id(impl)
    n = _dump_count.get(key, 0)
    if n >= 40:
        return
    _dump_count[key] = n + 1
    d = Path(_DUMP); d.mkdir(parents=True, exist_ok=True)
    torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu(), "scale": impl.scale,
                "qsl": md.query_start_loc.cpu(), "seq_lens": md.seq_lens.cpu()}, d / f"l{key}_{n:03d}.pt")


def ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        src = Path(__file__).parent / "csrc" / "int4attn" / "int4_attn.hip"
        build = Path(os.environ.get("BONSAI_INT4_BUILD", Path.home() / ".cache" / "bonsai-int4attn"))
        build.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1201")
        flags = [f for f in ("V3SIM", "K3SIM") if os.environ.get("BONSAI_" + f) == "1"]
        if V3:
            flags.append("V3")
        tag = "".join("_" + f.lower() for f in flags)
        if flags:
            build = build.with_name(build.name + tag)
            build.mkdir(parents=True, exist_ok=True)
        _ext = load(name="bonsai_int4_attn" + tag, sources=[str(src)], extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--offload-arch=gfx1201"] + [f"-DBONSAI_{f}" for f in flags],
                    build_directory=str(build), verbose=False)
    return _ext


class _Scratch:
    """Shared per-process scratch (layers run sequentially)."""
    def __init__(self):
        self.part = {}
        self.q = {}
        self.ids = {}

    def part_bufs(self, nseq, nkv, qtiles, device):
        """Fixed-size buffers (never reallocated: CUDA graphs keep their address)."""
        splits = max(1, min(NUM_SPLITS, SPLIT_CAP // (nseq * nkv * qtiles)))
        n = nseq * nkv * qtiles * splits * 16
        b = self.part.get(device)
        if b is None:
            rows = max(n, SPLIT_CAP * 16 + 4 * 4 * 3 * 16)
            b = (torch.empty(rows, 256, dtype=torch.float32, device=device),
                 torch.empty(rows, 2, dtype=torch.float32, device=device))
            self.part[device] = b
        assert n <= b[0].shape[0], (nseq, nkv, qtiles, splits)
        return b[0], b[1], splits

    def chunk_bufs(self, nseq, nkv, qtiles, device):
        splits = max(1, min(NUM_SPLITS, 2048 // (nseq * nkv * qtiles)))
        n = nseq * nkv * qtiles * splits * 16
        b = self.part.get(("chunk", device))
        if b is None or b[0].shape[0] < n:
            rows = max(n, 2048 * 16)
            b = (torch.empty(rows, 256, dtype=torch.float32, device=device),
                 torch.empty(rows, 2, dtype=torch.float32, device=device))
            self.part[("chunk", device)] = b
        return b[0], b[1], splits

    def q_bufs(self, T, nh, device):
        b = self.q.get(device)
        if b is None or b[0].shape[0] < T:
            n = max(T, 4096)
            b = (torch.empty(n, nh, 256, dtype=torch.int8, device=device),
                 torch.empty(n, nh, dtype=torch.float32, device=device),
                 torch.empty(n, nh, dtype=torch.int32, device=device))
            self.q[device] = b
        return b[0][:T], b[1][:T], b[2][:T]

    def side_stream(self, device):
        st = self.ids.get(("stream", device))
        if st is None:
            st = self.ids[("stream", device)] = torch.cuda.Stream(device=device)
        return st

    def arange(self, n, device):
        a = self.ids.get(device)
        if a is None or a.shape[0] < n:
            a = torch.arange(max(n, 1024), dtype=torch.int32, device=device)
            self.ids[device] = a
        return a[:n]


_S = _Scratch()


def decode_call(e, q8, qs, qsum, cache, block_table, seq_lens, qsl, ids, qtiles, out):
    """Launch the decode attention on all sequences `ids` (used by the decode microbench / tests)."""
    po, pml, splits = _S.part_bufs(ids.shape[0], cache.shape[1], qtiles, q8.device)
    e.attn_decode(q8, qs, qsum, cache, block_table, seq_lens, qsl, ids, qtiles, splits, po, pml, out)


def _prepare(md, common):
    """Attach the decode / prefill split to the metadata (once per step)."""
    num_reqs = common.num_reqs
    dev = common.query_start_loc.device
    if md.max_query_len <= DEC_MAX:
        md._b4 = ("dec", _S.arange(num_reqs, dev), (md.max_query_len * 6 + 15) // 16, None, None, 0)
        return
    qsl = common.query_start_loc_cpu[: num_reqs + 1].tolist()
    dec, chunk, work = [], [], []
    max_dec, max_chunk = 1, 1
    for s in range(num_reqs):
        ql = qsl[s + 1] - qsl[s]
        if ql <= 0:
            continue
        if ql <= DEC_MAX:
            dec.append(s)
            max_dec = max(max_dec, ql)
        elif ql <= CHUNK_SPLITK_MAX:
            # small prefill chunk mixed with decodes: the split-K kernel keeps the GPU busy
            chunk.append(s)
            max_chunk = max(max_chunk, ql)
        else:
            for r0 in range(0, ql * 6, PF_ROWS):
                work += [s, r0]
    dec_t = torch.tensor(dec, dtype=torch.int32).to(dev, non_blocking=True) if dec else None
    chunk_t = torch.tensor(chunk, dtype=torch.int32).to(dev, non_blocking=True) if chunk else None
    work_t = torch.tensor(work, dtype=torch.int32).view(-1, 2).to(dev, non_blocking=True) if work else None
    md._b4 = ("mix", dec_t, (max_dec * 6 + 15) // 16, work_t, chunk_t, (max_chunk * 6 + 15) // 16)


def patch():
    from vllm.v1.attention.backends import triton_attn as ta
    from vllm.v1.kv_cache_interface import KVQuantMode

    backend = ta.TritonAttentionBackend
    orig_customize = backend.customize_spec.__func__

    def customize_spec(cls, spec):
        spec = orig_customize(cls, spec)
        if spec.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            from dataclasses import replace

            spec = replace(spec, state_content_bytes=CONTENT_BYTES)
        return spec

    backend.customize_spec = classmethod(customize_spec)

    builder = ta.TritonAttentionMetadataBuilder
    orig_build = builder.build

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        _prepare(md, common_attn_metadata)
        return md

    builder.build = build

    impl = ta.TritonAttentionImpl
    orig_forward = impl.forward
    orig_update = impl.do_kv_cache_update

    def cache_u8(kv_cache):
        # logical (num_blocks, nkv, block_size, 264) uint8
        return kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self._kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD:
            return orig_update(self, layer, key, value, kv_cache, slot_mapping)
        n = slot_mapping.shape[0]
        ext().kv_write(key[:n], value[:n], slot_mapping, cache_u8(kv_cache))

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
                output_block_scale=None):
        if self._kv_quant_mode != KVQuantMode.INT4_PER_TOKEN_HEAD or attn_metadata is None:
            return orig_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                                output_block_scale)
        e = ext()
        T = attn_metadata.num_actual_tokens
        if _DUMP and T >= 1024:
            _dump(self, query[:T], key[:T], value[:T], attn_metadata)
        nh = query.shape[1]
        cache = cache_u8(kv_cache)
        q8, qs, qsum = _S.q_bufs(T, nh, query.device)
        e.q_prep(query[:T], q8, qs, qsum, float(self.scale))
        out = output[:T].view(T, nh, 256)
        mode, dec_ids, qtiles, work, chunk_ids, chunk_qtiles = attn_metadata._b4
        nkv = cache.shape[1]
        has_dec = dec_ids is not None and dec_ids.numel() > 0
        if chunk_ids is not None:
            main = torch.cuda.current_stream()
            if has_dec and OVERLAP:
                side = _S.side_stream(query.device)
                side.wait_stream(main)
                po, pml, splits = _S.part_bufs(dec_ids.shape[0], nkv, qtiles, query.device)
                with torch.cuda.stream(side):
                    e.attn_decode(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                  attn_metadata.query_start_loc, dec_ids, qtiles, splits, po, pml, out)
            elif has_dec:
                po, pml, splits = _S.part_bufs(dec_ids.shape[0], nkv, qtiles, query.device)
                e.attn_decode(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                  attn_metadata.query_start_loc, dec_ids, qtiles, splits, po, pml, out)
            co, cml, csplits = _S.chunk_bufs(chunk_ids.shape[0], nkv, chunk_qtiles, query.device)
            e.attn_decode(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                  attn_metadata.query_start_loc, chunk_ids, chunk_qtiles, csplits, co, cml, out)
            if work is not None:
                e.attn_prefill(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                               attn_metadata.query_start_loc, work, out)
            if has_dec and OVERLAP:
                main.wait_stream(side)
            return output
        if has_dec and work is not None and OVERLAP:
            # mixed step: the memory-bound decode attention runs on a side stream, overlapping the
            # compute-bound prefill attention
            main = torch.cuda.current_stream()
            side = _S.side_stream(query.device)
            side.wait_stream(main)
            po, pml, splits = _S.part_bufs(dec_ids.shape[0], nkv, qtiles, query.device)
            with torch.cuda.stream(side):
                e.attn_decode(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                  attn_metadata.query_start_loc, dec_ids, qtiles, splits, po, pml, out)
            e.attn_prefill(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                           attn_metadata.query_start_loc, work, out)
            main.wait_stream(side)
            return output
        if has_dec:
            po, pml, splits = _S.part_bufs(dec_ids.shape[0], nkv, qtiles, query.device)
            e.attn_decode(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                  attn_metadata.query_start_loc, dec_ids, qtiles, splits, po, pml, out)
        if work is not None:
            e.attn_prefill(q8, qs, qsum, cache, attn_metadata.block_table, attn_metadata.seq_lens,
                           attn_metadata.query_start_loc, work, out)
        return output

    impl.forward = forward
    impl.do_kv_cache_update = do_kv_cache_update
