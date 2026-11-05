# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast path for Hadamard-folded ternary linears (Q2_0 / PQ2_0 / PTQ1_0).

At load time the GGUF blocks are repacked into a byte-plane layout with one
fp16 scale per 128 weights (see ``csrc/ternary/ternary_kernels.cu``). At run
time one kernel applies the sign flip and Hadamard rotation and quantizes the
activations to int8, then a dot4 GEMV (decode) or an int8 Triton GEMM
(prefill) consumes them.
"""

from functools import cache

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from ..triton.gemm.ternary_quant.packed import ternary_packed_gemm
from ..triton.gemm.utils import GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0

GROUP = 128
CHUNK = 512
GEMV_MAX_ROWS = 8
# With gfx12 WMMA, the dot4 GEMV only wins up to this many rows.
WMMA_GEMV_MAX_ROWS = 4
_TERN16 = __import__("os").environ.get("BONSAI_TERN16", "1") == "1"
SUPPORTED_BLOCK_SIZES = (512, 1024)

_BLOCK_LAYOUT = {
    GGML_TYPE_Q2_0: (64, 18),
    GGML_TYPE_PQ2_0: (128, 34),
    GGML_TYPE_PTQ1_0: (128, 28),
}


def kernels_available() -> bool:
    namespace = getattr(torch.ops, "_C_gguf", None)
    return namespace is not None and hasattr(namespace, "ternary_gemv")


@cache
def _has_gfx12_wmma(device_index: int) -> bool:
    if torch.version.hip is None or not hasattr(torch.ops._C_gguf, "ternary_wmma_gemm"):
        return False
    arch = torch.cuda.get_device_properties(device_index).gcnArchName.split(":")[0]
    return arch in ("gfx1200", "gfx1201")


_SPLIT_COUNTERS: dict[int, torch.Tensor] = {}


def _split_counters(device: torch.device) -> torch.Tensor:
    """Self-resetting split-K tile counters (zeroed once per device)."""
    counters = _SPLIT_COUNTERS.get(device.index)
    if counters is None:
        counters = torch.zeros(1 << 16, dtype=torch.int32, device=device)
        _SPLIT_COUNTERS[device.index] = counters
    return counters


def _decode_blocks(
    blocks: torch.Tensor, quant_type: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return codes (int32 in {0,1,2}, ``[R, K]``) and block scales."""
    rows, num_blocks, _ = blocks.shape
    if quant_type == GGML_TYPE_PTQ1_0:
        scales = blocks[..., 26:28].contiguous().view(torch.float16)[..., 0]
        packed = blocks[..., :26].to(torch.int32)
        pow3 = torch.tensor([1, 3, 9, 27, 81], device=blocks.device, dtype=torch.int32)

        def trits(part: torch.Tensor, count: int) -> torch.Tensor:
            q = (part.unsqueeze(-2) * pow3[:count, None]) & 0xFF
            return ((q * 3) >> 8).flatten(-2)

        codes = torch.cat(
            [
                trits(packed[..., 0:16], 5),
                trits(packed[..., 16:24], 5),
                trits(packed[..., 24:26], 4),
            ],
            dim=-1,
        )
    else:
        scales = blocks[..., 0:2].contiguous().view(torch.float16)[..., 0]
        packed = blocks[..., 2:].to(torch.int32)
        shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32)
        codes = ((packed.unsqueeze(-1) >> shifts) & 3).flatten(-2)
    return codes.reshape(rows, -1), scales


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    rows, k = codes.shape
    planes = codes.view(rows, k // CHUNK, 4, 4, 32).permute(0, 1, 4, 3, 2)
    shifts = 2 * torch.arange(4, device=codes.device, dtype=torch.int32)
    return (planes << shifts).sum(-1, dtype=torch.int32).to(torch.uint8).view(rows, -1)


def repack_ternary(
    weight: torch.Tensor, quant_type: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Repack GGUF ternary rows; ``None`` if scales are not per-128 groups."""
    block_weights, block_bytes = _BLOCK_LAYOUT[quant_type]
    rows, row_bytes = weight.shape
    if row_bytes % block_bytes:
        return None
    k = row_bytes // block_bytes * block_weights
    if k % CHUNK:
        return None
    q = torch.empty((rows, k // 4), dtype=torch.uint8, device=weight.device)
    s = torch.empty((rows, k // GROUP), dtype=torch.float16, device=weight.device)
    step = max(1, (1 << 22) // k)
    for start in range(0, rows, step):
        end = min(rows, start + step)
        blocks = weight[start:end].view(end - start, -1, block_bytes)
        codes, scales = _decode_blocks(blocks, quant_type)
        if block_weights != GROUP:
            scales = scales.view(end - start, -1, GROUP // block_weights)
            if not torch.equal(scales, scales[..., :1].expand_as(scales)):
                return None
            scales = scales[..., 0]
        q[start:end] = _pack_codes(codes)
        s[start:end] = scales
    return q, s


def _ternary_linear(
    x: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    xs = torch.empty((m, k // GROUP), dtype=torch.float32, device=x.device)
    xsx = torch.empty_like(xs)
    torch.ops._C_gguf.ternary_act_quant(x, signs, block_size, xq, xs, xsx)
    wmma = _has_gfx12_wmma(x.device.index)
    if wmma and WMMA_GEMV_MAX_ROWS < m <= 16 and _TERN16:
        # bonsai: split-K small-M WMMA kernel (csrc/int4attn/int4_attn.hip), ~485 vs ~340 GB/s at M=16
        from vllm_gguf_plugin.bonsai_int4_attn import ext as _bonsai_ext

        n = q.shape[0]
        return _bonsai_ext().tern_m16(q, s, xq, xs, 2 if n >= 8192 else 4, 1, x.dtype == torch.bfloat16)
    if wmma and 16 < m <= 64 and _TERN16:
        from vllm_gguf_plugin.bonsai_int4_attn import ext as _bonsai_ext

        return _bonsai_ext().tern_m64(q, s, xq, xs, 2, 2, x.dtype == torch.bfloat16)
    if m <= (WMMA_GEMV_MAX_ROWS if wmma else GEMV_MAX_ROWS):
        return torch.ops._C_gguf.ternary_gemv(q, s, xq, xs, xsx, x.dtype)
    if wmma:
        return torch.ops._C_gguf.ternary_wmma_gemm(
            q, s, xq, xs, _split_counters(x.device), x.dtype
        )
    return ternary_packed_gemm(xq, xs, q, s, x.dtype)


def _ternary_linear_fake(
    x: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], q.shape[0]))


direct_register_custom_op(
    op_name="gguf_ternary_linear",
    op_func=_ternary_linear,
    fake_impl=_ternary_linear_fake,
)
ternary_linear = torch.ops.vllm.gguf_ternary_linear
