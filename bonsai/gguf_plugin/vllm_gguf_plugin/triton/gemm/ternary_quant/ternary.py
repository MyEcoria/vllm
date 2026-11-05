"""Q2_0, PQ2_0 and PTQ1_0 GEMM kernels.

All three types store one fp16 scale per block and codes that decode to
``code - 1``. ``Q2_0`` (64 weights) and ``PQ2_0`` (128 weights) pack four
2-bit codes per byte after the scale. ``PTQ1_0`` (128 weights) packs base-3
trits like ``TQ1_0``: bytes 0-15 hold five trits each (weights 0-79), bytes
16-23 five each (weights 80-119), bytes 24-25 four each (weights 120-127),
and the scale follows at byte 26.
"""

import torch
import triton
import triton.language as tl

from ..utils import (
    BLOCK_BYTES_BY_TYPE,
    BLOCK_QK_BY_TYPE,
    GGML_TYPE_PQ2_0,
    GGML_TYPE_PTQ1_0,
    GGML_TYPE_Q2_0,
    TRITON_SUPPORTED_ACTIVATION_DTYPES,
    load_f16_from_u8,
)

TERNARY_TYPES = (GGML_TYPE_Q2_0, GGML_TYPE_PQ2_0, GGML_TYPE_PTQ1_0)
PTQ1_0_SCALE_OFFSET = tl.constexpr(26)


@triton.jit
def ternary_codes(block_ptrs, pos, mask, IS_PTQ1: tl.constexpr):
    """Decode the signed code of weight ``pos`` in the block at ``block_ptrs``."""
    if IS_PTQ1:
        in_a = pos < 80
        in_b = pos < 120
        byte = tl.where(
            in_a, pos % 16, tl.where(in_b, 16 + (pos - 80) % 8, 24 + (pos - 120) % 2)
        )
        trit = tl.where(
            in_a, pos // 16, tl.where(in_b, (pos - 80) // 8, (pos - 120) // 2)
        )
        pow3 = tl.where(
            trit == 0,
            1,
            tl.where(trit == 1, 3, tl.where(trit == 2, 9, tl.where(trit == 3, 27, 81))),
        )
        packed = tl.load(block_ptrs + byte, mask=mask, other=0).to(tl.int32)
        code = (((packed * pow3) & 0xFF) * 3) >> 8
    else:
        packed = tl.load(block_ptrs + 2 + pos // 4, mask=mask, other=0).to(tl.int32)
        code = (packed >> ((pos % 4) * 2)) & 0x3
    return code - 1


@triton.jit
def ternary_gemm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    m,
    n,
    num_k_blocks,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_ym,
    stride_yn,
    QK: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    IS_PTQ1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, QK)
    m_mask = offs_m < m
    n_mask = offs_n < n

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_block_ptrs = w_ptr + offs_n[:, None] * stride_wn
    scale_ptrs = w_ptr + offs_n * stride_wn + (PTQ1_0_SCALE_OFFSET if IS_PTQ1 else 0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kb in range(num_k_blocks):
        x = tl.load(x_ptrs + kb * QK * stride_xk, mask=m_mask[:, None], other=0.0)
        codes = ternary_codes(
            w_block_ptrs + kb * BLOCK_BYTES, offs_k[None, :], n_mask[:, None], IS_PTQ1
        )
        scales = load_f16_from_u8(scale_ptrs + kb * BLOCK_BYTES, n_mask)
        # Codes are exact in any float dtype, so the per-block dot is exact up
        # to accumulation and the fp16 scale is applied in fp32.
        partial = tl.dot(x, tl.trans(codes.to(x.dtype)))
        acc += partial * scales.to(tl.float32)[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def ggml_gemm_ternary_triton(
    W: torch.Tensor,
    X: torch.Tensor,
    row: int,
    quant_type: int,
) -> torch.Tensor:
    quant_type = int(quant_type)
    if quant_type not in TERNARY_TYPES:
        raise ValueError(f"Unsupported ternary quant type: {quant_type}")
    if not W.is_cuda or not X.is_cuda:
        raise ValueError("Ternary Triton kernels require CUDA tensors")
    if W.dtype is not torch.uint8:
        raise TypeError(f"Ternary weights must be torch.uint8, got {W.dtype}")
    if X.dtype not in TRITON_SUPPORTED_ACTIVATION_DTYPES:
        raise TypeError(
            "Ternary Triton kernels support torch.float16, torch.bfloat16 and "
            f"torch.float32 activations, got {X.dtype}"
        )
    if row != W.shape[0]:
        raise ValueError(
            f"row must match W.shape[0], got row={row}, W.shape[0]={W.shape[0]}"
        )
    qk = BLOCK_QK_BY_TYPE[quant_type]
    block_bytes = BLOCK_BYTES_BY_TYPE[quant_type]
    num_k_blocks, remainder = divmod(W.shape[1], block_bytes)
    if remainder:
        raise ValueError(
            f"Invalid row width {W.shape[1]} bytes for quant type {quant_type}: "
            f"must be divisible by {block_bytes}"
        )
    hidden_size = num_k_blocks * qk
    if X.shape[-1] != hidden_size:
        raise ValueError(
            f"X hidden size {X.shape[-1]} does not match weight width {hidden_size}"
        )

    W = W.contiguous()
    X_2d = X.reshape(-1, hidden_size).contiguous()
    m = X_2d.shape[0]
    Y_2d = torch.empty((m, row), device=X.device, dtype=X.dtype)
    if m == 0:
        return Y_2d.view(*X.shape[:-1], row)

    block_m = 16 if m <= 16 else 64
    block_n = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(row, block_n))
    ternary_gemm_kernel[grid](
        X_2d,
        W,
        Y_2d,
        m,
        row,
        num_k_blocks,
        X_2d.stride(0),
        X_2d.stride(1),
        W.stride(0),
        Y_2d.stride(0),
        Y_2d.stride(1),
        QK=qk,
        BLOCK_BYTES=block_bytes,
        IS_PTQ1=quant_type == GGML_TYPE_PTQ1_0,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return Y_2d.view(*X.shape[:-1], row)
