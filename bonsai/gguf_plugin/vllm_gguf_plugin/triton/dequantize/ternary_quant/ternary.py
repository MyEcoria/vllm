import torch
import triton
import triton.language as tl

from ...gemm.ternary_quant.ternary import PTQ1_0_SCALE_OFFSET, ternary_codes
from ...gemm.utils import (
    BLOCK_BYTES_BY_TYPE,
    BLOCK_QK_BY_TYPE,
    GGML_TYPE_PTQ1_0,
    load_f16_from_u8,
)
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def ternary_dequantize_kernel(
    w_ptr,
    y_ptr,
    total,
    QK: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    IS_PTQ1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // QK) * BLOCK_BYTES
    code = ternary_codes(block_ptrs, offs % QK, mask, IS_PTQ1)
    scale_ptrs = block_ptrs + (PTQ1_0_SCALE_OFFSET if IS_PTQ1 else 0)
    d = load_f16_from_u8(scale_ptrs, mask).to(tl.float32)
    tl.store(y_ptr + offs, code.to(tl.float32) * d, mask=mask)


def ggml_dequantize_ternary_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None,
    quant_type: int,
) -> torch.Tensor:
    quant_type = int(quant_type)
    return run_dequantize_kernel(
        ternary_dequantize_kernel,
        W,
        m,
        n,
        dtype,
        quant_type,
        extra_args=(
            BLOCK_QK_BY_TYPE[quant_type],
            BLOCK_BYTES_BY_TYPE[quant_type],
            quant_type == GGML_TYPE_PTQ1_0,
        ),
    )
