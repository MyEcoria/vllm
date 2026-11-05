"""Int8 GEMM over repacked ternary weights (prefill path).

See ``csrc/ternary/ternary_kernels.cu`` for the byte-plane weight layout and the
permuted int8 activation layout produced by ``ternary_act_quant``. Shifting the
128 weight bytes of a chunk by ``2 * i`` yields the codes of scale group ``i``
in the same order as the 128 activations of that group, so every group is one
``128``-deep int8 dot.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def ternary_packed_gemm_kernel(
    xq_ptr,
    xs_ptr,
    q_ptr,
    s_ptr,
    y_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_n
    group_id = pid // width
    first_m = group_id * GROUP_M
    group_size = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % group_size
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    m_mask = offs_m < M
    n_mask = offs_n < N
    groups = K // 128

    x_ptrs = xq_ptr + offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
    w_ptrs = q_ptr + offs_n[None, :].to(tl.int64) * (K // 4) + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g in range(0, groups):
        x = tl.load(x_ptrs + g * 128, mask=m_mask[:, None], other=0)
        packed = tl.load(w_ptrs + (g // 4) * 128, mask=n_mask[None, :], other=0)
        w = (((packed.to(tl.int32) >> ((g % 4) * 2)) & 3) - 1).to(tl.int8)
        part = tl.dot(x, w)
        xs = tl.load(xs_ptr + offs_m * groups + g, mask=m_mask, other=0.0)
        ws = tl.load(s_ptr + offs_n * groups + g, mask=n_mask, other=0.0)
        acc += part.to(tl.float32) * xs[:, None] * ws.to(tl.float32)[None, :]

    y_ptrs = y_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(
        y_ptrs,
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def ternary_packed_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    m, k = xq.shape
    n = q.shape[0]
    y = torch.empty((m, n), device=xq.device, dtype=dtype)
    if m == 0:
        return y
    block_m, block_n = 128, 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    ternary_packed_gemm_kernel[grid](
        xq,
        xs,
        q,
        s,
        y,
        m,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=8,
        num_warps=8,
        num_stages=1,
    )
    return y
