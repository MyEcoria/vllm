# SPDX-License-Identifier: Apache-2.0
"""Fast small-batch path for the Q8_0 layers (the grafted MTP head of bonsai). BONSAI_MTP_W8=1.

At load, each all-Q8_0 GGUF linear is rotated by the same signed Hadamard transform T that the
ternary activation quantizer applies (W' = W T^T, so W x = W' (T x)) and requantized to int8 with
one fp16 scale per 128 columns, in the quantizer's permuted column order. The int8 weights are
then directly the WMMA B operand (w8_m16_kernel): 3-6x faster than the Q8_0 MMVQ/MMQ kernels for
the 4-16 row speculative batches. The rotation spreads outliers; the head only drafts tokens, so
the ~0.9 % relative error only touches the acceptance rate, not the output.
"""
from __future__ import annotations

import torch

BLOCK = 1024
W8_MAX_M = 64  # beyond: dequantize + hipBLASLt (prompt-size batches)
Q8_0 = 8
_T: dict = {}


def _act_quant(x: torch.Tensor):
    m, k = x.shape
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    xs = torch.empty((m, k // 128), dtype=torch.float32, device=x.device)
    xsx = torch.empty_like(xs)
    torch.ops._C_gguf.ternary_act_quant(x, _ones(k, x.device), BLOCK, xq, xs, xsx)
    return xq, xs


_ONES: dict = {}


def _ones(k, device):
    t = _ONES.get((k, device))
    if t is None:
        t = _ONES[(k, device)] = torch.ones(k, dtype=torch.float32, device=device)
    return t


def _transform(device):
    """T (x' = T x) for one block, recovered from the quantizer with unit vectors (exact: after the
    normalized transform all entries of T e_k have the same magnitude)."""
    T = _T.get(device)
    if T is None:
        eye = torch.eye(BLOCK, device=device, dtype=torch.bfloat16)
        xq, xs = _act_quant(eye)
        T = (xq.float().view(BLOCK, BLOCK // 128, 128) * xs.view(BLOCK, BLOCK // 128, 1)).view(BLOCK, BLOCK).T
        _T[device] = T = T.contiguous()
    return T


def _convert(weight: torch.Tensor, n: int, k: int):
    from . import ops

    T = _transform(weight.device)
    w8 = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    s = torch.empty((n, k // 128), dtype=torch.float16, device=weight.device)
    step = max(1, (64 << 20) // (k * 4))
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        W = ops.ggml_dequantize(weight[r0:r1].contiguous(), Q8_0, r1 - r0, k, torch.float16).float()
        Wp = (W.view(r1 - r0, k // BLOCK, BLOCK) @ T.T).view(r1 - r0, k // 128, 128)
        sc = (Wp.abs().amax(-1) / 127).clamp_min(1e-12)
        w8[r0:r1] = torch.round(Wp / sc[..., None]).clamp(-127, 127).to(torch.int8).view(r1 - r0, k)
        s[r0:r1] = sc.half()
    return w8, s


def _w8_linear(x: torch.Tensor, w8: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    from .bonsai_int4_attn import ext

    m = x.shape[0]
    n = w8.shape[0]
    if m == 0:
        return x.new_empty((0, n))
    xq, xs = _act_quant(x)
    if m <= W8_MAX_M:
        return ext().w8_gemm(w8, s, xq, xs, 4 if n <= 8192 else 2, x.dtype == torch.bfloat16)
    k = x.shape[1]
    xd = (xq.view(m, k // 128, 128).to(x.dtype) * xs.view(m, k // 128, 1).to(x.dtype)).view(m, k)
    return xd @ ext().w8_dequant(w8, s).T.to(x.dtype)


def _w8_linear_fake(x, w8, s):
    return x.new_empty((x.shape[0], w8.shape[0]))


def patch():
    from vllm.utils.torch_utils import direct_register_custom_op

    from .quantization.linear import GGUFLinearMethod

    direct_register_custom_op(op_name="bonsai_w8_linear", op_func=_w8_linear, fake_impl=_w8_linear_fake)
    op = torch.ops.vllm.bonsai_w8_linear

    orig_post = GGUFLinearMethod.process_weights_after_loading
    orig_apply = GGUFLinearMethod.apply

    def process_weights_after_loading(self, layer):
        orig_post(self, layer)
        if getattr(layer, "ternary_q", None) is not None or not layer.weight.is_cuda:
            return
        wt = layer.weight_type
        types = set(wt.shard_weight_type.values()) or {wt.weight_type}
        if types != {Q8_0} or self.layout is not None:
            return
        n = layer.weight.shape[0]
        k = layer.weight.shape[1] // 34 * 32
        if k % BLOCK:
            return
        w8, s = _convert(layer.weight.data, n, k)
        layer.register_buffer("b_w8", w8, persistent=False)
        layer.register_buffer("b_w8s", s, persistent=False)
        layer.weight.data = torch.empty(0, dtype=torch.uint8, device=w8.device)
        torch.cuda.empty_cache()

    def apply(self, layer, x, bias=None):
        if getattr(layer, "b_w8", None) is None:
            return orig_apply(self, layer, x, bias)
        x = self._forward_input(layer, x)
        lead = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        out = op(x, layer.b_w8, layer.b_w8s)
        if bias is not None:
            out.add_(bias)
        return out.view(*lead, -1)

    GGUFLinearMethod.process_weights_after_loading = process_weights_after_loading
    GGUFLinearMethod.apply = apply
