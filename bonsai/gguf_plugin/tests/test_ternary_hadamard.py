# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ternary GGUF types and ``prism.hadamard`` transforms (PrismML Bonsai)."""

from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.hadamard import (
    HadamardSpec,
    attach_hadamard,
    hadamard_forward,
    hadamard_inverse,
    read_prism_hadamard_config,
    sylvester_hadamard,
)
from vllm_gguf_plugin.quantization import GGUFConfig, fused_mul_mat_gguf, ternary_fast

TERNARY_TYPES = [
    GGMLQuantizationType.Q2_0,
    GGMLQuantizationType.PQ2_0,
    GGMLQuantizationType.PTQ1_0,
]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need CUDA"
)


def _reference_dequantize(raw: torch.Tensor, quant_type: int) -> torch.Tensor:
    """Decode following llama.cpp's ``dequantize_row_*`` reference loops."""
    qk, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    rows = raw.shape[0]
    blocks = raw.reshape(-1, type_size).to(torch.int64)
    out = torch.empty(blocks.shape[0], qk)
    if quant_type == GGMLQuantizationType.PTQ1_0:
        scale = blocks[:, 26:28].to(torch.uint8).contiguous().view(torch.float16)
        trits = []
        for start, count, n_trits in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
            for n in range(n_trits):
                q = (blocks[:, start : start + count] * 3**n) & 0xFF
                trits.append((q * 3) >> 8)
        codes = torch.cat(trits, dim=1)
    else:
        scale = blocks[:, 0:2].to(torch.uint8).contiguous().view(torch.float16)
        shifts = torch.arange(4) * 2
        codes = (blocks[:, 2:, None] >> shifts) & 0x3
        codes = codes.reshape(blocks.shape[0], qk)
    out = (codes - 1).float() * scale.float()
    return out.reshape(rows, -1)


def _random_blocks(quant_type: int, rows: int, cols: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    qk, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    num_blocks = rows * cols // qk
    raw = torch.randint(
        0, 256, (num_blocks, type_size), dtype=torch.uint8, generator=generator
    )
    scales = torch.rand(num_blocks, generator=generator) * 0.02 + 0.001
    scale_offset = 26 if quant_type == GGMLQuantizationType.PTQ1_0 else 0
    raw[:, scale_offset : scale_offset + 2] = (
        scales.to(torch.float16).view(torch.uint8).view(num_blocks, 2)
    )
    return raw.reshape(rows, -1)


def test_gguf_reader_accepts_ternary_types(tmp_path: Path):
    path = tmp_path / "ternary.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    for index, quant_type in enumerate(TERNARY_TYPES):
        raw = _random_blocks(quant_type, rows=3, cols=256).numpy()
        writer.add_tensor(f"t{index}.weight", raw, raw_dtype=quant_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    tensors = gguf.GGUFReader(path).tensors
    assert [t.tensor_type for t in tensors] == TERNARY_TYPES
    assert all(t.shape.tolist() == [256, 3] for t in tensors)


def _write_prism_gguf(path: Path, **overrides) -> None:
    """Write a tiny GGUF with prism.hadamard metadata (block 4, width 8)."""
    metadata = {
        "version": 1,
        "block_size": 4,
        "transform": "normalized-sylvester-walsh-hadamard",
        "axis": "input-last-dimension",
        "sign_mode": "explicit",
        "weight_names": ["blk.0.ffn_up.weight", "output.weight"],
        "inverse_weight_names": ["token_embd.weight"],
        "sign_widths": [8],
        "sign_values": [1, -1, -1, 1, 1, 1, -1, 1],
        "gdn_v_grouped": True,
    }
    metadata.update(overrides)
    writer = gguf.GGUFWriter(path, "qwen35")
    for key, value in metadata.items():
        if value is None:
            continue
        name = f"prism.hadamard.{key}"
        if isinstance(value, bool):
            writer.add_bool(name, value)
        elif isinstance(value, int):
            writer.add_uint32(name, value)
        elif isinstance(value, str):
            writer.add_string(name, value)
        else:
            writer.add_array(name, value)
    for name in ("blk.0.ffn_up.weight", "output.weight", "token_embd.weight"):
        writer.add_tensor(name, np.zeros((16, 8), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_read_prism_hadamard_config(tmp_path: Path):
    path = tmp_path / "prism.gguf"
    _write_prism_gguf(path)
    config = read_prism_hadamard_config(gguf.GGUFReader(path))

    assert config is not None
    assert config.block_size == 4
    assert config.gdn_v_grouped and not config.tied_output
    specs = config.weight_specs(
        {
            "blk.0.ffn_up.weight": "model.layers.0.mlp.up_proj.weight",
            "output.weight": "lm_head.weight",
            "token_embd.weight": "model.embed_tokens.weight",
        }
    )
    up, head, embed = (
        specs["model.layers.0.mlp.up_proj.weight"],
        specs["lm_head.weight"],
        specs["model.embed_tokens.weight"],
    )
    assert up.forward and not up.inverse
    assert head.forward and not head.inverse
    assert embed.inverse and not embed.forward
    assert up.signs.tolist() == [1, -1, -1, 1, 1, 1, -1, 1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 3},
        {"version": 2},  # version 2 requires a tied output
        {"transform": "random-orthogonal"},
        {"axis": "output-first-dimension"},
        {"sign_values": [1, -1, 0, 1, 1, 1, -1, 1]},
        {"sign_widths": [16]},
        {"block_size": 3},
        {"inverse_weight_names": ["output.weight"]},
        {"weight_names": ["blk.9.ffn_up.weight"]},
    ],
)
def test_read_prism_hadamard_config_rejects_unsupported(tmp_path: Path, overrides):
    path = tmp_path / "prism.gguf"
    _write_prism_gguf(path, **overrides)
    with pytest.raises(ValueError):
        read_prism_hadamard_config(gguf.GGUFReader(path))


def test_read_prism_hadamard_config_absent(tmp_path: Path):
    path = tmp_path / "plain.gguf"
    writer = gguf.GGUFWriter(path, "qwen35")
    writer.add_tensor("output.weight", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    assert read_prism_hadamard_config(gguf.GGUFReader(path)) is None


def _blockwise(x: torch.Tensor, block_size: int) -> torch.Tensor:
    matrix = sylvester_hadamard(block_size, torch.float64)
    return (x.reshape(*x.shape[:-1], -1, block_size) @ matrix).reshape(x.shape)


@pytest.mark.parametrize("use_signs", [True, False])
def test_hadamard_transforms_undo_folding(use_signs: bool):
    """Folded weights on transformed inputs reproduce the original layer."""
    torch.manual_seed(0)
    block_size, width = 16, 64
    signs = torch.randint(0, 2, (width,)).double() * 2 - 1 if use_signs else None
    sign_diag = signs if use_signs else torch.ones(width, dtype=torch.float64)
    weight = torch.randn(32, width, dtype=torch.float64)
    # W_f = W D H, and an embedding row e is stored as l = H D e.
    folded = _blockwise(weight * sign_diag, block_size)

    layer = torch.nn.Module()
    spec = HadamardSpec(block_size, signs, forward=True, inverse=True)
    attach_hadamard(
        layer,
        spec,
        offset=0,
        width=width,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    x = torch.randn(5, width, dtype=torch.float64)
    torch.testing.assert_close(hadamard_forward(layer, x) @ folded.T, x @ weight.T)
    rows = torch.randn(3, width, dtype=torch.float64)
    latent = _blockwise(rows * sign_diag, block_size)
    torch.testing.assert_close(hadamard_inverse(layer, latent), rows)


def test_hadamard_row_parallel_slice():
    """A rank owning whole blocks transforms its slice with its own signs."""
    torch.manual_seed(0)
    signs = torch.randint(0, 2, (16,)).double() * 2 - 1
    spec = HadamardSpec(4, signs, forward=True)
    x = torch.randn(2, 16, dtype=torch.float64)

    full = torch.nn.Module()
    attach_hadamard(
        full, spec, offset=0, width=16, dtype=torch.float64, device=torch.device("cpu")
    )
    shard = torch.nn.Module()
    attach_hadamard(
        shard, spec, offset=8, width=8, dtype=torch.float64, device=torch.device("cpu")
    )
    torch.testing.assert_close(
        hadamard_forward(shard, x[:, 8:]), hadamard_forward(full, x)[:, 8:]
    )

    with pytest.raises(NotImplementedError):
        attach_hadamard(
            torch.nn.Module(),
            spec,
            offset=6,
            width=6,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )


def _config_with_specs(specs: dict[str, HadamardSpec]) -> GGUFConfig:
    config = GGUFConfig()
    config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    config.register_hadamard_specs(specs)
    return config


def test_gguf_config_resolves_fused_hadamard_layers():
    spec = HadamardSpec(4, torch.ones(8), forward=True)
    config = _config_with_specs(
        {
            "layers.0.attn.q_proj.weight": spec,
            "layers.0.attn.k_proj.weight": spec,
            "layers.0.attn.v_proj.weight": spec,
            "layers.0.mlp.down_proj.weight": spec,
        }
    )

    assert config._get_hadamard_spec("layers.0.attn.qkv_proj") is spec
    assert config._get_hadamard_spec("layers.0.attn.o_proj") is None
    # A folded weight whose layer never asks for its transform must fail.
    with pytest.raises(ValueError, match="mlp.down_proj"):
        config.verify_hadamard_specs_claimed()
    assert config._get_hadamard_spec("layers.0.mlp.down_proj") is spec
    config.verify_hadamard_specs_claimed()

    partial = _config_with_specs({"layers.0.attn.q_proj.weight": spec})
    with pytest.raises(ValueError, match="Only some shards"):
        partial._get_hadamard_spec("layers.0.attn.qkv_proj")


@requires_cuda
@pytest.mark.parametrize("quant_type", TERNARY_TYPES)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_ternary_dequantize(quant_type: int, dtype: torch.dtype):
    raw = _random_blocks(quant_type, rows=48, cols=1024)
    expected = _reference_dequantize(raw, quant_type)
    out = ops.ggml_dequantize(raw.cuda(), quant_type, 48, 1024, dtype)
    torch.testing.assert_close(out.float().cpu(), expected.to(dtype).float())


@requires_cuda
@pytest.mark.parametrize("quant_type", TERNARY_TYPES)
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 7, 83])
@torch.inference_mode()
def test_ternary_mul_mat(quant_type: int, dtype: torch.dtype, num_tokens: int):
    """Covers the decode-sized path that other types send to MMVQ kernels."""
    rows, cols = 320, 1024
    raw = _random_blocks(quant_type, rows, cols)
    weight = _reference_dequantize(raw, quant_type).cuda()
    x = torch.randn(num_tokens, cols, device="cuda", dtype=dtype)

    out = fused_mul_mat_gguf(x, raw.cuda(), int(quant_type))
    expected = x.float() @ weight.T
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


def _paired_q2_0_scales(raw: torch.Tensor) -> torch.Tensor:
    """Give both 64-weight halves of each 128-weight group the same scale."""
    blocks = raw.reshape(-1, 2, 18).clone()
    blocks[:, 1, :2] = blocks[:, 0, :2]
    return blocks.reshape(raw.shape)


@requires_cuda
@pytest.mark.skipif(
    not ternary_fast.kernels_available(), reason="needs the compiled ternary kernels"
)
@pytest.mark.parametrize("quant_type", TERNARY_TYPES)
@pytest.mark.parametrize("block_size", [512, 1024])
@pytest.mark.parametrize("num_tokens", [1, 3, 8, 9, 200])
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_ternary_fast_linear(
    quant_type: int, block_size: int, num_tokens: int, dtype: torch.dtype
):
    """Repacked weights + rotated int8 activations match the folded layer."""
    rows, cols = 192, 2048
    raw = _random_blocks(quant_type, rows, cols, seed=1)
    if quant_type == GGMLQuantizationType.Q2_0:
        raw = _paired_q2_0_scales(raw)
    weight = _reference_dequantize(raw, quant_type).double()
    packed = ternary_fast.repack_ternary(raw.cuda(), int(quant_type))
    assert packed is not None
    q, s = packed

    generator = torch.Generator().manual_seed(2)
    signs = torch.randint(0, 2, (cols,), generator=generator).float() * 2 - 1
    x = torch.randn(num_tokens, cols, generator=generator).to(dtype)
    out = ternary_fast.ternary_linear(x.cuda(), q, s, signs.cuda(), block_size).double()
    expected = _blockwise(x.double() * signs.double(), block_size) @ weight.T
    rel = (out.cpu() - expected).norm() / expected.norm()
    assert out.dtype == torch.float64 and rel < 2e-2


@requires_cuda
def test_ternary_repack_rejects_unpaired_q2_0_scales():
    """Q2_0 rows whose 64-weight halves differ in scale keep the GGUF path."""
    raw = _random_blocks(GGMLQuantizationType.Q2_0, 4, 1024)
    assert ternary_fast.repack_ternary(raw.cuda(), GGMLQuantizationType.Q2_0) is None
