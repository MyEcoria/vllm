# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-side transforms for Hadamard-folded GGUF weights.

PrismML checkpoints (e.g. Ternary Bonsai) store matrices in a rotated basis
described by the ``prism.hadamard.*`` GGUF metadata. A folded weight is
``W_f = W D H``, with ``D`` a fixed diagonal of +-1 signs for the input width
and ``H`` the normalized Sylvester-Hadamard transform applied blockwise along
the input dimension. Since ``H H = I`` and ``D D = I``,
``W x = W_f (H (D x))``. Token-embedding rows are stored in the same latent
basis, so a lookup is mapped back with ``D (H e)``.

Loading such a file without applying the transforms silently produces
garbage, so any metadata this module does not understand is rejected.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import gguf
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

KEY_PREFIX = "prism.hadamard."
SUPPORTED_TRANSFORM = "normalized-sylvester-walsh-hadamard"
SUPPORTED_AXIS = "input-last-dimension"
TOKEN_EMBD = "token_embd.weight"
OUTPUT = "output.weight"


@dataclass(frozen=True, eq=False)
class HadamardSpec:
    """Transform a module applies around its folded weight.

    Attributes:
        block_size: Hadamard block size along the input dimension.
        signs: Full-width +-1 sign vector, or None for identity signs.
        forward: Rotate the matmul input (``D`` then ``H``).
        inverse: Rotate an embedding lookup back (``H`` then ``D``).
        required: Loading fails if no layer claims this weight.
    """

    block_size: int
    signs: torch.Tensor | None
    forward: bool = False
    inverse: bool = False
    required: bool = True

    def same_transform(self, other: HadamardSpec) -> bool:
        return (
            self.block_size == other.block_size
            and self.forward == other.forward
            and self.inverse == other.inverse
            and (
                self.signs is other.signs
                or (
                    self.signs is not None
                    and other.signs is not None
                    and torch.equal(self.signs, other.signs)
                )
            )
        )


@dataclass(frozen=True)
class PrismHadamardConfig:
    """Validated ``prism.hadamard.*`` metadata of one GGUF file."""

    version: int
    block_size: int
    signs: Mapping[int, torch.Tensor] | None
    forward_tensors: frozenset[str]
    inverse_tensors: frozenset[str]
    tensor_widths: Mapping[str, int]
    gdn_v_grouped: bool
    tied_output: bool

    def tensor_spec(self, tensor_name: str) -> HadamardSpec | None:
        forward = tensor_name in self.forward_tensors
        inverse = tensor_name in self.inverse_tensors
        if self.tied_output and tensor_name == TOKEN_EMBD:
            forward = True
        if not (forward or inverse):
            return None
        signs = None
        if self.signs is not None:
            signs = self.signs[self.tensor_widths[tensor_name]]
        return HadamardSpec(self.block_size, signs, forward, inverse)

    def weight_specs(self, name_map: Mapping[str, str]) -> dict[str, HadamardSpec]:
        """Return the transform of every loaded folded tensor, by mapped name."""
        specs: dict[str, HadamardSpec] = {}
        for tensor_name in sorted(self.forward_tensors | self.inverse_tensors):
            mapped = name_map.get(tensor_name)
            if mapped is None:
                continue
            spec = self.tensor_spec(tensor_name)
            assert spec is not None
            specs[mapped] = spec
        if self.tied_output and TOKEN_EMBD in name_map:
            # An lm_head that keeps its own module when tied (unquantized
            # tables) needs the forward transform; the GGUF embedding method
            # instead reuses embed_tokens, which already carries it.
            specs["lm_head.weight"] = HadamardSpec(
                self.block_size,
                specs[name_map[TOKEN_EMBD]].signs,
                forward=True,
                required=False,
            )
        return specs


def _field(reader: gguf.GGUFReader, key: str, default=None):
    field = reader.get_field(KEY_PREFIX + key)
    return default if field is None else field.contents()


def read_prism_hadamard_config(
    reader: gguf.GGUFReader,
) -> PrismHadamardConfig | None:
    """Parse and validate the ``prism.hadamard.*`` metadata of a GGUF file.

    Returns:
        The parsed config, or None when the file has no Hadamard metadata.

    Raises:
        ValueError: The metadata is malformed or uses an unsupported variant.
    """
    version = _field(reader, "version")
    tied_output = bool(_field(reader, "tied_output", False))
    if version is None:
        if any(name.startswith(KEY_PREFIX) for name in reader.fields):
            raise ValueError("prism.hadamard metadata without prism.hadamard.version")
        return None
    if version not in (1, 2):
        raise ValueError(f"Unsupported prism.hadamard.version: {version}")
    if (version == 2) != tied_output:
        raise ValueError(
            "prism.hadamard version 2 requires tied_output=true; version 1 forbids it"
        )

    block_size = _field(reader, "block_size")
    if not block_size or block_size & (block_size - 1):
        raise ValueError(f"Invalid prism.hadamard.block_size: {block_size}")
    transform = _field(reader, "transform")
    if transform != SUPPORTED_TRANSFORM:
        raise ValueError(f"Unsupported prism.hadamard.transform: {transform}")
    axis = _field(reader, "axis")
    if axis != SUPPORTED_AXIS:
        raise ValueError(f"Unsupported prism.hadamard.axis: {axis}")

    sign_mode = _field(reader, "sign_mode")
    signs: dict[int, torch.Tensor] | None
    if sign_mode == "identity":
        signs = None
    elif sign_mode == "explicit":
        signs = _parse_signs(
            _field(reader, "sign_widths", []),
            _field(reader, "sign_values", []),
            block_size,
        )
    else:
        raise ValueError(f"Unsupported prism.hadamard.sign_mode: {sign_mode}")

    forward_tensors = frozenset(_field(reader, "weight_names", []))
    inverse_tensors = frozenset(_field(reader, "inverse_weight_names", []))
    if not forward_tensors:
        raise ValueError("prism.hadamard.weight_names is empty")
    if not inverse_tensors <= {TOKEN_EMBD}:
        raise ValueError(
            "Only token_embd.weight may be an inverse-transformed lookup table, "
            f"got {sorted(inverse_tensors)}"
        )
    if forward_tensors & inverse_tensors:
        raise ValueError("prism.hadamard forward and inverse tensors overlap")

    tensor_shapes = {tensor.name: tensor.shape for tensor in reader.tensors}
    if tied_output and (OUTPUT in tensor_shapes or TOKEN_EMBD not in inverse_tensors):
        raise ValueError(
            "prism.hadamard.tied_output requires a latent token_embd.weight and "
            "no output.weight"
        )
    if (
        not tied_output
        and TOKEN_EMBD in inverse_tensors
        and OUTPUT not in tensor_shapes
    ):
        raise ValueError("A tied Hadamard output requires prism.hadamard version 2")

    tensor_widths: dict[str, int] = {}
    for name in forward_tensors | inverse_tensors:
        shape = tensor_shapes.get(name)
        if shape is None:
            raise ValueError(f"prism.hadamard references missing tensor {name}")
        if len(shape) != 2:
            raise ValueError(f"prism.hadamard tensor {name} is not a matrix")
        # GGUF shapes list the input dimension first.
        width = int(shape[0])
        if width % block_size:
            raise ValueError(
                f"prism.hadamard block size {block_size} does not divide the "
                f"input dimension {width} of {name}"
            )
        if signs is not None and width not in signs:
            raise ValueError(f"prism.hadamard has no sign vector for {name}")
        tensor_widths[name] = width

    return PrismHadamardConfig(
        version=version,
        block_size=block_size,
        signs=signs,
        forward_tensors=forward_tensors,
        inverse_tensors=inverse_tensors,
        tensor_widths=tensor_widths,
        gdn_v_grouped=bool(_field(reader, "gdn_v_grouped", False)),
        tied_output=tied_output,
    )


def _parse_signs(
    widths: list[int], values: list[int], block_size: int
) -> dict[int, torch.Tensor]:
    if not widths:
        raise ValueError(
            "prism.hadamard.sign_mode is explicit but sign_widths is empty"
        )
    signs: dict[int, torch.Tensor] = {}
    offset = 0
    for width in widths:
        if width <= 0 or width % block_size or offset + width > len(values):
            raise ValueError(f"Invalid prism.hadamard sign width: {width}")
        if width in signs:
            raise ValueError(f"Duplicate prism.hadamard sign width: {width}")
        vector = torch.tensor(values[offset : offset + width], dtype=torch.float32)
        if not torch.all(vector.abs() == 1):
            raise ValueError("prism.hadamard sign values must be +/-1")
        signs[width] = vector
        offset += width
    if offset != len(values):
        raise ValueError("prism.hadamard.sign_values length mismatch")
    return signs


@cache
def load_prism_hadamard_config(path: str | Path) -> PrismHadamardConfig | None:
    return read_prism_hadamard_config(gguf.GGUFReader(str(path)))


def sylvester_hadamard(
    size: int, dtype: torch.dtype = torch.float32, device: torch.device | None = None
) -> torch.Tensor:
    """Normalized Sylvester-Hadamard matrix; symmetric and its own inverse."""
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0
        )
    return (matrix / math.sqrt(size)).to(dtype=dtype, device=device)


_shared_matrices: dict[tuple[int, torch.dtype, torch.device], torch.Tensor] = {}


def _use_hadacore(dtype: torch.dtype, device: torch.device) -> bool:
    from vllm.platforms import current_platform

    return (
        device.type == "cuda"
        and current_platform.is_cuda()
        and dtype in (torch.float16, torch.bfloat16)
        and hasattr(torch.ops._C, "hadacore_transform")
    )


def attach_hadamard(
    layer: torch.nn.Module,
    spec: HadamardSpec,
    *,
    offset: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Register the transform state of ``spec`` on ``layer``.

    ``offset`` and ``width`` select the input slice owned by this rank; they
    must not split a Hadamard block.
    """
    block = spec.block_size
    if offset % block or width % block:
        raise NotImplementedError(
            f"The local input slice [{offset}, {offset + width}) of a "
            f"Hadamard-folded layer splits a {block}-wide rotation block; use a "
            "tensor-parallel size that keeps whole blocks on each rank"
        )
    signs = None
    if spec.signs is not None:
        if offset + width > spec.signs.numel():
            raise ValueError(
                f"Hadamard sign vector of width {spec.signs.numel()} does not "
                f"cover input slice [{offset}, {offset + width})"
            )
        signs = spec.signs[offset : offset + width].to(dtype=dtype, device=device)
    matrix = None
    if not _use_hadacore(dtype, device):
        key = (block, dtype, device)
        matrix = _shared_matrices.get(key)
        if matrix is None:
            matrix = sylvester_hadamard(block, dtype, device)
            _shared_matrices[key] = matrix
    layer.hadamard_block_size = block
    layer.register_buffer("hadamard_signs", signs, persistent=False)
    layer.register_buffer("hadamard_matrix", matrix, persistent=False)


def _blockwise_hadamard(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    blocks = x.reshape(-1, layer.hadamard_block_size)
    if layer.hadamard_matrix is None:
        from vllm import _custom_ops as ops

        out = ops.hadacore_transform(blocks, inplace=False)
    else:
        out = blocks @ layer.hadamard_matrix
    return out.view(x.shape)


def hadamard_forward(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map an activation into the folded basis: ``H (D x)``."""
    if layer.hadamard_signs is not None:
        x = x * layer.hadamard_signs
    return _blockwise_hadamard(layer, x)


def hadamard_inverse(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Map a latent embedding row back: ``D (H x)``."""
    x = _blockwise_hadamard(layer, x)
    if layer.hadamard_signs is not None:
        x = x * layer.hadamard_signs
    return x
