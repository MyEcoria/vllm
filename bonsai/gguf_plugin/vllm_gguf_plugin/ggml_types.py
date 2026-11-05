# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGML tensor types that released ``gguf`` packages may not know yet.

``Q2_0`` is a mainline llama.cpp type. ``PQ2_0`` and ``PTQ1_0`` are the
PrismML ternary packings used by the Bonsai models. ``GGUFReader`` rejects
tensor types missing from ``GGMLQuantizationType``, so they are added to the
enum and to ``GGML_QUANT_SIZES`` when absent.
"""

import enum

import gguf
from gguf import GGMLQuantizationType

# name -> (type id, elements per block, bytes per block)
EXTRA_GGML_TYPES: dict[str, tuple[int, int, int]] = {
    "Q2_0": (42, 64, 2 + 64 // 4),
    "PQ2_0": (142, 128, 2 + 128 // 4),
    "PTQ1_0": (143, 128, 2 + 2 + 24),
}


def _add_enum_member(enum_cls: type[enum.IntEnum], name: str, value: int) -> None:
    member = int.__new__(enum_cls, value)
    member._name_ = name
    member._value_ = value
    member._sort_order_ = len(enum_cls._member_names_)
    type.__setattr__(enum_cls, name, member)
    enum_cls._member_map_[name] = member
    enum_cls._member_names_.append(name)
    enum_cls._value2member_map_[value] = member


def register_extra_ggml_types() -> None:
    """Make ``gguf`` aware of the types in ``EXTRA_GGML_TYPES``."""
    for name, (value, block_size, type_size) in EXTRA_GGML_TYPES.items():
        member = GGMLQuantizationType._value2member_map_.get(value)
        if member is None:
            if name in GGMLQuantizationType._member_map_:
                raise RuntimeError(
                    f"gguf defines GGML type {name} with an unexpected id "
                    f"{GGMLQuantizationType[name].value}, expected {value}"
                )
            _add_enum_member(GGMLQuantizationType, name, value)
            member = GGMLQuantizationType(value)
        elif member.name != name:
            raise RuntimeError(
                f"gguf maps GGML type id {value} to {member.name}, expected {name}"
            )

        sizes = gguf.GGML_QUANT_SIZES.setdefault(member, (block_size, type_size))
        if sizes != (block_size, type_size):
            raise RuntimeError(
                f"gguf defines {name} with block layout {sizes}, "
                f"expected {(block_size, type_size)}"
            )


register_extra_ggml_types()
