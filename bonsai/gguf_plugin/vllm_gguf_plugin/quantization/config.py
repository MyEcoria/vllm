# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix

from .utils import is_layer_skipped_gguf

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

    from ..hadamard import HadamardSpec
    from .layout import GGUFLinearLayout


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []
        self.linear_layouts: dict[str, GGUFLinearLayout] = {}
        # Keyed by weight name (``<module>.weight``).
        self.hadamard_specs: dict[str, HadamardSpec] = {}
        self._claimed_hadamard_weights: set[str] = set()

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        del config
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from .fused_moe import GGUFMoEMethod
        from .hadamard_methods import (
            HadamardUnquantizedEmbeddingMethod,
            HadamardUnquantizedLinearMethod,
        )
        from .linear import GGUFLinearMethod
        from .vocal_embeds import GGUFEmbeddingMethod

        if isinstance(layer, LinearBase):
            hadamard = self._get_hadamard_spec(prefix)
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                if hadamard is not None:
                    return HadamardUnquantizedLinearMethod(hadamard)
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(
                self,
                layout=self.linear_layouts.get(prefix),
                hadamard=hadamard,
            )
        if isinstance(layer, VocabParallelEmbedding):
            hadamard = self._get_hadamard_spec(prefix)
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                if hadamard is not None:
                    return HadamardUnquantizedEmbeddingMethod(hadamard)
                return UnquantizedEmbeddingMethod()
            return GGUFEmbeddingMethod(self, hadamard=hadamard)
        if isinstance(layer, RoutedExperts):
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def register_linear_layouts(
        self,
        layouts: Mapping[str, "GGUFLinearLayout"],
        prefix: str = "",
    ) -> None:
        """Register GGUF linear layouts before model initialization."""
        self.linear_layouts.update(
            (maybe_prefix(prefix, module_name), layout)
            for module_name, layout in layouts.items()
        )

    def register_hadamard_specs(
        self,
        specs: Mapping[str, "HadamardSpec"],
        prefix: str = "",
    ) -> None:
        """Register Hadamard-folded weights before model initialization."""
        self.hadamard_specs.update(
            (maybe_prefix(prefix, weight_name), spec)
            for weight_name, spec in specs.items()
        )

    def _get_hadamard_spec(self, prefix: str) -> "HadamardSpec | None":
        """Return the transform of the module at ``prefix``, if folded.

        Fused modules (e.g. ``qkv_proj``) must have all or none of their
        shards folded, with one shared transform.
        """
        if not self.hadamard_specs:
            return None
        parent, _, proj_name = prefix.rpartition(".")
        shard_prefixes = [
            maybe_prefix(parent, shard)
            for shard in self.packed_modules_mapping.get(proj_name, [])
        ]
        weight_names = [f"{name}.weight" for name in (prefix, *shard_prefixes)]
        specs = [self.hadamard_specs.get(name) for name in weight_names]
        found = [spec for spec in specs if spec is not None]
        if not found:
            return None
        shard_specs = specs[1:]
        if any(spec is None for spec in shard_specs) and any(
            spec is not None for spec in shard_specs
        ):
            raise ValueError(
                f"Only some shards of {prefix} are Hadamard-folded; all shards "
                "of a fused layer must share one input transform"
            )
        if not all(spec.same_transform(found[0]) for spec in found[1:]):
            raise ValueError(f"Shards of {prefix} use different Hadamard transforms")
        self._claimed_hadamard_weights.update(
            name for name, spec in zip(weight_names, specs) if spec is not None
        )
        return found[0]

    def verify_hadamard_specs_claimed(self) -> None:
        """Fail if a folded weight's module never received its transform."""
        required = {name for name, spec in self.hadamard_specs.items() if spec.required}
        missing = required - self._claimed_hadamard_weights
        if missing:
            raise ValueError(
                "Hadamard-folded weights were not matched to any vLLM layer, so "
                f"their activation transform would be skipped: {sorted(missing)}"
            )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )
        if self.linear_layouts:
            layouts = self.linear_layouts
            mapped_names = hf_to_vllm_mapper.apply_list(list(layouts))
            self.linear_layouts = dict(zip(mapped_names, layouts.values(), strict=True))
        if self.hadamard_specs:
            self.hadamard_specs = _map_specs(hf_to_vllm_mapper, self.hadamard_specs)


def _map_specs(
    mapper: "WeightsMapper", specs: Mapping[str, "HadamardSpec"]
) -> dict[str, "HadamardSpec"]:
    mapped: dict[str, HadamardSpec] = {}
    for name, spec in specs.items():
        new_name = mapper.map_name(name)
        if new_name is None:
            raise ValueError(f"Hadamard-folded weight {name} is dropped by the model")
        previous = mapped.setdefault(new_name, spec)
        if not previous.same_transform(spec):
            raise ValueError(f"Conflicting Hadamard transforms for {new_name}")
    return mapped
