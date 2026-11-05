# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization methods that apply ``prism.hadamard`` transforms."""

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from ..hadamard import (
    HadamardSpec,
    attach_hadamard,
    hadamard_forward,
    hadamard_inverse,
)


class HadamardMixin:
    """Applies an optional ``HadamardSpec`` around a layer's weight."""

    hadamard: HadamardSpec | None = None

    def _record_hadamard_input(
        self, input_size_per_partition: int, input_size: int
    ) -> None:
        self._hadamard_input = (input_size_per_partition, input_size)

    def _attach_hadamard(
        self, layer: torch.nn.Module, dtype: torch.dtype, device: torch.device
    ) -> None:
        if self.hadamard is None:
            return
        width, full_width = self._hadamard_input
        offset = 0
        if width != full_width:
            offset = get_tensor_model_parallel_rank() * width
        attach_hadamard(
            layer,
            self.hadamard,
            offset=offset,
            width=width,
            dtype=dtype,
            device=device,
        )

    def _forward_input(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard is not None and self.hadamard.forward:
            return hadamard_forward(layer, x)
        return x

    def _inverse_output(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard is not None and self.hadamard.inverse:
            return hadamard_inverse(layer, x)
        return x


@register_weight_loader_v2_supported_method
class HadamardUnquantizedLinearMethod(HadamardMixin, UnquantizedLinearMethod):
    """Unquantized linear method for a Hadamard-folded weight."""

    def __init__(self, hadamard: HadamardSpec) -> None:
        super().__init__()
        self.hadamard = hadamard

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._record_hadamard_input(input_size_per_partition, input_size)
        self._params_dtype = params_dtype
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight.device
        super().process_weights_after_loading(layer)
        self._attach_hadamard(layer, self._params_dtype, device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply(layer, self._forward_input(layer, x), bias)


class HadamardUnquantizedEmbeddingMethod(HadamardMixin, UnquantizedEmbeddingMethod):
    """Unquantized embedding / LM head method for a Hadamard-folded table."""

    def __init__(self, hadamard: HadamardSpec) -> None:
        super().__init__()
        self.hadamard = hadamard

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self._record_hadamard_input(input_size_per_partition, input_size)
        self._params_dtype = params_dtype
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.weight.device
        super().process_weights_after_loading(layer)
        self._attach_hadamard(layer, self._params_dtype, device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply(layer, self._forward_input(layer, x), bias)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return self._inverse_output(layer, super().embedding(layer, input_))
