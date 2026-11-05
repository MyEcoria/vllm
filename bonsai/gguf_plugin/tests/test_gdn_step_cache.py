# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the per-step GDN prefill-state cache."""

import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    gather_prefill_initial_state,
)


def _reference(ssm_state, idx, has_init):
    """Original per-layer logic: gather then boolean-mask zeroing."""
    out = ssm_state[idx]
    out[~has_init, ...] = 0
    return out


def test_cached_mask_matches_reference_across_layers():
    """One mask built per step is reused by all layers, outputs identical."""
    idx = torch.tensor([3, 0, 2])
    has_init = torch.tensor([True, False, True])
    mask4d = has_init.view(-1, 1, 1, 1)  # built once per step
    for layer in range(4):
        torch.manual_seed(layer)
        state = torch.randn(5, 2, 4, 4)
        got = gather_prefill_initial_state(state, idx, mask4d, has_init)
        assert torch.equal(got, _reference(state, idx, has_init))
        assert not got[1].any()


def test_fallback_without_cached_mask_is_identical():
    """Cache off (mask4d None) gives the same result as cache on."""
    state = torch.randn(4, 2, 3, 3)
    idx = torch.tensor([1, 2])
    has_init = torch.tensor([False, True])
    on = gather_prefill_initial_state(state, idx, has_init.view(-1, 1, 1, 1), has_init)
    off = gather_prefill_initial_state(state, idx, None, has_init)
    assert torch.equal(on, off)
