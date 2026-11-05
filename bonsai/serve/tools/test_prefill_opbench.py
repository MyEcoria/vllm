# SPDX-License-Identifier: Apache-2.0
"""Smoke test of prefill_opbench (needs free GPU memory)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import prefill_opbench as pb  # noqa: E402


def _gpu_free(min_gb=10):
    """True if a GPU with at least min_gb free memory is available."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.mem_get_info()[0] > min_gb * 2**30


pytestmark = pytest.mark.skipif(not _gpu_free(), reason="needs free GPU memory")


def test_table_keys_and_positive_timings():
    """Every section reports a positive timing and the projection is present."""
    a = pb.parse_args(["--tokens", "512", "--depth", "4096", "--iters", "2"])
    r = pb.run_all(a)[512]
    assert set(r) == {"attn", "gemm", "act_quant", "gdn", "projected"}
    assert len(r["gemm"]) == len(pb.SHAPES)
    times = [r["attn"]["ms"], r["act_quant"]["ms"], r["gdn"]["ms"], r["projected"]["ms_per_token"]]
    times += [v["ms"] for v in r["gemm"].values()]
    assert all(t > 0 for t in times)
