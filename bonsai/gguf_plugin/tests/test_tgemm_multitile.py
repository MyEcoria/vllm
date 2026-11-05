# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-M-tile ternary GEMM (16 < M <= 64) must match the old path bit for bit.

The kernel choice is cached per process, so the old path (BONSAI_TGEMM_MT=0)
runs in a child process on identical seeded inputs.
"""

import os
import subprocess
import sys

import pytest
import torch

MIN_FREE_BYTES = 2 << 30
SHAPES = [(5120, 6144), (5120, 17408), (1000, 2048), (4096, 4096)]
CHILD = """
import sys, torch
import vllm_gguf_plugin.ops  # noqa: F401
m, n, k, dtype, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
g = torch.Generator().manual_seed(m * 7919 + n + k)
q = torch.randint(0, 256, (n, k // 4), generator=g, dtype=torch.uint8)
s = (torch.rand(n, k // 128, generator=g) * 0.1 + 0.01).half()
xq = torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8)
xs = torch.rand(m, k // 128, generator=g) * 0.1 + 0.01
dev = "cuda"
counters = torch.zeros(4096, dtype=torch.int32, device=dev)
y = torch.ops._C_gguf.ternary_wmma_gemm(
    q.to(dev), s.to(dev), xq.to(dev), xs.to(dev), counters, getattr(torch, dtype))
torch.save(y.cpu(), out)
"""


def _gpu_ready() -> bool:
    """Return True when a ROCm GPU with enough free memory is available."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return torch.cuda.mem_get_info()[0] >= MIN_FREE_BYTES
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a ROCm GPU with free memory"
)


def _run(m, n, k, dtype, out, multi: bool) -> torch.Tensor:
    """Run the GEMM in a child process with the multi-tile path on or off."""
    env = dict(os.environ, BONSAI_TGEMM_MT="1" if multi else "0")
    args = [sys.executable, "-c", CHILD, str(m), str(n), str(k), dtype, str(out)]
    subprocess.run(args, env=env, check=True)
    return torch.load(out)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [17, 32, 48, 60, 64])
def test_multitile_bit_exact(tmp_path, m, n, k, dtype):
    """The MT kernel output equals wmma_small's, bit for bit."""
    new = _run(m, n, k, dtype, tmp_path / "new.pt", multi=True)
    old = _run(m, n, k, dtype, tmp_path / "old.pt", multi=False)
    assert new.shape == (m, n)
    assert torch.equal(new.view(torch.int16), old.view(torch.int16))
