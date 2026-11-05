"""Tests for the ISA-text parser of tgemm_isa.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tgemm_isa as t  # noqa: E402

ASM = """
kern:
\tv_mov_b32 v0, 0
.LBB0_1:
\tglobal_load_b128 v[4:7], v[0:1], off
\ts_wait_loadcnt 0x0
\tv_and_b32 v8, 0x3030303, v4
\tv_lshrrev_b32 v9, 2, v4
\tv_wmma_i32_16x16x16_iu8 v[10:17], v[8:9], v[8:9], v[10:17]
\tds_load_b64 v[2:3], v1
\ts_cbranch_scc1 .LBB0_1
.Lfunc_end0:
\t.amdhsa_kernel kern
\t\t.amdhsa_next_free_vgpr 124
\t\t.amdhsa_next_free_sgpr 30
\t\t.amdhsa_group_segment_fixed_size 4096
\t.end_amdhsa_kernel
"""


def test_resources():
    """Resource block is parsed."""
    r = t.parse_resources(ASM)["kern"]
    assert (r["vgpr"], r["sgpr"], r["lds"]) == (124, 30, 4096)


def test_loop_mix():
    """Inner loop is found and classified."""
    loop = t.hottest_loop(t.kernel_lines(ASM, "kern"))
    c = t.mix(loop)
    assert c["wmma"] == 1 and c["unpackALU"] == 2 and c["gload"] == 1
    assert c["dsread"] == 1 and c["waitcnt"] == 1


def test_occupancy():
    """Occupancy follows VGPR and LDS limits."""
    assert t.occupancy(128, 0, 128) == 12
    assert t.occupancy(240, 0, 256) == 6
