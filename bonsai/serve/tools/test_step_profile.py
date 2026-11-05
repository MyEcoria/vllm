"""Tests for step_profile on a tiny synthetic trace."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import step_profile as sp


def _trace():
    """Two steps: decode (1 kernel, graph) and mixed (2 kernels)."""
    ann = lambda n, ts, d: {"cat": "user_annotation", "name": n, "ts": ts, "dur": d}
    ker = lambda n, ts, d, c: {"cat": "kernel", "name": n, "ts": ts, "dur": d,
                               "args": {"correlation": c}}
    rt = lambda n, ts, c: {"cat": "cuda_runtime", "name": n, "ts": ts, "args": {"correlation": c}}
    return [ann("execute_context_0(0)_generation_3(12)", 0, 1000),
            rt("hipGraphLaunch", 10, 1), ker("tern_m16", 100, 400, 1),
            ann("execute_context_1(16)_generation_3(12)", 2000, 3000),
            rt("hipLaunchKernel", 2010, 2), ker("int4_attn", 2100, 500, 2),
            rt("hipLaunchKernel", 2020, 3), ker("gdn_x", 3000, 200, 3)]


def test_kinds_and_times():
    """Decode vs mixed classification, busy, gap, graph and category."""
    res = sp.analyse(_trace())
    assert res["decode"]["busy_ms"] == 0.4 and res["decode"]["graph"] == 1
    assert res["mixed"]["kernels"] == 2 and res["mixed"]["graph"] == 0
    assert abs(res["mixed"]["gap_ms"] - 0.4) < 1e-9
    assert res["mixed"]["cats"]["attention int4"] == 0.5
    assert "decode" in sp.format_report(res)


def test_cpu_report_self_time():
    """Nested cpu_ops: parent self time excludes the child."""
    ev = _trace() + [
        {"cat": "cpu_op", "name": "parent", "ts": 2000, "dur": 100, "tid": 1},
        {"cat": "cpu_op", "name": "aten::child", "ts": 2010, "dur": 40, "tid": 1}]
    for e in ev:
        if e.get("cat") == "user_annotation":
            e["tid"] = 1
    out = sp.cpu_report(ev, "mixed")
    assert "2 ops/step (1 aten)" in out and "parent" in out


def test_gap_stats():
    """Small gaps are summed, big gaps listed."""
    k = lambda ts, d: {"cat": "kernel", "name": "k", "ts": ts, "dur": d}
    small, big = sp.gap_stats([k(0, 100), k(150, 100), k(1000, 10)])
    assert abs(small - 0.05) < 1e-9 and len(big) == 1
