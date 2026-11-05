"""Tests for step_profile on a tiny synthetic trace."""
import json
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


def test_prefill_rows_per_token(tmp_path):
    """Prefill-only step: GPU ms per category divided by its prefill tokens, JSON dumped."""
    ev = [{"cat": "user_annotation", "name": "execute_context_1(64)_generation_0(0)",
           "ts": 0, "dur": 1000, "tid": 1},
          {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 10, "args": {"correlation": 1}},
          {"cat": "kernel", "name": "tern_m16", "ts": 100, "dur": 400, "args": {"correlation": 1}},
          {"cat": "cuda_runtime", "name": "hipLaunchKernel", "ts": 20, "args": {"correlation": 2}},
          {"cat": "kernel", "name": "rms_norm", "ts": 600, "dur": 100, "args": {"correlation": 2}},
          {"cat": "cpu_op", "name": "aten::mm", "ts": 5, "dur": 200, "tid": 1}]
    rows = sp.prefill_rows(ev)
    assert len(rows) == 1 and rows[0]["tokens"] == 64 and rows[0]["launches"] == 2
    assert abs(rows[0]["busy_ms"] - 0.5) < 1e-9 and abs(rows[0]["idle_ms"] - 0.1) < 1e-9
    assert abs(rows[0]["cpu_ms"] - 0.2) < 1e-9 and not rows[0]["graph"]
    assert abs(rows[0]["ms_per_token"]["ternary GEMM/GEMV"] - 0.4 / 64) < 1e-12
    assert "norms/elementwise" in sp.prefill_table(rows)
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": ev}))
    out = tmp_path / "out.json"
    assert sp.main(["--prefill", "--json", str(out), str(path)]) == 0
    assert '"tokens": 64' in out.read_text()


def test_gap_stats():
    """Small gaps are summed, big gaps listed."""
    k = lambda ts, d: {"cat": "kernel", "name": "k", "ts": ts, "dur": d}
    small, big = sp.gap_stats([k(0, 100), k(150, 100), k(1000, 10)])
    assert abs(small - 0.05) < 1e-9 and len(big) == 1
