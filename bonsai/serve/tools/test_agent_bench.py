"""Offline tests for agent_bench (no server)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agent_bench as ab  # noqa: E402

PROM = """# HELP x
vllm:prefix_cache_queries_total{engine="0"} 1000
vllm:prefix_cache_hits_total{engine="0"} 250
other_metric 3
"""


def test_text_deterministic():
    assert ab.gen_text(random.Random(1), 500) == ab.gen_text(random.Random(1), 500)


def test_text_distinct_per_agent():
    a = ab.gen_text(random.Random(1000), 2000, "# a0 ")
    b = ab.gen_text(random.Random(1001), 2000, "# a1 ")
    assert a[:200] != b[:200] and len(a) >= 2000


def test_parse_prom_and_hit_rate():
    cur = ab.parse_prom(PROM)
    assert cur["vllm:prefix_cache_hits"] == 250
    before = {"vllm:prefix_cache_hits": 50.0, "vllm:prefix_cache_queries": 200.0}
    assert ab.hit_rate(before, cur) == 200 / 800
    assert ab.hit_rate({}, {}) is None


def row(turn, prefill, decode):
    return {"agent": 0, "turn": turn, "prefill_tps": prefill, "decode_tps": decode,
            "ttft": 1.0}


def test_verdict_pass_fail_ignores_cold_turn():
    ok = [row(1, 10, 10), row(2, 700, 65), row(2, 650, 61)]
    assert ab.verdict(ab.summarize(ok)) == "PASS"
    bad = ok + [row(3, 500, 70)]
    assert ab.verdict(ab.summarize(bad)) == "FAIL"
    assert ab.verdict(ab.summarize([row(1, 1, 1)])) is None


def test_parse_sse_line():
    assert ab.parse_sse_line(b'data: {"a": 1}\n') == {"a": 1}
    assert ab.parse_sse_line(b"data: [DONE]\n") == "[DONE]"
    assert ab.parse_sse_line(b": keepalive\n") is None
    assert ab.parse_sse_line(b"data: {bad\n") is None


def test_window_min_rate():
    times = [i * 0.1 for i in range(31)]
    assert abs(ab.window_min_rate(times, 31) - 10.0) < 1.0
    assert ab.window_min_rate([0.0, 0.5], 2) is None
