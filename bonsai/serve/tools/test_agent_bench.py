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


def test_window_stats_mean_and_worst_bucket():
    times = [0.1, 0.2, 0.3, 0.4, 1.5, 2.5, 2.6]
    mean, worst = ab.window_stats(times, 0.0, 3.0)
    assert mean == 7 / 3 and worst == 1.0


def test_window_stats_excludes_outside_and_short_window():
    mean, worst = ab.window_stats([0.0, 0.1, 0.2, 5.0], 0.1, 0.6)
    assert mean == 2 / 0.5 and worst == 2 / 0.5
    assert ab.window_stats([1.0], 2.0, 2.0) == (0.0, 0.0)


def test_summarize_mixed_and_verdict():
    dec = [[i * 0.01 for i in range(300)], [i * 0.1 for i in range(30)]]
    s = ab.summarize_mixed(99000, 100.0, 0.0, 3.0, dec)
    assert s["prefill_tps"] == 990.0
    assert s["decoders"][0]["mean"] == 100.0 and s["decoders"][1]["worst"] == 10.0
    assert not ab.mixed_verdict(s, 600, 60)
    assert ab.mixed_verdict(s, 600, 10)
    assert "prefill 990" in ab.format_mixed(s)


def test_mtp_chunks_weighted_by_cumulative_usage():
    assert ab.chunk_weights([3, 4, 8]) == ([3, 1, 4], True)
    assert ab.chunk_weights([3, None]) == ([1, 1], False)
    mean, worst = ab.window_stats([0.5, 1.5], 0.0, 2.0, [3, 1])
    assert mean == 2.0 and worst == 1.0


def test_uncached_prefill_used_by_check():
    dec = [[0.1, 0.2]] * 3
    s = ab.summarize_mixed(1000, 2.0, 0.0, 1.0, dec, [[2, 4]] * 3, cached=800)
    assert s["prefill_tps"] == 500 and s["uncached_prefill_tps"] == 100
    assert s["decoders"][0]["mean"] == 4 and "uncached 100" in ab.format_mixed(s)
    assert not ab.mixed_verdict(s, 200, 1)
    fb = ab.summarize_mixed(1000, 2.0, 0.0, 1.0, dec)
    assert not fb["exact_tokens"] and "no usage" in ab.format_mixed(fb)


def test_early_decoders_detected():
    assert ab.early_decoders([10.0, 4.0, None], 5.0) == [1, 2]
    assert ab.early_decoders([6.0, 7.0], 5.0) == []


def test_mixed_phase_order_with_fakes(monkeypatch):
    log, lock = [], __import__("threading").Lock()

    def fake_stream(args, headers, prompt, on_first=None, counts=None, stop=None):
        kind = {1: "warm", 16: "fourth"}.get(args.max_tokens, "decoder")
        with lock:
            log.append(kind)
        now = ab.time.perf_counter()
        if kind == "decoder":
            times = []
            while not stop.is_set():
                times.append(ab.time.perf_counter())
                counts.append(len(times) * 2)
                if len(times) == 1 and on_first:
                    on_first()
                ab.time.sleep(0.001)
            return "", times, None, now
        if kind == "fourth":
            ab.time.sleep(0.05)
            on_first()
            return "", [ab.time.perf_counter()], {"prompt_tokens": 100}, now
        return "", [now], None, now

    monkeypatch.setattr(ab, "stream_request", fake_stream)
    monkeypatch.setattr(ab, "make_text", lambda rng, a, h, n, prefix="": prefix)
    args = ab.parse_args(["--mixed"])
    summary, errors = ab.run_mixed(args, {})
    assert errors == [] and summary["early_decoders"] == []
    assert log[:3] == ["warm"] * 3 and log[-1] == "fourth"
    assert log[3:6] == ["decoder"] * 3


def test_partial_overlap_window_per_decoder():
    dec = [[i * 0.1 for i in range(100)]] * 2
    s = ab.summarize_mixed(1000, 20.0, 0.0, 20.0, dec, None, None, [5.0, 25.0])
    assert s["decoders"][0]["overlap"] == 5.0 and s["decoders"][1]["overlap"] == 20.0
    assert s["decoders"][0]["mean"] == 10.0 and s["decoders"][1]["mean"] == 5.0
    line = ab.format_mixed(s)
    assert "[partial window 5 s of 20 s]" in line and "d0 10.0" in line
    full = ab.summarize_mixed(1000, 20.0, 0.0, 20.0, dec, None, None, [30.0, 25.0])
    assert "partial" not in ab.format_mixed(full)
