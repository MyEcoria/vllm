"""Simulate N independent agents with long, growing contexts against a Bonsai server."""
import argparse
import copy
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8000"
TARGET_PREFILL = 600.0
TARGET_DECODE = 60.0
WORDS = ("alpha beta gamma delta kernel buffer socket thread queue token cache "
         "stream parse index merge split value error retry commit branch build "
         "module config schema router handler worker driver tensor shard").split()


def base_url(value):
    """Validate a server base URL and strip trailing slashes."""
    if not value.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(f"not an http(s) URL: {value}")
    return value.rstrip("/")


def build_headers():
    """Return JSON headers with a Bearer token from env or ~/.bonsai/api_key."""
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("BONSAI_API_KEY")
    key_file = Path.home() / ".bonsai" / "api_key"
    if not key and key_file.is_file():
        key = key_file.read_text().strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Multi-agent long-context benchmark.")
    p.add_argument("--url", type=base_url,
                   default=os.environ.get("BONSAI_URL", DEFAULT_URL))
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--start-tokens", type=int, default=90000)
    p.add_argument("--turns", type=int, default=4)
    p.add_argument("--delta-min", type=int, default=2000)
    p.add_argument("--delta-max", type=int, default=4000)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--stagger", type=float, default=0.0)
    p.add_argument("--sync-turns", action="store_true",
                   help="barrier: all agents start turn N together")
    p.add_argument("--max-context", type=int, default=100000,
                   help="stop an agent before prompt + max_tokens exceeds this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="bonsai")
    p.add_argument("--json", dest="json_path", default=None)
    p.add_argument("--mixed", action="store_true",
                   help="3 warm decoders + a 4th prefill arriving mid-decode")
    p.add_argument("--mixed-delta", type=int, default=None, metavar="N",
                   help="4th request is a warm agent sending only N new tokens")
    p.add_argument("--mixed-tokens", type=int, default=99000,
                   help="context size of the warm agents and the cold 4th prompt")
    p.add_argument("--mixed-max-tokens", type=int, default=1500)
    p.add_argument("--check", action="store_true",
                   help="--mixed: exit 1 if below the thresholds")
    p.add_argument("--min-prefill", type=float, default=TARGET_PREFILL)
    p.add_argument("--min-decode", type=float, default=TARGET_DECODE)
    return p.parse_args(argv)


def gen_line(rng, prefix=""):
    """Return one random code-like line."""
    w = [rng.choice(WORDS) for _ in range(3)]
    return (f"{prefix}def {w[0]}_{rng.randrange(16**6):06x}({w[1]}, {w[2]}): "
            f"return {rng.randrange(10**6)} + len('{rng.choice(WORDS)}')\n")


def gen_text(rng, n_chars, prefix=""):
    """Return random text of at least n_chars characters."""
    parts, total = [], 0
    while total < n_chars:
        line = gen_line(rng, prefix)
        parts.append(line)
        total += len(line)
    return "".join(parts)


def post_json(url, headers, payload, timeout=600):
    """POST a JSON payload and return the decoded JSON reply."""
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def count_tokens(args, headers, text):
    """Count tokens via /tokenize, falling back to 4 chars per token."""
    try:
        reply = post_json(args.url + "/tokenize", headers,
                          {"model": args.model, "prompt": text,
                           "add_special_tokens": False})
        return int(reply["count"])
    except Exception:
        return max(1, len(text) // 4)


def make_text(rng, args, headers, tokens, prefix=""):
    """Return random text of about `tokens` tokens, calibrated on a sample."""
    sample = gen_text(rng, 4000, prefix)
    ratio = len(sample) / count_tokens(args, headers, sample)
    return gen_text(rng, int(tokens * ratio), prefix)


def parse_sse_line(line):
    """Parse one SSE line: dict for data, "[DONE]" for the end, None otherwise."""
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(data)
    except ValueError:
        return None


def window_min_rate(times, generated, window=1.0):
    """Return the minimum tokens/s over full 1-s windows of chunk arrivals."""
    if len(times) < 2 or times[-1] - times[0] < window:
        return None
    weight = generated / len(times)
    rates, start = [], times[0]
    while start + window <= times[-1]:
        n = sum(weight for t in times if start <= t < start + window)
        rates.append(n / window)
        start += window
    return min(rates)


def stream_request(args, headers, prompt, on_first=None, counts=None):
    """Stream one completion; return text, chunk times, usage and t0."""
    payload = {"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
               "temperature": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True,
                              "continuous_usage_stats": True}}
    req = urllib.request.Request(args.url + "/v1/completions",
                                 json.dumps(payload).encode(), headers)
    text, times, usage, t0 = [], [], None, time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            obj = parse_sse_line(line)
            if not isinstance(obj, dict):
                continue
            usage = obj.get("usage") or usage
            chunk = (obj.get("choices") or [{}])[0].get("text")
            if chunk:
                text.append(chunk)
                times.append(time.perf_counter())
                if counts is not None:
                    counts.append((obj.get("usage") or {}).get("completion_tokens"))
                if len(times) == 1 and on_first:
                    on_first()
    return "".join(text), times, usage, t0


def measure(times, t0, usage, prompt_est, prev_prompt, max_tokens):
    """Build the metrics dict of one request."""
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", prompt_est)
    generated = usage.get("completion_tokens", len(times)) or len(times)
    ttft = times[0] - t0 if times else float("nan")
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    delta = prompt_tokens - prev_prompt
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"cached": cached, "ttft": ttft, "prompt_tokens": prompt_tokens, "delta": delta,
            "prefill_tps": delta / ttft if times else 0.0, "generated": generated,
            "decode_tps": (generated - 1) / span if span > 0 else 0.0,
            "min_window_tps": window_min_rate(times, generated)}


def run_agent(idx, args, headers, results, errors, barrier=None):
    """Run all turns of one agent sequentially, appending to results."""
    rng = random.Random(args.seed * 1000 + idx)
    try:
        prompt = make_text(rng, args, headers, args.start_tokens, f"# a{idx} ")
        prev = 0
        for turn in range(1, args.turns + 1):
            if barrier is not None:
                barrier.wait()   # raises BrokenBarrierError once any agent stopped
            if len(prompt) // 4 + args.max_tokens > args.max_context and \
                    prompt_len(args, headers, prompt) + args.max_tokens > args.max_context:
                print(f"agent {idx}: stop before turn {turn} (context cap)", flush=True)
                if barrier is not None:
                    barrier.abort()
                break
            answer, times, usage, t0 = stream_request(args, headers, prompt)
            m = measure(times, t0, usage, len(prompt) // 4, prev, args.max_tokens)
            results.append({"agent": idx, "turn": turn, **m})
            print(format_row(results[-1]), flush=True)
            prev = m["prompt_tokens"]
            n = rng.randint(args.delta_min, args.delta_max)
            prompt += answer + "\n" + make_text(rng, args, headers, n, "[tool] ")
    except threading.BrokenBarrierError:
        pass
    except Exception as exc:
        errors.append(f"agent {idx}: {exc!r}")
        if barrier is not None:
            barrier.abort()


def prompt_len(args, headers, prompt):
    """Return the exact token count of a prompt."""
    return count_tokens(args, headers, prompt)


def format_row(r):
    """Format one (agent, turn) result line."""
    w = r["min_window_tps"]
    c = r.get("cached")
    cache = "" if c is None else f"cached {c}/{r['prompt_tokens']} ({c / max(1, r['prompt_tokens']):.0%}) "
    return (f"agent {r['agent']} turn {r['turn']}: prompt {r['prompt_tokens']} {cache}"
            f"delta {r['delta']} ttft {r['ttft']:.2f}s prefill {r['prefill_tps']:.0f} "
            f"t/s decode {r['decode_tps']:.1f} t/s min1s "
            f"{'n/a' if w is None else f'{w:.1f}'} t/s")


def parse_prom(text):
    """Sum Prometheus samples by metric name (labels dropped, _total stripped)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        name = m.group(1).removesuffix("_total")
        try:
            out[name] = out.get(name, 0.0) + float(m.group(3))
        except ValueError:
            pass
    return out


def hit_rate(before, after):
    """Return the prefix cache hit rate between two parsed scrapes, or None."""
    h, q = "vllm:prefix_cache_hits", "vllm:prefix_cache_queries"
    if h not in after or q not in after:
        return None
    queries = after[q] - before.get(q, 0.0)
    return (after[h] - before.get(h, 0.0)) / queries if queries > 0 else None


def scrape(args, headers):
    """Fetch and parse /metrics; empty dict on failure."""
    try:
        req = urllib.request.Request(args.url + "/metrics", headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r:
            return parse_prom(r.read().decode())
    except Exception:
        return {}


def stats(values):
    """Return (mean, min) of a list."""
    return sum(values) / len(values), min(values)


def summarize(results):
    """Group results per turn index into mean/min statistics."""
    rows = {}
    for turn in sorted({r["turn"] for r in results}):
        rs = [r for r in results if r["turn"] == turn]
        rows[turn] = {"prefill": stats([r["prefill_tps"] for r in rs]),
                      "decode": stats([r["decode_tps"] for r in rs]),
                      "ttft": stats([r["ttft"] for r in rs])}
    return rows


def verdict(rows):
    """PASS if every warm turn (>=2) meets both minimum targets; None if no warm turn."""
    warm = [v for t, v in rows.items() if t >= 2]
    if not warm:
        return None
    ok = all(v["prefill"][1] >= TARGET_PREFILL and v["decode"][1] >= TARGET_DECODE
             for v in warm)
    return "PASS" if ok else "FAIL"


def print_summary(rows):
    """Print the per-turn summary table and the PASS/FAIL line."""
    print(f"{'turn':>4} {'prefill mean':>12} {'min':>7} {'decode mean':>12} "
          f"{'min':>6} {'ttft mean':>9} {'min':>7}")
    for t, v in rows.items():
        tag = " (cold)" if t == 1 else ""
        print(f"{t:>4} {v['prefill'][0]:>12.0f} {v['prefill'][1]:>7.0f} "
              f"{v['decode'][0]:>12.1f} {v['decode'][1]:>6.1f} "
              f"{v['ttft'][0]:>9.2f} {v['ttft'][1]:>7.2f}{tag}")
    print(f"targets: warm prefill >= {TARGET_PREFILL:.0f} t/s, decode >= "
          f"{TARGET_DECODE:.0f} t/s -> {verdict(rows) or 'N/A (need >=2 turns)'}")


def chunk_weights(cumulative):
    """Return (tokens per chunk, exact) from cumulative counts; 1/chunk if absent."""
    if not cumulative or any(c is None for c in cumulative):
        return [1] * len(cumulative or []), False
    return [c - p for p, c in zip([0] + cumulative, cumulative)], True


def window_stats(times, start, end, weights=None):
    """Return (mean, worst 1-s bucket) tokens/s of arrivals inside [start, end)."""
    if end <= start:
        return 0.0, 0.0
    weights = weights or [1] * len(times)
    n_buckets = max(1, int((end - start) // 1.0))
    width = min(1.0, end - start)
    counts = [0] * n_buckets
    for t, w in zip(times, weights):
        if start <= t < end:
            counts[min(n_buckets - 1, int((t - start) // width))] += w
    return sum(counts) / (end - start), min(counts) / width


def mixed_verdict(summary, min_prefill, min_decode):
    """Return True if prefill (uncached if known) and every decoder mean pass."""
    prefill = summary.get("uncached_prefill_tps")
    if prefill is None:
        prefill = summary["prefill_tps"]
    return prefill >= min_prefill and all(
        d["mean"] >= min_decode for d in summary["decoders"])


def summarize_mixed(prompt_tokens, ttft, start, end, decoder_times,
                    decoder_counts=None, cached=None):
    """Build the mixed-run summary dict from timestamps and cumulative counts."""
    decoders, exact = [], True
    for i, times in enumerate(decoder_times):
        w, ok = chunk_weights((decoder_counts or [None] * len(decoder_times))[i] or [])
        exact = exact and ok
        mean, worst = window_stats(times, start, end, w if ok else None)
        decoders.append({"mean": mean, "worst": worst})
    rate = lambda n: n / ttft if ttft > 0 else 0.0
    return {"prompt_tokens": prompt_tokens, "ttft": ttft, "window": end - start,
            "prefill_tps": rate(prompt_tokens), "cached_tokens": cached,
            "uncached_prefill_tps": None if cached is None
            else rate(prompt_tokens - cached),
            "exact_tokens": exact, "decoders": decoders}


def format_mixed(s):
    """Format the one-line mixed summary."""
    dec = " ".join(f"d{i} {d['mean']:.1f}/{d['worst']:.1f}"
                   for i, d in enumerate(s["decoders"]))
    unc = s.get("uncached_prefill_tps")
    unc = "" if unc is None else f", uncached {unc:.0f} t/s"
    note = "" if s.get("exact_tokens", True) else " [tokens=chunks: no usage]"
    return (f"mixed: prefill {s['prefill_tps']:.0f} t/s{unc} (prompt "
            f"{s['prompt_tokens']} ttft {s['ttft']:.2f}s) decoders mean/worst1s "
            f"t/s: {dec}{note}")


def start_decoders(args, headers, prompts, errors):
    """Start one thread per warm prompt; return threads, times, counts, events."""
    cfg = copy.copy(args)
    cfg.max_tokens = args.mixed_max_tokens
    events = [threading.Event() for _ in prompts]
    times = [[] for _ in prompts]
    counts = [[] for _ in prompts]

    def work(i):
        """Stream one warm decoder and store its chunk times."""
        try:
            times[i] = stream_request(cfg, headers, prompts[i], events[i].set,
                                    counts[i])[1]
        except Exception as exc:
            errors.append(f"decoder {i}: {exc!r}")
            events[i].set()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    return threads, times, counts, events


def fourth_prompt(args, headers, rng, warm_prompt_fn):
    """Return the 4th prompt: cold, or a cached prefix plus an N-token delta."""
    if args.mixed_delta is None:
        return make_text(rng, args, headers, args.mixed_tokens, "# a3 ")
    prefix = warm_prompt_fn()
    cfg = copy.copy(args)
    cfg.max_tokens = 1
    stream_request(cfg, headers, prefix)
    return prefix + make_text(rng, args, headers, args.mixed_delta, "[tool] ")


def run_mixed(args, headers):
    """Run the mixed scenario; return (summary or None, errors)."""
    rng = random.Random(args.seed)
    prompts = [make_text(random.Random(args.seed * 1000 + i), args, headers,
                         args.mixed_tokens, f"# a{i} ") for i in range(3)]
    prefix_fn = lambda: make_text(random.Random(args.seed * 1000 + 3), args,
                                  headers, args.mixed_tokens, "# a3 ")
    last = fourth_prompt(args, headers, rng, prefix_fn)
    errors = []
    threads, times, counts, events = start_decoders(args, headers, prompts, errors)
    for e in events:
        e.wait()
    cfg = copy.copy(args)
    cfg.max_tokens = 16
    summary = None
    try:
        _, t4, usage, start = stream_request(cfg, headers, last)
        usage = usage or {}
        tokens = usage.get("prompt_tokens", len(last) // 4)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        for t in threads:
            t.join()
        summary = summarize_mixed(tokens, t4[0] - start, start, t4[0], times,
                                  counts, cached)
    except Exception as exc:
        errors.append(f"4th request: {exc!r}")
    return summary, errors


def main_mixed(args, headers):
    """Run the mixed mode, print the summary line, write JSON, return exit code."""
    summary, errors = run_mixed(args, headers)
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    if summary is None:
        return 1
    print(format_mixed(summary))
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summary, indent=1))
    ok = mixed_verdict(summary, args.min_prefill, args.min_decode)
    return 1 if args.check and not ok else 0


def main(argv=None):
    """Run all agents concurrently, then print rows, summary and cache hit rate."""
    args = parse_args(argv)
    headers = build_headers()
    if args.mixed:
        return main_mixed(args, headers)
    before = scrape(args, headers)
    results, errors, threads = [], [], []
    barrier = threading.Barrier(args.agents) if args.sync_turns else None
    for i in range(args.agents):
        t = threading.Thread(target=run_agent, args=(i, args, headers, results, errors, barrier))
        t.start()
        threads.append(t)
        time.sleep(args.stagger)
    for t in threads:
        t.join()
    rate = hit_rate(before, scrape(args, headers))
    print_summary(summarize(results)) if results else None
    print("prefix cache hit rate:", "n/a" if rate is None else f"{rate:.1%}")
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(
            {"results": results, "errors": errors, "hit_rate": rate}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
