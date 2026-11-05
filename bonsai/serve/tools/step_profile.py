"""Summarise a vLLM torch-profiler trace per engine step type (decode-only vs mixed)."""
import gzip
import json
import re
import sys
from collections import defaultdict

STEP_RE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
CATEGORIES = (
    ("ternary GEMM/GEMV", ("tern", "wmma_small", "gemv", "gemm", "cijk", "mmvq", "mmq")),
    ("attention int4", ("int4", "attn", "attention", "splitk", "flash")),
    ("GDN/recurrent", ("gdn", "conv1d", "recurrent", "kda", "ssm", "delta", "fla")),
    ("MTP head", ("mtp", "w8_m16", "w8_")),
    ("sampling", ("sampl", "topk", "top_k", "topp", "argmax", "softmax", "logit", "rejection")),
    ("act-quant", ("quant",)),
    ("norms/elementwise", ("norm", "rms", "elementwise")),
)
CAT_LABELS = [label for label, _ in CATEGORIES] + ["other"]
MODE_FLAGS = ("--cpu", "--gaps", "--prefill")
CAT_WIDTH = max(len(label) for label in CAT_LABELS) + 2


def load_trace(path):
    """Load a (possibly gzipped) Chrome trace and return its event list."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)["traceEvents"]


def categorize(name):
    """Map a kernel name to one of the report categories."""
    low = name.lower()
    for label, keys in CATEGORIES:
        if any(k in low for k in keys):
            return label
    return "other"


def step_kind(name):
    """Return 'mixed' (prefill tokens present), 'decode' or None for a step annotation."""
    m = STEP_RE.search(name)
    if not m:
        return None
    return "mixed" if int(m.group(1)) > 0 and int(m.group(2)) > 0 and int(m.group(3)) > 0 \
        else ("decode" if int(m.group(1)) == 0 else "prefill")


def split_events(events):
    """Return (steps, kernels, runtime) sorted by start time."""
    steps = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and STEP_RE.search(e["name"])), key=lambda e: e["ts"])
    kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    runtime = [e for e in events if e.get("cat") == "cuda_runtime"]
    return steps, kernels, runtime


def assign_kernels(steps, kernels, runtime):
    """Group kernels per step via launch correlation (fallback: start time)."""
    starts = [s["ts"] for s in steps] + [float("inf")]
    launch = {}
    for r in runtime:
        cid = (r.get("args") or {}).get("correlation")
        if cid is not None:
            launch[cid] = r["ts"]
    groups = defaultdict(list)
    for k in kernels:
        t = launch.get((k.get("args") or {}).get("correlation"), k["ts"])
        for i in range(len(steps)):
            if starts[i] <= t < starts[i + 1]:
                groups[i].append(k)
                break
    return groups


def busy_time(ks):
    """Return (union busy us, span us) of kernel intervals."""
    ivs = sorted((k["ts"], k["ts"] + k["dur"]) for k in ks)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            busy += (cur_e - cur_s) if cur_e is not None else 0.0
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += (cur_e - cur_s) if cur_e is not None else 0.0
    return busy, (ivs[-1][1] - ivs[0][0]) if ivs else 0.0


def graph_launches(step, runtime):
    """Count hipGraphLaunch runtime calls inside the step annotation."""
    lo, hi = step["ts"], step["ts"] + step["dur"]
    return sum(1 for r in runtime if "graphlaunch" in r["name"].lower() and lo <= r["ts"] <= hi)


def summarize_step(step, ks, runtime):
    """Return a metrics dict (ms) for one step."""
    busy, span = busy_time(ks)
    cats = defaultdict(float)
    for k in ks:
        cats[categorize(k["name"])] += k["dur"] / 1000
    return {"kind": step_kind(step["name"]), "cpu_ms": step["dur"] / 1000,
            "span_ms": span / 1000, "busy_ms": busy / 1000,
            "gap_ms": (span - busy) / 1000, "kernels": len(ks),
            "graph": graph_launches(step, runtime) > 0, "cats": dict(cats)}


def aggregate(rows):
    """Average a list of step dicts of the same kind."""
    n = len(rows)
    out = {"n": n, "graph": sum(r["graph"] for r in rows)}
    for key in ("cpu_ms", "span_ms", "busy_ms", "gap_ms", "kernels"):
        out[key] = sum(r[key] for r in rows) / n
    cats = defaultdict(float)
    for r in rows:
        for c, v in r["cats"].items():
            cats[c] += v / n
    out["cats"] = dict(cats)
    return out


def analyse(events):
    """Return {kind: aggregate} for a trace's events."""
    steps, kernels, runtime = split_events(events)
    groups = assign_kernels(steps, kernels, runtime)
    by_kind = defaultdict(list)
    for i, s in enumerate(steps):
        if groups.get(i):
            by_kind[step_kind(s["name"])].append(summarize_step(s, groups[i], runtime))
    return {k: aggregate(v) for k, v in by_kind.items()}


def format_report(res):
    """Render the per-kind table as text."""
    lines = []
    for kind, a in sorted(res.items()):
        lines.append(f"{kind}: {a['n']} steps, cpu {a['cpu_ms']:.1f} ms, gpu span "
                     f"{a['span_ms']:.1f} ms, busy {a['busy_ms']:.1f} ms, idle gaps "
                     f"{a['gap_ms']:.1f} ms, {a['kernels']:.0f} kernels/step, "
                     f"hip graph in {a['graph']}/{a['n']} steps")
        for c, v in sorted(a["cats"].items(), key=lambda x: -x[1]):
            lines.append(f"    {c:<20}{v:8.2f} ms")
    return "\n".join(lines)


def step_self_times(step, ops):
    """Return ({op: self us}, {op: calls}) for cpu_ops nested in a step on its thread."""
    lo, hi = step["ts"], step["ts"] + step["dur"]
    mine = sorted((e for e in ops if e["tid"] == step["tid"] and lo <= e["ts"] <= hi),
                  key=lambda e: (e["ts"], -e["dur"]))
    self_us, calls, stack = defaultdict(float), defaultdict(int), []
    for e in mine:
        while stack and stack[-1][1] <= e["ts"]:
            stack.pop()
        if stack:
            self_us[stack[-1][0]] -= e["dur"]
        self_us[e["name"]] += e["dur"]
        calls[e["name"]] += 1
        stack.append((e["name"], e["ts"] + e["dur"]))
    return self_us, calls


def cpu_report(events, kind="mixed", top=15):
    """Per-step CPU self-time table (ms/step) for steps of one kind."""
    steps = [s for s in split_events(events)[0] if step_kind(s["name"]) == kind]
    ops = [e for e in events if e.get("cat") == "cpu_op"]
    tot, cnt = defaultdict(float), defaultdict(int)
    for s in steps:
        su, c = step_self_times(s, ops)
        for k, v in su.items():
            tot[k] += v
            cnt[k] += c[k]
    n = max(len(steps), 1)
    aten = sum(v for k, v in cnt.items() if k.startswith("aten::")) / n
    lines = [f"{kind}: {len(steps)} steps, {sum(cnt.values()) / n:.0f} ops/step ({aten:.0f} aten)"]
    for k, v in sorted(tot.items(), key=lambda x: -x[1])[:top]:
        lines.append(f"    {k[:60]:<60}{v / n / 1000:8.2f} ms {cnt[k] / n:7.0f} calls")
    return "\n".join(lines)


def gap_stats(events, big_us=200.0):
    """Return (small_gap_total_ms, [(start_ms, gap_ms, prev, next)] for gaps > big_us)."""
    ks = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    small, big, end, prev = 0.0, [], None, None
    for k in ks:
        if end is not None and k["ts"] > end:
            g = k["ts"] - end
            if g > big_us:
                big.append((end / 1000, g / 1000, prev["name"][:40], k["name"][:40]))
            else:
                small += g
        if end is None or k["ts"] + k["dur"] > end:
            end, prev = k["ts"] + k["dur"], k
    return small / 1000, big


def kernel_split(events, top=12):
    """Return [(name, calls, total_ms)] of the heaviest kernels."""
    acc = defaultdict(lambda: [0, 0.0])
    for e in events:
        if e.get("cat") == "kernel":
            acc[e["name"][:90]][0] += 1
            acc[e["name"][:90]][1] += e["dur"] / 1000
    return sorted(((n, c, t) for n, (c, t) in acc.items()), key=lambda x: -x[2])[:top]


def gap_report(events):
    """Text report of inter-kernel gaps and the heaviest kernels (totals over the trace)."""
    small, big = gap_stats(events)
    lines = [f"small gaps (<=0.2 ms): {small:.1f} ms total; {len(big)} gaps > 0.2 ms"]
    lines += [f"    @{t:.1f} ms gap {g:.2f} ms  {a} -> {b}" for t, g, a, b in big[:20]]
    lines += [f"    {t:8.2f} ms {c:6d}  {n}" for n, c, t in kernel_split(events)]
    return "\n".join(lines)


def prefill_tokens(name):
    """Return the prefill token count encoded in a step annotation."""
    return int(STEP_RE.search(name).group(2))


def step_cpu_ms(step, ops):
    """Return a step's CPU ms: top-level cpu_op time on its thread."""
    self_us, _ = step_self_times(step, ops)
    return sum(self_us.values()) / 1000


def prefill_rows(events):
    """Return one metrics dict per prefill-only step, with ms/token per category."""
    steps, kernels, runtime = split_events(events)
    ops = [e for e in events if e.get("cat") == "cpu_op"]
    groups = assign_kernels(steps, kernels, runtime)
    rows = []
    for i, s in enumerate(steps):
        if step_kind(s["name"]) != "prefill" or not groups.get(i):
            continue
        m, tokens = summarize_step(s, groups[i], runtime), prefill_tokens(s["name"])
        rows.append({"tokens": tokens, "wall_ms": m["cpu_ms"], "busy_ms": m["busy_ms"],
                     "idle_ms": m["gap_ms"], "cpu_ms": step_cpu_ms(s, ops),
                     "launches": m["kernels"], "graph": m["graph"], "gpu_ms": m["cats"],
                     "ms_per_token": {c: v / tokens for c, v in m["cats"].items()}})
    return rows


def _metric_cells(r):
    """Return the fixed-width metric columns of one prefill row."""
    mode = "graph" if r["graph"] else "eager"
    return (f"{r['tokens']:>7}{r['wall_ms']:>10.2f}{r['busy_ms']:>10.2f}"
            f"{r['idle_ms']:>10.2f}{r['cpu_ms']:>10.2f}{r['launches']:>10d}{mode:>8}")


def _cat_cells(values, fmt):
    """Return one fixed-width cell per report category (missing -> 0)."""
    return "".join(f"{values.get(c, 0.0):>{CAT_WIDTH}{fmt}}" for c in CAT_LABELS)


def prefill_table(rows):
    """Render prefill-only steps: metrics with GPU ms per category, then ms/token."""
    if not rows:
        return "no prefill-only steps"
    head = (f"{'step':>5}{'tokens':>7}{'wall_ms':>10}{'busy_ms':>10}{'idle_ms':>10}"
            f"{'cpu_ms':>10}{'launches':>10}{'mode':>8}")
    cats = "".join(f"{c:>{CAT_WIDTH}}" for c in CAT_LABELS)
    lines = [head + cats]
    lines += [f"{i:>5}{_metric_cells(r)}{_cat_cells(r['gpu_ms'], '.3f')}"
              for i, r in enumerate(rows)]
    lines += ["", "ms per prefill token", f"{'step':>5}{'tokens':>7}{cats}"]
    lines += [f"{i:>5}{r['tokens']:>7}{_cat_cells(r['ms_per_token'], '.5f')}"
              for i, r in enumerate(rows)]
    return "\n".join(lines)


def take_json_arg(argv):
    """Split out '--json PATH' and return (remaining argv, PATH or None)."""
    if "--json" not in argv:
        return argv, None
    i = argv.index("--json")
    if i + 1 >= len(argv):
        raise SystemExit("--json needs a PATH")
    return argv[:i] + argv[i + 2:], argv[i + 1]


def prefill_report(paths, json_out=None):
    """Print the prefill-only table per trace; optionally dump all rows as JSON."""
    report = []
    for path in paths:
        rows = prefill_rows(load_trace(path))
        print(f"== {path}")
        print(prefill_table(rows))
        report.append({"trace": str(path), "steps": rows})
    if json_out:
        with open(json_out, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0


def text_report(path, cpu, gaps):
    """Return the text report for one trace in the selected mode."""
    events = load_trace(path)
    if gaps:
        return gap_report(events)
    if cpu:
        return "\n".join((cpu_report(events, "mixed"), cpu_report(events, "decode")))
    return format_report(analyse(events))


def main(argv=None):
    """CLI: step_profile.py [--cpu|--gaps|--prefill [--json PATH]] TRACE [TRACE...]."""
    argv = sys.argv[1:] if argv is None else argv
    modes = {f for f in MODE_FLAGS if f in argv}
    argv, json_out = take_json_arg(argv)
    argv = [a for a in argv if a not in MODE_FLAGS]
    if not argv:
        print(__doc__)
        return 2
    if json_out and "--prefill" not in modes:
        print("--json requires --prefill", file=sys.stderr)
        return 2
    if "--prefill" in modes:
        return prefill_report(argv, json_out)
    for path in argv:
        print(f"== {path}")
        print(text_report(path, "--cpu" in modes, "--gaps" in modes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
