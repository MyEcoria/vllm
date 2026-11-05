"""Quality checks for a running bonsai server: nll, needle, greedy."""
import argparse
import json
import random
import subprocess
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

from agent_bench import build_headers, make_text, post_json

SEGMENTS = ((0, 2000), (2000, 6000), (6000, 12000))
GREEDY_PROMPTS = (
    "Write a Python function that merges two sorted lists.",
    "Explain in three sentences how a hash map handles collisions.",
    "List the first twelve prime numbers, separated by commas.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
)
REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_COMMIT = "c4c1a88696"
MAX_FILES = 400


def token_logprobs(prompt_logprobs):
    """Return the logprob of each prompt token, skipping the first (None)."""
    out = []
    for entry in prompt_logprobs:
        if entry is not None:
            out.append(next(iter(entry.values()))["logprob"])
    return out


def mean_nll(logprobs):
    """Return the mean negative log-likelihood of a logprob list."""
    if not logprobs:
        raise ValueError("no logprobs")
    return -sum(logprobs) / len(logprobs)


def segment_nll(logprobs, segments=SEGMENTS):
    """Return the mean NLL of each non-empty [start, end) slice."""
    return [mean_nll(logprobs[a:b]) for a, b in segments if logprobs[a:b]]


def insert_needle(text, code, depth):
    """Insert the needle sentence at fraction `depth` of text, on a line start."""
    pos = text.find("\n", int(len(text) * depth)) + 1
    return text[:pos] + f"# The secret code is {code}.\n" + text[pos:]


def needle_found(answer, code):
    """Return True when the answer contains the needle code."""
    return str(code) in (answer or "")


def first_divergence(a, b):
    """Return the index of the first differing char, or None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def match_ratio(a, b):
    """Return the similarity ratio of two strings in [0, 1]."""
    return 1.0 if a == b else SequenceMatcher(None, a, b, autojunk=False).ratio()


def select_files(ls_output):
    """Return the sorted first MAX_FILES vllm/*.py paths of a git ls-tree listing."""
    paths = [p for p in ls_output.splitlines() if p.endswith(".py")]
    return sorted(paths)[:MAX_FILES]


def git_out(*cmd):
    """Run git in the repo root and return stdout text."""
    return subprocess.run(["git", *cmd], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True,
                          errors="ignore").stdout


def default_texts(args, headers):
    """Return two code texts of about args.tokens tokens from the pinned commit."""
    listing = git_out("ls-tree", "-r", "--name-only", CODE_COMMIT, "vllm/")
    blob = "".join(git_out("show", f"{CODE_COMMIT}:{p}")
                   for p in select_files(listing))
    n = args.tokens * 4
    if len(blob) < 2 * n:
        raise SystemExit("not enough source text for --tokens")
    return [blob[:n], blob[n:2 * n]]


def run_nll(args, headers):
    """Measure mean NLL per token over the code texts."""
    texts = [Path(args.text_file).read_text()] if args.text_file \
        else default_texts(args, headers)
    lps = []
    for text in texts:
        reply = post_json(args.url + "/v1/completions", headers,
                          {"model": args.model, "prompt": text, "max_tokens": 1,
                           "temperature": 0, "prompt_logprobs": 0})
        lps += token_logprobs(reply["choices"][0]["prompt_logprobs"])
    segs = segment_nll(lps)
    line = f"NLL {mean_nll(lps):.4f} tokens={len(lps)} segments=" + \
        "/".join(f"{s:.4f}" for s in segs)
    return line, {"nll": mean_nll(lps), "tokens": len(lps), "segments": segs}


def chat_text(args, headers, prompt, max_tokens):
    """Send one greedy chat request and return content plus reasoning."""
    reply = post_json(args.url + "/v1/chat/completions", headers,
                      {"model": args.model, "temperature": 0,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]})
    m = reply["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + \
        (m.get("content") or "")


def server_count(args, headers, text):
    """Return the /tokenize token count of text, or None if unavailable."""
    try:
        reply = post_json(args.url + "/tokenize", headers,
                          {"model": args.model, "prompt": text})
        return int(reply["count"])
    except (urllib.error.URLError, KeyError, ValueError):
        return None


def fit_text(text, target, count):
    """Trim text to about `target` tokens; estimate with 3% margin if no count."""
    if count is None:
        count, target = max(1, len(text) // 4), int(target * 0.97)
    return text if count <= target else text[:len(text) * target // count]


def needle_request(args, headers, rng_seed):
    """Run one needle request and return (found, depth, code)."""
    rng = random.Random(rng_seed)
    code, depth = rng.randrange(1000, 10000), rng.uniform(0.05, 0.95)
    text = make_text(rng, args, headers, args.tokens)
    text = fit_text(text, args.tokens, server_count(args, headers, text))
    text = insert_needle(text, code, depth)
    prompt = (f"Read this code:\n{text}\n\nWhat is the secret code mentioned "
              "in a comment? Reply with just the number.")
    return needle_found(chat_text(args, headers, prompt, args.max_tokens),
                        code), depth, code


def run_needle(args, headers):
    """Run 4 parallel needle requests and report found k/4."""
    seeds = [args.seed * 100 + i for i in range(4)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda s: needle_request(args, headers, s), seeds))
    k = sum(r[0] for r in res)
    rows = [{"found": f, "depth": d, "code": c} for f, d, c in res]
    return f"NEEDLE found {k}/4 tokens~{args.tokens}", {"found": k, "runs": rows}


def run_greedy(args, headers):
    """Run the fixed greedy prompts; save or compare against a baseline."""
    outs = [chat_text(args, headers, p, args.max_tokens) for p in GREEDY_PROMPTS]
    data = {"prompts": list(GREEDY_PROMPTS), "outputs": outs}
    if args.save:
        Path(args.save).write_text(json.dumps(data, indent=1))
    if not args.compare:
        return f"GREEDY saved={bool(args.save)} prompts={len(outs)}", data
    base = json.loads(Path(args.compare).read_text())["outputs"]
    divs = [first_divergence(a, b) for a, b in zip(base, outs)]
    ratios = [match_ratio(a, b) for a, b in zip(base, outs)]
    data.update(divergence=divs, ratios=ratios)
    return (f"GREEDY match {sum(ratios) / len(ratios):.3f} "
            f"first_diverging_char={divs}"), data


def positive(value):
    """Argparse type for a positive int."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0: {value}")
    return n


def port_number(value):
    """Argparse type for a TCP port."""
    n = int(value)
    if not 1 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"bad port: {value}")
    return n


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Bonsai quality checks.")
    p.add_argument("--port", type=port_number, default=8000)
    p.add_argument("--model", default="bonsai")
    p.add_argument("--json", dest="json_path", default=None)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", dest="json_path", default=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("nll", parents=[common])
    n.add_argument("--tokens", type=positive, default=12000)
    n.add_argument("--text-file", default=None)
    d = sub.add_parser("needle", parents=[common])
    d.add_argument("--tokens", type=positive, default=97000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--max-tokens", type=positive, default=400)
    g = sub.add_parser("greedy", parents=[common])
    g.add_argument("--max-tokens", type=positive, default=128)
    g.add_argument("--save", default=None)
    g.add_argument("--compare", default=None)
    args = p.parse_args(argv)
    args.url = f"http://127.0.0.1:{args.port}"
    return args


def main(argv=None):
    """Run the chosen subcommand, print one summary line, optionally dump JSON."""
    args = parse_args(argv)
    runner = {"nll": run_nll, "needle": run_needle, "greedy": run_greedy}
    try:
        line, data = runner[args.cmd](args, build_headers())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        return 1
    print(line)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(data, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
