import argparse, json, os, random, sys, time, urllib.request
from pathlib import Path
DEFAULT_URL = "http://127.0.0.1:8000"
def base_url(value):
    """Validate a server base URL and strip trailing slashes."""
    if not value.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError(f"not an http(s) URL: {value}")
    return value.rstrip("/")
def load_api_key():
    """Return BONSAI_API_KEY, else the key in ~/.bonsai/api_key, else None."""
    key = os.environ.get("BONSAI_API_KEY")
    key_file = Path.home() / ".bonsai" / "api_key"
    if not key and key_file.is_file():
        key = key_file.read_text().strip()
    return key or None
def build_headers():
    """Return JSON headers, with a Bearer token when a key is configured."""
    headers = {"Content-Type": "application/json"}
    key = load_api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers
def parse_args():
    """Parse --url; the default is BONSAI_URL, else DEFAULT_URL."""
    parser = argparse.ArgumentParser(description="Benchmark a Bonsai server.")
    parser.add_argument("--url", type=base_url,
                        default=os.environ.get("BONSAI_URL", DEFAULT_URL),
                        help="server base URL (default: $BONSAI_URL or %(default)s)")
    return parser.parse_args()
ARGS = parse_args()
URL = ARGS.url + "/v1/completions"
HEADERS = build_headers()
def run(prompt, n):
    body = json.dumps({"model": "bonsai", "prompt": prompt, "max_tokens": n, "temperature": 0,
                       "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, body, HEADERS)
    t0 = time.perf_counter(); first = None
    with urllib.request.urlopen(req) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line and first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    return first - t0, (n - 1) / (end - first) if n > 1 else 0.0
random.seed(0)
for _ in range(2):
    run([random.randint(1000, 100000) for _ in range(512)], 4)
ttft = min(run([random.randint(1000, 100000) for _ in range(512)], 1)[0] for _ in range(3))
prompts = [
    "Write a detailed essay about the history of the Roman Empire, from its founding to its fall.",
    "Explain step by step how a compiler turns source code into machine code.",
    "def merge_sort(arr):\n    \"\"\"Sort a list using merge sort.\"\"\"\n",
]
tgs = [run(p, 512)[1] for p in prompts]
print(f"SERVE pp512 {512 / ttft:.0f} t/s (ttft {ttft * 1000:.0f} ms)  tg {sum(tgs) / len(tgs):.1f} t/s  per-prompt {[round(t, 1) for t in tgs]}")
