import argparse, json, os, time, urllib.request, random
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
    parser = argparse.ArgumentParser(description="Bonsai long-context check.")
    parser.add_argument("--url", type=base_url,
                        default=os.environ.get("BONSAI_URL", DEFAULT_URL),
                        help="server base URL (default: $BONSAI_URL or %(default)s)")
    return parser.parse_args()
ARGS = parse_args()
URL = ARGS.url + "/v1/chat/completions"
HEADERS = build_headers()
n_words = int(os.environ.get("LONGWORDS", "55000"))
random.seed(1)
words = ["alpha", "river", "stone", "light", "forest", "cloud", "music", "silver", "garden", "window"]
filler = [random.choice(words) for _ in range(n_words)]
filler.insert(n_words // 2, ". The secret code is 7429. ")
msg = "Read this text:\n" + " ".join(filler) + "\n\nWhat is the secret code mentioned in the text? Reply with just the number."
body = json.dumps({"model": "bonsai", "messages": [{"role": "user", "content": msg}], "max_tokens": 400, "temperature": 0}).encode()
t = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, HEADERS)))
dt = time.perf_counter() - t
n = r["usage"]["prompt_tokens"]; m = r["choices"][0]["message"]
ans = (m.get("content") or "") + " | reasoning: " + (m.get("reasoning_content") or m.get("reasoning") or "")[-120:]
print(f"LONG prompt {n} tokens, total {dt:.1f}s, found={'7429' in ans} -> {ans[:200]!r}")
