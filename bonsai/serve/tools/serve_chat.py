import json, time, urllib.request, threading, collections
URL = "http://127.0.0.1:8011/v1/chat/completions"
clk = collections.Counter(); stop = [False]
def sample():
    while not stop[0]:
        for f in ("pp_dpm_sclk", "pp_dpm_mclk"):
            for line in open(f"/sys/class/drm/card1/device/{f}"):
                if "*" in line: clk[(f, line.split(":")[1].strip())] += 1
        time.sleep(0.05)
def run(msg, n):
    body = json.dumps({"model": "bonsai", "messages": [{"role": "user", "content": msg}], "max_tokens": n,
                       "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    first = None
    with urllib.request.urlopen(req) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line and first is None:
                first = time.perf_counter()
    return (n - 1) / (time.perf_counter() - first)
prompts = ["Write a detailed essay about the history of the Roman Empire, from its founding to its fall.",
           "Explain step by step how a compiler turns source code into machine code.",
           "Write a Python implementation of merge sort with comments."]
run("hi", 8)
th = threading.Thread(target=sample); th.start()
tgs = [run(p, 512) for p in prompts]
stop[0] = True; th.join()
print(f"CHAT tg {sum(tgs)/len(tgs):.1f} t/s per-prompt {[round(t,1) for t in tgs]}")
print("CLOCKS", clk.most_common(8))
