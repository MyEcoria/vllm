import json, sys, urllib.request
URL = "http://127.0.0.1:8011/v1/chat/completions"
prompts = ["Write a detailed essay about the history of the Roman Empire, from its founding to its fall.",
           "Explain step by step how a compiler turns source code into machine code.",
           "Write a Python implementation of merge sort with comments.",
           "Quelle est la capitale de l'Australie et pourquoi n'est-ce pas Sydney ?"]
out = []
for p in prompts:
    body = json.dumps({"model": "bonsai", "messages": [{"role": "user", "content": p}], "max_tokens": 256, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"})))
    m = r["choices"][0]["message"]
    out.append((m.get("reasoning_content") or m.get("reasoning") or "") + "||" + (m.get("content") or ""))
json.dump(out, open(sys.argv[1], "w"))
print("dumped", [len(o) for o in out])
