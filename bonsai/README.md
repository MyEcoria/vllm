# bonsai

Local bonsai serving stack: launcher plus GGUF quantization plugin for vLLM.

## Layout

- `serve/`: launcher `bonsai`, `Opti.md`, `tools/`.
- `gguf_plugin/`: package `vllm_gguf_plugin`.

## Plugin origin

Upstream `vllm-project/vllm-gguf-plugin` at commit `e454affe0ca23c9dc8754ec3033270cb30622ed5`, plus local branch `bonsai-ternary-hadamard`. Includes uncommitted changes. Compiled `*.so` files are not included; they are rebuilt on install.

## Install

Editable plugin install, from the vLLM repo root:

```bash
uv pip install -e bonsai/gguf_plugin --no-build-isolation --no-deps --python .venv/bin/python
```

Put the launcher on your PATH:

```bash
ln -sfn /home/volta/Desktop/Perso/vllm/bonsai/serve/bonsai ~/.local/bin/bonsai
```

## Run

```bash
bonsai launch --detach
```

The server listens on port 8000. The API key is read from `~/.bonsai/api_key`.

## Benchmarks

All tools take `--url` (default `$BONSAI_URL`, else `http://127.0.0.1:8000`). The API key comes from `BONSAI_API_KEY`, else `~/.bonsai/api_key`.

```bash
.venv/bin/python bonsai/serve/tools/serve_bench.py --url http://127.0.0.1:8000
.venv/bin/python bonsai/serve/tools/longctx.py --url http://127.0.0.1:8000
.venv/bin/python bonsai/serve/tools/agent_bench.py --url http://127.0.0.1:8000 --sync-turns
```

- `serve_bench.py` prints one `SERVE` line: prefill speed (pp512, tokens/s, with TTFT) and decode speed (tg, tokens/s).
- `longctx.py` sends one long prompt (`LONGWORDS` words, default 55000) with a hidden number and prints a `LONG` line with the token count, time, and whether the number was found.
- `agent_bench.py` runs 4 agents with distinct growing histories (90k start tokens, +2-4k per turn, 256 output tokens, `--max-context` 100000). It prints one line per request with cached and prompt tokens, prefill and decode speeds, then a per-turn summary. `--sync-turns` makes all agents start each turn together. Needs the server flag `--enable-prompt-tokens-details`.

## Tests

```bash
cd bonsai/gguf_plugin && ../../.venv/bin/python -m pytest -q tests/test_ternary_hadamard.py
```

Expected result: 104 passed.

## Roadmap

- F1: agent benchmark and hybrid prefix cache.
- F2: adaptive prefill chunking.
- F3: Windowed-MTP.
- F4: HIP attention kernels.
- F5: optional sparse decode attention.

Targets: 4 agents at 100k context, at least 600 t/s delta prefill, at least 60 t/s decode.
