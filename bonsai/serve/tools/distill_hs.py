"""Phase 2: capture Bonsai's final (post-norm) hidden states for each sample.

Runs vLLM in-process and eager, hooks the text backbone and prefills every
prompt+response. Hidden states go to one bf16 memmap with an offsets index.
"""

import json
import os
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

CAPTURED: list[torch.Tensor] = []


def _backbone(model):
    target = None
    for name, module in model.named_modules():
        if name.endswith("language_model.model.norm") or name == "model.norm":
            target = (name, module)
    return target


def install_hook(model):
    name, module = _backbone(model)
    module._hs_buf = []

    def hook(mod, _inp, out):
        out = out[0] if isinstance(out, tuple) else out
        mod._hs_buf.append(out.detach().to("cpu", torch.bfloat16))

    module.register_forward_hook(hook)
    return name


def take(model):
    module = _backbone(model)[1]
    buf = list(module._hs_buf)
    module._hs_buf.clear()
    return buf


def main():
    d = os.path.expanduser("~/.cache/bonsai-distill")
    tag = os.environ.get("TAG", "")
    samples = json.load(open(f"{d}/samples{tag}.json"))
    llm = LLM(model=sys.argv[1], tokenizer="Qwen/Qwen3.8-27B", quantization="gguf",
              max_model_len=2048, max_num_batched_tokens=2048, gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=False,
              max_num_seqs=1, attention_backend="TRITON_ATTN", enforce_eager=True)
    print("hooked", llm.apply_model(install_hook), flush=True)
    total = sum(len(s["prompt"]) + len(s["output"]) for s in samples)
    hs = np.lib.format.open_memmap(f"{d}/hidden{tag}.npy", mode="w+", dtype=np.uint16, shape=(total, 5120))
    index, off = [], 0
    sp = SamplingParams(max_tokens=1)
    for i, s in enumerate(samples):
        ids = s["prompt"] + s["output"]
        llm.apply_model(take)
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        got = llm.apply_model(take)[0]
        h = got[0][: len(ids)]
        assert h.shape == (len(ids), 5120), (len(ids), [c.shape for c in got])
        hs[off:off + len(ids)] = h.view(torch.int16).numpy().view(np.uint16)
        index.append({"offset": off, "prompt_len": len(s["prompt"]), "tokens": ids})
        off += len(ids)
        if i % 100 == 0:
            print(f"{i}/{len(samples)} tokens {off}/{total}", flush=True)
    hs.flush()
    json.dump(index, open(f"{d}/index{tag}.json", "w"))
    print("done", off)


if __name__ == "__main__":
    main()
