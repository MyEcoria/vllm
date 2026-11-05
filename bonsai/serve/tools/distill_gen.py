"""Phase 1: Bonsai greedy responses for the distillation prompts."""
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

def main():
    d = os.path.expanduser("~/.cache/bonsai-distill")
    prompts = json.load(open(f"{d}/prompts.json"))
    llm = LLM(model=sys.argv[1], tokenizer="Qwen/Qwen3.8-27B", quantization="gguf", max_model_len=2048,
              gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=False, max_num_seqs=16, attention_backend="TRITON_ATTN",
              speculative_config={"method": "mtp", "num_speculative_tokens": 3})
    tok = llm.get_tokenizer()
    ids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                              tokenize=False), add_special_tokens=False) for p in prompts]
    print("example prompt:", repr(tok.decode(ids[0])), flush=True)
    sp = SamplingParams(temperature=0, max_tokens=640)
    t = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=i) for i in ids], sp)
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"generated {n} tokens in {time.perf_counter() - t:.0f}s", flush=True)
    json.dump([{"prompt": list(o.prompt_token_ids), "output": list(o.outputs[0].token_ids)} for o in outs],
              open(f"{d}/samples.json", "w"))
    print("sample:", tok.decode(outs[0].outputs[0].token_ids)[:400])

if __name__ == "__main__":
    main()
