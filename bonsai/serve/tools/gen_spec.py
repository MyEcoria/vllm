import os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
def main():
    spec = {"method": "mtp", "num_speculative_tokens": int(os.environ.get("K", "3"))}
    if os.environ.get("DFLASH"):
        spec = {"method": "dflash", "model": os.environ["DFLASH"], "num_speculative_tokens": int(os.environ.get("DK", "7"))}
        if os.environ.get("DQ"):
            spec["quantization"] = os.environ["DQ"]
    llm = LLM(model=sys.argv[1], tokenizer="Qwen/Qwen3.8-27B", quantization="gguf", max_model_len=int(os.environ.get("MAXLEN", "4096")),
              gpu_memory_utilization=float(os.environ.get("GMU", "0.92")), limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=False, max_num_seqs=4, attention_backend="TRITON_ATTN",
              speculative_config=spec if os.environ.get("K", "3") != "0" else None)
    p = "Write a detailed essay about the history of the Roman Empire, from its founding to its fall."
    sp = SamplingParams(temperature=0, max_tokens=int(os.environ.get("TG", "400")), ignore_eos=True)
    llm.generate([p], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
    t = time.perf_counter()
    o = llm.generate([p], sp, use_tqdm=False)
    dt = time.perf_counter() - t
    print("TG", len(o[0].outputs[0].token_ids) / dt, flush=True)
if __name__ == "__main__":
    main()
