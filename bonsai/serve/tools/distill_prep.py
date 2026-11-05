"""Prepare frozen tensors for MTP distillation (CPU).

- Bonsai token embedding and LM head, dequantized from Q2_0 and unfolded
  (rows times the block Hadamard, then the sign flip), saved as bf16.
- Qwen3.8-27B MTP weights (bf16), range-read from the Hugging Face shard.
"""

import json
import os
import struct
import sys
import urllib.request

import torch
from safetensors.torch import save_file

import vllm_gguf_plugin.ggml_types  # noqa: F401
from gguf import GGUFReader
from vllm_gguf_plugin.hadamard import load_prism_hadamard_config, sylvester_hadamard
from vllm_gguf_plugin.quantization.ternary_fast import _decode_blocks

SHARD = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model-00018-of-00018.safetensors"
D = os.path.expanduser("~/.cache/bonsai-distill")


def unfold(name, reader, signs, h):
    t = next(t for t in reader.tensors if t.name == name)
    cols, rows = (int(x) for x in t.shape)
    raw = torch.from_numpy(t.data.copy()).view(rows, -1)
    out = torch.empty(rows, cols, dtype=torch.bfloat16)
    step = 8192
    for r0 in range(0, rows, step):
        blocks = raw[r0:r0 + step].view(min(step, rows - r0), -1, 18)
        codes, scales = _decode_blocks(blocks, int(t.tensor_type))
        w = (codes.float() - 1).view(codes.shape[0], -1, 64) * scales.float()[..., None]
        w = w.view(codes.shape[0], cols)
        w = (w.view(-1, cols // 1024, 1024) @ h).view(-1, cols) * signs
        out[r0:r0 + step] = w.to(torch.bfloat16)
    return out


def fetch(start, length):
    req = urllib.request.Request(SHARD, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def main():
    torch.set_num_threads(10)
    path = sys.argv[1]
    reader = GGUFReader(path)
    cfg = load_prism_hadamard_config(path)
    signs = cfg.signs[5120].float()
    h = sylvester_hadamard(1024, torch.float32)
    for name, out in (("output.weight", "lm_head.pt"), ("token_embd.weight", "embed.pt")):
        if not os.path.exists(f"{D}/{out}"):
            w = unfold(name, reader, signs, h)
            torch.save(w, f"{D}/{out}")
            print(name, w.shape, w.float()[:2, :4], flush=True)
    (hlen,) = struct.unpack("<Q", fetch(0, 8))
    header = json.loads(fetch(8, hlen))
    base = 8 + hlen
    tensors = {}
    for k, info in header.items():
        if not k.startswith("mtp."):
            continue
        s, e = info["data_offsets"]
        raw = torch.frombuffer(bytearray(fetch(base + s, e - s)), dtype=torch.bfloat16)
        tensors[k] = raw.view(info["shape"]).clone()
    save_file(tensors, f"{D}/mtp_init.safetensors")
    print("mtp tensors", len(tensors))


if __name__ == "__main__":
    main()
