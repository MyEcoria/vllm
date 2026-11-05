"""Copy a Bonsai GGUF and append Qwen3.8-27B's MTP head as a blk.<n>.nextn block.

The MTP tensors are range-read from the Hugging Face safetensors shard. Linear
weights are stored as Q8_0 (or F16), norms as F32 with the GGUF +1 offset that
the plugin's Qwen3.5 MTP adapter removes.
"""

import json
import os
import struct
import sys
import urllib.request

import numpy as np

import vllm_gguf_plugin.ggml_types  # noqa: F401  (registers Q2_0/PQ2_0/PTQ1_0)
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import quantize

SHARD = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model-00018-of-00018.safetensors"
LINEAR_TYPE = GGMLQuantizationType[sys.argv[3]] if len(sys.argv) > 3 else GGMLQuantizationType.Q8_0


def fetch(start, length):
    req = urllib.request.Request(SHARD, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def main():
    src, dst = sys.argv[1], sys.argv[2]
    reader = GGUFReader(src)
    block = int(reader.fields["qwen35.block_count"].contents())

    if not os.environ.get("MTP_LOCAL"):
        (hlen,) = struct.unpack("<Q", fetch(0, 8))
        header = json.loads(fetch(8, hlen))
        base = 8 + hlen

    local = os.environ.get("MTP_LOCAL")
    if local:
        from safetensors.numpy import load_file as np_load
        import torch
        from safetensors.torch import load_file as t_load
        loaded = {k: v.float().numpy() for k, v in t_load(local).items()}
        header = {k: None for k in loaded}

    def tensor(name):
        if local:
            return loaded[name]
        info = header[name]
        assert info["dtype"] == "BF16", info
        s, e = info["data_offsets"]
        raw = np.frombuffer(fetch(base + s, e - s), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])

    mapping = {
        "mtp.fc.weight": "nextn.eh_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "nextn.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm.weight",
        "mtp.norm.weight": "nextn.shared_head_norm.weight",
        "mtp.layers.0.input_layernorm.weight": "attn_norm.weight",
        "mtp.layers.0.post_attention_layernorm.weight": "post_attention_norm.weight",
        "mtp.layers.0.self_attn.q_norm.weight": "attn_q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight": "attn_k_norm.weight",
        "mtp.layers.0.self_attn.q_proj.weight": "attn_q.weight",
        "mtp.layers.0.self_attn.k_proj.weight": "attn_k.weight",
        "mtp.layers.0.self_attn.v_proj.weight": "attn_v.weight",
        "mtp.layers.0.self_attn.o_proj.weight": "attn_output.weight",
        "mtp.layers.0.mlp.gate_proj.weight": "ffn_gate.weight",
        "mtp.layers.0.mlp.up_proj.weight": "ffn_up.weight",
        "mtp.layers.0.mlp.down_proj.weight": "ffn_down.weight",
    }
    assert set(mapping) == {k for k in header if k.startswith("mtp.")}, sorted(
        k for k in header if k.startswith("mtp.")
    )

    writer = GGUFWriter(dst, reader.fields["general.architecture"].contents())
    for name, f in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(name, f.contents(), vtype, sub)

    for t in reader.tensors:
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            writer.add_tensor(t.name, np.asarray(t.data))
        else:
            writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)

    for hf_name, suffix in mapping.items():
        w = tensor(hf_name)
        name = f"blk.{block}.{suffix}"
        if w.ndim == 1:
            writer.add_tensor(name, (w + 1.0).astype(np.float32))
        elif LINEAR_TYPE is None:
            writer.add_tensor(name, w.astype(np.float16))
        else:
            writer.add_tensor(name, quantize(w, LINEAR_TYPE), raw_dtype=LINEAR_TYPE)
        print(name, w.shape, flush=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    print("done")


if __name__ == "__main__":
    main()
