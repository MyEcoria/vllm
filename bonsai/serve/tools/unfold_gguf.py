"""Rewrite a prism.hadamard ternary GGUF as a plain F16 GGUF.

Dequantizes with the independent numpy reference (ternary_ref.py), undoes the
Hadamard folding and drops the prism.* metadata, so the result can be served by
the stock GGUF path as a reference for the ternary + Hadamard implementation.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vllm_gguf_plugin.ggml_types  # noqa: F401,E402  (enum registration only)
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402
from ternary_ref import SIZES, dequant, unfold_embedding_rows, unfold_weight  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
reader = GGUFReader(src)
fields = reader.fields


def field(name):
    return fields[name].contents()


block = field("prism.hadamard.block_size")
widths = field("prism.hadamard.sign_widths")
values = np.asarray(field("prism.hadamard.sign_values"), dtype=np.float32)
signs, off = {}, 0
for w in widths:
    signs[w] = values[off : off + w]
    off += w
assert off == len(values)
forward = set(field("prism.hadamard.weight_names"))
inverse = set(field("prism.hadamard.inverse_weight_names"))

arch = field("general.architecture")
writer = GGUFWriter(dst, arch)
for name, f in fields.items():
    if name.startswith(("GGUF.", "prism.")) or name == "general.architecture":
        continue
    vtype = f.types[0]
    sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
    writer.add_key_value(name, f.contents(), vtype, sub)

n_forward = n_inverse = 0
for t in reader.tensors:
    qtype = int(t.tensor_type)
    if qtype in SIZES:
        cols, rows = (int(x) for x in t.shape)
        w = dequant(np.asarray(t.data), qtype, rows, cols)
        if t.name in forward:
            w = unfold_weight(w, signs[cols], block)
            n_forward += 1
        elif t.name in inverse:
            w = unfold_embedding_rows(w, signs[cols], block)
            n_inverse += 1
        writer.add_tensor(t.name, w.astype(np.float16))
        if t.name == "token_embd.weight" and field("prism.hadamard.tied_output"):
            writer.add_tensor("output.weight", w.astype(np.float16))
    else:
        assert t.name not in forward and t.name not in inverse, t.name
        assert t.tensor_type.name == "F32"; writer.add_tensor(t.name, np.asarray(t.data, dtype=np.float32))
    print(t.name, t.tensor_type.name, flush=True) if t.name.startswith("blk.0.") else None

assert n_forward == len(forward), (n_forward, len(forward))
assert n_inverse == len(inverse)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("done", n_forward, n_inverse)
