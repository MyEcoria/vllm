"""Check Bonsai-2-27B GGUF conventions against the Qwen3.8-27B base weights.

Range-downloads one GDN layer and a few embedding rows from both files and
reports correlations between the unfolded ternary weights and the originals.
"""

import json
import struct
import sys
import urllib.request

import numpy as np

sys.path.insert(0, sys.argv[1])
from ternary_ref import dequant, unfold_embedding_rows, unfold_weight  # noqa: E402

GGUF = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-Q2_0.gguf"
BASE = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/"
HEAD = sys.argv[2]


def fetch(url, start, length):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req) as r:
        data = r.read()
    assert len(data) == length, (len(data), length)
    return data


def parse_gguf(path):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    rs = lambda: f.read(rd("Q")[0]).decode()
    sc = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rv(t):
        if t in sc:
            return rd(sc[t])[0]
        if t == 8:
            return rs()
        at, n = rd("I")[0], rd("Q")[0]
        return [rv(at) for _ in range(n)]

    f.read(4)
    _, nt, nkv = rd("IQQ")
    kv = {}
    for _ in range(nkv):
        k = rs()
        kv[k] = rv(rd("I")[0])
    tensors = {}
    for _ in range(nt):
        name = rs()
        nd = rd("I")[0]
        dims = rd("Q" * nd)
        tt = rd("I")[0]
        off = rd("Q")[0]
        tensors[name] = (dims, tt, off)
    data_start = (f.tell() + 31) // 32 * 32
    return kv, tensors, data_start


kv, tensors, data_start = parse_gguf(HEAD)
block = kv["prism.hadamard.block_size"]
signs, off = {}, 0
for w in kv["prism.hadamard.sign_widths"]:
    signs[w] = np.array(kv["prism.hadamard.sign_values"][off : off + w], dtype=np.float64)
    off += w
nv, nk = kv["qwen35.ssm.time_step_rank"], kv["qwen35.ssm.group_count"]
hd, hk = kv["qwen35.ssm.inner_size"] // nv, kv["qwen35.ssm.state_size"]
print("block", block, "nv", nv, "nk", nk, "gdn_v_grouped", kv.get("prism.hadamard.gdn_v_grouped"))


def gguf_matrix(name, rows=None):
    (cols, nrows), tt, toff = tensors[name]
    qk, nbytes = {42: (64, 18)}[tt]
    row_bytes = cols // qk * nbytes
    r0, r1 = rows if rows else (0, nrows)
    raw = fetch(GGUF, data_start + toff + r0 * row_bytes, (r1 - r0) * row_bytes)
    return dequant(np.frombuffer(raw, np.uint8), tt, r1 - r0, cols).astype(np.float64)


header_cache = {}


def base_tensor(name, shard, rows=None):
    url = BASE + shard
    if shard not in header_cache:
        n = struct.unpack("<Q", fetch(url, 0, 8))[0]
        header_cache[shard] = (json.loads(fetch(url, 8, n)), 8 + n)
    header, base = header_cache[shard]
    meta = header[name]
    assert meta["dtype"] == "BF16", meta["dtype"]
    shape = meta["shape"]
    start, end = meta["data_offsets"]
    row_bytes = 2 * int(np.prod(shape[1:]))
    if rows:
        start, end = start + rows[0] * row_bytes, start + rows[1] * row_bytes
        shape = [rows[1] - rows[0], *shape[1:]]
    raw = np.frombuffer(fetch(url, base + start, end - start), np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape).astype(np.float64)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def vperm(unit):
    return np.arange(nv * unit).reshape(nv // nk, nk, unit).transpose(1, 0, 2).reshape(-1)


S1 = "model-00001-of-00018.safetensors"
P = "model.language_model.layers.0.linear_attn."

w_f = gguf_matrix("blk.0.ssm_out.weight")
w = unfold_weight(w_f, signs[w_f.shape[1]], block)
ref = base_tensor(P + "out_proj.weight", S1)
w_tiled = w[:, vperm(hd)]
print(f"out_proj  as stored (grouped): corr={corr(w, ref):.3f}   "
      f"tiled->grouped permuted: corr={corr(w_tiled, ref):.3f}   "
      f"no hadamard: corr={corr(w_f, ref):.3f}")
w_nosign = unfold_weight(w_f, np.ones_like(signs[w_f.shape[1]]), block)
print(f"out_proj  without signs: corr={corr(w_nosign, ref):.3f}")

z_f = gguf_matrix("blk.0.attn_gate.weight")
z = unfold_weight(z_f, signs[z_f.shape[1]], block)
ref_z = base_tensor(P + "in_proj_z.weight", S1)
print(f"in_proj_z rows reordered: corr={corr(z[vperm(hd)], ref_z):.3f}   "
      f"rows as stored: corr={corr(z, ref_z):.3f}")

qkv_f = gguf_matrix("blk.0.attn_qkv.weight")
qkv = unfold_weight(qkv_f, signs[qkv_f.shape[1]], block)
ref_qkv = base_tensor(P + "in_proj_qkv.weight", S1)
qk_rows = 2 * nk * hk
v_fixed = np.concatenate([qkv[:qk_rows], qkv[qk_rows:][vperm(hd)]])
print(f"in_proj_qkv q/k part: corr={corr(qkv[:qk_rows], ref_qkv[:qk_rows]):.3f}   "
      f"v reordered: corr={corr(v_fixed[qk_rows:], ref_qkv[qk_rows:]):.3f}   "
      f"v as stored: corr={corr(qkv[qk_rows:], ref_qkv[qk_rows:]):.3f}")

rows = (1000, 1016)
lat = gguf_matrix("token_embd.weight", rows)
emb = unfold_embedding_rows(lat, signs[lat.shape[1]], block)
ref_e = base_tensor("model.language_model.embed_tokens.weight",
                    "model-00003-of-00018.safetensors", rows)
print(f"embed rows inverse (H then D): corr={corr(emb, ref_e):.3f}   "
      f"wrong order (D then H): corr={corr(unfold_embedding_rows(lat * signs[lat.shape[1]], np.ones(lat.shape[1]), block), ref_e):.3f}   "
      f"latent as is: corr={corr(lat, ref_e):.3f}")
