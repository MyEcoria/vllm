"""Independent numpy reference for Q2_0 / PQ2_0 / PTQ1_0 and prism.hadamard.

Written from the llama.cpp reference loops (ggml-quants.c), not from the
plugin code, so it can cross-check the plugin.
"""

import numpy as np

SIZES = {42: (64, 18), 142: (128, 34), 143: (128, 28)}


def dequant(raw: np.ndarray, qtype: int, rows: int, cols: int) -> np.ndarray:
    qk, nbytes = SIZES[qtype]
    blocks = np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(-1, nbytes)
    assert blocks.shape[0] == rows * cols // qk
    out = np.empty((blocks.shape[0], qk), dtype=np.float32)
    if qtype in (42, 142):
        d = blocks[:, :2].copy().view("<f2").astype(np.float32)[:, 0]
        for j in range(qk):
            q = (blocks[:, 2 + j // 4] >> ((j % 4) * 2)) & 3
            out[:, j] = (q.astype(np.int32) - 1) * d
    else:
        d = blocks[:, 26:28].copy().view("<f2").astype(np.float32)[:, 0]
        pow3 = [1, 3, 9, 27, 81, 243]
        y = 0
        for j0, c in ((0, 16), (16, 8)):
            for n in range(5):
                for m in range(c):
                    q = (blocks[:, j0 + m].astype(np.uint32) * pow3[n]) & 0xFF
                    out[:, y] = (((q * 3) >> 8).astype(np.int32) - 1) * d
                    y += 1
        for n in range(4):
            for h in range(2):
                q = (blocks[:, 24 + h].astype(np.uint32) * pow3[n]) & 0xFF
                out[:, y] = (((q * 3) >> 8).astype(np.int32) - 1) * d
                y += 1
    return out.reshape(rows, cols)


def hadamard(n: int) -> np.ndarray:
    idx = np.arange(n)
    parity = np.zeros((n, n), dtype=np.int64)
    a = idx[:, None] & idx[None, :]
    while a.any():
        parity ^= a & 1
        a = a >> 1
    return np.where(parity, -1.0, 1.0) / np.sqrt(n)


def unfold_weight(w_f: np.ndarray, signs: np.ndarray, block: int) -> np.ndarray:
    """W = W_f H D (blockwise along the input dimension)."""
    rows, cols = w_f.shape
    h = hadamard(block)
    w = (w_f.reshape(rows, cols // block, block) @ h).reshape(rows, cols)
    return w * signs[None, :]


def unfold_embedding_rows(latent: np.ndarray, signs: np.ndarray, block: int):
    """e = D (H l) per row."""
    rows, cols = latent.shape
    h = hadamard(block)
    e = (latent.reshape(rows, cols // block, block) @ h).reshape(rows, cols)
    return e * signs[None, :]
