"""Fine-tune the Qwen3.8 MTP head on Bonsai's own hidden states (self-distillation).

Mirrors vLLM's Qwen3.5 MTP drafting: step s at slot t takes the embedding of
x[t+s] and the previous hidden (Bonsai's post-norm hidden for s=1, the MTP
output otherwise) at rope position t+s-1, attends to step-1 keys of slots <= t
plus its own keys from steps 2..s at slot t, and predicts x[t+s+1].
"""

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

D = os.path.expanduser("~/.cache/bonsai-distill")
H, NH, NKV, HD, ROT, EPS, THETA = 5120, 24, 4, 256, 64, 1e-6, 1e7
STEPS = 3
dev = "cuda"


def rms(x, w):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    return (y * (1.0 + w.float())).to(x.dtype)


def rope(x, cos, sin):
    xr, xp = x[..., :ROT], x[..., ROT:]
    x1, x2 = xr[..., : ROT // 2], xr[..., ROT // 2:]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos + rot * sin, xp], -1)


class MTP(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.names = sorted(sd)
        for k in self.names:
            self.register_parameter(k.replace(".", "__"), nn.Parameter(sd[k].float()))

    def p(self, k):
        return getattr(self, ("mtp." + k).replace(".", "__"))

    def state(self):
        return {k: getattr(self, k.replace(".", "__")).detach().to(torch.bfloat16).contiguous()
                for k in self.names}

    def step(self, emb, hid, pos, keys, values, lm_head):
        L = emb.shape[0]
        x = torch.cat([rms(emb, self.p("pre_fc_norm_embedding.weight")),
                       rms(hid, self.p("pre_fc_norm_hidden.weight"))], -1)
        x = F.linear(x, self.p("fc.weight"))
        resid = x
        a = rms(x, self.p("layers.0.input_layernorm.weight"))
        q, gate = torch.chunk(F.linear(a, self.p("layers.0.self_attn.q_proj.weight")).view(L, NH, 2 * HD), 2, -1)
        gate = gate.reshape(L, NH * HD)
        q = rms(q, self.p("layers.0.self_attn.q_norm.weight")).transpose(0, 1)
        k = rms(F.linear(a, self.p("layers.0.self_attn.k_proj.weight")).view(L, NKV, HD),
                self.p("layers.0.self_attn.k_norm.weight")).transpose(0, 1)
        v = F.linear(a, self.p("layers.0.self_attn.v_proj.weight")).view(L, NKV, HD).transpose(0, 1)
        inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, device=emb.device).float() / ROT))
        f = pos.float()[:, None] * inv
        cos = torch.cat([f.cos(), f.cos()], -1).to(q.dtype)
        sin = torch.cat([f.sin(), f.sin()], -1).to(q.dtype)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        keys.append(k)
        values.append(v)
        s = len(keys)
        kc = torch.cat(keys, 1).repeat_interleave(NH // NKV, 0)
        vc = torch.cat(values, 1).repeat_interleave(NH // NKV, 0)
        idx = torch.arange(L, device=emb.device)
        mask = [idx[:, None] >= idx[None, :]] + [idx[:, None] == idx[None, :]] * (s - 1)
        mask = torch.cat(mask, 1)
        o = F.scaled_dot_product_attention(q[None], kc[None], vc[None], attn_mask=mask[None, None])[0]
        o = o.transpose(0, 1).reshape(L, NH * HD) * torch.sigmoid(gate)
        x = resid + F.linear(o, self.p("layers.0.self_attn.o_proj.weight"))
        m = rms(x, self.p("layers.0.post_attention_layernorm.weight"))
        m = F.linear(F.silu(F.linear(m, self.p("layers.0.mlp.gate_proj.weight")))
                     * F.linear(m, self.p("layers.0.mlp.up_proj.weight")),
                     self.p("layers.0.mlp.down_proj.weight"))
        x = x + m
        return rms(x, self.p("norm.weight"))


def chunk_loss(out, lm_head, tgt):
    logits = F.linear(out, lm_head).float()
    loss = F.cross_entropy(logits, tgt, reduction="sum")
    return loss, (logits.argmax(-1) == tgt).sum()


def lm_loss(out, lm_head, tgt, mask):
    total = out.new_zeros((), dtype=torch.float32)
    hits = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    for c0 in range(0, out.shape[0], 384):
        sl = slice(c0, c0 + 384)
        o = out[sl]
        loss = checkpoint(lambda o_, t_, m_: F.cross_entropy(F.linear(o_, lm_head).float(), t_,
                                                             reduction="none").mul(m_).sum(),
                          o, tgt[sl], mask[sl].float(), use_reentrant=False)
        total = total + loss
        with torch.no_grad():
            hits[sl] = F.linear(o, lm_head).argmax(-1) == tgt[sl]
    return total, hits


def run_sample(model, sample, hidden, embed, lm_head, train):
    toks = torch.tensor(sample["tokens"][: int(os.environ.get("MAXTOK", "100000"))], device=dev)
    T, P = len(toks), sample["prompt_len"]
    L = T - STEPS - 1
    if L < 8:
        return None
    h_arr = hidden[sample["tag"]] if isinstance(hidden, dict) else hidden
    hid = torch.from_numpy(np.asarray(h_arr[sample["offset"]: sample["offset"] + L])).view(torch.bfloat16).to(dev)
    t = torch.arange(L, device=dev)
    keys, values = [], []
    h = hid
    total = torch.zeros((), device=dev)
    stats = []
    for s in range(1, STEPS + 1):
        emb = embed[toks[s: s + L].cpu()].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.step(emb, h, t + s - 1, keys, values, lm_head)
            tgt = toks[t + s + 1]
            mask = (t + s + 1) >= P
            loss, hits = lm_loss(out, lm_head, tgt, mask)
        total = total + loss * (1.0 if s == 1 else 0.8 ** (s - 1))
        stats.append((hits & mask, mask))
        h = out
    n = int(stats[0][1].sum())
    return total, n, stats


def evaluate(model, samples, hidden, embed, lm_head):
    model.eval()
    acc = np.zeros(STEPS)
    chain = np.zeros(STEPS)
    n = 0
    with torch.no_grad():
        for smp in samples:
            r = run_sample(model, smp, hidden, embed, lm_head, False)
            if r is None:
                continue
            _, cnt, stats = r
            mask = stats[0][1]
            run = torch.ones_like(mask)
            for s, (hits, _) in enumerate(stats):
                run = run & hits
                acc[s] += float((hits & mask).sum())
                chain[s] += float((run & mask).sum())
            n += cnt
    model.train()
    torch.cuda.empty_cache()
    return acc / n, chain / n, 1 + (chain / n).sum()


def main():
    epochs = float(os.environ.get("EPOCHS", "2"))
    lr = float(os.environ.get("LR", "5e-5"))
    tags = os.environ.get("TAGS", "").split(",")
    hiddens = {t: np.load(f"{D}/hidden{t}.npy", mmap_mode="r") for t in tags}
    index = []
    for t in tags:
        for e in json.load(open(f"{D}/index{t}.json")):
            e["tag"] = t
            index.append(e)
    random.seed(0)
    first = [e for e in index if e["tag"] == tags[0]]
    random.shuffle(first)
    val = first[:80]
    val_ids = {id(e) for e in val}
    train = [e for e in index if id(e) not in val_ids]
    hidden = hiddens
    embed = torch.load(f"{D}/embed.pt")
    lm_head = torch.load(f"{D}/lm_head.pt").to(dev)
    model = MTP(load_file(f"{D}/" + os.environ.get("INIT", "mtp_init.safetensors"))).to(dev)
    a, c, al = evaluate(model, val, hidden, embed, lm_head)
    print(f"init: step acc {np.round(a, 3)} chain {np.round(c, 3)} expected accept len {al:.3f}", flush=True)
    if epochs == 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    tokens_per_update = int(os.environ.get("TPU", "12000"))
    total_updates = int(epochs * sum(len(s["tokens"]) - len(s["tokens"][:s["prompt_len"]]) for s in train)
                        / tokens_per_update) + 1
    step_i, acc_tokens, acc_loss = 0, 0, 0.0
    t0 = time.time()
    ep = 0.0
    order = []
    while ep < epochs:
        order += random.sample(train, len(train))
        ep += 1
    order = order[: int(len(train) * epochs)]
    for i, smp in enumerate(order):
        r = run_sample(model, smp, hidden, embed, lm_head, True)
        if r is None:
            continue
        loss, n, _ = r
        if n == 0:
            continue
        loss.backward()
        acc_tokens += n
        acc_loss += float(loss)
        if acc_tokens >= tokens_per_update or i == len(order) - 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= acc_tokens
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            warm = min(1.0, (step_i + 1) / 10)
            cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step_i / total_updates)))
            for g in opt.param_groups:
                g["lr"] = lr * warm * cos
            opt.step()
            opt.zero_grad(set_to_none=True)
            step_i += 1
            if step_i % 5 == 0:
                print(f"update {step_i}/{total_updates} sample {i}/{len(order)} loss/tok {acc_loss / acc_tokens:.3f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            acc_tokens, acc_loss = 0, 0.0
            if step_i % 60 == 0:
                a, c, al = evaluate(model, val, hidden, embed, lm_head)
                print(f"  eval: step acc {np.round(a, 3)} chain {np.round(c, 3)} accept len {al:.3f}", flush=True)
    a, c, al = evaluate(model, val, hidden, embed, lm_head)
    print(f"final: step acc {np.round(a, 3)} chain {np.round(c, 3)} expected accept len {al:.3f}", flush=True)
    save_file(model.state(), f"{D}/" + os.environ.get("OUT", "mtp_tuned.safetensors"))
    print("saved")


if __name__ == "__main__":
    main()
