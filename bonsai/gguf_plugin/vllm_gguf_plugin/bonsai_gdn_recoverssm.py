# SPDX-License-Identifier: Apache-2.0
"""RecoverSSM speculative decoding for Qwen3.5/3.6 Gated DeltaNet (bonsai).

Enabled with BONSAI_GDN_RECOVERSSM=1 (requires VLLM_USE_V2_MODEL_RUNNER=1 and MTP).

Stock vLLM keeps one full GDN state copy per draft token (num_speculative_blocks), i.e. with
k=3 and prefix caching 5 states per request and layer group. RecoverSSM (vLLM's KDA
implementation) verifies the draft window off a single checkpoint, records the per-token
corrections, and after sampling folds the accepted ones into the state. GDN is KDA with a
per-head (not per-channel) gate and GQA keys, so the KDA kernels are reused unchanged:
q/k are repeated to the value heads and the scalar gate input is broadcast over channels.
Saves 3 state copies per request (~0.26 GB per request at bf16 state for Qwen3.5-27B).
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch


def patch():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as qg
    from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
        KDARecoverSSMCommitContext,
        KDARecoverSSMRecords,
        kda_recoverssm_verify,
    )
    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.attention.backends.recoverssm_metadata import (
        RecoverSSMMetadata,
        RecoverSSMPostprocessMetadata,
    )
    from vllm.v1.worker import gpu_worker

    # ------------------------------------------------------------------ config
    orig_worker_init = gpu_worker.Worker.__init__

    def worker_init(self, vllm_config, *args, **kwargs):
        if vllm_config.speculative_config is not None and vllm_config.num_speculative_tokens > 0:
            vllm_config.cache_config.use_replayssm = True
            vllm_config.cache_config.use_kda_recoverssm = True
        orig_worker_init(self, vllm_config, *args, **kwargs)

    gpu_worker.Worker.__init__ = worker_init

    # ------------------------------------------------------------- layer state
    by_alog: dict[int, SimpleNamespace] = {}

    def make_proxy(layer, cfg):
        """Per-layer records, allocated at layer construction (never inside a CUDA graph capture:
        capture-time allocations live in the shared graph pool and get overwritten)."""
        hv = layer.num_v_heads // layer.tp_size
        kd = layer.head_k_dim
        q_len = 1 + cfg.num_speculative_tokens
        dev = layer.A_log.device
        records = KDARecoverSSMRecords.allocate(cfg.scheduler_config.max_num_seqs, hv, q_len, kd,
                                                torch.bfloat16, dev)
        return SimpleNamespace(layer=layer, kv_cache=None, recoverssm_records=records, A_log=layer.A_log,
                               dt_bias=None, local_num_heads=hv, head_dim=kd, gate_lower_bound=None,
                               recoverssm_backend="triton", q_len=q_len,
                               rep=hv // (layer.num_k_heads // layer.tp_size))

    def expanded_dt_bias(layer, hv, kd):
        return layer.dt_bias.detach().float()[:, None].expand(hv, kd).contiguous().view(-1)

    def layer_proxy(layer, cfg=None):
        return layer._b_recover

    orig_init = qg.QwenGatedDeltaNetAttention.__init__

    def gdn_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        from vllm.config import get_current_vllm_config

        self._b_cfg = cfg = get_current_vllm_config()
        self._b_recover = None
        if cfg.speculative_config is not None and cfg.num_speculative_tokens > 0:
            self._b_recover = make_proxy(self, cfg)

    qg.QwenGatedDeltaNetAttention.__init__ = gdn_init

    current = {"layer": None}
    orig_core = qg.QwenGatedDeltaNetAttention._forward_core

    def forward_core(self, *args, **kwargs):
        current["layer"] = self
        try:
            return orig_core(self, *args, **kwargs)
        finally:
            current["layer"] = None

    qg.QwenGatedDeltaNetAttention._forward_core = forward_core

    # The fused MTP decode kernel writes per-draft states itself: route spec decodes through
    # _forward_core (-> RecoverSSM verify) instead.
    orig_can_fused = qg.QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode

    def can_fused(self, attn_metadata):
        if self._b_recover is not None and self._b_cfg.cache_config.use_kda_recoverssm:
            return False
        return orig_can_fused(self, attn_metadata)

    qg.QwenGatedDeltaNetAttention._can_use_fused_gdn_mtp_decode = can_fused

    # ------------------------------------------------- verify (no state writes)
    orig_update = qg.fused_sigmoid_gating_delta_rule_update

    def fused_update(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0, scale=None,
                     initial_state=None, inplace_final_state=True, cu_seqlens=None,
                     ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
                     is_kda=False):
        layer = current["layer"]
        if num_accepted_tokens is None or layer is None or is_kda or not layer._b_cfg.cache_config.use_kda_recoverssm:
            return orig_update(A_log, a, b, dt_bias, q, k, v, beta, threshold, scale, initial_state,
                               inplace_final_state, cu_seqlens, ssm_state_indices, num_accepted_tokens,
                               use_qk_l2norm_in_kernel, is_kda)
        p = layer._b_recover
        T = q.shape[1]
        hv, kd = p.local_num_heads, p.head_dim
        out = torch.empty_like(v)
        n = cu_seqlens.shape[0] - 1
        if NATIVE_VERIFY:
            gdn_verify(q, k, v, a, b, A_log, dt_bias, initial_state, p.recoverssm_records, cu_seqlens,
                       ssm_state_indices[:n, 0], p.q_len, out)
        else:
            qe = q.repeat_interleave(p.rep, dim=2).contiguous()
            ke = k.repeat_interleave(p.rep, dim=2).contiguous()
            raw_g = a.reshape(1, T, hv, 1).expand(1, T, hv, kd).contiguous()
            raw_beta = b.reshape(1, T, hv).contiguous()
            kda_recoverssm_verify(qe, ke, v.contiguous(), raw_g, raw_beta, A_log, expanded_dt_bias(layer, hv, kd),
                                  None, initial_state, p.recoverssm_records, cu_seqlens, ssm_state_indices[:n, 0],
                                  p.q_len, out=out)
        if os.environ.get("BONSAI_RECOVER_CMP"):
            idx = ssm_state_indices[:n, 0].long()
            small4 = initial_state[idx].clone().repeat_interleave(p.q_len, 0)
            small4 = torch.cat([small4[:1], small4])  # slot 0 is the null block
            slots = 1 + torch.arange(n * p.q_len, device=q.device, dtype=torch.int32).view(n, p.q_len)
            ones = torch.ones(n, dtype=torch.int32, device=q.device)
            o_ref, _ = orig_update(A_log, a, b, dt_bias, q, k, v, beta, threshold, scale, small4, True, cu_seqlens,
                                   slots, ones, use_qk_l2norm_in_kernel, is_kda)
            o_ref = o_ref.reshape(out.shape)
            e = ((out.float() - o_ref.float()).norm() / (o_ref.float().norm() + 1e-9)).item()
            with open("/tmp/claude-1000/recover_cmp.txt", "a") as f:
                f.write(f"{layer.prefix} n={n} T={T} qshape={tuple(q.shape)} ashape={tuple(a.shape)} a_stride={a.stride()} "
                        f"idx={idx.tolist()} err={e:.3e} refnorm={o_ref.float().norm().item():.3e} outnorm={out.float().norm().item():.3e} st={initial_state[idx].float().norm().item():.3e} nan={bool(torch.isnan(initial_state[idx]).any())} acc_in={num_accepted_tokens[:n].tolist()} si_row={ssm_state_indices[0].tolist()} qsl={cu_seqlens.tolist()}\n")
        return out, initial_state

    NATIVE_VERIFY = os.environ.get("BONSAI_GDN_NATIVE_VERIFY", "1") == "1"
    gdn_verify = _make_gdn_verify()
    qg.fused_sigmoid_gating_delta_rule_update = fused_update

    # --------------------------------------------------------------- metadata
    @dataclass
    class GDNRecoverMetadata(gdn_attn.GDNAttentionMetadata, RecoverSSMMetadata):
        b_commit: tuple | None = None
        b_context: object = field(default=None, repr=False, compare=False)

        def commit_recoverssm_state(self, num_accepted_tokens):
            if self.b_commit is None:
                return None
            state_indices, qsl, request_indices, align = self.b_commit
            n = self.num_spec_decodes
            if os.environ.get("BONSAI_RECOVER_DEBUG"):
                with open("/tmp/claude-1000/recover_debug.txt", "a") as f:
                    f.write(f"n={n} acc={tuple(num_accepted_tokens.shape)} si={tuple(state_indices.shape)} "
                            f"qsl={tuple(qsl.shape)} ri={None if request_indices is None else tuple(request_indices.shape)} "
                            f"bt={None if not align else tuple(align[0].shape)} nc={None if not align else tuple(align[1].shape)} "
                            f"bs={None if not align else align[2]} pf={self.num_prefills} dec={self.num_decodes}\n")
            ctx = context_for(self.b_context)
            ctx.commit(num_accepted_tokens, state_indices[:n, 0], qsl[: n + 1],
                                  request_indices=request_indices,
                                  block_table=align[0] if align else None,
                                  num_computed_tokens=align[1] if align else None,
                                  mamba_block_size=align[2] if align else None)
            if os.environ.get("BONSAI_RECOVER_DEBUG"):
                with open("/tmp/claude-1000/recover_debug.txt", "a") as f:
                    f.write(f"  src={state_indices[:n, 0].tolist()} acc={num_accepted_tokens[:n].tolist()} "
                            f"len={ctx.commit_lens[:n].tolist()} final={ctx.final_state_indices[:n].tolist()} "
                            f"bnd={ctx.boundary_state_indices[:n].tolist()} nc={align[1][:n].tolist() if align else None}\n")
            if align is None:
                return None
            return RecoverSSMPostprocessMetadata(num_spec_decodes=n, request_indices=request_indices,
                                                 block_table=align[0], num_computed_tokens=align[1],
                                                 block_size=align[2])

    builder_cls = gdn_attn.GDNAttentionMetadataBuilder
    orig_build = builder_cls.build

    def context_for(self):
        fc = self.vllm_config.compilation_config.static_forward_context
        layers = [fc[name] for name in self.layer_names]
        key = tuple(layer.kv_cache[1].data_ptr() for layer in layers)
        ctx = getattr(self, "_b_ctx", None)
        if ctx is None or getattr(self, "_b_ctx_key", None) != key:
            proxies = [layer._b_recover for layer in layers]
            for pr in proxies:
                pr.kv_cache = pr.layer.kv_cache
                pr.dt_bias = expanded_dt_bias(pr.layer, pr.local_num_heads, pr.head_dim)
            self._b_ctx_key = key
            ctx = KDARecoverSSMCommitContext.create(proxies, spec_query_len=proxies[0].q_len,
                                                    max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
                                                    backend="triton")
            self._b_ctx = ctx
        return ctx

    def build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens=None,
              num_decode_draft_tokens_cpu=None, fast_build=False):
        md = orig_build(self, common_prefix_len, common_attn_metadata, num_accepted_tokens,
                        num_decode_draft_tokens_cpu, fast_build)
        if not self.vllm_config.cache_config.use_kda_recoverssm:
            return md
        commit = None
        ctx = None
        if md.num_spec_decodes > 0 and md.spec_state_indices_tensor is not None:
            # RecoverSSM folds the accepted tokens (and compacts the conv history) right after
            # sampling, so every verify starts from offset 0, like KDA's builder (always ones).
            na = md.num_accepted_tokens
            if na is not None:
                own = getattr(self, "num_accepted_tokens", None)
                if own is not None and own.untyped_storage().data_ptr() == na.untyped_storage().data_ptr():
                    na.fill_(1)  # builder-owned staging buffer (captured by FULL graphs)
                else:  # a view of the model runner's buffer: never mutate it
                    ones = getattr(self, "_b_ones", None)
                    if ones is None:
                        ones = self._b_ones = torch.ones(1024, dtype=na.dtype, device=na.device)
                    md.num_accepted_tokens = ones[: na.shape[0]]
            # Without speculative state blocks the mamba block table can be one column wide, but
            # GDN sizes the spec conv window from its width (max_query_len): keep it 1 + k wide.
            q_len = 1 + self.vllm_config.num_speculative_tokens
            si = md.spec_state_indices_tensor
            if si.shape[-1] < q_len:
                md.spec_state_indices_tensor = si[:, :1].expand(-1, q_len)
            m = common_attn_metadata
            request_indices = None
            if md.num_prefills > 0 or md.num_decodes > 0:
                mask = md.spec_sequence_masks_cpu
                if mask is None and num_decode_draft_tokens_cpu is not None:
                    mask = num_decode_draft_tokens_cpu >= 0
                request_indices = mask.nonzero(as_tuple=True)[0].to(torch.int32).to(m.query_start_loc.device,
                                                                                      non_blocking=True)
            align = None
            if self.kv_cache_spec.mamba_cache_mode == "align":
                align = (m.block_table_tensor, m.compute_num_computed_tokens(), self.kv_cache_spec.block_size)
            commit = (md.spec_state_indices_tensor, md.spec_query_start_loc, request_indices, align)
            ctx = self  # the commit context is created lazily at commit time (outside graph capture)
        vals = {f.name: getattr(md, f.name) for f in dataclasses.fields(md)}
        return GDNRecoverMetadata(**vals, b_commit=commit, b_context=ctx)

    builder_cls.build = build
    _patch_update_block_table()


def _patch_update_block_table():
    """MRV2 builds GDN metadata for the first mamba group and re-gathers the state indices for
    the others with update_block_table(): keep those 1 + k wide and point their commit at their
    own block table and layers (not the first group's)."""
    import dataclasses as dc

    from vllm.v1.attention.backends import gdn_attn

    builder_cls = gdn_attn.GDNAttentionMetadataBuilder
    orig = builder_cls.update_block_table

    def update_block_table(self, metadata, blk_table, slot_mapping):
        md = orig(self, metadata, blk_table, slot_mapping)
        commit = getattr(md, "b_commit", None)
        if commit is None:
            return md
        q_len = 1 + self.vllm_config.num_speculative_tokens
        si = md.spec_state_indices_tensor
        if si is not None and si.shape[-1] < q_len:
            si = si[:, :1].expand(-1, q_len)
        _, qsl, request_indices, align = commit
        if align is not None:
            align = (blk_table, align[1], align[2])
        return dc.replace(md, spec_state_indices_tensor=si, b_commit=(si, qsl, request_indices, align), b_context=self)

    builder_cls.update_block_table = update_block_table


# ---------------------------------------------------------------------------------------
# Native GDN verify: same math and record layout as KDA's verify (so KDA's commit can fold
# the records), but with GQA keys indexed directly and a per-head scalar gate, which avoids
# repeating q/k to the value heads and broadcasting the gate input on the host side.
def _make_gdn_verify():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _gdn_recoverssm_verify_kernel(
        q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, state_ptr, corr_ptr, kg_ptr, out_ptr,
        qsl_ptr, si_ptr,
        s_q_tok, s_q_head, s_k_tok, s_k_head, s_v_tok, s_v_head, s_a_tok, s_b_tok,
        s_st_blk, s_st_head, s_st_v, s_st_k,
        s_c_row, s_c_head, s_c_pos, s_c_dim, s_kg_row, s_kg_head, s_kg_pos, s_kg_dim,
        s_o_tok, s_o_head, s_si,
        REP: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, QLEN: tl.constexpr,
    ):
        pid_v = tl.program_id(0)
        pid_b = tl.program_id(1)
        hv = tl.program_id(2)
        hk = hv // REP
        bos = tl.load(qsl_ptr + pid_b).to(tl.int64)
        eos = tl.load(qsl_ptr + pid_b + 1).to(tl.int64)
        qlen = eos - bos
        sidx = tl.load(si_ptr + pid_b * s_si).to(tl.int64)
        offs_k = tl.arange(0, K)
        offs_v = pid_v * BV + tl.arange(0, BV)
        if sidx <= 0:
            for t in tl.static_range(QLEN):
                tl.store(out_ptr + (bos + t) * s_o_tok + hv * s_o_head + offs_v, tl.zeros([BV], tl.float32),
                         mask=t < qlen)
            return
        st = tl.load(state_ptr + sidx * s_st_blk + hv * s_st_head + offs_v[:, None] * s_st_v
                     + offs_k[None, :] * s_st_k).to(tl.float32)
        A = tl.exp(tl.load(A_log_ptr + hv).to(tl.float32))
        dtb = tl.load(dt_bias_ptr + hv).to(tl.float32)
        for t in tl.static_range(QLEN):
            valid = t < qlen
            tok = bos + t
            q = tl.load(q_ptr + tok * s_q_tok + hk * s_q_head + offs_k, mask=valid, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + tok * s_k_tok + hk * s_k_head + offs_k, mask=valid, other=0.0).to(tl.float32)
            v = tl.load(v_ptr + tok * s_v_tok + hv * s_v_head + offs_v, mask=valid, other=0.0).to(tl.float32)
            raw_a = tl.load(a_ptr + tok * s_a_tok + hv, mask=valid, other=0.0).to(tl.float32)
            raw_b = tl.load(b_ptr + tok * s_b_tok + hv, mask=valid, other=0.0).to(tl.float32)
            x = raw_a + dtb
            sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
            decay = tl.exp(-A * sp)
            kn = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
            qn = q * tl.rsqrt(tl.sum(q * q) + 1e-6) * (K ** -0.5)
            upd = st * decay
            corr = (v - tl.sum(upd * kn[None, :], axis=1)) * tl.sigmoid(raw_b)
            upd = upd + corr[:, None] * kn[None, :]
            st = tl.where(valid, upd, st)
            o = tl.sum(st * qn[None, :], axis=1)
            tl.store(out_ptr + tok * s_o_tok + hv * s_o_head + offs_v, o, mask=valid)
            tl.store(corr_ptr + pid_b * s_c_row + hv * s_c_head + t * s_c_pos + offs_v * s_c_dim, corr, mask=valid)
            if pid_v == 0:
                kgp = kg_ptr + pid_b * s_kg_row + hv * s_kg_head + t * s_kg_pos
                tl.store(kgp + offs_k * s_kg_dim, k, mask=valid)
                tl.store(kgp + (K + offs_k) * s_kg_dim, tl.zeros([K], tl.float32) + raw_a, mask=valid)

    def gdn_verify(q, k, v, a, b, A_log, dt_bias, state, records, qsl, state_indices, q_len, out):
        # q, k: [1, T, H, K]; v/out: [1, T, HV, V]; a, b: [T, HV]
        T, H, K = q.shape[1], q.shape[2], q.shape[3]
        HV, V = v.shape[2], v.shape[3]
        n = state_indices.shape[0]
        if T == 0 or n == 0:
            return out
        BV = 32
        corr, kg = records.correction, records.key_gate
        grid = (V // BV, n, HV)
        _gdn_recoverssm_verify_kernel[grid](
            q, k, v, a, b, A_log, dt_bias, state, corr, kg, out, qsl, state_indices,
            q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2), a.stride(0), b.stride(0),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            corr.stride(0), corr.stride(1), corr.stride(2), corr.stride(3),
            kg.stride(0), kg.stride(1), kg.stride(2), kg.stride(3),
            out.stride(1), out.stride(2), state_indices.stride(0),
            REP=HV // H, K=K, V=V, BV=BV, QLEN=q_len, num_warps=4, num_stages=2)
        return out

    return gdn_verify
