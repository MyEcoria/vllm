# SPDX-License-Identifier: Apache-2.0

import os
from functools import wraps
from pathlib import Path

import vllm.engine.arg_utils as arg_utils_module
import vllm.transformers_utils.config as config_module
from vllm.config.load import LoadConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.model_loader import (
    _LOAD_FORMAT_TO_MODEL_LOADER,
    get_model_loader,
    register_model_loader,
)
from vllm.transformers_utils.config import get_config_parser, register_config_parser

from .config_parser import GGUFConfigParser
from .gguf_utils import check_gguf_file, is_gguf, is_remote_gguf, split_remote_gguf
from .loader import GGUFModelLoader
from .quantization import DiffusionGGUFConfig, GGUFConfig
from .weights_adapter.diffusion.integration import _patch_diffusers_loader

OOTGGUFConfig = GGUFConfig
OOTGGUFModelLoader = GGUFModelLoader


def _is_gguf_reference(model: str | None) -> bool:
    if not model:
        return False
    return model.endswith(".gguf") or is_remote_gguf(model) or is_gguf(model)


def _get_gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str:
    if hf_config_path is not None:
        return hf_config_path
    if tokenizer is not None and not _is_gguf_reference(tokenizer):
        return tokenizer
    if is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        return repo_id
    if check_gguf_file(model):
        return str(Path(model).parent)
    return model


def _patch_engine_args() -> None:
    if getattr(EngineArgs, "_gguf_create_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        if _is_gguf_reference(self.model):
            gguf_model = self.model
            if self.quantization is None:
                self.quantization = "gguf"
            if self.load_format == "auto":
                self.load_format = "gguf"
            if self.config_format == "auto":
                self.config_format = "gguf"
            if not self.model_weights:
                self.model_weights = gguf_model
            if self.served_model_name is None:
                self.served_model_name = [gguf_model]
            self.model = _get_gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._gguf_create_model_config_patched = True

    original_create_speculative_config = EngineArgs.create_speculative_config

    @wraps(original_create_speculative_config)
    def create_speculative_config(self, *args, **kwargs):
        configured_model = getattr(self, "spec_model", None)
        if self.speculative_config is not None:
            configured_model = configured_model or self.speculative_config.get("model")

        gguf_model = self.model_weights
        target_model_config = args[0] if args else kwargs.get("target_model_config")
        if (
            configured_model is None
            and self.speculative_config is not None
            and self.speculative_config.get("method") == "mtp"
            and _is_gguf_reference(gguf_model)
            and target_model_config is not None
        ):
            # Like the target, the draft reads its HF config (and processor
            # configs) from the config source, not from the GGUF path.
            self.speculative_config["model"] = target_model_config.model
        # A separate non-GGUF draft checkpoint keeps its own config format.
        separate_draft = (
            configured_model is not None
            and not _is_gguf_reference(configured_model)
            and target_model_config is not None
            and target_model_config.config_format == "gguf"
        )
        if separate_draft:
            target_model_config.config_format = "auto"
        try:
            config = original_create_speculative_config(self, *args, **kwargs)
        finally:
            if separate_draft:
                target_model_config.config_format = "gguf"
        if separate_draft and config is not None and config.draft_load_config is None:
            # ...and its own (non-GGUF) weight loader.
            config.draft_load_config = LoadConfig(load_format="auto")
        if (
            config is not None
            and config.method == "mtp"
            and configured_model is None
            and _is_gguf_reference(gguf_model)
        ):
            config.draft_model_config.model_weights = gguf_model
        return config

    EngineArgs.create_speculative_config = create_speculative_config


def _patch_speculator_probe() -> None:
    if getattr(arg_utils_module, "_gguf_speculator_probe_patched", False):
        return

    original_maybe_override = arg_utils_module.maybe_override_with_speculators

    @wraps(original_maybe_override)
    def maybe_override_with_speculators(model, tokenizer, *args, **kwargs):
        if _is_gguf_reference(model):
            return model, tokenizer, kwargs.get("vllm_speculative_config")
        return original_maybe_override(model, tokenizer, *args, **kwargs)

    arg_utils_module.maybe_override_with_speculators = maybe_override_with_speculators
    config_module.maybe_override_with_speculators = maybe_override_with_speculators
    arg_utils_module._gguf_speculator_probe_patched = True
    config_module._gguf_speculator_probe_patched = True


def _patch_dflash_draft_vocab() -> None:
    """Build DFlash draft vocab tables on CPU (opt-in).

    Drafts without their own embeddings/LM head share the target's after
    loading, but vLLM first allocates full bf16 tables (2 x vocab x hidden).
    On small GPUs that transient allocation does not fit next to the target.
    """
    import os

    if os.environ.get("VLLM_GGUF_DRAFT_VOCAB_ON_CPU") != "1":
        return
    import torch
    from vllm.model_executor.layers import vocab_parallel_embedding as vpe
    from vllm.model_executor.models import qwen3_dflash

    if getattr(qwen3_dflash, "_gguf_vocab_on_cpu", False):
        return
    original_init = vpe.VocabParallelEmbedding.__init__
    state = {"active": False}

    @wraps(original_init)
    def vocab_init(self, *args, **kwargs):
        if state["active"]:
            with torch.device("cpu"):
                return original_init(self, *args, **kwargs)
        return original_init(self, *args, **kwargs)

    vpe.VocabParallelEmbedding.__init__ = vocab_init
    for cls in (qwen3_dflash.DFlashQwen3ForCausalLM,):
        cls_init = cls.__init__

        @wraps(cls_init)
        def init(self, *args, __orig=cls_init, **kwargs):
            state["active"] = True
            try:
                __orig(self, *args, **kwargs)
            finally:
                state["active"] = False

        cls.__init__ = init
    qwen3_dflash._gguf_vocab_on_cpu = True


def _register_omni_diffusion_quantization() -> None:
    try:
        from vllm_omni.quantization import register_quantization_override
    except ImportError:
        return

    register_quantization_override("gguf", lambda **kw: DiffusionGGUFConfig(**kw))


def register() -> None:
    """Register the out-of-tree GGUF integration."""
    register_quantization_config("gguf")(GGUFConfig)
    _register_omni_diffusion_quantization()
    _patch_dflash_draft_vocab()

    if "gguf" not in _LOAD_FORMAT_TO_MODEL_LOADER or not isinstance(
        get_model_loader(LoadConfig(load_format="gguf")), GGUFModelLoader
    ):
        register_model_loader("gguf")(GGUFModelLoader)

    try:
        parser = get_config_parser("gguf")
    except ValueError:
        parser = None
    if not isinstance(parser, GGUFConfigParser):
        register_config_parser("gguf")(GGUFConfigParser)
    _patch_engine_args()
    _patch_speculator_probe()
    _patch_diffusers_loader()
    if os.environ.get("BONSAI_INT4_ATTN") == "1":
        from .bonsai_int4_attn import patch as _patch_int4_attn

        _patch_int4_attn()
    if os.environ.get("BONSAI_MTP_W8") == "1":
        from .bonsai_mtp_w8 import patch as _patch_mtp_w8

        _patch_mtp_w8()
    if os.environ.get("BONSAI_GDN_RECOVERSSM") == "1":
        from .bonsai_gdn_recoverssm import patch as _patch_gdn_recoverssm

        _patch_gdn_recoverssm()
