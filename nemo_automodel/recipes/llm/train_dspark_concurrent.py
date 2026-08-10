# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DSpark draft-model training recipe (Qwen3, Gemma4, DeepSeek V4, GLM-5.2, and MiniMax M3 VL targets).

DSpark is a semi-autoregressive parallel drafter: a parallel backbone produces a
block of tokens per anchor in one pass, a serial Markov head injects intra-block
dependency, and a confidence head predicts per-position acceptance. This recipe
mirrors the EAGLE / DFlash scaffolding -- online target hidden-state capture,
gradient accumulation with a trailing-window flush, and the shared checkpointer
plumbing -- and trains the draft with the three-term DSpark objective.

Adapt train_dspark.py for training a DSpark drafter concurrently with a main model.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.checkpoint.utils import find_latest_checkpoint, resolve_restore_from_to_checkpoint_dir
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.dspark_cache import (
    DTYPE_MAP,
    build_cached_dspark_dataloader,
    read_manifest,
    read_target_weight_modules,
)
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.datasets.vlm.dspark_collate import build_dspark_vlm_dataloader
from nemo_automodel.components.distributed.activation_checkpointing import (
    apply_selective_checkpointing_to_layers,
    apply_submodule_checkpointing,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.metric_logger import MetricsSample, build_metric_logger
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run, suppress_wandb_log_messages
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.minimax_m3_vl.processing import build_minimax_m3_vl_processor
from nemo_automodel.components.optim.optimizer import build_optimizer
from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.speculative.dspark.config import (
    build_deepseek_v4_draft_config,
    build_gemma4_draft_config,
    build_glm_5_2_draft_config,
    build_minimax_m3_draft_config,
)
from nemo_automodel.components.speculative.dspark.core import DSparkTrainerModule
from nemo_automodel.components.speculative.dspark.registry import (
    build_target_layer_ids,
    resolve_dspark_draft_spec,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.components.speculative.dspark.target_utils import (
    DEEPSEEK_V4_MODEL_TYPE as _DEEPSEEK_V4_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GEMMA4_MODEL_TYPES as _GEMMA4_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    GLM_5_2_MODEL_TYPE as _GLM_5_2_MODEL_TYPE,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    MINIMAX_M3_MODEL_TYPES as _MINIMAX_M3_MODEL_TYPES,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    apply_target_chat_template as _apply_target_chat_template,
)
from nemo_automodel.components.speculative.dspark.target_utils import (
    read_target_model_type as _read_target_model_type,
)
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config, parse_distributed_section
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _is_checkpoint_model_config_compatible,
)
from nemo_automodel.recipes.llm._dspark_target_build import (
    build_deepseek_v4_target,
    build_glm_5_2_target,
    gather_full_weight_module,
    repair_glm_5_2_qk_rope_head_dim,
    resolve_reduced_target_layers,
)
from nemo_automodel.recipes.llm._spec_train_utils import (
    apply_draft_compile,
    apply_draft_fp8,
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
    raise_if_peft_configured,
)
from nemo_automodel.components._peft.lora import patch_linear_module

logger = logging.getLogger(__name__)

_DSPARK_MM_KEYS = tuple(k for k in VLM_INPUT_KEYS if k != "input_ids")


def _extract_mm_kwargs(batch: dict) -> dict:
    """Return only the multimodal keys present in *batch*, for ``generate_batch(**kwargs)``.

    Empty for a text-only batch (Qwen3, Gemma4, or MiniMax M3 without
    ``multimodal: true``), so the ``generate_batch`` call is unchanged in that case.
    """
    return {k: batch[k] for k in _DSPARK_MM_KEYS if k in batch}


def _packing_kwargs(batch: dict) -> dict:
    """Sequence-packing metadata from a dataloader batch (empty dict when unpacked)."""
    if "seq_lens" not in batch:
        return {}
    return {
        "position_ids": batch["position_ids"],
        "seq_lens": batch["seq_lens"],
        "doc_remaining": batch["doc_remaining"],
    }


def _validate_packing_gates(*, cp_size: int, target_attn_impl: str, micro_batch_size: int) -> None:
    """Reject sequence-packing configs the DSpark path cannot honor (fail fast at setup).

    Context parallelism shards the sequence and strips the block-causal mask packing
    relies on, and a FlashAttention target packs documents from per-document
    ``position_ids`` only at batch size 1.
    """
    if cp_size > 1:
        raise NotImplementedError(
            "Sequence packing (packed_sequence_size>0) is not supported with context parallelism "
            "(distributed.cp_size>1) in DSpark; CP shards the sequence and strips the block-causal mask "
            "packing relies on. Set cp_size=1 or packed_sequence_size=0."
        )
    if "flash" in target_attn_impl and micro_batch_size > 1:
        raise ValueError(
            "Sequence packing with a FlashAttention target requires micro_batch_size=1 "
            f"(got {micro_batch_size}); set micro_batch_size=1 or load the target with "
            "attn_implementation='sdpa'."
        )


def get_wsd_lambda(warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.0):
    total_steps = warmup_steps + stable_steps + decay_steps
    
    def lr_lambda(current_step):
        # 1. Warmup Phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Stable Phase
        elif current_step < (warmup_steps + stable_steps):
            return 1.0
        
        # 3. Decay Phase (Linear Decay example)
        elif current_step < total_steps:
            decay_current = current_step - (warmup_steps + stable_steps)
            # Linearly interpolate between 1.0 and min_lr_ratio
            factor = 1.0 - (float(decay_current) / float(max(1, decay_steps)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * factor
        
        # Post-decay baseline
        return min_lr_ratio

    return lr_lambda


def get_warmup_stable_lambda(warmup_steps):

    def lr_lambda(current_step):
        # 1. Warmup Phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # 2. Stable Phase
        return 1.0

    return lr_lambda
    
    
class _DraftArgs(dict):
    """Dict with attribute access for the per-architecture draft-config builders."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(key) from exc


def _resolve_wandb_kwargs(wandb_cfg: dict) -> dict | None:
    """Convert a ``wandb:`` config block into ``wandb.init`` kwargs, or ``None``.

    ``enable`` is the examples' documentation-only opt-in flag (W&B logging is
    opt-in: example configs ship the block with ``enable: false`` so users start
    logging by flipping it to ``true`` instead of commenting the block in/out);
    it is not a real ``wandb.init`` kwarg, so strip it before forwarding the rest
    -- passing it through raises ``TypeError: init() got an unexpected keyword
    argument 'enable'``. Returns ``None`` when ``enable`` is explicitly ``False``.
    """
    kwargs = dict(wandb_cfg)
    if kwargs.pop("enable", True) is False:
        return None
    return kwargs


def _init_dspark_wandb(*, is_main: bool, wandb_cfg, cfg_dict: dict, default_name: str):
    """Initialize the rank-zero W&B run for a DSpark training job, or return ``None``.

    Centralizes the ``is_main`` / block-presence / ``enable`` gating that
    ``TrainDSparkRecipe.setup`` previously inlined, so it is unit-testable
    without a distributed environment.
    """
    if not is_main or wandb_cfg is None:
        return None
    wandb_kwargs = _resolve_wandb_kwargs(wandb_cfg.to_dict())
    if wandb_kwargs is None:
        return None
    suppress_wandb_log_messages()
    return init_wandb_run(wandb_kwargs, cfg_dict, default_name=default_name)


def _resolve_dspark_optimizer_spec(opt_cfg) -> tuple[str, dict]:
    """Normalize the recipe's ``optimizer:`` config into a ``build_optimizer`` spec.

    Reads an optional ``_target_`` (a registry short name such as ``"fused_adam"``
    or a dotted import path, e.g. ``transformer_engine.pytorch.optimizers.FusedAdam``)
    plus whatever other fields the config carries -- ``lr``/``betas``/``weight_decay``
    and any optimizer-specific kwargs (``master_weights``, ``master_weight_dtype``,
    ``exp_avg_dtype``, ``exp_avg_sq_dtype``, ``store_param_remainders``, ...) -- and
    returns the ``(target, kwargs)`` tuple that ``build_optimizer`` resolves via its
    registry / dotted-import-path / ``OptimizerFromFactoryConfig`` escape hatch.

    Absent an explicit ``_target_``, this defaults to plain ``torch.optim.AdamW``
    with its prior ``betas``/``weight_decay`` defaults (matching the previous
    hardcoded behavior, so existing DSpark configs are unaffected). Those two
    AdamW-shaped defaults are only injected in that no-``_target_`` case: forcing
    them onto an arbitrary explicit ``_target_`` would break optimizers that do
    not accept a ``betas`` kwarg (e.g. plain SGD).
    """
    kwargs = dict(opt_cfg.to_dict())
    # ConfigNode resolves ``_target_`` to the callable in ``to_dict``; recover the original
    # import-path string via ``get_as_string``. Only call it when the key is actually
    # present: ``ConfigNode.get_as_string`` raises ``KeyError`` for an absent key even
    # with an explicit ``None`` default (a ``None`` default is never returned), which
    # crashed every DSpark config whose ``optimizer:`` block omits ``_target_``.
    target = kwargs.pop("_target_", None)
    if target is not None and hasattr(opt_cfg, "get_as_string"):
        target = opt_cfg.get_as_string("_target_")
    kwargs.pop("warmup_ratio", None)
    kwargs.pop("min_lr_ratio", None)
    kwargs["lr"] = float(kwargs["lr"])
    if target is None:
        target = "torch.optim.AdamW"
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("weight_decay", 0.0)
    return target, kwargs


def _build_dspark_optimizer(trainer_module, opt_cfg, device_mesh=None) -> torch.optim.Optimizer:
    """Build the DSpark trainer's optimizer from its ``optimizer:`` config.

    Thin wrapper around ``build_optimizer`` so ``TrainDSparkRecipe.setup`` has a
    single, unit-testable call site (``build_optimizer`` itself needs no
    distributed environment for a non-pipelined single-part model like the
    DSpark draft, so this is testable with a plain CPU module).
    """
    return build_optimizer(trainer_module, _resolve_dspark_optimizer_spec(opt_cfg), device_mesh=device_mesh)[0]


def _resolve_warmup_steps(warmup_ratio: float, total_optim_steps: int, min_warmup_steps: int = 20) -> int:
    """Return the LR warmup length in optimizer steps.

    ``warmup_ratio * total_optim_steps`` collapses to a handful of steps (or fewer)
    on short / small-dataset runs, dropping a freshly-initialized draft (random
    attention layers, Markov head, confidence head) to near-peak LR within the
    first few optimizer steps -- a reliable way to trigger an early loss spike.
    Floor the ratio-derived step count at ``min_warmup_steps`` unless the caller
    explicitly opts out of warmup with ``warmup_ratio<=0`` (e.g. the smoke config).
    """
    if warmup_ratio <= 0:
        return 1
    return max(min_warmup_steps, int(warmup_ratio * total_optim_steps))


def _apply_draft_activation_checkpointing(draft_model: torch.nn.Module, mode: bool | str) -> None:
    """Apply the recipe's AC mode to the trainable DSpark draft before FSDP."""
    if not mode or (isinstance(mode, str) and mode.lower() == "false"):
        return
    layers = list(getattr(draft_model, "layers", ()))
    if not layers:
        logger.warning("Draft activation checkpointing requested, but the draft exposes no layers.")
        return
    if is_selective_activation_checkpointing(mode):
        apply_selective_checkpointing_to_layers(draft_model, layers, has_kv_sharing=False)
        logger.info("Enabled selective activation checkpointing on %d draft layers", len(layers))
    else:
        # DSpark's native layers are not HF GradientCheckpointingLayer subclasses.
        # Checkpoint their attention/MLP/norm submodules before FSDP indexes params.
        apply_submodule_checkpointing(layers, has_kv_sharing=False)
        logger.info("Enabled full activation checkpointing on %d draft layers", len(layers))


def _validate_cached_dspark_manifest(
    cache_dir: str,
    manifest: dict,
    target_config,
    target_layer_ids: list[int],
    *,
    target_model: str,
    target_model_type: str,
    seq_length: int,
    compute_dtype: torch.dtype,
) -> None:
    """Validate that a DSpark offline cache matches the configured target/draft run."""
    if str(manifest["target_model"]) != str(target_model):
        logger.warning(
            "DSpark cache at %s was built for target_model=%r, but this run configured target_model=%r. "
            "Continuing because raw paths can differ across machines; structural cache fields will still be "
            "validated.",
            cache_dir,
            manifest["target_model"],
            target_model,
        )
    if str(manifest["target_model_type"]) != str(target_model_type):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_model_type={manifest['target_model_type']!r}, "
            f"but the configured target has model_type={target_model_type!r}."
        )
    if int(manifest["target_vocab_size"]) != int(target_config.vocab_size):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_vocab_size={manifest['target_vocab_size']}, "
            f"but the configured target has {target_config.vocab_size}. The cache does not match this target."
        )
    hidden_size = int(target_config.hidden_size)
    if int(manifest["hidden_size"]) != hidden_size:
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for hidden_size={manifest['hidden_size']}, "
            f"but the configured target has hidden_size={hidden_size}."
        )
    if int(manifest["num_hidden_layers"]) != int(target_config.num_hidden_layers):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for num_hidden_layers={manifest['num_hidden_layers']}, "
            f"but the configured target has num_hidden_layers={target_config.num_hidden_layers}."
        )
    if int(manifest["seq_length"]) != int(seq_length):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for seq_length={manifest['seq_length']}, "
            f"but this run configured seq_length={seq_length}."
        )
    cache_dtype = DTYPE_MAP.get(str(manifest["dtype"]))
    if cache_dtype is None:
        raise ValueError(f"DSpark cache at {cache_dir} has unsupported dtype={manifest['dtype']!r}.")
    if compute_dtype == torch.float32 and cache_dtype != torch.float32:
        raise ValueError(
            f"DSpark cache at {cache_dir} stores dtype={manifest['dtype']}, but CPU cached training "
            "requires fp32 cache tensors. Regenerate with --dtype fp32 or train on CUDA."
        )
    expected_hidden_dim = hidden_size * len(target_layer_ids)
    if int(manifest["target_hidden_dim"]) != expected_hidden_dim:
        raise ValueError(
            f"DSpark cache at {cache_dir} has target_hidden_dim={manifest['target_hidden_dim']}, "
            f"but the configured target/layers need {expected_hidden_dim} "
            f"(hidden_size {hidden_size} x {len(target_layer_ids)} target layers)."
        )
    if int(manifest["target_last_hidden_dim"]) != hidden_size:
        raise ValueError(
            f"DSpark cache at {cache_dir} has target_last_hidden_dim={manifest['target_last_hidden_dim']}, "
            f"but the configured target has hidden_size={hidden_size}."
        )
    recorded_layer_ids = [int(x) for x in manifest["target_layer_ids"]]
    if recorded_layer_ids != list(target_layer_ids):
        raise ValueError(
            f"DSpark cache at {cache_dir} was built for target_layer_ids={recorded_layer_ids}, "
            f"but this run requested target_layer_ids={target_layer_ids}."
        )


def _distributed_section_dict(cfg) -> dict:
    """Return the ``distributed:`` section of a top-level recipe config as a plain dict.

    A missing block yields ``{}``, i.e. the default FSDP2 configuration; this keeps the
    fail-loud missing-block contract of ``create_distributed_setup_from_config`` intact
    for callers that require an explicit block.
    """
    section = cfg.get("distributed", None)
    if section is None:
        return {}
    return section.to_dict() if hasattr(section, "to_dict") else dict(section)


def _add_accept_rate_per_position(
    metrics: dict[str, float],
    accept_num: torch.Tensor,
    accept_den: torch.Tensor,
) -> None:
    """Add measured per-position acceptance rates to a metrics dictionary."""
    for position, (num, den) in enumerate(zip(accept_num.tolist(), accept_den.tolist())):
        if den > 0:
            metrics[f"accept_rate@{position}"] = num / den


class TrainDSparkConcurrentRecipe(BaseRecipe):
    """Recipe for DSpark draft-model training on Qwen3, Gemma4, DeepSeek V4, GLM-5.2, and MiniMax M3 VL targets."""

    def __init__(self, cfg, dist_env=None, device_mesh=None, target_peft_config=None, target_config=None):
        """
        This recipe is expected to be called from the 
        TrainFinetuneRecipeForNextTokenPredictionDSpark recipe's setup method.
        The caller should pass in its device_mesh. The caller will have a separate 
        moe_mesh, which is not needed for this recipe.
        """
        self.cfg = cfg
        self.dist_env = dist_env
        self.device_mesh = device_mesh
        self.target_peft_config = target_peft_config
        self.target_config = target_config


    def setup(self):
        """Build the target model, DSpark draft, data, optimizer, and trainer module."""

        recipe_cfg = self.cfg.recipe_args
        self.device = self.dist_env.device or torch.device("cpu")
        raise_if_peft_configured(self.cfg, type(self).__name__)
        self.dp_mesh = get_flat_mesh(self.device_mesh, "dp")

        # Sequence packing is supported on the online LLM (text-only) path only; the
        # VLM and offline-cache paths do not carry the block-causal packing metadata.
        self.packed_sequence_size = int(recipe_cfg.get("packed_sequence_size", 0) or 0)
        self.cached_target_path = None
        if self.packed_sequence_size > 0 and (is_multimodal or self.cached_target_path is not None):
            raise NotImplementedError(
                "Sequence packing (packed_sequence_size>0) is only supported on the online text-only "
                "DSpark path; the VLM and cached-target paths do not carry the packing metadata."
            )


        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Resolve the captured target layers once and share them between the
        # target wrapper (what to capture) and the draft config (the ``fc`` input
        # width) so the two never disagree.
        # Gemma4 and MiniMax M3 VL nest their text fields (layer count, vocab)
        # under text_config.
        # 
        num_target_layers = int(recipe_cfg.get("num_target_layers", None) or self.target_config.num_hidden_layers)
        draft_num_hidden_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
        target_layer_ids = list(
            recipe_cfg.get("target_layer_ids", None)
            or build_target_layer_ids(num_target_layers, draft_num_hidden_layers)
        )
        target_layer_ids = validate_target_layer_ids(target_layer_ids, num_target_layers)
        # HFDSparkTargetModel validates target_layer_ids against the actual (possibly
        # reduced) layer count via common.validate_target_layer_ids, which also accepts
        # -1 (the embedding output) and enforces strictly-increasing ids.
        self.target_layer_ids = target_layer_ids

        self.block_size = int(recipe_cfg.get("block_size", 7))
        self.num_anchors = int(recipe_cfg.get("num_anchors", 512))
        target_text_config = self.target_config.text_config if hasattr(self.target_config, "text_config") else self.target_config
        self.mask_token_id = self._resolve_mask_token_id(recipe_cfg, target_text_config.vocab_size)

        embed_src = None
        head_src = None
        
        # The Qwen3 / Gemma4 drafts consume a flex_attention BlockMask during training.
        # The DeepSeek V4 and GLM-5.2 drafts instead consume a dense additive mask
        # (the DFlash SDPA path), so they are exempt from the flex_attention requirement.
        attention_backend = recipe_cfg.get("attention_backend", "flex_attention")
        if attention_backend != "flex_attention":
            raise ValueError(f"DSpark training requires attention_backend='flex_attention', got {attention_backend!r}.")
        confidence_head_alpha = float(recipe_cfg.get("confidence_head_alpha", 1.0))
        markov_rank = int(recipe_cfg.get("markov_rank", 256))

        # Qwen3-style draft: a small non-causal stack reusing the target's
        # architecture defaults plus the DSpark-specific fields.
        # Qwen3 is the only draft architecture currently supported by this recipe.
        draft_config = self.target_config.to_dict()
        draft_config["architectures"] = ["Qwen3DSparkModel"]
        draft_config["num_hidden_layers"] = draft_num_hidden_layers
        draft_config["layer_types"] = ["full_attention"] * draft_num_hidden_layers
        draft_config["max_window_layers"] = draft_num_hidden_layers
        draft_config["num_target_layers"] = num_target_layers
        draft_config["target_layer_ids"] = target_layer_ids
        draft_config["block_size"] = self.block_size
        draft_config["num_anchors"] = self.num_anchors
        draft_config["mask_token_id"] = self.mask_token_id
        draft_config["markov_rank"] = markov_rank
        if markov_rank > 0:
            draft_config["markov_head_type"] = str(recipe_cfg.get("markov_head_type", "vanilla"))
        draft_config["enable_confidence_head"] = confidence_head_alpha > 0.0
        if confidence_head_alpha > 0.0:
            draft_config["confidence_head_with_markov"] = bool(recipe_cfg.get("confidence_head_with_markov", True))
        # The draft owns an independent (frozen) lm_head seeded from the target.
        draft_config["tie_word_embeddings"] = False
        draft_config_obj = Qwen3Config.from_dict(draft_config)
        draft_config_obj._attn_implementation = attention_backend

        # TODO: hardcode arhcitectures for now
        architectures = ["Qwen3ForCausalLM"]
        draft_cls = resolve_dspark_draft_spec(architectures).draft_cls
        self.draft_model = draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)
        if self.target_peft_config is not None:
            patch_linear_module(self.draft_model.lm_head, 
                                dim = self.target_peft_config.dim, 
                                alpha = self.target_peft_config.alpha, 
                                use_triton = self.target_peft_config.use_triton,
                                dropout = self.target_peft_config.dropout,
                                dropout_position = self.target_peft_config.dropout_position,
                                use_dora = self.target_peft_config.use_dora)
        if self.packed_sequence_size > 0 and type(self.draft_model).__name__ != "Qwen3DSparkModel":
            # Only the Qwen3 draft forward threads the packing metadata so far; the
            # other DSpark drafts would silently let anchors cross document boundaries.
            raise NotImplementedError(
                f"Sequence packing (packed_sequence_size>0) is only supported by the Qwen3 DSpark draft, "
                f"not {type(self.draft_model).__name__}."
            )


        # training only the backbone, fc, Markov head, and confidence head.
        # Under this code path, the target's embeddings and lm_head wights 
        # will be copied to the draft later.
        # The draft's embedding and lm_head are still frozen here.
        self.draft_model.set_embedding_head_trainable(False)
        # Optional FP8 draft compute, in place (see apply_draft_fp8); must precede AC and the FSDP2/DDP wrap.
        apply_draft_fp8(self.draft_model, self.cfg.get("fp8", None))
        # Optional torch.compile of the draft, in place; after the fp8 swap.
        apply_draft_compile(self.draft_model, self.cfg.get("compile", None))

        dist_cfg = self.cfg.get("distributed", None)
        activation_checkpointing = dist_cfg.get("activation_checkpointing", False) if dist_cfg is not None else False
        # The target consumes this setting through its distributed setup, while
        # the separately constructed trainable draft must be wrapped explicitly.
        _apply_draft_activation_checkpointing(self.draft_model, activation_checkpointing)

        trainer_module = DSparkTrainerModule(
            self.draft_model,
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", None),
            ce_loss_alpha=float(recipe_cfg.get("ce_loss_alpha", 0.1)),
            l1_loss_alpha=float(recipe_cfg.get("l1_loss_alpha", 0.9)),
            confidence_head_alpha=confidence_head_alpha,
        ).to(self.device)
        # Multi-GPU strategy: FSDP2 (default) shards the draft per block, or DDP.
        self.parallel_strategy = "ddp"
        if self.dist_env.world_size > 1:
            # Case-fold to match parse_distributed_section's strategy normalization.
            strategy = str(dist_cfg.get("strategy", "fsdp2")).lower() if dist_cfg is not None else "fsdp2"
            self.parallel_strategy = strategy
            if strategy == "fsdp2":
                from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

                mp_policy = MixedPrecisionPolicy(param_dtype=self.compute_dtype, reduce_dtype=torch.float32)
                # Shard over "dp" (not the world) under CP so the draft stays replicated
                # across cp ranks; without a mesh (cp_size=1) this is the world default.
                shard_kwargs = {"mp_policy": mp_policy}
                if self.dp_mesh is not None:
                    shard_kwargs["mesh"] = self.dp_mesh
                for layer in trainer_module.draft_model.layers:
                    fully_shard(layer, **shard_kwargs)
                fully_shard(trainer_module, **shard_kwargs)
            elif strategy == "ddp":
                trainer_module = DistributedDataParallel(
                    trainer_module,
                    device_ids=[self.device.index] if self.device.type == "cuda" else None,
                    output_device=self.device.index if self.device.type == "cuda" else None,
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                    process_group=self.dp_mesh.get_group() if self.dp_mesh is not None else None,
                )
            else:
                raise ValueError(f"Unsupported distributed.strategy={strategy!r}; use 'fsdp2' or 'ddp'.")
        self.trainer_module = trainer_module
        # FP8 + FSDP2 float8 all-gather: amortize the per-parameter dynamic-scale
        # computation into one call after each optimizer step (mirrors train_ft).
        # apply_fp8_to_model already resolved whether per-step scale precompute
        # applies (enabled + tensorwise + fp8 all-gather) onto the draft module;
        # reuse that instead of re-deriving from raw YAML.
        self._precompute_fp8_scales = self.parallel_strategy == "fsdp2" and bool(
            getattr(self.draft_model, "precompute_float8_dynamic_scale_for_fsdp", False)
        )

        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = _build_dspark_optimizer(self.trainer_module, opt_cfg, device_mesh=self.dp_mesh)
        logger.info(
            "Optimizer=%s lr=%.3e master_weights=%s master_weight_dtype=%s "
            "store_param_remainders=%s exp_avg_dtype=%s exp_avg_sq_dtype=%s",
            type(self.optimizer).__name__,
            self.peak_lr,
            getattr(self.optimizer, "master_weights", False),
            getattr(self.optimizer, "master_weight_dtype", None),
            getattr(self.optimizer, "store_param_remainders", False),
            getattr(self.optimizer, "exp_avg_dtype", None),
            getattr(self.optimizer, "exp_avg_sq_dtype", None),
        )
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dist_cfg = self.cfg.get("distributed", None)
        self.defer_fsdp_grad_sync = bool(dist_cfg.get("defer_fsdp_grad_sync", True)) if dist_cfg is not None else True
        self.metric_logger = build_metric_logger(str(self.output_dir / "dspark_train_metrics.jsonl"))

        # TODO: For concurrent training, the DSpark module doesn't need its own dataloader.
        # However, the user should have the option to specify a separate lr_scheduler and 
        # optimizer for the DSpark module.
        total_optim_steps = opt_cfg.get("total_steps", 1)
        lr_scheduler_cfg = self.cfg.get("lr_scheduler", None)
        warmup_steps = int(lr_scheduler_cfg.get("warmup_steps", 100))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, get_warmup_stable_lambda(warmup_steps)
        )
        self.total_optim_steps = total_optim_steps
        self.runtime = SimpleNamespace(global_step=0, epoch=0)
        self._resume_epoch = 0

        # Seed by the dp coordinate, not the global rank: under CP the draft is
        # replicated across cp ranks and must sample the SAME anchor positions each
        # step, else the replicas diverge. _get_dp_rank() returns the global rank
        # when there is no mesh, so the plain world-sharded path is unchanged.
        self.rng = StatefulRNG(seed=int(recipe_cfg.get("shuffle_seed", 42)) + self._get_dp_rank(), ranked=False)
        target_path = recipe_cfg.get("target_model_name_or_path", None)
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

        self.wandb_run = _init_dspark_wandb(
            is_main=self.dist_env.is_main,
            wandb_cfg=self.cfg.get("wandb", None),
            cfg_dict=self.cfg.to_dict(),
            default_name="dspark_" + str(target_path).rstrip("/").split("/")[-1],
        )

    @staticmethod
    def _resolve_mask_token_id(recipe_cfg, vocab_size: int) -> int:
        """Resolve and validate the MASK token id filling non-anchor block positions.

        The draft's ``embed_tokens`` row at this id is the learned "predict here"
        signal. It must be a deliberately chosen reserved / unused token id (never a
        silent fallback to ``pad``, which is commonly aliased to ``eos``), and the
        inference runtime must fill block slots with the same id.
        """
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "DSpark requires recipe_args.mask_token_id to be set explicitly (the token used for "
                "non-anchor block positions). Pick a reserved / rarely-used token id so the mask-slot "
                "embedding does not collide with real content, and use the same id in the inference runtime."
            )
        mask_token_id = int(mask_token_id)
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of range for the vocab [0, {vocab_size}); "
                "it indexes the draft embed_tokens table."
            )
        return mask_token_id

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the EAGLE / DFlash recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
        draft_state_dict_keys = list(self.draft_model.state_dict().keys())
        ckpt_kwargs = dict(
            enabled=True,
            checkpoint_dir=default_dir,
            model_save_format="safetensors",
            model_repo_id=str(target_path),
            model_cache_dir=hf_constants.HF_HUB_CACHE,
            save_consolidated=True,
            is_peft=False,
            model_state_dict_keys=draft_state_dict_keys,
        )
        if ckpt_cfg is not None:
            user_cfg = ckpt_cfg.to_dict() if hasattr(ckpt_cfg, "to_dict") else dict(ckpt_cfg)
            user_cfg.pop("restore_from", None)
            ckpt_kwargs.update(user_cfg)
        if ckpt_kwargs.get("model_state_dict_keys") is None:
            ckpt_kwargs["model_state_dict_keys"] = draft_state_dict_keys

        self.checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
        # Under CP the draft is replicated across cp ranks, so key the shard on the dp
        # coordinate (identical for cp peers) rather than the global rank. Without a
        # mesh (cp_size=1) this returns the global rank, unchanged.
        dp_rank = self._get_dp_rank()
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config, dp_rank=dp_rank, tp_rank=0, pp_rank=0, moe_mesh=None
        )
        self._log_checkpoint_retention_policy(self.checkpoint_config)

    def _module(self):
        return (
            self.trainer_module.module
            if isinstance(self.trainer_module, DistributedDataParallel)
            else self.trainer_module
        )

    def _maybe_precompute_fp8_scales(self) -> None:
        """Precompute float8 dynamic scales after an optimizer step (FSDP2 fp8 all-gather only)."""
        if not getattr(self, "_precompute_fp8_scales", False):
            return
        precompute_float8_dynamic_scale_for_fsdp(self._module())

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float | None = None,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
        is_final_checkpoint: bool = False,
    ) -> None:
        """Persist the DSpark draft model, optimizer, scheduler, RNG, and meta."""
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        self.checkpointer.async_wait()

        ckpt_root = self.checkpoint_config.checkpoint_dir
        path = os.path.join(str(ckpt_root), f"epoch_{epoch}_step_{step}")
        is_dist_initialized = dist.is_initialized()
        is_rank_0 = (not is_dist_initialized) or dist.get_rank() == 0
        best_metric_name = next(iter(val_loss.keys())) if val_loss and len(val_loss) == 1 else best_metric_key
        best_val_metric = val_loss.get(best_metric_name) if val_loss else None

        self._complete_pending_checkpoint()

        if is_rank_0:
            if os.path.exists(path):
                raise FileExistsError(f"Checkpoint directory {path} already exists")
            os.makedirs(path, exist_ok=True)
            loss_dict: dict[str, float] = {}
            if train_loss is not None:
                loss_dict["train_loss"] = float(train_loss)
            if val_loss:
                for k, v in val_loss.items():
                    loss_dict[k] = float(v)
            if loss_dict:
                with open(os.path.join(path, "losses.json"), "w") as f:
                    json.dump(loss_dict, f)
        if is_dist_initialized:
            dist.barrier()

        draft_model = self._module().draft_model
        self.checkpointer.save_model(
            draft_model,
            path,
            tokenizer=self.tokenizer,
            is_final_checkpoint=is_final_checkpoint,
        )
        self.checkpointer.save_optimizer(self.optimizer, draft_model, path, self.lr_scheduler)
        # The checkpointer keys the rng file on dp_rank, but cp peers share a dp_rank
        # (and, being seeded per dp_rank, hold identical rng state), so every peer would
        # torch.save the same rng_dp_rank_N.pt and race on a shared FS; let only the
        # first cp peer write it.
        cp_mesh = getattr(self, "cp_mesh", None)
        if cp_mesh is None or cp_mesh.get_local_rank() == 0:
            self.checkpointer.save_on_dp_ranks(self.rng, "rng", path)

        if is_rank_0:
            self._save_extra_state(path, epoch=epoch)
            try:
                save_config(self.cfg.raw_config, path)
            except (AttributeError, OSError) as e:
                logger.warning("Failed to save config snapshot: %s", e)
        if is_dist_initialized:
            dist.barrier()

        if getattr(self.checkpointer.config, "is_async", False):
            setattr(self, "_last_pending_checkpoint_dir", path)
            setattr(
                self,
                "_last_pending_best_checkpoint_info",
                {
                    "path": path,
                    "val": float(best_val_metric) if best_val_metric is not None else None,
                    "metric_key": best_metric_name,
                },
            )
        else:
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric), best_metric_name)
                self._prune_old_checkpoints()
            if is_dist_initialized:
                dist.barrier()

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist DSpark meta: global_step, epoch, block_size, mask, and target layers."""
        torch.save(
            {
                "global_step": self.runtime.global_step,
                "epoch": int(epoch),
                "block_size": self.block_size,
                "num_anchors": self.num_anchors,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_layer_ids),
            },
            os.path.join(path, "dspark_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Restore the DSpark draft model, optimizer, scheduler, RNG, and global_step."""
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        is_rank_0 = (not dist.is_initialized()) or dist.get_rank() == 0
        ckpt_root = self.checkpoint_config.checkpoint_dir

        if restore_from:
            ckpt_dir = resolve_restore_from_to_checkpoint_dir(ckpt_root, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logger.warning("restore_from='LATEST' but no checkpoint found in %s", ckpt_root)
                return
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        else:
            auto = find_latest_checkpoint(ckpt_root)
            if auto is None:
                return
            ckpt_dir = str(auto)

        ok, reason = _is_checkpoint_model_config_compatible(self.cfg, ckpt_dir)
        if not ok and not restore_from:
            if is_rank_0:
                logger.warning(
                    "Auto-detected checkpoint at %s is incompatible: %s. Skipping restore.", ckpt_dir, reason
                )
            return

        if is_rank_0:
            logger.info("Resuming from checkpoint: %s", ckpt_dir)

        draft_model = self._module().draft_model
        self.checkpointer.load_model(draft_model, os.path.join(ckpt_dir, "model"))
        self.checkpointer.load_optimizer(self.optimizer, draft_model, ckpt_dir, self.lr_scheduler)
        try:
            self.checkpointer.load_on_dp_ranks(self.rng, "rng", ckpt_dir)
        except FileNotFoundError:
            logger.warning("RNG state not found in %s; continuing without restoring RNG.", ckpt_dir)
        self._load_extra_state(ckpt_dir)

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore DSpark meta: global_step and epoch."""
        meta_path = os.path.join(ckpt_dir, "dspark_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _forward_batch(self, batch):
        """Run one batch through live target capture or the offline cache."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        return self.trainer_module(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
            position_ids=batch["position_ids"],
            seq_lens=batch["seq_lens"],
            doc_remaining=batch["doc_remaining"],
        )

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured."""
        every = getattr(self, "ckpt_every_steps", None)
        if every is None or every <= 0 or self.runtime.global_step % every != 0:
            return False
        total_optim_steps = getattr(self, "total_optim_steps", None)
        is_final_checkpoint = total_optim_steps is not None and self.runtime.global_step >= total_optim_steps
        self.save_checkpoint(
            epoch=epoch,
            step=self.runtime.global_step,
            best_metric_key="val_loss",
            is_final_checkpoint=is_final_checkpoint,
        )
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Always save the fully-trained model at the end, unless a cadence already saved the final step."""
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        every = getattr(self, "ckpt_every_steps", None)
        saved_by_step = bool(every and every > 0 and gs % every == 0)
        saved_by_epoch = bool(getattr(self, "save_checkpoint_every_epoch", False))
        if saved_by_step or saved_by_epoch:
            return False
        self.save_checkpoint(epoch=completed_epochs, step=gs, best_metric_key="val_loss", is_final_checkpoint=True)
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                metrics = self._forward_batch(batch)
                total_loss += metrics.loss.detach()
                total_batches += 1
        total_loss = self._dp_allreduce(total_loss)
        total_batches = self._dp_allreduce(total_batches)
        self.trainer_module.train()
        return {"val_loss": (total_loss / total_batches.clamp_min(1)).item()}

    def _wandb_log(self, data: dict, step: int) -> None:
        """Log rank-zero metrics when a W&B run is active."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            run.log(data, step=step)

    def _finish_wandb(self) -> None:
        run = getattr(self, "wandb_run", None)
        if run is None:
            return
        try:
            run.finish()
        except Exception:
            logger.warning("Failed to finish W&B run cleanly.", exc_info=True)
        finally:
            self.wandb_run = None

    def run_train_micro_batch(self, micro_batch):
        """
        Run the DSpark forward and backward pass for the single micro-batch in the input, 
        followed by the optimizer step.
        Run the optimizer step after processing all micro-batches.

        """
        self.trainer_module.train()
        running_loss = 0.0
        running_ce = 0.0
        running_l1 = 0.0
        running_conf = 0.0
        # Acceptance diagnostics accumulate as (num, den) sums, not per-step
        # ratios: reducing the sums and dividing once gives the exact global
        # ratio regardless of per-micro-batch token imbalance, and keeps tau at
        # its >= 1 floor even when a micro-batch contributes no valid blocks.
        running_tau_num = 0.0
        running_tau_den = 0.0
        running_conf_abs_err_num = 0.0
        running_conf_bias_num = 0.0
        running_conf_cumprod_bias_num = 0.0
        running_conf_diag_den = 0.0
        running_accept_pos_num = torch.zeros(self.block_size, device=self.device)
        running_accept_pos_den = torch.zeros(self.block_size, device=self.device)
        running_micro = 0
        is_optim_step = True
        # get_sync_ctx handles both DDP (no_sync) and FSDP2 (set_requires_gradient_sync).
        with get_sync_ctx(self.trainer_module, is_optim_step, self.defer_fsdp_grad_sync):
            metrics = self._forward_batch(micro_batch)
            loss = metrics.loss
            loss.backward()

        running_loss += metrics.loss.detach().item()
        running_ce += metrics.ce_loss.detach().item()
        running_l1 += metrics.l1_loss.detach().item()
        running_conf += metrics.confidence_loss.detach().item()
        running_tau_num += metrics.tau_num.detach().item()
        running_tau_den += metrics.tau_den.detach().item()
        running_conf_abs_err_num += metrics.confidence_abs_error_num.detach().item()
        running_conf_bias_num += metrics.confidence_bias_num.detach().item()
        running_conf_cumprod_bias_num += metrics.confidence_cumprod_bias_num.detach().item()
        running_conf_diag_den += metrics.confidence_diag_den.detach().item()
        running_accept_pos_num += metrics.accept_rate_per_pos_num.detach()
        running_accept_pos_den += metrics.accept_rate_per_pos_den.detach()
        running_micro += 1

        if is_optim_step:
            torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            self._maybe_precompute_fp8_scales()
            self.runtime.global_step += 1
            self._maybe_save_step_checkpoint(self.runtime.epoch)

            if self.runtime.global_step % self.log_every_steps == 0:
                # One collective: the loss window sums and micro-batch count,
                # the acceptance-diagnostic (num, den) sums, and the per-position
                # accept sums, concatenated so a single all-reduce covers them.
                # Losses divide by the micro-batch count (window mean of already
                # normalized values); the diagnostics divide num by den for the
                # exact global ratio.
                scalars = torch.tensor(
                    [
                        running_loss,
                        running_ce,
                        running_l1,
                        running_conf,
                        running_tau_num,
                        running_tau_den,
                        running_conf_abs_err_num,
                        running_conf_bias_num,
                        running_conf_cumprod_bias_num,
                        running_conf_diag_den,
                        float(running_micro),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                reduced = self._dp_allreduce(
                    torch.cat([scalars, running_accept_pos_num, running_accept_pos_den])
                )
                n_scalars = scalars.numel()
                w = reduced[:n_scalars].tolist()
                pos_num = reduced[n_scalars : n_scalars + self.block_size]
                pos_den = reduced[n_scalars + self.block_size :]
                count = max(1.0, w[10])
                avg = {
                    "loss": w[0] / count,
                    "ce_loss": w[1] / count,
                    "l1_loss": w[2] / count,
                    "confidence_loss": w[3] / count,
                }
                # Log a diagnostic only when it was measured this window (its
                # denominator is positive), so an ablation without the TV signal
                # or the confidence head shows no curve rather than a flat zero
                # that reads like collapsed acceptance.
                accept_den = pos_den.sum().item()
                if accept_den > 0:
                    avg["accept_rate"] = pos_num.sum().item() / accept_den
                    _add_accept_rate_per_position(avg, pos_num, pos_den)
                if w[5] > 0:
                    avg["tau"] = w[4] / w[5]
                if w[9] > 0:
                    avg["confidence_abs_error"] = w[6] / w[9]
                    avg["confidence_bias"] = w[7] / w[9]
                    avg["confidence_cumprod_bias"] = w[8] / w[9]
                running_loss = running_ce = running_l1 = running_conf = 0.0
                running_tau_num = running_tau_den = 0.0
                running_conf_abs_err_num = running_conf_bias_num = 0.0
                running_conf_cumprod_bias_num = running_conf_diag_den = 0.0
                running_accept_pos_num = torch.zeros(self.block_size, device=self.device)
                running_accept_pos_den = torch.zeros(self.block_size, device=self.device)
                running_micro = 0
                if self.dist_env.is_main:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                    self.metric_logger.log(
                        MetricsSample(
                            step=self.runtime.global_step,
                            epoch=0,
                            metrics={**avg, "lr": current_lr, "mem": mem},
                        )
                    )
                    # ``avg`` renames l1_loss -> tv_loss and carries only the
                    # diagnostics measured this window, so mirror its keys under
                    # the train/ prefix rather than hard-coding each one.
                    wandb_metrics = {
                        "train/tv_loss" if key == "l1_loss" else f"train/{key}": value
                        for key, value in avg.items()
                    }
                    wandb_metrics.update(
                        {"train/lr": current_lr, "train/mem_gib": mem}
                    )
                    self._wandb_log(wandb_metrics, step=self.runtime.global_step)
                    accept = avg.get("accept_rate")
                    tau = avg.get("tau")
                    logger.info(
                        "step %d | loss %.4f | ce %.4f | tv %.4f | conf %.4f | "
                        "accept %s | tau %s | lr %.2e | mem %.2f GiB",
                        self.runtime.global_step,
                        avg["loss"],
                        avg["ce_loss"],
                        avg["l1_loss"],
                        avg["confidence_loss"],
                        "n/a" if accept is None else f"{accept:.3f}",
                        "n/a" if tau is None else f"{tau:.2f}",
                        current_lr,
                        mem,
                    )

    def run_train_validation_loop(self):
        """Run the DSpark training loop."""
        self.trainer_module.train()
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            if getattr(self, "metric_logger", None) is not None:
                self.metric_logger.close()
            self._finish_wandb()
            return

        pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch_idx)

                running_loss = 0.0
                running_ce = 0.0
                running_l1 = 0.0
                running_conf = 0.0
                # Acceptance diagnostics accumulate as (num, den) sums, not per-step
                # ratios: reducing the sums and dividing once gives the exact global
                # ratio regardless of per-micro-batch token imbalance, and keeps tau at
                # its >= 1 floor even when a micro-batch contributes no valid blocks.
                running_tau_num = 0.0
                running_tau_den = 0.0
                running_conf_abs_err_num = 0.0
                running_conf_bias_num = 0.0
                running_conf_cumprod_bias_num = 0.0
                running_conf_diag_den = 0.0
                running_accept_pos_num = torch.zeros(self.block_size, device=self.device)
                running_accept_pos_den = torch.zeros(self.block_size, device=self.device)
                running_micro = 0
                epoch_loss = 0.0
                micro_step = 0
                pending_micro_batches = 0
                completed_steps = 0
                last_batch_idx = -1
                num_batches = len(self.train_dataloader)
                for batch_idx, batch in enumerate(self.train_dataloader):
                    last_batch_idx = batch_idx
                    is_optim_step = (pending_micro_batches + 1 == self.grad_accumulation_steps) or (
                        batch_idx == num_batches - 1
                    )
                    # get_sync_ctx handles both DDP (no_sync) and FSDP2 (set_requires_gradient_sync).
                    with get_sync_ctx(self.trainer_module, is_optim_step, self.defer_fsdp_grad_sync):
                        metrics = self._forward_batch(batch)
                        loss = metrics.loss / self.grad_accumulation_steps
                        loss.backward()

                    running_loss += metrics.loss.detach().item()
                    running_ce += metrics.ce_loss.detach().item()
                    running_l1 += metrics.l1_loss.detach().item()
                    running_conf += metrics.confidence_loss.detach().item()
                    running_tau_num += metrics.tau_num.detach().item()
                    running_tau_den += metrics.tau_den.detach().item()
                    running_conf_abs_err_num += metrics.confidence_abs_error_num.detach().item()
                    running_conf_bias_num += metrics.confidence_bias_num.detach().item()
                    running_conf_cumprod_bias_num += metrics.confidence_cumprod_bias_num.detach().item()
                    running_conf_diag_den += metrics.confidence_diag_den.detach().item()
                    running_accept_pos_num += metrics.accept_rate_per_pos_num.detach()
                    running_accept_pos_den += metrics.accept_rate_per_pos_den.detach()
                    running_micro += 1
                    epoch_loss += metrics.loss.detach().item()
                    micro_step += 1
                    pending_micro_batches += 1

                    if pending_micro_batches == self.grad_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.lr_scheduler.step()
                        self._maybe_precompute_fp8_scales()
                        self.runtime.global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        completed_steps += 1
                        pending_micro_batches = 0
                        self._maybe_save_step_checkpoint(epoch_idx)

                        if self.runtime.global_step % self.log_every_steps == 0:
                            # One collective: the loss window sums and micro-batch count,
                            # the acceptance-diagnostic (num, den) sums, and the per-position
                            # accept sums, concatenated so a single all-reduce covers them.
                            # Losses divide by the micro-batch count (window mean of already
                            # normalized values); the diagnostics divide num by den for the
                            # exact global ratio.
                            scalars = torch.tensor(
                                [
                                    running_loss,
                                    running_ce,
                                    running_l1,
                                    running_conf,
                                    running_tau_num,
                                    running_tau_den,
                                    running_conf_abs_err_num,
                                    running_conf_bias_num,
                                    running_conf_cumprod_bias_num,
                                    running_conf_diag_den,
                                    float(running_micro),
                                ],
                                device=self.device,
                                dtype=torch.float32,
                            )
                            reduced = self._dp_allreduce(
                                torch.cat([scalars, running_accept_pos_num, running_accept_pos_den])
                            )
                            n_scalars = scalars.numel()
                            w = reduced[:n_scalars].tolist()
                            pos_num = reduced[n_scalars : n_scalars + self.block_size]
                            pos_den = reduced[n_scalars + self.block_size :]
                            count = max(1.0, w[10])
                            avg = {
                                "loss": w[0] / count,
                                "ce_loss": w[1] / count,
                                "l1_loss": w[2] / count,
                                "confidence_loss": w[3] / count,
                            }
                            # Log a diagnostic only when it was measured this window (its
                            # denominator is positive), so an ablation without the TV signal
                            # or the confidence head shows no curve rather than a flat zero
                            # that reads like collapsed acceptance.
                            accept_den = pos_den.sum().item()
                            if accept_den > 0:
                                avg["accept_rate"] = pos_num.sum().item() / accept_den
                                _add_accept_rate_per_position(avg, pos_num, pos_den)
                            if w[5] > 0:
                                avg["tau"] = w[4] / w[5]
                            if w[9] > 0:
                                avg["confidence_abs_error"] = w[6] / w[9]
                                avg["confidence_bias"] = w[7] / w[9]
                                avg["confidence_cumprod_bias"] = w[8] / w[9]
                            running_loss = running_ce = running_l1 = running_conf = 0.0
                            running_tau_num = running_tau_den = 0.0
                            running_conf_abs_err_num = running_conf_bias_num = 0.0
                            running_conf_cumprod_bias_num = running_conf_diag_den = 0.0
                            running_accept_pos_num = torch.zeros(self.block_size, device=self.device)
                            running_accept_pos_den = torch.zeros(self.block_size, device=self.device)
                            running_micro = 0
                            if self.dist_env.is_main:
                                current_lr = self.lr_scheduler.get_last_lr()[0]
                                mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                                self.metric_logger.log(
                                    MetricsSample(
                                        step=self.runtime.global_step,
                                        epoch=epoch_idx,
                                        metrics={**avg, "lr": current_lr, "mem": mem},
                                    )
                                )
                                # ``avg`` renames l1_loss -> tv_loss and carries only the
                                # diagnostics measured this window, so mirror its keys under
                                # the train/ prefix rather than hard-coding each one.
                                wandb_metrics = {
                                    "train/tv_loss" if key == "l1_loss" else f"train/{key}": value
                                    for key, value in avg.items()
                                }
                                wandb_metrics.update(
                                    {"train/lr": current_lr, "train/mem_gib": mem, "train/epoch": epoch_idx}
                                )
                                self._wandb_log(wandb_metrics, step=self.runtime.global_step)
                                if pbar is not None:
                                    pbar.set_postfix(loss=f"{avg['loss']:.4f}", lr=f"{current_lr:.2e}")
                                accept = avg.get("accept_rate")
                                tau = avg.get("tau")
                                logger.info(
                                    "step %d | epoch %d | loss %.4f | ce %.4f | tv %.4f | conf %.4f | "
                                    "accept %s | tau %s | lr %.2e | mem %.2f GiB",
                                    self.runtime.global_step,
                                    epoch_idx,
                                    avg["loss"],
                                    avg["ce_loss"],
                                    avg["l1_loss"],
                                    avg["confidence_loss"],
                                    "n/a" if accept is None else f"{accept:.3f}",
                                    "n/a" if tau is None else f"{tau:.2f}",
                                    current_lr,
                                    mem,
                                )

                # Flush the trailing partial accumulation window (see EAGLE recipes
                # for the rescale rationale).
                if pending_micro_batches > 0:
                    scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
                    for p in self.trainer_module.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    self._maybe_precompute_fp8_scales()
                    self.runtime.global_step += 1
                    if pbar is not None:
                        pbar.update(1)
                    completed_steps += 1
                    pending_micro_batches = 0
                    self._maybe_save_step_checkpoint(epoch_idx)

                eval_metrics = self._run_eval()
                if self.dist_env.is_main:
                    msg = f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps}"
                    if eval_metrics is not None:
                        msg += f" val_loss={eval_metrics['val_loss']:.4f}"
                        self._wandb_log(
                            {"val/loss": eval_metrics["val_loss"], "val/epoch": epoch_idx},
                            step=self.runtime.global_step,
                        )
                    logger.info(msg)

                if getattr(self, "save_checkpoint_every_epoch", False) and last_batch_idx >= 0:
                    avg_loss = epoch_loss / max(1, micro_step) if micro_step else None
                    self.save_checkpoint(
                        epoch=epoch_idx + 1,
                        step=self.runtime.global_step,
                        train_loss=avg_loss,
                        val_loss=eval_metrics,
                        best_metric_key="val_loss",
                        is_final_checkpoint=epoch_idx + 1 >= self.num_epochs,
                    )
                    self._log_saved_checkpoint("epoch", epoch_idx + 1, self.runtime.global_step)

            self._maybe_save_final_checkpoint(self.num_epochs)
            self._finalize_and_close_checkpointer()
        finally:
            if pbar is not None:
                pbar.close()
            if getattr(self, "metric_logger", None) is not None:
                self.metric_logger.close()
            self._finish_wandb()


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDSparkRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDSparkConcurrentRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
