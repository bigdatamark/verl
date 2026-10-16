# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# [MODULE DOC]
# VeOmni-backed training engine for verl. Subclasses the FSDP engine family but builds models,
# parallelism (FSDP2 + Ulysses sequence parallel + expert parallel), optimizers and LR
# schedulers through VeOmni; adds VeOmni-specific input transforms (vision-language masks,
# SP sharding, varlen flash-attention kwargs), MoE router replay (record/replay of routing
# decisions), weight export for rollout sync (full and sharded-delta), offload handling, and
# registers language-model and value-model variants under backend "veomni" for cuda/npu.
# Side effects at import: module logger; requires veomni installed; registry registration.


import logging
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from veomni.arguments import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed import parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model
from veomni.models.checkpoint_tensor_loading import get_checkpoint_tensor_converter
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
    slice_input_tensor,
)
from verl.utils.veomni.router_replay import RouterReplayAction, VeOmniRouterReplay
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig

from ..base import BaseEngineCtx, EngineRegistry
from ..fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead, FSDPEngineWithValueHead
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    VL_TYPE2INDEX,
    get_moe_param_handler,
    load_safetensors_index,
    load_veomni_model_to_gpu,
    load_veomni_optimizer,
    offload_veomni_model_to_cpu,
    offload_veomni_optimizer,
)

logger = logging.getLogger(__file__)


def _build_ops_implementation_config(engine_config: VeOmniEngineConfig) -> OpsImplementationConfig:
    # [FUNCTION DOC]
    # HELPER: Builds VeOmni's OpsImplementationConfig from the engine config.
    # Behavior: collects every engine config dataclass field whose name ends in "_implementation".
    # Fields known to the installed OpsImplementationConfig are forwarded; unknown fields still at
    # their dataclass default are skipped (logged at info); unknown fields set to non-default
    # values are collected as unsupported.
    # Inputs: engine_config - VeOmniEngineConfig dataclass.
    # Outputs: an OpsImplementationConfig constructed from forwarded fields.
    # Exceptions: ValueError listing unsupported explicitly-set selectors.
    # KNOWN ISSUE: fields whose default is supplied via default_factory have a sentinel default, so
    # they always compare as non-default and would be reported unsupported if unknown.
    """Forward every ``*_implementation`` selector that the installed VeOmni accepts.

    ``VeOmniEngineConfig`` mirrors VeOmni's ``OpsImplementationConfig`` field by field, so a
    verl build newer than the installed VeOmni would pass unknown keyword arguments and fail
    in the constructor. A selector the installed VeOmni does not know is skipped while it is
    still at its verl default, and rejected otherwise, so a kernel the user explicitly asked
    for is never silently downgraded to whatever that VeOmni version does by default.
    """
    accepted = {f.name for f in fields(OpsImplementationConfig)}

    kwargs, unsupported, skipped = {}, {}, []
    for f in fields(engine_config):
        if not f.name.endswith("_implementation"):
            continue
        value = getattr(engine_config, f.name)
        if f.name in accepted:
            kwargs[f.name] = value
        elif value != f.default:
            unsupported[f.name] = value
        else:
            skipped.append(f.name)

    if unsupported:
        raise ValueError(
            f"The installed VeOmni's OpsImplementationConfig has no {sorted(unsupported)}, but "
            f"they were explicitly set to {unsupported}. Upgrade VeOmni or unset these options."
        )
    if skipped:
        logger.info(f"Skipping {sorted(skipped)}: unknown to the installed VeOmni, left at the verl default.")

    return OpsImplementationConfig(**kwargs)


class VeOmniEngine(FSDPEngine):
    # [CLASS DOC]
    # Base VeOmni engine (FSDP2 only). Holds model/engine/optimizer/checkpoint configs, VeOmni
    # parallel state, offload flags, router-replay controller, and entropy function. Class flag
    # indicates VeOmni computes position-id-derived attention kwargs itself.
    _veomni_handles_position_ids = True

    def _apply_veomni_input_transforms(self, model_inputs: dict, micro_batch: TensorDict):
        # [FUNCTION DOC]
        # HELPER: Mutates model_inputs in place before the forward pass.
        # Behavior: if the model type is a known vision-language type, adds boolean image and video
        # masks (input ids equal to the type's image/video placeholder token index) and, when SP is
        # enabled, runs the sequence shard collator over model_inputs (padding pixel features and
        # slicing sequence features for this SP rank). Then if use_remove_padding (non-tensor flag,
        # default true) and position ids are present, adds varlen flash-attention kwargs derived from
        # the full position ids and, when SP is enabled, slices position ids along the last dim for
        # this SP rank.
        # Inputs: model_inputs - mapping prepared by the parent; micro_batch - TensorDict for flags.
        # Outputs: none. Exceptions: ValueError from position-id layout validation.
        """Apply VeOmni-specific input transforms shared by LM and value heads.

        Handles vision-language model masks, sequence parallel sharding,
        and flash attention kwargs from position_ids.
        """
        input_ids_rmpad = model_inputs["input_ids"]
        sp_enabled = parallel_state.get_parallel_state().sp_enabled
        sp_shard_collator = OmniSequenceShardCollator() if sp_enabled else None

        if self.module.config.model_type in VL_TYPE2INDEX.keys():
            image_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["IMAGE_INPUT_INDEX"]
            video_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["VIDEO_INPUT_INDEX"]
            model_inputs.update({"image_mask": image_mask, "video_mask": video_mask})

            if sp_enabled:
                sp_shard_collator(model_inputs)

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        if use_remove_padding and model_inputs.get("position_ids", None) is not None:
            model_inputs.update(_prepare_veomni_flash_attention_kwargs(model_inputs["position_ids"]))
            if sp_enabled:
                model_inputs["position_ids"] = sp_shard_collator.sp_slice(model_inputs["position_ids"], dim=-1)

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: VeOmniEngineConfig,
        optimizer_config: VeOmniOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: stores configs, fixes data-parallel mode to fsdp2, records rank. Computes
        # dp_size = world_size / ulysses size; if fsdp_size is negative or at least dp_size uses pure
        # sharding (replicate 1, shard dp_size), else requires dp_size divisible by fsdp_size and uses
        # hybrid sharding (replicate dp_size/fsdp_size, shard fsdp_size). Initializes VeOmni's global
        # parallel state with these sizes plus expert-parallel and ulysses sizes. Optionally enables
        # full determinism. Records remove-padding, param/optimizer offload, LoRA (rank > 0), and
        # whether FSDP2 CPU offload policy is used. Records SP enabled/size and the SP process group.
        # Chooses chunked or plain entropy-from-logits, optionally wrapped in torch.compile (dynamic).
        # Records router replay mode/enabled flag (controller created later) and pad-to-length options.
        # Inputs: model_config, engine_config, optimizer_config, checkpoint_config; kwargs ignored.
        # Exceptions: ValueError when dp_size is not divisible by fsdp_size.
        # Side effects: initializes global parallel state; may mutate env/RNG via determinism; logs.
        # Note: does not call the parent constructor.
        """
        Initialize the VeOmniEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with VeOmni and model settings.
        """

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        # VeOmniEngine only supports fsdp2.
        self.data_parallel_mode = "fsdp2"
        self.rank = dist.get_rank()

        fsdp_size = self.engine_config.fsdp_size
        world_size = dist.get_world_size()
        dp_size = world_size // self.engine_config.ulysses_parallel_size

        if fsdp_size < 0 or fsdp_size >= dp_size:
            data_parallel_replicate_size = 1
            data_parallel_shard_size = dp_size
        else:
            if dp_size % fsdp_size != 0:
                raise ValueError(
                    f"Data parallel size ({dp_size}) must be divisible by fsdp_size ({fsdp_size}). "
                    "Please adjust your parallel configuration."
                )
            data_parallel_replicate_size = dp_size // fsdp_size
            data_parallel_shard_size = fsdp_size

        parallel_state.init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=data_parallel_replicate_size,
            dp_shard_size=data_parallel_shard_size,
            extra_parallel_sizes=(self.engine_config.expert_parallel_size,),
            ulysses_size=self.engine_config.ulysses_parallel_size,
            dp_mode=self.data_parallel_mode,
        )

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self.use_remove_padding = self.model_config.use_remove_padding

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0
        # When VeOmni parallelizes with enable_fsdp_offload, FSDP2 uses CPUOffloadPolicy and
        # owns CPU<->accelerator placement. Manually calling model.to(device) then crashes
        # state_dict() with a DTensor storage device mismatch (see #5995 / #6604, which fixed
        # the FSDP engine; the VeOmni engine has the same paths and needs the same guard).
        self._uses_fsdp2_cpu_offload_policy = self.engine_config.enable_fsdp_offload

        self.use_ulysses_sp = parallel_state.get_parallel_state().sp_enabled
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_parallel_size

        if self.use_ulysses_sp:
            self.ulysses_parallel_group = parallel_state.get_parallel_state().device_mesh["sp"].get_group()
        else:
            self.ulysses_parallel_group = None

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

        # Router replay (R2 / R3) for MoE models. Controller is attached in
        # initialize() after the model is built; here we only record intent.
        self._router_replay_mode: str = self.engine_config.router_replay.mode
        self.enable_routing_replay: bool = self._router_replay_mode != "disabled"
        self._router_replay: VeOmniRouterReplay | None = None
        if self.enable_routing_replay:
            logger.info("VeOmniEngine: router_replay enabled, mode=%s", self._router_replay_mode)

        self.pad_to_length: bool = self.engine_config.pad_to_length
        self.pad_to_length_bucket: int = self.engine_config.pad_to_length_bucket

    def initialize(self):
        # [FUNCTION DOC]
        # Behavior: builds model/optimizer/scheduler; if router replay is enabled, requires
        # use_remove_padding in engine config, creates the replay controller, and installs it on the
        # module. Creates the FSDP checkpoint manager, then offloads model/optimizer/grads to CPU per
        # offload flags and logs memory.
        # Exceptions: RuntimeError if router replay enabled without remove padding; errors from replay
        # install if VeOmni lacks support.
        # Side effects: sets module, optimizer, lr_scheduler, checkpoint_manager, _router_replay.
        """
        Build the model, optimizer, and learning rate scheduler under VeOmni.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        self._build_model_optimizer()

        if self.enable_routing_replay:
            # Defense in depth: the VeOmniActorConfig check is the primary
            # fail-fast point and runs *before* engine init. By the time we get here,
            # ``_build_model_optimizer()`` has already finished — this
            # second check exists to catch direct ``VeOmniEngine``
            # instantiation paths that bypass the worker (e.g. unit tests,
            # standalone debug scripts) so the user gets a typed config
            # error instead of an opaque mid-step ``AttributeError`` on
            # ``input_ids.offsets()``.
            if not self.engine_config.use_remove_padding:
                raise RuntimeError(
                    "router_replay requires use_remove_padding=True. In VeOmni engine, "
                    "the non-remove-padding path also disables Ulysses SP slicing and "
                    "the fused-kernel log_probs path, and is not a tested production "
                    "configuration for MoE routing replay. Set "
                    "actor.model.use_remove_padding=True or "
                    "router_replay.mode='disabled'."
                )
            self._router_replay = VeOmniRouterReplay()
            # Fails loudly if the VeOmni build in the environment does not
            # export `set_active_replay` yet (plan requires upgrading VeOmni
            # or disabling router_replay).
            self._router_replay.install(self.module)

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_optimizer,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _build_optimizer(self, module):
        # [FUNCTION DOC]
        # HELPER: Builds a VeOmni optimizer for the module from optimizer config (lr, betas, weight
        # decay, type). If the module exposes a get_optimizer_pre_hook factory, registers the hook it
        # returns (given module, its config and dp mode) as a step pre-hook.
        # Outputs: the optimizer.
        optimizer = build_optimizer(
            module,
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
            optimizer_type=self.optimizer_config.optimizer,
        )
        get_optimizer_pre_hook = getattr(module, "get_optimizer_pre_hook", None)
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(module, module.config, self.data_parallel_mode)
            optimizer.register_step_pre_hook(optimizer_pre_hook)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        # [FUNCTION DOC]
        # HELPER: Builds a VeOmni LR scheduler from optimizer config (total steps, lr, min lr, decay
        # style, decay ratio, warmup ratio, start lr). Outputs: the scheduler.
        optim_config = self.optimizer_config
        lr_scheduler = build_lr_scheduler(
            optimizer,
            train_steps=optim_config.total_training_steps,
            lr=optim_config.lr,
            lr_min=optim_config.lr_min,
            lr_decay_style=optim_config.lr_scheduler_type,
            lr_decay_ratio=optim_config.lr_decay_ratio,
            lr_warmup_ratio=optim_config.lr_warmup_steps_ratio,
            lr_start=optim_config.lr_start,
        )

        return lr_scheduler

    def _get_model_config_path(self):
        # [FUNCTION DOC]
        # HELPER: Returns what to pass as config_path to model building; base returns the local HF config
        # path. Overridable hook.
        """Return the config path (or PretrainedConfig) for build_foundation_model.

        Subclasses can override to modify the HF config before model construction
        (e.g. VeOmniEngineWithValueHead rewrites architectures to ForTokenClassification).
        """
        return self.model_config.local_hf_config_path

    def _maybe_apply_async_activation_offload(self, module):
        # [FUNCTION DOC]
        # HELPER: If async activation offload is enabled, lazily imports VeOmni's offload API and applies
        # it to the unparallelized module for the configured module names, with host cache limit
        # converted from GB to bytes; logs memory. Otherwise no-op.
        # Exceptions: ImportError if VeOmni lacks the feature (only when enabled).
        # Side effects: wraps module layers' call behavior.
        """Attach VeOmni's stream-based activation offload to the decoder layers.

        Must run on the unparallelized module: the offload wraps each selected layer's
        ``__call__`` so that ``saved_tensors_hooks`` sits *outside* the gradient-checkpoint
        boundary and the FSDP2 sharding that ``build_parallelize_model`` installs. Applying it
        afterwards would put the hooks inside the checkpoint region, where the packed
        activations are recomputed rather than saved, so nothing would be offloaded.
        """
        if not self.engine_config.enable_async_activation_offload:
            return

        # Imported lazily so a VeOmni predating stream-based offload only fails when asked for it.
        from veomni.distributed.async_offload import apply_async_activation_offload

        apply_async_activation_offload(
            module,
            list(self.engine_config.activation_offload_modules),
            host_cache_limit_bytes=int(self.engine_config.activation_offload_host_cache_limit_gb * 1024**3),
        )
        log_gpu_memory_usage("After apply async activation offload", logger=logger)

    def _build_model_optimizer(self):
        # [FUNCTION DOC]
        # HELPER: Builds the ops implementation config and mixed-precision config; builds the foundation
        # model (float32 weights when mixed precision is enabled, else bfloat16) with configured
        # attention/ops implementation and init device; applies optional activation offload; then
        # parallelizes via VeOmni (full shard, mixed precision, gradient checkpointing, FSDP offload,
        # basic modules = union of model's no-split modules and configured ones, reentrant, forward
        # prefetch, broadcast weights from rank 0, and the safetensors index mapping). Builds optimizer
        # and scheduler unless forward_only, else both None.
        # Side effects: sets module, optimizer, lr_scheduler; logs memory.
        # build_foundation_model runs apply_ops_config(ops_implementation)
        # before constructing the model, so per-model device_patch files see
        # the resolved kernel backends.
        ops_implementation = _build_ops_implementation_config(self.engine_config)

        veomni_mixed_precision_config = MixedPrecisionConfig(enable=self.engine_config.mixed_precision)

        # Load base model with specified configuration and dtype
        module = build_foundation_model(
            config_path=self._get_model_config_path(),
            weights_path=self.model_config.local_path,
            torch_dtype="float32" if veomni_mixed_precision_config.enable else "bfloat16",
            attn_implementation=self.engine_config.attn_implementation,
            ops_implementation=ops_implementation,
            init_device=self.engine_config.init_device,
        )
        log_gpu_memory_usage("After load base model", logger=logger)

        self._maybe_apply_async_activation_offload(module)

        # Applies parallel strategies to the model.
        log_gpu_memory_usage("Before parallelize model", logger=logger)
        module = build_parallelize_model(
            module,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.local_path,
            enable_full_shard=self.engine_config.enable_full_shard,
            mixed_precision=veomni_mixed_precision_config,
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
            enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
            basic_modules=list(
                set(getattr(module, "_no_split_modules", None) or []) | set(self.engine_config.basic_modules)
            ),
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
            broadcast_model_weights_from_rank0=True,
            fqn_to_index_mapping=load_safetensors_index(self.model_config.local_path),
        )
        log_gpu_memory_usage("After parallelize model", logger=logger)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Behavior: clips gradients to clip_grad using the module's own clip method if present, else the
        # torch utility; materializes a DTensor norm. If the norm is not finite, prints a warning and
        # zeros grads without stepping; otherwise steps the optimizer.
        # Outputs: grad norm as a Python float.
        # Side effects: mutates params/optimizer state or grads; prints.
        """
        Perform an optimization step using the optimizer.
        """
        if hasattr(self.module, "clip_grad_norm_"):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        # [FUNCTION DOC]
        # Behavior: stores SP size in data; computes the global token count as the sum of loss_mask
        # all-reduced over the DP group and stores it plus dp_size. Splits into micro-batches (same count
        # across DP). If router replay is enabled and the batch flag enable_routing_replay is true, begins
        # RECORD (mode R2 and forward_only) or REPLAY (otherwise). For each micro-batch runs forward_step
        # under no-grad when forward_only, backpropagates the loss otherwise, and collects meta info;
        # postprocesses into the aggregated output. Always clears the replay controller afterward when
        # active.
        # Inputs: data - TensorDict with loss_mask and batching metadata; loss_function - callable;
        # forward_only - skip backward.
        # Outputs: dict from postprocess_batch_func.
        # Side effects: mutates data non-tensor fields; collectives; accumulates gradients.
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        tu.assign_non_tensor(data, sp_size=parallel_state.get_parallel_state().ulysses_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        # Router replay state machine: decide RECORD vs REPLAY for this step.
        # RECORD: R2 compute_log_prob (forward_only=True).
        # REPLAY: R2 actor update, or R3 always (forward_only=True and False).
        rr_active = self.enable_routing_replay and tu.get_non_tensor_data(data, "enable_routing_replay", default=False)
        if rr_active:
            assert self._router_replay is not None
            if self._router_replay_mode == "R2" and forward_only:
                self._router_replay.begin_record()
            else:
                self._router_replay.begin_replay()

        ctx = torch.no_grad() if forward_only else nullcontext()

        # Wrap the per-step body in try/finally so the controller is always
        # reset to DISABLED even if forward / backward / postprocess raises.
        # Without this, an exception leaves _recorded / _targets pinned
        # (GPU memory) until the next successful step's begin_record/replay
        # clears them, which may never happen if the caller (Ray actor) tears
        # down the worker after the failure.
        try:
            output_lst = []

            for micro_batch in micro_batches:
                with ctx:
                    loss, meta_info = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only
                    )
                if not forward_only:
                    loss.backward()

                output_lst.append(meta_info)

            result = postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
            return result
        finally:
            if rr_active:
                self._router_replay.clear()

    def get_data_parallel_rank(self):
        # [FUNCTION DOC]
        # Behavior: returns this rank's local index in the "dp" mesh dimension.
        return parallel_state.get_parallel_state().device_mesh.get_local_rank("dp")

    def get_data_parallel_size(self):
        # [FUNCTION DOC]
        # Behavior: returns world size divided by ulysses size.
        return torch.distributed.get_world_size() // parallel_state.get_parallel_state().ulysses_size

    def get_data_parallel_group(self):
        # [FUNCTION DOC]
        # Behavior: returns the dp mesh dimension group when ulysses size > 1, else the WORLD group.
        if parallel_state.get_parallel_state().ulysses_size > 1:
            return parallel_state.get_parallel_state().device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        # [FUNCTION DOC]
        # Behavior: not supported. Exceptions: always NotImplementedError.
        raise NotImplementedError

    def get_context_parallel_group(self):
        # [FUNCTION DOC]
        # Behavior: not supported. Exceptions: always NotImplementedError.
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        # [FUNCTION DOC]
        # Behavior: returns true when this rank is local rank 0 in the "ulysses" mesh dim (when ulysses
        # size > 1), otherwise true.
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        if parallel_state.get_parallel_state().ulysses_size > 1:
            is_collect = parallel_state.get_parallel_state().device_mesh["ulysses"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def train_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: returns a train-mode context bound to this engine; kwargs forwarded to the context.
        """
        Return a context manager that switches to training mode with VeOmni-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: returns an eval-mode context bound to this engine; kwargs forwarded.
        """
        Return a context manager that switches to evaluation mode with VeOmni-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # [FUNCTION DOC]
        # Behavior: first calls the grandparent (BaseEngine, skipping FSDPEngine) device-move logic, then
        # for the accelerator device loads model (if model) and optimizer state (if optimizer and one
        # exists) via VeOmni helpers; for "cpu" offloads them.
        # Inputs: device - accelerator name or "cpu"; model, optimizer, grad - which parts to move.
        # Exceptions: AssertionError for any other device string.
        # KNOWN ISSUE: the trailing ValueError branch is unreachable because of the preceding assert;
        # grad flag is only forwarded to the parent, not handled here.
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super(FSDPEngine, self).to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_veomni_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_veomni_optimizer(self.optimizer, device)
        elif device == "cpu":
            if model:
                offload_veomni_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_veomni_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: if params are offloaded or the module currently lives on CPU, and FSDP2 CPU offload
        # policy is not in use, loads the model to GPU. Saves via the checkpoint manager, barriers,
        # then re-offloads the model if param offload is enabled.
        # Inputs: local_path, optional hdfs_path, global_step, max_ckpt_to_keep; kwargs ignored.
        # Side effects: writes checkpoint files; collective barrier; device moves.
        """
        Save VeOmni checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if (self._is_offload_param or origin_module_device == "cpu") and not getattr(
            self, "_uses_fsdp2_cpu_offload_policy", False
        ):
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: loads the model to GPU if param-offloaded (and not using CPU offload policy), loads the
        # checkpoint, barriers, then re-offloads model and/or optimizer per offload flags.
        # Inputs: local_path, optional hdfs_path, del_local_after_load; kwargs ignored.
        # Side effects: mutates model/optimizer state; device moves; barrier.
        """
        Load VeOmni checkpoint, restoring parameters and optimizer state.
        """
        if self._is_offload_param and not getattr(self, "_uses_fsdp2_cpu_offload_policy", False):
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_veomni_optimizer(self.optimizer)

    def get_per_tensor_param_shard(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: unless FSDP2 CPU offload policy is used, loads the model to GPU, then obtains the
        # shard exporter generator and metadata from veomni_shard_export. Wraps the generator so that,
        # after it is exhausted, the model is offloaded back to CPU if manual offload and param offload
        # apply.
        # Outputs: (generator of (name, local shard, spec), metadata).
        # Side effects: device moves (deferred offload happens only when the generator is fully consumed).
        """Yield each rank's *local* shard ``(name, local_shard, ShardSpec)`` -- the
        DTensor export plus veomni's EP declarations. The mechanics live in
        :func:`verl.workers.engine.veomni.utils.veomni_shard_export`; this wrapper
        owns the offload dance (CPUOffloadPolicy manages placement itself -- see
        #5995 -- and the delta path returns early in update_weights, so the
        offload-back happens here, after the exporter is exhausted).
        """
        from .utils import veomni_shard_export

        manual_offload = not getattr(self, "_uses_fsdp2_cpu_offload_policy", False)
        if manual_offload:
            load_veomni_model_to_gpu(self.module)
        gen, meta = veomni_shard_export(self.module)

        def _with_offload_back():
            # [FUNCTION DOC]
            # HELPER: Generator yielding all exporter items then performing the deferred offload.
            yield from gen
            if manual_offload and self._is_offload_param:
                offload_veomni_model_to_cpu(self.module)

        return _with_offload_back(), meta

    def _hf_delta_entry(self, name, spec, place, lidx, lval):
        # [FUNCTION DOC]
        # HELPER: Per-param delta entry builder. If the spec has a to-HF chunk converter, a block
        # placement and declared HF slots, delegates to the veomni converter entry builder; if it has a
        # converter but no slots/block placement, raises; otherwise falls back to the FSDP identity
        # builder.
        # Inputs: name, spec, place, lidx, lval - as in the delta export contract.
        # Outputs: delta entry tuple. Exceptions: NotImplementedError when converter lacks slots.
        """veomni's per-param entry builder: EP/converter specs (fused expert
        stacks) go through this backend's own converter machinery (see
        :mod:`verl.workers.engine.veomni.utils`); everything else falls back to
        the FSDP engine's DTensor identity handling."""
        from ..spec import BlockPlacement
        from .utils import NO_SLOTS_MSG, hf_entry_converter

        if spec.to_hf_chunk is not None and isinstance(place, BlockPlacement) and spec.hf_slots is not None:
            return hf_entry_converter(name, spec, place, lidx, lval)
        if spec.to_hf_chunk is not None:
            raise NotImplementedError(f"{name}: {NO_SLOTS_MSG}")
        return super()._hf_delta_entry(name, spec, place, lidx, lval)

    # get_per_tensor_param_delta_shard is inherited from FSDPEngine and
    # prime_delta_snapshots from BaseEngine; both consume this class's
    # get_per_tensor_param_shard and _hf_delta_entry overrides.

    def get_per_tensor_param(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: loads model to GPU unless CPU offload policy is used. If the model has a checkpoint
        # tensor converter exposing export_weights, returns that export and None. Otherwise takes the
        # state dict, converts keys to HF names, re-offloads params if param offload is on, and selects
        # the MoE param handler for the model type. Returns a lazy generator: each param is materialized
        # (DTensor moved to device and full_tensor; others unchanged). Expert projection params (name has
        # "mlp.experts." and a down/gate/up/gate_up projection) under EP are broadcast from each EP rank in
        # turn (using this rank's tensor or a same-shaped buffer) and passed to the handler with expert id
        # base src_rank times local expert count; other expert params go through the handler with base 0;
        # non-expert params are yielded directly.
        # Outputs: (generator of (name, tensor), None).
        # Side effects: device moves; collective broadcasts during iteration.
        # KNOWN ISSUE: the model is offloaded before the generator runs, so non-DTensor params may be
        # yielded from CPU; LoRA not supported.
        # FSDP2 CPUOffloadPolicy owns CPU<->accelerator placement; calling model.to(device)
        # here leaves the module half-moved and crashes state_dict() below (#5995). The
        # per-DTensor .to(device).full_tensor() in param_generator() below stages each
        # shard instead, so the manual whole-model move is unnecessary under CPU offload.
        if not getattr(self, "_uses_fsdp2_cpu_offload_policy", False):
            load_veomni_model_to_gpu(self.module)

        # TODO: currently only for DeepseekV4, unify all models to export weights by converter.
        converter = get_checkpoint_tensor_converter(self.module)
        if converter is not None and hasattr(converter, "export_weights"):
            return converter.export_weights(self.module), None

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        ps = parallel_state.get_parallel_state()
        model_type = getattr(self.module.config, "model_type", "default")
        process_func = get_moe_param_handler(model_type, ps.ep_enabled)

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy

        def param_generator():
            # [FUNCTION DOC]
            # HELPER: Lazy generator implementing the per-param materialization/broadcast described above.
            for name, param in params.items():
                unsharded_tensor = (
                    param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param
                )

                is_expert_layer = "mlp.experts." in name
                is_proj = any(p in name for p in ["down_proj", "gate_proj", "up_proj", "gate_up_proj"])

                if is_expert_layer and is_proj and ps.ep_enabled:
                    ep_rank, ep_size = ps.ep_rank, ps.ep_size
                    buffer = torch.empty_like(unsharded_tensor)  # [num_experts/ep_size, H, I]
                    for src_ep_rank in range(ep_size):
                        tensor = unsharded_tensor if src_ep_rank == ep_rank else buffer
                        torch.distributed.broadcast(tensor, group_src=src_ep_rank, group=ps.ep_group)
                        yield from process_func(name, tensor, expert_id_base=src_ep_rank * tensor.size(0))

                else:
                    if is_expert_layer:
                        yield from process_func(name, unsharded_tensor, expert_id_base=0)
                    else:
                        yield name, unsharded_tensor

        # TODO: support VeOmni LoRA
        return param_generator(), None


class EngineEvalModeCtx(BaseEngineCtx):
    # [CLASS DOC]
    # Eval-mode context for VeOmniEngine: swaps in the engine's Ulysses SP group on enter and restores
    # the previous one on exit; reshards the root FSDP module on exit when shard size > 1.
    def __init__(self, engine: VeOmniEngine, **kwargs):
        # [FUNCTION DOC]
        # Behavior: initializes the base context with mode "eval"; kwargs forwarded.
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: asserts the engine type, runs base enter, saves the current SP group, installs the
        # engine's SP group, and puts the module in train mode.
        # KNOWN ISSUE: eval context calls module.train() rather than eval(), so dropout etc. remain active
        # (mirrors a TODO in the train context).
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Behavior: restores the previous SP group; if dp shard size > 1 reshards the root FSDP module
        # (FSDP1 handle reshard or FSDP2 reshard); then runs base exit.
        # Inputs: standard exception triple. Outputs: none (exceptions propagate).
        assert isinstance(self.engine, VeOmniEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if parallel_state.get_parallel_state().dp_shard_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    # [CLASS DOC]
    # Train-mode context for VeOmniEngine: SP group swap like eval, and zeroes grads on exit when
    # configured or when an exception occurred.
    def __init__(self, engine: VeOmniEngine, **kwargs):
        # [FUNCTION DOC]
        # Behavior: initializes the base context with mode "train"; kwargs forwarded.
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: asserts engine type, base enter, saves and installs SP group, sets module train mode.
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        # TODO: Switch to eval mode after Integrating the CI environment
        # VeOmni (ref: https://github.com/ByteDance-Seed/VeOmni/pull/421)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Behavior: restores SP group; zeroes optimizer grads if zero_grad_on_exit or an exception is
        # propagating; then base exit.
        assert isinstance(self.engine, VeOmniEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@dataclass
class OmniSequenceShardCollator:
    # [CLASS DOC]
    # Dataclass collator that pads and slices batch features along a sequence dimension for Ulysses
    # SP. Fields: mapping of feature name to slice dim (input_ids/labels last dim, pixel values dim 0),
    # mapping of feature to padding value, and mapping of feature to padding scale (pixel features 4).
    """
    Data collator to chunk inputs along the sequence length.
    """

    # features to slice sequence dimension
    sp_slice_features: dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": -1,
            "labels": -1,
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to slice sequence dimension."},
    )

    # features to padding sequence dimension
    padding_features: dict[str, int] = field(
        default_factory=lambda: {
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to padding sequence dimension."},
    )

    # padding scale for padding features
    padding_scale: dict[str, int] = field(
        default_factory=lambda: {"pixel_values": 4, "pixel_values_videos": 4},
        metadata={"help": "padding scale for padding features."},
    )

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: reads SP size and rank from VeOmni parallel state into attributes.
        self.sp_size = parallel_state.get_parallel_state().sp_size
        self.sp_rank = parallel_state.get_parallel_state().sp_rank

    def sp_slice(self, feature: torch.Tensor, dim: int = -1) -> dict[str, "torch.Tensor"]:
        # [FUNCTION DOC]
        # Behavior: splits the dim into ceil(len/sp_size) chunks and returns this rank's chunk via narrow.
        # Inputs: feature - tensor; dim - dimension. Outputs: a view tensor.
        # KNOWN ISSUE: return annotation says dict but a tensor is returned; if length is not divisible
        # the last rank's narrow can exceed bounds (callers must pre-pad).
        seq_length = feature.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_scale: int = 1
    ) -> "torch.Tensor":
        # [FUNCTION DOC]
        # Behavior: computes padding so the dim length is a multiple of sp_size*pad_scale and appends a
        # tensor filled with pad_value (same dtype/device); returns input unchanged if no pad needed.
        # Inputs: tensor; dim; pad_value; pad_scale. Outputs: padded tensor.
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        seq_length = tensor.size(dim)
        scale_sp_size = self.sp_size * pad_scale

        sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
        pad_size = sp_chunk_size * scale_sp_size - seq_length
        if pad_size == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[dict[str, "torch.Tensor"]]) -> dict[str, "torch.Tensor"]:
        # [FUNCTION DOC]
        # Behavior: pads every padding feature present (using its slice dim, default last), then slices
        # every slice feature present for this SP rank; mutates and returns the batch mapping.
        # Inputs: batch - mapping of feature name to tensor. Outputs: same mapping.
        # KNOWN ISSUE: annotated as a sequence of dicts but used as a single mapping.
        for key in batch.keys():
            if key in self.padding_features.keys():
                batch[key] = self.sp_padding(
                    batch[key],
                    dim=self.sp_slice_features.get(key, -1),
                    pad_value=self.padding_features[key],
                    pad_scale=self.padding_scale.get(key, 1),
                )

        # sp slice
        for key in batch.keys():
            if key in self.sp_slice_features.keys():
                batch[key] = self.sp_slice(batch[key], dim=self.sp_slice_features[key])

        return batch


def _prepare_veomni_flash_attention_kwargs(position_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    # [FUNCTION DOC]
    # HELPER: Accepts 2-D (1, nnz) position ids as-is, or 3-D (rope_dim, 1, nnz) taking the first rope
    # row; derives cumulative seqlens and max lengths via VeOmni.
    # Outputs: mapping with cu_seq_lens_q/k and max_length_q/k.
    # Exceptions: ValueError for 3-D with middle dim not 1 or other ranks.
    """Normalize packed position_ids layout and derive varlen FlashAttention kwargs.

    Supported formats for use_remove_padding=true:
        - 2D: (1, total_nnz) - standard packed format
        - 3D: (rope_dim, 1, total_nnz) - VeRL mRoPE packed format
    """
    if position_ids.dim() == 2:
        # (1, total_nnz) - standard packed format
        fa_position_ids = position_ids
    elif position_ids.dim() == 3:
        # (rope_dim, 1, total_nnz) - VeRL mRoPE packed format
        if position_ids.shape[1] == 1:
            fa_position_ids = position_ids[0]
        else:
            raise ValueError(
                f"Unsupported 3D position_ids shape: {tuple(position_ids.shape)}, expected (rope_dim, 1, total_nnz)"
            )
    else:
        raise ValueError(
            f"Unsupported position_ids rank: {position_ids.dim()}, "
            f"expected 2 (1, total_nnz) or 3 (rope_dim, 1, total_nnz)"
        )

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(fa_position_ids)
    return {
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_k,
        "max_length_q": max_length_q,
        "max_length_k": max_length_k,
    }


@EngineRegistry.register(model_type="language_model", backend=["veomni"], device=["cuda", "npu"])
class VeOmniEngineWithLMHead(VeOmniEngine, FSDPEngineWithLMHead):
    # [CLASS DOC]
    # Language-model VeOmni engine (registered as model_type language_model). Adds fused log-prob
    # /top-K distillation inputs and router-replay preparation and output capture.
    def prepare_model_inputs(self, micro_batch: TensorDict):
        # [FUNCTION DOC]
        # Behavior: gets parent inputs, applies VeOmni transforms. When use_fused_kernels and
        # use_remove_padding: sets labels to input ids, shift_labels to the rolled ids (unsqueezed), and
        # return_log_probs true; additionally if distillation_use_topk and teacher_ids exist, requires
        # teacher_logprobs, flattens both nested tensors to (1, nnz, K), SP-slices along dim 1 with padding
        # when SP is used, and adds them as teacher top-K kwargs. Finally arms router replay.
        # Outputs: (model_inputs, output_args).
        # Exceptions: ValueError when teacher_ids present without teacher_logprobs.
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        self._apply_veomni_input_transforms(model_inputs, micro_batch)

        # Activate VeOmni's chunk_logprobs path: ForCausalLMLoss short-circuits
        # to per-token log_probs/entropy on return_log_probs=True. Pass the
        # already-rolled labels as shift_labels so chunk_logprobs skips its
        # internal causal shift and the output seq length matches the input —
        # prepare_model_outputs().squeeze(0) then lands at (total_nnz,).
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        if use_fused_kernels and use_remove_padding:
            input_ids_rmpad = model_inputs["input_ids"]
            shift_labels = output_args["input_ids_rmpad_rolled"].unsqueeze(0)
            model_inputs["labels"] = input_ids_rmpad
            model_inputs["shift_labels"] = shift_labels
            model_inputs["return_log_probs"] = True

            # Pass teacher top-K tensors so ForCausalLMLoss routes to
            # chunk_topk_distill_function for fused distillation. TD keys
            # teacher_ids / teacher_logprobs are populated by verl's native
            # distillation pipeline (see verl/trainer/distillation/losses.py).
            distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)
            if distillation_use_topk and "teacher_ids" in micro_batch.keys():
                if "teacher_logprobs" not in micro_batch.keys():
                    raise ValueError(
                        "teacher_ids present without teacher_logprobs; "
                        "both must be provided together for fused top-K distillation."
                    )
                # Kernel kwarg names follow veomni's chunk_topk_distill_function API.
                teacher_topk_ids = micro_batch["teacher_ids"].values().unsqueeze(0)
                teacher_topk_log_probs = micro_batch["teacher_logprobs"].values().unsqueeze(0)
                # SP-slice along seqlen (dim=1); teacher tensors are 3D
                # (1, total_nnz, K) so use slice_input_tensor directly —
                # ulysses_pad_and_slice_inputs hardcodes 2D.
                if self.use_ulysses_sp:
                    from verl.utils.ulysses import slice_input_tensor

                    teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1, padding=True)
                    teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1, padding=True)
                model_inputs["teacher_topk_ids"] = teacher_topk_ids
                model_inputs["teacher_topk_log_probs"] = teacher_topk_log_probs

        # Arm router replay for this micro-batch. In REPLAY mode this also
        # reshapes routed_experts with the same pad + Ulysses rule that
        # super().prepare_model_inputs just applied to input_ids.
        self._prepare_router_replay_inputs(micro_batch, output_args)

        return model_inputs, output_args

    def _prepare_router_replay_inputs(self, micro_batch: TensorDict, output_args: dict) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op if no controller or disabled. In RECORD begins a micro-batch. In REPLAY requires
        # routed_experts in the micro-batch, flattens it (nested values) to int64 [nnz, layers, topk],
        # appends pad_size zero rows, SP-slices along dim 0 without padding, and begins the micro-batch
        # with per-layer target tensors (unbound along dim 1).
        # Exceptions: RuntimeError when routed_experts is missing in REPLAY.
        """Arm the router-replay controller for this micro-batch.

        RECORD only needs the controller reset. REPLAY additionally hands it
        the per-layer target indices in the layout the routers will see.
        ``routed_experts`` spans the whole sequence (prompt + response) in the
        same rmpad order as ``input_ids``, so all it takes is appending the
        pad suffix ``super().prepare_model_inputs`` added and slicing along
        the Ulysses SP group the same way it did.
        """
        rr = self._router_replay
        if rr is None or rr.action is RouterReplayAction.DISABLED:
            return

        if rr.action is RouterReplayAction.RECORD:
            rr.begin_microbatch()
            return

        routed = micro_batch.get("routed_experts", None)
        if routed is None:
            raise RuntimeError(
                "router_replay REPLAY: micro_batch missing 'routed_experts'. "
                "Verify that compute_log_prob (R2) or the rollout path (R3) "
                "attached routed_experts to the batch before this engine "
                "call, and that left_right_2_no_padding preserved it."
            )

        # Nested-jagged [bs, seq, L, topk] -> rmpad [total_nnz, L, topk].
        targets = (routed.values() if routed.is_nested else routed).to(torch.int64)
        pad_size = int(output_args.get("pad_size", 0))
        if pad_size:
            targets = torch.cat([targets, targets.new_zeros((pad_size, *targets.shape[1:]))])
        if self.use_ulysses_sp:
            targets = slice_input_tensor(targets, dim=0, padding=False)

        rr.begin_microbatch(targets=list(targets.unbind(dim=1)))

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict, logits_processor_func):
        # [FUNCTION DOC]
        # Behavior: gets parent outputs. In RECORD mode takes recorded routing (reinterpreted as bytes),
        # gathers SP shards and removes pad suffix, then stores it reinterpreted as int16 as a jagged
        # nested tensor using input ids offsets under "routed_experts". In REPLAY verifies the number of
        # routers fired equals number of target layers.
        # Outputs: model output mapping.
        # Exceptions: RuntimeError on fired/target count mismatch.
        """Attach this micro-batch's recorded MoE routing to the model output.

        Mirrors how ``super()`` post-processes ``log_probs``: gather the SP
        shards, drop the pad suffix, re-wrap per sample. From there
        ``postprocess_batch_func`` restores batch order like any other
        per-token output.
        """
        model_output = super().prepare_model_outputs(output, output_args, micro_batch, logits_processor_func)

        rr = self._router_replay
        if rr is not None and rr.action is RouterReplayAction.RECORD:
            recorded = self._gather_and_unpad_packed(rr.take_recorded().view(torch.uint8), output_args["pad_size"])
            model_output["routed_experts"] = torch.nested.nested_tensor_from_jagged(
                recorded.view(torch.int16), micro_batch["input_ids"].offsets()
            )
        elif rr is not None and rr.action is RouterReplayAction.REPLAY:
            if rr.num_fired != rr.num_targets:
                raise RuntimeError(
                    f"router_replay REPLAY: {rr.num_fired} routers fired but routed_experts "
                    f"carries {rr.num_targets} layer slots. Targets are matched to routers by "
                    "fire order, so unequal counts mean every layer replayed the wrong slot. "
                    "Rollout backends index this tensor by absolute decoder-layer index, so "
                    "each decoder layer must contribute exactly one router that calls "
                    "`maybe_replay_indices` -- check that no MoE variant in this model family "
                    "(e.g. a hash-routed or dense layer) is left unhooked."
                )

        return model_output


@EngineRegistry.register(model_type="value_model", backend=["veomni"], device=["cuda", "npu"])
class VeOmniEngineWithValueHead(VeOmniEngine, FSDPEngineWithValueHead):
    # [CLASS DOC]
    # Value-model VeOmni engine (registered as model_type value_model) loading a token
    # classification head with one label.
    """Value model engine using VeOmni's FSDP2 + sequence parallelism.

    Combines VeOmniEngine (model init, parallel state, activation offloading)
    with FSDPEngineWithValueHead (TokenClassification output -> per-token values).
    """

    def _get_model_config_path(self):
        # [FUNCTION DOC]
        # Behavior: builds the HF config via VeOmni, sets num_labels 1, disables dropouts, unties word
        # embeddings, looks up the transformers token-classification class for the config type and sets
        # architectures to it.
        # Outputs: modified config object.
        # Exceptions: ValueError if no token-classification class exists for the config type.
        # KNOWN ISSUE: hidden_dropout is set to the string "0" rather than a number.
        """Return a modified HF config that loads ForTokenClassification(num_labels=1).

        Uses HF's AutoModelForTokenClassification model mapping to resolve the
        canonical ForTokenClassification class name for this model family, then
        sets config.architectures so VeOmni's MODELING_REGISTRY dispatches to it.
        """
        from transformers import AutoModelForTokenClassification
        from veomni.models.auto import build_config

        config = build_config(self.model_config.local_hf_config_path)
        config.num_labels = 1
        config.classifier_dropout = 0.0
        config.hidden_dropout = "0"
        config.summary_dropout_prob = 0.0
        config.tie_word_embeddings = False
        token_cls = AutoModelForTokenClassification._model_mapping.get(type(config), None)
        if token_cls is None:
            raise ValueError(f"No ForTokenClassification class in transformers for {type(config).__name__}.")
        config.architectures = [token_cls.__name__]
        return config

    def prepare_model_inputs(self, micro_batch: TensorDict):
        # [FUNCTION DOC]
        # Behavior: gets parent inputs and applies VeOmni input transforms. Outputs: (inputs, output_args).
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        self._apply_veomni_input_transforms(model_inputs, micro_batch)
        return model_inputs, output_args
