# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# NeMo Automodel engine for verl: Automodel builds, parallelizes (FSDP2 / MegatronFSDP / DDP with TP/CP/EP meshes), shards the optimizer, schedules LR, clips gradients and checkpoints, while verl drives micro-batching, the loss and log-prob/entropy computation. Provides AutomodelEngine, its train/eval contexts, and the registered AutomodelEngineWithLMHead (language_model, backend "automodel", cuda).

"""Automodel (nemo_automodel) engine for verl SFT training.

This engine delegates model building, parallelization, optimizer sharding,
LR scheduling, gradient clipping, and checkpointing to Automodel's
infrastructure while using verl's training loop, data pipeline, and loss function.
"""

import logging
import os
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from huggingface_hub.constants import HF_HUB_CACHE
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from tensordict import TensorDict
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.model import convert_weight_keys, extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import AutomodelEngineConfig, AutomodelOptimizerConfig, HFModelConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import detach_tree, enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    build_automodel_model,
    build_distributed_config_from_engine_config,
    get_dp_group_size,
    get_dp_rank,
    get_pp_rank,
    get_tp_rank,
    load_automodel_model_to_gpu,
    load_automodel_optimizer,
    maybe_fully_shard_optimizer,
    offload_automodel_model_to_cpu,
    offload_automodel_optimizer,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AutomodelEngine(BaseEngine):
    # [CLASS DOC]
    # Base Automodel engine. Attributes: model/engine/optimizer/checkpoint configs, mode, rank, distributed_config, device_mesh, moe_mesh, offload flags, compute_entropy_from_logits; after initialize also module, optimizer, lr_scheduler, checkpointer. Subclasses must implement forward_step.
    """Engine implementation using Automodel for distributed training."""

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: AutomodelEngineConfig,
        optimizer_config: AutomodelOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Stores configs, applies Automodel cache and Transformer Engine compatibility patches, builds distributed config and meshes for the global world size, optionally enables full determinism with the configured seed, records offload flags, and selects (chunked or plain) entropy function, optionally torch.compiled with dynamic shapes.
        # Inputs: model_config, engine_config, optimizer_config, checkpoint_config; extra kwargs ignored.
        # Side effects: global monkey patches; process group/mesh creation; requires initialized torch.distributed.
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self.mode = None
        self.rank = torch.distributed.get_rank()

        # Apply compatibility patches early in the process
        from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
        from nemo_automodel.shared.te_patches import apply_te_patches

        apply_cache_compatibility_patches()
        apply_te_patches()

        world_size = torch.distributed.get_world_size()
        self.distributed_config, self.device_mesh, self.moe_mesh = build_distributed_config_from_engine_config(
            self.engine_config, world_size
        )

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Returns the param_offload flag.
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Returns the optimizer_offload flag.
        return self._is_offload_optimizer

    def initialize(self):
        # [FUNCTION DOC]
        # Builds model, optimizer (sharded for MegatronFSDP) and LR scheduler (both None when forward_only), builds the checkpointer, then offloads model/grad/optimizer to CPU per offload flags and empties the CUDA cache.
        # Side effects: GPU allocation, logging of memory usage.
        """Build the model, optimizer, LR scheduler, and checkpointer using Automodel infrastructure."""
        self.module = build_automodel_model(
            self.model_config, self.engine_config, self.distributed_config, self.device_mesh, self.moe_mesh
        )
        log_gpu_memory_usage("After Automodel model build", logger=logger)

        if not self.engine_config.forward_only:
            self.optimizer = self._build_optimizer(self.module)
            # maybe shard optimizer for MegatronFSDP
            maybe_fully_shard_optimizer(self.module, self.optimizer, self.distributed_config)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.lr_scheduler = None
        self._build_checkpointer()

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)
        torch.cuda.empty_cache()

    def _build_optimizer(self, module):
        # [FUNCTION DOC]
        # HELPER: Builds the optimizer through Automodel's build_optimizer.
        # Behavior: makes a config mapping with _target_ "<optimizer_impl>.<optimizer>", lr, weight_decay, eps, betas; adds master_weights and store_param_remainders when truthy; maps short dtype names (bf16/fp32/fp16) for exp_avg_dtype, exp_avg_sq_dtype, master_weight_dtype to torch dtype strings; merges override_optimizer_config last.
        # Inputs: module - model.
        # Outputs: the single optimizer.
        # Exceptions: AssertionError if Automodel returns not exactly one optimizer.
        """Build optimizer via Automodel's build_optimizer."""
        from nemo_automodel.components.config.loader import ConfigNode
        from nemo_automodel.recipes.llm.train_ft import build_optimizer as automodel_build_optimizer

        config = self.optimizer_config

        opt_dict = {
            "_target_": f"{config.optimizer_impl}.{config.optimizer}",
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "eps": config.eps,
            "betas": list(config.betas),
        }

        if config.master_weights:
            opt_dict["master_weights"] = config.master_weights
        if config.store_param_remainders:
            opt_dict["store_param_remainders"] = config.store_param_remainders

        _short_to_torch = {"bf16": "torch.bfloat16", "fp32": "torch.float32", "fp16": "torch.float16"}
        for attr in ("exp_avg_dtype", "exp_avg_sq_dtype", "master_weight_dtype"):
            val = getattr(config, attr, None)
            if val is not None:
                opt_dict[attr] = _short_to_torch.get(val, val)

        if config.override_optimizer_config:
            opt_dict.update(config.override_optimizer_config)

        cfg_opt = ConfigNode(opt_dict)
        optimizers = automodel_build_optimizer(module, cfg_opt, self.distributed_config, self.device_mesh)
        assert len(optimizers) == 1, f"Expected 1 optimizer, got {len(optimizers)}"
        return optimizers[0]

    def _build_lr_scheduler(self, optimizer):
        # [FUNCTION DOC]
        # HELPER: Builds Automodel's OptimizerParamScheduler.
        # Behavior: warmup steps = lr_warmup_steps, or lr_warmup_steps_ratio times total steps when non-positive; init/min LR = lr times init_lr_ratio (default 0.1) / min_lr_ratio (default 0.01); decay over total_training_steps with lr_scheduler_type; constant weight decay (wd_incr_style default "constant").
        # Outputs: scheduler. Side effects: prints the schedule on rank 0.
        cfg = self.optimizer_config
        total_steps = cfg.total_training_steps
        num_warmup_steps = cfg.lr_warmup_steps

        if num_warmup_steps <= 0:
            num_warmup_steps = int(cfg.lr_warmup_steps_ratio * total_steps)

        base_lr = cfg.lr
        init_lr_ratio = cfg.init_lr_ratio if cfg.init_lr_ratio is not None else 0.1
        min_lr_ratio = cfg.min_lr_ratio if cfg.min_lr_ratio is not None else 0.01

        if self.rank == 0:
            print(
                f"Automodel LR Scheduler: total_steps={total_steps}, warmup={num_warmup_steps}, "
                f"decay_style={cfg.lr_scheduler_type}, init_lr={base_lr * init_lr_ratio:.2e}, "
                f"max_lr={base_lr:.2e}, min_lr={base_lr * min_lr_ratio:.2e}"
            )

        scheduler = OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=base_lr * init_lr_ratio,
            max_lr=base_lr,
            min_lr=base_lr * min_lr_ratio,
            lr_warmup_steps=num_warmup_steps,
            lr_decay_steps=total_steps,
            lr_decay_style=cfg.lr_scheduler_type,
            start_wd=cfg.weight_decay,
            end_wd=cfg.weight_decay,
            wd_incr_steps=total_steps,
            wd_incr_style=getattr(cfg, "wd_incr_style", "constant"),
        )
        return scheduler

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        # [FUNCTION DOC]
        # Runs forward (and backward) over micro-batches.
        # Behavior: all-reduces the number of loss-mask tokens across the DP group and stores it plus dp_size as non-tensor data on the batch; splits into micro-batches with the same count across DP ranks; when training, prepares grad accumulation and, with EP above 1, sets the MoE aux-loss backward scale to the DP(with CP) group size; signals final backward before the last micro-batch; for each micro-batch calls forward_step (under no-grad if forward_only) and backpropagates the loss when training; finally post-processes collected outputs back into original order.
        # Inputs: data - TensorDict with loss_mask; loss_function; forward_only.
        # Outputs: dict from postprocess_batch_func.
        # Side effects: collective all-reduce; gradient accumulation; mutates data.
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        output_lst = []
        ctx = torch.no_grad() if forward_only else nullcontext()

        if not forward_only:
            prepare_for_grad_accumulation([self.module])

            # Set MoE aux loss backward scale to counteract FSDP's gradient allreduce.
            if self.engine_config.ep_size > 1:
                from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                    float(get_dp_group_size(self.device_mesh, include_cp=True))
                )

        num_micro_batches = len(micro_batches)
        for i, micro_batch in enumerate(micro_batches):
            # Signal final backward for MoE
            if not forward_only and i == num_micro_batches - 1:
                prepare_for_final_backward([self.module])

            with ctx:
                loss, meta_info = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(meta_info)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        # [FUNCTION DOC]
        # Abstract per-micro-batch forward; raises NotImplementedError.
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        # [FUNCTION DOC]
        # Zeroes optimizer gradients.
        # KNOWN ISSUE: fails with AttributeError when the engine is forward_only (optimizer is None), e.g. if a train-mode context exits.
        self.optimizer.zero_grad()

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Scales and clips gradients via Automodel (L2 norm, foreach, EP axis "ep" when the MoE mesh has it, DP group size including CP), converts the norm to float; if not finite prints a warning and zeroes grads, else steps the optimizer and updates MoE gate bias when the model supports it. Returns the grad norm as float.
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=self.optimizer_config.clip_grad,
            model_parts=[self.module],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=None,
            dp_group_size=get_dp_group_size(self.device_mesh, include_cp=True),
        )

        if isinstance(grad_norm, torch.Tensor):
            grad_norm_val = grad_norm.item()
        else:
            grad_norm_val = float(grad_norm)

        # If grad_norm is not finite, skip the update
        if not torch.isfinite(torch.tensor(grad_norm_val)):
            print(f"WARN: grad_norm is not finite: {grad_norm_val}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
            if hasattr(self.module, "update_moe_gate_bias"):
                self.module.update_moe_gate_bias()

        return grad_norm_val

    def lr_scheduler_step(self):
        # [FUNCTION DOC]
        # Steps the LR scheduler by one and returns the first param group's LR.
        """Step Automodel's OptimizerParamScheduler and return current LR."""
        self.lr_scheduler.step(increment=1)
        lr = self.optimizer.param_groups[0]["lr"]
        return lr

    def get_data_parallel_rank(self):
        # [FUNCTION DOC]
        # Returns the "dp" local rank of the device mesh, or the global rank without a mesh.
        if self.device_mesh is not None:
            return self.device_mesh.get_local_rank("dp")
        return torch.distributed.get_rank()

    def get_data_parallel_size(self):
        # [FUNCTION DOC]
        # Returns the "dp" mesh dim size, or the world size without a mesh.
        if self.device_mesh is not None:
            return self.device_mesh["dp"].size()
        return torch.distributed.get_world_size()

    def get_data_parallel_group(self):
        # [FUNCTION DOC]
        # Returns the "dp" mesh dim process group, or WORLD without a mesh.
        if self.device_mesh is not None:
            return self.device_mesh.get_group(mesh_dim="dp")
        return torch.distributed.group.WORLD

    def is_mp_src_rank_with_outputs(self):
        # [FUNCTION DOC]
        # Returns true only for TP rank 0 when a tp dim larger than 1 exists; true otherwise.
        if self.device_mesh is not None and "tp" in self.device_mesh.mesh_dim_names:
            if self.device_mesh["tp"].size() > 1:
                return self.device_mesh.get_local_rank("tp") == 0
        return True

    def train_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Returns an AutomodelTrainModeCtx for this engine (kwargs forwarded).
        return AutomodelTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Returns an AutomodelEvalModeCtx for this engine (kwargs forwarded).
        return AutomodelEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # [FUNCTION DOC]
        # Moves model/optimizer between CPU and the accelerator.
        # Behavior: validates arguments via the base; no-op for forward_only engines; for the accelerator device loads model and/or optimizer state; for "cpu" offloads model (reshard) and/or optimizer state. The grad flag is otherwise unused.
        # Exceptions: AssertionError for devices other than the accelerator name or "cpu" (the ValueError branch is unreachable).
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")

        if device == device_name:
            if model:
                load_automodel_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_automodel_optimizer(self.optimizer, get_device_id())
        elif device == "cpu":
            if model:
                offload_automodel_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_automodel_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def _build_checkpointer(self):
        # [FUNCTION DOC]
        # HELPER: Creates Automodel's Checkpointer with safetensors consolidated model saving, HF hub cache, model path as repo id, no PEFT, and DP(with CP)/TP/PP ranks plus MoE mesh.
        ckpt_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="checkpoints/",
            model_save_format="safetensors",
            model_cache_dir=HF_HUB_CACHE,
            model_repo_id=self.model_config.path,
            save_consolidated=True,
            is_peft=False,
        )
        self.checkpointer = Checkpointer(
            config=ckpt_config,
            dp_rank=get_dp_rank(self.device_mesh, include_cp=True),
            tp_rank=get_tp_rank(self.device_mesh),
            pp_rank=get_pp_rank(self.device_mesh),
            moe_mesh=self.moe_mesh,
        )

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Saves model and (if present) optimizer plus LR scheduler with the Checkpointer to local_path.
        # Behavior: loads the model to GPU first when param offload is on or params are on CPU; barriers; re-offloads when param offload is on.
        # KNOWN ISSUE: hdfs_path, global_step and max_ckpt_to_keep are ignored (no upload or checkpoint rotation).
        """Save model, optimizer, and LR scheduler using Automodel's Checkpointer."""
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_automodel_model_to_gpu(self.module)

        # Save model weights
        self.checkpointer.save_model(self.module, local_path)

        # Save optimizer and LR scheduler state
        if self.optimizer is not None:
            scheduler_list = [self.lr_scheduler] if self.lr_scheduler is not None else None
            self.checkpointer.save_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Loads model weights (from local_path/model if that directory exists, else local_path) and optimizer/scheduler state via the Checkpointer, then barriers and re-offloads model and optimizer per offload flags.
        # KNOWN ISSUE: hdfs_path and del_local_after_load are ignored; del_local_after_load is annotated int but defaults to True.
        """Load model, optimizer, and LR scheduler using Automodel's Checkpointer."""
        if self._is_offload_param:
            load_automodel_model_to_gpu(self.module)

        model_path = os.path.join(local_path, "model")
        if not os.path.isdir(model_path):
            model_path = local_path
        self.checkpointer.load_model(self.module, model_path)

        if self.optimizer is not None:
            scheduler_list = [self.lr_scheduler] if self.lr_scheduler is not None else None
            self.checkpointer.load_optimizer(self.optimizer, self.module, local_path, scheduler=scheduler_list)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

        if self._is_offload_optimizer and self.optimizer is not None:
            offload_automodel_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        # [FUNCTION DOC]
        # Exports full weights for rollout sync.
        # Behavior: loads model to GPU, takes the state dict and converts keys to HF naming, re-offloads if param offload is on, then returns a generator yielding (name, full tensor) with DTensors gathered lazily, and None as PEFT config.
        # Side effects: the lazy full_tensor gathers are collectives, so all ranks must consume the generator in lockstep.
        load_automodel_model_to_gpu(self.module)

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            offload_automodel_model_to_cpu(self.module)

        def param_generator():
            # [FUNCTION DOC]
            # HELPER: Yields (name, tensor) for each state-dict entry, materializing DTensors with full_tensor.
            for name, param in params.items():
                unsharded_tensor = param.full_tensor() if isinstance(param, DTensor) else param
                yield name, unsharded_tensor

        return param_generator(), None


class AutomodelEvalModeCtx(BaseEngineCtx):
    # [CLASS DOC]
    # Eval-mode context for Automodel: sets module.eval() on entry and reshards the root FSDP module on exit.
    def __init__(self, engine: AutomodelEngine, **kwargs):
        # [FUNCTION DOC]
        # Creates the context in "eval" mode; kwargs forwarded to BaseEngineCtx.
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        # [FUNCTION DOC]
        # Asserts the engine type, runs base entry (load per offload), then switches the module to eval mode.
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        self.engine.module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Reshards the root module if it supports reshard, then runs base exit (offload, clear mode).
        assert isinstance(self.engine, AutomodelEngine)
        # Reshard the root FSDP module
        if hasattr(self.engine.module, "reshard"):
            self.engine.module.reshard()
        super().__exit__(exc_type, exc_value, traceback)


class AutomodelTrainModeCtx(BaseEngineCtx):
    # [CLASS DOC]
    # Train-mode context for Automodel: sets module.train() on entry and zeroes grads on exit.
    def __init__(self, engine: AutomodelEngine, **kwargs):
        # [FUNCTION DOC]
        # Creates the context in "train" mode; kwargs forwarded to BaseEngineCtx.
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        # [FUNCTION DOC]
        # Asserts the engine type, runs base entry, then switches module to train mode.
        assert isinstance(self.engine, AutomodelEngine)
        super().__enter__()
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Zeroes gradients if zero_grad_on_exit is set or an exception occurred, then runs base exit.
        assert isinstance(self.engine, AutomodelEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["automodel"], device=["cuda"])
class AutomodelEngineWithLMHead(AutomodelEngine):
    # [CLASS DOC]
    # Causal-LM Automodel engine computing per-token log-probs (and optional entropy) of next tokens from nested (jagged) input batches.
    """Automodel engine for language model with LM head training."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        # [FUNCTION DOC]
        # Builds model inputs from a micro-batch of nested tensors.
        # Behavior: reads non-tensor flags use_remove_padding (default true), pad_mode (default NO_PADDING), use_fused_kernels; requires pad_mode NO_PADDING. Broadcasts a scalar temperature to a per-sample float32 tensor. Remove-padding path: packs input ids into shape (1, total_tokens) and position ids into (1, total) or (3or4, 1, total) for 3D; records labels as input ids rolled left by one and per-token temperatures; inputs have no attention mask; for TE attention adds thd format, int32 cu_seqlens (with leading batch dim) and max_seqlen. Padded path: right-pads input ids (pad_token_id) and position ids (0; 3D variant to shape (4, batch, len) after transpose) to the max length, builds an attention mask of ones over valid tokens, records rolled packed labels and per-sample temperature. Fused kernels add the scalar temperature and return_dict. Multi-modal inputs are merged in.
        # Inputs: micro_batch - TensorDict.
        # Outputs: (model_inputs mapping, output_args mapping).
        # Exceptions: AssertionError for per-sample temperature with fused kernels, non-NO_PADDING pad mode or batch-size mismatch.
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        temperature = micro_batch["temperature"]
        temperature_item = temperature
        if use_fused_kernels:
            assert not isinstance(temperature, torch.Tensor), (
                "use_fused_kernels does not support per sample temperature yet"
            )
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]

        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

        temperature = temperature.to(torch.float32)
        assert temperature.shape[0] == input_ids.shape[0]

        output_args = {}

        if use_remove_padding:
            temperature_rmpad = verl_F.expand_as_nested(temperature, input_ids).values()
            temperature_rmpad = temperature_rmpad.unsqueeze(0)

            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids_rmpad = input_ids.values().unsqueeze(0)
                if position_ids.dim() == 3:
                    position_ids_rmpad = position_ids.values().unsqueeze(1)
                else:
                    position_ids_rmpad = position_ids.values().unsqueeze(0)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            temperature_rmpad = temperature_rmpad.squeeze(0)
            output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
            output_args["temperature_rmpad"] = temperature_rmpad

            model_inputs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": None,
                "position_ids": position_ids_rmpad,
            }

            # For TE attention backend, pass cu_seqlens
            if self.engine_config.attn_implementation == "te":
                cu_seqlens = input_ids.offsets().to(torch.int32)
                max_seqlen = cu_seqlens.diff().max().item()
                model_inputs["qkv_format"] = "thd"
                model_inputs["cu_seqlens"] = cu_seqlens.unsqueeze(0)
                model_inputs["max_seqlen"] = max_seqlen

        else:
            if pad_mode == DatasetPadMode.NO_PADDING:
                input_ids = micro_batch["input_ids"]
                position_ids = micro_batch["position_ids"]
                loss_mask = micro_batch["loss_mask"]

                pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
                batch_size = micro_batch.batch_size[0]
                seq_len_effective = input_ids.offsets().diff()
                max_seq_len = max(seq_len_effective)

                input_ids_rmpad_rolled = torch.roll(input_ids.values(), shifts=-1, dims=0)
                output_args["input_ids_rmpad_rolled"] = input_ids_rmpad_rolled
                output_args["temperature"] = temperature

                input_ids = torch.nested.to_padded_tensor(
                    input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
                )

                if position_ids.dim() == 3:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                    ).transpose(0, 1)
                else:
                    position_ids = torch.nested.to_padded_tensor(
                        position_ids, padding=0, output_size=(batch_size, max_seq_len)
                    )

                attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
                attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
                attention_mask = torch.nested.to_padded_tensor(
                    attention_mask, padding=0, output_size=(batch_size, max_seq_len)
                )

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }

            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        extra_args = {}
        if use_fused_kernels:
            extra_args["temperature"] = temperature_item
            extra_args["return_dict"] = True

        model_inputs.update(multi_modal_inputs)
        model_inputs.update(extra_args)

        return model_inputs, output_args

    def prepare_model_outputs(self, output, output_args, micro_batch: TensorDict):
        # [FUNCTION DOC]
        # Converts raw model output into log_probs (and entropy) nested tensors aligned with input ids.
        # Behavior: wraps bare tensor output as an object with logits. Remove-padding: fused kernels supply log_probs/entropy directly; otherwise gathers DTensor logits, divides by clamped temperature, computes log-probs of the rolled labels (in-place backward unless entropy is needed) and entropy (chunked, plain, or activation-checkpointed); then re-nests by input offsets. Padded: fused kernels slice the last response_length positions; otherwise scales logits by temperature, optionally computes entropy, narrows each row to its valid length, concatenates, computes log-probs and re-nests both.
        # Inputs: output - model output; output_args - from prepare_model_inputs; micro_batch.
        # Outputs: mapping with "log_probs" and optionally "entropy".
        # Exceptions: NotImplementedError for unsupported pad modes.
        # KNOWN ISSUE: in the padded path with fused kernels, outputs are dense tensors sliced by max_response_length rather than nested per-sequence tensors, inconsistent with the other paths.
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)

        if isinstance(output, torch.Tensor):
            from types import SimpleNamespace

            output = SimpleNamespace(logits=output)

        model_output = {}
        input_ids = micro_batch["input_ids"]

        if use_remove_padding:
            input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
            temperature_rmpad = output_args["temperature_rmpad"]

            if use_fused_kernels:
                log_probs = output.log_probs.squeeze(0)
                entropy_rmpad = output.entropy.squeeze(0)
            else:
                logits_rmpad = output.logits.squeeze(0)
                # With TP, logits are DTensors sharded on vocab dim; gather for log_softmax.
                if isinstance(logits_rmpad, DTensor):
                    logits_rmpad = logits_rmpad.full_tensor()
                logits_rmpad = logits_rmpad / temperature_rmpad.clamp(min=1e-8).unsqueeze(-1).to(logits_rmpad.dtype)

                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    inplace_backward=inplace_backward,
                )

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        if self.engine_config.entropy_from_logits_with_chunking:
                            entropy_rmpad = self.compute_entropy_from_logits(
                                logits_rmpad,
                                chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                            )
                        else:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                    else:
                        entropy_rmpad = torch.utils.checkpoint.checkpoint(
                            self.compute_entropy_from_logits, logits_rmpad
                        )

            if pad_mode == DatasetPadMode.NO_PADDING:
                cu_seqlens = input_ids.offsets()
                log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                if calculate_entropy:
                    entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
            else:
                raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        else:
            response_length = tu.get_non_tensor_data(data=micro_batch, key="max_response_length", default=1024)
            if use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]
            else:
                logits = output.logits
                # With TP, logits are DTensors sharded on vocab dim; gather for log_softmax.
                if isinstance(logits, DTensor):
                    logits = logits.full_tensor()
                temperature = output_args["temperature"]
                temperature = temperature.unsqueeze(-1).unsqueeze(-1)
                logits = logits / temperature.clamp(min=1e-8).to(logits.dtype)

                if calculate_entropy:
                    if not self.engine_config.entropy_checkpointing:
                        entropy = verl_F.entropy_from_logits(logits)
                    else:
                        entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                if pad_mode == DatasetPadMode.NO_PADDING:
                    cu_seqlens = input_ids.offsets()
                    seq_lengths = cu_seqlens.diff()
                    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
                    logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
                    logits_rmpad = torch.cat([t for t in logits.unbind()])
                    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
                    log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
                    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
                    if calculate_entropy:
                        entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                        entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                        entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
                else:
                    raise NotImplementedError(f"pad_mode {pad_mode} not implemented")

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        # [FUNCTION DOC]
        # Runs one micro-batch forward under bf16 autocast and computes loss.
        # Behavior: moves the micro-batch to device, prepares inputs, calls the module with use_cache false, prepares outputs; calls loss_function with model output, data and DP group, or, when it is None (forward-only only), uses a dummy loss of 1.0 and empty metrics.
        # Outputs: (loss tensor, mapping with detached "model_output", float "loss", and "metrics").
        # Exceptions: AssertionError if loss_function is None and not forward_only.
        """Run forward pass, compute loss, and return outputs."""
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            raw_output = self.module(
                **model_inputs,
                use_cache=False,
            )

            model_output = self.prepare_model_outputs(
                output=raw_output, output_args=output_args, micro_batch=micro_batch
            )

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            # Detach before this lands in forward_backward_batch's output_lst; see detach_tree.
            output = {
                "model_output": detach_tree(model_output),
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output
