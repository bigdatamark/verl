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
# Helpers for the NeMo Automodel engine: rank/size queries on Automodel device meshes (dims named dp, dp_cp, cp, tp, pp), distributed-config and mesh construction, model building via NeMoAutoModelForCausalLM, and CPU/GPU offload of models and optimizer state.

"""Utility functions for the Automodel engine integration."""

import torch
import torch.distributed

from verl.utils.device import get_device_id, get_torch_device


def get_dp_rank(device_mesh, include_cp=False):
    # [FUNCTION DOC]
    # HELPER: Returns the data-parallel local rank: 0 for no mesh; the "dp_cp" rank when include_cp and a cp dim larger than 1 exists; otherwise the "dp" rank.
    """Get data-parallel rank from device mesh."""
    if device_mesh is None:
        return 0
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh.get_local_rank("dp_cp")
    return device_mesh.get_local_rank("dp")


def get_tp_rank(device_mesh):
    # [FUNCTION DOC]
    # HELPER: Returns the "tp" local rank, or 0 when there is no mesh, no tp dim, or tp size is 1.
    """Get tensor-parallel rank from device mesh."""
    if device_mesh is None or "tp" not in device_mesh.mesh_dim_names or device_mesh["tp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("tp")


def get_pp_rank(device_mesh):
    # [FUNCTION DOC]
    # HELPER: Returns the "pp" local rank, or 0 when there is no mesh, no pp dim, or pp size is 1.
    """Get pipeline-parallel rank from device mesh."""
    if device_mesh is None or "pp" not in device_mesh.mesh_dim_names or device_mesh["pp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("pp")


def get_dp_group_size(device_mesh, include_cp=False):
    # [FUNCTION DOC]
    # HELPER: Returns data-parallel group size: global world size without a mesh; "dp_cp" size when include_cp and cp is larger than 1; "dp" size if present; else global world size.
    """Get data-parallel group size from device mesh."""
    if device_mesh is None:
        return torch.distributed.get_world_size()
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh["dp_cp"].size()
    if "dp" in device_mesh.mesh_dim_names:
        return device_mesh["dp"].size()
    return torch.distributed.get_world_size()


def maybe_fully_shard_optimizer(model, optimizer, distributed_config):
    # [FUNCTION DOC]
    # HELPER: For MegatronFSDP configs with more than one process, calls megatron_fsdp's fully_shard_optimizer on model/optimizer; otherwise no-op. Side effects: rewires optimizer state sharding.
    """Call fully_shard_optimizer for MegatronFSDP strategy."""
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
        from megatron_fsdp.fully_shard import fully_shard_optimizer

        fully_shard_optimizer(model, optimizer)


def build_distributed_config_from_engine_config(engine_config, world_size):
    # [FUNCTION DOC]
    # Builds the Automodel distributed config and device/MoE meshes.
    # Behavior: strategy "fsdp2" - FSDP2Config with a MixedPrecisionPolicy from mp_param/reduce/output dtypes (cast_forward_inputs true), sequence_parallel, activation_checkpointing, defer_fsdp_grad_sync; "megatron_fsdp" - MegatronFSDPConfig; "ddp" - DDPConfig (both with activation_checkpointing). Then calls Automodel create_device_mesh with tp/pp/cp/ep/dp_replicate sizes and world_size.
    # Inputs: engine_config - AutomodelEngineConfig; world_size - int.
    # Outputs: (distributed_config, device_mesh, moe_mesh).
    # Exceptions: ValueError on unknown strategy.
    # Side effects: creates process groups.
    """Build v5 distributed config, device_mesh, and moe_mesh from engine config.

    Args:
        engine_config: AutomodelEngineConfig instance.
        world_size: Total number of processes in the job.

    Returns:
        Tuple of (distributed_config, device_mesh, moe_mesh).
    """
    from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config, MegatronFSDPConfig
    from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

    strategy = engine_config.distributed_strategy

    if strategy == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        mp_policy = MixedPrecisionPolicy(
            param_dtype=PrecisionType.to_dtype(engine_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(engine_config.mp_reduce_dtype),
            output_dtype=PrecisionType.to_dtype(engine_config.mp_output_dtype),
            cast_forward_inputs=True,
        )

        distributed_config = FSDP2Config(
            sequence_parallel=engine_config.sequence_parallel,
            mp_policy=mp_policy,
            activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=engine_config.defer_fsdp_grad_sync,
        )

    elif strategy == "megatron_fsdp":
        distributed_config = MegatronFSDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    elif strategy == "ddp":
        distributed_config = DDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    else:
        raise ValueError(f"Unsupported distributed_strategy: {strategy}")

    device_mesh, moe_mesh = create_device_mesh(
        distributed_config,
        tp_size=engine_config.tp_size,
        pp_size=engine_config.pp_size,
        cp_size=engine_config.cp_size,
        ep_size=engine_config.ep_size,
        dp_replicate_size=engine_config.dp_replicate_size,
        world_size=world_size,
    )

    return distributed_config, device_mesh, moe_mesh


def build_automodel_model(model_config, engine_config, distributed_config, device_mesh, moe_mesh):
    # [FUNCTION DOC]
    # Builds the distributed model with NeMoAutoModelForCausalLM.from_pretrained.
    # Behavior: assembles kwargs: fp8_config when enable_fp8; compile_config when enable_compile; force_hf true when ep_size is at most 1 and the first HF architecture name contains "qwen" or "llama" (reads AutoConfig from the model path); a BackendConfig from backend_config unless force_hf; a MoEParallelizerConfig from moe_config (defaulting mp_policy from distributed_config) when ep_size above 1; attn_implementation and torch_dtype from model_dtype. Then calls from_pretrained with meshes, distributed config, activation checkpointing and trust_remote_code.
    # Inputs: model_config - HFModelConfig; engine_config; distributed_config; device_mesh; moe_mesh.
    # Outputs: the parallelized HF model.
    # Side effects: reads model config/weights from disk; allocates device memory.
    """Build a model using NeMoAutoModelForCausalLM.from_pretrained().

    Args:
        model_config: HFModelConfig with model path and settings.
        engine_config: AutomodelEngineConfig with distributed settings.
        distributed_config: FSDP2Config, MegatronFSDPConfig, or DDPConfig instance.
        device_mesh: Pre-created device mesh (or None for DDP).
        moe_mesh: Pre-created MoE mesh (or None).

    Returns:
        A HuggingFace model with Automodel's distributed infrastructure applied.
    """
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM

    kwargs = {}

    if engine_config.enable_fp8:
        from nemo_automodel.components.quantization.fp8 import FP8Config

        kwargs["fp8_config"] = FP8Config()

    if engine_config.enable_compile:
        from nemo_automodel.components.utils.compile_utils import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    # Qwen/Llama with ep_size<=1: use HF implementation.
    from transformers import AutoConfig

    _cfg = AutoConfig.from_pretrained(model_config.path, trust_remote_code=model_config.trust_remote_code)
    _arch = (getattr(_cfg, "architectures", None) or [""])[0].lower()
    if engine_config.ep_size <= 1 and ("qwen" in _arch or "llama" in _arch):
        kwargs["force_hf"] = True

    if engine_config.backend_config and not kwargs.get("force_hf", False):
        from nemo_automodel.components.models.common.utils import BackendConfig

        backend_kwargs = dict(engine_config.backend_config)
        kwargs["backend"] = BackendConfig(**backend_kwargs)

    # MoE config for MoEParallelizerConfig
    if engine_config.ep_size > 1:
        from nemo_automodel.components.moe.config import MoEParallelizerConfig

        moe_kwargs = dict(engine_config.moe_config) if engine_config.moe_config else {}
        if hasattr(distributed_config, "mp_policy"):
            moe_kwargs.setdefault("mp_policy", distributed_config.mp_policy)

        kwargs["moe_config"] = MoEParallelizerConfig(**moe_kwargs)

    kwargs["attn_implementation"] = engine_config.attn_implementation

    from verl.utils.torch_dtypes import PrecisionType

    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    model = NeMoAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.path,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        distributed_config=distributed_config,
        activation_checkpointing=engine_config.activation_checkpointing,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )

    return model


@torch.no_grad()
def offload_automodel_model_to_cpu(model, empty_cache=True):
    # [FUNCTION DOC]
    # HELPER: Offloads an FSDP2 model to CPU under no-grad.
    # Behavior: forces every FSDP param group's training state to IDLE (so reshard is allowed), reshards the model, moves it to CPU and optionally empties the device cache.
    # Inputs: model - FSDP2-wrapped module; empty_cache - bool.
    # Side effects: mutates FSDP internal state; device memory release.
    """Offload an FSDP2-wrapped model to CPU (reshard, move to CPU, optional cache clear)."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_automodel_model_to_gpu(model):
    # [FUNCTION DOC]
    # HELPER: Moves the model to the current device id (non-blocking) under no-grad.
    """Load model back to GPU."""
    device = get_device_id()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_automodel_optimizer(optimizer):
    # [FUNCTION DOC]
    # HELPER: Moves every tensor in the optimizer's per-parameter state to CPU (non-blocking); no-op when state is empty. Mutates optimizer.state in place.
    """Offload optimizer state to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_automodel_optimizer(optimizer, device_id):
    # [FUNCTION DOC]
    # HELPER: Moves every tensor in the optimizer's per-parameter state to device_id (non-blocking); no-op when state is empty. Mutates optimizer.state in place.
    """Load optimizer state back to GPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
