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
# Helper utilities for the FSDP engine: NPU patching, device mesh creation, sharding-strategy selection and MoE weight unfusing for rollout weight sync.
import logging
import os

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name, is_npu_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_npu_fsdp_patches(model_config=None):
    # [FUNCTION DOC]
    # HELPER: Applies NPU-specific model patches for the FSDP backend.
    # Behavior: no-op unless an NPU is available; also returns early when model_config is given and has use_liger truthy; otherwise imports and calls apply_npu_patches.
    # Inputs: model_config - optional mapping-like config supporting get.
    # Outputs: none.
    # Side effects: monkey-patches transformers model code globally.
    """Apply NPU patches for FSDP backend if NPU is available."""
    if is_npu_available:
        if model_config is not None and model_config.get("use_liger", False):
            return
        from verl.models.transformers.npu_patch import apply_npu_patches

        apply_npu_patches()


def create_device_mesh(world_size, fsdp_size):
    # [FUNCTION DOC]
    # HELPER: Builds the device mesh for FSDP/HSDP.
    # Behavior: if fsdp_size is negative or at least world_size, creates a 1D mesh of size world_size named "fsdp"; otherwise a 2D mesh of shape (world_size // fsdp_size, fsdp_size) named ("ddp", "fsdp").
    # Inputs: world_size - total processes; fsdp_size - shard group size.
    # Outputs: the DeviceMesh on the current device type.
    # Side effects: initializes process-group mesh state.
    """
    Create a device mesh for distributed training based on the world size and FSDP size.

    Args:
        world_size (int): Total number of processes in the distributed training setup.
        fsdp_size (int): Size of the Fully Sharded Data Parallel (FSDP) group.

    Returns:
        torch.distributed.device_mesh.DeviceMesh: The initialized device mesh.
    """
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    # [FUNCTION DOC]
    # HELPER: Maps a device mesh to an FSDP1 ShardingStrategy.
    # Behavior: with zero3_enable chooses FULL_SHARD (1D) / HYBRID_SHARD (2D); without it SHARD_GRAD_OP (1D) / _HYBRID_SHARD_ZERO2 (2D).
    # Inputs: device_mesh - mesh whose ndim is inspected; zero3_enable - bool, default True (undocumented in docstring).
    # Outputs: the selected ShardingStrategy.
    # Exceptions: NotImplementedError when mesh ndim is not 1 or 2.
    # Side effects: none.
    """
    Determine the appropriate sharding strategy based on the number of dimensions of the device mesh.

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): The device mesh used for distributed training.

    Returns:
        torch.distributed.fsdp.ShardingStrategy: The sharding strategy to be used with FSDP.

    Raises:
        NotImplementedError: If the number of dimensions of the device mesh is neither 1 nor 2.
    """

    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def unfuse_moe_params(weights, model_type: str | None = None):
    # [FUNCTION DOC]
    # HELPER: Generator converting packed 3D MoE expert parameters into per-expert 2D checkpoint keys.
    # Behavior: for each (name, tensor) pair: if model_type is "gpt_oss" and the name contains ".mlp.experts." with a 3D tensor, yield unchanged; if the name ends with ".mlp.experts.gate_up_proj" and is 3D, split along dim 1 into gate and up halves and yield "<base>.<i>.gate_proj.weight" and "<base>.<i>.up_proj.weight" per expert i (contiguous slices); if it ends with ".mlp.experts.down_proj" and is 3D, yield "<base>.<i>.down_proj.weight" per expert; otherwise yield the pair unchanged.
    # Inputs: weights - iterable of (name, tensor); model_type - optional HF model type string.
    # Outputs: lazily yields (name, tensor) pairs.
    # Side effects: none (allocates contiguous copies of slices).
    """Expand Transformers 5 packed MoE expert tensors to vLLM checkpoint keys.

    Transformers 5 stores Qwen-style MoE experts as packed 3D parameters:
    ``mlp.experts.gate_up_proj`` with shape
    ``[num_experts, 2 * intermediate_size, hidden_size]`` and
    ``mlp.experts.down_proj`` with shape
    ``[num_experts, hidden_size, intermediate_size]``. vLLM's Qwen MoE reload
    path still accepts the original per-expert checkpoint keys during live
    weight sync, so stream those keys without materializing a full dict.
    """
    for name, tensor in weights:
        # GPT-OSS checkpoint weights use packed 3D expert tensors directly.
        # Splitting them into per-expert 2D tensors makes vLLM's GPT-OSS loader
        # index a 2D tensor as 3D during TP slicing.
        if model_type == "gpt_oss" and ".mlp.experts." in name and tensor.dim() == 3:
            yield name, tensor
            continue

        if name.endswith(".mlp.experts.gate_up_proj") and tensor.dim() == 3:
            gate, up = tensor.chunk(2, dim=1)
            base = name.removesuffix(".gate_up_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.gate_proj.weight", gate[expert_id].contiguous()
                yield f"{base}.{expert_id}.up_proj.weight", up[expert_id].contiguous()
            continue

        if name.endswith(".mlp.experts.down_proj") and tensor.dim() == 3:
            base = name.removesuffix(".down_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.down_proj.weight", tensor[expert_id].contiguous()
            continue

        yield name, tensor
