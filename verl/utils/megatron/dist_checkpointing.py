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
# Thin wrappers around Megatron-Core dist_checkpointing for saving/loading sharded state dicts
# with the torch_dist backend, parallelized across the data-parallel (+context-parallel) group.

import megatron.core
import torch
from megatron.core import dist_checkpointing, mpu
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistLoadShardedStrategy,
    TorchDistSaveShardedStrategy,
)
from packaging import version


def save_dist_checkpointing(
    sharded_state_dict,
    ckpt_path,
    async_save=False,
    content_metadata=None,
):
    # [FUNCTION DOC]
    # Behavior: builds a TorchDistSaveShardedStrategy (backend "torch_dist", version 1) wrapped in
    # FullyParallelSaveStrategyWrapper over the DP-with-CP group; calls dist_checkpointing.save with
    # access-integrity validation enabled, async_sharded_save=async_save, and content_metadata only
    # when provided AND megatron.core >= 0.14.0 (silently dropped otherwise).
    # Inputs: sharded_state_dict - Megatron sharded state dict; ckpt_path - target directory;
    # async_save - bool; content_metadata - optional metadata mapping.
    # Outputs: whatever dist_checkpointing.save returns (an async request when async, else None).
    # Exceptions: Megatron save/validation errors propagate.
    # Side effects: collective file writes to ckpt_path.
    validate_sharding_integrity = True
    # Get checkpointing strategies
    save_strategy = TorchDistSaveShardedStrategy(backend="torch_dist", version=1)
    save_strategy = FullyParallelSaveStrategyWrapper(
        save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/optimizer/distrib_optimizer.py#L1109-L1123
    mcore_ge_014 = version.parse(megatron.core.__version__) >= version.parse("0.14.0")
    # Save model sharded state dicts
    save_kwargs = dict(
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=validate_sharding_integrity,
    )
    if content_metadata is not None:
        if mcore_ge_014:
            save_kwargs["content_metadata"] = content_metadata
    return dist_checkpointing.save(sharded_state_dict, ckpt_path, **save_kwargs)


def load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    # [FUNCTION DOC]
    # Behavior: builds TorchDistLoadShardedStrategy wrapped in FullyParallelLoadStrategyWrapper over
    # the DP-with-CP group; best-effort registers torch.optim.AdamW and TransformerEngine FusedAdam as
    # torch.load safe globals (any failure ignored); loads via dist_checkpointing.load.
    # Inputs: sharded_state_dict - template sharded state dict; ckpt_dir - checkpoint directory.
    # Outputs: the loaded state dict.
    # Exceptions: load errors propagate. Side effects: mutates torch serialization safe-globals; file reads.
    # Get checkpointing strategies
    load_strategy = TorchDistLoadShardedStrategy()
    load_strategy = FullyParallelLoadStrategyWrapper(
        load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
    )

    # Fix torch.load weights only error
    try:
        import transformer_engine as te

        torch.serialization.add_safe_globals([torch.optim.AdamW])
        torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])
    except Exception:
        pass

    # Load model sharded state dicts
    state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)

    return state_dict
