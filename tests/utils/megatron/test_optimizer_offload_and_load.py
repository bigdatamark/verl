# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
# Single-GPU test for verl's offload_megatron_optimizer / load_megatron_optimizer on a Megatron precision-aware
# distributed Adam optimizer, checking all optimizer state moves to CPU and back to CUDA. Requires CUDA and
# Megatron-Core.

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as McoreDDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from verl.utils.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer

# ==== Helper functions ==== #


SEQUENCE_LENGTH = 64


def init_model():
    # [FUNCTION DOC]
    # HELPER: Builds a 2-layer, hidden 512, 4-head bf16 GPTModel (local layer spec, vocab 128, max seq length
    # SEQUENCE_LENGTH) on CUDA, wraps it in Megatron DDP with distributed optimizer, and returns a one-element list of
    # model chunks. Side effects: GPU allocation; requires initialized parallel state.
    """Initialize a small GPT model for the test, wrapped in Megatron DDP."""

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=SEQUENCE_LENGTH,
    ).cuda()

    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model_chunk = McoreDDP(transformer_config, ddp_config, gpt_model)
    return [model_chunk]


def init_precision_aware_optimizer(model):
    # [FUNCTION DOC]
    # HELPER: Builds a Megatron Adam optimizer (lr/min_lr 1e-6, clip 1.0, no weight decay, distributed, bf16,
    # precision-aware) for the given model chunks. Inputs: model - list of chunks. Outputs: the optimizer.
    """Initialize a precision-aware optimizer for the model."""

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-6,
        min_lr=1e-6,
        clip_grad=1.0,
        weight_decay=0.0,
        use_distributed_optimizer=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_precision_aware_optimizer=True,
    )
    return get_megatron_optimizer(optimizer_config, model)


def precision_aware_optimizer_is_on_device(optimizer, device):
    # [FUNCTION DOC]
    # HELPER: Treats the optimizer as a list of sub-optimizers (chained or single). First asserts, for each,
    # that every entry in shard_fp32_from_float16_groups is None and every per-parameter state has a non-None
    # "master_param" (the layout verl's offload code assumes). Then returns False if any tensor value in any
    # per-parameter state is not on device, else True.
    # Inputs: optimizer; device - torch device to compare against. Outputs: bool.
    # Exceptions: AssertionError if the layout assumption is violated.
    """Check that all optimizer state tracked by a given precision-aware
    optimizer is on the specified device."""

    opts = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]

    # Verify that "master_param" is populated for each parameter and not
    # shard_fp32_from_float16_groups (this is an assumption made by VeRL's
    # optimizer offloading code).
    for opt in opts:
        for group in opt.shard_fp32_from_float16_groups:
            for param in group:
                assert param is None
        param_to_param_opt_state = opt.optimizer.state
        for param_state in param_to_param_opt_state.values():
            assert param_state.get("master_param", None) is not None

    # Check device placement of optimizer state.
    for opt in opts:
        param_to_param_opt_state = opt.optimizer.state
        for param_state in param_to_param_opt_state.values():
            for v in param_state.values():
                if isinstance(v, torch.Tensor) and v.device != device:
                    return False

    return True


# ==== Tests ==== #


def test_precision_aware_optimizer_offload_and_load(tmp_path):
    # [FUNCTION DOC]
    # Behavior: sets CUDA device 0, initializes a single-rank process group (gloo for CPU, NCCL for CUDA) via a file
    # rendezvous in tmp_path, initializes Megatron model parallel and seeds with 123. Builds model and optimizer,
    # zeroes grad buffers and grads (kept as zeros, not None) and steps to materialize optimizer state, asserting
    # the step succeeded. Offloads and asserts all state is on CPU; loads and asserts all state is on cuda:0.
    # Always destroys model parallel state and the process group afterwards.
    # Inputs: tmp_path fixture. Outputs: none. Side effects: distributed init/teardown, GPU memory, rendezvous file.
    # Initialize torch distributed and Megatron parallel state.
    rendezvous_file = tmp_path / "rdzv_optimizer"

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"file://{rendezvous_file}",
        rank=0,
        world_size=1,
    )
    mpu.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)

    try:
        # Initialize model and optimizer.
        model_chunks = init_model()
        optimizer = init_precision_aware_optimizer(model_chunks)

        # Fully initialize the optimizer state by calling optimizer.step() on
        # dummy gradients set to 0.
        for model_chunk in model_chunks:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad(set_to_none=False)
        update_successful, _, _ = optimizer.step()
        assert update_successful

        # Offload optimizer state.
        offload_megatron_optimizer(optimizer)

        # Make sure everything has been offloaded.
        assert precision_aware_optimizer_is_on_device(
            optimizer,
            torch.device("cpu"),
        )

        # Load optimizer state.
        load_megatron_optimizer(optimizer)

        # Make sure everything has been loaded.
        assert precision_aware_optimizer_is_on_device(
            optimizer,
            torch.device("cuda:0"),
        )
    finally:
        # Tear down MPU state and torch.distributed.
        mpu.destroy_model_parallel()
        dist.destroy_process_group()
