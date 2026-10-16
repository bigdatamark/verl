# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Multi-GPU (CUDA or NPU) test that activation offloading (enable_activation_offloading) does not change numerics
# for FSDP1 and FSDP2: logits after a training step with offloading must exactly equal those without it, starting
# from the same checkpoint.
import os
import shutil
import tempfile

import pytest
import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.fsdp_utils import MixedPrecisionPolicy, apply_fsdp2, get_fsdp_wrap_policy


def create_random_input_ids(batch_size, seq_len, vocab_size):
    # [FUNCTION DOC]
    # HELPER: Generates random token ids of shape (batch_size, seq_len) on the current device, a random left-padded
    # attention mask (up to 10% left padding, 50-70% valid tokens) and position ids from it, then removes padding
    # (flash-attn unpad on CUDA, verl's unpad on NPU) and returns both as packed tensors of shape (1, total_valid).
    # Inputs: batch_size, seq_len, vocab_size - ints. Outputs: (input_ids, position_ids).
    # KNOWN ISSUE: on a device that is neither cuda nor npu, unpad_input is never imported and a NameError results.
    if get_device_name() == "cuda":
        from flash_attn.bert_padding import unpad_input
    elif get_device_name() == "npu":
        from verl.utils.attention_utils import unpad_input
    from verl.utils.model import compute_position_id_with_mask, create_random_mask

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=get_device_name())

    attention_mask = create_random_mask(
        input_ids, max_ratio_of_left_padding=0.1, min_ratio_of_valid_token=0.5, max_ratio_of_valid_token=0.7
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    input_ids = unpad_input(input_ids.unsqueeze(-1), attention_mask)[0].transpose(0, 1)
    position_ids = unpad_input(position_ids.unsqueeze(-1), attention_mask)[0].transpose(0, 1)
    return input_ids, position_ids


def _fsdp_activation_offloading_test(rank, world_size, rendezvous_file, strategy="fsdp"):
    # [FUNCTION DOC]
    # HELPER: Per-rank worker spawned by the test. Sets the device, initializes the process group through a file
    # rendezvous, builds a 1-D "dp" device mesh, creates a 4-layer Qwen2 model from config (bf16, flash attention 2)
    # on device, and wraps it with FSDP1 (full shard, bf16 params, fp32 reduce/buffers, verl wrap policy) when
    # strategy is "fsdp", otherwise applies FSDP2 with a bf16/fp32 mixed precision policy. Creates AdamW (lr 1e-4),
    # StepLR (gamma 0.9) and an FSDPCheckpointManager using the Qwen2.5-0.5B tokenizer from ~/models.
    # Procedure: step 1 on random batch 1 (loss = mean of logits) then save checkpoint to a fresh temp dir; step 2 on
    # batch 2 and record no-grad logits on batch 2 as the reference; enable activation offloading, reload the
    # checkpoint (model, optimizer, scheduler), repeat step 2 on batch 2 and record logits again; assert exact
    # equality (zero tolerance), print a success message, remove the temp dir, barrier and destroy the process group.
    # Inputs: rank, world_size - ints; rendezvous_file - path; strategy - "fsdp" or "fsdp2".
    # Outputs: none. Exceptions: AssertionError on logit mismatch.
    # Side effects: distributed init/teardown, GPU memory, temp checkpoint files, stdout print.
    get_torch_device().set_device(rank)
    torch.distributed.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

    model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct")
    config = Qwen2Config(num_hidden_layers=4)

    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        model = model.to(device=get_device_name())

    # Wrap model with FSDP
    mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

    if strategy == "fsdp":
        model = FSDP(
            model,
            use_orig_params=False,
            device_id=get_torch_device().current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            auto_wrap_policy=get_fsdp_wrap_policy(module=model),
        )
    else:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        fsdp_kwargs = {
            "mesh": device_mesh,
            "mp_policy": mp_policy,
        }
        apply_fsdp2(model, fsdp_kwargs, {})

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Create checkpoint manager
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    checkpoint_manager = FSDPCheckpointManager(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, tokenizer=tokenizer
    )

    # Generate sample input
    batch_size = 2
    seq_len = 32
    vocab_size = 32000
    # First input for initial update
    input_ids1, position_ids1 = create_random_input_ids(batch_size, seq_len, vocab_size)

    # Second input for verification
    input_ids2, position_ids2 = create_random_input_ids(batch_size, seq_len, vocab_size)

    # Step 1: Initial update and save checkpoint
    outputs1 = model(input_ids=input_ids1, position_ids=position_ids1)
    loss1 = outputs1.logits.mean()
    loss1.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Save checkpoint after first update
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "checkpoint")
    checkpoint_manager.save_checkpoint(local_path=checkpoint_path, hdfs_path=None, global_step=0)

    # Step 2: Second update and forward pass
    outputs2 = model(input_ids=input_ids2, position_ids=position_ids2)
    loss2 = outputs2.logits.mean()
    loss2.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Record logits after second update
    with torch.no_grad():
        logits_without_offloading = model(input_ids=input_ids2, position_ids=position_ids2).logits

    # Step 3: wrap module with activation offloading and load checkpoint
    enable_activation_offloading(model, strategy=strategy)
    checkpoint_manager.load_checkpoint(checkpoint_path)

    # Step 4: Repeat the second update with same input
    outputs3 = model(input_ids=input_ids2, position_ids=position_ids2)
    loss3 = outputs3.logits.mean()
    loss3.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    # Record logits after loaded checkpoint and update
    with torch.no_grad():
        logits_with_offloading = model(input_ids=input_ids2, position_ids=position_ids2).logits

    # Step 4: Verify outputs match
    torch.testing.assert_close(logits_without_offloading, logits_with_offloading, atol=0.0, rtol=0.0)
    print(f"Activaiton offloading for {strategy} test passed on {world_size} GPUs!")

    # Cleanup
    shutil.rmtree(temp_dir)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4))
@pytest.mark.parametrize("strategy", ("fsdp", "fsdp2"))
def test_activation_offloading(world_size, strategy, tmp_path):
    # [FUNCTION DOC]
    # Behavior: parametrized over world size 2 and 4 and strategy fsdp/fsdp2; spawns world_size processes running
    # the worker with a rendezvous file under tmp_path and waits for them.
    # Inputs: world_size, strategy, tmp_path. Outputs: none. Side effects: process spawning.
    rendezvous_file = str(tmp_path / "rdzv_file")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)

    mp.spawn(
        fn=_fsdp_activation_offloading_test,
        args=(world_size, rendezvous_file, strategy),
        nprocs=world_size,
        join=True,
    )
