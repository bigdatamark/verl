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
# Distributed (torchrun, >= 2 GPUs, ideally 8) tests for Ulysses sequence parallelism with verl's monkey-patched HF
# attention: forward/backward of SP-sliced packed inputs must match a non-SP forward. Configs whose head counts are
# incompatible with the SP size must raise AssertionError.

import contextlib
import copy
from dataclasses import dataclass

import pytest
import torch
import torch.distributed
import transformers
from packaging import version
from torch.distributed import init_device_mesh
from transformers import AutoModelForCausalLM, LlamaConfig, PretrainedConfig, Qwen2Config

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.protocol import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.ulysses import (
    FSDPUlyssesShardingManager,
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    set_ulysses_sequence_parallel_group,
    ulysses_pad_and_slice_inputs,
)

if get_device_name() == "cuda":
    from flash_attn.bert_padding import index_first_axis, rearrange, unpad_input
elif get_device_name() == "npu":
    from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input

# TODO(sgm): add more models for test
# we only need one scale for each model


@dataclass
class SequenceParallelConfig:
    # [CLASS DOC]
    # HELPER: Dataclass bundling a model config, the SP size, and whether the combination is expected to be valid.
    config: PretrainedConfig
    sp_size: int
    is_valid: bool


def test_configs():
    # [FUNCTION DOC]
    # HELPER: Returns the parametrization list: Llama (32 heads) sp 8, Qwen2 (28 heads / 4 KV, hidden 3584) sp 4 valid
    # and sp 8 invalid, Qwen2 (32 heads / 4 KV) sp 4 and 8, plus Apertus sp 8 when transformers >= 4.56.
    # KNOWN ISSUE: named test_* so pytest also collects it as a test function (returning a non-None value).
    configs = [
        SequenceParallelConfig(
            LlamaConfig(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=32), sp_size=8, is_valid=True
        ),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584),
            sp_size=4,
            is_valid=True,
        ),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=28, num_key_value_heads=4, hidden_size=3584),
            sp_size=8,
            is_valid=False,
        ),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4), sp_size=4, is_valid=True
        ),
        SequenceParallelConfig(
            Qwen2Config(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=4), sp_size=8, is_valid=True
        ),
    ]

    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        from transformers import ApertusConfig

        configs.append(
            SequenceParallelConfig(
                ApertusConfig(num_hidden_layers=2, num_attention_heads=32, num_key_value_heads=32, hidden_size=4096),
                sp_size=8,
                is_valid=True,
            )
        )

    return configs


def sync_model_parameters_global(layer):
    # [FUNCTION DOC]
    # HELPER: Broadcasts every parameter of the module from rank 0 so all ranks share weights.
    # Side effects: collective communication; in-place parameter overwrite.
    # synchronize weights
    for p in layer.parameters():
        torch.distributed.broadcast(tensor=p.data, src=0)


@pytest.mark.parametrize("test_config", test_configs())
def test_hf_casual_fwd_bwd(test_config):
    # [FUNCTION DOC]
    # Behavior: initializes the global process group if needed and runs _hf_casual_fwd_bwd with dp = world_size / sp,
    # expecting AssertionError for invalid configs. Process group is intentionally not destroyed.
    if not torch.distributed.is_initialized():
        initialize_global_process_group()

    context = contextlib.nullcontext() if test_config.is_valid else pytest.raises(AssertionError)
    with context:
        world_size = torch.distributed.get_world_size()
        _hf_casual_fwd_bwd(test_config.config, test_config.sp_size, world_size // test_config.sp_size)

    # TODO: seems not work, will cause `socketStartConnect: Connect to xxx failed : Software caused connection abort`
    # torch.distributed.destroy_process_group()


def _hf_casual_fwd(config, sp_size, dp_size):
    # [FUNCTION DOC]
    # HELPER: Forward-only variant: builds a dp x sp device mesh and Ulysses sharding manager, creates a bf16
    # flash-attention model, applies the monkey patch for sp_size, syncs weights, builds a random right-only-padded batch
    # (1 x 128), and inside the sharding manager preprocesses (all-gathers across SP group) the data, unpads, pads and
    # slices ids for SP, runs the model and gathers/unpads logits. Then disables the SP group and runs a full packed
    # forward, asserting mean logits match (rtol 1e-2). Currently unused by any test.
    # Exceptions: AssertionError if fewer than 2 devices.
    assert get_torch_device().device_count() >= 2, "need at least 2 gpus for test"

    ulysses_device_mesh = init_device_mesh(
        device_type=get_device_name(), mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp")
    )
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 1
    seqlen = 128
    # response_length = 127

    # patch before load
    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        apply_monkey_patch(model, sp_size)
        model = model.to(device=get_device_name())
        sync_model_parameters_global(model)

    # different rank will generate different input_ids following fsdp
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device=get_device_name())
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.8
    )
    position_ids = compute_position_id_with_mask(
        attention_mask
    )  # TODO(sgm): we can construct the position_ids_rmpad here

    model_inputs = {
        "input_ids": input_ids.to(get_device_name()),
        "attention_mask": attention_mask.to(get_device_name()),
        "position_ids": position_ids.int().to(get_device_name()),
    }

    model_inputs = DataProto.from_dict(model_inputs)

    # 1. perform ulysses forward
    with sharding_manager:
        model_inputs = sharding_manager.preprocess_data(model_inputs)
        input_ids = model_inputs.batch["input_ids"]
        attention_mask = model_inputs.batch["attention_mask"]
        position_ids = model_inputs.batch["position_ids"]
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        # slice input tensor for ulysses
        # input_ids are padded and sliced
        # postition_ids are only padded but not sliced
        input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
            input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
        )

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        logits_split_in_seq = model(
            input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded, use_cache=False
        ).logits  # (1, total_nnz/n, vocab_size)

        # all_gather output
        logits_full = gather_outputs_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

    # 2. perform normal forward
    set_ulysses_sequence_parallel_group(None)
    logits_rmpad_local = model(
        input_ids_rmpad, position_ids=position_ids_rmpad, use_cache=False
    ).logits  # (1, total_nnz, vocab_size)

    mean_local = logits_rmpad_local.mean()
    mean_full = logits_full.mean()
    torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=1e-5)


def _hf_casual_fwd_bwd(config, sp_size, dp_size):
    # [FUNCTION DOC]
    # HELPER: Same setup as _hf_casual_fwd; the non-SP reference forward uses deep copies of the inputs and model. After
    # computing both logit means it backpropagates each and asserts mean logits match (rtol 1e-2, atol 3e-5) and that the
    # first layer q_proj weight gradients of the SP model and the reference model match (rtol 1e-2, atol 1e-3).
    # Exceptions: AssertionError on device count, invalid head/SP combinations (from the patch), or mismatches.
    assert get_torch_device().device_count() >= 2, "need at least 2 gpus for test"

    ulysses_device_mesh = init_device_mesh(
        device_type=get_device_name(), mesh_shape=(dp_size, sp_size), mesh_dim_names=("dp", "sp")
    )
    sharding_manager = FSDPUlyssesShardingManager(ulysses_device_mesh)

    batch_size = 1
    seqlen = 128
    # response_length = 127

    # patch before load
    with torch.device(get_device_name()):
        model = AutoModelForCausalLM.from_config(
            config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        apply_monkey_patch(model, sp_size)
        model = model.to(device=get_device_name())
        sync_model_parameters_global(model)

    # different rank will generate different input_ids following fsdp
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seqlen), device=get_device_name())
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_left_padding=0, max_ratio_of_valid_token=0.9, min_ratio_of_valid_token=0.8
    )
    position_ids = compute_position_id_with_mask(
        attention_mask
    )  # TODO(sgm): we can construct the position_ids_rmpad here

    model_inputs = {
        "input_ids": input_ids.to(get_device_name()),
        "attention_mask": attention_mask.to(get_device_name()),
        "position_ids": position_ids.int().to(get_device_name()),
    }

    model_inputs = DataProto.from_dict(model_inputs)

    # 1. perform ulysses forward
    with sharding_manager:
        model_inputs = sharding_manager.preprocess_data(model_inputs)
        input_ids = model_inputs.batch["input_ids"]
        attention_mask = model_inputs.batch["attention_mask"]
        position_ids = model_inputs.batch["position_ids"]
        input_ids_rmpad, indices, *_ = unpad_input(
            input_ids.unsqueeze(-1), attention_mask
        )  # input_ids_rmpad (total_nnz, ...)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
        # unpad the position_ids to align the rotary
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        # slice input tensor for ulysses
        # input_ids are padded and sliced
        # postition_ids are only padded but not sliced
        input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
            input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
        )

        # input with input_ids_rmpad and postition_ids to enable flash attention varlen
        logits_split_in_seq = model(
            input_ids_rmpad_sliced, position_ids=position_ids_rmpad_padded, use_cache=False
        ).logits  # (1, total_nnz/n, vocab_size)

        # all_gather output
        logits_full = gather_outputs_and_unpad(logits_split_in_seq, gather_dim=1, unpad_dim=1, padding_size=pad_size)

    # 2. perform normal forward
    set_ulysses_sequence_parallel_group(None)
    input_ids_full = copy.deepcopy(input_ids_rmpad)
    position_ids_full = copy.deepcopy(position_ids_rmpad)
    model_no_sp = copy.deepcopy(model)
    logits_rmpad_local = model_no_sp(
        input_ids_full, position_ids=position_ids_full, use_cache=False
    ).logits  # (1, total_nnz, vocab_size)

    mean_local = logits_rmpad_local.mean()
    mean_full = logits_full.mean()

    mean_full.backward()
    mean_local.backward()

    # 3. check the gradients
    grad = model.model.layers[0].self_attn.q_proj.weight.grad
    grad_full = model_no_sp.model.layers[0].self_attn.q_proj.weight.grad
    torch.testing.assert_close(mean_local, mean_full, rtol=1e-2, atol=3e-5)
    # The check should be less strict because the gradient is not an averaged value.
    torch.testing.assert_close(grad, grad_full, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
