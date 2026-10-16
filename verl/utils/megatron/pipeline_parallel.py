# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
# Helpers for Megatron pipeline parallelism: precomputing per-micro-batch activation shapes for
# remove-padding inputs and building (virtual-)pipeline batch iterators.

import torch
from megatron.core import parallel_state as mpu

from .sequence_parallel import pad_to_sequence_parallel


def compute_transformers_input_shapes(batches, meta_info):
    # [FUNCTION DOC]
    # Behavior: for each micro-batch, unpads input_ids by attention_mask with flash_attn unpad_input
    # to get total non-pad tokens; if sequence parallel, pads that count to a multiple of the TP size
    # and records shape (padded_tokens / TP size, 1, hidden_size), else (tokens, 1, hidden_size).
    # Inputs: batches - iterable of mappings with "input_ids" and "attention_mask" (batch, seq);
    # meta_info - mapping with "sequence_parallel" (bool) and "hidden_size".
    # Outputs: list of torch.Size, one per micro-batch.
    # Exceptions: ImportError if flash_attn missing. Side effects: none (computes on device tensors).
    from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron

    # pre-compute input shapes for each micro-batch at each pp stage
    input_shapes = []
    for model_inputs in batches:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
        if meta_info["sequence_parallel"]:
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            # compute shapes for model_inputs
            input_shapes.append(
                torch.Size(
                    [
                        input_ids_rmpad.shape[0] // mpu.get_tensor_model_parallel_world_size(),
                        1,
                        meta_info["hidden_size"],
                    ]
                )
            )
        else:
            # compute shapes for model_inputs
            input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info["hidden_size"]]))
    return input_shapes


def make_batch_generator(batches, vpp_size):
    # [FUNCTION DOC]
    # Behavior: if vpp_size > 1 returns a list of vpp_size independent iterators over the same batches
    # sequence; otherwise a single iterator.
    # Inputs: batches - re-iterable sequence; vpp_size - int. Outputs: iterator or list of iterators.
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator
