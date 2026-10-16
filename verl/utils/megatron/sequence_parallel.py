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
# Sequence-parallel helpers for Megatron: tagging parameters as sequence-parallel and padding
# flattened token tensors to a multiple of the tensor-parallel world size.

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu


def mark_parameter_as_sequence_parallel(parameter):
    # [FUNCTION DOC]
    # Behavior: sets attribute sequence_parallel=True on the parameter. Side effects: mutates it.
    parameter.sequence_parallel = True


def is_sequence_parallel_param(param):
    # [FUNCTION DOC]
    # Behavior: returns whether the param has a truthy sequence_parallel attribute. Outputs: bool-ish.
    return hasattr(param, "sequence_parallel") and param.sequence_parallel


def pad_to_sequence_parallel(unpad_tokens: torch.Tensor):
    # [FUNCTION DOC]
    # Behavior: computes pad so the first dim is a multiple of the TP world size and right-pads with
    # zeros along dim 0 (supports 1-D and 2-D tensors).
    # Inputs: unpad_tokens - (total_nnz,) or (total_nnz, d). Outputs: padded tensor (or the input
    # unchanged when no padding is needed).
    # Exceptions: NotImplementedError for ndim > 2 (only when padding is needed).
    # KNOWN ISSUE: the error message calls unpad_tokens.ndim() but ndim is an int, so a TypeError is
    # raised instead of the intended NotImplementedError.
    """pad the tokens such that the total length is a multiple of sp world size

    Args:
        unpad_tokens: (total_nnz, ...). Tokens after removing padding

    Returns:
        the padded tokens: (total_nnz + pad_size,...)

    """
    total_nnz = unpad_tokens.shape[0]
    sp_world_size = mpu.get_tensor_model_parallel_world_size()

    pad_size = 0 if total_nnz % sp_world_size == 0 else sp_world_size - total_nnz % sp_world_size

    if pad_size > 0:
        if unpad_tokens.ndim == 1:
            unpad_tokens = F.pad(unpad_tokens, (0, pad_size))
        elif unpad_tokens.ndim == 2:
            unpad_tokens = F.pad(unpad_tokens, (0, 0, 0, pad_size))
        else:
            raise NotImplementedError(f"Padding dim {unpad_tokens.ndim()} is not supported")

    return unpad_tokens
