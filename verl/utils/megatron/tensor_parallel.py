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
# Tensor-parallel helpers for Megatron: default ModelParallelConfig / parallel-layer kwargs,
# TP parameter attribute queries, and vocab-parallel entropy, sum of squared probabilities and
# log-prob computations where logits are sharded along the vocab dimension across the TP group.

"""
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from torch.nn import init

if TYPE_CHECKING:
    from megatron.core import ModelParallelConfig


def update_kwargs_with_config(dictionary: dict, config: "ModelParallelConfig"):
    # [FUNCTION DOC]
    # HELPER: Sets dictionary["config"] = config and returns the same (mutated) dict.
    dictionary["config"] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    # [FUNCTION DOC]
    # HELPER: Returns a fresh dict of defaults: fp32 params_dtype, no CPU init, perform
    # initialization, no gradient-accumulation fusion, no sequence parallel.
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    # [FUNCTION DOC]
    # HELPER: Constructs a megatron ModelParallelConfig from the default kwargs.
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    # [FUNCTION DOC]
    # HELPER: Returns kwargs shared by column/row parallel linear layers: xavier-normal init_method,
    # stride 1, keep_master_weight_for_test False, and a default ModelParallelConfig.
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    # [FUNCTION DOC]
    # HELPER: Common parallel-linear kwargs whose config is a ModelParallelConfig built from the
    # defaults plus async_tensor_model_parallel_allreduce=False.
    # Exceptions: TypeError on Megatron versions whose ModelParallelConfig lacks that field.
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    # [FUNCTION DOC]
    # HELPER: Returns the common parallel-linear kwargs unchanged.
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    # [FUNCTION DOC]
    # HELPER: Returns kwargs with xavier-normal init_method and a default ModelParallelConfig.
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    # [FUNCTION DOC]
    # Behavior: whether the param has a truthy tensor_model_parallel attribute.
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_tensor_parallel_partition_dim(param):
    # [FUNCTION DOC]
    # Behavior: returns param.partition_dim. Exceptions: AssertionError if not a TP param.
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    # [FUNCTION DOC]
    # Behavior: returns param.partition_stride. Exceptions: AssertionError if not a TP param.
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):
    # [CLASS DOC]
    # HELPER: Autograd function computing softmax entropy over vocab-sharded logits with TP all-reduces.
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: global max over vocab via all-reduce MAX; exponentiates shifted logits; all-reduce SUM
        # of exp sums; softmax = exp / sum; local sum of softmax * logits then all-reduce SUM; entropy =
        # max + log(sum_exp) - sum(p * logits). Squeezes the trailing dim.
        # Inputs: ctx; vocab_parallel_logits - (..., vocab/tp).
        # Outputs: entropy of shape (...).
        # Side effects: three TP collectives; saves logits, softmax and the expectation for backward.
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            # [FUNCTION DOC]
            # HELPER: torch.compile'd (dynamic shapes) elementwise product summed over the last dim, keepdim.
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: gradient = -p * (logits - E_p[logits]) * grad_output, computed in place reusing the
        # saved softmax buffer; temporarily subtracts and then re-adds the expectation to the saved logits.
        # Inputs: ctx; grad_output (...). Outputs: gradient w.r.t. logits (..., vocab/tp).
        # Side effects: mutates saved tensors in place (logits restored up to float rounding).
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: applies _VocabParallelEntropy. Inputs: (total_nnz, vocab/tp). Outputs: (total_nnz,).
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_entropy_with_chunking(vocab_parallel_logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: allocates a zero tensor shaped like logits minus the last dim (default dtype), then for
    # each chunk of chunk_size positions along dim 1 applies _VocabParallelEntropy and writes the result.
    # Inputs: vocab_parallel_logits - (batch, seq, vocab/tp); chunk_size - positions per chunk.
    # Outputs: entropy (batch, seq).
    # KNOWN ISSUE: docstring claims 2-D (total_nnz, vocab/tp) input is supported, but the code slices
    # three dims and chunks along dim 1, so 2-D input fails.
    # KNOWN ISSUE: results are written into a preallocated tensor via in-place indexing, which breaks
    # the autograd link to the chunked entropy; output dtype is the default float dtype.
    """Memory-efficient entropy calculation using chunked processing when logits are sharded in tp ranks.
    Args:
        vocab_parallel_logits: (batch_size, seq_len, vocab_size // tp_size) or (total_nnz, vocab_size // tp_size)
        chunk_size: Number of sequence tokens to process at once. Defaults to 2048.
    Returns: (batch_size, seq_len)
    """

    output_shape = list(vocab_parallel_logits.shape[:-1])
    entropy = torch.zeros(output_shape, device=vocab_parallel_logits.device)

    for i in range(0, vocab_parallel_logits.shape[1], chunk_size):
        logits_chunk = vocab_parallel_logits[:, i : i + chunk_size, :]
        entropy_chunk = _VocabParallelEntropy.apply(logits_chunk)
        entropy[:, i : i + chunk_size] = entropy_chunk

    return entropy


def vocab_parallel_sum_pi_squared(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: numerically stable sum of squared softmax probabilities over sharded vocab: global
    # max (all-reduce MAX), exp of shifted logits (non in-place), all-reduce SUM of sum(exp) and of
    # sum(exp^2), returns sum(exp^2) / sum(exp)^2 squeezed.
    # Inputs: (..., vocab/tp) logits. Outputs: (...). Side effects: three TP collectives; input untouched.
    """Compute Σπ² (sum of squared probabilities) when logits are sharded across tp ranks.

    Used by ``optimal_token_baseline`` advantage estimators as the path-variance proxy:
    ``w_t = 1 - 2*π_t + Σπ²``.

    Args:
        vocab_parallel_logits: (..., vocab_size // tp_size)

    Returns: (...,)

    Implementation is non-destructive (does not mutate ``vocab_parallel_logits``) so it
    can be safely called before ``vocab_parallel_entropy`` / ``vocab_parallel_log_probs``
    which would otherwise consume the same tensor.
    """
    tp_group = mpu.get_tensor_model_parallel_group()

    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = vocab_parallel_logits - logits_max
    exp_shifted = shifted.exp()
    sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp, group=tp_group)
    sum_exp_squared = exp_shifted.pow(2).sum(dim=-1, keepdim=True)
    dist.all_reduce(sum_exp_squared, group=tp_group)
    return (sum_exp_squared / sum_exp.pow(2)).squeeze(dim=-1)


def vocab_parallel_log_probs_from_logits(logits, labels):
    # [FUNCTION DOC]
    # Behavior: returns the negated Megatron vocab_parallel_cross_entropy of logits vs labels, i.e. the
    # per-token label log-probability. Inputs: logits (..., vocab/tp); labels (...). Outputs: (...).
    # Side effects: Megatron's implementation may modify logits in place.
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    # [FUNCTION DOC]
    # Behavior: unpads input_ids with the attention mask, rolls the flattened ids by -1 to get
    # next-token labels, computes vocab-parallel log-probs against logits_rmpad, re-pads to
    # (batch, seqlen), and returns the slice [-response_length-1 : -1] along the sequence (log-probs of
    # response tokens, predicted from the previous position).
    # Inputs: input_ids, attention_mask - (batch, seqlen); logits_rmpad - (total_nnz, vocab/tp);
    # response_length - int. Outputs: (batch, response_length).
    # Exceptions: ImportError if flash_attn is missing.
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel
    region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output
