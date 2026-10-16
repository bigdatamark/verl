#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
# Autograd wrapper around the fused Triton "linear + cross-entropy" kernels in the sibling
# kernels module. Computes per-token log-probabilities of the label and the softmax entropy of
# (hidden @ weight^T) / temperature without materializing the full logits tensor; optionally
# the vocab dimension is sharded across a process group (tensor-parallel vocab).
# Exposes linear_cross_entropy, an alias for LinearCrossEntropy.apply.

import typing

import torch
import torch.distributed as dist


class LinearCrossEntropy(torch.autograd.Function):
    # [CLASS DOC]
    # torch.autograd.Function whose forward returns (logprobs, entropy) and whose backward returns
    # gradients for hidden and weight only (labels/temperature/reduction/group get None).
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: typing.Optional[float] = 1.0,
        reduction: typing.Optional[str] = "none",
        dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    ) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: asserts temperature is a float and reduction is a str; inside an NVTX range, lazily
        # imports the kernels module, maps reduction (lower-cased) to the kernel enum number, flattens
        # hidden to 2-D (tokens, hidden_size) and labels to 1-D if needed, then calls
        # efficient_entropy_forward(hidden, weight, labels, REDUCTION, temperature, group).
        # Inputs: ctx - autograd context; hidden - (..., hidden_size); weight - (vocab[/shard], hidden_size);
        # labels - token ids, any shape flattening to the token count; temperature - float divisor of
        # logits; reduction - "none"/"sum"/"mean" style string understood by the kernel module;
        # dist_process_group - optional group across which vocab is split.
        # Outputs: tuple (logprobs, entropy) as returned by the kernel (flattened / reduced per REDUCTION,
        # NOT reshaped back to the original hidden shape).
        # Exceptions: AssertionError on wrong temperature/reduction types; kernel errors propagate.
        # Side effects: saves hidden, weight, labels and kernel intermediates (max, accumulate, entropy_b)
        # for backward; stores original hidden shape, reduction enum, group, temperature and
        # should_return_fp32_grad=False on ctx.
        # KNOWN ISSUE: the default temperature 1.0 passes, but an int temperature (e.g. 1) fails the assert.
        """_summary_

        Args:
            ctx (_type_): _description_
            hidden (torch.Tensor): (batch_size, num_tokens, hidden_size) -> (batch_size * num_tokens, hidden_size)
            weight (torch.Tensor): (vocab_size, hidden_size)
            labels (torch.Tensor): (batch_size, num_tokens) -> (batch_size * num_tokens, )
            temperature (typing.Optional[float], optional): _description_. Defaults to 1.0.
            reduction (typing.Optional[str], optional): _description_. Defaults to "none".
            dist_process_group (typing.Optional[dist.ProcessGroup], optional): _description_. Defaults to None.

        Returns:
            typing.List[torch.Tensor]: _description_
        """

        assert isinstance(temperature, float), f"temperature must be a float, but got {type(temperature)}"
        assert isinstance(reduction, str), f"reduction must be a str, but got {type(reduction)}"
        with torch.cuda.nvtx.range("LinearCrossEntropy-forward"):
            from . import kernels

            REDUCTION = kernels.get_entropy_reduction_enum_number(reduction.lower())

            original_hidden_shape = hidden.shape
            if len(hidden.shape) != 2:
                hidden = hidden.view(-1, hidden.shape[-1])  # (batch_size * num_tokens, hidden_size)
            if len(labels.shape) != 1:
                labels = labels.view(-1)

            logprobs, entropy, _maximum, _accumulate, _entropy_b = kernels.efficient_entropy_forward(
                hidden, weight, labels, REDUCTION, temperature, dist_process_group
            )

            ctx.save_for_backward(hidden, weight, labels, _maximum, _accumulate, _entropy_b)
            ctx.original_hidden_shape = original_hidden_shape
            ctx.REDUCTION = REDUCTION
            ctx.dist_process_group = dist_process_group
            ctx.should_return_fp32_grad = False
            ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, dlogprobs: torch.Tensor, dentropy: torch.Tensor) -> list[torch.Tensor]:
        # [FUNCTION DOC]
        # Behavior: retrieves saved tensors and settings, calls efficient_entropy_backward with the
        # upstream gradients of logprobs and entropy, and reshapes d_hidden to the original hidden shape.
        # Inputs: ctx; dlogprobs, dentropy - upstream gradients matching forward outputs.
        # Outputs: tuple (d_hidden, d_weight, None, None, None, None).
        # Exceptions: kernel errors propagate.
        # Side effects: NVTX range only.
        from . import kernels

        with torch.cuda.nvtx.range("LinearCrossEntropy-backward"):
            (hidden, weight, labels, _maximum, _accumulate, _entropy_b) = ctx.saved_tensors
            REDUCTION = ctx.REDUCTION
            dist_process_group = ctx.dist_process_group
            should_return_fp32_grad = ctx.should_return_fp32_grad
            temperature = ctx.temperature

            d_hidden, d_weight = kernels.efficient_entropy_backward(
                dlogprobs,
                dentropy,
                hidden,
                weight,
                labels,
                _maximum,
                _accumulate,
                _entropy_b,
                REDUCTION,
                should_return_fp32_grad,
                temperature,
                dist_process_group,
            )
            d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return (d_hidden, d_weight, None, None, None, None)


linear_cross_entropy = LinearCrossEntropy.apply
