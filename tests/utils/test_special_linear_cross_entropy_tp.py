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
# Standalone torchrun script (not a pytest module; needs at least 2 CUDA ranks with
# NCCL) validating the fused linear_cross_entropy kernel in vocab tensor-parallel
# mode against a pure-PyTorch reference autograd function. For each of up to
# MAX_TEST_CASES model shapes it optionally checks the TP reference against a
# single-device reference, then compares kernel vs reference forward (logprobs,
# entropy) and backward (hidden and weight grads) values, latency and peak memory.
# Hidden states, labels and upstream gradients are broadcast from rank 0; each rank
# holds its own vocab shard of the weight. Env knobs: MAX_TEST_CASES (default 5),
# VERIFY_TORCH_SELF, LOW_MEMORY, LOW_MEMORY_DIV_FACTOR (default 16).
# KNOWN ISSUE: env-var knobs are read as raw strings when set, so a set
# MAX_TEST_CASES breaks range() and the "<= 5" assert, LOW_MEMORY_DIV_FACTOR breaks
# the division, and any non-empty VERIFY_TORCH_SELF/LOW_MEMORY (even "0") is truthy.

import os

import torch
import torch.distributed as dist

try:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
except ImportError:
    # FIXME: remove these manually included paths
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")))
finally:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

import verl.utils.torch_functional as verl_F

compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

MAX_TEST_CASES = os.environ.get("MAX_TEST_CASES", 5)
VERIFY_TORCH_SELF = os.environ.get("VERIFY_TORCH_SELF", False)
LOW_MEMORY = os.environ.get("LOW_MEMORY", False)
LOW_MEMORY_DIV_FACTOR = os.environ.get("LOW_MEMORY_DIV_FACTOR", 16)


def run_torch_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, temperature: float, reduction="none"
) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Single-device reference for logprobs and entropy of a linear head.
    # Behavior: flattens hidden to 2-D tokens-by-hidden and labels to 1-D; computes
    # float32 logits as hidden times weight, using weight as-is if its first dim equals
    # the hidden size, else its transpose; divides by temperature; entropy is
    # logsumexp of logits minus the softmax-weighted sum of logits; logprobs is the
    # negated cross-entropy against labels with the given reduction.
    # Inputs: hidden [.., hidden]; weight [vocab, hidden] or [hidden, vocab]; labels;
    # temperature; reduction - cross-entropy reduction, default "none".
    # Outputs: tuple (logprobs, entropy), float32, per token when reduction is none.
    # KNOWN ISSUE: annotated as returning a list but returns a tuple; the
    # orientation heuristic is ambiguous when vocab size equals hidden size.
    # [num_tokens, vocab_size]
    if len(hidden.shape) > 2:
        hidden = hidden.view(-1, hidden.shape[-1])  # [num_tokens, hidden_size]
    if len(labels.shape) > 1:
        labels = labels.view(-1)
    logits = torch.matmul(
        hidden.to(torch.float32),
        weight.to(torch.float32) if weight.size(0) == hidden.size(1) else weight.T.to(torch.float32),
    )
    logits /= temperature
    pd = torch.nn.functional.softmax(logits, dim=-1)  # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1)  # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1)  # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)  # [num_tokens]
    logprobs = torch.neg(logprobs)
    return logprobs, entropy


class TorchEntropyTP(torch.autograd.Function):
    # [CLASS DOC]
    # HELPER: Reference autograd function for vocab-parallel logprobs/entropy; each rank
    # holds a [local_vocab, hidden] weight shard. Correctness reference only, not
    # efficient (materializes full logits on every rank).
    """
    it is used for testing the correctness of the kernel
    it is not efficient and is not recommended to use in practice
    """

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        dist_process_group: torch.distributed.ProcessGroup,
    ):
        # [FUNCTION DOC]
        # Behavior: flattens inputs, computes local float32 logits divided by temperature,
        # all-gathers them along the vocab dim into a full [tokens, vocab*world] buffer
        # (rank i's shard in column block i), then computes entropy and negated
        # cross-entropy logprobs over the full logits exactly like run_torch_entropy.
        # Inputs: ctx; hidden; weight shard; labels (global vocab ids); temperature;
        # dist_process_group. Outputs: (logprobs, entropy) per token.
        # Side effects: saves hidden, weight, labels, full logits, and the softmax-weighted
        # logit sum for backward, plus the group, temperature and original hidden shape on
        # ctx; performs an all-gather collective.
        # weight has shape [vocab_size, hidden_size], hidden has shape [num_tokens, hidden_size]
        ctx.original_hidden_shape = hidden.shape
        if len(hidden.shape) > 2:
            hidden = hidden.view(-1, hidden.shape[-1])  # [num_tokens, hidden_size]
        if len(labels.shape) > 1:
            labels = labels.view(-1)

        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32).T)  # [num_tokens, vocab_size]
        logits /= temperature
        whole_logits = torch.empty(
            (logits.shape[0], logits.shape[1] * dist.get_world_size(dist_process_group)),
            dtype=logits.dtype,
            device=logits.device,
        )
        whole_logits_ref = [
            whole_logits[:, i * logits.shape[1] : (i + 1) * logits.shape[1]]
            for i in range(dist.get_world_size(dist_process_group))
        ]
        dist.all_gather(whole_logits_ref, logits, group=dist_process_group)

        pd = torch.nn.functional.softmax(whole_logits, dim=-1)
        entropy_a = torch.logsumexp(whole_logits, dim=-1)  # [num_tokens]
        entropy_b = torch.sum(pd * whole_logits, dim=-1)  # [num_tokens]
        entropy = entropy_a - entropy_b

        logprobs = torch.nn.functional.cross_entropy(whole_logits, labels, reduction="none")
        logprobs = torch.neg(logprobs)

        ctx.save_for_backward(hidden, weight, labels, whole_logits, entropy_b)
        ctx.dist_process_group = dist_process_group
        ctx.temperature = temperature
        return logprobs, entropy

    @staticmethod
    def backward(ctx, g_logprobs: torch.Tensor, g_entropy: torch.Tensor):
        # [FUNCTION DOC]
        # Behavior: recomputes softmax of full logits (max-subtracted); entropy grad wrt
        # logits is g_entropy times negative probs times (logits minus saved weighted sum);
        # logprob grad is negated g_logprobs times (probs minus one-hot of labels); sums
        # both and divides by temperature; slices this rank's vocab column block; returns
        # grad of hidden as local-dlogits times weight (reshaped to original hidden shape)
        # and grad of weight as local-dlogits transposed times hidden, both float32.
        # Inputs: ctx; g_logprobs, g_entropy - upstream grads per token.
        # Outputs: (d_hidden, d_weight, None, None, None).
        # Note: d_hidden is only the partial contribution of this shard; callers must
        # all-reduce (sum) it across ranks.
        hidden, weight, labels, whole_logits, entropy_b = ctx.saved_tensors
        dist_process_group = ctx.dist_process_group
        temperature = ctx.temperature
        batch_size, hidden_size = hidden.shape
        vocab_size, hidden_size = weight.shape
        rank = dist.get_rank(dist_process_group)

        # Compute softmax probabilities
        maximum, _ = torch.max(whole_logits, dim=-1, keepdim=True)
        exp_logits = torch.exp(whole_logits - maximum)
        accumulate = exp_logits.sum(dim=-1, keepdim=True)
        pd = exp_logits / accumulate

        # Gradient for entropy
        # entropy = entropy_a - entropy_b
        # entropy_a = log(sum(exp(logits)))
        # entropy_b = sum(pd * logits)
        # d_entropy_a/d_logits = pd
        # d_entropy_b/d_logits = pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = d_entropy_a - d_entropy_b
        # d_entropy/d_logits = pd - pd * (logits - b.unsqueeze(1) + 1)
        # d_entropy/d_logits = -pd * (logits - b.unsqueeze(1))
        d_logits_entropy = g_entropy.unsqueeze(1) * (-pd * (whole_logits - entropy_b.unsqueeze(1)))

        # Gradient for logprobs
        # logprobs = -cross_entropy = -log(pd[labels])
        # d_logprobs/d_logits = (pd - one_hot(labels))
        one_hot = torch.zeros_like(whole_logits)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)
        g_logprobs = torch.neg(g_logprobs)
        d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - one_hot)
        # NOTE: This will lead to wrong result
        # d_logits_logprobs = g_logprobs.unsqueeze(1) * (pd - 1) * one_hot

        # Combine gradients
        d_logits = d_logits_entropy + d_logits_logprobs
        d_logits /= temperature

        # Get local slice of gradients
        local_d_logits = d_logits[:, rank * vocab_size : (rank + 1) * vocab_size]

        # Compute gradients for hidden and weight
        d_hidden = torch.matmul(local_d_logits, weight.to(torch.float32))
        d_weight = torch.matmul(local_d_logits.T, hidden.to(torch.float32))
        d_hidden = d_hidden.view(ctx.original_hidden_shape)

        return d_hidden, d_weight, None, None, None


run_torch_entropy_tp = TorchEntropyTP.apply


class TestLinearCrossEntropy_TensorParallel:
    # [CLASS DOC]
    # Driver class (plain class, not collected by pytest) holding distributed state
    # and per-test-case shapes, with methods for reference self-check, correctness vs
    # kernel, and peak-memory reporting.
    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes the NCCL default process group, records the world group,
        # rank and world size, binds the CUDA device to the rank, prints rank info.
        # Side effects: global distributed init and CUDA device selection.
        dist.init_process_group(backend="nccl")
        self.group = dist.group.WORLD

        self.local_rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

    def initialize(self, test_case_idx: int, temperature: float = 1.5):
        # [FUNCTION DOC]
        # HELPER: Sets the current test case index and softmax temperature (default 1.5).
        self.test_case_idx = test_case_idx
        self.temperature = temperature

    def shutdown(self):
        # [FUNCTION DOC]
        # HELPER: Destroys the default process group.
        dist.destroy_process_group()

    def cleanup(self):
        # [FUNCTION DOC]
        # HELPER: Empties CUDA cache, resets peak memory stats, runs garbage collection and
        # synchronizes the device.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        # [FUNCTION DOC]
        # HELPER: Sets dtype bfloat16 and batch size, token count, hidden size and local
        # vocab size for test case 0..4 (shapes modeled on Qwen2/Qwen2.5/DeepSeek-like
        # heads; case 4 uses 8192 tokens). When LOW_MEMORY is truthy divides vocab size by
        # LOW_MEMORY_DIV_FACTOR.
        # Exceptions: ValueError for an index outside 0..4; AssertionError if
        # MAX_TEST_CASES exceeds 5.
        global LOW_MEMORY, LOW_MEMORY_DIV_FACTOR, MAX_TEST_CASES

        self.dtype = torch.bfloat16
        if self.test_case_idx == 0:
            self.batch_size = 1
            self.num_tokens = 1937
            self.hidden_size = 3584
            self.vocab_size = 152064
        elif self.test_case_idx == 1:
            self.batch_size = 1
            self.num_tokens = 2169
            self.hidden_size = 896
            self.vocab_size = 151936
        elif self.test_case_idx == 2:
            self.batch_size = 1
            self.num_tokens = 1530
            self.hidden_size = 2048
            self.vocab_size = 32256
        elif self.test_case_idx == 3:
            self.batch_size = 1
            self.num_tokens = 1388
            self.hidden_size = 4096
            self.vocab_size = 102400
        elif self.test_case_idx == 4:
            self.batch_size = 1
            self.num_tokens = 8192
            self.hidden_size = 4096
            self.vocab_size = 102400
        else:
            raise ValueError(f"Invalid test case index: {self.test_case_idx}")
        if LOW_MEMORY:
            self.vocab_size = int(self.vocab_size / LOW_MEMORY_DIV_FACTOR)
        assert MAX_TEST_CASES <= 5, "MAX_TEST_CASES should be less than or equal to 5."

    def generate_forward_inputs(self):
        # [FUNCTION DOC]
        # HELPER: Creates random inputs on CUDA: hidden [batch, tokens, hidden] and weight
        # [vocab, hidden] uniform in [-0.5, 0.5] with requires_grad, and integer labels
        # [batch, tokens] in [0, local vocab).
        # Outputs: (hidden, weight, labels).
        hidden = (
            torch.empty((self.batch_size, self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
            .uniform_(-0.5, 0.5)
            .requires_grad_()
        )
        weight = (
            torch.empty((self.vocab_size, self.hidden_size), dtype=self.dtype, device="cuda")
            .uniform_(-0.5, 0.5)
            .requires_grad_()
        )
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.num_tokens), device="cuda")
        return hidden, weight, labels

    def generate_backward_inputs(self):
        # [FUNCTION DOC]
        # HELPER: Creates random upstream grads per token: g_entropy uniform in
        # [-0.5, 0.5], g_logprobs uniform in [-1, 1].
        # Outputs: (g_entropy, g_logprobs).
        g_entropy = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-0.5, 0.5)
        g_logprobs = torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda").uniform_(-1, 1)
        return g_entropy, g_logprobs

    def verify_torch_itself(self, iterations: int = 5):
        # [FUNCTION DOC]
        # Behavior: for the given number of iterations, broadcasts hidden/labels from rank
        # 0, all-gathers weight shards into a full weight, and checks TorchEntropyTP
        # against run_torch_entropy on the full weight: forward outputs at 1e-4 tolerance;
        # after broadcasting upstream grads, hidden grad (TP grad all-reduced) and this
        # rank's slice of the full weight grad at atol 1e-2/rtol 1e-4.
        # Inputs: iterations - default 5. Side effects: collectives; prints PASS on rank 0.
        # Exceptions: AssertionError from assert_close on mismatch.
        self.cleanup()
        self.generate_hyper()

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            # forward pass
            # Create a tensor to hold the gathered weights from all ranks
            # weight has shape [vocab_size, hidden_size]
            # We want to gather along the first dimension to get [vocab_size * world_size, hidden_size]

            # Create a single contiguous tensor to hold all gathered weights
            whole_weight = torch.empty(
                (self.vocab_size * self.world_size, self.hidden_size), dtype=weight.dtype, device=weight.device
            )

            # Create views into the tensor for each rank's portion
            whole_weight_views = [
                whole_weight[i * self.vocab_size : (i + 1) * self.vocab_size] for i in range(self.world_size)
            ]

            # Perform all_gather operation using the views
            dist.all_gather(whole_weight_views, weight, group=self.group)

            # Set requires_grad for autograd
            whole_weight.requires_grad_()

            (single_logprobs, single_entropy) = run_torch_entropy(hidden, whole_weight, labels, self.temperature)

            (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)

            torch.testing.assert_close(single_logprobs, tp_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(single_entropy, tp_entropy, atol=1e-4, rtol=1e-4)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            (single_d_hidden, single_d_weight) = torch.autograd.grad(
                (single_entropy, single_logprobs), (hidden, whole_weight), (g_entropy, g_logprobs), retain_graph=False
            )

            (tp_d_hidden, tp_d_weight) = torch.autograd.grad(
                (tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(tp_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(tp_d_hidden, single_d_hidden, atol=1e-2, rtol=1e-4)
            # Extract the corresponding slice from single_d_weight for comparison
            # tp_d_weight has shape [vocab_size, hidden_size]
            # single_d_weight has shape [vocab_size * world_size, hidden_size]
            torch.testing.assert_close(
                tp_d_weight,
                single_d_weight[self.local_rank * self.vocab_size : (self.local_rank + 1) * self.vocab_size],
                atol=1e-2,
                rtol=1e-4,
            )

            # atol=1e-3, rtol=1e-4)
        if self.local_rank == 0:
            print("[PASS] torch TP correctness is verified")

    def check_torch_storage(self):
        # [FUNCTION DOC]
        # Behavior: runs one forward and backward of the TP reference with peak memory
        # stats reset before each phase, all-reduces the hidden grad, and prints forward
        # and backward peak allocated MB on rank 0.
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (tp_logprobs, tp_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_tp_hidden, d_tp_weight) = torch.autograd.grad(
            (tp_entropy, tp_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )
        torch.cuda.synchronize()
        backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_tp_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Torch Forward pass peak memory: {forward_max_memory:.2f} MB")
            print(f"[INFO]: Torch Backward pass peak memory: {backward_max_memory:.2f} MB")

    def verify_kernel_correctness(self, iterations: int = 5):
        # [FUNCTION DOC]
        # Behavior: for each iteration, times (CUDA events) the TP reference and the
        # kernel forward (reduction none, with the group), asserts logprobs/entropy close
        # (atol 1e-1, rtol 1e-2), then times both backward passes with broadcast upstream
        # grads, all-reduces both hidden grads, and asserts hidden and weight grads close
        # (atol 2e-2, rtol 4e-2). Drops the first (warm-up) latency of each list and on
        # rank 0 prints PASS and the four average latencies in ms.
        # Inputs: iterations - default 5.
        # Exceptions: AssertionError on mismatch; ZeroDivisionError if iterations is 1.
        self.cleanup()
        self.generate_hyper()

        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_inputs()

            # NOTE: we need to manually synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy_tp(hidden, weight, labels, self.temperature, self.group)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
                hidden, weight, labels, self.temperature, "none", self.group
            )
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-1, rtol=1e-2)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-1, rtol=1e-2)

            # backward pass
            g_entropy, g_logprobs = self.generate_backward_inputs()
            # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
            dist.broadcast(g_entropy, src=0, group=self.group)
            dist.broadcast(g_logprobs, src=0, group=self.group)

            start_event.record()
            (torch_d_hidden, torch_d_weight) = torch.autograd.grad(
                (torch_entropy, torch_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            start_event.record()
            (kernel_d_hidden, kernel_d_weight) = torch.autograd.grad(
                (kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))
            # NOTE: all-reduce on hidden is conducted outside the kernel
            dist.all_reduce(kernel_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            torch.testing.assert_close(torch_d_hidden, kernel_d_hidden, atol=2e-2, rtol=4e-2)
            torch.testing.assert_close(torch_d_weight, kernel_d_weight, atol=2e-2, rtol=4e-2)

        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]

        if self.local_rank == 0:
            print("\n[PASS]: Verified kernel forward & backward correctness.")

            print(
                f"[INFO]: Forward pass: Torch implementation average time: "
                f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Backward pass: torch implementation average time: "
                f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Forward pass: Kernel implementation average time: "
                f"{sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms"
            )
            print(
                f"[INFO]: Backward pass: kernel implementation average time: "
                f"{sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms"
            )

    def check_kernel_storage(self):
        # [FUNCTION DOC]
        # Behavior: same as check_torch_storage but for linear_cross_entropy: reports
        # kernel forward and backward peak memory in MB on rank 0.
        self.cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_inputs()

        # NOTE: we need to manually synchronize hidden and labels among Process Group
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(
            hidden, weight, labels, self.temperature, "none", self.group
        )
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        g_entropy, g_logprobs = self.generate_backward_inputs()
        # NOTE: we need to manually synchronize g_entropy and g_logprobs among Process Group
        dist.broadcast(g_entropy, src=0, group=self.group)
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad(
            (kernel_entropy, kernel_logprobs), (hidden, weight), (g_entropy, g_logprobs), retain_graph=False
        )
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        # NOTE: all-reduce on hidden is conducted outside the kernel
        dist.all_reduce(d_kernel_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")
            print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")


if __name__ == "__main__":
    # TP command: torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/kernels/test_linear_cross_entropy_tp.py

    # Check if running with torchrun (distributed mode)
    assert int(os.environ["WORLD_SIZE"]) > 1, (
        "[ERROR]: This test is designed to run in distributed mode with torchrun. Please use torchrun to "
        "execute this script."
    )
    torch.manual_seed(233376 + int(os.environ.get("RANK", 0)))

    # set_backward_method(BackwardEnum._Total_Fuse_MN)
    # set_backward_method(BackwardEnum._Split_Dlogits_N)

    test = TestLinearCrossEntropy_TensorParallel()
    for test_case_idx in range(MAX_TEST_CASES):
        print(f"[INFO] Running test case {test_case_idx}")
        test.initialize(test_case_idx)
        if VERIFY_TORCH_SELF:
            test.verify_torch_itself()
        test.check_torch_storage()
        test.verify_kernel_correctness()
        test.check_kernel_storage()

    test.shutdown()
