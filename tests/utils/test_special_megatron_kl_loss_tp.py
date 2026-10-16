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
# Standalone torchrun script (requires WORLD_SIZE > 1, CUDA, Megatron-core)
# checking that the Megatron vocab-parallel top-k forward-KL distillation loss
# matches the FSDP full-vocab reference: per-token losses, overlap metrics
# (overlap_token_advantage, overlap_count) and student-logit gradients, for both
# "thd" (packed) and "bshd" (padded) data formats. Tensor-parallel size equals the
# world size; the vocab is split evenly into per-rank shards. MAX_TEST_CASES env
# var (default 4, parsed as int) selects how many shape cases run.


import gc
import os

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk as compute_forward_kl_topk_ref
from verl.trainer.distillation.megatron.losses import compute_forward_kl_topk as compute_forward_kl_topk_vp
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.workers.config import DistillationConfig, DistillationLossConfig

MAX_TEST_CASES = int(os.environ.get("MAX_TEST_CASES", 4))


class TestVocabParallelKLDivergence:
    # [CLASS DOC]
    # Driver class (plain class, not pytest-collected) holding distributed/TP state,
    # per-case hyperparameters and the correctness check.
    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes the global process group and Megatron model parallel with
        # TP size equal to world size; stores local rank, world size, CUDA device and the
        # TP process group; prints rank info.
        # Side effects: global distributed and Megatron parallel-state initialization.
        local_rank, rank, world_size = initialize_global_process_group()
        mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)

        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        self.group = mpu.get_tensor_model_parallel_group()
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

    def initialize(self, test_case_idx: int):
        # [FUNCTION DOC]
        # HELPER: Records the current test case index.
        self.test_case_idx = test_case_idx

    def shutdown(self):
        # [FUNCTION DOC]
        # HELPER: Tears down the global process group.
        destroy_global_process_group()

    def cleanup(self):
        # [FUNCTION DOC]
        # HELPER: Empties CUDA cache, resets peak memory stats, collects garbage, syncs.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.synchronize()

    def generate_hyper(self):
        # [FUNCTION DOC]
        # HELPER: Sets batch size, sequence length, vocab size, top-k and the student
        # log-prob min clamp (None in case 3) for cases 0..3, then the per-rank shard size
        # as vocab divided by world size.
        # Exceptions: ValueError for an unknown case; AssertionError if vocab is not
        # divisible by world size.
        if self.test_case_idx == 0:
            self.batch_size = 2
            self.seq_len = 4
            self.vocab_size = 32
            self.topk = 6
            self.clamp = -6.0
        elif self.test_case_idx == 1:
            self.batch_size = 4
            self.seq_len = 8
            self.vocab_size = 64
            self.topk = 8
            self.clamp = -8.0
        elif self.test_case_idx == 2:
            self.batch_size = 2
            self.seq_len = 4
            self.vocab_size = 128
            self.topk = 10
            self.clamp = -4.0
        elif self.test_case_idx == 3:
            self.batch_size = 1
            self.seq_len = 16
            self.vocab_size = 64
            self.topk = 4
            self.clamp = None
        else:
            raise ValueError(f"Invalid test case index: {self.test_case_idx}")

        assert self.vocab_size % self.world_size == 0, "vocab_size must be divisible by world_size"
        self.shard_size = self.vocab_size // self.world_size

    def generate_forward_inputs(self):
        # [FUNCTION DOC]
        # HELPER: Builds random full-vocab student logits [B, S, V] (scaled 0.7) and
        # teacher top-k log-probs/ids from a random teacher log-softmax (scaled 0.9), then
        # injects edge cases: the first top-k id is forced to global index shard_size
        # (local index 0 on rank 1, colliding with the out-of-shard placeholder index) and
        # that student logit set to 3.0; the last two top-k ids are forced to 1 and 2 (in
        # rank 0's shard, out-of-shard for rank 1); the student logit at the second top-k
        # id is set to -50 so it hits the clamp. Teacher log-probs are kept consistent
        # with the forced ids.
        # Outputs: (student_logits, teacher_topk_logps, teacher_topk_ids).
        # KNOWN ISSUE: forcing ids can create duplicate ids within a token's top-k (and
        # the top-k log-probs are no longer sorted); the edge-case comments assume
        # exactly 2 ranks.
        B, S, V = self.batch_size, self.seq_len, self.vocab_size
        topk = self.topk
        shard_size = self.shard_size

        full_student_logits = torch.randn(B, S, V, device=self.device) * 0.7
        teacher_full_logits = torch.randn(B, S, V, device=self.device) * 0.9
        teacher_full_logps = F.log_softmax(teacher_full_logits, dim=-1)
        teacher_topk_logps, teacher_topk_ids = torch.topk(teacher_full_logps, k=topk, dim=-1)

        # Edge case 1: Force index 0 collision on rank 1 (on rank 1, global index shard_size maps to local index 0)
        # 1. When a teacher top-k index is not in the local shard, it's remapped to
        # local index 0 (as a dummy placeholder, with its prob set to 0)
        # 2. But local index 0 might also be a legitimate teacher
        # top-k index (e.g., on rank 1, global index shard_size maps to local index 0)
        teacher_topk_ids[..., 0] = shard_size
        teacher_topk_logps[..., 0] = teacher_full_logps[..., shard_size]

        # Edge case 2: Make the colliding token active (high probability, not clamped)
        full_student_logits[..., shard_size] = 3.0

        # Edge case 3: Force out-of-shard entries for rank 1 (indices 1 and 2 are in rank 0's shard)
        teacher_topk_ids[..., -1] = 1
        teacher_topk_logps[..., -1] = teacher_full_logps[..., 1]
        teacher_topk_ids[..., -2] = 2
        teacher_topk_logps[..., -2] = teacher_full_logps[..., 2]

        # Edge case 4: Force some student probs to be clamped (very low logits)
        full_student_logits.scatter_(
            dim=-1, index=teacher_topk_ids[..., 1:2], src=torch.full((B, S, 1), -50.0, device=self.device)
        )

        return full_student_logits, teacher_topk_logps, teacher_topk_ids

    def to_nested(self, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Converts a dense [B, ...] tensor into a jagged nested tensor with one
        # component per batch row.
        return torch.nested.as_nested_tensor([tensor[i] for i in range(tensor.shape[0])], layout=torch.jagged)

    def pad_for_bshd_preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Right-pads the sequence dim (dim 1) with zeros up to a multiple of the
        # TP world size, mirroring the BSHD preprocessing with context parallel size 1.
        # Inputs: tensor [B, S, ...]. Outputs: padded tensor or the input if aligned.
        # Exceptions: AssertionError if context parallel size is not 1.
        """Mirror preprocess_bshd_engine's CP=1 sequence padding for student logits."""
        assert mpu.get_context_parallel_world_size() == 1, "This TP KL test does not initialize context parallelism"
        align_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (align_size - tensor.shape[1] % align_size) % align_size
        if pad_size == 0:
            return tensor

        padded = torch.zeros(
            tensor.shape[0],
            tensor.shape[1] + pad_size,
            *tensor.shape[2:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, : tensor.shape[1]] = tensor
        return padded

    def verify_correctness(self, iterations: int = 5):
        # [FUNCTION DOC]
        # Behavior: builds a DistillationConfig with the case's clamp, computes this
        # rank's vocab slice, and for each iteration: seeds on rank 0 (seed from case and
        # iteration), generates inputs and broadcasts them from rank 0, converts teacher
        # tensors to nested; runs the VP loss on the local slice of the THD-flattened
        # student logits and backprops the summed loss; runs the reference on full logits
        # and takes this rank's slice of its grad; asserts losses, overlap advantage
        # (1e-4 tolerance), overlap count, and gradients match. Then repeats the VP loss
        # on padded BSHD logits, backprops only the unpadded positions, and asserts the
        # unpadded loss, metrics and grads (reshaped to THD) match the reference.
        # Inputs: iterations - default 5. Side effects: collectives, prints PASS on rank 0.
        # Exceptions: AssertionError on mismatch.
        # KNOWN ISSUE: shard offset uses the local rank rather than the TP rank, which is
        # only correct for single-node runs.
        self.cleanup()
        self.generate_hyper()

        cfg = DistillationConfig(distillation_loss=DistillationLossConfig(log_prob_min_clamp=self.clamp))

        shard_start = self.local_rank * self.shard_size
        shard_end = shard_start + self.shard_size

        for i in range(iterations):
            if self.local_rank == 0:
                torch.manual_seed(42 + self.test_case_idx * 100 + i)

            # Generate inputs and broadcast to all ranks
            full_student_logits, teacher_topk_logps, teacher_topk_ids = self.generate_forward_inputs()
            dist.broadcast(full_student_logits, src=0, group=self.group)
            dist.broadcast(teacher_topk_logps, src=0, group=self.group)
            dist.broadcast(teacher_topk_ids, src=0, group=self.group)
            full_student_logits_bshd = full_student_logits
            full_student_logits_thd = full_student_logits_bshd.reshape(1, -1, self.vocab_size)
            teacher_topk_logps = self.to_nested(teacher_topk_logps)
            teacher_topk_ids = self.to_nested(teacher_topk_ids)

            # VP implementation on sharded logits
            vp_logits = full_student_logits_thd[..., shard_start:shard_end].contiguous().detach().requires_grad_(True)
            loss_out = compute_forward_kl_topk_vp(
                student_logits=vp_logits,
                teacher_topk_log_probs=teacher_topk_logps,
                teacher_topk_ids=teacher_topk_ids,
                config=cfg,
                data_format="thd",
            )
            vp_loss = loss_out["distillation_losses"]
            vp_loss.sum().backward()
            grad_vp = vp_logits.grad.detach().clone()

            # Reference implementation on full logits
            full_ref = full_student_logits_thd.detach().clone().requires_grad_(True)
            fsdp_loss_out = compute_forward_kl_topk_ref(
                student_logits=full_ref,
                teacher_topk_log_probs=teacher_topk_logps,
                teacher_topk_ids=teacher_topk_ids,
                config=cfg,
                data_format="thd",
            )
            ref_loss = fsdp_loss_out["distillation_losses"]
            ref_loss.sum().backward()
            grad_ref_shard = full_ref.grad[..., shard_start:shard_end].detach().clone()

            # Compare losses and overlap logging metrics
            torch.testing.assert_close(vp_loss, ref_loss, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(
                loss_out["overlap_token_advantage"],
                fsdp_loss_out["overlap_token_advantage"],
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(loss_out["overlap_count"], fsdp_loss_out["overlap_count"])

            # Compare gradients
            torch.testing.assert_close(grad_vp, grad_ref_shard, atol=1e-4, rtol=1e-4)

            # BSHD path should preserve the top-k dense dimension when preprocessing teacher tensors.
            full_student_logits_bshd_padded = self.pad_for_bshd_preprocess(full_student_logits_bshd)
            vp_logits_bshd = (
                full_student_logits_bshd_padded[..., shard_start:shard_end].contiguous().detach().requires_grad_(True)
            )
            loss_out_bshd = compute_forward_kl_topk_vp(
                student_logits=vp_logits_bshd,
                teacher_topk_log_probs=teacher_topk_logps,
                teacher_topk_ids=teacher_topk_ids,
                config=cfg,
                data_format="bshd",
            )
            vp_loss_bshd = loss_out_bshd["distillation_losses"]
            vp_loss_bshd[:, : self.seq_len].sum().backward()
            grad_vp_bshd = vp_logits_bshd.grad.detach().clone()

            torch.testing.assert_close(vp_loss_bshd[:, : self.seq_len].reshape(1, -1), ref_loss, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(
                loss_out_bshd["overlap_token_advantage"][:, : self.seq_len].reshape(1, -1),
                fsdp_loss_out["overlap_token_advantage"],
                atol=1e-4,
                rtol=1e-4,
            )
            torch.testing.assert_close(
                loss_out_bshd["overlap_count"][:, : self.seq_len].reshape(1, -1), fsdp_loss_out["overlap_count"]
            )
            torch.testing.assert_close(
                grad_vp_bshd[:, : self.seq_len].reshape(1, -1, self.shard_size),
                grad_ref_shard,
                atol=1e-4,
                rtol=1e-4,
            )

        if self.local_rank == 0:
            print(f"[PASS] VP KL divergence correctness verified for test case {self.test_case_idx}")


if __name__ == "__main__":
    assert int(os.environ.get("WORLD_SIZE", 1)) > 1, (
        "[ERROR]: This test is designed to run in distributed mode with torchrun. "
        "Please use torchrun to execute this script."
    )
    torch.manual_seed(42 + int(os.environ.get("RANK", 0)))

    test = TestVocabParallelKLDivergence()
    try:
        for test_case_idx in range(MAX_TEST_CASES):
            if test.local_rank == 0:
                print(f"[INFO] Running test case {test_case_idx}")
            test.initialize(test_case_idx)
            test.verify_correctness()
    finally:
        test.shutdown()
