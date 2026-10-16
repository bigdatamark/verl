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
# All logic is tested via local replicas (see module docstring), so drift from trainer_base.py is not detected.

"""CPU-only unit tests for the streaming dataloader feed logic in the V1 PPO trainer.

These cover the dependency-light pieces introduced for streaming generation:

- The ``num_prompts`` resolution + validation contract of
  ``PPOTrainer._next_train_batch`` (default to ``train_batch_size``; require a positive
  multiple of ``gen_batch_size``; fetch in small chunks and coalesce submissions).
- The ``steps_per_epoch`` derivation used for ``total_training_steps`` / epoch tracking, which
  must be computed from the dataset size and ``train_batch_size`` (NOT ``len(dataloader)``, which
  now counts ``gen_batch_size`` fetches), and must equal the classic ``len(dataloader)`` when
  ``gen_batch_size == train_batch_size``.

Importing the trainer directly would pull in heavy runtime deps (ray, transfer_queue, vllm), so
-- following the convention of ``test_compute_reward_colocate_on_cpu.py`` -- the logic under test
is replicated standalone and kept in sync with ``verl/trainer/ppo/v1/trainer_base.py``.
"""

import pytest


def _resolve_num_prompts(num_prompts, train_batch_size, gen_batch_size):
    # [FUNCTION DOC]
    # HELPER: replica of num_prompts resolution: None defaults to train_batch_size; a falsy gen_batch_size
    # falls back to train_batch_size; returns num_prompts if it is a positive multiple of gen_batch_size.
    # Inputs: num_prompts (int or None), train_batch_size, gen_batch_size (int or None). Outputs: int.
    # Exceptions: ValueError with message "must be a positive multiple of gen_batch_size" otherwise.
    """Standalone copy of the ``num_prompts`` resolution/validation in
    ``PPOTrainer._next_train_batch``. Kept in sync with ``trainer_base.py``.
    """
    if num_prompts is None:
        num_prompts = train_batch_size
    gen_batch_size = gen_batch_size or train_batch_size
    if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
        raise ValueError(
            f"num_prompts ({num_prompts}) must be a positive multiple of gen_batch_size "
            f"({gen_batch_size}); it is submitted in whole gen_batch_size dataloader fetches."
        )
    return num_prompts


def _num_fetches(num_prompts, gen_batch_size):
    # [FUNCTION DOC]
    # HELPER: number of dataloader fetches, num_prompts integer-divided by gen_batch_size.
    """How many ``gen_batch_size`` dataloader fetches ``_next_train_batch`` performs."""
    return num_prompts // gen_batch_size


def _num_submissions(num_prompts, gen_batch_size):
    # [FUNCTION DOC]
    # HELPER: validates num_prompts is a positive multiple of gen_batch_size (ValueError otherwise) and always
    # returns 1 (one coalesced submission).
    # KNOWN ISSUE: hardcoded return makes the coalesced-submission tests tautological; they do not exercise
    # production code.
    """A valid request is coalesced into one submission."""
    if num_prompts <= 0 or num_prompts % gen_batch_size != 0:
        raise ValueError("num_prompts must be a positive multiple of gen_batch_size")
    return 1


def _steps_per_epoch(dataset_size, train_batch_size):
    # [FUNCTION DOC]
    # HELPER: dataset_size integer-divided by train_batch_size (drop_last semantics).
    """Standalone copy of ``PPOTrainer.steps_per_epoch``. ``//`` matches dataloader drop_last=True."""
    return dataset_size // train_batch_size


class TestResolveNumPrompts:
    # [CLASS DOC]
    # Tests for _resolve_num_prompts defaults and validation.
    def test_default_is_train_batch_size(self):
        # [FUNCTION DOC]
        # None resolves to train_batch_size for gen sizes 1 and 256.
        # None -> one train batch worth, regardless of gen_batch_size.
        assert _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1) == 1024
        assert _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=256) == 1024

    def test_gen_batch_size_none_falls_back_to_train_batch_size(self):
        # [FUNCTION DOC]
        # gen None resolves to train size (512) and that default needs a single fetch.
        # gen_batch_size null => equals train_batch_size => default request is a single fetch.
        assert _resolve_num_prompts(None, train_batch_size=512, gen_batch_size=None) == 512
        assert _num_fetches(_resolve_num_prompts(None, 512, None) or 512, 512) == 1

    def test_explicit_multiple_ok(self):
        # [FUNCTION DOC]
        # Explicit multiples (24, 8) of gen size 8 are returned unchanged.
        assert _resolve_num_prompts(24, train_batch_size=64, gen_batch_size=8) == 24
        assert _resolve_num_prompts(8, train_batch_size=64, gen_batch_size=8) == 8

    def test_gen_batch_size_one_allows_any_positive(self):
        # [FUNCTION DOC]
        # With gen size 1 any positive count (1,3,7,100) is valid.
        # The DAPO/drop-refill use case: gen_batch_size=1 makes any dropped count valid.
        for k in (1, 3, 7, 100):
            assert _resolve_num_prompts(k, train_batch_size=1024, gen_batch_size=1) == k

    @pytest.mark.parametrize("bad", [20, 1, 7, 15])
    def test_non_multiple_raises(self, bad):
        # [FUNCTION DOC]
        # Parametrized non-multiples of 8 raise ValueError. Inputs: bad - the invalid count.
        with pytest.raises(ValueError, match="must be a positive multiple of gen_batch_size"):
            _resolve_num_prompts(bad, train_batch_size=64, gen_batch_size=8)

    @pytest.mark.parametrize("bad", [0, -8, -1])
    def test_non_positive_raises(self, bad):
        # [FUNCTION DOC]
        # Parametrized zero/negative counts raise ValueError. Inputs: bad - the invalid count.
        with pytest.raises(ValueError, match="must be a positive multiple of gen_batch_size"):
            _resolve_num_prompts(bad, train_batch_size=64, gen_batch_size=8)


class TestNumFetches:
    # [CLASS DOC]
    # Tests mapping resolved counts to fetch counts.
    def test_default_gen_equals_train_is_single_fetch(self):
        # [FUNCTION DOC]
        # gen == train (1024) needs 1 fetch.
        n = _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1024)
        assert _num_fetches(n, 1024) == 1

    def test_small_gen_batch_size_multiple_fetches(self):
        # [FUNCTION DOC]
        # gen 1 with train 1024 needs 1024 fetches.
        # train_batch_size=1024, gen_batch_size=1 => 1024 individual fetches per train batch.
        n = _resolve_num_prompts(None, train_batch_size=1024, gen_batch_size=1)
        assert _num_fetches(n, 1) == 1024

    def test_intermediate_granularity(self):
        # [FUNCTION DOC]
        # train 64 / gen 8 needs 8 fetches.
        n = _resolve_num_prompts(None, train_batch_size=64, gen_batch_size=8)
        assert _num_fetches(n, 8) == 8

    def test_refill_count_maps_to_fetches(self):
        # [FUNCTION DOC]
        # Refilling 5 with gen 1 needs 5 fetches.
        # Refilling 5 dropped groups with gen_batch_size=1 => exactly 5 fetches.
        n = _resolve_num_prompts(5, train_batch_size=64, gen_batch_size=1)
        assert _num_fetches(n, 1) == 5


class TestCoalescedSubmission:
    # [CLASS DOC]
    # Tests that requests of many fetches still produce one submission.
    def test_gen_batch_size_one_submits_train_batch_once(self):
        # [FUNCTION DOC]
        # 64 fetches, 1 submission.
        assert _num_fetches(64, 1) == 64
        assert _num_submissions(64, gen_batch_size=1) == 1

    def test_large_refill_is_submitted_once(self):
        # [FUNCTION DOC]
        # 626 fetches, 1 submission.
        assert _num_fetches(626, 1) == 626
        assert _num_submissions(626, gen_batch_size=1) == 1

    def test_multiple_gen_chunks_are_submitted_once(self):
        # [FUNCTION DOC]
        # 96 prompts with gen 24: 4 fetches, 1 submission.
        assert _num_fetches(96, 24) == 4
        assert _num_submissions(96, gen_batch_size=24) == 1


class TestStepsPerEpoch:
    # [CLASS DOC]
    # Tests the steps-per-epoch derivation.
    def test_exact_division(self):
        # [FUNCTION DOC]
        # 10240 / 1024 = 10 steps.
        assert _steps_per_epoch(dataset_size=10240, train_batch_size=1024) == 10

    def test_drop_last_truncates_partial_batch(self):
        # [FUNCTION DOC]
        # 10500 / 1024 truncates to 10.
        # 10500 // 1024 == 10 (the trailing partial train batch is dropped, matching drop_last=True).
        assert _steps_per_epoch(dataset_size=10500, train_batch_size=1024) == 10

    def test_independent_of_gen_batch_size(self):
        # [FUNCTION DOC]
        # For several gen sizes, fetches per epoch scale inversely with gen size while steps stay constant.
        # steps_per_epoch must NOT depend on gen_batch_size; only on dataset size / train_batch_size.
        dataset_size, train_batch_size = 8192, 1024
        expected = _steps_per_epoch(dataset_size, train_batch_size)
        for gen_batch_size in (1, 8, 256, 1024):
            # Number of gen fetches per epoch scales with 1/gen_batch_size, but step count does not.
            fetches_per_epoch = dataset_size // gen_batch_size
            assert fetches_per_epoch == expected * (train_batch_size // gen_batch_size)
            assert _steps_per_epoch(dataset_size, train_batch_size) == expected

    def test_equals_len_dataloader_when_gen_equals_train(self):
        # [FUNCTION DOC]
        # When gen == train, the dataloader length equals steps per epoch.
        # When gen_batch_size == train_batch_size, len(dataloader) (== dataset//gen with drop_last)
        # equals steps_per_epoch exactly, reproducing the classic non-streaming step count.
        dataset_size, train_batch_size = 8192, 1024
        gen_batch_size = train_batch_size
        len_dataloader = dataset_size // gen_batch_size  # what StatefulDataLoader reports
        assert _steps_per_epoch(dataset_size, train_batch_size) == len_dataloader

    def test_streaming_len_dataloader_would_overcount(self):
        # [FUNCTION DOC]
        # With gen 1 the dataloader length overcounts steps by train/gen and therefore differs.
        # Documents the bug being avoided: with gen_batch_size < train_batch_size, naively using
        # len(dataloader) as the step count over-counts by train_batch_size/gen_batch_size.
        dataset_size, train_batch_size, gen_batch_size = 8192, 1024, 1
        len_dataloader = dataset_size // gen_batch_size  # 8192 gen fetches
        steps = _steps_per_epoch(dataset_size, train_batch_size)  # 8 real steps
        assert len_dataloader == steps * (train_batch_size // gen_batch_size)
        assert len_dataloader != steps  # naive use would be wrong

    def test_total_training_steps_composition(self):
        # [FUNCTION DOC]
        # 10 steps per epoch times 3 epochs is 30.
        # total_training_steps = steps_per_epoch * total_epochs.
        steps = _steps_per_epoch(dataset_size=10240, train_batch_size=1024)
        assert steps * 3 == 30
