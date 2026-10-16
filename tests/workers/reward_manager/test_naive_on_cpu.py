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
# CPU tests for the per-sample compute_score timeout of NaiveRewardManager and its _score_timeout
# context manager (SIGALRM-based; requires main thread on a POSIX system). Tests involve real sleeps.

"""CPU tests for NaiveRewardManager's per-sample compute_score timeout."""

import time

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.workers.reward_manager.naive import NaiveRewardManager, _score_timeout


class _DummyTokenizer:
    # [CLASS DOC]
    # HELPER: fake tokenizer whose decode always returns "dummy".
    def decode(self, token_ids, skip_special_tokens=True):
        # [FUNCTION DOC]
        # HELPER: ignores inputs and returns the string "dummy".
        return "dummy"


def _make_minimal_data(batch_size=1, prompt_len=2, response_len=2):
    # [FUNCTION DOC]
    # HELPER: builds a DataProto from a single dict with zero prompts/responses, an all-ones attention mask of
    # prompt plus response length, and data_source / reward_model (ground truth "1") object arrays.
    # Inputs: batch_size, prompt_len, response_len (defaults 1, 2, 2). Outputs: DataProto.
    total = prompt_len + response_len
    return DataProto.from_single_dict(
        {
            "prompts": torch.zeros((batch_size, prompt_len), dtype=torch.long),
            "responses": torch.zeros((batch_size, response_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, total), dtype=torch.long),
            "data_source": np.array(["dummy"] * batch_size, dtype=object),
            "reward_model": np.array([{"ground_truth": "1"} for _ in range(batch_size)], dtype=object),
        }
    )


def test_score_timeout_allows_fast_calls():
    # [FUNCTION DOC]
    # Asserts a fast computation inside a 2 s timeout completes normally.
    with _score_timeout(2.0):
        result = 1 + 1
    assert result == 2


def test_score_timeout_raises_on_slow_calls():
    # [FUNCTION DOC]
    # Asserts sleeping 5 s inside a 0.5 s timeout raises TimeoutError.
    with pytest.raises(TimeoutError):
        with _score_timeout(0.5):
            time.sleep(5)


def test_score_timeout_disabled_is_noop():
    # [FUNCTION DOC]
    # Asserts None and 0 timeouts disable the alarm and short sleeps complete.
    # None or 0 disables the timeout (no SIGALRM is installed).
    with _score_timeout(None):
        time.sleep(0.05)
    with _score_timeout(0):
        time.sleep(0.05)


def test_naive_reward_manager_times_out_slow_score():
    # [FUNCTION DOC]
    # Uses a compute_score that sleeps 10 s with compute_score_timeout 1.0; asserts the call returns in under 5 s
    # and the timed-out sample receives total reward 0.0.
    """A hanging compute_score (e.g. ReDoS / slow sympy) must not block the loop."""

    def slow_compute_score(data_source, solution_str, ground_truth, extra_info=None):
        # [FUNCTION DOC]
        # HELPER: scoring stub that sleeps 10 seconds then returns 1.0.
        time.sleep(10)
        return 1.0

    manager = NaiveRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=slow_compute_score,
        compute_score_timeout=1.0,
    )
    data = _make_minimal_data()

    start = time.time()
    out = manager(data, return_dict=True)
    elapsed = time.time() - start

    assert elapsed < 5.0, f"reward manager did not time out (took {elapsed:.1f}s)"
    # Timed-out sample is assigned reward 0.0 instead of hanging.
    assert out["reward_tensor"].sum().item() == 0.0


def test_naive_reward_manager_no_timeout_by_default():
    # [FUNCTION DOC]
    # Asserts that without a timeout the manager returns the fast score 0.5 as the reward sum.
    """Without compute_score_timeout, behavior is unchanged."""

    def fast_compute_score(data_source, solution_str, ground_truth, extra_info=None):
        # [FUNCTION DOC]
        # HELPER: scoring stub returning 0.5 immediately.
        return 0.5

    manager = NaiveRewardManager(
        tokenizer=_DummyTokenizer(),
        num_examine=0,
        compute_score=fast_compute_score,
    )
    data = _make_minimal_data()
    out = manager(data, return_dict=True)
    assert out["reward_tensor"].sum().item() == pytest.approx(0.5)
