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
# Tests for compute_advantage_for_multi_trajectories in the v1 PPO utils, which computes outcome advantages
# when a sample is split into multiple turn rows. Batch keys follow the pattern
# "<uid>_<trajectory index>_<turn index>"; advantages are computed on the final turn row of each
# trajectory and broadcast (masked by response_mask) to every row of that trajectory.
import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.v1.utils import compute_advantage, compute_advantage_for_multi_trajectories


@pytest.fixture
def batch_data() -> DataProto:
    # [FUNCTION DOC]
    # HELPER: fixture building a 6-row DataProto with 4-token float token_level_rewards, a long response_mask
    # (including a fully masked last row) and all uids "prompt_a". Outputs: the DataProto.
    tensors = {
        "token_level_rewards": torch.tensor(
            [
                [100, 0, 0, 0],
                [1, 2, 3, 4],
                [200, 200, 0, 0],
                [150, 0, 0, 150],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
            ],
            dtype=torch.float32,
        ),
        "response_mask": torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
    }
    non_tensors = {
        "uid": np.array(["prompt_a"] * 6, dtype=object),
    }
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_compute_advantage_for_single_trajectory(batch_data: DataProto):
    # [FUNCTION DOC]
    # Behavior: each row is its own trajectory (keys prompt_a_i_0); asserts advantages and returns equal plain
    # GRPO compute_advantage on the same batch.
    # Inputs: batch_data fixture. Outputs: none. Exceptions: AssertionError.
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=[f"prompt_a_{i}_0" for i in range(len(batch_data))],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data,
        adv_estimator=AdvantageEstimator.GRPO,
    )
    assert torch.equal(result.batch["advantages"], expected.batch["advantages"])
    assert torch.equal(result.batch["returns"], expected.batch["returns"])


def test_compute_advantage_for_multi_trajectories(batch_data: DataProto):
    # [FUNCTION DOC]
    # Behavior: rows 0-1 form trajectory 0, rows 2-3 trajectory 2, rows 4 and 5 single-row trajectories. Expected
    # values are GRPO advantages on the last rows [1,3,4,5]; for each trajectory the scalar advantage is read
    # at the first valid response position of its last row (columns 0,1,0,0) and multiplied by each row's
    # response_mask. Asserts both advantages and returns equal this expectation.
    # Inputs: batch_data fixture. Outputs: none. Exceptions: AssertionError.
    result = compute_advantage_for_multi_trajectories(
        data=batch_data,
        batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_2_1", "prompt_a_3_0", "prompt_a_4_0"],
        adv_estimator=AdvantageEstimator.GRPO,
    )
    expected = compute_advantage(
        batch_data.select_idxs([1, 3, 4, 5]),
        adv_estimator=AdvantageEstimator.GRPO,
    )
    gather_row_indices = [0, 0, 1, 1, 2, 3]
    gather_col_indices = [0, 0, 1, 1, 0, 0]
    adv_expected = (
        expected.batch["advantages"][gather_row_indices, gather_col_indices].unsqueeze(-1)
        * result.batch["response_mask"]
    )
    assert torch.equal(result.batch["advantages"], adv_expected)
    assert torch.equal(result.batch["returns"], adv_expected)
