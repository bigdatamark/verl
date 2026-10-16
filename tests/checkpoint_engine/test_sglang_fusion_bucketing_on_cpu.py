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
# CPU tests for get_named_tensor_buckets in the SGLang rollout utils, focusing on
# "fusion groups": sets of parameter-name suffixes (DeepSeek-V4) that must be
# delivered in the same bucket even if that exceeds the bucket byte limit.

import asyncio

import pytest
import torch

from verl.workers.rollout.sglang_rollout.utils import DEEPSEEK_V4_FUSION_GROUPS, get_named_tensor_buckets


def _tensor(nbytes):
    # [FUNCTION DOC]
    # HELPER: returns an uninitialized 1-D uint8 tensor of exactly nbytes bytes.
    return torch.empty(nbytes, dtype=torch.uint8)


def _buckets(items, fusion_groups=()):
    # [FUNCTION DOC]
    # HELPER: drains the async bucket generator synchronously.
    # Inputs: items - list of (name, tensor) pairs; fusion_groups - sequence of
    # suffix groups (default none). Bucket size is fixed at 1024 bytes.
    # Outputs: list of buckets, each a list of (name, tensor) pairs.
    # Exceptions: propagates errors from the generator (e.g. ValueError for incomplete groups).
    # Side effects: runs a new asyncio event loop.
    async def collect():
        # [FUNCTION DOC]
        # HELPER: async inner collector gathering every bucket into a list.
        return [bucket async for bucket in get_named_tensor_buckets(iter(items), 1024, fusion_groups)]

    return asyncio.run(collect())


@pytest.mark.parametrize("group", DEEPSEEK_V4_FUSION_GROUPS)
def test_deepseek_v4_fusion_group_stays_in_one_bucket(group):
    # [FUNCTION DOC]
    # Behavior: for each DeepSeek-V4 fusion group, places a 1000-byte filler first
    # and then 64-byte tensors for each suffix under layer 7; asserts some bucket
    # contains all members of the group (the group is not split across buckets).
    # Inputs: group - one fusion group (parametrized). Outputs: none (assertions).
    prefix = "model.layers.7"
    items = [("filler", _tensor(1000)), *((prefix + suffix, _tensor(64)) for suffix in group)]
    buckets = _buckets(items, DEEPSEEK_V4_FUSION_GROUPS)
    names = [[name for name, _ in bucket] for bucket in buckets]
    assert any(all(prefix + suffix in bucket for suffix in group) for bucket in names)


def test_fusion_grouping_is_opt_in_for_non_deepseek_models():
    # [FUNCTION DOC]
    # Behavior: with no fusion groups passed, a DeepSeek-style name is bucketed
    # normally, yielding exactly one bucket with that single name.
    name = "model.layers.7.attn.wq_a.weight"
    assert [[item_name for item_name, _ in bucket] for bucket in _buckets([(name, _tensor(64))])] == [[name]]


def test_deepseek_v4_incomplete_group_fails_loudly():
    # [FUNCTION DOC]
    # Behavior: passing only one member of a fusion group must raise ValueError
    # whose message contains "never completed" once the iterator is exhausted.
    with pytest.raises(ValueError, match="never completed"):
        _buckets(
            [("model.layers.7.attn.wq_a.weight", _tensor(64))],
            DEEPSEEK_V4_FUSION_GROUPS,
        )
