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
# Tests for pipeline-parallel helpers: make_batch_generator (virtual pipeline batch iterators) and
# get_dynamic_pipeline_shards (layer-to-stage partitioning used by the Megatron model merger).

import pytest

from verl.model_merger.megatron_model_merger import get_dynamic_pipeline_shards
from verl.utils.megatron.pipeline_parallel import make_batch_generator


def test_make_batch_generator_no_vpp():
    # [FUNCTION DOC]
    # Behavior: with vpp_size 1, asserts make_batch_generator returns a single iterator yielding the batches in order.
    # Inputs: none. Outputs: none.
    batches = [1, 2, 3]
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == batches


def test_make_batch_generator_with_vpp():
    # [FUNCTION DOC]
    # Behavior: with vpp_size 2, asserts the result is a list of 2 iterators, each independently yielding all batches.
    # Inputs: none. Outputs: none.
    batches = [{"data": 1}, {"data": 2}]
    vpp_size = 2
    generators = make_batch_generator(batches, vpp_size)
    assert isinstance(generators, list)
    assert len(generators) == vpp_size

    # Check each generator yields the original batches
    for gen in generators:
        assert list(gen) == batches


def test_make_batch_generator_empty():
    # [FUNCTION DOC]
    # Behavior: empty batch list; asserts vpp 1 yields nothing and vpp 3 returns 3 empty iterators.
    # Inputs: none. Outputs: none.
    batches = []
    vpp_size = 1
    generator = make_batch_generator(batches, vpp_size)
    assert list(generator) == []

    vpp_size = 3
    generators = make_batch_generator(batches, vpp_size)
    assert len(generators) == vpp_size
    for gen in generators:
        assert list(gen) == []


@pytest.mark.parametrize(
    "layer_num,pp_size,gt",
    [
        (61, 8, [6, 8, 8, 8, 8, 8, 8, 7]),
        (61, 7, [8, 9, 9, 9, 9, 9, 8]),
        (61, 1, [61]),
        (61, 0, ValueError),
        (10, 16, ValueError),
    ],
)
def test_get_dynamic_pipeline_shards(layer_num, pp_size, gt):
    # [FUNCTION DOC]
    # Behavior: parametrized; when gt is a list, asserts get_dynamic_pipeline_shards returns pp_size shard sizes equal
    # to gt element-wise (e.g. 61 layers on 8 stages gives 6,8,8,8,8,8,8,7 - first/last stages lighter); when gt is an
    # exception class (pp_size 0, or more stages than layers), asserts that exception is raised.
    # Inputs: layer_num - total layers; pp_size - pipeline stages; gt - expected shard list or exception type.
    # Outputs: none.
    if isinstance(gt, list):
        shards = get_dynamic_pipeline_shards(layer_num, pp_size)
        assert len(shards) == len(gt) == pp_size, f"Expected {pp_size} shards, got {len(shards)}"
        assert all([shard == gt[i] for i, shard in enumerate(shards)]), f"Expected shards {gt}, got {shards}"
    elif issubclass(gt, Exception):
        with pytest.raises(gt):
            shards = get_dynamic_pipeline_shards(layer_num, pp_size)
