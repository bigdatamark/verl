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
# CPU tests for verl.utils.ray_utils.parallel_put: object refs are returned in order, empty input is rejected, and a
# max_workers argument works.

import pytest
import ray

from verl.utils.ray_utils import parallel_put


# Initialize Ray for testing if not already done globally
@pytest.fixture()
def init_ray():
    # [FUNCTION DOC]
    # HELPER: Pytest fixture starting Ray with 4 CPUs and shutting it down afterwards.
    ray.init(num_cpus=4)
    yield
    ray.shutdown()


def test_parallel_put_basic(init_ray):
    # [FUNCTION DOC]
    # Behavior: puts a mixed list (int, str, dict, list) and asserts the refs count matches and ray.get yields the same values in order.
    data = [1, "hello", {"a": 2}, [3, 4]]
    refs = parallel_put(data)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data


def test_parallel_put_empty(init_ray):
    # [FUNCTION DOC]
    # Behavior: asserts parallel_put on an empty list raises AssertionError.
    data = []
    with pytest.raises(AssertionError):
        _ = parallel_put(data)


def test_parallel_put_workers(init_ray):
    # [FUNCTION DOC]
    # Behavior: puts 0..19 with max_workers=4 and with the default, asserting round-trip equality in order each time.
    data = list(range(20))
    # Test with specific number of workers
    refs = parallel_put(data, max_workers=4)
    assert len(refs) == len(data)
    retrieved_data = [ray.get(ref) for ref in refs]
    assert retrieved_data == data
    # Test with default workers (should cap)
    refs_default = parallel_put(data)
    assert len(refs_default) == len(data)
    retrieved_data_default = [ray.get(ref) for ref in refs_default]
    assert retrieved_data_default == data
