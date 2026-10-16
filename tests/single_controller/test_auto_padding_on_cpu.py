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
# CPU Ray test for DataProto auto padding with DP_COMPUTE_PROTO dispatch. Sets the global DataProtoConfig.auto_padding
# to True at import time, then checks local padding/chunking and that worker-group calls with batch sizes not
# divisible by the worker count return outputs of the original length (both positional and keyword arguments,
# with global and per-instance auto_padding).
# Side effects: mutates global DataProtoConfig.auto_padding on import and during the test.

import numpy as np
import ray
import torch

from verl import DataProto
from verl.protocol import DataProtoConfig
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

# or set env var VERL_AUTO_PADDING = "1" / "true"
DataProtoConfig.auto_padding = True


@ray.remote
class Actor(Worker):
    # [CLASS DOC]
    # HELPER: Minimal Ray worker used as the dispatch target.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls the base Worker initializer only. Side effects: Worker reads rank/world-size env.
        super().__init__()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        # [FUNCTION DOC]
        # HELPER: DP_COMPUTE_PROTO method; adds the worker's rank to batch field "a" in place and returns the same DataProto.
        # Inputs: data - DataProto with tensor "a". Outputs: the mutated data.
        data.batch["a"] += self.rank
        return data


def test_auto_padding():
    # [FUNCTION DOC]
    # Behavior: inits Ray with 100 CPUs and a 4-worker CPU group. For sizes 4..19 pads a local DataProto to a multiple of 4,
    # chunks it into 4 and checks each chunk length equals ceil(size/4). Then calls the remote add with sizes 10, 1 and 8
    # (args and kwargs) and asserts output length equals input length; then disables global auto padding and repeats
    # with auto_padding=True set on the individual DataProto (from_dict and from_single_dict).
    # Inputs: none. Outputs: none.
    # Exceptions: AssertionError on mismatches.
    # Side effects: starts and shuts down Ray, prints outputs, leaves DataProtoConfig.auto_padding False.
    # KNOWN ISSUE: the post-padding length assertion compares len(local_data) against itself plus len % chunk_size, which is trivially true once padding has made the length divisible, so it checks nothing useful.
    ray.init(num_cpus=100)

    chunk_size = 4
    actor_cls = RayClassWithInitArgs(cls=Actor)
    resource_pool = RayResourcePool(process_on_nodes=[chunk_size], use_gpu=False)
    actor_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=actor_cls)

    # test locally first
    for test_size in range(4, 20):
        local_data = DataProto.from_dict({"a": torch.zeros(test_size)}, {"na": np.zeros(test_size, dtype=object)})
        # print(f"before padding, local_data = {local_data}")
        padding_size = (chunk_size - (test_size % chunk_size)) if (test_size % chunk_size > 0) else 0
        local_data.padding(padding_size)
        # print(f"after padding, local_data = {local_data}")
        assert len(local_data) == len(local_data) + len(local_data) % chunk_size, (
            f"expecting padded length to be {len(local_data) + len(local_data) % chunk_size}, but got {len(local_data)}"
        )
        chunked = local_data.chunk(chunk_size)
        assert len(chunked) == chunk_size, f"during test_size = {test_size}, expecting {chunk_size}, got {chunked}"
        for dp in chunked:
            assert len(dp) == test_size // chunk_size + bool(test_size % chunk_size), (
                f"test size = {test_size}, expecting dp to be length of "
                f"{test_size // chunk_size + bool(test_size % chunk_size)}, but got {len(dp)}: {dp} {chunked}"
            )

    # test with RayWorkerGroup method decorated as dispatch_mode=Dispatch.DP_COMPUTE_PROTO
    data = DataProto.from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)})
    output = actor_wg.add(data)

    print(output.batch["a"])
    assert len(output) == 10, "Failed in args split and padding."

    data = DataProto.from_dict({"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output.batch["a"])
    assert len(output) == 10, "Failed in kwargs split and padding."

    data = DataProto.from_dict({"a": torch.zeros(1)}, {"na": np.array([str(i) for i in range(1)], dtype=object)})
    output = actor_wg.add(data)

    print(output.batch["a"])
    assert len(output) == 1, "Failed in args split and padding."

    data = DataProto.from_dict({"a": torch.zeros(1)}, {"na": np.array([str(i) for i in range(1)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output.batch["a"])
    assert len(output) == 1, "Failed in kwargs split and padding."

    data = DataProto.from_dict({"a": torch.zeros(8)}, {"na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data)

    print(output.batch["a"])
    assert len(output) == 8, "Failed in args split and padding."

    data = DataProto.from_dict({"a": torch.zeros(8)}, {"na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output.batch["a"])
    assert len(output) == 8, "Failed in kwargs split and padding."

    # test data proto specific config
    DataProtoConfig.auto_padding = False

    data = DataProto.from_dict(
        {"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data)
    print(output.batch["a"])
    assert len(output) == 10, "Failed in args split and padding."

    data = DataProto.from_dict(
        {"a": torch.zeros(10)}, {"na": np.array([str(i) for i in range(10)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data=data)
    print(output.batch["a"])
    assert len(output) == 10, "Failed in kwargs split and padding."

    data = DataProto.from_single_dict(
        {"a": torch.zeros(1), "na": np.array([str(i) for i in range(1)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data)

    print(output.batch["a"])
    assert len(output) == 1, "Failed in args split and padding."

    data = DataProto.from_single_dict(
        {"a": torch.zeros(1), "na": np.array([str(i) for i in range(1)], dtype=object)}, auto_padding=True
    )
    output = actor_wg.add(data=data)

    print(output.batch["a"])
    assert len(output) == 1, "Failed in kwargs split and padding."

    data = DataProto.from_single_dict({"a": torch.zeros(8), "na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data)

    print(output.batch["a"])
    assert len(output) == 8, "Failed in args split and padding."

    data = DataProto.from_single_dict({"a": torch.zeros(8), "na": np.array([str(i) for i in range(8)], dtype=object)})
    output = actor_wg.add(data=data)

    print(output.batch["a"])
    assert len(output) == 8, "Failed in kwargs split and padding."

    ray.shutdown()


if __name__ == "__main__":
    test_auto_padding()
