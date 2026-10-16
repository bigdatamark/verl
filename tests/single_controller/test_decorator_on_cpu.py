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
# CPU Ray tests for the register decorator: synchronous DP_COMPUTE_PROTO, async def methods returning a
# DataProtoFuture, and nd-compute dispatch over a registered mesh with TensorDict inputs containing jagged nested tensors.

import asyncio
import time

import pytest
import ray
import torch
from tensordict import TensorDict

from verl.protocol import DataProto, DataProtoFuture
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import tensordict_utils as tu


# Pytest fixture for Ray setup/teardown
@pytest.fixture
def ray_init_shutdown():
    # [FUNCTION DOC]
    # HELPER: Pytest fixture that starts Ray with 100 CPUs before a test and shuts it down after.
    ray.init(num_cpus=100)
    yield
    ray.shutdown()


# Define a simple worker for testing
@ray.remote
class DecoratorTestWorker(Worker):
    # [CLASS DOC]
    # HELPER: Ray worker holding an initial value, registering mesh "train" with dp_rank equal to its rank and collecting from every rank.
    def __init__(self, initial_value=0):
        # [FUNCTION DOC]
        # HELPER: Calls base init, stores initial_value (default 0), sleeps 0.1s, registers dispatch/collect info for mesh "train".
        super().__init__()
        self.value = initial_value
        # Simulate some setup if needed
        time.sleep(0.1)  # Ensure worker init completes

        self._register_dispatch_collect_info(mesh_name="train", dp_rank=self.rank, is_collect=True)

    # Test method for synchronous DP compute (default behavior)
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def dp_compute(self, data: DataProto) -> DataProto:
        # [FUNCTION DOC]
        # HELPER: Sync DP_COMPUTE_PROTO; after a 0.1s sleep writes batch "output" = input + value + rank (rank as a tensor
        # matching the input's device/dtype). Returns the mutated DataProto.
        time.sleep(0.1)  # Simulate work
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        data.batch["output"] = data.batch["input"] + self.value + rank_value
        return data

    # Test async def method with DP compute (default behavior)
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
    async def async_dp_compute(self, data: DataProto) -> DataProto:
        # Simulate async work
        # [FUNCTION DOC]
        # HELPER: Async, non-blocking DP_COMPUTE_PROTO; awaits 0.1s then writes "output_async" = input*2 + value + rank.
        await asyncio.sleep(0.1)  # Simulate async work
        rank_value = torch.tensor(self.rank, device=data.batch["input"].device, dtype=data.batch["input"].dtype)
        data.batch["output_async"] = data.batch["input"] * 2 + self.value + rank_value
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def dp_compute_td(self, data: TensorDict) -> TensorDict:
        # note that we have to call contiguous so that we can modify data in plac
        # [FUNCTION DOC]
        # HELPER: Non-blocking nd-compute over mesh "train" with a TensorDict. Makes the dict contiguous, writes
        # "output" = input + value + rank, pops the jagged "position_ids", sets its ragged dim to 2, and asserts each row i
        # equals an arange of length 4 + 2*rank + i broadcast to its shape (verifies correct nested-tensor splitting).
        # Outputs: the TensorDict without position_ids. Exceptions: AssertionError on mismatch.
        data = tu.contiguous(data)
        rank_value = torch.tensor(self.rank, device=data["input"].device, dtype=data["input"].dtype)
        data["output"] = data["input"] + self.value + rank_value
        position_ids = data.pop("position_ids")
        position_ids._ragged_idx = 2

        for i, position_id in enumerate(position_ids.unbind(dim=0)):
            assert (position_id == torch.arange(4 + rank_value * 2 + i).expand(position_id.shape)).all()

        return data


# Test function for synchronous DP compute
def test_decorator_dp_compute(ray_init_shutdown):
    # [FUNCTION DOC]
    # Behavior: 2 CPU workers with initial_value 10; dispatches a 4-element input and asserts the result is a DataProto
    # whose "output" equals input + 10 + rank for each half.
    """
    Tests the default behavior of a synchronous decorated method with DP_COMPUTE_PROTO.
    Verifies the result correctness.
    """
    num_workers = 2
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)  # Use CPU for simplicity
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=10)
    worker_group = RayWorkerGroup(
        resource_pool, cls_with_args, name_prefix=f"decorator_test_sync_dp_{int(time.time())}"
    )

    # Prepare input data (size 4, for 2 workers)
    input_tensor = torch.arange(4, dtype=torch.float32)
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    # Call the decorated method
    output = worker_group.dp_compute(data)

    # Assert the result correctness
    assert isinstance(output, DataProto), "Expected DataProto result"
    assert "output" in output.batch.keys()
    assert len(output) == len(data), "Output length should match input length"

    # Expected output calculation for DP_COMPUTE_PROTO with 2 workers
    # Worker 0 gets data[0:2], Worker 1 gets data[2:4]
    # Worker 0 adds initial_value(10) + rank(0) = 10
    # Worker 1 adds initial_value(10) + rank(1) = 11
    expected_output_part1 = torch.tensor([0, 1], dtype=torch.float32) + 10 + 0
    expected_output_part2 = torch.tensor([2, 3], dtype=torch.float32) + 10 + 1
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    torch.testing.assert_close(output.batch["output"], expected_output, msg="Sync DP compute output data mismatch")


# Test function for async def method with DP compute
def test_decorator_async_function(ray_init_shutdown):
    # [FUNCTION DOC]
    # Behavior: 2 workers with initial_value 5; calls the async method, asserts it returns a DataProtoFuture, gets it and
    # checks "output_async" equals input*2 + 5 + rank per half.
    """
    Tests the decorator with an `async def` method using DP_COMPUTE_PROTO.
    Verifies that the call returns a future and the result is correct after .get().
    """
    num_workers = 2
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=5)
    worker_group = RayWorkerGroup(
        resource_pool, cls_with_args, name_prefix=f"decorator_test_async_dp_{int(time.time())}"
    )

    # Prepare input data (size 4, for 2 workers)
    input_tensor = torch.arange(4, dtype=torch.float32)
    data = DataProto(batch=TensorDict({"input": input_tensor}, batch_size=[4]))

    # Call the async decorated method - this should return a future
    future_output: DataProtoFuture = worker_group.async_dp_compute(data)

    # Assert that the call returned a future
    assert isinstance(future_output, DataProtoFuture), "Expected DataProtoFuture for async def call"

    # Get the result (this should block)
    result_data = future_output.get()

    # Assert the result correctness
    assert isinstance(result_data, DataProto)
    assert "output_async" in result_data.batch.keys()
    assert len(result_data) == len(data), "Output length should match input length"

    # Expected output calculation for DP_COMPUTE_PROTO with 2 workers
    # Worker 0 gets data[0:2], Worker 1 gets data[2:4]
    # Worker 0 calculates: input * 2 + initial_value(5) + rank(0)
    # Worker 1 calculates: input * 2 + initial_value(5) + rank(1)
    expected_output_part1 = (torch.tensor([0, 1], dtype=torch.float32) * 2) + 5 + 0
    expected_output_part2 = (torch.tensor([2, 3], dtype=torch.float32) * 2) + 5 + 1
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    torch.testing.assert_close(
        result_data.batch["output_async"], expected_output, msg="Async DP compute output data mismatch"
    )


def test_decorator_dp_compute_td(ray_init_shutdown):
    # [FUNCTION DOC]
    # Behavior: builds a TensorDict with a 4-element input and jagged position_ids rows of lengths 4..7 (each 4 x L),
    # calls dp_compute_td, gets the future and checks output is a TensorDict whose "output" equals input + 10 + rank.
    num_workers = 2
    resource_pool = RayResourcePool([num_workers], use_gpu=False, max_colocate_count=1)  # Use CPU for simplicity
    cls_with_args = RayClassWithInitArgs(cls=DecoratorTestWorker, initial_value=10)
    worker_group = RayWorkerGroup(
        resource_pool, cls_with_args, name_prefix=f"decorator_test_sync_dp_{int(time.time())}"
    )

    # Prepare input data (size 4, for 2 workers)
    input_tensor = torch.arange(4, dtype=torch.float32)
    position_ids = torch.nested.as_nested_tensor(
        [
            torch.arange(4).expand(4, 4).contiguous(),
            torch.arange(5).expand(4, 5).contiguous(),
            torch.arange(6).expand(4, 6).contiguous(),
            torch.arange(7).expand(4, 7).contiguous(),
        ],
        layout=torch.jagged,
    )
    data = TensorDict({"input": input_tensor, "position_ids": position_ids}, batch_size=[4])

    # Call the decorated method
    output = worker_group.dp_compute_td(data)

    output = output.get()

    # Assert the result correctness
    assert isinstance(output, TensorDict), "Expected DataProto result"
    assert "output" in output.keys()
    assert len(output) == len(data), "Output length should match input length"

    # Expected output calculation for DP_COMPUTE_PROTO with 2 workers
    # Worker 0 gets data[0:2], Worker 1 gets data[2:4]
    # Worker 0 adds initial_value(10) + rank(0) = 10
    # Worker 1 adds initial_value(10) + rank(1) = 11
    expected_output_part1 = torch.tensor([0, 1], dtype=torch.float32) + 10 + 0
    expected_output_part2 = torch.tensor([2, 3], dtype=torch.float32) + 10 + 1
    expected_output = torch.cat([expected_output_part1, expected_output_part2])

    torch.testing.assert_close(output["output"], expected_output, msg="Sync DP compute output data mismatch")
