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
# GPU Ray tests for split_resource_pool: splitting a pool by a fixed size, by a list of sizes (within and across
# simulated nodes) and splitting a split pool again, then checking world sizes and DP outputs per sub-pool.

import os

import ray
import torch

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    split_resource_pool,
)
from verl.utils.device import get_device_name, get_nccl_backend


def get_local_gpus_num(division=1):
    # [FUNCTION DOC]
    # HELPER: Returns the local CUDA device count integer-divided by division, at least 1.
    return max(1, torch.cuda.device_count() // division)


@ray.remote
class Actor(Worker):
    # [CLASS DOC]
    # HELPER: Worker holding a worker_id offset; allocates a 4096x4096 random tensor on device (to occupy memory) and
    # initializes torch.distributed if needed.
    def __init__(self, worker_id) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base init, stores worker_id, allocates the device tensor and, if not initialized, inits the process
        # group with the device collective backend using RANK (default 0) and WORLD_SIZE (default 1).
        # Side effects: global distributed init, device memory.
        super().__init__()
        self.worker_id = worker_id
        self.temp_tensor = torch.rand(4096, 4096).to(get_device_name())

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend=get_nccl_backend(), world_size=world_size, rank=rank)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def add(self, data: DataProto):
        # [FUNCTION DOC]
        # HELPER: DP_COMPUTE_PROTO; adds rank + worker_id to batch "a" in place and returns it.
        data.batch["a"] += self.rank + self.worker_id
        return data


def test_split_resource_pool_with_split_size():
    # [FUNCTION DOC]
    # Behavior: builds a pool of two simulated nodes of half GPUs, creates placement groups, splits by half into two pools
    # with worker_id 0 and 100, asserts each world size is half and that adding to an ngpus-length zero tensor gives each
    # rank's value repeated twice (plus 100 for the second).
    # KNOWN ISSUE: the "repeated twice" expectation assumes ngpus equals 2 x half (i.e. an even GPU count >= 2).
    ray.init()
    ngpus = torch.cuda.device_count()
    half = get_local_gpus_num(2)
    # simulate 2 nodes of half GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[half, half])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(
        resource_pool=global_resource_pool, split_size=half
    )
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )
    assert actor_worker_1.world_size == half
    assert actor_worker_2.world_size == half

    data = DataProto.from_dict({"a": torch.zeros(ngpus)})
    actor_output_1 = actor_worker_1.add(data)
    actor_output_2 = actor_worker_2.add(data)
    assert actor_output_1.batch["a"].tolist() == [float(r) for r in range(half) for _ in range(2)]
    assert actor_output_2.batch["a"].tolist() == [float(r + 100) for r in range(half) for _ in range(2)]

    ray.shutdown()


def test_split_resource_pool_with_split_size_list():
    # [FUNCTION DOC]
    # Behavior: four simulated nodes of quarter GPUs, split into [quarter, 3*quarter]; asserts world sizes and outputs
    # 0..quarter-1 and 100..100+3*quarter-1.
    ray.init()
    quarter = get_local_gpus_num(4)
    # simulate 4 nodes of quarter GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[quarter] * 4)
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[quarter, 3 * quarter],
    )
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )
    assert actor_worker_1.world_size == quarter
    assert actor_worker_2.world_size == 3 * quarter

    data_1 = DataProto.from_dict({"a": torch.zeros(quarter)})
    data_2 = DataProto.from_dict({"a": torch.zeros(3 * quarter)})
    actor_output_1 = actor_worker_1.add(data_1)
    actor_output_2 = actor_worker_2.add(data_2)
    print(actor_output_1.batch["a"].tolist())
    print(actor_output_2.batch["a"].tolist())
    assert actor_output_1.batch["a"].tolist() == list(range(quarter))
    assert actor_output_2.batch["a"].tolist() == list(range(100, 100 + 3 * quarter))

    ray.shutdown()


def test_split_resource_pool_with_split_size_list_cross_nodes():
    # [FUNCTION DOC]
    # Behavior: same as the list test but the pool is two nodes of half GPUs, so the second split spans nodes.
    ray.init()
    half = get_local_gpus_num(2)
    quarter = get_local_gpus_num(4)
    # simulate 2 nodes of half GPUs each (cross-node split)
    global_resource_pool = RayResourcePool(process_on_nodes=[half, half])
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    actor_1_resource_pool, actor_2_resource_pool = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[quarter, 3 * quarter],
    )
    actor_cls_1 = RayClassWithInitArgs(cls=Actor, worker_id=0)
    actor_cls_2 = RayClassWithInitArgs(cls=Actor, worker_id=100)
    actor_worker_1 = RayWorkerGroup(
        resource_pool=actor_1_resource_pool, ray_cls_with_init=actor_cls_1, device_name=get_device_name()
    )
    actor_worker_2 = RayWorkerGroup(
        resource_pool=actor_2_resource_pool, ray_cls_with_init=actor_cls_2, device_name=get_device_name()
    )

    assert actor_worker_1.world_size == quarter
    assert actor_worker_2.world_size == 3 * quarter

    data_1 = DataProto.from_dict({"a": torch.zeros(quarter)})
    data_2 = DataProto.from_dict({"a": torch.zeros(3 * quarter)})
    actor_output_1 = actor_worker_1.add(data_1)
    actor_output_2 = actor_worker_2.add(data_2)
    print(actor_output_1.batch["a"].tolist())
    print(actor_output_2.batch["a"].tolist())
    assert actor_output_1.batch["a"].tolist() == list(range(quarter))
    assert actor_output_2.batch["a"].tolist() == list(range(100, 100 + 3 * quarter))

    ray.shutdown()


def test_split_resource_pool_with_split_twice():
    # [FUNCTION DOC]
    # Behavior: pool of ngpus/2 nodes with 2 GPUs each, split into [quarter, mid, quarter] with mid = ngpus - 2*quarter,
    # then the middle pool split into size-1 pools; for each final pool (worker_id = index*100) dispatches 4 zeros and
    # asserts world size and that each rank's value (rank + index*100) repeats 4 / world_size times.
    ray.init()
    ngpus = torch.cuda.device_count()
    quarter = get_local_gpus_num(4)
    mid = ngpus - 2 * quarter  # middle pool size
    # simulate ngpus//2 nodes of 2 GPUs each
    global_resource_pool = RayResourcePool(process_on_nodes=[2] * (ngpus // 2))
    global_resource_pool.get_placement_groups(device_name=get_device_name())

    rp_1, rp_2, rp_3 = split_resource_pool(
        resource_pool=global_resource_pool,
        split_size=[quarter, mid, quarter],
    )
    rp_2_subs = split_resource_pool(resource_pool=rp_2, split_size=1)
    fp_list = [rp_1] + list(rp_2_subs) + [rp_3]

    correct_world_size = [quarter] + [1] * mid + [quarter]
    correct_output = []
    for ws in correct_world_size:
        idx = len(correct_output)
        correct_output.append([float(r + idx * 100) for r in range(ws) for _ in range(4 // ws)])

    for idx, rp in enumerate(fp_list):
        actor_cls = RayClassWithInitArgs(cls=Actor, worker_id=idx * 100)
        actor_worker = RayWorkerGroup(resource_pool=rp, ray_cls_with_init=actor_cls, device_name=get_device_name())
        data = DataProto.from_dict({"a": torch.zeros(4)})
        actor_output = actor_worker.add(data)
        assert actor_worker.world_size == correct_world_size[idx]
        assert actor_output.batch["a"].tolist() == correct_output[idx]

    ray.shutdown()
