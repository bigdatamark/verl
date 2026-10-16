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
# Multi-GPU Ray test of the high-level scheduling API: several worker groups sharing one resource pool, and merged
# resource pools, checking each worker's CUDA_VISIBLE_DEVICES assignment.

import gc
import time

import ray

from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, merge_resource_pool
from verl.utils.device import get_device_name


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # [CLASS DOC]
    # HELPER: Worker that forwards an optional cuda_visible_devices to the base Worker and can report its node id.
    def __init__(self, cuda_visible_devices=None) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base Worker init with cuda_visible_devices passed positionally.
        super().__init__(cuda_visible_devices)

    def get_node_id(self):
        # [FUNCTION DOC]
        # HELPER: Returns the Ray node id of the current actor process.
        return ray.get_runtime_context().get_node_id()


def test():
    # [FUNCTION DOC]
    # Behavior: inits Ray; with one pool over all GPUs creates actor/critic/ref groups and asserts each sees devices
    # "0".."n-1"; deletes them, removes placement groups and sleeps 5s; then creates two half pools, merges them,
    # checks world sizes, and asserts groups on the merged pool see all GPUs while the ref group on the second half pool
    # sees "half".."n-1".
    # Side effects: starts/stops Ray, sleeps, removes placement groups, prints.
    import torch

    ray.init()
    ngpus = torch.cuda.device_count()
    half = ngpus // 2

    # test single-node-no-partition
    print("test single-node-no-partition")
    resource_pool = RayResourcePool([ngpus], use_gpu=True)

    class_with_args = RayClassWithInitArgs(cls=TestActor)

    print("create actor worker group")
    actor_wg = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="high_level_api_actor", device_name=get_device_name()
    )
    print("create critic worker group")
    critic_wg = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="hight_level_api_critic", device_name=get_device_name()
    )
    print("create ref worker group")
    ref_wg = RayWorkerGroup(
        resource_pool, class_with_args, name_prefix="high_level_api_ref", device_name=get_device_name()
    )

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(ngpus)]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(ngpus)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(ngpus)]

    del actor_wg
    del critic_wg
    del ref_wg
    gc.collect()  # make sure ray actors are deleted

    [ray.util.remove_placement_group(pg) for pg in resource_pool.get_placement_groups()]
    print("wait 5s to remove placemeng_group")
    time.sleep(5)
    # test single-node-multi-partition

    print("test single-node-multi-partition")
    rm_resource_pool = RayResourcePool([half], use_gpu=True, name_prefix="rm")
    ref_resource_pool = RayResourcePool([half], use_gpu=True, name_prefix="ref")
    total_resource_pool = merge_resource_pool(rm_resource_pool, ref_resource_pool)

    assert rm_resource_pool.world_size == half
    assert ref_resource_pool.world_size == half
    assert total_resource_pool.world_size == ngpus

    actor_wg = RayWorkerGroup(
        total_resource_pool, class_with_args, name_prefix="high_level_api_actor", device_name=get_device_name()
    )
    critic_wg = RayWorkerGroup(
        total_resource_pool, class_with_args, name_prefix="high_level_api_critic", device_name=get_device_name()
    )
    ref_wg = RayWorkerGroup(
        ref_resource_pool, class_with_args, name_prefix="high_level_api_ref", device_name=get_device_name()
    )

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(ngpus)]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(ngpus)]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(half, ngpus)]

    ray.shutdown()
