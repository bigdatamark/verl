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
# CPU Ray test for fused workers: builds a raw colocated base class from actor/critic workers at import time,
# subclasses it with a method calling both roles, and checks role-scoped calls via fuse() match the fused method.
# Side effects on import: creates RayClassWithInitArgs objects and the fused base class (no Ray init).

import ray

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
    create_colocated_worker_raw_cls,
)


@ray.remote
class Actor(Worker):
    # [CLASS DOC]
    # HELPER: Worker whose ONE_TO_ALL add returns x + rank.
    def __init__(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base Worker init only.
        super().__init__()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def add(self, x):
        # [FUNCTION DOC]
        # HELPER: Returns x plus the worker rank. Inputs: x - number.
        x += self.rank
        return x


@ray.remote
class Critic(Worker):
    # [CLASS DOC]
    # HELPER: Worker whose ALL_TO_ALL sub returns x - val.
    def __init__(self, val) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base init and stores val.
        super().__init__()
        self.val = val

    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def sub(self, x):
        # [FUNCTION DOC]
        # HELPER: Returns x minus the stored val (x is the per-worker element under ALL_TO_ALL).
        x -= self.val
        return x


actor_cls = RayClassWithInitArgs(cls=Actor)
critic_cls = RayClassWithInitArgs(cls=Critic, val=10)
cls_dict = {"actor": actor_cls, "critic": critic_cls}
FusedBaseClass = create_colocated_worker_raw_cls(cls_dict)


@ray.remote
class HybridWorker(FusedBaseClass):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    # [CLASS DOC]
    # HELPER: Ray worker built on the fused actor+critic base class, exposing role sub-objects self.actor and self.critic.
    def foo(self, x):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; returns critic.sub applied to actor.add(x) on the same process.
        return self.critic.sub(self.actor.add(x))


def test_fused_workers():
    # [FUNCTION DOC]
    # Behavior: inits Ray (100 CPUs), creates a 2-process CPU pool, marks the hybrid class as fused, creates the group and
    # fuses role names; computes actor.add(0.1) then critic.sub on the per-worker results, and asserts element-wise
    # equality with foo(0.1). Side effects: starts/stops Ray, prints.
    ray.init(num_cpus=100)

    # create separate workers on the same resource pool
    process_on_nodes = [2]
    resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=False)

    # create colocated workers
    hybrid_cls_with_init = RayClassWithInitArgs(cls=HybridWorker)
    hybrid_cls_with_init.fused_worker_used = True

    fused_wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=hybrid_cls_with_init)
    fused_wg.fuse(cls_dict.keys())

    x = fused_wg.actor.add(0.1)
    print(x)
    y = fused_wg.critic.sub(x)
    print(y)
    z = fused_wg.foo(0.1)
    print(z)
    for i, j in zip(y, z, strict=True):
        assert i == j

    ray.shutdown()


if __name__ == "__main__":
    test_fused_workers()
