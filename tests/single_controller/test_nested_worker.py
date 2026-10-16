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
# GPU Ray test that a Worker can hold another Worker instance in-process (nested worker) and that registered
# ONE_TO_ALL methods work for both the plain and the nesting worker groups.

import ray

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name


class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # [CLASS DOC]
    # HELPER: Worker storing value x; get returns x + rank.
    def __init__(self, x) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base Worker init and stores x as self.a.
        super().__init__()
        self.a = x

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get(self):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; returns self.a + self.rank.
        return self.a + self.rank


class TestHighLevelActor(Worker):
    # [CLASS DOC]
    # HELPER: Worker that constructs a local (non-remote) TestActor and delegates get to it.
    def __init__(self, x=None) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base init and creates an inner TestActor with the same x (default None); the inner worker reads the same rank env.
        super().__init__()
        self.test_actor = TestActor(x=x)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get(self):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; returns the inner actor's get(), i.e. x + rank.
        return self.test_actor.get()


def test_nested_worker():
    # [FUNCTION DOC]
    # Behavior: inits Ray, creates a 4-GPU pool, a group of TestActor(x=2) and asserts get returns [2,3,4,5]; then a group of
    # TestHighLevelActor(x=2) on the same pool returns the same. Side effects: starts/stops Ray.
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=ray.remote(TestActor), x=2)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_basic",
        device_name=get_device_name(),
    )

    output = worker_group.get()

    assert output == [2, 3, 4, 5]

    class_with_args = RayClassWithInitArgs(cls=ray.remote(TestHighLevelActor), x=2)
    high_level_worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_basic_2",
        device_name=get_device_name(),
    )

    output_1 = high_level_worker_group.get()

    assert output_1 == [2, 3, 4, 5]

    ray.shutdown()
