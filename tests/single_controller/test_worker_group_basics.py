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
# GPU Ray end-to-end tests of RayWorkerGroup basics: execute_all_sync/async, registered dispatch modes (ONE_TO_ALL,
# ALL_TO_ALL, RANK_ZERO execute, custom dispatch/collect), reattaching to workers from another remote task via
# from_detached, and execute_func_rank_zero.

"""
e2e test verl.single_controller.ray
"""

import ray
import torch

from verl.single_controller.base.decorator import Dispatch, Execute, collect_all_to_all, register
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.device import get_device_name


def two_to_all_dispatch_fn(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Custom dispatch function. Each positional and keyword argument must be a list of length 2; it is extended
    # in place by appending alternating original elements until it has world_size entries.
    # Inputs: worker_group (uses world_size), args, kwargs. Outputs: the (mutated) args tuple and kwargs mapping.
    # Exceptions: AssertionError if any argument length is not 2.
    # Side effects: mutates the caller's lists.
    """
    Assume the input is a list of 2. Duplicate the input interleaved and pass to each worker.
    """
    for arg in args:
        assert len(arg) == 2
        for i in range(worker_group.world_size - 2):
            arg.append(arg[i % 2])
    for k, v in kwargs.items():
        assert len(v) == 2
        for i in range(worker_group.world_size - 2):
            v.append(v[i % 2])
    return args, kwargs


def get_ray_remote_options() -> str:
    # [FUNCTION DOC]
    # HELPER: Returns Ray resource options for a tiny fractional allocation: 0.1 GPU on CUDA, 0.1 of the "NPU" custom
    # resource on NPU, otherwise 0.1 CPU.
    # Outputs: a mapping of Ray options. Not used elsewhere in this module.
    # KNOWN ISSUE: annotated and documented as returning a str/torch.device, but it returns a dict of Ray options.
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if get_device_name() == "cuda":
        return dict(num_gpus=0.1)
    elif get_device_name() == "npu":
        return dict(resources={"NPU": 0.1})
    return dict(num_cpus=0.1)


@ray.remote
class TestActor(Worker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    # [CLASS DOC]
    # HELPER: Worker storing x and exposing methods computing x + y (+ x arg) under various dispatch modes.
    def __init__(self, x) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls base init and stores x.
        super().__init__()
        self._x = x

    def foo(self, y):
        # [FUNCTION DOC]
        # HELPER: Unregistered method returning stored x + y.
        return self._x + y

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def foo_rank_zero(self, x, y):
        # [FUNCTION DOC]
        # HELPER: ALL_TO_ALL dispatch executed on rank zero only; returns stored x + y + x argument.
        return self._x + y + x

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def foo_one_to_all(self, x, y):
        # [FUNCTION DOC]
        # HELPER: Non-blocking ONE_TO_ALL; returns stored x + y + x argument.
        return self._x + y + x

    @register(Dispatch.ALL_TO_ALL, blocking=False)
    def foo_all_to_all(self, x, y):
        # [FUNCTION DOC]
        # HELPER: Non-blocking ALL_TO_ALL (per-rank elements); returns stored x + y + x argument.
        return self._x + y + x

    @register(dispatch_mode={"dispatch_fn": two_to_all_dispatch_fn, "collect_fn": collect_all_to_all})
    def foo_custom(self, x, y):
        # [FUNCTION DOC]
        # HELPER: Uses two_to_all_dispatch_fn and collect_all_to_all; returns stored x + y + x argument.
        return self._x + y + x


@ray.remote(num_cpus=0.1)
def remote_call_wg(worker_names):
    # [FUNCTION DOC]
    # HELPER: Ray remote task (0.1 CPU) that rebuilds a worker group from worker names, prints them, asserts foo_custom
    # with x=[1,2], y=[5,6] gives [8,10,8,10] and foo_rank_zero(1,2) gives 5, and returns the worker names.
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)
    worker_group = RayWorkerGroup.from_detached(
        worker_names=worker_names, ray_cls_with_init=class_with_args, name_prefix=None
    )
    print(worker_group.worker_names)

    output_ref = worker_group.foo_custom(x=[1, 2], y=[5, 6])
    assert output_ref == [8, 10, 8, 10]

    output_ref = worker_group.foo_rank_zero(x=1, y=2)
    assert output_ref == 5

    return worker_group.worker_names


def add_one(data):
    # [FUNCTION DOC]
    # HELPER: Moves a tensor to the current device, adds 1, moves it back to CPU and returns it.
    data = data.to(get_device_name())
    data += 1
    data = data.to("cpu")
    return data


def test_basics():
    # [FUNCTION DOC]
    # Behavior: inits Ray, creates 4 GPU workers with x=2; checks execute_all_sync/async of foo, the one-to-all and
    # all-to-all methods, the remote reattach task, and execute_func_rank_zero with add_one on a 2x2 ones tensor.
    # Side effects: starts/stops Ray, prints.
    ray.init(num_cpus=100)

    # create 4 workers, each hold a GPU
    resource_pool = RayResourcePool([4], use_gpu=True)
    class_with_args = RayClassWithInitArgs(cls=TestActor, x=2)

    worker_group = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=class_with_args,
        name_prefix="worker_group_basic",
        device_name=get_device_name(),
    )

    print(worker_group.worker_names)

    # this will wait for all the results
    output = worker_group.execute_all_sync("foo", y=3)
    assert output == [5, 5, 5, 5]

    # this is a list of object reference. It won't block.
    output_ref = worker_group.execute_all_async("foo", y=4)
    print(output_ref)

    assert ray.get(output_ref) == [6, 6, 6, 6]

    output_ref = worker_group.foo_one_to_all(x=1, y=2)
    assert ray.get(output_ref) == [5, 5, 5, 5]

    output_ref = worker_group.foo_all_to_all(x=[1, 2, 3, 4], y=[5, 6, 7, 8])
    assert ray.get(output_ref) == [8, 10, 12, 14]

    print(ray.get(remote_call_wg.remote(worker_group.worker_names)))

    output = worker_group.execute_func_rank_zero(add_one, torch.ones(2, 2))
    torch.testing.assert_close(output, torch.ones(2, 2) + 1)

    ray.shutdown()


if __name__ == "__main__":
    test_basics()
