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
# GPU Ray test for point-to-point NCCL transfer using ray.util.collective groups between an actor group and a rollout
# group with a 2:1 ratio: actor rank r pairs with rollout rank r // 2 in a 2-member group named "A{r}_R{r//2}"
# (actor is member 0, rollout member 1). Each actor sends a 4-element tensor filled with its rank.
# KNOWN ISSUE: the module docstring says "Rollout rank 2, 3 - Rollout rank 1" (should be Actor ranks 2, 3) and fixes counts at 4/2, while the test sizes them from the GPU count (rollout = max(1, ngpus // 3), actor = 2 x rollout).

"""
Test for using ray collective group.
Suppose we Actor and Rollout. Actor contains 4 workers and Rollout contains 2 workers. We established a Worker to
Rollout relationship by using collective groups
Actor: rank 0, 1 - Rollout rank 0
Rollout rank 2, 3 - Rollout rank 1
Then, we initiate 4 p2p comms from actor to rollout
"""

import ray
import ray.util.collective as collective
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@ray.remote
class Actor(Worker):
    @register(Dispatch.ONE_TO_ALL)
    # [CLASS DOC]
    # HELPER: Sending-side worker.
    def init(self):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; computes the paired rollout rank, stores the group name and joins it as rank 0 with NCCL.
        # Side effects: creates a collective group.
        remote_rank = self.rank // 2
        self.group_name = f"A{self.rank}_R{remote_rank}"
        collective.init_collective_group(world_size=2, rank=0, backend="nccl", group_name=self.group_name)

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def send_tensors(self):
        # [FUNCTION DOC]
        # HELPER: Non-blocking ONE_TO_ALL; sends a CUDA tensor of four copies of its rank to group member 1.
        tensor = torch.ones(size=(4,), dtype=torch.float32, device="cuda") * self.rank
        collective.send(tensor=tensor, dst_rank=1, group_name=self.group_name)


@ray.remote
class Rollout(Worker):
    @register(Dispatch.ONE_TO_ALL)
    # [CLASS DOC]
    # HELPER: Receiving-side worker paired with two actors.
    def init(self):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; computes paired actor ranks 2*rank and 2*rank+1, stores both group names and joins both as rank 1.
        self.remote_first_rank = self.rank * 2
        self.remote_second_rank = self.remote_first_rank + 1
        self.first_group_name = f"A{self.remote_first_rank}_R{self.rank}"
        self.second_group_name = f"A{self.remote_second_rank}_R{self.rank}"

        collective.init_collective_group(world_size=2, rank=1, backend="nccl", group_name=self.first_group_name)
        collective.init_collective_group(world_size=2, rank=1, backend="nccl", group_name=self.second_group_name)

    @register(Dispatch.ONE_TO_ALL, blocking=False)
    def receive_tensors(self):
        # [FUNCTION DOC]
        # HELPER: Non-blocking ONE_TO_ALL; allocates two random CUDA buffers and receives into them from member 0 of each group.
        # Side effects: stores tensor1/tensor2 on self.
        self.tensor1 = torch.randn(size=(4,), dtype=torch.float32, device="cuda")
        self.tensor2 = torch.randn(size=(4,), dtype=torch.float32, device="cuda")

        collective.recv(self.tensor1, src_rank=0, group_name=self.first_group_name)
        collective.recv(self.tensor2, src_rank=0, group_name=self.second_group_name)

    @register(Dispatch.ONE_TO_ALL)
    def get_tensors(self):
        # [FUNCTION DOC]
        # HELPER: ONE_TO_ALL; returns a mapping "src_{actor rank}" to the received tensor for both paired actors.
        return {f"src_{self.remote_first_rank}": self.tensor1, f"src_{self.remote_second_rank}": self.tensor2}


def test_ray_collective_group():
    # [FUNCTION DOC]
    # Behavior: inits Ray, sizes groups from GPU count, inits both groups, launches sends and receives concurrently, waits,
    # merges the per-rollout dictionaries and asserts each tensor from actor i sums to 4*i.
    # Side effects: starts/stops Ray, prints.
    ray.init()
    ngpus = torch.cuda.device_count()
    n_rollout = max(1, ngpus // 3)  # keep 2:1 actor:rollout ratio
    n_actor = n_rollout * 2

    actor_resource_pool = RayResourcePool([n_actor])
    rollout_resource_pool = RayResourcePool([n_rollout])

    actor_cls = RayClassWithInitArgs(cls=Actor)
    rollout_cls = RayClassWithInitArgs(cls=Rollout)

    actor_wg = RayWorkerGroup(
        resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls, name_prefix="collective_group_actor"
    )
    rollout_wg = RayWorkerGroup(
        resource_pool=rollout_resource_pool, ray_cls_with_init=rollout_cls, name_prefix="collective_group_rollout"
    )

    actor_wg.init()
    rollout_wg.init()

    out1 = actor_wg.send_tensors()
    out2 = rollout_wg.receive_tensors()

    # block to wait
    ray.get(out1)
    ray.get(out2)

    output = {}
    for d in rollout_wg.get_tensors():
        output |= d

    print(output)

    for i in range(n_actor):
        assert torch.sum(output[f"src_{i}"]).item() == 4 * i

    ray.shutdown()


if __name__ == "__main__":
    test_ray_collective_group()
