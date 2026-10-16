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
# GPU Ray test for verl's Ray-based NCCL rendezvous: two plain Ray actors create a communicator via
# create_nccl_communicator_in_ray and report their rank ids.

import ray


@ray.remote
class TestWorker:
    # [CLASS DOC]
    # HELPER: Plain Ray actor (not a verl Worker) holding rank, world size, group name and an optional communicator.
    def __init__(self, rank, world_size, group_name):
        # [FUNCTION DOC]
        # HELPER: Stores rank, world_size and group_name; communicator starts as None.
        self.rank = rank
        self.world_size = world_size
        self.group_name = group_name
        self.communicator = None

    def init(self):
        # [FUNCTION DOC]
        # HELPER: Lazily imports and calls create_nccl_communicator_in_ray, storing the communicator.
        from verl.utils.rendezvous.ray_backend import create_nccl_communicator_in_ray

        self.communicator = create_nccl_communicator_in_ray(self.rank, self.world_size, self.group_name)

    def test(self):
        # [FUNCTION DOC]
        # HELPER: Returns the communicator's rank id, or None if init was not called.
        if self.communicator is None:
            return None
        return self.communicator.rank_id()


def test_rvdz():
    # [FUNCTION DOC]
    # Behavior: inits Ray, creates 2 actors with 1 GPU each in group "test_group", inits both concurrently and asserts ranks are [0, 1].
    # Side effects: starts/stops Ray.
    ray.init()

    group_name = "test_group"
    world_size = 2

    workers = [TestWorker.options(num_gpus=1).remote(rank, world_size, group_name) for rank in range(world_size)]

    ray.get([worker.init.remote() for worker in workers])

    ranks = ray.get([worker.test.remote() for worker in workers])

    assert ranks == [0, 1], f"expecting [0, 1], got {ranks}"

    ray.shutdown()
