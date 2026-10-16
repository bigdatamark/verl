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
# GPU Ray test that a driver-side function executed on workers via execute_with_func_generator gives the same result
# as calling it directly on the driver.
# Side effects on import: sets RAY_DEDUP_LOGS=0 and NCCL_DEBUG=WARN.

import os

import ray
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base.worker import Worker
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool
from verl.utils.device import get_device_name

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"


@ray.remote
class ModelActor(Worker):
    # [CLASS DOC]
    # HELPER: Empty Ray worker used to execute driver-provided functions.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Does nothing; intentionally skips the base Worker initializer.
        pass


class HackSelf:
    # [CLASS DOC]
    # HELPER: Dummy stand-in for a worker "self" when calling get_aux_metrics on the driver.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Does nothing.
        pass


def get_aux_metrics(self, test_proto):
    # [FUNCTION DOC]
    # HELPER: Worker-style function computing per-sequence decode counts.
    # Inputs: self - unused worker (or stand-in); test_proto - DataProto with "sequence_ids" (batch x len).
    # Outputs: new DataProto with "sequence_ids" and "decode_count" (per row, the full row length).
    sequence_ids = test_proto.batch["sequence_ids"]
    decode_count = []
    for i in range(sequence_ids.size(0)):
        decode_count.append(len(sequence_ids[i].tolist()))
    ret_proto = DataProto(
        batch=TensorDict(
            {"sequence_ids": sequence_ids, "decode_count": torch.tensor(decode_count)}, batch_size=sequence_ids.size(0)
        )
    )
    return ret_proto


def test():
    # construct model
    # [FUNCTION DOC]
    # Behavior: inits Ray, creates 2 GPU workers, builds an 8 x 2048 ones DataProto with meta query_length, runs
    # get_aux_metrics sharded on the workers and locally, and asserts decode_count matches.
    # Side effects: starts/stops Ray.
    ray.init()

    # create 2 workers, each hold a GPU
    resource_pool = RayResourcePool([2], use_gpu=True, name_prefix="a")

    class_with_args = RayClassWithInitArgs(cls=ModelActor)
    shard_wg = RayWorkerGroup(resource_pool, class_with_args, device_name=get_device_name())

    test_bs = 8
    test_proto = DataProto(
        TensorDict(
            {
                "sequence_ids": torch.ones([test_bs, 2048], dtype=torch.int64),
            },
            batch_size=test_bs,
        ),
        meta_info={"query_length": 1536},
    )

    # Sharding among different ranks
    ret_proto1 = shard_wg.execute_with_func_generator(get_aux_metrics, test_proto)

    # compare execute on driver
    hs = HackSelf()
    ret_proto2 = get_aux_metrics(hs, test_proto)

    torch.testing.assert_close(ret_proto1.batch["decode_count"], ret_proto2.batch["decode_count"])

    ray.shutdown()
