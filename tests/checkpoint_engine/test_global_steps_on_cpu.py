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
# CPU unit tests that global_steps is propagated through weight sync: from
# ActorRolloutRefWorker.update_weights into checkpoint_engine.send_weights, and
# from CheckpointEngineWorker.update_weights into receive_weights and the rollout
# server adapter. Uses fake engines and bypasses worker __init__ and the
# registered-method decorator (calls the __wrapped__ function directly).

import asyncio
from types import SimpleNamespace

from verl.checkpoint_engine.base import CheckpointEngineWorker
from verl.workers.engine_workers import ActorRolloutRefWorker


class _FakeTrainerEngine:
    # [CLASS DOC]
    # HELPER: fake trainer engine exposing a single fixed weight entry.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: stores weights as a list with one ("w", sentinel object) pair.
        self.weights = [("w", object())]

    def get_per_tensor_param(self):
        # [FUNCTION DOC]
        # HELPER: returns an iterator over the weights and None (no extra metadata).
        return iter(self.weights), None


class _FakeCheckpointEngine:
    # [CLASS DOC]
    # HELPER: fake checkpoint engine recording global_steps and weights it sees.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: initializes recorded sent/received global_steps and sent weights to None.
        self.sent_global_steps = None
        self.received_global_steps = None
        self.sent_weights = None

    async def send_weights(self, weights, global_steps=None):
        # [FUNCTION DOC]
        # HELPER: async; records global_steps and materializes the weights iterable into a list.
        self.sent_global_steps = global_steps
        self.sent_weights = list(weights)

    def receive_weights(self, global_steps=None):
        # [FUNCTION DOC]
        # HELPER: records global_steps and returns an async generator yielding a single ("w", object) pair.
        self.received_global_steps = global_steps

        async def _weights():
            # [FUNCTION DOC]
            # HELPER: async generator yielding one fake weight.
            yield "w", object()

        return _weights()


class _FakeServerAdapter:
    # [CLASS DOC]
    # HELPER: fake rollout server adapter recording update_weights calls.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: initializes recorded global_steps and weights to None.
        self.global_steps = None
        self.weights = None

    async def update_weights(self, weights, global_steps=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: async; records global_steps and drains the async weights iterator into a list. Extra kwargs ignored.
        self.global_steps = global_steps
        self.weights = [item async for item in weights]


def test_actor_worker_passes_global_steps_to_checkpoint_engine_send():
    # [FUNCTION DOC]
    # Behavior: builds a bare ActorRolloutRefWorker with config selecting the
    # "modelexpress" checkpoint backend, a fake actor engine and fake checkpoint
    # engine; runs unwrapped update_weights(global_steps=17, mode="auto") and asserts
    # the engine got global_steps 17 and exactly the trainer's weights.
    checkpoint_engine = _FakeCheckpointEngine()
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="modelexpress"),
        ),
    )
    worker.actor = SimpleNamespace(engine=_FakeTrainerEngine())
    worker.checkpoint_engine = checkpoint_engine

    asyncio.run(
        ActorRolloutRefWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
            mode="auto",
        )
    )

    assert checkpoint_engine.sent_global_steps == 17
    assert checkpoint_engine.sent_weights == worker.actor.engine.weights


def test_checkpoint_worker_passes_global_steps_to_receive_and_rollout_update():
    # [FUNCTION DOC]
    # Behavior: builds a bare CheckpointEngineWorker with fake checkpoint engine and
    # server adapter; runs unwrapped update_weights(global_steps=17) and asserts both
    # receive_weights and the adapter got 17, and the adapter received weight "w".
    checkpoint_engine = _FakeCheckpointEngine()
    server_adapter = _FakeServerAdapter()
    worker = CheckpointEngineWorker.__new__(CheckpointEngineWorker)
    worker.checkpoint_engine = checkpoint_engine
    worker.server_adapter = server_adapter

    asyncio.run(
        CheckpointEngineWorker.update_weights.__wrapped__(
            worker,
            global_steps=17,
        )
    )

    assert checkpoint_engine.received_global_steps == 17
    assert server_adapter.global_steps == 17
    assert [name for name, _tensor in server_adapter.weights] == ["w"]
