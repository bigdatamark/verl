# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# CPU regression test that hybrid-mode sleep of vLLMHttpServer with MTP (multi-token prediction) speculative
# decoding enabled does not use a sleep level that discards the MTP drafter, which would zero the
# speculative acceptance rate. Requires ray and vllm (skipped otherwise).

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ray")
pytest.importorskip("vllm")

from verl.trainer.ppo.ray_trainer import compute_spec_decode_metrics
from verl.workers.rollout.vllm_rollout import vllm_async_server


class _FakeMtpEngine:
    # [CLASS DOC]
    # HELPER: fake engine modelling the failure mode: sleeping at level 2 marks the MTP drafter unavailable.
    """Minimal model of the MTP failure mode seen in hybrid sleep."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: drafter starts available; the set of drafter-discarding sleep levels is {2}.
        self.mtp_drafter_available = True
        self.sleep_levels_that_discard_mtp_drafter = {2}

    async def sleep(self, level: int):
        # [FUNCTION DOC]
        # HELPER: async; marks the drafter unavailable if the requested level discards it. Inputs: level.
        if level in self.sleep_levels_that_discard_mtp_drafter:
            self.mtp_drafter_available = False

    async def reset_encoder_cache(self):
        # [FUNCTION DOC]
        # HELPER: async no-op.
        pass

    def sync_actor_weights(self):
        # [FUNCTION DOC]
        # HELPER: no-op standing in for weight sync after wake.
        pass

    def generate_spec_decode_stats(self):
        # [FUNCTION DOC]
        # HELPER: returns (3 drafts, 3 accepted if drafter available else 0, 1 verify step).
        num_draft_tokens = 3
        num_accepted_tokens = num_draft_tokens if self.mtp_drafter_available else 0
        num_verify_steps = 1
        return num_draft_tokens, num_accepted_tokens, num_verify_steps


def test_mtp_hybrid_sleep_keeps_drafter_available_for_nonzero_acceptance(monkeypatch):
    # [FUNCTION DOC]
    # Patches NPU detection to False, creates a vLLMHttpServer without __init__ with MTP enabled for rollout, no LoRA
    # and the fake engine, runs _sleep_hybrid, syncs weights, computes spec decode metrics from the fake stats and
    # asserts acceptance rate > 0 and acceptance length > 1.
    monkeypatch.setattr(vllm_async_server, "is_torch_npu_available", lambda check_device=False: False)

    server = object.__new__(vllm_async_server.vLLMHttpServer)
    server.config = SimpleNamespace(mtp=SimpleNamespace(enable=True, enable_rollout=True))
    server.model_config = SimpleNamespace(lora_rank=0, lora={})
    server.engine = _FakeMtpEngine()

    asyncio.run(server._sleep_hybrid())
    server.engine.sync_actor_weights()
    drafts, accepts, verifies = server.engine.generate_spec_decode_stats()
    metrics = compute_spec_decode_metrics(
        spec_drafts=np.array([drafts]),
        spec_accepts=np.array([accepts]),
        spec_verifies=np.array([verifies]),
    )

    assert metrics["rollout/spec_accept_rate"] > 0.0
    assert metrics["rollout/spec_accept_length"] > 1.0
