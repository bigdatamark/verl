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
# Tests of the LoRA weight-sync ordering used by engine_workers.update_weights, exercised against a local
# re-implementation of that logic with mocked rollout and actor engine (no GPU/Ray).
# KNOWN ISSUE: the tests exercise a copy of the production logic, not the real method, so they will not detect drift
# if engine_workers.update_weights changes.

"""Unit tests for LoRA weight sync ordering in engine_workers.update_weights().

Tests the branching logic that controls how weights are synced to the rollout:
  - Adapter mode (peft_merge=False): base weights first, then adapter deltas
  - Merge mode (peft_merge=True): single sync with merged weights
  - Non-LoRA: single sync, standard weights

These tests mock the actor engine and rollout to verify call ordering and
arguments without requiring GPU, ray, or sglang infrastructure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

# ---------------------------------------------------------------------------
# Helper: simulate engine_workers.update_weights() logic
# ---------------------------------------------------------------------------


async def _update_weights(
    *,
    rollout,
    actor_engine,
    peft_merge: bool,
    base_sync_done: bool,
    free_cache_engine: bool,
    layered_summon: bool = False,
    global_steps: int = None,
    checkpoint_backend: str = "naive",
):
    # [FUNCTION DOC]
    # HELPER: Async re-implementation of the worker's update_weights. Steps: if checkpoint_backend is not "naive", fetch
    # per-tensor params once (no arguments) and return without syncing. Otherwise, if free_cache_engine and the rollout's
    # sleep_level (default 2 when absent) is not 1, resume the rollout's "weights". Probe params and peft_config with
    # base_sync_done=True. In adapter mode (peft_merge false and peft_config present) set rollout.sleep_level to 1 and,
    # if the base was not yet synced, fetch base params (base_sync_done=False) and push them first with
    # base_sync_done=False. Then push the probed params with base_sync_done=True. Finally, if free_cache_engine, resume
    # "kv_cache". global_steps is forwarded to every push.
    # Inputs: rollout - async mock-like object; actor_engine - object with get_per_tensor_param; flags as named.
    # Outputs: none. Side effects: calls on rollout/engine; may set rollout.sleep_level.
    """Reproduce the update_weights() logic from engine_workers.py.

    This mirrors the actual code so we can test the branching without
    importing the real class (which requires ray, torch, etc.).
    """
    # 0. early return for non-naive checkpoint backend
    if checkpoint_backend != "naive":
        per_tensor_param, _ = actor_engine.get_per_tensor_param()
        return

    # 1. resume weights (conditional on sleep_level)
    if free_cache_engine:
        if getattr(rollout, "sleep_level", 2) != 1:
            await rollout.resume(tags=["weights"])

    # 2. probe adapter-mode params first so we can discover peft_config
    per_tensor_param, peft_config = actor_engine.get_per_tensor_param(
        layered_summon=layered_summon, base_sync_done=True
    )

    # 3. determine base sync need
    do_lora_base_sync = False
    if not peft_merge and peft_config is not None:
        rollout.sleep_level = 1
        do_lora_base_sync = not base_sync_done

    # 4. sync weights
    if do_lora_base_sync:
        per_tensor_param_base, peft_config = actor_engine.get_per_tensor_param(
            layered_summon=layered_summon, base_sync_done=False
        )
        await rollout.update_weights(
            per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
        )

    await rollout.update_weights(
        per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
    )

    # 5. resume kv_cache
    if free_cache_engine:
        await rollout.resume(tags=["kv_cache"])


def _make_mocks(peft_config=None, params_by_base_sync_done=None):
    # [FUNCTION DOC]
    # HELPER: Builds an AsyncMock rollout without a sleep_level attribute and a MagicMock actor engine whose
    # get_per_tensor_param returns (params chosen by the base_sync_done keyword, default True, from the given mapping;
    # peft_config). Default mapping returns "fake_params" for both. Outputs: (rollout, actor_engine).
    """Create mock rollout and actor engine.

    Args:
        peft_config: If not None, get_per_tensor_param returns this as peft_config.
            Use a truthy value (e.g. MagicMock()) for LoRA, None for non-LoRA/merge.
        params_by_base_sync_done: Optional mapping used to return different params
            for probe (`True`) and base sync (`False`) calls.
    """
    rollout = AsyncMock()
    # Don't pre-set sleep_level — let the code set it via getattr default
    del rollout.sleep_level

    if params_by_base_sync_done is None:
        params_by_base_sync_done = {False: "fake_params", True: "fake_params"}

    actor_engine = MagicMock()

    def _get_per_tensor_param(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Side-effect function selecting params by the base_sync_done keyword (default True) and returning them with peft_config.
        base_sync_done = kwargs.get("base_sync_done", True)
        return params_by_base_sync_done[base_sync_done], peft_config

    actor_engine.get_per_tensor_param = MagicMock(side_effect=_get_per_tensor_param)

    return rollout, actor_engine


# ---------------------------------------------------------------------------
# Adapter mode tests (peft_merge=False, peft_config is not None)
# ---------------------------------------------------------------------------


class TestAdapterModeFirstIteration:
    # [CLASS DOC]
    # Adapter mode, first iteration (base not yet synced): base sync must precede the adapter sync.
    """First iteration in adapter mode: base_sync_done=False."""

    def test_sends_base_before_adapter(self):
        # [FUNCTION DOC]
        # Engine is queried twice: probe with base_sync_done=True, then base with base_sync_done=False.
        """Base weights must be sent before adapter deltas."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(
            peft_config=peft_cfg,
            params_by_base_sync_done={False: "fake_base_params", True: "fake_adapter_params"},
        )

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        # get_per_tensor_param called twice: first probe with base_sync_done=True, then fetch base weights
        assert engine.get_per_tensor_param.call_count == 2
        calls = engine.get_per_tensor_param.call_args_list
        assert calls[0] == call(layered_summon=False, base_sync_done=True)
        assert calls[1] == call(layered_summon=False, base_sync_done=False)

    def test_update_weights_called_twice(self):
        # [FUNCTION DOC]
        # Rollout receives base params (base_sync_done False) then adapter params (True).
        """Two update_weights calls: base (base_sync_done=False), then adapter (True)."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(
            peft_config=peft_cfg,
            params_by_base_sync_done={False: "fake_base_params", True: "fake_adapter_params"},
        )

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        assert rollout.update_weights.call_count == 2
        first_call = rollout.update_weights.call_args_list[0]
        second_call = rollout.update_weights.call_args_list[1]
        assert first_call.args[0] == "fake_base_params"
        assert second_call.args[0] == "fake_adapter_params"
        assert first_call.kwargs["base_sync_done"] is False
        assert second_call.kwargs["base_sync_done"] is True

    def test_sets_sleep_level_to_1(self):
        # [FUNCTION DOC]
        # sleep_level becomes 1 after the first adapter sync.
        """After first iteration, sleep_level should be set to 1."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        assert rollout.sleep_level == 1

    def test_first_call_resumes_weights(self):
        # [FUNCTION DOC]
        # Both weights and kv_cache resumes occur on the first iteration.
        """First iteration: sleep_level not yet set, so weight resume fires."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        # resume called twice: weights then kv_cache
        resume_calls = rollout.resume.call_args_list
        assert call(tags=["weights"]) in resume_calls
        assert call(tags=["kv_cache"]) in resume_calls


class TestAdapterModeSubsequentIterations:
    # [CLASS DOC]
    # Adapter mode, later iterations (base synced, sleep_level 1): adapter-only sync, no weight resume.
    """Subsequent iterations in adapter mode: base_sync_done=True, sleep_level=1."""

    def test_single_update_weights_call(self):
        # [FUNCTION DOC]
        # With base already synced, only one probe and one adapter push (base_sync_done True).
        """Only adapter deltas sent, no base sync."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)
        rollout.sleep_level = 1  # Set from previous iteration

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        assert engine.get_per_tensor_param.call_count == 1
        assert rollout.update_weights.call_count == 1
        assert rollout.update_weights.call_args.kwargs["base_sync_done"] is True

    def test_skips_weight_resume(self):
        # [FUNCTION DOC]
        # With sleep_level 1 only the kv_cache resume happens.
        """With sleep_level=1, weight resume is skipped."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)
        rollout.sleep_level = 1

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        # Only kv_cache resume, no weight resume
        resume_calls = rollout.resume.call_args_list
        assert call(tags=["weights"]) not in resume_calls
        assert call(tags=["kv_cache"]) in resume_calls


# ---------------------------------------------------------------------------
# Merge mode tests (peft_merge=True)
# ---------------------------------------------------------------------------


class TestMergeMode:
    # [CLASS DOC]
    # Merge mode (peft_merge True, no peft_config): one standard sync and normal weight resume.
    """Merge mode: peft_merge=True, peft_config=None (merged into base)."""

    def test_single_update_weights_call(self):
        # [FUNCTION DOC]
        # Merge mode performs one probe and one push.
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=True,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        assert engine.get_per_tensor_param.call_count == 1
        assert rollout.update_weights.call_count == 1

    def test_resumes_weights(self):
        # [FUNCTION DOC]
        # Merge mode resumes weights.
        """Merge mode always resumes weights (sleep_level stays at default 2)."""
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=True,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        resume_calls = rollout.resume.call_args_list
        assert call(tags=["weights"]) in resume_calls

    def test_does_not_set_sleep_level(self):
        # [FUNCTION DOC]
        # Merge mode leaves sleep_level unset.
        """Merge mode should not touch sleep_level."""
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=True,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        assert not hasattr(rollout, "sleep_level")


# ---------------------------------------------------------------------------
# Non-LoRA tests
# ---------------------------------------------------------------------------


class TestNonLora:
    # [CLASS DOC]
    # Non-LoRA models: one standard sync; the adapter path is never entered.
    """Non-LoRA model: peft_config=None, peft_merge=False."""

    def test_single_update_weights_call(self):
        # [FUNCTION DOC]
        # Non-LoRA path performs one probe and one push.
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=True,
                free_cache_engine=True,
            )
        )

        assert engine.get_per_tensor_param.call_count == 1
        assert rollout.update_weights.call_count == 1

    def test_peft_config_none_skips_adapter_path(self):
        # [FUNCTION DOC]
        # With peft_config None, no sleep_level is set and only one push happens even if base_sync_done is False.
        """Even with peft_merge=False, if peft_config is None, adapter path is not entered."""
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        # Should NOT set sleep_level or do double sync
        assert not hasattr(rollout, "sleep_level")
        assert rollout.update_weights.call_count == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    # [CLASS DOC]
    # Edge cases: no resume without free_cache_engine, full call ordering, global_steps forwarding, non-naive early return.
    def test_free_cache_engine_false_skips_resume(self):
        # [FUNCTION DOC]
        # No resume calls when free_cache_engine is False.
        """When free_cache_engine=False, no resume calls should happen."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=False,
            )
        )

        rollout.resume.assert_not_called()

    def test_adapter_full_call_ordering(self):
        # [FUNCTION DOC]
        # First adapter iteration produces exactly: resume weights, push base, push adapter, resume kv_cache, with global_steps 42.
        """Verify the complete call sequence on first adapter iteration."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(
            peft_config=peft_cfg,
            params_by_base_sync_done={False: "fake_base_params", True: "fake_adapter_params"},
        )

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
                global_steps=42,
            )
        )

        # Full expected ordering:
        # 1. resume(weights)  2. update_weights(base)  3. update_weights(adapter)  4. resume(kv_cache)
        expected = [
            call.resume(tags=["weights"]),
            call.update_weights("fake_base_params", peft_config=peft_cfg, base_sync_done=False, global_steps=42),
            call.update_weights("fake_adapter_params", peft_config=peft_cfg, base_sync_done=True, global_steps=42),
            call.resume(tags=["kv_cache"]),
        ]
        # Filter to only resume and update_weights calls
        actual = [c for c in rollout.mock_calls if c[0] in ("resume", "update_weights")]
        assert actual == expected

    def test_global_steps_forwarded(self):
        # [FUNCTION DOC]
        # global_steps reaches the push call.
        """Verify global_steps is passed through to update_weights."""
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=True,
                base_sync_done=True,
                free_cache_engine=True,
                global_steps=99,
            )
        )

        assert rollout.update_weights.call_args.kwargs["global_steps"] == 99

    def test_non_naive_backend_early_return(self):
        # [FUNCTION DOC]
        # A non-naive backend performs no push and no resume.
        """Non-naive checkpoint backend returns early, skips all LoRA logic."""
        peft_cfg = MagicMock()
        rollout, engine = _make_mocks(peft_config=peft_cfg)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
                checkpoint_backend="disaggregated",
            )
        )

        rollout.update_weights.assert_not_called()
        rollout.resume.assert_not_called()

    def test_non_lora_probe_still_uses_base_sync_done_true(self):
        # [FUNCTION DOC]
        # Non-LoRA probe is a single call with base_sync_done=True and the push uses base_sync_done True.
        """Non-LoRA path still probes with base_sync_done=True and sends a standard update."""
        rollout, engine = _make_mocks(peft_config=None)

        asyncio.run(
            _update_weights(
                rollout=rollout,
                actor_engine=engine,
                peft_merge=False,
                base_sync_done=False,
                free_cache_engine=True,
            )
        )

        engine.get_per_tensor_param.assert_called_once_with(layered_summon=False, base_sync_done=True)
        assert rollout.update_weights.call_args.kwargs["base_sync_done"] is True
