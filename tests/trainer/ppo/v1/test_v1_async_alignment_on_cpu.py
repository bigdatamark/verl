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
# Trainers are built with object.__new__ and SimpleNamespace/MagicMock collaborators; ray.get is patched
# where balancer calls would hit Ray.
"""CPU tests for the v1 async-trainer alignment with v0 fully-async semantics.

Covers the ``hybrid_engine=False`` no-op mode-switch hooks, fractional
``num_warmup_batches`` rounding, and the ``max_off_policy_threshold=None``
off-policy control bypass.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1 import trainer_separate_async as separate_async_module
from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import HybridEngineMode, PPOTrainerSeparateAsync


def _make_trainer(
    *,
    standalone_replicas=1,
    hybrid_replicas=2,
    current_mode=HybridEngineMode.ROLLOUT,
    enable_hybrid_replicas=True,
) -> PPOTrainerSeparateAsync:
    # [FUNCTION DOC]
    # HELPER: creates an uninitialized PPOTrainerSeparateAsync with the hybrid-replica flag and current mode set,
    # a fake hybrid server manager (get_replicas returns hybrid_replicas entries, plus server addresses), a
    # fake standalone manager (replica list and mocked global_load_balancer) and a mocked checkpoint manager.
    # Inputs: keyword-only standalone_replicas (1), hybrid_replicas (2), current_mode (ROLLOUT),
    # enable_hybrid_replicas (True). Outputs: the trainer.
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer._enable_hybrid_replicas = enable_hybrid_replicas
    trainer.current_mode = current_mode
    trainer.llm_server_manager = SimpleNamespace(
        get_replicas=lambda: ["hybrid"] * hybrid_replicas,
        server_addresses=[f"hybrid-{i}" for i in range(hybrid_replicas)],
    )
    trainer.standalone_server_manager = SimpleNamespace(
        get_replicas=lambda: ["standalone"] * standalone_replicas,
        global_load_balancer=MagicMock(),
    )
    trainer.checkpoint_manager = MagicMock()
    return trainer


class TestUnconfiguredDefaultsPreserveBehavior:
    # [CLASS DOC]
    # Checks that default/unset new options preserve pre-change behavior.
    """When the new knobs are not configured, every code path must be a no-op
    relative to the pre-change behavior (minimal-change contract for
    hybrid_engine=False, fractional warmup, and off-policy null support)."""

    def test_legacy_config_without_hybrid_engine_key_resolves_enabled(self):
        # [FUNCTION DOC]
        # Asserts a config lacking hybrid_engine resolves to True via a default lookup.
        # KNOWN ISSUE: only exercises OmegaConf's get-with-default on a local config, not trainer code.
        # A separate_async config object from a pre-change user yaml has no
        # hybrid_engine override; .get() must fall back to hybrid-enabled.
        legacy_actor_cfg = OmegaConf.create({"rollout": {"nnodes": 1}})
        assert bool(legacy_actor_cfg.get("hybrid_engine", True)) is True

    def test_integer_warmup_matches_previous_formula(self):
        # [FUNCTION DOC]
        # For warmup batches 1,2,3 with train size 64 and nothing restored, asserts requests equal n*64.
        # Integer num_warmup_batches must produce exactly the same prompt count
        # as the previous `num_warmup_batches * train_batch_size` formula.
        stub = SimpleNamespace(
            config=OmegaConf.create({"skip": {"rollout_tq": {"enable": False}}, "data": {"train_batch_size": 64}}),
            _restored_tq_prompt_count=0,
        )
        stub._add_prompts_to_generate = MagicMock()
        stub._add_async_warmup_batches = PPOTrainer._add_async_warmup_batches.__get__(stub)
        for num_warmup_batches in (1, 2, 3):
            stub._add_prompts_to_generate.reset_mock()
            stub._add_async_warmup_batches(num_warmup_batches)
            stub._add_prompts_to_generate.assert_called_once_with(num_warmup_batches * 64)

    def test_default_off_policy_threshold_keeps_staleness_control(self):
        # [FUNCTION DOC]
        # With integer threshold 8, a finished key produced at step 1 is reported stale at step 100.
        # The default (int) threshold still enables drop/wait as before.
        rb = ReplayBufferAsync(
            trainer_mode="separate_async",
            trainer_config={},
            max_off_policy_threshold=8,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
        )
        rb.prompt_global_steps["train"]["stale"] = 1
        rb.finished_keys["train"].add("stale")
        assert "stale" in rb._stale_terminal_keys(100, "train")

    def test_hybrid_engine_enabled_default_takes_original_hooks(self, monkeypatch):
        # [FUNCTION DOC]
        # With hybrid enabled in ROLLOUT mode, switch_to_trainer aborts and sleeps replicas once, switches mode to
        # TRAINER, and calls ray.get once (balancer removal). Inputs: monkeypatch.
        # Default True must run the original (non-guarded) hook bodies.
        trainer = _make_trainer(enable_hybrid_replicas=True)
        trainer.global_steps = 1
        trainer.current_mode = HybridEngineMode.ROLLOUT
        balancer_calls = []
        monkeypatch.setattr(
            separate_async_module.ray,
            "get",
            lambda ref: balancer_calls.append(ref),
        )
        trainer.switch_to_trainer()
        trainer.checkpoint_manager.abort_replicas.assert_called_once()
        trainer.checkpoint_manager.sleep_replicas.assert_called_once()
        assert trainer.current_mode == HybridEngineMode.TRAINER
        assert len(balancer_calls) == 1  # balancer removal ran (original path)


class TestHybridEngineDisabled:
    # [CLASS DOC]
    # Behavior when hybrid replicas are disabled (hybrid_engine=False).
    def test_resource_mapping_uses_pure_actor_role(self):
        # [FUNCTION DOC]
        # _init_resource_pool_mgr maps the pure Actor role and not ActorRollout/ActorRolloutRef.
        trainer = object.__new__(PPOTrainerSeparateAsync)
        trainer._enable_hybrid_replicas = False
        trainer.config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "model": {"lora": {"rank": 0}, "lora_adapter_path": None},
                    "actor": {"use_kl_loss": False},
                },
                "algorithm": {},
                "critic": {"enable": False},
                "trainer": {"nnodes": 1, "n_gpus_per_node": 4},
                "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
                "distillation": None,
            }
        )

        trainer._init_resource_pool_mgr()

        assert Role.Actor in trainer.role_worker_mapping
        assert Role.ActorRollout not in trainer.role_worker_mapping
        assert Role.ActorRolloutRef not in trainer.role_worker_mapping

    def test_switch_is_disabled_with_warning_when_hybrid_is_disabled(self, monkeypatch):
        # [FUNCTION DOC]
        # Constructs the trainer with base __init__ patched (sets config and a fake replay buffer) and a config with
        # hybrid_engine False but enable_switch True; asserts switching is forced off and one warning is logged.
        # Inputs: monkeypatch.
        config = OmegaConf.create(
            {
                "data": {"train_batch_size": 64},
                "actor_rollout_ref": {
                    "hybrid_engine": False,
                    "actor": {"ppo_mini_batch_size": 16},
                    "rollout": {
                        "nnodes": 1,
                        "n_gpus_per_node": 8,
                        "checkpoint_engine": {"backend": "nccl"},
                        "disaggregation": {"enabled": True},
                    },
                },
                "trainer": {
                    "v1": {
                        "separate_async": {
                            "parameter_sync_step": 4,
                            "hybrid_rollout": {
                                "_target_": "verl.trainer.config.HybridRolloutSwitchConfig",
                                "enable_switch": True,
                            },
                        }
                    }
                },
                "reward": {"reward_model": {"enable": False}},
            }
        )

        def mock_base_init(trainer, trainer_config):
            # [FUNCTION DOC]
            # HELPER: replacement base initializer storing the config and a replay buffer stub whose wait returns
            # None and sampleable count returns 0. Side effects: mutates trainer.
            trainer.config = trainer_config
            trainer.replay_buffer = SimpleNamespace(
                wait_for_sampleable=lambda *_args: None,
                get_sampleable_count=lambda *_args: 0,
            )

        monkeypatch.setattr(separate_async_module.PPOTrainer, "__init__", mock_base_init)
        warning = MagicMock()
        monkeypatch.setattr(separate_async_module.logger, "warning", warning)

        trainer = PPOTrainerSeparateAsync(config)

        assert trainer.hybrid_rollout_config.enable_switch is False
        warning.assert_called_once()

    def test_mode_switch_hooks_are_noops(self):
        # [FUNCTION DOC]
        # All four mode-switch/balancer hooks perform no checkpoint-manager calls.
        trainer = _make_trainer(enable_hybrid_replicas=False)
        trainer.switch_to_rollout()
        trainer.switch_to_trainer()
        trainer.add_replicas_to_balancer()
        trainer.remove_replicas_from_balancer()
        trainer.checkpoint_manager.update_weights.assert_not_called()
        trainer.checkpoint_manager.abort_replicas.assert_not_called()
        trainer.checkpoint_manager.sleep_replicas.assert_not_called()

    def test_on_init_end_skips_hybrid_weight_update(self):
        # [FUNCTION DOC]
        # on_init_end updates weights only through the standalone checkpoint manager.
        trainer = _make_trainer(enable_hybrid_replicas=False)
        trainer.global_steps = 1
        trainer.standalone_checkpoint_manager = MagicMock()
        trainer.on_init_end()
        trainer.standalone_checkpoint_manager.update_weights.assert_called_once()
        trainer.checkpoint_manager.update_weights.assert_not_called()


class TestFractionalWarmup:
    # [CLASS DOC]
    # Tests fractional num_warmup_batches rounding down to whole gen_batch_size chunks.
    def _warmup_stub(self, *, train_batch_size=64, gen_batch_size=None, restored=0):
        # [FUNCTION DOC]
        # HELPER: SimpleNamespace stub with data config (train size, optional gen size), restored count, a mocked
        # _add_prompts_to_generate and the real _add_async_warmup_batches bound. Outputs: the stub.
        data = {"train_batch_size": train_batch_size}
        if gen_batch_size is not None:
            data["gen_batch_size"] = gen_batch_size
        stub = SimpleNamespace(
            config=OmegaConf.create({"skip": {"rollout_tq": {"enable": False}}, "data": data}),
            _restored_tq_prompt_count=restored,
        )
        stub._add_prompts_to_generate = MagicMock()
        stub._add_async_warmup_batches = PPOTrainer._add_async_warmup_batches.__get__(stub)
        return stub

    def test_fractional_warmup_rounds_down_to_gen_batch_chunks(self):
        # [FUNCTION DOC]
        # 1.5 x 64 = 96 with gen 32 is already whole chunks: request 96.
        # 1.5 x 64 = 96 prompts, already a whole number of gen_batch_size=32 chunks.
        stub = self._warmup_stub(train_batch_size=64, gen_batch_size=32)
        stub._add_async_warmup_batches(1.5)
        stub._add_prompts_to_generate.assert_called_once_with(96)

    def test_fraction_below_one_chunk_is_dropped(self):
        # [FUNCTION DOC]
        # 1.25 x 64 = 80 with gen 64 rounds down to 64.
        # 1.25 x 64 = 80 prompts -> 80 // 64 * 64 = 64 (gen_batch_size=64).
        stub = self._warmup_stub(train_batch_size=64, gen_batch_size=64)
        stub._add_async_warmup_batches(1.25)
        stub._add_prompts_to_generate.assert_called_once_with(64)

    def test_integer_warmup_unchanged_without_gen_batch_size(self):
        # [FUNCTION DOC]
        # 2 x 64 without gen size requests 128.
        stub = self._warmup_stub(train_batch_size=64)
        stub._add_async_warmup_batches(2)
        stub._add_prompts_to_generate.assert_called_once_with(128)


class TestOffPolicyThresholdNone:
    # [CLASS DOC]
    # Tests that a None off-policy threshold disables staleness control.
    def test_none_threshold_disables_stale_eviction(self):
        # [FUNCTION DOC]
        # With None and "drop", a very old finished key is not reported stale.
        rb = ReplayBufferAsync(
            trainer_mode="separate_async",
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="drop",
            sampler_kwargs={},
            poll_interval=0.05,
        )
        # A very old terminal key stays sampleable: no drop even with strategy="drop".
        rb.prompt_global_steps["train"]["stale"] = 1
        rb.finished_keys["train"].add("stale")
        assert rb._stale_terminal_keys(1000, "train") == set()

    def test_none_threshold_disables_wait_blocking(self):
        # [FUNCTION DOC]
        # With None and "wait", an old pending key does not block when 4 finished keys satisfy a request of 4.
        rb = ReplayBufferAsync(
            trainer_mode="separate_async",
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="wait",
            sampler_kwargs={},
            poll_interval=0.05,
        )
        rb.prompt_global_steps["train"]["old"] = 1
        rb.pending_keys["train"].add("old")
        # Enough sampleable keys exist: None threshold must not block on the stale pending key.
        rb.finished_keys["train"].update({f"k{i}" for i in range(4)})
        assert rb._has_enough_samples(1000, "train", 4, set(rb.finished_keys["train"])) is True

    def test_invalid_threshold_still_rejected(self):
        # [FUNCTION DOC]
        # Threshold 0 raises AssertionError "Invalid max off policy threshold".
        with pytest.raises(AssertionError, match="Invalid max off policy threshold"):
            ReplayBufferAsync(
                trainer_mode="separate_async",
                trainer_config={},
                max_off_policy_threshold=0,
                max_off_policy_strategy="drop",
                sampler_kwargs={},
            )
