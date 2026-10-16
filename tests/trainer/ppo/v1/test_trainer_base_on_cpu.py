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
# Tests for PPOTrainer._build_replay_buffer and _init_dataloader in the v1 trainer base: choosing the
# built-in ReplayBuffer (sync) vs ReplayBufferAsync (async modes), forwarding filter_groups (DAPO) options,
# custom sampler loading, sync failed-group refill forcing gen_batch_size=1, and the warning for the
# ignored max_num_gen_batches option. Trainers are created via __new__ with minimal OmegaConf configs.

from unittest.mock import patch

from omegaconf import OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    # [CLASS DOC]
    # HELPER: concrete PPOTrainer subclass with no-op on_step_end/on_sample_end hooks.
    def on_step_end(self):
        # [FUNCTION DOC]
        # HELPER: no-op.
        pass

    def on_sample_end(self):
        # [FUNCTION DOC]
        # HELPER: no-op.
        pass


class _CustomSampler:
    # [CLASS DOC]
    # HELPER: stand-in custom sampler class that records the keyword arguments it was built with.
    def __init__(self, **kwargs):
        # [FUNCTION DOC]
        # HELPER: stores all keyword arguments in self.kwargs.
        self.kwargs = kwargs


def _trainer_with_filter_groups(filter_groups: dict, trainer_mode: str = "sync") -> _StubTrainer:
    # [FUNCTION DOC]
    # HELPER: builds an uninitialized _StubTrainer with the given trainer_mode and a config containing the given
    # algorithm.filter_groups mapping, train_batch_size 64, gen_batch_size 8, reward model disabled, an empty
    # section for the mode under trainer.v1, and default sampler settings (no custom sampler, off-policy
    # threshold 1 with "drop" strategy, empty sampler_kwargs).
    # Inputs: filter_groups mapping; trainer_mode string (default "sync"). Outputs: the trainer.
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = trainer_mode
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"filter_groups": filter_groups},
            "data": {"train_batch_size": 64, "gen_batch_size": 8},
            "reward": {"reward_model": {"enable": False, "enable_resource_pool": False}},
            "trainer": {
                "v1": {
                    trainer_mode: {},
                    "sampler": {
                        "custom_sampler": None,
                        "max_off_policy_threshold": 1,
                        "max_off_policy_strategy": "drop",
                        "sampler_kwargs": {},
                    },
                }
            },
        }
    )
    return trainer


def test_builtin_sampler_class_follows_trainer_mode():
    # [FUNCTION DOC]
    # Asserts sync mode yields exactly ReplayBuffer, and colocate_async/separate_async yield ReplayBufferAsync
    # with filter metric "acc" and train/gen batch sizes left as None.
    sync_sampler = _trainer_with_filter_groups({"enable": False}, trainer_mode="sync")._build_replay_buffer()
    async_samplers = [
        _trainer_with_filter_groups({"enable": True, "metric": "acc"}, trainer_mode=mode)._build_replay_buffer()
        for mode in ("colocate_async", "separate_async")
    ]

    assert type(sync_sampler) is ReplayBuffer
    assert all(type(sampler) is ReplayBufferAsync for sampler in async_samplers)
    assert all(sampler.filter_groups_metric == "acc" for sampler in async_samplers)
    assert all(sampler.train_batch_size is None for sampler in async_samplers)
    assert all(sampler.gen_batch_size is None for sampler in async_samplers)


def test_custom_sampler_skips_builtin_filter_groups_validation():
    # [FUNCTION DOC]
    # With a custom sampler configured (load_extern_type patched to return _CustomSampler), asserts
    # _resolve_filter_groups_metric is not called and none of the built-in-only kwargs (filter metric, batch
    # sizes, inflight limit, sync refill flag) are passed to the custom sampler.
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})
    trainer.config.trainer.v1.sampler.custom_sampler = {"path": "custom.py", "name": "CustomSampler"}

    with (
        patch("verl.trainer.ppo.v1.trainer_base.load_extern_type", return_value=_CustomSampler),
        patch.object(trainer, "_resolve_filter_groups_metric") as resolve_filter_groups_metric,
    ):
        sampler = trainer._build_replay_buffer()

    resolve_filter_groups_metric.assert_not_called()
    assert isinstance(sampler, _CustomSampler)
    assert "filter_groups_metric" not in sampler.kwargs
    assert "train_batch_size" not in sampler.kwargs
    assert "gen_batch_size" not in sampler.kwargs
    assert "max_inflight_gen_batches" not in sampler.kwargs
    assert "sync_refill_failed_groups" not in sampler.kwargs


def test_builtin_filter_groups_uses_default_inflight_limit():
    # [FUNCTION DOC]
    # With filter groups enabled in sync mode, asserts metric "acc", train_batch_size 64, gen_batch_size 1 and
    # default max_inflight_gen_batches 1.
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc"})

    sampler = trainer._build_replay_buffer()

    assert sampler.filter_groups_metric == "acc"
    assert sampler.train_batch_size == 64
    assert sampler.gen_batch_size == 1
    assert sampler.max_inflight_gen_batches == 1


def test_builtin_filter_groups_forwards_configured_inflight_limit():
    # [FUNCTION DOC]
    # Asserts a configured max_inflight_gen_batches of 3 is forwarded to the buffer.
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_inflight_gen_batches": 3})

    sampler = trainer._build_replay_buffer()

    assert sampler.max_inflight_gen_batches == 3


def test_builtin_sync_failure_refill_forces_single_prompt_generation():
    # [FUNCTION DOC]
    # With sync_refill_failed_groups true and filter groups disabled, asserts the flag is forwarded and
    # gen_batch_size is 1.
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True

    sampler = trainer._build_replay_buffer()

    assert sampler.sync_refill_failed_groups is True
    assert sampler.gen_batch_size == 1


def test_sync_failure_refill_overrides_dataloader_generation_batch_size():
    # [FUNCTION DOC]
    # Fills in the dataset config fields and trainer attributes needed by _init_dataloader, patches dataset,
    # sampler, StatefulDataLoader creation and logger.warning, then asserts data.gen_batch_size was rewritten
    # to 1, the first dataloader was built with batch_size 1, and the override warning
    # "data.gen_batch_size=8 is overridden to 1." was logged.
    trainer = _trainer_with_filter_groups({"enable": False})
    trainer.config.trainer.v1.sampler.sync_refill_failed_groups = True
    trainer.config.data.update(
        {
            "train_files": [],
            "val_files": [],
            "train_max_samples": -1,
            "val_max_samples": -1,
            "dataloader_num_workers": 0,
            "val_batch_size": 1,
            "validation_shuffle": False,
        }
    )
    trainer.config.trainer.total_epochs = 1
    trainer.config.trainer.total_training_steps = None
    trainer.parameter_sync_step = 1
    trainer.tokenizer = None
    trainer.processor = None

    with (
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_dataset", side_effect=[[{}, {}], [{}]]),
        patch("verl.trainer.ppo.v1.trainer_base.create_rl_sampler", return_value=None),
        patch("verl.trainer.ppo.v1.trainer_base.StatefulDataLoader") as dataloader,
        patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning,
    ):
        trainer._init_dataloader()

    assert trainer.config.data.gen_batch_size == 1
    assert dataloader.call_args_list[0].kwargs["batch_size"] == 1
    warning.assert_any_call("data.gen_batch_size=8 is overridden to 1.")


def test_builtin_filter_groups_warns_when_total_generation_limit_is_configured():
    # [FUNCTION DOC]
    # Asserts that configuring max_num_gen_batches=10 logs exactly one warning (exact format string and arg)
    # saying it is ignored by the built-in buffer.
    trainer = _trainer_with_filter_groups({"enable": True, "metric": "acc", "max_num_gen_batches": 10})

    with patch("verl.trainer.ppo.v1.trainer_base.logger.warning") as warning:
        trainer._build_replay_buffer()

    warning.assert_called_once_with(
        "algorithm.filter_groups.max_num_gen_batches=%s is ignored by the built-in V1 ReplayBuffer; "
        "use max_inflight_gen_batches to bound concurrent Sync DAPO generation.",
        10,
    )
