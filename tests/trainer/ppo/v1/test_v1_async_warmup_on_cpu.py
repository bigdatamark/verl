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
# Tests use SimpleNamespace stubs with real PPOTrainer / async trainer methods bound to them, so only the
# warmup arithmetic (window = warmup batches x train_batch_size, minus restored prompt groups, never
# negative) and the skip-rollout-TQ gate are exercised.
"""CPU tests for V1 async warmup-after-resume gating.

The prefetch window is measured in prompts: ``num_warmup_batches * train_batch_size``. Prompt groups
restored from a TransferQueue checkpoint already occupy that window, whatever their status, so warmup
only submits the remaining shortfall.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync


def _stub(
    *,
    skip_rollout_tq: bool = False,
    restored_tq_prompt_count: int = 0,
    num_warmup_batches: int = 2,
    train_batch_size: int = 4,
):
    # [FUNCTION DOC]
    # HELPER: builds a SimpleNamespace with a config (skip.rollout_tq.enable, data.train_batch_size, and
    # num_warmup_batches for both colocate_async and separate_async), a _restored_tq_prompt_count attribute,
    # a MagicMock _add_prompts_to_generate, and PPOTrainer._add_async_warmup_batches bound to it.
    # Inputs: keyword-only skip_rollout_tq (False), restored_tq_prompt_count (0), num_warmup_batches (2),
    # train_batch_size (4). Outputs: the stub.
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "skip": {"rollout_tq": {"enable": skip_rollout_tq}},
                "data": {"train_batch_size": train_batch_size},
                "trainer": {
                    "v1": {
                        "colocate_async": {"num_warmup_batches": num_warmup_batches},
                        "separate_async": {"num_warmup_batches": num_warmup_batches},
                    }
                },
            }
        ),
        _restored_tq_prompt_count=restored_tq_prompt_count,
    )
    stub._add_prompts_to_generate = MagicMock()
    stub._add_async_warmup_batches = PPOTrainer._add_async_warmup_batches.__get__(stub)
    return stub


def test_fresh_start_submits_the_whole_prefetch_window():
    # [FUNCTION DOC]
    # 3 batches x 4 with nothing restored: expects one request for 12 prompts.
    stub = _stub(restored_tq_prompt_count=0)
    stub._add_async_warmup_batches(3)
    stub._add_prompts_to_generate.assert_called_once_with(12)


def test_full_restored_prompt_pool_skips_warmup():
    # [FUNCTION DOC]
    # 12 restored equals the window: expects no request.
    stub = _stub(restored_tq_prompt_count=12)
    stub._add_async_warmup_batches(3)
    stub._add_prompts_to_generate.assert_not_called()


def test_partial_restored_prompt_pool_is_topped_up():
    # [FUNCTION DOC]
    # 6 restored of 12: expects a request for 6.
    stub = _stub(restored_tq_prompt_count=6)
    stub._add_async_warmup_batches(3)
    stub._add_prompts_to_generate.assert_called_once_with(6)


def test_overfilled_restored_prompt_pool_does_not_add_warmup():
    # [FUNCTION DOC]
    # 13 restored exceeds the window: expects no request.
    stub = _stub(restored_tq_prompt_count=13)
    stub._add_async_warmup_batches(3)
    stub._add_prompts_to_generate.assert_not_called()


def test_zero_warmup_batches_submits_nothing():
    # [FUNCTION DOC]
    # 0 warmup batches: expects no request.
    stub = _stub(restored_tq_prompt_count=0)
    stub._add_async_warmup_batches(0)
    stub._add_prompts_to_generate.assert_not_called()


def test_skip_rollout_tq_skips_warmup_even_without_restored_prompts():
    # [FUNCTION DOC]
    # With skip.rollout_tq enabled: expects no request even with nothing restored.
    stub = _stub(skip_rollout_tq=True, restored_tq_prompt_count=0)
    stub._add_async_warmup_batches(3)
    stub._add_prompts_to_generate.assert_not_called()


def test_colocate_async_on_train_begin_tops_up_configured_window():
    # [FUNCTION DOC]
    # Binds PPOTrainerColocateAsync.on_train_begin; with 3 configured batches and 4 restored expects a request
    # for 8 (12 - 4).
    stub = _stub(restored_tq_prompt_count=4, num_warmup_batches=3)
    stub.on_train_begin = PPOTrainerColocateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    stub._add_prompts_to_generate.assert_called_once_with(8)


def test_separate_async_on_train_begin_tops_up_configured_window():
    # [FUNCTION DOC]
    # Binds PPOTrainerSeparateAsync.on_train_begin; 2 batches x 4, none restored: expects a request for 8.
    stub = _stub(restored_tq_prompt_count=0, num_warmup_batches=2)
    stub.on_train_begin = PPOTrainerSeparateAsync.on_train_begin.__get__(stub)
    stub.on_train_begin()
    stub._add_prompts_to_generate.assert_called_once_with(8)


def test_loaded_prompt_count_drives_on_train_begin_top_up(monkeypatch, tmp_path):
    # [FUNCTION DOC]
    # Creates <tmp>/global_step_6/transfer_queue, configures resume_mode "resume_path" pointing at it, binds
    # the real _load_checkpoint and colocate on_train_begin, patches TQ checkpoint support to True, the prompt
    # group counter to 6 and tq.load_checkpoint to a mock. Asserts global_steps becomes 6, the restored count
    # becomes 6, the TQ checkpoint is loaded from the transfer_queue dir, and warmup requests 6 (12 - 6).
    # Inputs: monkeypatch, tmp_path. Side effects: creates directories under tmp_path.
    checkpoint_dir = tmp_path / "global_step_6"
    tq_checkpoint_dir = checkpoint_dir / "transfer_queue"
    tq_checkpoint_dir.mkdir(parents=True)

    stub = _stub(restored_tq_prompt_count=0, num_warmup_batches=3)
    stub.config.trainer.resume_mode = "resume_path"
    stub.config.trainer.resume_from_path = str(checkpoint_dir)
    stub.config.trainer.del_local_ckpt_after_load = False
    stub.trainer_mode = "colocate_async"
    stub.use_critic = False
    stub.actor_rollout_wg = MagicMock()
    stub.train_dataloader = MagicMock()
    stub._load_checkpoint = PPOTrainer._load_checkpoint.__get__(stub)
    stub.on_train_begin = PPOTrainerColocateAsync.on_train_begin.__get__(stub)

    load_tq_checkpoint = MagicMock()
    monkeypatch.setattr(trainer_base, "_tq_supports_checkpoint", lambda: True)
    monkeypatch.setattr(trainer_base, "_count_tq_prompt_groups", lambda: 6)
    monkeypatch.setattr(trainer_base.tq, "load_checkpoint", load_tq_checkpoint, raising=False)

    stub._load_checkpoint()
    stub.on_train_begin()

    assert stub.global_steps == 6
    assert stub._restored_tq_prompt_count == 6
    load_tq_checkpoint.assert_called_once_with(str(tq_checkpoint_dir))
    stub._add_prompts_to_generate.assert_called_once_with(6)
