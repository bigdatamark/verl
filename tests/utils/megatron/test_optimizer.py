# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# CPU tests for precision-dependent kwargs assembled by verl's init_megatron_optim_config, using a fixture that
# replaces the module-level OptimizerConfig with a kwargs recorder.
"""Tests for the precision-aware dispatch in ``init_megatron_optim_config``.

These tests stub out ``megatron.core.optimizer.OptimizerConfig`` so they can
run on CPU without TransformerEngine — the goal is to verify which kwargs
verl assembles for each precision mode, not Megatron's downstream validation.

The precision-aware optimizer is opt-in: the bf16 branch keeps the fp32
optimizer state unless ``use_precision_aware_optimizer`` is set on the config,
at which point the moment / grad dtypes follow the configured fields.
"""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from verl.utils.megatron import optimizer as opt_mod
from verl.utils.megatron.optimizer import init_megatron_optim_config


def _base_optim_config(**overrides):
    # [FUNCTION DOC]
    # HELPER: Returns an OmegaConf config with optimizer "adam", lr 1e-3, min_lr 0, clip_grad 1.0, weight_decay 0.01,
    # updated with any keyword overrides.
    cfg = {
        "optimizer": "adam",
        "lr": 1e-3,
        "min_lr": 0.0,
        "clip_grad": 1.0,
        "weight_decay": 0.01,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _precision_aware_optim_config(**overrides):
    # [FUNCTION DOC]
    # HELPER: Returns a base config with use_precision_aware_optimizer True and main_grads/exp_avg/exp_avg_sq dtypes
    # "bf16", plus overrides.
    """bf16 optimizer state explicitly opted in via the new config flags."""
    fields = {
        "use_precision_aware_optimizer": True,
        "main_grads_dtype": "bf16",
        "exp_avg_dtype": "bf16",
        "exp_avg_sq_dtype": "bf16",
    }
    fields.update(overrides)
    return _base_optim_config(**fields)


@pytest.fixture
def captured_args(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: pytest fixture; monkeypatches verl.utils.megatron.optimizer.OptimizerConfig with a function that
    # clears and refills a shared dict with the received kwargs and returns a MagicMock. Outputs: that dict, reflecting
    # the most recent construction. Side effects: module attribute patch for the test duration.
    """Replace ``OptimizerConfig`` with a recorder so we can inspect kwargs."""
    captured: dict = {}

    def _fake(**kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs into captured (replacing previous content) and returns a MagicMock built with them.
        captured.clear()
        captured.update(kwargs)
        return MagicMock(name="OptimizerConfig", **kwargs)

    monkeypatch.setattr(opt_mod, "OptimizerConfig", _fake)
    return captured


def test_bf16_branch_defaults_to_fp32_optimizer_state(captured_args):
    # [FUNCTION DOC]
    # Behavior: bf16 mode with base config; asserts bf16 True, params_dtype bfloat16 and that no precision-aware flag
    # or moment/grad dtype kwargs are passed. Inputs: captured_args. Outputs: none.
    """Opt-out default: bf16 params, but the precision-aware optimizer stays off."""
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # No precision-aware optimizer and no sub-fp32 moment / grad dtypes by default.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_bf16_branch_opt_in_enables_precision_aware_with_bf16_state(captured_args):
    # [FUNCTION DOC]
    # Behavior: bf16 mode with opt-in config; asserts precision-aware True, grads/moment dtypes bfloat16, and
    # main_params_dtype not passed. Inputs: captured_args. Outputs: none.
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=False, bf16=True)

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16
    # Master params dtype intentionally left at Megatron default (fp32) —
    # TE FusedAdam rejects bf16 master at init.
    assert "main_params_dtype" not in captured_args


def test_bf16_opt_in_respects_per_field_dtypes(captured_args):
    # [FUNCTION DOC]
    # Behavior: opt-in with exp_avg_sq_dtype "fp32"; asserts that field becomes float32 while others stay bfloat16.
    # Inputs: captured_args. Outputs: none.
    """Per-flag control: opting in but pinning a moment to fp32 is honored."""
    cfg = _precision_aware_optim_config(exp_avg_sq_dtype="fp32")
    init_megatron_optim_config(cfg, fp16=False, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.float32


def test_fp16_branch_uses_precision_aware_but_keeps_fp32_optimizer_state(captured_args):
    # [FUNCTION DOC]
    # Behavior: fp16 mode; asserts fp16 True, bf16 False, params float16, precision-aware True, initial loss scale
    # 32768, min loss scale 1, store_param_remainders False, and no grad/moment dtype kwargs.
    # Inputs: captured_args. Outputs: none.
    init_megatron_optim_config(_base_optim_config(), fp16=True, bf16=False)

    assert captured_args["fp16"] is True
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float16
    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["initial_loss_scale"] == 32768
    assert captured_args["min_loss_scale"] == 1
    assert captured_args["store_param_remainders"] is False
    # Adam moment / grad dtypes left at Megatron's fp32 default in fp16 mode.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_fp32_branch_disables_precision_aware_optimizer(captured_args):
    # [FUNCTION DOC]
    # Behavior: fp32 mode (fp16 and bf16 False); asserts params float32 and no precision-aware or dtype kwargs.
    # Inputs: captured_args. Outputs: none.
    init_megatron_optim_config(_base_optim_config(), fp16=False, bf16=False)

    assert captured_args["fp16"] is False
    assert captured_args["bf16"] is False
    assert captured_args["params_dtype"] is torch.float32
    # Precision-aware optimizer must stay off — Megatron asserts the dtype
    # fields equal fp32 when it's disabled.
    assert "use_precision_aware_optimizer" not in captured_args
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args
    assert "exp_avg_sq_dtype" not in captured_args


def test_default_kwargs_dispatch_to_bf16_branch(captured_args):
    # [FUNCTION DOC]
    # Behavior: calling with defaults selects the bf16 path (bf16 True, params bfloat16, no precision-aware flag).
    # Inputs: captured_args. Outputs: none.
    """Backward compatibility: callers that omit ``bf16`` get the bf16 path (fp32 state)."""
    init_megatron_optim_config(_base_optim_config())

    assert captured_args["bf16"] is True
    assert captured_args["params_dtype"] is torch.bfloat16
    # Opt-in default keeps the precision-aware optimizer off.
    assert "use_precision_aware_optimizer" not in captured_args


def test_fp16_wins_over_bf16_when_both_true(captured_args):
    # [FUNCTION DOC]
    # Behavior: fp16 and bf16 both True with opt-in config; asserts fp16 wins (params float16) and bf16-only dtype
    # kwargs are absent. Inputs: captured_args. Outputs: none.
    init_megatron_optim_config(_precision_aware_optim_config(), fp16=True, bf16=True)

    assert captured_args["fp16"] is True
    assert captured_args["params_dtype"] is torch.float16
    # bf16-branch-only fields must not appear when fp16 is selected, even when
    # the precision-aware config flags are present.
    assert "main_grads_dtype" not in captured_args
    assert "exp_avg_dtype" not in captured_args


def test_use_distributed_optimizer_passes_through(captured_args):
    # [FUNCTION DOC]
    # Behavior: asserts use_distributed_optimizer False and True are each passed through unchanged.
    # Inputs: captured_args. Outputs: none.
    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=False)
    assert captured_args["use_distributed_optimizer"] is False

    init_megatron_optim_config(_base_optim_config(), use_distributed_optimizer=True)
    assert captured_args["use_distributed_optimizer"] is True


def test_basic_optim_config_fields_pass_through(captured_args):
    # [FUNCTION DOC]
    # Behavior: asserts optimizer, lr, min_lr, clip_grad and weight_decay from the config pass through unchanged.
    # Inputs: captured_args. Outputs: none.
    cfg = _base_optim_config(optimizer="sgd", lr=5e-4, min_lr=1e-5, clip_grad=0.5, weight_decay=0.1)
    init_megatron_optim_config(cfg)

    assert captured_args["optimizer"] == "sgd"
    assert captured_args["lr"] == pytest.approx(5e-4)
    assert captured_args["min_lr"] == pytest.approx(1e-5)
    assert captured_args["clip_grad"] == pytest.approx(0.5)
    assert captured_args["weight_decay"] == pytest.approx(0.1)


def test_override_optimizer_config_overrides_branch_defaults(captured_args):
    # [FUNCTION DOC]
    # Behavior: opt-in config with override_optimizer_config setting precision-aware False and an arbitrary sentinel
    # exp_avg_dtype; asserts overrides win while non-overridden bf16 defaults remain bfloat16.
    # Inputs: captured_args. Outputs: none.
    cfg = _precision_aware_optim_config(
        override_optimizer_config={
            "use_precision_aware_optimizer": False,
            "exp_avg_dtype": "sentinel-override",
        },
    )
    init_megatron_optim_config(cfg, bf16=True)

    # User-supplied overrides win over the opted-in bf16 defaults …
    assert captured_args["use_precision_aware_optimizer"] is False
    assert captured_args["exp_avg_dtype"] == "sentinel-override"
    # … but non-overridden bf16 defaults remain.
    assert captured_args["main_grads_dtype"] is torch.bfloat16
    assert captured_args["exp_avg_sq_dtype"] is torch.bfloat16


def test_missing_override_config_leaves_branch_defaults_intact(captured_args):
    # [FUNCTION DOC]
    # Behavior: asserts the opt-in config has no override_optimizer_config key and conversion still succeeds with
    # precision-aware True and exp_avg_dtype bfloat16. Inputs: captured_args. Outputs: none.
    """``optim_config.get('override_optimizer_config', {})`` must not crash when absent."""
    cfg = _precision_aware_optim_config()
    assert "override_optimizer_config" not in cfg

    init_megatron_optim_config(cfg, bf16=True)

    assert captured_args["use_precision_aware_optimizer"] is True
    assert captured_args["exp_avg_dtype"] is torch.bfloat16
