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
# CPU unit tests for LoRA-only checkpointing: BaseCheckpointManager's
# should_save_lora_only flag and is_lora_only_state_dict classifier, and
# FSDPCheckpointManager saving only adapter weights (when LoRA is present) and
# loading a LoRA-only checkpoint without clobbering base weights. Uses fake
# FSDP/PEFT model objects with scalar parameters.

"""Unit tests for save_lora_only checkpoint support in FSDPCheckpointManager."""

from collections import namedtuple

import pytest

from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

# Return type mimicking torch.nn.Module.load_state_dict with strict=False
_LoadStateDictResult = namedtuple("_LoadStateDictResult", ["missing_keys", "unexpected_keys"])


class TestBaseCheckpointManagerLoraOnly:
    # [CLASS DOC]
    # Tests for the LoRA-only helpers on BaseCheckpointManager.
    """Tests for LoRA-only checkpoint properties on BaseCheckpointManager."""

    @pytest.fixture(autouse=True)
    def _patch_dist(self, monkeypatch):
        # [FUNCTION DOC]
        # HELPER: autouse fixture monkeypatching torch.distributed get_rank -> 0,
        # get_world_size -> 1 and barrier -> no-op so managers run without a process group.
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def test_should_save_lora_only_default(self):
        # [FUNCTION DOC]
        # Behavior: no checkpoint config means should_save_lora_only is False.
        mgr = self._make_manager(checkpoint_config=None)
        assert mgr.should_save_lora_only is False

    def test_should_save_lora_only_false(self):
        # [FUNCTION DOC]
        # Behavior: save_lora_only False in config yields False.
        mgr = self._make_manager(checkpoint_config={"save_lora_only": False})
        assert mgr.should_save_lora_only is False

    def test_should_save_lora_only_true(self):
        # [FUNCTION DOC]
        # Behavior: save_lora_only True in config yields True.
        mgr = self._make_manager(checkpoint_config={"save_lora_only": True})
        assert mgr.should_save_lora_only is True

    def test_is_lora_only_state_dict_empty(self):
        # [FUNCTION DOC]
        # Behavior: an empty state dict is not LoRA-only.
        assert BaseCheckpointManager.is_lora_only_state_dict({}) is False

    def test_is_lora_only_state_dict_none(self):
        # [FUNCTION DOC]
        # Behavior: None is not LoRA-only.
        assert BaseCheckpointManager.is_lora_only_state_dict(None) is False

    def test_is_lora_only_state_dict_full(self):
        # [FUNCTION DOC]
        # Behavior: base-only keys are not LoRA-only.
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": None,
            "model.layers.0.self_attn.k_proj.weight": None,
            "lm_head.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is False

    def test_is_lora_only_state_dict_lora(self):
        # [FUNCTION DOC]
        # Behavior: keys all containing lora_A/lora_B are LoRA-only.
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": None,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is True

    def test_is_lora_only_state_dict_adapter_key(self):
        # [FUNCTION DOC]
        # Behavior: keys all containing ".adapter_" are treated as LoRA-only.
        state_dict = {
            "model.layers.0.attn.adapter_A.weight": None,
            "model.layers.0.attn.adapter_B.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is True

    def test_is_lora_only_state_dict_mixed(self):
        # [FUNCTION DOC]
        # Behavior: a mix of LoRA and base keys is not LoRA-only.
        state_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": None,
            "model.layers.0.self_attn.q_proj.weight": None,
        }
        assert BaseCheckpointManager.is_lora_only_state_dict(state_dict) is False

    @staticmethod
    def _make_manager(checkpoint_config):
        # [FUNCTION DOC]
        # HELPER: builds a BaseCheckpointManager over placeholder model/optimizer objects with the given checkpoint_config.
        class MockModel:
            # [CLASS DOC]
            # HELPER: empty model placeholder.
            pass

        class MockOptimizer:
            # [CLASS DOC]
            # HELPER: empty optimizer placeholder.
            pass

        return BaseCheckpointManager(
            model=MockModel(),
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )


class TestFSDPCheckpointManagerLoraOnly:
    # [CLASS DOC]
    # Tests for LoRA detection, LoRA-only save, and load merging on FSDPCheckpointManager.
    """Tests for LoRA-only checkpoint logic on FSDPCheckpointManager."""

    @pytest.fixture(autouse=True)
    def _patch_dist(self, monkeypatch):
        # [FUNCTION DOC]
        # HELPER: autouse fixture monkeypatching torch.distributed get_rank -> 0,
        # get_world_size -> 1 and barrier -> no-op so managers run without a process group.
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def _make_fsdp_manager(self, checkpoint_config, model=None):
        # [FUNCTION DOC]
        # HELPER: builds an FSDPCheckpointManager (no optimizer/scheduler/processor)
        # around model, defaulting to a fake FSDP model with LoRA.
        from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

        if model is None:
            model = _FakeFSDPModel(has_lora=True)

        return FSDPCheckpointManager(
            model=model,
            optimizer=None,
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=checkpoint_config,
        )

    def test_has_lora_true(self):
        # [FUNCTION DOC]
        # Behavior: _has_lora is True when the wrapped module has a peft_config.
        mgr = self._make_fsdp_manager(checkpoint_config={"save_lora_only": True})
        assert mgr._has_lora() is True

    def test_has_lora_false(self):
        # [FUNCTION DOC]
        # Behavior: _has_lora is False when no peft_config exists.
        model = _FakeFSDPModel(has_lora=False)
        mgr = self._make_fsdp_manager(checkpoint_config={"save_lora_only": True}, model=model)
        assert mgr._has_lora() is False

    def test_save_lora_only_filters_state_dict(self, tmp_path):
        # [FUNCTION DOC]
        # Behavior: saving with save_lora_only and contents [model] writes
        # model_world_size_1_rank_0.pt containing only the 4 keys with "lora_" or ".adapter_".
        # Side effects: writes into pytest tmp_path.
        model = _FakeFSDPModel(has_lora=True)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        for key in state_dict:
            assert "lora_" in key or ".adapter_" in key, "Unexpected base key: {}".format(key)
        assert len(state_dict) == 4

    def test_save_lora_only_no_lora_saves_full(self, tmp_path):
        # [FUNCTION DOC]
        # Behavior: save_lora_only with a non-LoRA model falls back to a full save (base.weight present).
        model = _FakeFSDPModel(has_lora=False)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        assert "base.weight" in state_dict
        assert "lora_A.weight" in state_dict

    def test_save_lora_only_disabled_saves_full(self, tmp_path):
        # [FUNCTION DOC]
        # Behavior: save_lora_only disabled saves the full state dict even with LoRA.
        model = _FakeFSDPModel(has_lora=True)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": False, "save_contents": ["model"]},
            model=model,
        )

        save_dir = tmp_path / "ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        saved_path = save_dir / "model_world_size_1_rank_0.pt"
        import torch

        state_dict = torch.load(saved_path, weights_only=False)
        assert "base.weight" in state_dict
        assert "lora_A.weight" in state_dict

    def test_load_lora_only_checkpoint_merges(self, tmp_path):
        # [FUNCTION DOC]
        # Behavior: saves a LoRA-only checkpoint from a model with base 1.0, loads it
        # into a model with base 99.0 and asserts base stays 99.0 while lora_A becomes 0.5.
        model_before = _FakeFSDPModel(has_lora=True)
        model_before._fsdp_wrapped_module.base_weight.data.fill_(1.0)

        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": True, "save_contents": ["model"]},
            model=model_before,
        )
        save_dir = tmp_path / "lora_only"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        model_after = _FakeFSDPModel(has_lora=True)
        model_after._fsdp_wrapped_module.base_weight.data.fill_(99.0)

        mgr2 = self._make_fsdp_manager(
            checkpoint_config={"load_contents": ["model"]},
            model=model_after,
        )
        mgr2.load_checkpoint(local_path=str(save_dir))

        assert model_after._fsdp_wrapped_module.base_weight.item() == 99.0
        assert model_after._fsdp_wrapped_module.lora_A_weight.item() == 0.5

    def test_load_full_checkpoint_still_works(self, tmp_path):
        # [FUNCTION DOC]
        # Behavior: a full checkpoint (base 42.0) loaded into a fresh model restores base to 42.0.
        model = _FakeFSDPModel(has_lora=True)
        model._fsdp_wrapped_module.base_weight.data.fill_(42.0)
        mgr = self._make_fsdp_manager(
            checkpoint_config={"save_lora_only": False, "save_contents": ["model"]},
            model=model,
        )
        save_dir = tmp_path / "full_ckpt"
        mgr.save_checkpoint(local_path=str(save_dir), global_step=1)

        model_after = _FakeFSDPModel(has_lora=True)
        model_after._fsdp_wrapped_module.base_weight.data.fill_(0.0)
        mgr2 = self._make_fsdp_manager(
            checkpoint_config={"load_contents": ["model"]},
            model=model_after,
        )
        mgr2.load_checkpoint(local_path=str(save_dir))

        assert model_after._fsdp_wrapped_module.base_weight.item() == 42.0


class _FakeConfig:
    # [CLASS DOC]
    # HELPER: config stub whose save_pretrained does nothing.
    """Minimal config mock that supports save_pretrained."""

    def save_pretrained(self, path):
        # [FUNCTION DOC]
        # HELPER: no-op.
        pass


class _FakeFSDPModel:
    # [CLASS DOC]
    # HELPER: fake FSDP wrapper exposing _fsdp_wrapped_module (a fake PEFT model),
    # its config and can_generate, plus state_dict/load_state_dict/named_buffers.
    """A fake FSDP-wrapped model that mimics the interface used by FSDPCheckpointManager."""

    def __init__(self, has_lora=True):
        # [FUNCTION DOC]
        # HELPER: wraps a new fake PEFT model (with or without LoRA) and mirrors its config and can_generate.
        self._fsdp_wrapped_module = _FakePEFTModel(has_lora=has_lora)
        self.config = self._fsdp_wrapped_module.config
        self.can_generate = self._fsdp_wrapped_module.can_generate

    def state_dict(self):
        # [FUNCTION DOC]
        # HELPER: delegates to the wrapped module's state_dict.
        return self._fsdp_wrapped_module.state_dict()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # [FUNCTION DOC]
        # HELPER: delegates loading; normalizes a None result to empty missing/unexpected key lists.
        result = self._fsdp_wrapped_module.load_state_dict(state_dict, strict=strict, assign=assign)
        if result is None:
            return _LoadStateDictResult(missing_keys=[], unexpected_keys=[])
        return result

    def named_buffers(self):
        # [FUNCTION DOC]
        # HELPER: returns an empty items view (no buffers).
        return {}.items()


class _FakePEFTModel:
    # [CLASS DOC]
    # HELPER: fake PEFT model with three scalar parameters (base 1.0, lora_A 0.5, lora_B 0.3).
    """Fake PEFT-style model with a peft_config."""

    def __init__(self, has_lora=True):
        # [FUNCTION DOC]
        # HELPER: creates parameters, a fake config and a can_generate returning False;
        # when has_lora, adds a peft_config mapping "default" to an object with r, lora_alpha, task_type.
        import torch

        self.base_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.lora_A_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.lora_B_weight = torch.nn.Parameter(torch.tensor(0.3))
        self.config = _FakeConfig()
        self.can_generate = lambda: False

        if has_lora:
            self.peft_config = {"default": type("Cfg", (), {"r": 8, "lora_alpha": 16, "task_type": "CAUSAL_LM"})()}

    def state_dict(self):
        # [FUNCTION DOC]
        # HELPER: returns base.weight, lora_A.weight, lora_B.weight, plus PEFT-prefixed
        # aliases of the LoRA params when peft_config exists (4 LoRA keys total then).
        d = {
            "base.weight": self.base_weight,
            "lora_A.weight": self.lora_A_weight,
            "lora_B.weight": self.lora_B_weight,
        }
        if hasattr(self, "peft_config"):
            d["base_model.model.lora_A.lora_A.weight"] = self.lora_A_weight
            d["base_model.model.lora_B.lora_B.weight"] = self.lora_B_weight
        return d

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # [FUNCTION DOC]
        # HELPER: copies each tensor into base/lora_A/lora_B by substring match of the
        # key (base.weight first); others are collected as unexpected. strict/assign ignored.
        # Outputs: result with no missing keys and the unexpected keys.
        unexpected_keys = []
        for key, tensor in state_dict.items():
            if "base.weight" in key or key == "base.weight":
                self.base_weight.data.copy_(tensor)
            elif "lora_A" in key:
                self.lora_A_weight.data.copy_(tensor)
            elif "lora_B" in key:
                self.lora_B_weight.data.copy_(tensor)
            else:
                unexpected_keys.append(key)
        return _LoadStateDictResult(missing_keys=[], unexpected_keys=unexpected_keys)
