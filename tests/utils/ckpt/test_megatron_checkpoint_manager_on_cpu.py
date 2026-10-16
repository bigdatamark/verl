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
# Checkpoint layout under test ("v2"): <ckpt>/{model,optimizer,extra}/dist_ckpt for distributed-checkpoint
# shards, with HF-format model weights saved via an mbridge "bridge" object when not using
# dist-checkpointing. Save/load functions of megatron dist_checkpointing are patched with mocks;
# _PATCH_LOAD_META is the dotted path of the content-metadata loader patched in load tests.

"""Unit tests for MegatronCheckpointManager.

These tests verify the __init__ flag resolution, builder composition,
save_checkpoint / load_checkpoint dispatch, and edge cases.

Uses real megatron.core with gloo backend on a single CPU process.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from verl.trainer.config import CheckpointConfig
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager

# ---------------------------------------------------------------------------
# Session-scoped: initialize torch.distributed + megatron parallel state once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _init_distributed():
    # [FUNCTION DOC]
    # HELPER: session-scoped autouse fixture; sets default MASTER_ADDR/PORT (29599)/RANK/WORLD_SIZE env vars,
    # initializes a single-process gloo process group and Megatron model parallel state (TP=1, PP=1) if not
    # already, yields, then destroys model parallel state and the process group.
    # Side effects: mutates os.environ and global distributed state.
    """Initialize gloo process group and megatron parallel state for the
    entire test session (single process, world_size=1, CPU only)."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    yield

    mpu.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MegatronSub:
    # [CLASS DOC]
    # HELPER: stand-in megatron sub-config with both optimizer reshardability flags False.
    dist_ckpt_optim_fully_reshardable = False
    distrib_optim_fully_reshardable_mem_efficient = False


class _RoleCfg:
    # [CLASS DOC]
    # HELPER: stand-in role config exposing a megatron sub-config.
    megatron = _MegatronSub()


class _MinimalModel:
    # [CLASS DOC]
    # HELPER: stand-in model config with a fake path.
    path = "/tmp/fake_model"


class _MinimalConfig:
    # [CLASS DOC]
    # HELPER: stand-in trainer config exposing model and actor role configs.
    """Minimal stand-in for the trainer config object."""

    model = _MinimalModel()
    actor = _RoleCfg()


def _make_mock_model():
    # [FUNCTION DOC]
    # HELPER: returns a one-element list (single virtual pipeline stage) holding a MagicMock model with its
    # auto-created "module" attribute deleted (so DDP unwrapping does not redirect) and sharded_state_dict
    # returning {"layer.weight": mock}.
    """A mock model list (VPP length 1) with a working sharded_state_dict.

    ``_build_model_sharded_state_dict`` does ``if hasattr(model, "module"): model = model.module``
    (mirroring Megatron DDP unwrapping).  A bare ``MagicMock()`` auto-creates
    ``model.module`` on access, so we explicitly delete it to prevent the
    unwrap from redirecting to a different mock.
    """
    model = MagicMock()
    del model.module
    model.sharded_state_dict.return_value = {"layer.weight": MagicMock()}
    return [model]


def _make_manager(
    save_contents=None,
    load_contents=None,
    use_dist_checkpointing=False,
    bridge="auto",
    peft_cls=None,
    use_distributed_optimizer=False,
):
    # [FUNCTION DOC]
    # HELPER: builds a MegatronCheckpointManager with mocked model/optimizer (sharded state dict with step and
    # param_state)/lr scheduler (state_dict last_epoch 10), a synchronous CheckpointConfig, role "actor", fp16
    # params and other MagicMock collaborators (transformer_config is a spec-less mock with no attributes).
    # When bridge is "auto": a mock bridge if HF weights would be saved ("hf_model" in save contents, or
    # "model" without dist checkpointing), otherwise None for dist checkpointing or a mock bridge otherwise.
    # Inputs: save_contents/load_contents (default model, optimizer, extra), use_dist_checkpointing, bridge,
    # peft_cls, use_distributed_optimizer. Outputs: the manager.
    # Exceptions: propagates the manager's ValueError for invalid bridge combinations.
    if save_contents is None:
        save_contents = ["model", "optimizer", "extra"]
    if load_contents is None:
        load_contents = ["model", "optimizer", "extra"]

    ckpt_config = CheckpointConfig(
        save_contents=list(save_contents),
        load_contents=list(load_contents),
        async_save=False,
    )

    model = _make_mock_model()
    optimizer = MagicMock()
    optimizer.sharded_state_dict.return_value = {"step": 1, "param_state": MagicMock()}
    lr_scheduler = MagicMock()
    lr_scheduler.state_dict.return_value = {"last_epoch": 10}

    if bridge == "auto":
        save_c = list(save_contents)
        would_save_hf = "hf_model" in save_c or ("model" in save_c and not use_dist_checkpointing)
        bridge = MagicMock() if would_save_hf else (None if use_dist_checkpointing else MagicMock())

    return MegatronCheckpointManager(
        config=_MinimalConfig(),
        checkpoint_config=ckpt_config,
        model_config=MagicMock(),
        transformer_config=MagicMock(spec=[]),
        role="actor",
        model=model,
        arch="GPTForCausalLM",
        hf_config=MagicMock(),
        param_dtype=torch.float16,
        share_embeddings_and_output_weights=False,
        processing_class=MagicMock(),
        optimizer=optimizer,
        optimizer_scheduler=lr_scheduler,
        use_distributed_optimizer=use_distributed_optimizer,
        use_dist_checkpointing=use_dist_checkpointing,
        bridge=bridge,
        peft_cls=peft_cls,
    )


# ===========================================================================
# Tests: __init__ flag resolution
# ===========================================================================


class TestInitFlagResolution:
    # [CLASS DOC]
    # Tests for save/load flag resolution in the manager constructor.
    def test_hf_path_default(self):
        # [FUNCTION DOC]
        # Default contents without dist ckpt: HF model saved, no dist-ckpt model.
        mgr = _make_manager(save_contents=["model", "optimizer", "extra"])
        assert mgr.use_dist_checkpointing is False
        assert mgr.should_save_hf_model is True
        assert mgr.should_save_dist_ckpt_model is False

    def test_dist_ckpt_path(self):
        # [FUNCTION DOC]
        # With dist checkpointing: dist-ckpt model saved, no HF model.
        mgr = _make_manager(
            save_contents=["model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )
        assert mgr.use_dist_checkpointing is True
        assert mgr.should_save_hf_model is False
        assert mgr.should_save_dist_ckpt_model is True

    def test_hf_model_only_in_save_contents(self):
        # [FUNCTION DOC]
        # Only "hf_model": HF saved, no dist-ckpt model.
        mgr = _make_manager(save_contents=["hf_model"])
        assert mgr.should_save_hf_model is True
        assert mgr.should_save_dist_ckpt_model is False

    def test_no_model_in_save_contents(self):
        # [FUNCTION DOC]
        # No model entries: neither model format saved.
        mgr = _make_manager(save_contents=["optimizer", "extra"])
        assert mgr.should_save_hf_model is False
        assert mgr.should_save_dist_ckpt_model is False

    def test_load_flags_hf_path(self):
        # [FUNCTION DOC]
        # Load flags without dist ckpt: HF load True, dist-ckpt load False.
        mgr = _make_manager(load_contents=["model", "optimizer", "extra"])
        assert mgr.should_load_hf_model is True
        assert mgr.should_load_dist_ckpt_model is False

    def test_load_dist_ckpt_path(self):
        # [FUNCTION DOC]
        # Load flags with dist ckpt: HF load False, dist-ckpt load True.
        mgr = _make_manager(
            load_contents=["model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )
        assert mgr.should_load_hf_model is False
        assert mgr.should_load_dist_ckpt_model is True

    def test_hf_path_requires_bridge_instance(self):
        # [FUNCTION DOC]
        # HF path with bridge None raises ValueError "HF-format model weights require".
        with pytest.raises(ValueError, match="HF-format model weights require"):
            _make_manager(use_dist_checkpointing=False, bridge=None)

    def test_hf_model_requires_bridge(self):
        # [FUNCTION DOC]
        # "hf_model" with bridge None raises ValueError mentioning 'hf_model'.
        with pytest.raises(ValueError, match="'hf_model'"):
            _make_manager(save_contents=["hf_model"], bridge=None)

    def test_hf_model_with_dist_ckpt_when_mbridge(self):
        # [FUNCTION DOC]
        # model + hf_model with dist ckpt and a bridge: both formats saved.
        mgr = _make_manager(
            save_contents=["model", "hf_model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )
        assert mgr.should_save_hf_model is True
        assert mgr.should_save_dist_ckpt_model is True


# ===========================================================================
# Tests: builder composition
# ===========================================================================


class TestBuilders:
    # [CLASS DOC]
    # Tests for the private state-dict builder methods.
    def test_build_model_sharded_state_dict(self):
        # [FUNCTION DOC]
        # Single stage: result has key "model" and the model's sharded_state_dict was called once.
        mgr = _make_manager()
        result = mgr._build_model_sharded_state_dict(metadata={})
        assert "model" in result
        mgr.model[0].sharded_state_dict.assert_called_once()

    def test_build_model_sharded_state_dict_vpp(self):
        # [FUNCTION DOC]
        # Two stages: result has keys "model0" and "model1".
        mgr = _make_manager()
        second_model = MagicMock()
        second_model.sharded_state_dict.return_value = {"layer2.weight": MagicMock()}
        mgr.model.append(second_model)

        result = mgr._build_model_sharded_state_dict(metadata={})
        assert "model0" in result
        assert "model1" in result

    def test_build_optimizer_state_dict(self):
        # [FUNCTION DOC]
        # Result has "optimizer" and the optimizer's sharded_state_dict was called once.
        mgr = _make_manager()
        model_sd = mgr._build_model_sharded_state_dict(metadata={})
        result = mgr._build_optimizer_state_dict(model_sd, metadata={})
        assert "optimizer" in result
        mgr.optimizer.sharded_state_dict.assert_called_once()

    def test_build_optimizer_includes_lr_scheduler(self):
        # [FUNCTION DOC]
        # Result includes the lr scheduler state (last_epoch 10).
        mgr = _make_manager()
        model_sd = mgr._build_model_sharded_state_dict(metadata={})
        result = mgr._build_optimizer_state_dict(model_sd, metadata={})
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["last_epoch"] == 10

    def test_build_optimizer_no_lr_scheduler(self):
        # [FUNCTION DOC]
        # With no scheduler, "lr_scheduler" is absent.
        mgr = _make_manager()
        mgr.lr_scheduler = None
        model_sd = mgr._build_model_sharded_state_dict(metadata={})
        result = mgr._build_optimizer_state_dict(model_sd, metadata={})
        assert "optimizer" in result
        assert "lr_scheduler" not in result

    def test_build_extra_state_dict(self):
        # [FUNCTION DOC]
        # Extra state includes "rng_state".
        mgr = _make_manager()
        result = mgr._build_extra_state_dict()
        assert "rng_state" in result


# ===========================================================================
# Tests: save_checkpoint dispatch
# ===========================================================================


def _collect_save_calls(mock_save_dc):
    # [FUNCTION DOC]
    # HELPER: maps each mocked save call to {subdir: sharded_state_dict}, where subdir is the parent directory
    # name of the ckpt_path keyword (e.g. "optimizer" for .../optimizer/dist_ckpt). Later calls overwrite
    # earlier ones for the same subdir.
    """Return {relative_subdir: sharded_state_dict} for each save_dist_checkpointing call."""
    out = {}
    for call in mock_save_dc.call_args_list:
        kwargs = call.kwargs
        # Path format:   <root>/global_step_N/{model|optimizer|extra}/dist_ckpt
        ckpt_path = kwargs["ckpt_path"]
        normalized = os.path.normpath(ckpt_path)
        subdir = os.path.basename(os.path.dirname(normalized))
        out[subdir] = kwargs["sharded_state_dict"]
    return out


class TestSaveCheckpointDispatch:
    # [CLASS DOC]
    # Tests which subtrees save_checkpoint writes for each content/backend combination.
    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        # [FUNCTION DOC]
        # HELPER: autouse fixture creating and finally removing a temp directory in self.test_dir.
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _save_path(self, step=0):
        # [FUNCTION DOC]
        # HELPER: returns <test_dir>/global_step_<step> (default step 0).
        return os.path.join(self.test_dir, f"global_step_{step}")

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_mbridge_split_optimizer_extra(self, mock_save_dc):
        # [FUNCTION DOC]
        # Bridge path: dist saves only to optimizer and extra subdirs (with optimizer / rng_state contents, no model
        # key anywhere) and the bridge saves weights once. Inputs: mock_save_dc (patched save function).
        """With mbridge, optimizer and extra go to SEPARATE dist_ckpt directories; model via bridge."""
        mgr = _make_manager(save_contents=["model", "optimizer", "extra"])
        with patch.object(mgr, "_save_transformer_config"):
            mgr.save_checkpoint(self._save_path(), global_step=1)

        calls = _collect_save_calls(mock_save_dc)
        assert set(calls) == {"optimizer", "extra"}
        assert "optimizer" in calls["optimizer"]
        assert "rng_state" in calls["extra"]
        # model must NOT be in any dist_ckpt tree (it lives under model/huggingface/)
        assert all("model" not in sd for sd in calls.values())
        mgr.bridge.save_weights.assert_called_once()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_mbridge_model_only_no_dist_ckpt(self, mock_save_dc):
        # [FUNCTION DOC]
        # Model only with bridge: no dist save, bridge save once.
        """With mbridge and save_contents=['model'], dist_checkpointing is not called."""
        mgr = _make_manager(save_contents=["model"])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        mock_save_dc.assert_not_called()
        mgr.bridge.save_weights.assert_called_once()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_optimizer_only_writes_only_optimizer_subdir(self, mock_save_dc):
        # [FUNCTION DOC]
        # Optimizer only: only the optimizer subdir is written, with no model key, and no bridge save.
        """save_contents=['optimizer'] writes ONLY the optimizer/ subtree."""
        mgr = _make_manager(save_contents=["optimizer"])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        calls = _collect_save_calls(mock_save_dc)
        assert set(calls) == {"optimizer"}
        assert "optimizer" in calls["optimizer"]
        assert "model" not in calls["optimizer"]
        mgr.bridge.save_weights.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_extra_only(self, mock_save_dc):
        # [FUNCTION DOC]
        # Extra only: only extra subdir with rng_state, and the model sharded state dict is not built.
        """save_contents=['extra'] writes ONLY the extra/ subtree; model sharded SD not built."""
        mgr = _make_manager(save_contents=["extra"])
        with patch.object(mgr, "_save_transformer_config"):
            mgr.save_checkpoint(self._save_path(), global_step=1)

        calls = _collect_save_calls(mock_save_dc)
        assert set(calls) == {"extra"}
        assert "rng_state" in calls["extra"]
        assert "optimizer" not in calls["extra"]
        mgr.model[0].sharded_state_dict.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_dist_ckpt_backend_splits_into_three(self, mock_save_dc):
        # [FUNCTION DOC]
        # Dist backend: model, optimizer and extra subdirs each written with their respective keys.
        """With dist_ckpt backend, model/optimizer/extra go to three separate directories."""
        mgr = _make_manager(
            save_contents=["model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )
        with patch.object(mgr, "_save_transformer_config"):
            mgr.save_checkpoint(self._save_path(), global_step=1)

        calls = _collect_save_calls(mock_save_dc)
        assert set(calls) == {"model", "optimizer", "extra"}
        assert "model" in calls["model"]
        assert "optimizer" in calls["optimizer"]
        assert "rng_state" in calls["extra"]

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_hf_model_with_mbridge_deduplicates(self, mock_save_dc):
        # [FUNCTION DOC]
        # model + hf_model with bridge: weights saved once via bridge, no dist save.
        """save_contents=['model', 'hf_model'] with mbridge saves model once via bridge."""
        mgr = _make_manager(save_contents=["model", "hf_model"])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        mgr.bridge.save_weights.assert_called_once()
        mock_save_dc.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_hf_model_and_dist_ckpt_saves_both(self, mock_save_dc):
        # [FUNCTION DOC]
        # Dist + bridge with model, hf_model, optimizer, extra: bridge save once and all three dist subdirs written.
        """With mbridge + dist_checkpointing, ``model`` and ``hf_model`` write shards and HF tree."""
        mgr = _make_manager(
            save_contents=["model", "hf_model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )
        with patch.object(mgr, "_save_transformer_config"):
            mgr.save_checkpoint(self._save_path(), global_step=1)

        mgr.bridge.save_weights.assert_called_once()
        mock_save_dc.assert_called()
        calls = _collect_save_calls(mock_save_dc)
        assert "model" in calls and "optimizer" in calls and "extra" in calls

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_peft_adapters_under_model_subdir(self, mock_save_dc):
        # [FUNCTION DOC]
        # With a PEFT class (filter patched to identity), adapters are dist-saved under model and the optimizer
        # separately.
        """PEFT adapter shards live under model/dist_ckpt/ even with mbridge; optimizer stays separate."""
        mgr = _make_manager(save_contents=["model", "optimizer"], peft_cls=MagicMock())

        with patch.object(mgr, "_maybe_filter_peft_state_dict", side_effect=lambda sd: sd):
            mgr.save_checkpoint(self._save_path(), global_step=1)

        calls = _collect_save_calls(mock_save_dc)
        assert "model" in calls
        assert "optimizer" in calls
        assert "optimizer" in calls["optimizer"]

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_empty_save_contents(self, mock_save_dc):
        # [FUNCTION DOC]
        # Empty contents: no dist save and no bridge save.
        """Empty save_contents should not trigger any saving."""
        mgr = _make_manager(save_contents=[])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        mock_save_dc.assert_not_called()
        mgr.bridge.save_weights.assert_not_called()


# ===========================================================================
# Tests: load_checkpoint dispatch
# ===========================================================================

_PATCH_LOAD_META = "megatron.core.dist_checkpointing.load_content_metadata"


def _make_v2_layout(ckpt_path: str, *subdirs: str) -> None:
    # [FUNCTION DOC]
    # HELPER: for each given subdir name creates <ckpt_path>/<sub>/dist_ckpt with an empty "common.pt"
    # sentinel file. Side effects: writes to disk.
    """Create the v2 split layout with the given subdirs non-empty.

    ``subdirs`` is a subset of ``{"model", "optimizer", "extra"}``.  For each,
    we create ``<ckpt_path>/<sub>/dist_ckpt`` and put a sentinel file in it so
    ``_has_checkpoint_files`` returns True.
    """
    for sub in subdirs:
        dpath = os.path.join(ckpt_path, sub, "dist_ckpt")
        os.makedirs(dpath, exist_ok=True)
        # dist_checkpointing writes arbitrary per-shard files; a dummy
        # sentinel is enough for our in-repo _has_checkpoint_files gate.
        sentinel = os.path.join(dpath, "common.pt")
        with open(sentinel, "w") as f:
            f.write("")


def _load_calls_by_subdir(mock_load_dc):
    # [FUNCTION DOC]
    # HELPER: maps each mocked load call to {subdir: sharded_state_dict} using the parent directory name of the
    # ckpt_dir keyword.
    """Return {subdir: sharded_state_dict} from load_dist_checkpointing invocations."""
    out = {}
    for call in mock_load_dc.call_args_list:
        kwargs = call.kwargs
        subdir = os.path.basename(os.path.dirname(os.path.normpath(kwargs["ckpt_dir"])))
        out[subdir] = kwargs["sharded_state_dict"]
    return out


class TestLoadCheckpointDispatch:
    # [CLASS DOC]
    # Tests which subtrees load_checkpoint reads.
    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        # [FUNCTION DOC]
        # HELPER: autouse fixture creating a temp dir and a global_step_1 checkpoint dir (self.ckpt_path); removes
        # it afterwards.
        self.test_dir = tempfile.mkdtemp()
        self.ckpt_path = os.path.join(self.test_dir, "global_step_1")
        os.makedirs(self.ckpt_path, exist_ok=True)
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_hf_weight_load_model_via_bridge(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # HF path: model loaded via the bridge helper once; dist loads only from optimizer and extra subdirs.
        # Inputs: mock_load_dc, _mock_meta (patched functions).
        """On the HF path: model loads via bridge; optimizer/extra come from separate dist_ckpt dirs."""
        _make_v2_layout(self.ckpt_path, "optimizer", "extra")

        def fake_load(sharded_state_dict, ckpt_dir):
            # [FUNCTION DOC]
            # HELPER: returns optimizer+lr_scheduler state if the request contains "optimizer", an rng_state list if it
            # contains "rng_state", else an empty mapping.
            if "optimizer" in sharded_state_dict:
                return {"optimizer": {"step": 1}, "lr_scheduler": {"last_epoch": 5}}
            if "rng_state" in sharded_state_dict:
                return {"rng_state": [{"random_rng_state": None}]}
            return {}

        mock_load_dc.side_effect = fake_load
        mgr = _make_manager(load_contents=["model", "optimizer", "extra"])

        with (
            patch.object(mgr, "_load_model_as_hf_via_bridge") as mock_bridge_load,
            patch.object(mgr, "load_rng_states"),
        ):
            mgr.load_checkpoint(self.ckpt_path)
            mock_bridge_load.assert_called_once()

        calls = _load_calls_by_subdir(mock_load_dc)
        assert set(calls) == {"optimizer", "extra"}
        assert "optimizer" in calls["optimizer"]
        assert "rng_state" in calls["extra"]
        assert "model" not in calls["optimizer"]

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_dist_ckpt_load_model_from_each_subdir(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # Dist backend: loads from model, optimizer and extra subdirs and calls the model's load_state_dict once.
        """With dist_ckpt backend, all three subtrees are loaded independently."""
        _make_v2_layout(self.ckpt_path, "model", "optimizer", "extra")

        def fake_load(sharded_state_dict, ckpt_dir):
            # [FUNCTION DOC]
            # HELPER: returns a model state with a zero tensor, optimizer state, or rng_state depending on which key the
            # request contains; otherwise an empty mapping.
            if "model" in sharded_state_dict:
                return {"model": {"layer.weight": torch.zeros(1)}}
            if "optimizer" in sharded_state_dict:
                return {"optimizer": {"step": 1}}
            if "rng_state" in sharded_state_dict:
                return {"rng_state": [{"random_rng_state": None}]}
            return {}

        mock_load_dc.side_effect = fake_load
        mgr = _make_manager(
            load_contents=["model", "optimizer", "extra"],
            use_dist_checkpointing=True,
        )

        with patch.object(mgr, "load_rng_states"):
            mgr.load_checkpoint(self.ckpt_path)

        calls = _load_calls_by_subdir(mock_load_dc)
        assert set(calls) == {"model", "optimizer", "extra"}
        assert "model" in calls["model"]
        assert "optimizer" in calls["optimizer"]
        assert "rng_state" in calls["extra"]
        mgr.model[0].load_state_dict.assert_called_once()

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_load_optimizer_only(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # Optimizer only: loads only the optimizer subdir (no rng_state) and calls the optimizer's load_state_dict once.
        """load_contents=['optimizer'] touches only the optimizer subtree."""
        _make_v2_layout(self.ckpt_path, "optimizer")
        mock_load_dc.return_value = {"optimizer": {"step": 1}}
        mgr = _make_manager(load_contents=["optimizer"])
        mgr.load_checkpoint(self.ckpt_path)

        calls = _load_calls_by_subdir(mock_load_dc)
        assert set(calls) == {"optimizer"}
        assert "optimizer" in calls["optimizer"]
        assert "rng_state" not in calls["optimizer"]
        mgr.optimizer.load_state_dict.assert_called_once()

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_load_extra_only(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # Extra only: loads only extra and does not build the model sharded state dict.
        """load_contents=['extra'] loads only RNG states from the extra subtree."""
        _make_v2_layout(self.ckpt_path, "extra")
        mock_load_dc.return_value = {"rng_state": [{"random_rng_state": None}]}
        mgr = _make_manager(load_contents=["extra"])

        with patch.object(mgr, "load_rng_states"):
            mgr.load_checkpoint(self.ckpt_path)

        calls = _load_calls_by_subdir(mock_load_dc)
        assert set(calls) == {"extra"}
        assert "rng_state" in calls["extra"]
        mgr.model[0].sharded_state_dict.assert_not_called()

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_load_rejects_legacy_layout(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # A root-level dist_ckpt directory (legacy layout) raises RuntimeError "deprecated Megatron checkpoint
        # layout" before any load call.
        """Loading a pre-v2 checkpoint (bare dist_ckpt/ or huggingface/ at root) must fail fast."""
        # Legacy layout: dist_ckpt/ directly at the root, NO model/optimizer/extra.
        legacy_dir = os.path.join(self.ckpt_path, "dist_ckpt")
        os.makedirs(legacy_dir, exist_ok=True)
        with open(os.path.join(legacy_dir, "common.pt"), "w") as f:
            f.write("")

        mgr = _make_manager(load_contents=["optimizer"])
        with pytest.raises(RuntimeError, match="deprecated Megatron checkpoint layout"):
            mgr.load_checkpoint(self.ckpt_path)
        mock_load_dc.assert_not_called()


# ===========================================================================
# Tests: save_checkpoint side effects (HF config, transformer config)
# ===========================================================================


class TestSaveCheckpointSideEffects:
    # [CLASS DOC]
    # Tests HF config / tokenizer and transformer config side files.
    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        # [FUNCTION DOC]
        # HELPER: autouse temp-dir fixture (self.test_dir).
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _save_path(self, step=0):
        # [FUNCTION DOC]
        # HELPER: returns <test_dir>/global_step_<step>.
        return os.path.join(self.test_dir, f"global_step_{step}")

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_hf_config_saved_with_mbridge(self, mock_save_dc):
        # [FUNCTION DOC]
        # Saving model via bridge also saves the HF config and processing class once each.
        mgr = _make_manager(save_contents=["model"])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        mgr.hf_config.save_pretrained.assert_called_once()
        mgr.processing_class.save_pretrained.assert_called_once()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_hf_config_not_saved_without_model(self, mock_save_dc):
        # [FUNCTION DOC]
        # Optimizer-only save does not save the HF config.
        mgr = _make_manager(save_contents=["optimizer"])
        mgr.save_checkpoint(self._save_path(), global_step=1)

        mgr.hf_config.save_pretrained.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_transformer_config_saved_with_extra(self, mock_save_dc):
        # [FUNCTION DOC]
        # Extra save writes the transformer config once.
        mgr = _make_manager(save_contents=["extra"])
        with patch.object(mgr, "_save_transformer_config") as mock_tc:
            mgr.save_checkpoint(self._save_path(), global_step=1)
            mock_tc.assert_called_once()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_transformer_config_not_saved_without_extra(self, mock_save_dc):
        # [FUNCTION DOC]
        # Model-only save does not write the transformer config.
        mgr = _make_manager(save_contents=["model"])
        with patch.object(mgr, "_save_transformer_config") as mock_tc:
            mgr.save_checkpoint(self._save_path(), global_step=1)
            mock_tc.assert_not_called()


# ===========================================================================
# Tests: model sharded state dict is NOT built when unnecessary
# ===========================================================================


class TestModelShardedStateDictNotBuiltUnnecessarily:
    # [CLASS DOC]
    # Tests that the (expensive) model sharded state dict is built only when needed.
    @pytest.fixture(autouse=True)
    def _tmpdir(self):
        # [FUNCTION DOC]
        # HELPER: autouse temp-dir fixture (self.test_dir).
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_extra_only_does_not_build_model_sd(self, mock_save_dc):
        # [FUNCTION DOC]
        # Extra-only save does not build it.
        mgr = _make_manager(save_contents=["extra"])
        with patch.object(mgr, "_save_transformer_config"):
            mgr.save_checkpoint(os.path.join(self.test_dir, "step_1"), global_step=1)
        mgr.model[0].sharded_state_dict.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_model_only_mbridge_does_not_build_model_sd(self, mock_save_dc):
        # [FUNCTION DOC]
        # Bridge model-only save does not build it.
        """mbridge model save uses bridge.save_weights, not model.sharded_state_dict."""
        mgr = _make_manager(save_contents=["model"])
        mgr.save_checkpoint(os.path.join(self.test_dir, "step_1"), global_step=1)
        mgr.model[0].sharded_state_dict.assert_not_called()

    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.save_dist_checkpointing", return_value=None)
    def test_optimizer_does_build_model_sd(self, mock_save_dc):
        # [FUNCTION DOC]
        # Optimizer save builds it exactly once (needed as optimizer metadata).
        """Optimizer save needs model sharded SD as metadata input."""
        mgr = _make_manager(save_contents=["optimizer"])
        mgr.save_checkpoint(os.path.join(self.test_dir, "step_1"), global_step=1)
        mgr.model[0].sharded_state_dict.assert_called_once()

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_load_extra_only_does_not_build_model_sd(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # Extra-only load does not build it.
        mock_load_dc.return_value = {"rng_state": [{"random_rng_state": None}]}
        mgr = _make_manager(load_contents=["extra"])
        ckpt_path = os.path.join(self.test_dir, "step_1")
        _make_v2_layout(ckpt_path, "extra")

        with patch.object(mgr, "load_rng_states"):
            mgr.load_checkpoint(ckpt_path)
        mgr.model[0].sharded_state_dict.assert_not_called()

    @patch(_PATCH_LOAD_META, return_value=None)
    @patch("verl.utils.checkpoint.megatron_checkpoint_manager.load_dist_checkpointing")
    def test_load_optimizer_builds_model_sd(self, mock_load_dc, _mock_meta):
        # [FUNCTION DOC]
        # Optimizer load builds it exactly once.
        """Loading optimizer needs model sharded SD as metadata input."""
        mock_load_dc.return_value = {"optimizer": {"step": 1}}
        mgr = _make_manager(load_contents=["optimizer"])
        ckpt_path = os.path.join(self.test_dir, "step_1")
        _make_v2_layout(ckpt_path, "optimizer")

        mgr.load_checkpoint(ckpt_path)
        mgr.model[0].sharded_state_dict.assert_called_once()
