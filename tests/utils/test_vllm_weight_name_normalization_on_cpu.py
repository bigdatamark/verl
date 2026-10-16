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
# CPU tests for receiver-side weight-name normalization used by the vLLM colocated
# worker extension when syncing weights into a LoRA-enabled vLLM model. At import
# time the real weight_update_utils.py and vllm_rollout/utils.py (plus the real
# verl/utils/vllm/utils.py providing resolve_weight_name and the module-level flag
# _HAS_LORA_LOAD_WEIGHTS) are loaded with vLLM and heavy verl modules stubbed.
# Tests drive resolve_weight_name against fake models' parameter namespaces to
# check when a ".base_layer" path segment is stripped, kept or added (depending on
# the vLLM version flag, whether the parent module has a real base_layer child,
# the hf-to-vllm mapper, packed module mapping, and fused-MoE expert aliases), and
# check the worker's _update_weights / update_weights_from_ipc paths (base sync
# resolves names; adapter sync passes LoRA tensors through and accumulates them
# across buckets into a single add_lora call). The two IPC tests need real vLLM.

"""CPU tests for the receiver-side weight-name normalization in
``vLLMColocateWorkerExtension``.

Covers the regression that motivated reverting PR #5599: with LoRA enabled, vLLM
wraps *every* linear layer (exposing base weights under ``.base_layer``), while
Megatron only LoRA-wraps the user's target_modules and exports plain names for
the rest (e.g. the Qwen3.5-VL vision merger ``merger.linear_fc1``). The receiver
reconciles each name against the live vLLM namespace rather than a hard-coded
list. Imports *only* from ``verl.workers.rollout.vllm_rollout.utils`` (deps
stubbed) to prove the receiver is decoupled from ``megatron_peft_utils``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_weight_update_utils():
    # [FUNCTION DOC]
    # HELPER: Loads weight_update_utils.py directly from its file path as a module
    # named "weight_update_utils". Outputs: the module.
    # KNOWN ISSUE: module_from_spec is called before asserting the spec is non-None.
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/weight_update_utils.py"
    spec = importlib.util.spec_from_file_location("weight_update_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_vllm_rollout_utils():
    # [FUNCTION DOC]
    # HELPER: Installs fake modules for vllm (outputs, lora layers/base without
    # load_weights, lora request/utils/worker_manager/models), verl.third_party.vllm
    # (sleep level 1, version 0.8.0), verl.utils.vllm (fake TensorLoRARequest and
    # VLLMHijack), its patch and quant-utils submodules (no-op functions), and the
    # preloaded weight_update_utils; then loads the real verl/utils/vllm/utils.py,
    # exposes its resolve_weight_name and flag on the fake package, loads
    # vllm_rollout/utils.py under its real module name, attaches the real utils
    # module as _vllm_utils_real, and finally restores every replaced module entry.
    # Outputs: the loaded vllm_rollout utils module.
    # Side effects: temporary module-table mutation; deliberately does not stub the
    # platform module to avoid leaking a fake into verl.utils.device.
    """Load ``vllm_rollout/utils.py`` with heavyweight deps stubbed (no GPU/vLLM needed)."""
    module_name = "verl.workers.rollout.vllm_rollout.utils"
    module_path = _REPO_ROOT / "verl/workers/rollout/vllm_rollout/utils.py"

    fake_outputs = types.ModuleType("vllm.outputs")

    class _FakeRequestOutput:
        # [CLASS DOC]
        # HELPER: Placeholder for vLLM RequestOutput.
        pass

    fake_outputs.RequestOutput = _FakeRequestOutput
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.outputs = fake_outputs

    # Stub ``vllm.lora.layers.base.BaseLayerWithLoRA`` WITHOUT ``load_weights`` --
    # the module-level probe (``"load_weights" in BaseLayerWithLoRA.__dict__``)
    # then falls to False, matching released vLLM. Tests that exercise the
    # newer-vLLM path monkeypatch ``_HAS_LORA_LOAD_WEIGHTS`` so both branches
    # are covered.
    fake_lora_base = types.ModuleType("vllm.lora.layers.base")

    class _FakeBaseLayerWithLoRA:
        # [CLASS DOC]
        # HELPER: Placeholder BaseLayerWithLoRA lacking load_weights, so the version flag
        # evaluates False (released-vLLM behavior).
        pass

    fake_lora_base.BaseLayerWithLoRA = _FakeBaseLayerWithLoRA
    fake_lora_layers = types.ModuleType("vllm.lora.layers")
    fake_lora_layers.base = fake_lora_base
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora.layers = fake_lora_layers
    fake_vllm.lora = fake_lora

    fake_vllm_third_party = types.ModuleType("verl.third_party.vllm")
    fake_vllm_third_party.VLLM_SLEEP_LEVEL = 1
    fake_vllm_third_party.get_version = lambda pkg: "0.8.0"

    fake_vllm_utils = types.ModuleType("verl.utils.vllm")

    class _FakeTensorLoRARequest:
        # [CLASS DOC]
        # HELPER: Fake TensorLoRARequest storing lora_name, lora_int_id, lora_path,
        # peft_config and lora_tensors from keyword args.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Copies the five known keyword args onto attributes (None if absent).
            self.lora_name = kwargs.get("lora_name")
            self.lora_int_id = kwargs.get("lora_int_id")
            self.lora_path = kwargs.get("lora_path")
            self.peft_config = kwargs.get("peft_config")
            self.lora_tensors = kwargs.get("lora_tensors")

    class _FakeVLLMHijack:
        # [CLASS DOC]
        # HELPER: Fake VLLMHijack with a no-op hijack.
        @staticmethod
        def hijack():
            # [FUNCTION DOC]
            # HELPER: Does nothing; returns None.
            return None

    fake_vllm_utils.TensorLoRARequest = _FakeTensorLoRARequest
    fake_vllm_utils.VLLMHijack = _FakeVLLMHijack

    fake_vllm_patch = types.ModuleType("verl.utils.vllm.patch")
    fake_vllm_patch.patch_vllm_moe_model_weight_loader = lambda model: None

    fake_vllm_quant = types.ModuleType("verl.utils.vllm.vllm_quant_utils")
    fake_vllm_quant.apply_vllm_quant_patches = lambda: None
    fake_vllm_quant.is_quantized_model = lambda config: False
    fake_vllm_quant.load_quanted_weights = lambda *a, **k: []

    # NOTE: deliberately do NOT stub verl.plugin.platform. It is lightweight and
    # imports fine on CPU. verl.utils.device binds `get_platform` at import time, so
    # a `lambda: None` stub gets baked into verl.utils.device during this window;
    # because only the keys in `fakes` below are restored, that fake would leak
    # process-wide and later tests would crash in get_device_name() with
    # "'NoneType' object has no attribute 'device_name'".

    # Minimal stubs for the heavyweight vllm.lora.* imports pulled in by
    # ``verl/utils/vllm/utils.py`` (which also defines ``resolve_weight_name``).
    # Only the symbols touched at import time need to exist.
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = type("LoRARequest", (), {})
    fake_lora_utils = types.ModuleType("vllm.lora.utils")
    fake_lora_utils.get_adapter_absolute_path = lambda p: p
    fake_lora_worker_manager = types.ModuleType("vllm.lora.worker_manager")
    fake_lora_worker_manager.LRUCacheWorkerLoRAManager = type("LRUCacheWorkerLoRAManager", (), {})
    fake_lora_model = types.ModuleType("vllm.lora.lora_model")
    fake_lora_model.LoRAModel = type("LoRAModel", (), {})
    fake_lora_models = types.ModuleType("vllm.lora.models")
    fake_lora_models.LoRAModel = type("LoRAModel", (), {})

    fakes = {
        "vllm": fake_vllm,
        "vllm.outputs": fake_outputs,
        "vllm.lora": fake_lora,
        "vllm.lora.layers": fake_lora_layers,
        "vllm.lora.layers.base": fake_lora_base,
        "vllm.lora.request": fake_lora_request,
        "vllm.lora.utils": fake_lora_utils,
        "vllm.lora.worker_manager": fake_lora_worker_manager,
        "vllm.lora.lora_model": fake_lora_model,
        "vllm.lora.models": fake_lora_models,
        "verl.third_party.vllm": fake_vllm_third_party,
        "verl.utils.vllm": fake_vllm_utils,
        "verl.utils.vllm.patch": fake_vllm_patch,
        "verl.utils.vllm.vllm_quant_utils": fake_vllm_quant,
        "verl.workers.rollout.vllm_rollout.weight_update_utils": _weight_update_utils,
    }

    saved = {name: sys.modules.get(name) for name in fakes}
    try:
        sys.modules.update(fakes)
        # Load the real ``verl/utils/vllm/utils.py`` under the stubbed deps so
        # ``vllm_rollout/utils.py``'s ``from verl.utils.vllm import resolve_weight_name``
        # resolves to the genuine function (with its module-level ``_HAS_LORA_LOAD_WEIGHTS``
        # probe falling to False since the stub ``BaseLayerWithLoRA`` has no load_weights).
        utils_path = _REPO_ROOT / "verl/utils/vllm/utils.py"
        utils_spec = importlib.util.spec_from_file_location("verl.utils.vllm.utils_real", utils_path)
        utils_module = importlib.util.module_from_spec(utils_spec)
        assert utils_spec is not None and utils_spec.loader is not None
        utils_spec.loader.exec_module(utils_module)
        fake_vllm_utils.resolve_weight_name = utils_module.resolve_weight_name
        fake_vllm_utils._HAS_LORA_LOAD_WEIGHTS = utils_module._HAS_LORA_LOAD_WEIGHTS

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        module._vllm_utils_real = utils_module
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    return module


_weight_update_utils = _load_weight_update_utils()
_vllm_rollout_utils = _load_vllm_rollout_utils()
vLLMColocateWorkerExtension = _vllm_rollout_utils.vLLMColocateWorkerExtension
# The real ``verl/utils/vllm/utils.py`` module hosting ``resolve_weight_name``
# (loaded under stubbed deps); monkeypatch its ``_HAS_LORA_LOAD_WEIGHTS`` to flip
# the vLLM-version branch under test.
_vllm_utils_real = _vllm_rollout_utils._vllm_utils_real
resolve_weight_name = _vllm_utils_real.resolve_weight_name


class _FakeMapper:
    # [CLASS DOC]
    # HELPER: Minimal vLLM WeightsMapper stand-in: for each name, replaces the first
    # occurrence of each mapping key substring with its value, in mapping order.
    """Minimal stand-in for vLLM ``WeightsMapper``.

    Mimics substring-based ``orig_to_new_substr`` / ``orig_to_new_stacked``
    mapping: each key is a substring that is replaced (once) in the name. This
    matches how the real mapper rewrites names like ``in_proj_qkv`` ->
    ``in_proj_qkvz`` regardless of the trailing ``.weight`` / ``.base_layer``.
    """

    def __init__(self, mapping: dict[str, str]):
        # [FUNCTION DOC]
        # HELPER: Stores the substring mapping.
        self.mapping = mapping

    def apply_list(self, names: list[str]) -> list[str]:
        # [FUNCTION DOC]
        # HELPER: Applies the substring replacements to each name. Outputs: new list.
        out = []
        for name in names:
            mapped = name
            for substr, new in self.mapping.items():
                if substr in mapped:
                    mapped = mapped.replace(substr, new, 1)
            out.append(mapped)
        return out


class _FakeModel:
    # [CLASS DOC]
    # HELPER: Fake vLLM model exposing named_parameters/named_buffers, an optional
    # hf_to_vllm_mapper and packed_modules_mapping, and a recording load_weights.
    """A fake vLLM model exposing only the introspection surface the resolver uses."""

    def __init__(self, params: dict[str, torch.Tensor], *, mapper=None, packed=None):
        # [FUNCTION DOC]
        # HELPER: Copies the name-to-tensor mapping and stores mapper and packed mapping.
        # name -> tensor (for named_parameters / named_buffers)
        self._params = dict(params)
        self.hf_to_vllm_mapper = mapper
        self.packed_modules_mapping = packed

    def named_parameters(self, remove_duplicate: bool = False):
        # [FUNCTION DOC]
        # HELPER: Yields (name, tensor) pairs; remove_duplicate is ignored.
        del remove_duplicate
        yield from self._params.items()

    def named_buffers(self):
        # [FUNCTION DOC]
        # HELPER: Returns an empty iterator (no buffers).
        return iter(())

    def load_weights(self, weights):
        # [FUNCTION DOC]
        # HELPER: Records the names of the given (name, tensor) pairs in self.loaded.
        # Side effects: sets self.loaded.
        # Record resolved names for assertions.
        self.loaded = [name for name, _ in weights]


class _FakeVllmConfig:
    # [CLASS DOC]
    # HELPER: Fake vLLM config with speculative and quant configs and an empty model config.
    def __init__(self, speculative_config=None, quant_config=None):
        # [FUNCTION DOC]
        # HELPER: Stores speculative_config and quant_config; model_config is an empty namespace.
        self.speculative_config = speculative_config
        self.quant_config = quant_config
        self.model_config = types.SimpleNamespace()


def _make_worker(model):
    # [FUNCTION DOC]
    # HELPER: Creates a worker extension instance without running its constructor and
    # attaches a model runner namespace holding the model, a fake vLLM config and no drafter.
    worker = object.__new__(vLLMColocateWorkerExtension)
    worker.model_runner = types.SimpleNamespace(model=model, vllm_config=_FakeVllmConfig(), drafter=None)
    return worker


def _resolve(worker, model, name):
    # [FUNCTION DOC]
    # HELPER: Builds the set of parameter and buffer names of the model and calls
    # resolve_weight_name(model, name, names). The worker argument is unused.
    # Outputs: resolved name.
    del worker  # resolve_weight_name is a pure function (no worker state)
    names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    names.update(n for n, _ in model.named_buffers())
    return resolve_weight_name(model, name, names)


def _mapped(worker, model, name):
    # [FUNCTION DOC]
    # HELPER: Applies the model's hf_to_vllm mapper to one name (identity if no
    # mapper or empty result). The worker argument is unused.
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return name
    out = mapper.apply_list([name])
    return out[0] if out else name


# ---------------------------------------------------------------------------
# The exact nohup.out failure: vision merger leaf param with .base_layer must
# resolve to the unwrapped name.
# ---------------------------------------------------------------------------


def test_vision_merger_base_layer_is_stripped():
    # [FUNCTION DOC]
    # Behavior: a plain (non-LoRA-wrapped) vision merger linear receives
    # ".base_layer" names for weight and bias; both resolve to the unwrapped names.
    """Qwen3.5-VL ``merger.linear_fc1.base_layer.weight`` -> ``merger.linear_fc1.weight``."""
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "merger.linear_fc1.bias": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    assert _resolve(worker, model, "merger.linear_fc1.base_layer.weight") == "merger.linear_fc1.weight"
    assert _resolve(worker, model, "merger.linear_fc1.base_layer.bias") == "merger.linear_fc1.bias"


# ---------------------------------------------------------------------------
# Mixed LoRA model: language layers are LoRA-wrapped (have .base_layer), but
# the vision merger is a plain linear (no base_layer child). On vLLM 0.26.0
# the suffix must be stripped ONLY for the merger, not kept model-wide.
# ---------------------------------------------------------------------------


def test_mixed_model_vision_merger_strips_on_released_vllm():
    # [FUNCTION DOC]
    # Behavior: on released vLLM (flag False) in a mixed model, merger names are
    # stripped while a LoRA-wrapped language q_proj base_layer name is kept as is.
    """vLLM 0.26.0 (_HAS_LORA_LOAD_WEIGHTS=False) with a mixed model: the
    vision merger's ``.base_layer.`` is stripped (no real base_layer child),
    while a coexisting LoRA-wrapped language leaf keeps the suffix."""
    # _HAS_LORA_LOAD_WEIGHTS defaults to False here (stub base layer), matching
    # vLLM 0.26.0.
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "merger.linear_fc1.bias": torch.empty(0),
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    assert _resolve(worker, model, "merger.linear_fc1.base_layer.weight") == "merger.linear_fc1.weight"
    assert _resolve(worker, model, "merger.linear_fc1.base_layer.bias") == "merger.linear_fc1.bias"
    # The language leaf keeps .base_layer on vLLM 0.26.0.
    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == name


def test_mixed_model_with_prefix_rewriting_mapper():
    # [FUNCTION DOC]
    # Behavior: with a prefix-rewriting mapper, the has-base-layer check uses the
    # mapped name: the language gate's base_layer name (in Bridge prefix) is kept,
    # and merger names (Bridge prefix) are stripped, all returned in the incoming
    # unmapped prefix.
    """The Qwen3.5-VL regression: the Bridge exports language params under
    ``model.language_model.`` and the merger under ``model.visual.``, while the
    live vLLM namespace is ``language_model.model.`` / ``visual.``. vLLM's
    ``hf_to_vllm_mapper`` rewrites the prefixes (``orig_to_new_prefix``).

    The parent-has-base-layer check must consult the *mapper-rewritten* name --
    otherwise the LoRA-wrapped language leaf's real ``base_layer`` is invisible
    under the Bridge prefix and gets wrongly stripped, crashing
    ``AutoWeightsLoader`` (it then can't find ``layers.0.mlp.gate.weight``
    inside ``Qwen3_5Model``)."""
    mapper = _FakeMapper(
        {
            # substring replacements approximating WeightsMapper orig_to_new_prefix
            "model.visual.": "visual.",
            "model.language_model.": "language_model.model.",
        }
    )
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.gate.base_layer.weight": torch.empty(0),
            "visual.merger.linear_fc1.weight": torch.empty(0),
            "visual.merger.linear_fc1.bias": torch.empty(0),
        },
        mapper=mapper,
    )
    worker = _make_worker(model)

    # Language leaf: incoming Bridge prefix differs from vLLM's; the real
    # base_layer child exists under the rewritten prefix -> KEEP the suffix.
    name = "model.language_model.layers.0.mlp.gate.base_layer.weight"
    assert _resolve(worker, model, name) == name
    # Merger: no base_layer child under either prefix -> STRIP.
    assert _resolve(worker, model, "model.visual.merger.linear_fc1.base_layer.weight") == (
        "model.visual.merger.linear_fc1.weight"
    )
    assert _resolve(worker, model, "model.visual.merger.linear_fc1.base_layer.bias") == (
        "model.visual.merger.linear_fc1.bias"
    )


# ---------------------------------------------------------------------------
# LoRA-wrapped language params KEEP .base_layer (the suffix is real there).
# ---------------------------------------------------------------------------


def test_lora_wrapped_base_layer_gets_stripped(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM (flag True) a LoRA-wrapped leaf's ".base_layer" is
    # stripped to the inner name.
    # Inputs: monkeypatch - sets the flag.
    # KNOWN ISSUE: the preceding section banner says LoRA-wrapped params KEEP
    # .base_layer, which contradicts this test (that only holds on released vLLM).
    """A LoRA-wrapped leaf with .base_layer is stripped: vLLM's
    BaseLayerWithLoRA.load_weights delegates to AutoWeightsLoader(base_layer)
    which expects the inner name (no base_layer. prefix)."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == ("language_model.model.layers.0.self_attn.q_proj.weight")


def test_missing_base_layer_not_added_on_newer_vllm(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM, a name without ".base_layer" is not suffixed even
    # though the model only has the base_layer form.
    """On newer vLLM, incoming without .base_layer is NOT suffixed --
    BaseLayerWithLoRA.load_weights delegates to AutoWeightsLoader(base_layer)
    expecting inner names."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, name) == name


def test_missing_base_layer_is_added_on_released_vllm():
    # [FUNCTION DOC]
    # Behavior: on released vLLM, a name without ".base_layer" whose base_layer form
    # exists is suffixed with ".base_layer".
    """On released vLLM (no BaseLayerWithLoRA.load_weights), an incoming leaf
    without ``.base_layer`` is suffixed so the model's load_weights recurses
    into the ``base_layer`` child and matches the real param."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, name) == ("language_model.model.layers.0.self_attn.q_proj.base_layer.weight")


def test_suffixed_leaf_is_kept_on_released_vllm():
    # [FUNCTION DOC]
    # Behavior: on released vLLM, an incoming base_layer name matching a real
    # parameter is kept.
    """On released vLLM, an incoming ``.base_layer.`` leaf is kept as-is
    (the loader recurses and matches the suffixed param -- the strip is gated
    off when the model has ``.base_layer`` params and the version is old)."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _resolve(worker, model, name) == name


# ---------------------------------------------------------------------------
# Non-leaf fused-MoE expert alias: strip Bridge-inserted .base_layer.
# ---------------------------------------------------------------------------


def test_expert_alias_base_layer_gets_stripped():
    # [FUNCTION DOC]
    # Behavior: a non-leaf expert alias "experts.base_layer.gate_up_proj" is stripped
    # to "experts.gate_up_proj" when that parameter exists.
    """``experts.base_layer.gate_up_proj`` (non-leaf) -> ``experts.gate_up_proj`` when that exists."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.experts.gate_up_proj": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    name = "language_model.model.layers.0.mlp.experts.base_layer.gate_up_proj"
    assert _resolve(worker, model, name) == "language_model.model.layers.0.mlp.experts.gate_up_proj"


def test_packed_routed_expert_alias_routed_through_base_layer():
    # [FUNCTION DOC]
    # Behavior: when the model's experts are LoRA-wrapped (parameters exist under
    # experts.base_layer.routed_experts), packed aliases experts.gate_up_proj and
    # experts.down_proj are rerouted under experts.base_layer.
    """Qwen3.5 routed experts: Bridge exports packed ``experts.gate_up_proj``
    (non-leaf). vLLM's LoRA-wrapped MoE (``FusedMoE3DWithLoRA``) has no
    ``load_weights``, so AutoWeightsLoader can't dispatch it -- route it under
    ``experts.base_layer.gate_up_proj`` so the loader recurses into ``base_layer``
    (MoERunner.load_weights -> routed_experts.load_weights). The ``lora_base_layer_prefix``
    clear (see patch_vllm_moe_model_weight_loader) then makes the mapping resolve."""
    model = _FakeModel(
        {
            "language_model.model.layers.0.mlp.experts.base_layer.routed_experts.w13_weight": torch.empty(0),
            "language_model.model.layers.0.mlp.experts.base_layer.routed_experts.w2_weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == ("language_model.model.layers.0.mlp.experts.base_layer.gate_up_proj")
    assert _resolve(worker, model, "language_model.model.layers.0.mlp.experts.down_proj") == (
        "language_model.model.layers.0.mlp.experts.base_layer.down_proj"
    )


def test_packed_routed_expert_alias_not_routed_without_lora():
    # [FUNCTION DOC]
    # Behavior: without a LoRA experts.base_layer, the packed alias is unchanged.
    """Without a LoRA experts.base_layer, packed aliases stay unchanged
    (AutoWeightsLoader dispatches directly to the MoERunner)."""
    model = _FakeModel({"language_model.model.layers.0.mlp.experts.gate_up_proj": torch.empty(0)})
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == incoming


# ---------------------------------------------------------------------------
# Packed/stacked names: the resolver toggles ``.base_layer`` on the UNMAPPED
# name (deciding the toggle via the mapper AND ``packed_modules_mapping``), but
# returns the unmapped form so vLLM's own ``load_weights`` does the stacking
# (returning the mapped name would double-map, e.g. ``in_proj_qkvz`` ->
# ``in_proj_qkvzz`` with a wrong shard_id).
# ---------------------------------------------------------------------------


def test_unpacked_proj_exists_via_packed_owner():
    # [FUNCTION DOC]
    # Behavior: an unpacked q_proj name whose packed owner qkv_proj exists is
    # returned unchanged.
    """An unpacked alias whose packed owner exists is returned unchanged."""
    model = _FakeModel(
        {"language_model.model.layers.0.self_attn.qkv_proj.weight": torch.empty(0)},
        packed={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.self_attn.q_proj.weight"
    assert _resolve(worker, model, incoming) == incoming


def test_stacked_gdn_in_proj_not_double_mapped(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM, the GDN in_proj_qkv name (whose mapped stacked owner
    # in_proj_qkvz has a base_layer) is returned unchanged, so mapping it afterwards
    # yields in_proj_qkvz.weight exactly once (no double mapping).
    """GDN ``in_proj_qkv`` is NOT suffixed with ``.base_layer`` on newer vLLM --
    the name stays unmapped so vLLM's mapper stacks it cleanly."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    mapper = _FakeMapper({".in_proj_qkv.": ".in_proj_qkvz."})
    model = _FakeModel(
        {"language_model.model.layers.0.linear_attn.in_proj_qkvz.base_layer.weight": torch.empty(0)},
        mapper=mapper,
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.linear_attn.in_proj_qkv.weight"
    resolved = _resolve(worker, model, incoming)
    # No .base_layer added on newer vLLM.
    assert resolved == incoming
    # vLLM's mapper then maps it cleanly (single z):
    assert _mapped(_make_worker(model), model, resolved) == (
        "language_model.model.layers.0.linear_attn.in_proj_qkvz.weight"
    )


def test_shared_expert_packed_owner_no_base_layer_on_newer_vllm(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM, a shared-expert gate_proj name is returned unchanged
    # and maps to gate_up_proj.weight.
    """On newer vLLM, shared expert gate_proj.weight is NOT suffixed with
    .base_layer -- vLLM's mapper stacks gate_proj -> gate_up_proj and
    BaseLayerWithLoRA.load_weights handles the inner name."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    mapper = _FakeMapper({".shared_expert.gate_proj.": ".shared_expert.gate_up_proj."})
    model = _FakeModel(
        {"language_model.model.layers.0.mlp.shared_expert.gate_up_proj.base_layer.weight": torch.empty(0)},
        mapper=mapper,
        packed={"gate_up_proj": ["gate_proj", "up_proj"]},
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight"
    resolved = _resolve(worker, model, incoming)
    assert resolved == incoming
    assert _mapped(_make_worker(model), model, resolved) == (
        "language_model.model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    )


def test_mapper_alias_left_to_vllm_mapper():
    # [FUNCTION DOC]
    # Behavior: a logical alias absent from the model is returned unchanged (left
    # for vLLM's mapper) even though the mapper maps it to an existing parameter.
    """A logical alias absent on the model is returned unchanged (not pre-mapped)."""
    model = _FakeModel(
        {"language_model.model.layers.0.mlp.experts.base_layer.w13_weight": torch.empty(0)},
        mapper=_FakeMapper(
            {
                "language_model.model.layers.0.mlp.experts.gate_up_proj": (
                    "language_model.model.layers.0.mlp.experts.base_layer.w13_weight"
                )
            }
        ),
    )
    worker = _make_worker(model)

    incoming = "language_model.model.layers.0.mlp.experts.gate_up_proj"
    assert _resolve(worker, model, incoming) == incoming


# ---------------------------------------------------------------------------
# Already-correct names are a no-op (the common case for non-VLM LoRA).
# ---------------------------------------------------------------------------


def test_already_matching_name_is_noop():
    # [FUNCTION DOC]
    # Behavior: a name that exactly matches a parameter is returned unchanged.
    model = _FakeModel({"language_model.model.layers.0.self_attn.q_proj.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "language_model.model.layers.0.self_attn.q_proj.weight") == (
        "language_model.model.layers.0.self_attn.q_proj.weight"
    )


def test_truly_unknown_name_strips_base_layer(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM, an unknown name still has ".base_layer." stripped.
    """Even unknown names have ``.base_layer.`` stripped on newer vLLM."""
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel({"a.base_layer.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "does.not.exist.base_layer.weight") == "does.not.exist.weight"


def test_drafter_drops_base_layer_wholesale():
    # [FUNCTION DOC]
    # Behavior: for a model with no base_layer parameters (e.g. an MTP drafter), the
    # segment is stripped from every name whether or not the result exists.
    """A model with no ``.base_layer`` params (e.g. the MTP drafter) has the
    suffix stripped wholesale from every name, regardless of whether the
    stripped form ultimately exists."""
    model = _FakeModel({"a.weight": torch.empty(0)})
    worker = _make_worker(model)

    assert _resolve(worker, model, "does.not.exist.base_layer.weight") == "does.not.exist.weight"
    assert _resolve(worker, model, "fc.base_layer.weight") == "fc.weight"


# ---------------------------------------------------------------------------
# End-to-end through _update_weights: load_weights receives resolved names.
# ---------------------------------------------------------------------------


def test_update_weights_resolves_names_before_load(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: on newer vLLM, _update_weights in the LoRA base-sync phase (peft config
    # set, base sync not done) passes resolved names to load_weights: merger name
    # stripped and q_proj base_layer name stripped.
    # LoRA base-sync phase (peft_config set, base_sync_done=False): the vLLM model
    # is LoRA-wrapped (has ``.base_layer``), so the resolver reconciles incoming
    # names against it. (``peft_config=None`` is the non-LoRA / merge path where the
    # model is never wrapped and the resolver is skipped -- it can't reach here.)
    monkeypatch.setattr(_vllm_utils_real, "_HAS_LORA_LOAD_WEIGHTS", True)
    model = _FakeModel(
        {
            "merger.linear_fc1.weight": torch.empty(0),
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0),
        }
    )
    worker = _make_worker(model)

    weights = [
        ("merger.linear_fc1.base_layer.weight", torch.empty(0)),
        ("language_model.model.layers.0.self_attn.q_proj.base_layer.weight", torch.empty(0)),
    ]

    worker._update_weights(weights, peft_config={"r": 1}, base_sync_done=False)

    assert model.loaded == [
        "merger.linear_fc1.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    ]


def test_lora_adapter_path_skips_normalization():
    # [FUNCTION DOC]
    # Behavior: in the adapter phase (peft config set, base sync done) LoRA tensors
    # are not normalized; add_lora is called with exactly the incoming tensor names.
    """LoRA adapter tensors (peft_config + base_sync_done) keep .base_layer — vLLM expects it."""
    model = _FakeModel({"language_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.empty(0)})

    added = []

    def _add_lora(request):
        # [FUNCTION DOC]
        # HELPER: Records the set of tensor names of the LoRA request.
        added.append(set(request.lora_tensors))

    worker = _make_worker(model)
    worker.add_lora = _add_lora

    # These are LoRA tensors (lora_A/lora_B), not base weights; the .base_layer here
    # is irrelevant to normalization and must be passed through verbatim.
    weights = [("language_model.model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(1))]
    worker._update_weights(weights, peft_config={"r": 1}, base_sync_done=True)

    assert added == [{"language_model.model.layers.0.self_attn.q_proj.lora_A.weight"}]


# ---------------------------------------------------------------------------
# Multi-bucket LoRA accumulation: add_lora is called exactly once, with the
# complete adapter, after is_last — never per-bucket.
# ---------------------------------------------------------------------------


class _FakeBucketReceiver:
    # [CLASS DOC]
    # HELPER: Fake bucketed receiver invoking the callback once per scripted
    # (weights, is_last) bucket.
    """Drives the on_bucket_received callback with a scripted bucket sequence."""

    def __init__(self, buckets):
        # [FUNCTION DOC]
        # HELPER: Stores the bucket script.
        # buckets: list of (weights_list, is_last)
        self._buckets = buckets

    def receive_weights(self, on_bucket_received):
        # [FUNCTION DOC]
        # HELPER: Calls on_bucket_received(weights, is_last) for each bucket in order.
        for weights, is_last in self._buckets:
            on_bucket_received(weights, is_last)


def test_update_weights_from_ipc_accumulates_lora_across_buckets(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: with the receiver patched to deliver two LoRA buckets, the adapter
    # path calls add_lora exactly once with both tensors and the first bucket's value
    # intact (cloned, not a reused view). Skipped without vLLM.
    # Side effects: patches the receiver class and worker attributes.
    """A LoRA adapter split across two buckets yields one add_lora with all tensors."""
    # Unlike the resolver tests above (which fully stub sys.modules), this drives the
    # real update_weights_from_ipc path and imports the actual bucketed_weight_transfer
    # module, whose package __init__ hard-requires vllm. vllm isn't in the `cpu` extra,
    # so this is skipped under cpu_unit_tests and run in vllm.yml (the vllm venv).
    pytest.importorskip("vllm")
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    monkeypatch.setattr(
        bwt,
        "BucketedWeightReceiver",
        lambda *a, **k: _FakeBucketReceiver(
            [
                ([("lora.A.weight", torch.ones(1))], False),
                ([("lora.B.weight", torch.zeros(1))], True),
            ]
        ),
    )

    model = _FakeModel({"q.base_layer.weight": torch.empty(0)})

    added = []

    def _add_lora(request):
        # [FUNCTION DOC]
        # HELPER: Records a copy of the LoRA request's tensor mapping.
        added.append(dict(request.lora_tensors))

    worker = _make_worker(model)
    worker.add_lora = _add_lora
    worker.remove_lora = lambda lora_id: None
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-lora.sock"

    worker.update_weights_from_ipc(peft_config={"r": 1}, base_sync_done=True)

    # Exactly one add_lora, holding both buckets' tensors (accumulated, not
    # overwritten) with the first bucket's tensor preserved (cloned, not a view
    # into a reused buffer).
    assert len(added) == 1
    assert set(added[0]) == {"lora.A.weight", "lora.B.weight"}
    torch.testing.assert_close(added[0]["lora.A.weight"], torch.ones(1))


def test_update_weights_from_ipc_standard_loads_per_bucket(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: standard base sync (no peft config) loads each bucket immediately,
    # in order (q.weight then k.weight), with post-load processing stubbed out.
    # Skipped without vLLM.
    """Standard (non-LoRA) base sync loads every bucket immediately (no accumulation)."""
    # See the note above: needs the real bucketed_weight_transfer (vllm-backed), which
    # the `cpu` extra can't provide, so it is skipped here and run in vllm.yml.
    pytest.importorskip("vllm")
    import verl.workers.rollout.vllm_rollout.bucketed_weight_transfer as bwt

    monkeypatch.setattr(
        bwt,
        "BucketedWeightReceiver",
        lambda *a, **k: _FakeBucketReceiver(
            [
                ([("q.weight", torch.ones(1))], False),
                ([("k.weight", torch.zeros(1))], True),
            ]
        ),
    )

    model = _FakeModel({"q.weight": torch.empty(0), "k.weight": torch.empty(0)})
    loaded = []
    model.load_weights = lambda weights: loaded.extend(name for name, _ in weights)

    worker = _make_worker(model)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker._get_zmq_handle = lambda: "ipc:///tmp/test-bucketed-std.sock"

    # Bypass the post-load process_weights_after_loading (needs real vLLM).
    fake_loader_utils = types.ModuleType("vllm.model_executor.model_loader.utils")
    fake_loader_utils.process_weights_after_loading = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.utils", fake_loader_utils)

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=False)

    assert loaded == ["q.weight", "k.weight"]
