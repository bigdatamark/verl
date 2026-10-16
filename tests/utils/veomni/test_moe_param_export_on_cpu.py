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
# CPU tests for VeOmni MoE parameter export handlers, loading
# verl/workers/engine/veomni/utils.py directly from its file path under a private
# module name. Covers GPT-OSS packed expert tensors (gate_up_proj, down_proj and
# their biases): without expert parallelism the passthrough handler exports them
# unchanged; with expert parallelism the streaming handler splits the local expert
# dim into per-expert HF names offset by the global expert id base, and
# enumerate_hf_slots reports per-expert names with per-expert shapes.

import importlib.util
from pathlib import Path

import pytest
import torch

_UTILS_PATH = Path(__file__).parents[3] / "verl" / "workers" / "engine" / "veomni" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("_veomni_moe_export_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_UTILS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_UTILS)

MOE_PARAM_HANDERS = _UTILS.MOE_PARAM_HANDERS
enumerate_hf_slots = _UTILS.enumerate_hf_slots
get_moe_param_handler = _UTILS.get_moe_param_handler
passthrough_moe_param_handler = _UTILS.passthrough_moe_param_handler
streamed_gpt_oss_moe_param_handler = _UTILS.streamed_gpt_oss_moe_param_handler


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("model.layers.0.mlp.experts.gate_up_proj", (4, 8, 12)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias", (4, 12)),
        ("model.layers.0.mlp.experts.down_proj", (4, 6, 8)),
        ("model.layers.0.mlp.experts.down_proj_bias", (4, 8)),
    ],
)
def test_gpt_oss_non_ep_keeps_packed_expert_params(name, shape):
    # [FUNCTION DOC]
    # Behavior: the registered GPT-OSS handler is the passthrough handler; with EP
    # disabled it yields exactly one (name, tensor) pair with the original name and the
    # identical tensor object.
    # Inputs: name, shape - parametrized packed expert params.
    tensor = torch.randn(shape)
    handler = get_moe_param_handler("gpt_oss", ep_enabled=False)

    assert MOE_PARAM_HANDERS["gpt_oss"] is passthrough_moe_param_handler
    exported = list(handler(name, tensor, expert_id_base=0))

    assert len(exported) == 1
    assert exported[0][0] == name
    assert exported[0][1] is tensor


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("model.layers.0.mlp.experts.gate_up_proj", (2, 8, 12)),
        ("model.layers.0.mlp.experts.gate_up_proj_bias", (2, 12)),
        ("model.layers.0.mlp.experts.down_proj", (2, 6, 8)),
        ("model.layers.0.mlp.experts.down_proj_bias", (2, 8)),
    ],
)
def test_gpt_oss_ep_streams_global_experts_in_checkpoint_layout(name, shape, monkeypatch):
    # [FUNCTION DOC]
    # Behavior: with EP enabled (device lookup patched to the tensor's device) the
    # handler is the streaming GPT-OSS handler; for 2 local experts with base 4 it
    # yields names with "mlp.experts.4." and "mlp.experts.5." inserted (no ".weight"
    # suffix added) and slices equal to the tensor's first-dim entries; slot
    # enumeration yields names for experts 0 and 1 with the shape minus the expert dim.
    # Inputs: name, shape; monkeypatch.
    # Note: slot enumeration uses expert ids starting at 0, not the export base.
    tensor = torch.randn(shape)
    monkeypatch.setattr(_UTILS, "get_device_id", lambda: tensor.device)
    handler = get_moe_param_handler("gpt_oss", ep_enabled=True)

    assert handler is streamed_gpt_oss_moe_param_handler
    exported = list(handler(name, tensor, expert_id_base=4))

    assert [exported_name for exported_name, _ in exported] == [
        name.replace("mlp.experts.", "mlp.experts.4."),
        name.replace("mlp.experts.", "mlp.experts.5."),
    ]
    assert all(not exported_name.endswith(".weight") for exported_name, _ in exported)
    torch.testing.assert_close(exported[0][1], tensor[0])
    torch.testing.assert_close(exported[1][1], tensor[1])

    slots = enumerate_hf_slots(handler, name, shape, tensor.dtype, device=tensor.device)
    assert slots == [
        (name.replace("mlp.experts.", "mlp.experts.0."), shape[1:]),
        (name.replace("mlp.experts.", "mlp.experts.1."), shape[1:]),
    ]
