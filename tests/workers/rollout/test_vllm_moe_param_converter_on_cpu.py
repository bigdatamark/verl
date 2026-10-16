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
# CPU tests for unfuse_moe_params, which converts fused/packed MoE expert weights (one tensor holding all experts)
# into the per-expert parameter layout vLLM expects, depending on model type.


import torch

from verl.workers.engine.fsdp.utils import unfuse_moe_params


def _collect(weights, model_type):
    # [FUNCTION DOC]
    # HELPER: Materializes the unfuse_moe_params generator into a list of (name, tensor) pairs.
    # Inputs: weights - iterable of (name, tensor); model_type - HF model type string.
    return [item for item in unfuse_moe_params(weights, model_type)]


def test_qwen_moe_packed_weights_are_expanded_per_expert():
    # [FUNCTION DOC]
    # For qwen3_moe, gate_up_proj of shape (experts=2, 2*inter=6, hidden=8) is split into per-expert gate/up weights of
    # shape (3, 8) emitted expert by expert, and down_proj (2, 8, 3) into per-expert down weights (8, 3) emitted after all
    # gate/up weights; names follow "experts.<i>.<proj>.weight".
    gate_up = torch.randn(2, 6, 8)
    down = torch.randn(2, 8, 3)

    converted = _collect(
        [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ],
        "qwen3_moe",
    )

    assert [name for name, _ in converted] == [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
    ]
    assert [tensor.shape for _, tensor in converted] == [
        (3, 8),
        (3, 8),
        (3, 8),
        (3, 8),
        (8, 3),
        (8, 3),
    ]


def test_gpt_oss_packed_weights_are_not_expanded():
    # [FUNCTION DOC]
    # For gpt_oss, packed tensors pass through unchanged: same names, same tensor objects and shapes.
    gate_up = torch.randn(2, 8, 6)
    down = torch.randn(2, 3, 8)
    weights = [
        ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
        ("model.layers.0.mlp.experts.down_proj", down),
    ]

    converted = _collect(weights, "gpt_oss")

    assert [name for name, _ in converted] == [name for name, _ in weights]
    assert converted[0][1] is gate_up
    assert converted[1][1] is down
    assert converted[0][1].shape == (2, 8, 6)
    assert converted[1][1].shape == (2, 3, 8)
