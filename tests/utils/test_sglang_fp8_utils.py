# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# Tests for the SGLang FP8 quantization helpers: building the FP8 quant config
# (defaults, merging ignored layers from an HF config's quantization_config, and
# the SGLANG_FP8_IGNORED_LAYERS env var) and SGLangFP8QuantizerHelper's
# per-parameter should-quantize decision (substring and "re:" regex matching).
# All tests clear or set SGLANG_FP8_IGNORED_LAYERS via pytest's monkeypatch.
from types import SimpleNamespace

from verl.utils.sglang.sglang_fp8_utils import SGLangFP8QuantizerHelper, build_sglang_fp8_quant_config


class MappingLikeConfig:
    # [CLASS DOC]
    # HELPER: Minimal mapping-like config object exposing only a dict-style get,
    # used to check the config builder works with objects that are neither dicts
    # nor attribute namespaces.
    def __init__(self, values):
        # [FUNCTION DOC]
        # HELPER: Stores the given dict as the backing values.
        # Inputs: values - dict of key to value. Outputs: none. Side effects: sets self.values.
        self.values = values

    def get(self, key, default=None):
        # [FUNCTION DOC]
        # HELPER: Dict-style lookup delegating to the backing dict.
        # Inputs: key; default - returned when key absent. Outputs: the value or default.
        return self.values.get(key, default)


def test_build_sglang_fp8_quant_config_preserves_defaults(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: with the env var unset and no HF config, the built config must equal
    # exactly the default dict: activation_scheme dynamic, fmt e4m3, quant_method fp8,
    # weight_block_size [128, 128] (no ignored_layers key).
    # Inputs: monkeypatch fixture. Outputs: none; asserts. Side effects: unsets env var.
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)

    quant_config = build_sglang_fp8_quant_config()

    assert quant_config == {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }


def test_sglang_fp8_quant_config_merges_hf_ignored_layers(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: an HF config (attribute namespace) whose quantization_config dict has
    # both ignored_layers and modules_to_not_convert yields ignored_layers equal to the
    # concatenation (ignored_layers first); the helper then refuses to quantize weights
    # under those modules and still quantizes an unrelated layer's weight.
    # Inputs: monkeypatch. Outputs: none; asserts.
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["model.layers.0.self_attn.q_proj"],
            "modules_to_not_convert": ["model.layers.1.mlp.down_proj"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
    ]
    assert not helper.should_quantize_param("model.layers.0.self_attn.q_proj.weight")
    assert not helper.should_quantize_param("model.layers.1.mlp.down_proj.weight")
    assert helper.should_quantize_param("model.layers.2.mlp.down_proj.weight")


def test_sglang_fp8_quant_config_accepts_mapping_like_config(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: same merge works when the HF config and its quantization_config are
    # mapping-like objects only exposing get; ignored_layers is carried over and the
    # matching parameter is not quantized.
    # Inputs: monkeypatch. Outputs: none; asserts.
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = MappingLikeConfig(
        {
            "quantization_config": MappingLikeConfig(
                {
                    "ignored_layers": ["model.layers.0.linear_attn"],
                }
            )
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["model.layers.0.linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")


def test_sglang_fp8_quantizer_matches_regex_ignored_layers(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: an ignored layer entry prefixed with "re:" is treated as a regex;
    # parameters whose names match are excluded and non-matching ones (an MoE expert
    # up_proj) are quantized. The entry is kept verbatim in the config.
    # Inputs: monkeypatch. Outputs: none; asserts.
    monkeypatch.delenv("SGLANG_FP8_IGNORED_LAYERS", raising=False)
    hf_config = SimpleNamespace(
        quantization_config={
            "ignored_layers": ["re:.*linear_attn.*"],
        }
    )

    quant_config = build_sglang_fp8_quant_config(hf_config)
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["re:.*linear_attn.*"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")


def test_sglang_fp8_quantizer_reads_sglang_env_ignored_layers(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: with SGLANG_FP8_IGNORED_LAYERS set to "linear_attn" and no HF config,
    # ignored_layers becomes that single entry and parameters containing it are skipped
    # while other parameters are quantized.
    # Inputs: monkeypatch. Outputs: none; asserts. Side effects: sets env var.
    monkeypatch.setenv("SGLANG_FP8_IGNORED_LAYERS", "linear_attn")

    quant_config = build_sglang_fp8_quant_config()
    helper = SGLangFP8QuantizerHelper(quant_config)

    assert quant_config["ignored_layers"] == ["linear_attn"]
    assert not helper.should_quantize_param("model.layers.0.linear_attn.in_proj_ba.weight")
    assert not helper.should_quantize_param("model.layers.0.linear_attn.g_proj.weight")
    assert helper.should_quantize_param("model.layers.0.mlp.experts.0.up_proj.weight")
