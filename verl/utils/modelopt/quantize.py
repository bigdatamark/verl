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
# Builds ModelOpt (mtq) quantization configs for NVFP4 weight-only (W4A16) QAT and applies them
# to Megatron models. Holds a module-level quantizer config: weight quantizers use FP4 E2M1
# (num_bits (2,1)) with dynamic block size 16 along the last axis and FP8 E4M3 scales (scale_bits
# (4,3)); input quantizers are disabled. Ignore patterns given in HF naming are translated to
# Megatron naming before being disabled.
"""ModelOpt NVFP4 quantization config and application for Megatron QAT."""

import copy

import modelopt.torch.quantization as mtq
import torch.nn as nn
from modelopt.torch.quantization.config import _default_disabled_quantizer_cfg

_NVFP4_W4A16_QUANTIZER_CFG = {
    "*weight_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
        "enable": True,
    },
    "*input_quantizer": {"enable": False},
}


def _ignore_patterns_to_quant_cfg(ignore_patterns: list[str]) -> list[dict]:
    # [FUNCTION DOC]
    # HELPER: Converts user ignore patterns into ModelOpt quantizer-disable entries.
    # Behavior: for each pattern, translate known HF-style names to Megatron-style wildcards
    # ("lm_head" -> "*output_layer*", "*mlp.gate" -> "*router*", "*self_attn*" -> "*self_attention*");
    # other patterns pass through unchanged. Each yields an entry with quantizer_name=<pattern>, enable=False.
    # Inputs: ignore_patterns - list of glob-style module name patterns.
    # Outputs: list of dicts, in input order.
    # Exceptions: none. Side effects: none.
    cfg = []
    mapping = {
        "lm_head": "*output_layer*",
        "*mlp.gate": "*router*",
        "*self_attn*": "*self_attention*",
    }
    for pattern in ignore_patterns:
        key = pattern
        if key in mapping:
            key = mapping[key]
        cfg.append({"quantizer_name": key, "enable": False})
    return cfg


def build_quantize_config(
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> dict:
    # [FUNCTION DOC]
    # Behavior: validates mode, normalizes the NVFP4 W4A16 quantizer config into list form, appends a
    # deep copy of ModelOpt's default disabled-quantizer config (normalized if it is a dict), then the
    # ignore-pattern disable entries (later entries take precedence in ModelOpt).
    # Inputs: qat_mode - must be "w4a16"; ignore_patterns - optional list (None treated as empty).
    # Outputs: dict with "quant_cfg" (list of entries) and "algorithm" = "max" (max calibration).
    # Exceptions: ValueError when qat_mode is not "w4a16".
    # Side effects: none (defaults are deep-copied).
    """Build a complete ModelOpt quantization config for ``mtq.quantize``."""
    if qat_mode != "w4a16":
        raise ValueError(f"Only 'w4a16' is supported, got: {qat_mode}")

    if ignore_patterns is None:
        ignore_patterns = []

    ignore_cfg = _ignore_patterns_to_quant_cfg(ignore_patterns)

    quant_cfg = mtq.normalize_quant_cfg_list(_NVFP4_W4A16_QUANTIZER_CFG)
    disabled_cfg = copy.deepcopy(_default_disabled_quantizer_cfg)
    if isinstance(disabled_cfg, dict):
        disabled_cfg = mtq.normalize_quant_cfg_list(disabled_cfg)
    quant_cfg.extend(disabled_cfg)
    quant_cfg.extend(ignore_cfg)
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def apply_qat(
    model: nn.Module,
    qat_mode: str,
    ignore_patterns: list[str] | None = None,
) -> nn.Module:
    # [FUNCTION DOC]
    # Behavior: builds the config via build_quantize_config and calls mtq.quantize on the model with
    # no calibration forward loop, inserting fake-quant quantizer modules.
    # Inputs: model - Megatron nn.Module; qat_mode, ignore_patterns - as for build_quantize_config.
    # Outputs: the same model object (quantized in place).
    # Exceptions: ValueError from build_quantize_config; any ModelOpt errors propagate.
    # Side effects: mutates model in place.
    """Apply Quantization-Aware Training to a Megatron model."""
    config = build_quantize_config(qat_mode, ignore_patterns)
    mtq.quantize(model, config)
    return model
