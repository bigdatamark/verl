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
# High-level QAT workflow helpers for the Megatron backend: applying fake quantization to model
# chunks and wrapping exported weights in a quantizing exporter.

"""High-level QAT workflow helpers for Megatron backend."""


def _get_qat_field(qat_config, key, default=None):
    # [FUNCTION DOC]
    # HELPER: Returns qat_config[key] (dict .get) or the attribute key, with default when missing.
    """Extract a field from qat_config, supporting both dict and object-style access."""
    if isinstance(qat_config, dict):
        return qat_config.get(key, default)
    return getattr(qat_config, key, default)


def apply_qat_to_modules(modules, qat_config):
    # [FUNCTION DOC]
    # Behavior: reads mode (default "w4a16") and ignore_patterns (converted to a list if set) from
    # qat_config and replaces each module chunk in place with apply_qat(chunk, mode, ignore_patterns).
    # Inputs: modules - mutable list of Megatron chunks; qat_config - dict or object.
    # Outputs: the same list. Side effects: mutates the list and quantizes modules.
    """Apply ModelOpt fake quantization to a list of Megatron module chunks."""
    from verl.utils.modelopt.quantize import apply_qat

    qat_mode = _get_qat_field(qat_config, "mode", "w4a16")
    ignore_patterns = _get_qat_field(qat_config, "ignore_patterns", None)
    if ignore_patterns is not None:
        ignore_patterns = list(ignore_patterns)

    for i in range(len(modules)):
        modules[i] = apply_qat(modules[i], qat_mode, ignore_patterns=ignore_patterns)
    return modules


def export_qat_weights(per_tensor_param, modules, qat_mode, bridge):
    # [FUNCTION DOC]
    # Behavior: builds a QATWeightExporter(modules, bridge, qat_mode) and returns its quantizing
    # iterator over per_tensor_param.
    # Inputs: per_tensor_param - iterator of (hf_name, tensor); modules; qat_mode; bridge - Megatron
    # Bridge (AutoBridge-like) object. Outputs: iterator of (name, tensor).
    # Side effects: exporter construction performs PP/EP collectives.
    """Process exported weights through QATWeightExporter for quantized weight sync."""
    from verl.utils.modelopt.qat_weight_exporter import QATWeightExporter

    qat_weight_exporter = QATWeightExporter(modules, bridge, qat_mode)
    return qat_weight_exporter.process_weights_iterator(per_tensor_param)
