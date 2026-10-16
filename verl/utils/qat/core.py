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
# Training-side QAT setup: config dataclass, loading of the compressed-tensors style quantization JSON,
# replacement of eligible nn.Linear modules with fake-quantizing QATLinear, setup of sibling links between
# projections that vLLM fuses (q/k/v, gate/up) so they can share a global scale, and invalidation of
# cached weight scales after optimizer steps.
"""QAT (Quantization-Aware Training) utilities for verl FSDP training."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn

from verl.base_config import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class QATConfig(BaseConfig):
    # [CLASS DOC]
    # QAT settings: enable flag, mode ("w4a16" or "w4a4"), quantization group size (16), ignore patterns
    # (substring match, or regex when prefixed "re:"; defaults skip lm_head, embed_tokens and MoE router
    # gates), activation observer name for W4A4, and path to the quantization config JSON.
    """Unified configuration for QAT (Quantization-Aware Training)."""

    enable: bool = False
    mode: str = "w4a16"
    group_size: int = 16
    ignore_patterns: list[str] = field(default_factory=lambda: ["lm_head", "embed_tokens", "re:.*mlp.gate$"])
    activation_observer: str = "static_minmax"
    quantization_config_path: Optional[str] = None


def load_quantization_config(qat_config: QATConfig) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Behavior: reads the JSON at quantization_config_path; if the config has ignore_patterns, replaces the
    # JSON "ignore" field with them (logging when they differ).
    # Inputs: qat_config. Outputs: parsed dict.
    # Exceptions: ValueError if the path is empty; OSError / JSON decode errors propagate.
    # Side effects: file read, logging.
    """Load quantization config JSON file from QATConfig."""
    if not qat_config.quantization_config_path:
        raise ValueError("quantization_config_path is required when QAT is enabled")

    logger.info(f"Loading QAT quantization config from: {qat_config.quantization_config_path}")

    with open(qat_config.quantization_config_path) as f:
        quant_config = json.load(f)

    if qat_config.ignore_patterns:
        original_ignore = quant_config.get("ignore", [])
        quant_config["ignore"] = qat_config.ignore_patterns
        if original_ignore != qat_config.ignore_patterns:
            logger.info(f"Overriding JSON 'ignore' field: {original_ignore} -> {qat_config.ignore_patterns}")

    logger.info("Successfully loaded QAT quantization config")
    return quant_config


def _should_quantize(name: str, module: nn.Module, config: QATConfig) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True only for nn.Linear modules whose qualified name matches no ignore pattern
    # ("re:" patterns via regex match from the start of the name, others via substring) and whose
    # in_features is divisible by group_size (warning logged otherwise). Side effects: debug/warning logs.
    """Check if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    for pattern in config.ignore_patterns:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.match(regex, name):
                logger.debug(f"Ignoring {name} due to regex pattern: {regex}")
                return False
        else:
            if pattern in name:
                logger.debug(f"Ignoring {name} due to pattern: {pattern}")
                return False

    if module.in_features % config.group_size != 0:
        logger.warning(
            f"Skipping {name}: in_features={module.in_features} not divisible by group_size={config.group_size}"
        )
        return False

    return True


def apply_qat(
    model: nn.Module,
    config: QATConfig | dict[str, Any],
) -> nn.Module:
    # [FUNCTION DOC]
    # Behavior: converts dict config to QATConfig; returns the model unchanged if disabled. Parses mode into
    # QATMode (lower-cased), collects eligible modules first, then replaces each non-QATLinear one in its
    # parent with QATLinear.from_linear (mode, group_size, activation_observer).
    # Inputs: model; config - QATConfig or dict of its fields. Outputs: the same model (mutated).
    # Exceptions: ValueError for invalid mode; TypeError for unknown dict keys.
    # Side effects: mutates module tree; logging.
    """Apply QAT to a model by replacing nn.Linear with QATLinear."""
    from verl.utils.qat.linear import QATLinear, QATMode

    if not isinstance(config, QATConfig):
        config = QATConfig(**config)

    if not config.enable:
        logger.info("QAT is disabled, returning original model")
        return model

    mode = QATMode(config.mode.lower())
    logger.info(f"Applying QAT with mode={mode.value}, group_size={config.group_size}")

    modules_to_replace = []
    for name, module in model.named_modules():
        if _should_quantize(name, module, config):
            modules_to_replace.append((name, module))

    logger.info(f"Found {len(modules_to_replace)} Linear layers to convert to QAT")

    converted_count = 0
    for name, module in modules_to_replace:
        if isinstance(module, QATLinear):
            continue

        fake_quant_module = QATLinear.from_linear(
            module,
            mode=mode,
            group_size=config.group_size,
            activation_observer=config.activation_observer,
        )

        _set_module(model, name, fake_quant_module)
        converted_count += 1

    logger.info(f"Successfully applied QAT to {converted_count} layers")

    return model


def _set_module(model: nn.Module, name: str, new_module: nn.Module):
    # [FUNCTION DOC]
    # HELPER: Walks the dotted name through getattr from model to the parent and sets the last component to
    # new_module. Exceptions: AttributeError for invalid paths. Side effects: mutates model.
    """Set a module in the model by its full name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


FUSION_PATTERNS = {
    "qkv": ["q_proj", "k_proj", "v_proj"],
    "gate_up": ["gate_proj", "up_proj"],
}


def setup_fusion_siblings(model: nn.Module):
    # [FUNCTION DOC]
    # Behavior: for each fusion pattern (qkv: q/k/v_proj; gate_up: gate/up_proj), groups QATLinear modules
    # whose names end with a pattern suffix by parent path; for groups with at least two members, sets on
    # each module a list of weak references to its siblings (_fusion_siblings_ref).
    # Outputs: dict pattern name -> number of groups linked. Side effects: sets attributes; logs.
    """Setup fusion siblings for QKV and GateUp layers."""
    import weakref

    from verl.utils.qat.linear import QATLinear

    qat_modules = {name: m for name, m in model.named_modules() if isinstance(m, QATLinear)}

    counts = {}
    for group_name, suffixes in FUSION_PATTERNS.items():
        groups: dict[str, dict[str, nn.Module]] = {}
        for name, module in qat_modules.items():
            for suffix in suffixes:
                if name.endswith(suffix):
                    parent = name.rsplit(".", 1)[0]
                    groups.setdefault(parent, {})[suffix] = module

        count = 0
        for parent, projs in groups.items():
            if len(projs) >= 2:
                modules = list(projs.values())
                for i, m in enumerate(modules):
                    siblings = modules[:i] + modules[i + 1 :]
                    m._fusion_siblings_ref = [weakref.ref(s) for s in siblings]
                count += 1
        counts[group_name] = count

    logger.info(f"[QAT Fuse] Setup fusion siblings: {counts}")
    return counts


def enable_qat_fuse(model: nn.Module):
    # [FUNCTION DOC]
    # Behavior: sets up fusion siblings and marks the model with _qat_fuse_enabled = True.
    # Outputs: None. Side effects: mutates modules/model; logs.
    """Enable QAT fuse mode: sets up fusion siblings for weight scale fusion."""
    setup_fusion_siblings(model)
    model._qat_fuse_enabled = True
    logger.info("[QAT Fuse] Enabled QAT fuse mode")


def invalidate_all_scales(model: nn.Module):
    # [FUNCTION DOC]
    # Behavior: for every QATLinear in the model, clears cached blockwise scale, global scale and cached weight
    # amax so they are recomputed on the next forward. Side effects: mutates modules; debug log.
    """Clear all cached weight scales after optimizer.step()."""
    from verl.utils.qat.linear import QATLinear

    count = 0
    for module in model.modules():
        if isinstance(module, QATLinear):
            module._weight_blockwise_scale = None
            module._weight_global_scale = None
            module._cached_weight_amax = None
            count += 1

    logger.debug(f"[QAT Fuse] Invalidated scales for {count} QATLinear layers")
