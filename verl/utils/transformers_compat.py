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
# Version-compatibility shims for the HuggingFace transformers library: a fallback for
# flash_attn_supports_top_left_mask, a cached installed-version range check, a resolver for the
# vision-to-sequence auto model class, normalization of visual-encoder outputs, and removal of tied
# alias keys from state dicts before saving.

"""
Compatibility utilities for different versions of transformers library.
"""

import importlib.metadata
from functools import lru_cache
from typing import Optional

from packaging import version

# Handle version compatibility for flash_attn_supports_top_left_mask
# This function was added in newer versions of transformers
try:
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
except ImportError:
    # For older versions of transformers that don't have this function
    # Default to False as a safe fallback for older versions
    def flash_attn_supports_top_left_mask():
        # [FUNCTION DOC]
        # HELPER: Fallback defined only when transformers lacks flash_attn_supports_top_left_mask.
        # Behavior: always returns False. Inputs: none. Outputs: False. Exceptions/Side effects: none.
        """Fallback implementation for older transformers versions.
        Returns False to disable features that require this function.
        """
        return False


@lru_cache
def is_transformers_version_in_range(min_version: Optional[str] = None, max_version: Optional[str] = None) -> bool:
    # [FUNCTION DOC]
    # Behavior: reads the installed transformers version from package metadata and checks it lies in
    # the inclusive range [min_version, max_version]; a None bound is treated as unbounded. Result is
    # memoized per argument pair (lru_cache).
    # Inputs: min_version / max_version - optional version strings.
    # Outputs: True if both bounds are satisfied, else False.
    # Exceptions: raises ModuleNotFoundError if transformers is not installed; invalid version strings
    # propagate packaging's InvalidVersion.
    # Side effects: none (cached).
    try:
        # Get the installed version of the transformers library
        transformers_version_str = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as e:
        raise ModuleNotFoundError("The `transformers` package is not installed.") from e

    transformers_version = version.parse(transformers_version_str)

    lower_bound_check = True
    if min_version is not None:
        lower_bound_check = version.parse(min_version) <= transformers_version

    upper_bound_check = True
    if max_version is not None:
        upper_bound_check = transformers_version <= version.parse(max_version)

    return lower_bound_check and upper_bound_check


@lru_cache
def get_auto_model_for_vision2seq():
    # [FUNCTION DOC]
    # Behavior: tries to import AutoModelForImageTextToText (preferred, newer) and returns it; on
    # ImportError falls back to AutoModelForVision2Seq. Memoized.
    # Inputs: none. Outputs: the auto model class.
    # Exceptions: ImportError if neither class exists. Side effects: imports transformers.
    """Return the available VL auto model class across transformers versions."""

    try:
        # Prefer the newer class when available. In transformers 4.x this class has
        # a broader mapping than AutoModelForVision2Seq, and AutoModelForVision2Seq
        # is deprecated for removal in v5.
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq

    return AutoModelForImageTextToText


def unpack_visual_output(visual_output):
    # [FUNCTION DOC]
    # Behavior: normalizes visual encoder output into a pair. If the object has pooler_output, returns
    # (pooler_output, deepstack_features or None). Otherwise, a tuple is returned unchanged (assumed
    # already a pair); any other object is returned as (object, None).
    # Inputs: visual_output - encoder output object, tuple, or tensor.
    # Outputs: a 2-tuple (embeddings, deepstack features or None).
    # Exceptions/Side effects: none.
    """Unpack the output from the visual encoder, handling both tuple and object return types.

    Newer versions of transformers return an object with `pooler_output` and `deepstack_features`
    attributes instead of a plain tuple.
    """
    if hasattr(visual_output, "pooler_output"):
        # For newer versions(>=5.0.0) of transformers, return the pooler_output and deepstack_features
        if hasattr(visual_output, "deepstack_features"):
            return visual_output.pooler_output, visual_output.deepstack_features
        else:
            return visual_output.pooler_output, None
    if isinstance(visual_output, tuple):
        return visual_output
    else:
        return visual_output, None


def drop_tied_target_keys(state_dict, model, model_config) -> None:
    # [FUNCTION DOC]
    # Behavior: no-op unless model_config.tie_word_embeddings is truthy. Calls model.tie_weights(), then
    # iterates named parameters without de-duplication, keeping a mapping from parameter identity to its
    # first-seen (canonical) name. For each later alias of the same parameter: if the canonical name is in
    # state_dict, the alias key is removed from state_dict (if present); otherwise the alias becomes the
    # new canonical name.
    # Inputs: state_dict - mutable mapping name to tensor; model - nn.Module; model_config - HF config.
    # Outputs: None.
    # Exceptions: none expected.
    # Side effects: mutates state_dict in place; re-ties model weights.
    """Drop tied alias keys (e.g. ``lm_head.weight``) from ``state_dict``.

    FSDP gather and the model merger materialise one independent CPU tensor
    per state_dict key, which defeats ``save_pretrained``'s storage-pointer-
    based dedup. When both keys end up in safetensors, ``transformers>=5``
    silently refuses to re-tie on reload. Detects aliases by ``Parameter``
    identity after ``tie_weights()``.

    Only drops the alias when the canonical name has been confirmed to exist
    in ``state_dict``; otherwise the alias is promoted to canonical so we
    never accidentally remove every entry for a tied parameter.
    """
    if not getattr(model_config, "tie_word_embeddings", False):
        return
    model.tie_weights()
    canonical_by_id: dict[int, str] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        pid = id(param)
        if pid not in canonical_by_id:
            canonical_by_id[pid] = name
            continue
        if canonical_by_id[pid] in state_dict:
            state_dict.pop(name, None)
        else:
            # Canonical name absent: promote the alias instead of erasing it.
            canonical_by_id[pid] = name
