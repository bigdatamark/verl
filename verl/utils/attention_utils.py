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
# Device-agnostic padding/unpadding helpers for variable-length attention. Resolves implementations at call time: NPU
# helpers on Ascend, flash_attn.bert_padding on CUDA, or pure-torch fallbacks when flash-attn is not installed.
# Module globals cache the last resolved functions.

from typing import Callable

import torch
import torch.nn.functional as F

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _fallback_index_first_axis(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Selects rows of the tensor along the first axis by indices. Inputs: tensor (ndim >= 2), indices.
    # Outputs: indexed tensor. Exceptions: AssertionError if ndim < 2.
    """Pure-torch equivalent of `flash_attn.bert_padding.index_first_axis`."""
    assert tensor.ndim >= 2
    return tensor[indices]


def _fallback_pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Scatters unpadded rows back into a zero tensor of shape (batch * seqlen, ...) at indices, reshaped to
    # (batch, seqlen, ...). Inputs: hidden_states (total_tokens, ...), indices, batch, seqlen. Outputs: padded tensor.
    """Pure-torch equivalent of `flash_attn.bert_padding.pad_input`."""
    other_shape = hidden_states.shape[1:]
    output = hidden_states.new_zeros(batch * seqlen, *other_shape)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *other_shape)


def _fallback_unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask=None):
    # [FUNCTION DOC]
    # HELPER: Unpads a (batch, seqlen, ...) tensor. Effective mask is attention_mask plus optional unused_mask; computes
    # per-row lengths (int32), indices of nonzero flattened mask positions, cumulative lengths with a leading 0.
    # Outputs: (selected token rows, indices, cu_seqlens, max length in batch as Python int, per-row lengths from
    # attention_mask only).
    """Pure-torch equivalent of `flash_attn.bert_padding.unpad_input`."""
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        _fallback_index_first_axis(hidden_states.reshape(-1, *hidden_states.shape[2:]), indices),
        indices,
        cu_seqlens,
        seqlens_in_batch.max().item(),
        used_seqlens_in_batch,
    )


def _fallback_rearrange(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Lazily imports einops and forwards all arguments to einops.rearrange. Exceptions: ImportError without einops.
    """`einops.rearrange`, imported lazily so the other fallbacks stay einops-free."""
    from einops import rearrange as einops_rearrange

    return einops_rearrange(*args, **kwargs)


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    # [FUNCTION DOC]
    # HELPER: Imports NPU implementations if an NPU is available (no device check), else flash_attn.bert_padding ones,
    # falling back to the local pure-torch versions on ImportError; stores them in module globals.
    # Outputs: (index_first_axis, pad_input, rearrange, unpad_input). Side effects: sets globals.
    # KNOWN ISSUE: the globals are never consulted as a cache, so resolution (imports and device check) runs on every call.
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ImportError:
            # flash-attn only ships CUDA wheels, so CPU-only installs (unit tests, dev
            # boxes) have to use the unoptimized but equivalent torch implementations.
            index_first_axis = _fallback_index_first_axis
            pad_input = _fallback_pad_input
            rearrange = _fallback_rearrange
            unpad_input = _fallback_unpad_input

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: dispatches to the resolved index_first_axis with all arguments. Outputs: its result.
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).
      - Without flash-attn installed: `_fallback_index_first_axis`.

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: dispatches to the resolved pad_input with all arguments. Outputs: its result.
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).
      - Without flash-attn installed: `_fallback_pad_input`.

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: dispatches to the resolved rearrange with all arguments. Outputs: its result.
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).
      - Without flash-attn installed: `einops.rearrange`.

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: dispatches to the resolved unpad_input with all arguments. Outputs: its result.
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).
      - Without flash-attn installed: `_fallback_unpad_input`.

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
