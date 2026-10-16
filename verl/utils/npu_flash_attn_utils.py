# Copyright 2024 Bytedance Ltd. and/or its affiliates

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
# Unpad/pad utilities for variable-length (packed) attention on NPU, copied from flash-attention's
# bert_padding: custom autograd functions for gathering/scattering along the first axis and helpers
# to convert between padded (batch, seqlen, ...) and packed (total_nnz, ...) layouts.
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexFirstAxis(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function that gathers rows of a tensor along dim 0 by integer indices (faster than
    # advanced indexing); backward scatters gradients back into a zero tensor of the original shape.
    # Exposed as index_first_axis.
    @staticmethod
    def forward(ctx, input, indices):
        # [FUNCTION DOC]
        # Behavior: saves indices; flattens all trailing dims, gathers rows at indices, reshapes back to
        # (len(indices), *original trailing shape). Stores original first-dim size on ctx.
        # Inputs: ctx - autograd context; input - tensor with ndim >= 2; indices - 1-D long tensor.
        # Outputs: gathered tensor. Exceptions: AssertionError if input.ndim < 2. Side effects: mutates ctx.
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: creates zeros of shape (first_axis_dim, flattened trailing), scatters grad rows into
        # positions given by saved indices, reshapes to original shape.
        # Inputs: ctx; grad_output - gradient with ndim >= 2.
        # Outputs: (grad_input, None) - no gradient for indices.
        # Exceptions: AssertionError if grad_output.ndim < 2. Side effects: none.
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexPutFirstAxis(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function that scatters rows into a new zero tensor of first-dim size first_axis_dim at
    # given indices; backward gathers gradient rows at those indices. Exposed as index_put_first_axis.
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        # [FUNCTION DOC]
        # Behavior: saves indices; allocates zeros (first_axis_dim, *values.shape[1:]) on values' device/dtype
        # and assigns values into rows at indices.
        # Inputs: ctx; values - (n, ...) with ndim >= 2; indices - 1-D, length n; first_axis_dim - int.
        # Outputs: the filled tensor.
        # Exceptions: AssertionError if indices not 1-D or values.ndim < 2. Side effects: mutates ctx.
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: selects grad_output rows at saved indices.
        # Inputs: ctx; grad_output. Outputs: (grad_values, None, None). Exceptions: none. Side effects: none.
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def pad_input(hidden_states, indices, batch, seqlen):
    # [FUNCTION DOC]
    # Behavior: scatters packed hidden states into a flat (batch*seqlen, ...) zero tensor at indices and
    # reshapes to (batch, seqlen, ...). Differentiable via IndexPutFirstAxis.
    # Inputs/Outputs: see docstring. Exceptions: assertion errors from IndexPutFirstAxis. Side effects: none.
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
def unpad_input(hidden_states, attention_mask, unused_mask=None):
    # [FUNCTION DOC]
    # Behavior: combines attention_mask with optional unused_mask (sum), computes per-row token counts
    # (int32) from the combined mask and separately from attention_mask alone, flat nonzero indices of the
    # combined mask, max sequence length (as Python int, forces device sync), and cu_seqlens as cumulative
    # counts left-padded with 0. Gathers selected tokens from the flattened hidden states.
    # Inputs: see docstring.
    # Outputs: tuple (packed hidden_states, indices, cu_seqlens, max_seqlen_in_batch, used_seqlens_in_batch);
    # the last counts attention_mask only (excludes unused_mask).
    # Exceptions: none explicit. Side effects: GPU/NPU sync via item().
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
