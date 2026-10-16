# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Conversions between batch tensor layouts used by workers: left-right padded batches
# (prompt left-padded, response right-padded, shape batch x seq_len), "no-padding" batches made of
# jagged nested tensors (one variable-length row per sample), and padded response-only tensors.
# Model outputs computed on the no-padding layout are shifted left by one token (token t's logit
# predicts token t+1), which the response-extraction helpers account for.
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.attention_utils import index_first_axis, unpad_input


def left_right_2_no_padding(data: TensorDict) -> TensorDict:
    # [FUNCTION DOC]
    # Behavior: asserts the four required keys exist. Pops input_ids; records max_seq_len (input
    # width) and max_response_len (response_mask width) as non-tensor data. Removes padding from
    # input_ids using attention_mask (unpad_input), storing the flat-token gather indices as
    # non-tensor "indices", and rebuilds input_ids as a jagged nested tensor using the cumulative
    # sequence lengths as offsets. For each sample, selects position ids at valid mask positions
    # (supports 1-D per-sample positions or multi-row e.g. 4 x seq_len rope positions) and packs them
    # into a jagged nested tensor. Sets loss_mask to the same object as response_mask. If a
    # non-nested routed_experts tensor exists, flattens batch/seq dims, gathers by indices and nests
    # it with the same offsets. If both teacher_logprobs and teacher_ids exist, does the same for each.
    # Inputs: data - TensorDict with input_ids, attention_mask, response_mask, position_ids
    # (batch x seq_len, or batch x 4 x seq_len), optional routed_experts / teacher_logprobs / teacher_ids.
    # Outputs: the same TensorDict, mutated in place, with nested input_ids/position_ids/loss_mask.
    # Exceptions: AssertionError if a required key is missing.
    # Side effects: mutates and returns the input TensorDict.
    # KNOWN ISSUE: docstring says attention_mask (and "response") are removed, but the code leaves
    # attention_mask (and any other keys) in the TensorDict; only input_ids is replaced.
    """
    Convert TensorDict from left-right padding to no-padding format.

    Args:
        data: TensorDict with "input_ids", "attention_mask", "response_mask", "position_ids"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors like "input_ids", "loss_mask", "position_ids"
        - NonTensorData includes "max_seq_len", "max_response_len", "indices"

    Note:
    1. the return input_ids/position_ids/loss_mask are nested tensor.
    2. we will remove "attention_mask", "response" in the return data, but "response_mask" is kept.
    """
    assert "input_ids" in data, "input_ids is required in left-right padding data"
    assert "attention_mask" in data, "attention_mask is required in left-right padding data"
    assert "response_mask" in data, "response_mask is required in left-right padding data"
    assert "position_ids" in data, "position_ids is required in left-right padding data"

    input_ids = data.pop("input_ids")
    attention_mask = data["attention_mask"]
    response_mask = data["response_mask"]
    position_ids = data["position_ids"]  # (bs, seq_len) or # (bs, 4, seq_len)

    max_seq_len, max_response_len = input_ids.shape[1], response_mask.shape[1]
    tu.assign_non_tensor_data(data, "max_seq_len", max_seq_len)
    tu.assign_non_tensor_data(data, "max_response_len", max_response_len)

    input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
    tu.assign_non_tensor_data(data, "indices", indices)

    input_ids_nested = torch.nested.nested_tensor_from_jagged(input_ids_rmpad.squeeze(-1), offsets=cu_seqlens)

    position_ids_list = []
    for i in range(attention_mask.shape[0]):
        curr_mask = attention_mask[i].bool()
        curr_pos_ids = position_ids[i]
        if curr_pos_ids.dim() == 1:  # (seq_len,)
            valid_ids = curr_pos_ids[curr_mask]
        else:  # (4, seq_len)
            valid_ids = curr_pos_ids[:, curr_mask]
        position_ids_list.append(valid_ids)
    position_ids_nested = torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)

    data["input_ids"] = input_ids_nested
    data["position_ids"] = position_ids_nested
    data["loss_mask"] = data["response_mask"]

    routed_experts = data.get("routed_experts", None)
    if routed_experts is not None and not routed_experts.is_nested:
        routed_experts_rmpad = index_first_axis(routed_experts.unsqueeze(-1).flatten(0, 1), indices)
        routed_experts_nested = torch.nested.nested_tensor_from_jagged(
            routed_experts_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        data["routed_experts"] = routed_experts_nested

    # (bsz, seqlen, topk)
    teacher_logprobs = data.get("teacher_logprobs", None)
    teacher_ids = data.get("teacher_ids", None)
    if teacher_logprobs is not None and teacher_ids is not None:
        teacher_logprobs_rmpad = index_first_axis(teacher_logprobs.unsqueeze(-1).flatten(0, 1), indices)
        teacher_ids_rmpad = index_first_axis(teacher_ids.unsqueeze(-1).flatten(0, 1), indices)
        teacher_logprobs_nested = torch.nested.nested_tensor_from_jagged(
            teacher_logprobs_rmpad.squeeze(-1), offsets=cu_seqlens
        )
        teacher_ids_nested = torch.nested.nested_tensor_from_jagged(teacher_ids_rmpad.squeeze(-1), offsets=cu_seqlens)
        data["teacher_logprobs"] = teacher_logprobs_nested
        data["teacher_ids"] = teacher_ids_nested

    return data


def no_padding_2_padding(tensor: torch.Tensor, data: TensorDict) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: takes the flat token values (nested values, or the tensor itself). Computes per-sample
    # prompt and response lengths: if prompts are nested, from nested offsets (max_response_len taken
    # from non-tensor data, or the longest response if unset); otherwise from attention_mask sums over
    # the prompt and response column ranges, with max_response_len set to the responses width. Sums
    # lengths to cumulative sequence end offsets, asserts the total matches the token count and that no
    # prompt length is zero. For each sample, takes the slice ending one token before the sequence end
    # and of response length (the one-token left shift), right-pads it on the sequence dim to
    # max_response_len (trailing dims unpadded), and stacks the results.
    # Inputs: tensor - nested or flat (total_tokens, ...) model output; data - TensorDict with prompts,
    # responses, and attention_mask when not nested; optional non-tensor max_response_len.
    # Outputs: dense tensor of shape batch x max_response_len x trailing dims, zero padded.
    # Exceptions: AssertionError on token count mismatch, zero-length prompt, or nested attention_mask.
    # Side effects: none; .item() calls cause device sync.
    """Slice response from unpad model output.

    Args:
        tensor: a nested tensor or a tensor of shape (total_nnz,*),
            total_nnz is the total number of tokens across all sequences in the batch

        data: TensorDict with "prompts", "responses", "attention_mask"

    Returns:
        tensor: sliced response tensor of shape [bsz, max_response_len, *]
    """
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    if prompt_ids.is_nested:
        prompt_lens = prompt_ids.offsets().diff()
        response_lens = response_ids.offsets().diff()
        if max_response_len < 0:
            max_response_len = response_lens.max().item()
    else:
        attention_mask = data["attention_mask"]
        assert not attention_mask.is_nested
        prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
        max_response_len = response_ids.shape[1]

    sequence_lens = prompt_lens + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)
    assert sequence_offsets[-1].item() == values.shape[0]
    assert not prompt_lens.eq(0).any(), f"seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"

    response_list = []
    # Skip padding dimensions after sequence dimensions, if any.
    skip_padding = (0, 0) * (values.ndim - 1)
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        # left-shift model output by one token for log_probs/values
        response_list.append(F.pad(values[seq_offset - resp_len - 1 : seq_offset - 1], (*skip_padding, 0, pad_size)))

    output = torch.stack(response_list, dim=0)
    return output


def build_attention_mask_from_nested(input_ids: torch.Tensor, max_seq_len: int | None = None) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: from nested input ids, compute each sequence length from offsets, use max_seq_len or
    # the longest length, and produce a mask where position j of row i is 1 iff j is less than length i.
    # Inputs: input_ids - nested jagged tensor; max_seq_len - optional output width.
    # Outputs: int32 tensor batch x max_seq_len on the values' device (right-padded mask).
    # Exceptions: AssertionError if input_ids is not nested.
    # Side effects: none.
    """Build a padded full-sequence attention mask from nested input ids."""
    assert input_ids.is_nested, "input_ids must be a nested tensor"
    device = input_ids.values().device
    seq_lens = input_ids.offsets().diff().to(device=device)
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
    return (positions < seq_lens.unsqueeze(1)).to(torch.int32)


def embeds_padding_2_no_padding(data: TensorDict) -> TensorDict:
    # [FUNCTION DOC]
    # Behavior: converts prompt_embeds with its mask to nested form via the inner helper, and does the
    # same for negative_prompt_embeds/negative_prompt_embeds_mask when negative_prompt_embeds is a tensor.
    # Selection uses the boolean mask, so it works for any mask though the docstring expects a
    # left-contiguous valid region.
    # Inputs: data - TensorDict with prompt_embeds (batch x seq x dim) and prompt_embeds_mask, optional
    # negative counterparts.
    # Outputs: the same TensorDict, mutated, with nested embeds and nested all-true masks.
    # Exceptions: KeyError if the negative mask is missing while negative embeds exist.
    # Side effects: mutates the input TensorDict.
    """
    Convert TensorDict from prompt embeds with padding to no-padding format.

    Currently we expect the prompt embedding mask to be [1111000...] format,
    which means the valid tokens are continuous and start from the left.

    Args:
        data: TensorDict with "prompt_embeds", "prompt_embeds_mask",
              "negative_prompt_embeds", "negative_prompt_embeds_mask"

    Returns:
        data: TensorDict with
        - Tensor includes NestedTensors "prompt_embeds", "prompt_embeds_mask",
          "negative_prompt_embeds", "negative_prompt_embeds_mask"
    """

    def _to_nested(embeds: torch.Tensor, mask: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: for each row, select embeddings at positions where the mask is true and the
        # corresponding (all-true) mask values; return two jagged nested tensors (embeds, mask).
        # Inputs: embeds - batch x seq x dim; mask - batch x seq. Side effects: none.
        """Strip padding from (bs, seq_len, dim) embeds using the boolean mask and return nested tensors."""
        embeds_list, mask_list = [], []
        for i in range(mask.shape[0]):
            curr_mask = mask[i].bool()
            embeds_list.append(embeds[i, curr_mask, :])
            mask_list.append(curr_mask[curr_mask])
        return (
            torch.nested.as_nested_tensor(embeds_list, layout=torch.jagged),
            torch.nested.as_nested_tensor(mask_list, layout=torch.jagged),
        )

    data["prompt_embeds"], data["prompt_embeds_mask"] = _to_nested(data["prompt_embeds"], data["prompt_embeds_mask"])

    if isinstance(data.get("negative_prompt_embeds", None), torch.Tensor):
        data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"] = _to_nested(
            data["negative_prompt_embeds"], data["negative_prompt_embeds_mask"]
        )

    return data


def response_from_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: for each sample, using the nested tensor's offsets as sequence ends and response lengths
    # from response_mask offsets, take the response-length slice ending one token before the sequence
    # end (left-shift alignment) and pack slices into a jagged nested tensor.
    # Inputs: tensor - nested full-sequence model output; response_mask - nested response mask.
    # Outputs: nested tensor of per-sample response values.
    # Exceptions: ValueError from strict zip if batch sizes differ.
    # Side effects: none.
    """Extract response from nested model output.

    Args:
        tensor: a nested tensor with shape (bsz, prompt_len + response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    values, offsets = tensor.values(), tensor.offsets()
    response_lens = response_mask.offsets().diff()
    response_list = []
    for resp_len, seq_offset in zip(response_lens, offsets[1:], strict=True):
        # left-shift model output by one token for log_probs/values
        response_list.append(values[seq_offset - resp_len - 1 : seq_offset - 1])
    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)


def response_to_nested(tensor: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: asserts response_mask is nested; for each row of the padded tensor keeps the first
    # response-length entries and packs them into a jagged nested tensor.
    # Inputs: tensor - padded batch x response_len; response_mask - nested response mask.
    # Outputs: nested tensor batch x jagged response_len.
    # Exceptions: AssertionError if response_mask is not nested.
    # Side effects: none.
    """Convert padded response tensor to nested tensor.

    Args:
        tensor: a tensor with shape (bsz, response_len)
        response_mask: a nested tensor with shape (bsz, response_len)

    Returns:
        tensor: a nested tensor with shape (bsz, response_len)
    """
    assert response_mask.is_nested
    response_lens = response_mask.offsets().diff()
    response_list = []
    for i in range(tensor.shape[0]):
        response_list.append(tensor[i, : response_lens[i]])

    return torch.nested.as_nested_tensor(response_list, layout=torch.jagged)
