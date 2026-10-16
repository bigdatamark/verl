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
# Helpers for running actor forward passes with the PrefixGrouper library, which packs
# responses that share the same prompt (same uid) behind a single copy of the prefix so the
# shared prompt is computed once. Provides position-id construction, grouping a micro batch
# into a PrefixGrouper, the grouped forward, and a full micro-batch forward that returns
# per-token log-probs/entropy aligned to the response layout.

from __future__ import annotations

import torch
from prefix_grouper import PrefixGrouper

from verl.utils.torch_functional import logprobs_from_logits


def build_position_ids_for_prefix_grouper(prefix_grouper: PrefixGrouper) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: for each packed row (one group), positions 0..prefix_len-1 for the prefix, then
    # each non-empty suffix gets positions starting again at prefix_len (prefix_len ..
    # prefix_len+suffix_len-1), laid out consecutively. Remaining (padding) positions are 0.
    # Inputs: prefix_grouper - object exposing group_info (prefix_len, suffix_lens) and
    # padding_mask [groups, max_len]. Outputs: long tensor [groups, max_len].
    """Build position_ids for PrefixGrouper where each response restarts from prefix_len."""
    num_samples = len(prefix_grouper.group_info)
    max_len = prefix_grouper.padding_mask.size(1)
    device = prefix_grouper.padding_mask.device

    position_ids = torch.zeros(num_samples, max_len, dtype=torch.long, device=device)

    for i, group in enumerate(prefix_grouper.group_info):
        prefix_len = group.prefix_len

        position_ids[i, :prefix_len] = torch.arange(prefix_len, device=device)
        cur_pos = prefix_len
        for suffix_len in group.suffix_lens:
            if suffix_len > 0:
                position_ids[i, cur_pos : cur_pos + suffix_len] = torch.arange(
                    prefix_len, prefix_len + suffix_len, device=device
                )
                cur_pos += suffix_len

    return position_ids


def build_pg_from_micro_batch(
    micro_batch: dict,
    pad_token_id: int,
    padding_mode: str = "right",
):
    # [FUNCTION DOC]
    # Behavior: derives group sizes by run-length counting consecutive equal uids (assumes
    # same-uid samples are contiguous), takes the first sample's prompt of each group as the
    # prefix, prefix mask = tokens not equal to pad_token_id, builds the PrefixGrouper from the
    # prefix mask, response mask and group sizes, concatenates inputs, and builds the attention
    # mask (grouper padding mask) and position ids.
    # Inputs: micro_batch - mapping with prompts, responses, response_mask, uid; pad_token_id;
    # padding_mode - padding layout passed to PrefixGrouper (default right).
    # Outputs: tuple (prefix_grouper, concat_input_ids, attention_mask, position_ids, responses,
    # response_mask).
    # KNOWN ISSUE: non-contiguous samples sharing a uid are treated as separate groups; prefix
    # mask treats any pad_token_id occurrence as padding even inside the prompt.
    """Build PrefixGrouper from micro_batch dict containing prompts, responses, response_mask, uid."""
    prompts = micro_batch["prompts"]
    responses = micro_batch["responses"]
    response_mask = micro_batch["response_mask"]
    uids = micro_batch["uid"]

    bs = responses.size(0)

    group_sizes = []
    cur = 1
    for i in range(1, bs):
        if uids[i] == uids[i - 1]:
            cur += 1
        else:
            group_sizes.append(cur)
            cur = 1
    group_sizes.append(cur)

    prefix_indices = []
    cursor = 0
    for gs in group_sizes:
        prefix_indices.append(cursor)
        cursor += gs
    prefix_indices = torch.tensor(prefix_indices, device=prompts.device)

    prefix_ids = prompts.index_select(0, prefix_indices)
    prefix_mask = prefix_ids.ne(pad_token_id)

    prefix_grouper = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=response_mask,
        group_sizes=group_sizes,
        padding_mode=padding_mode,
        device=prompts.device,
    )

    concat_input_ids = prefix_grouper.concat_input(prefix_ids, prefix_mask, responses, response_mask)

    attention_mask = prefix_grouper.padding_mask

    position_ids = build_position_ids_for_prefix_grouper(prefix_grouper)

    return (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        responses,
        response_mask,
    )


def pg_forward(
    model,
    prefix_grouper,
    concat_input_ids,
    attention_mask,
    position_ids,
    completion_ids,
    completion_mask,
    *,
    temperature=1.0,
    padding_mode="right",
    include_prefix_last=1,
    calculate_entropy=False,
    entropy_fn=None,
):
    # [FUNCTION DOC]
    # Behavior: runs the model on the packed inputs (no KV cache, passing prefix_grouper), splits
    # logits into prefix/suffix parts (including the last prefix logit), converts completion ids
    # to the grouper's padding layout, shifts so logits at t predict token t+1 (drop last logit,
    # drop first mask entry), casts to float and divides by temperature in place, and computes
    # token log-probs; entropy optionally via entropy_fn.
    # Inputs: model; prefix_grouper; packed input ids, attention mask, position ids; completion
    # ids/mask; temperature; padding_mode; include_prefix_last; calculate_entropy; entropy_fn.
    # Outputs: (log_probs, entropy or None, suffix_mask).
    logits = model(
        input_ids=concat_input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        prefix_grouper=prefix_grouper,
    ).logits

    prefix_out, prefix_mask, suffix_out_raw, suffix_mask_raw = prefix_grouper.split_output(
        logits, include_prefix_last=include_prefix_last
    )

    completion_ids_right = prefix_grouper.convert_padding(
        completion_ids,
        completion_mask,
        padding_mode=padding_mode,
    )

    suffix_out = suffix_out_raw[:, :-1].float()
    suffix_mask = suffix_mask_raw[:, 1:]

    suffix_out /= temperature

    log_probs = logprobs_from_logits(suffix_out, completion_ids_right)

    entropy = None
    if calculate_entropy and entropy_fn is not None:
        entropy = entropy_fn(suffix_out)

    return log_probs, entropy, suffix_mask


def forward_micro_batch_with_prefix_grouper(
    micro_batch: dict,
    model,
    temperature: float,
    calculate_entropy: bool,
    device_name: str,
    param_dtype,
    use_chunking_entropy: bool = False,
):
    # [FUNCTION DOC]
    # Behavior: selects the entropy function (chunked or plain) if requested, reads pad_token_id
    # from the micro batch (default 0), builds the grouper, runs pg_forward under autocast,
    # zeroes log-probs/entropy at padded suffix positions, and right-pads them with zeros to the
    # response length when shorter.
    # Inputs: see docstring. Outputs: (entropy or None, log_probs), shapes [bs, response_len].
    """
    Forward pass using PrefixGrouper for shared-prefix optimization.

    Args:
        micro_batch: Dict containing prompts, responses, response_mask, uid, etc.
        model: The actor module.
        temperature: Temperature for logits scaling.
        calculate_entropy: Whether to compute entropy.
        device_name: Device name for autocast.
        param_dtype: Parameter dtype for autocast.
        use_chunking_entropy: Whether to use chunking entropy function.

    Returns:
        tuple: (entropy, log_probs) where entropy may be None if not calculated.
    """
    import verl.utils.torch_functional as verl_F

    entropy_fn = None
    if calculate_entropy:
        if use_chunking_entropy:
            entropy_fn = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_fn = verl_F.entropy_from_logits

    pad_token_id = micro_batch.get("pad_token_id", 0)

    (
        prefix_grouper,
        concat_input_ids,
        attention_mask,
        position_ids,
        responses,
        response_mask,
    ) = build_pg_from_micro_batch(
        micro_batch,
        pad_token_id=pad_token_id,
        padding_mode="right",
    )

    with torch.autocast(device_type=device_name, dtype=param_dtype):
        log_probs, entropy, suffix_mask_from_pg = pg_forward(
            model=model,
            prefix_grouper=prefix_grouper,
            concat_input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            completion_ids=responses,
            completion_mask=response_mask,
            temperature=temperature,
            padding_mode="right",
            include_prefix_last=1,
            calculate_entropy=calculate_entropy,
            entropy_fn=entropy_fn,
        )

    # Zero out padding positions
    padding_mask = suffix_mask_from_pg == 0
    log_probs = log_probs.masked_fill(padding_mask, 0.0)
    if entropy is not None:
        entropy = entropy.masked_fill(padding_mask, 0.0)

    # Pad to target response length if needed
    target_response_length = responses.size(1)
    if log_probs.size(1) != target_response_length:
        batch_size = log_probs.size(0)
        current_len = log_probs.size(1)

        full_log_probs = log_probs.new_zeros(batch_size, target_response_length)
        full_log_probs[:, :current_len] = log_probs
        log_probs = full_log_probs

        if entropy is not None:
            full_entropy = entropy.new_zeros(batch_size, target_response_length)
            full_entropy[:, :current_len] = entropy
            entropy = full_entropy

    return entropy, log_probs
