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
# Sequence layout utilities for Megatron-core models: converting padded (bshd) or jagged nested-tensor
# batches into packed THD buffers (with PackedSeqParams) or trimmed/padded BSHD batches, sharding the
# sequence dimension across context-parallel (CP) ranks (zigzag: each rank gets one chunk from the front
# and its mirror chunk from the back of each sequence; contiguous: each rank gets one slice of the
# global packed buffer), aligning lengths for TP sequence parallel, CP and FP8, and the inverse
# postprocessing (all-gather across CP and restore original per-sample layout). Also NPU (Ascend /
# MindSpeed) specific mask/position helpers and VLM attention-mask builders.
# Module state: logger; ContextParallelLayout literal type; a flag computed at import telling whether
# PackedSeqParams accepts cp_partition_mode (explicit parameter or **kwargs).

import inspect
import logging
import math
import os
from dataclasses import fields, is_dataclass
from typing import Literal

import torch
from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

from verl.utils.device import is_npu_available
from verl.utils.model import CausalLMOutputForPPO

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ContextParallelLayout = Literal["zigzag", "contiguous"]

# Older Megatron-core releases have no ``cp_partition_mode`` field on PackedSeqParams; they
# support only the zigzag CP layout. Inspect ``__init__`` rather than dataclass fields so that
# duck-typed replacements (e.g. test stubs taking ``**kwargs``) also count as supporting it.
_PACKED_SEQ_PARAMS_INIT_PARAMS = inspect.signature(PackedSeqParams.__init__).parameters
_PACKED_SEQ_PARAMS_HAS_CP_PARTITION_MODE = "cp_partition_mode" in _PACKED_SEQ_PARAMS_INIT_PARAMS or any(
    p.kind is inspect.Parameter.VAR_KEYWORD for p in _PACKED_SEQ_PARAMS_INIT_PARAMS.values()
)


def _packed_seq_params_supports(field_name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Reports whether PackedSeqParams declares a dataclass field of the given name.
    # Inputs: field_name. Outputs: bool. Side effects: none.
    if is_dataclass(PackedSeqParams):
        return field_name in {field.name for field in fields(PackedSeqParams)}
    return field_name in getattr(PackedSeqParams, "__dataclass_fields__", {})


def _compute_fp8_thd_align_size(align_size: int) -> tuple[int, int]:
    # [FUNCTION DOC]
    # HELPER: Computes FP8 THD alignment: per-sequence alignment lcm(16, align_size) and total-length
    # alignment align_size*128.
    # Inputs: align_size - base alignment. Outputs: tuple (per_seq_align, total_align). Side effects: none.
    """Compute FP8 alignment sizes for thd-format sequences.

    For FP8 block quantization, each sequence must be padded to a multiple of
    lcm(16, align_size), and the total padded length must be divisible by
    (align_size * 128) for TransformerEngine compatibility.

    Returns (per_seq_align_size, total_align_size).
    """
    return math.lcm(16, align_size), align_size * 128


def preprocess_packed_seqs(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pre_process: bool = True, use_fp8_padding: bool = False
) -> tuple[torch.Tensor, PackedSeqParams]:
    # [FUNCTION DOC]
    # Behavior: packs a right/left padded batch into a THD buffer. Valid lengths come from the mask sum;
    # each is padded up to align_size (tp*cp*2 when cp>1 else tp; replaced by FP8 per-seq alignment when
    # use_fp8_padding, with the last sequence additionally padded so the total is a multiple of
    # total_align). Builds padded cumulative lengths. When pre_process, creates a zero buffer of total
    # padded length / cp and copies masked tokens of each sample: without CP at its padded offset; with
    # CP, the rank's zigzag pair (front chunk cp_rank and back mirror chunk) of half-length
    # padded_len/(2*cp), clipped to the valid length.
    # Inputs: input_ids - [batch, seq, ...]; attention_mask - bool [batch, seq]; pre_process - whether to
    # build packed ids (first pp stage); use_fp8_padding - apply FP8 alignment.
    # Outputs: tuple (packed ids with leading dim 1 when pre_process else original input_ids,
    # PackedSeqParams in thd format using padded cu_seqlens for q/kv and padded variants, max padded len).
    # Exceptions: none explicit.
    # Side effects: one device-to-host sync for lengths; reads mpu state.
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if use_fp8_padding:
        per_seq_align, total_align = _compute_fp8_thd_align_size(align_size)
        align_size = per_seq_align

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    if use_fp8_padding:
        pad_size_last = (total_align - cu_seqlens_padded[-1] % total_align) % total_align
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            first_start = half_seqlen * cp_rank
            first_end = min(half_seqlen * (cp_rank + 1), d.shape[0])
            first_len = max(first_end - first_start, 0)
            if first_len > 0:
                input_ids_rmpad[start_idx : start_idx + first_len] = d[first_start:first_end]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = max(remain_end - remain_start, 0)
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: inverse of preprocess_packed_seqs. If post_process is false returns output unchanged.
    # Otherwise allocates zeros [batch, seq_len, ...]; with CP all-gathers each rank's packed output
    # (keeping the local tensor to preserve autograd), then for each sample reassembles the zigzag chunks
    # (rank j's front half to position j, back half to mirrored position) and scatters the first valid
    # length into the masked positions; without CP slices directly from padded offsets.
    # Inputs: output - [1, packed_len, ...]; packed_seq_params - from preprocess; attention_mask - original
    # bool mask; batch_size; seq_len - original padded length; post_process - last pp stage flag.
    # Outputs: tensor [batch, seq_len, ...].
    # Side effects: CP all_gather collective; host syncs.
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    shape = [batch_size, seq_len] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output, dtype=output.dtype) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=mpu.get_context_parallel_group())
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx : start_idx + s]
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device, dtype=output.dtype)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


def preprocess_bshd(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
):
    # [FUNCTION DOC]
    # Behavior: removes left padding by compacting each row's valid tokens to the front; target length is
    # the max valid length, rounded up to a multiple of TP size when sequence_parallel. Mask and
    # position ids are always compacted; input ids only when pre_process.
    # Inputs: input_ids [b, s, ...]; attention_mask [b, s] bool; position_ids [b, s]; sequence_parallel;
    # pre_process.
    # Outputs: (new or original input_ids, new_attention_mask, new_position_ids).
    # Exceptions: AssertionError if masks are not 2-D or CP size is not 1.
    # Side effects: host sync via item().
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    if sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
        seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len)
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def postprocess_bshd(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    # [FUNCTION DOC]
    # Behavior: inverse of preprocess_bshd: scatters each row's values at positions valid in the
    # compacted mask back to the positions valid in the original mask, into a zero tensor of length
    # origin_seqlen. Returns result unchanged when post_process is false.
    # Inputs: result [b, s', ...]; attention_mask - compacted mask; original_attention_mask; origin_seqlen;
    # post_process.
    # Outputs: tensor [b, origin_seqlen, ...]. Side effects: none.
    """
    Recover left padding from result
    return result
    """
    if not post_process:
        return result
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result


def postprocess_packed_seqs_for_dict_output(
    labels_mask: torch.Tensor,
    output: CausalLMOutputForPPO,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # Behavior: for fused-kernel outputs, flattens entropy and log_probs to [1, -1], zeroes log_probs where
    # labels_mask is false, and unpacks both via postprocess_packed_seqs.
    # Inputs: labels_mask - bool mask broadcastable to [1, packed]; output - object with entropy and
    # log_probs; packed_seq_params; attention_mask; batch_size; seq_len; post_process.
    # Outputs: dict with keys entropy and log_probs.
    # Side effects: mutates output.entropy and output.log_probs in place (reassigns attributes).
    # KNOWN ISSUE: docstring says it returns CausalLMOutputForPPO but it returns a plain dict; labels_mask
    # is undocumented.
    """_summary_
    For fused kernels, the output is a dictionary with keys like 'log_probs', 'entropy', etc.
    This function post-processes each tensor in the output dictionary.
    Args:
        output (CausalLMOutputForPPO): _description_
        packed_seq_params (PackedSeqParams): _description_
        attention_mask (torch.Tensor): _description_
        batch_size (int): _description_
        seq_len (int): _description_
        post_process (bool, optional): _description_. Defaults to True.
    Returns:
        CausalLMOutputForPPO: _description_
    """
    ret = {}
    output.entropy = output.entropy.view(1, -1)
    output.log_probs = output.log_probs.view(1, -1)
    output.log_probs = output.log_probs.masked_fill(~labels_mask, 0.0)
    ret["entropy"] = postprocess_packed_seqs(
        output.entropy, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
    )
    ret["log_probs"] = postprocess_packed_seqs(
        output.log_probs, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
    )
    return ret


def preprocess_for_mindspeed(input_ids, cu_seqlens_padded, seqlens_in_batch_padded, batch_size):
    # [FUNCTION DOC]
    # HELPER: On NPU only, passes padded cu_seqlens to MindSpeed's set_actual_seq_len and sets packed
    # position ids (0..len-1 restarting within each padded segment) via set_position_ids as a
    # [total, 1] tensor. No-op when NPU unavailable; logs a warning if mindspeed cannot be imported.
    # Inputs: input_ids (for device); cu_seqlens_padded; seqlens_in_batch_padded; batch_size.
    # Outputs: None. Side effects: sets MindSpeed global state; per-sample host syncs.
    # KNOWN ISSUE: called by preprocess_thd_engine before the FP8 / min_local_rows / length-bucket padding
    # of the last sequence, so MindSpeed sees shorter cu_seqlens than the final PackedSeqParams.
    if not is_npu_available:
        return
    try:
        from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len
        from mindspeed.utils import set_position_ids

        set_actual_seq_len(cu_seqlens_padded)
        # Generate position IDs within each padded segment
        pack_length = int(seqlens_in_batch_padded.sum().item())
        position_ids_packed = torch.zeros(pack_length, dtype=torch.int32, device=input_ids.device)
        for i in range(batch_size):
            start = cu_seqlens_padded[i].item()
            end = cu_seqlens_padded[i + 1].item()
            position_ids_packed[start:end] = torch.arange(end - start, dtype=torch.int32, device=input_ids.device)

        set_position_ids(position_ids_packed.unsqueeze(0).transpose(0, 1).contiguous())
    except ImportError as e:
        logger.warning(f"Could not import mindspeed modules, skipping position_id setting: {e}")


### No padding versions for model engine
### inputs are nested tensors
def preprocess_thd_engine(
    input_ids: torch.Tensor,
    pre_process: bool = True,
    need_roll: bool = False,
    use_fp8_padding: bool = False,
    local_cp_size: int | None = None,
    min_local_rows: int | None = None,
    pad_to_length_bucket: int | None = None,
    cp_layout: ContextParallelLayout = "zigzag",
) -> tuple[torch.Tensor, PackedSeqParams, torch.Tensor | None]:
    # [FUNCTION DOC]
    # Behavior: packs a jagged nested tensor of sequences into a THD buffer for the model engine and
    # shards it across CP. Steps: validate cp_layout; determine CP size/rank/group (dynamic CP groups when
    # local_cp_size is given, recorded in extra PackedSeqParams args); align_size = tp*cp*(2 for zigzag
    # with cp>1 else 1); pad each length to align_size (FP8 per-seq alignment optional); compute padded
    # cu_seqlens; call preprocess_for_mindspeed; then optionally extend the last sequence so that the
    # total is a multiple of FP8 total_align, at least cp*min_local_rows, and a multiple of
    # lcm(pad_to_length_bucket, cp[, fp8 total]). When pre_process, builds local ids buffer of
    # total/cp rows plus position ids:
    # - contiguous: rank owns global rows [rank*local, (rank+1)*local); copies the overlap of each valid
    # sequence; with need_roll, takes token at (index+1) modulo sequence length and uses that index as the
    # position id.
    # - no CP: copy each sequence at its padded offset with positions 0..len-1.
    # - zigzag CP: zero-pad the sequence to its padded length, copy front chunk cp_rank and mirrored back
    # chunk; positions follow original indices (back chunk positions clamped to valid length). With
    # need_roll, records replacement values for the last row of each chunk (next token across the chunk
    # boundary, or wrap to first token), then the whole buffer is rolled left by one and replacements
    # applied (also applies when cp<=1 in zigzag mode, with no replacements).
    # Finally sets cp_partition_mode when supported and builds PackedSeqParams.
    # Inputs: input_ids - jagged nested tensor [batch, ragged, ...]; pre_process; need_roll - produce
    # next-token-shifted ids (labels); use_fp8_padding; local_cp_size - dynamic CP size; min_local_rows -
    # minimum rows per rank; pad_to_length_bucket - total length bucket; cp_layout - zigzag or contiguous.
    # Outputs: (packed ids [1, rows, ...], PackedSeqParams, position ids [1, rows]) when pre_process, else
    # (input_ids, PackedSeqParams, None).
    # Exceptions: ValueError for unknown cp_layout, non-positive pad_to_length_bucket, or contiguous layout
    # with cp>1 on a Megatron without cp_partition_mode.
    # Side effects: host syncs, MindSpeed state on NPU, one-time warning when padding for CP.
    # KNOWN ISSUE: with need_roll and no CP in zigzag mode, the global left roll makes each sequence's last
    # row take the next sequence's first token (no fix-up), relying on loss masks to ignore it.
    """Pack nested THD sequences and shard their rows across CP ranks.

    ``zigzag`` is the default causal-attention layout: each rank receives a
    chunk from both ends of every sequence. ``contiguous`` assigns each rank
    one consecutive interval of the *global padded THD buffer*. The latter is
    required by attention variants whose CP kernels address rows through one
    rank-local ``global_start``.
    """
    if cp_layout not in ("zigzag", "contiguous"):
        raise ValueError(f"Unsupported context parallel layout: {cp_layout}")

    batch_size = input_ids.shape[0]

    tp_size = mpu.get_tensor_model_parallel_world_size()
    extra_packed_args = {}
    if local_cp_size is not None:
        # dynamic CP
        cp_size = local_cp_size
        cp_group = mpu.get_dynamic_data_context_parallel_groups(group_size=local_cp_size)
        cp_rank = torch.distributed.get_rank(group=cp_group)
        extra_packed_args["local_cp_size"] = local_cp_size
        extra_packed_args["cp_group"] = cp_group
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
    if _packed_seq_params_supports("cp_partition_mode"):
        extra_packed_args["cp_partition_mode"] = cp_layout
    cp_layout_factor = 2 if cp_layout == "zigzag" and cp_size > 1 else 1
    align_size = tp_size * cp_size * cp_layout_factor
    seqlens_in_batch = input_ids.offsets().diff()

    if use_fp8_padding:
        per_seq_align, total_align = _compute_fp8_thd_align_size(align_size)
        align_size = per_seq_align

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    preprocess_for_mindspeed(input_ids, cu_seqlens_padded, seqlens_in_batch_padded, batch_size)

    if use_fp8_padding:
        # Pad the last sequence so total length is divisible by total_align for TE
        pad_size_last = (total_align - cu_seqlens_padded[-1] % total_align) % total_align
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    if min_local_rows is not None:
        min_total_rows = cp_size * min_local_rows
        if cu_seqlens_padded[-1] < min_total_rows:
            pad_size_last = min_total_rows - cu_seqlens_padded[-1]
            cu_seqlens_padded[-1] += pad_size_last
            seqlens_in_batch_padded[-1] += pad_size_last

    if pad_to_length_bucket is not None:
        if pad_to_length_bucket <= 0:
            raise ValueError("pad_to_length_bucket must be a positive integer")
        total_alignment = math.lcm(pad_to_length_bucket, cp_size)
        if use_fp8_padding:
            total_alignment = math.lcm(total_alignment, total_align)
        pad_size_last = (-cu_seqlens_padded[-1]) % total_alignment
        cu_seqlens_padded[-1] += pad_size_last
        seqlens_in_batch_padded[-1] += pad_size_last

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        position_ids_rmpad = torch.zeros(shape[0], dtype=torch.long, device=input_ids.device)
        if need_roll:
            saved_roll_dict = {}
            saved_position_roll_dict = {}
        local_global_start = shape[0] * cp_rank
        local_global_end = local_global_start + shape[0]
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_layout == "contiguous":
                seq_global_start = cu_seqlens_padded_cpu[i]
                seq_valid_end = seq_global_start + seqlens_in_batch_cpu[i]
                copy_global_start = max(local_global_start, seq_global_start)
                copy_global_end = min(local_global_end, seq_valid_end)
                if copy_global_start >= copy_global_end:
                    continue

                source_start = copy_global_start - seq_global_start
                source_end = copy_global_end - seq_global_start
                destination_start = copy_global_start - local_global_start
                destination_end = copy_global_end - local_global_start
                d = input_ids[i]
                if need_roll:
                    source_indices = torch.arange(source_start, source_end, device=d.device)
                    source_indices = (source_indices + 1) % seqlens_in_batch_cpu[i]
                    input_ids_rmpad[destination_start:destination_end] = torch.index_select(d, 0, source_indices)
                    position_ids_rmpad[destination_start:destination_end] = source_indices
                else:
                    input_ids_rmpad[destination_start:destination_end] = d[source_start:source_end]
                    position_ids_rmpad[destination_start:destination_end] = torch.arange(
                        source_start, source_end, dtype=torch.long, device=input_ids.device
                    )
                continue

            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i]
                # Build position_ids: 0, 1, 2, ..., seqlen-1 for this sequence
                position_ids_rmpad[start_idx : start_idx + seqlen] = torch.arange(
                    seqlen, dtype=torch.long, device=input_ids.device
                )
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen_orig_i = seqlens_in_batch_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i]
            # Pad to the full per-seq padded length so every CP rank gets a
            # full slice; align_size alone is too short for the CP split end.
            pad_target = max(align_size, seqlen_padded_i)
            if d.shape[0] < pad_target:
                original_size = d.shape[0]
                pad = torch.zeros((pad_target - d.shape[0], *d.shape[1:]), dtype=d.dtype, device=d.device)
                d = torch.cat([d, pad], dim=0)
                logger.warning_once(
                    f"Padding tensor for context parallel alignment, original_size={original_size}, "
                    f"align_size={align_size}"
                )

            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            # Build position_ids for the first chunk
            position_ids_rmpad[start_idx : start_idx + half_seqlen] = torch.arange(
                half_seqlen * cp_rank, half_seqlen * (cp_rank + 1), dtype=torch.long, device=input_ids.device
            )

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]
                # Build position_ids for the remaining chunk: use remain_start as base,
                # clamped to original seqlen to avoid exceeding seqlen-1 for padded positions
                pos_end = min(remain_end, seqlen_orig_i)
                valid_pos_len = pos_end - remain_start
                if valid_pos_len > 0:
                    position_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + valid_pos_len] = (
                        torch.arange(remain_start, pos_end, dtype=torch.long, device=input_ids.device)
                    )

            if need_roll:
                # Handle roll for cp_size > 1 case
                saved_roll_dict[start_idx + half_seqlen - 1] = d[(cp_rank + 1) * half_seqlen]
                saved_position_roll_dict[start_idx + half_seqlen - 1] = position_ids_rmpad[start_idx + half_seqlen - 1]
                if remain_len > 0:
                    if remain_end == d.shape[0]:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[0]
                        saved_position_roll_dict[start_idx + half_seqlen + remain_len - 1] = 0
                    else:
                        saved_roll_dict[start_idx + half_seqlen + remain_len - 1] = d[remain_end]
                        saved_position_roll_dict[start_idx + half_seqlen + remain_len - 1] = position_ids_rmpad[
                            start_idx + half_seqlen + remain_len - 1
                        ]

        if need_roll and cp_layout == "zigzag":
            input_ids_rmpad = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
            position_ids_rmpad = torch.roll(position_ids_rmpad, shifts=-1, dims=0)
            if len(saved_roll_dict) > 0:
                for k, v in saved_roll_dict.items():
                    input_ids_rmpad[k] = v
                for k, v in saved_position_roll_dict.items():
                    position_ids_rmpad[k] = v

    if _PACKED_SEQ_PARAMS_HAS_CP_PARTITION_MODE:
        # Tell the attention kernels how the rows above were sharded across CP ranks. The rows
        # are already split according to `cp_layout`, but PackedSeqParams defaults to "zigzag",
        # so without this an attention variant that requires a contiguous split (DeepSeek-V4)
        # raises "DSv4 THD CP requires a contiguous CP partition." on every forward.
        extra_packed_args["cp_partition_mode"] = cp_layout
    elif cp_layout != "zigzag" and cp_size > 1:
        raise ValueError(
            f"cp_layout='{cp_layout}' requires PackedSeqParams.cp_partition_mode, which this "
            "Megatron-core version does not provide. Upgrade Megatron-core or use the zigzag layout."
        )

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        **extra_packed_args,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params, position_ids_rmpad.unsqueeze(0)
    else:
        return input_ids, packed_seq_params, None


def postprocess_thd_engine(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    input_ids: torch.Tensor,
    batch_size: int,
    post_process: bool = True,
    local_cp_size: int | None = None,
    cp_layout: ContextParallelLayout = "zigzag",
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: inverse of preprocess_thd_engine. Validates cp_layout; returns output if not post_process.
    # Determines CP group (dynamic from packed_seq_params.cp_group when local_cp_size set), all-gathers
    # outputs across CP (local tensor kept for autograd). Contiguous: concatenates rank outputs in rank
    # order and slices each sequence's valid length at its padded offset. Zigzag/no-CP: reassembles like
    # postprocess_packed_seqs, keeping the first valid length. Original lengths come from
    # input_ids.offsets().
    # Inputs: output [1, rows, ...]; packed_seq_params; input_ids - original nested tensor; batch_size;
    # post_process; local_cp_size; cp_layout.
    # Outputs: jagged nested tensor of per-sequence outputs.
    # Exceptions: ValueError for unknown cp_layout.
    # Side effects: CP all_gather; host syncs.
    """
    Postprocess packed sequences
    """
    if cp_layout not in ("zigzag", "contiguous"):
        raise ValueError(f"Unsupported context parallel layout: {cp_layout}")
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    # The reason why we use input_ids.offsets() instead of packed_seq_params.cu_seqlens_q.diff()
    # is that the latter one is the padded length, while the former one is the original length.
    cu_seqlens = input_ids.offsets()
    seq_lens_cpu: list[int] = cu_seqlens.diff().tolist()

    output_new = []

    if local_cp_size is not None:
        cp_size = local_cp_size
        cp_group = packed_seq_params.cp_group
        cp_rank = torch.distributed.get_rank(group=cp_group)
    else:
        cp_size = mpu.get_context_parallel_world_size()
        cp_group = mpu.get_context_parallel_group()
        cp_rank = mpu.get_context_parallel_rank()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=cp_group)
        output_list[cp_rank] = output
    else:
        output_list = [output]

    if cp_layout == "contiguous":
        packed_output = torch.cat([rank_output[0] for rank_output in output_list], dim=0)
        for i in range(batch_size):
            sequence_start = cu_padded_cpu[i]
            output_new.append(packed_output[sequence_start : sequence_start + seq_lens_cpu[i]])
        return torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new.append(output[0][start_idx : start_idx + s])
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device, dtype=output.dtype)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx : packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen : packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen : (j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen : s_len_padded - j * half_seqlen] = o1
        output_new.append(tmp[:s_len])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor


def _build_npu_attn_mask(original_attention_mask: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Builds an NPU fusion-attention mask [B, 1, S, S] where true means masked: combines pairwise
    # validity of the [B, S] mask with a lower-triangular causal mask and inverts it.
    # Inputs: original_attention_mask bool [B, S]. Outputs: bool tensor. Side effects: none.
    """Build attn_mask for torch_npu.npu_fusion_attention (B1SS / [B, 1, Sq, Skv])"""
    _, seq_len = original_attention_mask.shape
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=original_attention_mask.device, dtype=torch.bool))
    attn_mask = original_attention_mask.unsqueeze(-1) & original_attention_mask.unsqueeze(-2)
    attn_mask = attn_mask & causal_mask
    return (~attn_mask).unsqueeze(1).contiguous()


def preprocess_bshd_engine(
    input_ids: torch.Tensor,
    pre_process: bool = True,
    need_roll: bool = False,
    use_fp8_padding: bool = False,
    forced_max_seqlen: int | None = None,
):
    # [FUNCTION DOC]
    # Behavior: pads a jagged nested tensor to a dense BSHD batch for the engine. max length is
    # forced_max_seqlen or the batch max, rounded up to tp*cp*2 (cp>1) or tp; with FP8, further rounded so
    # that batch*len is divisible by 128*tp*cp and remains aligned. Without CP, copies each sequence to the
    # front and marks its mask. With CP, zero-pads each sequence to max length and keeps the zigzag pair of
    # chunks of length max/(2*cp) (front chunk cp_rank, back chunk 2*cp-rank-1); need_roll replaces tokens
    # with those at the next global position (mod max length); mask marks valid parts of each chunk.
    # Position ids are 0..local-1 without CP or the zigzag global positions with CP, broadcast over batch.
    # Without CP, need_roll rolls ids left by one along seq. On NPU the mask is converted to [B,1,S,S]
    # masked-form via _build_npu_attn_mask.
    # Inputs: input_ids nested [b, ragged, ...dense]; pre_process (unused); need_roll; use_fp8_padding;
    # forced_max_seqlen - raw unaligned max length override.
    # Outputs: (ids [b, local_len, ...], attention_mask [b, local_len] bool or NPU 4-D mask, position_ids).
    # Side effects: host sync.
    # KNOWN ISSUE: pre_process is accepted but never used. Without CP, need_roll's roll lets the last
    # padded row wrap to the first token of the row.
    """
    Preprocess bshd sequences
    return "input_ids, attention_mask, position_ids"

    The input is a jagged nested tensor with shape [batch, seq, ...]. Any
    dense dimensions after seq are preserved in the returned padded tensor.

    When ``forced_max_seqlen`` is given, it overrides the per-micro-batch
    ``seqlens_in_batch.max()`` as the raw padding target. Callers that want to
    align tensor shapes across micro-batches (e.g. to share a cuDNN fused-attention
    execution plan) pass the *mini-batch* global max here; any TP/CP/FP8 alignment
    is still applied below, so this argument should be the unaligned raw max.
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()

    batch_size = input_ids.shape[0]
    dense_shape = tuple(input_ids.shape[2:])
    seqlens_in_batch = input_ids.offsets().diff()
    max_seqlen = forced_max_seqlen if forced_max_seqlen is not None else seqlens_in_batch.max().item()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    # For CP (zigzag), sequence length must be divisible by (2 * cp_size).
    # After zigzag-CP split each rank holds s/cp_size tokens, which must also be
    # divisible by tp_size for sequence-parallel scatter.  Therefore the total
    # sequence length must be divisible by tp_size * cp_size * 2.
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if align_size > 1:
        pad_size = (align_size - max_seqlen % align_size) % align_size
        max_seqlen += pad_size
    if use_fp8_padding:
        # For FP8 block quantization, batch_size * max_seqlen / tp_size must be divisible by 128.
        # With CP, local sequence length is max_seqlen / cp_size.
        # We need:
        # 1) max_seqlen aligned for SP/CP splitting.
        # 2) batch_size * max_seqlen % (128 * tp_size * cp_size) == 0.
        # Compute the required alignment for max_seqlen:
        fp8_total_align = 128 * tp_size * cp_size
        fp8_seq_align = fp8_total_align // math.gcd(batch_size, fp8_total_align)
        # Also ensure SP and CP split alignment.
        fp8_seq_align = math.lcm(fp8_seq_align, align_size)
        max_seqlen = ((max_seqlen + fp8_seq_align - 1) // fp8_seq_align) * fp8_seq_align

    local_max_seqlen = max_seqlen // cp_size if cp_size > 1 else max_seqlen
    attention_mask = torch.zeros(batch_size, local_max_seqlen, dtype=torch.bool, device=input_ids.device)
    input_ids_bshd = torch.zeros(
        (batch_size, local_max_seqlen, *dense_shape), dtype=input_ids.dtype, device=input_ids.device
    )
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()
    for i in range(batch_size):
        seqlen_i = int(seqlens_in_batch_cpu[i])
        if cp_size <= 1:
            attention_mask[i, :seqlen_i] = True
            input_ids_bshd[i, :seqlen_i] = input_ids[i]
            continue

        seq = input_ids[i]
        if seqlen_i < max_seqlen:
            seq_padded = torch.zeros((max_seqlen, *dense_shape), dtype=seq.dtype, device=seq.device)
            seq_padded[:seqlen_i] = seq
            seq = seq_padded

        chunk_len = max_seqlen // (2 * cp_size)
        first_start = cp_rank * chunk_len
        second_start = (2 * cp_size - cp_rank - 1) * chunk_len
        first_chunk = seq[first_start : first_start + chunk_len]
        second_chunk = seq[second_start : second_start + chunk_len]
        local_seq = torch.cat((first_chunk, second_chunk), dim=0)
        if need_roll:
            local_pos = torch.cat(
                (
                    torch.arange(first_start, first_start + chunk_len, dtype=torch.long, device=seq.device),
                    torch.arange(second_start, second_start + chunk_len, dtype=torch.long, device=seq.device),
                ),
                dim=0,
            )
            local_seq = seq[(local_pos + 1) % max_seqlen]
        input_ids_bshd[i] = local_seq

        valid_first = max(0, min(seqlen_i - first_start, chunk_len))
        valid_second = max(0, min(seqlen_i - second_start, chunk_len))
        if valid_first > 0:
            attention_mask[i, :valid_first] = True
        if valid_second > 0:
            attention_mask[i, chunk_len : chunk_len + valid_second] = True

    if cp_size <= 1:
        position_ids = torch.arange(local_max_seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(attention_mask)
    else:
        chunk_len = max_seqlen // (2 * cp_size)
        first_pos = torch.arange(
            cp_rank * chunk_len, (cp_rank + 1) * chunk_len, dtype=torch.long, device=input_ids.device
        )
        second_pos = torch.arange(
            max_seqlen - (cp_rank + 1) * chunk_len,
            max_seqlen - cp_rank * chunk_len,
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = torch.cat((first_pos, second_pos), dim=0).unsqueeze(0).expand_as(attention_mask)
    if need_roll and cp_size <= 1:
        input_ids_bshd = torch.roll(input_ids_bshd, shifts=-1, dims=1)

    if is_npu_available:
        # Ascend npu_fusion_attention's attn_mask must be BNSS / B1SS / 11SS / SS; [B, S] is invalid.
        attention_mask = _build_npu_attn_mask(attention_mask)

    return input_ids_bshd, attention_mask, position_ids


def postprocess_bshd_engine(
    output: torch.Tensor,
    attention_mask: torch.Tensor,
    post_process: bool = True,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: inverse of preprocess_bshd_engine. Returns output if not post_process. On NPU, recovers a
    # [B, S] validity mask from the diagonal of the 4-D masked-form mask. Asserts output's first two dims
    # match the mask. With CP, all-gathers outputs and masks, rebuilds full-length rows by placing each
    # rank's front half at slot j and back half at the mirror slot, then selects valid positions; without
    # CP selects masked positions per row.
    # Inputs: output [b, local_len, ...]; attention_mask; post_process.
    # Outputs: jagged nested tensor of per-sample valid outputs.
    # Exceptions: AssertionError on shape mismatch or odd local length under CP.
    # Side effects: CP all_gather collectives.
    """
    Postprocess bshd sequences
    """
    if not post_process:
        return output

    if is_npu_available:
        attention_mask = attention_mask.diagonal(dim1=-2, dim2=-1).squeeze(1)
        attention_mask = ~attention_mask.bool()

    assert output.shape[:2] == attention_mask.shape, (
        f"output.shape: {output.shape}, attention_mask.shape: {attention_mask.shape}"
    )

    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    batch_size = output.shape[0]

    if cp_size > 1:
        output_list = [torch.empty_like(output, dtype=output.dtype) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, output.detach(), group=cp_group)
        output_list[cp_rank] = output

        mask_list = [torch.empty_like(attention_mask, dtype=attention_mask.dtype) for _ in range(cp_size)]
        torch.distributed.all_gather(mask_list, attention_mask, group=cp_group)
    else:
        output_list = [output]
        mask_list = [attention_mask]

    output_new = []

    for i in range(batch_size):
        if cp_size <= 1:
            mask = attention_mask[i].bool()
            output_new.append(output[i][mask])
            continue

        local_seqlen = output.shape[1]
        assert local_seqlen % 2 == 0, "CP bshd expects local sequence length to be divisible by 2"
        half_seqlen = local_seqlen // 2
        full_seqlen = local_seqlen * cp_size

        tmp = torch.empty(full_seqlen, *output.shape[2:], device=output.device, dtype=output.dtype)
        full_mask = torch.zeros(full_seqlen, device=attention_mask.device, dtype=torch.bool)

        for j in range(cp_size):
            o = output_list[j][i]
            m = mask_list[j][i].bool()

            o0, o1 = o[:half_seqlen], o[half_seqlen:]
            m0, m1 = m[:half_seqlen], m[half_seqlen:]

            front_start = j * half_seqlen
            front_end = (j + 1) * half_seqlen
            back_start = full_seqlen - (j + 1) * half_seqlen
            back_end = full_seqlen - j * half_seqlen

            tmp[front_start:front_end] = o0
            tmp[back_start:back_end] = o1
            full_mask[front_start:front_end] = m0
            full_mask[back_start:back_end] = m1

        output_new.append(tmp[full_mask])

    output_new_tensor = torch.nested.as_nested_tensor(output_new, layout=torch.jagged)

    return output_new_tensor


def build_vlm_attn_mask_thd(input_ids: torch.Tensor, pad_token_id: int = None):
    # [FUNCTION DOC]
    # Behavior: pads a nested input_ids tensor to (batch, max length aligned to tp*cp*2 or tp) using
    # pad_token_id and builds a bool mask of valid prefix per row.
    # Inputs: input_ids nested tensor; pad_token_id.
    # Outputs: (padded ids, bool attention mask). Side effects: host sync.
    seqlens_in_batch = input_ids.offsets().diff()

    # Align to TP/CP so ``combined_embeddings`` is divisible by tp_size before
    # the SP scatter. Mirrors ``build_vlm_attn_mask_bshd``.
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    max_seqlen = int(seqlens_in_batch.max().item())
    if align_size > 1:
        max_seqlen += (align_size - max_seqlen % align_size) % align_size

    batch_size = input_ids.shape[0]
    input_ids_with_pad = input_ids.to_padded_tensor(pad_token_id, output_size=(batch_size, max_seqlen))
    attention_mask = torch.zeros_like(input_ids_with_pad, dtype=torch.bool)
    for i, seqlen in enumerate(seqlens_in_batch):
        attention_mask[i, :seqlen] = True

    return input_ids_with_pad, attention_mask


def build_vlm_attn_mask_bshd(
    input_ids: torch.Tensor, batch_size: int, pad_token_id: int = None, forced_max_seqlen: int | None = None
):
    # [FUNCTION DOC]
    # Behavior: same as build_vlm_attn_mask_thd but the raw max may be overridden by forced_max_seqlen and
    # batch_size is given explicitly; on NPU returns no mask.
    # Inputs: input_ids nested; batch_size; pad_token_id; forced_max_seqlen.
    # Outputs: (padded ids [batch, len], bool mask or None on NPU). Side effects: host sync.
    seqlens_in_batch = input_ids.offsets().diff()
    # When ``forced_max_seqlen`` is given, pad to the mini-batch global max (raw, unaligned)
    # so the VLM padded tensors share the same `s_q` as the label/loss_mask produced by
    # preprocess_bshd_engine; otherwise logits.shape[:2] != label.shape[:2]. TP/CP alignment
    # is still applied below.
    max_seqlen = forced_max_seqlen if forced_max_seqlen is not None else seqlens_in_batch.max().item()

    # For CP (zigzag), sequence length must be divisible by (2 * cp_size).
    # After zigzag-CP split each rank holds s/cp_size tokens, which must also be
    # divisible by tp_size for sequence-parallel scatter.  Therefore the total
    # sequence length must be divisible by tp_size * cp_size * 2.
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if align_size > 1:
        pad_size = (align_size - max_seqlen % align_size) % align_size
        max_seqlen += pad_size

    input_ids_bshd = input_ids.to_padded_tensor(pad_token_id, output_size=(batch_size, max_seqlen))

    if is_npu_available:
        return input_ids_bshd, None

    attention_mask = torch.zeros_like(input_ids_bshd, dtype=torch.bool)
    for i, seqlen in enumerate(seqlens_in_batch):
        attention_mask[i, :seqlen] = True

    return input_ids_bshd, attention_mask
