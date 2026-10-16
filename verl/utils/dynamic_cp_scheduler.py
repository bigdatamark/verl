# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Adapter between verl TensorDict batches (jagged NestedTensor input_ids) and Megatron-Core's DefaultDynamicCPScheduler, which groups samples into micro-batches and assigns each to a dynamically sized context-parallel (CP) sub-group within the DPxCP group. Defines non-tensor metadata keys DCP_GROUP_LEADER, DCP_LOCAL_NUM_TOKENS, DCP_PADDING_MASK, DCP_SAMPLE_IDS attached to each scheduled micro-batch, and a post-processing step that gathers outputs from group leaders and restores original sample order.

"""Thin TensorDict adapter for Megatron-Core's Dynamic CP scheduler."""

import math

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.py_functional import append_to_dict

DCP_GROUP_LEADER = "_dcp_group_leader"
DCP_LOCAL_NUM_TOKENS = "_dcp_local_num_tokens"
DCP_PADDING_MASK = "_dcp_padding_mask"
DCP_SAMPLE_IDS = "_dcp_sample_ids"


def get_megatron_dynamic_cp_scheduler_cls() -> type:
    # [FUNCTION DOC]
    # Returns Megatron-Core's DefaultDynamicCPScheduler class.
    # Exceptions: RuntimeError (chained) if the import fails because Megatron-Core lacks the feature.
    """Return the scheduler implementation owned by Megatron-Core."""
    try:
        from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Dynamic CP requires a Megatron-Core build containing NVIDIA/Megatron-LM#5154 (merge commit d2e7ec5b)."
        ) from exc
    return DefaultDynamicCPScheduler


def _local_padding_mask(
    seq_lens: list[int],
    *,
    cp_size: int,
    cp_rank: int,
    tp_size: int,
    cp_layout: str = "zigzag",
    min_local_rows: int | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Builds a boolean mask (True = padding) for this rank's local shard of a packed THD batch.
    # Behavior: each sequence is padded up to a multiple of tp_size*cp_size*(2 if zigzag and cp_size>1 else 1). If min_local_rows is given, the last sequence's padded length is increased so total padded rows reach at least cp_size*min_local_rows. Contiguous layout: the concatenated padded stream is split evenly across cp ranks; positions overlapping each sequence's valid tokens in this rank's slice are False. Zigzag layout: each padded sequence is split into 2*cp_size chunks; this rank owns front chunk cp_rank and back chunk (2*cp_size-1-cp_rank); valid tokens in each chunk are False; per-sequence masks are concatenated. With cp_size 1, first seq_len positions of each sequence are False.
    # Inputs: seq_lens - valid lengths; cp_size, cp_rank, tp_size; cp_layout - "zigzag" or "contiguous"; min_local_rows - optional positive minimum rows per rank.
    # Outputs: 1-D bool tensor of local padded length.
    # Exceptions: ValueError for unsupported layout or non-positive min_local_rows.
    """Build the MCore router padding mask for one local THD shard."""
    if cp_layout not in ("zigzag", "contiguous"):
        raise ValueError(f"Unsupported context parallel layout: {cp_layout}")

    cp_layout_factor = 2 if cp_layout == "zigzag" and cp_size > 1 else 1
    align_size = tp_size * cp_size * cp_layout_factor
    padded_lens = [math.ceil(seq_len / align_size) * align_size for seq_len in seq_lens]

    if min_local_rows is not None:
        if min_local_rows <= 0:
            raise ValueError(f"min_local_rows must be positive, got {min_local_rows}")
        min_total_rows = cp_size * min_local_rows
        padded_total_rows = sum(padded_lens)
        if padded_total_rows < min_total_rows:
            padded_lens[-1] += min_total_rows - padded_total_rows

    if cp_layout == "contiguous":
        local_padded_len = sum(padded_lens) // cp_size
        local_global_start = local_padded_len * cp_rank
        local_global_end = local_global_start + local_padded_len
        mask = torch.ones(local_padded_len, dtype=torch.bool)
        seq_global_start = 0
        for seq_len, padded_len in zip(seq_lens, padded_lens, strict=True):
            seq_valid_end = seq_global_start + seq_len
            copy_global_start = max(local_global_start, seq_global_start)
            copy_global_end = min(local_global_end, seq_valid_end)
            if copy_global_start < copy_global_end:
                destination_start = copy_global_start - local_global_start
                destination_end = copy_global_end - local_global_start
                mask[destination_start:destination_end] = False
            seq_global_start += padded_len
        return mask

    masks = []
    for seq_len, padded_len in zip(seq_lens, padded_lens, strict=True):
        local_padded_len = padded_len // cp_size
        mask = torch.ones(local_padded_len, dtype=torch.bool)
        if cp_size == 1:
            mask[:seq_len] = False
            masks.append(mask)
            continue

        half_chunk = padded_len // (2 * cp_size)
        front_start = cp_rank * half_chunk
        back_start = padded_len - (cp_rank + 1) * half_chunk
        front_tokens = max(0, min(seq_len - front_start, half_chunk))
        back_tokens = max(0, min(seq_len - back_start, half_chunk))
        mask[:front_tokens] = False
        mask[half_chunk : half_chunk + back_tokens] = False
        masks.append(mask)
    return torch.cat(masks)


def _cp_members(rank_assignments: list[list[int]], rank: int) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Returns the list of ranks whose sample-id assignment equals that of the given rank (its CP group).
    # Exceptions: RuntimeError if the rank's assignment is empty.
    rank_sample_ids = rank_assignments[rank]
    if not rank_sample_ids:
        raise RuntimeError("Megatron-Core Dynamic CP returned an empty rank assignment")
    return [peer for peer, sample_ids in enumerate(rank_assignments) if sample_ids == rank_sample_ids]


class DynamicCPScheduler:
    # [CLASS DOC]
    # Validates sizes and turns a replicated batch into this rank's list of micro-batches with per-micro-batch CP metadata. Attributes: max_seqlen_per_dp_cp_rank, dp_size, cp_size, total_ranks (dp*cp).
    """Apply Megatron-Core scheduling to the replicated batch used by verl #5057."""

    def __init__(
        self,
        max_seqlen_per_dp_cp_rank: int,
        dp_size: int,
        cp_size: int,
    ):
        # [FUNCTION DOC]
        # Stores sizes and total_ranks.
        # Exceptions: ValueError unless all three are positive plain ints (bool rejected via exact type check).
        if any(type(value) is not int or value <= 0 for value in (max_seqlen_per_dp_cp_rank, dp_size, cp_size)):
            raise ValueError("Dynamic CP scheduler sizes must be positive integers")

        self.max_seqlen_per_dp_cp_rank = max_seqlen_per_dp_cp_rank
        self.dp_size = dp_size
        self.cp_size = cp_size
        self.total_ranks = dp_size * cp_size

    def schedule(
        self,
        batch: TensorDict,
        dcp_group,
        *,
        tp_size: int,
        cp_layout: str = "zigzag",
        min_local_rows: int | None = None,
    ) -> list[TensorDict]:
        # [FUNCTION DOC]
        # Behavior: validates input_ids is a jagged nested tensor, batch is non-empty, and dcp_group size equals total_ranks; computes per-sample lengths from nested offsets (all must be >0) and checks the longest sequence fits in total_ranks * max_seqlen_per_dp_cp_rank. Instantiates the Megatron scheduler and obtains, for each micro-batch, a per-rank list of sample ids. For each micro-batch: verifies all ranks have non-empty assignments; finds this rank's CP members (ranks sharing identical ids), which must be contiguous and start at a multiple of the group size; selects the samples from the batch; builds the local padding mask; attaches non-tensor data local_cp_size, sample ids, group-leader flag (first member), padding mask and number of valid local tokens (mask size minus padding count).
        # Inputs: batch - TensorDict; dcp_group - process group over DPxCP; tp_size; cp_layout; min_local_rows.
        # Outputs: list of TensorDict micro-batches for this rank.
        # Exceptions: ValueError for invalid input; RuntimeError for invalid scheduler output.
        """Return this rank's scheduled micro-batches without rerouting batch data."""
        input_ids = batch.get("input_ids", None)
        if not isinstance(input_ids, torch.Tensor) or not input_ids.is_nested:
            raise ValueError("Dynamic CP requires input_ids to be a jagged NestedTensor")
        if len(batch) == 0:
            raise ValueError("Dynamic CP cannot schedule an empty batch")
        if dcp_group.size() != self.total_ranks:
            raise ValueError(
                f"DPxCP group size {dcp_group.size()} does not match "
                f"dp_size * cp_size ({self.dp_size} * {self.cp_size})"
            )

        seq_lens = [int(seq_len) for seq_len in input_ids.offsets().diff().tolist()]
        if any(seq_len <= 0 for seq_len in seq_lens):
            raise ValueError("Dynamic CP requires every input sequence to contain at least one token")

        max_seq_len = max(seq_lens)
        required_ranks = math.ceil(max_seq_len / self.max_seqlen_per_dp_cp_rank)
        if required_ranks > self.total_ranks:
            raise ValueError(
                f"A sequence with {max_seq_len} tokens needs at least {required_ranks} ranks at "
                f"max_seqlen_per_dp_cp_rank={self.max_seqlen_per_dp_cp_rank}, but the DPxCP group only has "
                f"{self.total_ranks} ranks. Increase max_seqlen_per_dp_cp_rank or cap the rollout sequence length."
            )

        scheduler = get_megatron_dynamic_cp_scheduler_cls()(
            max_seqlen_per_dp_cp_rank=self.max_seqlen_per_dp_cp_rank,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            microbatch_group_size_per_vp_stage=None,
        )
        sample_id_groups = scheduler.get_groups_and_subsamples(list(enumerate(seq_lens)))

        dcp_rank = dcp_group.rank()
        micro_batches = []
        for rank_assignments in sample_id_groups:
            if len(rank_assignments) != self.total_ranks or any(not ids for ids in rank_assignments):
                raise RuntimeError(
                    f"Megatron-Core Dynamic CP must assign all {self.total_ranks} ranks in every micro-batch"
                )

            sample_ids = rank_assignments[dcp_rank]
            members = _cp_members(rank_assignments, dcp_rank)
            local_cp_size = len(members)
            cp_rank = members.index(dcp_rank)
            if members != list(range(members[0], members[0] + local_cp_size)):
                raise RuntimeError(f"Dynamic CP group members must be contiguous, got {members}")
            if members[0] % local_cp_size != 0:
                raise RuntimeError(f"Dynamic CP group {members} is not aligned to its size")

            micro_batch = tu.index_select_tensor_dict(batch, sample_ids)
            selected_lens = [seq_lens[sample_id] for sample_id in sample_ids]
            padding_mask = _local_padding_mask(
                selected_lens,
                cp_size=local_cp_size,
                cp_rank=cp_rank,
                tp_size=tp_size,
                cp_layout=cp_layout,
                min_local_rows=min_local_rows,
            )
            tu.assign_non_tensor_data(micro_batch, "local_cp_size", local_cp_size)
            tu.assign_non_tensor_data(micro_batch, DCP_SAMPLE_IDS, list(sample_ids))
            tu.assign_non_tensor_data(micro_batch, DCP_GROUP_LEADER, dcp_rank == members[0])
            tu.assign_non_tensor_data(micro_batch, DCP_PADDING_MASK, padding_mask)
            tu.assign_non_tensor_data(
                micro_batch,
                DCP_LOCAL_NUM_TOKENS,
                padding_mask.numel() - int(padding_mask.sum().item()),
            )
            micro_batches.append(micro_batch)
        return micro_batches


def _detach_model_output(model_output: dict, sample_ids: list[int]) -> dict[str, list[torch.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Splits each tensor value of a model-output dict into per-sample tensors (unbind for nested, unbind along dim 0 otherwise), detaches and moves them to CPU.
    # Inputs: model_output - dict of tensors; sample_ids - ids expected (used for count check).
    # Outputs: dict key -> list of CPU tensors.
    # Exceptions: TypeError for non-tensor values; ValueError if sample count mismatches.
    detached = {}
    for key, value in model_output.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Dynamic CP model output {key!r} must be a tensor, got {type(value).__name__}")
        parts = list(value.unbind()) if value.is_nested else list(value.unbind(0))
        if len(parts) != len(sample_ids):
            raise ValueError(f"Dynamic CP output {key!r} has {len(parts)} samples for ids {sample_ids}")
        detached[key] = [part.detach().cpu() for part in parts]
    return detached


def postprocess_dynamic_cp_batch(
    output_lst: list[dict], batch_size: int, dcp_group, include_model_output: bool = True
) -> dict:
    # [FUNCTION DOC]
    # Behavior: picks the device of the first tensor found in any output's model_output. Keeps only outputs flagged as group leader, recording sample ids, loss, metrics and detached per-sample model outputs; all-gathers these records across dcp_group; checks the union of ids is exactly 0..batch_size-1 each once; sorts records by min sample id; collects non-None losses in order, merges metrics with append_to_dict, and maps per-key sample id to tensor. If include_model_output, rebuilds each key as a jagged nested tensor in original sample order and moves it back to the original device.
    # Inputs: output_lst - per-micro-batch output dicts; batch_size; dcp_group; include_model_output.
    # Outputs: dict with model_output (dict, empty if excluded), loss (list), metrics (dict of lists).
    # Exceptions: RuntimeError on duplicated/missing ids or missing per-key samples.
    # Side effects: collective all_gather_object over dcp_group.
    """Collect one output per CP group and restore the original sample order."""
    output_device = next(
        (
            value.device
            for output in output_lst
            for value in output.get("model_output", {}).values()
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    local_records = []
    for output in output_lst:
        if not output.get(DCP_GROUP_LEADER, False):
            continue
        sample_ids = [int(sample_id) for sample_id in output[DCP_SAMPLE_IDS]]
        local_records.append(
            {
                "sample_ids": sample_ids,
                "loss": output.get("loss", None),
                "metrics": output.get("metrics", {}),
                "model_output": _detach_model_output(output.get("model_output", {}), sample_ids),
            }
        )

    records_by_rank = [None] * dcp_group.size()
    torch.distributed.all_gather_object(records_by_rank, local_records, group=dcp_group)
    records = [record for rank_records in records_by_rank for record in rank_records]

    collected_ids = [sample_id for record in records for sample_id in record["sample_ids"]]
    expected_ids = set(range(batch_size))
    if sorted(collected_ids) != list(range(batch_size)):
        raise RuntimeError(
            f"Dynamic CP output collection must contain each sample once, got ids {sorted(collected_ids)}"
        )

    records.sort(key=lambda record: min(record["sample_ids"]))
    by_key: dict[str, dict[int, torch.Tensor]] = {}
    losses = []
    metrics = {}
    for record in records:
        if record["loss"] is not None:
            losses.append(record["loss"])
        append_to_dict(metrics, record["metrics"])
        for key, parts in record["model_output"].items():
            values = by_key.setdefault(key, {})
            for sample_id, part in zip(record["sample_ids"], parts, strict=True):
                values[sample_id] = part

    model_output = {}
    if include_model_output:
        for key, values in by_key.items():
            missing = expected_ids.difference(values)
            if missing:
                raise RuntimeError(f"Dynamic CP output {key!r} is missing sample ids {sorted(missing)}")
            model_output[key] = torch.nested.as_nested_tensor(
                [values[sample_id] for sample_id in range(batch_size)], layout=torch.jagged
            )
            if output_device is not None:
                model_output[key] = model_output[key].to(output_device)

    return {"model_output": model_output, "loss": losses, "metrics": metrics}
