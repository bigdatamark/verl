# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# Utilities for the SGLang rollout: LoRA config helpers, DeepSeek-V4 fused-parameter
# grouping, a pickle-based object broadcast, and size-bounded bucketing of named tensors
# for weight sync. Constants: SGLANG_LORA_NAME (adapter name used in SGLang),
# DEEPSEEK_V4_FUSION_GROUPS (tuples of name suffixes, under "self_attn" and "attn", that
# must be loaded together in one bucket).

import pickle
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist

from verl.utils.device import get_device_name
from verl.workers.rollout.utils import ensure_async_iterator

SGLANG_LORA_NAME = "verl_actor_lora_name"

_DEEPSEEK_V4_FUSION_MEMBERS = (
    ("wq_a.weight", "wkv.weight"),
    ("wq_a.scale", "wkv.scale"),
    ("wq_a.weight_scale_inv", "wkv.weight_scale_inv"),
    ("compressor.wkv.weight", "compressor.wgate.weight"),
    ("indexer.compressor.wkv.weight", "indexer.compressor.wgate.weight"),
)
DEEPSEEK_V4_FUSION_GROUPS = tuple(
    tuple(f".{attention}.{member}" for member in members)
    for attention in ("self_attn", "attn")
    for members in _DEEPSEEK_V4_FUSION_MEMBERS
)


def _fusion_key(name: str, groups: tuple[tuple[str, ...], ...]):
    # [FUNCTION DOC]
    # HELPER: Find which fusion group a parameter name belongs to by suffix match.
    # Outputs: None if no match; else (name prefix without suffix, group index, group size).
    # Exceptions: ValueError if the name matches more than one suffix/group.
    matches = [
        (index, suffix, len(group)) for index, group in enumerate(groups) for suffix in group if name.endswith(suffix)
    ]
    if len(matches) > 1:
        raise ValueError(f"{name!r} matches multiple fusion groups: {matches}")
    if not matches:
        return None
    index, suffix, size = matches[0]
    return name[: -len(suffix)], index, size


def normalize_peft_config_for_sglang(peft_config: dict) -> dict:
    # [FUNCTION DOC]
    # Copy a PEFT adapter config dict, converting enum values of "task_type"/"peft_type" to
    # their .value, and making target_modules a list unless it is a single string.
    # Exceptions: ValueError if "peft_type" is missing; KeyError if "target_modules" is missing.
    """Normalize an engine's adapter config (enums to strings) for SGLang's adapter loader."""
    normalized = dict(peft_config)
    for key in ("task_type", "peft_type"):
        if key in normalized:
            normalized[key] = getattr(normalized[key], "value", normalized[key])
    if "peft_type" not in normalized:
        raise ValueError(
            "adapter config has no 'peft_type', which SGLang's adapter loader requires. See "
            "BaseEngine.get_per_tensor_param for the keys. Keys present: " + ", ".join(sorted(normalized))
        )
    # A bare string must stay one: list() would tear "all-linear" into characters.
    target_modules = normalized["target_modules"]
    normalized["target_modules"] = target_modules if isinstance(target_modules, str) else list(target_modules)
    return normalized


def lora_rank_of(model_config) -> int:
    # [FUNCTION DOC]
    # Return the LoRA rank as the max of the flat lora_rank attribute and lora["rank"]
    # (treating missing/None as 0).
    """The LoRA rank, from whichever block carries it: megatron sets ``model.lora.rank``, fsdp
    the flat ``model.lora_rank``."""
    return max(int(getattr(model_config, "lora_rank", 0) or 0), int(model_config.lora.get("rank", 0) or 0))


def lora_served_as_adapter(model_config) -> bool:
    # [FUNCTION DOC]
    # True when LoRA is enabled (rank > 0) and lora["merge"] is not true, i.e. SGLang must
    # serve the adapter separately.
    """Whether SGLang should serve LoRA as a hot-swappable adapter.

    ``HFModelConfig`` carries two LoRA config blocks that are never synced: megatron
    runs set ``model.lora.rank`` (dict) while fsdp runs set the flat ``model.lora_rank``,
    so both must be checked to detect that LoRA is enabled at all.

    With ``model.lora.merge=True`` the trainer merges the adapter into the base weights
    and pushes a full HF-keyed weight update (``peft_config=None``), so no adapter is ever
    loaded into SGLang: the engine must not be launched with ``enable_lora`` and requests
    must not carry a ``lora_path``.
    """
    lora_enabled = lora_rank_of(model_config) > 0
    return lora_enabled and not model_config.lora.get("merge", False)


def sglang_lora_target_modules(target_modules: Any) -> list[str]:
    # [FUNCTION DOC]
    # Map verl target_modules to SGLang lora_target_modules: "all-linear" becomes ["all"];
    # other strings (regex) are rejected; iterables are listed.
    # Exceptions: ValueError for non-"all-linear" strings.
    """Render verl's ``model.target_modules`` as SGLang's ``lora_target_modules``."""
    if target_modules == "all-linear":
        return ["all"]
    if isinstance(target_modules, str):
        raise ValueError(
            f"SGLang cannot serve a regex `target_modules` ({target_modules!r}); PEFT matches it "
            f"against the whole parameter key. Use `all-linear`, or list the module names."
        )
    return list(target_modules)


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    # [FUNCTION DOC]
    # HELPER: Broadcast a list of picklable objects from src via torch.distributed. Source
    # sends the byte length (0 for empty) then the pickled bytes as uint8 tensors on the
    # accelerator (or CPU if forced) and returns data unchanged; receivers read the length,
    # return an empty list for 0, else receive and unpickle the bytes.
    # Inputs: data (used only on src), rank, dist_group, src, force_cpu_device.
    # Outputs: the broadcast list. Side effects: collective communication.
    # KNOWN ISSUE: src is passed to dist.broadcast as a global rank while also compared to the
    # caller-supplied rank; receivers with a non-default group must pass the global src.
    # Unpickling received data is unsafe with untrusted peers.
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


def _compact_for_bucket(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Clone a tensor only if it is a view of another tensor's storage, so buckets do
    # not pin larger backing buffers; tensors owning their storage are returned as-is.
    """Return a tensor safe to retain in a weight-sync bucket without pinning extra memory.

    ``get_named_tensor_buckets`` keeps every tensor alive until its bucket is flushed. A tensor
    that is a *view* into a larger backing buffer would therefore keep that whole buffer resident
    (and ship the whole buffer downstream), so such views must be compacted with ``clone()``.

    However the weights synced here come from ``DTensor.full_tensor()`` (a fresh all-gather) and
    already own tight, contiguous storage. Cloning those allocates a second full-size buffer and
    transiently doubles the tensor's footprint -- which OOMs on multi-GiB fused MoE weights
    (e.g. ``[num_experts, ...]`` ``gate_up_proj``/``qkv``) while the actor params and rollout
    weights are both already resident. Skip the clone when the tensor already owns its storage.
    """
    return tensor.clone() if tensor._base is not None else tensor


async def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]],
    bucket_bytes: int,
    fusion_groups: tuple[tuple[str, ...], ...] = (),
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    # [FUNCTION DOC]
    # Async generator that groups (name, tensor) pairs into lists whose total byte size stays
    # at or under bucket_bytes where possible.
    # Behavior: accepts sync or async iterables. Each tensor is compacted. Tensors matching a
    # fusion group are staged until all members with the same prefix arrive, then placed
    # together. Placement of an indivisible run: if the current bucket is non-empty and adding
    # the run would exceed bucket_bytes, the current bucket is yielded first; the run is then
    # appended (a single run larger than the limit forms an oversize bucket). The final
    # partial bucket is yielded at the end.
    # Inputs: iterable; bucket_bytes (> 0); fusion_groups - tuples of suffixes.
    # Outputs: yields lists of (name, tensor).
    # Exceptions: ValueError if bucket_bytes <= 0 or some fused group never completed.
    # KNOWN ISSUE: the docstring says size is in megabytes and its example uses a
    # nonexistent bucket_size_mb argument with synchronous iteration; the function takes bytes
    # and is an async generator.
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    # Members of a fused destination param (SGLang's DSv4 wqkv_a / wkv_gate) are
    # held here until the group is whole, then placed in one bucket: that loader
    # cats both halves inside a single load_weights call and asserts its cache
    # empty on return, so a bucket boundary between them makes it fire. A full
    # sync contains every member by construction, so this only ever delays a
    # tensor by its sibling -- nothing is dropped.
    staged: dict = {}

    def _place(entries):
        # [FUNCTION DOC]
        # HELPER: Decide whether the current bucket must be cut before a run of entries: if the
        # bucket is non-empty and the run's total size would overflow, returns (old bucket,
        # entries) after resetting the enclosing bucket state; otherwise (None, entries).
        # Side effects: rebinds the enclosing current_bucket/current_size.
        """Put an indivisible run of entries in a bucket, cutting before it if
        it would not fit rather than partway through."""
        nonlocal current_bucket, current_size
        total = sum(sz for _, _, sz in entries)
        if current_bucket and current_size + total > bucket_bytes:
            out, current_bucket, current_size = current_bucket, [], 0
            return out, entries
        return None, entries

    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        match = _fusion_key(name, fusion_groups)
        if match is not None:
            prefix, group_index, expected_size = match
            key = (prefix, group_index)
            slot = staged.setdefault(key, [])
            slot.append((name, _compact_for_bucket(tensor), tensor_size))
            if len(slot) < expected_size:
                continue
            entries = staged.pop(key)
        else:
            entries = [(name, _compact_for_bucket(tensor), tensor_size)]

        flushed, entries = _place(entries)
        if flushed:
            yield flushed
        for n, t, sz in entries:
            current_bucket.append((n, t))
            current_size += sz

    if staged:
        raise ValueError(f"fused parameter groups never completed in this sync: {sorted(staged)}")
    if current_bucket:
        yield current_bucket
