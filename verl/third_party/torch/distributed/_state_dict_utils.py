# official torch 2.6.0 set_model_state_dict API leads to OOM
# this is a copy of torch/distributed/checkpoint from torch 2.7.0

# From PyTorch:

# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# From Caffe2:

# Copyright (c) 2016-present, Facebook Inc. All rights reserved.

# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.

# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.

# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.

# All contributions by Kakao Brain:
# Copyright 2019-2020 Kakao Brain

# All contributions by Cruise LLC:
# Copyright (c) 2022 Cruise LLC.
# All rights reserved.

# All contributions by Tri Dao:
# Copyright (c) 2024 Tri Dao.
# All rights reserved.

# All contributions by Arm:
# Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.

# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.

# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# ruff: noqa: B028, UP038, UP007, E721, E501
# mypy: allow-untyped-defs
# [MODULE DOC]
# Vendored copy of torch.distributed._state_dict_utils from torch 2.7 (used because the torch 2.6 set_model_state_dict
# path can OOM). Provides a generic recursive state-dict walker with per-type callbacks (ShardedTensor, DTensor, plain
# tensor), gather/offload/copy/CPU-clone helpers, a structure-similarity check, rank-0 broadcast or local distribution
# of a full state dict into a (possibly DTensor-sharded) local state dict, and DCP-style flatten/unflatten utilities.
# Distributed types are imported only when torch.distributed is available.

import copy
import io
import math
import weakref
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Union, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import AsyncCollectiveTensor

if dist.is_available() or TYPE_CHECKING:
    from torch.distributed import distributed_c10d
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset


def _identity_func(
    obj: torch.Tensor,
    pg: Optional[dist.ProcessGroup],
    device: Optional[torch.device],
    companion_obj: Any,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: callback that returns obj unchanged (ignores pg, device, companion).
    return obj


def _all_gather_sharded_tensor(
    sharded_tensor: "ShardedTensor",
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Behavior: all-gather a ShardedTensor sharded along dim 0 into a full tensor. Each rank contributes its first
    # local shard flattened (moved to the process group's device, zero-padded to a uniform chunk size of
    # ceil(dim0/world) rows' worth of elements), or zeros if it has no shard; gathers into one flat buffer, trims to total
    # numel and reshapes to the global size.
    # Inputs: sharded_tensor; pg - default group if None; device - gather device (default the group's default device).
    # Outputs: full tensor on the gather device. Side effects: collective communication.
    if pg is None:
        pg = distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    shards = sharded_tensor.local_shards()
    dim_0_size = sharded_tensor.size()[0]  # type: ignore[index]
    tensor_numel = sharded_tensor.size().numel()  # type: ignore[union-attr]
    chunk_size = math.ceil(dim_0_size / world_size) * tensor_numel // dim_0_size
    pg_device = distributed_c10d._get_pg_default_device(pg) if device is None else device
    if shards:
        local_tensor = shards[0].tensor.flatten()
        if local_tensor.device.type != pg_device.type:
            local_tensor = local_tensor.to(pg_device)
        num_padding = chunk_size - local_tensor.numel()
        if num_padding > 0:
            local_tensor = F.pad(local_tensor, [0, num_padding])
    else:
        local_tensor = torch.zeros(chunk_size, dtype=sharded_tensor.dtype, device=pg_device)

    tensor = torch.empty(
        chunk_size * world_size,
        dtype=local_tensor.dtype,
        device=pg_device,
    )
    dist.all_gather_into_tensor(tensor, local_tensor, group=pg)

    tensor = tensor.narrow(0, 0, tensor_numel).reshape(sharded_tensor.size())
    return tensor


class CompanionMismatch(Exception):
    # [CLASS DOC]
    # Exception raised when a companion object's structure, keys, dtype or size does not match the state dict.
    pass


def _iterate_state_dict(
    iter_object: Any,
    sharded_tensor_func: Callable,
    dtensor_func: Callable,
    tensor_func: Callable,
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    companion_obj: Any = None,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
    non_blocking: bool = True,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Behavior: recursively walk a state dict. ShardedTensor/DTensor/Tensor leaves go to their callback (called with
    # value, pg, device, companion); int/float/str/bytes/BytesIO/None pass through; dicts and lists/tuples recurse (with the
    # companion's matching child; tuples are rebuilt as tuples); other types are deep-copied if type_check is False, else
    # rejected. After processing, if ranks_only is empty or this rank is in it: tensor results are moved to CPU when
    # cpu_offload and no companion; with a companion, the result is copied into the companion (into its local tensor for
    # DTensor) and the companion is returned. Ranks not in ranks_only get an empty dict for dicts, else None.
    # Inputs: see docstring. Outputs: new structure with processed leaves.
    # Exceptions: CompanionMismatch on structural mismatch; ValueError on unsupported types when type_check.
    # KNOWN ISSUE: for a dict mismatch the message logic is inverted: it is empty when the companion is a dict (key
    # mismatch) and calls keys() on the non-dict companion otherwise, which raises AttributeError instead.
    """Iterate through the state dict, applying the given functions to each tensor type.

    Args:
        iter_object (Any): the target state_dict.
        sharded_tensor_func (Callable): the function to apply to ShardedTensor
        dtensor_func (Callable): the function to apply to DTensor
        tensor_func (Callable): the function to apply to Tensor
        pg (Optional[dist.ProcessGroup]): process group passed to tensor functions
        device (Optional[torch.device]): device passed to tensor functions
        cpu_offload (bool): whether to offload the tensors to CPU memory. This option is ignored
            if a companion_obj is supplied.
        companion_obj (Any): A companion object to the state dict. If this object
            is supplied, we attempt to copy the tensor to the companion object.
        ranks_only (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.
        non_blocking (bool): whether to use non-blocking copy when copying to the companion object.
    """
    # TODO: should we use pytree?
    cpu_device = torch.device("cpu")
    if isinstance(iter_object, ShardedTensor):
        ret = sharded_tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, DTensor):
        ret = dtensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, torch.Tensor):
        ret = tensor_func(iter_object, pg, device, companion_obj)
    elif isinstance(iter_object, (int, float, str, bytes, io.BytesIO)) or iter_object is None:
        ret = iter_object
    elif isinstance(iter_object, dict):
        if companion_obj is not None and (
            not isinstance(companion_obj, dict) or set(companion_obj.keys()) != set(iter_object.keys())
        ):
            msg = "" if isinstance(companion_obj, dict) else f"{set(companion_obj.keys())=} {set(iter_object.keys())=}"
            raise CompanionMismatch(msg)

        ret = {
            key: _iterate_state_dict(
                value,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[key] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for key, value in iter_object.items()
        }
    elif isinstance(iter_object, (list, tuple)):
        if companion_obj is not None and (
            not isinstance(companion_obj, (list, tuple)) or len(companion_obj) != len(iter_object)
        ):
            raise CompanionMismatch

        ret = [
            _iterate_state_dict(
                v,
                sharded_tensor_func,
                dtensor_func,
                tensor_func,
                pg=pg,
                device=device,
                cpu_offload=cpu_offload,
                companion_obj=companion_obj[idx] if companion_obj is not None else None,
                ranks_only=ranks_only,
                type_check=type_check,
                non_blocking=non_blocking,
            )
            for idx, v in enumerate(iter_object)
        ]
        if isinstance(iter_object, tuple):
            ret = tuple(ret)
    elif not type_check:
        ret = copy.deepcopy(iter_object)
    else:
        raise ValueError(f"Unexpected value type {type(iter_object)}")

    if not ranks_only or dist.get_rank(pg) in ranks_only:
        if isinstance(ret, torch.Tensor):
            if cpu_offload and companion_obj is None:
                ret = ret.to(cpu_device)

            if companion_obj is not None:
                if isinstance(companion_obj, DTensor):
                    assert isinstance(ret, DTensor)
                    companion_obj._local_tensor.copy_(ret._local_tensor, non_blocking=non_blocking)
                else:
                    companion_obj.copy_(ret, non_blocking=non_blocking)
                ret = companion_obj
    else:
        ret = {} if isinstance(ret, dict) else None

    return ret


def _gather_state_dict(
    state_dict: dict[str, Any],
    *,
    pg: Optional[dist.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    cpu_offload: bool = False,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Behavior: gather all ShardedTensors (all-gather, moved back to the local shard device) and DTensors
    # (redistributed to all-Replicate placements, converted to local, waiting on async collectives) into full tensors,
    # leaving plain tensors as-is; honors cpu_offload, ranks_only and type_check.
    # Outputs: gathered state dict. Side effects: collective communication.
    """
    Given a state_dict, this API gathers all the ShardedTensors or DTensors in
    the state_dict.


    Args:
        state_dict (Dict[str, Any]): the target sharded state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        device: (Optional[torch.device]): the device that is used to
            perform allgather for ShardedTensor. Note that gathering a DTensor
            will use the DeviceMesh. So this argument will be ignored when
            gathering a DTensor.
        cpu_offload (bool): whether to offload the tensors to CPU memory. The
            default value is False.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    def sharded_tensor_func(value, pg, device, companion_obj):
        # [FUNCTION DOC]
        # HELPER: ShardedTensor callback: all-gather and move to the local shard's device (CPU if no shards).
        # ShardedTensor does not seem to record the original device type.
        # So if the tensor is moved to CPU, we won't know the original type.
        # As a result, we have to rely on the user to tell us the correct one.
        cpu_device = torch.device("cpu")
        output_tensor = _all_gather_sharded_tensor(value, pg, device)
        local_shard_device = value.local_shards()[0].tensor.device if value.local_shards() else cpu_device
        if output_tensor.device != local_shard_device:
            value = output_tensor.to(local_shard_device)
        else:
            value = output_tensor
        return value

    def dtensor_func(value, pg, device, companion_obj):
        # [FUNCTION DOC]
        # HELPER: DTensor callback: move to the mesh device type if needed (compares a device with a device-type string, so it
        # effectively always calls the move), redistribute to Replicate on all mesh dims, take the local tensor and wait if
        # asynchronous.
        if value.device != value.device_mesh.device_type:
            value = value.to(value.device_mesh.device_type)
        # FSDP all_gather: [Shard(0)] -> [Replicate()]
        # HSDP all_gather: [Replicate(), Shard(0)] -> [Replicate(), Replicate()]
        # 2D FSDP + TP all_gather:
        # - [Shard(0), Shard(n)] -> [Replicate(), Replicate()]
        # - [Shard(0), Replicate()] -> [Replicate(), Replicate()]
        placements = [Replicate() for _ in value.placements]
        value = value.redistribute(
            device_mesh=value.device_mesh,
            placements=placements,
        )
        # Call `wait()` to force the tensor to be synchronous with respect
        # to the main stream.
        # See the discussion in https://github.com/pytorch/pytorch/pull/117799.
        value = value.to_local()
        if isinstance(value, AsyncCollectiveTensor):
            value = value.wait()
        return value

    return _iterate_state_dict(
        state_dict,
        sharded_tensor_func,
        dtensor_func,
        _identity_func,
        pg=pg,
        device=device,
        cpu_offload=cpu_offload,
        ranks_only=ranks_only,
        type_check=type_check,
    )


def _offload_state_dict_to_cpu(
    state_dict: dict[str, Any],
    *,
    ranks_only: tuple[int, ...] = (),
    type_check: bool = True,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Behavior: walk the state dict with identity callbacks and cpu_offload so all tensors are moved to CPU.
    # Inputs: state_dict; ranks_only; type_check. Outputs: new state dict. (The docstring's pg argument does not exist.)
    """
    Given a state_dict, this API offload all the tensors to CPU memory.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        pg (Optional[dist.ProcessGroup]): the process group that is used to
            gather ShardedTensor. Note that gathering a DTensor will use
            the DeviceMesh. So this argument will be ignored when gathering a
            DTensor.
        ranks_only: (Tuple[int, ...]): if this tuple is empty, all ranks will
            have the same state_dicts. Otherwise only ranks that in ``ranks_only``
            have the same state_dicts. Other ranks will get empty state_dicts.
        type_check: (bool): check if the instance data type is a supported type
            that can be saved by DCP.  The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        The gathered state dictionary.
    """

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=True,
        ranks_only=ranks_only,
        type_check=type_check,
    )
    return ret


@torch.no_grad()
def _copy_state_dict(
    state_dict: dict[str, Any],
    copy_state_dict: dict[str, Any],
    non_blocking: bool = False,
    type_check: bool = True,
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Behavior: under no_grad, copy every tensor of state_dict into the matching tensor of copy_state_dict (same
    # structure required), returning a structure whose leaves are the companion tensors.
    # Inputs: state_dict; copy_state_dict; non_blocking; type_check.
    # Exceptions: CompanionMismatch on structural mismatch. Side effects: writes into copy_state_dict tensors.
    """
    Copies all tensors in a given state dict into a different state_dict with the
    same structure. Additionally, a copied state dict with the same value references
    is returned. Editing the keys on this state dict will not affect the
    passed in copy_state_dict (but the value references are the same).

    .. warning::
        It is expected by this function that state_dict and copy_state_dict share
        the same structure and data types.

    .. warning::
        The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Args:
        state_dict (Dict[str, Any]): the target state_dict.
        copy_state_dict (Dict[str, Any]):
            The state dict we are copying into. This state_dict must have exactly
             the same structure as the source `state_dict`.
        non_blocking: (bool): Whether copy ops should be performed asynchronously
        type_check (bool): check if the instance data type is a supported type
            that can be saved by DCP. The current supported data types are
            torch.Tensor, DTensor, int, float, str, list, dict, None.

    Returns:
        State Dict copy
    """

    return _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        _identity_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        companion_obj=copy_state_dict,
        type_check=type_check,
        non_blocking=non_blocking,
    )


@torch.no_grad()
def _create_cpu_state_dict(
    state_dict: dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Behavior: under no_grad, create a structurally identical state dict where each tensor is a new uninitialized CPU
    # tensor of the same size and dtype, optionally in shared memory and/or pinned (shared+pinned registers the memory with
    # cudaHostRegister and unregisters on garbage collection). DTensors are moved/copied to CPU with a new local tensor.
    # Non-tensor values are deep-copied (type_check off).
    # Inputs: state_dict; pin_memory; share_memory. Outputs: new state dict (values not copied).
    # Exceptions: AssertionError if CUDA host registration fails.
    # Note: zero-dim tensors become a new scalar tensor holding 0 of the same dtype (contents are not preserved).
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: tensor callback allocating the empty CPU tensor as described above.
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:

                def unpin_memory(t):
                    # [FUNCTION DOC]
                    # HELPER: unregisters pinned host memory for a tensor. Exceptions: AssertionError on non-zero CUDA error code.
                    succ = int(torch.cuda.cudart().cudaHostUnregister(t.data_ptr()))
                    assert succ == 0, f"Unpinning shared memory failed with error-code: {succ}"

                weakref.finalize(t, unpin_memory, t)
                succ = int(
                    torch.cuda.cudart().cudaHostRegister(
                        t.data_ptr(),
                        t.numel() * t.element_size(),
                        1,  # lines up with 'cudaHostRegisterPortable'
                    )
                )
                assert succ == 0, f"Pinning shared memory failed with error-code: {succ}"
            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    def dtensor_func(
        obj: DTensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> DTensor:
        # [FUNCTION DOC]
        # HELPER: DTensor callback: CPU copy of the DTensor with its local tensor replaced by a fresh CPU tensor (zero-dim
        # returned as is).
        if len(obj.size()) == 0:
            return obj

        if obj.device != torch.device("cpu"):
            ret = cast(DTensor, obj.to(device="cpu"))
        else:
            ret = copy.deepcopy(obj)
        ret._local_tensor = tensor_func(ret._local_tensor, pg, device, None)
        return ret

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        dtensor_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )
    return ret


def _check_state_dict_similarity(
    state_dict: dict[str, Any],
    compared_state_dict: dict[str, Any],
) -> bool:
    # [FUNCTION DOC]
    # HELPER: Behavior: walk state_dict with compared_state_dict as companion, raising on structure, dtype or size mismatch
    # of plain tensors (DTensor/ShardedTensor leaves are not compared).
    # Outputs: True if similar, False on CompanionMismatch.
    # Note: when structures match the walk still copies matching tensors into the companion, so compared_state_dict is
    # overwritten with state_dict's tensor values as a side effect.
    """
    Given two state_dicts, check if the structures are the same. And
    if a [key, tensor] pair exist in one state_dict there must be
    the a corresponding pait, [key, other_tensor], in the other state_dict,
    where tensor and other_tensor have the same size and dtype.

    Return the check result.
    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        companion_obj: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: callback raising CompanionMismatch when dtype or size differ, else returning obj.
        if companion_obj.dtype != obj.dtype or companion_obj.size() != obj.size():
            raise CompanionMismatch
        return obj

    try:
        _iterate_state_dict(
            state_dict,
            _identity_func,
            _identity_func,
            tensor_func,
            pg=None,
            device=None,
            cpu_offload=False,
            ranks_only=(),
            companion_obj=compared_state_dict,
            type_check=False,
        )
    except CompanionMismatch:
        return False

    return True


class _TensorInfo(NamedTuple):
    # [CLASS DOC]
    # NamedTuple (size, dtype) describing a tensor without its data; broadcast in place of large tensors.
    size: torch.Size
    dtype: torch.dtype


def _broadcast_tensors(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    keys: list[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: for each key build a tensor on device: rank 0 uses its full tensor, other ranks allocate an empty
    # tensor from the broadcast tensor info. Record in local_state_dict either the full tensor or, for DTensor entries, a
    # pair of (local DTensor, full tensor); keys absent locally are skipped. Broadcast all tensors from rank 0 (coalesced if
    # more than one), then distribute pairs into shards.
    # Inputs: full_state_dict - rank0 tensors or tensor infos; local_state_dict; keys; device; pg (default group if None).
    # Side effects: collective communication; mutates local_state_dict.
    # Note: rank checks use the global rank, so rank 0 of the default group must be the source.
    tensors = []
    for key in keys:
        if dist.get_rank() == 0:
            full_state = full_state_dict[key]
            assert isinstance(full_state, torch.Tensor)
            full_tensor = full_state.detach().to(device)
        else:
            tensor_info = full_state_dict[key]
            full_tensor = torch.empty(
                size=tensor_info.size,
                device=device,
                dtype=tensor_info.dtype,
            )
        tensors.append(full_tensor)
        local_state = local_state_dict.get(key, None)
        if local_state is None:
            continue
        elif isinstance(local_state, DTensor):
            local_state_dict[key] = (local_state, full_tensor)
        else:
            local_state_dict[key] = full_tensor

    if pg is None:
        pg = dist.distributed_c10d._get_default_group()

    if len(tensors) > 1:
        dist._broadcast_coalesced(pg, tensors, 500, 0)
    else:
        dist.broadcast(tensors[0], src=0, group=pg)

    _distribute_tensors(local_state_dict, keys, device, pg)


def _distribute_tensors(
    local_state_dict: dict[str, Any],
    keys: list[str],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: for each key whose local entry is a (DTensor, full tensor) pair, compute this rank's local shard
    # shape and offset for the DTensor placements and slice the full tensor accordingly. If the DTensor is on meta device,
    # build a new DTensor from a cloned slice with the original shape and stride; otherwise copy the slice into the
    # existing local tensor. Store the result under the key.
    # Side effects: mutates local_state_dict and existing DTensor storage.
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()
    for key in keys:
        _local_state = local_state_dict.get(key, None)
        if _local_state is None or torch.is_tensor(_local_state):
            continue

        local_state = _local_state[0]
        full_tensor = _local_state[1]

        shape, offset = compute_local_shape_and_global_offset(
            full_tensor.shape, local_state.device_mesh, local_state.placements
        )
        slices = [
            slice(cur_offset, cur_offset + cur_shape) for cur_shape, cur_offset in zip(shape, offset, strict=False)
        ]
        if local_state.is_meta:
            # Use .clone() here rather than view to clone and return only the sliced portion, minimizing memory access and cost.
            local_tensor = full_tensor[slices].detach().clone()
            # TODO: currently, we cannot handle strided sharding if the dp dimension is not even. For example,
            # one of the case that is not yet supported is when placements = (Shard(0), _StridedShard(0, sf=2)).
            ret = DTensor.from_local(
                local_tensor,
                local_state.device_mesh,
                local_state.placements,
                shape=local_state.shape,
                stride=local_state.stride(),
            )
        else:
            ret = local_state
            # Copy full_tensor[slices] into local_state.to_local() to reduce memory footprint.
            ret.to_local().copy_(full_tensor[slices])
        local_state_dict[key] = ret


def _broadcast_state_dict(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
    strict: bool = False,
    cpu_offload: bool = False,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: rank 0 builds a summary of its full state dict (non-tensors as-is, scalars on CPU, other tensors
    # as size/dtype info) and broadcasts it as an object. Every rank copies non-tensor values into existing local keys, then
    # for each tensor key broadcasts it individually (to limit memory) into local_state_dict, optionally moving to CPU.
    # With strict, local keys absent from the global set are removed.
    # Inputs: full_state_dict (meaningful on rank 0); local_state_dict; device; pg; strict; cpu_offload.
    # Side effects: collectives; mutates local_state_dict.
    # Note: zero-dim tensors are carried in the object broadcast and assigned only if the key exists locally.
    # Broadcast from rank0's `full_state_dict` to all ranks' `local_state_dict`.
    # If strict is True, any keys in `local_state_dict` but not in `full_state_dict`
    # will be removed from `local_state_dict`.
    ret = {}
    if dist.get_rank() == 0:
        for key, value in full_state_dict.items():
            if not torch.is_tensor(value):
                ret[key] = value
            elif value.dim() == 0:
                ret[key] = value.cpu()
            else:
                ret[key] = _TensorInfo(value.size(), value.dtype)

    broadcast_list = [ret]
    dist.broadcast_object_list(broadcast_list, src=0, group=pg)
    ret = broadcast_list[0]
    # Gather values
    keys = []
    local_state_dict_keys = set(local_state_dict.keys())
    global_keys = set()
    for key, value in ret.items():
        global_keys.add(key)
        if not isinstance(value, _TensorInfo):
            if key in local_state_dict:
                local_state_dict[key] = value
            continue

        if dist.get_rank() == 0:
            ret[key] = full_state_dict[key]

        keys.append(key)
        # Broadcast every tensor to avoid OOM for now.
        if len(keys) >= 1:
            _broadcast_tensors(ret, local_state_dict, keys, device, pg)
            if cpu_offload:
                for key in keys:
                    local_state_dict[key] = local_state_dict[key].cpu()
            keys.clear()

    if strict:
        if missing_keys := (local_state_dict_keys - global_keys):
            for key in missing_keys:
                local_state_dict.pop(key)

    if keys:
        _broadcast_tensors(ret, local_state_dict, keys, device, pg)
        if cpu_offload:
            for key in keys:
                local_state_dict[key] = local_state_dict[key].cpu()


def _distribute_state_dict(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    device: torch.device,
    pg: Optional[dist.ProcessGroup] = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: every rank holds the full state dict; for each entry, non-tensors and scalars are written into the
    # local dict, tensors replace existing local entries: DTensor entries are re-sharded with distribute_tensor using the
    # local placements, plain tensors moved to device; keys missing locally are skipped.
    # Side effects: mutates local_state_dict; distribute_tensor may communicate.
    # KNOWN ISSUE: the check skipping keys not in full_state_dict is always false (iterating that same dict); it was likely
    # meant to check local_state_dict, and as written non-tensor values are added even for keys absent locally.
    # Full_state_dict = True, broadcast_from_rank0 = False here. Each rank has
    # full_state_dict. Skip the broadcast in ``_broadcast_state_dict`` and
    # distribute tensors in each rank
    for key, value in full_state_dict.items():
        if key not in full_state_dict:
            continue
        if not torch.is_tensor(value):
            local_state_dict[key] = value
        elif value.dim() == 0:
            local_state_dict[key] = value.cpu()
        else:
            assert isinstance(value, torch.Tensor)
            local_state = local_state_dict.get(key, None)
            if local_state is None:
                continue
            elif isinstance(local_state, DTensor):
                local_state_dict[key] = distribute_tensor(
                    value.detach().to(device),
                    local_state.device_mesh,
                    local_state.placements,
                )
            else:
                local_state_dict[key] = value.detach().to(device)


# These APIs are from torch.distributed.checkpoint.
# TODO: We should consolidate the code here as some not all modules can depend on
# DCP.
PATH_ITEM = Union[str, int]
OBJ_PATH = tuple[PATH_ITEM, ...]
FLATTEN_MAPPING = dict[str, OBJ_PATH]
STATE_DICT_TYPE = dict[str, Any]
CONTAINER_TYPE = MutableMapping[PATH_ITEM, Any]


def _traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, Any], None],
) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: depth-first walk; mappings extend the path with the string key, lists/tuples with the integer index;
    # any other value invokes visitor(path, value).
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping, list, and tuple will be flattened and other value types are treated
    as the terminal values and will invoke ``visitor``.
    """

    def _traverse_obj(path: OBJ_PATH, value: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: recursive step of the traversal.
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)
        else:
            visitor(path, value)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def _flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    # [FUNCTION DOC]
    # HELPER: Behavior: flatten nested containers into a single-level mapping keyed by the dot-joined path, recording each
    # key's original path.
    # Outputs: (flattened mapping, mapping from flat key to path tuple).
    # Exceptions: ValueError on duplicate flat keys.
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        # [FUNCTION DOC]
        # HELPER: visitor storing the value and path under the dot-joined key; raises on duplicates.
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    _traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def _set_element(root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: Any) -> None:
    # [FUNCTION DOC]
    # HELPER: Behavior: walk/create containers along path (dict when the next key is a string, list when an integer,
    # extending lists with None as needed) and set value at the final key.
    # Side effects: mutates root_dict.
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: list[Any], idx: int) -> None:
        # [FUNCTION DOC]
        # HELPER: pads a list with None until index idx is valid.
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val: CONTAINER_TYPE | list[Any] = {} if type(key) == str else []

        if isinstance(cur_container, Mapping):
            cur_container = cast(CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val))
        else:
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) == int:
        extend_list(cast(list[Any], cur_container), key)

    cur_container[key] = value


def _unflatten_state_dict(state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING) -> STATE_DICT_TYPE:
    # [FUNCTION DOC]
    # HELPER: Behavior: rebuild a nested state dict by setting each flat value at its recorded path.
    # Outputs: nested mapping. Exceptions: KeyError if a key lacks a mapping.
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        _set_element(nested, mapping[key], value)
    return nested
