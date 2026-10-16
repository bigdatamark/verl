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
# DeepSpeed-Ulysses sequence-parallel (SP) utilities. Holds a module-level global SP process group
# (None means SP disabled) with getters/setters; all-to-all and all-gather collectives wrapped as
# autograd Functions; helpers to pad/slice/gather sequence tensors; and sharding managers that swap
# the active SP group and all-gather/split DataProto batches across the SP group.

"""
Utilities for DeepSpeed Ulysses Sequence Parallelism.
DeepSpeed Ulysses Paper: https://arxiv.org/abs/2309.14509
Inspired from: https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/layer.py
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from verl import DataProto

_ULYSSES_SEQUENCE_PARALLEL_GROUP = None


def set_ulysses_sequence_parallel_group(group: dist.ProcessGroup):
    # [FUNCTION DOC]
    # Behavior: stores group in the module-global SP group slot. Inputs: group - process group or None.
    # Outputs: None. Side effects: mutates the module global.
    """
    Set ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    _ULYSSES_SEQUENCE_PARALLEL_GROUP = group


def get_ulysses_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    # [FUNCTION DOC]
    # Behavior: returns the module-global SP group (may be None). Inputs: none. Outputs: group or None.
    """
    Get ulysses sequence parallel process group.
    """
    global _ULYSSES_SEQUENCE_PARALLEL_GROUP
    return _ULYSSES_SEQUENCE_PARALLEL_GROUP


def get_ulysses_sequence_parallel_world_size(group: ProcessGroup = None) -> int:
    # [FUNCTION DOC]
    # Behavior: uses the given group or the global one; returns its world size, or 1 when no group.
    # Inputs: group - optional process group. Outputs: int.
    """
    Get ulysses sequence parallel world size.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_world_size(group) if group else 1


def get_ulysses_sequence_parallel_rank(group: ProcessGroup = None) -> int:
    # [FUNCTION DOC]
    # Behavior: uses the given group or the global one; returns this process's rank in it, or 0 when no
    # group. Inputs: group - optional process group. Outputs: int.
    """
    Get ulysses sequence parallel rank.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    return dist.get_rank(group) if group else 0


def gather_seq_scatter_heads(
    x: Tensor,
    seq_dim: int,
    head_dim: int,
    unpadded_dim_size: int = 0,
    group: ProcessGroup = None,
) -> Tensor:
    # [FUNCTION DOC]
    # Behavior: if no SP group, returns x unchanged. Otherwise performs a differentiable all-to-all that
    # splits along head_dim and concatenates along seq_dim (local seq shard to full seq, full heads to
    # head shard). If unpadded_dim_size is non-zero and not divisible by the SP world size, removes the
    # trailing (gathered seq length minus unpadded_dim_size) elements on seq_dim.
    # Inputs: x - tensor; seq_dim, head_dim - dims; unpadded_dim_size - true sequence length or 0;
    # group - optional process group (defaults to global).
    # Outputs: the transformed tensor.
    # Side effects: collective communication over the group.
    """
    A func to sync embedding input with alltoall in sequence parallel
    gather sequence dimension and scatter head dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq/n, h, ...] -> [bsz, seq, h/n, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    x = SeqAllToAll.apply(group, x, head_dim, seq_dim)
    if unpadded_dim_size and unpadded_dim_size % sp_world != 0:
        padding_size = x.size(seq_dim) - unpadded_dim_size
        x = _unpad_tensor(x, seq_dim, padding_size)
    return x


def gather_heads_scatter_seq(x: Tensor, head_dim: int, seq_dim: int, group: ProcessGroup = None) -> Tensor:
    # [FUNCTION DOC]
    # Behavior: inverse of gather_seq_scatter_heads. If no SP group, returns x. Otherwise zero-pads
    # seq_dim up to a multiple of the SP world size, then performs a synchronous differentiable
    # all-to-all splitting along seq_dim and concatenating along head_dim.
    # Inputs: x - tensor; head_dim, seq_dim - dims; group - optional process group.
    # Outputs: tensor with sequence sharded and heads gathered. Side effects: collective communication.
    """
    A func to sync attention result with alltoall in sequence parallel
    gather head dimension and scatter seq dim:
    e.g. seq_dim: 1, head_dim: 2
    [bsz, seq, h/n, ...] -> [bsz, seq/n, h, ...]
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if not group:
        return x
    dim_size = x.size(seq_dim)
    sp_world = get_ulysses_sequence_parallel_world_size(group)
    if dim_size % sp_world != 0:
        padding_size = sp_world - (dim_size % sp_world)
        x = _pad_tensor(x, seq_dim, padding_size)
    return SeqAllToAll.apply(group, x, seq_dim, head_dim, False)


def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # [FUNCTION DOC]
    # HELPER: Appends zeros of length padding_size along dim (same dtype/device) via concatenation.
    # Inputs: x, dim, padding_size. Outputs: new padded tensor.
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
    # [FUNCTION DOC]
    # HELPER: Removes the last padding_size elements along dim by slicing (returns a view).
    # Inputs: x, dim, padding_size (must be > 0; zero would yield an empty slice). Outputs: tensor view.
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(0, -padding_size)
    return x[tuple(slc)]


def slice_input_tensor(x: Tensor, dim: int, padding: bool = True, group: ProcessGroup = None) -> Tensor:
    # [FUNCTION DOC]
    # Behavior: optionally zero-pads x along dim to a multiple of the SP world size, then returns the
    # contiguous chunk for this rank (equal-size parts, rank-th part).
    # Inputs: x - tensor; dim - dim to slice; padding - whether to pad first; group - optional group.
    # Outputs: contiguous local slice.
    # Exceptions: dist.get_world_size fails if distributed is not initialized.
    # KNOWN ISSUE: the rank is always taken from the global SP group, ignoring the group argument, so a
    # custom group with different ranks would slice incorrectly.
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group)
    sp_rank = get_ulysses_sequence_parallel_rank()
    dim_size = x.size(dim)
    # pad before slice
    if padding and dim_size % sp_world_size:
        padding_size = sp_world_size - (dim_size % sp_world_size)
        x = _pad_tensor(x, dim, padding_size)
    # slice the input tensor
    parts = x.size(dim) // sp_world_size
    slc = [slice(None)] * len(x.shape)
    slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
    return x[tuple(slc)].contiguous()


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    # [FUNCTION DOC]
    # Behavior: splits local_input into world-size chunks along scatter_dim (made contiguous), runs
    # dist.all_to_all into same-shape empty buffers, and concatenates received chunks along gather_dim.
    # With async_op, returns a closure that waits on the work handle then returns the concatenation.
    # Inputs: local_input; scatter_dim; gather_dim; group (defaults to global SP group); async_op.
    # Outputs: contiguous tensor, or a zero-argument wait function when async_op.
    # Side effects: collective communication. Assumes scatter_dim divisible evenly by world size.
    group = get_ulysses_sequence_parallel_group() if group is None else group
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    comm = dist.all_to_all(output_list, input_list, group=group, async_op=async_op)
    if async_op:

        def wait():
            # [FUNCTION DOC]
            # HELPER: Waits on the async all-to-all handle, then returns the contiguous concatenation of the
            # received chunks along gather_dim.
            comm.wait()
            return torch.cat(output_list, dim=gather_dim).contiguous()

        return wait
    return torch.cat(output_list, dim=gather_dim).contiguous()


def all_gather_tensor(local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    # [FUNCTION DOC]
    # Behavior: all-gathers local_tensor into a new tensor whose first dimension is world-size times
    # larger (rank-ordered along dim 0).
    # Inputs: local_tensor; group (defaults to global SP group); async_op.
    # Outputs: the gathered output tensor.
    # Side effects: collective communication.
    # KNOWN ISSUE: with async_op True the work handle is discarded and the output is returned without
    # waiting, so it may be read before the gather completes.
    group = get_ulysses_sequence_parallel_group() if group is None else group
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


class SeqAllToAll(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd Function wrapping all_to_all_tensor; backward performs the reverse all-to-all (gather_dim
    # and scatter_dim swapped).
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
        async_op: bool = False,
    ) -> Tensor:
        # [FUNCTION DOC]
        # Behavior: saves group, scatter_dim, gather_dim, async_op on ctx and returns all_to_all_tensor
        # output. Inputs: ctx; group; local_input; scatter_dim; gather_dim; async_op.
        # Outputs: tensor (or wait closure when async_op, which is not a valid autograd output).
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group, async_op)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        # [FUNCTION DOC]
        # Behavior: in async mode concatenates grad_output entries after the first along gather_dim,
        # otherwise uses the first gradient; then runs a synchronous reverse all-to-all. Returns gradient
        # only for local_input, None for all others (one extra trailing None, tolerated by autograd).
        # Outputs: tuple of gradients. Side effects: collective communication.
        input_t = torch.cat(grad_output[1:], dim=ctx.gather_dim).contiguous() if ctx.async_op else grad_output[0]
        return (
            None,
            all_to_all_tensor(input_t, ctx.gather_dim, ctx.scatter_dim, ctx.group, False),
            None,
            None,
            None,
            None,
        )


class Gather(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd Function that all-gathers a tensor across the group and concatenates along gather_dim;
    # backward returns this rank's slice of the gradient, optionally scaled by world size.
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> Tensor:
        # [FUNCTION DOC]
        # Behavior: records group, gather_dim, grad_scaler, async_op, world size, rank and the local size on
        # gather_dim in ctx. All-gathers along dim 0, splits the result into per-rank blocks of the local
        # dim-0 size, and concatenates them along gather_dim.
        # Inputs: ctx; group; local_tensor; gather_dim; grad_scaler; async_op.
        # Outputs: gathered tensor. Side effects: collective communication.
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Any:
        # [FUNCTION DOC]
        # Behavior: multiplies grad by world size if grad_scaler, then returns this rank's contiguous chunk
        # (local part size along gather_dim) as the gradient for local_tensor; None for all other inputs.
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outpus_and_unpad(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Deprecated misspelled alias. Always raises RuntimeError directing callers to
    # gather_outputs_and_unpad. Inputs ignored.
    raise RuntimeError(
        "please use verl.utils.ulysses.gather_outputs_and_unpad instead of verl.utils.ulysses.gather_outpus_and_unpad"
    )


def gather_outputs_and_unpad(
    x: Tensor,
    gather_dim: int,
    unpad_dim: int = None,
    padding_size: int = 0,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    # [FUNCTION DOC]
    # Behavior: if no group (argument or global), returns x unchanged. Otherwise gathers x along
    # gather_dim with the differentiable Gather (grad_scaler controls backward scaling). If unpad_dim is
    # given and padding_size is non-zero, removes the trailing padding_size elements on unpad_dim.
    # Inputs: see docstring.
    # Outputs: gathered (and possibly unpadded) tensor.
    # Exceptions: AssertionError if unpad_dim given and padding_size is not an int.
    # Side effects: collective communication.
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    group = get_ulysses_sequence_parallel_group() if group is None else group
    if group is None:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    if unpad_dim is not None:
        assert isinstance(padding_size, int), "padding size is not given or is not an integer"
        if padding_size == 0:
            return x
        x = _unpad_tensor(x, unpad_dim, padding_size)
    return x


def ulysses_pad(
    input_ids_rmpad: torch.Tensor, position_ids_rmpad: Optional[torch.Tensor] = None, sp_size: int = 1, pad_value=0
):
    # [FUNCTION DOC]
    # Behavior: pads rmpad input ids (shape batch by total length) on the last dim with pad_value so the
    # length is divisible by sp_size. Position ids (shape with size 1 at dim -2; 2-D or 3-D for
    # multi-axis rope) are extended with positions 0 to pad_size-1 (repeated across the leading axis for
    # 3-D). If sp_size is at most 1, returns inputs unchanged with pad size 0.
    # Inputs: input_ids_rmpad; position_ids_rmpad - optional; sp_size; pad_value.
    # Outputs: tuple (input ids, position ids or None, pad_size).
    # Exceptions: AssertionError if position ids dim -2 is not 1 or lengths mismatch.
    if position_ids_rmpad is not None:
        assert position_ids_rmpad.size(-2) == 1
        assert input_ids_rmpad.size(-1) == position_ids_rmpad.size(-1)
    if sp_size <= 1:
        return input_ids_rmpad, position_ids_rmpad, 0
    _, total_seq_len = input_ids_rmpad.shape
    pad_size = (sp_size - total_seq_len % sp_size) % sp_size
    if pad_size > 0:
        input_ids_rmpad = torch.nn.functional.pad(input_ids_rmpad, (0, pad_size), value=pad_value)
        if position_ids_rmpad is not None:
            pad_pos_ids = torch.arange(pad_size, device=position_ids_rmpad.device).unsqueeze(0)
            if position_ids_rmpad.dim() == 3:
                pad_pos_ids = pad_pos_ids.unsqueeze(0).repeat(position_ids_rmpad.size(0), 1, 1)
            position_ids_rmpad = torch.cat((position_ids_rmpad, pad_pos_ids), dim=-1)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def ulysses_pad_and_slice_inputs(
    input_ids_rmpad: torch.Tensor,
    position_ids_rmpad: Optional[torch.Tensor] = None,
    sp_size: int = 1,
    skip_position_ids_rmpad: bool = False,
    pad_value=0,
):
    # [FUNCTION DOC]
    # Behavior: pads via ulysses_pad, then slices input ids along dim 1 to this rank's SP chunk (no extra
    # padding). Position ids are sliced too unless absent or skip_position_ids_rmpad is set (then they
    # are returned padded but full length).
    # Inputs/Outputs: see docstring. Exceptions: as ulysses_pad; requires initialized distributed.
    """
    Pad and slice input_ids to be divisible by sp_size
    Pad position_ids to be divisible by sp_size.

    Note both input_ids_rmpad and position_ids_rmpad will be padded and sliced.

    The is the utility of pre-forward for ulysses sequence parallelism

    Args:
        input_ids_rmpad: shape of [bsz, seqlen]
        position_ids_rmpad: shape of [bsz, seqlen], where bsz must be 1
        sp_size (int): ulysses sequence parallelism size
        skip_position_ids_rmpad: whether to skip position_ids_rmpad for VeOmniEngine

    Returns:
        torch.Tensor: padded and sliced input_ids
        torch.Tensor: padded and sliced position_ids
        int: pad size
    """
    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
        input_ids_rmpad, position_ids_rmpad, sp_size, pad_value=pad_value
    )
    input_ids_rmpad = slice_input_tensor(input_ids_rmpad, dim=1, padding=False)
    if position_ids_rmpad is not None and not skip_position_ids_rmpad:
        position_ids_rmpad = slice_input_tensor(position_ids_rmpad, dim=1, padding=False)
    return input_ids_rmpad, position_ids_rmpad, pad_size


def validate_ulysses_config(num_heads, ulysses_sequence_size):
    # [FUNCTION DOC]
    # Behavior: when ulysses_sequence_size > 1, asserts num_heads is divisible by it.
    # Inputs: num_heads; ulysses_sequence_size. Outputs: None. Exceptions: AssertionError.
    if ulysses_sequence_size > 1:
        assert num_heads % ulysses_sequence_size == 0, (
            f"num_heads ({num_heads}) must be divisible by ulysses sequence size({ulysses_sequence_size})"
        )


class BaseShardingManager:
    # [CLASS DOC]
    # Context-manager base for sharding managers: no-op enter/exit, identity data pre/post-processing,
    # and a timing dict for subclasses.
    """Base sharding manager used for resharding weights/data across parallel groups."""

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes an empty timing mapping. Side effects: sets attribute.
        self.timing = {}

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: no-op context entry; returns None.
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Behavior: no-op context exit; returns None so exceptions propagate.
        pass

    def preprocess_data(self, data: "DataProto") -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: identity; returns data unchanged.
        return data

    def postprocess_data(self, data: "DataProto") -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: identity; returns data unchanged.
        return data


class FSDPUlyssesShardingManager(BaseShardingManager):
    # [CLASS DOC]
    # Sharding manager for FSDP plus Ulysses: while active, sets the global SP group to the mesh's "sp"
    # sub-group; all-gathers data across the SP group before compute and re-splits it afterward.
    """
    Sharding manager to support data resharding when using FSDP + Ulysses sequence parallelism.
    """

    def __init__(self, device_mesh: DeviceMesh):
        # [FUNCTION DOC]
        # Behavior: stores device_mesh (may be None, meaning no SP) and a fixed seed_offset of 12345 (unused
        # here). Inputs: device_mesh - mesh with an "sp" dimension or None.
        super().__init__()
        self.device_mesh = device_mesh
        self.seed_offset = 12345

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: if a mesh is set, saves the current global SP group and installs the mesh "sp" group.
        # Side effects: mutates the module-global SP group.
        if self.device_mesh is not None:
            self.prev_sp_group = get_ulysses_sequence_parallel_group()
            set_ulysses_sequence_parallel_group(self.device_mesh["sp"].get_group())

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC]
        # Behavior: if a mesh is set, restores the previously saved SP group. Does not suppress exceptions.
        if self.device_mesh is not None:
            set_ulysses_sequence_parallel_group(self.prev_sp_group)

    def preprocess_data(self, data: "DataProto") -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: if a mesh is set, all-gathers the DataProto in place across the "sp" group. Returns data.
        # Side effects: collective communication; mutates data.
        """
        AllGather data from sp region.

        This is because the data is first sharded along the FSDP dimension as we utilize the DP_COMPUTE.
        In Ulysses, we need to make sure the same data is used across a SP group.
        """
        if self.device_mesh is not None:
            from verl.protocol import all_gather_data_proto

            group = self.device_mesh["sp"].get_group()
            all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: "DataProto") -> "DataProto":
        # [FUNCTION DOC]
        # Behavior: if a mesh is set, chunks the data into SP-size pieces and returns the piece at this
        # process's local SP rank; otherwise returns data unchanged.
        """
        Split the data to follow FSDP partition.
        """
        if self.device_mesh is not None:
            sp_size = self.device_mesh["sp"].size()
            sp_rank = self.device_mesh["sp"].get_local_rank()
            data = data.chunk(chunks=sp_size)[sp_rank]
        return data
