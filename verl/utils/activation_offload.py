# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Activation (saved-for-backward tensor) CPU offloading for FSDP/FSDP2 models. Uses autograd saved-tensor hooks to
# route saved tensors through an offload handler; activations are grouped per transformer layer and offloaded
# asynchronously on a side stream with double buffering, then prefetched back during backward. Entry point:
# enable_activation_offloading, which wraps each FSDP layer's forward.

"""Functionality for CPU offloading of tensors saved for backward pass."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import FSDPModule as FSDP2

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_unique_tensor_key(tensor):
    # [FUNCTION DOC]
    # HELPER: Returns a key identifying a tensor's underlying memory: (storage data pointer plus storage offset, dtype).
    # Used to deduplicate views of the same memory when offloading. Inputs: tensor. Outputs: tuple.
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    return key


class FSDPParameterFilter:
    # [CLASS DOC]
    # HELPER: Callable filter that returns True for tensors NOT backed by model parameter storage, so parameters are not
    # offloaded. Holds a set of parameter storage data pointers.
    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: initializes an empty set of parameter storage pointers. Inputs/Outputs: none.
        self.model_parameters_storage = set()

    def __call__(self, tensor):
        # [FUNCTION DOC]
        # Behavior: returns True if the tensor's untyped storage pointer is not a known parameter storage.
        # Inputs: tensor. Outputs: bool.
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    def update_model_parameters(self, model):
        # [FUNCTION DOC]
        # Behavior: rebuilds the parameter storage pointer set from all parameters of the module (called each forward, since
        # FSDP may reallocate unsharded parameter storage). Inputs: model - module. Outputs: none.
        new_storage = set()
        for p in model.parameters():
            new_storage.add(p.data.untyped_storage().data_ptr())
        self.model_parameters_storage = new_storage


class CpuOffloadHookWithOffloadHandler:
    # [CLASS DOC]
    # Context manager installing autograd default saved-tensor hooks: pack forwards tensors to handler.tensor_push and
    # unpack calls handler.tensor_pop, both with handler_extra_kwargs.
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores handler and extra kwargs (empty dict default) and inside_context False.
        # Inputs: offload_handler; handler_extra_kwargs - optional dict. Outputs: none.
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: dict[str, Any] = handler_extra_kwargs
        self.inside_context = False

    def __enter__(self):
        # [FUNCTION DOC]
        # Behavior: marks inside_context and pushes the save/get hooks onto the autograd default hook stack. Returns None.
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(self.on_save_for_backward, self.on_get_saved_tensor)

    def __exit__(self, *args: Any):
        # [FUNCTION DOC]
        # Behavior: clears inside_context and pops the autograd default hooks. Inputs: ignored exception info.
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        # [FUNCTION DOC]
        # Behavior: pack hook; returns the identifier produced by handler.tensor_push. Inputs: tensor. Outputs: identifier.
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: unpack hook; returns the tensor from handler.tensor_pop for the saved identifier.
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class OffloadHandler:
    # [CLASS DOC]
    # Abstract offload handler interface with tensor_push/tensor_pop.
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Behavior: no-op constructor.
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        # [FUNCTION DOC]
        # Behavior: abstract; always raises NotImplementedError. Inputs: tensor, kwargs.
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        # [FUNCTION DOC]
        # Behavior: abstract; always raises NotImplementedError. Inputs: tensor_tag, kwargs.
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    # [CLASS DOC]
    # Identity autograd function marking a layer-group boundary: forward calls handler.on_group_commit_forward, backward
    # calls handler.on_group_commit_backward. Exposed as group_prefetch_offload_commit.
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        # [FUNCTION DOC]
        # Behavior: notifies handler of forward group commit, stores handler on ctx, returns the input tensor unchanged.
        # Inputs: ctx, tensor, cpu_offload_handler. Outputs: same tensor.
        # pylint: disable=missing-function-docstring
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC]
        # Behavior: notifies handler of backward group commit, passes gradient through. Outputs: (grad_output, None).
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    # [CLASS DOC]
    # Synchronous handler: offloads tensors of the first num_offload_group groups on the compute stream at push time and
    # reloads them at pop. Tags are (group index, index within group); state map from tag to either a (device, cpu copy)
    # tuple or the original tensor.
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(self, num_offload_group, tensor_need_offloading_checker=(lambda _: True)) -> None:
        # [FUNCTION DOC]
        # Behavior: stores group count and offload checker (default accept all), resets counters.
        # Inputs: num_offload_group - int; tensor_need_offloading_checker - predicate on tensor. Outputs: none.
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker

        self.groupid_reset()

    def groupid_reset(self):
        # [FUNCTION DOC]
        # Behavior: resets current group and per-group tensor count to 0, torch_tensor_count 0, and empties the tag map.
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        # [FUNCTION DOC]
        # Behavior: advances to the next group and resets the per-group tensor count.
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        # [FUNCTION DOC]
        # Behavior: decrements current group. Exceptions: AssertionError if it goes negative.
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        # [FUNCTION DOC]
        # Behavior: allocates a CPU tensor (same size/dtype/layout, pinned by default) and copies the source into it
        # non-blocking. Inputs: src_tensor; pin_memory - bool. Outputs: (source device, cpu copy) tuple.
        """Offload."""

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=True)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        # [FUNCTION DOC]
        # Behavior: copies the CPU backup back to its original device; non_blocking defaults to whether the backup is pinned.
        # Inputs: state tuple; non_blocking - optional bool. Outputs: tensor on device.
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        # [FUNCTION DOC]
        # Behavior: assigns a new tag; if the current group is below num_offload_group and the checker accepts, stores an
        # offloaded state, else stores the tensor itself. Inputs: tensor. Outputs: tag tuple.
        # Exceptions: AssertionError on duplicate tag.
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        # [FUNCTION DOC]
        # Behavior: removes the tag's state; reloads if it is an offload tuple, else returns the stored tensor.
        # Exceptions: AssertionError if tag unknown.
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    # [CLASS DOC]
    # Asynchronous double-buffer handler: tensors are kept referenced at push; at group commits whole groups are bulk
    # offloaded on a D2H stream (deduplicated by storage key) and later bulk reloaded on an H2D stream in backward.
    # layer_window_map maps offload group index to the layer index at which that group's offload must be complete,
    # spreading num_layers evenly over num_offload_group groups.
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: initializes base state, stores number of model layers, buffers, offloaded-group counter, the layer window
        # map (for each group i: (num_layers // num_offload_group) * (i+1) - 1 plus a carry so remainder layers are spread
        # over the first groups), and creates D2H and H2D device streams.
        # Inputs: num_offload_group (must be <= number of commits); num_model_group - number of layers; checker predicate.
        # Exceptions: ZeroDivisionError if num_offload_group is 0.
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
        )
        # Number of layers in the model
        self.num_layers = num_model_group
        # Data Structure to maintain reference to activation tensors
        self.tensor_tag_to_buf = {}
        # Tracking the number of layers offloaded
        self.offloaded_group_count = 0
        # Core data structure that decides the window for offloading
        self.layer_window_map = {}
        self.group_offload_mapping = {}

        # Logic to make offloading load balance across computation
        # for optimal CPU/GPU interconnect usage
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        # allocate streams and events for synchronization
        self.d2h_stream = get_torch_device().Stream()
        self.h2d_stream = get_torch_device().Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        # [FUNCTION DOC]
        # Behavior: fake/functional tensors and tensors rejected by the checker are not offloaded and are returned as their own
        # identifier. Otherwise assigns a (group, index) tag, stores the tensor in the state map and, for offloadable groups,
        # in the buffer map (keeps a reference until freed). Inputs: tensor. Outputs: tag tuple or the tensor.
        torch_stray_tensor = isinstance(
            tensor,
            torch._subclasses.fake_tensor.FakeTensor | torch._subclasses.functional_tensor.FunctionalTensor,
        )
        need_offload = not torch_stray_tensor
        need_offload = need_offload and self.tensor_need_offloading_checker(tensor)

        if need_offload:
            # obtain a unique tensor tag
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1

            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor

            if self.current_group < self.num_offload_group:
                self.tensor_tag_to_buf[tensor_tag] = tensor
        else:
            tensor_tag = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        # [FUNCTION DOC]
        # Behavior: if the identifier is a tensor, returns it. Otherwise pops the state (which must already have been reloaded
        # to a tensor) and drops any buffer entry. Exceptions: AssertionError if tag unknown or state still offloaded.
        """Tensor pop."""
        if isinstance(tensor_tag, torch.Tensor):
            return tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        # [FUNCTION DOC]
        # Behavior: on the D2H stream, for every tagged tensor in the given group, deduplicates by storage key, replaces its
        # state with (key, shape), then offloads each unique tensor to CPU and records the key-to-state mapping for the group.
        # Inputs: group_to_offload - int. Outputs: none. Side effects: async device-to-host copies.
        """Bulk offload group."""
        offload_mapping = {}
        offload_size = 0
        with get_torch_device().stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    key = _get_unique_tensor_key(state)
                    if key not in offload_mapping:
                        offload_mapping[key] = state
                    # if offload, return the reference to cpu copy
                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)
            for key, tensor in offload_mapping.items():
                state = SynchronizedGroupOffloadHandler.offload(tensor)
                offload_size += tensor.numel() * tensor.element_size()
                offload_mapping[key] = state

            self.group_offload_mapping[group_to_offload] = offload_mapping

    def synchronize_on_group_commit_forward(self, current_group):
        # [FUNCTION DOC]
        # Behavior: at group 0, makes D2H wait on compute and starts offloading group 0. When current group equals the window
        # layer for the next pending offload group, syncs streams both ways, drops buffer references for that group (allowing
        # GPU memory to be freed), starts offloading the following group if any, and increments offloaded_group_count.
        # Inputs: current_group - int. Outputs: none.
        """Synchronize on group commit forward."""

        # For the first group, kickstart the offload after we have
        # the first compute completion
        if current_group == 0:
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            self.bulk_offload_group(current_group)

        # Window map data structure helps us synchronize based on number
        # of layers offloaded
        if self.layer_window_map[self.offloaded_group_count] == current_group:
            # Stream synchronization both ways
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.d2h_stream)

            # Time to free the activation memory after usage
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            # Time to offload the next group
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            # Increment the offload group count to keep track
            self.offloaded_group_count += 1

    def on_group_commit_forward(self):
        # [FUNCTION DOC]
        # Behavior: runs forward synchronization for the current group then advances the group via the base method.
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        super().on_group_commit_forward()

    @torch.no_grad
    def bulk_reload_group(self, group_to_reload):
        # [FUNCTION DOC]
        # Behavior: on the H2D stream, pops the group's offload mapping, reloads each unique CPU copy to device, and replaces
        # each tagged (key, shape) state in the group with a view of the reloaded tensor of that shape.
        # Inputs: group_to_reload - int. Exceptions: AssertionError if group index out of range or mapping missing.
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group

        with get_torch_device().stream(self.h2d_stream):
            # move back tensors
            offload_mapping = self.group_offload_mapping.pop(group_to_reload)
            assert offload_mapping is not None
            for key, state in offload_mapping.items():
                offload_mapping[key] = SynchronizedGroupOffloadHandler.reload(state)
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload and not isinstance(state, torch.Tensor):
                    assert isinstance(state, tuple), f"{group_id} {state}"
                    key, shape = state
                    recovered_tensor = offload_mapping[key].view(shape)
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        # [FUNCTION DOC]
        # Behavior: decrements current group (asserting non-negative); if it equals the window layer of the last offloaded
        # group, syncs streams both ways, reloads that group, and decrements offloaded_group_count (not below 1). At group 0,
        # compute waits for all reloads and the counter resets to 0. Inputs/Outputs: none.
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0

        # Layer window data structure helps us to reload at right times
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            # Stream synchronization both ways
            self.h2d_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.h2d_stream)

            # Time to reload the next group
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrease the offloading group counter
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # Last group computation needs to wait till all the reloads complete
        if self.current_group == 0:
            get_torch_device().current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


def get_activation_offload_context(
    num_layers: int = 1, model_layers: int = 1, tensor_need_offloading_checker=(lambda t: True)
):
    # [FUNCTION DOC]
    # Behavior: builds an AsyncDoubleBufferGroupOffloadHandler and returns (saved-tensor hook context manager, a function
    # that applies the group commit identity op with that handler to a tensor).
    # Inputs: num_layers - offload group count; model_layers - total layer count; checker predicate. Outputs: tuple.
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        # [FUNCTION DOC]
        # HELPER: Applies the group commit function to the tensor with the captured handler. Outputs: same tensor.
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )


class ActivationHandler:
    # [CLASS DOC]
    # Coordinates per-layer forward wrapping: enters the offload hook context during training forward, optionally runs
    # the layer under reentrant torch checkpointing, and passes the first output through the group-commit sync function.
    def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):
        # [FUNCTION DOC]
        # Behavior: stores context, sync function, tensor filter and ckpt flag; if ckpt enabled, prepares a reentrant
        # checkpoint function. Inputs: offload_ctx, sync_func, tensor_filter, enable_ckpt. Outputs: none.
        self._offload_ctx = offload_ctx
        self._sync_func = sync_func
        self._enable_ckpt = enable_ckpt
        self._tensor_filter = tensor_filter
        if enable_ckpt:
            self.checkpoint_fn = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=True,
            )

    def pre_forward(self, module):
        # [FUNCTION DOC]
        # Behavior: if module is training, enters the offload context and refreshes the parameter filter from the module.
        if module.training:
            self._offload_ctx.__enter__()
            self._tensor_filter.update_model_parameters(module)

    def post_forward(self, module):
        # [FUNCTION DOC]
        # Behavior: if module is training, exits the offload context.
        if module.training:
            self._offload_ctx.__exit__(None, None, None)

    def _pack_kwargs(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Flattens positional and keyword args into one tuple plus a tuple of keyword names (values appended in order).
        # Outputs: (flat_args, kwarg_keys).
        kwarg_keys = []
        flat_args = list(args)
        for k, v in kwargs.items():
            kwarg_keys.append(k)
            flat_args.append(v)

        return tuple(flat_args), tuple(kwarg_keys)

    def _unpack_kwargs(self, flat_args, kwarg_keys):
        # [FUNCTION DOC]
        # HELPER: Inverse of _pack_kwargs: last len(kwarg_keys) entries become kwargs. Outputs: (args, kwargs dict).
        # Exceptions: AssertionError if more keys than args.
        assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
        if len(kwarg_keys) == 0:
            return flat_args, {}
        args = flat_args[: -len(kwarg_keys)]
        kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :], strict=True))
        return args, kwargs

    def _ckpt_forward(self, forward_method, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Runs forward_method under the checkpoint function, packing kwargs into positional inputs (reentrant
        # checkpoint only accepts positional args) and unpacking inside. Outputs: forward result.
        flat_args, kwarg_keys = self._pack_kwargs(*args, **kwargs)

        def my_function(*inputs):
            # [FUNCTION DOC]
            # HELPER: Unpacks inputs to args/kwargs and calls the original forward method.
            # unpack back into args and kwargs
            nonlocal forward_method, kwarg_keys
            unpacked_args, unpacked_kwargs = self._unpack_kwargs(inputs, kwarg_keys)
            # run original module
            return forward_method(*unpacked_args, **unpacked_kwargs)

        return self.checkpoint_fn(
            my_function,
            *flat_args,
        )

    def forward(self, module, forward_method, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: in eval mode calls forward directly. In training, calls forward (checkpointed if enabled), then passes the
        # output (or first element of a tuple output) through sync_func to mark the group boundary, re-wrapping tuples.
        # Inputs: module, forward_method, args, kwargs. Outputs: forward result.
        if not module.training:
            return forward_method(*args, **kwargs)
        if not self._enable_ckpt:
            ret = forward_method(*args, **kwargs)
        else:
            ret = self._ckpt_forward(forward_method, *args, **kwargs)
        binded_tensor = ret
        if isinstance(ret, tuple):
            binded_tensor = ret[0]
        binded_tensor = self._sync_func(binded_tensor)
        final_ret = binded_tensor
        if isinstance(ret, tuple):
            final_ret = (final_ret,) + ret[1:]
        return final_ret

    def wrap_module_forward_method(self, module):
        # [FUNCTION DOC]
        # Behavior: replaces module.forward with a bound wrapper calling pre_forward, handler.forward with the original bound
        # method, and post_forward. Inputs: module. Outputs: none. Side effects: mutates module.forward.
        orig_method = module.forward
        handler = self

        @functools.wraps(orig_method)
        def wrapped_method(model_self, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Wrapper forward: pre-hook, handled forward, post-hook; returns output.
            nonlocal handler
            handler.pre_forward(model_self)
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            handler.post_forward(model_self)
            return out

        module.forward = wrapped_method.__get__(module, type(module))


def enable_activation_offloading(model, strategy, enable_ckpt=False):
    # [FUNCTION DOC]
    # Behavior: collects FSDP/FSDP2-wrapped children recursively (not descending into wrapped ones), skipping those that
    # wrap only an Embedding. With fewer than 3 layers logs a warning and returns. Otherwise creates a parameter filter
    # and an offload context with (layers - 1) offload groups (last layer not offloaded) over all layers; if enable_ckpt,
    # disables HF gradient checkpointing on all modules (incompatible) since the handler provides its own; wraps every
    # layer's forward (unwrapping FSDP1 to its inner module).
    # Inputs: model; strategy - "fsdp" or "fsdp2"; enable_ckpt - bool. Outputs: none.
    # Exceptions: AssertionError for other strategies. Side effects: mutates module forwards, logs.
    """
    Enable activation offloading for the model. It groups activations by TransformerLayer and offloads activation
    groups asynchronously. This means that the offloading of the i-th activation group and the computation of the i+1-th
    activation group happen at the same time, and there are at most two activation groups in GPU memory.

    Args:
        model: the model to enable activation offloading
        strategy: the training strategy of the model, such as "fsdp"
        enable_ckpt: whether activation checkpointing(also called gradient checkpointing) has been enabled for the model

    Note:
        For best efficiency, activation offloading is usually combined with activation checkpointing. However, this
        implementation of activation offloading is conflicted with the implementation of activation checkpointing in
        some training strategies. This function resolves this conflict, and therefore requires the "strategy" and
        "enable_ckpt" arguments.

    Returns:

    """

    assert strategy == "fsdp" or strategy == "fsdp2", "activation offloading only supports fsdp strategy"
    layers = []

    def get_layers(module):
        # [FUNCTION DOC]
        # HELPER: Recursive traversal appending FSDP-wrapped non-embedding children to the enclosing layers list.
        for name, child in module.named_children():
            if not isinstance(child, FSDP | FSDP2):
                get_layers(child)
            else:
                wrapped_module = child
                if isinstance(child, FSDP):
                    wrapped_module = child._fsdp_wrapped_module
                # In some cases, torch.nn.Embedding is wrapped with FSDP alone. However, the activation
                # size of torch.nn.Embedding is small, so it's not necessary to offload it.
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    layers.append(child)

    get_layers(model)
    if len(layers) < 3:
        logger.warning(f"Find only {len(layers)} fsdp layers, not necessary to enable async activation offloading")
        return

    tensor_filter = FSDPParameterFilter()
    context, sync_func = get_activation_offload_context(len(layers) - 1, len(layers), tensor_filter)
    if enable_ckpt:
        # The implementation of activation checkpointing in transformers library is incompatible with
        # activation offloading,
        # so it will be disabled, but this implementation supports another version of activation checkpointing, so that
        # these two features can be enabled at the same time.
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)
    for layer in layers:
        module = layer
        if isinstance(layer, FSDP):
            module = module._fsdp_wrapped_module
        handler.wrap_module_forward_method(module)
