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
# Dispatch/collect machinery for the single controller. Defines the Dispatch and Execute dynamic enums (populated at import
# with predefined modes), dispatch functions (split driver-side arguments into one entry per worker rank) and collect
# functions (merge per-rank outputs), a global registry from Dispatch mode to its dispatch/collect pair, lazy N-D
# mesh-aware dispatch, and the register decorator that tags worker methods with dispatch/execute/blocking metadata stored
# under a magic attribute name (MAGIC_ATTR) that worker groups read when binding methods.
# Convention: a dispatch function takes (worker_group, *args, **kwargs) and returns (args, kwargs) where every arg/kwarg
# value is a list of length world_size; a collect function takes (worker_group, output_list).

import inspect
from functools import partial, wraps
from types import FunctionType

from verl.protocol import DataProtoFuture, _padding_size_key
from verl.utils.py_functional import DynamicEnum
from verl.utils.transferqueue_utils import tqbridge

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    # [CLASS DOC]
    # Dynamic enum (members added via register) of dispatch modes. Has its own registry and counter.
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    # [FUNCTION DOC]
    # Behavior: registers the predefined dispatch modes: RANK_ZERO, ONE_TO_ALL, ALL_TO_ALL, DP_COMPUTE, DP_COMPUTE_PROTO,
    # DP_COMPUTE_PROTO_WITH_FUNC, DP_COMPUTE_METRIC, DIRECT_ROLLOUT_METHOD. Side effects: mutates the Dispatch enum.
    # KNOWN ISSUE: RANK_ZERO is registered but has no entry in the dispatch function registry, so using it fails lookup.
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    # [CLASS DOC]
    # Dynamic enum of execution modes (which ranks run the call).
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    # [FUNCTION DOC]
    # Behavior: registers Execute modes ALL and RANK_ZERO. Side effects: mutates the Execute enum.
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Behavior: wrap each positional and keyword value in BatchData, assert it is chunkable, chunk it into exactly
    # chunks pieces.
    # Inputs: chunks - number of pieces; args/kwargs - chunkable batch objects.
    # Outputs: (list of chunk-lists per arg, mapping key -> chunk-list).
    # Exceptions: AssertionError if a value is not chunkable or chunking yields a different count.
    from verl.protocol import BatchData

    splitted_args = []
    for arg in args:
        assert BatchData(arg).is_chunkable(), f"arg of type {type(arg)} is not chunkable"
        chunked_arg = BatchData(arg).chunk(chunks=chunks)
        assert len(chunked_arg) == chunks
        splitted_args.append(chunked_arg)

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert BatchData(val).is_chunkable(), f"kwarg '{key}' of type {type(val)} is not chunkable"
        chunked_kwarg = BatchData(val).chunk(chunks=chunks)
        assert len(chunked_kwarg) == chunks
        splitted_kwargs[key] = chunked_kwarg

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Behavior: like the plain splitter but every value must be a DataProto or DataProtoFuture. For DataProto with
    # padding enabled, all such values must share the same length; the padding needed to reach a multiple of chunks is
    # computed once and applied in place before chunking. If any padding was computed, the padding size is added to the kwargs
    # under the protocol's padding-size key so the worker-side wrapper/collector can strip it.
    # Outputs: (list of chunk-lists, mapping of chunk-lists plus optional padding size entry).
    # Exceptions: AssertionError on wrong types or mismatched lengths. Side effects: pads input DataProto objects in place.
    from verl.protocol import DataProto, DataProtoFuture

    data_proto_len = None
    padding_size = None

    def _padding_and_split_data(obj, chunks):
        # [FUNCTION DOC]
        # HELPER: pads (when enabled) and chunks one object, tracking shared length/padding via enclosing-scope variables.
        nonlocal data_proto_len, padding_size
        assert isinstance(obj, DataProto | DataProtoFuture)
        if isinstance(obj, DataProto) and obj.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(obj)
                padding_size = (chunks - (data_proto_len % chunks)) if (data_proto_len % chunks > 0) else 0
            else:
                assert data_proto_len == len(obj), (
                    f"expecting all arg share same length of {data_proto_len}, but got {len(obj)}"
                )
            obj.padding(padding_size=padding_size)
        return obj.chunk(chunks=chunks)

    splitted_args = [_padding_and_split_data(arg, chunks) for arg in args]
    splitted_kwargs = {key: _padding_and_split_data(val, chunks) for key, val in kwargs.items()}
    if padding_size is not None:
        splitted_kwargs[_padding_size_key] = padding_size

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: replicate every arg and kwarg value world_size times. Outputs: (tuple of lists, mapping of lists).
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: always raises NotImplementedError; used as both dispatch and collect for DIRECT_ROLLOUT_METHOD, which must
    # not be invoked through the worker group dispatch path.
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: identity - args/kwargs are assumed already per-rank. Outputs: (args, kwargs) unchanged.
    return args, kwargs


def collect_all_to_all(worker_group, output):
    # [FUNCTION DOC]
    # Behavior: identity collect. Outputs: the list of outputs unchanged.
    return output


def _concat_data_proto_or_future(output: list):
    # [FUNCTION DOC]
    # HELPER: Behavior: assert all items share the exact same type, then concatenate via BatchData.
    # Outputs: a single concatenated object (DataProto, future, etc.).
    # Exceptions: AssertionError on mixed types.
    from verl.protocol import BatchData

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    return BatchData(output).concat()


def dispatch_dp_compute(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: validates that every arg and kwarg value is a list/tuple of length world_size and returns them unchanged.
    # Exceptions: AssertionError on violation (or if worker_group is not a WorkerGroup).
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    # [FUNCTION DOC]
    # Behavior: asserts worker group type and output length equals world_size; returns output list.
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: splits DataProto/future args into world_size chunks with auto padding.
    # Outputs: (args chunks, kwargs chunks possibly with padding size).
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: treats the first positional argument as a function to broadcast to all ranks; the remaining args and kwargs
    # are chunked into world_size pieces (no padding). Outputs: (function replicated list followed by chunk-lists, kwargs).
    # Exceptions: AssertionError if the first argument is not a plain function.
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    # [FUNCTION DOC]
    # Behavior: assert the outputs are concatable, validate length, then concatenate into one object.
    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_nd_compute(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: dispatch already-split per-DP-rank data onto all ranks of an N-D mesh. Puts each arg/kwarg list into the Ray
    # object store in parallel (thread count = min of the first arg length and CPU count, at least 1), then for each global
    # rank picks the element at that rank's DP index.
    # Inputs: dp_rank_mapping - list mapping global rank to DP rank; dp_size - number of DP groups; worker_group; args/kwargs -
    # lists of length dp_size.
    # Outputs: (tuple of per-rank lists of object refs, mapping of per-rank lists).
    # Exceptions: AssertionError on wrong length. KNOWN ISSUE: indexes args[0] for the worker count, so calls with only
    # keyword arguments raise IndexError.
    # Side effects: ray.put of all data.
    import os

    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)

    max_workers = max(1, min(len(args[0]), os.cpu_count()))

    args = [parallel_put(arg, max_workers=max_workers) for arg in args]
    kwargs = {k: parallel_put(v, max_workers=max_workers) for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = dp_rank_mapping[i]
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_nd_compute(collect_mask: list[bool], worker_group, output):
    # [FUNCTION DOC]
    # Behavior: keep outputs only from global ranks whose collect mask entry is true (one representative per DP rank).
    # Outputs: filtered list.
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        collect_dp_rank = collect_mask[global_rank]
        if collect_dp_rank:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_nd_compute_dataproto(dp_rank_mapping: list[int], dp_size, worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: chunk args/kwargs into dp_size pieces then dispatch via dispatch_nd_compute.
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return dispatch_nd_compute(dp_rank_mapping, dp_size, worker_group, *splitted_args, **splitted_kwargs)


def collect_nd_compute_dataproto(collect_mask: list[bool], worker_group, output):
    # [FUNCTION DOC]
    # Behavior: filter by collect mask, assert concatable, concatenate.
    output = collect_nd_compute(collect_mask, worker_group, output)

    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )
    return _concat_data_proto_or_future(output)


def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: on first use per mesh name, query and cache the worker group's per-rank DP rank list (asserting length);
    # derive dp_size as max rank + 1, then do N-D DataProto dispatch.
    # Side effects: populates worker_group._dispatch_info.
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # query dispatch info of the worker group
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dp_rank_mapping = worker_group._dispatch_info[mesh_name]
    # perform dispatch
    dp_size = max(dp_rank_mapping) + 1
    return dispatch_nd_compute_dataproto(dp_rank_mapping, dp_size, worker_group, *args, **kwargs)


def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    # [FUNCTION DOC]
    # Behavior: requires dispatch info for the mesh already cached; queries and caches the collect mask on first use; then
    # collects via the masked N-D collector. Positional args after worker_group are the output list.
    # Side effects: populates worker_group._collect_info.
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # the dispatch info is stored in the worker group
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    # a boolean of whether the dp_rank is used for collect
    collect_mask = worker_group._collect_info[mesh_name]
    # perform dispatch
    return collect_nd_compute_dataproto(collect_mask, worker_group, *args, **kwargs)


def make_nd_compute_dataproto_dispatch_fn(mesh_name):
    # [FUNCTION DOC]
    # Behavior: build a dispatch-mode dictionary (dispatch_fn, collect_fn) with the mesh name bound into the lazy N-D
    # functions. Outputs: mapping usable as a dispatch_mode in register.
    return {
        "dispatch_fn": partial(dispatch_lazy_compute_data_proto, mesh_name),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    # [FUNCTION DOC]
    # Outputs: the dispatch/collect mapping for a predefined Dispatch member. Exceptions: KeyError if not registered.
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    # [FUNCTION DOC]
    # Behavior: creates a new Dispatch member and registers its dispatch/collect functions.
    # Exceptions: AssertionError if the mode already exists in the registry. Side effects: mutates enum and registry.
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    # [FUNCTION DOC]
    # Behavior: replace the functions of an existing dispatch mode. Exceptions: AssertionError if missing.
    # Side effects: mutates registry.
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def get_predefined_execute_fn(execute_mode):
    # [FUNCTION DOC]
    # Inputs: execute_mode - Execute member. Outputs: mapping with execute_fn_name (execute_all or execute_rank_zero), the
    # worker-group method name to call. Exceptions: KeyError for other modes.
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    # [FUNCTION DOC]
    # HELPER: asserts dispatch_mode is a Dispatch member or a dict that has dispatch_fn and collect_fn keys.
    assert isinstance(dispatch_mode, Dispatch | dict), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    # [FUNCTION DOC]
    # HELPER: asserts execute_mode is an Execute member.
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Behavior: replace any DataProtoFuture in args/kwargs with its resolved value (via get).
    # Outputs: (tuple args, kwargs). Side effects: mutates the kwargs mapping in place; blocks on futures.
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    # [FUNCTION DOC]
    # Behavior: validates modes and returns a decorator. The decorator first wraps the function with the TransferQueue
    # bridge for the dispatch mode, then wraps it (sync or async variant matching the function) so futures are
    # materialized before the call when materialize_futures is set, and attaches a mapping of dispatch_mode, execute_mode
    # and blocking under the magic attribute.
    # Inputs: dispatch_mode - Dispatch member or dict; execute_mode - Execute member; blocking - whether the driver waits;
    # materialize_futures - resolve DataProtoFuture args first.
    # Outputs: decorator. Exceptions: AssertionError on invalid modes.
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.


    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """

    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: the decorator returned by register.
        func = tqbridge(dispatch_mode=dispatch_mode)(func)

        @wraps(func)
        def inner(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: sync wrapper: materialize futures if configured, then call.
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: async wrapper: materialize futures if configured, then await.
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
