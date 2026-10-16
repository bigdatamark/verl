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
# TransferQueue (TQ) bridge utilities. Provides the tqbridge decorator that transparently replaces BatchMeta /
# KVBatchMeta arguments with the real TensorDict data fetched from TransferQueue, and writes TensorDict outputs back to
# TQ returning updated metadata. When transfer_queue is not installed, placeholder BatchMeta/KVBatchMeta classes and a
# mock tq module (raising RuntimeError on any call) are defined. Sync callers run TQ coroutines on a shared daemon
# event-loop thread (lazily created, shut down at interpreter exit). Module state: TQ_INITIALIZED flag (tq.init called
# once per process), the bridge loop/thread and a lock; logger level from VERL_LOGGING_LEVEL (default WARN).

import asyncio
import atexit
import copy
import functools
import inspect
import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import torch
from tensordict.tensorclass import NonTensorData, NonTensorStack

if TYPE_CHECKING:
    from verl.single_controller.base.decorator import Dispatch

from tensordict import TensorDict

try:
    import transfer_queue as tq
    from transfer_queue import (
        BatchMeta,
        KVBatchMeta,
    )

except ImportError:

    class BatchMeta:
        # [CLASS DOC]
        # HELPER: Placeholder used for isinstance checks when transfer_queue is not installed.
        pass

    class KVBatchMeta:
        # [CLASS DOC]
        # HELPER: Placeholder used for isinstance checks when transfer_queue is not installed.
        pass

    # Mock transfer_queue module when not installed
    class _MockTQ:
        # [CLASS DOC]
        # HELPER: Stand-in for the transfer_queue module: any attribute access returns a function that raises RuntimeError
        # explaining that transfer_queue must be installed.
        """Mock transfer_queue module that raises RuntimeError on any access."""

        def __getattr__(self, name: str) -> Any:
            # [FUNCTION DOC]
            # Returns a callable that raises RuntimeError naming the requested attribute when invoked.
            def _raise(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Raises RuntimeError with install instructions, regardless of arguments.
                raise RuntimeError(
                    f"transfer_queue is not installed. Cannot use tq.{name}(). "
                    "Please install it by calling `pip install TransferQueue==0.1.8`"
                )

            return _raise

    tq = _MockTQ()


from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TQ_INITIALIZED = False
_ASYNC_BRIDGE_LOOP: asyncio.AbstractEventLoop | None = None
_ASYNC_BRIDGE_THREAD: threading.Thread | None = None
_ASYNC_BRIDGE_LOCK = threading.Lock()


def _run_event_loop_forever(loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
    # [FUNCTION DOC]
    # HELPER: Thread target: sets loop as the thread's event loop, signals ready, and runs the loop until stopped.
    asyncio.set_event_loop(loop)
    ready.set()
    loop.run_forever()


def _shutdown_async_bridge_runtime(
    loop: asyncio.AbstractEventLoop | None = None,
    thread: threading.Thread | None = None,
) -> None:
    # [FUNCTION DOC]
    # HELPER: Stops and closes an event loop runtime. With no arguments, atomically takes and clears the global bridge
    # loop/thread. If the thread is alive, schedules loop stop thread-safely (ignoring RuntimeError for a closed loop) and
    # joins it; then closes the loop if not already closed. Registered with atexit.
    global _ASYNC_BRIDGE_LOOP, _ASYNC_BRIDGE_THREAD

    # No explicit runtime provided: shutdown the shared global runtime.
    if loop is None and thread is None:
        with _ASYNC_BRIDGE_LOCK:
            loop = _ASYNC_BRIDGE_LOOP
            thread = _ASYNC_BRIDGE_THREAD
            _ASYNC_BRIDGE_LOOP = None
            _ASYNC_BRIDGE_THREAD = None

    if thread is not None and thread.is_alive():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            # The loop may already be closed if shutdown races with process teardown.
            pass
        thread.join()

    if loop is not None and not loop.is_closed():
        loop.close()


def _get_async_bridge_loop() -> asyncio.AbstractEventLoop:
    # [FUNCTION DOC]
    # HELPER: Returns the shared bridge event loop, creating it under the lock on first use: a new loop run forever in a
    # daemon thread; waits up to 5 seconds for it to start.
    # Exceptions: RuntimeError (after cleanup) if the thread does not signal readiness in time. Side effects: starts a thread,
    # sets module globals.
    global _ASYNC_BRIDGE_LOOP, _ASYNC_BRIDGE_THREAD
    with _ASYNC_BRIDGE_LOCK:
        loop = _ASYNC_BRIDGE_LOOP
        if loop is not None:
            return loop

        # lazy init the event loop & thread
        ready = threading.Event()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_run_event_loop_forever,
            args=(loop, ready),
            name="batchmeta tensordict converter",
            daemon=True,
        )
        thread.start()
        if not ready.wait(timeout=5):
            _shutdown_async_bridge_runtime(loop, thread)
            raise RuntimeError("Failed to start TransferQueue async bridge event loop.")
        _ASYNC_BRIDGE_LOOP = loop
        _ASYNC_BRIDGE_THREAD = thread
        return loop


atexit.register(_shutdown_async_bridge_runtime)


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    # [FUNCTION DOC]
    # HELPER: Runs async_func(args) on the shared bridge loop and blocks until its result, closing the coroutine if
    # scheduling fails. Exceptions: re-raises scheduling RuntimeError and any exception from the coroutine.
    # KNOWN ISSUE: called from inside a coroutine running on the bridge loop itself it would deadlock.
    # Use a shared event loop in a background thread because event
    # loop may already exist in server mode.
    loop = _get_async_bridge_loop()
    coroutine = async_func(*args, **kwargs)
    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    except RuntimeError:
        coroutine.close()
        raise
    return future.result()


def _find_meta(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Returns the first positional, then keyword, argument that is a BatchMeta or KVBatchMeta, else None.
    for arg in args:
        if isinstance(arg, BatchMeta | KVBatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta | KVBatchMeta):
            return v
    return None


async def _async_meta_to_realdata(meta: BatchMeta | KVBatchMeta) -> TensorDict:
    # [FUNCTION DOC]
    # HELPER: Fetches real data for a metadata object: converts KVBatchMeta to BatchMeta first; deep-copies extra_info;
    # for size 0 returns an empty TensorDict carrying extra_info as non-tensor entries; otherwise gets the data from the TQ
    # client and attaches each extra_info entry (NonTensorData/Stack directly, others wrapped as non-tensor data).
    # Outputs: TensorDict. Side effects: network I/O.
    if isinstance(meta, KVBatchMeta):
        meta = await async_kv_batch_meta2batch_meta(meta)
    meta_info = copy.deepcopy(meta.extra_info)
    if meta.size == 0:
        empty_td = TensorDict({}, batch_size=(0,))
        tu.assign_non_tensor(empty_td, **meta_info)
        return empty_td

    tq_client = tq.get_client()
    tensordict = await tq_client.async_get_data(meta)

    for key, val in meta_info.items():
        if isinstance(val, (NonTensorData | NonTensorStack)):
            tensordict[key] = val
        else:
            tu.assign_non_tensor_data(tensor_dict=tensordict, key=key, val=val)
    return tensordict


def _meta_to_realdata(meta: BatchMeta) -> TensorDict:
    # [FUNCTION DOC]
    # HELPER: Synchronous wrapper running the async fetch on the bridge loop.
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


async def _async_update_meta_with_output(output: TensorDict, meta: BatchMeta, func_name=None) -> BatchMeta:
    # [FUNCTION DOC]
    # HELPER: Writes a function's TensorDict output to TQ. Tensor and NonTensorStack entries become stored fields;
    # NonTensorData values are collected as metadata. If any fields exist, puts the selected fields with the given meta,
    # logs timing and sets the returned meta's extra_info to the collected metadata. Returns the (possibly new) meta.
    # Exceptions: ValueError for entries of other types.
    # KNOWN ISSUE: when there are no tensor fields, the collected non-tensor metadata is dropped and the original meta is returned unchanged.
    fields, meta_data = [], {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor | NonTensorStack):
            fields.append(k)
        elif isinstance(v, NonTensorData):
            meta_data[k] = v.data
        else:
            raise ValueError(f"Unsupported type {type(v)} for key {k} in output TensorDict.")

    if fields:
        t1 = time.time()
        tq_client = tq.get_client()
        meta = await tq_client.async_put(data=output.select(*fields), metadata=meta)
        t2 = time.time()

        logger.info(f"Task {func_name} (pid={os.getpid()}) is writing to TransferQueue, cost time: {t2 - t1}s)")
        meta.extra_info = meta_data
    return meta


def _update_meta_with_output(output: TensorDict, meta: BatchMeta, func_name=None) -> BatchMeta:
    # [FUNCTION DOC]
    # HELPER: Synchronous wrapper running the async output write on the bridge loop.
    updated_meta = _run_async_in_temp_loop(_async_update_meta_with_output, output, meta, func_name)
    return updated_meta


def _compute_need_collect(dispatch_mode: "dict | Dispatch", args: list) -> bool:
    # [FUNCTION DOC]
    # HELPER: Decides whether this worker should collect (write back) the output. True for None or Dispatch enum modes.
    # For dict modes (must contain collect_fn, else AssertionError): only when collect_fn is a partial of
    # collect_lazy_compute_data_proto with a mesh name argument and the first arg is a Worker, returns that worker's
    # query_collect_info for the mesh; otherwise True.
    """Compute whether data collection is needed for the current worker.

    This function determines whether the current worker should collect data based on
    the dispatch mode configuration and worker parameters. It's used to optimize
    distributed data collection by ensuring only the appropriate rank collects data.

    Args:
        dispatch_mode: Controls data collection logic for the current worker. Can be None,
                      a Dispatch instance, or a dict with 'collect_fn' key. If None or Dispatch,
                      always returns True (current worker should collect). If dict, checks
                      collect_fn for lazy compute optimization.
        args: List of arguments passed to the function. Should contain a Worker instance
             as the first argument when using lazy compute mode.

    Returns:
        bool: True if data collection is needed, False otherwise.

    Note:
        Only checks worker attributes when dispatch_mode is a dict with 'collect_fn',
        the collect_fn is 'collect_lazy_compute_data_proto', and args[0] is a Worker.
        Otherwise, returns True. For the lazy compute case, checks the worker's
        data parallel rank for the mesh specified in collect_fn.args[0] to determine
        if this worker should collect data.
    """
    from verl.single_controller.base.decorator import Dispatch
    from verl.single_controller.base.worker import Worker

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    assert "collect_fn" in dispatch_mode.keys(), "collect_fn should be in dispatch_mode."

    collect_fn = dispatch_mode["collect_fn"]

    # Check if collect_fn is a functools.partial and handle gracefully
    if isinstance(collect_fn, functools.partial):
        collect_fn_name = collect_fn.func.__name__
        if collect_fn_name != "collect_lazy_compute_data_proto" or len(args) < 1 or not isinstance(args[0], Worker):
            return True

        collect_mesh_name = collect_fn.args[0] if collect_fn.args else None
        if collect_mesh_name is None:
            return True

        return args[0].query_collect_info(collect_mesh_name)
    else:
        # If collect_fn is not a partial, we can't extract mesh_name information
        # Fall back to default behavior (collect data)
        return True


def _postprocess_common(output, put_data, need_collect):
    # [FUNCTION DOC]
    # HELPER: Normalizes outputs when not writing back: if output should have been put but this rank does not collect,
    # returns an empty BatchMeta; if not put and not collected, returns an empty DataProto or empty TensorDict for those
    # types; else returns output unchanged.
    # KNOWN ISSUE: docstring mentions BatchMeta.empty() but the code constructs BatchMeta() directly.
    """Common post-processing logic for function outputs in TransferQueue bridge.

    This function handles the final return value based on whether data should be
    put into storage (put_data) and whether collection is needed (need_collect).
    It ensures proper return types based on the execution context.

    Args:
        output: The original output from the decorated function. Can be any type.
        put_data: bool, indicating whether the output should be put into TransferQueue.
                 If True, output will be put to TQ and return the corresponding BatchMeta;
                 if False, output will not be put into TQ.
        need_collect: bool, indicating whether this process needs to collect data.
                     If False, the output will be replaced by an empty BatchMeta or DataProto
                     to avoid redundant communication.

    Returns:
        - BatchMeta.empty(): When put_data=True but need_collect=False, indicating
          no data should be stored but BatchMeta structure is expected.
        - DataProto(): When put_data=False, need_collect=False, and output is DataProto,
          returning an empty DataProto.
        - output: In all other cases, returns the original output unchanged.

    Note:
        This function is used in the tqbridge decorator to normalize return values
        across different execution paths and avoid redundant data operations in
        distributed scenarios.
    """
    from verl.protocol import DataProto

    if put_data and not need_collect:
        return BatchMeta()
    elif not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    elif not put_data and not need_collect and isinstance(output, TensorDict):
        return TensorDict({}, batch_size=(0,))
    else:
        return output


async def async_kv_batch_meta2batch_meta(meta: KVBatchMeta) -> BatchMeta:
    # [FUNCTION DOC]
    # Converts a KVBatchMeta to BatchMeta: initializes tq once, retrieves metadata for its keys in its partition (without
    # creating), selects requested fields (string becomes a single-item list) and copies extra_info.
    # Side effects: tq init, network I/O, sets TQ_INITIALIZED.
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=meta.keys, partition_id=meta.partition_id, create=False)
    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    batch_meta.extra_info = meta.extra_info
    return batch_meta


def kv_batch_meta2batch_meta(meta: KVBatchMeta):
    # [FUNCTION DOC]
    # Synchronous wrapper of the KV-to-BatchMeta conversion on the bridge loop.
    return _run_async_in_temp_loop(async_kv_batch_meta2batch_meta, meta)


async def async_batch_meta2kv_batch_meta(meta: BatchMeta) -> KVBatchMeta:
    # [FUNCTION DOC]
    # Converts a BatchMeta to KVBatchMeta: initializes tq once, requires all partition ids to be equal (AssertionError
    # otherwise), retrieves keys for the global indexes, and builds a KVBatchMeta with those keys, empty tag dicts per sample,
    # the partition id, field names and extra_info.
    # KNOWN ISSUE: the tags list repeats the same dict object for every sample.
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]
    assert all([partition_id == pid for pid in meta.partition_ids])
    keys = await tq_client.async_kv_retrieve_keys(global_indexes=meta.global_indexes, partition_id=partition_id)

    kv_batch_meta = KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )
    return kv_batch_meta


def batch_meta2kv_batch_meta(meta: BatchMeta):
    # [FUNCTION DOC]
    # Synchronous wrapper of the BatchMeta-to-KV conversion on the bridge loop.
    return _run_async_in_temp_loop(async_batch_meta2kv_batch_meta, meta)


def tqbridge(dispatch_mode: "dict | Dispatch" = None):
    # [FUNCTION DOC]
    # Builds a decorator that bridges TQ metadata and TensorDict for sync or async functions.
    # Behavior: validates dispatch_mode when given. The wrapper finds a metadata argument; if none, calls the function
    # unchanged. Otherwise initializes tq once, remembers tags and converts KVBatchMeta to BatchMeta, replaces every metadata
    # argument with fetched data, logs timing and calls the function. If the output is a TensorDict with a non-empty batch
    # size (which must equal the meta size, AssertionError otherwise) it is a candidate for write-back. need_collect comes
    # from dispatch_mode (True when absent). When both hold, writes the output to TQ and returns updated BatchMeta (converted
    # back to KVBatchMeta with the original tags if the input was KV); otherwise returns the post-processed output.
    # Inputs: dispatch_mode - None, Dispatch or dict. Outputs: decorator. Exceptions: dispatch validation errors.
    """Creates a decorator for bridging KVBatchMeta and TensorDict.

    This decorator automatically handles conversions between `KVBatchMeta`
    and `TensorDict` in function parameters, and decides whether to sync function
    output back to `KVBatchMeta` based on configuration(`put_data`). It supports
    both synchronous and asynchronous functions (async def). When TQ is not enabled, it
    simply calls the original function as-is.

    Args:
        dispatch_mode: Controls data collection behavior for the current worker. Passed to
                      _compute_need_collect to determine if current worker should collect data.
                      If None, _compute_need_collect will return True to fallback default logics.


    Returns:
        A decorator function used to decorate target functions (synchronous or asynchronous).
    """
    # TODO: move to the top
    from verl.single_controller.base.decorator import _check_dispatch_mode

    if dispatch_mode is not None:
        _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: Wraps func, choosing the async wrapper for coroutine functions and the sync wrapper otherwise.
        pid = os.getpid()

        @wraps(func)
        def inner(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Synchronous bridged wrapper as described in tqbridge (uses the bridge-loop sync helpers).
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return func(*args, **kwargs)
            else:
                global TQ_INITIALIZED
                if not TQ_INITIALIZED:
                    tq.init()
                    TQ_INITIALIZED = True

                is_kv_batch_meta = isinstance(batch_meta, KVBatchMeta)
                if is_kv_batch_meta:
                    tags = batch_meta.tags
                    batch_meta = kv_batch_meta2batch_meta(batch_meta)
                t1 = time.time()
                args = [_meta_to_realdata(arg) if isinstance(arg, BatchMeta | KVBatchMeta) else arg for arg in args]
                kwargs = {
                    k: _meta_to_realdata(v) if isinstance(v, BatchMeta | KVBatchMeta) else v for k, v in kwargs.items()
                }
                t2 = time.time()
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batch_meta.size}, cost time: {t2 - t1}"
                )

                output = func(*args, **kwargs)

                put_data = False
                if isinstance(output, TensorDict):
                    if output.batch_size:
                        assert output.batch_size[0] == batch_meta.size, (
                            f"output batch size {output.batch_size} != meta size {batch_meta.size}"
                        )
                        put_data = True

                if dispatch_mode is not None:
                    need_collect = _compute_need_collect(dispatch_mode, args)
                else:
                    need_collect = True
                if put_data and need_collect:
                    updated_meta = _update_meta_with_output(output, batch_meta, func.__name__)
                    if is_kv_batch_meta:
                        updated_meta = batch_meta2kv_batch_meta(updated_meta)
                        updated_meta.tags = tags
                    return updated_meta
                return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asynchronous bridged wrapper as described in tqbridge (awaits the async helpers directly).
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return await func(*args, **kwargs)
            else:
                global TQ_INITIALIZED
                if not TQ_INITIALIZED:
                    tq.init()
                    TQ_INITIALIZED = True

                is_kv_batch_meta = isinstance(batch_meta, KVBatchMeta)
                if is_kv_batch_meta:
                    tags = batch_meta.tags
                    batch_meta = await async_kv_batch_meta2batch_meta(batch_meta)

                t1 = time.time()
                args = [
                    await _async_meta_to_realdata(arg) if isinstance(arg, BatchMeta | KVBatchMeta) else arg
                    for arg in args
                ]
                kwargs = {
                    k: await _async_meta_to_realdata(v) if isinstance(v, BatchMeta | KVBatchMeta) else v
                    for k, v in kwargs.items()
                }
                t2 = time.time()
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batch_meta.size}, cost time: {t2 - t1}"
                )

                output = await func(*args, **kwargs)

                put_data = False
                if isinstance(output, TensorDict):
                    if output.batch_size:
                        assert output.batch_size[0] == batch_meta.size, (
                            f"output batch size {output.batch_size} != meta size {batch_meta.size}"
                        )
                        put_data = True

                if dispatch_mode is not None:
                    need_collect = _compute_need_collect(dispatch_mode, args)
                else:
                    need_collect = True
                if put_data and need_collect:
                    updated_meta = await _async_update_meta_with_output(output, batch_meta, func.__name__)
                    if is_kv_batch_meta:
                        updated_meta = await async_batch_meta2kv_batch_meta(updated_meta)
                        updated_meta.tags = tags
                    return updated_meta
                return _postprocess_common(output, put_data, need_collect)

        wrapper_inner = inner
        wrapper_async_inner = async_inner

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator
