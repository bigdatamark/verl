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
# Ray helpers: detecting whether Ray was told not to set accelerator visibility env vars, parallel
# ray.put of many objects, getting/creating an asyncio event loop, and a decorator that lets async
# functions be called from both sync and async contexts.
"""
Contains commonly used utilities for ray
"""

import asyncio
import concurrent.futures
import functools
import inspect
import os
from typing import Any, Optional

import ray


def ray_noset_visible_devices(env_vars=os.environ):
    # [FUNCTION DOC]
    # Behavior: returns True if any RAY_EXPERIMENTAL_NOSET_*_VISIBLE_* variable (CUDA, ROCR, HIP, Ascend,
    # Habana, Neuron, TPU, oneAPI) is set to a non-empty value in env_vars.
    # Inputs: env_vars - mapping (default os.environ, bound at definition time). Outputs: bool.
    # Side effects: none.
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def parallel_put(data_list: list[Any], max_workers: Optional[int] = None):
    # [FUNCTION DOC]
    # Behavior: submits ray.put for each item to a thread pool (default min(len, 16) workers), collects
    # (index, ref) pairs as they complete, and returns refs ordered by input index.
    # Inputs: data_list - non-empty list; max_workers. Outputs: list of ObjectRefs.
    # Exceptions: AssertionError on empty list; ray.put errors propagate. Side effects: object store writes.
    """
    Puts a list of data into the Ray object store in parallel using a thread pool.

    Args:
        data_list (List[Any]): A list of Python objects to be put into the Ray object store.
        max_workers (int, optional): The maximum number of worker threads to use.
                                     Defaults to min(len(data_list), 16).

    Returns:
        List[ray.ObjectRef]: A list of Ray object references corresponding to the input data_list,
                             maintaining the original order.
    """
    assert len(data_list) > 0, "data_list must not be empty"

    def put_data(index, data):
        # [FUNCTION DOC]
        # HELPER: Returns (index, ray.put(data)).
        return index, ray.put(data)

    if max_workers is None:
        max_workers = min(len(data_list), 16)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        data_list_f = [executor.submit(put_data, i, data) for i, data in enumerate(data_list)]
        res_lst = []
        for future in concurrent.futures.as_completed(data_list_f):
            res_lst.append(future.result())

        # reorder based on index
        output = [None for _ in range(len(data_list))]
        for res in res_lst:
            index, data_ref = res
            output[index] = data_ref

    return output


def get_event_loop():
    # [FUNCTION DOC]
    # Behavior: returns asyncio.get_event_loop(); if that raises RuntimeError (no loop in this thread), creates
    # a new loop and sets it as current.
    # Outputs: event loop. Side effects: may set the thread's event loop.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def auto_await(func):
    # [FUNCTION DOC]
    # Behavior: returns a wrapper (see docstring; note the code's case numbering differs from the docstring's).
    # Outputs: decorated function.
    """Auto await a coroutine function.

    Handles three cases:
    1. When the decorated function is called with await: returns the coroutine
       so the caller can await it.
    2. When called directly and there is no running event loop: runs the
       coroutine with asyncio.run() and returns the result.
    3. When called directly and the event loop is already running: runs the
       coroutine (e.g. in a thread pool to avoid deadlock) and returns the result.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Calls func; non-coroutine results are returned as-is. With no running loop, runs the coroutine
        # via asyncio.run. With a running loop, returns the coroutine unawaited if the immediate caller frame is a
        # coroutine function; otherwise runs it with asyncio.run in a single worker thread and blocks for the result.
        # KNOWN ISSUE: in the last case the running loop's thread is blocked until completion, and the coroutine
        # runs on a different loop, so it must not depend on objects bound to the original loop. An async caller
        # that does not await gets an unawaited coroutine.
        # Exceptions: exceptions of the coroutine propagate. Side effects: may create threads/loops.
        coro = func(*args, **kwargs)

        if not inspect.iscoroutine(coro):
            return coro

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Case 1: No running loop -> run with asyncio.run()
        if loop is None:
            return asyncio.run(coro)

        # Case 2: Running loop -> return coro if caller will await
        caller_frame = inspect.currentframe()
        if caller_frame is not None:
            caller_frame = caller_frame.f_back
        caller_is_async = caller_frame is not None and (caller_frame.f_code.co_flags & inspect.CO_COROUTINE) != 0
        if caller_is_async:
            return coro

        # Case 3: Running loop -> run coro in thread pool
        # (cannot block the loop thread without deadlock)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    return wrapper
