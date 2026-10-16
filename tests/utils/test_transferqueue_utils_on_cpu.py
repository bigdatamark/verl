# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# CPU tests for the transferqueue async bridge: running a coroutine function
# synchronously via _run_async_in_temp_loop must not leak file descriptors over
# repeated calls, and must reuse a single background event-loop thread
# (module globals _ASYNC_BRIDGE_THREAD and _ASYNC_BRIDGE_LOOP) until
# _shutdown_async_bridge_runtime is called.

import gc
import os

import pytest

from verl.utils import transferqueue_utils as tqu


async def _noop():
    # [FUNCTION DOC]
    # HELPER: Coroutine that does nothing and returns None.
    return None


def _open_fd_count() -> int:
    # [FUNCTION DOC]
    # HELPER: Counts open file descriptors of the current process via /proc/self/fd.
    # Outputs: int. Side effects: skips the test when /proc/self/fd is missing
    # (non-Linux).
    if not os.path.isdir("/proc/self/fd"):
        pytest.skip("requires Linux /proc file descriptor accounting")
    return len(os.listdir("/proc/self/fd"))


def test_temp_event_loop_releases_file_descriptors():
    # [FUNCTION DOC]
    # Behavior: with garbage collection disabled (to avoid GC hiding leaks) and a fresh
    # bridge runtime, runs the noop once to warm up, records the fd count, runs it 32
    # more times, and asserts the fd count did not grow. Always shuts down the bridge
    # and restores the GC state.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        before = _open_fd_count()
        for _ in range(32):
            tqu._run_async_in_temp_loop(_noop)
        leaked = _open_fd_count() - before
    finally:
        tqu._shutdown_async_bridge_runtime()
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    assert leaked == 0


def test_async_bridge_loop_reused_between_calls():
    # [FUNCTION DOC]
    # Behavior: after a shutdown, the first call creates a live bridge thread and an
    # open loop; a second call reuses the identical thread and loop objects. Shuts the
    # bridge down afterwards.
    tqu._shutdown_async_bridge_runtime()
    try:
        tqu._run_async_in_temp_loop(_noop)
        first_thread = tqu._ASYNC_BRIDGE_THREAD
        first_loop = tqu._ASYNC_BRIDGE_LOOP

        assert first_thread is not None
        assert first_loop is not None
        assert first_thread.is_alive()
        assert not first_loop.is_closed()

        tqu._run_async_in_temp_loop(_noop)
        assert tqu._ASYNC_BRIDGE_THREAD is first_thread
        assert tqu._ASYNC_BRIDGE_LOOP is first_loop
    finally:
        tqu._shutdown_async_bridge_runtime()
