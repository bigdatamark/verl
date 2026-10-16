# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
# NVIDIA Nsight Systems profiling backend: NVTX implementations of the range-marker API, an NVTX-
# annotated timer, and NsightSystemsProfiler, which drives profiler start/stop via the active platform
# (e.g. CUDA profiler API) and can relocate nsys reports out of Ray's fixed log dir
# (RAY_NSIGHT_LOG_DIR). Importing requires the nvtx package.
import functools
import glob
import logging
import os
import shutil
import socket
from contextlib import contextmanager
from typing import Callable, Optional

import nvtx

from verl.plugin.platform import get_platform

from .config import NsightToolConfig
from .profile import DistProfiler, ProfilerConfig

logger = logging.getLogger(__name__)

# Ray writes Nsight Systems reports to a fixed, non-configurable directory
# (see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html):
#   /tmp/ray/session_latest/logs/nsight/worker_process_<pid>[.<range_id>].nsys-rep
# The ``<pid>`` is nsys's ``%p`` token, i.e. the PID of the profiled (worker) process,
# which matches ``os.getpid()`` inside the verl worker. We use that to relocate only the
# current process's own artifacts and avoid races between co-located ranks.
RAY_NSIGHT_LOG_DIR = "/tmp/ray/session_latest/logs/nsight"


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    # [FUNCTION DOC]
    # Behavior: starts an NVTX range with given message/color/domain/category.
    # Outputs: the NVTX range id (annotated None). Side effects: emits a marker.
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """
    return nvtx.start_range(message=message, color=color, domain=domain, category=category)


def mark_end_range(range_id: str) -> None:
    # [FUNCTION DOC]
    # Behavior: ends the NVTX range with the given id. Side effects: emits a marker.
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return nvtx.end_range(range_id)


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    # [FUNCTION DOC]
    # Behavior: returns a decorator that wraps a function in nvtx.annotate with message (or function name)
    # and the given color/domain/category. Outputs: decorator.
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
        color (str, optional):
            The color of the range. Defaults to None.
        domain (str, optional):
            The domain of the range. Defaults to None.
        category (str, optional):
            The category of the range. Defaults to None.
    """

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: Applies nvtx.annotate to func and returns the wrapped function.
        profile_message = message or func.__name__
        return nvtx.annotate(profile_message, color=color, domain=domain, category=category)(func)

    return decorator


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    # [FUNCTION DOC]
    # Behavior: opens an NVTX range named name, delegates to performance._timer to accumulate elapsed
    # seconds into timing_raw[name], then closes the range.
    # Inputs: name; timing_raw - mutable mapping; color/domain/category - NVTX attributes.
    # Outputs: generator context manager yielding None.
    # KNOWN ISSUE: range is not closed if the body raises (no try/finally).
    # Side effects: mutates timing_raw; emits markers.
    """Context manager for timing with NVTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds NVTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the NVTX marker. Defaults to None.
        domain (Optional[str]): Domain for the NVTX marker. Defaults to None.
        category (Optional[str]): Category for the NVTX marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    mark_range = mark_start_range(message=name, color=color, domain=domain, category=category)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


class NsightSystemsProfiler(DistProfiler):
    # [CLASS DOC]
    # DistProfiler backend for Nsight Systems. Continuous mode: start/stop toggle the platform profiler
    # around the whole step. Discrete mode: each annotated call toggles it individually.
    """Nsight system profiler. Installed in a worker to control the Nsight system profiler."""

    def __init__(self, rank: int, config: Optional[ProfilerConfig], tool_config: Optional[NsightToolConfig], **kwargs):
        # [FUNCTION DOC]
        # Behavior: defaults config to ProfilerConfig(ranks=[]) when falsy; asserts tool_config given when
        # enabled; stores discrete from tool_config.
        # Inputs: rank (unused); config; tool_config; kwargs ignored.
        # Exceptions: AssertionError when enabled without tool_config.
        # KNOWN ISSUE: with tool_config None and profiling disabled, reading tool_config.discrete raises
        # AttributeError.
        # Side effects: sets self.discrete.
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
        """
        # If no configuration is provided, create a default ProfilerConfig with an empty list of ranks
        if not config:
            config = ProfilerConfig(ranks=[])
        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"
        self.discrete: bool = tool_config.discrete

    def start(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: calls platform profiler_start unless discrete. Inputs: kwargs ignored. Side effects: starts capture.
        if not self.discrete:
            get_platform().profiler_start()

    def stop(self):
        # [FUNCTION DOC]
        # Behavior: calls platform profiler_stop unless discrete. Side effects: stops capture.
        if not self.discrete:
            get_platform().profiler_stop()

    def relocate_results(
        self,
        save_path: Optional[str],
        *,
        rank: Optional[int] = None,
        save_file_prefix: Optional[str] = None,
        source_dir: Optional[str] = None,
    ) -> list[str]:
        # [FUNCTION DOC]
        # Behavior: returns [] (with warning) if save_path falsy. Globs source_dir (default RAY_NSIGHT_LOG_DIR)
        # for files named worker_process_<current pid>.*; returns [] (info log) if none. Otherwise creates
        # save_path, and moves each match to save_path/<role prefix_><hostname>_<basename>; FileNotFoundError is
        # ignored, other OSError logged as warning. Logs a summary when anything moved.
        # Inputs/Outputs: see docstring. Exceptions: none propagated from moves; makedirs errors propagate.
        # Side effects: filesystem moves, directory creation, logging.
        """Move this process's Nsight reports out of Ray's fixed log dir into ``save_path``.

        Ray hardcodes the Nsight output directory and offers no way to change it, which makes the
        ``*.nsys-rep`` files awkward to collect. This moves the current worker process's artifacts
        (matched by PID, so co-located ranks never touch each other's files) into ``save_path``.

        This is best-effort and safe to call repeatedly: files that do not exist yet (nsys finalizes
        the report only after the profiling session shuts down) are simply skipped, and nothing is
        deleted. Destination filenames are prefixed with the hostname (and role, when known) to keep
        them unique across nodes that may reuse PIDs and to identify which worker produced them.

        Args:
            save_path: Destination directory. When falsy, relocation is skipped.
            rank: Owning rank, used only for logging.
            save_file_prefix: Optional role label embedded in the destination filename.
            source_dir: Override for the Ray Nsight log directory (defaults to ``RAY_NSIGHT_LOG_DIR``).

        Returns:
            The list of destination paths that were successfully moved.
        """
        if not save_path:
            logger.warning("nsys relocate_results: save_path is not set, skipping relocation (rank=%s)", rank)
            return []

        src_dir = source_dir or RAY_NSIGHT_LOG_DIR
        pid = os.getpid()
        # nsys may emit worker_process_<pid>.nsys-rep and, in discrete/capture-range mode,
        # worker_process_<pid>.<range_id>.nsys-rep, plus intermediate .qdstrm files.
        matches = sorted(glob.glob(os.path.join(src_dir, f"worker_process_{pid}.*")))
        if not matches:
            logger.info(
                "nsys relocate_results: no reports for pid=%s under %s yet (rank=%s); "
                "nsys writes the report after the profiling session shuts down.",
                pid,
                src_dir,
                rank,
            )
            return []

        os.makedirs(save_path, exist_ok=True)
        hostname = socket.gethostname()
        prefix = f"{save_file_prefix}_" if save_file_prefix else ""
        moved: list[str] = []
        for src in matches:
            dst = os.path.join(save_path, f"{prefix}{hostname}_{os.path.basename(src)}")
            try:
                shutil.move(src, dst)
                moved.append(dst)
            except FileNotFoundError:
                # Another attempt (or process) already moved it; ignore.
                continue
            except OSError as e:
                logger.warning("nsys relocate_results: failed to move %s -> %s (rank=%s): %s", src, dst, rank, e)
        if moved:
            logger.info("nsys relocate_results: moved %d file(s) to %s (rank=%s)", len(moved), save_path, rank)
        return moved

    def step(self):
        # [FUNCTION DOC]
        # No-op (see docstring). Outputs: None.
        """No-op per-mini-batch hook.

        The actor update loop calls this once per mini-batch to drive the torch profiler's
        schedule, but Nsight Systems profiling is controlled via start/stop, so it has nothing to
        advance here. It must still be defined here: without it, the dispatcher's
        ``getattr(self._impl, "step", lambda: None)`` resolves to the inherited
        ``DistProfiler.step`` (backend impls subclass ``DistProfiler`` but never run its
        ``__init__``), which then reads dispatcher-only state such as ``_enable`` and raises
        ``AttributeError``.
        """
        return

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ) -> Callable:
        # [FUNCTION DOC]
        # Behavior: returns a decorator; wrapped call is surrounded by an NVTX range named message, the "role"
        # kwarg, or function name; in discrete mode platform profiler start/stop wrap the call.
        # Inputs: message, color, domain, category; kwargs_outer may contain "role". Outputs: decorator.
        # KNOWN ISSUE: exceptions skip range end / profiler stop.
        # Side effects: profiling markers and capture toggling.
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker, which has a member field `profiler` with
        NightSystemsProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            color (str, optional):
                The color of the range. Defaults to None.
            domain (str, optional):
                The domain of the range. Defaults to None.
            category (str, optional):
                The category of the range. Defaults to None.
        """

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: Builds the functools.wraps wrapper for func.
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                # [FUNCTION DOC]
                # HELPER: Runs func with the profiling behavior described in annotate; returns its result.
                # Prefer the stage label (`role`, e.g. "actor_compute_log_prob"): it names both the
                # role and the function, which the method name alone cannot do for a colocated
                # worker. Fall back to the method name for stages that declare no role.
                profile_name = message or kwargs_outer.get("role") or func.__name__

                if self.discrete:
                    get_platform().profiler_start()
                mark_range = mark_start_range(message=profile_name, color=color, domain=domain, category=category)

                result = func(*args, **kwargs_inner)

                mark_end_range(mark_range)
                if self.discrete:
                    get_platform().profiler_stop()

                return result

            return wrapper

        return decorator
