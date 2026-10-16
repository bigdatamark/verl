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
# Ascend NPU profiling backend built on torch_npu and MSTX markers: MSTX implementations of the
# range-marker API, an MSTX-annotated timer, a factory for torch_npu profiler objects, and the
# NPUProfiler DistProfiler backend. Importing requires torch_npu.
# Inspired from https://gitee.com/ascend/MindSpeed-RL/blob/master/mindspeed_rl/utils/utils.py
import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch_npu
from packaging import version
from torch_npu.npu import mstx

from .config import NPUToolConfig
from .profile import DistProfiler, ProfilerConfig


def mark_start_range(message: Optional[str] = None) -> None:
    # [FUNCTION DOC]
    # Behavior: starts an MSTX range with the message. Outputs: the range id returned by mstx.range_start
    # (annotated None but actually returns the id). Exceptions: from torch_npu. Side effects: emits a marker.
    """Start a mark range in the profiler.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """
    return mstx.range_start(message=message)


def mark_end_range(range_id: str) -> None:
    # [FUNCTION DOC]
    # Behavior: ends the MSTX range with the given id. Outputs: mstx return value. Side effects: emits a marker.
    """End a mark range in the profiler.

    Args:
        range_id (str):
            The id of the mark range to end.
    """
    return mstx.range_end(range_id)


def mark_annotate(message: Optional[str] = None) -> Callable:
    # [FUNCTION DOC]
    # Behavior: returns a decorator wrapping a function in an MSTX range named message or the function name.
    # Outputs: decorator. Side effects: markers on each call of the decorated function.
    """Decorate a function to annotate a mark range along with the function life cycle.

    Args:
        message (str, optional):
            The message to be displayed in the profiler. Defaults to None.
    """

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: Wraps func with mstx.mstx_range(message or func.__name__) and returns the wrapped function.
        profile_message = message or func.__name__
        return mstx.mstx_range(profile_message)(func)

    return decorator


@contextmanager
def marked_timer(name: str, timing_raw: dict[str, float], *args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC]
    # Behavior: warns (logging) if extra positional/keyword args are passed (unsupported, e.g. NVTX color);
    # opens an MSTX range named name, delegates timing to performance._timer (which accumulates elapsed
    # seconds into timing_raw[name]), then closes the range.
    # Inputs: name; timing_raw - mutable mapping of name to seconds; extra args ignored.
    # Outputs: generator-based context manager yielding None.
    # KNOWN ISSUE: if the body raises, mark_end_range is skipped (no try/finally), leaving the range open.
    # Side effects: mutates timing_raw; emits markers; may log warnings.
    """Context manager for timing with MSTX markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds MSTX markers for profiling.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    if args:
        logging.warning(f"Args are not supported in mstx_profile, but received: {args}")
    if kwargs:
        logging.warning(f"Kwargs are not supported in mstx_profile, but received: {kwargs}")
    mark_range = mark_start_range(message=name)
    from .performance import _timer

    yield from _timer(name, timing_raw)
    mark_end_range(mark_range)


def get_npu_profiler(
    contents: list[str],
    profile_level: str,
    profile_save_path: str,
    analysis: bool,
    role: Optional[str] = None,
    profile_step: Optional[str] = None,
):
    # [FUNCTION DOC]
    # Behavior: maps profile_level string to torch_npu ProfilerLevel; appends profile_step then role as
    # sub-directories of the save path when given; requires torch_npu >= 2.1; builds an experimental config
    # (DB export, data simplification, msprof_tx on, excluding the "communication" mstx domain); always
    # collects CPU activity, plus NPU when contents is None or includes "npu"; enables modules/stack/shapes/
    # memory when contents is None or lists them; trace handler writes to the save path with analysis flag.
    # Inputs: see docstring. Outputs: an unstarted torch_npu.profiler.profile object.
    # Exceptions: ValueError for unknown level; RuntimeError for torch_npu < 2.1.
    # Side effects: none until started.
    """Generate and return an NPU profiler object.

    Args:
        contents (list[str]):
            A list of options to control the collection content,
            such as npu, memory, shapes, module, stack. CPU activity is always collected,
            since the per-stage mstx markers are CPU-side events.
        profile_level (str):
            The collection level, which can be set to level_none,
            level0, level1 and level2.
        profile_save_path (str):
            The path to save the collected data.
        analysis (bool):
            Whether to enables automatic data parsing.
        role (str, optional):
            The role of the current data collection. Defaults to None.
        profile_step(str, optional):
            The current training step. Defaults to None.
    """
    if profile_level == "level_none":
        level = torch_npu.profiler.ProfilerLevel.Level_none
    elif profile_level == "level0":
        level = torch_npu.profiler.ProfilerLevel.Level0
    elif profile_level == "level1":
        level = torch_npu.profiler.ProfilerLevel.Level1
    elif profile_level == "level2":
        level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"level only supports level0, 1, 2, and level_none, but gets {profile_level}")

    if profile_step:
        profile_save_path = os.path.join(profile_save_path, profile_step)
    if role:
        profile_save_path = os.path.join(profile_save_path, role)

    # The ability to filter communication via mstx_domain_exclude requires torch_npu==2.1 or higher.
    if version.parse(torch_npu.__version__) < version.parse("2.1"):
        raise RuntimeError("torch_npu==2.1 or higher is required to use mstx_domain_exclude")

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=level,
        export_type=torch_npu.profiler.ExportType.Db,
        data_simplification=True,
        msprof_tx=True,
        mstx_domain_exclude=["communication"],
    )

    # CPU activity is always collected, whatever `contents` selects: verl marks each stage with an
    # mstx range, and those markers -- like operator names -- are CPU-side events, so a device-only
    # trace would be bare kernels that cannot be attributed to any stage.
    activites = [torch_npu.profiler.ProfilerActivity.CPU]
    if contents is None or "npu" in contents:
        activites.append(torch_npu.profiler.ProfilerActivity.NPU)

    prof = torch_npu.profiler.profile(
        with_modules=contents is None or "module" in contents,
        with_stack=contents is None or "stack" in contents,
        record_shapes=contents is None or "shapes" in contents,
        profile_memory=contents is None or "memory" in contents,
        activities=activites,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path, analyse_flag=analysis),
        experimental_config=experimental_config,
    )
    return prof


class NPUProfiler(DistProfiler):
    # [CLASS DOC]
    # DistProfiler backend for Ascend NPU. In continuous mode (discrete False) one profiler spans start/stop,
    # guarded by a class-level counter so only one profiler is active per process; in discrete mode each
    # annotated function call creates, starts and stops its own profiler.
    """
    NPU profiler. Initialized in a worker to control the NPU profiler.
    """

    _define_count = 0

    def __init__(self, rank: int, config: ProfilerConfig, tool_config: NPUToolConfig, **kwargs):
        # [FUNCTION DOC]
        # Behavior: defaults config to a disabled ProfilerConfig when falsy; asserts tool_config is given when
        # enabled; stores discrete, contents, level, analysis from tool_config and save_path from config.
        # Inputs: rank (unused); config; tool_config; kwargs ignored.
        # Exceptions: AssertionError when enabled without tool_config.
        # KNOWN ISSUE: when tool_config is None and profiling is disabled, attribute access on tool_config raises
        # AttributeError right after the assertion passes. Docstring wrongly names NsightSystemsProfiler.
        # Side effects: sets instance attributes.
        """Initialize the NsightSystemsProfiler.

        Args:
            rank (int): The rank of the current process.
            config (Optional[ProfilerConfig]): Configuration for the profiler. If None, a default configuration is used.
            tool_config (NPUToolConfig): The config to control npu profiler behavior.
        """
        if not config:
            config = ProfilerConfig(ranks=[], enable=False)
        if not tool_config:
            assert not config.enable, "tool_config must be set when profiler is enabled"
        self.discrete: bool = tool_config.discrete
        self.profile_npu = None
        self.profile_contents = tool_config.contents
        self.profile_level = tool_config.level
        self.profile_save_path = config.save_path
        self.analysis = tool_config.analysis

    def start(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: in continuous mode and if no profiler is active process-wide (class counter 0), creates a
        # profiler (save path/role sub-dir), starts it and increments the counter. Discrete mode: no-op.
        # Inputs: kwargs - optional "role". Side effects: starts profiling; mutates class counter.
        role = kwargs.get("role", None)
        if not self.discrete and NPUProfiler._define_count == 0:
            self.profile_npu = get_npu_profiler(
                contents=self.profile_contents,
                profile_level=self.profile_level,
                profile_save_path=self.profile_save_path,
                analysis=self.analysis,
                role=role,
            )
            self.profile_npu.start()
            NPUProfiler._define_count += 1

    def stop(self):
        # [FUNCTION DOC]
        # Behavior: in continuous mode and when the counter is 1, steps and stops the active profiler and
        # decrements the counter. Otherwise no-op. Side effects: flushes trace via on_trace_ready.
        if not self.discrete and NPUProfiler._define_count == 1:
            self.profile_npu.step()
            self.profile_npu.stop()
            NPUProfiler._define_count -= 1

    def step(self):
        # [FUNCTION DOC]
        # No-op (see docstring for why it must exist). Outputs: None.
        """No-op per-mini-batch hook.

        The actor update loop calls this once per mini-batch to drive the torch profiler's
        schedule, but the NPU profiler is driven by explicit start/stop calls, so it has nothing
        to advance here. It must still be defined: without it, the dispatcher's
        ``getattr(self._impl, "step", lambda: None)`` resolves to the inherited
        ``DistProfiler.step`` (backend impls subclass ``DistProfiler`` but never run its
        ``__init__``), which then reads dispatcher-only state such as ``_enable`` and raises
        ``AttributeError``.
        """
        return

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        # [FUNCTION DOC]
        # Behavior: returns a decorator; the wrapped function is surrounded by an MSTX range named message, role,
        # or the function name. In discrete mode a fresh profiler (with role sub-dir) is started before and
        # stepped+stopped after the call.
        # Inputs: message, role; extra kwargs ignored. Outputs: decorator.
        # KNOWN ISSUE: exceptions in the wrapped function skip range end and profiler stop (no try/finally).
        # Side effects: profiling and file output.
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with NPUProfiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: Builds the functools.wraps wrapper for func.
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                # [FUNCTION DOC]
                # HELPER: Runs func with the profiling behavior described in annotate; returns func's result.
                # Prefer the stage label (`role`, e.g. "actor_compute_log_prob"): it names both the
                # role and the function, which the method name alone cannot do for a colocated
                # worker. Fall back to the method name for stages that declare no role.
                profile_name = message or role or func.__name__
                discrete_mode = self.discrete

                if not discrete_mode:
                    mark_range = mark_start_range(message=profile_name)
                else:
                    profile_npu = get_npu_profiler(
                        contents=self.profile_contents,
                        profile_level=self.profile_level,
                        profile_save_path=self.profile_save_path,
                        analysis=self.analysis,
                        role=role,
                    )
                    profile_npu.start()
                    mark_range = mark_start_range(message=profile_name)

                result = func(*args, **kwargs_inner)

                if not discrete_mode:
                    mark_end_range(mark_range)
                else:
                    mark_end_range(mark_range)
                    profile_npu.step()
                    profile_npu.stop()

                return result

            return wrapper

        return decorator
