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
# PyTorch torch.profiler backend: builds profilers that export gzipped chrome traces with descriptive
# per-process filenames (role, step, rank, parallel ranks, pid, timestamp) directly into save_path,
# optionally driven by a torch.profiler.schedule over update mini-batches, and the Profiler DistProfiler
# backend supporting continuous (one trace per step) and discrete (one trace per annotated stage) modes.
import functools
import os
import re
from datetime import datetime, timezone
from typing import Callable, Optional

import torch

from .config import ProfilerConfig, TorchProfilerToolConfig
from .profile import DistProfiler


def get_dist_topology() -> dict:
    # [FUNCTION DOC]
    # Behavior: collects rank/world_size from torch.distributed when initialized, and tp/pp/dp (and cp when
    # available) ranks from Megatron parallel_state when model parallel is initialized. Every lookup is
    # wrapped so failures (including missing Megatron) just omit keys.
    # Outputs: dict possibly containing rank, world_size, tp, pp, dp, cp. Exceptions: none. Side effects: none.
    """Best-effort snapshot of the current process's distributed topology.

    Used to make per-process profiler trace files self-describing. The returned dict
    may contain ``rank``/``world_size`` (from ``torch.distributed``) and the
    ``tp``/``pp``/``dp``/``cp`` parallel ranks (from Megatron's ``parallel_state`` when
    initialized). Every lookup is guarded, so this never raises and simply omits the
    pieces that are unavailable (e.g. plain FSDP data parallelism only exposes rank).
    """
    info: dict = {}
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            info["rank"] = dist.get_rank()
            info["world_size"] = dist.get_world_size()
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        if mpu.model_parallel_is_initialized():
            info["tp"] = mpu.get_tensor_model_parallel_rank()
            info["pp"] = mpu.get_pipeline_model_parallel_rank()
            info["dp"] = mpu.get_data_parallel_rank()
            try:
                info["cp"] = mpu.get_context_parallel_rank()
            except Exception:
                pass
    except Exception:
        pass

    return info


def _sanitize_name_part(text: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Converts to str, replaces each run of characters outside [0-9A-Za-z.=+-] with "-", strips
    # leading/trailing "-".
    """Make an arbitrary label safe to embed in a filename."""
    return re.sub(r"[^0-9A-Za-z.=+-]+", "-", str(text)).strip("-")


def build_trace_basename(
    rank: int,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    topology: Optional[dict] = None,
    profile_step: Optional[int] = None,
) -> str:
    # [FUNCTION DOC]
    # Behavior: uses the given topology or queries get_dist_topology. Joins with "_" the parts: sanitized
    # save_file_prefix, sanitized role, "step<N>" (if profile_step given), "rank<R>" (topology rank or given
    # rank) with "-of-<world_size>" when known, a "-"-joined "tp<x>-pp<y>-dp<z>-cp<w>" segment for present
    # dims, "pid<pid>", and a local-time timestamp YYYYmmddHHMMSS plus milliseconds.
    # Outputs: filename stem string. Exceptions: none. Side effects: none.
    """Build a descriptive, per-process trace filename stem.

    Encodes -- when available -- the worker role (``save_file_prefix``, e.g. ``actor``),
    the profiling scope role (``role``, e.g. ``train``), the RL step
    (``profile_step``), the global rank and world size, and the
    tensor/pipeline/data/context parallel ranks, followed by pid and a timestamp so that
    files written by different processes never collide.
    """
    topology = get_dist_topology() if topology is None else topology
    current_time = datetime.now(tz=timezone.utc).astimezone()
    timestamp = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]
    pid = os.getpid()

    parts: list[str] = []
    if save_file_prefix:
        parts.append(_sanitize_name_part(save_file_prefix))
    if role:
        parts.append(_sanitize_name_part(role))
    if profile_step is not None:
        parts.append(f"step{_sanitize_name_part(profile_step)}")

    global_rank = topology.get("rank", rank)
    world_size = topology.get("world_size")
    rank_part = f"rank{global_rank}"
    if world_size:
        rank_part += f"-of-{world_size}"
    parts.append(rank_part)

    parallel_part = "-".join(f"{dim}{topology[dim]}" for dim in ("tp", "pp", "dp", "cp") if dim in topology)
    if parallel_part:
        parts.append(parallel_part)

    parts.append(f"pid{pid}")
    parts.append(timestamp)
    return "_".join(parts)


def get_torch_profiler(
    contents: list[str],
    save_path: str,
    role: Optional[str] = None,
    save_file_prefix: Optional[str] = None,
    rank: int = 0,
    profile_step: Optional[int] = None,
    schedule: Optional[dict] = None,
    name_mini_batch_window: bool = False,
):
    # [FUNCTION DOC]
    # Behavior: creates save_path; computes base file name via build_trace_basename; defines a trace handler
    # with an invocation counter that writes <base><suffix>.json.gz via export_chrome_trace, where suffix is
    # "_mb<a>" / "_mb<a>-<b>" (mini-batch window, when scheduled and name_mini_batch_window and step_num is an
    # int), else "_part<idx>" for scheduled windows lacking step_num or for any call after the first.
    # Activities: always CPU, plus CUDA when contents empty or contains "cuda"; with_stack/record_shapes/
    # profile_memory from contents. Adds torch.profiler.schedule(**schedule) when given and then disables
    # ProfilerStep row recording.
    # Inputs: see docstring. Outputs: unstarted torch.profiler.profile.
    # Exceptions: OSError from makedirs; TypeError for bad schedule keys.
    # Side effects: creates directory; handler prints and writes trace files.
    """Build a ``torch.profiler.profile`` instance.

    Args:
        contents: Selects the other ``torch.profiler.profile`` arguments -- ``cuda`` maps to
            ``activities``, ``shapes`` to ``record_shapes``, ``memory`` to ``profile_memory`` and
            ``stack`` to ``with_stack``. CPU activity is always on, since verl's per-stage
            ``record_function`` markers are CPU-side events.
        save_path: Directory to write chrome traces to.
        role: Optional logical scope name (e.g. ``train`` for a worker's whole-step window, or
            a stage name in discrete mode), embedded in the filename.
        save_file_prefix: Optional filename prefix, typically the worker role (``actor``/
            ``critic``/``ref``) so per-process traces are distinguishable.
        rank: Global rank, embedded in the trace filename (a fallback when
            ``torch.distributed`` is not initialized).
        profile_step: Optional RL step being profiled, embedded in the filename.
        schedule: Optional kwargs for ``torch.profiler.schedule``
            (``skip_first``/``wait``/``warmup``/``active``/``repeat``). When provided, the caller
            drives ``prof.step()`` once per update mini-batch to advance it.
        name_mini_batch_window: When True (discrete update stage), tag each saved file with the
            range of mini-batches it holds, since a scheduled file is a window rather than the
            whole stage.
    """
    # All traces land directly in save_path: the role is already part of the filename, so an
    # extra directory level would only scatter one step's traces across sibling dirs and hide
    # them from finish_hook_cmd, which is handed save_path.
    os.makedirs(save_path, exist_ok=True)

    base_file_name = build_trace_basename(
        rank=rank, role=role, save_file_prefix=save_file_prefix, profile_step=profile_step
    )

    # A scheduled profiler can fire on_trace_ready more than once (one file per active cycle);
    # keep an invocation counter so a later window cannot overwrite an earlier one.
    handler_state = {"count": 0}

    def _scheduled_mini_batch_range(step_num: int) -> tuple[int, int]:
        # [FUNCTION DOC]
        # HELPER: Computes (first, last) mini-batch indices of the window flushed at step_num: last = step_num-1
        # (min 0); cycle length = wait+warmup+active (active at least 1); cycle index = (last - skip_first,
        # floored at 0) divided by cycle length; first = skip_first + cycle*cycle_len + wait + warmup, capped at last.
        # Inputs: step_num - profiler step count. Uses enclosing schedule dict.
        """Mini-batches held by the window being flushed at ``step_num``.

        step() is advanced once per mini-batch, so the last mini-batch in the window is the one
        that just ran, ``step_num - 1``. The window's first mini-batch is derived from the
        schedule rather than from ``active``, because a window can hold fewer than ``active``
        mini-batches when the update loop runs out mid-window and ``stop()`` flushes it.
        """
        skip_first = int(schedule.get("skip_first", 0) or 0)
        wait = int(schedule.get("wait", 0) or 0)
        warmup = int(schedule.get("warmup", 0) or 0)
        active = max(int(schedule.get("active", 1) or 1), 1)

        last_mb = max(step_num - 1, 0)
        cycle_len = wait + warmup + active
        cycle = max(last_mb - skip_first, 0) // cycle_len if cycle_len else 0
        # Recording starts after this cycle's skipped, idle and warmup mini-batches.
        first_mb = skip_first + cycle * cycle_len + wait + warmup
        return min(first_mb, last_mb), last_mb

    def _trace_handler(prof):
        # [FUNCTION DOC]
        # HELPER: on_trace_ready callback; increments counter, chooses suffix as described above, prints and
        # exports the chrome trace. Side effects: file write, stdout.
        idx = handler_state["count"]
        handler_state["count"] += 1
        suffix = ""
        if schedule and name_mini_batch_window:
            step_num = getattr(prof, "step_num", None)
            if isinstance(step_num, int):
                first_mb, last_mb = _scheduled_mini_batch_range(step_num)
                suffix = f"_mb{first_mb}" if first_mb == last_mb else f"_mb{first_mb}-{last_mb}"
            else:
                suffix = f"_part{idx}"
        elif idx:
            suffix = f"_part{idx}"
        out_path = os.path.join(save_path, f"{base_file_name}{suffix}.json.gz")
        print(f"[Profiler] Saving trace to {out_path}")
        prof.export_chrome_trace(out_path)

    contents = set(contents) if contents else set()
    # CPU activity is always collected, whatever `contents` selects: verl marks each stage with
    # record_function, and those markers -- like operator names -- are CPU-side events, so a
    # device-only trace would be bare kernels that cannot be attributed to any stage.
    activities = [torch.profiler.ProfilerActivity.CPU]
    if not contents or "cuda" in contents:
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profile_kwargs = dict(
        activities=activities,
        with_stack="stack" in contents,
        record_shapes="shapes" in contents,
        profile_memory="memory" in contents,
        on_trace_ready=_trace_handler,
    )
    # With a schedule the caller drives prof.step() per mini-batch to walk the
    # skip_first/wait/warmup/active/repeat state machine; without one, collection runs
    # continuously from start() to stop().
    if schedule:
        profile_kwargs["schedule"] = torch.profiler.schedule(**schedule)

    prof = torch.profiler.profile(**profile_kwargs)

    if schedule:
        # A schedule makes torch label every prof.step() boundary with a
        # "ProfilerStep#<n>" record_function row. In verl a step() is just one
        # update mini-batch (see TrainingWorker.train_mini_batch), so those rows
        # tag mini-batch boundaries rather than any meaningful RL step and only
        # add noise to the trace. Turn the labelling off: prof.step() still walks
        # the schedule and saves the active window, it just stops emitting the rows.
        prof.record_steps = False

    return prof


class Profiler(DistProfiler):
    # [CLASS DOC]
    # torch.profiler backend. Class-level _define_count ensures one continuous profiler per process and
    # _active_prof holds the process-global running profiler so step() from any instance advances it.
    """A PyTorch profiler wrapper class for collecting performance metrics.

    This profiler provides a convenient interface for profiling PyTorch operations,
    with support for:

    - CPU and CUDA activity profiling
    - Optional mini-batch-level scheduling of the update loop (wait/warmup/active steps)
    - Multi-rank profiling support
    - Chrome trace export

    Args:
        config: Configuration object containing profiling parameters
    """

    _define_count = 0
    # Process-global handle to the currently running torch profiler. torch.profiler is
    # process-wide, so a step() issued by one Profiler instance (e.g. the inner actor
    # TrainingWorker running the mini-batch loop) must advance the profiler that another
    # instance started (e.g. the outer ActorRolloutRefWorker), or opened for the update stage
    # in discrete mode.
    _active_prof = None

    def __init__(
        self,
        rank,
        config: ProfilerConfig,
        tool_config: Optional[TorchProfilerToolConfig] = None,
        save_file_prefix=None,
    ):
        # [FUNCTION DOC]
        # Behavior: defaults config to a disabled ProfilerConfig; asserts tool_config when enabled; stores rank,
        # config, tool_config, contents, save_path, discrete flag, and initializes schedule kwargs and profile
        # step to None.
        # Exceptions: AssertionError when enabled without tool_config.
        # KNOWN ISSUE: with tool_config None (disabled), reading tool_config.contents raises AttributeError.
        # Side effects: sets attributes.
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        config = config or ProfilerConfig(ranks=[], enable=False)
        self.save_file_prefix = save_file_prefix

        if not tool_config:
            assert not config.enable, "tool_config must be provided when profiler is enabled"

        self.prof = None
        self.rank = rank
        self.config = config
        self.tool_config = tool_config
        self.contents = self.tool_config.contents
        self.save_path = self.config.save_path
        # Align with other profilers: read discrete mode, default to False for torch profiler
        self.discrete = getattr(self.tool_config, "discrete", False)
        # Resolved torch.profiler.schedule kwargs for the active continuous run (None => record
        # the whole step). Only used in continuous mode; discrete mode resolves per stage.
        self._schedule_kwargs = None
        # RL step of the profiled window, reported by the trainer on start().
        self._profile_step = None

    def check(self):
        # [FUNCTION DOC]
        # Outputs: True if a continuous profiler instance has been created on this object.
        return self.prof is not None

    def _resolve_schedule_kwargs(self) -> Optional[dict]:
        # [FUNCTION DOC]
        # HELPER: Returns schedule kwargs (skip_first, wait, warmup, active, repeat as ints, None/0 treated as 0)
        # from tool_config.schedule, or None when schedule is absent or active <= 0.
        """Build the full torch.profiler.schedule kwargs from tool_config, or None to disable.

        This is the schedule as configured, used verbatim for the discrete update-stage trace
        where skip_first/wait/warmup/active/repeat all apply to mini-batches.
        """
        sched = getattr(self.tool_config, "schedule", None) if self.tool_config else None
        if sched is None:
            return None
        active = int(getattr(sched, "active", 0) or 0)
        if active <= 0:
            return None
        return {
            "skip_first": int(getattr(sched, "skip_first", 0) or 0),
            "wait": int(getattr(sched, "wait", 0) or 0),
            "warmup": int(getattr(sched, "warmup", 0) or 0),
            "active": active,
            "repeat": int(getattr(sched, "repeat", 0) or 0),
        }

    def _resolve_continuous_schedule_kwargs(self) -> Optional[dict]:
        # [FUNCTION DOC]
        # HELPER: Same as _resolve_schedule_kwargs but repeat 0 becomes 1 (single window per step); None passes through.
        """Schedule for the single continuous trace, or None to record the whole step.

        The schedule picks which update mini-batches land in the trace. ``skip_first``/``wait``/
        ``warmup`` are honored, so pointing it at a later mini-batch (e.g. ``skip_first: 1`` to
        capture the second one) records exactly that mini-batch instead of always starting from the
        first -- ``step()`` is advanced once per mini-batch, so a window boundary always falls
        between mini-batches and the captured ones stay intact.

        The cost of skipping ahead is that the stages before the active window -- the log-prob
        forwards and rollout that run before the update loop -- are not kept either, because torch
        only persists a window that ends in RECORD_AND_SAVE. Leaving ``skip_first``/``wait``/
        ``warmup`` at 0 (the default) records from the top of the step, so every earlier stage is
        kept in full plus the first ``active`` update mini-batches, exactly as before.

        ``repeat`` defaults to a single window so one step does not emit a file per cycle; set it
        explicitly to capture more than one active window within the step.
        """
        full = self._resolve_schedule_kwargs()
        if full is None:
            return None
        return {
            "skip_first": full["skip_first"],
            "wait": full["wait"],
            "warmup": full["warmup"],
            "active": full["active"],
            "repeat": full["repeat"] or 1,
        }

    def start(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: records profile_step (from profile_step or global_step kwarg) for later filenames. In
        # continuous mode and if no profiler is active process-wide, resolves continuous schedule, builds a
        # profiler (role from kwargs), prints a start message describing the capture scope, starts it, sets it as
        # the active process-global profiler and increments the counter.
        # Inputs: kwargs - role, profile_step/global_step. Side effects: starts profiling, class state, stdout.
        role = kwargs.get("role", None)
        # Recorded outside the discrete gate: discrete mode opens its profilers later, from
        # annotate(), and still needs to know which RL step it is collecting.
        profile_step = kwargs.get("profile_step", kwargs.get("global_step"))
        self._profile_step = profile_step
        if not self.discrete and Profiler._define_count == 0:
            self._schedule_kwargs = self._resolve_continuous_schedule_kwargs()
            self.prof = get_torch_profiler(
                contents=self.contents,
                save_path=self.save_path,
                role=role,
                save_file_prefix=self.save_file_prefix,
                rank=self.rank,
                profile_step=self._profile_step,
                schedule=self._schedule_kwargs,
            )
            if self._schedule_kwargs:
                sk = self._schedule_kwargs
                skipped = sk["skip_first"] + sk["wait"] + sk["warmup"]
                if skipped:
                    scope = (
                        f"capturing {sk['active']} update mini-batch(es) after the first {skipped} "
                        f"(earlier stages before that window are not kept)"
                    )
                else:
                    scope = f"keeping every stage in full plus the first {sk['active']} update mini-batch(es)"
                print(f"[Profiler] started for rank {self.rank}: {scope}")
            else:
                print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()
            Profiler._active_prof = self.prof
            Profiler._define_count += 1

    def step(self):
        # [FUNCTION DOC]
        # Behavior: calls step() on the process-global active profiler if any. Side effects: advances schedule,
        # may flush a trace window.
        """Advance the process-global active profiler by one update mini-batch.

        The actor/critic update loop calls this once per mini-batch. When a
        ``torch.profiler.schedule`` is configured it walks the wait/warmup/active state machine
        (and saves the active window); without one it does nothing meaningful. Either way no
        ``ProfilerStep#<n>`` row is written -- get_torch_profiler turns that labelling off, since
        a step() here is a mini-batch, not a meaningful RL step. The log-prob and rollout stages
        never call step(), so they are never sub-sampled.

        No-op when no torch profiler is currently running.
        """
        if Profiler._active_prof is not None:
            Profiler._active_prof.step()

    @staticmethod
    def _flush_partial_window(prof) -> None:
        # [FUNCTION DOC]
        # HELPER: If the profiler's current action is RECORD (and the enum has RECORD/RECORD_AND_SAVE), sets it to
        # RECORD_AND_SAVE so stopping writes the partially collected window. Relies on a private torch attribute.
        # Side effects: mutates profiler state.
        """Make torch write a collection window that the update loop ended in the middle of.

        torch only calls ``on_trace_ready`` once a window reaches its last (``RECORD_AND_SAVE``)
        mini-batch. An update loop with fewer mini-batches than ``wait + warmup + active`` stops
        while the schedule is still in ``RECORD``, and torch would drop everything collected.
        Promoting the pending action keeps that data: a short trace instead of no trace at all.
        """
        record = getattr(torch.profiler.ProfilerAction, "RECORD", None)
        record_and_save = getattr(torch.profiler.ProfilerAction, "RECORD_AND_SAVE", None)
        if record is None or record_and_save is None:
            return
        if getattr(prof, "current_action", None) is record:
            prof.current_action = record_and_save

    def stop(self):
        # [FUNCTION DOC]
        # Behavior: in continuous mode when the counter is 1: flushes a partial window if scheduled, else steps
        # once; prints, stops the profiler (triggering trace export), clears the active profiler and schedule,
        # decrements the counter.
        # KNOWN ISSUE: the counter is class-wide but self.prof is per instance; if a different instance than the
        # one that started calls stop, self.prof may be None.
        # Side effects: trace files written, class state.
        if not self.discrete and Profiler._define_count == 1:
            if self._schedule_kwargs:
                # Stepping is driven per mini-batch; flush a window the step ended mid-way.
                self._flush_partial_window(self.prof)
            else:
                # Close the last training step's window before tearing the profiler down.
                self.step()
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()
            Profiler._active_prof = None
            self._schedule_kwargs = None
            Profiler._define_count -= 1

    def annotate(self, message: Optional[str] = None, role: Optional[str] = None, **kwargs_outer) -> Callable:
        # [FUNCTION DOC]
        # Behavior: returns a decorator. "scheduled" kwarg marks update stages. Continuous mode: runs func inside
        # a record_function range named message, role, or func name. Discrete mode: builds a new profiler for the
        # stage (role or profile name; schedule only if scheduled, with mini-batch window naming), starts it, sets
        # it as active profiler when scheduled, runs func inside record_function, and in finally flushes a
        # partial window/clears the active profiler (if scheduled) and stops the profiler.
        # Inputs: message, role, kwargs_outer (scheduled). Outputs: decorator.
        # Exceptions: func's exceptions propagate after cleanup. Side effects: trace files.
        """Decorate a Worker member function to profile the current rank in the current training step.

        Requires the target function to be a member function of a Worker,
        which has a member field `profiler` with Profiler type.

        Args:
            message (str, optional):
                The message to be displayed in the profiler. Defaults to None.
            role (str, optional):
                The role of the current data collection. Defaults to None.
        """

        # Stages that iterate over update mini-batches (e.g. actor_update) mark themselves with
        # scheduled=True; only their trace is sub-sampled by the schedule, driven by the
        # per-mini-batch step() calls inside the update loop.
        scheduled = bool(kwargs_outer.get("scheduled", False))

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: Builds the functools.wraps wrapper for func.
            @functools.wraps(func)
            def wrapper(*args, **kwargs_inner):
                # [FUNCTION DOC]
                # HELPER: Per-call profiling logic described in annotate; returns func's result.
                # Prefer the stage label (`role`, e.g. "actor_compute_log_prob"), which names both
                # the role and the function it ran: in a colocated worker the method name alone
                # cannot say whether a log-prob forward belongs to the actor or the reference
                # model. Fall back to the method name for stages that declare no role.
                profile_name = message or role or func.__name__

                if not self.discrete:
                    # In continuous mode, we just record function, profiler started globally
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)

                # In discrete mode, we start/stop profiler around the function.
                # Only the update stage is scheduled (its mini-batches drive step()); every other
                # stage collects in full, so it must not carry a schedule -- a wait/warmup schedule
                # with no step() calls would start in NONE and record nothing.
                schedule_kwargs = self._resolve_schedule_kwargs() if scheduled else None
                # torch.profiler is process-global, so wrap the call in try/finally:
                # if func raises, we must still stop the profiler. Otherwise it leaks
                # and the next stage's prof.start() fails with "Profiler is already
                # enabled on this thread", plus the process aborts at teardown.
                prof = get_torch_profiler(
                    contents=self.contents,
                    save_path=self.save_path,
                    # Without an explicit role the stage is still identified by the wrapped
                    # function, which is what the reader needs to attribute the trace.
                    role=role or profile_name,
                    save_file_prefix=self.save_file_prefix,
                    rank=self.rank,
                    profile_step=self._profile_step,
                    schedule=schedule_kwargs,
                    name_mini_batch_window=schedule_kwargs is not None,
                )
                prof.start()
                # Expose the scheduled stage's profiler process-globally so the update loop's
                # per-mini-batch step() (issued from the inner TrainingWorker) advances it.
                if schedule_kwargs is not None:
                    Profiler._active_prof = prof
                try:
                    with torch.profiler.record_function(profile_name):
                        return func(*args, **kwargs_inner)
                finally:
                    if schedule_kwargs is not None:
                        self._flush_partial_window(prof)
                        Profiler._active_prof = None
                    prof.stop()

            return wrapper

        return decorator
