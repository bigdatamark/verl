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
# CPU unit tests (torch profiler mocked) for the PyTorch profiler backend: the
# get_torch_profiler factory (activities, options, optional schedule, trace file
# naming and on_trace_ready export), the Profiler class lifecycle in continuous
# (one process-global collection shared across colocated roles, tracked via the
# class-level _define_count and _active_prof) and discrete (one profiler per
# annotated stage) modes, schedule resolution, partial-window flushing,
# build_trace_basename naming, DistProfiler delegation/fallback semantics, forced
# CPU activity, and conversion of per-role (ref, critic) OmegaConf profiler
# configs into a working torch backend.

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.profiler.config import (
    ProfilerConfig,
    TorchProfilerScheduleConfig,
    TorchProfilerToolConfig,
)
from verl.utils.profiler.profile import DistProfiler, _NoOpProfiler
from verl.utils.profiler.torch_profile import (
    Profiler,
    build_trace_basename,
    get_torch_profiler,
)


class TestTorchProfile(unittest.TestCase):
    # [CLASS DOC]
    # Core tests for get_torch_profiler, Profiler and DistProfiler with the torch tool.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Resets the Profiler class-level reference count and active profiler.
        # Reset process-global Profiler class state so tests don't leak into each other.
        Profiler._define_count = 0
        Profiler._active_prof = None

    def tearDown(self):
        # [FUNCTION DOC]
        # HELPER: Same reset as setUp so state does not leak between tests.
        Profiler._define_count = 0
        Profiler._active_prof = None

    @patch("torch.profiler.profile")
    def test_get_torch_profiler(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: contents cpu, cuda, stack yields a torch profile call with CPU and
        # CUDA activities, with_stack True, record_shapes and profile_memory False.
        # Inputs: mock_profile - patched torch.profiler.profile.
        # Test wrapper function
        get_torch_profiler(contents=["cpu", "cuda", "stack"], save_path="/tmp/test", rank=0)
        mock_profile.assert_called_once()
        _, kwargs = mock_profile.call_args

        # Verify activities
        activities = kwargs["activities"]
        self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
        self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

        # Verify options
        self.assertTrue(kwargs["with_stack"])
        self.assertFalse(kwargs["record_shapes"])
        self.assertFalse(kwargs["profile_memory"])

    @patch("torch.profiler.profile")
    def test_role_goes_in_filename_not_a_directory(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: with role "train" and prefix "actor" no subdirectory is created; the
        # on_trace_ready callback exports a chrome trace directly in save_path whose name
        # starts with "actor_train_" and ends with ".json.gz".
        # Side effects: uses a temporary directory.
        # The role must not create a directory level: one step's traces would be scattered
        # across sibling dirs and hidden from finish_hook_cmd, which only sees save_path.
        with tempfile.TemporaryDirectory() as save_path:
            get_torch_profiler(contents=["cpu"], save_path=save_path, role="train", save_file_prefix="actor", rank=0)
            # No "train" sub-directory is created next to the traces.
            self.assertEqual(os.listdir(save_path), [])

            _, kwargs = mock_profile.call_args
            mock_prof = MagicMock()
            kwargs["on_trace_ready"](mock_prof)

            (out_path,), _ = mock_prof.export_chrome_trace.call_args
            self.assertEqual(os.path.dirname(out_path), save_path)
            self.assertTrue(os.path.basename(out_path).startswith("actor_train_"))
            self.assertTrue(out_path.endswith(".json.gz"))

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_profiler_lifecycle(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: continuous mode start creates and starts the underlying profiler once,
        # step forwards one step, stop stops it once.
        # Inputs: mock_get_profiler - patched get_torch_profiler.
        # Mock the underlying torch profiler object
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        # Initialize
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # Test Start
        profiler.start()
        mock_get_profiler.assert_called_once()
        mock_prof_instance.start.assert_called_once()

        # Test Step
        profiler.step()
        mock_prof_instance.step.assert_called_once()

        # Test Stop
        profiler.stop()
        mock_prof_instance.stop.assert_called_once()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_start_forwards_training_step(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: start(role, profile_step=3) forwards profile_step 3 and role "train"
        # to get_torch_profiler.
        # The trainer reports the profiled step as start_profile(profile_step=...); it must reach
        # the filename, otherwise traces from different steps cannot be told apart.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        profiler.start(role="train", profile_step=3)
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["profile_step"], 3)
        self.assertEqual(kwargs["role"], "train")
        profiler.stop()

    @patch("torch.profiler.record_function")
    def test_continuous_annotate_range_names_role_and_function(self, mock_record_function):
        # [FUNCTION DOC]
        # Behavior: in continuous mode annotate wraps functions in a torch record_function
        # range named after the given role, or the function name when no role is given;
        # wrapped functions return their values.
        # Continuous mode puts every stage in one trace, so the range name has to carry the role as
        # well as the function: a bare "compute_log_prob" cannot say whether the forward belonged to
        # the actor or to the reference model colocated in the same process.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        @profiler.annotate(role="ref_compute_log_prob")
        def compute_log_prob():
            # [FUNCTION DOC]
            # HELPER: Dummy stage function wrapped by annotate; returns "done".
            return "done"

        @profiler.annotate()
        def train_batch():
            # [FUNCTION DOC]
            # HELPER: Dummy stage function wrapped by annotate; returns "done".
            return "done"

        self.assertEqual(compute_log_prob(), "done")
        mock_record_function.assert_called_with("ref_compute_log_prob")

        # A stage that declares no role is still named after the function it wraps.
        self.assertEqual(train_batch(), "done")
        mock_record_function.assert_called_with("train_batch")

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_annotate_names_stage_and_step(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: in discrete mode each annotated call opens its own profiler with role
        # set to the stage role (or function name if absent) and the profile_step recorded
        # by the earlier start (5).
        # Discrete mode opens one profiler per stage from annotate(), long after start(): the
        # stage name and the training step recorded at start() must both reach the filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)
        profiler.start(role="train", profile_step=5)

        @profiler.annotate(role="actor_update")
        def update_actor():
            # [FUNCTION DOC]
            # HELPER: Dummy stage function wrapped by annotate; returns "done".
            return "done"

        # An unlabeled stage falls back to the wrapped function's name.
        @profiler.annotate()
        def compute_log_prob():
            # [FUNCTION DOC]
            # HELPER: Dummy stage function wrapped by annotate; returns "done".
            return "done"

        self.assertEqual(update_actor(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["role"], "actor_update")
        self.assertEqual(kwargs["profile_step"], 5)

        self.assertEqual(compute_log_prob(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["role"], "compute_log_prob")

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_mode(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: in discrete mode start does not create a profiler and stop does not
        # stop one.
        # Mock for discrete mode
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        # In discrete mode, start/stop shouldn't trigger global profiler immediately
        profiler.start()
        mock_get_profiler.assert_not_called()

        profiler.stop()
        mock_prof_instance.stop.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_annotate_stops_profiler_on_exception(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: an exception raised in a discrete annotated stage propagates, but the
        # stage profiler was started and stopped once, and the body ran exactly once.
        # A stage raising inside a discrete-mode annotate must still stop the
        # (process-global) torch profiler; otherwise it leaks, the next stage's
        # start() fails with "Profiler is already enabled" and the process aborts.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        calls = {"n": 0}

        @profiler.annotate(role="boom")
        def boom():
            # [FUNCTION DOC]
            # HELPER: Stage body counting its calls and always raising RuntimeError.
            calls["n"] += 1
            raise RuntimeError("stage failed on purpose")

        with self.assertRaises(RuntimeError):
            boom()

        # Profiler must be started and, crucially, stopped despite the exception,
        # and the stage body must run exactly once (no re-execution).
        mock_prof_instance.start.assert_called_once()
        mock_prof_instance.stop.assert_called_once()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_propagates_and_runs_func_once(self):
        # [FUNCTION DOC]
        # Behavior: DistProfiler.annotate on a worker method with an active step and a
        # pass-through backend propagates the method's RuntimeError and runs it once.
        # DistProfiler.annotate must not swallow errors from the wrapped function
        # nor re-run it (which would execute the stage twice). It only falls back
        # when *setting up* backend profiling fails.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class PassthroughImpl:
            # [CLASS DOC]
            # HELPER: Fake backend whose annotate returns an identity decorator.
            def annotate(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns a decorator that returns the function unchanged. Inputs: any kwargs (ignored).
                def decorator(fn):
                    # [FUNCTION DOC]
                    # HELPER: Identity decorator. Inputs: fn. Outputs: fn.
                    return fn

                return decorator

        calls = {"n": 0}

        class FakeWorker:
            # [CLASS DOC]
            # HELPER: Minimal worker exposing profiler and rank attributes, as DistProfiler.annotate expects.
            def __init__(self, profiler):
                # [FUNCTION DOC]
                # HELPER: Stores the profiler and sets rank 0.
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="boom")
            def boom(self):
                # [FUNCTION DOC]
                # HELPER: Annotated method counting calls and raising RuntimeError.
                calls["n"] += 1
                raise RuntimeError("stage failed on purpose")

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = PassthroughImpl()
        dp._this_step = True  # simulate an active profiled step

        worker = FakeWorker(dp)
        with self.assertRaises(RuntimeError):
            worker.boom()
        self.assertEqual(calls["n"], 1)

    def test_dist_annotate_falls_back_when_setup_fails(self):
        # [FUNCTION DOC]
        # Behavior: if the backend annotate setup itself raises, DistProfiler.annotate
        # falls back to running the method unprofiled exactly once and returns its result.
        # If backend annotate setup raises, the function still runs (once), unprofiled.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=True)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )

        class BadImpl:
            # [CLASS DOC]
            # HELPER: Fake backend whose annotate always raises RuntimeError.
            def annotate(self, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Raises RuntimeError("cannot set up profiling").
                raise RuntimeError("cannot set up profiling")

        calls = {"n": 0}

        class FakeWorker:
            # [CLASS DOC]
            # HELPER: Minimal worker with profiler and rank.
            def __init__(self, profiler):
                # [FUNCTION DOC]
                # HELPER: Stores the profiler and sets rank 0.
                self.profiler = profiler
                self.rank = 0

            @DistProfiler.annotate(role="x")
            def do_work(self):
                # [FUNCTION DOC]
                # HELPER: Annotated method counting calls and returning "ok".
                calls["n"] += 1
                return "ok"

        dp = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dp._impl = BadImpl()
        dp._this_step = True

        worker = FakeWorker(dp)
        self.assertEqual(worker.do_work(), "ok")
        self.assertEqual(calls["n"], 1)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_no_schedule_collects_continuously(self, mock_profile, mock_schedule):
        # [FUNCTION DOC]
        # Behavior: without a schedule, torch.profiler.schedule is never called and no
        # schedule kwarg is passed to profile.
        # Without a schedule, collection runs from start() to stop() with no torch.profiler.schedule.
        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0)

        mock_schedule.assert_not_called()
        _, kwargs = mock_profile.call_args
        self.assertNotIn("schedule", kwargs)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_schedule_is_passed_through_when_provided(self, mock_profile, mock_schedule):
        # [FUNCTION DOC]
        # Behavior: a schedule dict is passed as keyword args to torch.profiler.schedule
        # and a schedule kwarg is given to profile.
        # A schedule dict is forwarded to torch.profiler.schedule and attached to the profiler.
        sched = {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 0}
        get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0, schedule=sched)

        mock_schedule.assert_called_once_with(**sched)
        _, kwargs = mock_profile.call_args
        self.assertIn("schedule", kwargs)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_schedule_suppresses_profiler_step_rows(self, mock_profile, mock_schedule):
        # [FUNCTION DOC]
        # Behavior: with a schedule the returned profiler has record_steps set to False.
        # A schedule sub-samples mini-batches, but verl advances step() per mini-batch (not per RL
        # step), so torch's "ProfilerStep#<n>" rows are turned off via record_steps=False.
        sched = {"skip_first": 0, "wait": 0, "warmup": 0, "active": 2, "repeat": 1}
        prof = get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0, schedule=sched)

        self.assertIs(prof.record_steps, False)

    @patch("torch.profiler.schedule")
    @patch("torch.profiler.profile")
    def test_no_schedule_leaves_record_steps_untouched(self, mock_profile, mock_schedule):
        # [FUNCTION DOC]
        # Behavior: without a schedule record_steps is not forced to False.
        # Without a schedule there are no step boundaries to label, so record_steps is left alone
        # (torch already defaults it to False).
        prof = get_torch_profiler(contents=["cpu"], save_path="/tmp/test", rank=0)

        self.assertIsNot(prof.record_steps, False)

    def test_continuous_schedule_follows_the_full_schedule(self):
        # [FUNCTION DOC]
        # Behavior: in continuous mode a full schedule (skip_first 3, wait 2, warmup 1,
        # active 2, repeat 5) resolves unchanged for both continuous and generic schedule
        # resolution.
        # In continuous mode the schedule selects which update mini-batch is captured, so
        # skip_first/wait/warmup are honored (an explicit repeat is kept as-is).
        tool_config = TorchProfilerToolConfig(
            contents=["cpu"],
            discrete=False,
            schedule=TorchProfilerScheduleConfig(skip_first=3, wait=2, warmup=1, active=2, repeat=5),
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        self.assertEqual(
            profiler._resolve_continuous_schedule_kwargs(),
            {"skip_first": 3, "wait": 2, "warmup": 1, "active": 2, "repeat": 5},
        )
        # The full schedule (used for the discrete update stage) matches.
        self.assertEqual(
            profiler._resolve_schedule_kwargs(),
            {"skip_first": 3, "wait": 2, "warmup": 1, "active": 2, "repeat": 5},
        )

    def test_continuous_schedule_defaults_record_from_start(self):
        # [FUNCTION DOC]
        # Behavior: with only active 2 set, the continuous schedule resolves to zeros for
        # skip_first/wait/warmup and repeat capped to 1.
        # With skip_first/wait/warmup left at 0, recording still starts at the top of the step
        # (keeping the earlier stages in full) and repeat is capped to a single window per step.
        tool_config = TorchProfilerToolConfig(
            contents=["cpu"],
            discrete=False,
            schedule=TorchProfilerScheduleConfig(active=2),
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        self.assertEqual(
            profiler._resolve_continuous_schedule_kwargs(),
            {"skip_first": 0, "wait": 0, "warmup": 0, "active": 2, "repeat": 1},
        )

    def test_schedule_disabled_when_active_not_positive(self):
        # [FUNCTION DOC]
        # Behavior: active 0 disables scheduling: both resolvers return None.
        tool_config = TorchProfilerToolConfig(
            contents=["cpu"], discrete=False, schedule=TorchProfilerScheduleConfig(active=0)
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)
        self.assertIsNone(profiler._resolve_schedule_kwargs())
        self.assertIsNone(profiler._resolve_continuous_schedule_kwargs())

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_update_stage_is_scheduled_and_drives_active_prof(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: in discrete mode a stage annotated with scheduled True gets the
        # resolved schedule and name_mini_batch_window True, and its profiler is the
        # process-global active profiler during the call and cleared afterwards.
        # In discrete mode only the update stage (scheduled=True) carries the schedule, and it must
        # expose its profiler process-globally so the inner update loop's step() advances it.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(
            contents=["cpu"],
            discrete=True,
            schedule=TorchProfilerScheduleConfig(skip_first=1, wait=1, warmup=1, active=2, repeat=0),
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)
        profiler.start(role="train", profile_step=5)

        active_prof_during_stage = {}

        @profiler.annotate(role="actor_update", scheduled=True)
        def update_actor():
            # [FUNCTION DOC]
            # HELPER: Records the class-level active profiler seen during the stage and returns "done".
            # The update loop steps the process-global profiler while the stage runs.
            active_prof_during_stage["prof"] = Profiler._active_prof
            return "done"

        self.assertEqual(update_actor(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["schedule"], {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 0})
        self.assertTrue(kwargs["name_mini_batch_window"])
        # The stage's profiler was the active one during the call, and is cleared afterwards.
        self.assertIs(active_prof_during_stage["prof"], mock_prof_instance)
        self.assertIsNone(Profiler._active_prof)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_discrete_non_update_stage_is_not_scheduled(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: a discrete stage without scheduled True gets schedule None,
        # name_mini_batch_window False, and never sets the active profiler.
        # A stage that does not iterate mini-batches (no scheduled=True) must collect in full: it
        # never calls step(), so a wait/warmup schedule would leave it recording nothing.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(
            contents=["cpu"],
            discrete=True,
            schedule=TorchProfilerScheduleConfig(skip_first=1, wait=1, warmup=1, active=2, repeat=0),
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)
        profiler.start(role="train", profile_step=5)

        @profiler.annotate(role="ref_compute_log_prob")
        def compute_ref_log_prob():
            # [FUNCTION DOC]
            # HELPER: Dummy stage function wrapped by annotate; returns "done".
            return "done"

        self.assertEqual(compute_ref_log_prob(), "done")
        _, kwargs = mock_get_profiler.call_args
        self.assertIsNone(kwargs["schedule"])
        self.assertFalse(kwargs["name_mini_batch_window"])
        # A non-scheduled stage never claims the process-global slot.
        self.assertIsNone(Profiler._active_prof)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_continuous_stop_flushes_partial_window_when_scheduled(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: continuous mode with a schedule: stop promotes a pending RECORD
        # action to RECORD_AND_SAVE (flushing the partial window), issues no trailing
        # step, and stops the profiler once.
        # With a schedule, stop() must not emit a trailing step() (that would advance the schedule);
        # it flushes any window the update loop ended mid-way instead.
        mock_prof_instance = MagicMock()
        # current_action left as a plain attribute so _flush_partial_window can inspect it.
        mock_prof_instance.current_action = torch.profiler.ProfilerAction.RECORD
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(
            contents=["cpu"], discrete=False, schedule=TorchProfilerScheduleConfig(active=3)
        )
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        profiler = Profiler(rank=0, config=config, tool_config=tool_config)

        profiler.start(role="train", profile_step=1)
        profiler.stop()

        # A pending RECORD window is promoted so torch saves it on stop() instead of dropping it,
        # and no trailing step() is issued.
        self.assertIs(mock_prof_instance.current_action, torch.profiler.ProfilerAction.RECORD_AND_SAVE)
        mock_prof_instance.step.assert_not_called()
        mock_prof_instance.stop.assert_called_once()

    @patch("torch.profiler.profile")
    def test_scheduled_files_are_named_by_mini_batch_window(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: with name_mini_batch_window True and the schedule skip 1/wait 1/warmup 1,
        # a flush at step_num 5 exports a file whose name contains "_mb3-4".
        # A discrete scheduled file holds one window of mini-batches, so its name states which.
        sched = {"skip_first": 1, "wait": 1, "warmup": 1, "active": 2, "repeat": 0}
        with tempfile.TemporaryDirectory() as save_path:
            get_torch_profiler(
                contents=["cpu"], save_path=save_path, rank=0, schedule=sched, name_mini_batch_window=True
            )
            on_trace_ready = mock_profile.call_args[1]["on_trace_ready"]

            # First window flushed at step_num=5 -> recorded mini-batches 3 and 4.
            mock_prof = MagicMock()
            mock_prof.step_num = 5
            on_trace_ready(mock_prof)
            (out_path,), _ = mock_prof.export_chrome_trace.call_args
            self.assertIn("_mb3-4", os.path.basename(out_path))

    @patch("torch.profiler.profile")
    def test_second_flush_does_not_overwrite_the_first(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: two exports from the same profiler: the first name has no "_part"
        # suffix and the second ends with "_part1.json.gz".
        # Both windows share one filename stem, so the second needs a suffix of its own.
        with tempfile.TemporaryDirectory() as save_path:
            get_torch_profiler(contents=["cpu"], save_path=save_path, rank=0)
            on_trace_ready = mock_profile.call_args[1]["on_trace_ready"]

            names = []
            for _ in range(2):
                mock_prof = MagicMock()
                on_trace_ready(mock_prof)
                (out_path,), _ = mock_prof.export_chrome_trace.call_args
                names.append(os.path.basename(out_path))

            self.assertNotIn("_part", names[0])
            self.assertTrue(names[1].endswith("_part1.json.gz"), names[1])

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_colocated_roles_share_one_collection(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: two continuous Profiler instances in one process: only the first
        # start creates a collection (which becomes the active profiler); step from
        # either advances it (2 steps); stop on one emits a trailing step (3), stops the
        # collection once and clears the active profiler.
        # A colocated actor and reference model each own a Profiler but share one process-global
        # collection: only the first start() opens it, and step() from either advances that one
        # profiler (the actor update loop is what actually drives step() per mini-batch).
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(save_path="/tmp/test", enable=True, tool_config=tool_config)
        actor = Profiler(rank=0, config=config, tool_config=tool_config)
        colocated_ref = Profiler(rank=0, config=config, tool_config=tool_config)

        actor.start(role="train", profile_step=1)
        colocated_ref.start(profile_step=1)
        # One collection for the process, opened by the first role to start.
        mock_get_profiler.assert_called_once()
        self.assertIs(Profiler._active_prof, mock_prof_instance)

        # Any Profiler instance advances the shared process-global collection (per mini-batch).
        actor.step()
        colocated_ref.step()
        self.assertEqual(mock_prof_instance.step.call_count, 2)

        # Without a schedule, stop() emits a trailing step() to flush, then tears the collection down.
        actor.stop()
        self.assertEqual(mock_prof_instance.step.call_count, 3)
        mock_prof_instance.stop.assert_called_once()
        self.assertIsNone(Profiler._active_prof)

    def test_build_trace_basename_encodes_role_rank_and_parallelism(self):
        # [FUNCTION DOC]
        # Behavior: the basename starts with "actor_train_" and contains
        # "rank5-of-16", "tp1-pp0-dp2-cp0" and "pid<current pid>".
        # Filename stem must embed the worker role, scope role, rank/world size and the
        # tp/pp/dp/cp parallel ranks so per-process traces are self-describing.
        name = build_trace_basename(
            rank=5,
            role="train",
            save_file_prefix="actor",
            topology={"rank": 5, "world_size": 16, "tp": 1, "pp": 0, "dp": 2, "cp": 0},
        )
        self.assertTrue(name.startswith("actor_train_"))
        self.assertIn("rank5-of-16", name)
        self.assertIn("tp1-pp0-dp2-cp0", name)
        self.assertIn(f"pid{os.getpid()}", name)

    def test_build_trace_basename_encodes_training_step(self):
        # [FUNCTION DOC]
        # Behavior: with profile_step 7 the basename starts "actor_train_step7_rank0-of-8_".
        # Without the step, traces from different profiled steps are only distinguishable by
        # their wall-clock timestamp, which is unreadable when several steps are profiled.
        name = build_trace_basename(
            rank=0, role="train", save_file_prefix="actor", profile_step=7, topology={"rank": 0, "world_size": 8}
        )
        self.assertTrue(name.startswith("actor_train_step7_rank0-of-8_"))

    def test_build_trace_basename_omits_step_when_unknown(self):
        # [FUNCTION DOC]
        # Behavior: without a step and empty topology the name starts "actor_train_rank0_"
        # and contains no "step".
        name = build_trace_basename(rank=0, role="train", save_file_prefix="actor", topology={})
        self.assertTrue(name.startswith("actor_train_rank0_"))
        self.assertNotIn("step", name)

    def test_build_trace_basename_distinguishes_roles_same_rank(self):
        # [FUNCTION DOC]
        # Behavior: same rank with prefixes "ref" vs "value_model" yields distinct names;
        # underscores in labels become hyphens ("value-model_...").
        # The original scheme collided ref/critic at the same rank; the role prefix fixes it.
        topo = {"rank": 5, "world_size": 16}
        ref_name = build_trace_basename(rank=5, save_file_prefix="ref", topology=topo)
        critic_name = build_trace_basename(rank=5, save_file_prefix="value_model", topology=topo)
        self.assertTrue(ref_name.startswith("ref_rank5-of-16_"))
        # Underscores in labels are normalized to hyphens (underscore is the field separator).
        self.assertTrue(critic_name.startswith("value-model_rank5-of-16_"))
        self.assertNotEqual(ref_name, critic_name)

    def test_build_trace_basename_minimal_topology(self):
        # [FUNCTION DOC]
        # Behavior: with only rank and empty topology the name starts "rank3_", has no
        # "-of-" and no parallel-dim fields.
        # With no distributed topology, fall back to the passed rank and omit parallel dims.
        name = build_trace_basename(rank=3, topology={})
        self.assertTrue(name.startswith("rank3_"))
        self.assertNotIn("-of-", name)
        for dim in ("tp", "pp", "dp", "cp"):
            self.assertNotIn(f"{dim}0", name)

    def test_build_trace_basename_sanitizes_labels(self):
        # [FUNCTION DOC]
        # Behavior: slashes and spaces in role and prefix are replaced with hyphens.
        # Slashes/spaces in labels must not leak into the filename.
        name = build_trace_basename(rank=0, role="update actor", save_file_prefix="actor/rollout", topology={})
        self.assertNotIn("/", name)
        self.assertNotIn(" ", name)
        self.assertIn("actor-rollout", name)
        self.assertIn("update-actor", name)

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_forwards_save_file_prefix(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: DistProfiler stores save_file_prefix on its torch implementation and
        # forwards it to get_torch_profiler on start.
        # DistProfiler must forward save_file_prefix down to the torch backend so it
        # ends up in the trace filename.
        mock_get_profiler.return_value = MagicMock()
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config, save_file_prefix="actor")
        self.assertEqual(dist_profiler._impl.save_file_prefix, "actor")

        dist_profiler.start()
        _, kwargs = mock_get_profiler.call_args
        self.assertEqual(kwargs["save_file_prefix"], "actor")
        dist_profiler.stop()

    def test_dist_profiler_step_noop_backend(self):
        # [FUNCTION DOC]
        # Behavior: with no tool (no-op backend) DistProfiler.step returns None safely.
        # A backend without scheduling support (no-op impl) must make step() a safe no-op.
        config = ProfilerConfig(tool=None, enable=True, all_ranks=True, save_path="/tmp/test", tool_config=None)
        dist_profiler = DistProfiler(rank=0, config=config)
        self.assertIsNone(dist_profiler.step())

    def test_dist_profiler_step_disabled(self):
        # [FUNCTION DOC]
        # Behavior: with profiling disabled, step returns None and never calls the
        # backend step.
        # When disabled, step() must not touch the backend at all.
        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=False, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        dist_profiler._impl = MagicMock()
        self.assertIsNone(dist_profiler.step())
        dist_profiler._impl.step.assert_not_called()

    @patch("verl.utils.profiler.torch_profile.get_torch_profiler")
    def test_dist_profiler_step_torch_delegates(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: with torch tool enabled, start then step calls the underlying
        # profiler step exactly once.
        mock_prof_instance = MagicMock()
        mock_get_profiler.return_value = mock_prof_instance

        tool_config = TorchProfilerToolConfig(contents=["cpu"], discrete=False)
        config = ProfilerConfig(
            tool="torch", enable=True, all_ranks=True, save_path="/tmp/test", tool_config=tool_config
        )
        dist_profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)

        dist_profiler.start()
        dist_profiler.step()
        mock_prof_instance.step.assert_called_once()

        dist_profiler.stop()


class TestCpuActivityAlwaysCollected(unittest.TestCase):
    # [CLASS DOC]
    # Tests that CPU activity is always collected regardless of configured contents.
    """Stage markers are CPU-side events, so a device-only trace hides every verl stage.

    ``contents`` is left exactly as the user wrote it; CPU activity is enabled when the profiler is
    built, so no configuration can produce a trace of unattributable kernels.
    """

    @patch("torch.profiler.profile")
    def test_cpu_activity_added_for_device_only_contents(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: for contents ["cuda"], ["cuda","memory"] and [], the activities passed
        # to profile include both CPU and CUDA.
        for contents in (["cuda"], ["cuda", "memory"], []):
            with self.subTest(contents=contents):
                mock_profile.reset_mock()
                get_torch_profiler(contents=list(contents), save_path="/tmp/test", rank=0)
                activities = mock_profile.call_args[1]["activities"]
                self.assertIn(torch.profiler.ProfilerActivity.CPU, activities)
                self.assertIn(torch.profiler.ProfilerActivity.CUDA, activities)

    @patch("torch.profiler.profile")
    def test_contents_are_not_rewritten(self, mock_profile):
        # [FUNCTION DOC]
        # Behavior: building a profiler does not mutate the user's contents list or the
        # tool config's contents.
        # The user's selection is theirs: enabling the activity must not edit the config.
        contents = ["cuda"]
        tool_config = TorchProfilerToolConfig(contents=contents)
        get_torch_profiler(contents=tool_config.contents, save_path="/tmp/test", rank=0)
        self.assertEqual(tool_config.contents, ["cuda"])
        self.assertEqual(contents, ["cuda"])


def _role_profiler_omegaconf(tool="torch", enable=True, discrete=False, contents=("cpu", "cuda")):
    # [FUNCTION DOC]
    # HELPER: Builds an OmegaConf per-role profiler sub-tree with _target_ for
    # ProfilerConfig and a nested torch TorchProfilerToolConfig, rank 0 only, save path
    # /tmp/test_role_profile.
    # Inputs: tool - default "torch"; enable; discrete; contents - default cpu, cuda.
    # Outputs: an OmegaConf DictConfig.
    """Mimic a per-role ``profiler`` OmegaConf sub-tree (identical across ref/ref.yaml and
    critic/critic.yaml).

    The nested ``_target_`` entries are what the hydra instantiation path (omega_conf_to_dataclass
    without an explicit dataclass_type) uses to build real dataclass tool configs, as opposed to the
    plain dicts the torch profiler cannot consume.
    """
    return OmegaConf.create(
        {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": tool,
            "enable": enable,
            "all_ranks": False,
            "ranks": [0],
            "save_path": "/tmp/test_role_profile",
            "tool_config": {
                "torch": {
                    "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                    "contents": list(contents),
                    "discrete": discrete,
                },
            },
        }
    )


class TestRefWorkerProfilerConfig(unittest.TestCase):
    # [CLASS DOC]
    # Tests ref-worker profiler config conversion.
    """The reference model's inner TrainingWorker must receive a real, torch-consumable profiler
    config (mirroring the actor), instead of silently running with a disabled no-op profiler.

    ``ActorRolloutRefWorker.init_model`` now forwards the ref's own ``profiler`` config to the ref
    ``TrainingWorkerConfig`` via ``omega_conf_to_dataclass(self.config.ref.get("profiler", {}))``.
    These lock in that conversion path so a torch profiler config actually yields a torch backend on
    the ref worker, while an absent config degrades to a no-op (the previous ref behavior).
    """

    def test_ref_profiler_config_builds_torch_backend(self):
        # [FUNCTION DOC]
        # Behavior: converting the role sub-tree without an explicit dataclass type gives
        # an enabled torch ProfilerConfig whose tool_config entry for "torch" is a real
        # TorchProfilerToolConfig (contents cpu, cuda); a DistProfiler built from it uses
        # the torch Profiler backend and is enabled for rank 0.
        # Exercise the exact conversion init_model performs on actor_rollout_ref.ref.profiler:
        # omega_conf_to_dataclass(...) (no dataclass_type) must resolve the _target_ entries into
        # real nested dataclasses the torch Profiler can consume via attribute access.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        ref_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(ref_profiler_config, ProfilerConfig)
        self.assertTrue(ref_profiler_config.enable)
        self.assertEqual(ref_profiler_config.tool, "torch")

        # TrainingWorker.__init__ extracts the tool-specific config exactly like this; it must be a
        # real dataclass (not a plain dict) for the torch Profiler to read .contents/.discrete.
        tool_config = ref_profiler_config.tool_config.get(ref_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)
        self.assertEqual(tool_config.contents, ["cpu", "cuda"])

        dist_profiler = DistProfiler(
            rank=0, config=ref_profiler_config, tool_config=tool_config, save_file_prefix="ref"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())

    def test_absent_ref_profiler_config_is_disabled_noop(self):
        # [FUNCTION DOC]
        # Behavior: DistProfiler with config None uses the no-op backend and is disabled.
        # Contrast with the previous behavior: without a profiler_config the ref worker built a
        # disabled no-op profiler, so the reference model was never profiled by its own worker.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


class TestCriticWorkerProfilerConfig(unittest.TestCase):
    # [CLASS DOC]
    # Tests critic-worker profiler config conversion.
    """The critic is a standalone TrainingWorker (no outer ActorRolloutRefWorker wrapper): the
    trainer drives start_profile()/stop_profile() and the ``train_batch`` annotation directly on it.

    ``RayPPOTrainer._init_workers`` (and the v1 / separation trainer variants) now forward
    ``omega_conf_to_dataclass(self.config.critic.get("profiler", {}))`` into the critic
    ``TrainingWorkerConfig``. Without it the critic's DistProfiler silently degraded to a no-op, so
    the critic was never profiled by any backend. These lock in that wiring.
    """

    def test_critic_profiler_config_builds_torch_backend(self):
        # [FUNCTION DOC]
        # Behavior: same conversion as the ref test; the DistProfiler with prefix
        # "value_model" uses the torch backend, is enabled for rank 0, and stores the prefix.
        # critic/critic.yaml's profiler block is structurally identical to ref/ref.yaml; the trainer
        # converts it the same way. It must yield a real torch backend on the standalone critic worker.
        omega_cfg = _role_profiler_omegaconf(tool="torch", enable=True)
        critic_profiler_config = omega_conf_to_dataclass(omega_cfg)

        self.assertIsInstance(critic_profiler_config, ProfilerConfig)
        tool_config = critic_profiler_config.tool_config.get(critic_profiler_config.tool)
        self.assertIsInstance(tool_config, TorchProfilerToolConfig)

        # The critic TrainingWorker uses model_type="value_model" as the trace filename prefix.
        dist_profiler = DistProfiler(
            rank=0, config=critic_profiler_config, tool_config=tool_config, save_file_prefix="value_model"
        )
        self.assertIsInstance(dist_profiler._impl, Profiler)
        self.assertTrue(dist_profiler.check_enable())
        self.assertTrue(dist_profiler.check_this_rank())
        self.assertEqual(dist_profiler._impl.save_file_prefix, "value_model")

    def test_absent_critic_profiler_config_is_disabled_noop(self):
        # [FUNCTION DOC]
        # Behavior: DistProfiler with config None is a disabled no-op.
        # The previous behavior: the trainer built the critic TrainingWorkerConfig without a
        # profiler_config, so DistProfiler(config=None) degraded to a disabled no-op.
        dist_profiler = DistProfiler(rank=0, config=None)
        self.assertIsInstance(dist_profiler._impl, _NoOpProfiler)
        self.assertFalse(dist_profiler.check_enable())


if __name__ == "__main__":
    unittest.main()
