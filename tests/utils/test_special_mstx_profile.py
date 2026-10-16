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
# Unit tests (CPU, mocks the NPU backend) for the Ascend NPU profiler (NPUProfiler)
# as created through the DistProfiler facade with tool "npu": enable/rank
# selection, start/stop with the class-level shared _define_count reference count
# for non-discrete (global) profiling, the no-op per-mini-batch step hook, and the
# annotate decorator emitting mstx ranges and, in discrete mode, a per-call
# profiler session. get_npu_profiler, mark_start_range and mark_end_range in the
# mstx_profile module are patched.

import unittest
from unittest.mock import MagicMock, patch

from verl.utils.profiler.config import NPUToolConfig, ProfilerConfig
from verl.utils.profiler.mstx_profile import NPUProfiler
from verl.utils.profiler.profile import DistProfiler


class TestNPUProfilerInitialization(unittest.TestCase):
    # [CLASS DOC]
    # Tests profiler enablement and rank selection at construction.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Resets the class-wide NPUProfiler._define_count to 0.
        NPUProfiler._define_count = 0

    def test_init_with_default_config(self):
        # [FUNCTION DOC]
        # Behavior: default NPU tool config and ProfilerConfig(tool npu) means profiling
        # is disabled (check_enable is False).
        tool_config = NPUToolConfig()
        config = ProfilerConfig(tool="npu")
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_enable())

    def test_init_with_disabled_config(self):
        # [FUNCTION DOC]
        # Behavior: explicit enable False yields check_enable False.
        config = ProfilerConfig(enable=False, tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_enable())

    def test_init_with_all_ranks_true(self):
        # [FUNCTION DOC]
        # Behavior: enable with all_ranks True selects rank 0 (check_this_rank True).
        config = ProfilerConfig(enable=True, all_ranks=True, tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=0, config=config, tool_config=tool_config)
        self.assertTrue(profiler.check_this_rank())

    def test_init_with_ranks_list(self):
        # [FUNCTION DOC]
        # Behavior: enable with ranks [1, 2] selects rank 1.
        config = ProfilerConfig(enable=True, ranks=[1, 2], tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=1, config=config, tool_config=tool_config)
        self.assertTrue(profiler.check_this_rank())

    def test_init_with_rank_not_in_ranks(self):
        # [FUNCTION DOC]
        # Behavior: enable with ranks [1, 2] does not select rank 3.
        config = ProfilerConfig(enable=True, ranks=[1, 2], tool="npu")
        tool_config = NPUToolConfig()
        profiler = DistProfiler(rank=3, config=config, tool_config=tool_config)
        self.assertFalse(profiler.check_this_rank())


class TestNPUProfilerStart(unittest.TestCase):
    # [CLASS DOC]
    # Tests start() behavior for the global (non-discrete) NPU profiler on rank 0.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Resets _define_count and builds an enabled config for ranks [0] and a
        # non-discrete NPU tool config.
        NPUProfiler._define_count = 0
        self.config = ProfilerConfig(enable=True, ranks=[0], tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_when_enabled_and_this_rank(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: start with role and step on a selected rank marks the step active,
        # increments _define_count to 1 and creates the backend profiler once.
        # Inputs: mock_get_profiler - patched get_npu_profiler.
        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start(role="worker", profile_step="1")
        self.assertTrue(profiler.check_this_step())
        self.assertEqual(NPUProfiler._define_count, 1)
        mock_get_profiler.assert_called_once()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_when_not_this_rank(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: start on an unselected rank (1) leaves the step inactive, count 0, and
        # never creates a backend profiler.
        profiler = DistProfiler(rank=1, config=self.config, tool_config=self.tool_config)
        profiler.start()
        self.assertFalse(profiler.check_this_step())
        self.assertEqual(NPUProfiler._define_count, 0)
        mock_get_profiler.assert_not_called()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_discrete_mode_does_not_increase_count(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: in discrete mode start does not create a global profiler and leaves
        # the count at 0.
        tool_config = NPUToolConfig(discrete=True)
        profiler = DistProfiler(rank=0, config=self.config, tool_config=tool_config)
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 0)
        mock_get_profiler.assert_not_called()

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_multiple_start_calls_do_not_increase_count(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: calling start twice on the same instance keeps the count at 1 and
        # creates the backend once.
        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start()
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        mock_get_profiler.assert_called_once()


class TestNPUProfilerStartStopInteraction(unittest.TestCase):
    # [CLASS DOC]
    # Tests start/stop pairing and the shared reference count across instances.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Same fixture as TestNPUProfilerStart.setUp.
        NPUProfiler._define_count = 0
        self.config = ProfilerConfig(enable=True, ranks=[0], tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_start_stop_cycle(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: start calls backend start once and sets count 1; stop resets count to
        # 0 and calls backend step and stop once each.
        mock_profile_npu = MagicMock()
        mock_get_profiler.return_value = mock_profile_npu

        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        self.assertEqual(mock_profile_npu.start.call_count, 1)
        profiler.stop()
        self.assertEqual(NPUProfiler._define_count, 0)
        self.assertEqual(mock_profile_npu.step.call_count, 1)
        self.assertEqual(mock_profile_npu.stop.call_count, 1)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_multiple_instances_share_define_count(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: two profiler instances on the same rank share the class count; the
        # second start does not start the backend again (count stays 1, start called
        # once), and stopping the first resets count to 0.
        mock_profile_npu = MagicMock()
        mock_get_profiler.return_value = mock_profile_npu

        profiler1 = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler2 = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler1.start()
        profiler2.start()
        self.assertEqual(NPUProfiler._define_count, 1)
        self.assertEqual(mock_profile_npu.start.call_count, 1)
        profiler1.stop()
        self.assertEqual(NPUProfiler._define_count, 0)


class TestNPUProfilerStep(unittest.TestCase):
    # [CLASS DOC]
    # Regression tests for DistProfiler.step() delegating to the NPU implementation.
    """Regression tests for the per-mini-batch ``step()`` hook.

    ``DistProfiler.step()`` delegates to ``self._impl.step()``. Because ``NPUProfiler``
    subclasses ``DistProfiler`` without running its ``__init__``, a missing ``step()``
    override used to resolve to the inherited ``DistProfiler.step`` and crash with
    ``AttributeError: 'NPUProfiler' object has no attribute '_enable'``.
    """

    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Same fixture as TestNPUProfilerStart.setUp.
        NPUProfiler._define_count = 0
        self.config = ProfilerConfig(enable=True, ranks=[0], tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)

    def test_step_is_noop_and_does_not_raise(self):
        # [FUNCTION DOC]
        # Behavior: step without start must not raise and leaves count 0. Not patched, so
        # relies on step not touching the backend.
        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        # Must be a clean no-op (no active profiler started) and must not raise.
        profiler.step()
        self.assertEqual(NPUProfiler._define_count, 0)

    @patch("verl.utils.profiler.mstx_profile.get_npu_profiler")
    def test_step_after_start_does_not_drive_backend(self, mock_get_profiler):
        # [FUNCTION DOC]
        # Behavior: after start, step does not call backend step; a subsequent stop still
        # calls backend step and stop exactly once.
        mock_profile_npu = MagicMock()
        mock_get_profiler.return_value = mock_profile_npu

        profiler = DistProfiler(rank=0, config=self.config, tool_config=self.tool_config)
        profiler.start(role="worker")
        profiler.step()
        # NPU profiler has no step schedule, so step() must not advance the backend.
        mock_profile_npu.step.assert_not_called()
        profiler.stop()
        # stop() still emits the single trailing step/stop on the backend.
        mock_profile_npu.step.assert_called_once()
        mock_profile_npu.stop.assert_called_once()


class TestNPUProfilerAnnotate(unittest.TestCase):
    # [CLASS DOC]
    # Tests the annotate decorator under various enable/step/discrete settings.
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Builds an all-ranks enabled config, non-discrete tool config, rank 0.
        self.config = ProfilerConfig(enable=True, all_ranks=True, tool="npu")
        self.tool_config = NPUToolConfig(discrete=False)
        self.rank = 0

    def test_annotate_decorator_applied_correctly(self):
        # [FUNCTION DOC]
        # Behavior: with the step forced active, an annotated function returns its value,
        # mark_start_range is called once with the given message, mark_end_range once with
        # the returned handle, and no backend profiler is created (non-discrete).
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        # Manually set private attribute for testing annotation in active step
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
        ):
            mock_start_patch.return_value = mock_mark_range

            with patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler:
                decorator = mock_worker.profiler.annotate(message="test")

                @decorator
                def test_func(self, *args, **kwargs):
                    # [FUNCTION DOC]
                    # HELPER: Dummy worker method wrapped by the annotate decorator; ignores its arguments and returns the string "result".
                    return "result"

                result = test_func(mock_worker)

                self.assertEqual(result, "result")
                mock_start_patch.assert_called_once_with(message="test")
                mock_end_patch.assert_called_once_with(mock_mark_range)
                mock_get_profiler.assert_not_called()

    def test_annotate_when_profiler_disabled(self):
        # [FUNCTION DOC]
        # Behavior: with profiling disabled, the annotated function runs and no range or
        # profiler calls happen.
        disabled_config = ProfilerConfig(enable=False, tool="npu")
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=disabled_config, tool_config=self.tool_config)

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            decorator = mock_worker.profiler.annotate(message="test")

            @decorator
            def test_func(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Dummy worker method wrapped by the annotate decorator; ignores its arguments and returns the string "result".
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_not_called()
            mock_end_patch.assert_not_called()
            mock_get_profiler.assert_not_called()

    def test_annotate_when_this_step_disabled(self):
        # [FUNCTION DOC]
        # Behavior: with the step flag forced inactive, no range or profiler calls happen.
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        mock_worker.profiler._this_step = False

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            decorator = mock_worker.profiler.annotate(message="test")

            @decorator
            def test_func(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Dummy worker method wrapped by the annotate decorator; ignores its arguments and returns the string "result".
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_not_called()
            mock_end_patch.assert_not_called()
            mock_get_profiler.assert_not_called()

    def test_annotate_discrete_mode_enabled(self):
        # [FUNCTION DOC]
        # Behavior: in discrete mode with an active step, annotate with a role marks the
        # range and creates a per-call backend profiler with the implementation's
        # contents, level, save path, analysis flag and the given role, then calls its
        # start, step and stop once each.
        discrete_tool_config = NPUToolConfig(discrete=True)
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=discrete_tool_config)
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"
        mock_profile_npu = MagicMock()

        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
            patch("verl.utils.profiler.mstx_profile.get_npu_profiler") as mock_get_profiler,
        ):
            mock_start_patch.return_value = mock_mark_range
            mock_get_profiler.return_value = mock_profile_npu
            decorator = mock_worker.profiler.annotate(message="test", role="test_role")

            @decorator
            def test_func(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Dummy worker method wrapped by the annotate decorator; ignores its arguments and returns the string "result".
                return "result"

            result = test_func(mock_worker)

            self.assertEqual(result, "result")
            mock_start_patch.assert_called_once_with(message="test")
            mock_end_patch.assert_called_once_with(mock_mark_range)
            mock_get_profiler.assert_called_once_with(
                contents=mock_worker.profiler._impl.profile_contents,
                profile_level=mock_worker.profiler._impl.profile_level,
                profile_save_path=mock_worker.profiler._impl.profile_save_path,
                analysis=mock_worker.profiler._impl.analysis,
                role="test_role",
            )
            mock_profile_npu.start.assert_called_once()
            mock_profile_npu.step.assert_called_once()
            mock_profile_npu.stop.assert_called_once()

    def test_annotate_with_default_message(self):
        # [FUNCTION DOC]
        # Behavior: annotate without a message uses the wrapped function's name
        # ("test_func") as the range message.
        mock_worker = MagicMock()
        mock_worker.profiler = DistProfiler(rank=self.rank, config=self.config, tool_config=self.tool_config)
        mock_worker.profiler._this_step = True

        mock_mark_range = "mocked_range_handle"
        with (
            patch("verl.utils.profiler.mstx_profile.mark_start_range") as mock_start_patch,
            patch("verl.utils.profiler.mstx_profile.mark_end_range") as mock_end_patch,
        ):
            mock_start_patch.return_value = mock_mark_range
            decorator = mock_worker.profiler.annotate()

            @decorator
            def test_func(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Dummy worker method wrapped by the annotate decorator; ignores its arguments and returns the string "result".
                return "result"

            test_func(mock_worker)

            mock_start_patch.assert_called_once_with(message="test_func")
            mock_end_patch.assert_called_once_with(mock_mark_range)


if __name__ == "__main__":
    unittest.main()
