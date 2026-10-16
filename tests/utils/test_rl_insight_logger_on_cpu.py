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
# CPU tests for verl.utils.tracking.RLInsightLogger, a class-level facade over the optional rl_insight package. The
# package is replaced by a MagicMock via a patched _get_rl_insight. Covers env-gated no-op behavior, init with
# transfer-queue metrics registration, lazy init on log/trace_state (once even under concurrent async tasks),
# metric registration de-duplication, finish resetting state, and graceful fallback for old rl_insight versions
# lacking trace_span / agent_loop_session.

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from verl.utils.tracking import RLInsightLogger


@pytest.fixture(autouse=True)
def _reset_rl_insight_logger():
    # [FUNCTION DOC]
    # HELPER: Autouse fixture resetting RLInsightLogger's class-level state (init flag, cached module, registered
    # metrics, warned features) before and after each test.
    RLInsightLogger._init_done = False
    RLInsightLogger._rl_insight_module = None
    RLInsightLogger._registered_metrics.clear()
    RLInsightLogger._warned_unsupported_features.clear()
    yield
    RLInsightLogger._init_done = False
    RLInsightLogger._rl_insight_module = None
    RLInsightLogger._registered_metrics.clear()
    RLInsightLogger._warned_unsupported_features.clear()


@pytest.fixture
def mock_rl_insight(monkeypatch):
    # [FUNCTION DOC]
    # HELPER: Fixture returning a MagicMock rl_insight module whose trace_state acts as a no-op context manager, and
    # patches RLInsightLogger._get_rl_insight to return it.
    module = MagicMock()
    module.metric_gauge = MagicMock()

    @contextmanager
    def _trace_state(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: No-op generator-based context manager accepting any arguments.
        yield

    module.trace_state.side_effect = _trace_state
    monkeypatch.setattr(RLInsightLogger, "_get_rl_insight", classmethod(lambda cls: module))
    return module


def test_apis_are_noops_when_env_disabled(monkeypatch, mock_rl_insight):
    # [FUNCTION DOC]
    # With the enable env var unset, init/log/trace_state/register_metrics/finish are no-ops: no rl_insight call is made
    # and the init flag stays False.
    monkeypatch.delenv(RLInsightLogger.ENABLE_ENV, raising=False)

    RLInsightLogger.init(
        project_name="p", experiment_name="e", config={"transfer_queue": {"metrics": {"enabled": True}}}
    )
    RLInsightLogger.log({"reward/mean": 1.0}, step=1)
    with RLInsightLogger.trace_state("rollout", state_lane_id="replica_0"):
        pass
    RLInsightLogger.register_metrics(["127.0.0.1:8000"], "vllm", [{"replica": 0}])
    RLInsightLogger.finish()

    mock_rl_insight.init.assert_not_called()
    mock_rl_insight.metric_gauge.assert_not_called()
    mock_rl_insight.trace_state.assert_not_called()
    mock_rl_insight.update_prometheus_config.assert_not_called()
    mock_rl_insight.finish.assert_not_called()
    assert RLInsightLogger._init_done is False


def test_init_registers_transfer_queue_after_client_init(monkeypatch, mock_rl_insight):
    # [FUNCTION DOC]
    # With the env enabled and a fake transfer_queue module in sys.modules returning a metrics endpoint, init calls
    # rl_insight.init with the project, experiment name and trainer.rl_insight sub-config, then registers the transfer
    # queue endpoint for Prometheus with no labels; the init flag becomes True.
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    config = {
        "trainer": {"rl_insight": {"server": {"url": "http://127.0.0.1:18080"}}},
        "transfer_queue": {"metrics": {"enabled": True}},
    }
    fake_tq = MagicMock()
    fake_tq.get_metrics_endpoint.return_value = "127.0.0.1:9000"
    monkeypatch.setitem(__import__("sys").modules, "transfer_queue", fake_tq)

    RLInsightLogger.init(project_name="proj", experiment_name="exp", config=config)

    mock_rl_insight.init.assert_called_once_with(
        project="proj",
        experiment_name="exp",
        config=config["trainer"]["rl_insight"],
    )
    mock_rl_insight.update_prometheus_config.assert_called_once_with(["127.0.0.1:9000"], "transfer_queue", None)
    assert RLInsightLogger._init_done is True


def test_log_lazy_inits_and_writes_gauge(monkeypatch, mock_rl_insight):
    # [FUNCTION DOC]
    # With the env enabled, log without prior init lazily calls init with no arguments, writes a gauge for the numeric
    # metric with "/" replaced by "_", skips the non-numeric value, and sets the init flag.
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")

    RLInsightLogger.log({"reward/mean": 1.25, "bad": object()}, step=3)

    mock_rl_insight.init.assert_called_once_with()
    mock_rl_insight.metric_gauge.assert_called_once_with("reward_mean", 1.25)
    assert RLInsightLogger._init_done is True


def test_trace_state_lazy_inits_once_under_concurrent_async_tasks(monkeypatch, mock_rl_insight):
    # [FUNCTION DOC]
    # Runs 16 concurrent asyncio tasks each entering trace_state; asserts init was called exactly once, trace_state
    # 16 times, and the init flag is set.
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")

    async def _one(i: int):
        # [FUNCTION DOC]
        # HELPER: Enters a trace_state for index i and yields to the event loop once.
        with RLInsightLogger.trace_state(f"state_{i}", state_lane_id=f"lane_{i}", step=i):
            await asyncio.sleep(0)

    async def _run():
        # [FUNCTION DOC]
        # HELPER: Gathers the 16 tasks.
        await asyncio.gather(*[_one(i) for i in range(16)])

    asyncio.run(_run())

    mock_rl_insight.init.assert_called_once_with()
    assert mock_rl_insight.trace_state.call_count == 16
    assert RLInsightLogger._init_done is True


def test_register_metrics_dedups_and_finish_resets(monkeypatch, mock_rl_insight):
    # [FUNCTION DOC]
    # Registering the same addresses/kind/labels twice calls update_prometheus_config only once; finish then calls
    # rl_insight.finish, clears the init flag and the registered metrics set.
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    addresses = ["127.0.0.1:8000", "127.0.0.1:8001"]
    labels = [{"replica": 0}, {"replica": 1}]

    RLInsightLogger.register_metrics(addresses, "vllm", labels)
    RLInsightLogger.register_metrics(addresses, "vllm", labels)
    mock_rl_insight.update_prometheus_config.assert_called_once_with(addresses, "vllm", labels)

    RLInsightLogger._init_done = True
    RLInsightLogger.finish()
    mock_rl_insight.finish.assert_called_once_with()
    assert RLInsightLogger._init_done is False
    assert not RLInsightLogger._registered_metrics


def test_trace_span_warns_and_noops_with_old_rl_insight(monkeypatch, mock_rl_insight, caplog):
    # [FUNCTION DOC]
    # With an old version (0.2.1) lacking trace_span, trace_span logs a "does not support trace_span" warning and does
    # not initialize the client.
    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    mock_rl_insight.__version__ = "0.2.1"
    mock_rl_insight.trace_span = None

    with caplog.at_level("WARNING", logger="verl.utils.tracking"):
        RLInsightLogger.trace_span("agent_task", start_time_ns=1, end_time_ns=2)

    assert "RL-Insight does not support trace_span" in caplog.text
    mock_rl_insight.init.assert_not_called()


def test_agent_loop_session_falls_back_with_old_rl_insight(monkeypatch, mock_rl_insight, caplog):
    # [FUNCTION DOC]
    # With an old version, agent_loop_session falls back to a local session object whose identity contains a state lane
    # id built from experiment, sample, session and trajectory 0 and the global step; finish() on it works, and a
    # "does not support agent_loop_session" warning is logged.
    # Side effects: resets and re-initializes the global RolloutTraceConfig singleton (not reset afterwards).
    from verl.utils.rollout_trace import RolloutTraceConfig

    monkeypatch.setenv(RLInsightLogger.ENABLE_ENV, "1")
    mock_rl_insight.__version__ = "0.2.1"
    RolloutTraceConfig.reset()
    RolloutTraceConfig.init(project_name="project", experiment_name="experiment", backend=None)

    with caplog.at_level("WARNING", logger="verl.utils.tracking"):
        session = RLInsightLogger.agent_loop_session(
            sample=7,
            session=1,
            uid="uid",
            global_steps=3,
            session_id="session-id",
        )

    assert session.identity["state_lane_id"] == "experiment=experiment/sample=7/session=1/traj=0"
    assert session.identity["global_steps"] == 3
    session.finish(status="success", trajectories=[])
    assert "RL-Insight does not support agent_loop_session" in caplog.text
