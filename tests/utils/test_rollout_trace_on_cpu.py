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
# CPU tests for verl.utils.rollout_trace: the rollout_trace_op decorator, RolloutTraceConfig singleton and the
# rollout_trace_attr context manager across weave, mlflow, trackio and dummy backends (each mocked through
# sys.modules), including token-to-text decoding, chat-message traces, trace disabling (also for nested ops and
# after exceptions), plus opt-in integration tests against real weave / mlflow backends. Test functions are async
# and rely on the project's pytest asyncio configuration to run them.

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op


@pytest.fixture(autouse=True)
def reset_rollout_trace_config_singleton():
    # [FUNCTION DOC]
    # HELPER: Autouse fixture resetting the RolloutTraceConfig singleton before and after each test, so a configured
    # backend never leaks into other tests in the same process.
    """Reset the RolloutTraceConfig singleton around each test.

    Resetting afterwards matters as much as before: the whole CPU suite runs in one
    pytest process, so a leftover ``backend="weave"`` would make every later
    ``@rollout_trace_op`` call in any test file import the real (uninstalled) weave.
    """
    RolloutTraceConfig.reset()
    yield
    RolloutTraceConfig.reset()


@pytest.fixture
def mock_weave_client():
    # [FUNCTION DOC]
    # HELPER: Fixture that installs a MagicMock "weave" module (and weave.trace.context) whose init returns a mock client
    # whose create_call returns a mock call; yields the mock client and restores sys.modules afterward.
    """Mocks the weave module and its client, yielding the mock client."""
    mock_weave = MagicMock()
    mock_client = MagicMock()
    mock_call = MagicMock()
    mock_client.create_call.return_value = mock_call
    mock_weave.init.return_value = mock_client

    # Also mock the call_context if it's used internally by the decorator
    mock_weave.trace.context.call_context.return_value = MagicMock()

    with patch.dict(sys.modules, {"weave": mock_weave, "weave.trace.context": mock_weave.trace.context}):
        yield mock_client


@pytest.fixture
def mock_trackio_client():
    # [FUNCTION DOC]
    # HELPER: Fixture installing a MagicMock "trackio" module with no current run and a Trace constructor returning a
    # plain dict describing the messages and metadata; yields the mock module.
    """Mocks the trackio module, yielding the mock module."""
    mock_trackio = MagicMock()
    mock_trackio.context_vars = types.SimpleNamespace(current_run=MagicMock())
    mock_trackio.context_vars.current_run.get.return_value = None
    mock_trackio.Trace.side_effect = lambda messages, metadata=None: {
        "_type": "trackio.trace",
        "messages": messages,
        "metadata": metadata or {},
    }

    with patch.dict(sys.modules, {"trackio": mock_trackio}):
        yield mock_trackio


@pytest.fixture
def mock_mlflow_client():
    # [FUNCTION DOC]
    # HELPER: Fixture installing a MagicMock "mlflow" module whose start_span context manager yields a mock span; yields
    # (mock module, mock span).
    """Mocks the mlflow module, yielding the mock module and active span."""
    mock_mlflow = MagicMock()
    mock_span = MagicMock()
    mock_mlflow.start_span.return_value.__enter__.return_value = mock_span

    with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        yield mock_mlflow, mock_span


class TracedClass:
    # [CLASS DOC]
    # HELPER: Sample class with traced async methods, a nested traced call chain and a raising method.
    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def my_method(self, a, b="default"):
        # [FUNCTION DOC]
        # HELPER: Traced; returns "result: <a>, <b>".
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @weave.op
    # @mlflow.trace
    async def middle_method(self, a, b="default"):
        # [FUNCTION DOC]
        # HELPER: Traced; awaits my_method with fixed arguments, then returns "result: <a>, <b>".
        await self.my_method("test_a1", b="test_b1")
        return f"result: {a}, {b}"

    @rollout_trace_op
    # @mlflow.trace
    async def my_method_with_exception(self):
        # [FUNCTION DOC]
        # HELPER: Traced; always raises ValueError("Test Exception").
        raise ValueError("Test Exception")

    async def upper_method(self):
        # [FUNCTION DOC]
        # HELPER: Untraced; awaits my_method and middle_method with fixed arguments and returns True.
        await self.my_method("test_a0", b="test_b0")
        await self.middle_method("test_a2", b="test_b2")
        return True


class UntracedClass:
    # [CLASS DOC]
    # HELPER: Class with a single traced method, used with no backend configured.
    @rollout_trace_op
    async def my_method(self, x):
        # [FUNCTION DOC]
        # HELPER: Traced; returns x doubled.
        return x * 2


class ChatTraceClass:
    # [CLASS DOC]
    # HELPER: Class whose traced run takes chat messages and returns a dict output with prompt/response ids and text.
    @rollout_trace_op
    async def run(self, messages, sampling_params):
        # [FUNCTION DOC]
        # HELPER: Returns a fixed dict (prompt ids [1,2,3], response ids [4], response text "4", reward 1.0).
        return {
            "prompt_ids": [1, 2, 3],
            "response_ids": [4],
            "response_text": "4",
            "reward": 1.0,
        }


class GeneratedTokenOutput(BaseModel):
    # [CLASS DOC]
    # HELPER: Pydantic model of generated token ids, used as a structured traced output.
    token_ids: list[int]


class FakeTokenizer:
    # [CLASS DOC]
    # HELPER: Fake tokenizer used for token2text decoding.
    def decode(self, token_ids):
        # [FUNCTION DOC]
        # HELPER: Returns "decoded:" followed by the comma-joined token ids.
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


class LLMGenerateTraceClass:
    # [CLASS DOC]
    # HELPER: Class with a tokenizer attribute and a traced generate method, mimicking an LLM server wrapper.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets self.tokenizer to a FakeTokenizer.
        self.tokenizer = FakeTokenizer()

    @rollout_trace_op
    async def generate(self, *, prompt_ids):
        # [FUNCTION DOC]
        # HELPER: Traced; returns GeneratedTokenOutput with token ids [3, 4] regardless of input.
        return GeneratedTokenOutput(token_ids=[3, 4])


async def test_rollout_trace_on_untraced_class():
    # [FUNCTION DOC]
    # With no backend configured, the decorated method still returns its normal result.
    """Tests that the decorator works correctly when no backend is configured."""
    instance = UntracedClass()
    assert await instance.my_method(10) == 20


async def test_rollout_trace_with_tracer(mock_weave_client):
    # [FUNCTION DOC]
    # With the weave backend, asserts get_client returns the mock client, create_call is called once with op
    # "TracedClass.my_method" and inputs bound by parameter name, and finish_call is called with the call and output.
    """Tests that the decorator calls the tracer's methods correctly."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()
    assert RolloutTraceConfig.get_client() is mock_weave_client

    result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_weave_client.create_call.assert_called_once()
    call_kwargs = mock_weave_client.create_call.call_args.kwargs
    assert call_kwargs["op"] == "TracedClass.my_method"
    expected_inputs = {"a": "test_a", "b": "test_b"}
    assert call_kwargs["inputs"] == expected_inputs

    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once_with(mock_call, output=result)


async def test_token2text_decodes_llm_generate_input_and_output(mock_mlflow_client):
    # [FUNCTION DOC]
    # With the mlflow backend and token2text enabled, asserts span inputs are the raw prompt ids and span outputs are the
    # model-dumped output plus decoded prompt_text and response_text.
    """Per-turn LLM traces decode prompt input ids and generated output ids."""
    _, mock_span = mock_mlflow_client
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="mlflow",
        token2text=True,
    )

    instance = LLMGenerateTraceClass()
    result = await instance.generate(prompt_ids=[1, 2])

    assert result == GeneratedTokenOutput(token_ids=[3, 4])
    mock_span.set_inputs.assert_called_once_with({"prompt_ids": [1, 2]})
    mock_span.set_outputs.assert_called_once_with(
        {
            "token_ids": [3, 4],
            "prompt_text": "decoded:1,2",
            "response_text": "decoded:3,4",
        }
    )


async def test_rollout_trace_with_exception(mock_weave_client):
    # [FUNCTION DOC]
    # With weave, a raising traced method propagates ValueError and finish_call receives the call plus an "exception"
    # keyword holding the ValueError.
    """Tests that `finish` is called with the exception when one is raised."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")
    instance = TracedClass()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    mock_weave_client.create_call.assert_called_once()
    mock_call = mock_weave_client.create_call.return_value
    mock_weave_client.finish_call.assert_called_once()

    # Check that finish_call was called with the exception
    args, kwargs = mock_weave_client.finish_call.call_args
    assert args[0] == mock_call
    assert "exception" in kwargs
    assert isinstance(kwargs["exception"], ValueError)


async def test_rollout_trace_with_dummy_backend(mock_weave_client):
    # [FUNCTION DOC]
    # The "dummy" backend does not call weave's create_call.
    """Tests that the tracer is not called when the backend is 'dummy'."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="dummy")
    instance = TracedClass()

    await instance.my_method("test_a")

    mock_weave_client.create_call.assert_not_called()


async def test_rollout_trace_with_trackio_backend(mock_trackio_client):
    # [FUNCTION DOC]
    # With trackio inside rollout_trace_attr(step 1, sample 2, rollout_n 3): asserts trackio.init is called with project,
    # name and framework config, a Trace is built whose first message describes the operation and whose metadata has
    # step and sample index, and trackio.log is called once with step 1.
    """Tests that the trackio backend logs a Trackio Trace."""
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="trackio")
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        result = await instance.my_method("test_a", b="test_b")

    assert result == "result: test_a, test_b"
    mock_trackio_client.init.assert_called_once_with(
        project="my-project", name="my-experiment", config={"framework": "verl"}
    )
    mock_trackio_client.Trace.assert_called_once()
    trace_kwargs = mock_trackio_client.Trace.call_args.kwargs
    assert trace_kwargs["messages"][0]["content"] == "verl rollout trace operation: TracedClass.my_method"
    assert trace_kwargs["metadata"]["step"] == 1
    assert trace_kwargs["metadata"]["sample_index"] == 2
    mock_trackio_client.log.assert_called_once()
    log_kwargs = mock_trackio_client.log.call_args.kwargs
    assert log_kwargs["step"] == 1


async def test_trackio_backend_uses_chat_messages_when_available(mock_trackio_client):
    # [FUNCTION DOC]
    # With trackio and a chat-style method, asserts the Trace messages are the input messages plus an assistant message
    # with the response text, the remaining inputs (sampling params) go to metadata, and the output dict is in metadata.
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="trackio")
    instance = ChatTraceClass()
    input_messages = [
        {"role": "system", "content": "You are a concise math assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.run(
            messages=input_messages,
            sampling_params={"temperature": 0.0, "max_tokens": 16},
        )

    trace_kwargs = mock_trackio_client.Trace.call_args.kwargs
    assert trace_kwargs["messages"] == [
        *input_messages,
        {"role": "assistant", "content": "4"},
    ]
    assert trace_kwargs["metadata"]["inputs"] == {"sampling_params": {"temperature": 0.0, "max_tokens": 16}}
    assert trace_kwargs["metadata"]["output"]["prompt_ids"] == [1, 2, 3]
    assert trace_kwargs["metadata"]["output"]["response_text"] == "4"


async def test_trace_disabled_with_trace_false(mock_weave_client):
    # [FUNCTION DOC]
    # trace=False in rollout_trace_attr suppresses weave calls; a following default attr block traces again (exactly one
    # create_call).
    """Tests that tracing is disabled when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    # No tracing should have occurred
    mock_weave_client.create_call.assert_not_called()

    # Verify that tracing works again with trace=True (default)
    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_false_disables_nested_trace_ops(mock_weave_client):
    # [FUNCTION DOC]
    # trace=False also suppresses all nested traced calls under an untraced method; tracing works again afterwards.
    """Tests that trace=False disables all nested @rollout_trace_op calls."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
        # Call upper_method which internally calls my_method and middle_method
        # All of these are decorated with @rollout_trace_op
        result = await instance.upper_method()
        assert result is True

    # No tracing should have occurred for any of the nested calls
    mock_weave_client.create_call.assert_not_called()

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


async def test_trace_enabled_restored_after_exception(mock_weave_client):
    # [FUNCTION DOC]
    # An exception raised inside a trace=False block must still restore tracing; a later call is traced once.
    """Tests that trace state is restored even if an exception occurs when trace=False."""
    RolloutTraceConfig.init(
        project_name="my-project",
        experiment_name="my-experiment",
        backend="weave",
    )
    instance = TracedClass()

    assert RolloutTraceConfig.get_backend() == "weave"

    # Use trace=False and raise an exception
    try:
        with rollout_trace_attr(step=1, sample_index=0, rollout_n=0, trace=False):
            raise RuntimeError("Test exception with trace disabled")
    except RuntimeError:
        pass

    with rollout_trace_attr(step=1, sample_index=0, rollout_n=0):
        result = await instance.my_method("test_a", b="test_b")
        assert result == "result: test_a, test_b"

    assert mock_weave_client.create_call.call_count == 1


@pytest.mark.skipif(
    os.environ.get("RUN_WEAVE_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping weave integration test. Set RUN_WEAVE_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_weave_backend():
    # [FUNCTION DOC]
    # Opt-in (RUN_WEAVE_INTEGRATION_TESTS=true) test using a real weave backend: runs the nested chain inside an attr block
    # and the raising method, then prints a reminder to inspect the weave project.
    """Integration test with a real weave backend."""

    # This assumes that the weave environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="weave")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3):
        await instance.upper_method()

    with pytest.raises(ValueError, match="Test Exception"):
        await instance.my_method_with_exception()

    print("\nWeave integration test ran successfully. Check your weave project for the trace.")


@pytest.mark.skipif(
    os.environ.get("RUN_MLFLOW_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Skipping mlflow integration test. Set RUN_MLFLOW_INTEGRATION_TESTS=true to run.",
)
async def test_rollout_trace_with_real_mlflow_backend():
    # [FUNCTION DOC]
    # Opt-in (RUN_MLFLOW_INTEGRATION_TESTS=true) test using a real mlflow backend with a named attr block over the nested
    # chain.
    # KNOWN ISSUE: the final print message mentions weave although this is the mlflow integration test.
    """Integration test with a real mlflow backend."""

    # This assumes that the mlflow environment (e.g., project) is configured
    RolloutTraceConfig.init(project_name="my-project", experiment_name="my-experiment", backend="mlflow")

    instance = TracedClass()

    with rollout_trace_attr(step=1, sample_index=2, rollout_n=3, name="agent_run"):
        assert await instance.upper_method()

    # with pytest.raises(ValueError, match="Test Exception"):
    #     await instance.my_method_with_exception()

    print("\nWeave integration test ran successfully. Check your weave project for the trace.")
