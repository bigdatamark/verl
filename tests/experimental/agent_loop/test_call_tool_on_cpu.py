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
# CPU unit tests for ToolAgentLoop._call_tool: dispatching a parsed function call
# to a registered tool, error messages for unknown names / bad JSON / None
# arguments / execution failures (reward 0.0), and left/right truncation of
# long tool responses. Uses fake tools and a spec'd mock loop object with the
# real _call_tool bound to it.

"""Unit tests for ToolAgentLoop._call_tool error handling (no GPU required).

Tests that malformed tool calls return specific, actionable error messages
instead of generic exception strings.
"""

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from verl.tools.schemas import ToolResponse


@dataclass
class FakeFunctionCall:
    # [CLASS DOC]
    # HELPER: stand-in for a parsed function call with name and raw JSON arguments string.
    """Minimal FunctionCall for testing."""

    name: str
    arguments: str


@dataclass
class FakeAgentData:
    # [CLASS DOC]
    # HELPER: stand-in for AgentData carrying only per-tool kwargs (default empty mapping).
    """Minimal AgentData for testing."""

    tools_kwargs: dict = field(default_factory=dict)


class FakeTool:
    # [CLASS DOC]
    # HELPER: fake tool implementing the create/execute/release lifecycle successfully.
    """A fake tool that succeeds."""

    def __init__(self, name: str):
        # [FUNCTION DOC]
        # HELPER: stores the tool name.
        self.name = name

    async def create(self, create_kwargs=None):
        # [FUNCTION DOC]
        # HELPER: async; returns fixed instance id "instance_1" and an empty ToolResponse.
        return "instance_1", ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        # [FUNCTION DOC]
        # HELPER: async; returns (ToolResponse with text "OK: <parameters>", reward 1.0, empty metrics).
        return ToolResponse(text=f"OK: {parameters}"), 1.0, {}

    async def release(self, instance_id):
        # [FUNCTION DOC]
        # HELPER: async no-op.
        pass


class FakeFailingTool(FakeTool):
    # [CLASS DOC]
    # HELPER: fake tool whose execute always fails.
    """A fake tool that raises during execute."""

    async def execute(self, instance_id, parameters, **kwargs):
        # [FUNCTION DOC]
        # HELPER: async; always raises RuntimeError("database connection failed").
        raise RuntimeError("database connection failed")


class FakeLongResponseTool(FakeTool):
    # [CLASS DOC]
    # HELPER: fake tool returning a configured fixed text.
    """A fake tool that returns a long response."""

    def __init__(self, name: str, text: str):
        # [FUNCTION DOC]
        # HELPER: stores name and the response text.
        super().__init__(name)
        self.text = text

    async def execute(self, instance_id, parameters, **kwargs):
        # [FUNCTION DOC]
        # HELPER: async; returns (ToolResponse with the stored text, 1.0, empty metrics).
        return ToolResponse(text=self.text), 1.0, {}


def _make_tool_agent_loop(
    tools: dict[str, Any],
    max_tool_response_length: int = 10000,
    tool_response_truncate_side: str = "left",
):
    # [FUNCTION DOC]
    # HELPER: builds a MagicMock spec'd as ToolAgentLoop with tools,
    # max_tool_response_length and tool_response_truncate_side set, and the real
    # unbound _call_tool bound to the mock as a method. Returns the mock.
    """Create a minimal ToolAgentLoop instance with only the fields _call_tool needs."""
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    mock = MagicMock(spec=ToolAgentLoop)
    mock.tools = tools
    mock.max_tool_response_length = max_tool_response_length
    mock.tool_response_truncate_side = tool_response_truncate_side
    # Bind the real _call_tool method to our mock
    mock._call_tool = ToolAgentLoop._call_tool.__get__(mock, ToolAgentLoop)
    return mock


class TestCallToolErrorHandling(unittest.IsolatedAsyncioTestCase):
    # [CLASS DOC]
    # Async test case; each test calls _call_tool(tool_call, {}, agent_data) and
    # inspects the returned (response, reward, metrics) tuple.
    """Test ToolAgentLoop._call_tool error handling for malformed tool calls."""

    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: registers fake "calculator" and "search" tools, builds the mock loop and empty agent data.
        self.tools = {
            "calculator": FakeTool("calculator"),
            "search": FakeTool("search"),
        }
        self.loop = _make_tool_agent_loop(self.tools)
        self.agent_data = FakeAgentData()

    async def test_valid_tool_call(self):
        # [FUNCTION DOC]
        # Behavior: valid JSON args to a known tool yield reward 1.0 and "OK" text.
        """Valid tool call should succeed."""
        tool_call = FakeFunctionCall(name="calculator", arguments='{"a": 3, "b": 5}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert "OK" in response.text

    async def test_unknown_function_name(self):
        # [FUNCTION DOC]
        # Behavior: misspelled name gives reward 0.0 and text mentioning "Unknown function", the bad name and all available tool names.
        """Unknown function name should list available tools."""
        tool_call = FakeFunctionCall(name="calculater", arguments='{"a": 3}')
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Unknown function" in response.text
        assert "calculater" in response.text
        assert "calculator" in response.text
        assert "search" in response.text

    async def test_invalid_json_arguments(self):
        # [FUNCTION DOC]
        # Behavior: non-JSON args give reward 0.0 and "Invalid JSON" plus the tool name.
        """Invalid JSON arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="{a: 3}")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text
        assert "calculator" in response.text

    async def test_empty_arguments(self):
        # [FUNCTION DOC]
        # Behavior: empty-string args give reward 0.0 and "Invalid JSON".
        """Empty string arguments should report parse error."""
        tool_call = FakeFunctionCall(name="calculator", arguments="")
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_none_arguments(self):
        # [FUNCTION DOC]
        # Behavior: None args give reward 0.0 and "Invalid JSON".
        """None arguments should report error."""
        tool_call = FakeFunctionCall(name="calculator", arguments=None)
        response, reward, _ = await self.loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "Invalid JSON" in response.text

    async def test_tool_execution_error(self):
        # [FUNCTION DOC]
        # Behavior: an exception in execute yields reward 0.0 and text containing the tool name and exception message.
        """Tool execution failure should include tool name in error."""
        tools = {"failing_tool": FakeFailingTool("failing_tool")}
        loop = _make_tool_agent_loop(tools)
        tool_call = FakeFunctionCall(name="failing_tool", arguments='{"query": "test"}')
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 0.0
        assert "failing_tool" in response.text
        assert "database connection failed" in response.text

    async def test_left_truncation_keeps_response_tail(self):
        # [FUNCTION DOC]
        # Behavior: with max length 19 and left truncation, text starts with "(truncated)..." and ends with the original tail; reward stays 1.0.
        """Left truncation should drop the left side and preserve the response tail."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="left")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("(truncated)...")
        assert response.text.endswith("Final answer: Paris")

    async def test_right_truncation_keeps_response_head(self):
        # [FUNCTION DOC]
        # Behavior: with right truncation, text keeps the head, ends with "...(truncated)" and drops the tail.
        """Right truncation should drop the right side and preserve the response head."""
        tool_response = (
            "Search results for capital of France:\n"
            "1. Lyon is a major city with a long Roman history.\n"
            "2. Marseille is a large port city in southern France.\n"
            "3. The final retrieved snippet says the capital is Paris.\n"
            "Final answer: Paris"
        )
        tools = {"search": FakeLongResponseTool("search", tool_response)}
        loop = _make_tool_agent_loop(tools, max_tool_response_length=19, tool_response_truncate_side="right")
        tool_call = FakeFunctionCall(name="search", arguments="{}")
        response, reward, _ = await loop._call_tool(tool_call, {}, self.agent_data)
        assert reward == 1.0
        assert response.text.startswith("Search results")
        assert response.text.endswith("...(truncated)")
        assert "Final answer: Paris" not in response.text


if __name__ == "__main__":
    unittest.main()
