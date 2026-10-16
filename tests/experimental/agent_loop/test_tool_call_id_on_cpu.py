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
# CPU tests verifying that tool call ids emitted by the model are preserved through the Kimi tool parser,
# the assistant-message builder, and the tool-processing state of ToolAgentLoop (tool messages get tool_call_id).
# Uses SimpleNamespace stand-ins for the loop and agent data and calls unbound ToolAgentLoop methods.

import unittest
from types import SimpleNamespace
from typing import Any

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, KimiToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class _FakeTokenizer:
    # [CLASS DOC]
    # HELPER: Tokenizer stub that always decodes to a fixed text; pad_token is "<pad>".
    pad_token = "<pad>"

    def __init__(self, text: str):
        # [FUNCTION DOC]
        # HELPER: Stores the text to be returned by decode.
        self.text = text

    def decode(self, response_ids: list[int], skip_special_tokens: bool = False) -> str:
        # [FUNCTION DOC]
        # HELPER: Ignores the ids and flag; returns the stored text.
        del response_ids, skip_special_tokens
        return self.text

    def convert_tokens_to_ids(self, token: str) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns 151645 for "<|im_end|>", otherwise -1.
        return 151645 if token == "<|im_end|>" else -1


def _make_tool_schema(name: str, required: list[str] | None = None) -> OpenAIFunctionToolSchema:
    # [FUNCTION DOC]
    # HELPER: Builds a function tool schema with the given name, string properties city and unit, and the given
    # required list (empty if None).
    # Outputs: an OpenAIFunctionToolSchema.
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "tool used by CPU tests",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": required or [],
                },
            },
        }
    )


class TestToolCallIdOnCpu(unittest.IsolatedAsyncioTestCase):
    # [CLASS DOC]
    # Async unittest case grouping the tool-call-id preservation tests.
    async def test_kimi_tool_parser_preserves_model_emitted_tool_call_id(self) -> None:
        # [FUNCTION DOC]
        # Behavior: feeds a Kimi-format response (text, tool call section with id "call-get_weather-0" and JSON args,
        # im_end) to KimiToolParser and asserts content "Let me check.", one call named get_weather, re-serialized
        # arguments with spaces, and tool_call_id equal to the raw id.
        raw_tool_call_id = "call-get_weather-0"
        response_text = (
            "Let me check."
            "<|tool_calls_section_begin|>"
            f"<|tool_call_begin|>{raw_tool_call_id}"
            '<|tool_call_argument_begin|>{"city":"Seattle","unit":"celsius"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            "<|im_end|>"
        )
        parser = KimiToolParser(_FakeTokenizer(response_text))

        content, tool_calls = await parser.extract_tool_calls([1, 2, 3], tools=[_make_tool_schema("get_weather")])

        assert content == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == '{"city": "Seattle", "unit": "celsius"}'
        assert tool_calls[0].tool_call_id == raw_tool_call_id

    def test_build_assistant_message_preserves_tool_call_id(self) -> None:
        # [FUNCTION DOC]
        # Behavior: builds an assistant message from two FunctionCalls (one with id, one without) and asserts role,
        # content, that arguments are decoded into dicts, and that "id" is included only for the call that had one.
        loop = SimpleNamespace(max_parallel_calls=2)
        agent_data = SimpleNamespace(
            tool_calls=[
                FunctionCall(name="get_weather", arguments='{"city": "Seattle"}', tool_call_id="call-get_weather-0"),
                FunctionCall(name="search", arguments='{"query": "forecast"}'),
            ]
        )

        message = ToolAgentLoop._build_assistant_message(loop, "Checking.", agent_data)

        assert message["role"] == "assistant"
        assert message["content"] == "Checking."
        assert message["tool_calls"][0] == {
            "type": "function",
            "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
            "id": "call-get_weather-0",
        }
        assert message["tool_calls"][1] == {
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "forecast"}},
        }

    async def test_processing_tools_state_adds_tool_call_id_to_tool_message(self) -> None:
        # [FUNCTION DOC]
        # Behavior: runs _handle_processing_tools_state with a fake loop whose tool call returns a ToolResponse text
        # and reward 1.0, and whose context merge records updated messages and appends tokens [41, 42] with mask [0, 0].
        # Asserts the next state is GENERATING, the appended tool message carries tool_call_id, the merge received it,
        # prompt_ids were extended, response_mask is [0, 0], and tool_rewards is [1.0].
        # Side effects: mutates the local fake agent_data.
        captured_messages: list[dict[str, Any]] = []

        async def call_tool(
            tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: SimpleNamespace
        ) -> tuple[ToolResponse, float, dict[str, Any]]:
            # [FUNCTION DOC]
            # HELPER: Fake async tool executor; ignores args and returns (ToolResponse with temperature JSON text, 1.0, {}).
            del tool_call, tools_kwargs, agent_data
            return ToolResponse(text='{"temperature": 26.1}'), 1.0, {}

        async def ct_merge_context_msg(
            previous_messages: list[dict[str, Any]],
            updated_messages: list[dict[str, Any]],
            runtime_token_ids: list[int],
            response_mask: list[int],
            response_logprobs: Any = None,
            *,
            tools: Any = None,
        ):
            # [FUNCTION DOC]
            # HELPER: Fake async context merge; appends updated_messages to captured_messages and returns a tuple of
            # (object with token_ids = runtime ids + [41, 42], response_mask + [0, 0], None).
            # Side effects: mutates the enclosing captured_messages list.
            del previous_messages, response_logprobs, tools
            captured_messages.extend(updated_messages)
            merge_result = SimpleNamespace(token_ids=list(runtime_token_ids) + [41, 42])
            return merge_result, list(response_mask) + [0, 0], None

        loop = SimpleNamespace(
            max_parallel_calls=1,
            processor=None,
            tool_parser_name="qwen3_coder",
            response_length=128,
            tool_schemas=[],
            _assert_mm_supported=lambda has_multi_modal: None,
            ct_merge_context_msg=ct_merge_context_msg,
            _call_tool=call_tool,
        )
        agent_data = SimpleNamespace(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
                            "id": "call-get_weather-0",
                        }
                    ],
                },
            ],
            tool_calls=[
                FunctionCall(name="get_weather", arguments='{"city": "Seattle"}', tool_call_id="call-get_weather-0")
            ],
            tools_kwargs={},
            metrics={},
            tool_rewards=[],
            prompt_ids=[1, 2, 3],
            response_mask=[],
            response_logprobs=[],
            image_data=None,
            user_turns=0,
        )

        state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)

        assert state == AgentState.GENERATING
        assert agent_data.messages[-1] == {
            "role": "tool",
            "content": '{"temperature": 26.1}',
            "tool_call_id": "call-get_weather-0",
        }
        assert captured_messages[-1] == agent_data.messages[-1]
        assert agent_data.prompt_ids == [1, 2, 3, 41, 42]
        assert agent_data.response_mask == [0, 0]
        assert agent_data.tool_rewards == [1.0]


if __name__ == "__main__":
    unittest.main()
