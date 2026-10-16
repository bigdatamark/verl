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
# Local-only test of GptOssToolParser: parses a GPT-OSS harmony-format response
# (commentary channel addressed to functions.<name>) into function calls.
# Requires downloading the openai/gpt-oss-20b tokenizer, hence skipped by default.

import pytest
from transformers import AutoTokenizer

from verl.experimental.agent_loop.tool_parser import GptOssToolParser


@pytest.mark.asyncio
@pytest.mark.skip(reason="local test only")
async def test_gpt_oss_tool_parser():
    # [FUNCTION DOC]
    # Behavior: encodes a sample harmony transcript containing one tool call plus a
    # tool response, runs the parser's async extract_tool_calls on the token ids,
    # and asserts exactly one call named get_current_weather with JSON arguments
    # {"location": "Tokyo"} (the tool-response message must not be counted).
    # Inputs: none. Outputs: none (assertions).
    # Exceptions: AssertionError on mismatch; network/HF errors loading tokenizer.
    # Side effects: downloads tokenizer from the HF hub. Marked skip.
    example_text = """
<|start|>assistant<|channel|>commentary to=functions.get_current_weather \
<|constrain|>json<|message|>{"location": "Tokyo"}<|call|>
<|start|>functions.get_current_weather to=assistant<|channel|>commentary<|message|>\
{ "temperature": 20, "sunny": true }<|end|>"""
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    response_ids = tokenizer.encode(example_text)
    tool_parser = GptOssToolParser(tokenizer)
    _, function_calls = await tool_parser.extract_tool_calls(response_ids)
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_current_weather"
    assert function_calls[0].arguments == '{"location": "Tokyo"}'
