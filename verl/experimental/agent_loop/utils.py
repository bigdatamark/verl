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
# Module utils: small helpers for agent loops, mainly manual chat formatting for gpt-oss models (whose chat
# template cannot render tool responses without matching tool calls). Also re-exports resolve_config_path from
# verl.utils.import_utils for backward compatibility.

from typing import Any

# Moved to verl.utils.import_utils; re-exported for compat.
from verl.utils.import_utils import resolve_config_path  # noqa: F401


# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Behavior: wraps a tool response in gpt-oss harmony markup: start token, "functions.<name> to=assistant",
    # commentary channel, message token, content, end token. Inputs: tool_response text; tool_call_name.
    # Outputs: formatted string. No side effects.
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Behavior: appends the gpt-oss assistant start marker to the text. Outputs: new string.
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"


def build_gpt_oss_tool_response_text(messages: list[dict[str, Any]], tool_call_names: list[str]) -> str:
    # [FUNCTION DOC]
    # HELPER: Behavior: formats each tool message's "content" with the tool name at the same index, concatenates
    # them and appends the assistant generation prompt.
    # Inputs: messages - list of dicts with "content"; tool_call_names - names aligned by index.
    # Outputs: string. Exceptions: IndexError if fewer names than messages; KeyError if content missing.
    """Build gpt-oss tool response text (manual formatting + generation prompt)."""
    tool_response_texts: list[str] = []
    for i, tool_msg in enumerate(messages):
        actual_tool_name = tool_call_names[i]
        formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
        tool_response_texts.append(formatted)
    return add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
