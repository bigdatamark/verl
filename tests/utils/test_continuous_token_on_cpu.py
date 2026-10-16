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
# CPU tests for continuous-token building (verl.utils.tokenizer.continuous_token, continuous_token_wiring and the
# DeepSeek V4 builder). A continuous-token builder incrementally extends a multi-turn conversation's token ids by
# rendering only newly added messages and merging them onto the previous context token ids (inserting any
# boundary tokens the chat template would produce, e.g. a newline after Qwen's im_end), so the final ids match
# a full re-render. Tests cover the family registry and model-type inference, per-family boundary logic, special
# token validation, rollback/template-equivalence, tool-response handling and VL variants. Many fake tokenizers
# here encode text as one id per character (the character's code point) and render a simple role-tagged template.

import logging

import pytest

from verl.utils.tokenizer.continuous_token import (
    ContinuousTokenBuilder,
    DeepSeekContinuousTokenBuilder,
    DeepSeekVL2ContinuousTokenBuilder,
    Gemma4ContinuousTokenBuilder,
    GLM46VContinuousTokenBuilder,
    GLMContinuousTokenBuilder,
    GptOssContinuousTokenBuilder,
    KimiVLContinuousTokenBuilder,
    MergeResult,
    MiniMaxContinuousTokenBuilder,
    QwenContinuousTokenBuilder,
    QwenVLContinuousTokenBuilder,
)
from verl.utils.tokenizer.continuous_token_wiring import (
    CONTINUOUS_TOKEN_BUILDER_FAMILIES,
    ContinuousTokenModelFamily,
    create_continuous_token_builder,
    get_continuous_token_builder_class,
    infer_continuous_token_model_family,
    list_continuous_token_builder_families,
    resolve_continuous_token_model_family,
)
from verl.utils.tokenizer.deepseek import DeepSeekV4ContinuousTokenBuilder


class _DummyTokenizer:
    # [CLASS DOC]
    # HELPER: Bare tokenizer stand-in exposing only name_or_path "Qwen/Qwen3-8B".
    name_or_path = "Qwen/Qwen3-8B"


class _TemplateTokenizer:
    # [CLASS DOC]
    # HELPER: Base fake tokenizer: encodes text as the list of character code points and renders messages as
    # "<role>content" plus newline per message, with an optional trailing "<assistant>" generation prompt.
    name_or_path = "unit-test/default"

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # HELPER: Returns the code point of each character in text; add_special_tokens ignored.
        return [ord(char) for char in text]

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Concatenates "<role>" + content (empty if missing) + newline for each message, appends "<assistant>" when
        # add_generation_prompt, and returns encoded ids if tokenize else the string. tools, return_dict and extra
        # kwargs are ignored.
        rendered = "".join(f"<{message['role']}>{message.get('content', '')}\n" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _RecordingTemplateTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Template tokenizer that records each chat-template call.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the empty calls list.
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Appends a record of messages (copied list), add_generation_prompt, tools and kwargs (copied) to calls,
        # then delegates to the base rendering. Side effects: mutates calls.
        self.calls.append(
            {
                "messages": list(messages),
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
                "kwargs": dict(kwargs),
            }
        )
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )


class _NonPrefixStableTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Tokenizer whose rendering is not prefix-stable: renders like the base but prefixes "mutated-prefix:"
    # when more than one message is rendered.
    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Renders as a string via the base, prepends "mutated-prefix:" for multi-message input, then encodes if
        # tokenize.
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        if len(messages) > 1:
            rendered = "mutated-prefix:" + rendered
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _QwenBoundaryTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake Qwen tokenizer (name Qwen/Qwen3-8B) with im_end id 151645 and newline id 198.
    name_or_path = "Qwen/Qwen3-8B"

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets im_end_id and newline_id.
        self.im_end_id = 151645
        self.newline_id = 198

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # HELPER: Encodes a lone newline as [newline_id], otherwise per-character code points.
        if text == "\n":
            return [self.newline_id]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Maps "<|im_end|>" to im_end_id, any other token to 0.
        if token == "<|im_end|>":
            return self.im_end_id
        return 0


class _GLMBoundaryTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake GLM-4.7 tokenizer with observation id 151333 and user id 151336.
    name_or_path = "zai-org/GLM-4.7-Flash"

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets observation_id and user_id.
        self.observation_id = 151333
        self.user_id = 151336

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Maps "<|observation|>" and "<|user|>" to their ids, else 0.
        if token == "<|observation|>":
            return self.observation_id
        if token == "<|user|>":
            return self.user_id
        return 0


class _MiniMaxBoundaryTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake MiniMax-M2 tokenizer with eos id 200020 (token "[e~[") and newline id 10.
    name_or_path = "MiniMaxAI/MiniMax-M2"

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets eos_id and newline_id.
        self.eos_id = 200020
        self.newline_id = 10

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # HELPER: Encodes a lone newline as [newline_id], otherwise code points.
        if text == "\n":
            return [self.newline_id]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Maps "[e~[" to eos_id, else 0.
        if token == "[e~[":
            return self.eos_id
        return 0


class _Gemma4BoundaryTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake Gemma 4 tokenizer with tool-response token id 262144 and a Gemma-style renderer.
    name_or_path = "google/gemma-4-27b-it"

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets tool_response_id.
        self.tool_response_id = 262144

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Maps "<|tool_response>" to tool_response_id, else 0.
        if token == "<|tool_response>":
            return self.tool_response_id
        return 0

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Finds the tool-call function names of the latest assistant message that has tool_calls (default name
        # "unknown"). Renders each tool message as a tool_response block naming the positionally matching function (or
        # "unknown" past the end) with stringified content; assistant messages as "<assistant>content<turn|>" plus
        # newline; other roles as "<role>content" plus newline; optional "<assistant>" generation prompt; encodes if tokenize.
        """Minimal Gemma-style renderer: tool messages become ``<|tool_response>`` blocks
        whose function name is resolved positionally from the latest assistant tool_calls,
        mirroring the real Gemma template enough to exercise the builder's tool path.
        """
        assistant_tool_names: list[str] = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                assistant_tool_names = [
                    tool_call.get("function", {}).get("name", "unknown") for tool_call in message["tool_calls"]
                ]
        rendered = ""
        tool_index = 0
        for message in messages:
            role = message.get("role")
            if role == "tool":
                name = assistant_tool_names[tool_index] if tool_index < len(assistant_tool_names) else "unknown"
                tool_index += 1
                content = message.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                rendered += f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>'
            elif role == "assistant":
                rendered += f"<assistant>{message.get('content', '')}<turn|>\n"
            else:
                rendered += f"<{role}>{message.get('content', '')}\n"
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _DeepSeekToolAppendTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake DeepSeek-V3.1 tokenizer (eos id 1, unknown id 0) whose renderer concatenates tool call strings.
    name_or_path = "deepseek-ai/DeepSeek-V3.1"
    unk_token_id = 0

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Sets eos_id to 1.
        self.eos_id = 1

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Maps the DeepSeek end-of-sentence token to eos_id, else unk_token_id.
        if token == "<｜end▁of▁sentence｜>":
            return self.eos_id
        return self.unk_token_id

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Renders tool messages as tool_output_begin/content/tool_output_end; assistant messages with tool_calls as
        # "<assistant>" + content + each call's name and arguments (string concatenation, so arguments must be strings)
        # + "<eos>"; other messages "<role>content" plus newline. Appends "<assistant>" only if requested and the last
        # message was not a tool message. Encodes if tokenize.
        # Exceptions: TypeError if a tool call's arguments are not a string (by design, like the real templates).
        """Minimal DeepSeek-style renderer: tool calls are spliced in by string
        concatenation like the real R1 / V3.1 / V3.2 templates (a mapping raises
        TypeError there), and no generation prompt follows a tool message.
        """
        rendered = ""
        last_was_tool = False
        for message in messages:
            role = message.get("role")
            if role == "tool":
                rendered += f"<tool_output_begin>{message.get('content', '')}<tool_output_end>"
                last_was_tool = True
                continue
            last_was_tool = False
            if role == "assistant" and message.get("tool_calls"):
                calls = ""
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    calls += "<tool_call_begin>" + function["name"] + "<tool_sep>" + function["arguments"]
                    calls += "<tool_call_end>"
                rendered += "<assistant>" + message.get("content", "") + calls + "<eos>"
            else:
                rendered += f"<{role}>{message.get('content', '')}\n"
        if add_generation_prompt and not last_was_tool:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _GptOssGenerationRewriteTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake GPT-OSS tokenizer that ends the final assistant message with "<return>" when no generation prompt
    # is requested, but "<end>" when one is (modeling GPT-OSS rewriting the return token for continuation).
    """Model GPT-OSS replacing a final assistant return token for continuation."""

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Renders each message as "<role>content<end>", except the last assistant message which ends in "<end>" or
        # "<return>" depending on add_generation_prompt; appends "<assistant>" if requested; encodes if tokenize.
        del tools, return_dict, kwargs
        rendered = ""
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            if role == "assistant" and index == len(messages) - 1:
                stop = "<end>" if add_generation_prompt else "<return>"
                rendered += f"<assistant>{content}{stop}"
            else:
                rendered += f"<{role}>{content}<end>"
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _GptOssRollbackTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake GPT-OSS tokenizer using the harmony channel format, with generation prompt
    # "<|start|>assistant<|channel|>analysis<|message|>".
    """Model the GPT-OSS channel format used by rollback context encoding."""

    _generation_prompt = "<|start|>assistant<|channel|>analysis<|message|>"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Renders assistant messages as an optional analysis channel block (from thinking, reasoning_content or
        # reasoning) followed by a final channel block; system/user as start/role/message/content/end; appends the
        # generation prompt if requested; encodes if tokenize.
        # Exceptions: ValueError for any other role.
        del tools, return_dict, kwargs
        rendered = ""
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "assistant":
                thinking = message.get("thinking") or message.get("reasoning_content") or message.get("reasoning")
                if thinking:
                    rendered += f"<|start|>assistant<|channel|>analysis<|message|>{thinking}<|end|>"
                rendered += f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
            elif role in {"system", "user"}:
                rendered += f"<|start|>{role}<|message|>{content}<|end|>"
            else:
                raise ValueError(f"Unsupported GPT-OSS rollback test role: {role!r}")
        if add_generation_prompt:
            rendered += self._generation_prompt
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class _MissingSpecialTokenTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Tokenizer whose convert_tokens_to_ids always returns None (missing special tokens).
    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns None for any token.
        return None


class _ListSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    # [CLASS DOC]
    # HELPER: Qwen tokenizer returning im_end as a single-element list.
    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns [im_end_id] for "<|im_end|>", else defers to the parent.
        if token == "<|im_end|>":
            return [self.im_end_id]
        return super().convert_tokens_to_ids(token)


class _MultiIdSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    # [CLASS DOC]
    # HELPER: Qwen tokenizer returning two ids for im_end (invalid multi-id special token).
    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns [im_end_id, im_end_id + 1] for "<|im_end|>", else defers to the parent.
        if token == "<|im_end|>":
            return [self.im_end_id, self.im_end_id + 1]
        return super().convert_tokens_to_ids(token)


class _InvalidSpecialTokenQwenTokenizer(_QwenBoundaryTokenizer):
    # [CLASS DOC]
    # HELPER: Qwen tokenizer returning -1 for im_end (invalid id).
    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns -1 for "<|im_end|>", else defers to the parent.
        if token == "<|im_end|>":
            return -1
        return super().convert_tokens_to_ids(token)


class _MultiTokenNewlineQwenTokenizer(_QwenBoundaryTokenizer):
    # [CLASS DOC]
    # HELPER: Qwen tokenizer whose newline encodes to two ids (invalid boundary token).
    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # HELPER: Returns [newline_id, newline_id + 1] for a lone newline, else defers to the parent.
        if text == "\n":
            return [self.newline_id, self.newline_id + 1]
        return super().encode(text, add_special_tokens=add_special_tokens)


class _DeepSeekBoundaryTokenizer(_TemplateTokenizer):
    # [CLASS DOC]
    # HELPER: Fake DeepSeek-R1 tokenizer mapping the builder's EOS, BOS, user and assistant tokens to ids 200-203.
    name_or_path = "deepseek-ai/DeepSeek-R1"

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns the mapped id for those four tokens, else None.
        mapping = {
            DeepSeekContinuousTokenBuilder._EOS_TOKEN: 200,
            DeepSeekContinuousTokenBuilder._BOS_TOKEN: 201,
            DeepSeekContinuousTokenBuilder._USER_TOKEN: 202,
            DeepSeekContinuousTokenBuilder._ASSISTANT_TOKEN: 203,
        }
        return mapping.get(token)


class _DeepSeekToolOutputTokenizer(_DeepSeekBoundaryTokenizer):
    # [CLASS DOC]
    # HELPER: DeepSeek tokenizer whose renderer groups consecutive tool messages inside tool_outputs_begin/end markers.
    _TOOL_OUTPUTS_BEGIN = "<\uff5ctool\u2581outputs\u2581begin\uff5c>"
    _TOOL_OUTPUT_BEGIN = "<\uff5ctool\u2581output\u2581begin\uff5c>"
    _TOOL_OUTPUT_END = "<\uff5ctool\u2581output\u2581end\uff5c>"
    _TOOL_OUTPUTS_END = DeepSeekContinuousTokenBuilder._TOOL_OUTPUTS_END_TOKEN

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=False,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # HELPER: Renders assistant-with-tool_calls as assistant token + content + EOS; tool messages as individual
        # tool_output blocks, opening the tool_outputs group on the first of a run; user as user token + content; others
        # as "<role>content". Closes the tool group only if the last rendered message was a tool message (an intermediate
        # group is not closed, a simplification). Appends assistant token plus "<think>" and newline as generation prompt;
        # encodes if tokenize.
        del tools, return_dict, kwargs
        rendered = ""
        is_tool = False
        for message in messages:
            role = message["role"]
            content = message.get("content", "")
            if role == "assistant" and message.get("tool_calls"):
                is_tool = False
                rendered += f"{DeepSeekContinuousTokenBuilder._ASSISTANT_TOKEN}{content}"
                rendered += DeepSeekContinuousTokenBuilder._EOS_TOKEN
            elif role == "tool":
                if not is_tool:
                    rendered += self._TOOL_OUTPUTS_BEGIN
                rendered += f"{self._TOOL_OUTPUT_BEGIN}{content}{self._TOOL_OUTPUT_END}"
                is_tool = True
            elif role == "user":
                is_tool = False
                rendered += f"{DeepSeekContinuousTokenBuilder._USER_TOKEN}{content}"
            else:
                is_tool = False
                rendered += f"<{role}>{content}"
        if is_tool:
            rendered += self._TOOL_OUTPUTS_END
        if add_generation_prompt:
            rendered += f"{DeepSeekContinuousTokenBuilder._ASSISTANT_TOKEN}<think>\n"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


_ROLLBACK_TEMPLATE_EQUIVALENCE_FAMILIES = (
    "default",
    "qwen",
    "qwen25",
    "qwen3",
    "qwen35",
    "minimax",
    "minimaxm2",
    "minimaxm25",
    "minimaxm27",
    "glm47",
    "glm5",
    "gemma4",
    "deepseek",
)


def _tokenizer_for_text_family(family: str):
    # [FUNCTION DOC]
    # HELPER: Returns the fake tokenizer matching a text family name: Qwen boundary for qwen variants (and "mimo"),
    # MiniMax for minimax variants, GLM for glm47/glm5, Gemma4 for gemma4, DeepSeek boundary for deepseek, otherwise
    # the plain template tokenizer.
    if family in {"qwen", "qwen25", "qwen3", "qwen35", "mimo"}:
        return _QwenBoundaryTokenizer()
    if family in {"minimax", "minimaxm2", "minimaxm25", "minimaxm27"}:
        return _MiniMaxBoundaryTokenizer()
    if family in {"glm47", "glm5"}:
        return _GLMBoundaryTokenizer()
    if family == "gemma4":
        return _Gemma4BoundaryTokenizer()
    if family == "deepseek":
        return _DeepSeekBoundaryTokenizer()
    return _TemplateTokenizer()


def test_builtin_family_surface():
    # [FUNCTION DOC]
    # Behavior: asserts the exact ordered tuple of built-in family names and that the list function returns the same.
    assert CONTINUOUS_TOKEN_BUILDER_FAMILIES == (
        "default",
        "qwen",
        "qwen25",
        "qwen3",
        "qwen35",
        "minimax",
        "minimaxm2",
        "minimaxm25",
        "minimaxm27",
        "glm47",
        "glm5",
        "gemma4",
        "gptoss",
        "deepseek",
        "vldefault",
        "qwenvl",
        "qwen25vl",
        "qwen3vl",
        "minimaxvl",
        "gemma4vl",
        "kimivl",
        "glm4v",
        "deepseekvl2",
        "deepseekv4",
    )
    assert list_continuous_token_builder_families() == CONTINUOUS_TOKEN_BUILDER_FAMILIES


@pytest.mark.parametrize(
    ("family", "builder_cls"),
    [
        (ContinuousTokenModelFamily.DEFAULT, ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN25, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN3, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN35, QwenContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M2, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M25, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.MINIMAX_M27, MiniMaxContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM47, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM5, GLMContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GEMMA4, Gemma4ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GPTOSS, GptOssContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEK, DeepSeekContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN25_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.QWEN3_VL, QwenVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.KIMI_VL, KimiVLContinuousTokenBuilder),
        (ContinuousTokenModelFamily.GLM4V, GLM46VContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEK_VL2, DeepSeekVL2ContinuousTokenBuilder),
        (ContinuousTokenModelFamily.DEEPSEEKV4, DeepSeekV4ContinuousTokenBuilder),
    ],
)
def test_builtin_family_class_mapping(family, builder_cls):
    # [FUNCTION DOC]
    # Behavior: parametrized; asserts each family enum maps to the expected builder class.
    # Inputs: family - enum member; builder_cls - expected class.
    assert get_continuous_token_builder_class(family) is builder_cls


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("glm4_moe", ContinuousTokenModelFamily.GLM47),
        ("glm_moe_dsa", ContinuousTokenModelFamily.GLM5),
        ("gemma4", ContinuousTokenModelFamily.GEMMA4),
        ("gpt_oss", ContinuousTokenModelFamily.GPTOSS),
        ("minimax_m2", ContinuousTokenModelFamily.MINIMAX_M2),
        ("minimax_text_01", ContinuousTokenModelFamily.MINIMAX),
        ("qwen3_5", ContinuousTokenModelFamily.QWEN35),
        ("qwen3_5_moe", ContinuousTokenModelFamily.QWEN35),
        ("qwen3", ContinuousTokenModelFamily.QWEN3),
        ("qwen3_moe", ContinuousTokenModelFamily.QWEN3),
        ("qwen2", ContinuousTokenModelFamily.QWEN),
        ("deepseek_v2", ContinuousTokenModelFamily.DEEPSEEK),
        ("deepseek_v3", ContinuousTokenModelFamily.DEEPSEEK),
        ("deepseek_v4", ContinuousTokenModelFamily.DEEPSEEKV4),
        # VL families.
        ("qwen2_5_vl", ContinuousTokenModelFamily.QWEN25_VL),
        ("qwen3_vl", ContinuousTokenModelFamily.QWEN3_VL),
        ("qwen3_vl_moe", ContinuousTokenModelFamily.QWEN3_VL),
        ("qwen2_vl", ContinuousTokenModelFamily.QWEN_VL),
        ("minimax_vl_01", ContinuousTokenModelFamily.MINIMAX_VL),
        ("kimi_vl", ContinuousTokenModelFamily.KIMI_VL),
        ("glm4v_moe", ContinuousTokenModelFamily.GLM4V),
        ("deepseek_vl_v2", ContinuousTokenModelFamily.DEEPSEEK_VL2),
    ],
)
def test_auto_family_inference_uses_exact_root_model_type(model_type, expected):
    # [FUNCTION DOC]
    # Behavior: parametrized; asserts HF model_type strings (text and VL) infer the expected family exactly.
    # Inputs: model_type - string; expected - family enum.
    assert infer_continuous_token_model_family(hf_model_type=model_type) == expected


def test_auto_family_inference_normalizes_hf_model_type():
    # [FUNCTION DOC]
    # Behavior: asserts surrounding whitespace and casing in the model type are normalized (" DeepSeek_V4 " -> DEEPSEEKV4).
    assert infer_continuous_token_model_family(hf_model_type=" DeepSeek_V4 ") == (ContinuousTokenModelFamily.DEEPSEEKV4)


def test_auto_family_inference_does_not_guess_unregistered_model_type(caplog):
    # [FUNCTION DOC]
    # Behavior: asserts an unregistered model type falls back to DEFAULT and logs a warning containing the type.
    # Inputs: caplog fixture.
    with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
        family = infer_continuous_token_model_family(hf_model_type="unregistered_wrapper")
    assert family == ContinuousTokenModelFamily.DEFAULT
    assert "unregistered_wrapper" in caplog.text


def test_explicit_family_is_not_rewritten():
    # [FUNCTION DOC]
    # Behavior: asserts an explicitly given family is kept regardless of model type (DEFAULT stays DEFAULT) and alias
    # strings like "qwen_3.5" resolve to QWEN35 even with a deepseek model type.
    assert (
        resolve_continuous_token_model_family(
            ContinuousTokenModelFamily.DEFAULT,
            hf_model_type="qwen3",
        )
        == ContinuousTokenModelFamily.DEFAULT
    )
    assert resolve_continuous_token_model_family(
        "qwen_3.5",
        hf_model_type="deepseek_v3",
    ) == (ContinuousTokenModelFamily.QWEN35)


def test_unknown_model_type_with_multimodal_processor_resolves_to_vl_default():
    # [FUNCTION DOC]
    # Behavior: asserts an unknown model type with a multimodal processor infers VL_DEFAULT.
    assert (
        infer_continuous_token_model_family(
            hf_model_type="unknown_model",
            has_multimodal_processor=True,
        )
        == ContinuousTokenModelFamily.VL_DEFAULT
    )


def test_auto_family_is_resolved_at_factory_time():
    # [FUNCTION DOC]
    # Behavior: asserts the factory with model type qwen3 creates a QwenContinuousTokenBuilder.
    builder = create_continuous_token_builder(
        _QwenBoundaryTokenizer(),
        hf_model_type="qwen3",
    )
    assert isinstance(builder, QwenContinuousTokenBuilder)


def test_qwen2_auto_uses_qwen_builder_and_newline_boundary_logic():
    # [FUNCTION DOC]
    # Behavior: for qwen2, asserts the factory gives a Qwen builder and merging [1, im_end] with [2] inserts the newline
    # id between them, reported in inserted_token_ids.
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, hf_model_type="qwen2")
    result = builder._merge_context_token_ids([1, tokenizer.im_end_id], [2])

    assert isinstance(builder, QwenContinuousTokenBuilder)
    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2]
    assert result.inserted_token_ids == [tokenizer.newline_id]


def test_unknown_model_with_non_multimodal_processor_uses_default_text_builder(caplog):
    # [FUNCTION DOC]
    # Behavior: unknown model type with a processor that is not multimodal; asserts a default text builder is created
    # and a warning is logged (see assertions below).
    # Inputs: caplog fixture.
    class TextProcessor:
        # [CLASS DOC]
        # HELPER: Empty processor stand-in with no multimodal attributes.
        pass

    with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
        builder = create_continuous_token_builder(
            _TemplateTokenizer(),
            hf_model_type="unknown_text_model",
            processor=TextProcessor(),
        )

    assert isinstance(builder, ContinuousTokenBuilder)
    assert "unknown_text_model" in caplog.text
    assert "default" in caplog.text


def test_default_builder_creation_forwards_kwargs():
    # [FUNCTION DOC]
    # Behavior: asserts the factory with family "default" forwards chat_template_kwargs unchanged and converts
    # allowed_append_roles to a frozenset.
    builder = create_continuous_token_builder(
        _TemplateTokenizer(),
        model_family="default",
        chat_template_kwargs={"enable_thinking": False},
        allowed_append_roles=["tool"],
    )
    assert isinstance(builder, ContinuousTokenBuilder)
    assert builder.chat_template_kwargs == {"enable_thinking": False}
    assert builder.allowed_append_roles == frozenset({"tool"})


def test_builder_forwards_template_kwargs_and_tools_when_rendering_initial_prompt():
    # [FUNCTION DOC]
    # Behavior: builds initial tokens for one user message with tools and asserts the last chat-template call used a
    # generation prompt, the same tools object, and the configured template kwargs.
    tokenizer = _RecordingTemplateTokenizer()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    builder = create_continuous_token_builder(
        tokenizer,
        model_family="default",
        chat_template_kwargs={"enable_thinking": False},
    )

    builder.build_initial_tokens([{"role": "user", "content": "question"}], tools=tools)

    assert tokenizer.calls[-1]["add_generation_prompt"] is True
    assert tokenizer.calls[-1]["tools"] is tools
    assert tokenizer.calls[-1]["kwargs"] == {"enable_thinking": False}


def test_default_builder_is_available_from_builtin_registry():
    # [FUNCTION DOC]
    # Behavior: asserts family "default" creates a base ContinuousTokenBuilder.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="default")
    assert isinstance(builder, ContinuousTokenBuilder)


def test_qwen3_builder_inserts_missing_newline_after_im_end():
    # [FUNCTION DOC]
    # Behavior: qwen3 builder merging [1, im_end] with [2, 3] inserts a newline: asserts token ids, inserted ids,
    # appended count 2 and kind "context"; then asserts align_response_metadata extends a response mask [1, 1] and
    # logprobs with zeros for the 3 merged context tokens.
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="qwen3")

    assert isinstance(builder, QwenContinuousTokenBuilder)
    result = builder._merge_context_token_ids([1, tokenizer.im_end_id], [2, 3])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "context"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0]


def test_qwen35_builder_uses_qwen3_newline_boundary_logic():
    # [FUNCTION DOC]
    # Behavior: qwen35 builder uses the same newline insertion after im_end (appended count 1, kind "context").
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="qwen35")

    assert isinstance(builder, QwenContinuousTokenBuilder)
    result = builder._merge_context_token_ids([1, tokenizer.im_end_id], [2])

    assert result.token_ids == [1, tokenizer.im_end_id, tokenizer.newline_id, 2]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 1
    assert result.kind == "context"


def test_minimax_builder_inserts_missing_newline_after_eos():
    # [FUNCTION DOC]
    # Behavior: minimaxm2 builder inserts a newline after the "[e~[" eos id; asserts ids, inserted ids, counts, kind
    # and zero-padded aligned mask/logprobs.
    tokenizer = _MiniMaxBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="minimaxm2")

    assert isinstance(builder, MiniMaxContinuousTokenBuilder)
    result = builder._merge_context_token_ids([1, tokenizer.eos_id], [2, 3])

    assert result.token_ids == [1, tokenizer.eos_id, tokenizer.newline_id, 2, 3]
    assert result.inserted_token_ids == [tokenizer.newline_id]
    assert result.appended_token_count == 2
    assert result.kind == "context"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [-0.1, -0.2, 0.0, 0.0, 0.0]


def test_glm47_builder_removes_ambiguous_boundary_token():
    # [FUNCTION DOC]
    # Behavior: glm47 builder merging context ending with observation id onto new ids starting with user id removes
    # the trailing previous boundary token: asserts ids [1, user, 2], removed_prefix_token_count 1, appended 2, and
    # aligned mask/logprobs trimmed to [1] / [-0.1] then zero-padded.
    tokenizer = _GLMBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="glm47")

    assert isinstance(builder, GLMContinuousTokenBuilder)
    result = builder._merge_context_token_ids([1, tokenizer.observation_id], [tokenizer.user_id, 2])

    assert result.token_ids == [1, tokenizer.user_id, 2]
    assert result.removed_prefix_token_count == 1
    assert result.appended_token_count == 2
    assert result.kind == "context"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1],
        [-0.1, -0.2],
    )
    assert aligned_mask == [1, 0, 0]
    assert aligned_logprobs == [-0.1, 0.0, 0.0]


def test_gemma4_builder_inserts_tool_response_boundary_for_appended_messages():
    # [FUNCTION DOC]
    # Behavior: gemma4 builder appending a tool message after an assistant tool call inserts the tool_response id
    # after the previous ids [1, 2, 3]; asserts inserted ids, appended count and kind.
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="gemma4")
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
        },
    ]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_context_tokens(previous_messages, updated_messages, [1, 2, 3])

    assert isinstance(builder, Gemma4ContinuousTokenBuilder)
    assert result.token_ids[:4] == [1, 2, 3, tokenizer.tool_response_id]
    assert result.inserted_token_ids == [tokenizer.tool_response_id]
    assert result.appended_token_count == len(result.token_ids) - 4
    assert result.kind == "context"


def test_gemma4_builder_does_not_duplicate_existing_tool_response_boundary():
    # [FUNCTION DOC]
    # Behavior: when previous ids already end with the tool_response id, gemma4 inserts nothing.
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family=ContinuousTokenModelFamily.GEMMA4)
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
        },
    ]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "name": "lookup"}]

    result = builder.merge_context_tokens(previous_messages, updated_messages, [1, tokenizer.tool_response_id])

    assert result.token_ids[:2] == [1, tokenizer.tool_response_id]
    assert result.inserted_token_ids == []
    assert result.kind == "context"


def test_gemma4_builder_formats_tool_response_by_position_with_warning(caplog):
    # [FUNCTION DOC]
    # Behavior: gemma4 tool group for a tool message without a name resolves the name ("lookup") by position from the
    # previous assistant's tool calls; asserts the exact rendered tool_response text and a warning about resolving by
    # position. Inputs: caplog.
    builder = create_continuous_token_builder(_Gemma4BoundaryTokenizer(), model_family="gemma4")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ]
    tool_messages = [{"role": "tool", "content": "answer"}]

    with caplog.at_level(logging.WARNING):
        token_ids = builder._tokenize_tool_group(
            tool_messages,
            previous_messages=previous_messages,
        )

    expected = '<|tool_response>response:lookup{value:<|"|>answer<|"|>}<tool_response|>'
    assert token_ids == [ord(char) for char in expected]
    assert "resolving a tool response name by position" in caplog.text


def test_gemma4_builder_inserts_turn_separator_before_user_after_tool_group():
    # [FUNCTION DOC]
    # Behavior: gemma4 incremental tokenization of a tool then user message inserts "<turn|>" plus newline between the
    # tool response group and the user turn, followed by the generation prompt; asserts exact text.
    builder = Gemma4ContinuousTokenBuilder(_Gemma4BoundaryTokenizer())
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ]
    updated_messages = previous_messages + [
        {"role": "tool", "content": "answer"},
        {"role": "user", "content": "continue"},
    ]

    token_ids = builder.tokenize_context_incremental_messages(previous_messages, updated_messages)

    expected_tool = '<|tool_response>response:lookup{value:<|"|>answer<|"|>}<tool_response|>'
    expected = expected_tool + "<turn|>\n<user>continue\n<assistant>"
    assert token_ids == [ord(char) for char in expected]


def test_deepseek_builder_renders_tool_appends_through_string_concatenating_template():
    # [FUNCTION DOC]
    # Behavior: with a DeepSeek string-concatenating template, asserts the base builder raises TypeError (its synthetic
    # tool call has mapping arguments), while the deepseek builder succeeds and yields only the tool output text
    # (no generation prompt after a tool message).
    tokenizer = _DeepSeekToolAppendTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family="deepseek")
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": {"q": "x"}}}
            ],
        },
    ]
    updated_messages = previous_messages + [{"role": "tool", "content": "answer", "tool_call_id": "call_0"}]

    # The base synthetic tool call carries a mapping, which this template family cannot splice in.
    with pytest.raises(TypeError):
        ContinuousTokenBuilder(tokenizer).tokenize_context_incremental_messages(previous_messages, updated_messages)

    assert isinstance(builder, DeepSeekContinuousTokenBuilder)
    incremental = builder.tokenize_context_incremental_messages(previous_messages, updated_messages)

    # Only the tool output: DeepSeek templates add no generation prompt after a tool message.
    assert incremental == [ord(char) for char in "<tool_output_begin>answer<tool_output_end>"]


def test_deepseek_builder_synthetic_tool_call_arguments_are_a_json_string():
    # [FUNCTION DOC]
    # Behavior: asserts the DeepSeek synthetic assistant keeps the tool_call_ids of the tool messages and uses the
    # JSON string "{}" for every call's arguments.
    builder = DeepSeekContinuousTokenBuilder(_DeepSeekToolAppendTokenizer())
    tool_messages = [
        {"role": "tool", "content": "first", "tool_call_id": "call_0"},
        {"role": "tool", "content": "second", "tool_call_id": "call_1"},
    ]

    synthetic_assistant = builder._synthetic_assistant_for_tools(tool_messages)

    assert [tool_call["id"] for tool_call in synthetic_assistant["tool_calls"]] == ["call_0", "call_1"]
    assert all(tool_call["function"]["arguments"] == "{}" for tool_call in synthetic_assistant["tool_calls"])


def test_gpt_oss_builder_formats_tool_responses_with_resolved_tool_name():
    # [FUNCTION DOC]
    # Behavior: gptoss tool group resolves the tool name via tool_call_id from the previous assistant and renders the
    # harmony "functions.<name> to=assistant" commentary message with JSON-encoded text content "ok"; asserts exact text.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "lookup"},
                }
            ],
        }
    ]
    tool_messages = [{"role": "tool", "tool_call_id": "call_0", "content": [{"type": "text", "text": "ok"}]}]

    token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>"ok"<|end|>'
    assert isinstance(builder, GptOssContinuousTokenBuilder)
    assert token_ids == [ord(char) for char in expected]


def test_gpt_oss_builder_formats_plain_assistant_context_with_end_not_return():
    # [FUNCTION DOC]
    # Behavior: gptoss single assistant message with thinking renders an analysis block then a final block, both ending
    # with "<|end|>" (never "<|return|>"), and includes the reasoning.
    builder = GptOssContinuousTokenBuilder(_TemplateTokenizer())
    message = {"role": "assistant", "content": "assistant answer", "thinking": "rollback reasoning"}

    token_ids = builder._tokenize_single_non_tool(message)

    expected = (
        "<|start|>assistant<|channel|>analysis<|message|>rollback reasoning<|end|>"
        "<|start|>assistant<|channel|>final<|message|>assistant answer<|end|>"
    )
    assert token_ids == [ord(char) for char in expected]
    assert "<|return|>" not in "".join(chr(token_id) for token_id in token_ids)
    assert "rollback reasoning" in "".join(chr(token_id) for token_id in token_ids)


def test_gpt_oss_builder_prefers_tool_message_name_over_context_id():
    # [FUNCTION DOC]
    # Behavior: gptoss prefers the tool message's own name over the name found through tool_call_id.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "from_context"},
                }
            ],
        }
    ]
    tool_messages = [{"role": "tool", "tool_call_id": "call_0", "name": "from_message", "content": "ok"}]

    token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = '<|start|>functions.from_message to=assistant<|channel|>commentary<|message|>"ok"<|end|>'
    assert token_ids == [ord(char) for char in expected]


def test_gpt_oss_builder_formats_multiple_tool_responses_by_position_with_warning(caplog):
    # [FUNCTION DOC]
    # Behavior: gptoss resolves two unnamed tool messages by position to "search" and "calculate", rendering both, and
    # warns. Inputs: caplog.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
            ],
        }
    ]
    tool_messages = [
        {"role": "tool", "content": "hits"},
        {"role": "tool", "content": "42"},
    ]

    with caplog.at_level(logging.WARNING):
        token_ids = builder._tokenize_tool_group(tool_messages, previous_messages=previous_messages)

    expected = (
        '<|start|>functions.search to=assistant<|channel|>commentary<|message|>"hits"<|end|>'
        '<|start|>functions.calculate to=assistant<|channel|>commentary<|message|>"42"<|end|>'
    )
    assert token_ids == [ord(char) for char in expected]
    assert "resolving a tool response name by position" in caplog.text


def test_gpt_oss_builder_rejects_ambiguous_positional_tool_name_resolution():
    # [FUNCTION DOC]
    # Behavior: gptoss raises ValueError "cannot resolve tool name by position" when the number of unnamed tool messages
    # differs from the latest assistant's tool calls, or when there are no previous messages.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
            ],
        }
    ]

    with pytest.raises(ValueError, match="cannot resolve tool name by position"):
        builder._tokenize_tool_group([{"role": "tool", "content": "hits"}], previous_messages=previous_messages)

    with pytest.raises(ValueError, match="cannot resolve tool name by position"):
        builder._tokenize_tool_group([{"role": "tool", "content": "fallback"}], previous_messages=[])


def test_gpt_oss_builder_does_not_use_older_assistant_tool_calls_for_position():
    # [FUNCTION DOC]
    # Behavior: gptoss only considers the latest assistant message; if it has no tool calls a ValueError mentioning
    # "latest assistant has 0 tool calls" is raised even though an older assistant had one.
    builder = create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss")
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "old_lookup"}}],
        },
        {"role": "assistant", "content": "new answer without tools"},
    ]

    with pytest.raises(ValueError, match="latest assistant has 0 tool calls"):
        builder._tokenize_tool_group([{"role": "tool", "content": "answer"}], previous_messages=previous_messages)


@pytest.mark.parametrize(
    ("builder", "expected_error"),
    [
        (
            create_continuous_token_builder(_TemplateTokenizer(), model_family="gptoss"),
            "got 2 tool response messages but the latest assistant has 4 tool calls",
        ),
        (
            create_continuous_token_builder(_Gemma4BoundaryTokenizer(), model_family="gemma4"),
            "got 2 tool response messages but the latest assistant has 4 tool calls",
        ),
    ],
)
def test_strict_tool_name_builders_reject_split_positional_tool_groups(builder, expected_error):
    # [FUNCTION DOC]
    # Behavior: parametrized over gptoss and gemma4 builders (constructed at collection time); appending two tool
    # messages, a user message and another tool message after an assistant with 4 tool calls raises ValueError
    # because a tool group of 2 does not match 4 calls.
    # Inputs: builder - builder instance; expected_error - message regex.
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "search"}},
                {"type": "function", "function": {"name": "calculate"}},
                {"type": "function", "function": {"name": "lookup_order"}},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
        },
    ]
    appended_messages = [
        {"role": "tool", "content": "hits"},
        {"role": "tool", "content": "42"},
        {"role": "user", "content": "please continue"},
        {"role": "tool", "content": "order shipped"},
    ]

    with pytest.raises(ValueError, match=expected_error):
        builder.tokenize_context_incremental_messages(previous_messages, previous_messages + appended_messages)


def test_default_builder_builds_dummy_assistant_from_tool_messages_only():
    # [FUNCTION DOC]
    # Behavior: base builder tool group: asserts the synthetic assistant (third message rendered in the first template
    # call) has tool calls with ids "continuous_token_call_<i>", names taken from tool messages or defaulting to
    # "continuous_token_tool", and empty mapping arguments.
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    tool_messages = [
        {"role": "tool", "content": "answer", "name": "from_message"},
        {"role": "tool", "content": "fallback"},
    ]

    builder._tokenize_tool_group(tool_messages, previous_messages=[])

    synthetic_assistant = tokenizer.calls[0]["messages"][2]
    assert synthetic_assistant["tool_calls"][0] == {
        "id": "continuous_token_call_0",
        "type": "function",
        "function": {"name": "from_message", "arguments": {}},
    }
    assert synthetic_assistant["tool_calls"][1] == {
        "id": "continuous_token_call_1",
        "type": "function",
        "function": {"name": "continuous_token_tool", "arguments": {}},
    }


def test_deepseek_builder_uses_json_string_arguments_in_synthetic_assistant():
    # [FUNCTION DOC]
    # Behavior: asserts the DeepSeek synthetic assistant uses "{}" string arguments.
    builder = DeepSeekContinuousTokenBuilder(_DeepSeekBoundaryTokenizer())

    synthetic_assistant = builder._synthetic_assistant_for_tools(
        [{"role": "tool", "content": "result", "name": "lookup"}]
    )

    assert synthetic_assistant["tool_calls"][0]["function"]["arguments"] == "{}"


def test_deepseek_builder_inherits_context_orchestration():
    # [FUNCTION DOC]
    # Behavior: asserts DeepSeekContinuousTokenBuilder does not override tokenize_context_incremental_messages or
    # merge_context_tokens (inherits them from the base class).
    assert "tokenize_context_incremental_messages" not in DeepSeekContinuousTokenBuilder.__dict__
    assert "merge_context_tokens" not in DeepSeekContinuousTokenBuilder.__dict__


def test_deepseek_builder_strips_tool_outputs_end_before_user_in_same_incremental():
    # [FUNCTION DOC]
    # Behavior: DeepSeek incremental tokenization of tool then user strips the tool_outputs_end marker between them;
    # asserts the exact expected text (tool outputs begin, one output, user turn, assistant prompt with think) and that
    # the end marker is absent.
    tokenizer = _DeepSeekToolOutputTokenizer()
    builder = DeepSeekContinuousTokenBuilder(tokenizer)
    previous_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ]
    updated_messages = previous_messages + [
        {"role": "tool", "content": "answer"},
        {"role": "user", "content": "continue"},
    ]

    token_ids = builder.tokenize_context_incremental_messages(previous_messages, updated_messages)

    expected = (
        _DeepSeekToolOutputTokenizer._TOOL_OUTPUTS_BEGIN
        + _DeepSeekToolOutputTokenizer._TOOL_OUTPUT_BEGIN
        + "answer"
        + _DeepSeekToolOutputTokenizer._TOOL_OUTPUT_END
        + DeepSeekContinuousTokenBuilder._USER_TOKEN
        + "continue"
        + DeepSeekContinuousTokenBuilder._ASSISTANT_TOKEN
        + "<think>\n"
    )
    assert token_ids == [ord(char) for char in expected]
    assert _DeepSeekToolOutputTokenizer._TOOL_OUTPUTS_END not in "".join(chr(token_id) for token_id in token_ids)


def test_deepseek_builder_trims_runtime_tool_outputs_end_before_later_user_context():
    # [FUNCTION DOC]
    # Behavior: when runtime context ids end with the encoded tool_outputs_end marker and a user message is appended,
    # the DeepSeek merge trims the marker from the previous ids; asserts the results described by the assertions below
    # (aligned mask/logprobs and ids).
    tokenizer = _DeepSeekToolOutputTokenizer()
    builder = DeepSeekContinuousTokenBuilder(tokenizer)
    tool_outputs_end_ids = tokenizer.encode(_DeepSeekToolOutputTokenizer._TOOL_OUTPUTS_END)
    previous_messages = [{"role": "tool", "content": "answer"}]
    updated_messages = previous_messages + [{"role": "user", "content": "continue"}]
    runtime_ids = [1, 2, *tool_outputs_end_ids]

    result = builder.merge_context_tokens(previous_messages, updated_messages, runtime_ids)
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1] * len(runtime_ids),
        [-0.1] * len(runtime_ids),
    )

    assert result.token_ids[:2] == [1, 2]
    assert result.removed_prefix_token_count == len(tool_outputs_end_ids)
    assert result.kind == "context"
    assert aligned_mask[:2] == [1, 1]
    assert aligned_logprobs[:2] == [-0.1, -0.1]


def test_default_builder_merges_append_only_context_messages():
    # [FUNCTION DOC]
    # Behavior: base builder merging a tool message append: asserts a MergeResult whose ids are runtime ids followed by
    # the incremental ids, appended count equals the incremental length, kind "context", and aligned mask/logprobs
    # keep the original 3 values then pad zeros.
    tokenizer = _TemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer", "tool_call_id": "call_0", "name": "lookup"}]
    runtime_ids = [1, 2, 3]

    result = builder.merge_context_tokens(old_messages, new_messages, runtime_ids)
    expected_incremental = builder.tokenize_context_incremental_messages(old_messages, new_messages)

    assert isinstance(result, MergeResult)
    assert result.token_ids == runtime_ids + expected_incremental
    assert result.appended_token_count == len(expected_incremental)
    assert result.kind == "context"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 1] + [0] * len(expected_incremental)
    assert aligned_logprobs == [0.1, 0.2, 0.3] + [0.0] * len(expected_incremental)


def test_default_builder_merges_every_encoded_context_group_through_hook():
    # [FUNCTION DOC]
    # Behavior: subclass records the first role of each group passed to _merge_context_group; asserts tool and user
    # groups are each merged through the hook and the incremental text is "<tool>answer", newline, "<user>continue",
    # newline, "<assistant>".
    class RecordingGroupMergeBuilder(ContinuousTokenBuilder):
        # [CLASS DOC]
        # HELPER: Builder subclass recording roles of merged context groups.
        def __init__(self, tokenizer):
            # [FUNCTION DOC]
            # HELPER: Initializes the base builder and an empty merged_roles list.
            super().__init__(tokenizer)
            self.merged_roles = []

        def _merge_context_group(
            self,
            incremental_ids,
            group_token_ids,
            *,
            group,
            processed_messages,
            tools=None,
        ):
            # [FUNCTION DOC]
            # HELPER: Appends the group's first message role to merged_roles and delegates to the base implementation
            # (which mutates incremental_ids in place).
            self.merged_roles.append(group[0]["role"])
            super()._merge_context_group(
                incremental_ids,
                group_token_ids,
                group=group,
                processed_messages=processed_messages,
                tools=tools,
            )

    builder = RecordingGroupMergeBuilder(_TemplateTokenizer())
    old_messages = [{"role": "assistant", "content": "tool call"}]
    new_messages = old_messages + [
        {"role": "tool", "content": "answer", "name": "lookup"},
        {"role": "user", "content": "continue"},
    ]

    incremental = builder.tokenize_context_incremental_messages(old_messages, new_messages)

    assert builder.merged_roles == ["tool", "user"]
    assert incremental == [ord(char) for char in "<tool>answer\n<user>continue\n<assistant>"]


def test_default_builder_tokenizes_system_and_user_appends_with_generation_prompt():
    # [FUNCTION DOC]
    # Behavior: asserts appending a system and a user message renders both followed by the generation prompt.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "retry"},
    ]

    incremental = builder.tokenize_context_incremental_messages(old_messages, new_messages)

    expected = "<system>policy\n<user>retry\n<assistant>"
    assert incremental == [ord(char) for char in expected]


def test_default_builder_tokenizes_rewritten_assistant_as_context():
    # [FUNCTION DOC]
    # Behavior: after a rollback checkpoint ending in a tool message, appending a rewritten assistant and a user
    # message yields "<assistant>rewritten answer", user turn and generation prompt. Asserts the first template call
    # rendered only the system/user prefix and the second appended the rewritten assistant (context rendering keeps
    # only the leading system/user messages as base). Then asserts the merge is "context" with the response mask/logprobs
    # padded with zeros for all incremental tokens.
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    rollback_checkpoint = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "tool call"},
        {"role": "tool", "content": "previous result"},
    ]
    rewritten_assistant = {"role": "assistant", "content": "rewritten answer"}
    followup_user = {"role": "user", "content": "continue"}
    updated_messages = rollback_checkpoint + [rewritten_assistant, followup_user]

    incremental = builder.tokenize_context_incremental_messages(rollback_checkpoint, updated_messages)

    expected = "<assistant>rewritten answer\n<user>continue\n<assistant>"
    assert incremental == [ord(char) for char in expected]
    assert [message["role"] for message in tokenizer.calls[0]["messages"]] == ["system", "user"]
    assert [message["role"] for message in tokenizer.calls[1]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert tokenizer.calls[1]["messages"][-1] == rewritten_assistant

    result = builder.merge_context_tokens(rollback_checkpoint, updated_messages, [1, 2, 3])
    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1], [-0.1])
    assert result.kind == "context"
    assert aligned_mask == [1] + [0] * len(incremental)
    assert aligned_logprobs == [-0.1] + [0.0] * len(incremental)


def test_gpt_oss_rollback_context_followup_and_generation_prompt_end_to_end():
    # [FUNCTION DOC]
    # Behavior: GPT-OSS harmony tokenizer: merges a rewritten assistant (with thinking) and a user message onto the
    # rendered rollback checkpoint; asserts the result equals a full render with generation prompt, no inserted or
    # removed tokens, zero-padded alignment, and that the appended text contains the analysis (reasoning preserved),
    # final and user blocks followed by the generation prompt.
    tokenizer = _GptOssRollbackTokenizer()
    builder = GptOssContinuousTokenBuilder(tokenizer)
    rollback_checkpoint = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
    ]
    updated_messages = rollback_checkpoint + [
        {
            "role": "assistant",
            "content": "rewritten answer",
            "thinking": "preserved rollback reasoning",
        },
        {"role": "user", "content": "continue"},
    ]
    runtime_ids = builder._render_tokens(rollback_checkpoint, add_generation_prompt=False)
    expected_ids = builder._render_tokens(updated_messages, add_generation_prompt=True)

    result = builder.merge_context_tokens(rollback_checkpoint, updated_messages, runtime_ids)
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1] * len(runtime_ids),
        [-0.1] * len(runtime_ids),
    )

    assert result.token_ids == expected_ids
    assert result.kind == "context"
    assert result.inserted_token_ids == []
    assert result.removed_prefix_token_count == 0
    assert aligned_mask == [1] * len(runtime_ids) + [0] * result.appended_token_count
    assert aligned_logprobs == [-0.1] * len(runtime_ids) + [0.0] * result.appended_token_count

    appended_text = "".join(chr(token_id) for token_id in result.token_ids[len(runtime_ids) :])
    assert appended_text == (
        "<|start|>assistant<|channel|>analysis<|message|>preserved rollback reasoning<|end|>"
        "<|start|>assistant<|channel|>final<|message|>rewritten answer<|end|>"
        "<|start|>user<|message|>continue<|end|>"
        f"{tokenizer._generation_prompt}"
    )
    assert "preserved rollback reasoning" in appended_text


@pytest.mark.parametrize("family", _ROLLBACK_TEMPLATE_EQUIVALENCE_FAMILIES)
def test_template_compatible_families_merge_rewritten_assistant_as_zero_loss_context(family):
    # [FUNCTION DOC]
    # Behavior: parametrized over template-compatible text families; for each, merging a rewritten assistant and user
    # onto the rendered checkpoint must reproduce a full re-render, kind "context", with the aligned mask/logprobs
    # keeping the retained runtime prefix (minus removed tokens) and zeros for inserted plus appended tokens.
    # Inputs: family - family name.
    tokenizer = _tokenizer_for_text_family(family)
    builder = create_continuous_token_builder(tokenizer, model_family=family)
    rollback_checkpoint = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "question"},
    ]
    updated_messages = rollback_checkpoint + [
        {"role": "assistant", "content": "rewritten answer"},
        {"role": "user", "content": "continue"},
    ]
    runtime_ids = builder._render_tokens(rollback_checkpoint, add_generation_prompt=False)
    expected_ids = builder._render_tokens(updated_messages, add_generation_prompt=True)

    result = builder.merge_context_tokens(rollback_checkpoint, updated_messages, runtime_ids)
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1] * len(runtime_ids),
        [-0.1] * len(runtime_ids),
    )

    assert result.token_ids == expected_ids
    assert result.kind == "context"
    retained_length = len(runtime_ids) - result.removed_prefix_token_count
    appended_length = len(result.inserted_token_ids) + result.appended_token_count
    assert aligned_mask == [1] * retained_length + [0] * appended_length
    assert aligned_logprobs == [-0.1] * retained_length + [0.0] * appended_length


@pytest.mark.parametrize("family", ["qwen", "qwen25", "qwen3", "qwen35"])
@pytest.mark.parametrize("previous_role", ["assistant", "tool"])
def test_qwen_context_boundary_is_role_agnostic_for_rewritten_assistant(family, previous_role):
    # [FUNCTION DOC]
    # Behavior: parametrized over qwen families and previous role (assistant/tool); asserts that runtime ids ending in
    # im_end get a newline inserted while ids already ending with im_end + newline get nothing inserted, independent
    # of the previous role. Inputs: family; previous_role.
    tokenizer = _QwenBoundaryTokenizer()
    builder = create_continuous_token_builder(tokenizer, model_family=family)
    previous_messages = [{"role": previous_role, "content": "previous turn"}]
    updated_messages = previous_messages + [
        {"role": "assistant", "content": "rewritten"},
        {"role": "user", "content": "continue"},
    ]

    missing_separator = builder.merge_context_tokens(
        previous_messages,
        updated_messages,
        [7, tokenizer.im_end_id],
    )
    existing_separator = builder.merge_context_tokens(
        previous_messages,
        updated_messages,
        [7, tokenizer.im_end_id, tokenizer.newline_id],
    )

    assert missing_separator.token_ids[:3] == [7, tokenizer.im_end_id, tokenizer.newline_id]
    assert missing_separator.inserted_token_ids == [tokenizer.newline_id]
    assert existing_separator.token_ids[:3] == [7, tokenizer.im_end_id, tokenizer.newline_id]
    assert existing_separator.inserted_token_ids == []


def test_context_incremental_messages_reject_final_assistant():
    # [FUNCTION DOC]
    # Behavior: asserts incremental context ending with an assistant message raises ValueError "cannot end with assistant".
    tokenizer = _GptOssGenerationRewriteTokenizer()
    builder = GptOssContinuousTokenBuilder(tokenizer)
    previous_messages = [{"role": "tool", "content": "result"}]
    updated_messages = previous_messages + [{"role": "assistant", "content": "rewritten"}]

    with pytest.raises(ValueError, match="cannot end with assistant"):
        builder.tokenize_context_incremental_messages(previous_messages, updated_messages)


def test_gemma4_first_assistant_context_continues_previous_model_tool_turn():
    # [FUNCTION DOC]
    # Behavior: Gemma4 merging an appended assistant after a tool turn: runtime ends with the turn separator and
    # appended ids start with the model turn header; asserts both are dropped so the assistant continues the previous
    # model turn (ids [1, 2, 3, 4], removed count equals separator length, nothing inserted).
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer)
    previous_messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "calling lookup",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {}}}],
        },
        {"role": "tool", "content": "tool result"},
    ]
    appended_messages = [{"role": "assistant", "content": "rewritten"}]
    runtime_ids = [1, 2] + builder._turn_separator_ids
    appended_ids = builder._model_turn_header_ids + [3, 4]

    result = builder._merge_context_token_ids(
        runtime_ids,
        appended_ids,
        previous_messages=previous_messages,
        appended_messages=appended_messages,
    )

    assert result.token_ids == [1, 2, 3, 4]
    assert result.appended_token_count == 2
    assert result.removed_prefix_token_count == len(builder._turn_separator_ids)
    assert result.inserted_token_ids == []
    assert result.kind == "context"


def test_gemma4_first_assistant_context_starts_new_model_turn_after_user():
    # [FUNCTION DOC]
    # Behavior: Gemma4 appended assistant after a user message keeps separator and header (plain concatenation).
    tokenizer = _Gemma4BoundaryTokenizer()
    builder = Gemma4ContinuousTokenBuilder(tokenizer)
    previous_messages = [{"role": "user", "content": "question"}]
    appended_messages = [{"role": "assistant", "content": "answer"}]
    runtime_ids = [1, 2] + builder._turn_separator_ids
    appended_ids = builder._model_turn_header_ids + [3, 4]

    result = builder._merge_context_token_ids(
        runtime_ids,
        appended_ids,
        previous_messages=previous_messages,
        appended_messages=appended_messages,
    )

    assert result.token_ids == runtime_ids + appended_ids
    assert result.appended_token_count == len(appended_ids)
    assert result.removed_prefix_token_count == 0
    assert result.inserted_token_ids == []


@pytest.mark.parametrize(
    ("builder", "expected_tool_response"),
    [
        (
            GptOssContinuousTokenBuilder(_TemplateTokenizer()),
            '<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>"result"<|end|>',
        ),
        (
            Gemma4ContinuousTokenBuilder(_Gemma4BoundaryTokenizer()),
            '<|tool_response>response:lookup{value:<|"|>result<|"|>}<tool_response|>',
        ),
    ],
)
def test_rewritten_assistant_tool_call_is_visible_to_following_tool_group(builder, expected_tool_response):
    # [FUNCTION DOC]
    # Behavior: parametrized over GPT-OSS and Gemma4 builders; when a rewritten assistant with a tool call is followed
    # by its tool response in the same append, asserts the incremental ids contain the tool response rendered with the
    # name "lookup" (the new assistant's tool calls are visible to the tool group).
    # Inputs: builder; expected_tool_response - expected text.
    previous_messages = [{"role": "user", "content": "question"}]
    rewritten_assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "lookup", "arguments": {}},
            }
        ],
    }
    updated_messages = previous_messages + [
        rewritten_assistant,
        {"role": "tool", "tool_call_id": "call_0", "content": "result"},
    ]

    incremental = builder.tokenize_context_incremental_messages(previous_messages, updated_messages)

    expected_tool_ids = [ord(char) for char in expected_tool_response]
    assert any(
        incremental[index : index + len(expected_tool_ids)] == expected_tool_ids
        for index in range(len(incremental) - len(expected_tool_ids) + 1)
    )


def test_default_builder_fuses_generation_prompt_into_last_append_group():
    # [FUNCTION DOC]
    # Behavior: two tool messages appended after a user message are rendered with exactly two template calls (base 3
    # messages without prompt, full 5 with prompt), i.e. the generation prompt is fused into the last group; asserts
    # tools and template kwargs are forwarded on every call.
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer, chat_template_kwargs={"enable_thinking": False})
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    old_messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    new_messages = old_messages + [
        {"role": "tool", "content": "first result", "name": "lookup"},
        {"role": "tool", "content": "second result", "name": "lookup"},
    ]

    builder.tokenize_context_incremental_messages(old_messages, new_messages, tools=tools)

    assert len(tokenizer.calls) == 2
    assert [len(call["messages"]) for call in tokenizer.calls] == [3, 5]
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, True]
    assert all(call["tools"] is tools for call in tokenizer.calls)
    assert all(call["kwargs"] == {"enable_thinking": False} for call in tokenizer.calls)


def test_default_builder_only_fuses_generation_prompt_into_final_append_group():
    # [FUNCTION DOC]
    # Behavior: system then user appended produce 4 template calls with only the last using the generation prompt.
    tokenizer = _RecordingTemplateTokenizer()
    builder = ContinuousTokenBuilder(tokenizer)
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "retry"},
    ]

    builder.tokenize_context_incremental_messages(old_messages, new_messages)

    assert len(tokenizer.calls) == 4
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, False, False, True]


def test_special_builder_can_keep_separate_full_history_generation_prompt():
    # [FUNCTION DOC]
    # Behavior: subclass disables fusing the generation prompt; asserts 4 template calls with message counts 2, 3, 4, 4
    # and only the last with a generation prompt (separate full-history prompt render).
    class FullHistoryGenerationPromptBuilder(ContinuousTokenBuilder):
        # [CLASS DOC]
        # HELPER: Builder subclass that never fuses the generation prompt with the last group.
        def _should_fuse_generation_prompt_with_last_group(self):
            # [FUNCTION DOC]
            # HELPER: Returns False.
            return False

    tokenizer = _RecordingTemplateTokenizer()
    builder = FullHistoryGenerationPromptBuilder(tokenizer)
    old_messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    new_messages = old_messages + [{"role": "user", "content": "retry"}]

    builder.tokenize_context_incremental_messages(old_messages, new_messages)

    assert len(tokenizer.calls) == 4
    assert [len(call["messages"]) for call in tokenizer.calls] == [2, 3, len(new_messages), len(new_messages)]
    assert [call["add_generation_prompt"] for call in tokenizer.calls] == [False, False, False, True]


def test_default_builder_rejects_multi_message_user_or_system_groups():
    # [FUNCTION DOC]
    # Behavior: subclass groups all appended messages together; asserts two user messages in one group raise
    # ValueError "expects one 'user' message per append group".
    class BadGroupingBuilder(ContinuousTokenBuilder):
        # [CLASS DOC]
        # HELPER: Builder subclass with a broken grouping that returns all appended messages as one group.
        def _iter_append_groups(self, appended_messages):
            # [FUNCTION DOC]
            # HELPER: Returns a single group containing every appended message.
            return [appended_messages]

    builder = BadGroupingBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [
        {"role": "user", "content": "retry"},
        {"role": "user", "content": "more context"},
    ]

    with pytest.raises(ValueError, match="expects one 'user' message per append group"):
        builder.tokenize_context_incremental_messages(old_messages, new_messages)


def test_default_builder_appends_assistant_tokens_to_runtime_stream():
    # [FUNCTION DOC]
    # Behavior: merge_assistant_tokens appends generated ids to the runtime stream (kind "assistant"); asserts the
    # aligned mask marks the new tokens 1 and logprobs append the assistant logprobs.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())

    result = builder.merge_assistant_tokens([1, 2, 3], [4, 5])

    assert result.token_ids == [1, 2, 3, 4, 5]
    assert result.appended_token_count == 2
    assert result.kind == "assistant"
    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [0, 1],
        [0.0, -0.1],
        assistant_logprobs=[-0.2, -0.3],
    )
    assert aligned_mask == [0, 1, 1, 1]
    assert aligned_logprobs == [0.0, -0.1, -0.2, -0.3]


def test_assistant_alignment_validates_logprobs():
    # [FUNCTION DOC]
    # Behavior: for an assistant merge, asserts alignment without any logprobs pads the mask with ones and returns None
    # logprobs; and raises ValueError when only assistant logprobs are given, when only response logprobs are given,
    # and when assistant logprobs length differs from the appended count.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2, kind="assistant")

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1])
    assert aligned_mask == [1, 1, 1]
    assert aligned_logprobs is None

    with pytest.raises(ValueError, match="response_logprobs is required"):
        builder.align_response_metadata(result, [1], assistant_logprobs=[-0.1, -0.2])

    with pytest.raises(ValueError, match="assistant_logprobs is required"):
        builder.align_response_metadata(result, [1], [0.0])

    with pytest.raises(ValueError, match="assistant_logprobs length must match"):
        builder.align_response_metadata(result, [1], [0.0], assistant_logprobs=[-0.1])


def test_builder_align_response_metadata_handles_inserted_boundary_tokens():
    # [FUNCTION DOC]
    # Behavior: asserts a context merge with one inserted boundary token pads both mask and logprobs with zeros for the
    # inserted and appended tokens.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(
        token_ids=[1, 2, 99, 3],
        appended_token_count=1,
        kind="context",
        inserted_token_ids=[99],
    )

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1, 1], [0.1, 0.2])

    assert aligned_mask == [1, 1, 0, 0]
    assert aligned_logprobs == [0.1, 0.2, 0.0, 0.0]


def test_alignment_rejects_unknown_merge_kind():
    # [FUNCTION DOC]
    # Behavior: asserts an unknown merge kind raises ValueError "Unknown Continuous Token merge kind".
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(token_ids=[1], appended_token_count=0, kind="unknown")

    with pytest.raises(ValueError, match="Unknown Continuous Token merge kind"):
        builder.align_response_metadata(result, [1])


def test_default_builder_rejects_mutated_message_prefix():
    # [FUNCTION DOC]
    # Behavior: asserts changed prefix messages raise "prefix messages changed" and a shorter update raises
    # "updated_messages is shorter".
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    changed_messages = [{"role": "user", "content": "different"}]

    with pytest.raises(ValueError, match="prefix messages changed"):
        builder.tokenize_context_incremental_messages(old_messages, changed_messages)

    with pytest.raises(ValueError, match="updated_messages is shorter"):
        builder.tokenize_context_incremental_messages(old_messages, [])


def test_default_builder_returns_empty_delta_when_no_message_is_appended():
    # [FUNCTION DOC]
    # Behavior: asserts identical old and new messages produce an empty incremental list.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    messages = [{"role": "user", "content": "question"}]

    assert builder.tokenize_context_incremental_messages(messages, messages) == []


def test_default_builder_rejects_non_prefix_stable_template_deltas():
    # [FUNCTION DOC]
    # Behavior: with a non-prefix-stable tokenizer, asserts render_delta_token_id raises "token-id suffix diff failed".
    builder = ContinuousTokenBuilder(_NonPrefixStableTokenizer())

    with pytest.raises(ValueError, match="token-id suffix diff failed"):
        builder.render_delta_token_id(
            [{"role": "user", "content": "question"}],
            [{"role": "tool", "content": "answer"}],
            add_generation_prompt=True,
        )


def test_subclass_only_overrides_token_level_merge_hook():
    # [FUNCTION DOC]
    # Behavior: subclass overrides only _merge_context_token_ids to insert a 99 boundary and record kwargs; asserts the
    # base orchestration calls it (see assertions below for the forwarded context and result).
    received_context = {}

    class BoundaryBuilder(ContinuousTokenBuilder):
        # [CLASS DOC]
        # HELPER: Builder subclass with a custom token-level merge hook.
        def _merge_context_token_ids(self, runtime_token_ids, appended_token_ids, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs into received_context and returns a context MergeResult of runtime ids, 99, then appended ids,
            # with inserted ids [99]. Side effects: mutates received_context.
            received_context.update(kwargs)
            return MergeResult(
                token_ids=list(runtime_token_ids) + [99] + list(appended_token_ids),
                appended_token_count=len(appended_token_ids),
                kind="context",
                inserted_token_ids=[99],
            )

    builder = BoundaryBuilder(_TemplateTokenizer())
    old_messages = [{"role": "user", "content": "question"}]
    new_messages = old_messages + [{"role": "tool", "content": "answer"}]
    incremental = builder.tokenize_context_incremental_messages(old_messages, new_messages)

    result = builder.merge_context_tokens(old_messages, new_messages, [1, 2, 3])

    assert result.token_ids == [1, 2, 3, 99] + incremental
    assert result.appended_token_count == len(incremental)
    assert result.inserted_token_ids == [99]
    assert result.kind == "context"
    assert received_context == {
        "previous_messages": old_messages,
        "appended_messages": new_messages[len(old_messages) :],
    }


def test_context_alignment_handles_boundary_inserts_and_trims():
    # [FUNCTION DOC]
    # Behavior: for a context MergeResult with one inserted token and one removed prefix token, asserts alignment drops
    # the last runtime mask/logprob entry and pads zeros for inserted plus appended tokens; with no logprobs the mask
    # is aligned the same way and logprobs are None.
    builder = ContinuousTokenBuilder(_TemplateTokenizer())
    result = MergeResult(
        token_ids=[1, 2, 99, 3, 4],
        appended_token_count=2,
        kind="context",
        inserted_token_ids=[99],
        removed_prefix_token_count=1,
    )

    aligned_mask, aligned_logprobs = builder.align_response_metadata(
        result,
        [1, 1, 1],
        [0.1, 0.2, 0.3],
    )
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs == [0.1, 0.2, 0.0, 0.0, 0.0]

    aligned_mask, aligned_logprobs = builder.align_response_metadata(result, [1, 1, 1])
    assert aligned_mask == [1, 1, 0, 0, 0]
    assert aligned_logprobs is None


def test_builder_rejects_unsupported_append_roles():
    # [FUNCTION DOC]
    # Behavior: with allowed_append_roles ["tool"], asserts appending a user raises ValueError mentioning "got 'user'";
    # constructing with an unsupported role "developer" raises "Unsupported Continuous Token append roles"; and
    # "assistant" is accepted and stored as a frozenset.
    builder = ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["tool"])

    with pytest.raises(ValueError, match="got 'user'"):
        builder.tokenize_context_incremental_messages(
            [{"role": "user", "content": "question"}],
            [{"role": "user", "content": "question"}, {"role": "user", "content": "retry"}],
        )

    with pytest.raises(ValueError, match="Unsupported Continuous Token append roles"):
        ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["developer"])

    assistant_builder = ContinuousTokenBuilder(_TemplateTokenizer(), allowed_append_roles=["assistant"])
    assert assistant_builder.allowed_append_roles == frozenset({"assistant"})


def test_model_specific_builders_validate_required_special_tokens():
    # [FUNCTION DOC]
    # Behavior: asserts Qwen, MiniMax, GLM, Gemma4 and DeepSeekV4 builders raise ValueError naming their required special
    # token when the tokenizer cannot resolve it (returns None).
    with pytest.raises(ValueError, match="required token '<\\|im_end\\|>'"):
        QwenContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '\\[e~\\['"):
        MiniMaxContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|observation\\|>'"):
        GLMContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<\\|tool_response>'"):
        Gemma4ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())

    with pytest.raises(ValueError, match="required token '<｜end▁of▁sentence｜>'"):
        DeepSeekV4ContinuousTokenBuilder(_MissingSpecialTokenTokenizer())


def test_model_specific_builders_validate_special_token_id_shape():
    # [FUNCTION DOC]
    # Behavior: asserts a single-element list id for im_end is accepted (merge inserts newline 198), while multiple ids,
    # a negative id, or a newline encoding to multiple ids each raise a specific ValueError.
    builder = QwenContinuousTokenBuilder(_ListSpecialTokenQwenTokenizer())
    assert builder._merge_context_token_ids([1, builder._im_end_id], [2]).token_ids == [
        1,
        builder._im_end_id,
        198,
        2,
    ]

    with pytest.raises(ValueError, match="returned multiple ids"):
        QwenContinuousTokenBuilder(_MultiIdSpecialTokenQwenTokenizer())

    with pytest.raises(ValueError, match="returned invalid id"):
        QwenContinuousTokenBuilder(_InvalidSpecialTokenQwenTokenizer())

    with pytest.raises(ValueError, match="Expected Qwen newline"):
        QwenContinuousTokenBuilder(_MultiTokenNewlineQwenTokenizer())


@pytest.mark.parametrize("previous_role", ["user", "tool"])
def test_deepseek_v4_only_inserts_eos_after_assistant(previous_role):
    # [FUNCTION DOC]
    # Behavior: parametrized over previous role user/tool; asserts DeepSeekV4 inserts nothing between [10, 20] and
    # [30, 40] when the previous message is not an assistant. Inputs: previous_role.
    tokenizer = _DeepSeekToolAppendTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer)

    result = builder._merge_context_token_ids(
        [10, 20],
        [30, 40],
        previous_messages=[{"role": previous_role, "content": "previous"}],
    )

    assert result.token_ids == [10, 20, 30, 40]
    assert result.inserted_token_ids == []


def test_deepseek_v4_inserts_missing_eos_after_assistant():
    # [FUNCTION DOC]
    # Behavior: asserts DeepSeekV4 inserts the eos id after a previous assistant message whose ids lack it.
    tokenizer = _DeepSeekToolAppendTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer)

    result = builder._merge_context_token_ids(
        [10, 20],
        [30, 40],
        previous_messages=[{"role": "assistant", "content": "generated"}],
    )

    assert result.token_ids == [10, 20, tokenizer.eos_id, 30, 40]
    assert result.inserted_token_ids == [tokenizer.eos_id]


def test_deepseek_v4_does_not_duplicate_existing_assistant_eos():
    # [FUNCTION DOC]
    # Behavior: asserts DeepSeekV4 does not duplicate the eos when runtime ids already end with it.
    tokenizer = _DeepSeekToolAppendTokenizer()
    builder = DeepSeekV4ContinuousTokenBuilder(tokenizer)

    result = builder._merge_context_token_ids(
        [10, tokenizer.eos_id],
        [30, 40],
        previous_messages=[{"role": "assistant", "content": "generated"}],
    )

    assert result.token_ids == [10, tokenizer.eos_id, 30, 40]
    assert result.inserted_token_ids == []


def test_unknown_family_fails_during_resolution():
    # [FUNCTION DOC]
    # Behavior: asserts an unknown model_family raises ValueError "Unknown Continuous Token model_family".
    with pytest.raises(ValueError, match="Unknown Continuous Token model_family"):
        create_continuous_token_builder(_DummyTokenizer(), model_family="missing_custom_family")


@pytest.mark.parametrize("model_family", ["", "   ", None])
def test_empty_family_fails_during_resolution(model_family):
    # [FUNCTION DOC]
    # Behavior: parametrized over empty, whitespace and None; asserts resolution raises "model_family must be a
    # non-empty string". Inputs: model_family.
    with pytest.raises(ValueError, match="model_family must be a non-empty string"):
        resolve_continuous_token_model_family(model_family)


# =============================================================================
# Multimodal (VL) continuous token builders, base-class MM hooks, and VL wiring
# (merged from the former tests/utils/test_continuous_token_mm_on_cpu.py)
# =============================================================================


class TestMergeResultTokenFields:
    # [CLASS DOC]
    # Tests that MergeResult is a frozen, token-only dataclass with expected defaults.
    """Verify MergeResult stays token-only and works with VL builders."""

    def test_default_values_text_only(self):
        # [FUNCTION DOC]
        # Behavior: asserts constructing with only ids and count keeps them and defaults kind to "context".
        result = MergeResult(token_ids=[1, 2, 3], appended_token_count=2)
        assert result.token_ids == [1, 2, 3]
        assert result.appended_token_count == 2
        assert result.kind == "context"

    def test_backward_compat_construction(self):
        # [FUNCTION DOC]
        # Behavior: asserts construction with all keyword fields stores kind and inserted ids.
        result = MergeResult(
            token_ids=[10, 20, 30],
            appended_token_count=1,
            kind="assistant",
            inserted_token_ids=[99],
            removed_prefix_token_count=0,
        )
        assert result.token_ids == [10, 20, 30]
        assert result.kind == "assistant"
        assert result.inserted_token_ids == [99]

    def test_frozen_immutability(self):
        # [FUNCTION DOC]
        # Behavior: asserts assigning to a field after construction raises AttributeError (frozen).
        """MergeResult should remain frozen (no assignment after construction)."""
        result = MergeResult(token_ids=[1], appended_token_count=0)
        with pytest.raises(AttributeError):
            result.token_ids = [2]  # type: ignore[misc]


class TestBaseClassMMHooks:
    # [CLASS DOC]
    # Tests for the base builder's multimodal hooks.
    """Verify base class MM hooks behave correctly (NotImplementedError / False)."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates self.builder from a minimal mock tokenizer whose chat template always returns [1, 2, 3].
        """Create a minimal mock tokenizer for base class instantiation."""

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Minimal tokenizer mock.
            def apply_chat_template(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Returns [1, 2, 3] regardless of input.
                return [1, 2, 3]

        self.builder = ContinuousTokenBuilder(MockTokenizer())

    def test_supports_multimodal_default_false(self):
        # [FUNCTION DOC]
        # Behavior: asserts supports_multimodal is False on the class and instance.
        """Base class should return False for supports_multimodal."""
        assert ContinuousTokenBuilder.supports_multimodal() is False
        assert self.builder.supports_multimodal() is False

    def test_render_tokens_with_mm_raises(self):
        # [FUNCTION DOC]
        # Behavior: asserts render_tokens_with_mm raises NotImplementedError on the base builder.
        """Base class render_tokens_with_mm should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not implement render_tokens_with_mm"):
            self.builder.render_tokens_with_mm(
                messages=[{"role": "user", "content": "hi"}],
                images=["fake_image.png"],
            )

    def test_supports_multimodal_classmethod(self):
        # [FUNCTION DOC]
        # Behavior: asserts supports_multimodal works as a classmethod without an instance.
        """supports_multimodal should be callable as classmethod without instance."""
        assert ContinuousTokenBuilder.supports_multimodal() is False

    def test_subclass_can_override_supports_multimodal(self):
        # [FUNCTION DOC]
        # Behavior: asserts a subclass overriding supports_multimodal returns True.
        """A VL subclass that overrides supports_multimodal should return True."""

        class FakeVLBuilder(ContinuousTokenBuilder):
            # [CLASS DOC]
            # HELPER: Fake VL builder subclass.
            @classmethod
            def supports_multimodal(cls) -> bool:
                # [FUNCTION DOC]
                # HELPER: Returns True.
                return True

        assert FakeVLBuilder.supports_multimodal() is True


class TestMultimodalMergeResultWithExistingSubclasses:
    # [CLASS DOC]
    # Tests that existing text subclasses still produce valid MergeResult objects.
    """Ensure existing text subclass _merge_token_ids still produce valid MergeResult."""

    def test_qwen_merge_still_works(self):
        # [FUNCTION DOC]
        # Behavior: with a mock Qwen tokenizer, asserts merging [100, 200, im_end] with [10, 20] inserts newline 198 and
        # reports 2 appended tokens.
        """QwenContinuousTokenBuilder merge should produce token-only MergeResult."""
        from verl.utils.tokenizer.continuous_token import QwenContinuousTokenBuilder

        class MockQwenTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Qwen tokenizer.
            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for a newline, else [1, 2, 3].
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Returns 151645 for im_end, else 0.
                if token == "<|im_end|>":
                    return 151645
                return 0

        builder = QwenContinuousTokenBuilder(MockQwenTokenizer())
        # Simulate: prefix ends with <|im_end|>, appended is [10, 20]
        result = builder._merge_context_token_ids([100, 200, 151645], [10, 20])
        assert result.token_ids == [100, 200, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]
        assert result.appended_token_count == 2


# =============================================================================
# Tests for VL subclasses
# =============================================================================


class TestQwenVLContinuousTokenBuilder:
    # [CLASS DOC]
    # Tests for QwenVLContinuousTokenBuilder.
    """Test QwenVL vision token handling."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates a mock Qwen-VL tokenizer (newline 198, im_end/vision/image_pad ids), a mock processor whose image
        # processor has merge_size 2, and the builder under test.
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        class MockQwenVLTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Qwen-VL tokenizer.
            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for a newline, else [1, 2, 3].
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Maps im_end, vision_start, vision_end and image_pad tokens to Qwen ids, else 0.
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            # [CLASS DOC]
            # HELPER: Mock image processor with merge_size 2.
            merge_size = 2

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor exposing the mock image processor.
            image_processor = MockImageProcessor()

        self.tokenizer = MockQwenVLTokenizer()
        self.processor = MockProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_supports_multimodal(self):
        # [FUNCTION DOC]
        # Behavior: asserts the VL builder supports multimodal.
        assert self.builder.supports_multimodal() is True

    def test_merge_inherits_qwen_newline_patch(self):
        # [FUNCTION DOC]
        # Behavior: asserts the VL builder inherits Qwen newline insertion after im_end.
        """VL builder should still insert newline after im_end (from QwenBuilder)."""
        result = self.builder._merge_context_token_ids([100, 151645], [10, 20])
        assert result.token_ids == [100, 151645, 198, 10, 20]
        assert result.inserted_token_ids == [198]


# =============================================================================
# Tests for wiring factory with VL families
# =============================================================================


class TestWiringVLFactory:
    # [CLASS DOC]
    # Tests for factory behavior with VL families and processors.
    """Test that create_continuous_token_builder handles VL families correctly."""

    def test_vl_family_requires_processor(self):
        # [FUNCTION DOC]
        # Behavior: asserts family "qwen25vl" without a processor raises ValueError "requires a processor".
        """VL families should raise if processor not provided."""
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Qwen2.5-VL tokenizer.
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for newline, else [1, 2, 3].
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Maps Qwen special tokens to ids, else 0.
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        with pytest.raises(ValueError, match="requires a processor"):
            create_continuous_token_builder(
                MockTokenizer(),
                model_family="qwen25vl",
            )

    def test_vl_family_succeeds_with_processor(self):
        # [FUNCTION DOC]
        # Behavior: asserts "qwen25vl" with a processor creates a multimodal QwenVLContinuousTokenBuilder.
        """VL families should instantiate correctly with processor provided."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Qwen2.5-VL tokenizer.
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for newline, else [1, 2, 3].
                if text == "\n":
                    return [198]
                return [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Maps Qwen special tokens to ids, else 0.
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockImageProcessor:
            # [CLASS DOC]
            # HELPER: Mock image processor with merge_size 2.
            merge_size = 2

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor exposing the image processor.
            image_processor = MockImageProcessor()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            model_family="qwen25vl",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_vl_family_inferred_from_model_type_with_processor(self):
        # [FUNCTION DOC]
        # Behavior: asserts model type "qwen2_5_vl" with a processor infers the Qwen VL builder.
        """A registered VL model_type resolves to its processor-backed builder."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock tokenizer.
            name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for newline, else [1, 2, 3].
                return [198] if text == "\n" else [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Returns 151645 for im_end, else 0.
                return {"<|im_end|>": 151645}.get(token, 0)

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor whose image_processor is an ad hoc object with merge_size 2.
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="qwen2_5_vl",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)

    def test_unknown_model_with_processor_falls_back_to_default_vl(self, caplog):
        # [FUNCTION DOC]
        # Behavior: asserts an unknown model type with a multimodal processor creates the default VL builder and logs a
        # warning naming the model type and "vldefault". Inputs: self, caplog.
        """Unrecognized model + multimodal processor -> default VL builder, with a warning."""
        from verl.utils.tokenizer.continuous_token import VLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock tokenizer with only a name.
            name_or_path = "acme/foobar-7b-instruct"

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor with an image processor.
            image_processor = type("IP", (), {"merge_size": 2})()

        with caplog.at_level(logging.WARNING, logger="verl.utils.tokenizer.continuous_token_wiring"):
            builder = create_continuous_token_builder(
                MockTokenizer(),
                hf_model_type="acme_model",
                processor=MockProcessor(),
            )
        assert isinstance(builder, VLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True
        assert "acme_model" in caplog.text
        assert "vldefault" in caplog.text

    def test_gemma4_unified_with_processor_upgrades_to_vl(self):
        # [FUNCTION DOC]
        # Behavior: asserts model type "gemma4" with a processor upgrades to Gemma4VLContinuousTokenBuilder.
        """Gemma4 (unified checkpoint, no vl marker) + processor -> Gemma4 VL builder."""
        from verl.utils.tokenizer.continuous_token import Gemma4VLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Gemma4 tokenizer.
            name_or_path = "google/gemma-4-27b-it"

            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for "<turn|>" plus newline, else [1, 2, 3].
                del add_special_tokens
                return [198] if text == "<turn|>\n" else [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Returns 12345 for "<|tool_response>", else 0.
                return {"<|tool_response>": 12345}.get(token, 0)

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor with an image processor.
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="gemma4",
            processor=MockProcessor(),
        )
        assert isinstance(builder, Gemma4VLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_qwen35_unified_with_processor_upgrades_to_vl(self):
        # [FUNCTION DOC]
        # Behavior: asserts model type "qwen3_5_moe" with a processor upgrades to the Qwen VL builder.
        """Qwen3.5 (unified checkpoint, no vl marker) + processor -> Qwen VL builder."""
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock Qwen3.5 tokenizer.
            name_or_path = "Qwen/Qwen3.5-35B-A3B"

            def encode(self, text, add_special_tokens=False):
                # [FUNCTION DOC]
                # HELPER: Returns [198] for newline, else [1, 2, 3].
                return [198] if text == "\n" else [1, 2, 3]

            def convert_tokens_to_ids(self, token):
                # [FUNCTION DOC]
                # HELPER: Maps Qwen special tokens to ids, else 0.
                mapping = {
                    "<|im_end|>": 151645,
                    "<|vision_start|>": 151652,
                    "<|vision_end|>": 151653,
                    "<|image_pad|>": 151655,
                }
                return mapping.get(token, 0)

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor with an image processor.
            image_processor = type("IP", (), {"merge_size": 2})()

        builder = create_continuous_token_builder(
            MockTokenizer(),
            hf_model_type="qwen3_5_moe",
            processor=MockProcessor(),
        )
        assert isinstance(builder, QwenVLContinuousTokenBuilder)
        assert builder.supports_multimodal() is True

    def test_text_specific_family_with_processor_raises(self):
        # [FUNCTION DOC]
        # Behavior: asserts a text-only family (qwen3) with a multimodal processor raises "multimodal processor was provided".
        """A recognized text-only family paired with a multimodal processor is a misconfiguration."""
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        class MockTokenizer:
            # [CLASS DOC]
            # HELPER: Mock tokenizer with only a name.
            name_or_path = "Qwen/Qwen3-8B"

        class MockProcessor:
            # [CLASS DOC]
            # HELPER: Mock processor with an image processor.
            image_processor = type("IP", (), {"merge_size": 2})()

        with pytest.raises(ValueError, match="multimodal processor was provided"):
            create_continuous_token_builder(
                MockTokenizer(),
                hf_model_type="qwen3",
                processor=MockProcessor(),
            )


# =============================================================================
# Integration tests: VL builder build_initial_tokens + merge_context_tokens end-to-end
# =============================================================================


class _MockQwenVLProcessor:
    # [CLASS DOC]
    # HELPER: Mock Qwen2.5-VL processor with a two-step render: chat template to text with an image placeholder in
    # place of each image block, then a call that encodes characters as code points and expands each placeholder into
    # vision_start, 4 image_pad ids and vision_end (prefix-stable for appended turns).
    """Faithful-ish mock of a Qwen2.5-VL processor's two-step render.

    Mirrors how the real processor works so incremental renders stay prefix-stable:

    1. ``apply_chat_template`` renders the message list to text, emitting an
       ``<|image_pad|>`` placeholder *in place* wherever an image content block
       appears (never at some fixed offset).
    2. ``__call__`` tokenizes that text (each char -> its ``ord``) and expands each
       ``<|image_pad|>`` placeholder in place into a vision span
       (``<|vision_start|>`` + 4 ``<|image_pad|>`` pads + ``<|vision_end|>``),
       simulating merge_size=2 on a 1x4x4 grid -> 4 pad tokens per image.

    Because a newly appended turn (and its placeholder) lands at the end of the
    text and is expanded in place, ``render(prefix)`` is always a token prefix of
    ``render(prefix + new_turn)``.
    """

    _IMAGE_PLACEHOLDER = "<|image_pad|>"

    class _ImageProcessor:
        # [CLASS DOC]
        # HELPER: Nested mock image processor with merge_size 2.
        merge_size = 2

    image_processor = _ImageProcessor()

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: For each message emits "<role>", then for list content the placeholder for image blocks and the text of
        # text blocks (other blocks ignored), or the stringified content; then a newline. Appends "<assistant>" when
        # add_generation_prompt. Always returns text (tokenize and other options ignored).
        parts: list[str] = []
        for message in messages:
            parts.append(f"<{message.get('role')}>")
            content = message.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        parts.append(self._IMAGE_PLACEHOLDER)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
            else:
                parts.append(str(content))
            parts.append("\n")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Takes the first text if a list/tuple is given, splits on the placeholder, encodes each segment as code
        # points and inserts the 6-id vision span between segments. Returns a mapping with input_ids (batch of one) and,
        # when images were present, zero pixel_values with 16 rows per image and image_grid_thw rows (1, 4, 4).
        # The images argument is not inspected; image count comes from the placeholders.
        rendered = text[0] if isinstance(text, list | tuple) else (text or "")
        segments = rendered.split(self._IMAGE_PLACEHOLDER)
        num_images = len(segments) - 1

        token_ids: list[int] = []
        for index, segment in enumerate(segments):
            token_ids.extend(ord(char) for char in segment)
            if index < num_images:
                # Expand this image's placeholder in place: vision_start + 4 pads + vision_end
                token_ids.extend([151652, 151655, 151655, 151655, 151655, 151653])

        result = {"input_ids": [token_ids]}
        if num_images > 0:
            # pixel_values dim0 = raw patches (t*h*w = 1*4*4 = 16 per image)
            import numpy as np

            result["pixel_values"] = np.zeros((num_images * 16, 3, 14, 14), dtype=np.float32)
            # image_grid_thw: each image is (1, 4, 4)
            result["image_grid_thw"] = np.array([[1, 4, 4]] * num_images, dtype=np.int64)

        return result


class _MockQwenVLTokenizer:
    # [CLASS DOC]
    # HELPER: Mock VL tokenizer: newline encodes to [198], any other text to [1000, 1001, 1002]; maps a set of Qwen, GLM,
    # image and media special tokens to fixed ids; simple chat template emitting fixed ids.
    """Mock tokenizer for VL integration tests."""

    name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    def encode(self, text, add_special_tokens=False):
        # [FUNCTION DOC]
        # HELPER: Returns [198] for a newline, else [1000, 1001, 1002].
        if text == "\n":
            return [198]
        return [1000, 1001, 1002]

    def convert_tokens_to_ids(self, token):
        # [FUNCTION DOC]
        # HELPER: Returns the fixed id for known special tokens, else 0.
        mapping = {
            "<|im_end|>": 151645,
            "<|im_start|>": 151644,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|image_pad|>": 151655,
            "<|observation|>": 151333,
            "<|user|>": 151336,
            "<|begin_of_image|>": 151700,
            "<|end_of_image|>": 151701,
            "<|media_start|>": 151800,
            "<|media_end|>": 151801,
        }
        return mapping.get(token, 0)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Emits im_start, then for each message: a vision placeholder triple per image block and [1000, 1001] per
        # text block for list content, or [1000, 1001, 1002] for string content, then im_end; appends im_start for a
        # generation prompt. Returns the string "mock_text_render" when not tokenizing, else the id list.
        """Simple mock chat template."""
        tokens = [151644]  # im_start
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        tokens.extend([151652, 151655, 151653])  # vision placeholder
                    elif isinstance(block, dict) and block.get("type") == "text":
                        tokens.extend([1000, 1001])
            else:
                tokens.extend([1000, 1001, 1002])
            tokens.append(151645)  # im_end
        if add_generation_prompt:
            tokens.append(151644)  # im_start for assistant
        if not tokenize:
            return "mock_text_render"
        return tokens


class TestQwenVLBuildInitialTokens:
    # [CLASS DOC]
    # Integration tests for QwenVL build_initial_tokens.
    """Integration test for QwenVL build_initial_tokens with images."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates the mock tokenizer, mock processor and a QwenVLContinuousTokenBuilder.
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_build_initial_no_images(self):
        # [FUNCTION DOC]
        # Behavior: asserts a text-only message produces a list of ints.
        """Without images, should use text-only path."""
        messages = [{"role": "user", "content": "Hello"}]
        token_ids = self.builder.build_initial_tokens(messages)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)

    def test_build_initial_with_images(self):
        # [FUNCTION DOC]
        # Behavior: asserts a message with one image produces ids containing exactly 4 image_pad tokens (processor expansion).
        """With images, should use processor-expanded token IDs."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "fake_image.png"},
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ]
        token_ids = self.builder.build_initial_tokens(messages, images=["fake_image.png"])
        assert isinstance(token_ids, list)
        assert token_ids.count(151655) == 4


class TestQwenVLMergeContextTokens:
    # [CLASS DOC]
    # Integration tests for QwenVL merge_context_tokens.
    """Integration test for QwenVL merge_context_tokens with images in appended messages."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Creates the mock tokenizer, mock processor and builder.
        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        self.tokenizer = _MockQwenVLTokenizer()
        self.processor = _MockQwenVLProcessor()
        self.builder = QwenVLContinuousTokenBuilder(self.tokenizer, self.processor)

    def test_merge_no_new_images(self):
        # [FUNCTION DOC]
        # Behavior: asserts appending a tool message without images yields a context MergeResult.
        """Without new images in appended messages, should use text-only merge."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_context_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        assert result.kind == "context"

    def test_merge_with_new_images(self):
        # [FUNCTION DOC]
        # Behavior: asserts appending a user message with an image yields a context MergeResult containing image_pad ids.
        """With new images in appended messages, should merge processor-expanded token IDs."""
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "new_image.png"},
                    {"type": "text", "text": "Look at this"},
                ],
            },
        ]
        # Simulate runtime token state
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        result = self.builder.merge_context_tokens(previous, updated, runtime_ids)
        assert isinstance(result, MergeResult)
        assert result.kind == "context"
        assert 151655 in result.token_ids

    def test_merge_with_new_images_rejects_non_prefix_processor_output(self):
        # [FUNCTION DOC]
        # Behavior: with a processor that corrupts the first id of image-bearing renders, asserts merging raises ValueError
        # "suffix diff failed" (prefix invariant check).
        """Incremental rendering should fail fast if the processor output is not append-only."""

        class BadPrefixProcessor(_MockQwenVLProcessor):
            # [CLASS DOC]
            # HELPER: Processor subclass breaking prefix stability for image-bearing calls.
            def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Calls the parent, then if images were passed sets the first input id to 9999. Returns the result.
                result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
                # Corrupt only the image-bearing (full) render so it diverges from the
                # image-free prefix render, breaking the append-only prefix invariant.
                if images:
                    result["input_ids"][0][0] = 9999
                return result

        from verl.utils.tokenizer.continuous_token import QwenVLContinuousTokenBuilder

        builder = QwenVLContinuousTokenBuilder(self.tokenizer, BadPrefixProcessor())
        previous = [{"role": "user", "content": "Hi"}]
        updated = [
            {"role": "user", "content": "Hi"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "new_image.png"},
                    {"type": "text", "text": "Look at this"},
                ],
            },
        ]
        runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
        with pytest.raises(ValueError, match="suffix diff failed"):
            builder.merge_context_tokens(previous, updated, runtime_ids)


@pytest.mark.parametrize(
    "builder_name",
    [
        "GLM46VContinuousTokenBuilder",
        "KimiVLContinuousTokenBuilder",
    ],
)
def test_other_vl_builders_reject_non_prefix_processor_output(builder_name):
    # [FUNCTION DOC]
    # Behavior: parametrized over GLM46V and KimiVL builder class names; constructs each with the mock tokenizer and the
    # corrupting processor and asserts merging an image-bearing append raises "suffix diff failed".
    # Inputs: builder_name - attribute name in the continuous_token module.
    """All VL builders should validate the append-only prefix invariant during merge."""

    class BadPrefixProcessor(_MockQwenVLProcessor):
        # [CLASS DOC]
        # HELPER: Processor subclass breaking prefix stability for image-bearing calls.
        def __call__(self, *, text=None, images=None, return_tensors=None, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Calls the parent, then if images were passed sets the first input id to 9999.
            result = super().__call__(text=text, images=images, return_tensors=return_tensors, **kwargs)
            # Corrupt only the image-bearing (full) render so it diverges from the
            # image-free prefix render, breaking the append-only prefix invariant.
            if images:
                result["input_ids"][0][0] = 9999
            return result

    import verl.utils.tokenizer.continuous_token as continuous_token

    builder_cls = getattr(continuous_token, builder_name)
    builder = builder_cls(_MockQwenVLTokenizer(), BadPrefixProcessor())
    previous = [{"role": "user", "content": "Hi"}]
    updated = [
        {"role": "user", "content": "Hi"},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "new_image.png"},
                {"type": "text", "text": "Look at this"},
            ],
        },
    ]
    runtime_ids = [151644, 1000, 1001, 1002, 151645, 151644]
    with pytest.raises(ValueError, match="suffix diff failed"):
        builder.merge_context_tokens(previous, updated, runtime_ids)


# =============================================================================
# Tests: chat_template_kwargs / mm_processor_kwargs are wired to the VL builder
# at construction time AND actually take effect at render time.
# =============================================================================


class _ConfigurablePadProcessor(_MockQwenVLProcessor):
    # [CLASS DOC]
    # HELPER: Mock VL processor whose number of pad tokens per image is controlled by a pads_per_image kwarg (default 4),
    # recording kwargs of every call.
    """VL processor whose per-image pad count is driven by the ``pads_per_image``
    mm kwarg, mirroring how real ``max_pixels``/``min_pixels`` change how many
    vision tokens an image expands into. Also records the kwargs each call
    receives so tests can assert they were forwarded verbatim.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the empty call_kwargs list (does not call the parent initializer; none needed).
        self.call_kwargs: list[dict] = []

    def __call__(self, *, text=None, images=None, return_tensors=None, pads_per_image=4, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records pads_per_image plus extra kwargs, then renders like the parent but expanding each placeholder to
        # vision_start, pads_per_image image_pad ids, vision_end; adds pixel_values/image_grid_thw when images exist.
        # Side effects: appends to call_kwargs.
        self.call_kwargs.append({"pads_per_image": pads_per_image, **kwargs})
        rendered = text[0] if isinstance(text, list | tuple) else (text or "")
        segments = rendered.split(self._IMAGE_PLACEHOLDER)
        num_images = len(segments) - 1

        token_ids: list[int] = []
        for index, segment in enumerate(segments):
            token_ids.extend(ord(char) for char in segment)
            if index < num_images:
                token_ids.append(151652)
                token_ids.extend([151655] * pads_per_image)
                token_ids.append(151653)

        result = {"input_ids": [token_ids]}
        if num_images > 0:
            import numpy as np

            result["pixel_values"] = np.zeros((num_images * 16, 3, 14, 14), dtype=np.float32)
            result["image_grid_thw"] = np.array([[1, 4, 4]] * num_images, dtype=np.int64)
        return result


class _RecordingTemplateProcessor(_MockQwenVLProcessor):
    # [CLASS DOC]
    # HELPER: Mock VL processor recording the extra kwargs passed to apply_chat_template.
    """VL processor that records the kwargs its ``apply_chat_template`` receives."""

    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Initializes the empty template_kwargs list.
        self.template_kwargs: list[dict] = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, return_dict=False, **kwargs
    ):
        # [FUNCTION DOC]
        # HELPER: Appends a copy of kwargs to template_kwargs, then delegates to the parent template.
        self.template_kwargs.append(dict(kwargs))
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )


def test_vl_builder_creation_forwards_chat_template_and_mm_processor_kwargs():
    # [FUNCTION DOC]
    # Behavior: asserts the factory stores chat_template_kwargs and mm_processor_kwargs on a qwen25vl builder.
    """create_continuous_token_builder must store both kwarg dicts on a VL builder."""
    builder = create_continuous_token_builder(
        _MockQwenVLTokenizer(),
        model_family="qwen25vl",
        processor=_MockQwenVLProcessor(),
        chat_template_kwargs={"enable_thinking": False},
        mm_processor_kwargs={"max_pixels": 12345, "min_pixels": 3136},
    )

    assert isinstance(builder, QwenVLContinuousTokenBuilder)
    assert builder.chat_template_kwargs == {"enable_thinking": False}
    assert builder.mm_processor_kwargs == {"max_pixels": 12345, "min_pixels": 3136}


def test_text_builder_creation_ignores_mm_processor_kwargs():
    # [FUNCTION DOC]
    # Behavior: asserts a default text builder created with mm_processor_kwargs does not have that attribute.
    """mm_processor_kwargs is multimodal-only: a text builder must not carry it."""
    builder = create_continuous_token_builder(
        _TemplateTokenizer(),
        model_family="default",
        mm_processor_kwargs={"max_pixels": 12345},
    )

    assert isinstance(builder, ContinuousTokenBuilder)
    assert not hasattr(builder, "mm_processor_kwargs")


def test_vl_builder_forwards_mm_processor_kwargs_to_processor_call_at_render():
    # [FUNCTION DOC]
    # Behavior: asserts mm_processor_kwargs (pads_per_image 3, max_pixels 999) reach the processor call during
    # build_initial_tokens with an image.
    """mm_processor_kwargs must be forwarded verbatim into the processor call."""
    processor = _ConfigurablePadProcessor()
    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        processor,
        mm_processor_kwargs={"pads_per_image": 3, "max_pixels": 999},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]

    builder.build_initial_tokens(messages, images=["x.png"])

    assert processor.call_kwargs
    assert processor.call_kwargs[-1]["pads_per_image"] == 3
    assert processor.call_kwargs[-1]["max_pixels"] == 999


def test_vl_builder_mm_processor_kwargs_actually_change_rendered_token_count():
    # [FUNCTION DOC]
    # Behavior: asserts builders configured with pads_per_image 2 and 6 produce 2 and 6 image_pad ids respectively.
    """Different mm_processor_kwargs must produce a different number of vision pad
    tokens, proving the kwargs genuinely take effect (not merely stored)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "x.png"},
                {"type": "text", "text": "hi"},
            ],
        }
    ]

    small = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(), _ConfigurablePadProcessor(), mm_processor_kwargs={"pads_per_image": 2}
    )
    large = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(), _ConfigurablePadProcessor(), mm_processor_kwargs={"pads_per_image": 6}
    )

    small_ids = small.build_initial_tokens(messages, images=["x.png"])
    large_ids = large.build_initial_tokens(messages, images=["x.png"])

    assert small_ids.count(151655) == 2
    assert large_ids.count(151655) == 6


def test_vl_builder_forwards_chat_template_kwargs_to_processor_template():
    # [FUNCTION DOC]
    # Behavior: asserts chat_template_kwargs (enable_thinking False) reach the processor's apply_chat_template.
    """chat_template_kwargs must reach the processor's apply_chat_template (VL path),
    not just the tokenizer path exercised by the text-only builder test."""
    processor = _RecordingTemplateProcessor()
    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        processor,
        chat_template_kwargs={"enable_thinking": False},
    )

    builder.build_initial_tokens([{"role": "user", "content": "question"}])

    assert processor.template_kwargs
    assert processor.template_kwargs[-1].get("enable_thinking") is False


def test_vl_builder_folds_processor_sampling_rate_into_mm_processor_kwargs():
    # [FUNCTION DOC]
    # Behavior: asserts a processor exposing feature_extractor.sampling_rate 16000 has it added to mm_processor_kwargs
    # alongside existing keys.
    """A processor exposing feature_extractor.sampling_rate should have that value
    folded into mm_processor_kwargs so audio renders stay aligned."""

    class _AudioProcessor(_MockQwenVLProcessor):
        # [CLASS DOC]
        # HELPER: Mock processor with a feature extractor whose sampling_rate is 16000.
        feature_extractor = type("FE", (), {"sampling_rate": 16000})()

    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        _AudioProcessor(),
        mm_processor_kwargs={"max_pixels": 111},
    )

    assert builder.mm_processor_kwargs["sampling_rate"] == 16000
    assert builder.mm_processor_kwargs["max_pixels"] == 111


def test_vl_builder_preserves_explicit_sampling_rate_over_processor_default():
    # [FUNCTION DOC]
    # Behavior: asserts an explicit sampling_rate 24000 in mm_processor_kwargs is not overwritten by the processor default.
    """An explicit sampling_rate in mm_processor_kwargs must not be overwritten by
    the processor's feature_extractor default."""

    class _AudioProcessor(_MockQwenVLProcessor):
        # [CLASS DOC]
        # HELPER: Mock processor with a feature extractor whose sampling_rate is 16000.
        feature_extractor = type("FE", (), {"sampling_rate": 16000})()

    builder = QwenVLContinuousTokenBuilder(
        _MockQwenVLTokenizer(),
        _AudioProcessor(),
        mm_processor_kwargs={"sampling_rate": 24000},
    )

    assert builder.mm_processor_kwargs["sampling_rate"] == 24000
