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
# Continuous Token (CT) builders: keep a multi-turn rollout's runtime token stream consistent with what the chat
# template would render for the full conversation, by rendering only appended messages as token-level suffix diffs
# against synthetic prefixes and patching the junction between model-generated tokens and appended context
# (inserting/removing boundary tokens). Contains the base text builder, model-specific text builders (GPT-OSS, Qwen,
# MiniMax, GLM, Gemma4, DeepSeek), a VL base and VL builders, plus helpers for tool-name resolution and token lookup.
# Module constants: allowed append roles (tool/user/system/assistant), synthetic system/user messages used as render
# prefixes, a placeholder assistant reasoning string and a dummy tool name.
"""Continuous Token builder implementations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from .chat_template import apply_chat_template
from .tokenizer import build_multimodal_processor_inputs, normalize_token_ids

_SUPPORTED_APPEND_ROLES = frozenset({"tool", "user", "system", "assistant"})
_SYNTHETIC_SYSTEM_MESSAGE: dict[str, Any] = {"role": "system", "content": "continuous token synthetic system"}
_SYNTHETIC_USER_MESSAGE: dict[str, Any] = {"role": "user", "content": "continuous token synthetic user"}
_ASSISTANT_REASONING_CONTENT: str = "reasoning"
_DUMMY_TOOL_NAME = "continuous_token_tool"
MergeKind = Literal["assistant", "context"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeResult:
    # [CLASS DOC]
    # Immutable record of one merge: token_ids (new runtime stream), appended_token_count (appended model/context tokens,
    # excluding inserted boundary tokens), kind ("assistant" or "context", default "context"), inserted_token_ids
    # (CT-inserted boundary tokens, default empty) and removed_prefix_token_count (stale prefix tokens dropped, default 0).
    """Merged runtime tokens plus the edits callers need to align metadata.

    ``token_ids`` is the updated runtime token stream. The other fields describe
    how the stream changed at the merge junction: ``inserted_token_ids`` are
    CT-created boundary tokens, ``appended_token_count`` counts newly appended
    model-generated assistant or context tokens excluding those inserted boundary tokens,
    and ``removed_prefix_token_count`` counts stale prefix tokens dropped before
    appending. Boundary tokens are not model-generated and therefore must not
    carry loss or model logprobs.
    """

    token_ids: list[int]
    appended_token_count: int
    kind: MergeKind = "context"
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class ContinuousTokenBuilder:
    # [CLASS DOC]
    # Base text-only CT builder. Holds the tokenizer, chat template kwargs and allowed append roles. Provides the runtime
    # API (initial render, context merge, assistant merge, metadata alignment) and overridable hooks for per-template
    # grouping, rendering and junction handling. Text builders do not support multimodal rendering.
    """Build and update continuous-token runtime prompts for multi-turn rollouts.

    This class exposes two API layers:

    AgentLoop-facing runtime APIs:
        ``build_initial_tokens`` renders the first prompt, ``merge_context_tokens``
        merges append-only context messages, ``merge_assistant_tokens``
        appends model-generated assistant tokens, and ``align_response_metadata``
        applies the recorded token edits to masks/logprobs.

    Developer extension APIs:
        Model-specific builders should subclass this class and keep the runtime
        API contracts above stable. Chat template specific behavior belongs in hooks
        such as ``_tokenize_tool_group``, ``_tokenize_single_non_tool``,
        ``_merge_context_group``,
        ``_should_fuse_generation_prompt_with_last_group``,
        ``_tokenize_generation_prompt_delta``, and ``_merge_context_token_ids``.
        ``render_delta_token_id`` is the shared suffix-diff helper those hooks can reuse.
    """

    allowed_append_roles: frozenset[str] = _SUPPORTED_APPEND_ROLES

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        allowed_append_roles: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        # [FUNCTION DOC]
        # Stores tokenizer and chat_template_kwargs (default empty mapping). If allowed_append_roles is given, converts to a
        # frozen set and validates it is a subset of the supported roles, storing it on the instance (otherwise the class
        # default is used).
        # Exceptions: ValueError listing unknown roles.
        # Text-only base: no processor / mm_processor_kwargs. All multimodal state
        # (processor, mm_processor_kwargs, sampling-rate defaults) lives in the VL
        # layer so a text builder never carries multimodal parameters it cannot use.
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}
        if allowed_append_roles is not None:
            allowed_roles = frozenset(allowed_append_roles)
            unknown_roles = allowed_roles - _SUPPORTED_APPEND_ROLES
            if unknown_roles:
                raise ValueError(f"Unsupported Continuous Token append roles: {sorted(unknown_roles)}")
            self.allowed_append_roles = allowed_roles

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Renders the full initial prompt token ids with a generation prompt and optional tools via the chat template.
        # Multimodal arguments are ignored by the text base. Outputs: list of int.
        # Text-only builders ignore multimodal inputs; VL builders override this.
        del mm_inputs_out
        return self._render_tokens(messages, add_generation_prompt=True, tools=tools)

    def tokenize_context_incremental_messages(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Tokenizes messages appended after previous_messages so the result equals what the template adds.
        # Behavior: validates append-only; returns [] if nothing appended. Splits appended messages into groups (runs of tool
        # messages form one group, everything else one per group). Asks whether to fuse the generation prompt into the last
        # group. For each group: tool groups use the tool-group hook with the messages processed so far; user/system/assistant
        # groups must contain one message, an assistant cannot be the last group, and use the single-message hook; any other
        # role raises. Each group's ids are merged into the incremental list via the merge-group hook, then the group is added
        # to the processed messages. If not fused, appends the generation-prompt delta computed from the full updated list.
        # Outputs: list of int. Exceptions: ValueError for append violations, bad groups, trailing assistant, unknown roles.
        self._assert_append_only(previous_messages, updated_messages)
        appended_messages = updated_messages[len(previous_messages) :]
        if not appended_messages:
            return []
        incremental_ids: list[int] = []
        processed_messages = list(previous_messages)

        groups = self._iter_append_groups(appended_messages)
        fuse_generation_prompt = self._should_fuse_generation_prompt_with_last_group()

        for index, group in enumerate(groups):
            add_generation_prompt = fuse_generation_prompt and index == len(groups) - 1
            role = group[0].get("role")
            if role == "tool":
                group_token_ids = self._tokenize_tool_group(
                    group,
                    previous_messages=processed_messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            elif role in {"user", "system", "assistant"}:
                # System appends can represent retry/control messages; unsupported templates will fail in suffix diff.
                if len(group) != 1:
                    raise ValueError(
                        f"Continuous Token expects one {role!r} message per append group, got {len(group)}"
                    )
                if role == "assistant" and index == len(groups) - 1:
                    raise ValueError("Continuous Token context incremental messages cannot end with assistant")
                group_token_ids = self._tokenize_single_non_tool(
                    group[0],
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
            else:
                raise ValueError(f"Unsupported Continuous Token append role: {role!r}")
            self._merge_context_group(
                incremental_ids,
                group_token_ids,
                group=group,
                processed_messages=processed_messages,
                tools=tools,
            )
            processed_messages.extend(group)

        if not fuse_generation_prompt:
            incremental_ids.extend(self._tokenize_generation_prompt_delta(updated_messages, tools=tools))
        return incremental_ids

    def _merge_context_group(
        self,
        incremental_ids: list[int],
        group_token_ids: list[int],
        *,
        group: list[dict[str, Any]],
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Hook: appends a group's token ids to the incremental list. Side effects: mutates incremental_ids.
        """Merge one encoded context group into the incremental token stream."""
        incremental_ids.extend(group_token_ids)

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Computes appended context token ids and merges them with the runtime stream via the junction hook, passing
        # previous_messages and appended_messages as keyword context. Outputs: MergeResult of kind "context".
        appended_messages = updated_messages[len(previous_messages) :]
        appended_ids = self.tokenize_context_incremental_messages(previous_messages, updated_messages, tools=tools)
        return self._merge_context_token_ids(
            runtime_token_ids,
            appended_ids,
            previous_messages=previous_messages,
            appended_messages=appended_messages,
        )

    def merge_assistant_tokens(self, runtime_token_ids: list[int], assistant_token_ids: list[int]) -> MergeResult:
        # [FUNCTION DOC]
        # Concatenates runtime ids with model-generated assistant ids; returns a MergeResult of kind "assistant" whose
        # appended count is the assistant token count. No boundary edits.
        """Merge model-generated assistant tokens into the runtime token stream."""
        merged_token_ids = list(runtime_token_ids) + list(assistant_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(assistant_token_ids),
            kind="assistant",
        )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Hook: plain concatenation of runtime and appended context ids, ignoring extra kwargs; returns a "context" MergeResult.
        """Merge runtime prefix tokens and appended context tokens.

        Model-specific builders usually override this hook for boundary handling,
        such as inserting or trimming tokens at the prefix/appended-token junction.
        ``merge_context_tokens`` also supplies message-level boundary context as
        keyword arguments for builders whose token edits depend on message roles.
        """
        merged_token_ids = list(runtime_token_ids) + list(appended_token_ids)
        return MergeResult(
            token_ids=merged_token_ids,
            appended_token_count=len(appended_token_ids),
            kind="context",
        )

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Renders messages through the module's apply_chat_template wrapper with tokenization, the given generation
        # prompt flag, tools and the stored chat template kwargs, and normalizes to a list of ints.
        tokenized = apply_chat_template(
            self.tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **self.chat_template_kwargs,
        )
        return normalize_token_ids(tokenized)

    def render_delta_token_id(
        self,
        prefix_messages: list[dict[str, Any]],
        appended_messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Renders prefix_messages alone (no generation prompt) and prefix plus appended (with the given generation flag) and
        # returns the token suffix of the full render beyond the prefix render.
        # Exceptions: ValueError naming the appended roles (or "generation_prompt" if none) when the full render does not start
        # with the prefix render.
        """Render prefix/full prompts as token IDs and return the token-level suffix."""
        prefix_token_ids = self._render_tokens(prefix_messages, add_generation_prompt=False, tools=tools)
        full_token_ids = self._render_tokens(
            prefix_messages + appended_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if full_token_ids[: len(prefix_token_ids)] != prefix_token_ids:
            roles = [message.get("role") for message in appended_messages] or ["generation_prompt"]
            raise ValueError(f"Continuous Token token-id suffix diff failed for roles: {roles}")
        return full_token_ids[len(prefix_token_ids) :]

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Hook: tokenizes a run of tool messages as the delta after a synthetic system, synthetic user and a synthetic
        # assistant whose tool calls match the tool messages.
        synthetic_assistant = self._synthetic_assistant_for_tools(tool_messages)
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            tool_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Hook: tokenizes one non-tool message as the delta after the synthetic system and user messages.
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            [message],
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # [FUNCTION DOC]
        # Hook: returns True, meaning the generation prompt is rendered together with the last appended group.
        """Whether the last append-group render should also add the generation prompt.

        Builders whose chat template derives the generation scaffold from more than
        the appended group may return ``False`` and implement
        :meth:`_tokenize_generation_prompt_delta` using the required history.
        """
        return True

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Hook: returns the tokens added only by the generation prompt for the full updated conversation (delta with no
        # appended messages).
        """Tokenize the tokens added only by ``add_generation_prompt=True``."""
        return self.render_delta_token_id(updated_messages, [], add_generation_prompt=True, tools=tools)

    def _iter_append_groups(self, appended_messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        # [FUNCTION DOC]
        # HELPER: Groups appended messages: consecutive tool messages become one group; each other message is its own group.
        # Outputs: list of message lists preserving order.
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(appended_messages):
            role = appended_messages[index].get("role")
            if role == "tool":
                end = index + 1
                while end < len(appended_messages) and appended_messages[end].get("role") == "tool":
                    end += 1
                groups.append(appended_messages[index:end])
                index = end
            else:
                groups.append([appended_messages[index]])
                index += 1
        return groups

    def _assert_append_only(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Validates updated_messages is previous_messages plus appended messages whose roles are allowed.
        # Exceptions: ValueError if shorter, if the prefix differs, or if an appended role is not allowed.
        if len(updated_messages) < len(previous_messages):
            raise ValueError("Continuous Token messages must be append-only; updated_messages is shorter")
        if updated_messages[: len(previous_messages)] != previous_messages:
            raise ValueError("Continuous Token messages must be append-only; prefix messages changed")
        for message in updated_messages[len(previous_messages) :]:
            role = message.get("role")
            if role not in self.allowed_append_roles:
                raise ValueError(
                    f"Continuous Token only supports appending roles {sorted(self.allowed_append_roles)}, got {role!r}"
                )

    def _synthetic_assistant_for_tools(
        self,
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Builds a synthetic assistant message with empty content, placeholder reasoning_content and one function tool
        # call per tool message (id from the message's tool_call_id or a dummy based on index, name from the message or the
        # dummy tool name, empty arguments).
        tool_calls = []
        for index, tool_message in enumerate(tool_messages):
            tool_call = {
                "id": _tool_call_id_or_dummy(tool_message, index),
                "type": "function",
                "function": {
                    "name": _tool_message_name_or_dummy(tool_message),
                    "arguments": {},
                },
            }
            tool_calls.append(tool_call)
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }

    def align_response_metadata(
        self,
        merge_result: MergeResult,
        response_mask: list[int],
        response_logprobs: list[float] | None = None,
        *,
        assistant_logprobs: list[float] | None = None,
    ) -> tuple[list[int], list[float] | None]:
        # [FUNCTION DOC]
        # Applies a MergeResult's edits to response-side metadata.
        # Behavior: copies mask and logprobs; if logprobs are absent but assistant_logprobs given, raises. Trims the last
        # removed_prefix_token_count entries from both. Appends 0 / 0.0 for each inserted boundary token. For kind assistant,
        # appends mask 1 per appended token and, when logprobs are tracked, the assistant logprobs (which are required when
        # tokens were appended and must match the appended count; treated as empty when nothing was appended). For kind
        # context, appends 0 / 0.0 per appended token. Other kinds raise.
        # Outputs: tuple (mask list, logprob list or None). Exceptions: ValueError as described.
        """Align response masks and logprobs after a Continuous Token merge.

        ``MergeResult`` records token edits at the runtime-prefix boundary. This
        method applies the same edits to response-side metadata: trimming
        metadata for removed prefix tokens, assigning zero mask/logprob to
        inserted boundary or context tokens, and assigning model-generated assistant
        mask/logprobs to appended assistant tokens.
        """
        aligned_mask = list(response_mask)
        aligned_logprobs = list(response_logprobs) if response_logprobs is not None else None
        if aligned_logprobs is None and assistant_logprobs is not None:
            raise ValueError("response_logprobs is required when assistant_logprobs is provided")

        # If merge trimmed tokens from the current prefix, trim their metadata too.
        if merge_result.removed_prefix_token_count:
            aligned_mask = aligned_mask[: -merge_result.removed_prefix_token_count]
            if aligned_logprobs is not None:
                aligned_logprobs = aligned_logprobs[: -merge_result.removed_prefix_token_count]

        # Boundary tokens are added by CT itself, so they get mask/logprob 0.
        inserted_token_count = len(merge_result.inserted_token_ids)
        aligned_mask += [0] * inserted_token_count
        if aligned_logprobs is not None:
            aligned_logprobs += [0.0] * inserted_token_count

        # Model-generated assistant tokens get mask 1 and their logprobs; context tokens get mask/logprob 0.
        if merge_result.kind == "assistant":
            aligned_mask += [1] * merge_result.appended_token_count
            if aligned_logprobs is not None:
                if assistant_logprobs is None:
                    if merge_result.appended_token_count:
                        raise ValueError("assistant_logprobs is required for assistant Continuous Token alignment")
                    assistant_logprobs = []
                if len(assistant_logprobs) != merge_result.appended_token_count:
                    raise ValueError(
                        "assistant_logprobs length must match appended assistant token count, "
                        f"got {len(assistant_logprobs)} and {merge_result.appended_token_count}"
                    )
                aligned_logprobs += list(assistant_logprobs)
        elif merge_result.kind == "context":
            aligned_mask += [0] * merge_result.appended_token_count
            if aligned_logprobs is not None:
                aligned_logprobs += [0.0] * merge_result.appended_token_count
        else:
            raise ValueError(f"Unknown Continuous Token merge kind: {merge_result.kind!r}")

        return aligned_mask, aligned_logprobs

    # === Multimodal hooks (VL subclasses override these) ===

    @classmethod
    def supports_multimodal(cls) -> bool:
        # [FUNCTION DOC]
        # Class-level flag: returns False for text builders (VL subclasses return True).
        """Whether this builder handles vision inputs.

        VL subclasses override this to return True. Used by the wiring layer
        to decide whether to pass images through the CT pipeline.
        """
        return False

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Multimodal rendering hook; not implemented in the text base.
        # Exceptions: always raises NotImplementedError.
        """Render messages with images through the processor.

        Unlike ``_render_tokens`` which uses only the tokenizer, this method
        invokes the full multimodal processor so image placeholders are expanded
        into the same token IDs the rollout backend will consume. VL subclasses apply
        their ``self.mm_processor_kwargs`` (min/max pixels, sampling rate, ...) captured
        at construction; the text base does not implement this method.

        Args:
            messages: OpenAI-format message list with image content items.
            images: List of PIL images (or paths), one per image content item.
            add_generation_prompt: Whether to append the generation prompt.

        Returns:
            Token IDs rendered by the multimodal processor. Pixel tensors are
            intentionally not returned here; final multimodal tensors are built
            from the full image list during agent-loop postprocessing.

        Raises:
            NotImplementedError: Unless overridden by a VL subclass.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement render_tokens_with_mm.")


class GptOssContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # GPT-OSS (harmony format) builder: encodes tool responses and plain historical assistant turns directly as harmony
    # text instead of template suffix diffs, and computes the generation prompt as a separate delta.
    """GPT-OSS tool-response formatting."""

    def _tokenize_single_non_tool(
        self,
        message: dict[str, Any],
        *,
        add_generation_prompt: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # For a plain assistant message without tool calls, directly encodes the harmony text for its analysis (if any) and
        # final channels without special tokens; otherwise delegates to the base hook.
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            del add_generation_prompt, tools
            # Context incremental messages currently reject a final assistant turn,
            # so a plain GPT-OSS assistant here is always historical context. Keep
            # its analysis/reasoning channel before the final channel, and close
            # both with <|end|> rather than the training-only final <|return|>.
            # If final assistant context is supported in the future, this function may need change.
            return self.tokenizer.encode(self._format_plain_assistant_context(message), add_special_tokens=False)
        return super()._tokenize_single_non_tool(
            message,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # [FUNCTION DOC]
        # Returns False so the generation prompt is derived separately.
        # Tool responses are encoded directly rather than through a chat-template
        # suffix diff, so keep generation-prompt derivation as a separate step.
        return False

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Encodes each tool message as a harmony tool response (tool name resolved from the message, its call id, or the
        # matching preceding assistant tool call; required) and tokenizes the concatenated text without special tokens.
        # Exceptions: ValueError from tool-name resolution when no name can be determined.
        del tools, add_generation_prompt
        response_text = "".join(
            self._format_tool_response(
                tool_message,
                _resolve_required_tool_name(
                    tool_message,
                    index,
                    tool_messages,
                    previous_messages,
                ),
            )
            for index, tool_message in enumerate(tool_messages)
        )
        return self.tokenizer.encode(response_text, add_special_tokens=False)

    @staticmethod
    def _format_tool_response(tool_message: dict[str, Any], tool_name: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Formats a tool response: stringifies content, JSON-encodes it compactly (non-ASCII kept) and wraps it in a
        # harmony message from functions.<tool_name> to the assistant on the commentary channel ending with the end token.
        content = _stringify_tool_content(tool_message.get("content", ""))
        json_content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
        return f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{json_content}<|end|>"

    @staticmethod
    def _format_plain_assistant_context(message: dict[str, Any]) -> str:
        # [FUNCTION DOC]
        # HELPER: Formats a plain historical assistant turn: picks thinking from "thinking", then "reasoning_content", then
        # "reasoning"; if non-empty emits an analysis-channel message, then always a final-channel message with the
        # stringified content, each closed with the end token (not the return token).
        thinking = message.get("thinking")
        if thinking is None:
            thinking = message.get("reasoning_content")
        if thinking is None:
            thinking = message.get("reasoning")

        rendered = ""
        if thinking:
            thinking_text = _stringify_tool_content(thinking)
            rendered += f"<|start|>assistant<|channel|>analysis<|message|>{thinking_text}<|end|>"
        content = _stringify_tool_content(message.get("content", ""))
        return rendered + f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"


class QwenContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # Qwen ChatML builder: if the runtime stream ends with the im_end token, inserts the newline token the template
    # renders after it before appending context.
    """Qwen ChatML boundary handling.

    Qwen2, Qwen2.5, Qwen3, and Qwen3.5 templates render ``<|im_end|>\n`` after a turn,
    while generation may stop at ``<|im_end|>``. When the runtime prefix ends
    there, insert the missing newline before appending context tokens.

    Note: when a Qwen3/Qwen3.5 context assistant message has no thinking, grouped
    CT encoding renders that assistant in a synthetic context and may produce an
    empty ``<think></think>`` block. Rendering that assistant together with the
    following context messages through the full chat template does not produce
    the empty thinking block.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the base, verifies "\n" encodes to exactly one token (ValueError otherwise) and resolves the im_end id.
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected Qwen newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._im_end_id = require_token_id(tokenizer, "<|im_end|>")

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Junction hook: copies runtime ids, appends the newline (recorded as inserted) when the last runtime token is im_end,
        # then appends context ids; returns a "context" MergeResult.
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._im_end_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted_token_ids,
        )


class MiniMaxContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # MiniMax builder: same newline-insertion rule as Qwen but keyed on the MiniMax end-of-turn token "[e~[".
    """MiniMax boundary handling.

    MiniMax templates render ``[e~[\n`` after a turn, while generation may stop
    at ``[e~[``. When the runtime prefix ends there, insert the missing newline
    before appending context tokens.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the base, verifies "\n" encodes to exactly one token (ValueError otherwise) and resolves the "[e~[" id.
        super().__init__(tokenizer, **kwargs)
        newline_ids = tokenizer.encode("\n", add_special_tokens=False)
        if len(newline_ids) != 1:
            raise ValueError(f"Expected MiniMax newline to tokenize to one token, got {newline_ids!r}")
        self._newline_id = int(newline_ids[0])
        self._eos_id = require_token_id(tokenizer, "[e~[")

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Junction hook: inserts the newline token when the runtime stream ends with the MiniMax end token, then appends the
        # context ids; returns a "context" MergeResult.
        prefix = list(runtime_token_ids)
        inserted_token_ids: list[int] = []
        if prefix and prefix[-1] == self._eos_id:
            prefix.append(self._newline_id)
            inserted_token_ids.append(self._newline_id)
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            inserted_token_ids=inserted_token_ids,
        )


class GLMContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # GLM builder: observation/user tokens act both as stop tokens and as the next message's start token, so a trailing one
    # in the runtime stream is removed before appending context (which re-renders it).
    """GLM observation/user boundary handling.

    ``<|observation|>`` and ``<|user|>`` can be both assistant stop tokens and
    next-message start tokens. If the runtime prefix ends with either, remove
    that token before appending the next context segment.
    """

    def __init__(self, tokenizer: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the base and resolves the observation and user token ids into a set of ambiguous boundary ids.
        super().__init__(tokenizer, **kwargs)
        self._observation_id = require_token_id(tokenizer, "<|observation|>")
        self._user_id = require_token_id(tokenizer, "<|user|>")
        self._ambiguous_boundary_ids = {self._observation_id, self._user_id}

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Junction hook: if the last runtime token is an ambiguous boundary id, drops it and records one removed prefix token;
        # returns a "context" MergeResult with the concatenation.
        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if prefix and prefix[-1] in self._ambiguous_boundary_ids:
            prefix = prefix[:-1]
            removed_prefix_token_count = 1
        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            removed_prefix_token_count=removed_prefix_token_count,
        )


class Gemma4ContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # Gemma4 builder: inserts turn separators before appended user turns when needed, renders tool responses as
    # continuations of the model turn and derives the generation scaffold separately based on the preceding message type.
    """Gemma4 turn-continuation and tool-response boundary handling."""

    def __init__(self, tokenizer: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the base, resolves the tool_response token id and tokenizes the "<turn|>\n" separator and the
        # "<|turn>model\n" header (each must be non-empty, else ValueError).
        super().__init__(tokenizer, **kwargs)
        self._tool_response_id = require_token_id(tokenizer, "<|tool_response>")
        turn_separator_ids = normalize_token_ids(tokenizer.encode("<turn|>\n", add_special_tokens=False))
        if not turn_separator_ids:
            raise ValueError("Expected Gemma4 '<turn|>\\n' turn separator to tokenize to at least one token")
        self._turn_separator_ids = turn_separator_ids
        model_turn_header_ids = normalize_token_ids(tokenizer.encode("<|turn>model\n", add_special_tokens=False))
        if not model_turn_header_ids:
            raise ValueError("Expected Gemma4 '<|turn>model\\n' header to tokenize to at least one token")
        self._model_turn_header_ids = model_turn_header_ids

    def _merge_context_group(
        self,
        incremental_ids: list[int],
        group_token_ids: list[int],
        *,
        group: list[dict[str, Any]],
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Merge-group hook: before a user group, appends the turn separator ids when the context so far does not already end
        # with them, then delegates to the base merge (append group ids).
        if group[0].get("role") == "user" and self._needs_user_turn_separator(
            incremental_ids,
            processed_messages=processed_messages,
            tools=tools,
        ):
            incremental_ids.extend(self._turn_separator_ids)
        super()._merge_context_group(
            incremental_ids,
            group_token_ids,
            group=group,
            processed_messages=processed_messages,
            tools=tools,
        )

    def _needs_user_turn_separator(
        self,
        incremental_ids: list[int],
        *,
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns False when there is no incremental output and no processed history; otherwise True iff the context
        # prefix does not end with the turn separator.
        if not incremental_ids and not processed_messages:
            return False
        return not self._context_prefix_ends_with_turn_separator(
            incremental_ids,
            processed_messages=processed_messages,
            tools=tools,
        )

    def _context_prefix_ends_with_turn_separator(
        self,
        incremental_ids: list[int],
        *,
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        # [FUNCTION DOC]
        # HELPER: Checks whether the context rendered so far ends with the turn separator: if there are incremental ids, checks
        # their tail; else if there is processed history, renders it (no generation prompt) and checks the tail; else False.
        separator_ids = self._turn_separator_ids
        if incremental_ids:
            return incremental_ids[-len(separator_ids) :] == separator_ids
        if not processed_messages:
            return False
        rendered_ids = self._render_tokens(processed_messages, add_generation_prompt=False, tools=tools)
        return rendered_ids[-len(separator_ids) :] == separator_ids

    def _should_fuse_generation_prompt_with_last_group(self) -> bool:
        # [FUNCTION DOC]
        # Returns False: Gemma4's generation scaffold is computed separately from the preceding message type.
        # Gemma4's generation scaffold depends on the preceding message type and
        # is handled by the model-specific hook below.
        return False

    def _tokenize_tool_group(
        self,
        tool_messages: list[dict[str, Any]],
        *,
        previous_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Tokenizes a run of tool messages as the delta after synthetic system/user plus a synthetic assistant whose tool
        # calls carry each message's call id (or a dummy) and its required resolved tool name. Each tool message is copied with
        # tool_call_id set to that id and name filled in when missing. Generation prompt is never added here.
        # Exceptions: ValueError when a tool name cannot be resolved or the suffix diff fails.
        del add_generation_prompt
        tool_calls = []
        rendered_tool_messages = []
        for index, tool_message in enumerate(tool_messages):
            call_id = _tool_call_id_or_dummy(tool_message, index)
            resolved_name = _resolve_required_tool_name(
                tool_message,
                index,
                tool_messages,
                previous_messages,
            )
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": resolved_name,
                        "arguments": {},
                    },
                }
            )
            rendered_message = dict(tool_message)
            rendered_message["tool_call_id"] = call_id
            if not rendered_message.get("name"):
                rendered_message["name"] = resolved_name
            rendered_tool_messages.append(rendered_message)
        synthetic_assistant = {
            "role": "assistant",
            "content": "",
            "reasoning_content": _ASSISTANT_REASONING_CONTENT,
            "tool_calls": tool_calls,
        }
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, synthetic_assistant],
            rendered_tool_messages,
            tools=tools,
        )

    def _tokenize_generation_prompt_delta(
        self,
        updated_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Returns [] unless the last updated message is user or system; otherwise the generation-prompt delta rendered after
        # synthetic system/user followed by that last message.
        last_message = updated_messages[-1]
        if last_message.get("role") not in {"user", "system"}:
            return []
        return self.render_delta_token_id(
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE, last_message],
            [],
            add_generation_prompt=True,
            tools=tools,
        )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Junction hook for Gemma4.
        # Behavior: when the first appended message is assistant and the last non-tool previous message is assistant (the
        # assistant continues the same model turn), removes a trailing turn separator from the runtime prefix (recorded as
        # removed prefix tokens) and drops a leading model-turn header from the appended ids. When the first appended message
        # is a tool message and neither the prefix end nor the appended start is already the tool_response token, inserts that
        # token (recorded as inserted). Returns a "context" MergeResult; appended count reflects the possibly trimmed appended ids.
        # Inputs: kwargs previous_messages and appended_messages (default empty).
        previous_messages = kwargs.get("previous_messages", [])
        appended_messages = kwargs.get("appended_messages", [])
        prefix = list(runtime_token_ids)
        appended = list(appended_token_ids)
        inserted_token_ids: list[int] = []
        removed_prefix_token_count = 0

        # Gemma renders tool messages inside the preceding model turn. An
        # appended assistant therefore continues that turn only when the latest
        # non-tool message is also assistant; after user/system it must retain
        # both the previous turn separator and its own model-turn header.
        if (
            appended_messages
            and appended_messages[0].get("role") == "assistant"
            and self._last_non_tool_role(previous_messages) == "assistant"
        ):
            separator_ids = self._turn_separator_ids
            if prefix[-len(separator_ids) :] == separator_ids:
                del prefix[-len(separator_ids) :]
                removed_prefix_token_count = len(separator_ids)

            header_ids = self._model_turn_header_ids
            if appended[: len(header_ids)] == header_ids:
                del appended[: len(header_ids)]

        # Gemma's tool block opens with <|tool_response>. The synthetic-prefix
        # suffix diff attributes that boundary token to the diffed-away assistant
        # turn, so it is missing from ``appended``; re-insert it at the
        # junction. Guard against double insertion in case the prefix already ends
        # with it or the diff happens to retain it.
        if (
            appended_messages
            and appended_messages[0].get("role") == "tool"
            and prefix[-1:] != [self._tool_response_id]
            and appended[:1] != [self._tool_response_id]
        ):
            prefix.append(self._tool_response_id)
            inserted_token_ids.append(self._tool_response_id)

        return MergeResult(
            token_ids=prefix + appended,
            appended_token_count=len(appended),
            kind="context",
            inserted_token_ids=inserted_token_ids,
            removed_prefix_token_count=removed_prefix_token_count,
        )

    @staticmethod
    def _last_non_tool_role(messages: list[dict[str, Any]]) -> str | None:
        # [FUNCTION DOC]
        # HELPER: Returns the role of the last message whose role is not "tool", or None.
        for message in reversed(messages):
            role = message.get("role")
            if role != "tool":
                return role
        return None


def require_token_id(tokenizer: Any, token: str) -> int:
    # [FUNCTION DOC]
    # Resolves a required special token to a single non-negative int id via convert_tokens_to_ids, unwrapping a
    # one-element list.
    # Exceptions: ValueError when the id is None, a list of other length, or not a non-negative int.
    # KNOWN ISSUE: many tokenizers return the unk id rather than None for unknown tokens, which this check accepts.
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not define required token {token!r}")
    if isinstance(token_id, list):
        if len(token_id) != 1:
            raise ValueError(f"Tokenizer returned multiple ids for required token {token!r}: {token_id!r}")
        token_id = token_id[0]
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Tokenizer returned invalid id for required token {token!r}: {token_id!r}")
    return token_id


def _stringify_tool_content(content: Any) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "" for None, strings unchanged, concatenated text of dict parts of type "text" for lists, else the
    # string conversion.
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _tool_message_name_or_dummy(tool_message: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the tool message's name as string if truthy, else the dummy tool name.
    if tool_message.get("name"):
        return str(tool_message["name"])
    return _DUMMY_TOOL_NAME


def _tool_call_id_or_dummy(tool_message: dict[str, Any], index: int) -> Any:
    # [FUNCTION DOC]
    # HELPER: Returns the message's tool_call_id when not None, else "continuous_token_call_<index>".
    if tool_message.get("tool_call_id") is not None:
        return tool_message["tool_call_id"]
    return f"continuous_token_call_{index}"


def _latest_assistant_tool_call_names(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, str], list[str | None]]:
    # [FUNCTION DOC]
    # HELPER: Finds the most recent assistant message and extracts its tool call names.
    # Outputs: (mapping from tool call id string to function name for calls having both, list of names by position with None
    # for non-dict calls or nameless calls). Returns (empty mapping, empty list) if no assistant exists or its tool_calls is
    # not a list.
    tool_names_by_id: dict[str, str] = {}
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            return tool_names_by_id, []
        positional_tool_names: list[str | None] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                positional_tool_names.append(None)
                continue
            name = _tool_call_function_name(tool_call)
            positional_tool_names.append(name)
            tool_call_id = tool_call.get("id")
            if name is not None and tool_call_id is not None:
                tool_names_by_id.setdefault(str(tool_call_id), name)
        return tool_names_by_id, positional_tool_names
    return tool_names_by_id, []


def _resolve_required_tool_name(
    tool_message: dict[str, Any],
    index: int,
    tool_messages: list[dict[str, Any]],
    previous_messages: list[dict[str, Any]],
) -> str:
    # [FUNCTION DOC]
    # HELPER: Determines a tool response's function name: the message's own name if set; else the name of the latest
    # assistant tool call with matching id; else positional matching, which requires the tool group size to equal the
    # latest assistant's tool call count and a name at this index, and logs a warning.
    # Exceptions: ValueError when positional matching is impossible.
    if tool_message.get("name"):
        return str(tool_message["name"])

    tool_names_by_id, positional_tool_names = _latest_assistant_tool_call_names(previous_messages)
    tool_call_id = tool_message.get("tool_call_id")
    if tool_call_id is not None and str(tool_call_id) in tool_names_by_id:
        return tool_names_by_id[str(tool_call_id)]

    if len(tool_messages) != len(positional_tool_names):
        raise ValueError(
            "Continuous Token cannot resolve tool name by position: "
            f"got {len(tool_messages)} tool response messages but the latest assistant has "
            f"{len(positional_tool_names)} tool calls"
        )
    if index >= len(positional_tool_names) or positional_tool_names[index] is None:
        raise ValueError(
            "Continuous Token cannot resolve tool name by position: "
            f"assistant tool call at index {index} has no function name"
        )

    # ToolAgentLoop uses asyncio.gather and appends responses in the original
    # tool-call order, so positional matching is safe for its full response
    # batches. Black-box agent loops may return responses in another order; they
    # must provide tool message name or tool_call_id instead of relying on this.
    logger.warning(
        "Continuous Token is resolving a tool response name by position; this is only safe when "
        "tool responses are appended in the same order as the latest assistant tool_calls"
    )
    return positional_tool_names[index]


def _tool_call_function_name(tool_call: dict[str, Any]) -> str | None:
    # [FUNCTION DOC]
    # HELPER: Returns the tool call's function name as string when present in a dict "function" entry, else None.
    function = tool_call.get("function")
    if isinstance(function, dict) and function.get("name") is not None:
        return str(function["name"])
    return None


class DeepSeekContinuousTokenBuilder(ContinuousTokenBuilder):
    # [CLASS DOC]
    # DeepSeek V2/V3/R1 builder. Resolves EOS (required) and optional BOS/User/Assistant ids, tokenizes the
    # tool-outputs-end marker, uses string-encoded synthetic tool arguments, and removes a trailing tool-outputs-end marker
    # before a user turn is appended (both inside incremental output and at the runtime junction).
    """DeepSeek V3/R1 boundary handling.

    DeepSeek uses direct concatenation at boundaries (no separator between
    ``<|end_of_sentence|>`` and the next role marker). The subclass validates
    key special tokens use correct Unicode (fullwidth vertical line U+FF5C
    and lower one-eighth block U+2581) to catch encoding regressions early.
    """

    # DeepSeek special tokens use fullwidth vertical line and lower one-eighth block
    _EOS_TOKEN = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    _BOS_TOKEN = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
    _USER_TOKEN = "<\uff5cUser\uff5c>"
    _ASSISTANT_TOKEN = "<\uff5cAssistant\uff5c>"
    _TOOL_OUTPUTS_END_TOKEN = "<\uff5ctool\u2581outputs\u2581end\uff5c>"

    def __init__(self, tokenizer: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the base, resolves required EOS id, optional BOS/User/Assistant ids (None when missing/unk) and tokenizes
        # the tool-outputs-end marker without special tokens. Exceptions: ValueError if EOS is missing or the marker tokenizes
        # to nothing.
        super().__init__(tokenizer, **kwargs)
        # EOS is the only token guaranteed across V2/V3/R1
        self._eos_id = require_token_id(tokenizer, self._EOS_TOKEN)
        # V3/R1-specific tokens — lookup but tolerate absence (V2-Lite has none)
        self._bos_id = self._optional_token_id(tokenizer, self._BOS_TOKEN)
        self._user_id = self._optional_token_id(tokenizer, self._USER_TOKEN)
        self._assistant_id = self._optional_token_id(tokenizer, self._ASSISTANT_TOKEN)
        self._tool_outputs_end_ids = normalize_token_ids(
            tokenizer.encode(self._TOOL_OUTPUTS_END_TOKEN, add_special_tokens=False)
        )
        if not self._tool_outputs_end_ids:
            raise ValueError(f"Expected DeepSeek {self._TOOL_OUTPUTS_END_TOKEN!r} to tokenize to at least one token")

    @staticmethod
    def _optional_token_id(tokenizer: Any, token: str) -> int | None:
        # [FUNCTION DOC]
        # HELPER: Returns the token id as int, or None when the tokenizer returns None or the unk id.
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk:
            return None
        return int(token_id)

    def _synthetic_assistant_for_tools(
        self,
        tool_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Builds the base synthetic assistant but replaces each tool call's arguments with the JSON string "{}" because these
        # templates concatenate arguments as strings.
        # The R1 / V3.1 / V3.2 templates splice ``tool['function']['arguments']`` into
        # the prompt by string concatenation, so a mapping raises TypeError there. The
        # synthetic tool call only exists to be diffed away; carry its arguments as the
        # JSON string those templates expect.
        synthetic_assistant = super()._synthetic_assistant_for_tools(tool_messages)
        for tool_call in synthetic_assistant["tool_calls"]:
            tool_call["function"]["arguments"] = "{}"
        return synthetic_assistant

    def _merge_context_group(
        self,
        incremental_ids: list[int],
        group_token_ids: list[int],
        *,
        group: list[dict[str, Any]],
        processed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Merge-group hook: before a user group, strips a trailing tool-outputs-end marker from the incremental ids, then
        # appends the group ids.
        if group[0].get("role") == "user":
            self._strip_trailing_tool_outputs_end(incremental_ids)
        super()._merge_context_group(
            incremental_ids,
            group_token_ids,
            group=group,
            processed_messages=processed_messages,
            tools=tools,
        )

    def _merge_context_token_ids(
        self,
        runtime_token_ids: list[int],
        appended_token_ids: list[int],
        **kwargs: Any,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Junction hook: if the first appended message is user, strips a trailing tool-outputs-end marker from the runtime
        # prefix and records the removed count; returns a "context" MergeResult with the concatenation.
        appended_messages = kwargs.get("appended_messages", [])
        prefix = list(runtime_token_ids)
        removed_prefix_token_count = 0
        if appended_messages and appended_messages[0].get("role") == "user":
            removed_prefix_token_count = self._strip_trailing_tool_outputs_end(prefix)

        return MergeResult(
            token_ids=prefix + list(appended_token_ids),
            appended_token_count=len(appended_token_ids),
            kind="context",
            removed_prefix_token_count=removed_prefix_token_count,
        )

    def _strip_trailing_tool_outputs_end(self, token_ids: list[int]) -> int:
        # [FUNCTION DOC]
        # HELPER: If token_ids ends with the tool-outputs-end marker ids, deletes them in place and returns their count;
        # otherwise returns 0. Side effects: mutates token_ids.
        suffix = self._tool_outputs_end_ids
        if token_ids[-len(suffix) :] != suffix:
            return 0
        del token_ids[-len(suffix) :]
        return len(suffix)


# =============================================================================
# Multimodal (VL) subclasses
# =============================================================================


class VLContinuousTokenMixin:
    # [CLASS DOC]
    # Mixin adding processor-backed multimodal rendering to a text builder (placed first in the MRO so text-family junction
    # handling still applies). Stores the processor and constant mm_processor_kwargs, extracts media references from
    # OpenAI-style content blocks and routes all renders through the processor's chat template and call.
    """Shared processor-backed logic for vision-language continuous token builders.

    Provides the multimodal workflow (image extraction, processor rendering,
    incremental dummy+trim encoding) common to all VL builders. Subclasses
    combine this mixin with a text-family builder (e.g. QwenContinuousTokenBuilder)
    via Python MRO so that boundary handling like Qwen's newline insertion or
    GLM's observation/user trim still applies through ``_merge_context_token_ids``.
    """

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # [FUNCTION DOC]
        # Initializes the next class in the MRO with tokenizer and kwargs, stores processor and mm_processor_kwargs (default
        # empty), and when no sampling_rate is set adds the processor feature extractor's sampling rate (as int) if available.
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        # Processor kwargs (e.g. max_pixels, do_pan_and_scan) that control how media
        # expands into tokens. Constant for the builder's whole lifetime, so renders
        # (initial prompt and incremental tool/user) stay aligned.
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        # Fold in the processor's audio sampling rate (a static processor property) so
        # mm_processor_kwargs is complete. Image-only processors have no
        # feature_extractor -> no-op.
        if "sampling_rate" not in self.mm_processor_kwargs:
            sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
            if sampling_rate is not None:
                self.mm_processor_kwargs = {**self.mm_processor_kwargs, "sampling_rate": int(sampling_rate)}

    @classmethod
    def supports_multimodal(cls) -> bool:
        # [FUNCTION DOC]
        # Returns True: VL builders support multimodal inputs.
        return True

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: Collects image references from list-valued message contents: for blocks of type image/image_url, uses the
        # "image" field, or falls back to image_url (dict url or string). Outputs: list of references in order.
        """Extract image references from OpenAI-style content blocks."""
        images: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                        image_ref = block.get("image")
                        if not image_ref:
                            image_url = block.get("image_url")
                            if isinstance(image_url, dict):
                                image_ref = image_url.get("url")
                            elif isinstance(image_url, str):
                                image_ref = image_url
                        if image_ref is not None:
                            images.append(image_ref)
        return images

    def _extract_videos_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: Collects non-None "video" fields from content blocks of type video. Outputs: list.
        """Extract video references from OpenAI-style content blocks."""
        videos: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "video":
                        video_ref = block.get("video")
                        if video_ref is not None:
                            videos.append(video_ref)
        return videos

    def _extract_audios_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: Collects "audio" (or fallback "audio_url") fields from content blocks of type audio. Outputs: list.
        """Extract audio references from OpenAI-style content blocks."""
        audios: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_ref = block.get("audio")
                        if audio_ref is None:
                            audio_ref = block.get("audio_url")
                        if audio_ref is not None:
                            audios.append(audio_ref)
        return audios

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Renders messages to token ids through the processor.
        # Behavior: copies chat template kwargs, adds tools when given, renders text (untokenized) with the processor chat
        # template via the module's apply_chat_template wrapper, then runs the processor on the single text with the given media
        # (empty lists become None) and the stored processor kwargs. If mm_inputs_out is given, updates it with all processor
        # outputs except input_ids. Returns normalized input_ids.
        # Side effects: mutates mm_inputs_out.
        """Render messages through the processor (full render with all media).

        ``mm_inputs_out``, when given, receives the processor features other than ``input_ids``
        (``pixel_values_videos``, ``video_grid_thw``, ...). SGLang needs them at request time
        because its ``video_data`` cannot take raw frames; without this hook they are dropped.
        """
        template_kwargs = dict(self.chat_template_kwargs)
        if tools:
            template_kwargs["tools"] = tools

        # Render the chat template through the processor (not the tokenizer) so the
        # placeholder text matches the legacy rollout path. VL models may ship a
        # processor chat template that differs from the tokenizer one, so it is
        # necessary to use the processor chat template for VL models.
        text = apply_chat_template(
            self.processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        # Processor kwargs are the builder-level constant captured at construction,
        # so initial-prompt and incremental (tool/user) renders stay aligned.
        proc_kwargs = dict(self.mm_processor_kwargs or {})
        processor_output = build_multimodal_processor_inputs(
            self.processor,
            text=[text],
            images=images if images else None,
            videos=videos if videos else None,
            audio=audios if audios else None,
            mm_processor_kwargs=proc_kwargs if proc_kwargs else None,
        )
        if mm_inputs_out is not None:
            mm_inputs_out.update({k: v for k, v in processor_output.items() if k != "input_ids"})
        return normalize_token_ids(processor_output["input_ids"])

    def _render_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Overrides the base text render to go through render_tokens_with_mm, extracting images, videos and audios from the
        # messages themselves.
        """Render messages to token IDs through the processor (VL override).

        Routes the base text renderer through the processor chat template +
        processor call so list-of-blocks content is handled and vision
        placeholders are expanded into per-image pad tokens. Media references are
        extracted from ``messages`` themselves.
        """
        return self.render_tokens_with_mm(
            messages,
            self._extract_images_from_messages(messages),
            videos=self._extract_videos_from_messages(messages),
            audios=self._extract_audios_from_messages(messages),
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Renders the initial prompt through the processor with the provided media, generation prompt and tools, optionally
        # filling mm_inputs_out.
        return self.render_tokens_with_mm(
            messages,
            images,
            videos=videos,
            audios=audios,
            add_generation_prompt=True,
            tools=tools,
            mm_inputs_out=mm_inputs_out,
        )


class VLContinuousTokenBuilder(VLContinuousTokenMixin, ContinuousTokenBuilder):
    # [CLASS DOC]
    # Default VL builder: mixin rendering plus the base plain-concatenation junction handling.
    """Generic vision-language builder used as the default for VL models that
    have no model-specific builder.

    Combines the shared processor-backed VL rendering (from the mixin) with the
    base, family-agnostic boundary handling (from ContinuousTokenBuilder).
    """


class QwenVLContinuousTokenBuilder(VLContinuousTokenMixin, QwenContinuousTokenBuilder):
    # [CLASS DOC]
    # Qwen VL builder (Qwen2-VL, 2.5-VL, 3-VL, 3-VL-MoE): mixin rendering plus Qwen newline insertion after im_end.
    """Qwen Vision-Language: Qwen ChatML newline patch + VL processor logic.

    Handles Qwen2-VL, Qwen2.5-VL, Qwen3-VL, and Qwen3-VL-MoE.
    """


class MiniMaxVLContinuousTokenBuilder(VLContinuousTokenMixin, MiniMaxContinuousTokenBuilder):
    # [CLASS DOC]
    # MiniMax-VL builder: mixin rendering plus MiniMax newline insertion, keyed on the end_of_sentence terminator, and
    # stripping of the assistant scaffold the processor template always appends when no generation prompt is requested.
    """MiniMax-VL (e.g. MiniMax-VL-01): MiniMax ``[e~[`` newline patch + VL processor logic.

    MiniMax-VL-01's *processor* chat template ignores ``add_generation_prompt`` and
    unconditionally appends an assistant scaffold ``<beginning_of_sentence>ai
    name=assistant\\n`` after every render. That breaks Continuous Token's
    append-only / suffix-diff invariant (``render(prefix)`` is no longer a token
    prefix of ``render(prefix + msg)``). We normalize the template here: strip the
    auto-appended scaffold when ``add_generation_prompt=False`` and keep it when
    ``True`` (where it legitimately is the generation prompt).
    """

    def __init__(self, tokenizer: Any, processor: Any, **kwargs: Any):
        # [FUNCTION DOC]
        # Initializes the mixin/MiniMax chain, repoints the EOS id to "<end_of_sentence>" and precomputes the scaffold ids.
        # Exceptions: ValueError when tokens are missing or scaffold detection fails.
        super().__init__(tokenizer, processor, **kwargs)
        # MiniMax-VL-01 uses ``<end_of_sentence>`` as its turn terminator, not the
        # MiniMax-Text-01 ``[e~[`` token the base builder resolves. Repoint the EOS
        # so the boundary-newline reinsertion in ``_merge_context_token_ids``
        # fires for VL turns.
        self._eos_id = require_token_id(tokenizer, "<end_of_sentence>")
        self._vl_scaffold_ids = self._compute_generation_scaffold_ids()

    def _compute_generation_scaffold_ids(self) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Renders synthetic system+user through the processor template (no generation prompt) and processor call
        # without images, and returns the ids from the last beginning_of_sentence token to the end.
        # Exceptions: ValueError when no such token exists or the result is empty.
        """Extract the unconditional trailing assistant scaffold token ids.

        Rendered via the processor chat template (bypassing this class's override),
        the scaffold is the final ``<beginning_of_sentence>...`` block, i.e. every
        token from the last ``<beginning_of_sentence>`` to the end.
        """
        text = apply_chat_template(
            self.processor,
            [_SYNTHETIC_SYSTEM_MESSAGE, _SYNTHETIC_USER_MESSAGE],
            tokenize=False,
            add_generation_prompt=False,
            **self.chat_template_kwargs,
        )
        ids = normalize_token_ids(
            build_multimodal_processor_inputs(self.processor, text=[text], images=None)["input_ids"]
        )
        bos_id = require_token_id(self.tokenizer, "<beginning_of_sentence>")
        bos_positions = [i for i, t in enumerate(ids) if t == bos_id]
        if not bos_positions:
            raise ValueError("MiniMax-VL scaffold detection failed: no <beginning_of_sentence> token")
        scaffold = ids[bos_positions[-1] :]
        if not scaffold:
            raise ValueError("MiniMax-VL scaffold detection produced an empty scaffold")
        return scaffold

    def render_tokens_with_mm(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        *,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Calls the mixin render, then removes the trailing scaffold ids when present and no generation prompt was requested.
        # The processor template always appends the scaffold; strip it unless a
        # generation prompt was requested, restoring append-only rendering.
        token_ids = super().render_tokens_with_mm(
            messages,
            images,
            videos=videos,
            audios=audios,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            mm_inputs_out=mm_inputs_out,
        )
        scaffold = self._vl_scaffold_ids
        if token_ids[-len(scaffold) :] == scaffold and not add_generation_prompt:
            token_ids = token_ids[: -len(scaffold)]
        return token_ids


class Gemma4VLContinuousTokenBuilder(VLContinuousTokenMixin, Gemma4ContinuousTokenBuilder):
    # [CLASS DOC]
    # Gemma4 VL builder: mixin processor rendering combined with Gemma4 turn/tool-response boundary handling.
    """Gemma4 (unified) vision-language: Gemma4 ``<|tool_response>`` boundary handling
    + VL processor rendering.

    Gemma4 is a unified text+vision architecture, so the same checkpoint serves
    both modalities. The mixin routes user/system/assistant rendering through the
    multimodal processor chat template, while tool-response boundary handling is
    inherited from :class:`Gemma4ContinuousTokenBuilder`.
    """


class GLM46VContinuousTokenBuilder(VLContinuousTokenMixin, GLMContinuousTokenBuilder):
    # [CLASS DOC]
    # GLM-4.6V builder: mixin processor rendering combined with GLM observation/user trimming.
    """GLM-4.6V: GLM observation/user trim + VL processor logic."""


class KimiVLContinuousTokenBuilder(VLContinuousTokenMixin, ContinuousTokenBuilder):
    # [CLASS DOC]
    # Kimi-VL builder: mixin processor rendering with plain concatenation at junctions.
    """Kimi-VL (MoonViT): direct concatenation + VL processor logic."""


class DeepSeekVL2ContinuousTokenBuilder(DeepSeekContinuousTokenBuilder):
    # [CLASS DOC]
    # DeepSeek-VL2 builder. Not using the mixin: renders via the DeepseekVLV2Processor conversation API (no chat template)
    # and computes context merges by full re-render plus prefix diff; inherits DeepSeek junction handling.
    """DeepSeek-VL2 continuous token builder.

    VL2 uses its own DeepseekVLV2Processor that handles both conversation
    formatting and image token expansion in a single __call__. It does NOT
    support standard apply_chat_template, so all rendering goes through the
    processor directly.

    The processor produces stable prefixes: full_render[:len(prev)] == prev,
    so we use full render + prefix diff (like the original CT approach).
    """

    def __init__(
        self,
        tokenizer: Any,
        processor: Any,
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # [FUNCTION DOC]
        # Initializes the DeepSeek text builder and stores the processor and mm_processor_kwargs (unused, kept for symmetry).
        super().__init__(tokenizer, **kwargs)
        self.processor = processor
        # VL2 renders through DeepseekVLV2Processor directly and does not consume
        # mm_processor_kwargs, but it is stored for API symmetry with other VL builders.
        self.mm_processor_kwargs = mm_processor_kwargs or {}

    @classmethod
    def supports_multimodal(cls) -> bool:
        # [FUNCTION DOC]
        # Returns True.
        return True

    def _extract_images_from_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        # [FUNCTION DOC]
        # HELPER: Same image-reference extraction as the VL mixin (image field or image_url dict/string).
        """Extract image references from content blocks."""
        images: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                        image_ref = block.get("image")
                        if not image_ref:
                            image_url = block.get("image_url")
                            if isinstance(image_url, dict):
                                image_ref = image_url.get("url")
                            elif isinstance(image_url, str):
                                image_ref = image_url
                        if image_ref is not None:
                            images.append(image_ref)
        return images

    def _to_vl2_conversation(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        add_generation_prompt: bool = True,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        # [FUNCTION DOC]
        # HELPER: Converts OpenAI-style messages to VL2 conversation dicts: list contents are flattened to text with an
        # "<image>" placeholder per image block (consuming images in order while available) and the message's images collected;
        # user and system map to the User role (system with no images), assistant to the Assistant role; when a generation prompt
        # is requested, appends an empty Assistant entry unless the last entry already is an empty Assistant.
        # Outputs: (conversation list, the original images list).
        # KNOWN ISSUE: messages with other roles (e.g. tool) are silently dropped.
        """Convert OpenAI-style messages to VL2 conversation format."""
        conv: list[dict[str, Any]] = []
        img_idx = 0
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                msg_images: list[Any] = []
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype in ("image", "image_url") and img_idx < len(images):
                            parts.append("<image>")
                            msg_images.append(images[img_idx])
                            img_idx += 1
                        elif btype == "text":
                            parts.append(block.get("text", ""))
                content = "".join(parts)
            else:
                msg_images = []

            if role == "user":
                conv.append({"role": "<|User|>", "content": content, "images": msg_images})
            elif role == "assistant":
                conv.append({"role": "<|Assistant|>", "content": content})
            elif role == "system":
                conv.append({"role": "<|User|>", "content": content, "images": []})

        if add_generation_prompt:
            if not conv or conv[-1].get("role") != "<|Assistant|>" or conv[-1].get("content"):
                conv.append({"role": "<|Assistant|>", "content": ""})
        return conv, images

    def _render_via_processor(
        self,
        messages: list[dict[str, Any]],
        images: list[Any],
        add_generation_prompt: bool = True,
    ) -> list[int]:
        # [FUNCTION DOC]
        # HELPER: Converts messages to a VL2 conversation and calls the processor with force_batchify, returning the first
        # row of input_ids as a list of ints.
        """Render messages through DeepseekVLV2Processor."""
        conv, all_images = self._to_vl2_conversation(messages, images, add_generation_prompt)
        out = self.processor.__call__(conversations=conv, images=all_images, force_batchify=True)
        return normalize_token_ids(out.input_ids[0].tolist())

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
        mm_inputs_out: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC]
        # Initial prompt: uses the given images or extracts them from messages; with no images falls back to the text template
        # render, otherwise renders through the processor. Video/audio inputs are ignored.
        # DeepSeek-VL2 renders through its own processor and has no video path here.
        del mm_inputs_out
        if images is None:
            images = self._extract_images_from_messages(messages)
        if not images:
            return self._render_tokens(messages, add_generation_prompt=True, tools=tools)
        return self._render_via_processor(messages, images, add_generation_prompt=True)

    def merge_context_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> MergeResult:
        # [FUNCTION DOC]
        # Validates append-only, re-renders the full updated conversation through the processor (with all extracted images
        # and a generation prompt) and takes everything past the runtime stream length as appended ids, then applies the
        # DeepSeek junction hook.
        # KNOWN ISSUE: the prefix is assumed to equal the runtime stream by length only (no equality check), tools are ignored,
        # and appended_messages is not passed to the junction hook so the tool-outputs-end stripping never triggers.
        """Merge tokens: always use processor + prefix diff for VL2.

        VL2 tokenizer has no chat_template, so all rendering goes through
        the processor. Prefix stability is guaranteed by the processor.
        """
        self._assert_append_only(previous_messages, updated_messages)

        # Always use full render + prefix diff (VL2 has no apply_chat_template)
        all_images = self._extract_images_from_messages(updated_messages)
        full_token_ids = self._render_via_processor(updated_messages, all_images, add_generation_prompt=True)

        prefix_len = len(runtime_token_ids)
        appended_token_ids = full_token_ids[prefix_len:]
        return self._merge_context_token_ids(runtime_token_ids, appended_token_ids)
