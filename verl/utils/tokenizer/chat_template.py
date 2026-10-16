# Copyright 2025 Bytedance Ltd. and/or its affiliates
# [MODULE DOC]
# Chat-template probing helpers. Derive, by rendering small synthetic conversations through a tokenizer's or
# processor's chat template, the token ids of the template's system-prompt prefix, its generation prompt and
# its inter-turn separator; also provides a wrapper around apply_chat_template that tolerates templates
# (e.g. Qwen3.5) that refuse conversations without a user message. Configures a module logger whose
# level comes from the VERL_LOGGING_LEVEL env var (default WARN).
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from .tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    # [FUNCTION DOC]
    # Computes the default system-prompt prefix tokens of a chat template.
    # Behavior: tokenizes a conversation of one empty user message and one of two empty user messages (no
    # generation prompt); the length difference is the per-turn size, and the system prompt is the one-turn rendering
    # with that many trailing tokens removed.
    # Inputs: tokenizer - tokenizer/processor with apply_chat_template; extra kwargs forwarded to apply_chat_template.
    # Outputs: list of int token ids (empty if the template has no implicit system prompt).
    # Side effects: none.
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def initialize_turn_separator(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    # [FUNCTION DOC]
    # HELPER: Computes the turn separator tokens a template emits after a message's closing token.
    # Behavior: renders a single user turn with empty and with "x" content, first with string content and, on any
    # exception, with list-of-text-parts content; returns [] if neither form renders. Finds the maximal common trailing
    # token run of the two renderings. If empty, returns []. Otherwise determines eos id(s) from the tokenizer (or its
    # wrapped tokenizer attribute), accepting an int or a list/tuple; if any eos id is present in the common suffix,
    # returns everything after the last occurrence; otherwise drops the first suffix token and returns the rest.
    # Inputs: tokenizer - tokenizer or processor; extra kwargs forwarded to apply_chat_template.
    # Outputs: list of int token ids (possibly empty).
    # Exceptions: swallows rendering exceptions. Side effects: none.
    """Tokens a chat template inserts after a message's closing token, before the next turn.

    Multi-turn agent rollouts build the token sequence incrementally. The model stops at the
    assistant close token (e.g. ``<|im_end|>``) and never emits the template's trailing
    turn-separator (e.g. ``"\\n"``, id 198 for Qwen). Rendering the following tool/user turn in
    isolation also omits that separator, so every turn boundary silently drops it and the rollout
    token sequence diverges from ``apply_chat_template`` of the equivalent full conversation.
    This returns the separator so callers can restore it at turn boundaries.

    Derivation: rendering the same (user) turn with empty vs non-empty content only differs in the
    content region, so the maximal common trailing run is exactly ``[close_token, *separator]``.
    A user turn is used deliberately -- probing with an assistant turn would inject reasoning
    scaffolding (e.g. Qwen3's ``<think></think>``) that is not part of the separator. The model
    emits the close token itself (it is the stop token), so the separator is everything after it.

    Returns an empty list when the template has no turn separator or an unexpected structure, so
    callers keep their previous behavior instead of crashing.
    """
    # Render two user turns that differ only in body text; the shared trailing run is the separator.
    # A bare string ``content`` is rejected by some multimodal processors (they iterate ``content``
    # expecting a list of typed parts), so fall back to the list-of-parts form, and return ``[]`` if
    # neither renders. Both probes must use the same form so only the body differs.
    empty = filled = None
    for as_parts in (False, True):
        if as_parts:
            body_empty, body_filled = [{"type": "text", "text": ""}], [{"type": "text", "text": "x"}]
        else:
            body_empty, body_filled = "", "x"
        try:
            empty = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_empty}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            filled = normalize_token_ids(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": body_filled}],
                    add_generation_prompt=False,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
            )
            break
        except Exception:
            empty = filled = None
    if empty is None or filled is None:
        return []
    # Maximal common trailing run == the message closing token(s) + inter-turn separator (identical
    # regardless of content).
    i = 0
    while i < len(empty) and i < len(filled) and empty[-1 - i] == filled[-1 - i]:
        i += 1
    suffix = empty[len(empty) - i :]
    if not suffix:
        return []
    # Split off the closing token the model already emits; the remainder is the dropped separator.
    # A processor (VLM path) exposes ``eos_token_id`` on its wrapped tokenizer rather than itself,
    # and some tokenizers (e.g. Llama 3) expose it as a list/tuple of ids rather than a single int.
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
    last_close = max((i for i, tok_id in enumerate(suffix) if tok_id in eos_ids), default=None)
    if last_close is not None:
        return suffix[last_close + 1 :]
    return suffix[1:]


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    # [FUNCTION DOC]
    # Computes both the system-prompt prefix tokens (same method as initialize_system_prompt) and the generation-prompt
    # tokens, the latter being the tokens appended when rendering one empty user message with add_generation_prompt=True
    # beyond the rendering without it.
    # Inputs: tokenizer; extra kwargs forwarded to apply_chat_template.
    # Outputs: tuple (system_prompt token list, generate_prompt token list). Exceptions: template errors propagate.
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        )
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
        )
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    # [FUNCTION DOC]
    # Calls processor.apply_chat_template with the given arguments; if that raises any exception, retries by prepending a
    # dummy user message (list-of-parts with empty text) and stripping the dummy prefix from the result.
    # Prefix stripping: render the dummy message alone (no generation prompt) as prefix; then
    # - tokenize false: slice the text output after the prefix string length;
    # - tokenize true, return_dict false: if output is a nested list (transformers 5 batch form) assert a single row and
    # unwrap both prefix and output, then slice after the prefix length;
    # - return_dict true: convert to plain dicts, take the prefix length from input_ids dimension 1 (assumes tensors with
    # return_tensors="pt") and slice input_ids, attention_mask and, if present, mm_token_type_ids along dimension 1.
    # Inputs: processor - tokenizer or processor; messages - list of message dicts; tokenize, add_generation_prompt, tools,
    # return_dict; extra kwargs forwarded.
    # Outputs: token id list, text string, or dict of tensors.
    # Exceptions: errors from the retry path propagate; AssertionError if nested output has more than one row.
    # KNOWN ISSUE: return_dict path assumes tensor outputs; other keys (e.g. pixel values) are left unchanged, and the first exception is silently swallowed regardless of its cause.
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        # Qwen3.5 apply_chat_template needs messages with at least one user message
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:  # tokenize=False
            return output[len(dummy_user_prefix) :]
        elif not return_dict:  # tokenize=True and return_dict=False
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            return output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
            return output
