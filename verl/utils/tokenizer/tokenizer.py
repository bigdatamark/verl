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
# Tokenizer and multimodal processor helpers: loading HuggingFace tokenizers/processors with verl-specific fixes
# (pad token defaults, gemma-2 eos fix, binding VLM rope-index methods onto processors), normalizing token-id outputs
# across transformers 4/5, resolving multimodal special token ids and building processor call kwargs.
"""Utils for tokenization."""

import types
import warnings

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "normalize_token_ids",
    "build_multimodal_processor_inputs",
    "get_processor_token_id",
]


def normalize_token_ids(tokenized_output) -> list[int]:
    # [FUNCTION DOC]
    # Converts various tokenizer outputs into a flat list of Python ints.
    # Behavior: if the input is a dict with input_ids, or any object with an input_ids attribute, take that field; convert
    # tensors/arrays via their list conversion; convert tuples to lists; unwrap a single-row nested list/tuple; then convert
    # every element to int (calling item() first on scalar-tensor-like elements).
    # Inputs: tokenized_output - list, tuple, tensor, dict or BatchEncoding. Outputs: list of int.
    # Exceptions: TypeError if the result is not list-like or an element is not int-convertible.
    # Side effects: none.
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def get_processor_token_id(processor, token_name: str) -> int | None:
    # [FUNCTION DOC]
    # Resolves a special token id named by token_name (e.g. "image", "video") from a processor.
    # Behavior: return None if processor is None; if the processor has an attribute token_name+"_token_id" that is not None,
    # return it as int; else if it has token_name+"_token" and a tokenizer attribute, convert the token string to an id and
    # return it as int if not None; otherwise None.
    # Outputs: int or None. Side effects: none.
    """Resolve a multimodal special token id from a processor.

    Newer processors may expose ``image_token``/``video_token`` strings instead
    of ``image_token_id``/``video_token_id`` integers. Fall back to tokenizer
    conversion so rollout code can stay processor-agnostic.
    """

    if processor is None:
        return None

    token_id_attr = f"{token_name}_token_id"
    token_id = getattr(processor, token_id_attr, None)
    if token_id is not None:
        return int(token_id)

    token_attr = f"{token_name}_token"
    token = getattr(processor, token_attr, None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token is not None and tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None:
            return int(converted)

    return None


def _split_videos_and_metadata(videos):
    # [FUNCTION DOC]
    # HELPER: Splits a video list that may contain (frames, metadata) tuples.
    # Inputs: videos - None or iterable. Outputs: (None, None) for None; if the first item is a tuple, returns
    # (list of values, list of metadata) by unzipping all items; otherwise (list of videos, None).
    if videos is None:
        return None, None
    videos = list(videos)
    if len(videos) > 0 and isinstance(videos[0], tuple):
        video_values, video_metadata = zip(*videos, strict=False)
        return list(video_values), list(video_metadata)
    return videos, None


def build_multimodal_processor_inputs(
    processor,
    *,
    text,
    images=None,
    videos=None,
    audio=None,
    mm_processor_kwargs=None,
    return_tensors: str = "pt",
):
    # [FUNCTION DOC]
    # Builds the keyword arguments and invokes the processor for a multimodal sample.
    # Behavior: copies mm_processor_kwargs; if audio is given and no sampling_rate was set, takes the processor's
    # feature_extractor sampling_rate (when available) as int; splits videos into values and metadata; defaults
    # return_tensors; if metadata exists, defaults video_metadata to it and do_sample_frames to False; assembles text,
    # images, videos plus kwargs, adds audio only when not None, then calls the processor.
    # Inputs: processor - callable HF processor; text, images, videos, audio, mm_processor_kwargs, return_tensors (default "pt").
    # Outputs: whatever the processor returns (typically a BatchFeature). Exceptions: processor errors propagate.
    # Side effects: none beyond the processor call.
    """Build kwargs for multimodal processor calls.

    This keeps the existing VL flow intact while extending it with audio-aware
    paths for processors that accept audio inputs.
    """
    processor_kwargs = dict(mm_processor_kwargs or {})
    if audio is not None and "sampling_rate" not in processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = int(sampling_rate)

    videos, video_metadata = _split_videos_and_metadata(videos)
    processor_kwargs.setdefault("return_tensors", return_tensors)

    if video_metadata is not None:
        processor_kwargs.setdefault("video_metadata", video_metadata)
        processor_kwargs.setdefault("do_sample_frames", False)

    processor_inputs = {"text": text, "images": images, "videos": videos, **processor_kwargs}
    if audio is not None:
        processor_inputs["audio"] = audio

    return processor(**processor_inputs)


def set_pad_token_id(tokenizer):
    # [FUNCTION DOC]
    # Ensures a tokenizer has a pad token: if pad_token_id is None sets it to eos_token_id, and if pad_token is None sets
    # it to eos_token, emitting a warning in each case.
    # Inputs: tokenizer. Outputs: none. Side effects: mutates tokenizer, issues warnings.
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    # [FUNCTION DOC]
    # Loads a HuggingFace tokenizer via AutoTokenizer.
    # Behavior: if correct_gemma2 and the path string contains "gemma-2-2b-it", warns and forces eos_token to
    # "<end_of_turn>" and eos_token_id to 107 via kwargs; loads the tokenizer with kwargs; if correct_pad_token, applies
    # set_pad_token_id.
    # Inputs: name_or_path; correct_pad_token (default True); correct_gemma2 (default True); extra kwargs for from_pretrained.
    # Outputs: tokenizer. Exceptions: loading errors propagate. Side effects: warnings, may download files.
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    # [FUNCTION DOC]
    # Loads a HuggingFace multimodal processor, returning None for text-only models.
    # Behavior: loads AutoProcessor; returns None immediately if it is a plain tokenizer. Loads AutoConfig and attaches it
    # as processor.config. Chooses a model class by processor class name (Qwen2VL, Qwen2.5VL, Qwen3VL, Glm4v, Glm46V); for
    # Mllama does nothing; for Gemma4 replaces validate_inputs with a no-op; any other name raises ValueError. If a model
    # class was chosen, binds its get_rope_index (and get_vision_position_ids if present) as methods of the processor.
    # Any exception (including the ValueError above) is caught, a warning issued and processor set to None. Finally, a
    # processor whose class name lacks "Processor" is discarded (None).
    # Inputs: name_or_path; kwargs forwarded to from_pretrained. Outputs: processor or None.
    # Side effects: mutates the processor object, warnings, may download files.
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            return None

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor.
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "Glm46VProcessor":
                from transformers.models.glm46v import Glm46VModel

                model_class = Glm46VModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case "Gemma4Processor":
                # Gemma4 uses standard 1D RoPE -> no get_rope_index to bind. Disable its strict
                # per-image-token check (which Qwen's processor lacks).
                processor.validate_inputs = lambda *args, **kwargs: None
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
