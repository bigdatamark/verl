# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
# Fused-kernel forward for Megatron GPT models: instead of materializing vocabulary logits, the last stage computes
# per-token log-probs and entropy directly from hidden states with a fused linear cross-entropy kernel. Two mechanisms:
# on Megatron Core >= 0.18 the native output_processor hook is used ("hook" mode); otherwise GPTModel.forward is
# monkey-patched ("legacy" mode). The chosen mode is cached on the model under a private attribute.

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import megatron.core as mcore
import torch
from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.utils import deprecate_inference_params
from packaging import version
from torch import Tensor

from verl.models.mcore.util import build_vlm_attn_mask_thd, preprocess_packed_seqs, preprocess_thd_engine
from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy
from verl.utils.megatron_utils import unwrap_model
from verl.utils.model import CausalLMOutputForPPO

from .util import postprocess_packed_seqs_for_dict_output, postprocess_thd_engine

_FUSED_FORWARD_MODE_ATTR = "_verl_fused_forward_mode"
_HOOK_MODE = "hook"
_LEGACY_MODE = "legacy"


def _supports_output_processor_hook(patching_model: torch.nn.Module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns true if the model's forward signature accepts both output_processor and output_processor_context.
    """Check whether the model supports Megatron's native output-processor hook.

    The ``output_processor`` / ``output_processor_context`` contract was
    introduced in Megatron Core 0.18.0. Models without this contract fall back
    to the legacy ``forward`` monkey patch.

    TODO: Remove this check and the legacy patch once all supported Megatron
    stacks require Megatron Core 0.18.0 or newer.
    """
    parameters = inspect.signature(patching_model.forward).parameters
    return {"output_processor", "output_processor_context"}.issubset(parameters)


def _resolve_fused_forward_mode(patching_model: torch.nn.Module) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "hook" if the model supports the output-processor hook, else "legacy".
    return _HOOK_MODE if _supports_output_processor_hook(patching_model) else _LEGACY_MODE


def _get_fused_forward_mode(model: torch.nn.Module) -> str:
    # [FUNCTION DOC]
    # HELPER: Reads the cached fused-forward mode from the unwrapped model, or from its language_model when absent;
    # defaults to "legacy" if missing or invalid.
    model = unwrap_model(model)
    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None and hasattr(model, "language_model"):
        mode = getattr(model.language_model, _FUSED_FORWARD_MODE_ATTR, None)
    return mode if mode in (_HOOK_MODE, _LEGACY_MODE) else _LEGACY_MODE


def _use_output_processor_hook(model: torch.nn.Module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns true if the cached mode for the model is "hook".
    return _get_fused_forward_mode(model) == _HOOK_MODE


@dataclass
class FusedOutputProcessorContext:
    # [CLASS DOC]
    # Dataclass carried through Megatron's output-processor hook; holds the sampling temperature for log-prob computation.
    """Context passed through Megatron's native output-processor hook."""

    temperature: float


def fused_output_processor(
    *,
    hidden_states,
    output_layer,
    output_weight,
    labels,
    context,
    config,
    **_ignored,
):
    # [FUNCTION DOC]
    # Behavior: output-processor hook implementation. Creates a CausalLMOutputForPPO holding the (un-gathered) hidden
    # states, gathers sequence-parallel hidden states if enabled, uses output_weight when given (tied embeddings) else the
    # output layer weight, computes log-probs and entropy with the fused kernel at context.temperature over the TP group,
    # and optionally logs inputs and results to disk when Megatron's config logger is enabled.
    # Inputs (keyword only): hidden_states, output_layer, output_weight, labels (packed), context
    # (FusedOutputProcessorContext), config; other keywords ignored (input_ids, position_ids, etc. used only for logging).
    # Outputs: CausalLMOutputForPPO with log_probs and entropy set.
    # Side effects: optional disk logging.
    """Compute fused log probabilities and entropy at Megatron's postprocess boundary."""
    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Megatron passes the shared embedding as output_weight for tied models. For
    # untied models the weight lives on output_layer.
    weight = output_weight if output_weight is not None else output_layer.weight

    temperature = context.temperature
    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(config):
        payload = OrderedDict(
            {
                "input_ids": _ignored.get("input_ids"),
                "position_ids": _ignored.get("position_ids"),
                "attention_mask": _ignored.get("attention_mask"),
                "decoder_input": _ignored.get("decoder_input"),
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs
    return output


def _get_patching_model(model: torch.nn.Module):
    # [FUNCTION DOC]
    # HELPER: Returns the GPTModel to patch: the unwrapped model itself, or its language_model if that is a GPTModel;
    # otherwise prints an unsupported message and returns None.
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_fused_forward(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: resolves the GPTModel, determines and caches the fused-forward mode on it; in hook mode does nothing more.
    # In legacy mode asserts Megatron Core >= 0.13.0 and, if not already patched, stores the original forward as
    # forward_backup and binds the fused forward in its place.
    # Exceptions: AssertionError for older Megatron versions.
    # Side effects: mutates the model instance.
    model = _get_patching_model(model)
    if model is None:
        return

    mode = getattr(model, _FUSED_FORWARD_MODE_ATTR, None)
    if mode is None:
        mode = _resolve_fused_forward_mode(model)
        setattr(model, _FUSED_FORWARD_MODE_ATTR, mode)

    if mode == _HOOK_MODE:
        return

    assert version.parse(mcore.__version__) >= version.parse("0.13.0"), (
        "Fused forward patching requires mecore >= 0.13.0"
    )
    if not hasattr(model, "forward_backup"):
        model.forward_backup = model.forward
        model.forward = _fused_GPTModel_forward.__get__(model, model.__class__)


def unpatch_fused_forward(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: restores the original forward from forward_backup and removes the backup, unless the model is unsupported
    # or in hook mode.
    # Side effects: mutates the model instance.
    model = _get_patching_model(model)
    if model is None or _get_fused_forward_mode(model) == _HOOK_MODE:
        return
    if hasattr(model, "forward_backup"):
        model.forward = model.forward_backup
        delattr(model, "forward_backup")


def fused_forward_model_gen(vision_model: bool = False):
    # [FUNCTION DOC]
    # Behavior: factory returning a fused forward closure for padded inputs (legacy actor path), specialized for VLMs.
    # Outputs: fused_forward_model function.
    def fused_forward_model(
        model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        labels_mask: Tensor,
        temperature: float,
        multi_modal_inputs: dict,
    ):
        # [FUNCTION DOC]
        # Behavior: moves multimodal tensors to device; packs input_ids (pre-processing except for VLMs), labels and
        # labels_mask into THD; calls the model with packed ids, packed_seq_params, packed labels and position ids (VLMs get
        # the padded input_ids and attention mask and compute positions themselves). In hook mode passes the fused output
        # processor and temperature context; in legacy mode passes temperature to the patched forward. On the last stage
        # unpacks log_probs/entropy to padded layout masked by labels_mask.
        # Inputs: model; input_ids, position_ids, attention_mask, labels, labels_mask - padded [batch, seq]; temperature;
        # multi_modal_inputs - dict.
        # Outputs: dict-like with log_probs and entropy (last stage) or hidden states (other stages).
        pre_process: bool = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process: bool = unwrap_model(model).post_process

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=pre_process)
        input_ids_rmpad = input_ids_rmpad.contiguous()
        labels_rmpad, _ = preprocess_packed_seqs(labels, attention_mask, pre_process=True)
        labels_mask_rmpad, _ = preprocess_packed_seqs(labels_mask, attention_mask, pre_process=True)
        labels_rmpad = labels_rmpad.contiguous()
        labels_mask_rmpad = labels_mask_rmpad.contiguous()

        input_args = dict(
            input_ids=input_ids_rmpad,
            attention_mask=None,
            position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            temperature=temperature,
            **model_kwargs,
        )

        if vision_model:
            # workaround for supporting sequence packing with context parallelism
            # cp split with sequence packing will make model lose vision token information, so we need to keep
            # the original input_ids and pack them after vision embedding is calculated,
            # cooporate with mbridge
            input_args["input_ids"] = input_ids
            input_args["attention_mask"] = attention_mask

        if _use_output_processor_hook(model):
            input_args.pop("temperature", None)
            output_orig: CausalLMOutputForPPO = model(
                **input_args,
                output_processor=fused_output_processor,
                output_processor_context=FusedOutputProcessorContext(temperature=temperature),
            )
        else:
            output_orig: CausalLMOutputForPPO = model(**input_args)

        if post_process:
            # output_orig is in type of CausalLMOutputForPPO
            output = postprocess_packed_seqs_for_dict_output(
                labels_mask_rmpad,
                output_orig,
                packed_seq_params,
                attention_mask,
                batch_size,
                seq_len,
                post_process=post_process,
            )
        else:
            output = output_orig
        return output

    return fused_forward_model


def fused_forward_model_engine(vision_model: bool = False):
    # [FUNCTION DOC]
    # Behavior: factory returning a fused forward closure for the model-engine path (nested inputs).
    # Outputs: fused_forward_model_engine_inner function.
    def fused_forward_model_engine_inner(
        model,
        input_ids: Tensor,
        labels: Tensor,
        multi_modal_inputs: dict,
        temperature: float,
        calculate_entropy: bool,
        pad_token_id: int,
        cp_layout: str = "zigzag",
        local_cp_size: int | None = None,
        router_padding_mask: Tensor | None = None,
        pad_to_length_bucket: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: packs nested input_ids with preprocess_thd_engine (fp8 padding, minimum local rows equal to
        # csa_window_size for the "dsv4_hybrid" attention variant, length bucketing, CP layout and dynamic CP size), collects
        # router padding mask and multimodal inputs, builds VLM ids/mask when needed, packs labels with a left roll, and calls
        # the model with no position ids (hook mode via output processor, legacy mode via temperature kwarg). On non-last
        # stages returns the raw output. Otherwise unpacks log_probs (adding a leading dim if 1-D) to the nested input layout
        # and, when calculate_entropy, the entropy likewise.
        # Inputs: model; input_ids, labels - nested tensors; multi_modal_inputs; temperature; calculate_entropy; pad_token_id;
        # cp_layout; local_cp_size; router_padding_mask; pad_to_length_bucket.
        # Outputs: dict with log_probs and optionally entropy, or the raw output on non-last stages.
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process

        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        config = unwrap_model(model).config
        min_local_rows = (
            config.csa_window_size if getattr(config, "experimental_attention_variant", None) == "dsv4_hybrid" else None
        )

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids,
            pre_process=pre_process,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        model_kwargs = {}
        if router_padding_mask is not None:
            model_kwargs["padding_mask"] = router_padding_mask
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        attention_mask = None
        if vision_model:
            input_ids_rmpad, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id)

        labels_rmpad, _, _ = preprocess_thd_engine(
            labels,
            pre_process=True,
            need_roll=True,
            use_fp8_padding=use_fp8_padding,
            min_local_rows=min_local_rows,
            pad_to_length_bucket=pad_to_length_bucket,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )
        labels_rmpad = labels_rmpad.contiguous()
        forward_kwargs = dict(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=None,
            packed_seq_params=packed_seq_params,
            labels=labels_rmpad,
            **model_kwargs,
        )
        if _use_output_processor_hook(model):
            output_orig: CausalLMOutputForPPO = model(
                **forward_kwargs,
                output_processor=fused_output_processor,
                output_processor_context=FusedOutputProcessorContext(temperature=temperature),
            )
        else:
            output_orig: CausalLMOutputForPPO = model(temperature=temperature, **forward_kwargs)

        if not post_process:
            return output_orig

        log_probs = output_orig.log_probs
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(0)
        log_probs = postprocess_thd_engine(
            log_probs,
            packed_seq_params,
            input_ids,
            input_ids.shape[0],
            post_process=post_process,
            cp_layout=cp_layout,
            local_cp_size=local_cp_size,
        )

        output = {"log_probs": log_probs}

        if calculate_entropy:
            entropy = output_orig.entropy
            if entropy.dim() == 1:
                entropy = entropy.unsqueeze(0)
            entropy = postprocess_thd_engine(
                entropy,
                packed_seq_params,
                input_ids,
                input_ids.shape[0],
                post_process=post_process,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
            )
            output["entropy"] = entropy

        return output

    return fused_forward_model_engine_inner


def _fused_GPTModel_forward(
    model,
    input_ids: Tensor,
    position_ids: Tensor,
    attention_mask: Tensor,
    decoder_input: Tensor = None,
    labels: Tensor = None,
    inference_context: BaseInferenceContext = None,
    packed_seq_params: PackedSeqParams = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    loss_mask: Optional[Tensor] = None,
    temperature: float = 1.0,
    padding_mask: Tensor | None = None,
    **kwargs,
) -> CausalLMOutputForPPO:
    # [FUNCTION DOC]
    # HELPER: Legacy replacement for GPTModel.forward (bound to the model instance).
    # Behavior: runs the model's _preprocess (forwarding padding_mask only when provided), passes the sequence-parallel
    # scattered padding mask and, for models with hash MoE layers, input_ids to the decoder as extra block kwargs, runs
    # the decoder, and returns hidden states on non-last stages. On the last stage gathers sequence-parallel hidden states,
    # selects the output layer weight (or the word embedding weight when no output weight exists), computes fused
    # log-probs/entropy at the given temperature, optionally logs to disk, and returns a CausalLMOutputForPPO.
    # Inputs: standard GPTModel forward args plus temperature and padding_mask; loss_mask is accepted but unused; extra
    # kwargs go to the decoder.
    # Outputs: hidden states tensor or CausalLMOutputForPPO with log_probs and entropy.
    # KNOWN ISSUE: when extra_block_kwargs is supplied, padding_mask/input_ids are written into the caller's dict in place.
    """
    Patch self._postprocess in forward for GPT models to enable fused kernel support.
    https://github.com/NVIDIA/Megatron-LM/blob/core_v0.13.0/megatron/core/models/gpt/gpt_model.py

    TODO: Currently we still need to patch `forward` because we need to pass `temperature`
    explicitly to `self._postprocess` when calling, maybe there can be a better way to handle this?
    """

    inference_context = deprecate_inference_params(inference_context, inference_params)

    preprocess_kwargs = {}
    if padding_mask is not None:
        # Only forward the kwarg when set: older Megatron-Core _preprocess
        # signatures (without MoE router padding support) stay compatible.
        preprocess_kwargs["padding_mask"] = padding_mask
    preproc_output = model._preprocess(
        input_ids=input_ids,
        position_ids=position_ids,
        decoder_input=decoder_input,
        inference_context=inference_context,
        packed_seq_params=packed_seq_params,
        **preprocess_kwargs,
    )

    (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = preproc_output[:5]

    decoder_extra_block_kwargs = extra_block_kwargs or {}
    if padding_mask is not None:
        # _preprocess scatters the mask across sequence-parallel ranks.
        decoder_extra_block_kwargs["padding_mask"] = preproc_output[5]
    if getattr(model.config, "moe_n_hash_layers", 0) > 0 and input_ids is not None:
        decoder_extra_block_kwargs["input_ids"] = input_ids

    # Run decoder.
    decoder_output = model.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask,
        inference_context=inference_context,
        rotary_pos_emb=rotary_pos_emb,
        rotary_pos_cos=rotary_pos_cos,
        rotary_pos_sin=rotary_pos_sin,
        packed_seq_params=packed_seq_params,
        sequence_len_offset=sequence_len_offset,
        **decoder_extra_block_kwargs,
        **kwargs,
    )
    hidden_states = decoder_output[0] if isinstance(decoder_output, tuple) else decoder_output

    if not model.post_process:
        return hidden_states

    output = CausalLMOutputForPPO(
        loss=None,
        logits=None,
        past_key_values=None,
        hidden_states=hidden_states,
        attentions=None,
    )

    if model.config.sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    # Get the output weight - use embedding weight if output_layer is None or weight is shared
    if hasattr(model, "output_layer") and model.output_layer is not None and model.output_layer.weight is not None:
        output_weight = model.output_layer.weight
    else:
        # When embeddings are tied, use the embedding weight
        output_weight = model.embedding.word_embeddings.weight

    logprobs, entropy = linear_cross_entropy(
        hidden_states,
        output_weight,
        labels,
        temperature,
        "none",
        parallel_state.get_tensor_model_parallel_group(),
    )

    if has_config_logger_enabled(model.config):
        payload = OrderedDict(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "decoder_input": decoder_input,
                "logprobs": logprobs,
                "entropy": entropy,
            }
        )
        log_config_to_disk(model.config, payload, prefix="input_and_logits")

    output.entropy = entropy
    output.log_probs = logprobs

    return output
