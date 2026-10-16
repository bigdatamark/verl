# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
# Monkey patches enabling Multi-Token Prediction (MTP) training in Megatron GPT models within verl: a replacement
# GPTModel._postprocess that computes MTP auxiliary losses (via Megatron's process_mtp_loss when available, else a
# legacy manual path that keeps the lm_head out of the MTP gradient), a patched MTP-layer _get_embeddings that detaches
# embeddings, and a patched MTP-layer _checkpointed_forward for activation recompute. Feature flags for the installed
# Megatron API are detected at import time.

import copy
import warnings
from inspect import signature
from typing import Callable

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper, roll_tensor

try:
    from megatron.core.transformer.multi_token_prediction import process_mtp_loss as _process_mtp_loss

    _HAS_PROCESS_MTP_LOSS = True
    _PROCESS_MTP_LOSS_PARAMS = set(signature(_process_mtp_loss).parameters)
except ImportError:
    _HAS_PROCESS_MTP_LOSS = False
    _PROCESS_MTP_LOSS_PARAMS = set()

_ROLL_TENSOR_HAS_FILL_VALUE = "fill_value" in signature(roll_tensor).parameters
_MTP_LOSS_NORMALIZATION_FACTOR_ATTR = "_verl_mtp_loss_normalization_factor"

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    from verl.utils.megatron_utils import unwrap_model


def _resolve_cp_group(module, packed_seq_params=None):
    # [FUNCTION DOC]
    # HELPER: Returns the context-parallel group to use: the dynamic CP group from packed_seq_params if present, else the
    # module's cp_group attribute, else module.pg_collection.cp (may be None).
    """Use the per-microbatch Dynamic CP group when packed metadata provides one."""
    dynamic_cp_group = getattr(packed_seq_params, "cp_group", None)
    if dynamic_cp_group is not None:
        return dynamic_cp_group
    static_cp_group = getattr(module, "cp_group", None)
    if static_cp_group is None:
        static_cp_group = getattr(getattr(module, "pg_collection", None), "cp", None)
    return static_cp_group


def _get_mtp_loss_config(config, packed_seq_params=None):
    # [FUNCTION DOC]
    # HELPER: When per-token loss is enabled and packed_seq_params carries a verl MTP loss normalization factor, returns a
    # shallow copy of config with mtp_loss_scaling_factor multiplied by that factor; otherwise returns config unchanged.
    # Exceptions: ValueError if the factor is not positive.
    """Adjust MTP scaling when MCore finalizes against routed rather than loss tokens."""
    factor = getattr(packed_seq_params, _MTP_LOSS_NORMALIZATION_FACTOR_ATTR, None)
    if not config.calculate_per_token_loss or factor is None:
        return config
    if factor <= 0:
        raise ValueError(f"MTP loss normalization factor must be positive, got {factor}")

    adjusted_config = copy.copy(config)
    adjusted_config.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor * factor
    return adjusted_config


def _get_patching_model(model: torch.nn.Module):
    # [FUNCTION DOC]
    # HELPER: Returns the GPTModel to patch (the unwrapped model or its language_model), or None with a printed message if unsupported.
    model = unwrap_model(model)
    if isinstance(model, GPTModel):
        return model

    if not (hasattr(model, "language_model") and isinstance(model.language_model, GPTModel)):
        print(f"Model {model.__class__.__name__} is not a supported for fused forward")
        return None

    return model.language_model


def patch_postprocess(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: saves the GPTModel's _postprocess as _postprocess_backup and binds _megatron_gptmodel_postprocess in its place.
    # Side effects: mutates the model instance.
    # KNOWN ISSUE: calling it twice overwrites the backup with the already-patched method, making unpatch ineffective.
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess_backup = model._postprocess
        model._postprocess = _megatron_gptmodel_postprocess.__get__(model, model.__class__)


def unpatch_postprocess(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: restores _postprocess from _postprocess_backup.
    # Exceptions: AttributeError if the model was never patched.
    model = _get_patching_model(model)
    if model is not None:
        model._postprocess = model._postprocess_backup


# copy from https://github.com/NVIDIA/Megatron-LM/blob/23e092f41ec8bc659020e401ddac9576c1cfed7e/megatron/core/models/gpt/gpt_model.py
# patch the postprocess method of GPTModel to support advanced features like MTP, 1f1b overlap, etc.
def _megatron_gptmodel_postprocess(
    self,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    mtp_in_postprocess=None,
    loss_mask=None,
    decoder_input=None,
    attention_mask=None,
    padding_mask=None,
    inference_params=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    runtime_gather_output=None,
    extra_block_kwargs=None,
    inference_context=None,
    output_processor=None,
    output_processor_context=None,
    is_spec_decode=None,
    mhc_multistream=None,
):
    # [FUNCTION DOC]
    # HELPER: Replacement GPTModel._postprocess (bound to the model).
    # Behavior: uses the shared embedding as output weight when tied. If MTP runs in postprocess and labels are given,
    # runs self.mtp with extra block kwargs, padding_mask (only if the MTP forward accepts it; cached on the MTP module)
    # and mhc_multistream when provided. Returns hidden states on non-last stages. When mtp_num_layers > 0 and labels are
    # given: with process_mtp_loss available, calls it with only the keyword arguments its signature accepts (hidden
    # states, labels, loss mask, output layer/weight, training flag, loss function, normalization-adjusted config, CP
    # group, packed params, muP logit scaling fn, input ids, and TP group when supported) and uses its returned hidden
    # states. Otherwise (legacy): splits hidden states into main and per-MTP-layer chunks; for each MTP layer rolls labels
    # and loss mask left by one, computes logits through the output layer with detached parameters/weight (so MTP loss
    # does not update lm_head), computes masked LM loss, logs the mean loss in training, and attaches the scaled loss to
    # the main hidden states via MTPLossAutoScaler (per-token mode rescales by original/rolled token counts; otherwise
    # divides by rolled token count, min 1). Finally computes logits from the main hidden states.
    # Inputs: Megatron postprocess arguments; output_processor, output_processor_context, is_spec_decode,
    # inference_context accepted for signature compatibility.
    # Outputs: hidden states (non-last stage) or logits transposed to [batch, seq, vocab].
    # KNOWN ISSUE: output_processor / output_processor_context are accepted but ignored, so the fused output-processor hook
    # (see model_forward_fused) is bypassed while this patch is active.
    # Side effects: MTP loss logging; caches a flag on self.mtp.
    """Postprocesses decoder hidden states to generate logits or compute loss.

    Applies Multi-Token Prediction if enabled, generates output logits through
    the output layer, and computes language model loss when labels are provided.
    """

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    if mtp_in_postprocess and labels is not None:
        mtp_kwargs = dict(extra_block_kwargs or {})
        if not hasattr(self.mtp, "_forward_has_padding_mask"):
            self.mtp._forward_has_padding_mask = "padding_mask" in signature(self.mtp.forward).parameters
        if self.mtp._forward_has_padding_mask:
            mtp_kwargs["padding_mask"] = padding_mask
        if mhc_multistream is not None:
            mtp_kwargs["mhc_multistream"] = mhc_multistream
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            embedding=self.embedding,
            **mtp_kwargs,
        )

    if not self.post_process:
        return hidden_states

    # Skip when mtp_num_layers is None or 0
    if self.config.mtp_num_layers and labels is not None:
        if _HAS_PROCESS_MTP_LOSS:
            # New Megatron API (>= verl megatron fork with process_mtp_loss):
            # process_mtp_loss handles chunking, rolling, loss scaling all internally.
            pg_collection = getattr(self, "pg_collection", None)
            cp_group = _resolve_cp_group(self, packed_seq_params)
            mtp_loss_config = _get_mtp_loss_config(self.config, packed_seq_params)
            scale_logits_fn = self._scale_logits if (hasattr(self, "_scale_logits") and self.config.use_mup) else None
            process_mtp_loss_kwargs = {
                "hidden_states": hidden_states,
                "labels": labels,
                "loss_mask": loss_mask,
                "output_layer": self.output_layer,
                "output_weight": output_weight,
                "runtime_gather_output": runtime_gather_output,
                "is_training": self.training,
                "compute_language_model_loss": self.compute_language_model_loss,
                "config": mtp_loss_config,
                "cp_group": cp_group,
                "packed_seq_params": packed_seq_params,
                "scale_logits_fn": scale_logits_fn,
                "input_ids": input_ids,
            }
            if "tp_group" in _PROCESS_MTP_LOSS_PARAMS:
                process_mtp_loss_kwargs["tp_group"] = getattr(self, "tp_group", None) or (
                    pg_collection.tp if pg_collection is not None else None
                )
            process_mtp_loss_kwargs = {
                key: val for key, val in process_mtp_loss_kwargs.items() if key in _PROCESS_MTP_LOSS_PARAMS
            }
            hidden_states = _process_mtp_loss(**process_mtp_loss_kwargs)
        else:
            # Legacy Megatron API: manual rolling + detached output-layer functional_call.
            mtp_labels = labels.clone()
            hidden_states_list = torch.chunk(hidden_states, 1 + self.config.mtp_num_layers, dim=0)
            hidden_states = hidden_states_list[0]
            if loss_mask is None:
                loss_mask = torch.ones_like(mtp_labels)
            cp_group = getattr(self, "cp_group", None)
            # Capture pre-roll token count so calculate_per_token_loss=True can
            # re-normalize MTP gradients against the main-loss token count.
            original_num_tokens = loss_mask.sum()
            for mtp_layer_number in range(self.config.mtp_num_layers):
                mtp_labels, _ = roll_tensor(
                    mtp_labels,
                    shifts=-1,
                    dims=-1,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
                loss_mask, num_tokens = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
                # Detach output-layer params so MTP loss does not update lm_head.
                output_layer_params = {k: v.detach() for k, v in self.output_layer.named_parameters()}
                output_layer_buffers = dict(self.output_layer.named_buffers())
                mtp_logits, _ = torch.func.functional_call(
                    self.output_layer,
                    {**output_layer_params, **output_layer_buffers},
                    args=(hidden_states_list[mtp_layer_number + 1],),
                    kwargs={
                        "weight": output_weight.detach() if output_weight is not None else None,
                        "runtime_gather_output": runtime_gather_output,
                    },
                )
                mtp_loss = self.compute_language_model_loss(mtp_labels, mtp_logits)
                mtp_loss = loss_mask * mtp_loss
                mtp_loss_for_log = torch.sum(mtp_loss) / num_tokens if num_tokens > 0 else mtp_loss.new_tensor(0.0)
                if self.training:
                    MTPLossLoggingHelper.save_loss_to_tracker(
                        mtp_loss_for_log,
                        mtp_layer_number,
                        self.config.mtp_num_layers,
                        avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                mtp_loss_scale = self.config.mtp_loss_scaling_factor / self.config.mtp_num_layers
                if self.config.calculate_per_token_loss:
                    # When calculate_per_token_loss is enabled, finalize_model_grads will
                    # divide all gradients by total_num_tokens (from main loss).
                    # However, MTP has fewer valid tokens due to rolling. To ensure correct
                    # per-token gradient weighting, we normalize by the rolled token count
                    # and re-scale by the original token count.
                    # Avoid division by zero
                    num_tokens_safe = torch.clamp(num_tokens, min=1)
                    hidden_states = MTPLossAutoScaler.apply(
                        hidden_states,
                        mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe),
                    )
                else:
                    safe_num_tokens = num_tokens.clamp(min=1)
                    hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens)

    logits, _ = self.output_layer(hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output)
    # [s b h] => [b s h]
    return logits.transpose(0, 1).contiguous()


def patch_mtp_layer_get_embeddings(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: collects MultiTokenPredictionLayer instances (from GPTModel.mtp.layers, or model.layers otherwise) and, for
    # each not yet patched, records whether its _get_embeddings accepts padding_mask, backs up the original and binds
    # _patched_get_embeddings_for_detach.
    # Outputs: true if at least one layer was patched; false otherwise.
    # Side effects: mutates layers; prints counts.
    """Patch the _get_embeddings method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)
    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    if target_layers:
        patched_count = 0
        for layer in target_layers:
            if hasattr(layer, "_get_embeddings_backup"):
                continue
            layer._get_embeddings_has_padding_mask = "padding_mask" in signature(layer._get_embeddings).parameters
            layer._get_embeddings_backup = layer._get_embeddings
            layer._get_embeddings = _patched_get_embeddings_for_detach.__get__(layer, layer.__class__)
            patched_count += 1
        if patched_count:
            print(f"Found and patched {patched_count} MTP layer(s) in any of the actor modules")
        return patched_count > 0
    else:
        print("No MTP layers found to patch in any of the actor modules")
        return False


def patch_mtp_layer_checkpointed_forward(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: collects MTP layers as above and, for each not yet patched whose _checkpointed_forward's first parameter is
    # forward_func (assumed so if the signature cannot be inspected), backs it up and binds _patched_checkpointed_forward.
    # Outputs: true if any layer was patched.
    # Side effects: mutates layers; prints counts.
    """Patch the _checkpointed_forward method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)
    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    if target_layers:
        patched_count = 0
        for layer in target_layers:
            if hasattr(layer, "_checkpointed_forward_backup"):
                continue
            try:
                params = list(signature(layer._checkpointed_forward).parameters)
            except (TypeError, ValueError):
                params = ["forward_func"]
            if not params or params[0] != "forward_func":
                continue
            layer._checkpointed_forward_backup = layer._checkpointed_forward
            layer._checkpointed_forward = _patched_checkpointed_forward.__get__(layer, layer.__class__)
            patched_count += 1
        if patched_count:
            print(f"Found and patched checkpointed forward for {patched_count} MTP layer(s)")
        return patched_count > 0
    else:
        print("No MTP layers found to patch checkpointed forward")
        return False


def unpatch_mtp_layer_get_embeddings(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: restores _get_embeddings from the backup on every patched MTP layer and removes the backup and the cached
    # padding-mask flag.
    # Outputs: true if any layer was unpatched. Side effects: prints count.
    """Unpatch the _get_embeddings method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)

    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    unpatched_count = 0
    for layer in target_layers:
        if hasattr(layer, "_get_embeddings_backup"):
            layer._get_embeddings = layer._get_embeddings_backup
            delattr(layer, "_get_embeddings_backup")
            if hasattr(layer, "_get_embeddings_has_padding_mask"):
                delattr(layer, "_get_embeddings_has_padding_mask")
            unpatched_count += 1

    if unpatched_count > 0:
        print(f"Unpatched {unpatched_count} MTP layer(s)")
        return True
    return False


def unpatch_mtp_layer_checkpointed_forward(model: torch.nn.Module):
    # [FUNCTION DOC]
    # Behavior: restores _checkpointed_forward from the backup on every patched MTP layer and removes the backup.
    # Outputs: true if any layer was unpatched. Side effects: prints count.
    """Unpatch the _checkpointed_forward method of MultiTokenPredictionLayer"""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

    # Unwrap each model in the actor_module to get the actual GPTModel
    model = _get_patching_model(model)

    # Collect all MultiTokenPredictionLayer instances
    target_layers = []

    if isinstance(model, GPTModel):
        # Check if GPTModel has MTP and find the layers
        if hasattr(model, "mtp") and hasattr(model.mtp, "layers"):
            for layer in model.mtp.layers:
                if isinstance(layer, MultiTokenPredictionLayer):
                    target_layers.append(layer)
    elif hasattr(model, "layers"):
        # Check if any layer in the model is MultiTokenPredictionLayer
        for layer in model.layers:
            if isinstance(layer, MultiTokenPredictionLayer):
                target_layers.append(layer)

    unpatched_count = 0
    for layer in target_layers:
        if hasattr(layer, "_checkpointed_forward_backup"):
            layer._checkpointed_forward = layer._checkpointed_forward_backup
            delattr(layer, "_checkpointed_forward_backup")
            unpatched_count += 1

    if unpatched_count > 0:
        print(f"Unpatched checkpointed forward for {unpatched_count} MTP layer(s)")
        return True
    return False


def _patched_get_embeddings_for_detach(
    self,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    embedding: Callable,
    hidden_states: torch.Tensor,
    packed_seq_params=None,
    padding_mask=None,
):
    # [FUNCTION DOC]
    # HELPER: Replacement MultiTokenPredictionLayer._get_embeddings (bound to the layer).
    # Behavior: resolves the CP group, rolls input_ids and position_ids (and padding_mask if given, filling with true when
    # supported) left by one across packed sequences, embeds the rolled ids, detaches the embedding output, and turns
    # hidden_states into a viewless tensor with gradients but without keeping its graph.
    # Outputs: (input_ids, position_ids, padding_mask, decoder_input, hidden_states) if the original accepted
    # padding_mask, else (input_ids, position_ids, decoder_input, hidden_states).
    # KNOWN ISSUE: the docstring is template text and does not describe the detach behavior; also using keep_graph=False on
    # hidden_states may cut gradient flow from MTP into the main decoder, which seems to be the intent (detach_encoder).
    """
    Patched version of _get_embeddings method for MultiTokenPredictionLayer.

    This is a modified version that you can customize according to your needs.
    The original implementation is preserved below with modifications.
    """

    # You can modify the logic here as needed
    # For example, you could:
    # - Change the shift amount in roll_tensor
    # - Apply custom transformations to input_ids or position_ids
    # - Add debugging information
    # - Modify the embedding computation

    # Original logic with custom modifications
    from megatron.core.utils import make_viewless_tensor

    cp_group = _resolve_cp_group(self, packed_seq_params)

    # Calc logits for the current Multi-Token Prediction (MTP) layers.
    input_ids, _ = roll_tensor(
        input_ids,
        shifts=-1,  # You can modify this shift value
        dims=-1,
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
    )
    position_ids, _ = roll_tensor(
        position_ids,
        shifts=-1,  # You can modify this shift value
        dims=-1,
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
    )
    if padding_mask is not None:
        roll_kwargs = {"fill_value": True} if _ROLL_TENSOR_HAS_FILL_VALUE else {}
        padding_mask, _ = roll_tensor(
            padding_mask,
            shifts=-1,
            dims=-1,
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            **roll_kwargs,
        )

    # embedding computation - you can modify this part
    decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)

    # Apply custom transformations if needed
    # For example: decoder_input = some_custom_function(decoder_input)

    # Detach token embeddings and main-decoder hidden states for detach_encoder.
    decoder_input = decoder_input.detach()
    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=False)

    if getattr(self, "_get_embeddings_has_padding_mask", False):
        return input_ids, position_ids, padding_mask, decoder_input, hidden_states
    return input_ids, position_ids, decoder_input, hidden_states


def _patched_checkpointed_forward(self, forward_func, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Replacement MultiTokenPredictionLayer._checkpointed_forward.
    # Behavior: separates tensor arguments (positional and keyword) from constants, recording a rebuild spec, so only
    # tensors are passed through activation checkpointing. With recompute_method "uniform" (requires recompute_num_layers
    # 1) checkpoints the rebuild-and-call function via Transformer Engine checkpointing when fp8 is on, else Megatron's
    # tensor_parallel checkpoint. With "block", warns and runs without recompute.
    # Inputs: forward_func and its args/kwargs.
    # Outputs: forward_func's result.
    # Exceptions: AssertionError for uniform with recompute_num_layers != 1; ValueError for other recompute methods.
    """Checkpoint MTP forward while keeping non-tensor args out of saved tensors."""
    # Reference: THUDM/slime Megatron MTP recompute patch.
    # https://github.com/THUDM/slime/blob/6961f5970e9dbb4716a10ba4a54a28fa3876d274/docker/patch/latest/megatron.patch#L723
    positional_specs: list = []
    kw_specs: list = []
    tensor_args: list[torch.Tensor] = []

    for arg in args:
        if torch.is_tensor(arg):
            positional_specs.append(("tensor", len(tensor_args)))
            tensor_args.append(arg)
        else:
            positional_specs.append(("const", arg))

    for key, value in kwargs.items():
        if torch.is_tensor(value):
            kw_specs.append((key, ("tensor", len(tensor_args))))
            tensor_args.append(value)
        else:
            kw_specs.append((key, ("const", value)))

    def run(*flat_tensor_args):
        # [FUNCTION DOC]
        # HELPER: Rebuilds the original positional and keyword arguments from the flat tensor list and the recorded specs and
        # calls forward_func.
        rebuilt_args = []
        for spec_type, payload in positional_specs:
            if spec_type == "tensor":
                rebuilt_args.append(flat_tensor_args[payload])
            else:
                rebuilt_args.append(payload)

        rebuilt_kwargs = {}
        for key, (spec_type, payload) in kw_specs:
            if spec_type == "tensor":
                rebuilt_kwargs[key] = flat_tensor_args[payload]
            else:
                rebuilt_kwargs[key] = payload

        return forward_func(*rebuilt_args, **rebuilt_kwargs)

    tensor_args_tuple = tuple(tensor_args)

    def checkpoint_handler():
        # [FUNCTION DOC]
        # HELPER: Runs the checkpointed call, using te_checkpoint (with the CUDA RNG tracker and TP group) when fp8 is enabled,
        # otherwise tensor_parallel.checkpoint; both honor distribute_saved_activations.
        if self.config.fp8:
            from megatron.core.extensions.transformer_engine import te_checkpoint

            return te_checkpoint(
                run,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                *tensor_args_tuple,
            )
        return tensor_parallel.checkpoint(run, self.config.distribute_saved_activations, *tensor_args_tuple)

    if self.config.recompute_method == "uniform":
        assert self.config.recompute_num_layers == 1, "recompute_num_layers must be 1 for MTP recompute"
        return checkpoint_handler()
    if self.config.recompute_method == "block":
        warnings.warn("recompute_method == 'block' is not supported for MTP yet. Skipping recompute.", stacklevel=2)
        return forward_func(*args, **kwargs)
    raise ValueError(f"Invalid activation recompute method: {self.config.recompute_method}")
