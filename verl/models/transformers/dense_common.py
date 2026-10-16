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
# Generic replacement forward functions for HF text-only causal LMs that compute per-token log-probs
# and entropy with fused linear+softmax kernels instead of materializing full logits. Defines the
# PPO output dataclass and torch / triton backend variants, bound as model.forward by the monkey
# patcher.

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    # [CLASS DOC]
    # Dataclass extending CausalLMOutputWithPast with optional log_probs and entropy tensors (per token);
    # logits is left unset by the fused forwards.
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> CausalLMOutputWithPast:
    # [FUNCTION DOC]
    # HELPER: Runs only the decoder backbone (self.model) with all given arguments, resolving
    # output_attentions / output_hidden_states defaults from self.config.
    # Inputs: self - HF *ForCausalLM module with .model and .config; standard HF forward arguments.
    # Outputs: the backbone output (BaseModelOutputWithPast or tuple depending on return_dict).
    # Side effects: none beyond the forward pass / cache updates.
    # KNOWN ISSUE: the docstring ends with a stray code-fence marker.
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    # [FUNCTION DOC]
    # Behavior: runs the backbone, requires return_dict, chooses target ids (shift_labels if given, else
    # labels rolled left by one along the last dim, else input_ids rolled left), and computes log_probs and
    # entropy of the targets under lm_head weights scaled by temperature via FusedLinearForPPO (backend
    # from self._verl_fused_kernels_backend, default torch).
    # Inputs: self - causal LM; HF forward args; labels / shift_labels - targets (shift_labels already
    # shifted, e.g. under Ulysses SP); temperature; logits_to_keep and loss_kwargs ignored.
    # Outputs: CausalLMOutputForPPO with log_probs, entropy, past_key_values, hidden_states, attentions.
    # Exceptions: NotImplementedError if return_dict is falsy; RuntimeError when no labels or input_ids.
    # Side effects: none.
    # KNOWN ISSUE: return_dict is not forwarded to the backbone (unlike the triton variant), and the check
    # happens after the backbone call.
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    # Loss calculations.
    # When the engine has already prepared globally-rolled labels (e.g. the FSDP
    # path under Ulysses SP, see issue #6068), it passes them as `shift_labels`
    # so we don't redo `torch.roll` on a sequence-parallel-sliced shard.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO(impl_backend=getattr(self, "_verl_fused_kernels_backend", "torch"))
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    # [FUNCTION DOC]
    # Behavior: same as the torch backend but forwards return_dict to the backbone and computes log_probs
    # and entropy with the triton linear_cross_entropy kernel (reduction none).
    # Inputs: same as forward_with_torch_backend.
    # Outputs: CausalLMOutputForPPO.
    # Exceptions: NotImplementedError if return_dict is falsy; RuntimeError when no targets available.
    # Side effects: none.
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations. See `forward_with_torch_backend` for why `shift_labels`
    # takes precedence over local `torch.roll` (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
