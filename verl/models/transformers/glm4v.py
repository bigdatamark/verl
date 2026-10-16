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
# verl patches for HF GLM-4V (Glm4vForConditionalGeneration): padding-free 3D (t/h/w) mRoPE position
# id computation, a flash-attention text attention forward supporting packed sequences (varlen via
# position ids) and Ulysses sequence parallelism, multimodal input-embedding merging, and forward
# variants returning logits or fused log_probs/entropy for PPO.
# Module state: flash-attention feature flags (window size, deterministic, top-left mask) chosen from
# flash_attn or the NPU flash-attention integration; deterministic mode from env
# FLASH_ATTENTION_DETERMINISTIC.

import inspect
import itertools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import _flash_attention_forward, fa_peft_integration_check
from transformers.models.glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vForConditionalGeneration,
    Glm4vTextAttention,
)
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

from verl.utils.device import is_npu_available
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_group,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

if is_npu_available:
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

    _flash_supports_window_size = "window_size" in inspect.signature(flash_attn_func).parameters
    _flash_supports_deterministic = "deterministic" in inspect.signature(flash_attn_func).parameters
    _flash_use_top_left_mask = flash_attn_supports_top_left_mask()

_flash_deterministic_enabled = os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: computes mRoPE position ids for one unbatched example. If ids and any grid are given:
    # classifies each valid (mask==1) token as image (image token outside video markers), video (image
    # token between begin/end-of-video markers) or text, groups consecutive equal types, and for each
    # group, starting from previous max position + 1: image -> 3D grid of (t, h/merge, w/merge) indices;
    # video -> one frame (t index advancing per video group) of (h/merge, w/merge), advancing to the next
    # video after its frame count; text -> same linear range on all three rows. Results fill positions of
    # valid tokens; others keep 1. Without grids: cumulative mask positions (masked = 1) or a plain range,
    # expanded to three rows.
    # Inputs: processor - with image_processor.merge_size and tokenizer; input_ids 1-D; image_grid_thw /
    # video_grid_thw [n, 3]; attention_mask 1-D optional.
    # Outputs: long tensor [3, seq_len].
    # Side effects: host syncs.
    # KNOWN ISSUE: video temporal position uses a frame counter reset by any non-video group rather than
    # the grid's t value.
    """
    Gets the position ids for GLM4V in padding-free format.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image|>")
    video_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
    video_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|end_of_video|>")

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)  # (3, seqlen)
        image_index, video_index = 0, 0
        video_group_index = 0

        input_ids_filtered = input_ids[attention_mask == 1]
        input_tokens = input_ids_filtered.tolist()

        input_token_type = []
        video_check_flg = False
        for token in input_tokens:
            if token == video_start_token_id:
                video_check_flg = True
            elif token == video_end_token_id:
                video_check_flg = False

            if token == image_token_id and not video_check_flg:
                input_token_type.append("image")
            elif token == image_token_id and video_check_flg:
                input_token_type.append("video")
            else:
                input_token_type.append("text")

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        llm_pos_ids_list = []
        video_frame_num = 1

        for modality_type, start_idx, end_idx in input_type_group:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0

            if modality_type == "image":
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                image_index += 1
                video_frame_num = 1

            elif modality_type == "video":
                t, h, w = (
                    video_frame_num,
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )

                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t,
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                for t_idx in range(llm_grid_t):
                    t_index = torch.tensor(t_idx).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(1, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(1, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

                video_group_index += 1

                if video_group_index >= video_grid_thw[video_index][0]:
                    video_index += 1
                    video_group_index = 0

                video_frame_num += 1

            else:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                video_frame_num = 1

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def prepare_fa2_from_position_ids(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, position_ids: torch.Tensor
):
    # [FUNCTION DOC]
    # HELPER: Flattens batch and sequence of q/k/v to [total, heads, dim] and derives cu_seqlens from
    # positions where position id equals 0 plus the total length; max length is the max segment length.
    # Inputs: query, key, value [b, s, h, d]; position_ids [b, s].
    # Outputs: (q, k, v, (cu_seqlens, cu_seqlens), (max_len, max_len)).
    # Exceptions: AssertionError if position_ids is not 2-D. Side effects: none.
    assert position_ids.ndim == 2  # (batch_size, seq_length)
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.view(-1)
    cu_seqlens = torch.cat(
        (
            (position_ids == 0).nonzero().view(-1).to(torch.int32),
            torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
        )
    )
    max_length = cu_seqlens.diff().max()  # use cu_seqlens to infer max_length for qwen2vl mrope
    return (query, key, value, (cu_seqlens, cu_seqlens), (max_length, max_length))


def _custom_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    position_ids: Optional[torch.Tensor] = None,
    use_top_left_mask: bool = False,
    deterministic: Optional[bool] = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Flash attention wrapper. Sets deterministic flag (argument or env default) and softcap when
    # supported, casts via PEFT dtype check to bf16, and under Ulysses SP all-to-alls q/k/v (gather
    # sequence, scatter heads) and all-gathers position ids along the sequence. If position ids are
    # non-monotonic (packed multiple sequences) and not decoding, runs flash_attn_varlen_func with
    # cu_seqlens from prepare_fa2_from_position_ids (dropout and softmax_scale taken from kwargs),
    # otherwise the HF _flash_attention_forward. Under SP, all-to-alls the output back.
    # Inputs: query/key/value [b, s_local, h, d]; attention_mask; query_length; is_causal; position_ids
    # [b, s_local]; use_top_left_mask; deterministic; kwargs (dropout, softmax_scale, softcap, others).
    # Outputs: attention output [b, s_local, h, d].
    # Exceptions: AssertionError if position_ids is not 2-D; Ulysses config validation errors.
    # Side effects: SP collectives.
    # KNOWN ISSUE: docstring mentions 3D mrope position ids but code asserts 2-D; with SP > 1 and
    # position_ids None the all-gather path fails.
    """
    Patches flash attention forward to handle 3D position ids in mrope. (3, batch_size, seq_length)
    """
    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    flash_kwargs = {}

    if _flash_supports_deterministic:
        flash_kwargs["deterministic"] = deterministic if deterministic is not None else _flash_deterministic_enabled

    if kwargs.get("softcap") is not None:
        flash_kwargs["softcap"] = kwargs.pop("softcap")

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype=torch.bfloat16
    )

    if position_ids is not None:
        assert position_ids.ndim == 2  # (batch_size, seq_length / sp_size)

    sp_size = get_ulysses_sequence_parallel_world_size()
    if sp_size > 1:
        # qkv: (batch_size, seq_length / sp_size, num_head, head_size)
        validate_ulysses_config(query_states.size(2), sp_size)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)
        position_ids_lst = [torch.empty_like(position_ids) for _ in range(sp_size)]
        position_ids = dist.all_gather(position_ids_lst, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.cat(position_ids_lst, dim=-1)  # (batch_size, seq_length)

    if position_ids is not None and query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all():
        batch_size = query_states.size(0)
        q, k, v, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids
        )
        attn_output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=kwargs.pop("dropout", 0.0),
            softmax_scale=kwargs.pop("softmax_scale", None),
            causal=is_causal,
            **flash_kwargs,
        )
        attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    else:
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=is_causal,
            use_top_left_mask=use_top_left_mask,
            deterministic=deterministic,
            **kwargs,
        )  # do not pass position_ids to old flash_attention_forward

    if sp_size > 1:
        # (batch_size, seq_length, num_head, head_size)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    return attn_output


def glm4v_attn_forward(
    self: "Glm4vTextAttention",
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> tuple[torch.Tensor, None, None]:
    # [FUNCTION DOC]
    # Behavior: replacement for Glm4vTextAttention.forward: projects q/k/v, applies rotary embedding with
    # the provided cos/sin (mrope sections already folded), repeats KV heads to match query heads, uses
    # dropout only in training, transposes to [b, s, h, d] and calls the custom flash-attention forward
    # with position ids, then reshapes to hidden size and applies o_proj.
    # Inputs: self - attention module; hidden_states [b, s_local, hidden]; attention_mask; position_ids;
    # position_embeddings (cos, sin); kwargs ignored.
    # Outputs: (attn_output, None).
    # Side effects: SP collectives inside attention.
    # KNOWN ISSUE: return annotation is a 3-tuple but a 2-tuple is returned.
    from transformers.models.glm4v.modeling_glm4v import apply_rotary_pos_emb, repeat_kv

    bsz, q_len, _ = hidden_states.size()  # q_len = seq_length / sp_size
    query_states = self.q_proj(hidden_states)  # (batch_size, seq_length / sp_size, num_heads * head_size)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Because the input can be padded, the absolute sequence length depends on the max position id.
    # Glm4vTextRotaryEmbedding already folds mrope_section into cos/sin, so this is the plain
    # application, exactly as Glm4vTextAttention.forward does it.
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    # This is before the transpose
    q_len = query_states.shape[2]

    # FA2 uses non-transposed inputs
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = _custom_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length=q_len,
        is_causal=getattr(self, "is_causal", True),
        dropout=dropout_rate,
        use_top_left_mask=_flash_use_top_left_mask,
        position_ids=position_ids,  # important: pass position ids
    )  # (batch_size, seq_length / sp_size, num_head, head_size)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _get_input_embeds(
    model: "Glm4vForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    # [FUNCTION DOC]
    # HELPER: Builds input embeddings by embedding tokens and masked-scattering vision encoder outputs into
    # image-token and video-token positions (after casting pixels to the visual dtype). For text-only
    # batches runs the vision tower on a dummy 16x1176 input (grid 1x4x4) and adds zero times its mean so
    # vision parameters stay in the autograd graph (needed by FSDP).
    # Inputs: model; input_ids; attention_mask; pixel_values; pixel_values_videos; image_grid_thw;
    # video_grid_thw.
    # Outputs: (inputs_embeds, attention_mask moved to embeds device or None).
    # Exceptions: ValueError when number of image/video tokens differs from feature count.
    # Side effects: vision forward passes; host sync.
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:  # handle mixed text-image data
        pixel_values = torch.zeros((16, 1176), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return inputs_embeds, attention_mask


def process_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Validates that position ids have shape [4, batch, seq] (text positions concatenated with the
    # 3 mRoPE rows) and returns them unchanged.
    # Inputs: position_ids. Outputs: same tensor. Exceptions: ValueError otherwise.
    if position_ids.ndim != 3 or position_ids.size(0) != 4:
        # we concat the text position ids with the 3D vision position ids by default
        # see https://github.com/huggingface/transformers/pull/39447
        raise ValueError("position_ids should be a 3D tensor of shape (4, batch_size, seq_length).")

    return position_ids


@dataclass
class Glm4vCausalLMOutputForPPO(Glm4vCausalLMOutputWithPast):
    # [CLASS DOC]
    # Dataclass extending Glm4vCausalLMOutputWithPast with optional per-token log_probs and entropy.
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None


def glm4v_base_forward(
    self: "Glm4vForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # Behavior: computes merged multimodal embeddings via _get_input_embeds and runs self.language_model
    # with input_ids None and inputs_embeds / attention_mask put into kwargs.
    # Inputs: self - GLM-4V model; input_ids; attention_mask; labels (unused); pixel/grid inputs; kwargs.
    # Outputs: language model output.
    # Side effects: mutates kwargs dict.
    kwargs["inputs_embeds"], kwargs["attention_mask"] = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
    return self.language_model(
        input_ids=None,
        **kwargs,
    )


def glm4v_forward(
    self: "Glm4vForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # [FUNCTION DOC]
    # HELPER: Calls self.model with validated 4-row position ids and all multimodal arguments.
    # Inputs: self - Glm4vForConditionalGeneration; input_ids; attention_mask; position_ids; pixel/grid
    # inputs; kwargs. Outputs: base model output.
    # Exceptions: ValueError from process_position_ids (including when position_ids is None).
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=process_position_ids(position_ids),
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        **kwargs,
    )


def forward_with_normal_backend(
    self: Glm4vForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Glm4vCausalLMOutputWithPast":
    # [FUNCTION DOC]
    # Behavior: runs the base model and computes full logits with lm_head.
    # Inputs: self; input_ids; labels and temperature (both unused); kwargs to glm4v_forward.
    # Outputs: Glm4vCausalLMOutputWithPast with logits and hidden_states.
    # KNOWN ISSUE: temperature is accepted but not applied to logits.
    outputs = glm4v_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Glm4vCausalLMOutputWithPast(
        logits=logits,
        hidden_states=outputs.hidden_states,
    )


def forward_with_torch_backend(
    self: Glm4vForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple | Glm4vCausalLMOutputForPPO:
    # [FUNCTION DOC]
    # Behavior: runs the base model, picks targets (shift_labels, else labels or input_ids rolled left by
    # one) and computes log_probs/entropy with FusedLinearForPPO (backend from
    # self._verl_fused_kernels_backend, default torch) at the given temperature.
    # Inputs: self; input_ids; labels; temperature; shift_labels; kwargs.
    # Outputs: Glm4vCausalLMOutputForPPO (log_probs, entropy, hidden_states).
    # Exceptions: RuntimeError when no targets are available.
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = glm4v_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
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
    return Glm4vCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: Glm4vForConditionalGeneration,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple | Glm4vCausalLMOutputForPPO:
    # [FUNCTION DOC]
    # Behavior: same as the torch backend but uses the triton linear_cross_entropy kernel (no reduction).
    # Inputs: same. Outputs: Glm4vCausalLMOutputForPPO.
    # Exceptions: RuntimeError when no targets are available.
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = glm4v_forward(self, input_ids, **kwargs)
    hidden_states = outputs[0]

    # See `dense_common.forward_with_torch_backend` for the `shift_labels`
    # rationale (issue #6068).
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
    return Glm4vCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
