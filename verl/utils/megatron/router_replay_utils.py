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
# Glue between the RouterReplay state (router_replay_patch) and Megatron pipeline/sequence/context
# parallel layouts. Provides: per-stage layer/MoE-layer counting, gathering recorded top-k routes
# from local routers into (batch, seq, layers, topk) int16 maps (R2 record), building replay masks
# and aligning rollout-captured routes (R3), scattering routes back to local routers for replay,
# VPP reordering, PP gathering, and helpers to query replay actions. Route indices are carried as
# int16 and bit-reinterpreted as uint8 for collectives. Module-level device_name is the current
# accelerator name.

"""
Router Replay Utilities
Utilities for handling router replay functionality in Megatron models.
"""

import inspect
import warnings

import torch

try:
    from megatron.core.pipeline_parallel.utils import is_vp_first_stage, is_vp_last_stage
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    pass

from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel.schedules import get_schedule_table
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region, scatter_to_sequence_parallel_region
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from verl.models.mcore.util import (
    postprocess_packed_seqs,
    postprocess_thd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)
from verl.utils.device import get_device_name
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction

device_name = get_device_name()


def _context_parallel_layout(tf_config) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "contiguous" when tf_config.experimental_attention_variant is "dsv4_hybrid",
    # else "zigzag".
    # KNOWN ISSUE: this definition is shadowed by an identical, documented redefinition later in the module.
    if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid":
        return "contiguous"
    return "zigzag"


# from megatron.core.transformer.transformer_block import get_num_layers_to_build
def get_num_layers_to_build(config: TransformerConfig, vp_stage: int | None = None, pp_rank: int | None = None) -> int:
    # [FUNCTION DOC]
    # Behavior: local copy of Megatron's get_num_layers_to_build. If a custom pipeline layout exists,
    # delegate to it for decoder layers at vp_stage. Otherwise (pp_rank defaults to current PP rank):
    # with uneven first/last stage sizes (incompatible with embedding/loss accounting), distribute
    # remaining layers evenly over middle stages and use the explicit counts on first/last stages;
    # otherwise count embedding/loss as extra layers when configured and split evenly. With VPP and
    # PP > 1, divide the per-rank count by vp_size. Subtract one on the first (last) virtual+pipeline
    # stage when the embedding (loss) is counted in the split.
    # Inputs: config - TransformerConfig; vp_stage; pp_rank.
    # Outputs: int number of transformer layers for that stage.
    # Exceptions: AssertionError on non-divisible splits or negative results; NameError if
    # is_vp_first_stage/is_vp_last_stage failed to import and embedding/loss accounting is on.
    """
    Determine the number of transformer layers to build for the current pipeline stage.
    Args:
        config (TransformerConfig): Configuration object containing transformer model parameters.
        vp_stage (Optional[int]): Virtual pipeline stage number.
        pp_rank (Optional[int]): Pipeline parallel rank.

    Returns:
        int: The number of layers to be built for the current pipeline stage.
    """
    # If we have a custom PP layout, straightforwardly
    # return the number of decoders in the layout array.
    if hasattr(config, "pipeline_model_parallel_layout") and config.pipeline_model_parallel_layout is not None:
        from megatron.core.transformer.enums import LayerType

        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )

    # Fallback for legacy tests.
    if pp_rank is None:
        pp_rank = mpu.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0
    is_last_pp_stage = pp_rank == config.pipeline_model_parallel_size - 1

    if config.num_layers_in_first_pipeline_stage is not None or config.num_layers_in_last_pipeline_stage is not None:
        assert not (config.account_for_embedding_in_pipeline_split or config.account_for_loss_in_pipeline_split), (
            " \
        Does not support standalone embedding stage and standalone loss stage with uneven pp"
        )
        # Number of layers to distribute over rest of pipeline stages
        layers_to_distribute = config.num_layers
        # Number of pipeline stages left for distributing transformer layers
        pipeline_stages_left = config.pipeline_model_parallel_size

        # If the uneven first (last) pipeline stage is enabled, remove the specified number
        # of layers to calculate the number of layers on each middle pipeline stage.
        if config.num_layers_in_first_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_first_pipeline_stage
            pipeline_stages_left -= 1

        if config.num_layers_in_last_pipeline_stage is not None:
            layers_to_distribute -= config.num_layers_in_last_pipeline_stage
            pipeline_stages_left -= 1

        # If pp_size <= 2, we do not have any intermediate pipeline stages, and we do not
        # need to check if the left over layers are divisible by the left over stages.
        if pipeline_stages_left > 0:
            assert layers_to_distribute % pipeline_stages_left == 0, (
                "With uneven pipelineing the left over layers must be divisible by left over stages"
            )
            num_layers_per_pipeline_rank = layers_to_distribute // pipeline_stages_left
        else:
            num_layers_per_pipeline_rank = 0

        # If the uneven first (last) pipeline stage is enabled, return the specified number
        # of layers for all virtual pipeline parallel stages within the first (last) pipeline
        # parallel stage.

        if is_first_pp_stage and config.num_layers_in_first_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_first_pipeline_stage

        if is_last_pp_stage and config.num_layers_in_last_pipeline_stage is not None:
            num_layers_per_pipeline_rank = config.num_layers_in_last_pipeline_stage
    else:
        # Include the embedding layer and loss layer into pipeline parallelism partition
        num_layers = config.num_layers
        if config.account_for_embedding_in_pipeline_split:
            num_layers += 1

        if config.account_for_loss_in_pipeline_split:
            num_layers += 1

        assert num_layers % config.pipeline_model_parallel_size == 0, (
            "num_layers should be divisible by pipeline_model_parallel_size"
        )
        num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

    vp_size = config.virtual_pipeline_model_parallel_size
    if vp_size is not None and config.pipeline_model_parallel_size > 1:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]

        assert num_layers_per_pipeline_rank % vp_size == 0, (
            f"num_layers_per_pipeline_rank {num_layers_per_pipeline_rank} \
            should be divisible by vp_size {vp_size}"
        )
        num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size

        num_layers_to_build = num_layers_per_virtual_stage

    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        num_layers_to_build = num_layers_per_pipeline_rank

    # The embedding (or loss) layer cannot function as a standalone transformer layer
    # Reduce the number of layers to construct by 1 on the first (or last) stage if the
    # embedding (or loss) layer is included in the pipeline parallelism partition and placement.
    if config.account_for_embedding_in_pipeline_split:
        if is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the first virtual pipeline stage"

    if config.account_for_loss_in_pipeline_split:
        if is_vp_last_stage(vp_stage, vp_size) and is_last_pp_stage:
            num_layers_to_build -= 1
            assert num_layers_to_build >= 0, "Not enough layers in the last virtual pipeline stage"

    return num_layers_to_build


def is_moe_layer(tf_config, layer_idx):
    # [FUNCTION DOC]
    # Behavior: with integer moe_layer_freq, layer is MoE when layer_idx % freq == 0; with a list,
    # when the entry at layer_idx equals 1.
    # Inputs: tf_config; layer_idx (0-based global). Outputs: bool.
    # Exceptions: ValueError for other/missing freq types; IndexError for out-of-range list access.
    moe_layer_freq = getattr(tf_config, "moe_layer_freq", None)

    if isinstance(moe_layer_freq, int):
        return layer_idx % moe_layer_freq == 0
    elif isinstance(moe_layer_freq, list):
        return moe_layer_freq[layer_idx] == 1
    else:
        raise ValueError(f"Unsupported moe_layer_freq type: {type(moe_layer_freq)}")


def get_moe_num_layers_to_build(
    config: TransformerConfig, vp_stage: int | None = None, pp_rank: int | None = None
) -> int:
    # [FUNCTION DOC]
    # Behavior: computes the stage's layer count and global layer offset (calling
    # get_transformer_layer_offset with whichever of vp_stage/pp_rank its signature supports), then
    # counts MoE layers in that global index range.
    # Inputs: config; vp_stage; pp_rank. Outputs: int.
    # KNOWN ISSUE: when get_transformer_layer_offset lacks pp_rank, an explicit pp_rank is ignored for
    # the offset (current rank is used) while the layer count still uses it.
    """Count the number of MoE layers assigned to the current rank.
    When ``moe_layer_freq`` is 1 or unset, every transformer layer is an MoE
    layer, so the count equals the total layer count. Otherwise only layers
    whose global index satisfies the frequency predicate are counted.
    Args:
        config: Megatron TransformerConfig providing layer layout information.
        vp_stage: Virtual-pipeline stage index (None defaults to current).
        pp_rank: Pipeline-parallel rank (None defaults to current).
    Returns:
        Number of MoE layers on the specified rank/stage.
    """
    total_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)

    sig = inspect.signature(get_transformer_layer_offset)
    # core 0.12.1 is not support vp_stage and pp_rank as parameters
    if "vp_stage" in sig.parameters and "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    elif "pp_rank" in sig.parameters:
        layer_offset = get_transformer_layer_offset(config, pp_rank=pp_rank)
    else:
        layer_offset = get_transformer_layer_offset(config)

    local_global_indices = range(layer_offset, layer_offset + total_layers)

    num_moe_layers = sum(1 for idx in local_global_indices if is_moe_layer(config, idx))

    return num_moe_layers


def _context_parallel_layout(tf_config) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the CP row layout ("contiguous" for dsv4_hybrid attention, else "zigzag") used
    # for THD packing; must match the engine's layout so routes align with token rows.
    """Return the CP row layout used for THD packing, matching MegatronEngine.

    Router-replay tensors must be sharded exactly like the model's own token rows; otherwise the
    replayed routes are attached to the wrong tokens (or the shapes disagree outright, since the
    zigzag layout doubles the alignment). Keep this in sync with
    ``MegatronEngine._get_context_parallel_layout``.
    """
    if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid":
        return "contiguous"
    return "zigzag"


def merge_router_topk_indices(
    attention_mask,
    input_ids,
    mini_layer_topk_idx_list,
    tf_config,
    vp_rank=None,
    local_cp_size=None,
    model=None,
):
    # [FUNCTION DOC]
    # Behavior (no grad): requires attention_mask for non-nested inputs. Selects routers either from
    # the given model (sorted by layer number; duplicates are an error) or from the global registry
    # slice for (tf_config, vp_rank; None -> 0). If none are found: error if the stage should contain
    # MoE layers, else append an empty int16 route map with zero layers (jagged per-sequence for
    # nested input_ids, else (batch, seq, 0, topk) on CPU) using a positive moe_router_topk. Otherwise
    # every router must have recorded indices; stacks them to (tokens, layers, topk) int16 on device,
    # views as uint8, all-gathers across the sequence-parallel (TP) region and adds a leading dim.
    # Then unpacks to the original layout: for nested inputs via preprocess/postprocess_thd_engine
    # (with fp8 padding for e4m3/hybrid, CP layout, csa_window_size min rows for dsv4_hybrid,
    # local_cp_size) and re-views values as int16 in a jagged tensor; for padded inputs via
    # preprocess/postprocess_packed_seqs to (batch, seq, layers, topk) then int16 view. Appends the
    # CPU result to mini_layer_topk_idx_list.
    # Inputs: attention_mask (batch, seq) or None for nested; input_ids (batch, seq) or jagged nested;
    # mini_layer_topk_idx_list - output list; tf_config; vp_rank; local_cp_size; model.
    # Outputs: none.
    # Exceptions: RuntimeError for missing mask, duplicate layers, missing expected routers, bad
    # topk, or routers without recorded indices.
    # Side effects: appends to the list; TP collective.
    # KNOWN ISSUE: docstring says vp_rank None means the current VP rank, but the code uses 0.
    """
    Merge recorded router top-k indices across sequence-parallel ranks for all router instances,
    then pack/unpack them to align with the original (batch, seq_len) layout and append the result.

    Args:
        attention_mask (torch.Tensor): Attention mask of shape [batch_size, seq_len]. Used to determine
            the valid token positions during pack/unpack.
        input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len]. Used together with
            attention_mask for sequence packing/unpacking.
        mini_layer_topk_idx_list (list): A Python list to which the merged top-k indices tensor will be appended.
        tf_config: Megatron/Transformer engine configuration object. Used to locate router instances for
            the current micro-batch.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        model: The forwarded model chunk. When given, routes are collected from its own routers instead
            of inferring their location in the process-global router registry.

    Returns:
        None: The function has side effects only; it appends a tensor of shape
        [1, dynamic_bs_all, layer_num, topk] to mini_layer_topk_idx_list.
    """
    with torch.no_grad():
        if not input_ids.is_nested and attention_mask is None:
            raise RuntimeError("router replay RECORD requires attention_mask for non-nested BSHD inputs.")
        vp_rank = 0 if vp_rank is None else vp_rank
        if model is not None:
            model_routers = sorted(iter_model_routers(model), key=lambda item: item[0])
            layer_numbers = [layer_number for layer_number, _ in model_routers]
            if len(layer_numbers) != len(set(layer_numbers)):
                raise RuntimeError(
                    "router replay RECORD found duplicate layer numbers in the forwarded model: "
                    f"layer_numbers={layer_numbers}, vp_rank={vp_rank}"
                )
            router_instances_list = [router for _, router in model_routers]
        else:
            router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

        if not router_instances_list:
            expected = get_moe_num_layers_to_build(tf_config, vp_rank)
            if expected:
                raise RuntimeError(
                    "router replay RECORD found no routers in a stage that expects MoE layers: "
                    f"expected={expected}, vp_rank={vp_rank}"
                )
            topk = getattr(tf_config, "moe_router_topk", None)
            if not isinstance(topk, int) or topk <= 0:
                raise RuntimeError(
                    "router replay RECORD cannot build an empty local route map without a positive "
                    f"moe_router_topk, got {topk!r}."
                )
            if input_ids.is_nested:
                empty_routes = [
                    torch.empty((tokens.shape[0], 0, topk), dtype=torch.int16) for tokens in input_ids.unbind()
                ]
                mini_layer_topk_idx_list.append(torch.nested.as_nested_tensor(empty_routes, layout=torch.jagged))
            else:
                batch_size, seq_len = attention_mask.shape[:2]
                mini_layer_topk_idx_list.append(
                    torch.empty((batch_size, seq_len, 0, topk), dtype=torch.int16, device="cpu")
                )
            return

        recorded_local_positions = [
            idx for idx, router in enumerate(router_instances_list) if router.recorded_topk_idx is not None
        ]
        missing = [idx for idx, router in enumerate(router_instances_list) if router.recorded_topk_idx is None]
        if missing:
            local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
            raise RuntimeError(
                "router replay RECORD did not capture all local routers: "
                f"missing_local_positions={missing}, selected={len(router_instances_list)}, "
                f"registry={len(RouterReplay.router_instances)}, "
                f"recorded_local_positions={recorded_local_positions}, local_layer_range={local_rank_info}, "
                f"pp_rank={mpu.get_pipeline_model_parallel_rank()}, vp_rank={vp_rank}"
            )

        layers_topk_idx = []
        for router in router_instances_list:
            layers_topk_idx.append(router.recorded_topk_idx.to(torch.int16))  # dynamic_bs, topk

        # layer_num, dynamic_bs, topk  -> dynamic_bs, layer_num, topk
        layers_topk_idx = torch.stack(layers_topk_idx).permute(1, 0, 2).to(device_name)
        layers_topk_idx = layers_topk_idx.contiguous().view(torch.uint8)
        # dynamic_bs, layer_num, topk -> 1, dynamic_bs_all, layer_num, topk
        layers_topk_idx = (
            gather_from_sequence_parallel_region(layers_topk_idx, tensor_parallel_output_grad=False)
            .unsqueeze(0)
            .contiguous()
        )

        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        if input_ids.is_nested:
            batch_size = input_ids.shape[0]
            _, packed_seq_params, _ = preprocess_thd_engine(
                input_ids,
                pre_process=True,
                use_fp8_padding=use_fp8_padding,
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            layers_topk_idx = postprocess_thd_engine(
                layers_topk_idx,
                packed_seq_params,
                input_ids,
                batch_size,
                post_process=True,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            # Undo the uint8 view. Nested tensors have no view(dtype), so reinterpret
            # the packed values buffer and re-wrap it on the same offsets.
            layers_topk_idx = torch.nested.nested_tensor_from_jagged(
                layers_topk_idx.values().contiguous().view(torch.int16), offsets=layers_topk_idx.offsets()
            )
        else:
            batch_size, seq_len = attention_mask.shape[:2]
            _, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
            )
            layers_topk_idx = postprocess_packed_seqs(
                layers_topk_idx, packed_seq_params, attention_mask, batch_size, seq_len, post_process=True
            )
            layers_topk_idx = layers_topk_idx.view(torch.int16)
        mini_layer_topk_idx_list.append(layers_topk_idx.cpu())


def build_r3_replay_mask(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: for each jagged sequence of total length L with a non-empty response, mark the first
    # L-1 rows True and the last row False; sequences without response tokens get all False. Returns a
    # jagged bool tensor with the same offsets as input_ids.
    # Inputs: input_ids - jagged nested tensor; response_mask (batch, resp_len).
    # Outputs: jagged bool mask.
    """Build a full-sequence replay mask for rollout-captured routes.

    Response logprobs are read from shifted model rows, but those rows attend to
    earlier hidden states whose MoE routing also affects the result. Replay every
    causal row that can influence response logprobs and skip only the final
    response row, whose logits are not used by ``no_padding_2_padding``.
    """
    total_lens = input_ids.offsets().diff()
    response_lens = response_mask.sum(dim=-1).to(device=total_lens.device, dtype=total_lens.dtype)
    bs = total_lens.size(0)
    values = torch.tensor([True, False], dtype=torch.bool, device=total_lens.device).repeat(bs)
    replay_lens = torch.where(response_lens > 0, total_lens - 1, torch.zeros_like(total_lens))
    suffix_lens = total_lens - replay_lens
    counts = torch.stack([replay_lens, suffix_lens], dim=1).flatten()
    mask_values = torch.repeat_interleave(values, counts)
    return torch.nested.nested_tensor_from_jagged(mask_values, offsets=input_ids.offsets())


def align_r3_router_replay_data(layers_topk_idx: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: per sequence, keeps routes whose length equals the input length, or appends a single
    # zero placeholder row when routes are exactly one shorter; returns a jagged tensor.
    # Inputs: layers_topk_idx - jagged (seq_i, layers, topk); input_ids - jagged.
    # Outputs: aligned jagged route tensor.
    # Exceptions: TypeError if either is not nested; RuntimeError on sequence-count mismatch or any
    # other length difference.
    """Align rollout routes with full training inputs one sequence at a time.

    Autoregressive rollout records the routes used to predict each generated
    token, so it has no route for the final generated token. The R3 replay mask
    leaves that final model row native; append one ignored placeholder per
    affected sequence so route and model tensors still have identical shapes.
    """
    if not layers_topk_idx.is_nested or not input_ids.is_nested:
        raise TypeError("R3 router replay requires jagged route targets and input_ids")

    route_parts = list(layers_topk_idx.unbind())
    input_lens = [int(length) for length in input_ids.offsets().diff().tolist()]
    if len(route_parts) != len(input_lens):
        raise RuntimeError(
            f"R3 router replay has {len(route_parts)} route sequences for {len(input_lens)} input sequences"
        )

    aligned_parts = []
    for sample_id, (routes, input_len) in enumerate(zip(route_parts, input_lens, strict=True)):
        route_len = routes.shape[0]
        if route_len == input_len:
            aligned_parts.append(routes)
        elif route_len == input_len - 1:
            placeholder = torch.zeros((1, *routes.shape[1:]), dtype=routes.dtype, device=routes.device)
            aligned_parts.append(torch.cat((routes, placeholder), dim=0))
        else:
            raise RuntimeError(
                f"R3 router replay sample {sample_id} has {route_len} route rows for {input_len} input tokens; "
                "expected equal lengths or exactly one missing final route"
            )

    return torch.nested.as_nested_tensor(aligned_parts, layout=torch.jagged)


def set_router_replay_data(
    layers_topk_idx,
    attention_mask,
    tf_config,
    vp_rank=None,
    replay_mask=None,
    local_cp_size=None,
    model=None,
):
    # [FUNCTION DOC]
    # Behavior (no grad): errors if layers_topk_idx is None. Packs routes (and optional replay mask)
    # to rmpad layout: nested via preprocess_thd_engine (same fp8/CP/min-rows options as merge),
    # padded via preprocess_packed_seqs (requires attention_mask; mask is ignored in this path). Moves
    # to device and scatters along tokens to the local sequence-parallel shard; reshapes to
    # (layers, local_tokens, topk). If the layer dim equals num_layers index by absolute layer,
    # otherwise by MoE ordinal. With model: for each (layer_number, router) of the model, set int64
    # target indices (+ mask) for its index; error listing layers out of range. Without model: walk the
    # local layer range from get_current_rank_layer_info, assigning to registry routers in order for
    # each MoE layer.
    # Inputs: layers_topk_idx (batch, seq, layers, topk) or jagged; attention_mask; tf_config;
    # vp_rank (None -> 0); replay_mask - optional per-token mask; local_cp_size; model.
    # Outputs: none.
    # Exceptions: RuntimeError on missing data/mask/routers/coverage; IndexError if registry is short.
    # Side effects: mutates RouterReplay instances (targets and backward queues); collectives.
    # KNOWN ISSUE: replay_mask is silently ignored for non-nested (padded) inputs.
    """
    Scatter the packed router top-k indices back to sequence-parallel ranks and update each local
    RouterReplay instance with target indices for replay mode.

    This function prepares the per-layer, per-sample top-k routing decisions (recorded during an earlier
    forward) so that subsequent replay passes can follow exactly the same routing.

    Args:
        layers_topk_idx (torch.Tensor): Router top-k indices with shape [bs, max_seq_len, layer_num, topk].
            This should be the merged output produced by merge_router_topk_indices.
        attention_mask (torch.Tensor): Attention mask [batch_size, seq_len] used for pack/unpack alignment.
        tf_config: Megatron/Transformer engine configuration object.
        vp_rank (Optional[int]): Virtual pipeline stage rank override. If None, the current VP rank from
            Megatron parallel state will be used.
        replay_mask (Optional[torch.Tensor]): Optional per-token mask. Masked tokens use replayed routes;
            unmasked tokens keep native Megatron routes.
        model: The forwarded model (or list of VPP chunks). When given, targets are written to its own
            routers instead of a positional slice of the global RouterReplay.router_instances list.

    Returns:
        None: The function updates internal RouterReplay instances in-place.
    """
    if layers_topk_idx is None:
        raise RuntimeError("router_replay REPLAY requires routed_experts from the preceding RECORD forward.")

    with torch.no_grad():
        vp_rank = 0 if vp_rank is None else vp_rank
        fp8 = tf_config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        cp_layout = _context_parallel_layout(tf_config)
        min_local_rows = (
            tf_config.csa_window_size
            if getattr(tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
            else None
        )

        replay_mask_rmpad = None
        if layers_topk_idx.is_nested:
            layers_topk_idx_rmpad, _, _ = preprocess_thd_engine(
                layers_topk_idx,
                pre_process=True,
                use_fp8_padding=use_fp8_padding,
                min_local_rows=min_local_rows,
                local_cp_size=local_cp_size,
                cp_layout=cp_layout,
            )
            if replay_mask is not None:
                replay_mask_rmpad, _, _ = preprocess_thd_engine(
                    replay_mask,
                    pre_process=True,
                    use_fp8_padding=use_fp8_padding,
                    min_local_rows=min_local_rows,
                    local_cp_size=local_cp_size,
                    cp_layout=cp_layout,
                )
        else:
            if attention_mask is None:
                raise RuntimeError("router replay REPLAY requires attention_mask for non-nested BSHD inputs.")
            layers_topk_idx_rmpad, _ = preprocess_packed_seqs(
                layers_topk_idx, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding
            )
        layers_topk_idx_rmpad = layers_topk_idx_rmpad.contiguous()  # 1, dynamic_bs_all, layer_num, topk

        # 1, dynamic_bs_split, layer_num, topk
        layers_topk_idx_rmpad_split = scatter_to_sequence_parallel_region(
            layers_topk_idx_rmpad.to(device_name).squeeze(dim=0)
        ).unsqueeze(dim=0)
        replay_mask_rmpad_split = None
        if replay_mask_rmpad is not None:
            replay_mask_rmpad_split = scatter_to_sequence_parallel_region(
                replay_mask_rmpad.to(device_name).squeeze(dim=0)
            )

        # dynamic_bs_split, layer_num, topk -> layer_num, dynamic_bs_split, topk
        layers_topk_idx_reshape = layers_topk_idx_rmpad_split.permute(0, 2, 1, 3).squeeze(
            dim=0
        )  # layer_num, dynamic_bs_all, topk
        # When dim-0 covers all layers (e.g. R3, or R2 with all-MoE models),
        # index by absolute layer_idx; otherwise (R2 with mixed dense/MoE),
        # dim-0 only contains MoE layers, index by MoE-layer ordinal.
        index_by_layer = len(layers_topk_idx_reshape) == tf_config.num_layers

        if model is not None:
            model_routers = list(iter_model_routers(model))
            if not model_routers:
                raise RuntimeError("router_replay REPLAY found no routers in the forwarded model.")

            missing_layer_numbers = []
            for layer_number, router in model_routers:
                layer_idx = layer_number - 1
                idx = layer_idx if index_by_layer else sum(1 for i in range(layer_idx) if is_moe_layer(tf_config, i))
                if not 0 <= idx < layers_topk_idx_reshape.shape[0]:
                    missing_layer_numbers.append(layer_number)
                    continue
                router.set_target_indices(
                    layers_topk_idx_reshape[idx].to(torch.int64),
                    replay_mask=replay_mask_rmpad_split,
                )
            if missing_layer_numbers:
                raise RuntimeError(
                    "router_replay REPLAY data does not cover every forwarded MoE layer: "
                    f"missing_layer_numbers={missing_layer_numbers}, route_layers={layers_topk_idx_reshape.shape[0]}, "
                    f"vp_rank={vp_rank}"
                )
            return

        local_rank_info = get_current_rank_layer_info(tf_config, vp_rank)
        offset, end = local_rank_info["start"], local_rank_info["end"]
        router_instances_list = RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

        # For R2: count MoE layers before `offset` as the starting position.
        moe_idx = sum(1 for i in range(offset) if is_moe_layer(tf_config, i))

        router_offset = 0
        for layer_idx in range(offset, end):
            if not is_moe_layer(tf_config, layer_idx):
                continue
            router = router_instances_list[router_offset]
            idx = layer_idx if index_by_layer else moe_idx
            router.set_target_indices(
                layers_topk_idx_reshape[idx].to(torch.int64),
                replay_mask=replay_mask_rmpad_split,
            )
            router_offset += 1
            moe_idx += 1


def iter_model_routers(model):
    # [FUNCTION DOC]
    # Behavior: generator over model chunks (list/tuple or single module); for each module in the
    # chunk's decoder (or the chunk itself), yields (layer_number, router_replay) for TopKRouter
    # modules having a non-None router_replay and layer_number. MTP layers are excluded by scoping.
    # Inputs: model. Outputs: iterator of (int, RouterReplay).
    """Yield ``(layer_number, RouterReplay)`` for every replay-enabled router owned by ``model``.

    ``model`` is a single module or the list of virtual-pipeline chunks. The process-global
    ``RouterReplay.router_instances`` list can hold more routers than this model has local layers —
    a model that rebuilds its decoder registers a set it then discards — so a positional slice of it
    can address the wrong objects. Walking the forwarded model reaches each layer's own router.
    """
    for chunk in model if isinstance(model, list | tuple) else [model]:
        # Scope to the decoder: MTP layers restart layer numbering at 1, so they alias decoder
        # layers, and the replay tensor carries no MTP rows. ``mtp`` is a sibling of ``decoder``
        # on both GPTModel and HybridModel, the only two classes that build one.
        for module in getattr(chunk, "decoder", chunk).modules():
            router = getattr(module, "router_replay", None)
            if isinstance(module, TopKRouter) and router is not None and module.layer_number is not None:
                yield module.layer_number, router


def set_model_router_replay_action(model, router_replay_action):
    # [FUNCTION DOC]
    # Behavior: sets the given action on every router yielded by iter_model_routers(model).
    # Side effects: mutates router state.
    """Set the router replay action on the forwarded model's own routers.

    Mirrors the module walk in ``set_router_replay_data``. Toggling through the positional slice can
    miss the real routers, leaving them stuck in REPLAY_FORWARD so the backward recompute reads a
    stale cross-micro-batch ``target_topk_idx`` instead of popping its own entry from
    ``replay_backward_list``.
    """
    for _, router in iter_model_routers(model):
        router.set_router_replay_action(router_replay_action)


def reorder_and_merge_vpp_layers(
    micro_batch_tensor_list,
    num_microbatches: int,
    vpp_size: int,
    microbatch_group_size_per_vp_stage: int,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: builds Megatron's VPP schedule table (virtual microbatch -> (microbatch, chunk)),
    # requires one captured tensor per schedule entry, groups captured tensors by chunk id in schedule
    # order, concatenates each chunk's tensors along the batch dim (or merges jagged maps), then
    # concatenates chunks along the layer dim (dim 2).
    # Inputs: micro_batch_tensor_list - list of route maps in capture order; num_microbatches;
    # vpp_size; microbatch_group_size_per_vp_stage.
    # Outputs: (batch, seq, layers, topk) tensor (or jagged equivalent).
    # Exceptions: RuntimeError when list length != schedule length.
    # KNOWN ISSUE: the docstring describes a single input tensor and reshape; the code takes a list and
    # concatenates, and does not raise ValueError as documented.
    """
    Reorder and merge per-VPP layer blocks into a contiguous layer dimension.

    Given a tensor shaped as [bs*vpp_size, max_token_len, layer_num_per_vpp, topk], this function:
    1) Builds the schedule table for virtual microbatches and reorders the first dimension so that entries
       belonging to the same model chunk (VPP stage) become contiguous.
    2) Reshapes and merges the (vpp_size, layer_num_per_vpp) into a single layer dimension, producing
       [bs, max_token_len, layer_num, topk].

    Args:
        micro_batch_tensor_list : the list of Input tensor.
        num_microbatches (int): Number of microbatches per pipeline stage (bs).
        vpp_size (int): Virtual pipeline parallel size (number of model chunks).
        microbatch_group_size_per_vp_stage (int): Number of consecutive microbatches processed per VPP stage.

    Returns:
        torch.Tensor: Output tensor of shape [bs, max_token_len, layer_num, topk].

    Raises:
        ValueError: If input tensor dimensionality or expected sizes do not match.
        RuntimeError: If the computed output shape is unexpected or the schedule length mismatches.
    """
    # 1) Build schedule table: map each virtual_microbatch_id -> (microbatch_id, model_chunk_id)
    schedule_table = get_schedule_table(num_microbatches, vpp_size, microbatch_group_size_per_vp_stage)
    if len(micro_batch_tensor_list) != len(schedule_table):
        raise RuntimeError(
            "R2 RECORD must capture every scheduled microbatch/chunk, including zero-layer chunks: "
            f"captured={len(micro_batch_tensor_list)}, expected={len(schedule_table)}"
        )

    # 2) Group by model_chunk_id to build reorder indices so entries of the same chunk become contiguous along dim 0
    tensor_by_chunk = [[] for _ in range(vpp_size)]
    mini_tensor_list = []

    for vidx, (_mb, chunk_id) in enumerate(schedule_table):
        tensor_by_chunk[chunk_id].append(micro_batch_tensor_list[vidx])

    if micro_batch_tensor_list[0].is_nested:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(merge_nested_router_maps(tensor_by_chunk[chunk_id]))
    else:
        for chunk_id in range(vpp_size):
            mini_tensor_list.append(torch.cat(tensor_by_chunk[chunk_id], dim=0))

    out = torch.cat(mini_tensor_list, dim=2)

    return out


def merge_nested_router_maps(router_maps) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Unbinds each jagged route map into per-sequence tensors, casts to int16 and rebuilds a
    # single jagged tensor (sequence order preserved).
    """Flatten recorded micro-batch routes into one int16 jagged tensor."""
    tensors = [tensor.to(torch.int16) for router_map in router_maps for tensor in router_map.unbind()]
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def get_current_rank_layer_info(tf_config, vp_rank=None):
    # [FUNCTION DOC]
    # Behavior: vp_rank None -> 0; computes the layer count for this stage and the global layer offset
    # (passing vp_stage if supported). Returns a dict with "start", "end" (exclusive) and "count".
    # Inputs: tf_config; vp_rank. Outputs: dict.
    # KNOWN ISSUE: docstring says it returns a tuple (local, all_assignments) and uses the current VP
    # rank when None; the code returns only the local dict and defaults to 0.
    # When vp_rank is None, default to the current VP rank (or 0 if VP is disabled).
    """Return the local layer range/count for the current process and the full assignment table.

    Args:
        tf_config: Configuration object used by compute_pipeline_layer_assignment.
        vp_rank (Optional[int]): Explicit virtual pipeline stage rank to query. If None, uses
            mpu.get_virtual_pipeline_model_parallel_rank() when VP is enabled; otherwise 0.

    Returns:
        Tuple[dict, dict]: A tuple of (local_assignment, all_assignments) where local_assignment contains
        keys {"start", "end", "count"} for the current (pp_rank, vp_stage).
    """
    if vp_rank is None:
        vp_rank = 0
    num_layers_to_build = get_num_layers_to_build(tf_config, vp_stage=vp_rank)

    sig = inspect.signature(get_transformer_layer_offset)

    if "vp_stage" in sig.parameters:
        offset = get_transformer_layer_offset(tf_config, vp_stage=vp_rank)
    else:
        offset = get_transformer_layer_offset(tf_config)
    local = {}
    local["start"] = offset
    local["end"] = offset + num_layers_to_build
    local["count"] = num_layers_to_build
    return local


def pp_gather(local_layers_router_map, tf_config):
    # [FUNCTION DOC]
    # Behavior: returns the input when PP size <= 1. Otherwise checks PP group size; for jagged maps,
    # moves to CPU and all_gather_object; for dense maps, views as uint8, all-gathers per-rank layer
    # counts, pads the layer dim to the max count, all-gathers payloads and trims/reinterprets each as
    # int16 (if every count is 0, uses the local map for all ranks). With VPP, reassembles layers in
    # global order: for each vp stage then each pp stage, slices that rank's MoE layers for the chunk
    # (offsets from get_moe_num_layers_to_build) and concatenates along layer dim; without VPP
    # concatenates PP ranks in order. Result moved to CPU.
    # Inputs: local_layers_router_map (batch, seq, local_layers, topk) int16 or jagged; tf_config.
    # Outputs: global router map (batch, seq, moe_layers, topk) on CPU.
    # Exceptions: RuntimeError on PP group size mismatch. Side effects: PP collectives.
    # KNOWN ISSUE: the pp_size <= 1 early return does not move the map to CPU, unlike the docstring.
    """
    Gather local router maps from all PP ranks into a global router map.

    Args:
        local_layers_router_map (torch.Tensor): Local router map of shape
            [bs, max_seq_len, local_num_layers, topk].
        tf_config: Configuration providing pipeline_model_parallel_size.

    Returns:
        torch.Tensor: Global router map of shape [bs, max_seq_len, num_layers, topk] placed on CPU.
    """
    pp_size = tf_config.pipeline_model_parallel_size
    if pp_size <= 1:
        return local_layers_router_map

    pp_group = mpu.get_pipeline_model_parallel_group()
    world_size = torch.distributed.get_world_size(pp_group)
    if world_size != pp_size:
        raise RuntimeError(f"pipeline group size {world_size} does not match configured PP size {pp_size}.")
    if local_layers_router_map.is_nested:
        # all_gather_object preserves each sender's CUDA device, so normalize
        # jagged route maps before concatenating routes from local PP ranks.
        local_layers_router_map = local_layers_router_map.to("cpu")
        layers_topk_idx_global_list = [None] * world_size
        torch.distributed.all_gather_object(layers_topk_idx_global_list, local_layers_router_map, pp_group)
    else:
        # PP stages may own different numbers of MoE layers, including zero. Exchange
        # layer counts, pad only that dimension for NCCL, then remove the padding.
        # NCCL has no int16 datatype, so payloads travel as bit-identical uint8 views.
        payload = local_layers_router_map.to(device_name).contiguous().view(torch.uint8)
        local_layer_count = torch.tensor([local_layers_router_map.shape[2]], dtype=torch.int64, device=payload.device)
        gathered_layer_counts = [torch.empty_like(local_layer_count) for _ in range(world_size)]
        torch.distributed.all_gather(
            tensor=local_layer_count,
            tensor_list=gathered_layer_counts,
            group=pp_group,
            async_op=False,
        )
        layer_counts = [int(count.item()) for count in gathered_layer_counts]
        max_layer_count = max(layer_counts)
        if max_layer_count == 0:
            layers_topk_idx_global_list = [local_layers_router_map for _ in range(world_size)]
        else:
            if payload.shape[2] != max_layer_count:
                padded_shape = list(payload.shape)
                padded_shape[2] = max_layer_count
                padded_payload = torch.zeros(padded_shape, dtype=payload.dtype, device=payload.device)
                padded_payload[:, :, : payload.shape[2], :].copy_(payload)
                payload = padded_payload
            gathered_payloads = [torch.empty_like(payload) for _ in range(world_size)]
            torch.distributed.all_gather(
                tensor=payload,
                tensor_list=gathered_payloads,
                group=pp_group,
                async_op=False,
            )
            layers_topk_idx_global_list = [
                tensor[:, :, :layer_count, :].contiguous().view(torch.int16)
                for tensor, layer_count in zip(gathered_payloads, layer_counts, strict=True)
            ]
    vp_size = tf_config.virtual_pipeline_model_parallel_size
    if vp_size is not None:
        vpp_router_map_offset = [[] for _ in range(pp_size)]
        for pp_stage in range(pp_size):
            vpp_router_map_offset[pp_stage].append(0)
            for vp_stage in range(vp_size):
                num_layers_to_build = get_moe_num_layers_to_build(tf_config, vp_stage, pp_stage)
                vpp_router_map_offset[pp_stage].append(num_layers_to_build + vpp_router_map_offset[pp_stage][-1])
        layers_topk_idx_global = []
        for vp_stage in range(vp_size):
            for pp_stage in range(pp_size):
                piece = slice(vpp_router_map_offset[pp_stage][vp_stage], vpp_router_map_offset[pp_stage][vp_stage + 1])
                if layers_topk_idx_global_list[pp_stage].is_nested:
                    nested_item = layers_topk_idx_global_list[pp_stage]
                    sliced_tensors = [t[:, piece, :] for t in nested_item.unbind()]
                    sliced_nested = torch.nested.as_nested_tensor(sliced_tensors, layout=torch.jagged)
                    layers_topk_idx_global.append(sliced_nested)
                else:
                    layers_topk_idx_global.append(layers_topk_idx_global_list[pp_stage][:, :, piece, :])
        global_router_map = torch.cat(layers_topk_idx_global, dim=2).to("cpu")
    else:
        global_router_map = torch.cat(layers_topk_idx_global_list, dim=2).to("cpu")

    return global_router_map


class RouterReplayHelper:
    # [CLASS DOC]
    # Static helpers locating local RouterReplay instances and querying their current action.
    """Helper class to query router replay state and locate local RouterReplay instances."""

    @staticmethod
    def get_micro_batch_router_list(tf_config, vp_rank=None):
        # [FUNCTION DOC]
        # Behavior: empty list if no registry. Validates vp_rank (None -> 0) against VPP size; computes
        # MoE-layer counts per local VPP stage and the global MoE-layer count. If the registry size equals
        # the local total, offset = sum of counts of earlier local stages; if it equals the global total,
        # offset = number of MoE layers before this stage's start layer; else error. Returns the registry
        # slice of length local_counts[vp_rank] at that offset.
        # Inputs: tf_config; vp_rank. Outputs: list of RouterReplay.
        # Exceptions: ValueError for out-of-range vp_rank; RuntimeError for unrecognized registry size.
        """
        Return the list of RouterReplay instances corresponding to the current micro-batch and local
        (pp_rank, vp_stage) layer range.

        Accept either one PP rank's complete registry in local VPP order, or one complete
        model's registry in global layer order. These positional layouts must not contain
        stale routers or routers from other models; use the model-scoped helpers for those cases.

        Args:
            tf_config: Configuration object used to compute layer assignments.
            vp_rank (Optional[int]): Explicit virtual pipeline stage to query. If None, the current VP
                rank from Megatron parallel state is used when available.
        Returns:
            list: A contiguous sublist of RouterReplay.router_instances for the local layer range.
        """
        if not RouterReplay.router_instances:
            return []

        vp_size = tf_config.virtual_pipeline_model_parallel_size or 1
        vp_rank = 0 if vp_rank is None else vp_rank
        if not 0 <= vp_rank < vp_size:
            raise ValueError(f"vp_rank={vp_rank} is outside the configured VPP size {vp_size}.")

        local_counts = [get_moe_num_layers_to_build(tf_config, stage) for stage in range(vp_size)]
        local_total = sum(local_counts)
        global_total = sum(is_moe_layer(tf_config, idx) for idx in range(tf_config.num_layers))
        registry_size = len(RouterReplay.router_instances)
        if registry_size == local_total:
            offset = sum(local_counts[:vp_rank])
        elif registry_size == global_total:
            layer_start = get_current_rank_layer_info(tf_config, vp_rank)["start"]
            offset = sum(is_moe_layer(tf_config, idx) for idx in range(layer_start))
        else:
            raise RuntimeError(
                "router replay registry does not cover a complete local PP/VPP or global model layout: "
                f"registry={registry_size}, local_expected={local_total}, global_expected={global_total}, "
                f"vp_rank={vp_rank}. Use model-scoped replay for non-positional registries."
            )
        return RouterReplay.router_instances[offset : offset + local_counts[vp_rank]]

    @staticmethod
    def _get_action_router_list(tf_config, vp_rank=None, model=None):
        # [FUNCTION DOC]
        # HELPER: Routers of the given model if provided, else get_micro_batch_router_list.
        if model is not None:
            return [router for _, router in iter_model_routers(model)]
        return RouterReplayHelper.get_micro_batch_router_list(tf_config, vp_rank)

    @staticmethod
    def is_r2_record_action(tf_config, vp_rank=None, model=None) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if routers exist and the first one's action is RECORD.
        """Return True if the current router_replay_action is RECORD (R2) for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.RECORD.
        """
        router_instances_list = RouterReplayHelper._get_action_router_list(tf_config, vp_rank, model)
        return bool(router_instances_list) and (
            router_instances_list[0].router_replay_action == RouterReplayAction.RECORD
        )

    @staticmethod
    def is_replay_forward_action(tf_config, vp_rank=None, model=None) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if routers exist and the first one's action is REPLAY_FORWARD.
        """Return True if the current router_replay_action is REPLAY_FORWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_FORWARD.
        """
        router_instances_list = RouterReplayHelper._get_action_router_list(tf_config, vp_rank, model)
        return (
            bool(router_instances_list)
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_FORWARD
        )

    @staticmethod
    def is_replay_backward_action(tf_config, vp_rank=None, model=None) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if routers exist and the first one's action is REPLAY_BACKWARD.
        """Return True if the current router_replay_action is REPLAY_BACKWARD for the local router instances.

        This inspects the first local RouterReplay instance's router_replay_action and compares it to
        RouterReplayAction.REPLAY_BACKWARD.
        """
        router_instances_list = RouterReplayHelper._get_action_router_list(tf_config, vp_rank, model)
        return (
            bool(router_instances_list)
            and router_instances_list[0].router_replay_action == RouterReplayAction.REPLAY_BACKWARD
        )
