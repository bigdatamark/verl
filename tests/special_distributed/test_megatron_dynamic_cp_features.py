# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# 4-rank CUDA pytest for Megatron dynamic context parallelism features: ranks 0-1 process one sequence with CP=2 while
# ranks 2-3 each process their own sequence with CP=1. Covers router replay recording/replay and MTP label rolling and
# backward under per-micro-batch CP groups. Applies Ascend patches on NPU before importing Megatron; skipped without
# CUDA or a Megatron-Core lacking dynamic_context_parallel.

"""Exercise CP2 on ranks 0-1 and independent CP1 micro-batches on ranks 2-3."""

from inspect import signature
from types import SimpleNamespace

import pytest
import torch
import torch.distributed
import torch.nn.functional as F

from verl.utils.device import is_npu_available

if is_npu_available:
    # Apply Ascend patches before importing Megatron-Core.
    import megatron_adaptor  # noqa: F401

pytest.importorskip("megatron.core")

from megatron.core import parallel_state  # noqa: E402
from megatron.core.transformer.multi_token_prediction import roll_tensor  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402

from verl.models.mcore.mtp_patch import (  # noqa: E402
    _megatron_gptmodel_postprocess,
    _patched_get_embeddings_for_detach,
)
from verl.models.mcore.util import preprocess_thd_engine  # noqa: E402
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group  # noqa: E402
from verl.utils.megatron.router_replay_patch import (  # noqa: E402
    RouterReplay,
    RouterReplayAction,
    _patched_topk_routing_with_score_function,
)
from verl.utils.megatron.router_replay_utils import merge_router_topk_indices, set_router_replay_data  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Dynamic CP feature coverage requires CUDA"),
    pytest.mark.skipif(
        "dynamic_context_parallel" not in signature(parallel_state.initialize_model_parallel).parameters,
        reason="Installed Megatron-Core does not support dynamic context parallel initialization",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    # [FUNCTION DOC]
    # HELPER: Module-scoped autouse fixture: initializes the global process group (asserting world size 4) and Megatron
    # model parallel with CP=2 and dynamic CP; on teardown barriers, destroys model parallel and the process group.
    _, _, world_size = initialize_global_process_group()
    assert world_size == 4
    parallel_state.initialize_model_parallel(context_parallel_size=2, dynamic_context_parallel=True)
    yield
    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    destroy_global_process_group()


def _sample():
    # [FUNCTION DOC]
    # HELPER: Returns a one-sequence jagged nested tensor of CUDA tokens and its local CP size: ranks 0-1 get tokens 1..8 with
    # CP size 2; rank r >= 2 gets 4 tokens starting at 1 + 8r with CP size 1.
    rank = torch.distributed.get_rank()
    if rank < 2:
        tokens, cp_size = torch.arange(1, 9, device="cuda"), 2
    else:
        tokens, cp_size = torch.arange(1 + rank * 8, 5 + rank * 8, device="cuda"), 1
    return torch.nested.as_nested_tensor([tokens], layout=torch.jagged), cp_size


def _route(logits, router):
    # [FUNCTION DOC]
    # HELPER: Calls the patched top-k routing (top-2, softmax scores, no groups/bias/scaling, unfused) with the given router
    # replay object. Outputs: the patched function's (probs, routing_map) result.
    return _patched_topk_routing_with_score_function(
        logits=logits,
        topk=2,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=router,
        scaling_factor=None,
    )


def test_router_replay_uses_each_microbatch_cp_group():
    # [FUNCTION DOC]
    # Behavior: preprocesses the sample to its CP-local THD ids, builds 4-expert logits from the tokens, records top-k with a
    # fresh RouterReplay (clearing the global instance registry), then merges router indices for the full sample and sets
    # replay data with the local CP size and asserts the router's target indices equal the recorded ones exactly; then
    # replays forward with different (negated, flipped) logits and asserts the routing map still selects the recorded
    # experts. Clears the global router registry at the end.
    input_ids, cp_size = _sample()
    local_ids, _, _ = preprocess_thd_engine(input_ids, local_cp_size=cp_size)
    tokens = local_ids.flatten().float()
    logits = torch.stack((tokens, -tokens, tokens.remainder(3), tokens.remainder(5)), dim=-1)

    RouterReplay.router_instances.clear()
    router = RouterReplay()
    router.set_router_replay_action(RouterReplayAction.RECORD)
    _route(logits, router)
    recorded = router.recorded_topk_idx.clone()

    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        num_moe_experts=4,
        moe_router_topk=2,
        context_parallel_size=2,
        use_cpu_initialization=True,
    )
    routes = []
    merge_router_topk_indices(None, input_ids, routes, config, local_cp_size=cp_size)
    set_router_replay_data(routes[0], None, config, local_cp_size=cp_size)
    torch.testing.assert_close(router.target_topk_idx, recorded, atol=0, rtol=0)

    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, routing_map = _route(-logits.flip(-1), router)
    assert routing_map.gather(1, recorded).all()
    RouterReplay.router_instances.clear()


class _OutputLayer(torch.nn.Module):
    # [CLASS DOC]
    # HELPER: Minimal stand-in for a Megatron output layer with a random (vocab x hidden) CUDA weight scaled by 0.02.
    def __init__(self, hidden_size, vocab_size):
        # [FUNCTION DOC]
        # HELPER: Registers the weight parameter. Inputs: hidden_size, vocab_size.
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size, device="cuda") * 0.02)

    def forward(self, hidden_states, weight=None, runtime_gather_output=None):
        # [FUNCTION DOC]
        # HELPER: Returns (linear projection of hidden_states with the given weight or its own, None); runtime_gather_output ignored.
        return F.linear(hidden_states, self.weight if weight is None else weight), None


def test_mtp_roll_and_backward_use_each_microbatch_cp_group():
    # [FUNCTION DOC]
    # Behavior: preprocesses the sample, computes expected rolled labels and rolled padding mask (fill True) with Megatron
    # roll_tensor on the micro-batch CP group; calls the patched MTP embedding helper with a stub layer and identity-like
    # embedding and asserts its rolled ids/padding match exactly. Then builds a stub GPT model (no shared embeddings, MTP
    # 1 layer, per-token loss, native CE) with a loss function capturing labels, runs the patched postprocess on hidden states
    # of twice the local length (main + MTP), asserts the first captured labels equal the expected rolled labels, backprops
    # the summed output and asserts gradients are finite and non-zero in the MTP half.
    input_ids, cp_size = _sample()
    local_ids, packed, position_ids = preprocess_thd_engine(input_ids, local_cp_size=cp_size)
    expected_labels, _ = roll_tensor(local_ids, cp_group=packed.cp_group, packed_seq_params=packed)
    padding_mask = torch.zeros_like(local_ids, dtype=torch.bool)
    expected_padding, _ = roll_tensor(
        padding_mask,
        cp_group=packed.cp_group,
        packed_seq_params=packed,
        fill_value=True,
    )

    layer = SimpleNamespace(cp_group=parallel_state.get_context_parallel_group(), _get_embeddings_has_padding_mask=True)
    rolled_ids, _, rolled_padding, _, _ = _patched_get_embeddings_for_detach(
        layer,
        input_ids=local_ids,
        position_ids=position_ids,
        padding_mask=padding_mask,
        embedding=lambda input_ids, position_ids: input_ids.float().unsqueeze(-1),
        hidden_states=torch.randn(local_ids.shape[-1], 1, 4, device="cuda"),
        packed_seq_params=packed,
    )
    torch.testing.assert_close(rolled_ids, expected_labels, atol=0, rtol=0)
    torch.testing.assert_close(rolled_padding, expected_padding, atol=0, rtol=0)

    local_tokens, hidden_size, vocab_size = local_ids.shape[-1], 8, 64
    hidden_states = torch.randn(2 * local_tokens, 1, hidden_size, device="cuda", requires_grad=True)
    captured_labels = []

    def language_model_loss(labels, logits):
        # [FUNCTION DOC]
        # HELPER: Records detached labels into the enclosing list and returns per-token cross entropy of logits (seq-first,
        # transposed to batch-first and flattened over vocab) against labels, reshaped to the labels' shape.
        captured_labels.append(labels.detach())
        return F.cross_entropy(
            logits.transpose(0, 1).reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    model = SimpleNamespace(
        share_embeddings_and_output_weights=False,
        post_process=True,
        training=False,
        output_layer=_OutputLayer(hidden_size, vocab_size),
        compute_language_model_loss=language_model_loss,
        cp_group=parallel_state.get_context_parallel_group(),
        pg_collection=None,
        config=SimpleNamespace(
            mtp_num_layers=1,
            mtp_loss_scaling_factor=0.1,
            mtp_detach_heads=False,
            calculate_per_token_loss=True,
            cross_entropy_loss_fusion=False,
            cross_entropy_fusion_impl="native",
            use_mup=False,
        ),
    )
    output = _megatron_gptmodel_postprocess(
        model,
        hidden_states=hidden_states,
        input_ids=local_ids,
        position_ids=position_ids,
        labels=local_ids,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        loss_mask=torch.ones_like(local_ids, dtype=torch.float32),
        packed_seq_params=packed,
    )

    torch.testing.assert_close(captured_labels[0], expected_labels, atol=0, rtol=0)
    output.float().sum().backward()
    assert torch.isfinite(hidden_states.grad).all()
    assert hidden_states.grad[local_tokens:].abs().sum() > 0
