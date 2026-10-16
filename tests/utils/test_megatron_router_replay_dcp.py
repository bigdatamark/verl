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
# Tests (skipped without megatron) for router replay utilities under dynamic context parallelism and pipeline
# parallelism: recording/replaying top-k routes with a dynamic local CP size, R3 alignment and replay masks,
# attaching routes to DCP outputs, pipeline-parallel gathering of routes, and VPP chunk merging.

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import verl.utils.megatron.router_replay_utils as router_utils
from verl.utils.dynamic_cp_scheduler import DCP_SAMPLE_IDS
from verl.workers.engine.megatron.transformer_impl import _attach_dcp_recorded_routes


def _nested(parts):
    # [FUNCTION DOC]
    # HELPER: Builds a jagged nested tensor from a list of tensors.
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def test_router_record_and_replay_use_dynamic_cp_size(monkeypatch):
    # [FUNCTION DOC]
    # Patches router helpers so a single fake router has recorded indices, collectives are identity (recording
    # gathered dtypes), and preprocess/postprocess record the local_cp_size kwarg. Calls merge_router_topk_indices
    # with local_cp_size 2 and asserts the gather saw a uint8 view (int16 has no NCCL type) while the recorded route
    # is int16; then calls set_router_replay_data and asserts the router's target indices equal the original and that
    # local_cp_size 2 was passed to all three pre/postprocess calls.
    calls = []
    gathered_dtypes = []
    router = SimpleNamespace(recorded_topk_idx=torch.tensor([[1], [2]]))
    monkeypatch.setattr(router_utils.RouterReplayHelper, "get_micro_batch_router_list", lambda *_args: [router])
    monkeypatch.setattr(
        router_utils,
        "gather_from_sequence_parallel_region",
        lambda tensor, **_kwargs: gathered_dtypes.append(tensor.dtype) or tensor,
    )
    monkeypatch.setattr(router_utils, "scatter_to_sequence_parallel_region", lambda tensor: tensor)
    monkeypatch.setattr(router_utils, "get_current_rank_layer_info", lambda *_args: {"start": 0, "end": 1})
    monkeypatch.setattr(router_utils, "device_name", "cpu")

    # The route tensor reaches preprocess_thd_engine only on the replay side, and by
    # then it is back to int16; input_ids arrive as int64.
    def fake_preprocess(value, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records local_cp_size; for int16 replay input replaces the value with a fixed route tensor; returns
        # (value, dummy packed params, None).
        calls.append(kwargs["local_cp_size"])
        if value.dtype == torch.int16:
            value = torch.tensor([[[[1]], [[2]]]], dtype=torch.int16)
        return value, object(), None

    # postprocess_thd_engine is dtype- and trailing-shape-preserving, so it hands
    # back the uint8 payload merge_router_topk_indices fed into the collectives.
    payload = torch.tensor([[[1]], [[2]]], dtype=torch.int16).view(torch.uint8)
    monkeypatch.setattr(router_utils, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(
        router_utils,
        "postprocess_thd_engine",
        lambda output, _packed, _input_ids, _batch_size, **kwargs: (
            calls.append(kwargs["local_cp_size"]) or _nested([payload])
        ),
    )

    input_ids = _nested([torch.arange(2)])
    recorded = []
    config = SimpleNamespace(fp8=None, num_layers=1, moe_layer_freq=1)
    router_utils.merge_router_topk_indices(None, input_ids, recorded, config, local_cp_size=2)

    # int16 has no NCCL datatype, so the collective must see the uint8 view while the
    # recorded routes land back as int16.
    assert gathered_dtypes == [torch.uint8]
    assert recorded[0].dtype == torch.int16

    target = {}
    router.set_target_indices = lambda indices, replay_mask=None: target.update(indices=indices, mask=replay_mask)
    router_utils.set_router_replay_data(recorded[0], None, config, local_cp_size=2)

    assert calls == [2, 2, 2]
    torch.testing.assert_close(target["indices"], torch.tensor([[1], [2]]))


def test_r3_alignment_mask_and_dcp_collection():
    # [FUNCTION DOC]
    # Aligns 3-row routes to input ids of lengths 4 and 3 with align_r3_router_replay_data and asserts per-sample
    # lengths [4, 3]; builds an R3 replay mask from a per-sample boolean tensor and asserts the expected per-token
    # masks; then _attach_dcp_recorded_routes must place the route for sample id 1 into the first loss entry and
    # sample 0 into the second under model_output.routed_experts.
    input_ids = _nested([torch.arange(4), torch.arange(3)])
    routes = _nested(
        [
            torch.full((3, 1, 1), 1, dtype=torch.int16),
            torch.full((3, 1, 1), 2, dtype=torch.int16),
        ]
    )
    aligned = router_utils.align_r3_router_replay_data(routes, input_ids)
    mask = router_utils.build_r3_replay_mask(
        input_ids,
        torch.tensor([[1, 1], [0, 0]], dtype=torch.bool),
    )

    assert [part.shape[0] for part in aligned.unbind()] == [4, 3]
    assert [part.tolist() for part in mask.unbind()] == [
        [True, True, True, False],
        [False, False, False],
    ]

    losses = [{DCP_SAMPLE_IDS: [1]}, {DCP_SAMPLE_IDS: [0]}]
    _attach_dcp_recorded_routes(losses, aligned)
    assert [part.shape[0] for part in losses[0]["model_output"]["routed_experts"].unbind()] == [4]
    assert [part.shape[0] for part in losses[1]["model_output"]["routed_experts"].unbind()] == [3]


def test_pp_gather_normalizes_nested_routes_to_cpu(monkeypatch):
    # [FUNCTION DOC]
    # With PP size 2 and a patched all_gather_object, pp_gather on nested int16 routes must send CPU tensors and
    # return a CPU int16 result concatenating local and remote layers along the layer dimension.
    # The nested branch rides all_gather_object, which pickles the tensor, so int16
    # needs no uint8 reinterpretation here.
    local = _nested([torch.tensor([[[1]], [[2]]], dtype=torch.int16)])
    remote = _nested([torch.tensor([[[3]], [[4]]], dtype=torch.int16)])
    monkeypatch.setattr(router_utils.mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)

    def fake_all_gather(output, value, _group):
        # [FUNCTION DOC]
        # HELPER: Asserts the value is on CPU and fills output with [value, remote].
        assert value.device.type == "cpu"
        output[:] = [value, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)
    config = SimpleNamespace(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=None)
    routes = router_utils.pp_gather(local, config)

    assert routes.device.type == "cpu"
    assert routes.dtype == torch.int16
    assert [part.tolist() for part in routes.unbind()] == [[[[1], [3]], [[2], [4]]]]


def test_pp_gather_supports_zero_layer_bshd_stage(monkeypatch):
    # [FUNCTION DOC]
    # A stage with zero MoE layers (BSHD shape with 0 layers) gathers with a remote stage holding 2 layers. Asserts two
    # all_gather calls (first a 1-element layer-count exchange, then the payload as a uint8 view sized to the max layer
    # count) and that the result equals the remote routes.
    local = torch.empty((1, 3, 0, 2), dtype=torch.int16)
    remote = torch.arange(12, dtype=torch.int16).reshape(1, 3, 2, 2)
    collective_calls = []

    monkeypatch.setattr(router_utils, "device_name", "cpu")
    monkeypatch.setattr(router_utils.mpu, "get_pipeline_model_parallel_group", lambda: object())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)

    def fake_all_gather(tensor_list, tensor, group, async_op=False):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather recording (shape, async_op); for the 1-element count exchange writes counts 0 and 2,
        # otherwise zeros the local slot and copies the remote payload's uint8 view into the second slot.
        collective_calls.append((tuple(tensor.shape), async_op))
        if tensor.numel() == 1:
            tensor_list[0].fill_(0)
            tensor_list[1].fill_(2)
            return
        tensor_list[0].zero_()
        tensor_list[1].copy_(remote.contiguous().view(torch.uint8))

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    config = SimpleNamespace(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=None)

    routes = router_utils.pp_gather(local, config)

    assert collective_calls == [((1,), False), ((1, 3, 2, 4), False)]
    assert routes.shape == remote.shape
    torch.testing.assert_close(routes, remote)


def test_reorder_and_merge_vpp_layers_keeps_dense_chunk_placeholder(monkeypatch):
    # [FUNCTION DOC]
    # With a patched schedule table, reorder_and_merge_vpp_layers over an empty dense chunk and a one-layer MoE chunk
    # returns exactly the MoE chunk.
    empty_chunk = torch.empty((1, 3, 0, 2), dtype=torch.int16)
    moe_chunk = torch.arange(6, dtype=torch.int16).reshape(1, 3, 1, 2)
    monkeypatch.setattr(router_utils, "get_schedule_table", lambda *_args: [(0, 0), (0, 1)])

    routes = router_utils.reorder_and_merge_vpp_layers([empty_chunk, moe_chunk], 1, 2, 1)

    assert routes.shape == moe_chunk.shape
    torch.testing.assert_close(routes, moe_chunk)
