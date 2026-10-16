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
# Unit tests for verl.utils.dynamic_cp_scheduler (dynamic context-parallel scheduling).
# Covers: adapting Megatron-core scheduler assignments into per-rank micro batches with DCP
# metadata, padding-mask token accounting, input validation errors, Megatron engine DCP
# contract helpers, threading of local_cp_size through the fused forward, and gathering /
# reordering of outputs in postprocess_dynamic_cp_batch. Megatron is faked via monkeypatch
# except where the test explicitly importorskips "megatron".

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import verl.utils.dynamic_cp_scheduler as dcp_module
from verl.utils import tensordict_utils as tu
from verl.utils.dynamic_cp_scheduler import (
    DCP_GROUP_LEADER,
    DCP_LOCAL_NUM_TOKENS,
    DCP_PADDING_MASK,
    DCP_SAMPLE_IDS,
    DynamicCPScheduler,
    _local_padding_mask,
    postprocess_dynamic_cp_batch,
)


class _FakeGroup:
    # [CLASS DOC]
    # HELPER: Minimal stand-in for a torch.distributed process group exposing only size() and rank().
    def __init__(self, size, rank=0):
        # [FUNCTION DOC]
        # HELPER: Stores the group size and this process's rank (default 0). No side effects.
        self._size = size
        self._rank = rank

    def size(self):
        # [FUNCTION DOC]
        # HELPER: Returns the stored group size.
        return self._size

    def rank(self):
        # [FUNCTION DOC]
        # HELPER: Returns the stored rank.
        return self._rank


def _nested(parts):
    # [FUNCTION DOC]
    # HELPER: Builds a jagged-layout nested tensor from a list of 1-D tensors.
    # Inputs: parts - list of tensors of varying length. Outputs: nested tensor.
    return torch.nested.as_nested_tensor(parts, layout=torch.jagged)


def _batch(lengths):
    # [FUNCTION DOC]
    # HELPER: Builds a TensorDict batch with a jagged "input_ids" (each sample is the range 0..length-1)
    # and a "sample" index tensor 0..N-1, batch size N.
    # Inputs: lengths - list of sequence lengths. Outputs: the TensorDict.
    return TensorDict(
        {
            "input_ids": _nested([torch.arange(length) for length in lengths]),
            "sample": torch.arange(len(lengths)),
        },
        batch_size=[len(lengths)],
    )


def test_schedule_adapts_mcore_assignments(monkeypatch):
    # [FUNCTION DOC]
    # Verifies DynamicCPScheduler.schedule converts a fake mcore scheduler's group assignment
    # (4 DP*CP ranks: ranks 0 and 1 share sample 0, ranks 2 and 3 get samples 1 and 2) into per-rank
    # micro batches. For each rank checks selected sample ids, local_cp_size (2 for shared, 1 otherwise),
    # DCP sample-id metadata, group-leader flag (only first rank in a shared group), local token count
    # (sample 0 of length 7 split as 3 and 4) and a boolean padding mask.
    # Side effects: monkeypatches dcp_module.get_megatron_dynamic_cp_scheduler_cls.
    class _Scheduler:
        # [CLASS DOC]
        # HELPER: Fake mcore dynamic-CP scheduler class returned by the patched factory.
        def __init__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts the constructor receives exactly the expected keyword arguments
            # (max seqlen 4, cp_size 2, dp_size 2, no microbatch group size). Raises AssertionError otherwise.
            assert kwargs == {
                "max_seqlen_per_dp_cp_rank": 4,
                "cp_size": 2,
                "dp_size": 2,
                "microbatch_group_size_per_vp_stage": None,
            }

        def get_groups_and_subsamples(self, samples):
            # [FUNCTION DOC]
            # HELPER: Asserts samples are (index, length) pairs for lengths 7,3,5 and returns one group
            # assigning sample lists [0],[0],[1],[2] to the four ranks.
            assert samples == [(0, 7), (1, 3), (2, 5)]
            return [[[0], [0], [1], [2]]]

    monkeypatch.setattr(dcp_module, "get_megatron_dynamic_cp_scheduler_cls", lambda: _Scheduler)
    scheduler = DynamicCPScheduler(max_seqlen_per_dp_cp_rank=4, dp_size=2, cp_size=2)
    expected = [
        ([0], 2, True, 3),
        ([0], 2, False, 4),
        ([1], 1, True, 3),
        ([2], 1, True, 5),
    ]

    for rank, (sample_ids, cp_size, leader, local_tokens) in enumerate(expected):
        micro_batch = scheduler.schedule(_batch([7, 3, 5]), _FakeGroup(4, rank), tp_size=1)[0]
        assert micro_batch["sample"].tolist() == sample_ids
        assert tu.get_non_tensor_data(micro_batch, "local_cp_size", None) == cp_size
        assert tu.get_non_tensor_data(micro_batch, DCP_SAMPLE_IDS, None) == sample_ids
        assert tu.get_non_tensor_data(micro_batch, DCP_GROUP_LEADER, None) is leader
        assert tu.get_non_tensor_data(micro_batch, DCP_LOCAL_NUM_TOKENS, None) == local_tokens
        assert tu.get_non_tensor_data(micro_batch, DCP_PADDING_MASK, None).dtype == torch.bool


@pytest.mark.parametrize("cp_size,tp_size", [(2, 1), (4, 2)])
def test_padding_masks_count_real_tokens(cp_size, tp_size):
    # [FUNCTION DOC]
    # Parametrized over (cp_size, tp_size). Builds a local padding mask per CP rank for sequence
    # lengths 7,3,17 and asserts the total count of non-padding (False) entries across ranks equals
    # the real token count 27.
    masks = [_local_padding_mask([7, 3, 17], cp_size=cp_size, cp_rank=rank, tp_size=tp_size) for rank in range(cp_size)]
    assert sum(mask.numel() - mask.sum().item() for mask in masks) == 27


def test_scheduler_rejects_invalid_capacity_and_group(monkeypatch):
    # [FUNCTION DOC]
    # Checks validation errors: constructing DynamicCPScheduler with a zero capacity raises ValueError
    # ("positive integers"); scheduling a 100-token sample with capacity 4*2 raises ValueError
    # ("Increase max_seqlen"); and a fake mcore scheduler whose sample split across ranks is not
    # aligned to a contiguous CP group raises RuntimeError ("not aligned").
    # Side effects: monkeypatches the scheduler factory.
    with pytest.raises(ValueError, match="positive integers"):
        DynamicCPScheduler(0, 1, 1)

    scheduler = DynamicCPScheduler(4, 2, 2)
    with pytest.raises(ValueError, match="Increase max_seqlen"):
        scheduler.schedule(_batch([100]), _FakeGroup(4), tp_size=1)

    class _UnalignedScheduler:
        # [CLASS DOC]
        # HELPER: Fake mcore scheduler that returns a misaligned assignment.
        def __init__(self, **_kwargs):
            # [FUNCTION DOC]
            # HELPER: Accepts and ignores any keyword arguments.
            pass

        def get_groups_and_subsamples(self, _samples):
            # [FUNCTION DOC]
            # HELPER: Returns one group assigning [0],[1],[1],[2] to four ranks (sample 1 split across ranks 1 and 2,
            # which is not a valid aligned CP group).
            return [[[0], [1], [1], [2]]]

    monkeypatch.setattr(dcp_module, "get_megatron_dynamic_cp_scheduler_cls", lambda: _UnalignedScheduler)
    with pytest.raises(RuntimeError, match="not aligned"):
        scheduler.schedule(_batch([3, 5, 7]), _FakeGroup(4, rank=1), tp_size=1)


def test_engine_dcp_contracts():
    # [FUNCTION DOC]
    # Requires megatron (skipped otherwise). Checks Megatron engine DCP helpers: world sizes 2, 6, 14 are
    # accepted and 3 raises ValueError ("even integer"); a uniform temperature tensor resolves to its scalar
    # while a non-uniform one raises NotImplementedError; and _check_dcp_unsupported_features raises
    # NotImplementedError mentioning moe_router_fusion=False when router fusion is enabled with DCP.
    pytest.importorskip("megatron")
    from verl.workers.engine.megatron.transformer_impl import (
        _check_dcp_unsupported_features,
        _resolve_fused_temperature,
        _validate_dcp_world_size,
    )

    for size in (2, 6, 14):
        _validate_dcp_world_size(size)
    with pytest.raises(ValueError, match="even integer"):
        _validate_dcp_world_size(3)

    assert _resolve_fused_temperature(torch.tensor([0.5, 0.5])) == 0.5
    with pytest.raises(NotImplementedError, match="uniform temperature"):
        _resolve_fused_temperature(torch.tensor([0.5, 1.0]))

    engine_config = SimpleNamespace(
        dynamic_context_parallel=True,
        use_remove_padding=True,
        pad_to_length=False,
        virtual_pipeline_model_parallel_size=None,
        router_replay=SimpleNamespace(mode="R2"),
    )
    model_config = SimpleNamespace(model_type="language_model", hf_config=SimpleNamespace())
    with pytest.raises(NotImplementedError, match="moe_router_fusion=False"):
        _check_dcp_unsupported_features(
            engine_config,
            model_config,
            tf_config=SimpleNamespace(fp8=None, moe_router_fusion=True),
        )


def test_fused_forward_threads_dynamic_cp_arguments(monkeypatch):
    # [FUNCTION DOC]
    # Requires megatron. Patches preprocess/postprocess/unwrap_model in the fused forward module and calls
    # the fused forward with local_cp_size=2 and a router padding mask. Asserts local_cp_size was passed to
    # both preprocess calls (input ids and labels), that the padding mask reached the model call as
    # "padding_mask", and that the output contains "log_probs".
    pytest.importorskip("megatron")
    import verl.models.mcore.model_forward_fused as fused_module

    captured = {}

    def fake_preprocess(value, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake preprocess that records the local_cp_size kwarg, and returns the first 4 flattened
        # values of the nested input as a 1xN tensor, a packed-params namespace with fixed padded cu_seqlens,
        # and None.
        captured.setdefault("cp_sizes", []).append(kwargs.get("local_cp_size"))
        packed = SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 8], dtype=torch.int32))
        return value.values().reshape(1, -1)[:, :4], packed, None

    monkeypatch.setattr(fused_module, "preprocess_thd_engine", fake_preprocess)
    monkeypatch.setattr(fused_module, "postprocess_thd_engine", lambda output, *_args, **_kwargs: output)
    monkeypatch.setattr(fused_module, "unwrap_model", lambda model: model)

    class _Model:
        # [CLASS DOC]
        # HELPER: Fake Megatron model with pre/post process flags and a config without fp8.
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None)

        def __call__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records the received kwargs and returns a namespace with zero log_probs of shape (1, 4).
            captured["kwargs"] = kwargs
            return SimpleNamespace(log_probs=torch.zeros(1, 4))

    input_ids = _nested([torch.arange(7)])
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    output = fused_module.fused_forward_model_engine()(
        model=_Model(),
        input_ids=input_ids,
        labels=input_ids,
        multi_modal_inputs={},
        temperature=1.0,
        calculate_entropy=False,
        pad_token_id=0,
        local_cp_size=2,
        router_padding_mask=padding_mask,
    )

    assert captured["cp_sizes"] == [2, 2]
    assert captured["kwargs"]["padding_mask"] is padding_mask
    assert "log_probs" in output


def test_postprocess_restores_original_sample_order(monkeypatch):
    # [FUNCTION DOC]
    # Verifies postprocess_dynamic_cp_batch gathers per-rank outputs (local leader holds sample 1, a remote
    # rank holds samples 0 and 2, others are empty), and reassembles log_probs in original sample order
    # as a nested tensor, and gathers loss and metric values in sample order ([1.0, 2.0] and [0.0, 1.0]).
    # Side effects: monkeypatches torch.distributed.all_gather_object.
    local = [
        {
            DCP_GROUP_LEADER: True,
            DCP_SAMPLE_IDS: [1],
            "model_output": {"log_probs": _nested([torch.tensor([10.0, 11.0])])},
            "loss": 2.0,
            "metrics": {"metric": 1.0},
        }
    ]
    remote = [
        {
            "sample_ids": [0, 2],
            "model_output": {"log_probs": [torch.tensor([0.0]), torch.tensor([20.0, 21.0])]},
            "loss": 1.0,
            "metrics": {"metric": 0.0},
        }
    ]

    def fake_all_gather(output, value, group):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather_object that fills the output list with [local payload, remote payload, empty, empty]
        # in rank order. Mutates output in place.
        output[:] = [value, remote, [], []]

    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather)
    result = postprocess_dynamic_cp_batch(local, batch_size=3, dcp_group=_FakeGroup(4))

    assert [part.tolist() for part in result["model_output"]["log_probs"].unbind()] == [
        [0.0],
        [10.0, 11.0],
        [20.0, 21.0],
    ]
    assert result["loss"] == [1.0, 2.0]
    assert result["metrics"] == {"metric": [0.0, 1.0]}
