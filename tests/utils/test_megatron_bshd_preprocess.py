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
# CPU (and optional GPU) tests for verl/models/mcore/util.py preprocessing/postprocessing of jagged batches into
# BSHD (padded batch) and THD (packed) layouts, with megatron stubbed. Covers dense trailing dimensions
# (top-k teacher log-probs / ids, router data), length bucketing, minimum local rows, router-replay padding,
# zigzag and contiguous context-parallel layouts, rolling of labels, and gathering back in postprocess.

"""Regression coverage for verl#6492."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

import verl.utils.device as device_module


def _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size: int = 4, cp_size: int = 1, cp_rank: int = 0):
    # [FUNCTION DOC]
    # HELPER: Installs fake megatron, megatron.core, parallel_state and packed_seq_params modules into sys.modules
    # (via monkeypatch) whose parallel_state functions return the given TP size, CP size and CP rank, a dummy CP group,
    # and a tuple for dynamic DCP groups; PackedSeqParams is a simple kwargs-to-attributes class. Forces
    # device_module.is_npu_available False, then loads util.py by file path as a fresh module.
    # Inputs: monkeypatch, tp_size (default 4), cp_size (default 1), cp_rank (default 0).
    # Outputs: the loaded util module. Side effects: sys.modules and attribute patches (undone by monkeypatch).
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")

    parallel_state.get_context_parallel_world_size = lambda: cp_size
    parallel_state.get_context_parallel_rank = lambda: cp_rank
    parallel_state.get_context_parallel_group = lambda: object()
    parallel_state.get_tensor_model_parallel_world_size = lambda: tp_size

    parallel_state.get_dynamic_data_context_parallel_groups = lambda group_size: ("dcp", group_size)

    class PackedSeqParams:
        # [CLASS DOC]
        # HELPER: Stand-in for megatron's PackedSeqParams; a plain attribute bag.
        def __init__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Stores every keyword argument as an instance attribute.
            self.__dict__.update(kwargs)

    packed_seq_params.PackedSeqParams = PackedSeqParams

    core.parallel_state = parallel_state
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed_seq_params)
    monkeypatch.setattr(device_module, "is_npu_available", False)

    util_path = Path(__file__).parents[2] / "verl" / "models" / "mcore" / "util.py"
    spec = importlib.util.spec_from_file_location("mcore_util_regression", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_tensor(rows: list[torch.Tensor]) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Builds a jagged nested tensor from a list of row tensors.
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _check_topk_preprocess(monkeypatch, device: torch.device):
    # [FUNCTION DOC]
    # HELPER: Shared body for the top-k BSHD tests on a given device. Two nested samples of 3 and 2 rows with a
    # dense trailing dim of 64 are converted by preprocess_bshd_engine; asserts output shape (2, 4, 64) (padded to a
    # TP-aligned length of 4), device placement, rows copied in place with zeros in padding, the boolean attention
    # mask, and positions 0..3 for each row. Repeats for long-typed ids, asserting dtype is preserved and the mask is
    # identical.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    topk = 64

    logprob_rows = [
        torch.arange(3 * topk, dtype=torch.float32, device=device).reshape(3, topk),
        torch.arange(2 * topk, dtype=torch.float32, device=device).reshape(2, topk) + 1000,
    ]
    teacher_logprobs = _nested_tensor(logprob_rows)

    logprobs_bshd, attention_mask, position_ids = mcore_util.preprocess_bshd_engine(teacher_logprobs)

    assert logprobs_bshd.shape == (2, 4, topk)
    assert logprobs_bshd.device.type == device.type
    assert attention_mask.device.type == device.type
    assert position_ids.shape == (2, 4)
    torch.testing.assert_close(logprobs_bshd[0, :3], logprob_rows[0])
    torch.testing.assert_close(logprobs_bshd[1, :2], logprob_rows[1])
    torch.testing.assert_close(logprobs_bshd[0, 3], torch.zeros(topk, dtype=torch.float32, device=device))
    torch.testing.assert_close(logprobs_bshd[1, 2:], torch.zeros(2, topk, dtype=torch.float32, device=device))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True, False], [True, True, False, False]], device=device),
    )
    torch.testing.assert_close(
        position_ids,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long, device=device),
    )

    id_rows = [
        torch.arange(3 * topk, dtype=torch.long, device=device).reshape(3, topk),
        torch.arange(2 * topk, dtype=torch.long, device=device).reshape(2, topk) + 2000,
    ]
    teacher_ids = _nested_tensor(id_rows)
    ids_bshd, ids_attention_mask, _ = mcore_util.preprocess_bshd_engine(teacher_ids)

    assert ids_bshd.shape == (2, 4, topk)
    assert ids_bshd.dtype == torch.long
    torch.testing.assert_close(ids_bshd[0, :3], id_rows[0])
    torch.testing.assert_close(ids_bshd[1, :2], id_rows[1])
    torch.testing.assert_close(ids_attention_mask, attention_mask)


def test_preprocess_bshd_engine_preserves_1d_input_shape_on_cpu(monkeypatch):
    # [FUNCTION DOC]
    # Plain 1-D token rows of length 3 and 2 are padded to (2, 4) with zeros; checks the mask and position ids.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch)
    rows = [
        torch.tensor([11, 12, 13], dtype=torch.long),
        torch.tensor([21, 22], dtype=torch.long),
    ]
    input_ids = _nested_tensor(rows)

    input_ids_bshd, attention_mask, position_ids = mcore_util.preprocess_bshd_engine(input_ids)

    assert input_ids_bshd.shape == (2, 4)
    torch.testing.assert_close(input_ids_bshd[0], torch.tensor([11, 12, 13, 0], dtype=torch.long))
    torch.testing.assert_close(input_ids_bshd[1], torch.tensor([21, 22, 0, 0], dtype=torch.long))
    torch.testing.assert_close(
        attention_mask,
        torch.tensor([[True, True, True, False], [True, True, False, False]]),
    )
    torch.testing.assert_close(position_ids, torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long))


def test_preprocess_thd_engine_rounds_packed_length_to_bucket_without_cp(monkeypatch):
    # [FUNCTION DOC]
    # With TP=CP=1, a 513-token sample packed with pad_to_length_bucket 512 yields a packed shape of (1, 1024).
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=1)
    input_ids = _nested_tensor([torch.arange(513)])

    packed, _, _ = mcore_util.preprocess_thd_engine(input_ids, pad_to_length_bucket=512)

    assert packed.shape == (1, 1024)


def test_preprocess_bshd_engine_preserves_topk_dense_dim_on_cpu(monkeypatch):
    # [FUNCTION DOC]
    # Runs the top-k BSHD check on CPU.
    _check_topk_preprocess(monkeypatch, torch.device("cpu"))


def test_preprocess_bshd_engine_preserves_topk_dense_dim_on_gpu(monkeypatch):
    # [FUNCTION DOC]
    # Runs the top-k BSHD check on CUDA; skipped when CUDA is unavailable.
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    _check_topk_preprocess(monkeypatch, torch.device("cuda"))


def test_preprocess_thd_engine_pads_to_minimum_rows(monkeypatch):
    # [FUNCTION DOC]
    # A 100-token sample with min_local_rows 128 is padded to 128 rows: checks shape, padded cu_seqlens [0, 128],
    # original ids/positions in the first 100 slots, and zeros afterward.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    input_ids = _nested_tensor([torch.arange(100, dtype=torch.long)])

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        min_local_rows=128,
    )

    assert local_ids.shape == (1, 128)
    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 128]
    torch.testing.assert_close(local_ids[0, :100], torch.arange(100, dtype=torch.long))
    torch.testing.assert_close(local_ids[0, 100:], torch.zeros(28, dtype=torch.long))
    torch.testing.assert_close(local_positions[0, :100], torch.arange(100, dtype=torch.long))
    torch.testing.assert_close(local_positions[0, 100:], torch.zeros(28, dtype=torch.long))


def test_thd_router_replay_drops_record_padding_and_restores_expert_zero(monkeypatch):
    # [FUNCTION DOC]
    # Router replay round trip with TP=2: a 7-token sample is packed (padded to 8), a recorded route tensor with a
    # non-zero value in the padding slot is postprocessed back to per-sample form and re-packed; asserts the first
    # 7 routes survive and the padding slot becomes 0 (record padding dropped, expert id 0 restored).
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=2, cp_size=1)
    input_ids = _nested_tensor([torch.arange(7, dtype=torch.long)])
    recorded_routes = torch.tensor([[[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[9]]]], dtype=torch.uint8)

    _, packed_seq_params, _ = mcore_util.preprocess_thd_engine(input_ids)
    exported_routes = mcore_util.postprocess_thd_engine(
        recorded_routes,
        packed_seq_params,
        input_ids,
        batch_size=1,
    )
    replay_routes, _, _ = mcore_util.preprocess_thd_engine(exported_routes)

    torch.testing.assert_close(replay_routes[0, :7], recorded_routes[0, :7])
    assert replay_routes[0, 7].item() == 0


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_preprocess_thd_engine_pads_multidimensional_router_data(monkeypatch, cp_rank):
    # [FUNCTION DOC]
    # Parametrized over CP rank with CP=2: a single sample of 48 rows of 8 router entries is packed; asserts the
    # local result shape (1, 2, 48, 8), padded cu_seqlens [0, 4], and that rank 0 holds the original data in its
    # first row with zeros elsewhere while rank 1 gets all zeros.
    mcore_util = _load_mcore_util_with_stubbed_megatron(
        monkeypatch,
        tp_size=1,
        cp_size=2,
        cp_rank=cp_rank,
    )
    route = torch.arange(48 * 8, dtype=torch.long).reshape(1, 48, 8)
    routed_experts = _nested_tensor([route])

    local_routes, packed_seq_params, _ = mcore_util.preprocess_thd_engine(routed_experts)

    expected = torch.zeros((2, 48, 8), dtype=torch.long)
    if cp_rank == 0:
        expected[0] = route[0]
    assert local_routes.shape == (1, 2, 48, 8)
    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 4]
    torch.testing.assert_close(local_routes[0], expected)


# [1..7] is padded to [1, 2, 3, 4, 5, 6, 7, 0], and rolling it by -1 gives [2, 3, 4, 5, 6, 7, 0, 1].
# Zigzag gives rank 0 the global rows {0, 1, 6, 7} and rank 1 the rows {2, 3, 4, 5}.
@pytest.mark.parametrize(
    ("cp_rank", "expected"),
    [
        (0, [2, 3, 0, 1]),
        (1, [4, 5, 6, 7]),
    ],
)
def test_preprocess_thd_engine_preserves_1d_zigzag_roll_alignment(monkeypatch, cp_rank, expected):
    # [FUNCTION DOC]
    # With CP=2 and need_roll, labels 1..7 are padded to 8, rolled left by one, then zigzag split; asserts padded
    # cu_seqlens [0, 8] and the per-rank local labels given by the parametrization (see the comment above).
    mcore_util = _load_mcore_util_with_stubbed_megatron(
        monkeypatch,
        tp_size=1,
        cp_size=2,
        cp_rank=cp_rank,
    )
    labels = _nested_tensor([torch.arange(1, 8, dtype=torch.long)])

    local_labels, packed_seq_params, _ = mcore_util.preprocess_thd_engine(labels, need_roll=True)

    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 8]
    torch.testing.assert_close(local_labels[0], torch.tensor(expected, dtype=torch.long))


@pytest.mark.parametrize(
    ("cp_rank", "expected_ids", "expected_positions"),
    [
        (0, [10, 11, 12, 13, 14], [0, 1, 2, 3, 4]),
        (1, [0, 20, 21, 22, 0], [0, 0, 1, 2, 0]),
    ],
)
def test_preprocess_thd_engine_contiguous_uses_global_packed_intervals(
    monkeypatch, cp_rank, expected_ids, expected_positions
):
    # [FUNCTION DOC]
    # Contiguous CP layout with two samples (5 and 3 tokens): asserts padded cu_seqlens are [0, 6, 10] (padding only
    # to a multiple of CP size, not 2x as zigzag would) and the per-rank local ids and positions taken from
    # contiguous global packed intervals.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2, cp_rank=cp_rank)
    input_ids = _nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ]
    )

    local_ids, packed_seq_params, local_positions = mcore_util.preprocess_thd_engine(
        input_ids,
        cp_layout="contiguous",
    )

    assert packed_seq_params.cu_seqlens_q_padded.tolist() == [0, 6, 10]
    # A zigzag-only factor of two would incorrectly pad these rows to 8 and 4.
    assert packed_seq_params.cu_seqlens_q_padded.tolist() != [0, 8, 12]
    torch.testing.assert_close(local_ids[0], torch.tensor(expected_ids, dtype=torch.long))
    torch.testing.assert_close(local_positions[0], torch.tensor(expected_positions, dtype=torch.long))


def test_preprocess_thd_engine_rejects_unknown_cp_layout(monkeypatch):
    # [FUNCTION DOC]
    # preprocess_thd_engine raises ValueError for cp_layout "interleaved".
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2)
    input_ids = _nested_tensor([torch.tensor([10, 11], dtype=torch.long)])

    with pytest.raises(ValueError, match="Unsupported context parallel layout: interleaved"):
        mcore_util.preprocess_thd_engine(input_ids, cp_layout="interleaved")


def test_preprocess_thd_engine_contiguous_rolls_each_sequence_independently(monkeypatch):
    # [FUNCTION DOC]
    # Contiguous layout with need_roll: each sequence is rolled independently, so rank 0's local labels are
    # [2, 3, 4, 1].
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2, cp_rank=0)
    labels = _nested_tensor(
        [
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            torch.tensor([5, 6, 7, 8], dtype=torch.long),
        ]
    )

    local_labels, _, _ = mcore_util.preprocess_thd_engine(
        labels,
        need_roll=True,
        cp_layout="contiguous",
    )

    torch.testing.assert_close(local_labels[0], torch.tensor([2, 3, 4, 1], dtype=torch.long))


def test_postprocess_thd_engine_contiguous_inverts_global_intervals(monkeypatch):
    # [FUNCTION DOC]
    # After contiguous preprocessing, patches torch.distributed.all_gather to fill per-rank outputs, then asserts
    # postprocess_thd_engine reconstructs each sample's values from the global packed intervals (sample 0 is the
    # first 5 values, sample 1 the 3 values after its padding slot).
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2, cp_rank=0)
    input_ids = _nested_tensor(
        [
            torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
            torch.tensor([20, 21, 22], dtype=torch.long),
        ]
    )
    _, packed_seq_params, _ = mcore_util.preprocess_thd_engine(input_ids, cp_layout="contiguous")
    rank_outputs = [
        torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.float32),
        torch.tensor([[105, 106, 107, 108, 109]], dtype=torch.float32),
    ]

    def fake_all_gather(outputs, local_output, group):
        # [FUNCTION DOC]
        # HELPER: Fake all_gather that copies the predefined per-rank outputs into the destination tensors in place.
        del local_output, group
        for destination, source in zip(outputs, rank_outputs, strict=True):
            destination.copy_(source)

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    restored = mcore_util.postprocess_thd_engine(
        rank_outputs[0],
        packed_seq_params,
        input_ids,
        batch_size=2,
        cp_layout="contiguous",
    )

    torch.testing.assert_close(restored[0], torch.tensor([100, 101, 102, 103, 104], dtype=torch.float32))
    torch.testing.assert_close(restored[1], torch.tensor([106, 107, 108], dtype=torch.float32))


def test_postprocess_thd_engine_rejects_unknown_cp_layout(monkeypatch):
    # [FUNCTION DOC]
    # postprocess_thd_engine raises ValueError for cp_layout "interleaved" even with post_process False.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1, cp_size=2)

    with pytest.raises(ValueError, match="Unsupported context parallel layout: interleaved"):
        mcore_util.postprocess_thd_engine(
            torch.empty(1, 1),
            packed_seq_params=None,
            input_ids=torch.empty(1),
            batch_size=1,
            post_process=False,
            cp_layout="interleaved",
        )


def test_preprocess_thd_engine_pads_short_topk_sequence_dimension(monkeypatch):
    # [FUNCTION DOC]
    # With a dynamic local_cp_size of 2 and torch.distributed.get_rank patched to 1, a single-row top-k sample is
    # packed to shape (1, 2, 64); asserts this rank's local data is all zeros, positions are [1, 0], and the packed
    # params record local_cp_size 2.
    mcore_util = _load_mcore_util_with_stubbed_megatron(monkeypatch, tp_size=1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    topk = 64
    teacher_logprobs = _nested_tensor([torch.arange(topk, dtype=torch.float32).reshape(1, topk)])

    packed, packed_seq_params, position_ids = mcore_util.preprocess_thd_engine(
        teacher_logprobs,
        local_cp_size=2,
    )

    assert packed.shape == (1, 2, topk)
    torch.testing.assert_close(packed, torch.zeros_like(packed))
    torch.testing.assert_close(position_ids, torch.tensor([[1, 0]], dtype=torch.long))
    assert packed_seq_params.local_cp_size == 2
