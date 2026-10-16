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
# Constants HIDDEN_SIZE (8) and VOCAB_SIZE (64) size the fake LM. ALL_ADAPTERS lists the torch and triton fused
# forwards of dense_common, qwen3_5, qwen3_vl, qwen2_vl and glm4v; TORCH_ADAPTERS lists only the torch backends.

"""Regression test for issue #6068 (use_fused_kernels=True + ulysses_sp>1).

Under Ulysses sequence parallelism, the fused-kernel forward functions in
`verl/models/transformers/*.py` were computing
`rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)` *after* Ulysses had
already SP-sliced the input. `torch.roll` wraps around the local-shard
boundary rather than the global sequence, so the last position on every SP
rank ended up predicting the wrong label. This biased ~1 position per rank
per micro-batch and manifested as a slow training-quality regression at
SP > 1 (issue #6068).

The fix plumbs the engine's pre-rolled `input_ids_rmpad_rolled` into the fused
forwards via a new `shift_labels` kwarg, mirroring what the veomni engine
already does at `verl/workers/engine/veomni/transformer_impl.py:659`.

These tests:
  1. Demonstrate the root cause directly (no model needed) — slice-then-roll
     diverges from roll-then-slice at every shard boundary.
  2. Verify each adapter's fused-forward (torch + triton backends) honors
     the new `shift_labels` kwarg — i.e., the *routing* fix.

The fused kernels themselves are Triton + CUDA only, so we patch them out
for the CPU tests; their numerical correctness is covered by
`tests/utils/test_linear_cross_entropy.py`. The end-to-end SP=2 integration
test in `tests/special_distributed/test_fused_kernels_ulysses_sp.py`
exercises the full path on 2 GPUs.
"""

import contextlib
from typing import Any
from unittest import mock

import pytest
import torch

from verl.models.transformers import dense_common, glm4v, qwen2_vl, qwen3_5, qwen3_vl

# ---------------------------------------------------------------------------
# 1. Root-cause demonstration: slice-then-local-roll != global-roll-then-slice.
# ---------------------------------------------------------------------------


def _global_then_slice(input_ids: torch.Tensor, sp_size: int) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Rolls ids left by one along the last dim then splits into sp_size chunks.
    # Outputs: list of shard tensors.
    """Correct behavior: roll on the full sequence, then SP-slice. Mirrors the
    engine's `input_ids_rmpad_rolled` -> `ulysses_pad_and_slice_inputs` path.
    """
    rolled = torch.roll(input_ids, shifts=-1, dims=-1)
    return list(torch.chunk(rolled, sp_size, dim=-1))


def _slice_then_local_roll(input_ids: torch.Tensor, sp_size: int) -> list[torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Splits ids into sp_size chunks then rolls each chunk left by one.
    # Outputs: list of shard tensors.
    """Buggy behavior: SP-slice first, then roll on the local shard."""
    sliced = list(torch.chunk(input_ids, sp_size, dim=-1))
    return [torch.roll(s, shifts=-1, dims=-1) for s in sliced]


def test_local_roll_diverges_from_global_roll_under_sp():
    # [FUNCTION DOC]
    # Behavior: for seeded random ids (1 x 32) and sp 4, asserts all but the last position of each shard agree between
    # the two methods and the last position differs on every rank.
    """For SP > 1, slice-then-local-roll produces different labels than
    global-roll-then-slice at exactly the shard-boundary position on every rank.
    """
    torch.manual_seed(0)
    total_nnz, sp_size = 32, 4
    input_ids = torch.randint(0, 10000, (1, total_nnz))

    correct_shards = _global_then_slice(input_ids, sp_size)
    buggy_shards = _slice_then_local_roll(input_ids, sp_size)

    # Interior positions match — the bug is only at the shard boundary.
    for correct, buggy in zip(correct_shards, buggy_shards, strict=True):
        torch.testing.assert_close(correct[..., :-1], buggy[..., :-1])

    # Last position of every shard differs — that's the bias term that
    # accumulates across training steps.
    for rank in range(sp_size):
        correct_last = correct_shards[rank][..., -1]
        buggy_last = buggy_shards[rank][..., -1]
        assert not torch.equal(correct_last, buggy_last), (
            f"rank {rank}: expected divergence at shard boundary but got {correct_last}=={buggy_last}"
        )


# ---------------------------------------------------------------------------
# 2. Adapter routing tests: each fused-forward (torch + triton) must use
#    `shift_labels` verbatim when the engine passes it, and must fall back to
#    local roll when it's absent (preserves SP=1 behavior).
# ---------------------------------------------------------------------------


HIDDEN_SIZE = 8
VOCAB_SIZE = 64


class _FakeConfig:
    # [CLASS DOC]
    # HELPER: Model config stub with output_attentions and output_hidden_states False.
    """Minimal stand-in for `model.config` used by `forward_base_model`."""

    output_attentions = False
    output_hidden_states = False


class _FakeBaseOutput:
    # [CLASS DOC]
    # HELPER: Model-output stub indexable at 0 (hidden states) with hidden_states, past_key_values, attentions None.
    """Mimics the HF model output: tuple-indexable and attribute-accessible."""

    def __init__(self, hidden: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Stores hidden and sets the optional attributes to None.
        self._hidden = hidden
        self.hidden_states = None
        self.past_key_values = None
        self.attentions = None

    def __getitem__(self, idx):
        # [FUNCTION DOC]
        # HELPER: Returns hidden for index 0; raises IndexError otherwise.
        if idx == 0:
            return self._hidden
        raise IndexError(idx)


def _make_fake_lm() -> torch.nn.Module:
    # [FUNCTION DOC]
    # HELPER: Seeds torch with 42 and returns a tiny LM module with config, a base model (embedding to float32 hidden
    # states wrapped in _FakeBaseOutput, ignoring extra kwargs) and a bias-free linear lm_head.
    """Minimal stand-in for the language-model wrapper used by every adapter.

    Provides `self.config`, `self.model(...)`, and `self.lm_head`. The fake
    base model accepts the full kwarg set that `forward_base_model` passes.
    """

    class FakeBaseModel(torch.nn.Module):
        # [CLASS DOC]
        # HELPER: Embedding-only base model.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Creates the VOCAB_SIZE x HIDDEN_SIZE embedding.
            super().__init__()
            self.embed = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)

        def forward(self, input_ids=None, **_kwargs):
            # [FUNCTION DOC]
            # HELPER: Embeds input_ids, casts to float32, wraps in _FakeBaseOutput.
            hidden = self.embed(input_ids).to(torch.float32)
            return _FakeBaseOutput(hidden)

    class FakeLM(torch.nn.Module):
        # [CLASS DOC]
        # HELPER: Wrapper exposing config, model and lm_head as the adapters expect.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Builds the config stub, base model and lm_head.
            super().__init__()
            self.config = _FakeConfig()
            self.model = FakeBaseModel()
            self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    torch.manual_seed(42)
    return FakeLM()


@contextlib.contextmanager
def _patch_fused_kernels():
    # [FUNCTION DOC]
    # HELPER: Context manager patching FusedLinearForPPO.forward, linear_cross_entropy, qwen2_vl_forward and
    # glm4v_forward with fakes. Yields a dict where fakes record the label ids ("input_ids") the kernel received and,
    # for the FusedLinearForPPO path, its impl_backend. Fake kernels return zero log-probs and entropy shaped like ids;
    # fake VLM forwards return zero hidden states (1 x T x hidden).
    # Side effects: temporary monkeypatching restored on exit.
    """Patch the fused kernels (Triton, CUDA-only) and the VLM body wrappers
    so the CPU test exercises only the routing layer. Yields a dict that
    captures the `input_ids` arg the kernel would have received.
    """
    captured: dict[str, Any] = {}

    def _fake_log_probs_and_entropy(input_ids: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Returns two float32 zero tensors with the shape of input_ids (both branches are identical).
        if input_ids.dim() == 1:
            shape = input_ids.shape
        else:
            shape = input_ids.shape  # already (B, T)
        log_probs = torch.zeros(shape, dtype=torch.float32)
        entropy = torch.zeros(shape, dtype=torch.float32)
        return log_probs, entropy

    def fake_fused_linear_forward(self, hidden_states, vocab_weights, input_ids, temperature=1.0):
        # [FUNCTION DOC]
        # HELPER: Records a clone of input_ids and self.impl_backend into captured, returns zero outputs.
        captured["input_ids"] = input_ids.detach().clone()
        captured["impl_backend"] = self.impl_backend
        return _fake_log_probs_and_entropy(input_ids)

    def fake_linear_ce(hidden_states, vocab_weights, input_ids, temperature, reduction):
        # [FUNCTION DOC]
        # HELPER: Records a clone of input_ids into captured, returns zero outputs.
        captured["input_ids"] = input_ids.detach().clone()
        return _fake_log_probs_and_entropy(input_ids)

    def fake_qwen2_vl_forward(self, input_ids, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns zero hidden states of shape (1, seq_len, hidden size taken from lm_head weight).
        # Bypass `process_position_ids` and the real VLM body; we only care
        # about the routing layer that runs *after* the model forward.
        return _FakeBaseOutput(torch.zeros(1, input_ids.shape[-1], self.lm_head.weight.shape[1]))

    def fake_glm4v_forward(self, input_ids, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Same as the qwen2_vl fake, for glm4v.
        return _FakeBaseOutput(torch.zeros(1, input_ids.shape[-1], self.lm_head.weight.shape[1]))

    with (
        mock.patch(
            "verl.utils.experimental.torch_functional.FusedLinearForPPO.forward",
            new=fake_fused_linear_forward,
        ),
        mock.patch(
            "verl.utils.kernel.linear_cross_entropy.linear_cross_entropy",
            new=fake_linear_ce,
        ),
        mock.patch(
            "verl.models.transformers.qwen2_vl.qwen2_vl_forward",
            new=fake_qwen2_vl_forward,
        ),
        mock.patch(
            "verl.models.transformers.glm4v.glm4v_forward",
            new=fake_glm4v_forward,
        ),
    ):
        yield captured


ALL_ADAPTERS = [
    dense_common.forward_with_torch_backend,
    dense_common.forward_with_triton_backend,
    qwen3_5.forward_with_torch_backend,
    qwen3_5.forward_with_triton_backend,
    qwen3_vl.forward_with_torch_backend,
    qwen3_vl.forward_with_triton_backend,
    qwen2_vl.forward_with_torch_backend,
    qwen2_vl.forward_with_triton_backend,
    glm4v.forward_with_torch_backend,
    glm4v.forward_with_triton_backend,
]

TORCH_ADAPTERS = [
    dense_common.forward_with_torch_backend,
    qwen3_5.forward_with_torch_backend,
    qwen3_vl.forward_with_torch_backend,
    qwen2_vl.forward_with_torch_backend,
    glm4v.forward_with_torch_backend,
]


def _adapter_id(forward_fn) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns "MODULE_LEAF.FUNCTION_NAME" for pytest ids.
    return f"{forward_fn.__module__.rsplit('.', 1)[-1]}.{forward_fn.__name__}"


@pytest.mark.parametrize("forward_fn", TORCH_ADAPTERS, ids=_adapter_id)
def test_torch_adapter_propagates_fused_linear_backend(forward_fn):
    # [FUNCTION DOC]
    # Behavior: sets the model's _verl_fused_kernels_backend to "liger", runs a torch-backend adapter, and asserts the
    # fused linear module saw impl_backend "liger".
    model = _make_fake_lm()
    model._verl_fused_kernels_backend = "liger"
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)

    with _patch_fused_kernels() as captured:
        forward_fn(model, input_ids=input_ids, labels=None, temperature=1.0, return_dict=True)

    assert captured["impl_backend"] == "liger"


@pytest.mark.parametrize("forward_fn", ALL_ADAPTERS, ids=_adapter_id)
def test_adapter_honors_shift_labels(forward_fn):
    # [FUNCTION DOC]
    # Behavior: passes explicit shift_labels that differ from rolled ids and asserts the kernel was called and received
    # exactly shift_labels.
    """When `shift_labels` is provided, every fused adapter must pass it
    through to the kernel verbatim — no local re-rolling.
    """
    model = _make_fake_lm()
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    # Deliberately != torch.roll(input_ids); the only way these labels reach
    # the kernel is if the adapter honors `shift_labels`.
    shift_labels = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)

    with _patch_fused_kernels() as captured:
        forward_fn(
            model,
            input_ids=input_ids,
            labels=None,
            temperature=1.0,
            shift_labels=shift_labels,
            return_dict=True,
        )

    assert "input_ids" in captured, "fused kernel was never called"
    assert torch.equal(captured["input_ids"], shift_labels), (
        f"{_adapter_id(forward_fn)}: expected kernel to see {shift_labels.tolist()}, "
        f"got {captured['input_ids'].tolist()}"
    )


@pytest.mark.parametrize("forward_fn", ALL_ADAPTERS, ids=_adapter_id)
def test_adapter_falls_back_to_local_roll_when_shift_labels_absent(forward_fn):
    # [FUNCTION DOC]
    # Behavior: without shift_labels, asserts the kernel received the ids rolled left by one.
    """Backward-compat: callers that don't pass `shift_labels` (e.g. the SP=1
    code path or any non-engine consumer) see unchanged behavior.
    """
    model = _make_fake_lm()
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    expected = torch.roll(input_ids, shifts=-1, dims=-1)

    with _patch_fused_kernels() as captured:
        forward_fn(
            model,
            input_ids=input_ids,
            labels=None,
            temperature=1.0,
            return_dict=True,
        )

    assert torch.equal(captured["input_ids"], expected), (
        f"{_adapter_id(forward_fn)}: expected fallback to torch.roll(input_ids), got {captured['input_ids'].tolist()}"
    )
