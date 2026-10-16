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
# CPU unit tests for the VeOmniRouterReplay controller state machine (actions
# DISABLED, RECORD, REPLAY). The controller assigns each router module a layer
# position keyed by object identity on first fire within a micro-batch; in RECORD
# it snapshots the native top-k expert indices once per layer (recompute does not
# overwrite), in REPLAY it returns the per-layer target indices (falling back to
# native for rows with duplicate top-k entries). Also covers take_recorded
# stacking, micro-batch mode validation, clear, and install/uninstall against a
# stubbed VeOmni global hook module. Toy shapes: 16 tokens, top-2 of 8 experts.

"""Unit tests for ``verl.utils.veomni.router_replay.VeOmniRouterReplay``.

Pure-CPU coverage for the controller state machine. The patched
``SparseMoeBlock`` integration (the actual end-to-end forward through a
real router) lives in VeOmni's invariant test suite; the engine-side
pad / Ulysses / nested-rebuild glue lives in
``tests/workers/test_router_replay_engine_helpers_on_cpu.py``.

What's covered
--------------
* RECORD lifecycle for a single micro-batch, and the reset between
  micro-batches.
* Recompute under per-layer activation checkpointing, which fires each
  layer *independently in reverse order* — the failure mode that breaks
  any monotonic-cursor design.
* ``take_recorded`` stacking order and shape.
* REPLAY first step (R3 case): targets must work before any RECORD has
  populated the id table.
* REPLAY strict missing-target and shape-mismatch error paths, and the
  duplicate-top-k fallback that guards positions with no recorded
  routing.
* ``begin_microbatch`` mode validation.
* Snapshot clone semantics.
* ``install`` / ``uninstall`` against a stubbed VeOmni hook surface.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from verl.utils.veomni.router_replay import RouterReplayAction, VeOmniRouterReplay

# ----------------------------------------------------------------- fixtures


@pytest.fixture
def ctrl():
    # [FUNCTION DOC]
    # HELPER: Fixture returning a fresh, not-installed controller.
    """Fresh controller per test. Does NOT call ``install()`` — that
    would need a stubbed ``veomni.utils.moe_router_replay`` module.
    The controller's ``on_router_forward`` / ``begin_*`` / ``clear``
    methods do not depend on ``install()`` having been called, so most
    tests don't need it. The ``install`` / ``uninstall`` paths get
    their own dedicated tests below."""
    return VeOmniRouterReplay()


# Each FakeRouter is a distinct nn.Module so ``id(router)`` is unique
# and stable for the test's lifetime — exactly what the production code
# relies on for FSDP2-wrapped MoE routers.
class _FakeRouter(nn.Module):
    # [CLASS DOC]
    # HELPER: Empty nn.Module standing in for an MoE router; distinct instances give
    # distinct identities.
    pass


@pytest.fixture
def routers():
    # [FUNCTION DOC]
    # HELPER: Fixture returning three distinct fake routers (three MoE layers).
    """Three distinct router instances, mimicking three MoE layers."""
    return [_FakeRouter() for _ in range(3)]


# Toy shapes — small enough that any failure prints readable tensors.
_NNZ = 16
_TOPK = 2


def _scores():
    # [FUNCTION DOC]
    # HELPER: Returns random router scores of shape [16, 8].
    return torch.randn(_NNZ, 8)


def _idx():
    # [FUNCTION DOC]
    # HELPER: Returns int64 indices [16, 2] with distinct experts per row (random
    # permutation prefix per row), matching real top-k output.
    """Return ``[_NNZ, _TOPK]`` int indices with DISTINCT entries per
    row. Distinct-per-row matters for tests that assert REPLAY returns
    the target verbatim — the duplicate-detection fallback in
    ``on_router_forward`` would otherwise treat duplicate-top-k rows
    as corrupted and return native instead. Real router top-k output
    always picks distinct experts, so this matches production
    semantics."""
    # Random distinct top-k per row via sampling without replacement.
    return torch.stack(
        [torch.randperm(8)[:_TOPK] for _ in range(_NNZ)],
        dim=0,
    ).to(torch.int64)


def _fire_all(ctrl, routers):
    # [FUNCTION DOC]
    # HELPER: Fires each router once in order through on_router_forward with fresh
    # random scores and indices. Outputs: list of returned index tensors.
    """Fire each router once in order and return the controller's outputs."""
    return [ctrl.on_router_forward(r, _scores(), _idx()) for r in routers]


# ===========================================================
# RECORD lifecycle
# ===========================================================


def test_take_recorded_is_layer_major(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: after one micro-batch in RECORD, take_recorded returns [16, layers, 2]
    # with each layer's slice equal to what that layer fired.
    """``take_recorded`` stacks layers on dim=1, so the result matches the
    trainer-side ``routed_experts`` layout ``[nnz, L, topk]``."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    fired = _fire_all(ctrl, routers)

    recorded = ctrl.take_recorded()
    assert recorded.shape == (_NNZ, len(routers), _TOPK)
    for pos, native in enumerate(fired):
        assert torch.equal(recorded[:, pos, :], native), f"layer {pos} landed on the wrong slice"


def test_record_begin_microbatch_drops_previous_microbatch(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: repeatedly starting a micro-batch and firing all routers keeps exactly
    # one entry per layer (previous micro-batch dropped) and the action stays RECORD.
    """The controller holds exactly one micro-batch. Re-arming must clear
    the previous one, otherwise ``take_recorded`` would stack stale layers
    and the engine's per-mb gather would see the wrong nnz."""
    ctrl.begin_record()
    for _ in range(3):
        ctrl.begin_microbatch()
        _fire_all(ctrl, routers)
        assert len(ctrl._recorded) == len(routers)
    assert ctrl.action is RouterReplayAction.RECORD


def test_take_recorded_before_any_router_fired_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: take_recorded with nothing fired raises RuntimeError "no router fired".
    """An empty buffer means the model's SparseMoeBlock never called the
    hook — fail with a message that points at the wiring."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    with pytest.raises(RuntimeError, match="no router fired"):
        ctrl.take_recorded()


def test_take_recorded_outside_record_mode_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: take_recorded in REPLAY mode raises RuntimeError "requires RECORD".
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="requires RECORD"):
        ctrl.take_recorded()


# ===========================================================
# Activation-checkpointing recompute (the bug class id-keying solves)
# ===========================================================


def test_record_recompute_per_layer_reverse(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: after a forward pass, re-firing routers in reverse order with new
    # indices (per-layer checkpoint recompute) adds no entries and keeps the
    # original snapshots.
    """Per-layer checkpointing: backward replays each layer independently
    in REVERSE order. This is the realistic VeOmni MoE training case
    that breaks any monotonic-cursor design. Subsumes whole-model
    recompute (which is just sequential forward order)."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    fired = _fire_all(ctrl, routers)  # forward layer 0..L-1
    for r in reversed(routers):  # backward recompute, reverse order
        ctrl.on_router_forward(r, _scores(), _idx())

    assert len(ctrl._recorded) == len(routers), f"per-layer reverse recompute leaked extra slots: {len(ctrl._recorded)}"
    # The recompute fired with *different* indices; the original snapshot wins.
    for pos, native in enumerate(fired):
        assert torch.equal(ctrl._recorded[pos], native), f"layer {pos} was overwritten by recompute"


# ===========================================================
# REPLAY
# ===========================================================


def test_replay_first_step_without_prior_discovery(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: REPLAY on the very first step (no prior RECORD) returns each layer's
    # target in fire order.
    """R3 first step: REPLAY runs before any RECORD has populated the id
    table. begin_microbatch just stashes the list; lookup happens
    lazily during forward."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.begin_microbatch(targets=targets)
    returned = _fire_all(ctrl, routers)
    for i, ret in enumerate(returned):
        assert torch.equal(ret, targets[i]), f"REPLAY layer {i} returned wrong target"


def test_replay_per_layer_reverse_recompute(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: after a forward pass in REPLAY, reverse-order recompute returns each
    # router's own target.
    """REPLAY recompute under per-layer checkpointing: same id ->
    same target, regardless of fire order."""
    ctrl.begin_replay()
    targets = [_idx() for _ in routers]
    ctrl.begin_microbatch(targets=targets)
    _fire_all(ctrl, routers)  # forward populates id mapping
    # backward recompute in reverse — each layer must hit its OWN target
    for r, want_pos in zip(reversed(routers), reversed(range(len(routers))), strict=True):
        ret = ctrl.on_router_forward(r, _scores(), _idx())
        assert torch.equal(ret, targets[want_pos]), f"REPLAY recompute layer {want_pos} returned wrong target"


def test_replay_strict_missing_target_pos_raises(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: with one target for three layers, the second distinct router raises
    # RuntimeError matching "pos=1 ... no target".
    """Layer position with no target must raise — no silent fallback."""
    ctrl.begin_replay()
    ctrl.begin_microbatch(targets=[_idx()])  # only 1 target for 3 layers
    ctrl.on_router_forward(routers[0], _scores(), _idx())  # pos 0 OK
    with pytest.raises(RuntimeError, match="pos=1.*no target"):
        ctrl.on_router_forward(routers[1], _scores(), _idx())


def test_replay_duplicate_topk_falls_back_to_native(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: targets with an all-zero row and a row with a repeated expert fall
    # back to the native indices for those rows, while clean rows use the target.
    """Rows whose target top-k contains a duplicate must fall through to
    native routing. This is the only guard for positions that have no
    recorded routing: the pad suffix the engine appends, and the trailing
    rows the rollout backend leaves at zero in R3 (final generated token,
    tool-response tokens). All of them arrive as all-zero rows.

    Without the fallback, VeOmni's MoE expert dispatch silently dedupes the
    duplicate top-k slots inside ``permute()``, while ``input_splits`` keeps
    counting all of them — the EP all-to-all then crashes with ``Split sizes
    doesn't match total dim 0 size`` several layers deep.
    """
    ctrl.begin_replay()

    # Build a target with two corruption patterns:
    #   row 0: all-zeros (what a pad token or an unrecorded R3 row looks like)
    #   row 1: partial duplicate (slots 0 and 1 both expert 5)
    # The other rows are clean (distinct top-k).
    targets = []
    for _ in routers:
        t = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + k for k in range(_TOPK)],
            dim=-1,
        )  # row i: [i, i+1] — distinct
        t[0] = 0  # all-zero row
        t[1, 0] = 5
        t[1, 1] = 5  # duplicate row
        targets.append(t)
    ctrl.begin_microbatch(targets=targets)

    for r, t in zip(routers, targets, strict=True):
        # Native: distinct per row, distinct per slot.
        native = torch.stack(
            [torch.arange(_NNZ, dtype=torch.int64) + 100 + k for k in range(_TOPK)],
            dim=-1,
        )
        out = ctrl.on_router_forward(r, _scores(), native)

        # Corrupted rows fall back to native.
        assert torch.equal(out[0], native[0]), "all-zero row must fall back to native"
        assert torch.equal(out[1], native[1]), "partial-duplicate row must fall back to native"
        # Clean rows substitute normally.
        for i in range(2, _NNZ):
            assert torch.equal(out[i], t[i]), f"clean row {i} must use replay target"


def test_replay_target_row_count_mismatch_raises(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: a target whose row count differs from the router's token count raises
    # RuntimeError "target has shape".
    """Defensive: a target sliced with a different SP rule than input_ids
    would silently misalign routing. The controller refuses rather than
    letting the downstream ``torch.where`` broadcast."""
    ctrl.begin_replay()
    ctrl.begin_microbatch(targets=[torch.zeros(_NNZ + 4, _TOPK, dtype=torch.int64) for _ in routers])
    with pytest.raises(RuntimeError, match="target has shape"):
        ctrl.on_router_forward(routers[0], _scores(), _idx())


# ===========================================================
# begin_microbatch mode validation
# ===========================================================


def test_begin_microbatch_replay_without_targets_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: begin_microbatch without targets in REPLAY raises RuntimeError.
    ctrl.begin_replay()
    with pytest.raises(RuntimeError, match="REPLAY requires per-layer targets"):
        ctrl.begin_microbatch()


def test_begin_microbatch_record_with_targets_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: begin_microbatch with targets in RECORD raises RuntimeError.
    ctrl.begin_record()
    with pytest.raises(RuntimeError, match="RECORD does not take targets"):
        ctrl.begin_microbatch(targets=[_idx()])


def test_begin_microbatch_while_disabled_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: begin_microbatch while DISABLED raises RuntimeError "requires RECORD or REPLAY".
    with pytest.raises(RuntimeError, match="requires RECORD or REPLAY"):
        ctrl.begin_microbatch()


# ===========================================================
# State management
# ===========================================================


def test_clear_resets_state(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: clear resets action to DISABLED and empties recorded, targets and the
    # identity-to-position map.
    """clear() is the always-safe reset; state must be empty after."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    _fire_all(ctrl, routers)
    assert ctrl.action is RouterReplayAction.RECORD
    assert ctrl._recorded
    ctrl.clear()
    assert ctrl.action is RouterReplayAction.DISABLED
    assert ctrl._recorded == []
    assert ctrl._targets == []
    assert ctrl._id_to_pos == {}


# ===========================================================
# Snapshot clone semantics
# ===========================================================


def test_record_snapshot_independent_of_source_tensor(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: mutating the source index tensor after recording does not change the
    # recorded snapshot (it is a detached clone).
    """The captured tensor must NOT alias the source — otherwise
    autograd-graph mutations would corrupt recorded indices."""
    ctrl.begin_record()
    ctrl.begin_microbatch()
    src = _idx()
    ctrl.on_router_forward(routers[0], _scores(), src)
    src.fill_(99)
    captured = ctrl._recorded[0]
    assert (captured != 99).all(), "snapshot must be independent of the source tensor (.detach().clone())"


# ===========================================================
# Disabled state
# ===========================================================


def test_disabled_passes_through_indices(ctrl, routers):
    # [FUNCTION DOC]
    # Behavior: while DISABLED, on_router_forward returns the input indices and leaves
    # all internal state empty.
    """When no begin_*() has been called (DISABLED), on_router_forward
    is a no-op pass-through and does not even assign layer positions."""
    src = _idx()
    out = ctrl.on_router_forward(routers[0], _scores(), src)
    assert torch.equal(out, src)
    assert ctrl._recorded == []
    assert ctrl._targets == []
    assert ctrl._id_to_pos == {}


# ===========================================================
# install / uninstall against a stubbed VeOmni hook surface
# ===========================================================


def _make_veomni_stub():
    # [FUNCTION DOC]
    # HELPER: Builds a mock of VeOmni's router replay hook module whose
    # set_active_replay stores the value in a captured mapping, get_active_replay
    # returns it, and validate_model_for_replay returns None.
    # Outputs: (stub module, captured mapping with key "active").
    """Return a (stub_module, captured_state) tuple where ``stub_module``
    is the fake ``veomni.utils.moe_router_replay`` and
    ``captured_state['active']`` records the last value passed to
    ``set_active_replay``."""
    stub = MagicMock()
    captured = {"active": None}

    def _set(x):
        # [FUNCTION DOC]
        # HELPER: Stores x as the active replay controller.
        captured["active"] = x

    stub.set_active_replay.side_effect = _set
    stub.get_active_replay.side_effect = lambda: captured["active"]
    stub.validate_model_for_replay.return_value = None
    return stub, captured


def test_install_uninstall_roundtrip(ctrl):
    # [FUNCTION DOC]
    # Behavior: with the stub installed in the module table, install registers the
    # controller as active and calls the model validator once; uninstall resets the
    # active slot to None.
    """install() registers the controller in VeOmni's global slot and
    runs the model validator; uninstall() clears the slot back to None."""
    stub, captured = _make_veomni_stub()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "veomni.utils.moe_router_replay", stub)
        ctrl.install(nn.Linear(1, 1))
        assert captured["active"] is ctrl
        stub.validate_model_for_replay.assert_called_once()
        ctrl.uninstall()
        assert captured["active"] is None


def test_install_without_veomni_raises(ctrl):
    # [FUNCTION DOC]
    # Behavior: with the hook module import blocked, install raises RuntimeError
    # mentioning "VeOmni build" instead of ImportError.
    """If the VeOmni hook surface is missing, install() must raise a
    typed RuntimeError pointing the user at the dependency, not a raw
    ImportError."""
    with pytest.MonkeyPatch.context() as mp:
        # Setting sys.modules[name] = None makes Python's import machinery
        # raise ImportError on `from veomni.utils.moe_router_replay import ...`
        # without needing to manipulate sys.path (which doesn't help when
        # veomni is installed in site-packages).
        mp.setitem(sys.modules, "veomni.utils.moe_router_replay", None)
        with pytest.raises(RuntimeError, match="VeOmni build"):
            ctrl.install(nn.Linear(1, 1))
