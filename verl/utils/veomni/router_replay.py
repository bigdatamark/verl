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
# MoE router record/replay controller for the VeOmni engine (see the module docstring for the
# lifecycle, layer-indexing contract and handling of positions without recorded routing). Exposes
# the RouterReplayAction enum and the VeOmniRouterReplay state machine.

"""VeOmni-flavored MoE router replay.

Record/replay controller for verl's VeOmni engine. Tapped into VeOmni's
MoE ``SparseMoeBlock.forward`` via
``veomni.utils.moe_router_replay.set_active_replay`` (a module-level
singleton slot); the patched forward calls
``maybe_replay_indices(module, routing_scores, top_indices)`` on each MoE
layer, which delegates to :meth:`VeOmniRouterReplay.on_router_forward`.

Scope
-----
The controller holds **one micro-batch** worth of routing indices and
knows nothing about padding, sequence parallelism, or batch layout.
Everything shape-related lives in the engine, which already does the
same work for ``input_ids`` / ``log_probs``:

* REPLAY targets are padded and Ulysses-sliced in
  ``VeOmniEngineWithLMHead.prepare_model_inputs``, right after
  ``super().prepare_model_inputs`` applied that same rule to
  ``input_ids``.
* RECORD output is all-gathered, unpadded and re-wrapped per sample in
  ``VeOmniEngineWithLMHead.prepare_model_outputs``, exactly like
  ``log_probs``.

Lifecycle::

    replay.begin_record()          # R2 compute_log_prob
        or replay.begin_replay()   # R2 actor update, or R3 everywhere

    # per micro-batch, in prepare_model_inputs:
    replay.begin_microbatch()                      # RECORD
    replay.begin_microbatch(targets=per_layer)     # REPLAY

    # per micro-batch, in prepare_model_outputs:
    indices = replay.take_recorded()               # RECORD

    replay.clear()                                 # after the step

Layer indexing
--------------
Positions are assigned by ``id(module)`` on first fire within a
micro-batch. Under activation checkpointing, backward recompute fires the
same routers again in arbitrary order (per-layer checkpoint segments run
from L-1 down to 0), so the id lookup is what keeps each layer on its own
position. It also makes recompute detection free: a module that is
already in the table is by definition not firing for the first time, so
RECORD simply skips it.

Fire order is only meaningful because of the contract it implies: slot
``i`` of a ``routed_experts`` tensor must belong to the ``i``-th router to
fire. Rollout backends fill that tensor by **absolute decoder-layer
index** over all ``num_hidden_layers`` (vLLM sizes its capture buffer that
way and writes at ``extract_layer_index(layer_name)``), so R3 replay is
only correct when every decoder layer contributes exactly one hooked
router. A model that hooks a subset -- e.g. DeepSeek-V4, whose first three
layers are hash-routed -- silently shifts every layer's target unless its
skipped layers are hooked too. :meth:`num_fired` vs :meth:`num_targets`
lets the engine assert the contract after each forward instead of
mis-routing quietly.

Positions without recorded routing
----------------------------------
Some positions the routers are asked to route have no recorded routing:
the pad suffix the engine appends, and the trailing rows the rollout
backend leaves at zero in R3. They all arrive as all-zero target rows,
and :meth:`VeOmniRouterReplay.on_router_forward` routes any row with a
duplicate top-k natively -- which covers them without a separate mask.
The engine rejects topk=1 models, where that check cannot fire.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


__all__ = ["RouterReplayAction", "VeOmniRouterReplay"]


class RouterReplayAction(Enum):
    # [CLASS DOC]
    # Enum of controller modes: DISABLED (pass-through), RECORD (capture router top-k indices),
    # REPLAY (substitute supplied target indices).
    DISABLED = "disabled"
    RECORD = "record"
    REPLAY = "replay"


class VeOmniRouterReplay:
    # [CLASS DOC]
    # Stateful controller holding one micro-batch of routing. State: current action; a mapping from
    # router module identity to layer position (rebuilt per micro-batch); recorded per-layer index
    # tensors (RECORD); per-layer targets (REPLAY); a debug flag enabled when env var
    # VERL_ROUTER_REPLAY_DEBUG equals "1"; and whether it is installed into VeOmni's global hook slot.
    """Per-micro-batch router replay controller for VeOmni.

    See the module docstring for the lifecycle and for the division of
    labour between this controller and the engine.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC]
        # Behavior: initializes action DISABLED, empty position table, recorded and target lists, reads the
        # debug env flag, and marks not installed. Side effects: reads environment.
        self._action: RouterReplayAction = RouterReplayAction.DISABLED
        # id(router_module) -> position, rebuilt every micro-batch. Absence
        # from this table is what identifies a router's first fire.
        self._id_to_pos: dict[int, int] = {}
        # RECORD: one [nnz, topk] tensor per layer position, in fire order.
        self._recorded: list[torch.Tensor] = []
        # REPLAY: one [nnz, topk] target per layer position.
        self._targets: list[torch.Tensor] = []
        # Env-gated shape sanity check.
        self._debug: bool = os.environ.get("VERL_ROUTER_REPLAY_DEBUG") == "1"
        self._installed: bool = False

    @property
    def action(self) -> RouterReplayAction:
        # [FUNCTION DOC]
        # Behavior: returns the current action. Outputs: RouterReplayAction.
        return self._action

    @property
    def num_fired(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns the number of distinct routers that fired since the last reset.
        """How many distinct routers have fired since ``begin_microbatch``."""
        return len(self._id_to_pos)

    @property
    def num_targets(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns the number of REPLAY targets armed for this micro-batch.
        """How many layer targets REPLAY was armed with for this micro-batch."""
        return len(self._targets)

    def install(self, model: nn.Module) -> None:
        # [FUNCTION DOC]
        # Behavior: imports VeOmni's set_active_replay and validate_model_for_replay, validates the model
        # is wired for replay, registers self as the active replay controller, and marks installed.
        # Inputs: model - module containing MoE layers. Outputs: None.
        # Exceptions: RuntimeError (chained) if VeOmni lacks the hook module; whatever validation raises.
        # Side effects: mutates VeOmni's global replay slot.
        """Register this controller with VeOmni's global ``set_active_replay`` slot.

        After returning, every MoE router forward in ``model`` should reach
        :meth:`on_router_forward` via ``maybe_replay_indices``.
        """
        try:
            from veomni.utils.moe_router_replay import set_active_replay, validate_model_for_replay  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "router_replay.mode != 'disabled' requires a VeOmni build that "
                "exposes `veomni.utils.moe_router_replay.set_active_replay`. "
                "Either upgrade VeOmni or set router_replay.mode='disabled'."
            ) from e
        # Fail fast if this model family has not been wired for replay
        # (would otherwise surface as a cryptic mid-forward error).
        validate_model_for_replay(model)
        set_active_replay(self)
        self._installed = True

    def uninstall(self) -> None:
        # [FUNCTION DOC]
        # Behavior: if installed, clears VeOmni's active replay slot (ignoring ImportError) and marks not
        # installed; otherwise no-op. Side effects: mutates VeOmni global slot.
        """Reverse :meth:`install`. Idempotent."""
        if not self._installed:
            return
        try:
            from veomni.utils.moe_router_replay import set_active_replay  # type: ignore

            set_active_replay(None)
        except ImportError:
            pass
        self._installed = False

    def on_router_forward(
        self,
        module: nn.Module,
        routing_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: DISABLED returns top_indices unchanged. Otherwise assigns the module a position by
        # identity on first fire in this micro-batch. If debug is on, asserts both tensors are 2-D with
        # matching row counts. RECORD: on first fire appends a detached int16 copy of top_indices; always
        # returns top_indices. REPLAY: looks up the target at the module's position, checks it exists and has
        # the same shape as top_indices, then per row returns the native top_indices where the target row
        # contains any duplicate expert id (detected by sorting and comparing neighbours), else the target row.
        # Inputs: module - router module; routing_scores - [rows, experts] scores (debug only);
        # top_indices - [rows, topk] native choices.
        # Outputs: [rows, topk] index tensor.
        # Exceptions: AssertionError on debug shape mismatch; RuntimeError when a REPLAY target is missing
        # or shape mismatched.
        # Side effects: mutates position table and recorded list.
        """Called from each MoE router forward via VeOmni's hook surface.

        Indices-only: records ``top_indices`` in RECORD mode or returns
        substituted target indices in REPLAY mode. All model-specific
        post-topk weight math (gather, renorm, scaling, dtype cast) lives
        in the per-family patched ``SparseMoeBlock.forward``, not here --
        that keeps the controller model-agnostic across MoE families
        (softmax/sigmoid gating, group topk, expert bias, scaling factors,
        etc.). ``routing_scores`` is accepted for optional debug inspection
        but NOT used to derive weights.
        """
        if self._action is RouterReplayAction.DISABLED:
            return top_indices

        mid = id(module)
        pos = self._id_to_pos.get(mid)
        first_fire = pos is None
        if first_fire:
            pos = len(self._id_to_pos)
            self._id_to_pos[mid] = pos

        if self._debug:
            # Cheap shape sanity check -- cross-family safe (no weight math).
            if routing_scores.dim() != 2 or top_indices.dim() != 2:
                raise AssertionError(
                    f"router_replay: expected 2D tensors, got routing_scores "
                    f"{tuple(routing_scores.shape)} and top_indices "
                    f"{tuple(top_indices.shape)}."
                )
            if routing_scores.shape[0] != top_indices.shape[0]:
                raise AssertionError(
                    f"router_replay: routing_scores / top_indices row count "
                    f"mismatch: {routing_scores.shape[0]} vs {top_indices.shape[0]}."
                )

        if self._action is RouterReplayAction.RECORD:
            # Only the first fire records. Backward recompute under activation
            # checkpointing re-enters with an already-mapped module, so it is
            # skipped without needing a separate counter. The snapshot is
            # detached so it outlives the autograd graph that produced it.
            if first_fire:
                self._recorded.append(top_indices.detach().to(torch.int16, copy=True))
            return top_indices

        # REPLAY. Strict: every layer position must have a target. There is no
        # silent fallback -- a missing target indicates a real plumbing bug
        # (routed_experts not propagated, layer count mismatch between the
        # RECORD and REPLAY models, or the engine forgot to call
        # begin_microbatch before this forward).
        if pos >= len(self._targets):
            raise RuntimeError(
                f"router_replay REPLAY: layer pos={pos} has no target "
                f"(only {len(self._targets)} targets set for this "
                "micro-batch). Likely cause: model has more MoE layers "
                "than the recorded routed_experts tensor describes, or "
                "begin_microbatch was not called before forward."
            )
        target = self._targets[pos]
        if target.shape != top_indices.shape:
            raise RuntimeError(
                f"router_replay REPLAY: layer pos={pos} target has shape "
                f"{tuple(target.shape)} but the router produced "
                f"{tuple(top_indices.shape)}. The engine must pad and Ulysses-slice "
                "the targets with the same rule it applies to input_ids."
            )

        # Defensive duplicate-detection.
        #
        # VeOmni's MoE expert dispatch (``permute()`` in
        # ``veomni/distributed/moe/moe_utils.py``) builds the permuted
        # tensor via ``routing_map.bool().masked_select(...)``, which
        # collapses duplicate top-k slots within one token to a single
        # entry. ``input_splits`` keeps counting every slot, so the two
        # diverge whenever ANY token has duplicate top-k experts and the
        # EP all-to-all crashes with
        # ``RuntimeError: Split sizes doesn't match total dim 0 size``.
        #
        # This is also the *only* mechanism protecting positions that have no
        # recorded routing at all: the pad suffix the engine appends, and the
        # trailing rows the rollout backend leaves at zero in R3 (the final
        # generated token, tool-response tokens in a multi-turn trajectory).
        # All of those are all-zero rows, hence duplicates. It is why the
        # engine rejects topk=1 models, where a single slot can never be a
        # duplicate and those rows would silently all route to expert 0.
        #
        # Native indices are always distinct top-k choices, so falling back to
        # them is correct regardless of what the rollout produced.
        sorted_target, _ = target.sort(dim=-1)
        has_duplicate = (sorted_target[:, 1:] == sorted_target[:, :-1]).any(dim=-1)
        return torch.where(has_duplicate.unsqueeze(-1), top_indices, target)

    def begin_record(self) -> None:
        # [FUNCTION DOC]
        # Behavior: sets action RECORD and resets micro-batch state.
        """Enter RECORD mode for a step."""
        self._action = RouterReplayAction.RECORD
        self._reset_microbatch()

    def begin_replay(self) -> None:
        # [FUNCTION DOC]
        # Behavior: sets action REPLAY and resets micro-batch state.
        """Enter REPLAY mode for a step."""
        self._action = RouterReplayAction.REPLAY
        self._reset_microbatch()

    def clear(self) -> None:
        # [FUNCTION DOC]
        # Behavior: sets action DISABLED and resets micro-batch state.
        """Reset the state machine between steps."""
        self._action = RouterReplayAction.DISABLED
        self._reset_microbatch()

    def _reset_microbatch(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Clears the position table, recorded list and targets list.
        self._id_to_pos = {}
        self._recorded = []
        self._targets = []

    def begin_microbatch(self, targets: list[torch.Tensor] | None = None) -> None:
        # [FUNCTION DOC]
        # Behavior: validates arguments against the mode (RECORD forbids targets, REPLAY requires them,
        # any other mode is an error), resets micro-batch state, and stores a copy of the targets list.
        # Inputs: targets - optional list of per-layer [nnz, topk] tensors.
        # Outputs: None. Exceptions: RuntimeError on mode/argument mismatch.
        """Arm the controller for one micro-batch forward.

        Must be called from the engine before every forward, once RECORD or
        REPLAY is active. Drops the previous micro-batch's state, so a
        router's first fire after this call is unambiguous -- which is what
        makes recompute detection work without a counter.

        RECORD takes no arguments. REPLAY requires ``targets``, a list of
        ``[nnz, topk]`` int64 device tensors ordered by layer position (the
        order routers fire in during forward). ``nnz`` must match what the
        routers will see on this rank, i.e. the engine has already applied
        the same padding and Ulysses slicing it applied to ``input_ids``.
        """
        if self._action is RouterReplayAction.RECORD:
            if targets is not None:
                raise RuntimeError("begin_microbatch: RECORD does not take targets.")
        elif self._action is RouterReplayAction.REPLAY:
            if targets is None:
                raise RuntimeError("begin_microbatch: REPLAY requires per-layer targets.")
        else:
            raise RuntimeError(f"begin_microbatch requires RECORD or REPLAY action, got {self._action}")

        self._reset_microbatch()
        if targets is not None:
            self._targets = list(targets)

    def take_recorded(self) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: requires RECORD mode and at least one recorded layer; stacks per-layer [nnz, topk]
        # int16 tensors along a new dim 1 to [nnz, layers, topk]. Does not clear state.
        # Exceptions: RuntimeError if not RECORD or nothing recorded.
        """Return this micro-batch's routing as ``[nnz, L, topk]``.

        Layer-major stacking matches the trainer-side ``routed_experts``
        layout, so the engine can hand the result straight to
        ``_gather_and_unpad_packed`` + ``nested_tensor_from_jagged`` the way
        it does for ``log_probs``.
        """
        if self._action is not RouterReplayAction.RECORD:
            raise RuntimeError(f"take_recorded requires RECORD action, got {self._action}")
        if not self._recorded:
            raise RuntimeError(
                "router_replay RECORD: no router fired during this forward. "
                "Either the model has no MoE layers or its SparseMoeBlock is "
                "not wired to `veomni.utils.moe_router_replay.maybe_replay_indices`."
            )
        return torch.stack(self._recorded, dim=1)
