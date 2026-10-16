# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# SGLang custom weight loader that applies sparse/dense delta weight flushes from the
# delta checkpoint engine in place inside each SGLang TP worker (see module docstring for
# the wire contract). Module state: logger; CHUNK_BYTES (512 MiB cap per load_weights
# call); LOADER_FQN (import path registered as custom_weight_loader and used as
# load_format); _VERIFY_STATS (process-global accumulator for verify sweeps).
"""In-process sparse delta apply for SGLang, loaded via ``--custom-weight-loader``.

SGLang's ``update_weights_from_tensor`` supports pluggable loaders: when the
request's ``load_format`` names an import path registered in
``--custom-weight-loader``, SGLang ``dynamic_import``s it **inside every TP
worker process** and calls ``loader(model, named_tensors)``. That is exactly
the hook a sparse delta needs: the delta payload is decoded and applied *in
place* onto SGLang's live weights (masked overwrite of only the changed
positions), so the receiver never stages a full-model mirror anywhere —
peak memory is one decode chunk, independent of model size — and no SGLang
fork or patch is required.

Wire contract (what the delta checkpoint engine sends per flush):

* ``__delta_spec__`` — uint8 tensor holding a JSON manifest
  ``{"encoding", "params": [DeltaParam-dict...], "checksum"}``.
* ``__positions__`` — uint8 blob of packed positions (per-param slices are
  byte offsets ``pos_start:pos_end``; ``indices`` packs little-endian int32
  absolute positions).
* ``__values__``  — the changed values in the flush's (uniform) dtype;
  per-param slices are element offsets ``val_start:val_end``.

Register at server launch (verl config)::

    +actor_rollout_ref.rollout.engine_kwargs.sglang.custom_weight_loader='["verl.workers.rollout.sglang_rollout.delta_loader.apply_delta"]'
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

# Cap on the densified tensors handed to one model.load_weights call, matching
# SGLang's own delta-apply chunking default.
CHUNK_BYTES = 512 << 20

# Import path callers pass as both --custom-weight-loader and load_format.
LOADER_FQN = "verl.workers.rollout.sglang_rollout.delta_loader.apply_delta"


def apply_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    # [FUNCTION DOC]
    # Entry point invoked by SGLang as loader(model, named_tensors) for one flush.
    # Behavior: collects named tensors into a mapping; decodes the "__delta_spec__" uint8
    # tensor as a UTF-8 JSON manifest; takes "__values__" and optional "__positions__" (empty
    # uint8 tensor on the values' device when absent, e.g. the seed flush). Computes the
    # checksum over positions and values with the delta_sync encoder's checksum and compares
    # with the manifest. Encoding "dense": if spec "verify" is truthy runs the verification
    # sweep (passing "is_last"), else a plain chunked dense load. Any other encoding: under the
    # NaN-masked copy patch, densifies each param into a NaN-masked full tensor and feeds them
    # to model.load_weights in chunks bounded by CHUNK_BYTES (a single oversize tensor still
    # forms its own chunk).
    # Inputs: model - SGLang model exposing load_weights; named_tensors - iterable of
    # (name, tensor) pairs with the three reserved names.
    # Outputs: None.
    # Exceptions: KeyError if spec/values missing; RuntimeError on checksum mismatch;
    # ValueError from decoding for unsupported encodings.
    # Side effects: mutates model weights; temporarily patches torch.Tensor.copy_ globally.
    """Decode one sparse delta flush and masked-apply it onto ``model`` in place."""
    from verl.checkpoint_engine.delta_sync.encode import checksum as _checksum

    tensors = dict(named_tensors)
    spec = json.loads(bytes(tensors["__delta_spec__"].cpu().numpy().tobytes()).decode())
    values = tensors["__values__"]
    positions = tensors.get("__positions__")
    if positions is None:  # values-only flush (the seed) carries no positions
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)

    got = _checksum(positions, values)
    if got != int(spec["checksum"]):
        raise RuntimeError(
            f"delta checksum mismatch in sglang loader: got {got}, expected {spec['checksum']}; "
            "indicates corruption between sender encode and receiver apply"
        )

    if spec["encoding"] == "dense":
        if spec.get("verify"):
            _verify_dense(model, spec["params"], values, bool(spec.get("is_last")))
            return
        _apply_dense(model, spec["params"], values)
        return

    encoding = spec["encoding"]
    with _masked_copy():
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for p in spec["params"]:
            t = _decode_one(encoding, positions, values, p)
            nbytes = t.numel() * t.element_size()
            if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
                model.load_weights(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append((p["name"], t))
            chunk_bytes += nbytes
        if chunk:
            model.load_weights(chunk)


def _apply_dense(model: torch.nn.Module, params: list[dict], values: torch.Tensor) -> None:
    # [FUNCTION DOC]
    # HELPER: Load a dense flush: for each param dict, slice values[val_start:val_end], cast to
    # the named torch dtype, reshape to shape, and load into the model in chunks of at most
    # CHUNK_BYTES. Side effects: mutates model weights.
    """Apply a dense (full-coverage) flush: plain chunked load, no masking needed."""
    chunk: list[tuple[str, torch.Tensor]] = []
    chunk_bytes = 0
    for p in params:
        dtype = getattr(torch, p["dtype"])
        t = values[p["val_start"] : p["val_end"]].to(dtype).view(p["shape"])
        nbytes = t.numel() * t.element_size()
        if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
            model.load_weights(chunk)
            chunk, chunk_bytes = [], 0
        chunk.append((p["name"], t))
        chunk_bytes += nbytes
    if chunk:
        model.load_weights(chunk)


_VERIFY_STATS: dict = {"params": 0, "pieces": []}


def _verify_dense(model: torch.nn.Module, params: list[dict], values: torch.Tensor, is_last: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Verification sweep confirming that reloading the trainer's full weights is a
    # no-op on the server (see docstring for rationale).
    # Behavior: for each param, temporarily replaces torch.Tensor.copy_ with a wrapper that,
    # the first time each destination (keyed by data pointer, numel, dtype) is written,
    # snapshots a clone of it, then performs the real copy; runs the dense load for that one
    # param; restores copy_. Counts differing elements between each destination and its
    # snapshot (bit-level comparison via int16 view for 2-byte floats, so NaN/-0 differences
    # count), records per-param offender counts and per-param totals in _VERIFY_STATS. When
    # is_last, logs a warning summarizing params checked, mismatch total and top 12
    # offenders, resets the stats, and raises if any mismatches occurred.
    # Inputs: model; params - list of param dicts; values - value tensor; is_last - final
    # flush of the sweep.
    # Exceptions: RuntimeError on any mismatch at the end of the sweep.
    # Side effects: mutates model (idempotent if in sync), mutates global _VERIFY_STATS,
    # temporarily patches torch.Tensor.copy_, logs.
    """State-equivalence sweep, phrased as an IDEMPOTENCE check: replaying the
    trainer's FULL current weights onto an in-sync server must be a no-op. For
    each parameter we snapshot every ``copy_`` destination (the exact internal
    slices sglang's own ``load_weights`` name-mapping resolves) BEFORE the
    load, run the real load path -- including multi-stage loaders that first
    write raw values and then transform in place (e.g. mamba's
    ``A = -exp(A_log)`` composed loader) -- and bit-compare the post-load state
    against the snapshot. Any changed element means the server's
    delta-accumulated state disagreed with the trainer's; that fails loud.
    Comparing per copy_ call instead would false-positive on every
    transform-loaded param (raw-vs-transformed at each stage)."""
    orig_copy = torch.Tensor.copy_

    by_param = _VERIFY_STATS.setdefault("by_param", {})
    for p in params:
        touched: dict = {}

        def snap_then_copy_(self: torch.Tensor, src: torch.Tensor, *args, _touched=touched, **kwargs) -> torch.Tensor:
            # [FUNCTION DOC]
            # HELPER: Patched copy_ that snapshots the destination once per unique key into the
            # bound _touched mapping before delegating to the original copy_. Returns its result.
            key = (self.data_ptr(), self.numel(), self.dtype)
            if key not in _touched:
                _touched[key] = (self, self.detach().clone())
            return orig_copy(self, src, *args, **kwargs)

        torch.Tensor.copy_ = snap_then_copy_
        try:
            _apply_dense(model, [p], values)
        finally:
            torch.Tensor.copy_ = orig_copy
        bad = 0
        for dst, pre in touched.values():
            if dst.is_floating_point() and dst.element_size() == 2:
                bad += int((dst.view(torch.int16) != pre.view(torch.int16)).sum())
            else:
                bad += int((dst != pre).sum())
        if bad:
            by_param[p["name"]] = by_param.get(p["name"], 0) + bad
        _VERIFY_STATS["pieces"].append(bad)
    _VERIFY_STATS["params"] += len(params)
    if is_last:
        total = sum(_VERIFY_STATS["pieces"])
        n = _VERIFY_STATS["params"]
        _VERIFY_STATS["params"] = 0
        _VERIFY_STATS["pieces"] = []
        offenders = _VERIFY_STATS.pop("by_param", {})
        top = sorted(offenders.items(), key=lambda kv: -kv[1])[:12]
        logger.warning("DELTA-VERIFY sweep: params=%d mismatch_elems=%d offenders=%s", n, total, top)
        if total:
            raise RuntimeError(
                f"delta state verification FAILED: {total} elements differ between the "
                f"server's delta-accumulated weights and the trainer's full export; "
                f"top offenders: {top}"
            )


def _decode_one(encoding: str, positions: torch.Tensor, values: torch.Tensor, p: dict) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Densify one param's sparse delta into a full-shape tensor of the param's dtype
    # filled with NaN except at changed positions.
    # Behavior: returns all-NaN when the param has no values; otherwise slices the position
    # bytes pos_start:pos_end, reinterprets them as int32 indices ("indices" encoding only),
    # widens to int64 and scatters the cast values at those flat positions.
    # Inputs: encoding name; positions uint8 blob; values tensor; p - param dict (shape, dtype,
    # val_start/val_end, pos_start/pos_end).
    # Outputs: tensor of shape p["shape"] on values' device.
    # Exceptions: ValueError for any encoding other than "indices".
    """Densify one param's sparse delta into a full-shape NaN-masked tensor.

    ``indices`` positions are reinterpreted via an int32 view (8 B/element
    transient) rather than a per-byte int64 unpack (32 B/element), so even a
    full-seed flush of a large embedding stays within a few GiB.
    """
    numel = math.prod(p["shape"])
    dtype = getattr(torch, p["dtype"])
    flat = torch.full((numel,), float("nan"), dtype=dtype, device=values.device)
    vals = values[p["val_start"] : p["val_end"]]
    if vals.numel() == 0:
        return flat.view(p["shape"])

    pos_b = positions[p["pos_start"] : p["pos_end"]]
    if encoding == "indices":
        # pos_start is always int32-aligned (each entry packs nnz * 4 bytes).
        idx = pos_b.view(torch.int32).to(torch.int64)
    else:
        raise ValueError(f"unsupported delta encoding: {encoding!r}")

    flat.index_copy_(0, idx, vals.to(dtype))
    return flat.view(p["shape"])


@contextmanager
def _masked_copy() -> Iterator[None]:
    # [FUNCTION DOC]
    # HELPER: Context manager that patches torch.Tensor.copy_ so that, when the source is a
    # floating tensor of the same shape as the destination, positions where the source (cast
    # to destination dtype) is NaN keep the destination's value; otherwise falls through to
    # the original copy_. Restores the original on exit, even on error.
    # Outputs: yields None. Side effects: global monkey-patch of torch.Tensor.copy_ for the
    # duration (affects all threads in the process).
    """Temporarily make ``Tensor.copy_`` skip NaN positions in the source.

    SGLang's per-model ``load_weights`` ultimately lands on ``param.copy_(loaded)``
    (possibly on a narrowed TP slice). Under this context a NaN-masked source
    overwrites only the changed positions; fully dense sources (e.g. the first
    full-seed flush) take the original fast path untouched.
    """
    orig_copy = torch.Tensor.copy_

    def masked_copy_(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Masked copy_ replacement; uses an elementwise select (no host sync) to keep
        # destination values where the casted source is NaN. Extra args are dropped on the masked
        # path and forwarded on the fallback path.
        # Sync-free masked overwrite: boolean advanced indexing (and a
        # ``mask.all()`` early-out) would force a device->host sync per
        # parameter -- ruinous for MoE flushes carrying >10k per-expert
        # entries. ``torch.where`` keeps everything on-stream; a NaN-free
        # (dense) source degenerates to a plain copy.
        if isinstance(src, torch.Tensor) and src.is_floating_point() and self.shape == src.shape:
            cast = src.to(self.dtype)
            return orig_copy(self, torch.where(torch.isnan(cast), self, cast))
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy
