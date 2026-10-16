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
# Megatron delta-export machinery: builds a per-parameter export index from Megatron-Bridge conversion tasks, creates communication-free "probe" copies of param mappings (size-faithful fake groups, stubbed gather helpers), exchanges per-row HF slot tables across ranks, and converts a shard-local sparse delta to HF-coordinate (slot, index, value) entries via NaN sentinels.
# Module state: a pool of read-only NaN tensors keyed by (shape, dtype, device), kept for the process lifetime.
"""Megatron-side delta export machinery, built on Megatron-Bridge param mappings.

The delta engine consumes final HF-coordinate entries; everything mcore-specific
lives here: enumerating parameters through :meth:`AutoBridge.get_conversion_tasks`,
routing each parameter's entries to its wire merge group, and probing the
bridge's own ``megatron_to_hf`` converters with NaN sentinels to translate a
shard-local delta into HF coordinates.

The probe never runs a collective, yet executes the mapping's REAL parallel
code paths: a "group" carries two separable meanings -- its SIZE/RANK (which
value math like Mamba's ``local_dim = global // tp_size`` consumes) and its
COMMUNICATION. The probe copies keep the first faithful and replace only the
second: groups become :class:`_ProbeGroup` (real size/rank, any actual use for
communication raises), and the bridge's comm helpers are stubbed with local
synthesis -- ``gather_from_tp_ranks`` returns this rank's shard at its true
rank index with NaN placeholders for every other rank (their contributions are
exported by those ranks' own probes). Feeding ``megatron_to_hf`` the LOCAL
shard as a NaN buffer with the rank's own delta scattered in yields HF tensors
whose non-NaN survivors are exactly this rank's contributions in final HF
coordinates.

Scope (asserted in the exporter): TP + EP + PP/VPP, no LoRA. Under PP the
bridge's conversion tasks already enumerate the GLOBAL parameter directory
(identical order on every rank, tied embeddings deduped, placeholders for
other stages); non-owner ranks ship zero-count lockstep rows and every
param merges over the WORLD group, so the wire master needs no relay.

Safety boundary: communication must stay confined to the four stubbed helpers
(``gather_from_tp_ranks`` / ``gather_from_ep_ranks[_scale]`` / the PP
broadcasts, whose ``pp_size == 1`` fast path the probe keeps) -- anything else
touching a probe group raises instead of skewing silently. The remaining
assumption is that transforms REARRANGE elements rather than arithmetically
BLEND chunks (blending would eat the NaN sentinels); the TP>1 differential
test (real ``megatron_to_hf`` vs probe assembly, bitwise) is the regression
oracle for both assumptions on every Megatron-Bridge upgrade.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from verl.workers.engine.spec import ShardSpec

logger = logging.getLogger(__name__)


class _ProbeGroup:
    # [CLASS DOC]
    # HELPER: Fake process group exposing only size and rank; any other attribute access raises RuntimeError to catch unstubbed communication.
    """Size/rank-faithful stand-in for a process group on probe copies: value
    math inside ``megatron_to_hf`` (e.g. Mamba's per-rank de-interleave dims)
    sees the REAL parallel sizes, while any attempt to actually communicate
    through the group fails loud (the probe stubs the bridge's comm helpers;
    anything else reaching a group is an unstubbed communication pattern)."""

    def __init__(self, size: int, rank: int):
        # [FUNCTION DOC]
        # Stores size and rank as ints.
        self._size = int(size)
        self._rank = int(rank)

    def size(self) -> int:
        # [FUNCTION DOC]
        # Returns the stored group size.
        return self._size

    def rank(self) -> int:
        # [FUNCTION DOC]
        # Returns the stored rank.
        return self._rank

    def __getattr__(self, name):
        # [FUNCTION DOC]
        # Called only for missing attributes; always raises RuntimeError naming the attribute (signals communication outside the stubbed helpers).
        raise RuntimeError(
            f"probe process group asked for {name!r}: this mapping communicates outside "
            "the stubbed helpers (gather_from_tp_ranks / gather_from_ep_ranks[_scale] / "
            "pp broadcasts) -- extend make_probe's comm stubs before trusting its export"
        )


_NAN_POOL: dict = {}


def _nan_block(shape, dtype, device) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns a cached all-NaN tensor for (shape, dtype, device), creating and pooling it on first request. Callers must treat it as read-only.
    # Side effects: grows the module-level pool (device memory).
    """Read-only all-NaN placeholder for another rank's gather chunk, pooled by
    (shape, dtype, device): the gather stub hands these out on every probe call
    for every param, and they are only ever READ (the transforms are functional),
    so one block per distinct shape serves the whole model for the run's
    lifetime instead of a fresh cudaMalloc+fill per param per sync."""
    key = (tuple(shape), dtype, str(device))
    t = _NAN_POOL.get(key)
    if t is None:
        t = torch.full(tuple(shape), float("nan"), dtype=dtype, device=device)
        _NAN_POOL[key] = t
    return t


def _warm_lazy_mappings(mapping, module) -> None:
    # [FUNCTION DOC]
    # HELPER: Eagerly resolves an AutoMapping's lazy concrete delegate before probing.
    # Behavior: if the mapping has the detection method and no resolved delegate, detects the parallelism type from the module, creates the delegate and records the detected type; failures are logged as warnings and swallowed. Only the mapping itself is warmed, not children.
    # Side effects: mutates the mapping object.
    """Force AutoMapping's lazy concrete delegate into existence: AutoMapping
    resolves its Column/Row/Replicated delegate on first use, snapshotting
    whatever process groups exist at that moment -- if that first use happened
    inside the probe, the delegate would be born with the REAL groups and
    gather for real (the qkv-bias double-size bug). Self-only on purpose:
    ``_inject`` warms every child right before copying it, so each node of the
    mapping tree is warmed exactly once."""
    if hasattr(mapping, "_detect_parallelism_type") and getattr(mapping, "_mapping", None) is None:
        try:
            t = mapping._detect_parallelism_type(module)
            mapping._mapping = mapping._get_or_create_mapping(t)
            mapping._detected_type = t
        except Exception as e:  # pragma: no cover - defensive; probe falls back to real groups
            logger.warning("could not warm lazy mapping %s: %s", type(mapping).__name__, e)


def make_probe(mapping, module):
    # [FUNCTION DOC]
    # Creates a communication-free probe copy of a Megatron-Bridge mapping tree.
    # Behavior: warms the root's lazy delegate, then recursively shallow-copies the mapping: on each copy sets pp_group to a (1, 0) probe group, replaces ep_group, _tp_group and _etp_group with probe groups carrying the original groups' real size/rank, replaces gather_from_tp_ranks with a function returning a list of tp_size entries (pooled NaN blocks, own tensor at tp_rank), gather_from_ep_ranks with a function returning a one-entry mapping from the bound name to the tensor, and gather_from_ep_ranks_scale likewise after unsqueeze/squeeze/unsqueeze(-1) reshaping. Child attributes that are MegatronParamMapping instances are warmed and recursively replaced by probe copies.
    # Inputs: mapping - Megatron-Bridge param mapping; module - Megatron module the mapping reads.
    # Outputs: the probe mapping copy (original untouched aside from warming).
    """Copy a Megatron-Bridge param mapping tree and turn ``megatron_to_hf``
    into a communication-free LOCAL transform that still runs the REAL
    (tp_size > 1) code paths: every copy's groups become size-faithful
    :class:`_ProbeGroup` stand-ins and the bridge's comm helpers are stubbed
    with local synthesis. The copy is recursive (composite mappings delegate to
    inner mappings -- QKVMapping._tp_mapping is an AutoMapping which itself
    delegates to a lazily-created concrete mapping; none of these receive the
    outer stubbing on their own)."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.core.utils import get_pg_rank, get_pg_size

    _warm_lazy_mappings(mapping, module)

    def _stub(c):
        # [FUNCTION DOC]
        # HELPER: Mutates a mapping copy in place to use probe groups and local gather stubs (see make_probe) and returns it.
        # size-faithful groups, sizes/ranks read from the ORIGINAL (real) groups
        # BEFORE replacement. The probe only ever runs on the stage that OWNS
        # the param, where a pp broadcast is the identity -- a (1, 0) group
        # keeps the pp helpers on that fast path regardless of the real pp size.
        c.pp_group = _ProbeGroup(1, 0)
        for attr in ("ep_group", "_tp_group", "_etp_group"):
            g = getattr(c, attr, None)
            setattr(c, attr, _ProbeGroup(get_pg_size(g), get_pg_rank(g)))

        def _gather_tp(tensor, _c=c):
            # [FUNCTION DOC]
            # HELPER: Local stand-in for a TP all-gather: list of length tp_size with NaN blocks and the given tensor at this rank's index.
            # what a real all_gather over the tp group would produce, minus the
            # other ranks' data: our shard rides at its true rank index, every
            # other chunk is a pooled read-only NaN block (those ranks export
            # their own contributions; transforms never write into their inputs,
            # which the TP>1 bitwise differential revalidates).
            nan = _nan_block(tensor.shape, tensor.dtype, tensor.device)
            out = [nan] * _c.tp_size
            out[_c.tp_rank] = tensor
            return out

        c.gather_from_tp_ranks = _gather_tp
        # each rank's probe emits only its own local experts, under the hf name
        # bound with the GLOBAL expert id at task construction; the engine
        # merges entries over the etp x ep group, so the ep fan-out reduces to
        # the bound name.
        c.gather_from_ep_ranks = lambda w, mod, name: {str(name): w}
        # mirrors the real helper's tail (unsqueeze(0) ... squeeze().unsqueeze(-1))
        c.gather_from_ep_ranks_scale = lambda w, mod, name: {str(name): w.unsqueeze(0).squeeze().unsqueeze(-1)}
        return c

    def _inject(m):
        # [FUNCTION DOC]
        # HELPER: Stubs a shallow copy of a mapping and recursively replaces its child mappings with probe copies; returns the stubbed copy.
        c = _stub(copy.copy(m))
        for attr, value in list(vars(c).items()):
            if isinstance(value, MegatronParamMapping):
                # warm BEFORE copying the child: the lazy delegate must exist so
                # the child copy's vars() include it and the recursion reaches it.
                _warm_lazy_mappings(value, module)
                setattr(c, attr, _inject(value))
        return c

    return _inject(mapping)


@dataclass
class McoreParamExport:
    # [CLASS DOC]
    # Dataclass export record for one Megatron parameter: megatron_name, param (None for params owned by another pipeline stage), spec (ShardSpec routing info), probe mapping, module handle, and slots (global (hf_name, hf_shape) table, filled by the slot exchange).
    """One mcore parameter's export record: geometry + probe + module handle.
    ``param is None`` marks a parameter owned by another pipeline stage: the
    rank ships a zero-count lockstep row for it (probe/module unused)."""

    megatron_name: str
    param: Optional[torch.Tensor]
    spec: ShardSpec
    probe: Any  # comm-stubbed mapping copy (local megatron_to_hf evaluator)
    module: Any  # module handle megatron_to_hf reads config from
    # GLOBAL slot table for this directory row (set by the index-build
    # exchange): identical on every rank of the row's merge group.
    slots: Optional[list] = None


def build_export_index(bridge, megatron_model, slot_cache: dict | None = None) -> list[McoreParamExport]:
    # [FUNCTION DOC]
    # Builds the export index of all parameters in global, rank-identical order.
    # Behavior: iterates over bridge conversion tasks. Tasks with no param are skipped when PP is 1, else recorded as non-contributing lockstep rows with WORLD as gather group. For owned params: expert-ness requires mapping.is_expert and (EP or TP above 1); TP-sharded means the param's tensor_model_parallel flag and TP above 1. With PP above 1 all rows merge over WORLD and contributes is: expert-DP rank 0 for experts; DP(with CP) rank 0 for TP-sharded; TP rank 0 and DP rank 0 otherwise. With PP 1: experts gather over the expert tensor-and-model-parallel group; TP-sharded over the TP group; others are unsharded specs. Each owned row gets a probe. Then runs the slot-table exchange and drops (with a warning) rows that have no slots on any rank.
    # Inputs: bridge - AutoBridge; megatron_model - model chunks; slot_cache - mapping from param name to slot list, reused across calls.
    # Outputs: list of McoreParamExport.
    # Side effects: collective all_gather_object over WORLD; populates slot_cache; logs.
    # KNOWN ISSUE: slot_cache defaults to None, but the slot exchange performs membership tests on it, so calling without a cache raises TypeError. The docstring also references a nonexistent preseed-slot-tables function; the work is done by the slot exchange.
    """Enumerate every mcore parameter through the bridge's conversion tasks
    and precompute its probe + wire routing.

    No shard geometry is hand-computed here: the comm-stubbed probe emits final
    HF coordinates straight from the local shard, so the engine only needs to
    know WHICH group's entries to merge per parameter (the spec's
    ``gather_group``; ``full_shape`` is the nominal local shape) and which
    ranks contribute. The index is built once (parameter sets are static) and
    reused by both the shard export and the delta entry hook.

    Lockstep: ``get_conversion_tasks`` already enumerates the GLOBAL parameter
    list in an order identical on every rank (the bridge allgathers and sorts
    names across pp ranks, dedupes tied embeddings, and leaves
    ``param_weight=None`` placeholders for parameters owned by other pipeline
    stages). Under PP>1 those placeholders become zero-count lockstep rows and
    every param's entries merge over the WORLD group (owner-stage ranks
    contribute, dp/cp replicas and other stages stay empty); their slot tables
    are pre-seeded once via :func:`_preseed_slot_tables`. Under PP=1 the
    routing is unchanged (tp / etp x ep subgroups, replicated rank-0 direct).
    """
    from megatron.core import parallel_state as mpu

    pp_world = mpu.get_pipeline_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    world = torch.distributed.group.WORLD

    tasks = bridge.get_conversion_tasks(megatron_model)
    index: list[McoreParamExport] = []
    for task in tasks:
        mapping = task.mapping
        param = task.param_weight
        name = task.global_param_name
        if param is None:
            if pp_world == 1:
                # not a pp placeholder (e.g. a skipped mapping); nothing to ship
                continue
            # owned by another pipeline stage: zero-count lockstep row -- the
            # rank walks the same global directory and contributes nothing.
            index.append(
                McoreParamExport(
                    megatron_name=name,
                    param=None,
                    spec=ShardSpec(full_shape=(0,), place=0, gather_group=world, contributes=False),
                    probe=None,
                    module=None,
                )
            )
            continue
        module = task.megatron_module
        local_shape = tuple(int(x) for x in param.shape)

        is_expert = mapping.is_expert and (ep_size > 1 or tp_world > 1)
        is_tp_sharded = getattr(param, "tensor_model_parallel", False) and tp_world > 1

        if pp_world > 1:
            # single WORLD merge group for every param: the wire master (global
            # rank 0) sits in every gather regardless of which stage owns the
            # param, so no relay hop is needed. Owner-stage ranks contribute
            # their shard pieces; dp/cp replicas dedupe via ``contributes``
            # (identical copies -- exactly one replica set ships).
            if is_expert:
                contributes = mpu.get_expert_data_parallel_rank() == 0
            elif is_tp_sharded:
                contributes = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            else:
                contributes = (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                )
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=world, contributes=contributes)
        elif is_expert:
            # every rank holding a piece of this expert set contributes; the
            # engine merges their probe entries over the joint etp x ep group.
            spec = ShardSpec(
                full_shape=local_shape,
                place=0,
                gather_group=mpu.get_expert_tensor_and_model_parallel_group(),
            )
        elif is_tp_sharded:
            spec = ShardSpec(full_shape=local_shape, place=0, gather_group=tp_group)
        else:
            # replicated: engine's pg=None path (rank 0 consumes its own entry
            # directly, replicas stay in lockstep via zero counts).
            spec = ShardSpec(full_shape=local_shape)

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                spec=spec,
                probe=make_probe(mapping, module),
                module=module,
            )
        )

    _exchange_slot_tables(index, slot_cache)
    # a row with an empty union has no owner on ANY rank: the bridge skipped
    # it everywhere (no mapping / missing HF key), so it is outside the
    # bridge's own export scope -- exactly what PP=1 does by skipping
    # param_weight=None tasks. Drop it (all ranks agree: the exchange result
    # is identical) instead of shipping unsized rows.
    unowned = [rec.megatron_name for rec in index if rec.slots is None]
    if unowned:
        logger.warning(
            "delta export drops %d row(s) the bridge skipped on every rank "
            "(no mapping or missing HF key; out of export scope): %s",
            len(unowned),
            unowned[:8],
        )
        index = [rec for rec in index if rec.slots is not None]
    return index


def _exchange_slot_tables(index: list[McoreParamExport], slot_cache: dict) -> None:
    # [FUNCTION DOC]
    # HELPER: Collectively computes the global per-row slot tables.
    # Behavior: for each owned row without a cached slot list, probes an empty delta (int64 indices, bf16 values) to populate the cache; builds a per-row local list (None for non-owned rows), all-gathers the lists over WORLD, asserts all ranks have the same row count, and sets each record's slots to the ordered first-seen union of (name, shape) across ranks in rank order, or None if empty.
    # Inputs: index - records (mutated); slot_cache - name-to-slot-list mapping (mutated).
    # Exceptions: AssertionError on diverging row counts.
    # Side effects: collective communication; mutates records and cache.
    """One-time GLOBAL slot-table exchange, run at every world size, keyed by
    DIRECTORY ROW (not by param name).

    The engine's batched gather merges entries BY SLOT POSITION within each
    directory row and rank 0 names the merged pieces from ITS OWN list, so
    the per-row list must be identical on every rank of the row's merge
    group. Name-keyed merging is not enough on mcore: expert param NAMES
    embed the expert ids (``...experts.linear_fc1.weight0`` on ep rank 0 vs
    ``weight1`` on ep rank 1), so the same row carries different names per
    rank while still gathering together. Each rank therefore probes a zero
    delta through its OWNED rows to reveal its local ``(hf_name, hf_shape)``
    list, one ``all_gather_object`` over WORLD exchanges the ordered row
    lists, and every row's table becomes the rank-order first-seen union --
    identical everywhere by construction. Dense/TP rows (same list on every
    rank) dedup to themselves; expert rows concatenate the per-ep-rank
    lists; PP placeholder rows inherit the owners' union."""
    local_rows: list = []
    for rec in index:
        if rec.param is None:
            local_rows.append(None)
            continue
        if rec.megatron_name not in slot_cache:
            empty_idx = torch.empty(0, dtype=torch.int64, device=rec.param.device)
            empty_val = torch.empty(0, dtype=torch.bfloat16, device=rec.param.device)
            mcore_hf_delta_entry(rec, 0, empty_idx, empty_val, slot_cache)
        local_rows.append(slot_cache[rec.megatron_name])

    world = torch.distributed.get_world_size()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, local_rows)
    n_rows = len(local_rows)
    assert all(len(rows) == n_rows for rows in gathered), (
        f"directory row counts diverge across ranks: {[len(r) for r in gathered]} -- "
        "the bridge's global enumeration is expected to be structurally parallel"
    )
    for k, rec in enumerate(index):
        union: dict = {}
        for rows in gathered:  # rank order -> identical result on every rank
            row = rows[k]
            if row is None:
                continue
            for slot in row:
                union[(slot[0], tuple(slot[1]))] = None  # ordered-set semantics
        rec.slots = [(n, tuple(shape)) for (n, shape) in union.keys()] if union else None


def mcore_hf_delta_entry(rec: McoreParamExport, _place, lidx: torch.Tensor, lval: torch.Tensor, slot_cache: dict):
    # [FUNCTION DOC]
    # Converts one parameter's shard-local delta into an HF-coordinate entry.
    # Behavior: dtype_str is the value dtype name without "torch.". For non-owned rows asserts a slot table exists and the delta is empty. If the delta is empty and slots are known, returns zero counts per slot with empty index/value tensors without probing. Otherwise builds a NaN buffer of the local param shape, scatters the delta values at the flat indices, runs the probe's megatron_to_hf, and determines the slot list (record slots, else cache, else derived from the outputs and stored in the cache). Asserts every output name is in the slot list. For each slot present in the outputs, finds the non-NaN flat positions; counts per slot (int64 on CPU), concatenated int32 indices and values.
    # Inputs: rec - McoreParamExport; _place - unused; lidx - flat local indices; lval - delta values (floating point); slot_cache - mapping.
    # Outputs: tuple (slots, dtype_str, counts, hf_idx, hf_val).
    # Exceptions: AssertionError for non-float values, missing pre-seeded slots, delta on non-owned rows, or unknown output slots.
    # Side effects: may populate slot_cache.
    # KNOWN ISSUE: changed elements whose new value is itself NaN are indistinguishable from sentinels and are silently dropped from the delta.
    """Probe one mcore param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the LOCAL shard shape (exactly what
    the real ``megatron_to_hf`` receives), runs the comm-stubbed probe -- real
    group sizes, gathers synthesized locally, so the mapping executes its real
    TP>1 code paths -- and extracts each output slot's surviving positions.
    The slot list is cached after the first call (the converter's output names
    are deterministic, so every rank's cache agrees and the batched gather
    stays aligned)."""
    dtype_str = str(lval.dtype).replace("torch.", "")
    assert lval.numel() == 0 or lval.is_floating_point(), (
        f"{rec.megatron_name}: NaN sentinels require a floating-point param, got {lval.dtype}"
    )

    cached = rec.slots if rec.slots is not None else slot_cache.get(rec.megatron_name)
    if rec.param is None:
        # owned by another pipeline stage: pure lockstep row. The slot table
        # was pre-seeded from the owner stage (fail loud if not -- an
        # unsized entry would silently misalign the batched gather).
        assert cached is not None, f"{rec.megatron_name}: slot table not pre-seeded for a non-owned PP param"
        assert lidx.numel() == 0, f"{rec.megatron_name}: delta reported for a param this rank does not own"
    if lidx.numel() == 0 and cached is not None:
        # empty delta: the slot table froze after the first probe, so the
        # zero-count lockstep entry needs no probe run at all -- skip the
        # buffer build, the transform and the full-output NaN scan.
        return (
            cached,
            dtype_str,
            torch.zeros(len(cached), dtype=torch.int64),
            torch.empty(0, dtype=torch.int32, device=lval.device),
            torch.empty(0, dtype=lval.dtype, device=lval.device),
        )

    buf = torch.full(tuple(rec.param.shape), float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        buf.view(-1)[lidx] = lval

    outs = rec.probe.megatron_to_hf(buf, rec.module)

    key = rec.megatron_name
    slots = rec.slots if rec.slots is not None else slot_cache.get(key)
    if slots is None:
        # only reachable from the index-build exchange itself, which probes a
        # zero delta to reveal this rank's local list before the row unions
        # are installed on the recs; steady entries always find rec.slots.
        slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
        slot_cache[key] = slots
    unknown = set(outs) - {n for n, _ in slots}
    assert not unknown, (
        f"{key}: probe emitted slots missing from the global table {sorted(unknown)[:4]} -- "
        "non-deterministic converter naming would misalign the batched gather"
    )
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, _sshape) in enumerate(slots):
        out = outs.get(sname)
        if out is None:
            continue  # another rank's slot (e.g. its ep-local experts): zero count here
        fl = out.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            idx_pieces.append(p_.to(torch.int32))
            val_pieces.append(fl[p_])
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val
