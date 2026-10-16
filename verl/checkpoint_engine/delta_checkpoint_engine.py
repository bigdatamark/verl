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
# Delta-sharded checkpoint engine (registered as "delta_sharded"): NCCL weight sync for disaggregated rollout that
# ships only changed (position, value) pairs. Extends NCCLCheckpointEngine's ZMQ-manifest + ray.util.collective
# broadcast machinery. Sender: the first (seed) sync streams the full HF export values-only; later syncs batch the
# backend's per-shard HF deltas, gather them to rank 0, bucket them into size-bounded flushes and broadcast each
# immediately. Receiver: yields each flush as (named tensors, is_last) for the rollout adapter's delta loader.
# Each flush = one ZMQ manifest (topic + pickled meta dict: is_full, encoding, is_last, terminal_empty, element counts,
# value dtype, spec with params/checksum) followed by up to two NCCL broadcasts (positions bytes, then values bytes).
# cupy is imported with importlib.metadata.distributions patched to return nothing (avoids slow/conflicting plugin scans).

"""Delta weight-sync checkpoint engine (NCCL transport) for DISAGGREGATED rollout.

Puts the delta on the trainer->rollout wire: the trainer byte-diffs against a
pinned-CPU snapshot and broadcasts only the changed ``(position, value)`` pairs
over the same ``ray.util.collective`` NCCL group the full-weight
:class:`NCCLCheckpointEngine` uses (actor rank0 -> rollout CheckpointEngineWorkers).
With SGLang, each rollout worker hands its local copy of the sparse payload to
its colocated SGLang TP worker via same-GPU ``update_weights_from_tensor`` IPC,
where the verl-shipped :mod:`verl.workers.rollout.sglang_rollout.delta_loader`
(registered through SGLang's stock ``--custom-weight-loader`` hook — no SGLang
fork or patch needed) decodes and masked-applies it *in place* onto the live
weights. No full-model mirror is staged anywhere on the rollout side: receiver
peak memory is one bucket plus one decode chunk, independent of model size.

With vLLM, the rollout adapter forwards the same payload over same-GPU IPC to
VERL's registered weight-transfer backend. It decodes each flush into vLLM
checkpoint patches, whose loader reuses native ``model.load_weights()`` mapping.

The first (seed) sync streams the backend's FULL HF export (``get_per_tensor_
param()``) over the values-only wire -- every backend already knows how to
assemble and convert its own full tensors, so resume works by construction and
the seed inherits Megatron/veomni assembly for free. After the seed the caller
primes the backend's pinned shard snapshots; every later sync ships the
backend-computed sparse HF delta.

Data ladder (sender side, steady) -- the names in this file anchor to these::

    backend HF delta ENTRY (slots, dtype, counts, hf_idx, hf_val, group)
    --_GatherQueue--> per-SLOT delta on rank 0 --_bucket_*--> _FlushPiece
    --_FlushBucket--> FLUSH (DeltaFlush) --_publish_flush--> wire

Wire encodings: ``indices`` (int32 positions + values; every steady sync) and
``values`` (values only; the seed). The values meta tag is ``"dense"`` for
protocol continuity with the receiver's delta_loader.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import ray.util.collective as collective
import torch
import zmq

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

from .base import CheckpointEngineRegistry
from .delta_sync.encode import DeltaFlush, DeltaParam
from .delta_sync.encode import checksum as _checksum
from .delta_sync.sparse_gather import gather_slot_entries_to_rank0
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine, WorkerMetadata

logger = logging.getLogger(__name__)


def _prodshape(shape) -> int:
    # [FUNCTION DOC]
    # HELPER: Returns the product of the dimensions of shape as an int (1 for an empty shape).
    n = 1
    for x in shape:
        n *= int(x)
    return n


@dataclass(slots=True)
class _FlushPiece:
    # [CLASS DOC]
    # HELPER: Slotted record of one (possibly sliced) parameter delta awaiting an indices flush: name, dtype string, full
    # shape list, flat in-parameter positions tensor and matching values tensor.
    """One (possibly sliced) per-parameter piece buffered for a pending indices flush."""

    name: str
    dtype_str: str
    shape: list
    idx: torch.Tensor
    val: torch.Tensor


@dataclass(slots=True)
class _ValuesPiece:
    # [CLASS DOC]
    # HELPER: Slotted record of one whole parameter's flat values awaiting a values-only (seed) flush: name, dtype string,
    # shape list, flat values tensor.
    """One whole parameter's flat values buffered for the seed sync's values-only flush."""

    name: str
    dtype_str: str
    shape: list
    flat: torch.Tensor


class _FlushBucket:
    # [CLASS DOC]
    # HELPER: Byte-capped bucketing pipeline with one-flush lookahead. Pieces accumulate until their byte total reaches cap;
    # seal then publishes the previously pending flush with is_last False and assembles the current pieces into the new
    # pending flush. The caller emits the final pending flush with is_last True. assemble(pieces) builds a flush;
    # publish(flush, is_last) sends it.
    """One-flush-lookahead bucket pipeline, shared by the steady loop and both
    seed streams. Pieces accumulate until ``cap`` bytes; ``seal`` assembles them
    into the single pending flush, first emitting the previous pending with
    ``is_last=False`` (the lookahead: only the caller's finale knows which flush
    is last and emits it with ``is_last=True``). ``assemble`` and ``publish``
    carry the only real differences between the streams -- the wire format
    (indexed flush vs values-only flush) and the flush counters."""

    __slots__ = ("cap", "pieces", "nbytes", "pending", "_assemble", "_publish")

    def __init__(self, cap: int, assemble, publish):
        # [FUNCTION DOC]
        # HELPER: Stores cap (as int), the assemble and publish callbacks, and empty piece/byte/pending state.
        self.cap = int(cap)
        self.pieces: list = []
        self.nbytes = 0
        self.pending = None
        self._assemble = assemble
        self._publish = publish

    def add(self, piece, nbytes: int) -> None:
        # [FUNCTION DOC]
        # HELPER: Appends a piece, adds its byte count, and seals when the total reaches or exceeds cap.
        self.pieces.append(piece)
        self.nbytes += int(nbytes)
        if self.nbytes >= self.cap:
            self.seal()

    def seal(self) -> None:
        # [FUNCTION DOC]
        # HELPER: No-op with no pieces; otherwise emits any pending flush as non-last, assembles the pieces into the new
        # pending flush, and resets pieces and byte count.
        if not self.pieces:
            return
        self.emit(is_last=False)
        self.pending = self._assemble(self.pieces)
        self.pieces, self.nbytes = [], 0

    def emit(self, is_last: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: If a pending flush exists, publishes it with the given is_last and clears it.
        if self.pending is not None:
            self._publish(self.pending, is_last)
            self.pending = None


def _bucket_sliced(bkt: _FlushBucket, name: str, dtype_str: str, shape, aidx: torch.Tensor, aval: torch.Tensor) -> None:
    # [FUNCTION DOC]
    # HELPER: Splits one parameter's positions/values into consecutive slices of at most MAX_ENTRY_ELEMS elements and adds
    # each to the bucket as a _FlushPiece, with byte cost = elements x (4 position bytes + value element size).
    # Inputs: bkt - bucket; name, dtype_str, shape - parameter identity; aidx/aval - aligned 1-D tensors.
    """Slice one param's (idx, val) delta into <= MAX_ENTRY_ELEMS pieces and bucket
    them (bounds the receiver-side decode transient; the masked apply is sequential,
    so splitting is transparent). Bucket bytes = actual wire bytes (int32 positions
    + values)."""
    max_elems = DeltaShardedCheckpointEngine.MAX_ENTRY_ELEMS
    for s in range(0, aidx.numel(), max_elems):
        e = min(s + max_elems, aidx.numel())
        bkt.add(
            _FlushPiece(name, dtype_str, list(shape), aidx[s:e], aval[s:e]),
            (e - s) * (4 + aval.element_size()),
        )


class _GatherQueue:
    # [CLASS DOC]
    # HELPER: Batches slot-keyed delta entries (slots list of (name, shape), dtype string, per-slot counts, idx, val) per
    # process group, flushing a group's queue once it holds batch_k entries (count-only trigger, identical on every rank so
    # collectives stay in lockstep). A flush gathers all entries of the batch to rank 0 in one batched sparse gather
    # (bounded per round by max_round_bytes) and, on rank 0, calls consume(name, dtype_str, shape, numel, idx, val) per slot.
    """Per-gather-group batching of slot-keyed queue entries
    ``(slots, dtype_str, counts, idx, val)``. Entries carry FINAL-coordinate
    payloads (identity specs: one slot = the param itself; converter specs: the
    spec's hf_slots), so rank 0 never converts -- ``consume`` receives assembled
    per-slot pieces straight from the gather.

    One queue per ProcessGroup: separate queues stop pg alternation (dense fsdp
    group vs expert world group per layer) from shattering batches. The flush
    trigger is COUNT-ONLY: entry counts are identical on every rank while byte
    totals are not, so a count trigger is the only one that keeps the collective
    sequence identical across ranks (a per-rank byte trigger desyncs the gathers
    and deadlocks NCCL). Byte bounding happens INSIDE the batched gather via
    ``max_round_bytes``, decided from the all-gathered counts every rank sees."""

    __slots__ = ("batch_k", "max_round_bytes", "is_r0", "_consume", "_queues")

    def __init__(self, batch_k: int, max_round_bytes: int, is_r0: bool, consume):
        # [FUNCTION DOC]
        # HELPER: Stores batch size (at least 1), per-round byte cap, whether this is rank 0, the consume callback, and an empty
        # mapping from process-group identity to (group, entry list).
        self.batch_k = max(int(batch_k), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_r0 = is_r0
        self._consume = consume
        self._queues: dict[int, tuple] = {}  # id(pg) -> (pg, [entries])

    def put(self, pg, slots: list, dtype_str: str, counts: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Appends an entry to the queue for pg (creating it on first use, keyed by object identity) and flushes that
        # queue when it reaches batch_k entries.
        _pg, entries = self._queues.setdefault(id(pg), (pg, []))
        entries.append((slots, dtype_str, counts, idx, val))
        if len(entries) >= self.batch_k:
            self._flush(pg, entries)

    def flush_all(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Flushes every group's remaining entries (in insertion order of groups).
        for pg, entries in self._queues.values():
            self._flush(pg, entries)

    def _flush(self, pg, entries: list) -> None:
        # [FUNCTION DOC]
        # HELPER: Takes a copy of the entries and clears the queue. For pg None (unsharded params) rank 0 consumes each slot
        # directly by slicing idx/val by the per-slot counts; other ranks drop the entries. Otherwise concatenates counts (moved
        # to the idx device), idx and val across the batch, calls gather_slot_entries_to_rank0 on pg, and on rank 0 consumes the
        # gathered (idx, val) pairs in slot order. No-op for an empty queue.
        # Side effects: collective communication on pg for every rank.
        """One gather round for one group's queue."""
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if pg is None:
            # unsharded/replicated params: rank 0's local delta already is global
            if self.is_r0:
                for slots, dtype_str, counts, idx, val in batch:
                    off = 0
                    for (name, shape), c in zip(slots, counts.tolist(), strict=True):
                        self._consume(
                            name, dtype_str, tuple(shape), _prodshape(shape), idx[off : off + c], val[off : off + c]
                        )
                        off += c
            return
        dev = batch[0][3].device
        counts_concat = torch.cat([c for _, _, c, _, _ in batch]).to(dev)
        idx_concat = torch.cat([i for _, _, _, i, _ in batch])
        val_concat = torch.cat([v for _, _, _, _, v in batch])
        gathered = gather_slot_entries_to_rank0(
            idx_concat, val_concat, counts_concat, group=pg, max_round_bytes=self.max_round_bytes
        )
        if self.is_r0 and gathered is not None:
            slot_i = 0
            for slots, dtype_str, _counts, _i, _v in batch:
                for name, shape in slots:
                    aidx, aval = gathered[slot_i]
                    slot_i += 1
                    self._consume(name, dtype_str, tuple(shape), _prodshape(shape), aidx, aval)


@CheckpointEngineRegistry.register("delta_sharded")
class DeltaShardedCheckpointEngine(NCCLCheckpointEngine):
    # [CLASS DOC]
    # Sparse-delta NCCL checkpoint engine. wire_format is "delta_flush". MAX_ENTRY_ELEMS (64M) caps changed elements per
    # DeltaParam entry to bound receiver decode memory. Instance state adds: encoding ("indices" only), verify_every K
    # (dense verification sweep every K-th steady sync; 0 disables), batch_gather (params per gather batch) and whether the
    # seed sync has been done. Relies on parent state: rank, is_master, ip/listen_port, socket, topic, group_name,
    # bucket_size and rollout_dtype.
    """Sparse delta weight sync over NCCL, diffed on each rank's local shard.

    Reuses NCCLCheckpointEngine's group/zmq machinery but moves only changed
    positions+values: each actor rank keeps a pinned-CPU snapshot of only *its*
    shard, byte-diffs the shard, and only the changed ``(position, value)`` pairs
    are gathered to rank 0 and streamed to the rollout side -- no rank ever holds
    a full-model snapshot.

    ``send_weights`` takes the TRAINING ENGINE and drives the sync itself: the
    seed (first sync) streams the backend's full ``get_per_tensor_param()``
    export values-only and pins the diff base (``prime_delta_snapshots``); every
    steady sync consumes the backend's HF delta export
    ``get_per_tensor_param_delta_shard()`` — per-parameter FINAL-HF-coordinate
    entries ``(slots, dtype_str, counts, hf_idx, hf_val, gather_group)``. Naming,
    to-HF conversion, diff and snapshot all live on the backend side (see
    :mod:`verl.workers.engine.utils`); this engine only batches, gathers,
    buckets and ships, and so serves any backend that can produce HF deltas.
    """

    # Cap on changed elements per DeltaParam entry. The receiver-side decode
    # densifies per entry with an int64 index transient (8 B/element), so an
    # uncapped entry (e.g. a 7B model's whole embedding on the full seed, ~545M
    # elements) would spike several GiB at once. Oversized per-param deltas are
    # sliced into multiple entries (the masked apply is sequential, so splitting
    # is transparent); 64M elements bounds the transient to ~512 MiB.
    MAX_ENTRY_ELEMS = 64 << 20

    wire_format = "delta_flush"

    def prepare(self) -> WorkerMetadata:
        # [FUNCTION DOC]
        # Returns WorkerMetadata with this node id and, on the master, a MasterMetadata (zmq ip/port, single sender); skips the
        # parent's fixed double-bucket buffer allocation.
        # Delta broadcasts small per-flush buffers directly, so skip the parent's
        # 2 * bucket_size fixed buffers. Still hand back the master zmq endpoint
        # that build_topology() distributes to the rollout workers.
        #
        # Only rank 0 broadcasts here (see the assert in send_weights) and this engine has no relay
        # path, so it always takes the parent's single-sender topology.
        master = (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port, multi_sender=False) if self.is_master else None
        )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    # ---- trainer side ----
    # ---- shared STREAMING wire ----
    # Stream each flush as soon as it is produced so trainer peak memory stays near two buckets,
    # including during the full seed. Each ZMQ manifest + NCCL broadcast carries ``is_last``;
    # rollout adapters then forward that flush over same-GPU IPC to their backend-specific consumer.
    def _publish_flush(self, flush: DeltaFlush, first: bool, is_last: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: Sends an indices flush: ZMQ topic plus meta (is_full = first, encoding, is_last, counts, value dtype, spec with
        # param descriptors and checksum), then copies the positions (moved to GPU) and values as bytes into fresh cupy buffers
        # and broadcasts positions then values from rank 0 over the collective group. Staging into cupy-owned memory avoids
        # allocator reuse races with the asynchronous broadcast.
        meta = {
            "is_full": first,
            "encoding": self.encoding,
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": int(flush.positions_cpu.numel()),
            "val_numel": int(flush.values_gpu.numel()),
            "val_dtype": str(flush.values_gpu.dtype).replace("torch.", ""),
            "spec": {
                "encoding": self.encoding,
                "params": [vars(p) for p in flush.params],
                "checksum": int(flush.checksum),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        pos_u8 = flush.positions_cpu.to("cuda", non_blocking=True).contiguous().view(torch.uint8)
        val_u8 = flush.values_gpu.contiguous().view(torch.uint8)
        # Stage into cupy-owned buffers: ray's NCCL broadcast is enqueued on a separate
        # stream with no recordStream on its inputs, so broadcasting a zero-copy view of
        # these torch tensors (freed right after this call) would race with allocator reuse.
        pos_cp = cp.empty(pos_u8.numel(), dtype=cp.uint8)
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        pos_cp[:] = cp.asarray(pos_u8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(pos_cp, src_rank=0, group_name=self.group_name)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _publish_values_flush(
        self, params: list[DeltaParam], values: torch.Tensor, is_last: bool, verify: bool = False
    ) -> None:
        # [FUNCTION DOC]
        # HELPER: Sends a values-only seed/verify flush tagged encoding "dense" (is_full True, zero positions, spec including
        # verify and is_last flags and a checksum over empty positions plus values), then broadcasts only the value bytes via a
        # cupy staging buffer.
        """Publish a values-only (full-coverage, positions-free) flush -- used by the first
        sync. The wire encoding tag stays ``"dense"`` -- it is protocol, shared with the
        receiver's delta_loader decode."""
        values = values.contiguous()
        empty_pos = torch.empty(0, dtype=torch.uint8, device=values.device)
        meta = {
            "is_full": True,
            "encoding": "dense",
            "is_last": is_last,
            "terminal_empty": False,
            "pos_numel": 0,
            "val_numel": int(values.numel()),
            "val_dtype": str(values.dtype).replace("torch.", ""),
            "spec": {
                "encoding": "dense",
                "verify": verify,
                "is_last": is_last,
                "params": [vars(p) for p in params],
                "checksum": int(_checksum(empty_pos, values)),
            },
        }
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)
        val_u8 = values.view(torch.uint8)
        # cupy-owned staging: same lifetime rationale as _publish_flush.
        val_cp = cp.empty(val_u8.numel(), dtype=cp.uint8)
        val_cp[:] = cp.asarray(val_u8)
        collective.broadcast(val_cp, src_rank=0, group_name=self.group_name)

    def _release_staging_pool(self, phase: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Frees all cached blocks of cupy's default memory pool back to CUDA and logs (warning level) the bytes held and
        # device free memory before/after. Inputs: phase - label for the log.
        """Return the cupy staging pool's blocks to CUDA and log the evidence:
        ``held`` is what the pool would have kept from the device without this
        release, and the device-free delta shows the memory actually coming
        back (warning level so the default worker log level records it)."""
        from verl.utils.device import get_torch_device

        pool = cp.get_default_memory_pool()
        held = pool.total_bytes()
        free_before, _ = get_torch_device().mem_get_info()
        pool.free_all_blocks()
        free_after, _ = get_torch_device().mem_get_info()
        logger.warning(
            "cupy staging pool after %s send: held %.2fGB; device free %.2f->%.2fGB on release",
            phase,
            held / (1 << 30),
            free_before / (1 << 30),
            free_after / (1 << 30),
        )

    def _publish_terminal(self, first: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: Sends only a manifest marking the end of the stream with no data (terminal_empty True, is_last True); no broadcast.
        """End-of-stream marker when zero flushes were produced (no broadcast, just a signal)."""
        meta = {"is_full": first, "encoding": self.encoding, "is_last": True, "terminal_empty": True}
        self.socket.send_string(self.topic, flags=zmq.SNDMORE)
        self.socket.send_pyobj(meta)

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        # [FUNCTION DOC]
        # Rollout-side generator. Loops: receives topic and meta; stops on terminal_empty; allocates a GPU uint8 values buffer
        # (val_numel x element size of val_dtype) and, for non-dense encodings, a positions buffer of pos_numel bytes; receives
        # positions (if any) then values by broadcast; yields a list [("__delta_spec__", JSON spec bytes on GPU),
        # ("__positions__", position bytes) if present, ("__values__", values viewed as val_dtype)] together with is_last; drops
        # references after the consumer resumes; stops after the last flush. Logs the flush count.
        # Exceptions: AssertionError if called on rank 0 (rank must be > 0).
        """Yield delta flushes for the rollout adapter to consume.

        Each ``(named_tensors, is_last)`` item contains a JSON manifest, optional
        int32 position bytes, and patch values. The generator keeps at most one
        received GPU flush live at a time.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        applied = 0
        while True:
            self.socket.recv_string()
            meta = self.socket.recv_pyobj()
            if meta.get("terminal_empty"):
                break

            dense = meta.get("encoding") == "dense"
            val_dtype = getattr(torch, meta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device="cuda")
            if dense:
                pos = None
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            else:
                pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device="cuda")
                collective.broadcast(pos, src_rank=0, group_name=self.group_name)
                collective.broadcast(val_u8, src_rank=0, group_name=self.group_name)
            val = val_u8.view(val_dtype)
            spec_bytes = json.dumps(meta["spec"]).encode()
            spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to("cuda")
            named = [("__delta_spec__", spec_t), ("__values__", val)]
            if pos is not None:
                named.insert(1, ("__positions__", pos))
            is_last = bool(meta["is_last"])
            yield named, is_last
            applied += 1
            del pos, val_u8, val, spec_t
            if is_last:
                break
        logger.info("delta recv v=%s flushes=%d (yielded to server adapter)", global_steps, applied)

    def __init__(
        self, *args, encoding: str = "indices", batch_gather: int = 32, verify_every: int = 0, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Calls the parent constructor with remaining args, then stores encoding (asserting it is "indices"), verify_every and
        # batch_gather as ints and marks the seed as not yet done. Exceptions: AssertionError for other encodings.
        super().__init__(*args, **kwargs)
        assert encoding == "indices", f"delta_sharded ships only the 'indices' position encoding; got {encoding!r}"
        self.encoding = encoding
        # SGLang supports verify_every > 0; vLLM rejects it at startup.
        self.verify_every = int(verify_every)
        self._shard_seeded = False
        # Gather the per-param sparse deltas in groups of this many parameters
        # (one count-matrix all_gather + two padded gathers per group instead of
        # three collectives per parameter).
        self.batch_gather = int(batch_gather)

    def _verify_due(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns False when verify_every <= 0; otherwise increments a lazily created steady-sync counter and returns
        # True every verify_every-th call.
        """True on every K-th steady sync when constructed with ``verify_every=K``."""
        if self.verify_every <= 0:
            return False
        self._steady_count = getattr(self, "_steady_count", 0) + 1
        return self._steady_count % self.verify_every == 0

    def _assemble_flush(self, per_param: list[_FlushPiece]) -> DeltaFlush:
        # [FUNCTION DOC]
        # HELPER: Builds an indices DeltaFlush from pieces: for each piece asserts the parameter has fewer than 2^31 elements,
        # converts positions to int32, and records a DeltaParam with byte ranges into the positions buffer (4 bytes each) and
        # element ranges into the values buffer. Concatenates values (or an empty rollout-dtype GPU tensor) and positions viewed
        # as bytes (or empty), computes the checksum over both, and returns the flush (positions stay on GPU despite the field name).
        # Exceptions: AssertionError for oversized parameters.
        """Build one DeltaFlush (indices encoding) from rank 0's gathered per-param deltas.

        ``per_param``: :class:`_FlushPiece` entries whose ``idx`` are within-parameter
        flat positions (== what the receiver decodes).

        Positions stay on the GPU end to end (int32 pieces -> one cat -> uint8 view);
        the wire broadcasts from the GPU anyway, and a host round-trip here
        (``.cpu().numpy().tobytes()`` + join) dominated the whole send at scale
        (~2.4s/sync at 7B steady state, ~83s on the full seed).
        """
        idx_pieces: list[torch.Tensor] = []
        val_pieces: list[torch.Tensor] = []
        params: list[DeltaParam] = []
        pos_off = val_off = 0
        for piece in per_param:
            nnz = int(piece.idx.numel())
            # positions ride the wire as int32 (pos_width=4); a parameter bigger than
            # 2^31 elements would silently wrap, so fail loud instead. DeltaParam
            # carries pos_width for a future 8-byte escalation if a model needs it.
            assert _prodshape(piece.shape) < (1 << 31), (
                f"{piece.name}: {_prodshape(piece.shape)} elements exceeds the int32 position encoding"
            )
            idx_pieces.append(piece.idx.to(torch.int32))
            val_pieces.append(piece.val)
            params.append(
                DeltaParam(
                    name=piece.name,
                    dtype=piece.dtype_str,
                    shape=list(piece.shape),
                    pos_start=pos_off,
                    pos_end=pos_off + nnz * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + nnz,
                )
            )
            pos_off += nnz * 4
            val_off += nnz

        values_gpu = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=self.rollout_dtype, device="cuda")
        positions_u8 = (
            torch.cat(idx_pieces).contiguous().view(torch.uint8)
            if idx_pieces
            else torch.empty(0, dtype=torch.uint8, device=values_gpu.device)
        )
        cks = _checksum(positions_u8, values_gpu)
        return DeltaFlush(
            encoding=self.encoding, params=params, positions_cpu=positions_u8, values_gpu=values_gpu, checksum=cks
        )

    def _send_full_seed(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        verify: bool = False,
    ) -> dict[str, float] | None:
        # [FUNCTION DOC]
        # HELPER: Streams a full HF export values-only. Every rank iterates weights (assembly may be collective), asserting no
        # duplicate names and casting floating tensors to rollout_dtype; non-master ranks discard tensors and return None.
        # Rank 0 buckets flat values into bucket_size flushes published via _publish_values_flush, seals and emits the last
        # flush with is_last True (or a terminal marker when nothing was produced), releases the cupy pool, and logs timing.
        # Inputs: weights - iterator of (name, tensor); global_steps - for logging; verify - marks a verification sweep.
        # Outputs: None when no elements (or non-master), else metrics dict (changed_ratio 1.0, changed_elems, payload MiB, flushes).
        # Exceptions: AssertionError on duplicate names.
        """First sync: stream the backend's FULL HF export over the values-only wire.

        ``weights`` is ``get_per_tensor_param()`` -- every backend already knows how
        to assemble and convert its own full tensors (FSDP all-gather, veomni expert
        restack, Megatron TP/PP fusion), so the seed inherits all of that for free
        and this engine only buckets and broadcasts. Every trainer rank iterates the
        generator (the per-tensor assembly is collective); rank 0 buckets. Resume
        works by construction: whatever the trainer restored is what ships."""
        is_r0 = self.is_master
        t0 = time.time()
        n_flushes = 0
        total_elems = 0
        wire_bytes = 0

        def _assemble_values(pieces: list[_ValuesPiece]):
            # [FUNCTION DOC]
            # HELPER: Concatenates piece values into one flat tensor and builds DeltaParam descriptors with empty position ranges
            # and consecutive value ranges. Outputs: (params list, values tensor).
            params = []
            val_off = 0
            for piece in pieces:
                n = int(piece.flat.numel())
                params.append(
                    DeltaParam(
                        name=piece.name,
                        dtype=piece.dtype_str,
                        shape=list(piece.shape),
                        pos_start=0,
                        pos_end=0,
                        pos_width=4,
                        val_start=val_off,
                        val_end=val_off + n,
                    )
                )
                val_off += n
            return params, torch.cat([piece.flat for piece in pieces])

        def _publish_values(pending, is_last: bool) -> None:
            # [FUNCTION DOC]
            # HELPER: Publishes a pending (params, values) as a values-only flush and updates the flush and byte counters.
            nonlocal n_flushes, wire_bytes
            params, values = pending
            self._publish_values_flush(params, values, is_last=is_last, verify=verify)
            n_flushes += 1
            wire_bytes += int(values.nbytes)

        bkt = _FlushBucket(self.bucket_size, _assemble_values, _publish_values)

        seen_names: set = set()
        for name, tensor in weights:
            # duplicate names in a full export mean two source params mapped to
            # the same HF tensor -- the receiver would apply whichever came last
            # and the delta diff base would silently disagree with the trainer
            # (observed: NemotronH A_log). Fail loud instead.
            assert name not in seen_names, f"full export yields duplicate HF tensor {name!r}"
            seen_names.add(name)
            tensor = tensor.detach()
            if tensor.is_floating_point() and tensor.dtype != self.rollout_dtype:
                tensor = tensor.to(self.rollout_dtype)
            if not is_r0:
                del tensor
                continue
            flat = tensor.contiguous().reshape(-1)
            total_elems += int(flat.numel())
            bkt.add(_ValuesPiece(name, str(flat.dtype).replace("torch.", ""), list(tensor.shape), flat), flat.nbytes)

        if not is_r0:
            return
        bkt.seal()
        if bkt.pending is not None:
            bkt.emit(is_last=True)
        else:
            self._publish_terminal(True)
        # warning level on purpose: worker default log level swallows info, and the
        # one-off seed cost is the number people ask for when sizing a run.
        # the cupy staging pool does not return its blocks to CUDA on its own;
        # after streaming up to 2x bucket_size through it, give the memory back
        # so the trainer's next optimizer/forward pass can use it (raw cudaMalloc
        # OOMs on tight mcore shapes otherwise).
        self._release_staging_pool("seed")
        logger.warning(
            "delta-sharded FULL-%s v=%s done in %.1fs (flushes=%d elems=%d wire=%.1fGB)",
            "VERIFY" if verify else "SEED",
            global_steps,
            time.time() - t0,
            n_flushes,
            total_elems,
            wire_bytes / (1 << 30),
        )
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": 1.0,
            "checkpoint_engine/changed_elems": float(total_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }

    async def send_weights(
        self,
        engine,
        global_steps: int | None = None,
    ) -> dict[str, float] | None:
        # [FUNCTION DOC]
        # Drives one sync from the training engine object (not a weight iterator). All actor ranks call it.
        # Seed (first call): streams engine.get_per_tensor_param() via _send_full_seed, then calls engine.prime_delta_snapshots()
        # to pin the diff base, marks seeded, and returns the seed metrics.
        # Steady: iterates engine.get_per_tensor_param_delta_shard() entries (slots, dtype, counts, idx, val, group) into a
        # gather queue (batch size batch_gather, round cap bucket_size); rank 0 consumes gathered slot deltas, counting total and
        # changed elements and wire bytes and slicing them into the bucket. After flushing the queue, decides whether a
        # verification sweep is due; rank 0 seals and emits the last delta flush with is_last True unless a sweep follows (or
        # sends a terminal marker if nothing was produced and no sweep follows). If due, every rank runs a full values-only
        # sweep with verify set. Rank 0 then releases the cupy pool.
        # Outputs: None for non-master ranks or when no elements were seen; else metrics (changed ratio, changed elems, payload
        # MiB, flushes). Exceptions: AssertionError if rank > 0 (non-zero trainer ranks are expected to have negative rank).
        # KNOWN ISSUE: when a verification sweep runs, its metrics are discarded and the cupy pool is released twice; harmless.
        """Drive one weight sync from the TRAINING ENGINE (unlike the full-sync
        engines, which consume a weights iterator): the seed/steady phase choice
        and the snapshot prime are this engine's own state machine, so the worker
        stays delta-agnostic. The seed (first sync) streams the backend's full
        ``get_per_tensor_param()`` export over the values-only wire, then pins the
        diff base via ``engine.prime_delta_snapshots()``; every later sync consumes
        ``get_per_tensor_param_delta_shard()`` (backend-computed final-HF deltas).
        """
        # All actor ranks participate (gather-v is collective); only torch rank 0 broadcasts.
        # rank 0 accumulates the gathered per-param deltas into bucket_size-sized flushes and streams
        # each one as soon as it fills (then frees it), so peak memory is ~2 buckets rather than the
        # whole model.
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."
        if not self._shard_seeded:
            full, _ = engine.get_per_tensor_param()
            metrics = self._send_full_seed(full, global_steps)
            # weights do not move during the sync, so the snapshots equal exactly
            # what the rollout just received.
            engine.prime_delta_snapshots()
            # The seed is complete only after its baseline snapshots are ready.
            self._shard_seeded = True
            return metrics
        weights, _ = engine.get_per_tensor_param_delta_shard()
        is_r0 = self.is_master
        n_flushes = 0
        changed_elems = 0
        total_elems = 0
        wire_bytes = 0

        def _publish_steady(flush, is_last: bool) -> None:
            # [FUNCTION DOC]
            # HELPER: Publishes an indices flush (first False) and increments the flush counter.
            nonlocal n_flushes
            self._publish_flush(flush, first=False, is_last=is_last)
            n_flushes += 1

        bkt = _FlushBucket(self.bucket_size, self._assemble_flush, _publish_steady)

        batch_k = self.batch_gather

        def _bucket_slot_delta(
            name: str,
            dtype_str: str,
            full_shape: tuple,
            full_numel: int,
            aidx: torch.Tensor | None,
            aval: torch.Tensor | None,
        ) -> None:
            # [FUNCTION DOC]
            # HELPER: Rank-0 consume callback: adds the slot's full element count to the total; if it has changes, adds changed
            # elements and wire bytes and slices the delta into the bucket.
            nonlocal total_elems, changed_elems, wire_bytes
            total_elems += int(full_numel)
            if aidx is None or aidx.numel() == 0:
                return
            changed_elems += int(aidx.numel())
            wire_bytes += int(aidx.numel()) * (4 + aval.element_size())
            _bucket_sliced(bkt, name, dtype_str, full_shape, aidx, aval)

        gq = _GatherQueue(batch_k, self.bucket_size, is_r0, _bucket_slot_delta)

        # ``weights`` is the BACKEND's HF delta stream (hf_delta_export): entries
        # already carry final HF coordinates -- naming, conversion, diff and
        # snapshot all happened on the backend side. This engine only batches,
        # gathers and ships.
        for slots, dtype_str, counts, hf_idx, hf_val, pg in weights:
            gq.put(pg, slots, dtype_str, counts, hf_idx, hf_val)
        gq.flush_all()

        # For SGLang, verify_every=K appends a dense state-verification sweep to
        # every K-th steady sync inside the same receive stream. The steady bucket
        # keeps ``is_last`` unset, and the sweep's final flush carries it. The
        # receiver bit-compares each destination before overwriting it and fails
        # on any mismatch (see delta_loader._verify_dense).
        verify = self._verify_due()
        if is_r0:
            bkt.seal()  # seal the final partial bucket into the pending flush
            if bkt.pending is not None:
                bkt.emit(is_last=not verify)
            elif not verify:
                self._publish_terminal(False)
        if verify:
            # collective on every rank: the full export assembles per tensor.
            full, _ = engine.get_per_tensor_param()
            self._send_full_seed(full, global_steps, verify=True)
        if not is_r0:
            return
        self._release_staging_pool("steady")  # return staging blocks to CUDA between syncs
        logger.info("delta-sharded send v=%s delta flushes=%d (streamed)", global_steps, n_flushes)
        if not total_elems:
            return None
        return {
            "checkpoint_engine/changed_ratio": changed_elems / total_elems,
            "checkpoint_engine/changed_elems": float(changed_elems),
            "checkpoint_engine/payload_mbytes": wire_bytes / (1 << 20),
            "checkpoint_engine/flushes": float(n_flushes),
        }
