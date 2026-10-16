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
# Module encode: defines the delta-sync wire schema. A flush consists of a uint8 positions blob
# (absolute element positions encoded as int32, 4 bytes per changed element), a values tensor in the
# parameter dtype, a manifest list of per-parameter entries, and an integer checksum. Only the "indices"
# encoding name exists (DeltaEncodingName literal type). No side effects at import beyond torch import.
"""On-wire schema for delta sync: per-parameter manifest, flush container, checksum.

One layout: a uint8 positions blob (``indices`` encoding -- int32 absolute
positions, 4 bytes / nnz) plus a parameter-dtype values tensor, described by a
per-parameter manifest. Values are sent verbatim in the parameter's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices"]


# ---------- diff ----------------------------------------------------------


@dataclass
class DeltaParam:
    # [CLASS DOC]
    # Dataclass manifest entry for one parameter inside a flush.
    # Fields: name - parameter name; dtype - dtype string; shape - full parameter shape;
    # pos_start/pos_end - byte range in the positions blob; pos_width - bytes per position;
    # val_start/val_end - element range in the values tensor.
    # KNOWN ISSUE: pos_width is annotated as "2 or 4", but the module doc states the only encoding uses 4-byte int32 positions.
    """Per-parameter manifest entry for a single chunk / bucket.

    Offsets are byte offsets into the surrounding ``__positions__`` blob and
    element offsets into the surrounding ``__values__`` tensor.
    """

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    # [FUNCTION DOC]
    # Behavior: computes a corruption-detection hash of a positions tensor and a values tensor using
    # torch.hash_tensor (an XOR reduction); an empty tensor contributes 0. Combines as p XOR (v shifted left by 1).
    # Inputs: positions, values - tensors (any device). Outputs: Python int.
    # Exceptions: propagates torch errors (e.g. if hash_tensor is unavailable in the installed torch).
    # Side effects: one device-to-host sync per non-empty argument.
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    Uses ``torch.hash_tensor`` (XOR-reduce over uint64 bitcast); one reduction
    plus one ``.item()`` sync per argument.
    """
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return p ^ (v << 1)


# ---------- encode --------------------------------------------------------


@dataclass
class DeltaFlush:
    # [CLASS DOC]
    # Dataclass describing one ready-to-dispatch flush: encoding name, manifest params, positions blob
    # (positions_cpu, actually may live on GPU), values tensor (values_gpu) and the sender-side checksum.
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. Despite the name it lives on
      the GPU in the sharded engine (the wire broadcasts from the GPU, so a
      host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        # [FUNCTION DOC]
        # Behavior: number of changed elements in this flush. Outputs: element count of values_gpu. No side effects.
        return self.values_gpu.numel()

    @property
    def wire_bytes(self) -> int:
        # [FUNCTION DOC]
        # Behavior: total payload size on the wire. Outputs: bytes of positions blob plus values element count times
        # element size. No side effects.
        return self.positions_cpu.numel() + self.values_gpu.numel() * self.values_gpu.element_size()
