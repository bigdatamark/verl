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

import torch

WeightUpdate = tuple[str, torch.Tensor]


def split_buffer_updates(
    model: torch.nn.Module, weights: list[WeightUpdate]
) -> tuple[list[WeightUpdate], list[WeightUpdate], dict[str, torch.Tensor]]:
    """Split incoming weight updates into parameter and buffer updates.

    Returns the parameter updates, the buffer updates, and the model's
    ``named_buffers`` map so callers can reuse it without re-iterating.
    """
# [MODULE DOC]
# Helpers for applying incoming weight updates (name, tensor pairs) to a rollout model,
# separating registered buffers (which weight loaders typically skip) from parameters
# so buffers can be copied in-place directly. Defines the alias WeightUpdate as a pair of
# (parameter/buffer name, tensor).
    named_buffers = dict(model.named_buffers())
    param_updates, buffer_updates = [], []
    for name, tensor in weights:
        if name in named_buffers:
            buffer_updates.append((name, tensor))
        else:
            param_updates.append((name, tensor))
    return param_updates, buffer_updates, named_buffers
    # [FUNCTION DOC]
    # Behavior: build a name-to-tensor mapping of the model's named buffers, then partition the
    # incoming updates, preserving order, into those whose name is a buffer and all others.
    # Inputs: model - module whose buffers are inspected; weights - list of (name, tensor) pairs.
    # Outputs: tuple of (parameter updates list, buffer updates list, buffer name-to-tensor mapping).
    # Exceptions: none raised directly.
    # Side effects: none (the weight list is consumed only by iteration).


@torch.no_grad()
def apply_buffer_updates(
    model: torch.nn.Module,
    buffer_updates: list[WeightUpdate],
    named_buffers: dict[str, torch.Tensor] | None = None,
) -> int:
    """Copy updated buffer tensors into the target model in-place."""
    if not buffer_updates:
        return 0

    if named_buffers is None:
        named_buffers = dict(model.named_buffers())
    loaded = 0
    for name, tensor in buffer_updates:
        if name not in named_buffers:
            continue

        target = named_buffers[name]
        if target.shape != tensor.shape:
    # [FUNCTION DOC]
    # Behavior: runs under no-grad. Returns 0 immediately if there are no buffer updates. Otherwise
    # uses the supplied buffer mapping (or builds one from the model), and for each update whose name
    # exists in the mapping, verifies the shapes match, casts the source tensor to the target buffer's
    # device and dtype, and copies it into the buffer in place. Unknown names are silently skipped.
    # Inputs: model - target module; buffer_updates - list of (name, tensor); named_buffers - optional
    # precomputed buffer mapping (e.g. from split_buffer_updates).
    # Outputs: number of buffers actually copied.
    # Exceptions: ValueError when a buffer's shape differs from the update tensor's shape.
    # Side effects: mutates model buffers in place; blocking device transfers.
            raise ValueError(
                f"Buffer shape mismatch for {name}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}"
            )

        source = tensor.to(device=target.device, dtype=target.dtype, non_blocking=False)
        target.copy_(source, non_blocking=False)
        loaded += 1

    return loaded
