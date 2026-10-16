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
# Abstract rollout interface used by hybrid workers plus a lazy registry mapping (backend name, mode)
# to the fully qualified class path of each server adapter (vllm, sglang, trtllm in async mode).

import importlib
from abc import ABC, abstractmethod
from typing import Generator

import torch
from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig

__all__ = ["BaseRollout"]


class BaseRollout(ABC):
    # [CLASS DOC]
    # Abstract rollout: holds rollout config, model config (both converted to dataclasses) and device
    # mesh; subclasses implement async resume, weight update and release, and optionally sync generation.
    """Base class for rollout."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
        *args,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: converts config and model_config to dataclasses and stores them with device_mesh.
        # Inputs: config; model_config; device_mesh; extra args/kwargs ignored.
        self.config = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.device_mesh = device_mesh

    @abstractmethod
    async def resume(self, tags: list[str]):
        # [FUNCTION DOC]
        # Behavior: abstract; resume GPU memory for the given tags ("weights" and/or "kv_cache").
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        pass

    @abstractmethod
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: abstract; load new weights. Inputs: weights - generator of (name, tensor) for
        # "named_tensors", or of (named tensors, is_last) flushes for "delta_flush"; wire_format; kwargs.
        """Update the weights of the rollout model.

        Args:
            weights: A generator yielding ``(name, tensor)`` pairs for
                ``named_tensors``, or ``(named_tensors, is_last)`` flushes for
                ``delta_flush``.
            wire_format: ``named_tensors`` for full tensors (the default), or
                ``delta_flush`` for delta-engine flushes. Adapters must branch on
                this value instead of treating a delta flush as named weights.
        """
        pass

    @abstractmethod
    async def release(self):
        # [FUNCTION DOC]
        # Behavior: abstract; release weights and KV cache from GPU memory.
        """Release weights and kv cache in GPU memory."""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # [FUNCTION DOC]
        # Behavior: synchronous batch generation; default raises NotImplementedError.
        """Batch generate sequences in sync mode.

        Args:
            prompts: The input prompts.

        Returns:
            The output sequences.
        """
        raise NotImplementedError


_ROLLOUT_REGISTRY = {
    ("vllm", "async"): "verl.workers.rollout.vllm_rollout.ServerAdapter",
    ("sglang", "async"): "verl.workers.rollout.sglang_rollout.sglang_rollout.ServerAdapter",
    ("trtllm", "async"): "verl.workers.rollout.trtllm_rollout.trtllm_rollout.ServerAdapter",
}


def get_rollout_class(rollout_name: str, mode: str = "async") -> type[BaseRollout]:
    # [FUNCTION DOC]
    # Behavior: looks up the class path for (rollout_name, mode), imports its module and returns the class.
    # Inputs: rollout_name - backend; mode - default "async".
    # Outputs: rollout class. Exceptions: AssertionError for unknown pairs; ImportError/AttributeError.
    # Side effects: imports the backend module.
    """Get the rollout class by name.

    Args:
        rollout_name: The name of the rollout.
        mode: The mode of the rollout, async: server mode.

    Returns:
        The rollout class.
    """
    assert (rollout_name, mode) in _ROLLOUT_REGISTRY, f"Rollout {rollout_name} with mode {mode} not found"
    fqdn = _ROLLOUT_REGISTRY[(rollout_name, mode)]
    module_name, class_name = fqdn.rsplit(".", 1)
    rollout_module = importlib.import_module(module_name)
    return getattr(rollout_module, class_name)
