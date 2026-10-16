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
# Rollout replica abstraction for server-mode inference: the token output model, rollout deployment
# modes, the abstract RolloutReplica (one possibly multi-node server instance with lifecycle and
# control fan-out to its server actors), a lazy name-to-class registry for built-in backends
# (vllm, sglang, trtllm), and resolution including prefill/decode disaggregated variants.
# Constant: extra concurrency reserved for control-method calls (16).
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional

import ray
from omegaconf import DictConfig
from pydantic import BaseModel
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.config import HFModelConfig, RolloutConfig

logger = logging.getLogger(__file__)


# Max number of concurrent calls to the methods of Rollout,
# excluding calls to generate method.
CONTROL_METHOD_CONCURRENCY = 16


class TokenOutput(BaseModel):
    # [CLASS DOC]
    # Pydantic model of one generation result: response token ids, optional log-probs, optional routed
    # experts, stop reason, preemption count, and a free-form extra fields mapping.
    token_ids: list[int]
    """response token ids"""
    log_probs: Optional[list[float]] = None
    """logprobs of response token ids"""
    routed_experts: Optional[Any] = None
    """routed experts of response token ids"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class RolloutMode(Enum):
    # [CLASS DOC]
    # Deployment modes: HYBRID (rollout fused with trainer process, weight sync), COLOCATED (same
    # placement group, separate process), STANDALONE (dedicated GPUs).
    # Rollout engine and training engine(fsdp/megatron) fused in same process
    # Rollout and trainer share GPUs, switch context with weight synchronization.
    # Usage scenarios: on-policy training.
    HYBRID = "hybrid"

    # Rollout engine colocated with hybrid engine in same ray placement group but in separate process.
    # Rollout and hybrid processes share GPUs, switch context without weight synchronization.
    # Usage scenarios: GRM (LLM as a judge).
    COLOCATED = "colocated"

    # Standalone rollout server with separate GPU resource, disaggregated architecture.
    # Usage scenarios: off-policy training.
    STANDALONE = "standalone"


class RolloutReplica(ABC):
    # [CLASS DOC]
    # Abstract replica: tracks rank, config, model config, world size (tp*dp*pp), nodes, role flags,
    # mode, worker actors, resource pool, bundle indices, server actors, and the main server's address
    # and handle. Subclasses implement launch_servers and set those fields.
    """Rollout replica is an individual server instance, which may be deployed on single or multiple nodes.
    It is equivalent to launch server in each node with command line:

    SGLang:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 2 ...
    python -m sglang.launch_server --node-rank 1 --nnode 2 ...
    ```

    vLLM:
    ```
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 0 ...
    vllm serve --data-parallel-size 16 --data-parallel-size-local 8 --data-parallel-start-rank 8 ...
    ```

    Args:
        replica_rank: int, rank of this rollout replica.
        config: RolloutConfig, full config.
        model_config: DictConfig, model config.
        gpus_per_node: int, number of gpus per node.
    """

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores rank, converts config to dataclass, stores model config; computes world size,
        # GPUs per replica node = min(gpus_per_node, world size) (must divide world size), node count, role
        # flags, and name suffix ("_" + suffix when non-empty); initializes empty state.
        # Exceptions: AssertionError when world size is not divisible.
        self.replica_rank = replica_rank
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = model_config

        self.world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.gpus_per_node = gpus_per_node
        self.gpus_per_replica_node = min(gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0, (
            f"world_size {self.world_size} must be divisible by gpus_per_node {self.gpus_per_replica_node}"
        )
        self.nnodes = self.world_size // self.gpus_per_replica_node
        self.is_reward_model = is_reward_model
        self.is_teacher_model = is_teacher_model
        self.name_suffix = f"_{name_suffix}" if name_suffix else ""

        self.rollout_mode: RolloutMode = None
        self.workers: list[ActorHandle] = []
        self.resource_pool: RayResourcePool = None
        self.bundle_indices: list[int] = []

        self.servers: list[ActorHandle] = []
        self._server_address: str = None
        self._server_handle: ActorHandle = None

    async def init_hybrid(self, worker_group: RayWorkerGroup):
        # [FUNCTION DOC]
        # Behavior: sets HYBRID mode, takes this replica's contiguous slice of the worker group's workers
        # (world_size workers starting at rank*world_size), and launches servers.
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        await self.launch_servers()

    async def init_hybrid_colocated(self, worker_group: RayWorkerGroup, resource_pool: RayResourcePool):
        # [FUNCTION DOC]
        # Behavior: like init_hybrid, but also records the resource pool and bundle indices
        # rank*world_size + 0..world_size-1, then launches servers.
        # KNOWN ISSUE: docstring lists a bundle_indices argument that does not exist (computed internally),
        # and it sets HYBRID rather than COLOCATED mode.
        """Init hybrid rollout server, rollout engine and training engine(fsdp/megatron) fused in same process.

        Args:
            worker_group: RayWorkerGroup, fused workers where training engine(fsdp/megatron) have been initialized.
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
            bundle_indices: list[int], bundle indices for this rollout replica.
        """
        self.rollout_mode = RolloutMode.HYBRID
        self.workers = worker_group.workers[
            self.world_size * self.replica_rank : self.world_size * (self.replica_rank + 1)
        ]
        self.resource_pool = resource_pool
        self.bundle_indices = [self.replica_rank * self.world_size + idx for idx in range(self.world_size)]
        await self.launch_servers()

    # TODO(sgm): this should be the default solution, but need to make the RolloutMode more clear.
    async def init_colocated(self, resource_pool: RayResourcePool):
        # [FUNCTION DOC]
        # Behavior: sets COLOCATED mode, records the pool, builds a worker group on it with a role-specific
        # name prefix (reward/teacher/default) using the checkpoint-engine worker actor class, without bin
        # packing, with GPU use per rollout_worker_use_gpu; stores its workers and launches servers.
        # Side effects: spawns Ray actors.
        """Init colocated rollout server, rollout engine and hybrid engine colocated in same ray placement group
        but in separate processes.

        Args:
            resource_pool: RayResourcePool, ray placement group where hybrid engine processes have been launched.
        """
        self.rollout_mode = RolloutMode.COLOCATED
        self.resource_pool = resource_pool
        use_gpu = self.rollout_worker_use_gpu()

        if self.is_reward_model:
            name_prefix = f"rollout_reward_colocate_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_colocate_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_colocate_{self.replica_rank}{self.name_suffix}"

        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=name_prefix,
            use_gpu=use_gpu,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    async def init_standalone(self):
        # [FUNCTION DOC]
        # Behavior: sets STANDALONE mode; creates a dedicated resource pool named by role and rank with
        # gpus_per_replica_node per node across nnodes (max colocate 2); builds a GPU worker group on it with
        # a role-specific prefix; stores workers and launches servers.
        # Side effects: creates placement groups and Ray actors.
        """Init standalone rollout server, create new resource pool for this rollout."""
        # create resource pool for this rollout
        self.rollout_mode = RolloutMode.STANDALONE
        if self.is_reward_model:
            resource_pool_name = f"rollout_pool_reward_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            resource_pool_name = f"rollout_pool_teacher_{self.replica_rank}{self.name_suffix}"
        else:
            resource_pool_name = f"rollout_pool_{self.replica_rank}{self.name_suffix}"
        resource_pool_spec = {
            resource_pool_name: [self.gpus_per_replica_node] * self.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=None,
            max_colocate_count=2,
        )
        resource_pool_manager.create_resource_pool()
        self.resource_pool = resource_pool_manager.resource_pool_dict[resource_pool_name]

        # create worker group for this rollout
        if self.is_reward_model:
            name_prefix = f"rollout_reward_standalone_{self.replica_rank}{self.name_suffix}"
        elif self.is_teacher_model:
            name_prefix = f"rollout_teacher_standalone_{self.replica_rank}{self.name_suffix}"
        else:
            name_prefix = f"rollout_standalone_{self.replica_rank}{self.name_suffix}"
        worker_group = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=self.get_ray_class_with_init_args(),
            bin_pack=False,
            name_prefix=name_prefix,
            use_gpu=True,
            device_name=get_device_name(),
        )
        self.workers = worker_group.workers
        await self.launch_servers()

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        # [FUNCTION DOC]
        # Behavior: wraps the checkpoint-engine worker class as a Ray actor class and returns it with init
        # args (rollout config, model config, replica rank).
        """Get rollout worker actor class for colocated and standalone mode."""
        from verl.checkpoint_engine.base import CheckpointEngineWorker

        rollout_worker_actor_cls = ray.remote(CheckpointEngineWorker)

        return RayClassWithInitArgs(
            cls=rollout_worker_actor_cls,
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )

    @abstractmethod
    async def launch_servers(self):
        # [FUNCTION DOC]
        # Behavior: abstract; launch the HTTP server(s) and set servers/address/handle.
        # Exceptions: NotImplementedError if called on the base.
        """Launch http server in each node."""
        raise NotImplementedError

    @property
    def server_address(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns the OpenAI-compatible server address.
        """Get rollout server address for OpenAI chat completion."""
        return self._server_address

    @property
    def server_handle(self) -> ActorHandle:
        # [FUNCTION DOC]
        # Behavior: returns the main server actor handle.
        """Get rollout server handle for Token-in-token-out generation."""
        return self._server_handle

    @property
    def max_concurrency(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns max(1000, max_num_seqs + control concurrency) for Ray actor concurrency.
        # 1000 is Ray's default max_concurrency for async execution.
        # Add some margin to account for control method call.
        return max(1000, self.config.max_num_seqs + CONTROL_METHOD_CONCURRENCY)

    def rollout_worker_use_gpu(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: whether colocated rollout workers request GPUs; base returns true.
        return True

    async def wake_up(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls wake_up on all server actors.
        """Wake up each rollout server."""
        await asyncio.gather(*[server.wake_up.remote() for server in self.servers])

    async def sleep(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls sleep on all server actors.
        """Sleep each rollout server."""
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self, reject_request: bool = False):
        # [FUNCTION DOC]
        # Behavior: concurrently calls abort_all_requests(reject_request) on all server actors.
        """Partial rollout: abort and save all unfinished requests in each rollout server.

        Args:
            reject_request: Fail requests that arrive while generation is blocked instead
                of holding them until the next resume_generation(). Pass True when the
                replica is leaving the load balancer and no resume is coming soon.
                Backends that cannot intercept their own admission path log a warning
                and ignore it.
        """
        await asyncio.gather(
            *[server.abort_all_requests.remote(reject_request=reject_request) for server in self.servers]
        )

    async def resume_generation(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls resume_generation on all server actors.
        """Resume generation on all servers after abort_all_requests."""
        await asyncio.gather(*[server.resume_generation.remote() for server in self.servers])

    async def clear_kv_cache(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls clear_kv_cache on all server actors.
        """reset kv cache in each rollout server."""
        await asyncio.gather(*[server.clear_kv_cache.remote() for server in self.servers])

    async def release_kv_cache(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls release_kv_cache on all server actors.
        """Release only the kv_cache GPU memory, keeping model weights in place."""
        await asyncio.gather(*[server.release_kv_cache.remote() for server in self.servers])

    async def resume_kv_cache(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls resume_kv_cache on all server actors.
        """Restore the kv_cache GPU memory after a weight sync."""
        await asyncio.gather(*[server.resume_kv_cache.remote() for server in self.servers])

    async def start_profile(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: concurrently calls start_profile(kwargs) on all server actors.
        """Start profiling on the replica."""
        await asyncio.gather(*[server.start_profile.remote(**kwargs) for server in self.servers])

    async def stop_profile(self):
        # [FUNCTION DOC]
        # Behavior: concurrently calls stop_profile on all server actors.
        """Stop profiling on the replica."""
        await asyncio.gather(*[server.stop_profile.remote() for server in self.servers])


class RolloutReplicaRegistry:
    # [CLASS DOC]
    # Class-level registry mapping backend name to a zero-argument loader returning the replica class.
    """Factory for managing rollout replica implementations."""

    _registry: dict[str, Callable[[], type[RolloutReplica]]] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], type[RolloutReplica]]) -> None:
        # [FUNCTION DOC]
        # Behavior: (class method) stores/overwrites the loader under name.
        """Register a new rollout replica type."""
        cls._registry[name] = loader

    @classmethod
    def get(cls, name: str) -> type[RolloutReplica]:
        # [FUNCTION DOC]
        # Behavior: (class method) calls the loader for name and returns the class.
        # Exceptions: ValueError for unknown names (lists available).
        """Get a rollout replica class by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown rollout mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()


# Loader functions for built-in types
def _load_vllm():
    # [FUNCTION DOC]
    # HELPER: Lazily imports and returns the vLLM replica class.
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    return vLLMReplica


def _load_sglang():
    # [FUNCTION DOC]
    # HELPER: Sets SGLANG_USE_CPU_ENGINE=1 during import. If vllm is not importable, installs stub
    # modules for vllm, its custom ops (mock fp8 quant) and model executor activation layers (empty
    # GeluAndMul/SiluAndMul classes) into the module table so SGLang's transitive imports succeed.
    # Imports and returns the SGLang replica class, deleting the env var afterwards.
    # Side effects: mutates process environment and global module table.
    # KNOWN ISSUE: the env var is deleted rather than restored (a pre-existing value is lost) and stays
    # set if the import raises.
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"

    try:
        import vllm  # noqa: F401
    except ImportError:
        import sys
        import types
        from unittest.mock import Mock

        mock_vllm = types.ModuleType("vllm")

        mock_custom_ops = types.ModuleType("vllm._custom_ops")
        mock_custom_ops.scaled_fp8_quant = Mock()
        mock_vllm._custom_ops = mock_custom_ops

        mock_model_executor = types.ModuleType("vllm.model_executor")
        mock_layers = types.ModuleType("vllm.model_executor.layers")
        mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

        class GeluAndMul:  # noqa: N801
            # [CLASS DOC]
            # HELPER: Empty stub activation class for the mock vllm module.
            pass

        class SiluAndMul:  # noqa: N801
            # [CLASS DOC]
            # HELPER: Empty stub activation class for the mock vllm module.
            pass

        mock_activation.GeluAndMul = GeluAndMul
        mock_activation.SiluAndMul = SiluAndMul
        mock_layers.activation = mock_activation
        mock_model_executor.layers = mock_layers
        mock_vllm.model_executor = mock_model_executor

        sys.modules["vllm"] = mock_vllm
        sys.modules["vllm._custom_ops"] = mock_custom_ops
        sys.modules["vllm.model_executor"] = mock_model_executor
        sys.modules["vllm.model_executor.layers"] = mock_layers
        sys.modules["vllm.model_executor.layers.activation"] = mock_activation

    from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangReplica

    del os.environ["SGLANG_USE_CPU_ENGINE"]
    return SGLangReplica


def _load_trtllm():
    # [FUNCTION DOC]
    # HELPER: Lazily imports and returns the TensorRT-LLM replica class.
    from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica

    return TRTLLMReplica


# Register built-in types
RolloutReplicaRegistry.register("vllm", _load_vllm)
RolloutReplicaRegistry.register("sglang", _load_sglang)
RolloutReplicaRegistry.register("trtllm", _load_trtllm)


def get_rollout_replica_class(rollout: str, disaggregation_enabled: bool = False) -> type[RolloutReplica]:
    # [FUNCTION DOC]
    # Behavior: with disaggregation enabled, returns the SGLang PD replica (after loading the sglang
    # entry for its import side effects) or the vLLM PD replica; other backends raise. Otherwise
    # returns the registry class for the name.
    # Exceptions: NotImplementedError for unsupported PD backends; ValueError for unknown names.
    """Resolve a replica class by backend name.

    PD-disaggregated rollouts reuse the base backend name (``sglang`` /
    ``vllm``); the dispatch here picks the PD class only when the caller
    asserts ``disaggregation_enabled=True`` (sourced from
    ``RolloutConfig.disaggregation.enabled``). Validation in
    ``RolloutConfig.__post_init__`` rejects the flag for backends without a
    PD class.
    """
    if disaggregation_enabled:
        if rollout == "sglang":
            # _load_sglang side-effect: installs vllm mocks needed by SGLangPDReplica's
            # transitive imports. Cheap if already installed.
            RolloutReplicaRegistry.get("sglang")
            from verl.workers.rollout.sglang_rollout.sglang_pd_replica import SGLangPDReplica

            return SGLangPDReplica
        if rollout == "vllm":
            from verl.workers.rollout.vllm_rollout.vllm_pd_replica import vLLMPDReplica

            return vLLMPDReplica
        raise NotImplementedError(
            f"PD disaggregation is only supported with rollout in ('sglang', 'vllm'); got {rollout!r}."
        )
    return RolloutReplicaRegistry.get(rollout)
