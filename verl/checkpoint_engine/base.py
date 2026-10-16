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
# Core checkpoint-engine abstractions for transferring trainer (actor) weights to rollout replicas: tensor chunk
# metadata, a backend registry, the abstract engine interface (plus a cached variant), the colocated "naive" engine,
# the Ray worker that hosts an engine next to an inference server, the driver-side manager orchestrating a sync, and
# async helpers that split weights into byte chunks and merge them back.

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

import ray
import torch

from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.import_utils import import_external_libs
from verl.utils.ray_utils import auto_await
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, RolloutReplica, get_rollout_class
from verl.workers.rollout.utils import ensure_async_iterator


@dataclass
class TensorMeta:
    # [CLASS DOC]
    # Metadata describing one byte-chunk of a weight tensor in transit: tensor name, full shape and dtype, the chunk's byte
    # offset and size within the flattened tensor, and the chunk's byte offset within the transport bucket (None when not
    # yet placed).
    name: str
    """The name of the weight tensor."""
    shape: torch.Size
    """The shape of the weight tensor."""
    dtype: torch.dtype
    """The dtype of the weight tensor."""
    chunk_offset: int
    """The chunk offset of the weight tensor."""
    chunk_size: int
    """The chunk size of the weight tensor."""
    offset: int
    """The offset of the weight tensor in the bucket."""


class CheckpointEngineRegistry:
    # [CLASS DOC]
    # Process-global, class-level registry mapping backend name -> engine class, plus a mapping of engine module name ->
    # ImportError for optional backends that failed to import (used to enrich lookup errors).
    """Checkpoint engine registry."""

    _registry: dict[str, type["CheckpointEngine"]] = {}

    # Engine modules whose import failed, keyed by module name. Each engine pulls
    # its own transport dependency (cupy for nccl/nixl, nixl, torch_npu for hccl,
    # ...) and `verl.checkpoint_engine` imports them all optionally, so a missing
    # dependency would otherwise only show up as an unregistered backend.
    _import_errors: dict[str, ImportError] = {}

    def register(backend: str):
        # [FUNCTION DOC]
        # Decorator factory (called on the class, not an instance): returns a decorator that stores the decorated class under
        # backend in the registry (overwriting any previous entry) and returns the class unchanged.
        # Inputs: backend - name string. Outputs: the decorator.
        """Register a checkpoint engine.

        Args:
            backend: The backend of the checkpoint engine.
        """

        def wrapper(cls: type["CheckpointEngine"]):
            # [FUNCTION DOC]
            # HELPER: Registers cls under the captured backend name and returns cls.
            CheckpointEngineRegistry._registry[backend] = cls
            return cls

        return wrapper

    @classmethod
    def record_import_error(cls, module: str, error: ImportError):
        # [FUNCTION DOC]
        # Stores the ImportError for the named engine module (overwrites an earlier entry). Side effects: mutates class state.
        """Record an engine module that could not be imported.

        Args:
            module: The name of the checkpoint engine module.
            error: The import error raised by the module.
        """
        cls._import_errors[module] = error

    @classmethod
    def get(cls, backend: str) -> type["CheckpointEngine"]:
        # [FUNCTION DOC]
        # Returns the class registered for backend.
        # Exceptions: ValueError listing the sorted registered backends and, if any were recorded, every failed engine module
        # with its error, when backend is not registered.
        """Get the checkpoint engine class.

        Args:
            backend: The backend of the checkpoint engine.

        Returns:
            The checkpoint engine class.
        """
        if backend not in cls._registry:
            message = f"Checkpoint engine {backend} not registered, registered backends: {sorted(cls._registry)}"
            if cls._import_errors:
                unavailable = ", ".join(f"{module}: {error}" for module, error in sorted(cls._import_errors.items()))
                message += f". Engine modules that failed to import: {unavailable}"
            raise ValueError(message)
        return cls._registry[backend]

    @classmethod
    def new(cls, backend: str, *args, **kwargs) -> "CheckpointEngine":
        # [FUNCTION DOC]
        # Looks up backend via get and instantiates it with the given positional and keyword arguments. Outputs: the instance.
        # Exceptions: as get, plus whatever the constructor raises.
        """Create a new checkpoint engine instance.

        Args:
            backend: The backend of the checkpoint engine.
            *args: Variable length argument pass to the checkpoint engine constructor.
            **kwargs: Arbitrary keyword arguments pass to the checkpoint engine constructor.

        Returns:
            A new checkpoint engine instance.
        """
        return cls.get(backend)(*args, **kwargs)


class CheckpointEngine(ABC):
    # [CLASS DOC]
    # Abstract weight-transfer engine. Lifecycle per sync: prepare (allocate/register buffers, return metadata) on every
    # worker, build_topology (class-level, from all metadata) to produce per-rank init kwargs, init_process_group, then
    # send_weights on actor ranks / receive_weights on rollout ranks, and finally finalize.
    # Class attribute wire_format tells the receiving server adapter how receive_weights yields data: "named_tensors"
    # (name, tensor) pairs, or "delta_flush" (named_tensors, is_last) flushes.
    """CheckpointEngine is an abstraction to transfer weights from actor to rollout.

    In actor process:
    >>> actor = EngineRegistry.new(...) # FSDP, Megatron, VeOmini, TorchTitan, ...
    >>> engine = CheckpointEngine.new(...) # NCCLCheckpointEngine, NIXLCheckpointEngine, ...
    >>> await engine.send_weights(actor.get_per_tensor_param())

    In rollout process:
    >>> engine = CheckpointEngine.new(...)
    >>> server_adapter = ServerAdapter()
    >>> await server_adapter.update_weights(engine.get_weights()) # update weights via cuda ipc
    """

    # How receive_weights yields weights to the server adapter:
    #   "named_tensors" -- (name, tensor) pairs, bucketed into full-tensor loads.
    #   "delta_flush"   -- (named_tensors, is_last) flushes; the seed may be dense,
    #                       while steady updates carry sparse patches.
    wire_format = "named_tensors"

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Abstract. Allocates (and optionally registers) the transfer bucket and returns this worker's metadata dict used for
        # topology building. Exceptions: NotImplementedError in the base.
        # KNOWN ISSUE: the docstring documents a worker_group argument that the signature does not take.
        """Prepare checkpoint engine before each step send_weights/receive_weights.

        1. Allocate weight bucket.
        2. [Optional] Register weight bucket for RDMA.
        3. Return metadata to build communication topology: master ip:port, register RDMA description, etc.

        Args:
            worker_group: The worker group that the checkpoint engine will be used.

        Returns:
            A dictionary that contains the metadata of the worker group.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        # [FUNCTION DOC]
        # Abstract class method. From the actor world size, rollout world size and the list of prepare metadata (actor ranks
        # first, then rollout ranks), returns two dicts (actor kwargs, rollout kwargs) whose values are lists with one entry
        # per rank of the respective group, later dispatched to init_process_group. Exceptions: NotImplementedError in the base.
        """Build communication topology between all workers.

        Args:
            actor_wg_world_size: The world size of the actor worker group.
            rollout_world_size: The world size of the rollout replica.
            metadata: A list of metadata `prepare` from all workers.

        Returns:
            A tuple of two dictionaries that contains the communication topology for actor and rollout worker group.
            Each dict value should be a list argument equal to the world size of the worker group to dispatch to
            `init_process_group`.

            ```
            world_size = rollout.world_size + actor_wg.world_size
            kwargs = {
                "rank": list(range(world_size)),
                "world_size": [world_size] * world_size,
                "master_metadata": [metadata[0]] * world_size,
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def init_process_group(self, **kwargs):
        # [FUNCTION DOC]
        # Abstract. Joins the communication group using one rank's slice of build_topology kwargs.
        """Init process group for checkpoint engine.

        Args:
            **kwargs: Keyword arguments from `build_topology`.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        # [FUNCTION DOC]
        # Abstract. Frees (and optionally deregisters) buffers and may destroy the process group after a sync.
        """Finalize checkpoint engine after each step send_weights/receive_weights.

        1. Free weight bucket.
        1. [Optional] Deregister weight bucket for RDMA.
        2. [Optional] Destroy process group.
        """
        raise NotImplementedError

    @abstractmethod
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Abstract async. Consumes an iterator of (name, tensor) and transmits them; global_steps optionally tags the version.
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        raise NotImplementedError

    @abstractmethod
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # [FUNCTION DOC]
        # Abstract async generator. Yields received weights in the format given by wire_format; global_steps optionally tags the version.
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


class CheckpointEngineWithCache(CheckpointEngine):
    # [CLASS DOC]
    # Abstract engine that additionally keeps received weights in a local cache (shared memory, disk, ...) so rollout can
    # pull them later without interrupting in-flight requests.
    """Checkpoint engine with local cache: shm, disk, etc. This allow to synchronize weights without interrupting
    rollout ongoing requests (partial rollout). After requests exhausted, rollout can get weights from local cache.

    Laminar: https://arxiv.org/abs/2510.12633
    """

    @abstractmethod
    async def get_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        # [FUNCTION DOC]
        # Abstract async generator yielding (name, tensor) from the local cache.
        """Get the weights of the model from local cache.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        raise NotImplementedError


@CheckpointEngineRegistry.register("naive")
class ColocatedCheckpointEngine(CheckpointEngine):
    # [CLASS DOC]
    # Registered as "naive". For actor and rollout sharing a GPU/process: send_weights merely stores the generator and
    # receive_weights yields from it. Topology/process-group methods are unsupported. Methods are synchronous.
    """Checkpoint engine for actor and rollout colocated on same GPU.

    In actor process:
    >>> engine = ColocatedCheckpointEngine()
    >>> actor = Actor()
    >>> server_adapter = ServerAdapter()
    >>> engine.send_weights(actor.get_per_tensor_param())
    >>> server_adapter.update_weights(engine.receive_weights())
    """

    def __init__(self, bucket_size: int, is_master: bool = False) -> None:
        # [FUNCTION DOC]
        # Stores bucket_size (bytes) and is_master flag; neither is used by the transfer logic.
        self.bucket_size = bucket_size
        self.is_master = is_master

    def prepare(self):
        # [FUNCTION DOC]
        # Unsupported; raises NotImplementedError.
        raise NotImplementedError

    def init_process_group(self, **kwargs):
        # [FUNCTION DOC]
        # Unsupported; raises NotImplementedError.
        raise NotImplementedError

    def finalize(self):
        # [FUNCTION DOC]
        # Unsupported; raises NotImplementedError.
        raise NotImplementedError

    @classmethod
    def build_topology(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Unsupported; raises NotImplementedError.
        raise NotImplementedError

    def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Stores the weights iterator on the instance (lazy; nothing is consumed). global_steps is ignored.
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
            global_steps: Optional trainer step/version associated with this weight update.
        """
        self.weights = weights

    def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        # [FUNCTION DOC]
        # Generator yielding every item from the stored weights iterator, then clears it. global_steps is ignored.
        # Exceptions: TypeError if send_weights was not called first (stored value missing raises AttributeError, or None after a previous receive).
        """Receive the weights of the model.

        Args:
            global_steps: Optional trainer step/version associated with this weight update.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        yield from self.weights
        self.weights = None


class CheckpointEngineWorker(Worker):
    # [CLASS DOC]
    # Single-controller Worker (made a Ray actor below as _worker_cls) colocated with an inference engine process. Creates
    # the configured checkpoint engine and a server adapter (rollout class) used to push received weights into the engine.
    """CheckpointEngineWorker colocated with inference engine's WorkerProc on same GPU.

    Args:
        rollout_config: The rollout configuration.
        model_config: The model configuration.
        server_adapter: The server adapter to update weights.
    """

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Stores configs; validates that backend "delta_sharded" is only used with sglang or vllm rollouts (else
        # NotImplementedError); computes bucket size as update_weights_bucket_megabytes x 2^20 bytes; imports
        # custom_backend_module (if set) so plugins can register; instantiates the engine from the registry with the bucket size
        # and backend-specific engine_kwargs; keeps extra args; if no server_adapter is given, constructs the rollout class for
        # (rollout name, mode) with those extra args, configs and no device mesh; finally initializes a global CPU gloo process
        # group via Ray (no timeout).
        # Exceptions: NotImplementedError as above; ValueError for unknown backends.
        super().__init__()
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        if backend == "delta_sharded" and self.rollout_config.name not in {"sglang", "vllm"}:
            raise NotImplementedError(
                f"checkpoint_engine.backend={backend!r} has no delta weight consumer for "
                f"rollout.name={self.rollout_config.name!r}; use sglang or vllm"
            )
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        # If custom_backend_module is set, import it so plugins can register
        # in CheckpointEngineRegistry before the backend is instantiated.
        import_external_libs(self.rollout_config.checkpoint_engine.custom_backend_module or None)
        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )
        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        # [FUNCTION DOC]
        # Ray method dispatched to all ranks, non-blocking. Gets the engine's receive_weights generator and awaits the server
        # adapter's update_weights with it, the global_steps, and the engine's wire_format (default "named_tensors").
        weights = self.checkpoint_engine.receive_weights(global_steps=global_steps)
        await self.server_adapter.update_weights(
            weights,
            global_steps=global_steps,
            wire_format=getattr(self.checkpoint_engine, "wire_format", "named_tensors"),
        )

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Ray method with DP_COMPUTE dispatch, non-blocking: calls the named method on the engine with the given arguments and
        # returns its result (used for prepare / init_process_group / finalize).
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_replica_rank(self) -> int:
        # [FUNCTION DOC]
        # Returns the server adapter's replica_rank.
        """Get replica rank from the underlying rollout server adapter."""
        return self.server_adapter.replica_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def is_leader_rank(self) -> bool:
        # [FUNCTION DOC]
        # Returns the server adapter's is_leader_rank flag.
        """Get leader rank flag from the underlying rollout server adapter."""
        return self.server_adapter.is_leader_rank


_worker_cls = ray.remote(CheckpointEngineWorker)


class CheckpointEngineManager:
    # [CLASS DOC]
    # Driver-side coordinator of weight sync between the actor worker group and rollout replicas. Holds the config, backend
    # name and resolved backend class, the actor worker group, and a mutable replica list. Async methods are wrapped with
    # auto_await so they can be called from sync or async code.
    """Checkpoint engine manager to coordinate weight synchronization between actor and rollout replicas.

    - ME: model engine, FSDP, MCore, VeOmni, export full tensor generator `get_per_tensor_param`
    - CE: checkpoint engine, NCCL, NIXL, etc

    In actor, model engine and checkpoint engine are in same process.
    In rollout, checkpoint engine and rollout worker are in separate process, update weights via cuda ipc.

    ```
    ┌────────┬────────┬─────┬────────┐         ┌───────────────────┬───────────────────┐
    │ ┌────┐ │ ┌────┐ │     │ ┌────┐ │         │     Replica 0     │     Replica 1     │
    │ │ ME0│ │ │ ME1│ │     │ │ MEn│ │         ├────┬────┬────┬────┼────┬────┬────┬────┤
    │ └──┬─┘ │ └────┘ │ ... │ └────┘ │         │ 0  │ 1  │ 2  │ 3  │ 0  │ 1  │ 2  │ 3  │
    │    v   |        |     |        |         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
    | ┌──┴─┐ │ ┌────┐ │     │ ┌────┐ │            ^    ^    ^   cuda ipc   ^    ^    ^
    │ │ CE │ │ │ CE │ │     │ │ CE │ │         ┌──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┬──┴─┐
    │ └──┬─┘ │ └────┘ │     │ └────┘ │         │ CE │ CE │ CE │ CE │ CE │ CE │ CE │ CE |
    └────┼───┴────────┴─────┴────────┘         └──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┴──┬─┘
         v                                        |    |    |    |    |    |    |    |
         └─────────────(nccl/nixl/..)─────────────┴────┴────┴────┴────┴────┴────┴────┘
    ```

    Args:
        config: The checkpoint engine config.
        actor_wg: The actor worker group (the training side that produces weights).
        replicas: The list of rollout replicas.
    """

    def __init__(
        self,
        config: CheckpointEngineConfig,
        actor_wg: RayWorkerGroup,
        replicas: list[RolloutReplica],
    ) -> None:
        # [FUNCTION DOC]
        # Stores config and backend, imports custom_backend_module (if set), resolves the backend class from the registry
        # (ValueError if unknown), and stores actor_wg and replicas (the list is kept by reference).
        self.config = config
        self.backend = config.backend
        import_external_libs(self.config.custom_backend_module or None)
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.actor_wg = actor_wg
        self.replicas = replicas

    def build_process_group(self, rollout: RayWorkerGroup):
        # [FUNCTION DOC]
        # Calls prepare on every actor and rollout rank and gathers metadata (actor ranks first); builds topology via the
        # backend class; asserts every kwargs list matches its group's world size (AssertionError otherwise); adds method
        # "init_process_group" for every rank and invokes it on all ranks of both groups, waiting for completion.
        # Inputs: rollout - temporary worker group of all replica workers.
        """Build process group for actor worker group and rollout replicas."""
        actor_wg = self.actor_wg

        # 1. prepare all workers
        metadata = ray.get(
            actor_wg.execute_checkpoint_engine(["prepare"] * actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["prepare"] * rollout.world_size)
        )

        # 2. build communication topology between all workers
        actor_wg_kwargs, rollout_kwargs = self.backend_cls.build_topology(
            actor_wg.world_size, rollout.world_size, metadata
        )
        for k, v in actor_wg_kwargs.items():
            assert len(v) == actor_wg.world_size, f"actor_wg_kwargs[{k}] must have length of {actor_wg.world_size}"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] must have length of {rollout.world_size}"

        actor_wg_kwargs["method"] = ["init_process_group"] * actor_wg.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size

        # 3. init process group between all workers
        ray.get(
            actor_wg.execute_checkpoint_engine(**actor_wg_kwargs) + rollout.execute_checkpoint_engine(**rollout_kwargs)
        )

    def add_replicas(self, replicas: list[RolloutReplica]):
        # [FUNCTION DOC]
        # Appends replicas to the managed list (the process group is rebuilt on the next update_weights).
        """Add rollout replicas to the manager for elastic scale up, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to add.
        """
        self.replicas.extend(replicas)

    def remove_replicas(self, replicas: list[RolloutReplica]):
        # [FUNCTION DOC]
        # Removes the given replicas (by identity/hash) from the managed list, building a new list.
        """Remove rollout replicas from the manager for elastic scale down, will rebuild process group.

        Args:
            replicas: The list of rollout replicas to remove.
        """
        replicas_set = set(replicas)
        self.replicas = [r for r in self.replicas if r not in replicas_set]

    @auto_await
    async def sleep_replicas(self):
        # [FUNCTION DOC]
        # Concurrently calls sleep on all replicas (frees weights and KV cache).
        """Sleep all rollout replicas: free weight and kv_cache device memory."""
        await asyncio.gather(*[r.sleep() for r in self.replicas])

    @auto_await
    async def wake_up_replicas(self):
        # [FUNCTION DOC]
        # Concurrently calls wake_up on all replicas.
        """Resume all rollout replicas: recover kv_cache and weights device memory."""
        await asyncio.gather(*[r.wake_up() for r in self.replicas])

    @auto_await
    async def abort_replicas(self, reject_request: bool = False):
        # [FUNCTION DOC]
        # Concurrently aborts all in-flight requests on every replica, passing reject_request (default False).
        """Abort all in-flight requests on every replica.

        Args:
            reject_request: Fail requests arriving behind the closed gate instead of
                parking them, for replicas that will not resume generation soon.
        """
        await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])

    @auto_await
    async def resume_generation_replicas(self):
        # [FUNCTION DOC]
        # Concurrently resumes generation on all replicas.
        """Resume generation on all replicas after abort_all_requests."""
        await asyncio.gather(*[r.resume_generation() for r in self.replicas])

    @auto_await
    async def release_kv_cache_replicas(self):
        # [FUNCTION DOC]
        # Concurrently releases only the KV cache on all replicas (weights stay in place).
        """Release kv_cache of all rollout replicas before NCCL weight sync.

        Unlike sleep_replicas(), this only frees the kv_cache and leaves model
        weights untouched, so the NCCL transfer can write directly into the
        existing weight buffers.  Call resume_kv_cache_replicas() after sync.
        """
        await asyncio.gather(*[r.release_kv_cache() for r in self.replicas])

    @auto_await
    async def resume_kv_cache_replicas(self):
        # [FUNCTION DOC]
        # Concurrently restores the KV cache on all replicas.
        """Restore kv_cache of all rollout replicas after NCCL weight sync.

        Counterpart to release_kv_cache_replicas().
        """
        await asyncio.gather(*[r.resume_kv_cache() for r in self.replicas])

    @auto_await
    async def update_weights(self, global_steps: int = None):
        # [FUNCTION DOC]
        # Performs one weight sync. For backend "naive": calls the actor group's update_weights (colocated mode) and returns {}.
        # Otherwise: abort replicas' requests; build a temporary Ray worker group from all replicas' workers using the
        # checkpoint-engine worker class; release KV caches; build the process group; run actor update_weights and rollout
        # update_weights concurrently and wait; merge dict results from the actor ranks into sync metrics; call finalize on
        # all ranks; resume KV caches; resume generation.
        # Inputs: global_steps - optional version tag. Outputs: dict of merged sync metrics (later ranks overwrite same keys).
        # Side effects: extensive remote calls; replica request state is interrupted and resumed.
        """Update weights from actor worker group to rollout replicas.

        Args:
            global_steps: The global steps of the actor worker group.
        """

        # 0. update weights for sync training with colocated actor and rollout
        if self.backend == "naive":
            ray.get(self.actor_wg.update_weights(global_steps=global_steps, mode=self.backend))
            return {}

        # 1. abort and save all unfinished requests for partial rollout
        await self.abort_replicas()

        # 2. create a temporay worker group for all replicas
        workers = []
        for replica in self.replicas:
            workers.extend(replica.workers)
        rollout = RayWorkerGroup(worker_handles=workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_cls))
        actor_wg = self.actor_wg

        # 3. release kv_cache before weight sync (weights stay in place)
        await self.release_kv_cache_replicas()

        # 4. build process group
        self.build_process_group(rollout)

        # 5. update weights of all workers
        results = ray.get(
            actor_wg.update_weights(global_steps=global_steps, mode=self.backend)
            + rollout.update_weights(global_steps=global_steps)
        )
        # The sender workers return the engine's per-sync metrics (empty for
        # backends that don't track any); merge and hand them to the trainer.
        sync_metrics: dict = {}
        for result in results[: actor_wg.world_size]:
            if isinstance(result, dict):
                sync_metrics.update(result)

        # 6. finalize all workers
        ray.get(
            actor_wg.execute_checkpoint_engine(["finalize"] * actor_wg.world_size)
            + rollout.execute_checkpoint_engine(["finalize"] * rollout.world_size)
        )

        # 7. restore kv_cache after weight sync
        await self.resume_kv_cache_replicas()

        # 8. resume all unfinished requests for partial rollout
        await self.resume_generation_replicas()

        return sync_metrics


async def split_weight_chunks(
    weights: Generator[tuple[str, torch.Tensor], None, None], bucket_size: int, meta_only: bool = False
) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor | None], None]:
    # [FUNCTION DOC]
    # Async generator splitting each (name, tensor) from a sync or async iterator into byte chunks of at most bucket_size
    # bytes, yielding (TensorMeta with chunk offset/size and offset None, chunk as a uint8 view or None when meta_only).
    # Zero-byte tensors yield nothing. The tensor must be viewable as flat (contiguous).
    """Split the weight into chunks.

    Args:
        weights: The weights generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the weight chunk metadata and the buffer.
    """
    async for name, weight in ensure_async_iterator(weights):
        buffer = weight.view(-1).view(torch.uint8)
        chunk_offset = 0
        while chunk_offset < weight.nbytes:
            chunk_size = min(bucket_size, weight.nbytes - chunk_offset)
            tensor_meta = TensorMeta(
                name=name,
                shape=weight.shape,
                dtype=weight.dtype,
                chunk_offset=chunk_offset,
                chunk_size=chunk_size,
                offset=None,
            )
            yield (tensor_meta, None if meta_only else buffer[chunk_offset : chunk_offset + chunk_size])
            chunk_offset += chunk_size


async def merge_weight_chunks(
    chunks: Generator[tuple[TensorMeta, torch.Tensor], None, None], bucket_size: int
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    # [FUNCTION DOC]
    # Async generator inverse of split_weight_chunks over an async iterator of (TensorMeta, uint8 chunk). Tensors whose total
    # byte size fits in bucket_size are yielded immediately as a dtype/shape view of the chunk (no copy). Larger tensors are
    # assembled into a freshly allocated tensor on the chunk's device, requiring chunks in order starting at offset 0, and
    # yielded once the last byte arrives.
    # Exceptions: AssertionError on non-uint8 chunks, interleaved tensors, or out-of-order offsets.
    # KNOWN ISSUE: annotated as taking a Generator but iterated with async-for, so it requires an async iterable.
    """Merge the weight chunks into the original weight.

    Args:
        chunks: The chunks generator.
        bucket_size: Max bucket size in bytes.

    Yields:
        A tuple of the name of the weight tensor and the tensor itself.
    """
    merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
    async for tensor_meta, chunk in chunks:
        assert chunk.dtype == torch.uint8, f"Chunk dtype must be uint8, but got {chunk.dtype}"
        nbytes = tensor_meta.shape.numel() * tensor_meta.dtype.itemsize

        # weight is small enough to fit in one bucket
        if nbytes <= bucket_size:
            assert merge_weight is None, f"Weight must be None, but got {merge_name}"
            name, weight = tensor_meta.name, chunk.view(tensor_meta.dtype).view(tensor_meta.shape)
            yield (name, weight)
            continue

        if merge_weight is None:
            assert tensor_meta.chunk_offset == 0, f"Chunk offset must be 0, but got {tensor_meta}"
            merge_name, merge_weight = (
                tensor_meta.name,
                torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=chunk.device),
            )
            merge_buffer = merge_weight.view(-1).view(torch.uint8)
            merge_offset = 0

        assert tensor_meta.name == merge_name
        assert merge_offset == tensor_meta.chunk_offset
        merge_buffer[tensor_meta.chunk_offset : tensor_meta.chunk_offset + tensor_meta.chunk_size] = chunk
        merge_offset += tensor_meta.chunk_size
        if tensor_meta.chunk_offset + tensor_meta.chunk_size == nbytes:
            yield (merge_name, merge_weight)
            merge_name, merge_weight, merge_buffer, merge_offset = None, None, None, 0
