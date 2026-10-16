# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Trainer-side TensorRT-LLM rollout client. ServerAdapter (a BaseRollout) lives in each
# hybrid or standalone checkpoint worker, and drives the replica's TRTLLMHttpServer over
# HTTP through AsyncTRTLLMHttpAdapter: memory release/resume by tag and weight sync by
# gathering CUDA IPC handles from every rank of the replica to the leader, which posts
# them per device UUID. Also provides NVML helpers for device UUID lookup.
# Module state: logger (level from VERL_LOGGING_LEVEL, default WARN); ExecutorMemoryType
# (None when TRT-LLM cannot be imported, e.g. CPU-only actors); HTTP defaults (timeout 60s,
# 3 attempts, 2s retry delay, 2000 connections, 300s max wait); _NVML_INITIALIZED flag and
# _NVML_LOCK guarding one-time NVML init.
from __future__ import annotations

import asyncio
import base64
import contextlib
import gc
import logging
import os
import pickle
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import aiohttp
import pynvml
import ray
import torch
import torch.distributed as dist

try:
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType
except (ImportError, RuntimeError):
    # RuntimeError: FlashInfer's check_cuda_arch() crashes on CPU-only actors
    ExecutorMemoryType = None
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_torch_device
from verl.utils.net_utils import is_valid_ipv6_address
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_CONNECTIONS = 2000
DEFAULT_MAX_WAIT_TIME = 300.0


@contextlib.contextmanager
def nvml_context():
    # [FUNCTION DOC]
    # Context manager that initializes NVML, yields, and always attempts NVML shutdown
    # (shutdown errors ignored).
    # Exceptions: any NVMLError (from init or raised inside the with-block) is re-raised as
    # RuntimeError("Failed to initialize NVML ...").
    # KNOWN ISSUE: NVML errors raised by the caller's block are mislabeled as init failures.
    """Context manager for NVML initialization and shutdown.

    Raises:
        RuntimeError: If NVML initialization fails
    """
    try:
        pynvml.nvmlInit()
        yield
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to initialize NVML: {e}") from e
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


_NVML_INITIALIZED = False
_NVML_LOCK = threading.Lock()


def get_device_uuid(id: str | int) -> str:
    # [FUNCTION DOC]
    # Return the NVML UUID string of the GPU at index id (int or numeric string).
    # Behavior: initializes NVML once per process under a lock (never shut down), gets the
    # device handle by index and its UUID, decoding bytes to str.
    # Exceptions: RuntimeError on NVML init failure, NVML lookup failure, or unexpected UUID
    # type. Side effects: process-global NVML initialization.
    """Get the UUID of a CUDA device using NVML."""
    id = int(id)  # pynvml expects int; ray.get_gpu_ids() may return str
    global _NVML_INITIALIZED
    with _NVML_LOCK:
        if not _NVML_INITIALIZED:
            try:
                pynvml.nvmlInit()
                _NVML_INITIALIZED = True
            except pynvml.NVMLError as e:
                raise RuntimeError(f"Failed to initialize NVML: {e}") from e

    # Get the device handle and UUID
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        # Ensure the UUID is returned as a string, not bytes
        if isinstance(uuid, bytes):
            return uuid.decode("utf-8")
        elif isinstance(uuid, str):
            return uuid
        else:
            raise RuntimeError(f"Unexpected UUID type: {type(uuid)} for device {id} (global index: {id})")
    except pynvml.NVMLError as e:
        raise RuntimeError(f"Failed to get device UUID for device {id} (global index: {id}): {e}") from e


async def _read_async_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Decode an aiohttp response: {} for 204/zero length; JSON (any content type) if
    # parseable; else {"content_type", "text"}; {} if the text cannot be read.
    if resp.status == 204 or (resp.content_length == 0):
        return {}

    try:
        return await resp.json(content_type=None)
    except Exception:
        try:
            text = await resp.text()
        except Exception:
            return {}
        return {
            "content_type": (resp.headers.get("Content-Type") or ""),
            "text": text,
        }


class AsyncTRTLLMHttpAdapter:
    # [CLASS DOC]
    # Minimal async HTTP client for the verl-specific endpoints of the TRT-LLM server
    # (resume_memory, release_memory, update_weights) with retry/backoff; one new pooled
    # aiohttp session per request.
    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ):
        # [FUNCTION DOC]
        # Store host, port, total request timeout, retry count, base retry delay and connection
        # pool size. No I/O.
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.max_connections = max_connections

    @asynccontextmanager
    async def _get_session(self) -> aiohttp.ClientSession:
        # [FUNCTION DOC]
        # HELPER: Async context manager yielding a fresh ClientSession (connector limit
        # max_connections, per-host max_connections // 4, 300 s DNS cache, total timeout
        # self.timeout) and closing it afterwards.
        """Context manager for safe session access with proper connection pooling.

        Yields:
            aiohttp.ClientSession: Session instance for making HTTP requests

        Note:
            This method creates a new session for each request to avoid resource competition
            while still maintaining proper connection pooling through the shared connector.
        """
        # Create a new session for each request to avoid resource competition
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections // 4,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        try:
            yield session
        finally:
            # Always close the session to free up resources
            if not session.closed:
                await session.close()

    async def _make_async_request(
        self,
        endpoint: str,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        method: str = "POST",
        return_status: bool = False,
    ) -> dict[str, Any] | int:
        # [FUNCTION DOC]
        # HELPER: GET or POST (JSON body, default {}) to http://host:port/endpoint with the given
        # timeout, retrying timeouts and connection errors with exponential backoff
        # (retry_delay * 2^attempt). HTTP errors (status >= 400) are logged and re-raised
        # immediately; other errors are re-raised on the last attempt.
        # Outputs: the HTTP status code when return_status, else the decoded response dict.
        # Exceptions: aiohttp.ClientResponseError; RuntimeError after exhausting attempts.
        """Make an async HTTP request with retry logic and consistent error handling.

        Args:
            endpoint (str): The API endpoint to call (without leading slash)
            payload (Optional[Dict[str, Any]], optional): The JSON payload to send.
                Defaults to empty dict if None.
            method (str, optional): HTTP method to use. Defaults to "POST".

        Returns:
            Dict[str, Any]: The JSON response from the server

        Raises:
            aiohttp.ClientResponseError: If the HTTP request fails with a client/server error
            RuntimeError: If all retry attempts are exhausted

        Note:
            - Uses exponential backoff for retries
            - Logs warnings for timeout and connection errors, errors for HTTP errors
        """

        url = f"http://{self.host}:{self.port}/{endpoint}"

        for attempt in range(self.max_attempts):
            try:
                async with self._get_session() as session:
                    if method.upper() == "GET":
                        async with session.get(url, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)
                    else:
                        async with session.post(url, json=payload or {}, timeout=timeout) as response:
                            response.raise_for_status()
                            return response.status if return_status else await _read_async_response(response)

            except asyncio.TimeoutError:
                logger.warning(f"Async request to {endpoint} timed out (attempt {attempt + 1})")
            except aiohttp.ClientConnectorError:
                logger.warning(f"Connection error for {endpoint} (attempt {attempt + 1})")
            except aiohttp.ClientResponseError as e:
                logger.error(f"HTTP error for {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error for {endpoint}: {e}")
                if attempt == self.max_attempts - 1:
                    raise

            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(f"Failed to complete async request to {endpoint} after {self.max_attempts} attempts")

    async def resume_memory_occupation(self, tags: list[str]):
        # [FUNCTION DOC]
        # POST "resume_memory" with the given memory tags; returns the response dict.
        # KNOWN ISSUE: docstring mentions first-time weight reloading via release and optional
        # tags, neither of which the code does.
        """Resume GPU memory occupation (async version).

        Similar to AsyncEngine, this method handles first-time weight reloading
        by calling release_memory_occupation if needed.

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to resume.
                If None, resumes all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory resume status
        """
        return await self._make_async_request("resume_memory", {"tags": tags})

    async def release_memory_occupation(self, tags: list[str]):
        # [FUNCTION DOC]
        # POST "release_memory" with the given memory tags; returns the response dict.
        """Release GPU memory occupation temporarily (async version).

        Args:
            tags (Optional[List[str]], optional): List of tags to specify which memory to release.
                If None, releases all memory. Defaults to None. ["weights", "kv_cache"]

        Returns:
            Dict[str, Any]: Server response indicating memory release status
        """
        return await self._make_async_request("release_memory", {"tags": tags})

    async def update_weights(self, weights: dict[str, str]):
        # [FUNCTION DOC]
        # POST "update_weights" with weights (mapping device UUID to base64 pickled handle list,
        # or None to finalize the update); returns the response dict.
        """Update model weights from tensor data asynchronously.

        Args:
            weights: A dictionary that maps the device uuid of the weight handles.

        Returns:
            Dict[str, Any]: Server response containing update status
        """
        return await self._make_async_request("update_weights", {"weights": weights})


class ServerAdapter(BaseRollout):
    # [CLASS DOC]
    # BaseRollout implementation for TRT-LLM. Class attribute _WEIGHTS_TAGS lists every
    # non-kv-cache, non-internal ExecutorMemoryType value (or a hard-coded fallback list when
    # TRT-LLM is unavailable). Instance state: _adapter (HTTP client, lazy), hybrid_device_mesh
    # (CPU mesh with a flattened "exclude_dp" dim), gpu_id, is_leader_rank (rank 0 of the
    # replica group), replica_rank, server_actor, cached partial-loading support, node_ip.
    # All releasable/resumable weight tags: every ExecutorMemoryType except kv_cache
    # (handled separately) and internal tags prefixed with "_".
    # Fallback to hard-coded list for trtllm versions that don't export ExecutorMemoryType.
    _WEIGHTS_TAGS = (
        [t.value for t in ExecutorMemoryType if t is not ExecutorMemoryType.KV_CACHE and not t.value.startswith("_")]
        if ExecutorMemoryType is not None
        else [
            "sampler",
            "drafter",
            "guided_decoder",
            "spec_resource_manager",
            "model_extra",
            "executor_extra",
            "model",
            "draft_model",
        ]
    )

    @staticmethod
    def get_full_tags() -> list[str]:
        # [FUNCTION DOC]
        # Return all memory tags: the weight tags plus "kv_cache".
        return ServerAdapter._WEIGHTS_TAGS + ["kv_cache"]

    def __init__(
        self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh, replica_rank: int = -1
    ):
        # [FUNCTION DOC]
        # Behavior: initializes BaseRollout; for fp8 sets hf_config.quantization_config to the
        # dynamic e4m3 128x128 block quant dict. Hybrid mode (device_mesh given): requires "dp" to
        # be the first mesh dim, clones the mesh on CPU, flattens the non-dp dims into
        # "exclude_dp", sets is_leader_rank to exclude_dp local rank 0, gpu_id to the single Ray
        # GPU id, and replica_rank to the dp local rank. Standalone mode: replica_rank from the
        # argument, leader iff env RANK is 0, gpu_id from Ray. Records the node IP.
        # Exceptions: AssertionError if dp is not first, not exactly one GPU (hybrid), or
        # replica_rank < 0; KeyError without RANK in standalone; IndexError if Ray assigned no GPU.
        # KNOWN ISSUE: is_dp_rank is initialized but never used; in hybrid mode ray.get_gpu_ids()
        # is indexed before the single-GPU assertion is checked.
        super().__init__(config, model_config, device_mesh)
        if self.config.quantization == "fp8" and self.model_config.hf_config is not None:
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            self.model_config.hf_config.quantization_config = dict(FP8_BLOCK_QUANT_KWARGS)
        self._adapter = None
        self.hybrid_device_mesh = None
        self.gpu_id = None
        self.is_leader_rank = None
        self.replica_rank = None
        self.is_dp_rank = None
        self._supports_partial_loading = None

        # hybrid mode
        if self.device_mesh is not None:
            assert device_mesh.mesh_dim_names.index("dp") == 0, "DP dim should always be the first dimension"

            # Clone a new device mesh for CPU backend only (used for internal ranks communication)
            device_mesh_kwargs = dict(
                mesh_shape=device_mesh.mesh.shape,
                mesh_dim_names=device_mesh.mesh_dim_names,
            )
            self.hybrid_device_mesh = init_device_mesh("cpu", **device_mesh_kwargs)

            self.hybrid_device_mesh[self.hybrid_device_mesh.mesh_dim_names[1:]]._flatten(mesh_dim_name="exclude_dp")
            self.is_leader_rank = self.hybrid_device_mesh["exclude_dp"].get_local_rank() == 0
            logger.info(f"is_dp_leader: {self.is_leader_rank}")
            logger.info(f"exclude_dp_rank = {self.hybrid_device_mesh['exclude_dp'].get_local_rank()}")
            logger.info(f"exclude_dp_size = {self.hybrid_device_mesh['exclude_dp'].size()}")
            self.gpu_id = ray.get_gpu_ids()[0]
            self.replica_rank = self.hybrid_device_mesh["dp"].get_local_rank()
            assert len(ray.get_gpu_ids()) == 1, "ServerAdapter should run on a single GPU node"
        else:
            rank = int(os.environ["RANK"])
            self.replica_rank = replica_rank
            self.is_leader_rank = rank == 0
            # Required for CUDA IPC handle creation during weight sync for Async RL.
            # Reward/ref models skip weight sync so this can be None.
            self.gpu_id = ray.get_gpu_ids()[0]

        # Below is required for all modes.
        assert self.replica_rank >= 0, "replica_rank is not set"
        assert self.is_leader_rank is not None, "is_leader_rank is not set"

        self.node_ip = ray.util.get_node_ip_address().strip("[]")

    async def get_supports_partial_loading(self) -> bool:
        # [FUNCTION DOC]
        # Return (and cache) whether the server's model supports partial weight loading by
        # asking the server actor; failures are logged and treated as False. Lazily initializes
        # the HTTP adapter.
        # KNOWN ISSUE: calls server_actor.supports_partial_loading, which TRTLLMHttpServer does not
        # define (it is a worker-extension RPC), so it presumably always falls back to False.
        """Query and cache whether the model supports partial weight loading."""
        if self._supports_partial_loading is not None:
            return self._supports_partial_loading

        await self._init_server_adapter()
        try:
            self._supports_partial_loading = await self.server_actor.supports_partial_loading.remote()
        except Exception as e:
            logger.warning(f"Failed to query partial loading support: {e}, defaulting to False")
            self._supports_partial_loading = False

        logger.info(f"Model supports partial loading: {self._supports_partial_loading}")
        return self._supports_partial_loading

    async def _init_server_adapter(self):
        # [FUNCTION DOC]
        # HELPER: Lazily create the HTTP adapter (idempotent).
        # Behavior: in standalone mode without a mesh, builds a CPU mesh (dp, infer_tp, infer_pp)
        # from the initialized process group, flattens exclude_dp and recomputes is_leader_rank.
        # Looks up actor "trtllm_server_{replica_rank}", gets its address, asserts it is on this
        # node, and creates AsyncTRTLLMHttpAdapter with the server timeouts from config.
        # Exceptions: AssertionError if distributed is not initialized or the server is on
        # another node.
        if self._adapter is not None:
            return

        # Standalone mode: lazily build the CPU device mesh from the gloo process group
        # (initialized by initialize_global_process_group_ray before ServerAdapter construction).
        # Reward/ref models that never call resume(), release(), or update_weights() will never build the mesh.
        if self.hybrid_device_mesh is None and self.device_mesh is None:
            assert dist.is_initialized(), "gloo process group must be initialized before building device mesh"
            infer_tp = self.config.tensor_model_parallel_size
            infer_pp = getattr(self.config, "pipeline_model_parallel_size", 1)
            world_size = dist.get_world_size()
            dp = world_size // (infer_tp * infer_pp)
            self.hybrid_device_mesh = init_device_mesh(
                "cpu", mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )
            self.hybrid_device_mesh[self.hybrid_device_mesh.mesh_dim_names[1:]]._flatten(mesh_dim_name="exclude_dp")
            self.is_leader_rank = self.hybrid_device_mesh["exclude_dp"].get_local_rank() == 0

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"trtllm_server_{self.replica_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        assert server_address == self.node_ip, f"server address: {server_address} != node_ip: {self.node_ip}"

        logger.debug(f"replica_rank={self.replica_rank}, server address: {server_address}, port: {server_port}")
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._adapter = AsyncTRTLLMHttpAdapter(
            host=host,
            port=server_port,
            timeout=self.config.server.timeout,
            max_attempts=self.config.server.max_attempts,
            retry_delay=self.config.server.retry_delay,
            max_connections=self.config.server.max_connections,
        )

    async def resume(self, tags: list[str]):
        # [FUNCTION DOC]
        # Resume memory on the server. When "kv_cache" is requested and free_cache_engine is set,
        # first barriers across the replica group (in a thread) so other ranks finish offloading.
        # Then the leader (if free_cache_engine) maps "weights" to all weight tags or "kv_cache" to
        # ["kv_cache"] and posts resume.
        # Exceptions: ValueError (leader only) if tags contain neither "weights" nor "kv_cache".
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        # Synchronize all ranks before resuming KV cache to ensure non-leader ranks
        # have completed actor offloading to CPU, preventing OOM issue.
        if "kv_cache" in tags and self.config.free_cache_engine:
            group = self.hybrid_device_mesh["exclude_dp"].get_group() if self.hybrid_device_mesh is not None else None
            await asyncio.to_thread(dist.barrier, group=group)
        if self.is_leader_rank and self.config.free_cache_engine:
            if "weights" in tags:
                tags = self._WEIGHTS_TAGS
            elif "kv_cache" in tags:
                tags = ["kv_cache"]
            else:
                raise ValueError(f"Invalid tag: {tags}")
            await self._init_server_adapter()
            await self._adapter.resume_memory_occupation(tags=tags)

    async def release(self):
        # [FUNCTION DOC]
        # On the leader with free_cache_engine set, release all weight tags and kv_cache.
        """Release weights and kv cache in GPU memory."""
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._init_server_adapter()
            tags = self._WEIGHTS_TAGS + ["kv_cache"]
            await self._adapter.release_memory_occupation(tags=tags)

    async def update_weights_from_ipc_handles(self, device_handles):
        # [FUNCTION DOC]
        # Gather each rank's mapping of device UUID to serialized handles to the replica group's
        # rank 0 (in a thread), merge them on the leader and post them to update_weights, then
        # barrier the group.
        # Inputs: device_handles - mapping from device UUID to base64 pickled handle list.
        # Side effects: collectives, HTTP call.
        """Update weights from IPC handles."""
        if self.hybrid_device_mesh is not None:
            world_size = self.hybrid_device_mesh["exclude_dp"].size()
            group = self.hybrid_device_mesh["exclude_dp"].get_group()
        else:
            world_size = dist.get_world_size()
            group = None

        if self.is_leader_rank:
            gathered_handles = [None for _ in range(world_size)]
        else:
            gathered_handles = None

        await asyncio.to_thread(
            dist.gather_object,
            obj=device_handles,
            object_gather_list=gathered_handles,
            group_dst=0,
            group=group,
        )

        if self.is_leader_rank:
            all_handles = {k: v for d in gathered_handles for k, v in d.items()}
            await self._adapter.update_weights(all_handles)

        await asyncio.to_thread(dist.barrier, group=group)

    async def update_weights(
        self,
        weights: AsyncGenerator[tuple[str, torch.Tensor], None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Stream weights to the TRT-LLM server via CUDA IPC handles.
        # Behavior: only "named_tensors" wire format accepted. The leader lazily initializes the
        # adapter. Optionally quantizes to FP8 by name. Gets this GPU's UUID. Partial loading is
        # always used for non-VLMs; for VLMs the leader queries support and broadcasts it over the
        # exclude_dp group. Iterates weights: with partial loading, flushes the accumulated handles
        # when the next tensor would exceed the bucket budget
        # (checkpoint_engine.update_weights_bucket_megabytes MiB); each tensor is cloned and
        # reduced to an IPC handle. Without partial loading all handles accumulate into one flush.
        # Each flush serializes handles (pickle + base64) under this GPU's UUID and calls
        # update_weights_from_ipc_handles. Finally the leader posts update_weights(None) to
        # finalize and sets global_steps on the server actor; all ranks barrier; frees memory.
        # Exceptions: AssertionError for other wire formats or a tensor larger than the bucket;
        # re-raises device UUID lookup errors.
        # KNOWN ISSUE: in standalone mode the CPU mesh is built only inside _init_server_adapter,
        # which only the leader calls, so non-leaders use the default group (or hit a None mesh
        # in the VLM branch) while the leader uses the exclude_dp group for the same collectives.
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert wire_format == "named_tensors", (
            f"TensorRT-LLM rollout only consumes full named tensors; got wire_format={wire_format!r}"
        )
        if self.is_leader_rank:
            await self._init_server_adapter()

        total_available_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) * 1024 * 1024

        if self.config.get("quantization", None) == "fp8":
            from verl.utils.trtllm.trtllm_fp8_utils import TRTLLMFP8QuantizerHelper

            fp8_quantizer_helper = TRTLLMFP8QuantizerHelper(self.model_config.hf_config.quantization_config)
            weights = fp8_quantizer_helper.quant_weights_by_name(
                weights,
                dtype=self.model_config.hf_config.dtype,
            )

        try:
            device_uuid = get_device_uuid(int(self.gpu_id))
        except Exception as e:
            logger.error(f"Failed to get device UUID in update_weights(): {e}")
            device_uuid = None
            raise e

        cur_available_bytes = total_available_bytes
        cur_handles = []

        async def flush():
            # [FUNCTION DOC]
            # HELPER: If any handles are pending, send them via update_weights_from_ipc_handles and
            # reset the pending list and remaining byte budget (enclosing state).
            nonlocal cur_available_bytes, cur_handles
            if not cur_handles:
                return
            serialized_device_handles = {device_uuid: base64.b64encode(pickle.dumps(cur_handles)).decode("utf-8")}
            await self.update_weights_from_ipc_handles(serialized_device_handles)
            cur_available_bytes = total_available_bytes
            cur_handles = []

        # For non-VLM, always use partial loading. For VLM, leader queries and broadcasts to all
        # ranks in the DP replica; use get_global_rank(group, 0) since each replica has a different leader.
        is_vlm = self.model_config.hf_config is not None and hasattr(self.model_config.hf_config, "vision_config")
        if not is_vlm:
            supports_partial_loading = True
        else:
            exclude_dp_group = self.hybrid_device_mesh["exclude_dp"].get_group()
            spl_tensor = torch.zeros(1, dtype=torch.int32)
            if self.is_leader_rank:
                supports_partial_loading = await self.get_supports_partial_loading()
                spl_tensor[0] = int(supports_partial_loading)
            leader_global_rank = dist.get_global_rank(exclude_dp_group, 0)
            await asyncio.to_thread(dist.broadcast, spl_tensor, src=leader_global_rank, group=exclude_dp_group)
            supports_partial_loading = bool(spl_tensor.item())

        async for name, param in ensure_async_iterator(weights):
            if supports_partial_loading:
                size_in_bytes = param.element_size() * param.numel()
                if size_in_bytes > cur_available_bytes:
                    await flush()

                assert cur_available_bytes >= size_in_bytes, (
                    f"cur_available_bytes: {cur_available_bytes:,} size_in_bytes: {size_in_bytes:,} name: {name}"
                )
                cur_available_bytes -= size_in_bytes

            # Clone so the IPC handle owns the data to avoid buffer reuse
            # before TRT-LLM reads it, which will silently corrupt weights.
            handle = reduce_tensor(param.detach().clone())
            cur_handles.append((name, handle))

        await flush()

        if self.is_leader_rank:
            # Finalize update weights
            await self._adapter.update_weights(None)
            if global_steps is not None:
                await self.server_actor.set_global_steps.remote(global_steps)
        group = self.hybrid_device_mesh["exclude_dp"].get_group() if self.hybrid_device_mesh is not None else None
        await asyncio.to_thread(dist.barrier, group=group)

        del weights
        gc.collect()
        get_torch_device().empty_cache()

    def _get_attribute(self, name: str):
        # [FUNCTION DOC]
        # HELPER: Return the attribute called name (for remote introspection).
        return getattr(self, name)
