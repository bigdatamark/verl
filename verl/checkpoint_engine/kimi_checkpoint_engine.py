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
# Module kimi_checkpoint_engine: checkpoint engine built on Moonshot's checkpoint_engine package
# (ParameterServer). Trainer ranks offload their share of weights to CPU and register them as a named
# checkpoint; rollout ranks pull them via RDMA/P2P into a device H2D buffer and broadcast bucket by bucket
# inside a rollout-only process group. Provides a replacement receive_tensor coroutine that is bound onto the
# ParameterServer instance. Side effects: module logger level from VERL_LOGGING_LEVEL.
import asyncio
import concurrent.futures
import logging
import os
import time
import types
from collections import defaultdict
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import checkpoint_engine.distributed as dist
import ray
import torch
from checkpoint_engine.ps import H2DBucket, ParameterMeta, ParameterServer, _gen_h2d_buckets, _to_named_tensor

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry
from verl.utils.device import get_nccl_backend, get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def ckpt_get_named_tensor_buckets(
    iterable: Generator[tuple[str, torch.Tensor], None, None],
    bucket_bytes: int,
    world_size: int,
    rank_id: int,
    rollout_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Behavior: generator that round-robin assigns tensors to ranks by enumeration index modulo world_size;
    # every tensor is cast to rollout_dtype, and those owned by rank_id are packed into dict buckets whose total
    # bytes do not exceed bucket_bytes (a single oversize tensor forms its own bucket). Yields each non-empty bucket.
    # Inputs: iterable of (name, tensor); bucket_bytes; world_size; rank_id; rollout_dtype.
    # Outputs: yields mappings name to tensor. Exceptions: ValueError if bucket_bytes <= 0 (raised on first iteration).
    # KNOWN ISSUE: annotated as returning a dict but is a generator; also casts every tensor, including those not owned by this rank.
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = {}
    current_size = 0
    for tensor_idx, (name, tensor) in enumerate(iterable):
        tensor = tensor.to(rollout_dtype)
        if tensor_idx % world_size == rank_id:
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                    current_bucket = {}
                    current_size = 0

            current_bucket[name] = tensor
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


async def receive_tensor(
    self,
    checkpoint_name: str,
    ranks_group: int,
    ranks: list[int] | None = None,
    bucket_size: int = 2 << 30,
    disable_h2d_buffer: bool = False,
) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
    # [FUNCTION DOC]
    # Behavior: replacement for ParameterServer.receive_tensor, bound as a method (self is the ParameterServer).
    # Barriers on ranks_group, plans H2D buckets from the global parameter metas across the given ranks, optionally
    # allocates and registers an H2D staging buffer in the P2P store, and allocates a 2x bucket_size double buffer.
    # For each bucket round i: copies this rank's i-th owned bucket into the H2D buffer; then for each receiver rank
    # with an i-th bucket, the owning receiver copies data into the alternating half of the double buffer and a
    # broadcast from that rank is started on a worker thread. After the first broadcast, each later step yields the
    # previous bucket's tensors (views into the double buffer at the previous half's offset, by name, dtype, shape)
    # before waiting for the current broadcast and synchronizing. After the loop, yields the last bucket.
    # In a finally block: barrier, unregister the H2D buffer, empty the device cache.
    # Inputs: checkpoint_name; ranks_group - process group; ranks - participating ranks; bucket_size - bytes;
    # disable_h2d_buffer - copy directly into the broadcast buffer instead of staging.
    # Outputs: async generator of (name, tensor view). Yielded tensors alias the reusable buffer.
    # Exceptions: AssertionError if metas empty, dist not initialized, P2P store missing, or ranks empty.
    # Side effects: collectives, RDMA reads, device allocations, P2P registration.
    assert len(self._current_global_parameter_metas) != 0, "parameter metas is empty"
    assert dist.is_initialized(), "process group is not initialized"
    assert self._p2p_store is not None, "p2p store is not initialized"
    assert ranks, "ranks should be set"

    # first execute a barrier to avoid subsequent device oom
    dist.barrier(group=ranks_group)
    buckets = _gen_h2d_buckets(
        self._current_global_parameter_metas,
        bucket_size,
        self._local_rdma_devices,
        self._remote_rdma_devices,
        ranks,
    )
    h2d_buffer: torch.Tensor | None = (
        None
        if disable_h2d_buffer
        else torch.empty(bucket_size, dtype=torch.uint8, device=self.device_manager.device_type)
    )
    # p2p store need to register h2d_buffer to let other ranks read
    if ranks:
        h2d_buffer_name = "__h2d_buffer__"
        if h2d_buffer is not None and self._p2p_store is not None:
            self._p2p_store.register_named_tensors({h2d_buffer_name: h2d_buffer})
    receiver_rank_buckets: list[tuple[int, H2DBucket]] = []
    for receiver_rank, owner_rank, bucket in buckets:
        if receiver_rank != self._rank:
            continue
        receiver_rank_buckets.append((owner_rank, bucket))
    buffer = torch.empty(bucket_size * 2, dtype=torch.uint8, device=self.device_manager.device_type)
    buckets_by_receiver_rank: dict[int, list[H2DBucket]] = defaultdict(list)

    max_len = 0
    for receiver_rank, _, bucket in buckets:
        buckets_by_receiver_rank[receiver_rank].append(bucket)
        if len(buckets_by_receiver_rank[receiver_rank]) > max_len:
            max_len = len(buckets_by_receiver_rank[receiver_rank])
    gidx = 0
    metadata: list[ParameterMeta]
    try:
        for i in range(max_len):
            if i < len(receiver_rank_buckets) and not disable_h2d_buffer:
                self._copy_to_buffer(
                    checkpoint_name,
                    receiver_rank_buckets[i][1],
                    h2d_buffer,
                    receiver_rank_buckets[i][0] if ranks else None,
                )
            for receiver_rank, _buckets in buckets_by_receiver_rank.items():
                if i >= len(_buckets):
                    continue
                bucket = _buckets[i]
                start = gidx % 2 * bucket_size
                buffer_b: torch.Tensor = buffer[start : start + bucket.size]
                if receiver_rank == self._rank:
                    if disable_h2d_buffer:
                        self._copy_to_buffer(checkpoint_name, bucket, buffer_b)
                    else:
                        buffer_b.data.copy_(h2d_buffer[: bucket.size])
                broadcast_op = BroadcastOperation(
                    rank=receiver_rank,
                    ranks_group=ranks_group,
                    bucket=buffer_b,
                    metadata=bucket.items,
                )
                if gidx == 0:
                    metadata = await broadcast_op.wait_for_complete()
                    gidx += 1
                    continue
                meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
                for item in meta_list:
                    shape = item["shape"]
                    if isinstance(shape, list | tuple):
                        shape = torch.Size(shape)
                    assert isinstance(shape, torch.Size)
                    dtype, offset = item["dtype"], item["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    yield item["name"], tensor
                metadata = await broadcast_op.wait_for_complete()
                self.device_manager.device_module.synchronize()
                gidx += 1

        meta_list = _to_named_tensor(metadata, (gidx - 1) % 2 * bucket_size)
        for item in meta_list:
            shape = item["shape"]
            if isinstance(shape, list | tuple):
                shape = torch.Size(shape)
            assert isinstance(shape, torch.Size)
            dtype, offset = item["dtype"], item["offset"]
            size = dtype.itemsize * shape.numel()
            tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
            yield item["name"], tensor

    finally:
        dist.barrier(group=ranks_group)
        if ranks and h2d_buffer is not None:
            self._p2p_store.unregister_named_tensors([h2d_buffer_name])
        self.device_manager.device_module.empty_cache()


@dataclass
class MasterMetadata:
    # [CLASS DOC]
    # Dataclass of master endpoints; only dist_ip/dist_port are used here (zmq fields are None).
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


class BroadcastOperation:
    # [CLASS DOC]
    # Wraps one broadcast that runs on the default thread-pool executor of the running asyncio loop.
    """Async broadcast operation in separate thread.

    Args:
        rank (int): The rank of the current process.
        ranks_group (int): The process group's value.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (list[ParameterMeta]): The metadata of the tensor.
    """

    def __init__(
        self,
        rank: int,
        ranks_group: int,
        bucket: torch.Tensor,
        metadata: list[ParameterMeta],
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores arguments and schedules the broadcast in an executor. Inputs: rank - source rank;
        # ranks_group - process group; bucket - tensor; metadata - list of ParameterMeta to hand back.
        # Exceptions: RuntimeError if no running event loop. Side effects: starts a background broadcast.
        self.rank = rank
        self.ranks_group = ranks_group
        self.bucket = bucket
        self.metadata = metadata

        loop = asyncio.get_running_loop()
        self._task = loop.run_in_executor(None, self._run)

    def _run(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: broadcasts the bucket from self.rank within ranks_group. Side effects: collective communication.
        # broadcast tensor
        dist.broadcast(self.bucket, src=self.rank, group=self.ranks_group)

    async def wait_for_complete(self) -> list[ParameterMeta]:
        # [FUNCTION DOC]
        # Behavior: awaits the executor task. Outputs: the stored metadata list. Exceptions: propagates broadcast errors.
        """Wait for the broadcast operation to complete.

        Returns:
            list[ParameterMeta]: The bucket meta after broadcast.
        """
        await self._task
        return self.metadata


@CheckpointEngineRegistry.register("kimi_ckpt_engine")
class KIMICheckpointEngine(CheckpointEngine):
    # [CLASS DOC]
    # CheckpointEngine using the kimi ParameterServer. All trainer ranks (0..A-1) and rollout ranks (A..A+R-1)
    # join one global group; a sub-group of rollout ranks is used for broadcasts.
    """kimi checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        rebuild_group (bool): Whether to rebuild the process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores configuration; marks not initialized; fixed checkpoint name "kimi_checkpoint_engine".
        # Inputs: bucket_size, rebuild_group, is_master, rollout_dtype. Side effects: none.
        self.bucket_size = bucket_size
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.initialized = False
        self.checkpoint_name = "kimi_checkpoint_engine"

    def prepare(self) -> MasterMetadata:
        # [FUNCTION DOC]
        # Behavior: on master, resolves node IP and a free port. Outputs: MasterMetadata (dist fields only) on master, else None.
        if self.is_master:
            self.ip = ray.util.get_node_ip_address().strip("[]")
            self.listen_port, _ = get_free_port(self.ip)

        return (
            MasterMetadata(zmq_ip=None, zmq_port=None, dist_ip=self.ip, dist_port=self.listen_port)
            if self.is_master
            else None
        )

    def finalize(self):
        # [FUNCTION DOC]
        # Behavior: if rebuild_group, destroys the default process group and resets rank, world size and initialized flag.
        """Destroy the ckpt engine process group if rebuild_group is True."""
        if self.rebuild_group:
            dist.destroy_process_group()
            self.rank = None
            self.world_size = None
            self.initialized = False

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        # [FUNCTION DOC]
        # Behavior: builds init kwargs: trainer ranks 0..A-1, rollout ranks A..A+R-1; every entry gets method name
        # "init_process_group", both world sizes and master metadata (metadata[0]).
        # Outputs: tuple of (actor kwargs, rollout kwargs), each a dict of per-worker lists.
        actor_wg_kwargs = {
            "method": ["init_process_group"] * actor_wg_world_size,
            "rank": list(range(0, actor_wg_world_size)),
            "actor_wg_world_size": [actor_wg_world_size] * actor_wg_world_size,
            "rollout_world_size": [rollout_world_size] * actor_wg_world_size,
            "master_metadata": [metadata[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "method": ["init_process_group"] * rollout_world_size,
            "rank": list(range(actor_wg_world_size, actor_wg_world_size + rollout_world_size)),
            "actor_wg_world_size": [actor_wg_world_size] * rollout_world_size,
            "rollout_world_size": [rollout_world_size] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(
        self,
        rank: int,
        actor_wg_world_size: int,
        rollout_world_size: int,
        master_metadata: MasterMetadata,
    ):
        # [FUNCTION DOC]
        # Behavior: records rank and sizes; on first call creates a ParameterServer (manual process group, master
        # address from metadata), binds the custom receive_tensor onto it, selects the vllm NCCL-family backend,
        # initializes the process group, and creates the rollout sub-group. Subsequent calls only update attributes.
        # Inputs: rank, actor_wg_world_size, rollout_world_size, master_metadata. Side effects: distributed init.
        """Initialize the ckpt engine process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
        """
        self.rank = rank
        self.actor_wg_world_size = actor_wg_world_size
        self.rollout_world_size = rollout_world_size
        self.world_size = actor_wg_world_size + rollout_world_size

        if not self.initialized:
            self.parameter_server = ParameterServer(
                rank=rank,
                world_size=self.world_size,
                auto_pg=False,
                master_addr=master_metadata.dist_ip,
                master_port=master_metadata.dist_port,
            )
            self.parameter_server.receive_tensor = types.MethodType(receive_tensor, self.parameter_server)

            dist.use_backend(f"vllm_{get_nccl_backend()}")
            self.parameter_server.init_process_group()

            self.rollout_ranks = list(range(self.actor_wg_world_size, self.world_size))
            self.rollout_group = dist.new_group(self.rollout_ranks)
            self.initialized = True

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: for each bucket of this rank's tensors (round-robin by index), copies tensors to CPU non-blocking
        # with a 32-thread pool and collects them; synchronizes, registers the checkpoint with the parameter server,
        # empties device cache, gathers metas, barriers globally (waiting for receivers to finish) and unregisters.
        # Inputs: weights generator; global_steps unused. Side effects: CPU memory, collectives, logging.
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """

        def offload_cpu(name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
            # [FUNCTION DOC]
            # HELPER: Behavior: returns the name and a non-blocking CPU copy of tensor.
            return name, tensor.to("cpu", non_blocking=True)

        start_time = time.time()
        named_tensors = {}
        for named_tensors_gpu in ckpt_get_named_tensor_buckets(
            weights, self.bucket_size, self.actor_wg_world_size, self.rank, self.rollout_dtype
        ):
            with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                futures = [
                    executor.submit(
                        offload_cpu,
                        name,
                        tensor,
                    )
                    for name, tensor in named_tensors_gpu.items()
                ]
            for future in concurrent.futures.as_completed(futures):
                name, tensor_cpu = future.result()
                named_tensors[name] = tensor_cpu

        get_torch_device().synchronize()

        self.parameter_server.register_checkpoint(self.checkpoint_name, named_tensors=named_tensors)
        named_tensors = {}
        get_torch_device().empty_cache()
        logger.info(f"Rank {self.rank} offload and register, time cost: {time.time() - start_time:.2f}s")

        self.parameter_server.gather_metas(self.checkpoint_name)
        dist.barrier()
        self.parameter_server.unregister_checkpoint(self.checkpoint_name)
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        # [FUNCTION DOC]
        # Behavior: gathers metas, then iterates the bound receive_tensor over the rollout group, counting bytes and
        # params, yielding (name, tensor); then barriers globally and logs time and bandwidth.
        # Inputs: global_steps unused. Outputs: async generator of (name, tensor). Side effects: collectives, logging.
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        self.parameter_server.gather_metas(self.checkpoint_name)

        start_time = time.time()
        total_bytes, total_params = 0, 0
        async for name, tensor in self.parameter_server.receive_tensor(
            self.checkpoint_name, self.rollout_group, self.rollout_ranks, self.bucket_size
        ):
            total_bytes += tensor.element_size() * tensor.nelement()
            total_params += 1
            yield name, tensor
        dist.barrier()
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
