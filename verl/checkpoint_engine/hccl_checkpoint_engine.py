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
# Module hccl_checkpoint_engine: Ascend NPU checkpoint engine that streams model weights from trainer
# rank 0 to all rollout workers via HCCL broadcast (through a vLLM StatelessProcessGroup), with per-bucket
# tensor metadata published over a ZeroMQ PUB/SUB side channel. Uses double buffering (send/recv buffers of
# bucket_size bytes) so packing/consumption overlaps with communication.
# Exceptions: raises ImportError at import time if torch.npu is unavailable.
# Side effects: configures a module logger whose level comes from VERL_LOGGING_LEVEL (default WARN).
import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import ray
import torch
import zmq
from vllm.distributed.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.device import is_torch_npu_available
from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

if not is_torch_npu_available(check_device=False):
    raise ImportError("HCCLCheckpointEngine is unavailable because the torch.npu module is not available.")

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class MasterMetadata:
    # [CLASS DOC]
    # Dataclass of master endpoints shared with all participants: zmq_ip/zmq_port for the metadata PUB socket,
    # dist_ip/dist_port for the HCCL rendezvous.
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


class BroadcastOperation:
    # [CLASS DOC]
    # Wraps one bucket broadcast. Construction immediately performs the operation synchronously in the calling
    # thread (despite the docstring's mention of a separate thread): metadata via zmq, then tensor via HCCL.
    # KNOWN ISSUE: docstring says it runs in a separate thread and documents a group_name argument; the code
    # runs synchronously and takes process_group instead.
    """Async broadcast operation with HCCL in separate thread.

    Args:
        rank (int): The rank of the current process.
        group_name (str): The name of the HCCL process group.
        bucket (torch.Tensor): The tensor to broadcast.
        metadata (dict[str, TensorMeta]): The metadata of the tensor.
        socket (zmq.Socket): The zeromq socket to communicate with master.
        topic (str): The topic to subscribe.
    """

    def __init__(
        self,
        rank: int,
        process_group: StatelessProcessGroup | str,
        bucket: torch.Tensor,
        metadata: dict[str, TensorMeta],
        socket: zmq.Socket,
        topic: str,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores arguments and runs the broadcast immediately.
        # Inputs: rank - this process's rank in the group (0 is sender); process_group - HCCL communicator with a
        # broadcast method; bucket - uint8 buffer to send (rank 0) or fill (others); metadata - dict to publish
        # (rank 0) or None; socket - zmq PUB (rank 0) or SUB socket; topic - topic string.
        # Outputs: none. Side effects: network I/O and an enqueued HCCL broadcast.
        self.rank = rank
        self.pyhccl = process_group
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic

        self._run()

    def _run(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: rank 0 sends the topic frame then the pickled metadata on the PUB socket; other ranks
        # read the topic frame and the pickled metadata and store it on self.metadata. Then broadcasts the bucket
        # from source rank 0. Inputs/outputs: none (uses instance state). Side effects: socket I/O, overwrites
        # bucket contents on receivers, mutates self.metadata.
        # broadcast tensor meta via zeromq PUB/SUB
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()

        # broadcast tensor via HCCL
        self.pyhccl.broadcast(self.bucket, src=0)

    async def wait_for_complete(self) -> dict[str, TensorMeta]:
        # [FUNCTION DOC]
        # Behavior: returns the metadata (received or sent). Outputs: dict with "bucket_meta" (name to TensorMeta)
        # and "is_last" flag. Side effects: none; does not wait for the kernel.
        """Wait for the broadcast operation to complete.
        (This does not guarantee that the HCCL kernel has finished, only that it has been enqueued.)

        Returns:
            dict[str, TensorMeta]: The bucket meta after broadcast.
        """
        return self.metadata


@CheckpointEngineRegistry.register("nccl")
class HCCLCheckpointEngine(CheckpointEngine):
    # [CLASS DOC]
    # CheckpointEngine implementation for NPUs. Holds bucket size, group options, the HCCL communicator,
    # the zmq socket, and rank/world size. Master (trainer rank 0) hosts the zmq server and HCCL rendezvous port.
    # KNOWN ISSUE: it is registered under the name "nccl", the same key the CUDA NCCL engine uses; registering
    # both would collide (it presumably relies on only one module being importable per platform, but "hccl" seems intended).
    """HCCL checkpoint engine with collective communication.

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        group_name (str): The name of the HCCL process group. Defaults to "default".
        rebuild_group (bool): Whether to rebuild the HCCL process group in each update. Defaults to False.
        is_master (bool): Whether the current process is the master process. Defaults to False.
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: records configuration, sets device to current NPU, and if master, starts the zmq PUB server
        # and picks a free port for HCCL rendezvous.
        # Inputs: bucket_size - bytes per bucket; group_name - unused label; rebuild_group - recreate communicator every
        # update; is_master - whether this is the trainer rank 0; rollout_dtype - stored dtype hint.
        # Outputs: none. Side effects: binds a socket on master.
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.pyhccl = None
        self.device = torch.npu.current_device()

        # start zeromq server for broadcasting bucket tensor metadata
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()
            self.dist_port, _ = get_free_port(self.ip)

    def prepare(self) -> MasterMetadata:
        # [FUNCTION DOC]
        # Behavior: allocates two zeroed uint8 NPU buffers of bucket_size bytes. Outputs: MasterMetadata on master,
        # else None. Side effects: device memory allocation.
        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device="npu")

        return (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)
            if self.is_master
            else None
        )

    def finalize(self):
        # [FUNCTION DOC]
        # Behavior: if rebuild_group, destroys the communicator on participating ranks (rank >= 0) and resets
        # rank/world size; always drops both buffers and empties the NPU cache. Outputs: none.
        """Destroy the HCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            if self.rank >= 0:
                self.pyhccl.destroyComm(self.pyhccl.comm)
                self.pyhccl = None
            self.rank = None
            self.world_size = None

        self.send_buf = None
        self.recv_buf = None
        torch.npu.empty_cache()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        # [FUNCTION DOC]
        # Behavior: builds per-worker init kwargs. Trainer workers: rank 0 for the first, -1 (non-participant) for the
        # rest; rollout workers get ranks 1..N. All get world_size N+1 and the master metadata (metadata[0]).
        # Inputs: actor_wg_world_size, rollout_world_size - worker counts; metadata - list of prepare() results.
        # Outputs: tuple (actor kwargs dict of lists, rollout kwargs dict of lists).
        actor_wg_kwargs = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "master_metadata": [metadata[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [metadata[0]] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def _start_zmq_server(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: resolves this node's IP (stripping IPv6 brackets), picks a free port, creates a zmq PUB
        # socket (enabling IPv6 when needed) and binds it. Side effects: sets self.ip, self.zmq_port, self.socket.
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.zmq_port, _ = get_free_port(self.ip)

        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.zmq_port}"

        self.socket.bind(address)

    def _connect_zmq_client(self, metadata: MasterMetadata):
        # [FUNCTION DOC]
        # HELPER: Behavior: creates a zmq SUB socket, connects to the master's address (IPv6-aware) and subscribes to
        # the topic. Inputs: metadata - MasterMetadata. Exceptions: AssertionError if called on master.
        # Side effects: sets self.socket.
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        # [FUNCTION DOC]
        # Behavior: ranks below 0 just record rank/world size and return. Otherwise creates the HCCL communicator
        # when rebuild_group is set or none exists (else asserts rank/world size unchanged), connects the zmq client on
        # ranks > 0, and performs an all-reduce barrier on a 1-element tensor.
        # Inputs: rank, world_size, master_metadata. Exceptions: AssertionError on rank/world mismatch.
        # Side effects: network setup, logging.
        """Initialize the HCCL process group.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
        """
        # For actor workers other than rank 0, their rank should be -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or self.pyhccl is None:
            self.pyhccl = stateless_init_process_group(
                master_metadata.dist_ip, master_metadata.dist_port, rank, world_size, self.device
            )
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        if self.rank > 0:
            self._connect_zmq_client(master_metadata)

        # barrier
        signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
        self.pyhccl.all_reduce(signal)

        logger.info(f"init_process_group rank: {self.rank}, world_size: {self.world_size}")

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: trainer side. Non-participating ranks (-1) drain the generator and return. Rank 0 iterates
        # chunks produced by split_weight_chunks; when the next chunk would overflow the bucket it synchronizes the
        # device, waits on the previous operation, broadcasts the current buffer with its metadata (is_last False),
        # swaps buffers and resets. Each chunk gets its offset recorded and is copied into the send buffer. Finally
        # broadcasts the last bucket with is_last True and synchronizes.
        # Inputs: weights - generator of (name, tensor); global_steps - unused.
        # Exceptions: AssertionError if rank > 0, on bucket overflow, or duplicate chunk names in a bucket.
        # Side effects: collective communication, logging of elapsed time.
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        # For actor rank other than 0, consume weights without sending.
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # fill the tensor bucket
            if offset + tensor_meta.chunk_size > self.bucket_size:
                torch.npu.synchronize()

                # wait previous broadcast op finish
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()

                broadcast_op = BroadcastOperation(
                    rank=self.rank,
                    process_group=self.pyhccl,
                    bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False},
                    socket=self.socket,
                    topic=self.topic,
                )

                # swap send_buf and recv_buf
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size] = chunk
            offset += tensor_meta.chunk_size

        # broadcast last bucket
        torch.npu.synchronize()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()

        broadcast_op = BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True},
            socket=self.socket,
            topic=self.topic,
        )
        await broadcast_op.wait_for_complete()

        # the wait_for_complete() function just waits for the HCCL kernel to be enqueued,
        # not for the kernel to finish, hence we need to synchronize to make sure the
        # buffer does not get freed before the kernel finishes.
        torch.npu.synchronize()

        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        # [FUNCTION DOC]
        # Behavior: rollout side; reassembles full tensors from received chunks via merge_weight_chunks and yields
        # (name, tensor). Inputs: global_steps - unused. Outputs: async generator.
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        async for name, weight in merge_weight_chunks(self._receive_weight_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        # [FUNCTION DOC]
        # HELPER: Behavior: receives the first bucket into the recv buffer, syncs, swaps buffers; then while the
        # latest metadata is not last, starts receiving the next bucket into the other buffer, yields slices of the
        # current buffer for each chunk in its metadata, gets the next metadata, syncs and swaps. Finally yields
        # the last bucket's chunks and logs total time and bandwidth (counting full bucket_size per bucket).
        # Outputs: async generator of (TensorMeta, uint8 slice view of the buffer).
        # Exceptions: AssertionError if rank is 0 or below.
        # KNOWN ISSUE: docstring says it yields the weight name, but it yields the TensorMeta object.
        # Note: since the "next" broadcast runs synchronously at construction, the yielded slices of the previous buffer
        # are consumed after that broadcast completes; consumers must copy before the next swap.
        """Receive the weight chunks of the model.

        Yields:
            A tuple of the name of the weight tensor and the chunk itself.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0

        # receive first bucket
        start_time = time.time()
        broadcast_op = BroadcastOperation(
            rank=self.rank,
            process_group=self.pyhccl,
            bucket=recv_buf,
            metadata=None,
            socket=self.socket,
            topic=self.topic,
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # wait for the HCCL broadcast kernel to finish before we yield the tensors
        # otherwise if the buffer is clone using a non-blocking copy, it may
        # lead to data corruption
        torch.npu.synchronize()

        # swap send_buf and recv_buf
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            # 1. receive next bucket
            broadcast_op = BroadcastOperation(
                rank=self.rank,
                process_group=self.pyhccl,
                bucket=recv_buf,
                metadata=None,
                socket=self.socket,
                topic=self.topic,
            )

            # 2. yield tensor from send_buf
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 3. wait for next bucket broadcast finish
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(metadata["bucket_meta"])

            # 4. swap send_buf and recv_buf
            torch.npu.synchronize()  # sync non-blocking copy
            send_buf, recv_buf = recv_buf, send_buf

        # yield tensor from send_buf
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
