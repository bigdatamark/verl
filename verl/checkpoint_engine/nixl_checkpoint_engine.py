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
# Module nixl_checkpoint_engine: checkpoint engine using NVIDIA NIXL one-sided reads arranged as a chain
# (trainer rank 0 -> rollout 1 -> ... -> rollout N). Each hop advertises its filled buffer's transfer descriptors
# plus bucket metadata to the next hop over a ZeroMQ PUSH/PULL channel; the next hop reads the buffer remotely
# and NIXL notifies the producer with a per-operation key when the read completes. Double-buffered at every hop.
# cupy is imported with package-metadata discovery patched out. Side effects: module logger level from env.
import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import AsyncGenerator, Generator
from unittest.mock import patch

with patch("importlib.metadata.distributions", return_value=[]):
    import cupy as cp

import nixl._api as nixl_api
import nixl._bindings as nixl_bindings
import ray
import torch
import zmq
import zmq.asyncio

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class NixlAgentMetadata:
    # [CLASS DOC]
    # Dataclass exchanged between agents: agent name, serialized NIXL agent metadata bytes, and the agent's ZMQ PULL endpoint.
    agent_name: str
    agent_metadata: bytes
    zmq_ip: str
    zmq_port: int


class NixlAgent:
    # [CLASS DOC]
    # Wraps a nixl_agent (named by a random UUID) and adds a ZMQ PULL server for receiving messages, PUSH clients per
    # remote agent, a per-sender message queue and a per-remote notification queue. Unknown attributes are forwarded
    # to the underlying nixl agent.
    """This is a wrapper class for nixl_agent, the main purpose is to use ZeroMQ instead of
    `nixl_agent.send_notif` to send bucket tensor metadata.
    """

    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: creates the NIXL agent with a UUID name, initializes queues, starts the asyncio ZMQ PULL server.
        # Side effects: binds a socket on a free port.
        self.agent_name = str(uuid.uuid4())
        self.agent = nixl_api.nixl_agent(self.agent_name)
        self.notifications: dict[str, deque[bytes]] = defaultdict(deque)

        self.start_zmq_server()
        self.zmq_clients: dict[str, zmq.Socket] = {}
        self.messages: dict[str, deque[bytes]] = defaultdict(deque)

    def __getattr__(self, name):
        # [FUNCTION DOC]
        # Behavior: attribute fallback that fetches the named attribute from the wrapped agent; callables are returned
        # wrapped in a pass-through function, others as-is. Exceptions: AttributeError if the agent lacks it.
        attr = getattr(self.agent, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Behavior: forwards all arguments to the captured attribute and returns its result.
                return attr(*args, **kwargs)

            return wrapper
        else:
            return attr

    def get_agent_metadata(self) -> NixlAgentMetadata:
        # [FUNCTION DOC]
        # Behavior: builds NixlAgentMetadata from this agent's name, NIXL metadata, IP and listen port.
        return NixlAgentMetadata(
            agent_name=self.agent_name,
            agent_metadata=self.agent.get_agent_metadata(),
            zmq_ip=self.ip,
            zmq_port=self.listen_port,
        )

    def start_zmq_server(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: resolves node IP, picks a free port, binds an asyncio ZMQ PULL socket (IPv6-aware).
        # Side effects: sets ip, listen_port, socket.
        self.ip = ray.util.get_node_ip_address().strip("[]")
        self.listen_port, _ = get_free_port(self.ip)

        context = zmq.asyncio.Context()
        self.socket = context.socket(zmq.PULL)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.listen_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.listen_port}"

        self.socket.bind(address)

    def add_remote_agent(self, metadata: NixlAgentMetadata) -> str:
        # [FUNCTION DOC]
        # Behavior: registers the remote agent with NIXL, checks the returned name matches, opens a blocking ZMQ PUSH
        # socket to its endpoint and stores it. Outputs: remote agent name. Exceptions: AssertionError on name mismatch.
        agent_name = self.agent.add_remote_agent(metadata.agent_metadata).decode("utf-8")
        assert agent_name == metadata.agent_name, f"Agent name {agent_name} not equal to {metadata.agent_name}"

        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"

        socket.connect(address)
        self.zmq_clients[agent_name] = socket
        return agent_name

    def remove_remote_agent(self, agent_name: str):
        # [FUNCTION DOC]
        # Behavior: removes the remote agent from NIXL and closes and forgets its PUSH socket. Exceptions: KeyError if unknown.
        self.agent.remove_remote_agent(agent_name)
        socket = self.zmq_clients.pop(agent_name)
        socket.close()

    def send_message(self, agent_name, message: dict):
        # [FUNCTION DOC]
        # Behavior: sends a pickled tuple (own agent name, message) to the named agent without blocking.
        # Exceptions: KeyError for unknown agent; zmq.Again if it cannot be queued immediately.
        socket = self.zmq_clients[agent_name]
        socket.send_pyobj((self.agent_name, message), zmq.DONTWAIT)

    async def read_message(self, agent_name: str) -> dict:
        # [FUNCTION DOC]
        # Behavior: until a message from agent_name is queued, receives pickled (sender, message) tuples from the PULL
        # socket and queues them by sender; then pops the oldest for agent_name. Outputs: message dict.
        while len(self.messages[agent_name]) == 0:
            recv_agent_name, message = await self.socket.recv_pyobj()
            self.messages[recv_agent_name].append(message)
        return self.messages[agent_name].popleft()

    async def get_notification(self, remote_name: str) -> bytes:
        # [FUNCTION DOC]
        # Behavior: polls NIXL for new notifications (yielding to the event loop between polls), appending them per remote
        # agent, until one is queued for remote_name, then pops the oldest. Outputs: notification bytes.
        # KNOWN ISSUE: the inner loop variable reuses the name remote_name, overwriting the argument; the wait condition and
        # the returned queue then refer to whichever agent last sent a notification, not the requested one.
        while len(self.notifications[remote_name]) == 0:
            notifs = self.agent.get_new_notifs()
            for remote_name, notif in notifs.items():
                self.notifications[remote_name].extend(notif)
            await asyncio.sleep(0)
        return self.notifications[remote_name].popleft()


class ReadableOperation:
    # [CLASS DOC]
    # Producer side of one bucket hand-off: on construction sends the remote agent a message containing a fresh UUID
    # notify key, this side's transfer descriptors and the given metadata; waiting completes when the remote's read
    # notification arrives.
    # KNOWN ISSUE: the docstring lists a bucket_size argument that the constructor does not take.
    """Encapsulates a readable operation to remote agent.
       1. send metadata to remote agent
       2. wait until remote agent read complete.

    Args:
        agent (NixlAgent): The Nixl agent.
        remote_agent (str): The name of the remote agent.
        local_descs (nixl_bindings.nixlXferDList): The local transfer descriptors.
        metadata (dict): Metadata for the read operation.
        bucket_size (int): The size of the bucket in bytes.
    """

    def __init__(
        self,
        agent: NixlAgent,
        remote_agent: str,
        local_descs: nixl_bindings.nixlXferDList,
        metadata: dict,
    ):
        # [FUNCTION DOC]
        # Behavior: stores arguments, generates the notify key and sends the message. Inputs: agent, remote_agent name,
        # local_descs - descriptors of the buffer to be read, metadata - extra fields merged into the message.
        # Side effects: ZMQ send.
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.notify_key = uuid.uuid4().bytes
        message = {"notify_key": self.notify_key, "remote_descs": self.local_descs, **metadata}
        self.agent.send_message(self.remote_agent, message)

    async def wait_for_complete(self):
        # [FUNCTION DOC]
        # Behavior: awaits the next notification from the remote agent and checks it equals this operation's key.
        # Exceptions: AssertionError on key mismatch. Side effects: debug log.
        """Block until remote agent read complete."""
        notification = await self.agent.get_notification(self.remote_agent)
        assert self.notify_key == notification, f"Notify key {self.notify_key} not equal to {notification}"
        logger.debug(f"ReadableOperation to {self.remote_agent} complete")


class ReadOperation:
    # [CLASS DOC]
    # Consumer side of one bucket hand-off: reads the producer's message, issues a NIXL READ into local descriptors
    # with the notify key, and polls until done.
    """Encapsulates a read operation from remote agent.
    1. read medata from remote agent
    2. start read transfer operation
    3. wait until read complete

    Args:
        agent (NixlAgent): The Nixl agent.
        remote_agent (str): The name of the remote agent.
        local_descs (nixl_bindings.nixlXferDList): The local transfer descriptors.
        bucket_size (int): The size of the bucket in bytes.
    """

    def __init__(self, agent: NixlAgent, remote_agent: str, local_descs: nixl_bindings.nixlXferDList, bucket_size: int):
        # [FUNCTION DOC]
        # Behavior: stores agent, remote name, local descriptors and bucket_size (for bandwidth logging); other state None.
        self.agent = agent
        self.remote_agent = remote_agent
        self.local_descs = local_descs
        self.remote_descs = None
        self.xfer_handle = None
        self.notify_key = None
        self.bucket_size = bucket_size
        self.start_time = None

    async def read_metadata(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: awaits the next message from the remote agent, extracts and stores remote_descs and notify_key, and
        # returns the remaining metadata (bucket_meta, is_last).
        """Block until the remote agent sends the metadata.

        Returns:
            dict: Metadata from the remote agent.
        """
        metadata = await self.agent.read_message(self.remote_agent)
        self.remote_descs = metadata.pop("remote_descs")
        self.notify_key = metadata.pop("notify_key")
        return metadata

    def begin_read(self):
        # [FUNCTION DOC]
        # Behavior: creates a READ transfer from remote descriptors into local descriptors carrying the notify key, starts
        # it and records the start time. Exceptions: AssertionError if metadata not read yet or the transfer state is "ERR".
        """Start the read operation."""
        assert self.remote_descs is not None and self.notify_key is not None
        self.xfer_handle = self.agent.initialize_xfer(
            "READ", self.local_descs, self.remote_descs, self.remote_agent, self.notify_key
        )
        state = self.agent.transfer(self.xfer_handle)
        assert state != "ERR", f"Read from {self.remote_agent} got to {state} state."
        self.start_time = time.time()

    async def wait_for_complete(self):
        # [FUNCTION DOC]
        # Behavior: polls the transfer state, yielding to the loop, until "DONE"; on "ERR" logs and terminates the process
        # with exit(-1). Then releases the handle and logs bandwidth based on the full bucket size.
        # Side effects: may terminate the process.
        """Block until the read operation complete."""
        while True:
            state = self.agent.check_xfer_state(self.xfer_handle)
            if state == "ERR":
                logger.error(f"Read from {self.remote_agent} got to {state} state.")
                exit(-1)
            elif state == "DONE":
                break
            else:
                await asyncio.sleep(0)
        self.agent.release_xfer_handle(self.xfer_handle)
        end_time = time.time()
        bandwidth = self.bucket_size / (end_time - self.start_time) / (1024 * 1024 * 1024)
        logger.debug(f"ReadOperation read data from {self.remote_agent} complete, bandwidth: {bandwidth:.2f} GB/s")


@CheckpointEngineRegistry.register("nixl")
class NIXLCheckpointEngine(CheckpointEngine):
    # [CLASS DOC]
    # CheckpointEngine over NIXL in a chain. Holds a NixlAgent, two registered uint8 buffers (send/recv) with their
    # registration and transfer descriptors, and the names of the previous and next agents.
    """NIXL checkpoint engine with p2p communication, support various backends: ucx, uccl, mooncacke, etc.

    For UCX backend, some environment variables need to be set: UCX_TLS, UCX_IB_GID_INDEX, UCX_IB_DEVICES, etc.
    Please refer to: https://openucx.readthedocs.io/en/master/faq.html

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time. Note that we use
            two buffer to send and recv weights at same time, so the device memory overhead is 2 * bucket_size.
        device (str): The device to use for the checkpoint engine, "cpu" or "cuda".
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        is_master: bool = False,
    ):
        # [FUNCTION DOC]
        # Behavior: stores bucket_size, device, rollout_dtype (unused here), is_master, and creates a NixlAgent.
        # Side effects: binds a ZMQ socket.
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.agent = NixlAgent()
        self.is_master = is_master

    def prepare(self) -> NixlAgentMetadata:
        # [FUNCTION DOC]
        # Behavior: allocates two bucket_size uint8 buffers (cupy-backed CUDA tensors for "cuda", otherwise pinned
        # tensors on the given device), registers them with NIXL and builds transfer descriptors.
        # Outputs: this agent's NixlAgentMetadata.
        """Prepare send and recv bucket.

        Returns:
            NixlAgentMetadata: The metadata of the current nixl agent.
        """
        # For master process, use cupy instead of torch to avoid memory register error
        # when `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
        if self.device == "cuda":
            send_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            recv_buf = cp.zeros(self.bucket_size, dtype=cp.uint8)
            self.send_buf = torch.as_tensor(send_buf, dtype=torch.uint8)
            self.recv_buf = torch.as_tensor(recv_buf, dtype=torch.uint8)
        else:
            self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device, pin_memory=True)
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device=self.device, pin_memory=True)
        self.send_reg_descs = self.agent.register_memory(self.send_buf)
        self.recv_reg_descs = self.agent.register_memory(self.recv_buf)
        self.send_descs = self.agent.get_xfer_descs(self.send_buf)
        self.recv_descs = self.agent.get_xfer_descs(self.recv_buf)

        return self.agent.get_agent_metadata()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]):
        # [FUNCTION DOC]
        # Behavior: chain topology assuming metadata lists trainer workers followed by rollout workers. Trainer rank 0's
        # next agent is the first rollout worker; other trainer workers get rank -1 with no neighbors. Rollout i's previous
        # is trainer 0 (for the first) or rollout i-1, next is rollout i+1 (None for the last). World size N+1.
        # Outputs: (actor kwargs, rollout kwargs) with method, rank, world_size, prev/next metadata lists.
        actor_wg_kwargs = {
            "method": ["init_process_group"] * actor_wg_world_size,
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "prev_agent_metadata": [None] * actor_wg_world_size,
            "next_agent_metadata": [metadata[-rollout_world_size]] + [None] * (actor_wg_world_size - 1),
        }

        rollout_kwargs = {
            "method": ["init_process_group"] * rollout_world_size,
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "prev_agent_metadata": [metadata[0]] + metadata[-rollout_world_size:-1],
            "next_agent_metadata": metadata[-rollout_world_size + 1 :] + [None],
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(
        self, rank: int, world_size: int, prev_agent_metadata: NixlAgentMetadata, next_agent_metadata: NixlAgentMetadata
    ):
        # [FUNCTION DOC]
        # Behavior: validates neighbor presence per position (none for rank<0, only next for rank 0, both for middle,
        # only prev for last), stores rank/world size, and adds the neighbor agents as remotes. Logs the setup.
        # Exceptions: AssertionError on inconsistent neighbor metadata. Side effects: NIXL remote registration, ZMQ connects.
        """Setup the communication with the previous and next agent.

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            prev_agent_metadata (NixlAgentMetadata): The metadata of the previous nixl agent.
            next_agent_metadata (NixlAgentMetadata): The metadata of the next nixl agent.
        """
        if rank < 0:
            assert not prev_agent_metadata and not next_agent_metadata, (
                f"rank {rank} should not have prev_agent_metadata or next_agent_metadata"
            )
        elif rank == 0:
            assert not prev_agent_metadata and next_agent_metadata, f"rank {rank} should have next_agent_metadata"
        elif 0 < rank < world_size - 1:
            assert prev_agent_metadata and next_agent_metadata, (
                f"rank {rank} should have prev_agent_metadata and next_agent_metadata"
            )
        elif rank == world_size - 1:
            assert prev_agent_metadata and not next_agent_metadata, (
                f"rank {rank} should have prev_agent_metadata and not next_agent_metadata"
            )

        self.rank = rank
        self.world_size = world_size
        self.prev_agent = None
        self.next_agent = None

        if prev_agent_metadata is not None:
            self.prev_agent = self.agent.add_remote_agent(prev_agent_metadata)

        if next_agent_metadata is not None:
            self.next_agent = self.agent.add_remote_agent(next_agent_metadata)

        logger.info(
            f"init_process_group rank: {self.rank}, world_size: {self.world_size}, "
            f"prev_agent: {self.prev_agent}, next_agent: {self.next_agent}"
        )

    def finalize(self):
        # [FUNCTION DOC]
        # Behavior: removes neighbor agents, deregisters both buffers and clears buffers, descriptors, rank and neighbors.
        """Cleanup communication with the previous and next agent, and deregister the memory."""
        if self.prev_agent:
            self.agent.remove_remote_agent(self.prev_agent)
        if self.next_agent:
            self.agent.remove_remote_agent(self.next_agent)

        self.agent.deregister_memory(self.send_reg_descs)
        self.agent.deregister_memory(self.recv_reg_descs)
        self.send_buf = None
        self.recv_buf = None
        self.send_reg_descs = None
        self.recv_reg_descs = None
        self.send_descs = None
        self.recv_descs = None

        self.rank = None
        self.world_size = None
        self.prev_agent = None
        self.next_agent = None

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: rank -1 drains the generator. Rank 0 packs chunks into the send buffer (non-blocking copy); on overflow
        # synchronizes, waits for the previous bucket to be read by the next agent, advertises the filled buffer with
        # metadata (is_last False) and swaps buffers/descriptors. Finally synchronizes, waits for the previous, advertises
        # the last bucket with is_last True and waits for it to be read.
        # Inputs: weights generator; global_steps unused.
        # Exceptions: AssertionError for rank > 0, missing next agent, overflow, or duplicate chunk name.
        """Send the weights of the model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        assert self.rank <= 0, "Trainer workers other than rank 0 should not send weights."

        # For actor workers other than rank 0, just consume weights and do nothing.
        if self.rank < 0:
            for name, weight in weights:
                pass
            return

        assert self.next_agent is not None, "Next agent is not set."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs
        readable_op = None

        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            # fill the tensor bucket
            if offset + tensor_meta.chunk_size > self.bucket_size:
                torch.cuda.synchronize()

                # wait previous bucket to be received
                if readable_op is not None:
                    await readable_op.wait_for_complete()

                # send bucket meta to next agent
                readable_op = ReadableOperation(
                    self.agent,
                    self.next_agent,
                    send_descs,
                    {"bucket_meta": bucket_meta, "is_last": False},
                )

                # swap send and recv buf
                send_buf, recv_buf = recv_buf, send_buf
                send_descs, recv_descs = recv_descs, send_descs
                bucket_meta = {}
                offset = 0

            assert offset + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta

            tensor_meta.offset = offset
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[offset : offset + tensor_meta.chunk_size].copy_(chunk, non_blocking=True)
            offset += tensor_meta.chunk_size

        # send last bucket meta to next agent
        torch.cuda.synchronize()
        if readable_op is not None:
            await readable_op.wait_for_complete()

        readable_op = ReadableOperation(
            self.agent, self.next_agent, send_descs, {"bucket_meta": bucket_meta, "is_last": True}
        )
        await readable_op.wait_for_complete()
        logger.info(f"Rank {self.rank} send weights done, time cost: {time.time() - start_time:.2f}s")

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        # [FUNCTION DOC]
        # Behavior: wraps _receive_weight_chunks with merge_weight_chunks and yields reassembled (name, tensor).
        """Receive the weights of the model.

        Yields:
            A tuple of the name of the weight tensor and the tensor itself.
        """
        async for name, weight in merge_weight_chunks(self._receive_weight_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        # [FUNCTION DOC]
        # HELPER: Behavior: reads the first bucket from the previous agent into the recv buffer and swaps. Then while not
        # last: advertise the current buffer to the next agent (if any), read the next metadata and start reading into the
        # other buffer, yield chunk views of the current buffer, wait for the next agent's read and our read, synchronize,
        # swap. Finally advertise and yield the last bucket and wait for the next agent's read. Logs bandwidth counting
        # full bucket_size per bucket. Outputs: async generator of (TensorMeta, buffer slice).
        # Exceptions: AssertionError without a previous agent.
        """Receive the weight chunks of the model.

        Yields:
            A tuple of the chunk metadata and the chunk buffer view in send_buf.
        """
        assert self.prev_agent is not None, "Previous agent is not set."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        send_descs, recv_descs = self.send_descs, self.recv_descs
        total_bytes, total_params = 0, 0

        # receive first bucket from previous agent
        start_time = time.time()
        read_op = ReadOperation(self.agent, self.prev_agent, recv_descs, self.bucket_size)
        metadata = await read_op.read_metadata()
        read_op.begin_read()
        await read_op.wait_for_complete()
        total_bytes += self.bucket_size
        total_params += len(metadata["bucket_meta"])

        # swap send and recv buf
        send_buf, recv_buf = recv_buf, send_buf
        send_descs, recv_descs = recv_descs, send_descs
        while not metadata["is_last"]:
            # 1. send bucket to next agent
            readable_op = None
            if self.next_agent is not None:
                readable_op = ReadableOperation(
                    self.agent,
                    self.next_agent,
                    send_descs,
                    metadata,
                )

            # 2. receive bucket from previous agent
            read_op = ReadOperation(self.agent, self.prev_agent, recv_descs, self.bucket_size)
            next_metadata = await read_op.read_metadata()
            read_op.begin_read()

            # 3. yield tensor from send_buf
            for name, tensor_meta in metadata["bucket_meta"].items():
                tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
                yield tensor_meta, tensor

            # 4. wait for next agent read complete and read from previous agent complete
            if readable_op is not None:
                await readable_op.wait_for_complete()
            await read_op.wait_for_complete()
            total_bytes += self.bucket_size
            total_params += len(next_metadata["bucket_meta"])

            # 5. swap send and recv buf
            torch.cuda.synchronize()  # sync non-blocking copy
            metadata = next_metadata
            send_buf, recv_buf = recv_buf, send_buf
            send_descs, recv_descs = recv_descs, send_descs

        # send last bucket to next agent
        readable_op = None
        if self.next_agent is not None:
            readable_op = ReadableOperation(
                self.agent,
                self.next_agent,
                send_descs,
                metadata,
            )

        # yield tensor from send_buf
        for name, tensor_meta in metadata["bucket_meta"].items():
            tensor = send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            yield tensor_meta, tensor

        # wait for next agent read complete
        if readable_op is not None:
            await readable_op.wait_for_complete()
        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, total_params: {total_params}, "
            f"time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
