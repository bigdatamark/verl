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
# Module mooncake_checkpoint_engine: checkpoint engine using Mooncake TransferEngine for one-sided RDMA
# (or Ascend direct) transfers arranged as a chain pipeline: trainer rank 0 -> rollout 1 -> rollout 2 -> ...
# Each hop RDMA-reads the previous rank's double buffer, forwards metadata over a StatelessProcessGroup
# (vLLM, falling back to SGLang), and signals completion by writing a 4-byte magic value into the previous
# rank's magic slot. Side effects: module logger level from VERL_LOGGING_LEVEL (default INFO).
import asyncio
import gc
import logging
import os
import time
from typing import Any, AsyncGenerator, Generator

import ray
import torch
from mooncake.engine import TransferEngine

try:
    from vllm.distributed.utils import StatelessProcessGroup
except ImportError:
    from sglang.srt.distributed.utils import StatelessProcessGroup

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_torch_device
from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@CheckpointEngineRegistry.register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    # [CLASS DOC]
    # CheckpointEngine with p2p chain transfer. Owns a TransferEngine, a 2 x bucket_size registered data buffer,
    # a 4 KiB buffer holding the magic pattern to write, and an 8-byte magic receive area (two 4-byte slots).
    """Mooncake checkpoint engine with p2p communication using TransferEngine

    Args:
        bucket_size (int): Bucket size in bytes to transfer multiple weights at one time.
        device (str): The device to use for the checkpoint engine, "cpu" or "cuda".
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers.
        device_name (str): Mooncake device name filter.
    """

    def __init__(
        self,
        bucket_size: int,
        device: str = "cuda",
        rollout_dtype: torch.dtype = torch.bfloat16,
        device_name: str = "",
        is_master: bool = False,
        rebuild_group: bool = False,
    ):
        # [FUNCTION DOC]
        # Behavior: stores config; reads RANK env var and sets the local device (rank modulo device count); initializes
        # the TransferEngine with P2P handshake on the node IP using "ascend_direct" transport for npu else "rdma";
        # builds session id "host:rpc_port"; allocates and registers the three buffers.
        # Inputs: bucket_size; device - "cuda"/"npu"/"cpu"; rollout_dtype - cast dtype for sent weights; device_name -
        # Mooncake NIC filter; is_master, rebuild_group - stored only.
        # Exceptions: KeyError if RANK unset; AssertionError if initialization or registration fails.
        # Side effects: device selection, memory registration, logging.
        self.bucket_size = bucket_size
        self.device = device
        self.rollout_dtype = rollout_dtype
        self.is_master = is_master
        self.rebuild_group = rebuild_group

        rank = int(os.environ["RANK"])
        device_count = get_torch_device().device_count()
        local_rank = rank % device_count
        get_torch_device().set_device(local_rank)

        self.engine = TransferEngine()
        hostname = ray.util.get_node_ip_address().strip("[]")
        ret = self.engine.initialize(
            hostname,
            "P2PHANDSHAKE",
            "ascend_direct" if self.device == "npu" else "rdma",
            device_name,
        )
        assert ret == 0, f"TransferEngine initialize failed ret={ret}"

        rpc_port = self.engine.get_rpc_port()
        self.session_id = f"{hostname}:{rpc_port}"
        self.hostname = hostname

        self.buf = torch.empty(2 * self.bucket_size, dtype=torch.uint8, device=self.device)
        self.magic_buf = torch.empty(4 * 1024, dtype=torch.uint8, device=self.device)
        # Separate buffer for receiving magic completion signals (one 4-byte slot per double-buffer)
        # This prevents the next rank's magic write from corrupting data buffer contents.
        self.magic_recv = torch.zeros(8, dtype=torch.uint8, device=self.device)
        ret = self.engine.batch_register_memory(
            [self.buf.data_ptr(), self.magic_buf.data_ptr(), self.magic_recv.data_ptr()],
            [2 * self.bucket_size, 4 * 1024, 8],
        )
        assert ret == 0, f"batch_register_memory failed ret={ret}"
        logger.info(f"__init__ session_id={self.session_id}")

    def prepare(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: logs buffer pointers and picks a free port on this host. Outputs: dict with "addr" and "port".
        """Prepare send and recv buckets"""
        logger.info(
            f"prepare ptr={self.buf.data_ptr():#x} len={2 * self.bucket_size} "
            f"magic_buf_ptr={self.magic_buf.data_ptr():#x}"
        )
        port, _ = get_free_port(self.hostname)
        return {"addr": self.hostname, "port": port}

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadatas: list[dict]):
        # [FUNCTION DOC]
        # Behavior: same chain topology as other engines: trainer rank 0 participates (others -1), rollout ranks 1..N,
        # world size N+1, all receive metadatas[0]. Outputs: (actor kwargs, rollout kwargs) dicts of per-worker lists.
        actor_wg_kwargs = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "metadata": [metadatas[0]] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "metadata": [metadatas[0]] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, metadata: dict[str, Any]):
        # [FUNCTION DOC]
        # Behavior: records rank/world size; ranks < 0 return early. Otherwise creates a StatelessProcessGroup at the
        # given addr/port, all-gathers each rank's session id and buffer pointer, and stores the previous rank's entry
        # as buffer_info (None on rank 0).
        # Inputs: rank, world_size, metadata dict with "addr"/"port". Side effects: network rendezvous, logging.
        self.rank = rank
        self.world_size = world_size
        if rank < 0:
            logger.info(f"init_process_group rank={rank}")
            return

        self.store = StatelessProcessGroup.create(
            host=metadata["addr"],
            port=metadata["port"],
            rank=rank,
            world_size=world_size,
        )

        info = {
            "session_id": self.session_id,
            "ptr": self.buf.data_ptr(),
        }

        info_list = self.store.all_gather_obj(info)
        self.buffer_info = None if rank == 0 else info_list[rank - 1]

        logger.info(f"init_process_group rank={rank} world_size={world_size} buffer_info={self.buffer_info}")

    def finalize(self):
        # [FUNCTION DOC]
        # Behavior: drops the store, empties device cache, runs garbage collection, logs.
        # KNOWN ISSUE: docstring says it deregisters memory but no memory is deregistered.
        """Cleanup communication and deregister memory"""
        self.store = None
        get_torch_device().empty_cache()
        gc.collect()
        logger.info(f"finalize rank={self.rank}")

    async def wait_for_complete(self, buf: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Behavior: busy-polls (yielding to the event loop each iteration) until the first 4 bytes of buf equal
        # the magic pattern AB DC EF 88, then zeroes them. Inputs: buf - uint8 tensor slot. Side effects: mutates buf;
        # each check is a device sync. Never times out.
        magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)
        while True:
            if torch.equal(buf[:4], magic):
                buf[:4] = 0  # reset for next use
                break
            await asyncio.sleep(0)

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        # [FUNCTION DOC]
        # Behavior: non-participating ranks drain the generator and return. Rank 0 casts each weight to rollout_dtype and
        # packs raw bytes into the current half of the double buffer, recording name/shape/dtype/offset. When a weight
        # would overflow, it synchronizes and sends rank 1 the bucket metadata, data pointer, magic slot pointer, length,
        # is_last False; switches halves and, from the second switch on, waits for the magic completion on the slot it is
        # about to reuse. After all weights, sends the final bucket with is_last True and waits for both slots' completion.
        # Inputs: weights generator; global_steps unused.
        # Exceptions: AssertionError if a single weight exceeds bucket_size.
        # Side effects: network sends, logs bandwidth.
        # KNOWN ISSUE: total_bytes excludes the final bucket, so the logged bandwidth undercounts.
        """Send weights using Mooncake TransferEngine"""
        if self.rank < 0:
            for name, weight in weights:
                pass
            logger.info(f"send_weights rank={self.rank}")
            return

        total_bytes = 0
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        should_wait = False
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        magic_slots = [self.magic_recv[:4], self.magic_recv[4:]]
        idx = 0
        current = bufs[idx]

        for name, weight in weights:
            weight = weight.to(self.rollout_dtype)

            if offset + weight.nbytes > self.bucket_size:
                total_bytes += offset
                get_torch_device().synchronize()
                info = {
                    "bucket_meta": bucket_meta,
                    "ptr": current.data_ptr(),
                    "magic_ptr": magic_slots[idx].data_ptr(),
                    "len": offset,
                    "is_last": False,
                }
                # send to rank 1
                self.store.send_obj(info, 1)

                idx ^= 1
                current = bufs[idx]
                bucket_meta = {}
                offset = 0

                if should_wait:
                    await self.wait_for_complete(magic_slots[idx])
                should_wait = True

            assert offset + weight.nbytes <= self.bucket_size, (
                f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
            )

            bucket_meta[name] = {
                "name": name,
                "shape": weight.shape,
                "dtype": weight.dtype,
                "offset": offset,
            }
            current[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
            offset += weight.nbytes

        get_torch_device().synchronize()
        info = {
            "bucket_meta": bucket_meta,
            "ptr": current.data_ptr(),
            "magic_ptr": magic_slots[idx].data_ptr(),
            "len": offset,
            "is_last": True,
        }
        self.store.send_obj(info, 1)
        await self.wait_for_complete(magic_slots[idx])
        if should_wait:
            await self.wait_for_complete(magic_slots[idx ^ 1])

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} send weights done, "
            f"total bytes: {total_bytes} time cost: {time_cost:.2f}s bandwidth: {bandwidth:.2f} GB/s"
        )

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        # [FUNCTION DOC]
        # Behavior: loop: receive metadata from rank-1; when reusing a half (from the third bucket) and not the last rank,
        # wait until the next rank signaled completion on that slot; RDMA-read len bytes from the previous rank's buffer
        # into the current half; rewrite pointers to this rank's half and slot and forward to rank+1 unless last; yield
        # views (name, tensor) for each entry; synchronize; write the magic to the previous rank's magic slot (or, if absent,
        # to the start of its data buffer); advance halves; stop after the is_last bucket. Non-last ranks then wait for the
        # outstanding slots' completion. Logs time and bandwidth.
        # Outputs: async generator of (name, tensor view into the internal buffer; valid only until the half is reused).
        # Exceptions: AssertionError on failed transfers. Side effects: RDMA, network, logging.
        """Receive weights from the previous rank via RDMA.

        Protocol:
        1. Recv metadata from prev rank via TCPStore
        2. RDMA-read data from prev rank's buffer
        3. Forward metadata to next rank (if not last)
        4. Yield weights to consumer
        5. Write magic completion signal to prev rank's magic_ptr
           (a dedicated magic_recv slot, NOT the data buffer, to avoid
           transfer_sync_write local GPU side-effect corruption)
        """
        _magic = torch.tensor([0xAB, 0xDC, 0xEF, 0x88], dtype=torch.uint8, device=self.device)

        start_time = time.time()
        total_bytes = 0
        bufs = [self.buf[: self.bucket_size], self.buf[self.bucket_size :]]
        magic_slots = [self.magic_recv[:4], self.magic_recv[4:]]
        idx = 0
        current = bufs[idx]
        self.magic_buf[:4] = _magic.clone()

        while True:
            info = self.store.recv_obj(self.rank - 1)
            if idx >= 2 and self.rank < self.world_size - 1:
                await self.wait_for_complete(magic_slots[idx % 2])

            prev_ptr = info["ptr"]
            prev_magic_ptr = info.get("magic_ptr")

            ret = self.engine.transfer_sync_read(
                self.buffer_info["session_id"],
                current.data_ptr(),
                prev_ptr,
                info["len"],
            )
            assert ret == 0

            total_bytes += info["len"]

            info["ptr"] = current.data_ptr()
            info["magic_ptr"] = magic_slots[idx % 2].data_ptr()
            if self.rank < self.world_size - 1:
                self.store.send_obj(info, self.rank + 1)

            for name, meta in info["bucket_meta"].items():
                dtype, shape = meta["dtype"], meta["shape"]
                size = dtype.itemsize * shape.numel()
                tensor = current[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                yield name, tensor

            get_torch_device().synchronize()

            if prev_magic_ptr is not None:
                ret = self.engine.transfer_sync_write(
                    self.buffer_info["session_id"],
                    self.magic_buf.data_ptr(),
                    prev_magic_ptr,
                    4,
                )
                assert ret == 0
            else:
                ret = self.engine.transfer_sync_write(
                    self.buffer_info["session_id"],
                    self.magic_buf.data_ptr(),
                    prev_ptr,
                    4,
                )
                assert ret == 0

            idx += 1
            current = bufs[idx % 2]
            get_torch_device().synchronize()

            if info["is_last"]:
                break

        if self.rank < self.world_size - 1:
            for slot in range(min(idx, 2)):
                await self.wait_for_complete(magic_slots[slot])

        time_cost = time.time() - start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        logger.info(
            f"Rank {self.rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s"
        )
