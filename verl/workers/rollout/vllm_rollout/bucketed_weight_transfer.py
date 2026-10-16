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
# Bucketed weight transfer between a trainer-side sender and a vLLM-worker-side receiver
# over a ZMQ REQ/REP pair, using one fixed-size byte buffer shared via CUDA IPC (or POSIX
# shared memory, e.g. for NPU). Protocol: sender binds REQ and sends the buffer handle (or
# {name, size} of the shm segment); receiver connects REP, rebuilds the buffer and acks.
# Then per bucket the sender fills the buffer, synchronizes the device and sends
# {"bucket_meta": mapping name -> TensorMetadata, "is_last": bool}; the receiver views each
# tensor from the buffer (or rebuilds a per-tensor IPC handle for oversize weights), calls
# the callback and acks. The final ack is deferred until the receiver released its buffer.
# Module state: logger (level from VERL_LOGGING_LEVEL, default INFO).
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import logging
import os
from multiprocessing import shared_memory
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device, is_support_ipc

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    # [CLASS DOC]
    # Typed mapping describing one tensor in a bucket: name, shape, dtype, byte offset in the
    # shared buffer, and handle (an IPC rebuild handle for directly sent tensors, else None).
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int
    handle: tuple


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Rebuild a tensor from a torch IPC handle (rebuild function, args), replacing
    # argument index 6 (the device index) with device_id when given so processes with
    # different visible-device sets map to the right GPU. Returns the tensor.
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    # [FUNCTION DOC]
    # Create a POSIX shared-memory segment of size bytes with the given name, or attach to an
    # existing one of that name.
    # Exceptions: AssertionError if an existing segment is smaller than size.
    # Outputs: the SharedMemory object (caller must close/unlink).
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    # [FUNCTION DOC]
    # Attach to the shared-memory segment name and expose its first size bytes as a 1-D
    # tensor of dtype (zero-copy). Outputs: (tensor, SharedMemory handle kept alive by caller).
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    # [CLASS DOC]
    # Sender side of the protocol. Attributes: zmq_handle, bucket_size_mb, bucket_size
    # (bytes), use_shm, shared ZMQ context, socket/buffer/shm created per send.
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        # [FUNCTION DOC]
        # Store the endpoint, bucket size (MiB converted to bytes) and transport mode; obtain the
        # process-wide ZMQ context. No sockets are opened yet.
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights):
        # [FUNCTION DOC]
        # Send all weights as one sync round.
        # Behavior: opens the socket and buffer. For each (name, tensor): rounds the offset up to
        # the tensor's element size; if it would overflow a non-empty bucket, synchronizes and
        # sends the bucket (is_last False), waits for the ack and resets. If the tensor alone
        # exceeds the bucket, sends it directly by IPC handle (not allowed with shm). Otherwise
        # records its metadata and asynchronously copies it into the buffer at the offset. Finally
        # synchronizes and sends the remaining bucket with is_last True and waits for the ack.
        # Always cleans up.
        # Inputs: weights - sync or async iterable of (name, tensor) on the device.
        # Exceptions: AssertionError for an oversize tensor in shm mode; ZMQ errors.
        # Side effects: binds an IPC socket, allocates device/shared memory.
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # Align each tensor before the receiver views the byte buffer
                # using that tensor's dtype.
                alignment = weight.element_size()
                offset = (offset + alignment - 1) // alignment * alignment

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size and len(bucket_meta) > 0:
                    get_torch_device().synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                if offset + weight.nbytes > self.bucket_size:
                    assert not self.use_shm, (
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                        f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                    )
                    self._direct_send_large_weight(name, weight)
                    continue

                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                    "handle": None,
                }
                self.buffer[offset : offset + weight.nbytes].view(dtype=weight.dtype).view(weight.shape).copy_(
                    weight, non_blocking=True
                )
                offset += weight.nbytes

            # send the last bucket
            name = weight = None
            get_torch_device().synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        # [FUNCTION DOC]
        # HELPER: Remove any stale IPC socket file for an "ipc://" endpoint, then create a REQ
        # socket and bind it.
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        # [FUNCTION DOC]
        # HELPER: Allocate the transfer buffer: a uint8 device tensor of bucket_size bytes whose
        # IPC handle is sent, or a uniquely named shared-memory segment whose {name, size} is
        # sent; wait for the receiver's ack and store buffer/shm.
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        # [FUNCTION DOC]
        # HELPER: Close the socket, remove the IPC socket file, drop the buffer, close and unlink
        # the shared memory, collect IPC memory if supported and empty the device cache.
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        if is_support_ipc():
            get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

    def _direct_send_large_weight(self, name: str, weight: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Send one oversize weight as its own bucket containing an IPC handle to the
        # tensor itself (offset 0, is_last False) and wait for the ack.
        """Send a weight larger than the bucket size via cuda ipc or share memory."""
        logger.debug(f"Direct sending large weight {name}({weight.shape}, {weight.dtype})")
        # TODO: support fallback to shared memory
        handle = reduce_tensor(weight)
        bucket_meta: dict[str, TensorMetadata] = {}
        bucket_meta[name] = {
            "name": name,
            "shape": weight.shape,
            "dtype": weight.dtype,
            "offset": 0,
            "handle": handle,
        }
        self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
        self.socket.recv()


class BucketedWeightReceiver:
    # [CLASS DOC]
    # Receiver side of the protocol. Attributes: zmq_handle, target device, use_shm, ZMQ
    # context, socket, buffer, shm, and _ack_pending (final ack deferred to cleanup).
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        # [FUNCTION DOC]
        # Store endpoint, target device and transport mode; obtain the ZMQ context.
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None
        self._ack_pending = False

    def receive_weights(self, on_bucket_received: callable):
        # [FUNCTION DOC]
        # Receive one sync round, invoking on_bucket_received(weights, is_last) per bucket.
        # Behavior: connects and rebuilds the buffer. Loops: receives metadata; for each entry
        # rebuilds the tensor from its own IPC handle if present, else views it out of the shared
        # buffer at its offset (moved to the device in shm mode). Calls the callback, synchronizes
        # the device, then acks unless it was the last bucket, in which case the ack is deferred to
        # cleanup. Tensors passed to the callback alias the shared buffer and are only valid during
        # the callback. Always cleans up.
        # Side effects: device synchronization, ZMQ traffic.
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)],
            is_last: bool) called per bucket. ``is_last`` marks the final bucket
            of a weight-sync round so consumers that need the complete tensor set
            (e.g. vLLM ``add_lora``, which takes one adapter dict per call) can
            defer their finalization until the whole adapter has arrived.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset, handle = meta["shape"], meta["dtype"], meta["offset"], meta["handle"]
                    if handle is not None:
                        tensor = rebuild_ipc(handle, self.device.index)
                        weights.append((name, tensor))
                        continue
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                is_last = metadata["is_last"]
                on_bucket_received(weights, is_last)
                get_torch_device().synchronize()
                del weights, tensor
                if not is_last:
                    self.socket.send(b"")
                else:
                    self._ack_pending = True
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        # [FUNCTION DOC]
        # HELPER: Create a REP socket and connect to the sender's endpoint.
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        # [FUNCTION DOC]
        # HELPER: Receive the buffer descriptor, rebuild the CUDA IPC uint8 buffer on this device
        # (asserting uint8) or attach to the shared-memory segment, ack, and store buffer/shm.
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        # [FUNCTION DOC]
        # HELPER: Synchronize the device, drop the buffer, close (not unlink) shared memory,
        # collect IPC memory and empty the cache, then send the deferred final ack if pending and
        # close the socket.
        """clean up"""
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        if is_support_ipc():
            get_torch_device().ipc_collect()
        get_torch_device().empty_cache()
        # Ack last, after the buffer is released: the sender reclaims its bucket
        # buffer the moment this ack returns, and CUDA IPC only frees the sender's
        # block once our ref counter reaches zero. Acking earlier strands it in
        # CudaIPCSentDataLimbo until some later round drains the limbo.
        # The guard keeps the REP socket's strict recv/send alternation valid.
        if self.socket is not None:
            if self._ack_pending:
                self.socket.send(b"")
            self.socket.close()
            self.socket = None
