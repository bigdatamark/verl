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
# Distributed-training setup helpers: NUMA/CPU affinity, torch.distributed process group init/teardown (torchrun-style and Ray-style), TCPStore creation from a pre-bound socket, and stateless NCCL/HCCL group creation for weight sync with vLLM workers.

"""Utilities for distributed training."""

import ctypes
import os
import socket
from datetime import timedelta
from typing import Any

import ray
import torch.distributed
from torch.distributed import TCPStore

from verl.utils.device import get_device_name, get_nccl_backend, get_resource_name, get_torch_device, is_npu_available
from verl.utils.net_utils import is_ipv6


def set_numa_affinity():
    # [FUNCTION DOC]
    # Behavior: no-op on NPU. Otherwise loads libnuma; if NUMA is available, initializes NVML, determines the local GPU index (from Ray's assigned accelerator ids for the platform resource when Ray is initialized, else LOCAL_RANK env), and binds the process CPU affinity to that GPU's NUMA node.
    # Exceptions: none raised; ImportError and other exceptions are printed as warnings.
    # Side effects: changes process CPU affinity; NVML init/shutdown; prints.
    if is_npu_available:
        # TODO (FightingZhen) libnuma.so is not available in e2e_ascend CI image, remove this code after image update.
        return

    initialized = False
    try:
        libnuma = ctypes.CDLL("libnuma.so")
        if libnuma.numa_available() < 0:
            return

        import pynvml

        pynvml.nvmlInit()
        initialized = True
        device_name = get_resource_name()
        # Avoid ray.init in SFT trainer.
        if ray.is_initialized():
            local_rank = int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        pynvml.nvmlDeviceSetCpuAffinity(handle)
    except ImportError:
        print("Warning: pynvml not available, skipping NUMA affinity setup")
    except Exception as e:
        print(f"Warning: Failed to set NUMA affinity: {e}")
    finally:
        if initialized:
            pynvml.nvmlShutdown()


def initialize_global_process_group(timeout_second=36000):
    # [FUNCTION DOC]
    # Behavior: initializes the default process group with a mixed backend "cpu:gloo,<device>:<backend>", the given timeout and optional DIST_INIT_METHOD, reads LOCAL_RANK/RANK/WORLD_SIZE from env, and sets the current device to local_rank.
    # Inputs: timeout_second - collective timeout in seconds.
    # Outputs: tuple (local_rank, rank, world_size).
    # Exceptions: KeyError if env vars missing; init errors propagate.
    backend = f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
    torch.distributed.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=timeout_second),
        init_method=os.environ.get("DIST_INIT_METHOD", None),
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size


def destroy_global_process_group():
    # [FUNCTION DOC]
    # Destroys the default process group if initialized.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def initialize_global_process_group_ray(timeout_second=None, backend=None):
    # in current ray environment, LOCAL_RANK is always zero.

    # [FUNCTION DOC]
    # Behavior: if not already initialized, initializes the default process group using RANK/WORLD_SIZE env (defaults 0/1), optional timeout, backend (default mixed gloo + device backend), and DIST_INIT_METHOD. Does not set the device.
    # Inputs: timeout_second - optional seconds; backend - optional backend string.
    import torch.distributed

    timeout = timedelta(seconds=timeout_second) if timeout_second is not None else None
    backend = backend or f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
            init_method=os.environ.get("DIST_INIT_METHOD", None),
        )


def create_tcp_store(
    host: str,
    port: int,
    listen_socket: socket.socket | None = None,
    **kwargs: Any,
) -> TCPStore:
    # [FUNCTION DOC]
    # Behavior: creates a torch TCPStore; when listen_socket is given, detaches its fd and passes it as master_listen_fd so the store takes ownership; on construction failure closes the fd and re-raises.
    # Inputs: host, port, listen_socket (optional), extra TCPStore kwargs.
    # Outputs: TCPStore.
    # Side effects: detaches (invalidates) listen_socket object.
    """Create a TCPStore, optionally taking ownership of ``listen_socket``."""
    if listen_socket is None:
        return TCPStore(host_name=host, port=port, **kwargs)

    listen_fd = listen_socket.detach()
    try:
        return TCPStore(
            host_name=host,
            port=port,
            master_listen_fd=listen_fd,
            **kwargs,
        )
    except Exception:
        socket.close(listen_fd)
        raise


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    # [FUNCTION DOC]
    # Behavior: creates a vLLM StatelessProcessGroup (independent of the global torch group) at master_address:master_port, then wraps it with PyNcclCommunicator (or vllm_ascend PyHcclCommunicator on NPU) bound to device.
    # Inputs: master_address, master_port, rank, world_size, device.
    # Outputs: the communicator object.
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    # NOTE: If it is necessary to support weight synchronization with the sglang backend in the future,
    # the following can be used:
    # from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
    # from sglang.srt.distributed.utils import statelessprocessgroup

    from torch.distributed import TCPStore
    from vllm.distributed.utils import StatelessProcessGroup

    from verl.utils.device import is_npu_available

    if is_npu_available:
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator
    else:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    def create_process_group(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
        store_timeout: int = 300,
        listen_socket: socket.socket | None = None,
    ) -> "StatelessProcessGroup":
        # [FUNCTION DOC]
        # HELPER: IPv6-capable copy of vLLM StatelessProcessGroup.create. Rank 0 creates, binds and listens on a TCP socket (AF_INET6 if master_address is IPv6, else AF_INET, with SO_REUSEADDR) unless one is provided; other ranks use none. For vLLM >= 0.19 the store is built via create_tcp_store (socket ownership transferred) and the group without a socket argument; for older vLLM a TCPStore is built with master_listen_fd and the socket is passed to the group. Store uses is_master for rank 0, timeout store_timeout, use_libuv False.
        # Inputs: host, port, rank, world_size, data_expiration_seconds, store_timeout, listen_socket.
        # Outputs: StatelessProcessGroup.
        # Note: decides IPv6 from the enclosing master_address rather than host.
        """
        This is copied from vllm/distributed/utils.py:StatelessProcessGroup.create
        Modified to support ipv6 stateless communication groups."""
        launch_server = rank == 0
        if launch_server and listen_socket is None:
            if is_ipv6(master_address):
                listen_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            else:
                listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listen_socket.bind((host, port))
            listen_socket.listen()
            listen_fd = listen_socket.fileno()
        else:
            listen_socket = None
            listen_fd = None

        import vllm
        from packaging import version

        _VLLM_VERSION = version.parse(vllm.__version__)
        if _VLLM_VERSION >= version.parse("0.19.0"):
            store = create_tcp_store(
                host=host,
                port=port,
                listen_socket=listen_socket,
                world_size=world_size,
                is_master=launch_server,
                timeout=timedelta(seconds=store_timeout),
                use_libuv=False,
            )
            pg = StatelessProcessGroup(
                rank=rank,
                world_size=world_size,
                store=store,
                data_expiration_seconds=data_expiration_seconds,
            )
        else:
            store = TCPStore(
                host_name=host,
                port=port,
                world_size=world_size,
                is_master=launch_server,
                timeout=timedelta(seconds=store_timeout),
                use_libuv=False,  # for now: github.com/pytorch/pytorch/pull/150215
                master_listen_fd=listen_fd,
            )

            pg = StatelessProcessGroup(
                rank=rank,
                world_size=world_size,
                store=store,
                socket=listen_socket,
                data_expiration_seconds=data_expiration_seconds,
            )
        return pg

    pg = create_process_group(host=master_address, port=master_port, rank=rank, world_size=world_size)

    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl
