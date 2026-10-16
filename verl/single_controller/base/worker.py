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
# Base Worker class run inside each remote process (Ray actor) of a worker group, plus small rank/size dataclasses and a
# helper base providing node IP and free-port discovery. Workers read their distributed identity from environment
# variables set by the worker group, normalize visible-device env vars (CUDA/HIP/ROCR), and hold per-mesh dispatch and
# collect info used by the lazy N-D dispatch in the decorator module.

"""
the class for Worker
"""

import os
import socket
import warnings
from dataclasses import dataclass

import ray

from verl.utils.device import (
    get_resource_name,
    get_torch_device,
    get_visible_devices_keyword,
)

from .decorator import Dispatch, Execute, register


@dataclass
class DistRankInfo:
    # [CLASS DOC]
    # Dataclass of a worker's tensor/data/pipeline/context parallel ranks (ints).
    tp_rank: int
    dp_rank: int
    pp_rank: int
    cp_rank: int


@dataclass
class DistGlobalInfo:
    # [CLASS DOC]
    # Dataclass of tensor/data/pipeline/context parallel sizes (ints).
    tp_size: int
    dp_size: int
    pp_size: int
    cp_size: int


class WorkerHelper:
    # [CLASS DOC]
    # Mixin with networking helpers for choosing a rendezvous master address and port.
    @staticmethod
    def _get_node_ip():
        # [FUNCTION DOC]
        # HELPER: Outputs: this node's IP from Ray when env WG_BACKEND is ray. Exceptions: NotImplementedError otherwise.
        if os.getenv("WG_BACKEND", None) == "ray":
            return ray.util.get_node_ip_address()
        else:
            raise NotImplementedError("WG_BACKEND now just support ray mode.")

    @staticmethod
    def _get_free_port():
        # [FUNCTION DOC]
        # HELPER: Behavior: bind a socket to port 0 on all interfaces and return the OS-assigned port (socket closed after).
        # Outputs: int port. Note the port may be reused by others after closing (race).
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        # [FUNCTION DOC]
        # Deprecated misspelled alias: emits a warning and returns get_available_master_addr_port().
        warnings.warn(
            "This function is deprecated due to typo in name; Please use `get_available_master_addr_port` instead",
            stacklevel=2,
        )
        return self.get_available_master_addr_port()

    def get_available_master_addr_port(self):
        # [FUNCTION DOC]
        # Outputs: tuple (node IP with any IPv6 brackets stripped, free port as string).
        # Exceptions: NotImplementedError when not using the Ray backend.
        return self._get_node_ip().strip("[]"), str(self._get_free_port())


# we assume that in each WorkerGroup, there is a Master Worker
class Worker(WorkerHelper):
    # [CLASS DOC]
    # Base distributed worker. Class attribute fused_worker_attr_name names the instance mapping of fused sub-workers.
    # Instance state: _rank, _world_size, _local_rank, _local_world_size, _master_addr, _master_port, the lowercase
    # visible-devices attribute, fused_worker_dict, and two name-mangled private mappings from mesh name to dispatch DP rank
    # and to collect flag. Provides registered methods _query_dispatch_info/_query_collect_info (ONE_TO_ALL) and generic
    # execution helpers.
    """A distributed worker that handles initialization and configuration for distributed training.

    This class manages worker initialization, configuration, and provides methods for executing
    distributed operations. It handles communication settings, device configuration, and worker
    metadata management.
    """

    fused_worker_attr_name = "fused_worker_dict"

    def _register_dispatch_collect_info(self, mesh_name: str, dp_rank: int, is_collect: bool):
        # [FUNCTION DOC]
        # Behavior: record dp_rank and collect flag for mesh_name on this worker (called by subclasses during setup).
        # Exceptions: ValueError if the mesh is already registered in either mapping. Side effects: mutates private mappings.
        """Register the dp_rank for a given mesh name. This function is meant to be called by the worker

        Args:
            mesh_name (str):
                Name of the mesh to register dp_rank for.
            dp_rank (int):
                dp_rank to register for the given mesh name.
            is_collect (bool):
                Whether the dp_rank is used for collect.
        """
        if mesh_name in self.__dispatch_dp_rank or mesh_name in self.__collect_dp_rank:
            raise ValueError(f"mesh_name {mesh_name} has been registered")
        self.__dispatch_dp_rank[mesh_name] = dp_rank
        self.__collect_dp_rank[mesh_name] = is_collect

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_dispatch_info(self, mesh_name: str):
        # [FUNCTION DOC]
        # Remote (ONE_TO_ALL) method. Outputs: this rank's DP rank for mesh_name. Exceptions: AssertionError if unregistered.
        """Query the dispatch info for a given mesh name.

        Args:
            mesh_name (str):
                Name of the mesh to query dispatch info for.

        Returns:
            int:
                The dp_rank for the given mesh name.
        """
        assert mesh_name in self.__dispatch_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        # note that each rank store its own dp_rank
        return self.__dispatch_dp_rank[mesh_name]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _query_collect_info(self, mesh_name: str):
        # [FUNCTION DOC]
        # Remote (ONE_TO_ALL) wrapper over query_collect_info.
        return self.query_collect_info(mesh_name)

    def query_collect_info(self, mesh_name: str):
        # [FUNCTION DOC]
        # Outputs: this rank's collect flag for mesh_name. Exceptions: AssertionError if unregistered.
        """Query the collect info for a given mesh name.

        Args:
            mesh_name (str):
                Name of the mesh to query collect info for.

        Returns:
            bool:
                Whether the dp_rank is used for collect.
        """
        assert mesh_name in self.__collect_dp_rank, f"{mesh_name} is not registered in {self.__class__.__name__}"
        return self.__collect_dp_rank[mesh_name]

    def get_dispatch_collect(self):
        # [FUNCTION DOC]
        # Outputs: mapping with dispatch_dp_rank and collect_dp_rank entries referencing the private mappings (not copies).
        """Get all registered dispatch and collect dp_ranks.

        Returns:
            dict[str, int]:
                A dictionary mapping mesh names to their dispatch dp_ranks.
            dict[str, bool]:
                A dictionary mapping mesh names to whether they are used for collect.
        """
        return {"dispatch_dp_rank": self.__dispatch_dp_rank, "collect_dp_rank": self.__collect_dp_rank}

    def set_dispatch_collect(self, mesh_name: str, dispatch_dp_rank: dict[str, int], collect_dp_rank: dict[str, bool]):
        # [FUNCTION DOC]
        # Behavior: register mesh_name using values from the given mappings (typically copied from another worker's
        # get_dispatch_collect result).
        # Exceptions: AssertionError if mesh_name already registered.
        # KNOWN ISSUE: it iterates over all values of the input mappings and assigns each to the same mesh_name key, so only the
        # last value survives and the input mesh names are ignored; the docstring implies all meshes are set.
        """Set the dispatch and collect dp_ranks for all registered meshes.

        Args:
            mesh_name (str): Mesh name to set dispatch and collect dp_ranks for.
            dispatch_dp_rank (dict[str, int]):
                A dictionary mapping mesh names to their dispatch dp_ranks.
            collect_dp_rank (dict[str, bool]):
                A dictionary mapping mesh names to whether they are used for collect.
        """
        assert mesh_name not in self.__dispatch_dp_rank, (
            f"{mesh_name} is already registered, {self.__dispatch_dp_rank.keys()}"
        )
        assert mesh_name not in self.__collect_dp_rank, (
            f"{mesh_name} is already registered, {self.__collect_dp_rank.keys()}"
        )
        for dp_rank in dispatch_dp_rank.values():
            self.__dispatch_dp_rank[mesh_name] = dp_rank
        for is_collect in collect_dp_rank.values():
            self.__collect_dp_rank[mesh_name] = is_collect

    @classmethod
    def env_keys(cls):
        # [FUNCTION DOC]
        # Outputs: list of env var names used to configure the worker: WORLD_SIZE, RANK, LOCAL_WORLD_SIZE, LOCAL_RANK,
        # MASTER_ADDR, MASTER_PORT and the platform's visible-devices variable (uppercased).
        """The keys of the environment variables that are used to configure the Worker."""
        return [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_WORLD_SIZE",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
            get_visible_devices_keyword().upper(),
        ]

    def __init__(self, cuda_visible_devices=None) -> None:
        # [FUNCTION DOC]
        # Behavior: normalizes visible-device env vars, reads WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT (required) and
        # LOCAL_WORLD_SIZE/LOCAL_RANK (default 1/0) from the environment, builds a store mapping of underscore-prefixed
        # lowercase keys (plus visible devices if provided), applies it via _configure_with_store, and initializes the fused
        # worker mapping and the private dispatch/collect mappings.
        # Inputs: cuda_visible_devices - optional visible devices string to record.
        # Exceptions: KeyError if required env vars are missing; errors from env normalization.
        # Side effects: sets env vars and instance attributes; may set the torch device.
        """Initialize the worker with environment settings and device configuration.

        Args:
            cuda_visible_devices (str, optional):
                CUDA visible devices configuration. Defaults to None.
        """
        # construct a meta from environment variable. Note that the import must be inside the class because
        # it is executed remotely
        import os

        self._setup_env_cuda_visible_devices()

        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        self._rank = rank
        self._world_size = world_size

        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]

        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        store = {
            "_world_size": world_size,
            "_rank": rank,
            "_local_world_size": local_world_size,
            "_local_rank": local_rank,
            "_master_addr": master_addr,
            "_master_port": master_port,
        }
        if cuda_visible_devices is not None:
            store[f"_{get_visible_devices_keyword()}".lower()] = cuda_visible_devices

        self._configure_with_store(store=store)

        self.fused_worker_dict = {}
        self.__dispatch_dp_rank = {}
        self.__collect_dp_rank = {}

    def get_fused_worker_by_name(self, worker_name: str):
        # [FUNCTION DOC]
        # Outputs: the fused sub-worker registered under worker_name, or None.
        """Get a fused worker by its name.

        Args:
            worker_name (str):
                Name of the worker to retrieve
        """
        return self.fused_worker_dict.get(worker_name, None)

    def _setup_env_cuda_visible_devices(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: if HIP_VISIBLE_DEVICES is set, remove it and move its value to CUDA_VISIBLE_DEVICES (asserting equality
        # if CUDA is also set). If ROCR_VISIBLE_DEVICES is set, error if CUDA/HIP is also set, else move it to
        # CUDA_VISIBLE_DEVICES. If Ray was told not to set visible devices, take this actor's first assigned accelerator id from
        # the Ray runtime context, write it to LOCAL_RANK and set it as the current torch device.
        # Exceptions: AssertionError on HIP/CUDA mismatch; ValueError when ROCR and CUDA/HIP are both set.
        # Side effects: mutates os.environ and current device.
        from verl.utils.ray_utils import ray_noset_visible_devices

        is_ray_noset_visible_devices = ray_noset_visible_devices()

        # Prevent use of clashing `{CUDA/HIP/ROCR}_VISIBLE_DEVICES``
        rocr_val = os.environ.get("ROCR_VISIBLE_DEVICES", None)
        hip_val = os.environ.get("HIP_VISIBLE_DEVICES", None)
        cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if hip_val:
            # Switch the use of HIP_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES for consistency.
            # Make sure that the HIP_VISIBLE_DEVICES is set to the same value as CUDA_VISIBLE_DEVICES
            # at this point.
            val = os.environ.pop("HIP_VISIBLE_DEVICES")
            hip_val = None
            if cuda_val:
                assert val == cuda_val, (
                    f"Please use the same HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES, inconsistant values "
                    f"found: {val} and {cuda_val}."
                )
            else:
                cuda_val = val
                os.environ["CUDA_VISIBLE_DEVICES"] = val
                # os.environ["HIP_VISIBLE_DEVICES"] = val

        if rocr_val:
            # You must take care if both HIP/CUDA and ROCR env vars are set as they have
            # different meanings. Both env vars accept either a list of ints or a
            # list of UUIDs. The ROCR env var is processed first which then reduces
            # the number of GPUs that HIP can select from.
            # https://github.com/pytorch/pytorch/pull/144026
            # To avoid the complexity of this, we simply gives out error if both are set
            # (Also to keep consistency with ray's practice with 2.45.0).
            # Otherwise, we will set ROCR_VISIBLE_DEVICES to CUDA_VISIBLE_DEVICES
            # and remove ROCR_VISIBLE_DEVICES.
            if cuda_val:
                raise ValueError("Please don't set ROCR_VISIBLE_DEVICES when HIP/CUDA_VISIBLE_DEVICES is set.")

            cuda_val = os.environ.pop("ROCR_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_val
            rocr_val = None

        if is_ray_noset_visible_devices:
            # NOTE: Ray will automatically set the *_VISIBLE_DEVICES
            # environment variable for each actor, unless
            # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set,
            # so we need to set local rank when the flag is set.
            device_name = get_resource_name()
            local_rank = ray.get_runtime_context().get_accelerator_ids()[device_name][0]
            os.environ["LOCAL_RANK"] = local_rank
            get_torch_device().set_device(int(local_rank))

    def _configure_with_store(self, store: dict):
        # [FUNCTION DOC]
        # HELPER: Behavior: for each env key, set an instance attribute named underscore plus the lowercase key from the store
        # (None if absent); then export every non-None value to os.environ under the original key. Sets
        # REDIS_STORE_SERVER_HOST to the master address without brackets (empty if none).
        # Inputs: store - mapping of underscore-lowercase keys to values. Side effects: mutates instance dict and env.
        """
        This function should only be called inside by WorkerGroup
        """
        store_env_dict = {f"_{key.lower()}": store.get(f"_{key.lower()}", None) for key in type(self).env_keys()}
        self.__dict__.update(store_env_dict)  # this is hacky
        # print(f"__dict__: {self.__dict__}")
        for key in type(self).env_keys():
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                # print(f"set {key} to {val}")
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = (
            str(self._master_addr).replace("[", "").replace("]", "") if self._master_addr else ""
        )

    def get_master_addr_port(self):
        # [FUNCTION DOC]
        # Outputs: (master address, master port).
        """Get the master address and port for distributed communication."""
        return self._master_addr, self._master_port

    def get_cuda_visible_devices(self):
        # [FUNCTION DOC]
        # Outputs: current value of the visible-devices env var, or the string not set.
        """Get the CUDA visible devices configuration."""
        import os

        visible_devices = os.environ.get(get_visible_devices_keyword().upper(), "not set")
        return visible_devices

    @property
    def world_size(self):
        # [FUNCTION DOC]
        # Outputs: world size.
        """Get the total number of workers in the distributed setup."""
        return self._world_size

    @property
    def rank(self):
        # [FUNCTION DOC]
        # Outputs: global rank.
        """Get the rank of this worker in the distributed setup."""
        return self._rank

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        # [FUNCTION DOC]
        # Remote (DP_COMPUTE_PROTO_WITH_FUNC) method: calls func(self, *args, **kwargs) on this rank's data chunk and returns it.
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        # [FUNCTION DOC]
        # Remote (ALL_TO_ALL dispatch, RANK_ZERO execute) method: calls func(*args, **kwargs) (without self) and returns it.
        """Execute a function in rank zero execution mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        result = func(*args, **kwargs)
        return result
