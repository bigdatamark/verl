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
# Ray backend of the single controller. Provides: Ray placement-group based resource pools (including splitting into
# sub-pools and merging), a manager creating named pools from a spec, a Ray actor class wrapper that applies scheduling
# and resource options, RayWorkerGroup (creates/attaches Ray actor workers, binds registered methods via func_generator,
# executes remote calls on all workers or rank zero, supports spawning per-role groups and fused colocated workers),
# and factories that build colocated worker classes (deprecated WorkerDict and the newer FusedWorker). Logger level from
# VERL_LOGGING_LEVEL (default WARN).

import inspect
import logging
import os
import socket
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import ray
from ray.experimental.state.api import get_actor
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from verl.plugin.platform import get_platform
from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base import ClassWithInitArgs, ResourcePool, Worker, WorkerGroup
from verl.single_controller.base.decorator import MAGIC_ATTR, Dispatch
from verl.utils.device import get_device_name
from verl.utils.py_functional import temp_env_var

__all__ = ["Worker"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_random_string(length: int) -> str:
    # [FUNCTION DOC]
    # HELPER: Outputs: random string of the given length drawn from ASCII letters and digits (not cryptographically secure).
    import random
    import string

    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    # [FUNCTION DOC]
    # HELPER: Behavior: builds a callable object, whose class is dynamically named after method_name for observability, that
    # when called: runs dispatch_fn(worker_group, args, kwargs) to get per-rank arguments; pops the padding-size key from
    # kwargs (default 0); calls execute_fn(method_name, args, kwargs); ray.get the result if blocking; runs collect_fn; and if
    # padding was added, strips the trailing padding_count entries (index selection for DataProto, slicing for lists).
    # Inputs: self - the worker group; method_name; dispatch_fn; collect_fn; execute_fn; blocking.
    # Outputs: instance of the dynamically created functor class.
    class Functor:
        # [CLASS DOC]
        # HELPER: callable wrapper class; see func_generator.
        def __call__(this, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: performs dispatch, execute, optional blocking get, collect and padding removal; see func_generator.
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


def sort_placement_group_by_node_ip(pgs: list[PlacementGroup]) -> list[PlacementGroup]:
    # [FUNCTION DOC]
    # Behavior: maps node id to node IP from ray.nodes(); for each placement group, looks up the node of its first bundle via
    # Ray's internal placement group table, and returns the groups sorted by that IP string (stable rank assignment across
    # restarts for checkpoint resume).
    # Inputs: pgs - placement groups (each assumed to live on a single node). Outputs: sorted list.
    # Side effects: queries Ray GCS state.
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])


@ray.remote
def get_master_addr_port(master_port_range: Optional[list[int]] = None) -> tuple[str, str]:
    # [FUNCTION DOC]
    # Ray remote task. Behavior: get the node IP (brackets stripped). Without a range, bind port 0 to pick a free port.
    # With a range [start, end), try binding ports sequentially from start until one succeeds, logging collisions.
    # Inputs: master_port_range - optional two-element list. Outputs: (address, port as string).
    # Exceptions: RuntimeError if no port in range is free. Note: the port is released before use (race possible).
    addr = ray.util.get_node_ip_address().strip("[]")

    if master_port_range is None:
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
    else:
        port = master_port_range[0]
        while port < master_port_range[1]:
            try:
                with socket.socket() as s:
                    s.bind(("", port))
                    break
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
        else:
            raise RuntimeError(f"Could not find a free port in range {master_port_range}")
    return addr, str(port)


class RayResourcePool(ResourcePool):
    # [CLASS DOC]
    # Ray resource pool backed by placement groups, one per node. Extra state: use_gpu, name_prefix (random 6 chars if none),
    # cached placement groups (pgs), detached flag, optional accelerator_type.
    def __init__(
        self,
        process_on_nodes: Optional[list[int]] = None,
        use_gpu: bool = True,
        name_prefix: str = None,
        max_colocate_count: int = 10,
        detached=False,
        accelerator_type: Optional[str] = None,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: process_on_nodes - per-node process counts; use_gpu; name_prefix; max_colocate_count; detached - create
        # placement groups with detached lifetime; accelerator_type - optional Ray accelerator-type resource to require.
        # Side effects: none (pgs created lazily). Note n_gpus_per_node is left at the base default.
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self.pgs = None
        self.detached = detached
        self.accelerator_type = accelerator_type

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        # [FUNCTION DOC]
        # Behavior: returns cached placement groups if present. Otherwise builds one placement group per node, each with one
        # bundle per process: CPU equal to max_colocate_count plus, if use_gpu, one unit of the platform's Ray accelerator
        # resource (and a tiny 1e-4 amount of accelerator_type if set). Names are the given name or the prefix plus
        # verl_group_ plus the per-node counts joined by underscores and a colon, then the node index. Waits for all groups to
        # be ready, caches the IP-sorted list.
        # Inputs: strategy - Ray placement strategy (default STRICT_PACK); name - optional name prefix; device_name - requested
        # device (only a warning if it differs from the platform's device).
        # Outputs: list of placement groups.
        # Side effects: creates Ray placement groups (blocking).
        # KNOWN ISSUE: on the first call it returns the unsorted creation-order list while caching the sorted list, so the first
        # and later calls may return different orders.
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = (
            name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        )
        # print(f"pg_name_prefix = {pg_name_prefix}")
        current_platform = get_platform()
        if device_name != current_platform.device_name:
            logger.warning(
                f"Requested device {device_name} does not match current platform device {current_platform.device_name}"
            )
        device_name = current_platform.ray_resource_name()

        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = 1
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]

        lifetime = "detached" if self.detached else None

        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = sort_placement_group_by_node_ip(pgs)
        return pgs


class SubRayResourcePool(RayResourcePool):
    # [CLASS DOC]
    # Sub-pool view over a contiguous range of bundles of an existing pool's placement groups. Extra state: start_bundle_index
    # (global bundle offset across the pool's groups) and subgroup_world_size, which overrides world_size.
    def __init__(
        self,
        placement_groups: list[PlacementGroup],
        start_bundle_index: int,
        subgroup_world_size: int,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Inputs: placement_groups - parent's groups (shared, set as cached pgs); start_bundle_index; subgroup_world_size;
        # kwargs - forwarded to RayResourcePool.
        super().__init__(**kwargs)
        self.pgs = placement_groups
        self.start_bundle_index = start_bundle_index
        self.subgroup_world_size = subgroup_world_size

    @property
    def world_size(self):
        # [FUNCTION DOC]
        # Outputs: subgroup_world_size.
        return self.subgroup_world_size


@dataclass
class ResourcePoolManager:
    # [CLASS DOC]
    # Dataclass. Fields: resource_pool_spec - mapping pool name to per-node process counts; mapping - role to pool name;
    # max_colocate_count (default 3); resource_pool_dict - created pools by name.
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[int, str]
    max_colocate_count: int = 3
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        # [FUNCTION DOC]
        # Behavior: create a RayResourcePool (use_gpu, the manager's colocate count, name prefix = pool name) per spec entry, store
        # it, then verify the cluster has enough accelerators.
        # Exceptions: ValueError from the availability check. Side effects: fills resource_pool_dict.
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=self.max_colocate_count,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role) -> RayResourcePool:
        # [FUNCTION DOC]
        # Inputs: role - role key. Outputs: its RayResourcePool. Exceptions: KeyError if unmapped or not created.
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        # [FUNCTION DOC]
        # Outputs: total processes (accelerators) across all pool specs.
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        # [FUNCTION DOC]
        # HELPER: Behavior: read available resources per node from Ray internals, counting GPU if present else NPU per node; raise
        # if the cluster total is below the sum of all spec counts. Only totals are compared, not per-node fit.
        # Exceptions: ValueError when insufficient.
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def extract_pg_from_exist(
    resource_pools: dict[str, RayResourcePool], src_role_names: list[str], resource_pool: RayResourcePool
) -> list:
    # [FUNCTION DOC]
    # Behavior: collect the placement groups of all pools whose role name is in src_role_names, sort by bundle count
    # descending; sort the target pool's per-node process counts descending; greedily assign the i-th largest request to the
    # i-th largest group, then return the groups ordered by the target's original node index.
    # Inputs: resource_pools - name to pool; src_role_names - names to reuse; resource_pool - target pool.
    # Outputs: list of placement groups for the target's nodes.
    # Exceptions: AssertionError if there are too few groups or a group has too few bundles.
    # Side effects: may create placement groups of source pools.
    src_pgs = [
        pg
        for role_name, resource_pool in resource_pools.items()
        for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: list[tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, (
            f"requesting {request_process} processes, bundle count cannot satisfy"
        )
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


# split a RayResourcePool or SubRayResourcePool into multiple SubRayResourcePool
def split_resource_pool(
    resource_pool: RayResourcePool | SubRayResourcePool, split_size: int | list[int]
) -> list[SubRayResourcePool]:
    # [FUNCTION DOC]
    # Behavior: normalize split_size into a list (an int must divide world_size, producing equal splits), assert the sum
    # equals world_size, compute start bundle indices as cumulative offsets (starting at the sub-pool's own start for an
    # already-split pool), ensure placement groups exist, and build one SubRayResourcePool per split sharing those groups
    # with name prefix suffixed by _split_ and the index.
    # Outputs: list of sub-pools. Exceptions: AssertionError on bad sizes.
    # Side effects: may create placement groups.
    """
    Split a RayResourcePool into multiple SubRayResourcePool.
    resouce_pool can also be a SubRayResourcePool (have been splited) for multiple-time spliting.

    Args:
        resource_pool (RayResourcePool | SubRayResourcePool): The resource pool to split.
        split_size (int | list[int]): The size of each split. If int, all splits will have the same size.
            If list[int], each element in the list represents the size of a split.

    Returns:
        list[SubRayResourcePool]: A list of SubRayResourcePool after splitting.
    """
    # convert split_size to list[int]
    if isinstance(split_size, int):
        assert resource_pool.world_size % split_size == 0, "split_size must be a divisor of world_size"
        num_replica = resource_pool.world_size // split_size
        split_size_list = [split_size] * num_replica
    else:
        split_size_list = split_size

    assert sum(split_size_list) == resource_pool.world_size, "split_size must sum up to world_size"

    # judge if this resource pool has been splited
    if isinstance(resource_pool, SubRayResourcePool):
        start_bundle_idx_list = np.cumsum([resource_pool.start_bundle_index] + split_size_list[:-1])
    else:
        start_bundle_idx_list = np.cumsum([0] + split_size_list[:-1])

    # ensure resource_pool.pgs has been initialized
    device = get_device_name()
    placement_groups = resource_pool.get_placement_groups(device_name=device)
    split_resource_pools = [
        SubRayResourcePool(
            process_on_nodes=resource_pool.store,
            use_gpu=resource_pool.use_gpu,
            name_prefix=f"{resource_pool.name_prefix}_split_{split_idx}",
            max_colocate_count=resource_pool.max_colocate_count,
            placement_groups=placement_groups,
            start_bundle_index=start_bundle_idx_list[split_idx],
            subgroup_world_size=split_size_list[split_idx],
        )
        for split_idx in range(len(split_size_list))
    ]
    return split_resource_pools


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    # [FUNCTION DOC]
    # Behavior: assert both pools agree on use_gpu, max_colocate_count, n_gpus_per_node and detached; create a new pool of the
    # same type as rp1 with concatenated per-node counts and combined name prefix, and set its cached placement groups to the
    # concatenation of both pools' groups.
    # Outputs: merged pool. Exceptions: AssertionError on mismatch. Side effects: may create placement groups.
    # Note: accelerator_type is not carried over to the merged pool.
    assert rp1.use_gpu == rp2.use_gpu, "Both RayResourcePool must either use_gpu or not"
    assert rp1.max_colocate_count == rp2.max_colocate_count, "Both RayResourcePool must has the same max_colocate_count"
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayResourcePool must has the same n_gpus_per_node"
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    new_store = rp1.store + rp2.store

    merged = type(rp1)(
        new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}", rp1.max_colocate_count, rp1.detached
    )
    merged.pgs = rp1.get_placement_groups(device_name=get_device_name()) + rp2.get_placement_groups(
        device_name=get_device_name()
    )

    return merged


class RayClassWithInitArgs(ClassWithInitArgs):
    # [CLASS DOC]
    # Ray actor wrapper adding actor options (_options) and additional custom resources (_additional_resource).
    """A wrapper class for Ray actors with initialization arguments.

    This class extends ClassWithInitArgs to provide additional functionality for
    configuring and creating Ray actors with specific resource requirements and
    scheduling strategies.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # [FUNCTION DOC]
        # Inputs: cls - Ray remote class; args/kwargs - constructor arguments. Initializes empty options and resources.
        # self._options = kwargs.pop('options', dict())
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        # [FUNCTION DOC]
        # Side effects: replaces the additional resource mapping.
        """Set additional resource requirements for the actor.

        Args:
            additional_resource: Dictionary specifying additional resource requirements
        """
        self._additional_resource = additional_resource

    def update_options(self, options: dict):
        # [FUNCTION DOC]
        # Side effects: merges options into the stored Ray options.
        """Update the Ray actor creation options.

        Args:
            options: Dictionary of options to update
        """
        self._options.update(options)

    def __call__(
        self,
        placement_group,
        placement_group_bundle_idx,
        use_gpu: bool = True,
        num_gpus=1,
        sharing_with=None,
        device_name="cuda",
    ) -> Any:
        # [FUNCTION DOC]
        # Behavior: if sharing_with is given, fetch that actor's node id and visible devices, and create the actor with strict
        # node affinity to that node, passing those visible devices as cuda_visible_devices (stored options are ignored in this
        # path). Otherwise schedule into the given placement group bundle, merge stored options, add the platform's accelerator
        # resource options for num_gpus when use_gpu, add additional resources, and create the actor.
        # Inputs: placement_group; placement_group_bundle_idx; use_gpu; num_gpus (fractional allowed); sharing_with - optional
        # actor handle; device_name (unused).
        # Outputs: Ray actor handle.
        # KNOWN ISSUE: additional resources are only applied when the mapping has more than one entry, so a single additional
        # resource is silently ignored.
        """Create and return a Ray actor with the configured options.

        Args:
            placement_group: Ray placement group for scheduling
            placement_group_bundle_idx: Index of the bundle in the placement group
            use_gpu: Whether to use GPU resources
            num_gpus: Number of GPUs to allocate
            sharing_with: Actor to share resources with
            device_name: Device for training

        Returns:
            A Ray actor handle with the configured options
        """
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args, cuda_visible_devices=visible_devices, **self.kwargs)

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_idx
            )
        }
        options.update(self._options)

        if use_gpu:
            resource_opts = get_platform().ray_resource_options(num_gpus)
            options.update(resource_opts)

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup):
    # [CLASS DOC]
    # Ray implementation of WorkerGroup. Extra state: ray_cls_with_init, name_prefix, use_gpu, master port range, register
    # center timeout, fused_worker_used, sub_cls_name (role bound when spawned from a fused group), device_name,
    # profiling settings (profile_steps, worker_nsight_options), customized worker env, resource_pool, _world_size,
    # wg_dict (spawned role groups) and method_names (methods bound on the fused group itself).
    """A group of Ray workers that can be managed collectively.

    This class extends WorkerGroup to provide Ray-specific functionality for
    creating and managing groups of Ray actors with specific resource requirements
    and scheduling strategies.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool = None,
        ray_cls_with_init: RayClassWithInitArgs = None,
        bin_pack: bool = True,
        name_prefix: str = None,
        detached=False,
        worker_names=None,
        worker_handles: list[ray.actor.ActorHandle] = None,
        ray_wait_register_center_timeout: int = 300,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: pops master_addr, master_port, use_gpu (default from pool or True) and master_port_range from kwargs, calls
        # the base init, then records settings; device_name, profile_steps, worker_nsight_options and worker_env are read from
        # the remaining kwargs. If nsight options have no capture-range-end, sets it to repeat-shutdown with 6 times the number
        # of profile steps. Initializes workers by attaching to detached workers (no pool), from a sub-pool, or from a regular
        # pool; then binds registered methods of the worker class via func_generator if a class is given.
        # Inputs: resource_pool; ray_cls_with_init; bin_pack - STRICT_PACK vs PACK; name_prefix; detached; worker_names /
        # worker_handles - for attaching; ray_wait_register_center_timeout (stored only); kwargs as above.
        # Exceptions: AssertionError if worker_names are given with a pool (non-fused); errors from worker creation.
        # Side effects: creates placement groups and Ray actors.
        # KNOWN ISSUE: master_addr/master_port popped from kwargs are assigned before the base initializer, which resets both to
        # None, so manually specified master address/port are discarded.
        """Initialize a RayWorkerGroup.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            name_prefix: Prefix for worker names
            detached: Whether workers should be detached
            worker_names: Names of existing workers to attach to
            ray_wait_register_center_timeout: Timeout for waiting on register center
            **kwargs: Additional keyword arguments
        """
        self._master_addr = kwargs.pop("master_addr", None)
        self._master_port = kwargs.pop("master_port", None)
        self.use_gpu = kwargs.pop("use_gpu", resource_pool.use_gpu if resource_pool is not None else True)
        self._ray_master_port_range = kwargs.pop("master_port_range", None)
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix
        self._ray_wait_register_center_timeout = ray_wait_register_center_timeout
        # Whether the WorkerGroup is a Colocate WorkerGroup created by FusedWorker.
        self.fused_worker_used = False if ray_cls_with_init is None else ray_cls_with_init.fused_worker_used
        # if a WorkerGroup is spawned from Colocate WorkerGroup, this indicates which sub-class is binded to
        # this WorkerGroup.
        self.sub_cls_name = ""
        self.device_name = kwargs.get("device_name", "cuda")
        self.profile_steps = kwargs.get("profile_steps", None)
        self.worker_nsight_options = kwargs.get("worker_nsight_options", None)
        self.customized_worker_env = kwargs.get("worker_env", {})
        if self.worker_nsight_options is not None and self.worker_nsight_options["capture-range-end"] is None:
            self.worker_nsight_options["capture-range-end"] = f"repeat-shutdown:{6 * len(self.profile_steps)}"

        if worker_names is not None and (not self.fused_worker_used):
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names, worker_handles=worker_handles)
        elif isinstance(resource_pool, SubRayResourcePool):
            self._init_with_subresource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )
        else:
            self._init_with_resource_pool(
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                bin_pack=bin_pack,
                detached=detached,
                worker_env=self.customized_worker_env,
            )

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

        self.wg_dict = None
        self.method_names = []

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        # [FUNCTION DOC]
        # Behavior: query Ray state API for the actor; True iff its state is ALIVE (False if not found).
        """Check if a worker actor is still alive.

        Args:
            worker: Ray actor handle to check

        Returns:
            bool: True if the worker is alive, False otherwise
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names, worker_handles):
        # [FUNCTION DOC]
        # HELPER: use the given handles (strong references) or look up named actors; set workers and world size.
        # ray.get_actor holds a weak reference to the actor, which causes actors garbage collected unexpectedly
        # if we only hold spawn RayWorkerGroup. By passing actor handle explicitly, spawn RayWorkerGroup have
        # strong reference to these actors.
        # https://github.com/ray-project/ray/pull/45699
        workers = worker_handles if worker_handles else [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(workers)

    def _get_master_addr_port(self, pg, bundle_index=0, master_port_range=None):
        # [FUNCTION DOC]
        # HELPER: Behavior: if neither master addr nor port is set, run the get_master_addr_port remote task in the given placement
        # group bundle and store the result; if both are set, keep them; otherwise raise.
        # Exceptions: ValueError when only one of addr/port is set. Side effects: launches a Ray task (blocking).
        """Get master addr and port for this worker group"""
        if self._master_addr is None and self._master_port is None:
            self._master_addr, self._master_port = ray.get(
                get_master_addr_port.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=bundle_index
                    ),
                ).remote(master_port_range=master_port_range)
            )
        elif self._master_addr is not None and self._master_port is not None:
            logger.debug(f"{self._master_addr=} {self._master_port=}")
        else:
            raise ValueError(
                "Both 'master_addr' and 'master_port' must be provided if you intend to manually specify them, "
                "or neither should be provided to use Ray's default assignment."
            )

    def _init_with_resource_pool(
        self,
        resource_pool,
        ray_cls_with_init,
        bin_pack,
        detached,
        worker_env=None,
    ):
        # [FUNCTION DOC]
        # HELPER: Behavior: get placement groups (STRICT_PACK if bin_pack else PACK), set world size, then iterate over the
        # IP-sorted groups: get master addr/port from bundle 0 of the first group; for each local rank up to the first node's
        # process count create a worker with incrementing global rank.
        # Side effects: stores resource_pool; creates actors.
        # Note: uses the first node's process count for all nodes (assumes homogeneous nodes).
        """Initialize the worker group by creating new workers from a resource pool.

        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        self.resource_pool = resource_pool
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_group_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "
            if pg_idx == 0:
                self._get_master_addr_port(pg, bundle_index=0, master_port_range=self._ray_master_port_range)

            for local_rank in range(local_world_size):
                rank += 1
                self._create_worker(
                    rank=rank,
                    pg_idx=pg_idx,
                    pg=pg,
                    local_rank=local_rank,
                    resource_pool=resource_pool,
                    ray_cls_with_init=ray_cls_with_init,
                    worker_env=worker_env,
                    detached=detached,
                )

    def _init_with_subresource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached, worker_env=None):
        # [FUNCTION DOC]
        # HELPER: Behavior: like the regular pool init, but over global bundle indices from start_bundle_index for world_size
        # workers; the group index and local rank are derived by dividing by and taking the remainder against the first node's
        # process count; master addr/port come from the start bundle.
        # Side effects: creates actors.
        """Initialize the worker group by creating new workers from a resource pool or sub resource pool.
        Args:
            resource_pool: Resource pool for worker allocation
            ray_cls_with_init: Class with initialization arguments for workers
            bin_pack: Whether to use strict bin packing for resource allocation
            detached: Whether workers should be detached
        """
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy, device_name=self.device_name)
        world_size = resource_pool.world_size
        self._world_size = world_size

        rank = -1
        local_world_size = resource_pool.store[0]
        self._get_master_addr_port(
            pgs[resource_pool.start_bundle_index // local_world_size],
            bundle_index=resource_pool.start_bundle_index % local_world_size,
            master_port_range=self._ray_master_port_range,
        )
        for curr_rank in range(resource_pool.start_bundle_index, resource_pool.start_bundle_index + world_size):
            pg_idx = curr_rank // local_world_size
            pg = pgs[pg_idx]
            local_rank = curr_rank % local_world_size
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix}, for the "

            rank += 1
            self._create_worker(
                rank=rank,
                pg_idx=pg_idx,
                pg=pg,
                local_rank=local_rank,
                resource_pool=resource_pool,
                ray_cls_with_init=ray_cls_with_init,
                worker_env=worker_env,
                detached=detached,
            )

    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, ray_cls_with_init, worker_env, detached):
        # [FUNCTION DOC]
        # HELPER: Behavior: builds env vars WORLD_SIZE, RANK, WG_PREFIX, WG_BACKEND=ray, RAY_LOCAL_WORLD_SIZE, MASTER_ADDR,
        # MASTER_PORT, merges user worker_env (refusing overrides of these keys), derives the actor name as prefix + underlying
        # class name (parsed from the Ray ActorClass(...) type name) + _<pg index>:<local rank>, sets runtime_env (adding nsight
        # options when profiling on cuda), name and optional detached lifetime on the wrapper, and creates the actor in the given
        # bundle with 1/max_colocate_count accelerators.
        # Exceptions: ValueError if use_gpu is requested but the pool has no GPUs, or on env conflicts.
        # Side effects: mutates wrapper options; creates an actor; appends to workers and names.
        world_size = resource_pool.world_size
        use_gpu = resource_pool.use_gpu
        if self.use_gpu and not use_gpu:
            raise ValueError("use_gpu is True but resource_pool.use_gpu is False")
        local_world_size = resource_pool.store[0]
        num_gpus = 1 / resource_pool.max_colocate_count

        # we pass in environment variable at option so that Worker can use environment variable to set
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "WG_PREFIX": self.name_prefix,
            "WG_BACKEND": "ray",
            "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
            "MASTER_ADDR": self._master_addr,
            "MASTER_PORT": self._master_port,
        }
        if worker_env is not None:
            logging.debug(f"Appending ray class env, origin: {env_vars}, customized env: {worker_env}")
            conflict_env_vars = set(env_vars.keys()) & set(worker_env.keys())
            if len(conflict_env_vars) > 0:
                logging.error(
                    f"User customized env vars conflict with system env: {conflict_env_vars} "
                    f"Overriding may cause unexpected behavior."
                )
                raise ValueError(f"Cannot override protected system env: {conflict_env_vars}")
            env_vars.update(worker_env)
        import re

        cia_name = type(ray_cls_with_init.cls).__name__
        match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
        cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
        name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

        if self.profile_steps and self.device_name == "cuda":
            runtime_env = {
                "env_vars": env_vars,
                "nsight": self.worker_nsight_options,
            }
        else:
            runtime_env = {"env_vars": env_vars}
        ray_cls_with_init.update_options({"runtime_env": runtime_env, "name": name})

        if detached:
            ray_cls_with_init.update_options({"lifetime": "detached"})

        # create a worker
        worker = ray_cls_with_init(
            placement_group=pg,
            placement_group_bundle_idx=local_rank,
            use_gpu=self.use_gpu,
            num_gpus=num_gpus,
            device_name=self.device_name,
        )
        self._workers.append(worker)
        self._worker_names.append(name)

    @property
    def worker_names(self):
        # [FUNCTION DOC]
        # Outputs: list of worker actor names.
        return self._worker_names

    @classmethod
    def from_detached(
        cls,
        name_prefix=None,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Classmethod. Behavior: construct a group with no resource pool attached to existing workers by names/handles.
        # Outputs: new group.
        """Create a worker group from existing detached workers.

        Args:
            name_prefix: Prefix for worker names
            worker_names: Names of existing workers to attach to
            ray_cls_with_init: Class with initialization arguments for workers

        Returns:
            A new RayWorkerGroup instance
        """
        worker_group = cls(
            resource_pool=None,
            ray_cls_with_init=ray_cls_with_init,
            name_prefix=name_prefix,
            worker_names=worker_names,
            worker_handles=worker_handles,
            **kwargs,
        )
        return worker_group

    def spawn(self, prefix_set):
        # [FUNCTION DOC]
        # Behavior: for fused groups delegate to spawn_fused. Otherwise for each prefix, create a detached-attached copy sharing
        # the same workers and class, then for every attribute name starting with prefix plus underscore, also expose it under
        # the name without the prefix.
        # Inputs: prefix_set - role names. Outputs: mapping role -> worker group.
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        if self.fused_worker_used:
            return self.spawn_fused(prefix_set)

        def _rebind_actor_methods(worker_group, actor_name):
            # [FUNCTION DOC]
            # HELPER: aliases prefixed methods of worker_group under their unprefixed names.
            prefix: str = actor_name + "_"
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.from_detached(
                name_prefix=self.name_prefix,
                worker_names=self._worker_names,
                worker_handles=self._workers,
                ray_cls_with_init=self.ray_cls_with_init,
                profile_steps=self.profile_steps,
                worker_nsight_options=self.worker_nsight_options,
            )

            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def spawn_fused(self, prefix_set):
        # [FUNCTION DOC]
        # Behavior: for each role key, deep-copy this group, bind methods of that role's raw class (from the fused class's
        # raw_cls_dict) and record the role as sub_cls_name. Outputs: mapping role -> group.
        """Create a dictionary of worker groups for fused workers.

        Args:
            prefix_set: Set of prefixes to create worker groups for

        Returns:
            Dictionary of worker groups keyed by prefix
        """
        wg_dict = dict()
        for key in prefix_set:
            new_wg = deepcopy(self)
            new_wg._bind_worker_method(self.ray_cls_with_init.cls.raw_cls_dict[key], func_generator)
            new_wg.sub_cls_name = key
            wg_dict[key] = new_wg
        return wg_dict

    def fuse(self, prefix_set):
        # [FUNCTION DOC]
        # Behavior: spawn role groups once (cached in wg_dict), set each as an attribute named by role, and bind the fused class's
        # own registered methods, recording their names in method_names.
        # Side effects: adds attributes to self.
        """Fuse multiple worker groups into the current worker group.

        Args:
            prefix_set: Set of prefixes to fuse into the worker group
        """
        if self.wg_dict is None:
            self.wg_dict = self.spawn(prefix_set)
        for role_name, role_wg in self.wg_dict.items():
            setattr(self, role_name, role_wg)
        self.method_names = self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _execute_remote_single_worker(self, worker, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Behavior: for fused groups when the method is not one of the fused class's own methods, call the actor's fused
        # routing method with the encoded name sub-role + _fwmn_ + method; otherwise call the actor method directly.
        # Outputs: Ray object ref.
        """Execute a method on a single worker remotely.

        Args:
            worker: The worker actor handle
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        if self.fused_worker_used and method_name not in self.method_names:
            remote_call = getattr(worker, self.fused_worker_execute_fn_name)
            return remote_call.remote(f"{self.sub_cls_name}_fwmn_{method_name}", *args, **kwargs)
        # fused worker not used
        remote_call = getattr(worker, method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Outputs: result of calling method on rank 0 (blocking).
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        return ray.get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Outputs: object ref of calling method on rank 0.
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self._execute_remote_single_worker(self._workers[0], method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Alias of execute_rank_zero_async.
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Alias of execute_all_async.
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Outputs: list of results from all workers (blocking).
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: if every arg and kwarg value is a list of length equal to the worker count, send element i to worker i;
        # otherwise broadcast all arguments unchanged to every worker.
        # Outputs: list of object refs, one per worker.
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self._execute_remote_single_worker(self._workers[i], method_name, *sliced_args, **sliced_kwargs)
                    )
                return result

        return [self._execute_remote_single_worker(worker, method_name, *args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        # [FUNCTION DOC]
        # Outputs: master address.
        return self._master_addr

    @property
    def master_port(self):
        # [FUNCTION DOC]
        # Outputs: master port.
        return self._master_port

    @property
    def workers(self):
        # [FUNCTION DOC]
        # Outputs: list of worker handles.
        return self._workers

    @property
    def world_size(self):
        # [FUNCTION DOC]
        # Outputs: world size.
        return self._world_size


"""
Utilities that enables creating workers inside the same ray.Actor,
with code written in separate ray.Actors.
"""


# deprecated, switching to FusedWorker
def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    # [FUNCTION DOC]
    # HELPER (deprecated): Behavior: for each register-decorated method of user_defined_cls, create a forwarding function
    # (sync or async matching the original) that calls the same method on the inner worker stored under key, copy the
    # register metadata onto it, and attach it to cls: DIRECT_ROLLOUT_METHOD methods of roles whose key contains rollout
    # are attached unprefixed (asserting no name conflict), others as key + underscore + method name.
    # Exceptions: ValueError if attaching fails (including the conflict assertion). Side effects: mutates cls; prints.
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name, key=key):
                # [FUNCTION DOC]
                # HELPER: builds the forwarding function for method name (key bound by default arg).
                def func(self, *args, **kwargs):
                    # [FUNCTION DOC]
                    # HELPER: sync forwarder to the inner worker method.
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                async def async_func(self, *args, **kwargs):
                    # [FUNCTION DOC]
                    # HELPER: async forwarder awaiting the inner worker method.
                    # dispatch to the actual worker
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                return wrapper

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR)
            setattr(func, MAGIC_ATTR, attrs)
            try:
                # bind direct rollout method to class without prefix
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    assert not hasattr(cls, method_name), (
                        f"conflict direct rollout method {method_name} with role {key}"
                    )
                    setattr(cls, method_name, func)
                    print(f"bind role {key} method {method_name} to class {cls}")
                else:
                    method_name_with_prefix = key + "_" + method_name
                    setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def _unwrap_ray_remote(cls):
    # [FUNCTION DOC]
    # HELPER: Outputs: the underlying Python class of a Ray remote class (via its actor class attribute), or the input.
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def _determine_fsdp_megatron_base_class(mros: list):
    # [FUNCTION DOC]
    # HELPER: Behavior: walk the MRO of the first class only and return the first class named MegatronWorker or Worker.
    # Exceptions: ValueError if neither appears.
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker":
            return cls
        if cls.__name__ == "Worker":
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


# deprecated, switching to FusedWorker
def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # [FUNCTION DOC]
    # Deprecated colocation factory. Behavior: determine base worker class, record each role's class and init args, define a
    # WorkerDict subclass whose init instantiates each underlying class locally (with DISABLE_WORKER_INIT=1) into worker_dict,
    # bind forwarding methods for each role, and wrap WorkerDict as a Ray remote class in RayClassWithInitArgs.
    # Inputs: class_dict - role -> RayClassWithInitArgs. Outputs: RayClassWithInitArgs of the combined actor.
    # Exceptions: AssertionError if base is not a Worker subclass; ValueError from base detection.
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_fsdp_megatron_base_class(
        [cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()]
    )
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    print(f"colocated worker base class {worker_cls}")

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):
        # [CLASS DOC]
        # Combined actor holding one locally constructed worker per role in worker_dict.
        def __init__(self):
            # [FUNCTION DOC]
            # Behavior: initialize base worker, then construct each role's raw class with its stored args under DISABLE_WORKER_INIT=1.
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls


FusedWorkerCLSName = "FusedWorker"


def create_colocated_worker_raw_cls(class_dict: dict[str, RayClassWithInitArgs]):
    # [FUNCTION DOC]
    # Behavior: builds a FusedWorker class (dynamically renamed FusedWorker_<roles joined by underscore>) from unwrapped role
    # classes and their init args. Its instances construct every role class (with DISABLE_WORKER_INIT=1, after patching each
    # role class's Ray actor-name and method-prefix getters to the fused class name and role prefix), expose each as an
    # attribute named by role and in fused_worker_dict, and give every sub-worker a reference to that mapping. The class gets
    # is_fused_worker True and raw_cls_dict.
    # Inputs: class_dict - role -> RayClassWithInitArgs. Outputs: the renamed class (not Ray-wrapped).
    # Side effects: the role classes are patched globally when an instance is constructed.
    """
    This function returns a FusedWorker class.

    `FusedWorker.{class_name}` -> FusedClass
        Use `class_name` as a param to directly access the underlying class.

    `FusedWorker._fuw_execute("{class_name}_fwmn_{method_name}", *args, **kwargs)`
        First param must be "{class_name}_fwmn_{method_name}" in order to access `method_name`
        of underlying class `{class_name}`.

    `FusedWorker.fused_worker_dict` -> {"class_name": FusedClass}
        Stores all underlying classes.

    `FusedClass.fused_worker_dict` -> {"class_name": FusedClass}
        The same as `FusedWorker.fused_worker_dict`, enables underlying class to access other
        underlying classes.
    """
    raw_cls_dict = {cls_name: _unwrap_ray_remote(cia.cls) for cls_name, cia in class_dict.items()}
    init_args_dict = {cls_name: cia.args for cls_name, cia in class_dict.items()}
    init_kwargs_dict = {cls_name: cia.kwargs for cls_name, cia in class_dict.items()}
    cls_names = list(class_dict.keys())

    # FusedWorker_Actor_Critic
    class_name_renamed = "_".join([FusedWorkerCLSName] + cls_names)

    class FusedWorker(Worker):
        # [CLASS DOC]
        # Worker hosting multiple role workers in a single process; routes calls via _fuw_execute.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC]
            # Behavior: base Worker init with given args, then build sub-workers as described in create_colocated_worker_raw_cls.
            super().__init__(*args, **kwargs)
            self.cls_names = cls_names
            self.raw_cls_dict = raw_cls_dict
            self.init_args_dict = init_args_dict
            self.init_kwargs_dict = init_kwargs_dict

            for cls_name, udc, ud_args, ud_kwargs in zip(
                self.cls_names,
                self.raw_cls_dict.values(),
                self.init_args_dict.values(),
                self.init_kwargs_dict.values(),
                strict=True,
            ):
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    udc._get_ray_actor_cls_name = lambda x, name_renamed=class_name_renamed: name_renamed
                    udc._get_ray_method_prefix = lambda x, name_prefixed=cls_name: f"{name_prefixed}_"
                    # cls_name = "actor", "critic", udc = ActorWorker, CriticWorker
                    self.fused_worker_dict[cls_name] = udc(*ud_args, **ud_kwargs)
                    setattr(self, cls_name, self.fused_worker_dict[cls_name])

            # injecting fused_worker to each sub worker so they can be aware of existence of each other
            for _, worker in self.fused_worker_dict.items():
                setattr(worker, Worker.fused_worker_attr_name, self.fused_worker_dict)

        def _fuw_execute(self, method_name: str, *args, **kwargs):
            # [FUNCTION DOC]
            # Behavior: split the encoded name on _fwmn_ into role and method name, assert role exists, call the method on that
            # sub-worker with the arguments. Outputs: the method's result. Exceptions: AssertionError for unknown role.
            # for fused_worker, method_name is in a form of "{cls_name}_fwmn_{method_name}"
            # where fwmn stands "fused worker method name"
            names = method_name.split("_fwmn_")
            cls_name = names[0]
            method_name = names[1]

            assert cls_name in self.fused_worker_dict, (
                f"calling {cls_name}'s {method_name}, but {cls_name} not in fused_worker_dict"
            )
            udc_method = getattr(self.fused_worker_dict[cls_name], method_name)
            return udc_method(*args, **kwargs)

    renamed_fused_worker_cls = type(class_name_renamed, (FusedWorker,), {})
    renamed_fused_worker_cls.is_fused_worker = True
    renamed_fused_worker_cls.raw_cls_dict = raw_cls_dict

    return renamed_fused_worker_cls


def create_colocated_worker_cls_fused(class_dict: dict[str, RayClassWithInitArgs]):
    # [FUNCTION DOC]
    # Behavior: build the fused raw class, wrap with ray.remote, wrap in RayClassWithInitArgs with fused_worker_used True.
    # Outputs: RayClassWithInitArgs. Groups built from it support spawn/fuse per role.
    """
    This function returns a RayClassWithInitArgs instance of FusedWorker, which is an replacement
    of `create_colocated_worker_cls`. WorkerGroup constructed using this class will be a colocated
    WorkerGroup, which will be referenced as `ColocateWorkerGroup` below.

    `ColocateWorkerGroup.spawn(prefix_set)`
        returns a dict of WorkerGroup {"class_name": WorkerGroup}, WorkerGroup in this dict will
        have methods of underlying class `class_name` attached.

    `ColocateWorkerGroup.fuse(prefix_set)`
        After executing this function, `ColocateWorkerGroup.{class_name}` will return WorkerGroup
        with methods of underlying class `class_name` attached.
    """
    raw_colocated_worker_cls = create_colocated_worker_raw_cls(class_dict)

    remote_cls = ray.remote(raw_colocated_worker_cls)
    cia = RayClassWithInitArgs(cls=remote_cls)
    cia.fused_worker_used = True

    return cia
