# Copyright 2025 Meituan Ltd. and/or its affiliates
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
# Rollout side of fully-async PPO. Defines a server manager that handles hybrid (trainer-GPU, normally sleeping)
# and standalone rollout replicas, a single-sample agent loop manager, and the FullyAsyncRollouter Ray actor
# that streams prompts through generation and pushes each finished sample into the message queue, pausing
# when the queue is full or staleness exceeds its budget.
# Side effects on import: creates a module logger whose level comes from VERL_LOGGING_LEVEL (default WARN).

import asyncio
import collections
import logging
import os
import time
from pprint import pformat

import numpy as np
import ray
import torch
from omegaconf import DictConfig, open_dict

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.fully_async_policy.detach_utils import (
    RolloutSample,
    prepare_single_generation_data,
    safe_create_task,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.utils import (
    create_rl_dataset,
    create_rl_sampler,
    need_reward_model,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.skip import SkipManager
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient, LLMServerManager
from verl.workers.rollout.replica import RolloutReplica
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FullyAsyncLLMServerManager(LLMServerManager):
    # [CLASS DOC]
    # Server manager that keeps three extra structures: all registered hybrid replicas keyed by resource id
    # ("hybrid_<i>"), the currently active subset, and the active subset's server addresses; plus Prometheus
    # addresses and last add/remove timestamps. Hybrid replicas are added/removed from the global load balancer
    # at runtime.
    """Extension of :class:`LLMServerManager` for fully async training with hybrid scheduling."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        # [FUNCTION DOC]
        # Behavior: calls parent init, then creates empty hybrid/alive/address mappings, empty Prometheus address list,
        # and zero timestamps.
        # Inputs: config; optional worker_group (trainer worker group backing hybrid replicas); optional rollout pool.
        # Outputs: none.
        super().__init__(config, worker_group, rollout_resource_pool)
        # Pre-registered hybrid replicas: bound at init time but still sleeping.
        # Keyed by resource_id; populated during _initialize_llm_servers().
        self.hybrid_replicas: dict[str, RolloutReplica] = {}
        # Currently active (awake + in LB) subset of hybrid replicas.
        self.alive_replicas: dict[str, RolloutReplica] = {}
        # resource_id → server_address for alive hybrid replicas.
        self.alive_addresses: dict[str, str] = {}
        # Prometheus server addresses
        self.prometheus_server_addresses = []

        # Timing / counters
        self.last_hybrid_add_time: float = 0.0
        self.last_hybrid_remove_time: float = 0.0

    async def _initialize_llm_servers(self, start_rank: int = 0):
        # [FUNCTION DOC]
        # HELPER: Two-phase server initialization overriding the parent.
        # Behavior: Phase 1 (only if a worker group exists): run the parent init from rank 0 to create hybrid replicas,
        # move every created replica into the hybrid mapping as "hybrid_<index>", remember their addresses for
        # Prometheus, then clear the parent's replica/handle/address lists. Phase 2: temporarily set worker_group to none
        # so the parent takes its standalone path, starting at rank equal to the hybrid count; if
        # standalone_gpu_memory_utilization and gpu_memory_utilization are both configured, temporarily override the
        # latter with the former; always restore both in a finally block. Then, if Prometheus is enabled, require log
        # stats to be enabled and update the Prometheus config with hybrid plus standalone addresses. Logs counts.
        # Inputs: start_rank - ignored (phases choose their own ranks). Outputs: none.
        # Exceptions: ValueError when Prometheus is enabled but disable_log_stats is true; parent init errors.
        # Side effects: creates Ray server actors; mutates rollout config temporarily; writes Prometheus config.
        # ── Step 1: hybrid replicas first (replica_rank 0 … N_e-1) ──────────
        # Use parent class to create + init_hybrid all hybrid replicas, then
        # migrate them from rollup_replicas → hybrid_replicas (sleeping, not
        # yet in the load balancer).  Starting from rank 0 gives hybrid actors
        # the lowest-numbered placement-group bundles which are co-located with
        # the training engine, maximising GPU affinity on multi-node deployments.
        num_hybrid = 0
        if self.worker_group is not None:
            await super()._initialize_llm_servers(start_rank=0)
            num_hybrid = len(self.rollout_replicas)
            # Migrate hybrid replicas out of the parent's tracking lists.
            for i, replica in enumerate(self.rollout_replicas):
                resource_id = f"hybrid_{i}"
                self.hybrid_replicas[resource_id] = replica
                print(
                    f"[FullyAsyncAgentLoopManager] Hybrid replica '{resource_id}' "
                    f"(rank={i}) initialised at {replica._server_address} "
                )
            self.prometheus_server_addresses.extend(self.server_addresses)
            print(f"AgentLoopManager Hybrid: {self.server_addresses}")
            # Clear parent state so Step 2 starts clean.
            self.rollout_replicas = []
            self.server_handles = []
            self.server_addresses = []

        # ── Step 2: standalone replicas via parent class ─────────────────────
        # Temporarily clear worker_group so that super()._initialize_llm_servers()
        # takes the standalone branch (init_standalone).  Pass start_rank=num_hybrid
        # so that Ray actor names remain globally unique and never collide with the
        # hybrid actors created above.
        #
        # If standalone_gpu_memory_utilization is configured, temporarily override
        # gpu_memory_utilization so that standalone replicas can use a higher value
        # (e.g. 0.85) than hybrid replicas (which must share GPU with the training
        # engine).  The original value is restored in the finally block regardless
        # of whether the initialisation succeeds or fails.
        standalone_gmu = self.rollout_config.get("standalone_gpu_memory_utilization", None)
        original_gmu = self.rollout_config.get("gpu_memory_utilization", None)
        saved_worker_group = self.worker_group
        self.worker_group = None
        try:
            if standalone_gmu is not None and original_gmu is not None:
                with open_dict(self.rollout_config):
                    self.rollout_config.gpu_memory_utilization = standalone_gmu
            await super()._initialize_llm_servers(start_rank=num_hybrid)
        finally:
            # Always restore the original value to avoid affecting downstream logic.
            if standalone_gmu is not None and original_gmu is not None:
                with open_dict(self.rollout_config):
                    self.rollout_config.gpu_memory_utilization = original_gmu
            self.worker_group = saved_worker_group

        # Update Prometheus with the final (standalone) addresses.
        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            all_addresses = self.prometheus_server_addresses + self.server_addresses
            update_prometheus_config(self.rollout_config.prometheus, all_addresses, self.rollout_config.name)

        print(
            f"[FullyAsyncLLMServerManager] Created: "
            f"{len(self.rollout_replicas)} standalone replicas (rank {num_hybrid}+), "
            f"{num_hybrid} hybrid replicas registered (sleeping, rank 0-{num_hybrid - 1})"
        )

    async def add_replicas(self, resource_ids: list[str]) -> int:
        # [FUNCTION DOC]
        # Behavior: for each requested id, skip (with warning) if already active, skip (with error log) if not
        # registered; collect address-to-handle pairs. If none, return 0. Otherwise one RPC to the global load balancer
        # to add all servers, then for each valid id append its handle/address to the server lists (if not present),
        # append the replica to rollout_replicas (if not present), and record it in alive mappings; update last add time.
        # Inputs: resource_ids - list of hybrid ids. Outputs: count added (0 on failure).
        # Exceptions: any exception from the RPC/tracking is caught, logged, and 0 returned.
        # Side effects: load balancer and local state mutation; prints.
        # KNOWN ISSUE: active hybrid replicas are appended to rollout_replicas, contradicting get_standalone_replicas'
        # claim that rollout_replicas holds only standalone replicas and causing get_active_server_count to double-count.
        """Activate multiple pre-registered hybrid replicas in a single batch RPC.

        Uses ``batch_add_servers`` on the GlobalRequestLoadBalancer for atomic
        bulk registration, which is more efficient than calling :meth:`add_replica`
        in a loop.

        Args:
            resource_ids: List of resource identifiers to activate.

        Returns:
            Number of successfully activated replicas.
        """
        # Filter out already-active and missing replicas.
        servers_to_add: dict[str, ray.actor.ActorHandle] = {}
        valid_resource_ids: list[str] = []
        for rid in resource_ids:
            if rid in self.alive_replicas:
                logger.warning("[FullyAsyncLLMServerManager] Replica '%s' already active, skipping", rid)
                continue
            replica = self.hybrid_replicas.get(rid)
            if replica is None:
                logger.error(
                    "[FullyAsyncLLMServerManager] Replica '%s' is not registered, skipping",
                    rid,
                )
                continue
            servers_to_add[replica._server_address] = replica._server_handle
            valid_resource_ids.append(rid)

        if not servers_to_add:
            return 0

        try:
            # Single atomic batch RPC: register all handles + add all to LB pool.
            await self.global_load_balancer.add_servers.remote(servers=servers_to_add)

            # Track locally for introspection / Prometheus.
            for rid in valid_resource_ids:
                replica = self.hybrid_replicas[rid]
                server_address = replica._server_address
                server_handle = replica._server_handle
                if server_address not in self.server_addresses:
                    self.server_handles.append(server_handle)
                    self.server_addresses.append(server_address)
                if replica not in self.rollout_replicas:
                    self.rollout_replicas.append(replica)
                self.alive_replicas[rid] = replica
                self.alive_addresses[rid] = server_address

            self.last_hybrid_add_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] added {len(valid_resource_ids)} replicas: {valid_resource_ids}. "
                f"Active hybrid replicas ({len(self.alive_replicas)}): {list(self.alive_replicas.keys())}"
            )
            return len(valid_resource_ids)

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to batch activate replicas: %s", e)
            return 0

    async def remove_replicas(self, resource_ids: list[str]) -> int:
        # [FUNCTION DOC]
        # Behavior: for each id, skip with warning if not active; collect its address. If none, return 0. Otherwise one RPC
        # to remove those server ids from the load balancer, then for each valid id remove its address and the handle at
        # the same position from the server lists, remove the replica from rollout_replicas, and drop it from the alive
        # mappings; update last remove time.
        # Inputs: resource_ids. Outputs: count removed (0 on failure).
        # Exceptions: caught and logged, returning 0. Side effects: load balancer and local state mutation; prints.
        """Deactivate multiple active hybrid replicas in a single batch RPC.

        Uses ``batch_remove_servers`` on the GlobalRequestLoadBalancer for atomic
        bulk removal, which is more efficient than calling :meth:`remove_replica`
        in a loop.

        Args:
            resource_ids: List of resource identifiers to deactivate.

        Returns:
            Number of successfully deactivated replicas.
        """
        # Filter out missing replicas and collect server addresses.
        server_ids_to_remove: list[str] = []
        valid_resource_ids: list[str] = []
        for rid in resource_ids:
            if rid not in self.alive_replicas:
                logger.warning("[FullyAsyncLLMServerManager] Replica '%s' not active, skipping", rid)
                continue
            server_ids_to_remove.append(self.alive_addresses[rid])
            valid_resource_ids.append(rid)

        if not server_ids_to_remove:
            return 0

        try:
            # Single atomic batch RPC: remove all from LB pool + purge handles.
            await self.global_load_balancer.remove_servers.remote(server_ids=server_ids_to_remove)

            # Clean up local tracking lists.
            for rid in valid_resource_ids:
                server_address = self.alive_addresses[rid]
                replica = self.alive_replicas[rid]
                if server_address in self.server_addresses:
                    idx = self.server_addresses.index(server_address)
                    self.server_addresses.pop(idx)
                    self.server_handles.pop(idx)
                if replica in self.rollout_replicas:
                    self.rollout_replicas.remove(replica)
                self.alive_replicas.pop(rid)
                self.alive_addresses.pop(rid)

            self.last_hybrid_remove_time = time.time()

            print(
                f"[FullyAsyncLLMServerManager] removed {len(valid_resource_ids)} replicas: {valid_resource_ids}. "
                f"Remaining hybrid replicas ({len(self.alive_replicas)}): {list(self.alive_replicas.keys())}"
            )
            return len(valid_resource_ids)

        except Exception as e:
            logger.error("[FullyAsyncLLMServerManager] Failed to batch remove replicas: %s", e)
            return 0

    # -------------------------------------------------------------------------
    # Statistics / introspection
    # -------------------------------------------------------------------------
    def get_num_hybrid_replicas(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns the number of active hybrid replicas. Outputs: int.
        """Return the number of currently active hybrid replicas."""
        return len(self.alive_replicas)

    def get_hybrid_replicas_info(self) -> list[dict]:
        # [FUNCTION DOC]
        # Behavior: returns a list of records with resource_id and server_address for each active hybrid replica.
        """Return metadata for all active hybrid replicas."""
        return [{"resource_id": rid, "server_address": addr} for rid, addr in self.alive_addresses.items()]

    def get_hybrid_statistics(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns a mapping with active hybrid count and last add/remove timestamps under "hybrid/" keys.
        """Return hybrid-specific counters for monitoring."""
        return {
            "hybrid/num_hybrid_replicas": len(self.alive_replicas),
            "hybrid/last_add_time": self.last_hybrid_add_time,
            "hybrid/last_remove_time": self.last_hybrid_remove_time,
        }

    def get_active_server_count(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns length of rollout_replicas plus number of active hybrid replicas. Outputs: int.
        # KNOWN ISSUE: add_replicas also puts active hybrid replicas into rollout_replicas, so they are counted twice.
        """Total active rollout servers (standalone + hybrid)."""
        return len(self.rollout_replicas) + len(self.alive_replicas)

    def get_standalone_replicas(self) -> list:
        # [FUNCTION DOC]
        # Behavior: returns rollout_replicas as-is. Outputs: list.
        # KNOWN ISSUE: after add_replicas this list also contains active hybrid replicas, contrary to the docstring.
        """Return standalone-only replicas (hybrid replicas excluded).

        ``rollout_replicas`` only ever holds standalone replicas; hybrid
        replicas live separately in ``alive_replicas``/``hybrid_replicas``.
        """
        return self.rollout_replicas

    def get_client(self, client_cls=FullyAsyncLLMServerClient, **kwargs) -> LLMServerClient:
        # [FUNCTION DOC]
        # Behavior: computes only_hybrid as "rollout_replicas is empty" and delegates to the parent get_client with that
        # flag, default client class FullyAsyncLLMServerClient, and extra kwargs. Outputs: an LLM server client.
        """Override to automatically inject ``only_hybrid`` into the client.

        When there are no standalone replicas, hybrid replicas are the sole
        rollout resource.  During weight-sync windows the load balancer is
        temporarily empty, so the client should keep retrying instead of
        raising immediately. Defaults to :class:`FullyAsyncLLMServerClient` since
        ``only_hybrid`` retry support is only implemented there.
        """
        only_hybrid = len(self.rollout_replicas) == 0
        return super().get_client(client_cls=client_cls, only_hybrid=only_hybrid, **kwargs)


class FullyAsyncAgentLoopManager(AgentLoopManager):
    # [CLASS DOC]
    # Agent loop manager variant that generates one sample at a time by dispatching to agent loop workers in
    # round-robin order.
    @SkipManager.annotate(role="async_rollout")
    async def generate_sequences_single(self, prompts: DataProto) -> DataProto:
        # [FUNCTION DOC]
        # Behavior: picks the next worker round-robin and awaits its remote generate_sequences call (bridging the Ray
        # future into asyncio). Inputs: prompts - single-sample DataProto. Outputs: generated DataProto.
        # Exceptions: propagates remote errors. Decorated for skip-manager "async_rollout" role.
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch. Single sample data
        Returns:
            DataProto: Output batch.
        """
        worker = self._select_best_worker()
        output_future = worker.generate_sequences.remote(prompts)
        return await asyncio.wrap_future(output_future.future())

    def _select_best_worker(self):
        # [FUNCTION DOC]
        # HELPER: Lazily initializes a worker index to 0, returns the worker at that index, and advances it modulo the
        # number of workers. Side effects: mutates the index attribute.
        """Select the best worker, simple round-robin load balancing"""
        if not hasattr(self, "_worker_index"):
            self._worker_index = 0

        worker = self.agent_loop_workers[self._worker_index]
        self._worker_index = (self._worker_index + 1) % len(self.agent_loop_workers)
        return worker


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(SeparateRayPPOTrainer):
    # [CLASS DOC]
    # Ray actor (10 CPUs, up to 100 concurrent calls) that continuously generates samples. Reuses the separated
    # PPO trainer for dataset/worker utilities. Key state: dataloader, staleness_samples (samples generated or queued
    # since last parameter update), max_required_samples budget, max_concurrent_samples, pending queue of prompts,
    # set of active generation tasks, pause/resume event, per-step counters and a history of (time, active count,
    # capacity) used for resource utilization metrics.
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor=None,
        device_name=None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores tokenizer/processor/config; asserts no hybrid engine, train_batch_size 0, gen_batch_size 1,
        # staleness_threshold at least 0, trigger_parameter_sync_step at least 1; determines reward model need and, if
        # needed, asserts the RM uses its own resource pool; disables reference policy and critic; sets device; creates
        # a validation generations logger; reads prefix-grouper flag; initializes dump executor; builds train/val
        # datasets, sampler, validates config, creates dataloaders; total_rollout_steps is dataloader length times
        # epochs, capped by rollout.total_rollout_steps if set; required_samples is ppo_mini_batch_size times
        # require_batches; initializes counters, global_steps 1, completed steps 1, timing, pause/running flags,
        # dataloader lock, pending queue (capacity 128), active task set and the utilization history seeded with a
        # zero-capacity point.
        # Inputs: config; tokenizer; optional processor; optional device_name.
        # Exceptions: AssertionError for invalid config; ValueError from config validation.
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger or equal than 1"
        )

        self.use_reference_policy = False

        self.use_rm = need_reward_model(self.config)
        if self.use_rm:
            assert self.config.reward.reward_model.enable_resource_pool, (
                "GenRM/DisRM in fully async mode requires standalone mode (enable_resource_pool=True). "
                "Colocate mode is not supported because async rollout never pauses."
            )

        self.use_critic = False
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._init_dump_executor()

        # ==================== fully async config ====================

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        self.async_rollout_manager = None

        # Elastic worker group (injected via set_hybrid_worker_group before init_workers)
        # When set, its GPUs back hybrid replicas for trainer-side validation.
        self._hybrid_worker_group = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.concurrent_samples_per_replica: int = config.async_training.get("concurrent_samples_per_replica", 16)
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # Per-step sample counter: counts fully-generated samples in the current param version.
        # Reset to 0 at each reset_staleness() call.
        self._step_generated_samples: int = 0
        # Rolling history of per-step sample counts: (param_version, sample_count).
        # Keeps the most recent _STEP_HISTORY_SIZE entries for throughput diff analysis.
        self._STEP_HISTORY_SIZE: int = 10
        self._step_samples_history: collections.deque = collections.deque(maxlen=self._STEP_HISTORY_SIZE)
        # Monotonically increasing counter of completed param-sync steps.
        # Must NOT be capped by _STEP_HISTORY_SIZE; used by get_completed_steps() to
        # compute expected_samples in the trainer, which grows without bound.
        self._completed_steps: int = 1
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = time.time()
        self.step_start_time = time.time()

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Initialize async queues
        self.pending_queue = asyncio.Queue(maxsize=128)
        self.active_tasks = set()

        # Event-driven history of (len(self.active_tasks), max_concurrent_samples) over time,
        # used to compute dynamic_resource/rollout_resource_utilization in reset_staleness().
        # Each entry is (timestamp, active_count_after_change, max_concurrent_at_that_time),
        # appended whenever active_tasks' size changes (a sample is submitted, completes, or is
        # drained during a pause) OR when max_concurrent_samples itself changes (replicas
        # added/removed under dynamic resource scheduling). Recording the capacity alongside the
        # active count is required because max_concurrent_samples is NOT constant over a step
        # window when rollout resources are dynamically (de)activated — using a single
        # end-of-window capacity for the whole window would misattribute utilization for
        # intervals that had a different capacity. Seeded with a single (t0, 0, 0) point so the
        # very first interval has a well-defined start; the capacity will be corrected to the
        # real value as soon as max_concurrent_samples is known (see set_max_required_samples).
        self._active_count_history: list[tuple[float, int, int]] = [(time.time(), 0, 0)]

    def _init_async_objects(self):
        # [FUNCTION DOC]
        # HELPER: Creates the state lock and the resume event (initially set). Must run inside the actor's event loop.
        # Initialize asyncio synchronization primitives.
        # `lock` protects shared state: paused / active_tasks / staleness_samples / timing fields.
        self.lock = asyncio.Lock()
        # `_resume_event` signals that the rollouter is currently running (paused == False).
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        # [FUNCTION DOC]
        # Behavior: stores the message queue client under the lock. Outputs: none.
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        # [FUNCTION DOC]
        # Behavior: under the lock computes max_required_samples as required_samples times (staleness_threshold+1) times
        # trigger_parameter_sync_step; total_train_steps as total_rollout_steps divided by (required_samples times
        # trigger step), truncated; max_concurrent_samples as active server count times per-replica concurrency, capped
        # by max_required_samples; max_queue_size equal to max_required_samples. Logs, then if the utilization history
        # has only its seed point, replaces its capacity with the real max_concurrent_samples.
        # Outputs: none. Side effects: state mutation, print.
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = (
                self.llm_server_manager.get_active_server_count() * self.concurrent_samples_per_replica
            )
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

            # The initial seed point in _active_count_history was recorded in __init__ before
            # max_concurrent_samples was known (placeholder capacity 0). Now that the real
            # capacity is available, correct that seed point so the very first interval isn't
            # spuriously treated as having zero capacity.
            if len(self._active_count_history) == 1:
                t0, active0, _capacity0 = self._active_count_history[0]
                self._active_count_history[0] = (t0, active0, self.max_concurrent_samples)

    def get_replicas(self):
        # [FUNCTION DOC]
        # Behavior: returns the server manager's active replicas.
        """Get rollout worker group"""
        return self.llm_server_manager.get_replicas()

    def get_max_queue_size(self):
        # [FUNCTION DOC]
        # Behavior: returns max_queue_size (none before set_max_required_samples).
        return self.max_queue_size

    def get_total_train_steps(self):
        # [FUNCTION DOC]
        # Behavior: returns total_train_steps (none before set_max_required_samples).
        return self.total_train_steps

    def _compute_rollout_resource_utilization(self) -> float:
        # [FUNCTION DOC]
        # HELPER: Appends a closing point (now, active count, current capacity or 0) to the history, logs the raw
        # history, and computes a time- and capacity-weighted utilization: for each consecutive pair of points with
        # positive duration and positive capacity, add duration times capacity times min(1, active/capacity) to a
        # numerator and duration times capacity to a denominator. Returns numerator/denominator, or 0 when the
        # denominator is 0. Then resets the history to a single point (now, active count, capacity).
        # Outputs: float in [0,1]. Side effects: mutates history; prints.
        """Estimate rollout-resource utilization over the just-finished rollouter step
        (the window between the previous and current reset_staleness() call), using the
        event-driven (active_count, max_concurrent) history recorded by _record_active_count().

        Let t_0 < t_1 < ... < t_m be the timestamps in self._active_count_history (t_0 is
        the previous step's end / this step's start; t_m is "now", appended right below to
        close out the window — including any trailing drain/idle time before this call,
        e.g. from partial-rollout pause). Let A_k be the active_tasks size recorded at t_k and
        S_k be the max-concurrency capacity (max_concurrent_samples) recorded at t_k; both hold
        for the whole interval [t_k, t_{k+1}).

        Under dynamic resource scheduling, replicas can be activated/deactivated mid-window, so the
        capacity S_k is NOT constant across the window — each interval must use its own S_k
        (the capacity that was actually in effect during that interval), rather than a single
        end-of-window value. _record_active_count() is called both when active_tasks changes
        and whenever max_concurrent_samples changes (see _update_max_concurrent_samples), so
        every interval boundary reflects a change in at least one of A_k or S_k.

        For each interval k: u_k = min(1, A_k / S_k) (utilization can exceed 1 transiently
        under dynamic resource scheduling, so it is clamped). Intervals with S_k == 0 (no capacity
        available yet) contribute zero weight and are skipped.

        utilization = sum_k[(t_{k+1} - t_k) * S_k * u_k] / sum_k[(t_{k+1} - t_k) * S_k]

        Returns 0.0 if there is no time span to integrate over (e.g. total weight is 0).
        """
        # Close out the window with a final point at "now" so the trailing interval
        # (including any drain/idle time right before this reset_staleness() call) is
        # included in the integral.
        current_max_concurrent = self.max_concurrent_samples if self.max_concurrent_samples is not None else 0
        self._active_count_history.append((time.time(), len(self.active_tasks), current_max_concurrent))

        print(
            f"[FullyAsyncRollouter][RolloutResourceUtilization] raw active_count_history "
            f"(timestamp, active_count, max_concurrent): {self._active_count_history}"
        )

        numerator = 0.0
        denominator = 0.0
        for (t_k, active_k, capacity_k), (t_k1, _active_k1, _capacity_k1) in zip(
            self._active_count_history, self._active_count_history[1:], strict=False
        ):
            dt = t_k1 - t_k
            if dt <= 0 or capacity_k <= 0:
                continue
            u_k = min(1.0, active_k / capacity_k)
            numerator += dt * capacity_k * u_k
            denominator += dt * capacity_k

        utilization = numerator / denominator if denominator > 0 else 0.0

        # Reset history for the next step, seeded with the current active count/capacity so the
        # next window's first interval has a well-defined start.
        self._active_count_history = [(time.time(), len(self.active_tasks), current_max_concurrent)]

        return utilization

    async def reset_staleness(self):
        # [FUNCTION DOC]
        # Behavior: under the lock: unpauses and sets the resume event; sets staleness_samples to the number of active
        # tasks plus the current message queue size; computes version time (since step start, at least 1e-6), active
        # time (until idle start if idle began after step start, else whole window) and idle ratio; if any samples were
        # generated this step, increments completed steps and appends (completed steps, count) to the history; records
        # step sample count, resets it to 0; computes rollout resource utilization; logs; resets step start time.
        # Outputs: mapping of timing/metric names to floats.
        # Side effects: state mutation; remote queue-size call; prints.
        """
        Reset staleness samples after parameter update.
        Returns timing_raw dictionary for metrics.
        """
        async with self.lock:
            self.paused = False
            # Wake the drain loop in _processor_worker so it can exit early and resume submitting
            # new samples to idle replicas instead of waiting for long-tail in-flight tasks.
            self._resume_event.set()
            # every time param change, reset staleness_samples
            self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
            timing_raw = {}
            rollout_version_time = max(time.time() - self.step_start_time, 1e-6)
            if self.idle_start_time > self.step_start_time:
                rollout_active_time = self.idle_start_time - self.step_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
            else:
                rollout_active_time = rollout_version_time
                idle_ratio = 0
            timing_raw["fully_async/rollouter/active_time"] = rollout_active_time
            timing_raw["fully_async/rollouter/version_time"] = rollout_version_time
            timing_raw["fully_async/rollouter/idle_ratio"] = idle_ratio

            if self._step_generated_samples > 0:
                self._completed_steps += 1
                self._step_samples_history.append((self._completed_steps, self._step_generated_samples))
            timing_raw["fully_async/rollouter/step_generated_samples"] = self._step_generated_samples
            # Reset per-step counter for the next param version.
            self._step_generated_samples = 0

            timing_raw["dynamic_resource/rollout_resource_utilization"] = self._compute_rollout_resource_utilization()

            print(
                f"[FullyAsyncRollouter][Public][reset_staleness] "
                f"reset staleness_samples to: {self.staleness_samples} "
                f"idle_ratio: {timing_raw['fully_async/rollouter/idle_ratio']:.4f} "
                f"step_generated_samples(this_step): {timing_raw['fully_async/rollouter/step_generated_samples']} "
                f"rollout_resource_utilization: "
                f"{timing_raw['dynamic_resource/rollout_resource_utilization']:.4f} "
                f"recent_history(last {self._STEP_HISTORY_SIZE} steps): {list(self._step_samples_history)}"
            )
            self.step_start_time = time.time()

        return timing_raw

    async def _start_profiling(self):
        # [FUNCTION DOC]
        # HELPER: Awaits start of profiling on all replicas via the server manager.
        """Start rollout profiling on all replicas via LLMServerManager after weight sync."""
        await self.llm_server_manager.start_profile()

    async def _stop_profiling(self):
        # [FUNCTION DOC]
        # HELPER: Awaits stop of profiling on all replicas via the server manager.
        """Stop rollout profiling on all replicas before the next weight sync."""
        await self.llm_server_manager.stop_profile()

    def do_validate(self):
        # [FUNCTION DOC]
        # Behavior: runs the inherited validation inside a timer and returns the merged timing and validation metrics.
        # Outputs: mapping of metric names to values.
        """Run validation and return metrics"""
        timing_raw = {}
        with marked_timer("rollouter/validate_time", timing_raw, color="green"):
            val_metrics: dict = self._validate()
        return timing_raw | val_metrics

    async def save_checkpoint(self, local_global_step_folder: str):
        # [FUNCTION DOC]
        # Behavior: creates the folder, snapshots the train dataloader state under the dataloader lock, and saves it
        # with torch to "data.pt" in the folder. In-flight samples are not saved (they will be lost on resume).
        # Inputs: local_global_step_folder - path. Outputs: none. Side effects: filesystem write, print.
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        # [FUNCTION DOC]
        # Behavior: returns 0 when resume_mode is "disable". Raises if an HDFS dir is configured. Otherwise finds the latest
        # checkpoint under default_local_dir (made absolute). In "auto" mode returns 0 if none found; in "resume_path"
        # mode asserts the path is a string containing "global_step_" and uses it (made absolute); other modes raise.
        # Parses the trainer step from the folder name and sets global_steps to that step times required_samples times
        # trigger step, plus 1. Loads the dataloader state from "data.pt" if present, else warns.
        # Outputs: 0 on early exits, otherwise none.
        # Exceptions: NotImplementedError for HDFS; AssertionError for bad resume path; ValueError for unknown mode.
        # Side effects: mutates global_steps and dataloader state; reads files; prints.
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # [FUNCTION DOC]
        # HELPER: Raises ValueError if async_training config is missing; asserts rollout calculate_log_probs is enabled.
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        # [FUNCTION DOC]
        # Behavior: creates async primitives, worker classes, reward loop manager, optional teacher model manager,
        # the async rollout manager, and initializes the skip manager from config. Outputs: none.
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._create_worker_classes()
        await self._create_reward_loop_manager()
        await self._create_teacher_model_manager()
        await self._init_async_rollout_manager()
        SkipManager.init(self.config)

    async def _create_reward_loop_manager(self):
        # [FUNCTION DOC]
        # HELPER: Constructs a RewardLoopManager (with no RM resource pool) in a thread executor, since its constructor
        # runs its own event loop; stores it. Side effects: creates reward workers.
        """Create RewardLoopManager for the rollouter.

        TODO: RewardModelManager.__init__ uses asyncio.run() which forces us to use
        run_in_executor here. Upstream should provide an async init method so this
        can be a simple await call instead.
        """
        import asyncio

        from verl.experimental.reward_loop import RewardLoopManager

        loop = asyncio.get_running_loop()
        self.reward_loop_manager = await loop.run_in_executor(
            None,
            lambda: RewardLoopManager(config=self.config, rm_resource_pool=None),
        )

    async def _create_teacher_model_manager(self):
        # [FUNCTION DOC]
        # HELPER: Sets teacher manager to none; if distillation is enabled, builds a single "teacher_pool" resource pool
        # of n_gpus_per_node per node across nnodes, maps the TeacherModel role to it, creates it, and constructs a
        # MultiTeacherModelManager in a thread executor.
        # Exceptions: AssertionError if distillation n_gpus_per_node is not positive.
        # Side effects: allocates Ray resource pool and teacher workers.
        """Create MultiTeacherModelManager for distillation if enabled.

        Allocates a big resource pool for all teachers and passes it to
        MultiTeacherModelManager, which splits it internally per teacher.

        NOTE: MultiTeacherModelManager.__init__ calls _run_all internally which uses
        asyncio.run(), conflicting with the already-running event loop. Run in a thread executor.
        """
        from verl.trainer.distillation.losses import is_distillation_enabled
        from verl.trainer.ppo.utils import Role

        self.teacher_model_manager = None
        if is_distillation_enabled(self.config.get("distillation")):
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            resource_pool_spec = {}
            mapping = {}
            distillation_cfg = self.config.get("distillation", {})
            n_gpus = distillation_cfg.get("n_gpus_per_node", 0)
            nnodes = distillation_cfg.get("nnodes", 1)
            assert n_gpus > 0, "distillation.n_gpus_per_node must be greater than 0 for TeacherModel"
            teacher_pool = [n_gpus] * nnodes
            resource_pool_spec["teacher_pool"] = teacher_pool
            mapping[Role.TeacherModel] = "teacher_pool"

            resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
            resource_pool_manager.create_resource_pool()
            teacher_resource_pool = resource_pool_manager.get_resource_pool(Role.TeacherModel)

            loop = asyncio.get_running_loop()
            self.teacher_model_manager = await loop.run_in_executor(
                None,
                lambda: MultiTeacherModelManager(config=self.config, resource_pool=teacher_resource_pool),
            )

    def _create_actor_rollout_classes(self):
        # [FUNCTION DOC]
        # HELPER: No-op override; rollout workers are created by the agent loop instead.
        # Skip rollout creation and let agentloop handle it
        pass

    def _create_reward_model_class(self):
        # [FUNCTION DOC]
        # HELPER: No-op override; reward models are managed by the reward loop manager.
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _create_continuous_iterator(self):
        # [FUNCTION DOC]
        # HELPER: Generator yielding (epoch, batch) pairs over all batches for each configured epoch.
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # [FUNCTION DOC]
        # HELPER: Determines whether reward computation streams alongside rollout (no RM, or RM with its own pool) and
        # passes reward workers accordingly; asserts rollout mode is "async"; creates the FullyAsyncLLMServerManager with
        # the hybrid worker group, then the FullyAsyncAgentLoopManager with its client, reward worker handles and an
        # optional teacher client. Exceptions: AssertionError if rollout mode is not async.
        """
        Create the server manager and agent loop manager for fully async training.

        Uses :class:`FullyAsyncLLMServerManager` which supports two-phase init:
        - Phase 1: hybrid replicas on trainer GPUs (sleeping)
        - Phase 2: standalone replicas on rollout GPUs

        The ``GlobalRequestLoadBalancer`` (which also holds the server-handle
        registry) serves as the single source of truth for handle mapping and
        routing.  Clients look up handles atomically — no per-worker notification
        needed on hybrid add/remove.
        """
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        self.async_rollout_mode = True
        # Use FullyAsyncLLMServerManager for two-phase (hybrid + standalone) init.
        # It creates GlobalRequestLoadBalancer (with merged handle registry) internally.
        self.llm_server_manager = await FullyAsyncLLMServerManager.create(
            config=self.config,
            worker_group=self.get_hybrid_worker_group(),
        )
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        # [FUNCTION DOC]
        # HELPER: Producer coroutine. For each batch from the continuous iterator, prepares single-sample generation data,
        # wraps it in a RolloutSample with id "sample_<epoch>_<global_steps>" and empty status, and puts it in the pending
        # queue (blocking when full). Stops after the sample whose global_steps reached total_rollout_steps, otherwise
        # increments global_steps. Finally puts a none end signal and logs.
        continuous_iterator = self._create_continuous_iterator()

        for epoch, batch_dict in continuous_iterator:
            # Similar to _prepare_generate_batch: Separate data
            full_batch = prepare_single_generation_data(batch_dict, self.config)

            sample_id = f"sample_{epoch}_{self.global_steps}"

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                sample_id=sample_id,
                epoch=epoch,
                rollout_status={},
            )

            await self.pending_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples: "
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.pending_queue.put(None)
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    def _record_active_count(self):
        # [FUNCTION DOC]
        # HELPER: Appends (now, active task count, capacity or 0) to the utilization history.
        """Append current (len(self.active_tasks), max_concurrent_samples) to the history used
        for computing dynamic_resource/rollout_resource_utilization. Call this right after any
        change (increase or decrease) to self.active_tasks, AND whenever max_concurrent_samples
        changes (see _update_max_concurrent_samples), so each recorded interval carries the
        capacity that was actually in effect during it.
        """
        max_concurrent = self.max_concurrent_samples if self.max_concurrent_samples is not None else 0
        self._active_count_history.append((time.time(), len(self.active_tasks), max_concurrent))

    async def _processor_worker(self):
        # [FUNCTION DOC]
        # HELPER: Consumer coroutine. Loop: if paused or the pause condition holds, mark paused and clear the resume event,
        # then drain: repeatedly wait for any active task or the resume event; remove/await finished tasks under the lock
        # and record counts; break early on resume. If still not resumed, record idle start and wait for resume; cancel the
        # resume waiter on exit; restart loop. Otherwise take the next sample from the pending queue (marking task done)
        # and increment staleness_samples. On the end signal, wait for all active tasks to finish and exit. While active
        # tasks reach max_concurrent_samples, wait for at least one to finish. If paused, wait for resume. Then under the
        # lock create a task processing the sample (registered in the active set) and record counts.
        # Exceptions: propagates exceptions raised by awaited finished tasks.
        # KNOWN ISSUE: staleness_samples is also incremented for the none end signal.
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            if self.paused or await self._should_pause_generation():
                print(
                    "[FullyAsyncRollouter][Processor] Received pause signal, waiting for remaining tasks to return..."
                )
                async with self.lock:
                    self.paused = True
                    self._resume_event.clear()

                resume_future = asyncio.ensure_future(self._resume_event.wait())
                try:
                    # Drain: wait for either (a) at least one active task to finish, or
                    # (b) a resume signal (reset_staleness / monitor flipping paused=False) to
                    # break the drain early so new samples can be submitted to free replicas.
                    # We do NOT hold the lock during the wait, so publishers can acquire it to
                    # update paused / staleness_samples concurrently.
                    while self.active_tasks and not resume_future.done():
                        wait_set = set(self.active_tasks) | {resume_future}
                        done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
                        actual_done = done - {resume_future}
                        if actual_done:
                            async with self.lock:
                                for task in actual_done:
                                    self.active_tasks.discard(task)
                                    await task
                                self._record_active_count()
                        if resume_future in done:
                            print(
                                "[FullyAsyncRollouter][Processor] "
                                "Drain interrupted by resume signal, resuming generation early "
                                f"(active tasks remaining: {len(self.active_tasks)})"
                            )
                            break

                    # block until resuming
                    if not resume_future.done():
                        self.idle_start_time = time.time()
                        await resume_future
                finally:
                    if not resume_future.done():
                        resume_future.cancel()
                        await asyncio.gather(resume_future, return_exceptions=True)
                continue
            # Get sample from appropriate queue and immediately mark task as done
            rollout_sample = await self.pending_queue.get()
            self.pending_queue.task_done()
            self.staleness_samples += 1

            if rollout_sample is None:
                print(
                    "[FullyAsyncRollouter][Processor] Received end signal, waiting for remaining tasks to complete..."
                )
                while self.active_tasks:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done_tasks:
                                await task
                            self._record_active_count()
                break

            # Check whether the number of concurrent tasks exceeds the limit
            while len(self.active_tasks) >= self.max_concurrent_samples:
                async with self.lock:
                    if self.active_tasks:
                        done_tasks, self.active_tasks = await asyncio.wait(
                            self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done_tasks:
                            await task
                        self._record_active_count()

            # Submit single sample processing
            if self.paused:
                await self._resume_event.wait()
            async with self.lock:
                task = safe_create_task(
                    self._process_single_sample_streaming(rollout_sample),
                    name=rollout_sample.sample_id,
                    task_set=self.active_tasks,
                )
                self._record_active_count()

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        # [FUNCTION DOC]
        # HELPER: Sets the sample's uid array to "uid_<sample_id>" for each row, awaits single-sample generation,
        # replaces the batch with the output, re-sets the uid, attaches current statistics as rollout status, serializes
        # the sample with cloudpickle and puts it into the message queue. Increments generated counters on success or the
        # dropped counter on failure; always increments processed count.
        """Process a single sample streamingly"""
        # Calling asynchronous generation methods
        # Embed sample_id into prompts for skip management
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)

        rollout_sample.full_batch = ret
        # Re-set uid on output — agent loop worker returns a new DataProto without the input's non_tensor_batch
        rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
            [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
        )
        rollout_sample.rollout_status = await self.get_statistics()

        success = await self.message_queue_client.put_sample(
            sample=ray.cloudpickle.dumps(rollout_sample),
        )
        if success:
            self.total_generated_samples += 1
            self._step_generated_samples += 1
        else:
            self.dropped_stale_samples += 1
        self.processed_sample_count += 1

    async def _streaming_generation_main(self):
        # [FUNCTION DOC]
        # HELPER: Ensures the rollout manager exists, starts feed and processor tasks, waits for the first to finish;
        # re-raises its exception; raises RuntimeError if the processor ended before the feed. Then awaits the processor
        # and joins the pending queue. Finally cancels any unfinished tasks, clears references, sends a none sample to the
        # message queue as finish signal, and sets running false under the lock.
        # Exceptions: re-raises task exceptions and RuntimeError.
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = safe_create_task(self._feed_samples(), name="feed_task")
        self.processor_task = safe_create_task(self._processor_worker(), name="processor_task")

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

            await self.pending_queue.join()
            print("[FullyAsyncRollouter] pending_queue joined")

        except Exception as e:
            print(f"[FullyAsyncRollouter] Streaming process exception: {e}")
            raise e

        finally:
            if self.feed_task and not self.feed_task.done():
                self.feed_task.cancel()
                await asyncio.gather(self.feed_task, return_exceptions=True)

            if self.processor_task and not self.processor_task.done():
                self.processor_task.cancel()
                await asyncio.gather(self.processor_task, return_exceptions=True)

            self.feed_task = None
            self.processor_task = None

            # Send a finish signal
            await self.message_queue_client.put_sample(sample=None)

            async with self.lock:
                self.running = False

    async def fit(self):
        # [FUNCTION DOC]
        # Behavior: requires a message queue client (ValueError otherwise); under the lock clears pause, sets running and
        # the resume event; starts the generation main and monitor tasks and gathers them with exceptions captured; in
        # finally cancels any unfinished task and gathers again. Logs completion.
        # Outputs: none.
        # KNOWN ISSUE: because gather uses return_exceptions, errors from generation/monitor are swallowed silently and the
        # except branch is effectively unreachable.
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True
            self._resume_event.set()

        # Create the main asynchronous task
        generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
        monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        # [FUNCTION DOC]
        # HELPER: Every 10 s while running: every 60 s print statistics; if paused and the pause condition no longer holds,
        # unpause and set the resume event under the lock. Exits when running is false.
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.paused and not await self._should_pause_generation():
                async with self.lock:
                    self.paused = False
                    print("[FullyAsyncRollouter][ShouldPause] resume rollouter.")
                    self._resume_event.set()

    async def _should_pause_generation(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns True (logging only on transition) when message queue size is at least max_queue_size or
        # staleness_samples is at least max_required_samples; otherwise False. Side effects: remote stats call.
        """Determine whether the build should be paused"""
        queue_stats = await self.message_queue_client.get_statistics()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def get_statistics(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns a mapping of monitor, count and static statistics: active task count, pending queue size,
        # message queue size, generated/staleness/dropped counts, required and max samples, staleness threshold, max queue
        # size, max concurrency. Side effects: remote stats call.
        queue_stats = await self.message_queue_client.get_statistics()

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.pending_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            # counting stats
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats

    # -------------------------------------------------------------------------
    # Elastic worker group injection
    # -------------------------------------------------------------------------
    def set_hybrid_worker_group(self, worker_group: RayWorkerGroup):
        # [FUNCTION DOC]
        # Behavior: stores the worker group used for hybrid replicas (must be called before init_workers).
        """Inject the hybrid worker group."""
        self._hybrid_worker_group = worker_group

    def get_hybrid_worker_group(self):
        # [FUNCTION DOC]
        # Behavior: returns the stored hybrid worker group or none.
        """Return the worker group for hybrid replicas."""
        return self._hybrid_worker_group

    async def add_replicas(self, resource_ids: list[str]) -> int:
        # [FUNCTION DOC]
        # Behavior: delegates activation to the server manager; if any were added, recomputes max concurrency.
        # Outputs: number added.
        n = await self.llm_server_manager.add_replicas(resource_ids)
        if n > 0:
            self._update_max_concurrent_samples()
        return n

    async def remove_replicas(self, resource_ids: list[str]) -> int:
        # [FUNCTION DOC]
        # Behavior: delegates removal to the server manager; if any were removed, recomputes max concurrency.
        # Outputs: number removed.
        n = await self.llm_server_manager.remove_replicas(resource_ids)
        if n > 0:
            self._update_max_concurrent_samples()
        return n

    async def rebalance_requests(
        self, wait_inflight_timeout_s: float = 30.0, wait_inflight_poll_interval_s: float = 0.2
    ) -> dict:
        # [FUNCTION DOC]
        # Behavior: four steps: clear the load balancer's sticky routing cache (keep its diagnostics result); abort all
        # requests on all active replicas concurrently; poll the load balancer's total in-flight count every poll interval
        # until zero or timeout (log on timeout, no error); resume generation on the same replicas.
        # Inputs: wait_inflight_timeout_s; wait_inflight_poll_interval_s. Outputs: the sticky-cache diagnostics mapping
        # (with cleared_entries and server_loads). Side effects: aborts requests (clients retry), remote calls, prints.
        """Redistribute in-flight requests evenly across all active replicas.

        This performs a full rebalance cycle:

        1. **Clear sticky cache** — so subsequent ``acquire_server()`` calls use
           least-loaded selection rather than sticky routing.
        2. **Abort all replicas** — interrupt in-flight requests on all active
           replicas (standalone + hybrid).  :class:`FullyAsyncLLMServerClient`
           catches the abort and retries automatically.
        3. **Wait for in-flight requests to drain** — ``abort_all_requests()`` only
           waits for the rollout engine to acknowledge the abort; it does NOT wait
           for the client-side ``generate()`` call to receive the aborted output and
           run its ``finally: release_server(...)``.  ``release_server`` is a
           fire-and-forget RPC, so there is a real (if usually short) window where
           ``_inflight_requests`` has not yet dropped to 0 for the aborted requests.
           Poll :meth:`GlobalRequestLoadBalancer.get_total_inflight` until it reaches
           0 (or timeout) so Step 4's least-loaded routing is based on accurate counts.
        4. **Resume generation** — unpause all replicas so they accept retried
           requests, which are now routed via least-loaded selection to the
           replicas with the fewest in-flight requests (typically the newly
           activated hybrid replicas, which start at 0).

        Returns:
            Diagnostics dict from :meth:`GlobalRequestLoadBalancer.clear_sticky_cache`.
        """
        # Step 1: Clear sticky cache so retried requests use least-loaded routing.
        result = await self.llm_server_manager.global_load_balancer.clear_sticky_cache.remote()
        print(
            f"[FullyAsyncRollouter] Rebalance step 1/4: sticky cache cleared, "
            f"{result['cleared_entries']} entries, loads={result['server_loads']}"
        )

        # Step 2: Abort in-flight requests on all active replicas.
        active_replicas = self.llm_server_manager.get_replicas()
        if active_replicas:
            await asyncio.gather(*[replica.abort_all_requests() for replica in active_replicas])
            print(f"[FullyAsyncRollouter] Rebalance step 2/4: aborted requests on {len(active_replicas)} replicas")

        # Step 3: Wait until release_server() from all aborted requests has actually
        # landed on the load balancer, i.e. _inflight_requests has drained to 0.
        deadline = time.time() + wait_inflight_timeout_s
        total_inflight = await self.llm_server_manager.global_load_balancer.get_total_inflight.remote()
        while total_inflight > 0 and time.time() < deadline:
            await asyncio.sleep(wait_inflight_poll_interval_s)
            total_inflight = await self.llm_server_manager.global_load_balancer.get_total_inflight.remote()
        if total_inflight > 0:
            print(
                f"[FullyAsyncRollouter] Rebalance step 3/4: timed out waiting for in-flight requests to "
                f"drain, total_inflight={total_inflight} still outstanding after {wait_inflight_timeout_s}s"
            )
        else:
            print("[FullyAsyncRollouter] Rebalance step 3/4: all in-flight requests drained to 0")

        # Step 4: Resume generation so retried requests can be accepted.
        if active_replicas:
            await asyncio.gather(*[replica.resume_generation() for replica in active_replicas])
            print(f"[FullyAsyncRollouter] Rebalance step 4/4: resumed generation on {len(active_replicas)} replicas")

        return result

    def _update_max_concurrent_samples(self):
        # [FUNCTION DOC]
        # HELPER: If max_required_samples is unset, return. Otherwise set max_concurrent_samples to active replica count
        # times per-replica concurrency, capped by max_required_samples, log, and record a utilization history point.
        """Recompute max_concurrent_samples based on current active replica count.

        This is called whenever rollout resources are dynamically activated/deactivated
        (see add_replicas / remove_replicas). Since the capacity can change in the middle of a
        rollouter step window, we must record a new (timestamp, active_count, max_concurrent)
        point in _active_count_history right away — otherwise _compute_rollout_resource_utilization
        would incorrectly apply the OLD capacity to the time span between this change and the next
        active_tasks change.
        """
        if self.max_required_samples is None:
            return
        new_val = len(self.llm_server_manager.get_replicas()) * self.concurrent_samples_per_replica
        new_val = min(new_val, self.max_required_samples)
        print(
            f"[FullyAsyncRollouter] max_concurrent_samples updated: "
            f"{self.max_concurrent_samples} -> {new_val} "
            f"(active_replicas={len(self.llm_server_manager.get_replicas())})"
        )
        self.max_concurrent_samples = new_val
        self._record_active_count()

    def get_hybrid_replica(self, resource_id: str):
        # [FUNCTION DOC]
        # Behavior: returns the registered hybrid replica for an id, or none.
        """Return the RolloutReplica object for a registered hybrid resource."""
        return self.llm_server_manager.hybrid_replicas.get(resource_id)

    def get_all_hybrid_replicas(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns a shallow copy of the hybrid replica mapping (all registered, sleeping or active).
        """Return all registered hybrid replicas (sleeping + active)."""
        return dict(self.llm_server_manager.hybrid_replicas)

    # -------------------------------------------------------------------------
    # Statistics / introspection – delegate to llm_server_manager
    # -------------------------------------------------------------------------
    async def get_hybrid_statistics(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns the general statistics merged with hybrid statistics.
        """Combined rollout + hybrid statistics."""
        base_stats = await self.get_statistics()
        hybrid_stats = self.llm_server_manager.get_hybrid_statistics()
        return {**base_stats, **hybrid_stats}

    def get_num_active_replicas(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns the server manager's active server count.
        """Total active rollout replicas (standalone + active hybrid)."""
        return self.llm_server_manager.get_active_server_count()

    def get_standalone_replicas(self) -> list:
        # [FUNCTION DOC]
        # Behavior: returns the server manager's standalone replica list.
        """Return standalone-only replicas (hybrid replicas excluded)."""
        return self.llm_server_manager.get_standalone_replicas()

    def get_hybrid_replicas_info(self) -> list[dict]:
        # [FUNCTION DOC]
        # Behavior: returns the server manager's active hybrid replica metadata.
        """Metadata for all active hybrid replicas."""
        return self.llm_server_manager.get_hybrid_replicas_info()

    def get_total_produced_samples(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns total_generated_samples.
        """Total samples produced (uses base class counter)."""
        return self.total_generated_samples

    def get_completed_steps(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns the completed parameter-sync step counter.
        """Number of param-sync steps completed (monotonically increasing)."""
        return self._completed_steps

    async def get_sample_collection_ratio(self) -> float:
        # [FUNCTION DOC]
        # Behavior: returns message queue size divided by required_samples, logging the values. Outputs: float.
        """Return the fraction of required samples already collected for the current step."""
        queue_size = await self.message_queue_client.get_queue_size()
        ratio = queue_size / self.required_samples
        print(
            f"[FullyAsyncRollouter] get_sample_collection_ratio: "
            f"queue_size={queue_size}, required_samples={self.required_samples}, ratio={ratio:.4f}",
            flush=True,
        )
        return ratio

    async def wait_for_enough_samples(
        self,
        required_count: int,
        poll_interval: float = 1.0,
        timeout: float | None = None,
    ) -> int:
        # [FUNCTION DOC]
        # Behavior: polls the message queue size every poll_interval seconds; returns the size once it reaches
        # required_count; raises TimeoutError once elapsed time reaches timeout (if given).
        # Inputs: required_count; poll_interval; optional timeout. Outputs: observed queue size.
        # KNOWN ISSUE: docstring says a none required_count defaults to required_samples, but the code does not handle
        # none and the comparison would fail.
        """Block until the message queue contains at least ``required_count`` samples.

        Polls ``message_queue_client.get_queue_size()`` every ``poll_interval``
        seconds and returns only when the queue size reaches or exceeds
        ``required_count``.  If ``required_count`` is ``None``, defaults to
        ``self.required_samples``.

        The Trainer uses this to confirm that the queue truly holds enough
        completed trajectories *before* deactivating hybrid replicas — avoiding a
        race where the ratio-based check passes but the queue hasn't caught up yet.

        Args:
            required_count: Minimum number of samples to wait for.
                Defaults to ``self.required_samples``.
            poll_interval: Seconds between consecutive queue-size checks.
            timeout: Maximum seconds to wait.  Raises ``TimeoutError`` if
                exceeded.  ``None`` means no timeout.

        Returns:
            The final queue size observed (≥ required_count).

        Raises:
            TimeoutError: If ``timeout`` is set and the wait exceeds it.
        """

        start_time = time.time()
        while True:
            queue_size = await self.message_queue_client.get_queue_size()
            if queue_size >= required_count:
                print(
                    f"[FullyAsyncRollouter] wait_for_enough_samples: "
                    f"queue_size={queue_size} >= required={required_count}, done. "
                    f"waited={time.time() - start_time:.2f}s",
                    flush=True,
                )
                return queue_size

            if timeout is not None and (time.time() - start_time) >= timeout:
                raise TimeoutError(
                    f"Timed out waiting for {required_count} samples in queue (current={queue_size}, waited={timeout}s)"
                )

            await asyncio.sleep(poll_interval)
