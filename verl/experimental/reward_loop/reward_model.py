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
# Manager that launches reward model inference servers (as rollout replicas flagged as reward models) and an HTTP
# router in front of them, with wake/sleep control.
# Side effects on import: module logger with level from VERL_LOGGING_LEVEL (default WARN).

import asyncio
import logging
import os

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    # [CLASS DOC]
    # Owns reward model rollout replicas, their server handles/addresses, the RM tokenizer/processor, and the router
    # address.
    """Reward model manager."""

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores config and pool, launches servers and the router; asserts skip_tokenizer_init is false; sleeps the
        # servers if free_cache_engine is set.
        # Inputs: config - reward model config; optional resource_pool (colocated mode) or none (standalone).
        # Exceptions: AssertionError on bad config or insufficient GPUs. Side effects: launches servers and router process.
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """
        self.config = config
        self.resource_pool = resource_pool

        self._initialize_llm_servers()
        self._initialize_router()
        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        # [FUNCTION DOC]
        # HELPER: Replica world size is TP times DP times PP; total world size is the pool's world size, or n_gpus_per_node
        # times nnodes in standalone mode; replica count is their integer ratio (asserted positive). Builds an HF model config
        # for the RM path and its processor, creates that many replicas of the configured engine class, then initializes them
        # colocated on equal splits of the pool (count asserted equal) or standalone, and records handles and addresses.
        # Exceptions: AssertionError when not enough GPUs.
        rollout_config = self.config.rollout
        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size
        assert num_replicas > 0, (
            f"Not enough GPUs to run the reward model. "
            f"world_size ({world_size}) < rollout_world_size ({rollout_world_size}). "
            f"Check your resource pool or standalone config (n_gpus_per_node, nnodes)."
        )

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        model_config = HFModelConfig(path=self.config.model_path)
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            self._run_all(
                [
                    server.init_colocated(resource_pool)
                    for server, resource_pool in zip(self.rollout_replicas, split_resource_pools, strict=True)
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        # [FUNCTION DOC]
        # HELPER: Builds http worker URLs from server addresses and launches the naive router process; stores its address.
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        # TODO (dyy): sglang router is not ready yet.
        # if self.config.rollout.name == "sglang":
        #     from .router.inner_sglang_router import launch_router_process
        # else:
        #     from .router.naive_router import launch_router_process

        from .router.naive_router import launch_router_process

        self.router_address, _ = launch_router_process(worker_urls=worker_urls)

    def get_router_address(self):
        # [FUNCTION DOC]
        # Behavior: returns the router address string.
        return self.router_address

    def wake_up(self):
        # [FUNCTION DOC]
        # Behavior: wakes up all replicas (blocking until done).
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        # [FUNCTION DOC]
        # Behavior: puts all replicas to sleep (blocking until done).
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        # [FUNCTION DOC]
        # HELPER: Runs all given awaitables concurrently in a fresh event loop and waits for completion; returns nothing.
        # Exceptions: fails if called from within a running event loop.
        async def run_all():
            # [FUNCTION DOC]
            # HELPER: Coroutine awaiting all tasks with gather.
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
