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
# HELPER: shared test utility that stands up a full agent-loop stack on Ray
# (hybrid actor/rollout workers, optional reward model pool, reward loop, LLM
# server manager, agent loop manager, checkpoint engine) for integration tests.

import ray
from omegaconf import DictConfig

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from verl.utils import omega_conf_to_dataclass
from verl.utils.device import get_device_name
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager


def init_agent_loop_manager(config: DictConfig) -> AgentLoopManager | RayWorkerGroup:
    # [FUNCTION DOC]
    # HELPER: builds and returns an AgentLoopManager ready to generate.
    # Behavior: maps Role.ActorRollout to a "global_pool" sized
    # n_gpus_per_node x nnodes. If the reward model is enabled, either assigns it a
    # separate "reward_pool" (validating its gpus/nodes are > 0) or colocates it in
    # the global pool, overwriting its nnodes/n_gpus_per_node with the trainer's.
    # Creates resource pools, spawns colocated ActorRolloutRefWorker groups per pool,
    # calls init_model on the actor_rollout group. Requires async rollout mode. Then
    # creates RewardLoopManager (with the reward pool if enabled), LLMServerManager,
    # AgentLoopManager (using the server client and reward loop worker handles), and
    # a CheckpointEngineManager which puts replicas to sleep and syncs weights.
    # Inputs: config - full PPO-style DictConfig (trainer, reward, actor_rollout_ref).
    # Outputs: the AgentLoopManager.
    # Exceptions: ValueError for non-positive reward pool sizes or rollout.mode "sync".
    # Side effects: allocates Ray actors/GPUs; mutates config.reward.reward_model when colocated.
    # KNOWN ISSUE: the sync-mode check happens only after workers are created and models initialized, wasting resources; the return annotation includes RayWorkerGroup but only the manager is returned.
    # =========================== 1. Create hybrid ActorRollout workers ===========================
    # The unified model-engine ActorRolloutRefWorker supports both sync and async rollout modes.
    actor_rollout_cls = ActorRolloutRefWorker
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    if config.reward.reward_model.enable:
        if config.reward.reward_model.enable_resource_pool:
            mapping[Role.RewardModel] = "reward_pool"
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
        else:
            mapping[Role.RewardModel] = global_pool_id
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    resource_pool_manager.create_resource_pool()
    resource_pool_to_cls = {pool: {} for pool in resource_pool_manager.resource_pool_dict.values()}

    # create actor and rollout
    resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)
    actor_rollout_cls = RayClassWithInitArgs(
        cls=role_worker_mapping[Role.ActorRollout], config=config.actor_rollout_ref, role="actor_rollout"
    )
    resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

    device_name = get_device_name()
    all_wg = {}
    for resource_pool, class_dict in resource_pool_to_cls.items():
        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_dict = RayWorkerGroup(
            resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=device_name
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        all_wg.update(spawn_wg)
    actor_rollout_wg = all_wg["actor_rollout"]
    actor_rollout_wg.init_model()

    if config.actor_rollout_ref.rollout.mode == "sync":
        raise ValueError("Agent loop tests require async rollout mode. Please set rollout.mode=async.")

    # =========================== 2. Create AgentLoopManager ===========================
    rm_resource_pool = (
        resource_pool_manager.get_resource_pool(Role.RewardModel) if config.reward.reward_model.enable else None
    )
    reward_loop_manager = RewardLoopManager(
        config=config,
        rm_resource_pool=rm_resource_pool,
    )
    llm_server_manager = LLMServerManager.create(config=config, worker_group=actor_rollout_wg)
    agent_loop_manager = AgentLoopManager.create(
        config=config,
        llm_client=llm_server_manager.get_client(),
        reward_loop_worker_handles=reward_loop_manager.reward_loop_workers,
    )
    checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine),
        actor_wg=actor_rollout_wg,
        replicas=llm_server_manager.get_replicas(),
    )
    checkpoint_manager.sleep_replicas()
    checkpoint_manager.update_weights()

    return agent_loop_manager
