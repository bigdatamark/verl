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
# Helpers for the "separation" experimental trainer that decouples training (actor/critic/ref) from rollout.
# Builds the Ray resource-pool manager and the role -> worker-class mapping used by the separated trainer.


import ray

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_reference_policy


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    # [FUNCTION DOC]
    # Behavior: builds a resource pool spec for training roles only. If any of Actor, ActorRollout, Critic or RefPolicy
    # is in roles, validates trainer GPU/node counts and creates a single "trainer_pool" (n_gpus_per_node repeated nnodes
    # times), mapping every present training role to it. For Rollout and RewardModel it only validates their gpu/node
    # counts (> 0); no pools are created for them here (they are managed elsewhere).
    # Inputs: config - full trainer config (trainer.*, rollout.*, reward.reward_model.*); roles - list of Role values.
    # Outputs: a ResourcePoolManager built from the spec and role->pool mapping.
    # Exceptions: AssertionError when a required n_gpus_per_node / nnodes is not positive.
    # Side effects: none.
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    # Actor/Critic resource pool
    training_roles = [Role.Actor, Role.ActorRollout, Role.Critic, Role.RefPolicy]
    if any(role in roles for role in training_roles):
        assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
        assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

        trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
        resource_pool_spec["trainer_pool"] = trainer_pool

        for role in training_roles:
            if role in roles:
                mapping[role] = "trainer_pool"

    # Rollout resource pool
    if Role.Rollout in roles:
        assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
        assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

    if Role.RewardModel in roles:
        rm_cfg = config.reward.reward_model
        assert rm_cfg.n_gpus_per_node > 0, "config.reward.reward_model.n_gpus_per_node must be greater than 0"
        assert rm_cfg.nnodes > 0, "config.reward.reward_model.nnodes must be greater than 0"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    # [FUNCTION DOC]
    # Behavior: lazily imports worker classes and returns the role->Ray-remote-class mapping. The training role is
    # Role.Actor by default, or Role.ActorRollout when async_training.use_trainer_do_validate or
    # async_training.use_dynamic_resource_scheduling is enabled; it maps to DetachActorWorker. Critic maps to
    # TrainingWorker. RefPolicy (DetachActorWorker) is added only when need_reference_policy(config) is true.
    # Inputs: config - trainer config (optional async_training section).
    # Outputs: tuple (mapping from Role to ray.remote-wrapped class, RayWorkerGroup class).
    # KNOWN ISSUE: docstring says it returns a dict, but it actually returns a (dict, worker-group class) tuple.
    # Side effects: none.
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # Always use the unified model engine worker implementation.
    from verl.experimental.separation.engine_workers import DetachActorWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.workers.engine_workers import TrainingWorker

    ray_worker_group_cls = RayWorkerGroup

    train_role = Role.Actor
    async_training = config.get("async_training", {})
    if async_training.get("use_trainer_do_validate", False) or async_training.get(
        "use_dynamic_resource_scheduling", False
    ):
        train_role = Role.ActorRollout

    role_worker_mapping = {
        train_role: ray.remote(DetachActorWorker),
        Role.Critic: ray.remote(TrainingWorker),
    }

    # Add reference policy (if KL loss or reward is required)
    if need_reference_policy(config):
        role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)

    return role_worker_mapping, ray_worker_group_cls
