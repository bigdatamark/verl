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
# Shared PPO trainer utilities: the Role enum identifying worker roles, the WorkerType alias
# (a type of single-controller Worker), config predicates deciding which auxiliary models
# (reference policy, teacher, reward model, critic) are needed, and factories for the RL
# dataset and its sampler.

import warnings
from enum import Enum

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.distillation import is_distillation_enabled
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    # [CLASS DOC]
    # Enumeration of worker roles in the PPO system (Actor, Rollout, ActorRollout, Critic,
    # RefPolicy, RewardModel, ActorRolloutRef, Env, TeacherModel with integer values 0..8).
    # Can be subclassed to add roles. String form is a short lowercase name used for worker
    # group prefixes / config keys.
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    TeacherModel = 8

    def __str__(self):
        # [FUNCTION DOC]
        # Behavior: returns the short role string (delegates to _get_role_string).
        # Outputs: str.
        return self._get_role_string()

    def _get_role_string(self):
        # [FUNCTION DOC]
        # HELPER: Maps a role to its short name: actor, rollout, actor_rollout, critic, ref, rm,
        # actor_rollout_ref, teacher. Any unmapped member (e.g. Env or subclass members) falls back
        # to the lowercased member name.
        # Outputs: str. Side effects: none.
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.TeacherModel: "teacher",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        # [FUNCTION DOC]
        # Behavior: inverse of the string mapping; looks up the lowercased name among actor, rollout,
        # actor_rollout, critic, ref, rm, actor_rollout_ref.
        # Inputs: name - role string (case-insensitive).
        # Outputs: the matching Role member.
        # Exceptions: ValueError if no role matches.
        # KNOWN ISSUE: not a full inverse of __str__ - "teacher" and "env" (TeacherModel, Env) are not
        # accepted, so round-tripping those roles raises ValueError.
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    # [FUNCTION DOC]
    # Behavior: true when KL is used either in the reward (algorithm.use_kl_in_reward, default
    # false) or as an actor loss term (actor_rollout_ref.actor.use_kl_loss).
    # Inputs: config - full trainer config. Outputs: bool. Side effects: none.
    """Given the config, do we need ref policy."""
    return config.algorithm.get("use_kl_in_reward", False) or config.actor_rollout_ref.actor.use_kl_loss


def need_teacher_policy(
    config: DictConfig,
) -> bool:
    # [FUNCTION DOC]
    # Behavior: true when distillation is enabled per the optional "distillation" config section
    # (delegates to is_distillation_enabled).
    # Inputs: config - full trainer config. Outputs: bool.
    """Given the config, do we need distillation policy."""
    return is_distillation_enabled(config.get("distillation"))


def need_reward_model(
    config: DictConfig,
) -> bool:
    # [FUNCTION DOC]
    # Behavior: returns config.reward.reward_model.enable.
    # Inputs: config - full trainer config. Outputs: the enable flag (truthy).
    """Given the config, do we need reward model."""
    return config.reward.reward_model.enable


def need_critic(config: DictConfig) -> bool:
    # [FUNCTION DOC]
    # Behavior: if critic.enable is explicitly set (not null) return it as bool; otherwise
    # enable the critic only when the advantage estimator is GAE.
    # Inputs: config - full trainer config. Outputs: bool.
    # Side effects: emits a warning when the critic is implicitly disabled.
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    # [FUNCTION DOC]
    # Behavior: picks the dataset class via get_dataset_class(data_config) (lazy import) and
    # instantiates it with the files, tokenizer, processor, config and max_samples.
    # Inputs: data_paths - file path(s); data_config - data config; tokenizer; processor
    # (may be None for text-only); is_train - accepted but unused; max_samples - cap on samples
    # (-1 means all).
    # Outputs: dataset instance. Exceptions: propagates dataset construction errors.
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    # Get the dataset class
    dataset_cls = get_dataset_class(data_config)

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    # [FUNCTION DOC]
    # Behavior: if data_config.shuffle, returns a stateful (checkpoint-resumable) torchdata
    # RandomSampler driven by a torch Generator seeded with data_config.seed when provided;
    # otherwise a SequentialSampler.
    # Inputs: data_config - data config with shuffle and optional seed; dataset.
    # Outputs: sampler. Side effects: none.
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler
