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
# Dataclass configs for rewards: reward manager selection, sandbox fusion execution service, the
# generative/discriminative reward model (served via a rollout config), and the top-level RewardConfig.
# Logger level from VERL_LOGGING_LEVEL (default WARN).

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.trainer.config.config import ModuleConfig

from .rollout import RolloutConfig

__all__ = ["SandboxFusionConfig", "RewardConfig", "RewardModelConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class RewardManagerConfig(BaseConfig):
    # [CLASS DOC]
    # Reward manager selection: source ("register" or "importlib"), name, and optional external module
    # config. Not exported in the module export list.
    """Configuration for reward manager.

        A reward manager defines the mechanism of computing rule-based reward and handling different reward sources.

    Args:
        source (str): Source of the reward manager. Options: ``"register"``, ``"importlib"``. Default: ``"register"``.
        name (str, optional):
            - When ``source`` is ``"register"``, the name is used in `get_reward_manager_cls(name)``.
                See ``verl/experimental/reward/reward_manager.py`` for options. Default: ``"naive"``.
            - When ``source`` is ``"importlib"``, the name is used in ``getattr(module, name)``,
                e.g., ``"DAPORewardManager"``.
        module (ModuleConfig, optional): Optional configuration for the external module defining the reward manager,
    """

    source: str = "register"
    name: str = "naive"
    module: Optional[ModuleConfig] = field(default_factory=ModuleConfig)

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: runs base post-init. For "register", asserts the name is present in the reward-loop
        # reward manager registry. For "importlib", asserts a module with a path is configured. Other source
        # values are not rejected.
        # Exceptions: AssertionError. Side effects: imports the registry.
        # KNOWN ISSUE: the docstring points to verl/experimental/reward/reward_manager.py, but the registry
        # consulted lives under verl.experimental.reward_loop.reward_manager.
        super().__post_init__()
        if self.source == "register":
            from verl.experimental.reward_loop.reward_manager.registry import REWARD_MANAGER

            assert self.name in REWARD_MANAGER, (
                f"Reward manager is not registered: {self.name=} ,{REWARD_MANAGER.keys()=}"
            )
        elif self.source == "importlib":
            # NOTE: The existence is not checked since it depends on which machine the config is initialized on.
            assert self.module is not None and self.module.path is not None, (
                "When source is importlib, module.path should be set."
            )


@dataclass
class SandboxFusionConfig(BaseConfig):
    # [CLASS DOC]
    # Sandbox fusion settings: optional service URL, max concurrent requests (64) and per-process memory
    # limit in MB (1024). No validation.
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (Optional[str]): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: Optional[str] = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    # [CLASS DOC]
    # Reward model settings: enable flag, dedicated resource pool flag and size (GPUs per node, nodes),
    # model path, and the RolloutConfig used to serve it. No validation.
    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 8
    nnodes: int = 0
    model_path: Optional[str] = None
    rollout: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class RewardConfig(BaseConfig):
    # [CLASS DOC]
    # Top-level reward config: number of reward workers, reward manager, reward model and sandbox
    # fusion sub-configs. No validation.
    _mutable_fields = BaseConfig._mutable_fields

    # reward manager args
    num_workers: int = 8
    reward_manager: RewardManagerConfig = field(default_factory=RewardManagerConfig)

    # reward model args
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)

    # sandbox fusion args
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
