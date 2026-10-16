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
# Name-to-class registry for reward-loop reward managers, holding a module-level mapping populated by a decorator.

from typing import Callable

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER: dict[str, type[RewardManagerBase]] = {}


def register(name: str) -> Callable[[type[RewardManagerBase]], type[RewardManagerBase]]:
    # [FUNCTION DOC]
    # Behavior: returns a decorator that records the class under name and returns the class unchanged.
    # Inputs: name - str. Outputs: decorator.
    # Exceptions: the decorator raises ValueError if the name is already registered to a different class.
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls: type[RewardManagerBase]) -> type[RewardManagerBase]:
        # [FUNCTION DOC]
        # HELPER: Registers cls under the captured name (re-registering the same class is allowed) and returns it.
        # Exceptions: ValueError for conflicting registration.
        if name in REWARD_MANAGER and REWARD_MANAGER[name] != cls:
            raise ValueError(f"reward manager {name} has already been registered: {REWARD_MANAGER[name]} vs {cls}")
        REWARD_MANAGER[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[RewardManagerBase]:
    # [FUNCTION DOC]
    # Behavior: looks up the class registered under name. Outputs: class. Exceptions: ValueError for unknown name.
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER[name]
