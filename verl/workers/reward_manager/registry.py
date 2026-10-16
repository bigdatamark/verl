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
# Name-based registry for reward manager classes: a module-level mapping from name to class, a
# registration decorator, and a lookup function.

from typing import Callable

from verl.workers.reward_manager.abstract import AbstractRewardManager

__all__ = ["register", "get_reward_manager_cls"]

REWARD_MANAGER_REGISTRY: dict[str, type[AbstractRewardManager]] = {}


def register(name: str) -> Callable[[type[AbstractRewardManager]], type[AbstractRewardManager]]:
    # [FUNCTION DOC]
    # Behavior: returns a class decorator that records the class under name and returns the class
    # unchanged. Re-registering the same class under the same name is allowed.
    # Inputs: name - registry key. Outputs: decorator.
    # Exceptions: decorator raises ValueError when name is already bound to a different class.
    # Side effects: mutates the global registry.
    """Decorator to register a reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.
    """

    def decorator(cls: type[AbstractRewardManager]) -> type[AbstractRewardManager]:
        # [FUNCTION DOC]
        # HELPER: Registers cls under the enclosing name (see above) and returns it.
        if name in REWARD_MANAGER_REGISTRY and REWARD_MANAGER_REGISTRY[name] != cls:
            raise ValueError(
                f"Reward manager {name} has already been registered: {REWARD_MANAGER_REGISTRY[name]} vs {cls}"
            )
        REWARD_MANAGER_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_manager_cls(name: str) -> type[AbstractRewardManager]:
    # [FUNCTION DOC]
    # Behavior: returns the class registered under name.
    # Exceptions: ValueError for unknown names.
    """Get the reward manager class with a given name.

    Args:
        name: `(str)`
            The name of the reward manager.

    Returns:
        `(type)`: The reward manager class.
    """
    if name not in REWARD_MANAGER_REGISTRY:
        raise ValueError(f"Unknown reward manager: {name}")
    return REWARD_MANAGER_REGISTRY[name]
