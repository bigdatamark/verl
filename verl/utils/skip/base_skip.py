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
# Core abstractions for the skip framework: the SkipAction enum, the BaseSkip interface that concrete
# skips implement, and a name-to-class registry populated by the register_skip decorator.

from enum import Enum
from typing import Callable


class SkipAction(Enum):
    # [CLASS DOC]
    # Supported skip behaviors: CACHE (replay the cached result for each step, matching a previous run),
    # REPEAT (reuse the latest available cached result), RANDOM (random result) and EMPTY (do nothing
    # and return empty). Each concrete skip declares which it supports.
    CACHE = "cache"
    # Replace the result with cached data
    # ensuring that the output of each step is consistent with the non-skip version from the previous run,
    # while relying on cached result in every steps.
    REPEAT = "repeat"  # repeat the result
    # Replace with the latest cached results
    # while relying on one cached result at least
    RANDOM = "random"  # random the result
    EMPTY = "empty"  # do thing, and return empty


class BaseSkip:
    # [CLASS DOC]
    # Abstract skip. Class attributes: support_actions (allowed SkipAction values) and
    # support_online_step (whether the step can be extracted from call arguments at runtime).
    """Base class for skip.

    Implementations are shared per role via ``SkipManager.skip_instances`` and may receive
    overlapping calls in async pipelines; avoid mutable per-request fields on ``self``.

    Args:
        local_config: The local configuration object, refer to verl.utils.skip.SkipManagerConfig.
        global_config: The global configuration object.
    """

    support_actions = []
    support_online_step = False

    def __init__(self, local_config, global_config):
        # [FUNCTION DOC]
        # Reads action (converted to SkipAction), enable, dump_dir and steps from local_config and stores
        # global_config. Exceptions: ValueError if the action is invalid for SkipAction or not in
        # support_actions.
        self.action = SkipAction(local_config.action)
        self.enable = local_config.enable
        self.dump_dir = local_config.dump_dir
        self.steps = local_config.steps
        self.global_config = global_config
        if self.action not in self.support_actions:
            raise ValueError(f"Unsupported action: {self.action}. Supported actions are: {self.support_actions}")

    def is_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Returns the enable flag.
        return self.enable

    def meet_precondition(self, step: int, func: Callable, *args, **kwargs) -> bool:
        # [FUNCTION DOC]
        # Abstract: whether skipping should apply for this step/call. Raises NotImplementedError.
        raise NotImplementedError("meet_precondition is not implemented")

    def warp_function(self, step: int, func: Callable, *args, **kwargs):
        # [FUNCTION DOC]
        # Abstract: produce the skipped (replacement) result for a call. Raises NotImplementedError.
        raise NotImplementedError("warp_function is not implemented")

    def prepare_data(self, step: int, result, *args, **kwargs):
        # [FUNCTION DOC]
        # Abstract: handle/persist the real result of a call (e.g. dump to cache). Raises NotImplementedError.
        raise NotImplementedError("prepare_data is not implemented")

    def extract_step(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Abstract: derive the current step from call arguments. Raises NotImplementedError.
        raise NotImplementedError("extract_step is not implemented")


SKIP_REGISTRY: dict[str, type[BaseSkip]] = {}


def register_skip(
    name: str,
) -> Callable[[type[BaseSkip]], type[BaseSkip]]:
    # [FUNCTION DOC]
    # Returns a class decorator that stores the class in SKIP_REGISTRY under name (overwriting any
    # existing entry) and returns the class unchanged.
    def decorator(cls: type[BaseSkip]) -> type[BaseSkip]:
        # [FUNCTION DOC]
        # HELPER: Registers cls under the enclosing name and returns cls.
        SKIP_REGISTRY[name] = cls
        return cls

    return decorator
