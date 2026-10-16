# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Dataclass configs for the skip framework: one per skip implementation (sync rollout, async
# rollout, TransferQueue rollout) and a top-level SkipManagerConfig grouping them.

from dataclasses import dataclass, field

from verl.base_config import BaseConfig


@dataclass
class RolloutSkipConfig(BaseConfig):
    # [CLASS DOC]
    # Settings for the synchronous rollout skip: enable, dump_dir (cache directory, may contain "~"),
    # steps (training steps to which skipping applies), action ("cache" or "repeat").
    """Config for rollout skip behavior."""

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"  # cache | repeat | random | empty, refer to SkipAction in base_skip.py

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validates field types and that action is "cache" or "repeat". Exceptions: AssertionError.
        # KNOWN ISSUE: the inline comment on action lists random/empty too, but validation rejects them.
        assert isinstance(self.enable, bool), f"`enable` must be bool, got {type(self.enable)}"
        assert isinstance(self.dump_dir, str), f"`dump_dir` must be str, got {type(self.dump_dir)}"
        assert isinstance(self.steps, list), f"`steps` must be list[int], got {type(self.steps)}"
        assert all(isinstance(step, int) for step in self.steps), "`steps` must contain int only"
        assert self.action in {"cache", "repeat"}, f"`action` must be one of cache/repeat, got {self.action}"


@dataclass
class AsyncRolloutSkipConfig(BaseConfig):
    # [CLASS DOC]
    # Same fields and defaults as RolloutSkipConfig, for the async (agent loop) rollout skip.
    """Config for rollout skip behavior."""

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"  # cache | repeat | random | empty, refer to SkipAction in base_skip.py

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Same validation as RolloutSkipConfig. Exceptions: AssertionError.
        assert isinstance(self.enable, bool), f"`enable` must be bool, got {type(self.enable)}"
        assert isinstance(self.dump_dir, str), f"`dump_dir` must be str, got {type(self.dump_dir)}"
        assert isinstance(self.steps, list), f"`steps` must be list[int], got {type(self.steps)}"
        assert all(isinstance(step, int) for step in self.steps), "`steps` must contain int only"
        assert self.action in {"cache", "repeat"}, f"`action` must be one of cache/repeat, got {self.action}"


@dataclass
class RolloutTqSkipConfig(BaseConfig):
    # [CLASS DOC]
    # Same fields and defaults, for the TransferQueue-based rollout skip.
    """Config for V1 TransferQueue-based rollout skip behavior."""

    enable: bool = False
    dump_dir: str = "~/.verl/rollout_dump"
    steps: list[int] = field(default_factory=list)
    action: str = "cache"  # cache | repeat, refer to SkipAction in base_skip.py

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Same validation as RolloutSkipConfig. Exceptions: AssertionError.
        assert isinstance(self.enable, bool), f"`enable` must be bool, got {type(self.enable)}"
        assert isinstance(self.dump_dir, str), f"`dump_dir` must be str, got {type(self.dump_dir)}"
        assert isinstance(self.steps, list), f"`steps` must be list[int], got {type(self.steps)}"
        assert all(isinstance(step, int) for step in self.steps), "`steps` must contain int only"
        assert self.action in {"cache", "repeat"}, f"`action` must be one of cache/repeat, got {self.action}"


@dataclass
class SkipManagerConfig(BaseConfig):
    # [CLASS DOC]
    # Top-level skip config with sub-configs rollout, async_rollout and rollout_tq (default instances).
    """Top-level config for skip modules."""

    rollout: RolloutSkipConfig = field(default_factory=RolloutSkipConfig)
    async_rollout: AsyncRolloutSkipConfig = field(default_factory=AsyncRolloutSkipConfig)
    rollout_tq: RolloutTqSkipConfig = field(default_factory=RolloutTqSkipConfig)
