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

# [PACKAGE DOC]
# Reward manager package for the reward loop. Importing it registers the built-in managers ("dapo", "gdpo", "naive",
# "rate_limited"/limited, "remote") in the registry, and re-exports them with register and get_reward_manager_cls.
# The registry import must come first (import order matters).

from .registry import get_reward_manager_cls, register  # noqa: I001
from .dapo import DAPORewardManager
from .gdpo import GDPORewardManager
from .naive import NaiveRewardManager
from .limited import RateLimitedRewardManager
from .remote import RemoteRewardManager

__all__ = [
    "DAPORewardManager",
    "GDPORewardManager",
    "NaiveRewardManager",
    "RateLimitedRewardManager",
    "RemoteRewardManager",
    "register",
    "get_reward_manager_cls",
]
