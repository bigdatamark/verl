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
# CPU tests for verl.utils.config._validate_router_replay_config: router replay mode R2 must be rejected when the
# algorithm uses bypass mode (no separate old log-prob recomputation), while other modes are allowed.

from types import SimpleNamespace

import pytest

from verl.utils.config import _validate_router_replay_config


def _actor_config(mode):
    # [FUNCTION DOC]
    # HELPER: Returns a nested namespace actor config whose engine.router_replay.mode equals mode.
    return SimpleNamespace(engine=SimpleNamespace(router_replay=SimpleNamespace(mode=mode)))


def test_r2_rejects_bypass_mode_before_worker_startup():
    # [FUNCTION DOC]
    # Behavior: asserts R2 with bypass_mode True raises ValueError with the message about recording routes while
    # recomputing actor old_log_probs. Inputs: none. Outputs: none.
    with pytest.raises(ValueError, match="R2 records routes while recomputing actor old_log_probs"):
        _validate_router_replay_config(_actor_config("R2"), {"bypass_mode": True})


@pytest.mark.parametrize("mode", ["disabled", "R3"])
def test_non_r2_modes_allow_bypass(mode):
    # [FUNCTION DOC]
    # Behavior: parametrized over "disabled" and "R3"; asserts validation passes with bypass_mode True.
    # Inputs: mode - replay mode string. Outputs: none.
    _validate_router_replay_config(_actor_config(mode), {"bypass_mode": True})


def test_r2_allows_decoupled_old_log_prob():
    # [FUNCTION DOC]
    # Behavior: asserts R2 with bypass_mode False passes validation. Inputs: none. Outputs: none.
    _validate_router_replay_config(_actor_config("R2"), {"bypass_mode": False})
