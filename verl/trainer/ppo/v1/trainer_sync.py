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
# Registers the "sync" PPO trainer variant: colocated trainer/rollout without partial rollout; weight sync and replica
# sleep are driven by base PPOTrainer lifecycle hooks.

import logging
import os

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("sync")
class PPOTrainerSync(PPOTrainer):
    # [CLASS DOC]
    # Synchronous colocated PPO trainer registered as "sync"; only overrides lifecycle hooks.
    """Synchronous PPO trainer
    1. Trainer and rollout are colocated
    2. Partial rollout is disabled
    """

    def on_init_end(self):
        # [FUNCTION DOC]
        # Behavior: after init, pushes weights at global_steps to rollout replicas (covers checkpoint resume).
        # Inputs/Outputs: none. Side effects: weight sync.
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_step_end(self):
        # [FUNCTION DOC]
        # Behavior: after each step, under an "update_weights" timer, syncs weights to replicas (which wakes them).
        # Inputs/Outputs: none. Side effects: timing recorded in timing_raw.
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_sample_end(self):
        # [FUNCTION DOC]
        # Behavior: after sampling, sleeps replicas (discarding weights/KV cache) and stops rollout profiling if active.
        # Inputs/Outputs: none. Side effects: replica state change.
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
