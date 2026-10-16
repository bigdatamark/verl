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
# Registers the "colocate_async" PPO trainer variant: trainer and rollout share GPUs, partial rollout enabled,
# weights synced via the checkpoint manager around each step using lifecycle hooks of the base PPOTrainer.

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient


@register_trainer("colocate_async")
class PPOTrainerColocateAsync(PPOTrainer):
    # [CLASS DOC]
    # Asynchronous colocated PPO trainer; overrides base-trainer hooks to manage rollout replica sleep/wake/abort.
    # Registered in the trainer registry under the name "colocate_async".
    """Asynchronous PPO trainer
    1. Trainer and rollout are colocated.
    2. Partial rollout is enabled.
    """

    def get_llm_client(self):
        # [FUNCTION DOC]
        # Behavior: returns an LLM server client of class FullyAsyncLLMServerClient from the llm_server_manager.
        # Inputs: none. Outputs: the client instance. Side effects: none beyond the manager call.
        """Get the LLM server client for rollout generation."""
        return self.llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def on_init_end(self):
        # [FUNCTION DOC]
        # Behavior: hook after init; pushes current weights (at global_steps) to rollout replicas, e.g. after checkpoint load.
        # Inputs/Outputs: none. Side effects: weight sync via checkpoint_manager.
        # update weights after loading checkpoint
        self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        # [FUNCTION DOC]
        # Behavior: hook at train start; schedules config.trainer.v1.colocate_async.num_warmup_batches async warmup batches
        # via the base-class helper. Inputs/Outputs: none. Side effects: enqueues warmup generation.
        self._add_async_warmup_batches(self.config.trainer.v1.colocate_async.num_warmup_batches)

    def on_step_end(self):
        # [FUNCTION DOC]
        # Behavior: hook after each step; inside an "update_weights" timer, syncs weights to replicas for global_steps then
        # resumes generation on replicas. Inputs/Outputs: none. Side effects: records timing in timing_raw; replica state change.
        with marked_timer("update_weights", self.timing_raw, color="red"):
            # wake up all replicas to update weights
            self.checkpoint_manager.update_weights(self.global_steps)
            # resume generation
            self.checkpoint_manager.resume_generation_replicas()

    def on_sample_end(self):
        # [FUNCTION DOC]
        # Behavior: hook after sampling; aborts unfinished requests (pausing generation), sleeps replicas to free weights and
        # KV cache, and stops rollout profiling if the current step is profiled. Inputs/Outputs: none. Side effects: replica state.
        # abort all unfinished requests and pause generation
        self.checkpoint_manager.abort_replicas()
        # sleep all replicas to discard weights and kv cache
        self.checkpoint_manager.sleep_replicas()
        if self.curr_step_profile:
            self._stop_rollout_profiling()
