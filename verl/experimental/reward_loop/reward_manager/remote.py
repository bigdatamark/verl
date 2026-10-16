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
# Registers the "remote" reward manager, which runs synchronous score functions in separate Ray worker processes
# (avoids thread-pool issues with libraries like math-verify and suits CPU-heavy scoring).

import inspect
import itertools

import ray

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score


@ray.remote(num_cpus=1)
class RewardComputeWorker:
    # [CLASS DOC]
    # Ray actor (1 CPU) wrapping a synchronous score function; must not have async methods.
    """
    WARNING: This class cannot have async methods.
    """

    def __init__(self, compute_score_fn):
        # [FUNCTION DOC]
        # Behavior: stores the score function. Inputs: compute_score_fn - sync callable.
        # since the reward function may not be pickleable, we need to init it in the worker
        self.compute_score_fn = compute_score_fn

    def compute_score(self, **kwargs) -> dict:
        # [FUNCTION DOC]
        # Behavior: calls the score function with the given keyword arguments and returns its result (mapping or float).
        return self.compute_score_fn(**kwargs)


@register("remote")
class RemoteRewardManager(RewardManagerBase):
    # [CLASS DOC]
    # Reward manager that dispatches each sample's scoring round-robin to a pool of RewardComputeWorker actors on the
    # same node.
    """
    The reward manager.
    Some errors exist when using default thread pool to compute reward score, e.g., math-verify.
    https://github.com/verl-project/verl/issues/3407
    To avoid the above issues, we use a separate process to compute reward score.
    Moreover, process may be more suitable for cpu-intensive requests.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        # [FUNCTION DOC]
        # Behavior: base init; uses compute_score or the default; asserts it is not async; stores router address and RM
        # tokenizer; creates reward.num_workers RewardComputeWorker actors with soft affinity to the current node and a
        # round-robin cycle over them.
        # Exceptions: AssertionError for async score functions. Side effects: creates Ray actors.
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        assert not self.is_async_reward_score, "Async reward score is not supported in remote reward manager. "
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        num_reward_workers = config.reward.num_workers
        # in the rollout & reward parallel mode
        # the sum of final reward workers will be agent_loop_workers * num_reward_workers
        self.reward_worker = [
            # register the reward worker in the same node
            RewardComputeWorker.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=True,
                ),
            ).remote(self.compute_score)
            for _ in range(num_reward_workers)
        ]
        self.reward_worker_pool = itertools.cycle(self.reward_worker)

    def choose_reward_worker(self):
        # [FUNCTION DOC]
        # Behavior: returns the next worker from the round-robin cycle.
        return next(self.reward_worker_pool)

    async def run_single(self, data: DataProto) -> dict:
        # [FUNCTION DOC]
        # Behavior: same input preparation as the naive manager (last item, valid response decode, data_source, ground truth,
        # extra_info merged with tool fields, num_turns and rollout_reward_scores), then awaits the score on a chosen remote
        # worker and converts the result (mapping gives score and extra info; scalar gives score and "acc").
        # Outputs: mapping with reward_score and reward_extra_info.
        # KNOWN ISSUE: extra_info stored on the sample is mutated in place.
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        reward_worker = self.choose_reward_worker()
        result = await reward_worker.compute_score.remote(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **extra_reward_kwargs,
        )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
