# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
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
# One-step off-policy PPO trainer. Generation for step N+1 is launched as an asyncio task before training on step
# N, so rollout (on dedicated rollout GPUs via an agent loop) overlaps with training; weights are synced to rollout
# after each batch is received.

"""
This trainer supports model-agonistic model initialization with huggingface
"""

import asyncio
import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl import DataProto
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    ResourcePoolManager,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.llm_server import LLMServerManager


class OneStepOffRayTrainer(SeparateRayPPOTrainer):
    # [CLASS DOC]
    # Trainer subclassing the separated PPO trainer; the pipeline stages (reward, log-probs, advantage, updates,
    # validation, checkpoint) come from the base class, this class adds the overlapping generation scheduling.
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores tokenizer/processor/config; asserts hybrid engine is off; removes the Rollout role from the
        # mapping (agent loop creates rollout); determines reference policy, reward model and critic use; sets device and a
        # validation generations logger; ref_in_actor when a LoRA rank (nested or legacy) is positive or an adapter path is
        # set; KL controller when KL-in-reward is enabled; prefix-grouper flag; creates dataloaders; initializes step,
        # profile, metric and reward bookkeeping.
        # Inputs: see docstring. Exceptions: AssertionError if hybrid engine is on.
        # Side effects: mutates role_worker_mapping passed in.
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        # Skip rollout worker mapping and let agentloop create it.
        role_worker_mapping.pop(Role.Rollout, None)
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)

        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== SeparateRayPPOTrainer config ====================

        self.global_steps = 0
        self.epoch = 0
        self.max_steps_duration = 0
        self.progress_bar = None
        self.logger = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

    def _create_actor_rollout_classes(self):
        # [FUNCTION DOC]
        # HELPER: Registers the Actor role worker class (with actor_rollout_ref config and role name) in its resource pool.
        for role in [Role.Actor]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        # [FUNCTION DOC]
        # HELPER: Initializes critic, separate reference policy, and reward model worker groups when used; initializes the
        # actor worker group and aliases it as actor_rollout_wg; sets rm_wg to none when no reward model.
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = self.all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_wg = self.all_wg[str(Role.Actor)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    def _init_async_rollout_manager(self):
        # [FUNCTION DOC]
        # HELPER: Decides whether reward workers stream with rollout (no RM, or RM with own pool); asserts rollout mode is
        # "async"; loads a custom agent loop manager class from config if given, else the default; creates an LLM server
        # manager and the agent loop manager with its client and reward worker handles.
        # Exceptions: AssertionError if mode is not async; import errors for a bad class path.
        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        self.llm_server_manager = LLMServerManager.create(config=self.config)
        self.async_rollout_mode = True
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    def _create_continuous_iterator(self):
        # [FUNCTION DOC]
        # HELPER: Generator yielding (epoch, batch) for every batch of each configured epoch.
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.trainer.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _async_gen_next_batch(self, continuous_iterator):
        # [FUNCTION DOC]
        # HELPER: Takes the next (epoch, batch) from the iterator, returning none on exhaustion or any error (logged). Builds
        # a DataProto, assigns a random uuid per row, derives the generation batch, tags it with global_steps, repeats it n
        # times (interleaved), awaits async generation under a timer, repeats the original batch the same way and unions the
        # output; adds a response mask if missing, optionally balances the batch, and records per-sample token counts.
        # Outputs: (metrics, timing, epoch, batch, none future reward) or none.
        # KNOWN ISSUE: docstring mentions parameter synchronization, which this function does not do.
        """
        Call parameter synchronization and asynchronous sequence generation.
        """
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error in async_gen_next_batch: {e}")
            return None

        metrics = {}
        timing_raw = {}

        # Create the initial batch from the data loader
        batch = DataProto.from_single_dict(batch_dict)

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)

        # pass global_steps to trace
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        # async generation
        with marked_timer("generate_async", timing_raw, color="purple"):
            gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)

        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Launch individual reward computations as each generation completes
        future_reward = None

        # Return the original, now-modified `batch` and the `future_reward`
        return metrics, timing_raw, epoch, batch, future_reward

    @staticmethod
    @ray.remote
    def _launch_individual_rewards(batch, config, tokenizer):
        # [FUNCTION DOC]
        # HELPER: Ray remote static task that extracts the reward tensor and extra info from a batch. Inputs: batch;
        # config and tokenizer unused. Outputs: (reward tensor, extra info).
        reward_tensor, reward_extra_info = extract_reward(batch)
        return reward_tensor, reward_extra_info

    async def fit(self):
        # [FUNCTION DOC]
        # Behavior: creates the Tracking logger; resets global_steps to 0; loads checkpoint and syncs weights; optionally
        # validates before training (asserting metrics are non-empty) and returns early if val_only; creates progress bar;
        # starts at step 1, sets profile flags; creates the continuous iterator, launches the first generation task, then
        # repeatedly runs fit_step with the pending future until it returns none or the last step is reached.
        # Exceptions: AssertionError for empty validation metrics.
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self._fit_update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        # across epoch iterator
        continuous_iterator = self._create_continuous_iterator()
        # Start the first asynchronous generation task.
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return

    async def fit_step(self, batch_data_future, continuous_iterator):
        # [FUNCTION DOC]
        # Behavior: resets per-step metrics/timing/reward state, prepares the step and starts profiling; under the "step"
        # timer awaits the current batch (which also syncs weights and launches the next generation), then runs reward,
        # old log-prob, ref log-prob, critic values, advantage, critic update, actor update, and data dump, yielding to the
        # event loop between stages; then validates, saves checkpoint, stops profiling, collects metrics, runs experimental
        # hooks, and post-processes the step.
        # Inputs: batch_data_future - task producing the batch; continuous_iterator. Outputs: the next batch task or none.
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_data_future: batch future
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

            # await asyncio.sleep(0) ensures:
            # Asynchronous tasks can start executing immediately
            # The event loop can handle other pending coroutines
            # Prevents computations in a certain phase from blocking the entire asynchronous workflow
            #
            # The purpose here is to ensure that after triggering
            # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
            # the subsequent relevant logic can proceed in a timely manner
            await asyncio.sleep(0)
            batch = self._fit_compute_reward(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_ref_log_prob(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_compute_advantage(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_critic(batch)
            await asyncio.sleep(0)
            batch = self._fit_update_actor(batch)
            await asyncio.sleep(0)
            self._fit_dump_data(batch)
            await asyncio.sleep(0)

        self._fit_validate()
        await asyncio.sleep(0)
        self._fit_save_checkpoint()
        await asyncio.sleep(0)
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_experimental(batch)
        self._fit_postprocess_step()

        return batch_data_future

    async def _fit_generate(self, batch_data_future, continuous_iterator):
        # [FUNCTION DOC]
        # HELPER: Under the "gen" timer awaits the batch future and merges its metrics and timing (including the batch's
        # own "timing" meta_info, which is then removed), setting the rollout temperature. Then syncs weights to rollout
        # under a timer. Unless this is the last step, launches the next generation task and yields once.
        # Outputs: (batch, next future or none).
        # KNOWN ISSUE: if the iterator is exhausted the awaited future yields none and the tuple unpacking raises a TypeError.
        metrics = self.metrics
        timing_raw = self.timing_raw

        with marked_timer("gen", timing_raw, color="red"):
            _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            timing_raw.update(batch.meta_info["timing"])
            timing_raw.update(_timing_raw)
            metrics.update(_metrics)
            batch.meta_info.pop("timing", None)

        # sync weights from actor to rollout
        with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
            self._fit_update_weights()

        # async next generation
        if not self.is_last_step:
            batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
            await asyncio.sleep(0)
        else:
            batch_data_future = None

        return batch, batch_data_future
