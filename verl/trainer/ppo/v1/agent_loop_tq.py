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
# TransferQueue (TQ) variants of the agent loop worker and manager for the v1 trainer.
# Generation is fire-and-forget: each prompt uid gets a TQ tag with status running ->
# finished/failure, and every finished trajectory is written to TQ under key
# "<uid>_<session_id>_<output_index>" in partition "train" or "val" with length and
# staleness (global step) tags, instead of being returned to the caller.

# TODO: move this file to verl.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for AgentLoopManager and AgentLoopWorker"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def apply_greedy_sampling_params(params: dict[str, Any]) -> None:
    # [FUNCTION DOC]
    # HELPER: Mutates a sampling params mapping for greedy decoding: top_p 1.0, top_k -1,
    # temperature 0. Outputs: none.
    params["top_p"] = 1.0
    params["top_k"] = -1
    params["temperature"] = 0


async def _settle_session_tasks(tasks: list[asyncio.Task[Any]]) -> list[BaseException]:
    # [FUNCTION DOC]
    # HELPER: Awaits all tasks with exceptions captured and returns only the exceptions raised
    # (empty list when all succeed). Never raises from task failures.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if isinstance(result, BaseException)]


@ray.remote
class AgentLoopWorkerTQ(AgentLoopWorker):
    # [CLASS DOC]
    # Ray actor extending AgentLoopWorker: initializes the TQ client and keeps strong references
    # to background prompt tasks so they are not garbage-collected.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: delegates to the base worker init, calls tq.init(), creates an empty
        # background task set. Side effects: TQ client initialization.
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        # [FUNCTION DOC]
        # Behavior: reads and removes the optional "validate" flag; builds sampling params from the
        # rollout config (temperature, top_p, top_k, repetition_penalty 1.0, logprobs flag), using
        # val_kwargs top_p/top_k/temperature for validation; sets agent_name to the default agent
        # loop when absent; fetches per-sample trajectory info; for each row builds a per-sample
        # prompt mapping (tensor rows, NonTensorStack items, NonTensorData broadcast values; other
        # types are logged as errors and skipped) and schedules _run_prompt as a background task.
        # Inputs: batch - TensorDict with global_steps, index, uid, etc. Outputs: none (returns
        # before generation finishes). Side effects: mutates batch, creates asyncio tasks.
        """Spawn agent loop for each sample in the batch without waiting for the results."""
        validate = batch["validate"] if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if validate:
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        trajectory_info = await get_trajectory_info(batch["global_steps"], batch["index"], validate)

        # create background tasks for each sample in the batch
        for i in range(len(batch)):
            # TODO(wuxibin): add trace support
            trace_this_sample = False
            prompt = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[i]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[i].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    logger.exception(f"Unsupported type {type(v)} for key {k}")

            # “fire-and-forget” background tasks
            task = asyncio.create_task(
                self._run_prompt(prompt, sampling_params, trajectory=trajectory_info[i], trace=trace_this_sample)
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def _run_prompt(self, prompt: dict, sampling_params: dict, trajectory: dict, trace: bool = False) -> None:
        # [FUNCTION DOC]
        # Behavior: tags the uid as running in TQ partition train/val. Chooses n from an optional
        # per-sample "__rollout_n__" (default rollout.n or val_kwargs.n) and "__do_sample__" (default
        # true; for training, false switches to greedy params). Launches n agent loops concurrently
        # with session_id 0..n-1, waits for all to settle, logs errors, then tags the uid finished or
        # failure. On an unexpected exception it logs, settles any started tasks, and tags failure.
        # Inputs: prompt - per-sample mapping (mutated by popping control keys); sampling_params;
        # trajectory - info including validate; trace. Outputs: none.
        # Side effects: TQ tag writes; error logging. Never raises.
        """Spawn multiple agent loops in parallel according to rollout.n or rollout.val_kwargs.n."""
        uid, partition_id = prompt["uid"], "train" if not trajectory["validate"] else "val"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        tasks = []
        try:
            # NOTE: user can dynamically adjust n for each sample here, e.g according to task difficulty.
            config = self.config.actor_rollout_ref.rollout
            n = prompt.pop("__rollout_n__", config.n if not trajectory["validate"] else config.val_kwargs.n)
            do_sample = prompt.pop("__do_sample__", True)

            run_sampling_params = dict(sampling_params)
            if not trajectory["validate"] and not do_sample:
                apply_greedy_sampling_params(run_sampling_params)

            tasks = []
            for i in range(n):
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params, trajectory=trajectory, trace=trace, session_id=i, **prompt
                    )
                )
                tasks.append(task)

            # Publish a terminal status only after every session settles, so no sibling can write after
            # ReplayBuffer clears a failed group.
            session_errors = await _settle_session_tasks(tasks)
            if session_errors:
                for error in session_errors:
                    logger.error(
                        f"Error in _run_prompt for uid={uid}",
                        exc_info=(type(error), error, error.__traceback__),
                    )
                status = "failure"
            else:
                status = "finished"
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": status})
        except Exception as e:
            logger.exception(f"Error in _run_prompt: {e}")
            if tasks:
                await _settle_session_tasks(tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _agent_loop_postprocess(
        self, output: AgentLoopOutput | list[AgentLoopOutput], validate, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: normalizes output to a list (warns and returns when empty), computes reward scores,
        # computes teacher logprobs for the final output, and propagates the final reward score and
        # reward_extra_info to earlier outputs. For each output builds int64 prompts/responses tensors,
        # input_ids, all-ones attention mask, multimodal inputs and position ids; the stored field
        # mapping is the output's dict plus all sample kwargs, with multi_modal_data dropped,
        # loss_mask = response_mask, and input_ids/position_ids/multi_modal_inputs set. Tags record
        # status success, prompt/response/seq lengths, global_steps and min/max_global_steps from
        # extra_fields. Writes all outputs to TQ in one batch put.
        # Inputs: output - one or more agent loop outputs; validate - partition selector; kwargs -
        # sample fields including uid, session_id, global_steps. Outputs: none.
        # Side effects: TQ writes; mutates outputs.
        """Put agent loop outputs into TransferQueue."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        # TODO: Support output:list[AgentLoopOutput]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # NOTE: agent loop may has multiple outputs, put each output into TransferQueue.
        # key format: {uid}_{session_id}_{index}
        # - uid: raw prompt uid from dataset
        # - session_id: session id for rollout.n sampling
        # - index: index of agent loop output
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            # TODO: uniform response_mask and loss_mask
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    # These tags are used for off-policy staleness control, if a trajectory
                    # spans too many global steps, we need to filter it out.
                    # global_steps: which global steps this sample is from dataloader
                    "global_steps": kwargs["global_steps"],
                    # min_global_steps: start generation model weights version of this trajectory
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    # max_global_steps: end generation model weights version of this trajectory
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


class AgentLoopManagerTQ(AgentLoopManager):
    # [CLASS DOC]
    # Agent loop manager that spawns AgentLoopWorkerTQ workers and dispatches batches without
    # collecting outputs (results flow through TQ).
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: sets the worker class to AgentLoopWorkerTQ before base init.
        self.agent_loop_workers_class = AgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: async factory (sync-callable via auto_await): constructs the manager and
        # initializes its workers. Outputs: the manager instance.
        """Create agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        # [FUNCTION DOC]
        # Behavior: splits the batch into one chunk per worker and waits (ray.get) for each worker's
        # generate_sequences call, which only schedules generation and returns quickly.
        # Inputs: prompts - TensorDict batch. Outputs: none.
        # Exceptions: propagates Ray remote errors.
        """
        Dispatch input batch to agent loop workers without blocking. Workers should put agent loop outputs
        into TransferQueue once an agent loop finished.

        Args:
            prompts (TensorDict): Input batch from train or validation dataset.
        """
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=False)
            ]
        )
