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
# GPU integration test for real rollout server adapters (vLLM/SGLang selected by
# env ROLLOUT_NAME) with a standalone LLM server, an FSDP trainer worker group
# and an nccl checkpoint engine. Verifies global_steps tagging of generation
# outputs, abort-on-weight-update behavior without resume, and partial-rollout
# resume tracking min/max global steps. Needs 4 GPUs and local Qwen3-VL-2B-Instruct.

import asyncio
import os

import pytest
import ray
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from tests.checkpoint_engine.test_utils import create_trainer_worker_group
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import (
    RayResourcePool,
)
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import CheckpointEngineConfig, HFModelConfig
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient, LLMServerManager


@pytest.fixture
def init_config() -> DictConfig:
    # [FUNCTION DOC]
    # HELPER: pytest fixture. Composes Hydra ppo_trainer from verl/trainer/config
    # (relative to cwd) with async_training.partial_rollout=True, then overrides model
    # path, rollout name from env ROLLOUT_NAME, max_num_seqs 256, response_length 4096,
    # nccl checkpoint backend, 1 node x 4 GPUs. Outputs: DictConfig.
    # Exceptions: KeyError if ROLLOUT_NAME is unset.
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+async_training.partial_rollout=True",
            ],
        )

    config.actor_rollout_ref.model.path = os.path.expanduser("~/models/Qwen/Qwen3-VL-2B-Instruct")
    config.actor_rollout_ref.rollout.name = os.environ["ROLLOUT_NAME"]
    config.actor_rollout_ref.rollout.max_num_seqs = 256
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.checkpoint_engine.backend = "nccl"
    config.actor_rollout_ref.rollout.nnodes = 1
    config.trainer.n_gpus_per_node = 4
    config.trainer.nnodes = 1

    return config


async def _run_update_weights_with_global_steps_none(
    server_manager: LLMServerClient,
    checkpoint_manager: CheckpointEngineManager,
    tokenizer: PreTrainedTokenizer,
):
    # [FUNCTION DOC]
    # HELPER: syncs weights with global_steps None, generates one chat prompt and
    # asserts the output was not aborted and its extra_fields global_steps is None.
    # Side effects: prints decoded response.
    await checkpoint_manager.update_weights(global_steps=None)
    prompt = [{"role": "user", "content": "How to make a sandwich?"}]
    prompt_ids = normalize_token_ids(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))
    output = await server_manager.generate(
        request_id="test_0",
        prompt_ids=prompt_ids,
        sampling_params={
            "temperature": 1.0,
            "logprobs": True,
        },
    )
    assert output.stop_reason not in ("aborted", "abort"), (
        f"output.stop_reason is {output.stop_reason}, expected not abort"
    )
    assert output.extra_fields["global_steps"] is None, (
        f"output.extra_fields['global_steps'] is {output.extra_fields['global_steps']}, expected None"
    )
    print("========== [update_weights with global_steps=None] ==========")
    print("[RESPONSE]", tokenizer.decode(output.token_ids, skip_special_tokens=True))


async def _run_server_manager_without_resume(
    initial_steps: int,
    train_steps: int,
    server_manager: LLMServerClient,
    checkpoint_manager: CheckpointEngineManager,
    prompts: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
):
    # [FUNCTION DOC]
    # HELPER: for each step from initial_steps over train_steps: launch generation
    # tasks for all prompts, sleep 2s, update weights with that step (interrupting
    # generation), gather outputs and assert each was aborted and tagged with the
    # previous step (step - 1). Prints the first response per step.
    # KNOWN ISSUE: the loop variable global_steps is reassigned inside the output loop; harmless only because range iteration resets it and expected_steps is computed beforehand.
    for global_steps in range(initial_steps, initial_steps + train_steps):
        tasks = []
        for i, prompt in enumerate(prompts):
            prompt_ids = normalize_token_ids(
                tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
            )
            tasks.append(
                asyncio.create_task(
                    server_manager.generate(
                        request_id=f"test_{global_steps}_{i}",
                        prompt_ids=prompt_ids,
                        sampling_params={
                            "temperature": 1.0,
                            "logprobs": True,
                        },
                    )
                )
            )

        # wait a while and update weights to interrupt the generation
        await asyncio.sleep(2)
        await checkpoint_manager.update_weights(global_steps=global_steps)

        outputs = await asyncio.gather(*tasks)
        expected_steps = global_steps - 1
        for output in outputs:
            global_steps = output.extra_fields["global_steps"]
            assert output.stop_reason in ("aborted", "abort"), (
                f"output.stop_reason is {output.stop_reason}, expected in abort"
            )
            assert global_steps == expected_steps, f"output.global_steps is {global_steps}, expected {expected_steps}"
        print(f"========== [{initial_steps=}, {train_steps=}] ==========")
        print("[RESPONSE]", tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))


async def _run_server_manager_with_resume(
    initial_steps: int,
    train_steps: int,
    server_manager: FullyAsyncLLMServerClient,
    checkpoint_manager: CheckpointEngineManager,
    prompts: list[list[dict]],
    tokenizer: PreTrainedTokenizer,
):
    # [FUNCTION DOC]
    # HELPER: launches generation for all prompts once using the fully-async client
    # (which resumes after aborts), then performs train_steps weight updates 2s apart,
    # gathers outputs and asserts each has min_global_steps = initial_steps - 1,
    # max_global_steps greater than that, and was not aborted. Prints the first response.
    # 1. rollout generate responses
    tasks = []
    for i, prompt in enumerate(prompts):
        prompt_ids = normalize_token_ids(
            tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        )
        tasks.append(
            asyncio.create_task(
                server_manager.generate(
                    request_id=f"test_{initial_steps}_{i}",
                    prompt_ids=prompt_ids,
                    sampling_params={
                        "temperature": 1.0,
                        "logprobs": True,
                    },
                )
            )
        )

    # 2. trainer update weights to rollout multiple times
    for global_steps in range(initial_steps, initial_steps + train_steps):
        # wait a while and update weights to interrupt the generation
        await asyncio.sleep(2)
        await checkpoint_manager.update_weights(global_steps=global_steps)

    # 3. wait for rollout generate responses finished
    outputs = await asyncio.gather(*tasks)
    expected_min_steps = initial_steps - 1
    for output in outputs:
        min_global_steps = output.extra_fields["min_global_steps"]
        max_global_steps = output.extra_fields["max_global_steps"]
        assert min_global_steps == expected_min_steps, (
            f"output.min_global_steps is {min_global_steps}, expected {expected_min_steps}"
        )
        assert max_global_steps > expected_min_steps, (
            f"output.max_global_steps is {max_global_steps}, expected > {expected_min_steps}"
        )
        assert output.stop_reason not in ("aborted", "abort"), (
            f"output.stop_reason is {output.stop_reason}, expected not abort"
        )
    print(f"========== [{initial_steps=}, {train_steps=}] ==========")
    print("[RESPONSE]", tokenizer.decode(outputs[0].token_ids, skip_special_tokens=True))


@pytest.mark.asyncio
async def test_server_adapter(init_config):
    # [FUNCTION DOC]
    # Behavior: init Ray with tokenizer/NCCL/vLLM env vars; create trainer group over
    # a single-node pool and reset it; create a standalone LLMServerManager and a
    # CheckpointEngineManager over its replicas; with 16 long-article prompts, run the
    # global_steps-None check, then a step-0 sync, the no-resume check (steps 1..3),
    # and the resume check (steps 4..6) with FullyAsyncLLMServerClient. Shuts down Ray.
    # Inputs: init_config fixture. Side effects: GPU/Ray resources, model loading.
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
            }
        }
    )

    # 1. create trainer worker group
    model_config: HFModelConfig = omega_conf_to_dataclass(init_config.actor_rollout_ref.model)
    checkpoint_engine_config: CheckpointEngineConfig = omega_conf_to_dataclass(
        init_config.actor_rollout_ref.rollout.checkpoint_engine
    )
    trainer_pool = RayResourcePool(process_on_nodes=[init_config.trainer.n_gpus_per_node], max_colocate_count=3)
    actor_wg = create_trainer_worker_group(trainer_pool, model_config, checkpoint_engine_config)
    actor_wg.reset()

    # 2. create standalone rollout with AgentLoopManager
    llm_server_manager = await LLMServerManager.create(config=init_config)

    # 3. create checkpoint engine manager
    checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config, actor_wg=actor_wg, replicas=llm_server_manager.get_replicas()
    )

    n = 4
    prompts = [
        [{"role": "user", "content": "Please write an article about the history of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the history of America, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of China, at least 1000 words."}],
        [{"role": "user", "content": "Please write an article about the geography of America, at least 1000 words."}],
    ] * n

    # 4. test update_weights with global_steps=None
    await _run_update_weights_with_global_steps_none(
        server_manager=llm_server_manager.get_client(),
        checkpoint_manager=checkpoint_manager,
        tokenizer=model_config.tokenizer,
    )

    # 5. test LLMServerClient without partial rollout resume
    await checkpoint_manager.update_weights(global_steps=0)
    await _run_server_manager_without_resume(
        initial_steps=1,
        train_steps=3,
        server_manager=llm_server_manager.get_client(),
        checkpoint_manager=checkpoint_manager,
        prompts=prompts,
        tokenizer=model_config.tokenizer,
    )

    # 6. test FullyLLMServerClient with partial rollout resume
    await _run_server_manager_with_resume(
        initial_steps=4,
        train_steps=3,
        server_manager=llm_server_manager.get_client(client_cls=FullyAsyncLLMServerClient),
        checkpoint_manager=checkpoint_manager,
        prompts=prompts,
        tokenizer=model_config.tokenizer,
    )

    ray.shutdown()
