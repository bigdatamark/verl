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
# Standalone GPU performance script (not a pytest test) comparing vLLM async rollout distributed backends
# by timing AgentLoopManager.generate_sequences on GSM8K prompts. Requires the GSM8K train parquet under
# ~/data/gsm8k, 8 GPUs, a Ray cluster and the Qwen2.5-7B-Instruct model. Prints per-step timings to stdout.
# When run as main, executes 1 step for each of ("async","zeromq") and ("async","ray") on 8 GPUs per node.

"""
Compare vLLM AsyncLLM backend: ExternalRayDistributedExecutor(remote call) vs RayDistributedExecutor(compiled graph)

1. Prepare openai/gsm8k dataset
python3 examples/data_preprocess/gsm8k.py

2. Run perf test
python3 tests/workers/rollout/perf/vllm_async_rollout.py >perf.log 2>&1

hardware: Nvidia 8*H20
packages:
- torch==2.6.0
- vllm==0.8.5

[DEBUG] backend: sync, n_gpus_per_node: 8, batch_size: 2048, step: 0, step_time: 21.27 secs
[DEBUG] backend: zeromq, n_gpus_per_node: 8, batch_size: 2048, step: 0, step_time: 23.40 secs
[DEBUG] backend: ray, n_gpus_per_node: 8, batch_size: 2048, step: 0, step_time: 25.33 secs
"""

import os
import time

import ray
from omegaconf import DictConfig
from torch.utils.data import SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from tests.experimental.agent_loop.agent_utils import AgentLoopManager, RayWorkerGroup, init_agent_loop_manager
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.dataset import RLHFDataset
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn


def init_config(n_gpus_per_node) -> DictConfig:
    # [FUNCTION DOC]
    # Composes the ppo_trainer Hydra config (config dir resolved relative to CWD, so must run from repo root)
    # with dynamic bsz and FSDP param/optimizer offload, then overrides: GPUs per node, train batch 128, raw chat
    # returned, Qwen2.5-7B-Instruct model, async rollout mode, TP 2, GPU memory utilization 0.9, hermes multi-turn
    # format, prompt/response lengths 4096, n=16.
    # Inputs: n_gpus_per_node. Outputs: DictConfig. Side effects: Hydra global init.
    import os

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ],
        )
    config.trainer.n_gpus_per_node = n_gpus_per_node
    config.data.train_batch_size = 128
    config.data.return_raw_chat = True
    config.actor_rollout_ref.model.path = "Qwen/Qwen2.5-7B-Instruct"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
    config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    config.actor_rollout_ref.rollout.prompt_length = 4096
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 16

    return config


def initialize(config, backend) -> tuple[AgentLoopManager | RayWorkerGroup, StatefulDataLoader]:
    # [FUNCTION DOC]
    # Starts Ray with env vars (NCCL_DEBUG WARN, VLLM_USE_V1 1, VERL_VLLM_DISTRIBUTED_BACKEND=backend), creates
    # the agent loop manager (which launches async LLM servers), loads the tokenizer, builds an RLHFDataset from
    # ~/data/gsm8k/train.parquet and a sequential StatefulDataLoader (batch size gen_batch_size or train_batch_size,
    # default 8 workers, drop_last, default collate).
    # Inputs: config; backend name. Outputs: (agent loop manager, dataloader). Side effects: ray.init, GPU servers.
    env_vars = {
        "NCCL_DEBUG": "WARN",
        "VLLM_USE_V1": "1",
        "VERL_VLLM_DISTRIBUTED_BACKEND": backend,
    }
    ray.init(runtime_env={"env_vars": env_vars})

    # STEP 1: init async llm server
    server = init_agent_loop_manager(config)

    # STEP 2: create dataloader
    tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path)
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/gsm8k/train.parquet"),
        tokenizer=tokenizer,
        config=config.data,
    )
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=config.data.get("gen_batch_size", None) or config.data.train_batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        drop_last=True,
        collate_fn=default_collate_fn,
        sampler=SequentialSampler(dataset),
    )

    return server, dataloader


def perf_rollout(mode, backend, n_gpus_per_node, num_steps):
    # [FUNCTION DOC]
    # Builds config, sets rollout mode, initializes, then for each batch converts it to DataProto, keeps only
    # input_ids/attention_mask/position_ids plus raw_prompt_ids/raw_prompt, times generate_sequences and prints a
    # DEBUG line with backend, GPU count, generated batch size, step and elapsed seconds; stops after num_steps.
    # Inputs: mode, backend, n_gpus_per_node, num_steps. Side effects: prints; shuts down Ray at end.
    config = init_config(n_gpus_per_node)
    config.actor_rollout_ref.rollout.mode = mode
    agent_loop_manager, dataloader = initialize(config, backend)

    for step, batch in enumerate(dataloader):
        batch: DataProto = DataProto.from_single_dict(batch)
        batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "raw_prompt"],
        )
        t_start = time.time()
        gen_batch = agent_loop_manager.generate_sequences(batch)
        t_end = time.time()
        print(
            f"[DEBUG] backend: {backend}, n_gpus_per_node: {n_gpus_per_node}, batch_size: {len(gen_batch)}, "
            f"step: {step}, step_time: {t_end - t_start:.2f} secs"
        )
        if step + 1 >= num_steps:
            break

    ray.shutdown()


if __name__ == "__main__":
    num_steps = 1
    n_gpus_per_node = 8

    # test_cases = [("sync", "sync"), ("async", "zeromq"), ("async", "ray")]
    test_cases = [("async", "zeromq"), ("async", "ray")]
    for mode, backend in test_cases:
        perf_rollout(mode=mode, backend=backend, n_gpus_per_node=n_gpus_per_node, num_steps=num_steps)
