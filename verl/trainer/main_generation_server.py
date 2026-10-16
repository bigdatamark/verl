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
# Hydra entry point (config "ppo_trainer") that starts standalone rollout server replicas, sends chat-completion
# requests for every prompt in the training parquet files (n samples each), and writes responses to a parquet file.
# Side effects at import: sets NCCL_DEBUG=WARN and TOKENIZERS_PARALLELISM=true in the process environment.

"""
Generate responses given a dataset of prompts
"""

import os

import aiohttp
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

import asyncio
from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletion

from verl.utils.hdfs_io import makedirs
from verl.workers.rollout.replica import get_rollout_replica_class


async def start_server(config):
    # [FUNCTION DOC]
    # Behavior: computes number of replicas as total GPUs (gpus per node times nodes) integer-divided by rollout TP size;
    # instantiates the rollout replica class for the configured rollout backend per replica rank, initializes all in
    # standalone mode concurrently, and collects their server handles and addresses (asserting count).
    # Inputs: full config. Outputs: (list of handles, list of "host:port" addresses). Side effects: launches servers/actors.
    tp_size = config.actor_rollout_ref.rollout.tensor_model_parallel_size
    num_replicas = (config.trainer.n_gpus_per_node * config.trainer.nnodes) // tp_size
    rollout_config = config.actor_rollout_ref.rollout
    model_config = config.actor_rollout_ref.model
    # create standalone rollout server
    rollout_server_class = get_rollout_replica_class(config.actor_rollout_ref.rollout.name)
    rollout_servers = [
        rollout_server_class(
            replica_rank=replica_rank,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=config.trainer.n_gpus_per_node,
        )
        for replica_rank in range(num_replicas)
    ]
    await asyncio.gather(*[server.init_standalone() for server in rollout_servers])

    server_handles = [server._server_handle for server in rollout_servers]
    server_addresses = [server._server_address for server in rollout_servers]
    assert len(server_handles) == num_replicas
    assert len(server_addresses) == num_replicas

    return server_handles, server_addresses


async def submit_request(server_address, **chat_complete_request):
    # [FUNCTION DOC]
    # HELPER: Behavior: pops optional extra_headers from the request, opens an aiohttp session without total timeout,
    # POSTs the request JSON to the server's OpenAI-compatible chat completions endpoint with a fixed bearer token plus
    # extra headers, parses JSON into a ChatCompletion, and always closes the session.
    # Inputs: server_address; request fields as keywords. Outputs: ChatCompletion.
    # Exceptions: network/HTTP/JSON errors and validation errors propagate.
    # KNOWN ISSUE: the session is created inside the try block, so if anything fails before it is assigned the finally
    # clause raises UnboundLocalError and masks the original error. HTTP error statuses are not checked.
    try:
        extra_headers = chat_complete_request.pop("extra_headers", {})
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(
            url=f"http://{server_address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=chat_complete_request,
        ) as resp:
            data = await resp.json()
            return ChatCompletion(**data)
    finally:
        await session.close()


async def generate_per_replica(server_address, model_path: str, n_samples: int, sampling_params: dict, chat_lst: list):
    # [FUNCTION DOC]
    # HELPER: Builds one request per (prompt, sample) pair (n_samples consecutive copies per prompt) with model path and
    # sampling params, submits all concurrently to one server, returns the list of ChatCompletions in request order.
    # here we should sample n_samples for each chat_lst.
    # we use aiohttp to avoid hang in AsyncOpenAI when the number of requests is large.

    # client = AsyncOpenAI(
    #     api_key="123-abc",
    #     base_url=f"http://{server_address}/v1",
    # )

    chat_complete_request = [
        {
            "model": model_path,
            "messages": messages,
            **sampling_params,
        }
        for messages in chat_lst
        for _ in range(n_samples)
    ]

    tasks = [submit_request(server_address, **req) for req in chat_complete_request]
    results = await asyncio.gather(*tasks)
    return results


async def generate(
    server_addresses: list, model_path: str, n_samples: int, sampling_params: dict, chat_numpy: np.ndarray
):
    # [FUNCTION DOC]
    # Behavior: splits the prompt array into as many near-equal contiguous chunks as there are servers, converts each to
    # a list, generates on each server concurrently, returns a list (per replica) of result lists, preserving order.
    # Exceptions: AssertionError if chunk count mismatches.
    num_replicas = len(server_addresses)
    chat_sub_array = np.array_split(chat_numpy, num_replicas)
    chat_sub_array = [chat.tolist() for chat in chat_sub_array]
    assert len(server_addresses) == len(chat_sub_array)
    results = await asyncio.gather(
        *[
            generate_per_replica(server_addresses[i], model_path, n_samples, sampling_params, chat_sub_array[i])
            for i in range(num_replicas)
        ]
    )
    return results


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # [FUNCTION DOC]
    # Behavior: initializes Ray with fixed env vars, prints and resolves config; asserts n samples >= 1 and that greedy
    # (temperature 0) implies n = 1; builds sampling params (temperature, top_p, max_tokens = response_length); reads and
    # concatenates all train parquet files; converts the prompt column (chat message arrays) into lists and a numpy
    # array; starts servers, generates, flattens results, extracts the first choice's message content, reshapes into
    # (num prompts, n_samples) and asserts shape; stores them as a "responses" column and writes parquet to
    # data.output_path (creating its directory).
    # Inputs: Hydra config. Outputs: None. Side effects: Ray init, servers, network calls, file writes, stdout.
    # KNOWN ISSUE: building a numpy array from per-prompt message lists of different lengths yields a ragged/object or
    # failing array conversion; and the servers are started and used under two separate event loops (two asyncio.run calls).
    ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_USE_V1": "1"}})

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    n_samples = config.actor_rollout_ref.rollout.n

    if config.actor_rollout_ref.rollout.temperature == 0.0:
        assert n_samples == 1, "When temperature=0, n_samples must be 1."
    assert n_samples >= 1, "n_samples should always >= 1"

    sampling_params = {
        "temperature": config.actor_rollout_ref.rollout.temperature,
        "top_p": config.actor_rollout_ref.rollout.top_p,
        # "top_k": config.actor_rollout_ref.rollout.top_k,
        "max_tokens": config.actor_rollout_ref.rollout.response_length,
    }

    from omegaconf import ListConfig

    train_files = config.data.train_files
    if not isinstance(train_files, list | ListConfig):
        train_files = [train_files]

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)

    datasets = []
    for train_file in train_files:
        dataset = pd.read_parquet(train_file)
        datasets.append(dataset)

    # concat dataset
    dataset = pd.concat(datasets, axis=0, ignore_index=True)
    chat_lst = dataset[config.data.prompt_key].tolist()
    chat_lst = [chat.tolist() for chat in chat_lst]
    chat_numpy = np.array(chat_lst)

    # start native server
    server_handles, server_addresses = asyncio.run(start_server(config))

    # run generate
    gen_results = asyncio.run(
        generate(server_addresses, config.actor_rollout_ref.model.path, n_samples, sampling_params, chat_numpy)
    )

    # reshape results into a numpy array
    import itertools

    results = list(itertools.chain.from_iterable(gen_results))

    # extract content from results
    results = np.array([result.choices[0].message.content for result in results])
    results = np.reshape(results, (-1, n_samples))

    assert results.shape == (len(chat_lst), n_samples)

    results = results.tolist()

    # add to the data frame
    dataset["responses"] = results

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {config.data.output_path}")
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
