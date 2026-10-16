# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# Script that merges a Megatron-trained LoRA adapter into the base model and
# exports merged HuggingFace weights. Reuses the Hydra ppo_megatron_trainer config;
# spins up Ray actor workers that build the actor engine (which loads base
# weights and applies the adapter), then writes HF weights to a "huggingface"
# directory that is a sibling of the adapter directory.
# Side effects at import: sets NCCL_DEBUG=WARN and TOKENIZERS_PARALLELISM=true.

"""LoRA merger for the Megatron backend, built on the unified model engine.

The legacy :mod:`verl.workers.megatron_workers` module was removed in favour of
:mod:`verl.workers.engine_workers`. This script ports the old ``CustomSaveWorker``
onto the new ``ActorRolloutRefWorker`` / ``TrainingWorker`` stack while keeping
the same CLI contract: reuse the training Hydra config, supply an
``actor_rollout_ref.model.lora.adapter_path``, and the merged HuggingFace
weights are written next to the adapter checkpoint.
"""

import os
from pprint import pprint

import hydra
import ray
import torch
from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.engine_workers import ActorRolloutRefWorker

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CustomSaveWorker(ActorRolloutRefWorker):
    # [CLASS DOC]
    # Ray worker subclass adding one RPC that dumps the LoRA-merged actor weights to HF format.
    """Extends the unified :class:`ActorRolloutRefWorker` with a merge RPC.

    The actor ``TrainingWorker`` built by ``init_model`` already loads the base
    HuggingFace weights via the Megatron bridge and applies the LoRA adapter
    from ``config.model.lora.adapter_path`` (see
    :func:`verl.utils.megatron_utils.make_megatron_module`). Here we just need
    to dump the merged weights back to HuggingFace format.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_merged_weights(self, hf_ckpt_path):
        # [FUNCTION DOC]
        # Behavior: runs on every rank (ONE_TO_ALL dispatch). Reloads actor params to
        # the accelerator if param offload is enabled, barriers across ranks, prints a
        # rank-tagged message, then exports through the engine's bridge: save_weights
        # (distributed fs, memory efficient) when vanilla_bridge, else save_hf_weights.
        # Inputs: hf_ckpt_path - output directory. Outputs: True.
        # Exceptions: bridge/distributed errors propagate.
        # Side effects: device memory movement, collective barrier, file writes, prints.
        engine = self.actor.engine

        # Reload model parameters to the accelerator if they were offloaded.
        if engine.is_param_offload_enabled:
            self.actor.to(device="device", model=True, optimizer=False, grad=False)

        torch.distributed.barrier()

        print(f"[Rank {os.environ.get('RANK', '?')}] Saving merged weights to {hf_ckpt_path}...")

        # ``MegatronEngine`` exposes the bridge that was used to load the base
        # weights; reuse it to export merged HF-format weights.
        if engine.vanilla_bridge:
            engine.bridge.save_weights(engine.module, hf_ckpt_path, distributed_filesystem=True, memory_efficient=True)
        else:
            engine.bridge.save_hf_weights(engine.module, hf_ckpt_path)

        return True


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_megatron_trainer", version_base=None)
def main(config):
    # [FUNCTION DOC]
    # Behavior: Hydra entry point. Asserts lora.adapter_path is set; if the actor
    # optimizer's lr_decay_steps is None or < 1, overwrites it with 100000 to pass
    # scheduler validation; then calls run_merge.
    # Inputs: config - composed Hydra DictConfig. Exceptions: AssertionError if no adapter path.
    # Side effects: mutates config.
    assert config.actor_rollout_ref.model.lora.adapter_path is not None, "adapter_path must be specified"

    if (
        config.actor_rollout_ref.actor.optim.lr_decay_steps is None
        or config.actor_rollout_ref.actor.optim.lr_decay_steps < 1
    ):
        # set to bypass OptimizerParamScheduler checks
        config.actor_rollout_ref.actor.optim.lr_decay_steps = 100000

    run_merge(config)


def run_merge(config) -> None:
    # [FUNCTION DOC]
    # Behavior: initializes Ray if not already running, merging default runtime env
    # vars (TOKENIZERS_PARALLELISM, NCCL_DEBUG) with config.ray_kwargs.ray_init.runtime_env
    # and passing the remaining ray_init kwargs; then blocks on the remote main_task.
    # Inputs: config. Outputs: none. Side effects: Ray init, prints the init kwargs.
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    # [FUNCTION DOC]
    # Behavior: Ray remote task (1 CPU). Prints and resolves the config in place,
    # creates a resource pool of n_gpus_per_node x nnodes, a worker group of
    # CustomSaveWorker with role "actor", initializes models, computes the output
    # path as parent-of(normalized adapter_path)/huggingface, creates it and calls
    # save_merged_weights on all workers.
    # Inputs: config. Outputs: none. Side effects: allocates GPUs, writes files.
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(CustomSaveWorker), config=config.actor_rollout_ref, role="actor"
    )
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)

    worker = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    worker.init_model()

    # ``adapter_path`` points at the model/dist_ckpt/ tree that holds the
    # PEFT adapter shards.  The merged HF weights are written as the sibling
    # huggingface/ subtree inside the same model/ directory.
    adapter_path = config.actor_rollout_ref.model.lora.adapter_path
    model_dir = os.path.dirname(os.path.normpath(adapter_path))
    hf_ckpt_path = os.path.join(model_dir, "huggingface")
    os.makedirs(hf_ckpt_path, exist_ok=True)
    worker.save_merged_weights(hf_ckpt_path)


if __name__ == "__main__":
    """
    Use the same config as your training script, besides **specifying the adapter_path**.

    For example, your training script starts with:
        `python3 -m verl.trainer.main_ppo --config-name=ppo_megatron_trainer ...`
    Now replace it with
        `python3 ./scripts/megatron_merge_lora.py --config-name=ppo_megatron_trainer ...`
    """
    main()
