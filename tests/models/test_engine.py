# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Multi-GPU integration tests for the unified model engines (FSDP/FSDP2/Megatron) behind TrainingWorker and the
# Tinker split-step workers: actor log-prob parity with HF plus PPO updates, critic value parity plus value-loss
# update, per-tensor parameter export vs a reference state dict, FSDP2 autocast dtype resolution, and separate
# zero_grad / forward_backward / optimizer_step primitives with gradient accumulation.
# Sets NCCL_DEBUG=WARN at import; device_name is the detected accelerator name. Local models under ~/models needed.

import os

os.environ["NCCL_DEBUG"] = "WARN"

from functools import partial

import numpy as np
import pytest
import ray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Qwen3Config,
    Qwen3MoeConfig,
)

try:
    from transformers.core_model_loading import revert_weight_conversion
except ImportError:
    revert_weight_conversion = None
    pass

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.utils.torch_functional import logprobs_from_logits_naive
from verl.workers.config import (
    ActorConfig,
    CriticConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker, TrainingWorkerConfig
from verl.workers.engine_workers_tinker import (
    OptimStepParams,
    TinkerActorRolloutRefWorker,
    TinkerTrainingWorker,
    _apply_optim_step_params,
)
from verl.workers.utils.losses import ppo_loss, sft_loss, value_loss
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

device_name = get_device_name()


def get_test_language_model(device_count):
    # [FUNCTION DOC]
    # HELPER: Returns the expanded path of SmolLM2-135M-Instruct when device_count is 1, else Qwen2.5-0.5B.
    if device_count == 1:
        model = "~/models/HuggingFaceTB/SmolLM2-135M-Instruct"
    else:
        model = "~/models/Qwen/Qwen2.5-0.5B"
    model = os.path.expanduser(model)
    return model


def test_tinker_workers_expose_split_training_primitives():
    # [FUNCTION DOC]
    # Behavior: asserts the Tinker workers subclass TrainingWorker / ActorRolloutRefWorker and that
    # optimizer_zero_grad, forward_backward and optimizer_step are defined directly on the Tinker classes but not on
    # the base classes. CPU-only.
    assert issubclass(TinkerTrainingWorker, TrainingWorker)
    assert issubclass(TinkerActorRolloutRefWorker, ActorRolloutRefWorker)

    for name in ("optimizer_zero_grad", "forward_backward", "optimizer_step"):
        assert name not in TrainingWorker.__dict__
        assert name in TinkerTrainingWorker.__dict__
        assert name not in ActorRolloutRefWorker.__dict__
        assert name in TinkerActorRolloutRefWorker.__dict__


def create_training_config(model_type, strategy, device_count, model):
    # [FUNCTION DOC]
    # HELPER: Builds a TrainingWorkerConfig. Parallel sizes are all 1 on one device, else TP=PP=CP=2 and fsdp_size 4.
    # Shared kwargs: param/optimizer offload, dynamic bsz, remove padding, 500/1000 max tokens per GPU (train/infer).
    # Megatron uses McoreEngineConfig with mbridge (vanilla per VANILLA_MBRIDGE env) and McoreOptimizerConfig with
    # lr_decay_steps 10; fsdp/fsdp2 use FSDPEngineConfig with ulysses size = cp and default FSDP optimizer config.
    # Inputs: model_type ("language_model"/"value_model"), strategy, device_count, model path.
    # Outputs: TrainingWorkerConfig with no checkpoint config.
    # Exceptions: NotImplementedError for unknown strategies.
    if device_count == 1:
        tp = pp = cp = fsdp_size = 1
    else:
        tp = pp = cp = 2
        fsdp_size = 4

    path = os.path.expanduser(model)
    model_config = HFModelConfig(path=path, use_remove_padding=True)

    kwargs = dict(
        param_offload=True,
        optimizer_offload=True,
        use_dynamic_bsz=True,
        use_remove_padding=True,
        max_token_len_per_gpu=500,
        infer_max_token_len_per_gpu=1000,
    )

    if strategy == "megatron":
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            vanilla_mbridge=os.environ.get("VANILLA_MBRIDGE", "False").lower() == "true",
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            **kwargs,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    elif strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=fsdp_size, strategy=strategy, ulysses_sequence_parallel_size=cp, **kwargs
        )
        optimizer_config = FSDPOptimizerConfig()
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    config = TrainingWorkerConfig(
        model_type=model_type,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )
    return config


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "megatron"])
def test_actor_engine(strategy):
    # [FUNCTION DOC]
    # Behavior: starts Ray and a RayWorkerGroup of TrainingWorker over all local devices, resets (inits) the model,
    # sets an SFT loss, builds a seeded random batch (8 x 32, random left-padded mask, second half as responses),
    # converts to no-padding TensorDict, runs infer_batch, re-pads log_probs and asserts their masked mean matches HF
    # bf16 log-probs within atol 1e-3 / rtol 1e-2. Then adds random advantages/ref log probs, sets a PPO loss and runs
    # train_batch twice: once with auto offload and once after manually moving to device with auto offload disabled,
    # printing metrics, then moves back to CPU.
    # Inputs: strategy - fsdp, fsdp2 or megatron.
    # Side effects: Ray lifecycle, GPU training, prints.
    ray.init()
    device_count = get_torch_device().device_count()
    config = create_training_config(
        model_type="language_model",
        strategy=strategy,
        device_count=device_count,
        model=get_test_language_model(device_count),
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    wg.reset()

    sft_loss_ = partial(sft_loss, config=config)

    wg.set_loss_fn(sft_loss_)

    batch_size = 8
    seqlen = 32

    response_length = seqlen // 2

    torch.manual_seed(1)
    np.random.seed(1)

    input_ids = torch.randint(0, config.model_config.hf_config.vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    global_token_num = torch.sum(attention_mask, dim=-1).tolist()

    print(input_ids.float().mean(), attention_mask.float().mean())

    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # eval
    output = wg.infer_batch(data_td)
    output = output.get()
    logprobs_unpad = tu.get(output, "log_probs").cpu()
    logprobs = no_padding_2_padding(logprobs_unpad, data_td)

    output = DataProto.from_single_dict({"old_log_probs": logprobs})

    # load hf model and compare results with hf model
    path = config.model_config.path
    hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    hf_output = hf_model(input_ids, attention_mask=attention_mask)
    hf_logprobs = logprobs_from_logits_naive(
        hf_output.logits[:, -response_length - 1 : -1, :].float(), input_ids[:, -response_length:]
    )
    hf_logprobs_mean = torch.mean(hf_logprobs * response_mask)
    mcore_logprobs_mean = torch.mean(output.batch["old_log_probs"] * response_mask)

    torch.testing.assert_close(hf_logprobs_mean, mcore_logprobs_mean, atol=1e-3, rtol=1e-2, check_dtype=False)

    data = data.union(output)

    # TODO: sft_loss_ is not compatible with ActorWorker until we replace DataProto with torch.jagged TensorDict
    # wg.set_loss_fn(sft_loss_)

    # train for one step
    # metrics = wg.update_actor(data)
    # print(metrics)

    # add ppo data
    data.batch["advantages"] = torch.rand_like(responses, dtype=torch.float32)
    data.batch["ref_log_prob"] = torch.rand_like(responses, dtype=torch.float32)

    # construct actor config
    actor_config = ActorConfig(strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1)

    # set ppo loss
    ppo_loss_ = partial(ppo_loss, config=actor_config)
    wg.set_loss_fn(ppo_loss_)

    # update again
    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # auto load/offload
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)

    # test manual load/offload
    tu.assign_non_tensor(data_td, disable_auto_offload=True)
    wg.to("device")
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)
    wg.to("cpu")

    ray.shutdown()


def create_value_model(language_model_path, output_path):
    # [FUNCTION DOC]
    # HELPER: Creates a randomly initialized single-label token-classification (value) model from a language model
    # config (no dropout, untied embeddings), and saves model, tokenizer and config to output_path.
    # Outputs: the expanded output path.
    # Side effects: writes model files.
    config = AutoConfig.from_pretrained(language_model_path)
    config.num_labels = 1
    config.classifier_dropout = 0
    config.tie_word_embeddings = False
    model = AutoModelForTokenClassification.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(language_model_path))
    assert model.config.num_labels == 1
    path = os.path.expanduser(output_path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    return path


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_critic_engine(strategy):
    # [FUNCTION DOC]
    # Behavior: writes a value model to ~/models/test_model, starts Ray and a TrainingWorker group with
    # model_type value_model, infers values on a seeded random batch and asserts the masked mean matches an HF
    # flash-attention token-classification forward under bf16 autocast (atol/rtol 1e-2), then adds random returns,
    # sets a value loss with a CriticConfig and runs one train_batch, printing metrics.
    # Inputs: strategy - fsdp or fsdp2.
    # Side effects: writes into the user's ~/models directory, Ray lifecycle, GPU usage.
    device_count = get_torch_device().device_count()
    value_model_path = os.path.expanduser("~/models/test_model")
    language_model_path = get_test_language_model(device_count=device_count)
    create_value_model(language_model_path, value_model_path)

    torch.manual_seed(1)
    np.random.seed(1)

    ray.init()

    config = create_training_config(
        model_type="value_model", strategy=strategy, device_count=device_count, model=value_model_path
    )
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=config)
    resource_pool = RayResourcePool(process_on_nodes=[device_count])
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    # init model
    wg.reset()

    batch_size = 8
    seqlen = 32

    response_length = seqlen // 2
    input_ids = torch.randint(0, config.model_config.hf_config.vocab_size, (batch_size, seqlen))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.8, max_ratio_of_left_padding=0.2, min_ratio_of_valid_token=0.6
    )
    position_ids = compute_position_id_with_mask(attention_mask)

    global_token_num = torch.sum(attention_mask, dim=-1).tolist()

    print(input_ids.float().mean(), attention_mask.float().mean())

    responses = input_ids[:, response_length:]
    response_mask = attention_mask[:, response_length:]

    assert torch.all(response_mask[:, 0] == 1)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
        },
        meta_info={"temperature": 1.0, "global_token_num": global_token_num, "compute_loss": False},
    )

    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # eval
    output = wg.infer_batch(data_td)
    output = output.get()

    values_unpad = tu.get(output, "values").float().cpu()
    values = no_padding_2_padding(values_unpad, data_td)

    output = DataProto.from_single_dict({"values": values})

    # load hf model and compare results with hf model
    with torch.device(device_name), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        hf_model = AutoModelForTokenClassification.from_pretrained(
            value_model_path, torch_dtype=torch.float32, attn_implementation="flash_attention_2"
        )
        hf_output = hf_model(input_ids.to(device=device_name), attention_mask=attention_mask.to(device=device_name))
        hf_values = hf_output.logits[:, -response_length - 1 : -1, :].float().squeeze(-1).cpu()

    hf_values_mean = torch.mean(hf_values * response_mask)
    engine_values = torch.mean(output.batch["values"] * response_mask)

    torch.testing.assert_close(hf_values_mean, engine_values, atol=1e-2, rtol=1e-2)

    data = data.union(output)

    # add ppo data
    data.batch["returns"] = torch.rand_like(responses, dtype=torch.float32)

    # update again
    # create critic config
    critic_config = CriticConfig(
        strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1, model=config.model_config
    )
    value_loss_ = partial(value_loss, config=critic_config)
    wg.set_loss_fn(value_loss_)

    # update again
    data_td = data.to_tensordict()
    data_td = left_right_2_no_padding(data_td)

    # auto load/offload
    tu.assign_non_tensor(data_td, global_batch_size=data_td.shape[0])
    ppo_metrics = wg.train_batch(data_td)
    ppo_metrics = ppo_metrics.get()
    ppo_metrics = tu.get(ppo_metrics, "metrics")
    print(ppo_metrics)

    ray.shutdown()


def create_actor_model(tmp_path, config):
    # [FUNCTION DOC]
    # HELPER: Saves a randomly initialized causal LM built from config (plus the config) to TMP/test_model.
    # Outputs: that path.
    model = AutoModelForCausalLM.from_config(config)
    path = os.path.join(tmp_path, "test_model")
    model.save_pretrained(path)
    config.save_pretrained(path)
    return path


def _worker(rank: int, world_size: int, rendezvous_file: str, strategy: str, model_path: str):
    # [FUNCTION DOC]
    # HELPER: Spawned per-rank process for test_per_tensor_generator. Initializes a file-rendezvous process group,
    # builds a meta-device reference model from the saved config, constructs an engine (megatron TP=2 PP=2 or FSDP
    # size 4 with ulysses 2) via EngineRegistry, initializes it, gets per-tensor params and asserts each key exists in
    # the reference state dict (after revert_weight_conversion) with equal shape; rank 0 prints keys. Ends with a
    # barrier and process-group teardown.
    # Exceptions: NotImplementedError for unknown strategy; AssertionError on mismatch.
    # KNOWN ISSUE: ref_state_dict is only assigned when revert_weight_conversion is importable; with older
    # transformers the comparison loop raises NameError.
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    ref_model_config = AutoConfig.from_pretrained(model_path)
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_config(ref_model_config)

    from verl.workers.engine import BaseEngine, EngineRegistry

    # construct configs
    model_config = HFModelConfig(path=model_path, load_tokenizer=False)

    if strategy == "megatron":
        engine_config = McoreEngineConfig(
            forward_only=False,
            use_mbridge=True,
            vanilla_mbridge=os.environ.get("VANILLA_MBRIDGE", "False").lower() == "true",
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
        )
        optimizer_config = McoreOptimizerConfig(lr_decay_steps=10)
    elif strategy in ["fsdp", "fsdp2"]:
        engine_config = FSDPEngineConfig(
            forward_only=False, fsdp_size=4, strategy=strategy, ulysses_sequence_parallel_size=2
        )
        optimizer_config = FSDPOptimizerConfig()
    else:
        raise NotImplementedError(f"strategy {strategy} is not supported")

    checkpoint_config = CheckpointConfig()

    # build model engine
    engine: BaseEngine = EngineRegistry.new(
        model_type="language_model",
        backend=engine_config.strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )

    engine.initialize()

    # get per tensor parameter
    per_tensor_params, _ = engine.get_per_tensor_param()

    if revert_weight_conversion is not None:
        ref_state_dict = revert_weight_conversion(ref_model, ref_model.state_dict())

    # load ground truth and compare
    for key, value in per_tensor_params:
        assert key in ref_state_dict, f"{key} not in ref_state_dict"
        assert value.shape == ref_state_dict[key].shape, (
            f"{key} shape not equal, {value.shape} != {ref_state_dict[key].shape}"
        )
        if rank == 0:
            print(key, value.shape)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("config", [Qwen3Config(num_hidden_layers=2), Qwen3MoeConfig(num_hidden_layers=2)])
@pytest.mark.parametrize("strategy", ["megatron", "fsdp", "fsdp2"])
def test_per_tensor_generator(world_size, tmp_path, config, strategy):
    # [FUNCTION DOC]
    # Behavior: saves a 2-layer Qwen3 or Qwen3-MoE model to tmp_path and spawns world_size (8) _worker processes.
    # Inputs: world_size, tmp_path, config, strategy parametrized.
    rendezvous_file = str(tmp_path / "rdzv_mask")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)
    # create a model
    model_path = create_actor_model(tmp_path, config)
    # spawn workers
    mp.spawn(
        fn=_worker,
        args=(world_size, rendezvous_file, strategy, model_path),
        nprocs=world_size,
        join=True,
    )


def _autocast_dtype_worker(rank: int, world_size: int, rendezvous_file: str, model_path: str):
    # [FUNCTION DOC]
    # HELPER: Spawned per-rank process (regression for issue 5932). Builds FSDP2 engines with sdpa attention for
    # param_dtype bf16, fp32 and fp16 and asserts _autocast_dtype is bfloat16/float32/float16 respectively, with no
    # grad scaler for bf16/fp32 and a ShardedGradScaler for fp16.
    # Side effects: process group init/destroy.
    # Regression test for #5932: FSDP engine must resolve autocast dtype from
    # mixed_precision.param_dtype rather than hardcoding bfloat16.
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    from verl.workers.engine import BaseEngine, EngineRegistry

    model_config = HFModelConfig(
        path=model_path,
        load_tokenizer=False,
        override_config={"attn_implementation": "sdpa"},
    )

    def build_engine(mixed_precision):
        # [FUNCTION DOC]
        # HELPER: Creates and initializes an FSDP2 language-model engine with the given mixed_precision mapping,
        # fsdp_size = world_size, no ulysses, default optimizer and checkpoint configs.
        # Outputs: initialized engine.
        engine_config = FSDPEngineConfig(
            forward_only=False,
            fsdp_size=world_size,
            strategy="fsdp2",
            ulysses_sequence_parallel_size=1,
            mixed_precision=mixed_precision,
        )
        engine: BaseEngine = EngineRegistry.new(
            model_type="language_model",
            backend=engine_config.strategy,
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=FSDPOptimizerConfig(),
            checkpoint_config=CheckpointConfig(),
        )
        engine.initialize()
        return engine

    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    # bf16 (default) resolves to torch.bfloat16, no scaler needed
    engine = build_engine({"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"})
    assert engine._autocast_dtype == torch.bfloat16, f"expected bf16, got {engine._autocast_dtype}"
    assert engine.scaler is None, "bf16 should not create a scaler"

    # fp32 resolves to torch.float32 (forward_step will use nullcontext), no scaler
    engine = build_engine({"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"})
    assert engine._autocast_dtype == torch.float32, f"expected fp32, got {engine._autocast_dtype}"
    assert engine.scaler is None, "fp32 should not create a scaler"

    # fp16 gets a ShardedGradScaler for loss scaling during backward
    engine = build_engine({"param_dtype": "fp16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"})
    assert engine._autocast_dtype == torch.float16, f"expected fp16, got {engine._autocast_dtype}"
    assert isinstance(engine.scaler, ShardedGradScaler), "fp16 must create a ShardedGradScaler"

    dist.barrier()
    dist.destroy_process_group()


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns the local shard for a DTensor, otherwise the tensor itself.
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _snapshot_trainable_params(module):
    # [FUNCTION DOC]
    # HELPER: Returns detached float32 clones of the local shards of all trainable parameters.
    return [_local_tensor(param).detach().float().clone() for param in module.parameters() if param.requires_grad]


def _param_delta_norm(module, before) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Sums squared differences between current trainable local shards and the snapshot on the current device,
    # all-reduces the sum across ranks, and returns its square root.
    # Exceptions: ValueError if parameter count differs from the snapshot (strict zip).
    device = torch.device(device_name, get_torch_device().current_device())
    total = torch.zeros((), device=device)
    for param, param_before in zip((p for p in module.parameters() if p.requires_grad), before, strict=True):
        param_local = _local_tensor(param).detach().float()
        total += (param_local - param_before.to(param_local.device)).pow(2).sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total.sqrt()


def _grad_norm(module) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Global L2 norm of all existing gradients (local shards squared-summed, all-reduced, square root).
    device = torch.device(device_name, get_torch_device().current_device())
    total = torch.zeros((), device=device)
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = _local_tensor(param.grad).detach().float()
        total += grad.pow(2).sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total.sqrt()


def _has_any_grad(module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns True if any rank has any parameter with a non-None gradient (all-reduced integer flag).
    # KNOWN ISSUE: hardcodes the cuda device rather than using the device abstraction used elsewhere.
    any_grad = torch.tensor(
        int(any(param.grad is not None for param in module.parameters())),
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(any_grad, op=dist.ReduceOp.SUM)
    return any_grad.item() > 0


def _make_split_step_batch(config: HFModelConfig, engine_config: FSDPEngineConfig, world_size: int) -> TensorDict:
    # [FUNCTION DOC]
    # HELPER: Builds a deterministic 2 x 8 batch (arange ids mod vocab, full masks, last 4 tokens as responses, zero
    # old log probs, unit advantages), converts to no-padding TensorDict and attaches metadata: global batch size
    # (2 x world_size), remove-padding flag from the model config, and dynamic bsz / token limit / micro batch /
    # fused kernel settings from the engine config.
    # Outputs: TensorDict.
    batch_size = 2
    seq_len = 8
    response_length = 4
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    input_ids = input_ids.remainder(config.hf_config.vocab_size)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(seq_len, dtype=torch.long).expand(batch_size, seq_len)
    responses = input_ids[:, -response_length:]
    response_mask = torch.ones_like(responses)

    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, : seq_len - response_length],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": torch.zeros_like(responses, dtype=torch.float32),
            "advantages": torch.ones_like(responses, dtype=torch.float32),
        },
        meta_info={"temperature": 1.0},
    )
    data_td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        data_td,
        global_batch_size=batch_size * world_size,
        use_remove_padding=config.get("use_remove_padding", False),
        use_dynamic_bsz=engine_config.use_dynamic_bsz,
        max_token_len_per_gpu=engine_config.max_token_len_per_gpu,
        micro_batch_size_per_gpu=engine_config.micro_batch_size_per_gpu,
        use_fused_kernels=engine_config.use_fused_kernels,
    )
    return data_td


def _split_training_primitives_fsdp_worker(
    rank: int, world_size: int, rendezvous_file: str, model_path: str, strategy: str
):
    # [FUNCTION DOC]
    # HELPER: Spawned per-rank process. Builds an FSDP/FSDP2 engine (no dynamic bsz, micro batch 2, lr 1e-3 constant)
    # with a PPO loss and snapshots parameters. First, in train mode without zeroing on exit, zeroes grads and runs
    # forward_backward with return_model_output: asserts no grad_norm metric, log_probs present, positive grad norm,
    # and unchanged parameters. Second forward_backward (no model output) must accumulate (grad norm > 1.5x first),
    # with parameters still unchanged. Then applies custom optimizer hyperparameters (lr, betas, eps, weight decay) and
    # steps with zero-grad on exit: asserts positive grad norm, every param group carries the new hyperparameters,
    # parameters changed, and no gradients remain.
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )

    from verl.workers.engine import BaseEngine, EngineRegistry

    model_config = HFModelConfig(
        path=model_path,
        load_tokenizer=False,
        override_config={"attn_implementation": "sdpa"},
        use_remove_padding=True,
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        fsdp_size=world_size,
        strategy=strategy,
        ulysses_sequence_parallel_size=1,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        max_token_len_per_gpu=128,
    )
    engine: BaseEngine = EngineRegistry.new(
        model_type="language_model",
        backend=engine_config.strategy,
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(lr=1e-3, lr_scheduler_type="constant"),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()

    actor_config = ActorConfig(strategy=strategy, rollout_n=1, ppo_micro_batch_size_per_gpu=-1)
    loss_fn = partial(ppo_loss, config=actor_config)
    before = _snapshot_trainable_params(engine.module)

    with engine.train_mode(zero_grad_on_exit=False):
        engine.optimizer_zero_grad()
        first_batch = _make_split_step_batch(model_config, engine_config, world_size)
        tu.assign_non_tensor(first_batch, return_model_output=True)
        output_1 = engine.forward_backward_batch(
            first_batch,
            loss_function=loss_fn,
            forward_only=False,
        )
    if engine.is_mp_src_rank_with_outputs():
        assert "grad_norm" not in output_1["metrics"]
        assert "log_probs" in output_1["model_output"]
    grad_norm_1 = _grad_norm(engine.module)
    assert grad_norm_1.item() > 0
    assert _param_delta_norm(engine.module, before).item() == pytest.approx(0.0, abs=0.0)

    with engine.train_mode(zero_grad_on_exit=False):
        output_2 = engine.forward_backward_batch(
            _make_split_step_batch(model_config, engine_config, world_size),
            loss_function=loss_fn,
            forward_only=False,
        )
    if engine.is_mp_src_rank_with_outputs():
        assert "grad_norm" not in output_2["metrics"]
        assert output_2["model_output"] == {}
    grad_norm_2 = _grad_norm(engine.module)
    assert grad_norm_2.item() > grad_norm_1.item() * 1.5
    assert _param_delta_norm(engine.module, before).item() == pytest.approx(0.0, abs=0.0)

    optim_step_params: OptimStepParams = {
        "lr": 5e-4,
        "betas": (0.8, 0.9),
        "eps": 1e-7,
        "weight_decay": 0.02,
    }
    with engine.train_mode(zero_grad_on_exit=True):
        _apply_optim_step_params(engine.optimizer, optim_step_params)
        grad_norm = engine.optimizer_step()

    assert grad_norm > 0
    for param_group in engine.optimizer.param_groups:
        assert param_group["lr"] == pytest.approx(optim_step_params["lr"])
        assert param_group["betas"] == pytest.approx(optim_step_params["betas"])
        assert param_group["eps"] == pytest.approx(optim_step_params["eps"])
        assert param_group["weight_decay"] == pytest.approx(optim_step_params["weight_decay"])
    assert _param_delta_norm(engine.module, before).item() > 0
    assert not _has_any_grad(engine.module)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_fsdp_engine_split_forward_backward_and_optimizer_step(strategy, tmp_path):
    # [FUNCTION DOC]
    # Behavior: requires 2 CUDA devices; saves a tiny 2-layer Qwen3 model and spawns 2 split-primitive workers.
    # Inputs: strategy (fsdp/fsdp2), tmp_path.
    world_size = 2
    rendezvous_file = str(tmp_path / f"rdzv_split_training_primitives_{strategy}")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)
    model_path = create_actor_model(
        tmp_path,
        Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
    )
    mp.spawn(
        fn=_split_training_primitives_fsdp_worker,
        args=(world_size, rendezvous_file, model_path, strategy),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("config", [Qwen3Config(num_hidden_layers=2)])
def test_fsdp2_autocast_dtype_honors_mixed_precision(world_size, tmp_path, config):
    # [FUNCTION DOC]
    # Behavior: saves a 2-layer Qwen3 model and spawns world_size (8) _autocast_dtype_worker processes.
    rendezvous_file = str(tmp_path / "rdzv_autocast")
    os.makedirs(os.path.dirname(rendezvous_file), exist_ok=True)
    model_path = create_actor_model(tmp_path, config)
    mp.spawn(
        fn=_autocast_dtype_worker,
        args=(world_size, rendezvous_file, model_path),
        nprocs=world_size,
        join=True,
    )
