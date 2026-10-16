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
# Engine-based Ray workers for verl's single-controller design. TrainingWorker wraps one
# training engine (selected from the engine registry by model type and backend) behind
# coarse-grained remote APIs (train a batch, train mini-batches over epochs, inference,
# checkpointing, offload). ActorRolloutRefWorker composes an actor TrainingWorker, an optional
# reference TrainingWorker, a rollout engine and a checkpoint engine into a hybrid worker, and
# implements weight synchronization from trainer to rollout. Module logger level comes from
# VERL_LOGGING_LEVEL (default WARN).
import asyncio
import functools
import logging
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    # [FUNCTION DOC]
    # HELPER: Decorator factory that, when the worker has routing replay enabled, stores the given
    # boolean as non-tensor field "enable_routing_replay" on the data TensorDict before calling the
    # wrapped method. Inputs: enabled - flag value to set. Outputs: a decorator.
    # Side effects: mutates the data argument.
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        # [FUNCTION DOC]
        # HELPER: Wraps func with the flag-setting wrapper, preserving metadata.
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Sets the flag on data if self.enable_routing_replay, then delegates to func with all args.
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    # [CLASS DOC]
    # Ray worker holding one training engine with dispatch/collect info for the "train" mesh, an
    # optional FLOPs counter, and a pluggable loss function.
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        # [FUNCTION DOC]
        # Behavior: initializes the base worker and the Ray-backed global process group (no timeout), sets
        # NUMA affinity, stores sub-configs and device name. If no engine config is given, requires no
        # optimizer config and an auto-select function, and calls it with (model config, device name) to
        # obtain both. Copies use_remove_padding and use_fused_kernels from the model config onto the engine
        # config. Builds the profiler extension (tool config looked up by tool name; file prefix = model
        # type). Sets the model config's model_type and creates the engine via the registry using the engine
        # strategy as backend. Registers dispatch/collect info for mesh "train" from the engine's DP rank and
        # output-holder status. Creates a FLOPs counter when the model config has an HF config.
        # Inputs: config - TrainingWorkerConfig.
        # Exceptions: ValueError when neither engine config nor auto-select function is available;
        # AssertionError if optimizer config given without engine config.
        # Side effects: initializes distributed state, process affinity, and profiler.
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank,
                config=self.profiler_config,
                tool_config=self.profiler_tool_config,
                # Embed the model role (e.g. language_model/value_model) in trace filenames
                # so standalone (e.g. SFT) traces are self-describing per process.
                save_file_prefix=getattr(self.config, "model_type", None),
            ),
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) asserts device is "cpu" or "device" (the latter mapped to the
        # accelerator name) and moves model/optimizer/grads via the engine.
        # Exceptions: AssertionError for other device strings.
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) stores the loss function used by train/infer.
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) calls engine.initialize (builds model/optimizer etc.).
        # KNOWN ISSUE: docstring says it reloads checkpoints/resets state when already initialized, but the
        # code always just calls initialize.
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        # [FUNCTION DOC]
        # HELPER: Converts raw engine output into a TensorDict of model outputs plus aggregated metrics.
        # Behavior: pops metrics; sums the per-micro-batch losses into a tensor and averages across the DP
        # group (if any). Pops grad_norm (converted to float if tensor) and lr. All-gathers remaining
        # metrics across DP (if group exists), then sets loss, grad_norm and lr when present. Adds peak
        # allocated/reserved device memory and used host memory in GB. For keys starting with "mtp_losses",
        # replaces the value with the mean of each sublist's first element. If token counts and a FLOPs
        # counter exist, computes MFU as estimated/promised FLOPs divided by world size (divided by 3 more
        # for forward-only). Builds a TensorDict from the model outputs with metrics as non-tensor data.
        # Inputs: output - dict with loss, metrics, model_output; global_token_num - list of token counts
        # or None; delta_time - seconds; forward_only - bool; images_seqlens - optional for FLOPs.
        # Outputs: TensorDict. Side effects: collectives over DP group.
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, non-blocking, train-mesh dispatch) fixes 3-D position ids, pops controls from
        # data: disable_auto_offload (false), mini_batch_size, num_mini_batch, epochs (1), seed (42),
        # dataloader_kwargs. Requires one of mini_batch_size/num_mini_batch; per-GPU mini-batch size is
        # batch/num_mini_batch (must divide) or mini_batch_size/dp_size (must divide). Builds a shuffling
        # iterator over epochs seeded with seed + DP rank. Inside train mode and a timer, for each
        # mini-batch: fixes position ids; if input_ids exists, all-gathers per-sequence token counts across
        # DP into one flat list; stores token counts, update_lr_scheduler (true only on the last
        # iteration), and disable_auto_offload true; runs train_batch inside a named profiler range;
        # steps the profiler; collects outputs. On output-holder ranks, flattens list metrics per output
        # (aggregating Metric objects across DP, otherwise chaining nested lists) and merges them into one
        # metrics dict; returns a CPU TensorDict with metrics; other ranks return None.
        # Inputs: data - TensorDict with batch and control fields.
        # Exceptions: AssertionError on missing/indivisible sizes.
        # Side effects: training updates; collectives; profiler steps.
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        dataloader = tu.make_iterator(
            data,
            mini_batch_size=mini_batch_size_per_gpu,
            epochs=epochs,
            seed=seed + self.engine.get_data_parallel_rank(),
            dataloader_kwargs=dataloader_kwargs,
        )

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # update
            output_lst = []
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

            for batch_idx, mini_batch_td in enumerate(dataloader):
                maybe_fix_3d_position_ids(mini_batch_td)
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                # Name the mini-batch in the trace so each iteration of this otherwise identical
                # loop is distinguishable when the whole step is recorded (continuous mode).
                with torch.profiler.record_function(f"mini_batch{batch_idx}"):
                    actor_output = self.train_batch(mini_batch_td)
                # Advance the profiler once per mini-batch. This drives any
                # torch.profiler.schedule (the schedule's unit is one update mini-batch) and,
                # with no schedule, only labels torch's ProfilerStep#<n> boundary. start()/stop()
                # happen on the outer worker, but the profiler is process-global so this reaches it.
                self.profiler.step()
                output_lst.append(actor_output)

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, non-blocking, profiled) asserts a loss function exists and the engine is not
        # forward-only. Reads global_token_num, disable_auto_offload, images_seqlens. Fills missing
        # non-tensor defaults (use_remove_padding, use_dynamic_bsz, max_token_len_per_gpu,
        # micro_batch_size_per_gpu, use_fused_kernels) from configs. Runs engine.train_batch under train
        # mode and a timer. Steps the LR scheduler when update_lr_scheduler is set. On output-holder ranks,
        # drops model_output, adds lr to metrics if stepped, and postprocesses (forward_only false) to a
        # CPU TensorDict; other ranks return None.
        # Exceptions: AssertionError on missing loss or forward-only engine.
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, non-blocking) reads global_token_num, compute_loss (default true),
        # disable_auto_offload, pops no_lora_adapter, reads images_seqlens; fills missing defaults using
        # the infer token-length and infer micro-batch-size configs. Uses the loss function only when
        # compute_loss. Runs engine.infer_batch in eval mode under a timer, inside the engine's
        # disable-adapter context when no_lora_adapter. On output-holder ranks postprocesses with
        # forward_only true to a CPU TensorDict; otherwise None.
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) delegates checkpoint save to the engine and returns its result.
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) delegates checkpoint load to the engine and returns its result.
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    # [CLASS DOC]
    # Hybrid Ray worker for a role among actor, rollout, ref, actor_rollout, actor_rollout_ref.
    # Class attributes choose the worker classes for actor and ref (both TrainingWorker by default).
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    actor_worker_cls = TrainingWorker
    ref_worker_cls = TrainingWorker

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        # [FUNCTION DOC]
        # Behavior: initializes base worker; stores config, distillation config and whether distillation is
        # enabled, and role; derives is-actor/rollout/ref flags. Chooses the profiler config from actor,
        # else rollout, else ref section; converts it to ProfilerConfig and, for known tools (npu, nsys,
        # torch, torch_memory, precision_debugger), converts the tool-specific config. Determines router
        # replay from actor strategy (megatron or veomni router_replay.mode; otherwise disabled). Keeps the
        # raw profiler config for inner workers and builds the profiler extension prefixed by role.
        # Inputs: config - DictConfig with actor/rollout/ref/model sections; role; distillation_config;
        # kwargs ignored. Exceptions: AssertionError for unknown role.
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker | None = None
        self.ref: TrainingWorker | None = None
        self.rollout: BaseRollout = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        # Router replay is supported on the megatron engine and on the veomni
        # engine. Both expose `router_replay` on their per-strategy engine
        # config (the field lives on the shared `EngineConfig` base).
        actor_strategy = self.config.actor.strategy
        if actor_strategy == "megatron":
            rr_mode = self.config.actor.megatron.router_replay.mode
        elif actor_strategy == "veomni":
            rr_mode = self.config.actor.veomni.router_replay.mode
        else:
            rr_mode = "disabled"
        self.enable_routing_replay = rr_mode != "disabled"

        # Keep the raw (un-dataclassed) role profiler config so the inner actor
        # TrainingWorker can build a matching DistProfiler in init_model. Its stages then
        # annotate themselves in the (process-global) torch profiler even though start/stop
        # happen on this outer worker.
        # NOTE: we must rebuild via the hydra path (omega_conf_to_dataclass without
        # dataclass_type) so that tool_config entries are real dataclasses with
        # attribute access; the dataclass_type=ProfilerConfig variant above yields a
        # plain-dict tool_config that the inner torch profiler cannot consume.
        self._omega_profiler_config = omega_profiler_config

        DistProfilerExtension.__init__(
            self,
            DistProfiler(
                rank=self.rank,
                config=profiler_config,
                tool_config=tool_config,
                # Embed the worker role (actor/rollout/ref/...) in trace filenames so
                # per-process results are distinguishable across roles and ranks.
                save_file_prefix=self.role,
            ),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) forwards the loss function to the actor worker.
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) forwards device move to the actor worker.
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) converts model config. (1) If role includes ref: rewrites ref config
        # fields from log_prob_* keys (mini batch from actor, micro batch sizes, dynamic bsz default false,
        # max token len), builds the ref ActorConfig with a deep-copied model config with MTP disabled,
        # builds a ref profiler config, a TrainingWorkerConfig, assigns infer batching and remove-padding to
        # its engine config, creates and resets the ref worker, and registers "ref" dispatch info.
        # (2) If role includes actor: builds actor config and optional distillation config and profiler,
        # TrainingWorkerConfig; asserts actor dynamic bsz equals rollout log_prob dynamic bsz; assigns
        # infer (from rollout log_prob settings) and train batching settings and remove padding; asserts the
        # required token-length or micro-batch sizes are set; sets the loss function to the distillation PPO
        # loss or the PPO loss bound to the config; creates/resets the actor worker, sets its loss and
        # registers "actor" dispatch info. (3) If role includes rollout: builds a device mesh
        # (dp, infer_tp = tp*dp_size, infer_pp) requiring world size divisible by tp*pp, creates the rollout
        # via get_rollout_class, records base_sync_done (false when load format contains "dummy"),
        # layered_summon, and whether LoRA merge is set. (4) If actor: creates the checkpoint engine from
        # rollout.checkpoint_engine config (bucket size MB to bytes, backend kwargs, optional plugin import;
        # master is global rank 0). Finally aggressively empties device cache.
        # Exceptions: AssertionError for config mismatches.
        # Side effects: mutates self.config.ref; builds models; allocates device memory.
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # Build the inner ref profiler config via the hydra path (same as the actor / SFT),
            # so its tool_config entries are real dataclass instances the torch profiler can read.
            # This puts the reference model's inner TrainingWorker on par with the actor's, so the
            # torch profiler (and the nsys/npu backends) support the reference model too, instead
            # of the ref silently running with a disabled no-op profiler.
            ref_omega_profiler_config = self.config.ref.get("profiler", {})
            ref_profiler_config = (
                omega_conf_to_dataclass(ref_omega_profiler_config) if ref_omega_profiler_config else None
            )

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
                profiler_config=ref_profiler_config,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = self.ref_worker_cls(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            # Build the inner actor profiler config via the hydra path (same as SFT), so
            # its tool_config entries are real dataclass instances the torch profiler can
            # read. This gives the inner TrainingWorker a DistProfiler that shares the
            # process-global torch profiler, so its stages annotate the same trace.
            actor_profiler_config = (
                omega_conf_to_dataclass(self._omega_profiler_config) if self._omega_profiler_config else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
                profiler_config=actor_profiler_config,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
            self.actor = self.actor_worker_cls(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, ref mesh, profiled; routing replay flag false) runs ref inference on data and
        # returns the output on CPU, or None on non-holder ranks.
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, actor mesh, profiled; routing replay flag true) runs actor inference and returns
        # CPU output or None.
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    # scheduled=True: this stage iterates over update mini-batches, so in discrete mode its trace
    # is the one a torch.profiler.schedule sub-samples (other stages stay full).
    @DistProfiler.annotate(color="red", role="actor_update", scheduled=True)
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, actor mesh, profiled as scheduled stage; routing replay flag true) runs
        # train_mini_batch on the actor and returns CPU output or None.
        output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        # [FUNCTION DOC]
        # Behavior: (remote) asserts role includes actor and loads actor checkpoint.
        # Exceptions: AssertionError otherwise.
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # [FUNCTION DOC]
        # Behavior: (remote) asserts role includes actor and saves actor checkpoint.
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        # [FUNCTION DOC]
        # Behavior: (remote async, non-blocking) resolves mode ("auto" uses configured checkpoint engine
        # backend). For non-"naive" modes: "delta_sharded" lets the checkpoint engine drive the actor engine
        # directly; otherwise obtains the actor's per-tensor param generator and sends it through the
        # checkpoint engine; returns metrics or empty dict. For "naive": disables expandable segments and
        # empties cache; resumes rollout weights memory when free_cache_engine is set (for sglang only if
        # sleep level is not 1); gets per-tensor params (base_sync_done true); if LoRA is not merged and a
        # peft config exists, sets rollout sleep level 1 and does a base sync first when not yet done
        # (params with base_sync_done false); then updates rollout weights with the main params. Offloads
        # the actor model to CPU if param offload is enabled, synchronizes and empties cache; if
        # free_cache_engine, barriers all ranks (in a thread) and resumes the KV cache. Marks base sync done
        # and re-enables expandable segments.
        # Inputs: global_steps - step count for tracking; mode - "auto", "naive", or a backend name.
        # Outputs: metrics dict for non-naive modes; None for naive.
        # Side effects: GPU memory changes, collective barrier, rollout state changes.
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            if effective_mode == "delta_sharded":
                # the delta engine owns the sync state machine (seed vs steady,
                # snapshot prime), so it drives the training engine itself.
                metrics = await self.checkpoint_engine.send_weights(self.actor.engine, global_steps=global_steps)
                return metrics or {}
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            metrics = await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return metrics or {}

        set_expandable_segments(False)
        get_torch_device().empty_cache()
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        # sleep_level=1 (adapter mode) in SGLang never released weights, so do not resume them.
        # vLLM is different: its level-1 sleep goes through CuMemAllocator.sleep(offload_tags=("weights",))
        is_sglang = self.config.rollout.get("name", "") == "sglang"
        if is_sglang:
            resume_weights = self.config.rollout.free_cache_engine and getattr(self.rollout, "sleep_level", 2) != 1
        else:
            # vLLM: level-1 sleep still unmaps weights → must resume.
            resume_weights = self.config.rollout.free_cache_engine
        if resume_weights:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, _base_peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=_base_peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        get_torch_device().synchronize()
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After offload model to cpu", logger=logger)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            # All trainer ranks enter this naive sync path via ONE_TO_ALL. Wait for
            # their offload and device cleanup before a leader wakes the rollout
            # workers; local device synchronization alone cannot prevent OOM on peers.
            await asyncio.to_thread(torch.distributed.barrier)
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: (remote, DP-compute dispatch, non-blocking) calls the named method on the checkpoint
        # engine with the given args and returns its result.
        # Exceptions: AttributeError for unknown method names.
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
