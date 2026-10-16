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
# Ray single-controller variant of the SFT trainer: a driver process holds the dataloaders and dispatches batches to
# a RayWorkerGroup of TrainingWorker actors. Sets NCCL_DEBUG=WARN and TOKENIZERS_PARALLELISM=true at import.
# Provides SFTTrainer, run_sft (initializes Ray), a Hydra main ("sft_trainer_engine") and create_sft_dataset.

import os
from functools import partial

from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import ray
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.logger import log_with_rank
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    # [CLASS DOC]
    # Driver-side SFT trainer. Loads data on the driver (single sampler replica), optionally reorders each batch for
    # sequence-length balance across DP ranks, and calls the worker group's train/infer methods, collecting futures.
    def __init__(
        self,
        config,
    ):
        # [FUNCTION DOC]
        # Behavior: builds configs, datasets, dataloaders (also computing total steps and frequencies), the Ray worker group
        # engine, and the RAY-mode checkpoint handler; loads checkpoint into resume_global_step; sets device_name; prints config.
        # Inputs: config - full OmegaConf config. Outputs: none.
        self.config = config

        self._build_config()
        self._build_dataset()
        self._build_dataloader()

        self._build_engine()
        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        print(self.config)

    def _build_ckpt_handler(self):
        # [FUNCTION DOC]
        # HELPER: Creates CheckpointHandler in OrchestrationMode.RAY with the worker group as engine, the train dataloader,
        # default_local_dir and optional resume_mode ("auto"), resume_from_path, max_ckpt_to_keep, default_hdfs_dir.
        # Inputs/Outputs: none. Side effects: sets self.ckpt_handler.
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.training_client,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            mode=OrchestrationMode.RAY,
        )

    def _build_config(self):
        # [FUNCTION DOC]
        # HELPER: Converts model/engine/optim/checkpoint/profiler sections to dataclasses and validates trainer.profile_interval.
        # Inputs/Outputs: none.
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        # [FUNCTION DOC]
        # HELPER: Asserts the profile interval has 2 entries, end >= start, and end < 0 when start < 0; stores start/end.
        # Exceptions: AssertionError.
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        # [FUNCTION DOC]
        # HELPER: Builds the sft_loss function and a language_model TrainingWorkerConfig. If start_profile_step is not -1,
        # passes profile_steps (inclusive range start..end) to the worker group and, when profiler tool is "nsys", the nsys
        # worker_nsight_options from global_profiler. Creates a RayResourcePool of n_gpus_per_node per node over nnodes, a
        # RayWorkerGroup of remote TrainingWorker, sets the loss fn on all workers, and resets (initializes) them.
        # Inputs/Outputs: none. Side effects: spawns Ray actors.
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        wg_kwargs = {}
        if self.start_profile_step != -1:
            wg_kwargs["profile_steps"] = list(range(self.start_profile_step, self.end_profile_step + 1))
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        # create resource pool and worker group
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        nnodes = self.config.trainer.nnodes
        self.resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)
        ray_cls_with_init = RayClassWithInitArgs(ray.remote(TrainingWorker), config=config)
        self.training_client = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.config.trainer.device,
            **wg_kwargs,
        )
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.training_client.reset()

    def _build_dataset(self):
        # [FUNCTION DOC]
        # HELPER: Builds train and optional val datasets via create_sft_dataset with train/val_max_samples (default -1).
        # Inputs/Outputs: none.
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor=processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # [FUNCTION DOC]
        # HELPER: Builds driver-side DistributedSampler (1 replica, rank 0; train shuffled, val not; drop_last) and
        # StatefulDataLoaders with batch size = full train_batch_size, SFTTensorCollator(pad_mode), 8 workers, no pin memory.
        # Then sets total_training_steps (config or len(loader) * total_epochs) into the optimizer config, steps_per_epoch, and
        # resolves save_freq/test_freq ("after_each_epoch" means steps_per_epoch). Inputs/Outputs: none.
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = 0
        dp_size = 1

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=8,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

        # update
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

    def _get_batch_seqlens(self, data):
        # [FUNCTION DOC]
        # HELPER: Returns per-sample sequence lengths as a tensor: nested input_ids offset diffs, else attention-mask row sums.
        # Inputs: data - tensordict. Outputs: 1-D tensor.
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        return batch_seqlens

    def fit(self):
        # [FUNCTION DOC]
        # Behavior: training loop on the driver. Creates Tracking, resumes from resume_global_step and computes start epoch;
        # builds per-batch meta info (remove padding, dynamic bsz, token limits, micro batch size, temperature 1.0, global batch
        # size, pad mode/token). Per step: increments step, converts to tensordict, attaches seqlens and update_lr_scheduler;
        # if trainer.balance_batch, computes workloads from per-sample lengths, gets DP size from the worker group's "train"
        # dispatch info, partitions into equal-size balanced partitions, orders each partition with smaller items at both ends
        # (sorted, then even positions followed by reversed odd positions) and reindexes the batch accordingly; starts/stops
        # profiling at configured steps; trains (waiting on the future); renames loss/grad_norm/lr/mfu to "train/..." and adds
        # global/cumulative token metrics; logs. Validates on last step with val loader or every test_freq steps (mean of
        # infer losses), saves checkpoints on last step or every save_freq steps, prints summary and returns at last step.
        # Inputs/Outputs: none. Side effects: remote training, logging, checkpoints.
        # Exceptions: KeyError if worker metrics lack loss/grad_norm/lr/mfu.
        # KNOWN ISSUE: validation condition precedence means a test_freq step with no val loader iterates over None; freq 0
        # causes modulo-by-zero; train_time is never updated so printed time is always 0.
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                )
            ):
                global_step += 1
                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data).tolist()
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                if self.config.trainer.balance_batch:
                    global_seqlen_lst = torch.Tensor([item.size()[0] for item in data["input_ids"]])
                    global_seqlen_lst = calculate_workload(global_seqlen_lst)
                    dp_size = max(self.training_client._query_dispatch_info("train")) + 1

                    global_partition_lst = get_seqlen_balanced_partitions(
                        global_seqlen_lst, k_partitions=dp_size, equal_size=True
                    )
                    # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
                    for idx, partition in enumerate(global_partition_lst):
                        partition.sort(key=lambda x: (global_seqlen_lst[x], x))
                        ordered_partition = partition[::2] + partition[1::2][::-1]
                        global_partition_lst[idx] = ordered_partition

                    global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])

                    data = tu.index_select_tensor_dict(data, global_idx)

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()

                # train for on batch
                output = self.training_client.train_batch(data)
                output = output.get()

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                metrics = tu.get(output, "metrics")

                # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                metrics["train/loss"] = metrics.pop("loss")
                metrics["train/grad_norm"] = metrics.pop("grad_norm")
                metrics["train/lr"] = metrics.pop("lr")
                metrics["train/mfu"] = metrics.pop("mfu")
                metrics["train/global_tokens"] = torch.sum(torch.tensor(batch_seqlens, device=self.device_name)).item()
                total_tokens += metrics["train/global_tokens"]
                metrics["train/total_tokens(B)"] = total_tokens / 1e9
                tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)
                        output = output.get()
                        metrics = tu.get(output, "metrics")
                        val_losses.append(metrics["loss"])

                    val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))

                    metric = {"val/loss": val_loss.detach().item()}
                    tracking.log(data=metric, step=global_step)
                    last_valid_metric = metric

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    print(f"Total time for train steps: {train_time:.2f}s")
                    print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    # [FUNCTION DOC]
    # Behavior: initializes Ray, builds SFTTrainer and runs fit. Inputs: config. Outputs: none.
    ray.init()
    trainer = SFTTrainer(config=config)
    trainer.fit()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # [FUNCTION DOC]
    # Behavior: Hydra entry point; auto-sets device then runs run_sft. Inputs: config.
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    # [FUNCTION DOC]
    # Behavior: selects dataset class (external type from custom_cls path/name, else MultiTurnSFTDataset) and instantiates
    # it with parquet_files, tokenizer, config, processor, max_samples. Outputs: dataset instance.
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
