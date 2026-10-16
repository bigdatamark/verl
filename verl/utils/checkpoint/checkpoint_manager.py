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
# Base checkpoint manager shared by FSDP/Megatron managers (save/load content selection, retention of old checkpoint
# directories, RNG state capture/restore), plus helpers for the latest-step tracker file and for deciding whether to
# save before a capacity-block (ESI) expiration.

import os
import random
import shutil

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.trainer.config import CheckpointConfig
from verl.utils.device import get_device_name, get_torch_device, is_device_available


class BaseCheckpointManager:
    # [CLASS DOC]
    # Abstract SPMD checkpoint manager. Stores model, optimizer, lr scheduler, processing class (tokenizer/processor),
    # checkpoint config, which contents to load/save (default model, optimizer, extra), rank/world size, and the list of
    # previously saved paths used for retention. Subclasses implement load_checkpoint/save_checkpoint.
    """
    A checkpoint manager that saves and loads the following states in a SPMD way:
    - model
    - optimizer
    - lr_scheduler
    - extra_states

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig | CheckpointConfig = None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores inputs; reads load_contents/save_contents from config (default ["model", "optimizer", "extra"]);
        # initializes previous_global_step None and an empty saved-path list; reads rank and world size.
        # Inputs: model, optimizer, lr_scheduler, processing_class, checkpoint_config (DictConfig or dataclass, may be None).
        # Outputs: none. Exceptions: requires torch.distributed initialized.
        self.checkpoint_config = checkpoint_config
        checkpoint_load_contents = checkpoint_config.get("load_contents", None) if checkpoint_config else None
        checkpoint_save_contents = checkpoint_config.get("save_contents", None) if checkpoint_config else None
        if checkpoint_load_contents is None:
            checkpoint_load_contents = ["model", "optimizer", "extra"]
        if checkpoint_save_contents is None:
            checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.previous_global_step = None
        self.previous_saved_paths = []

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.processing_class = processing_class
        self.checkpoint_load_contents = checkpoint_load_contents
        self.checkpoint_save_contents = checkpoint_save_contents

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    @property
    def should_save_model(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "model" is in save contents.
        """
        Returns True if 'model' is in checkpoint_save_contents, indicating the model state should be saved.
        """
        return "model" in self.checkpoint_save_contents

    @property
    def should_save_optimizer(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "optimizer" is in save contents.
        """
        Returns True if 'optimizer' is in checkpoint_save_contents, indicating the optimizer state should be saved.
        """
        return "optimizer" in self.checkpoint_save_contents

    @property
    def should_save_extra(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "extra" is in save contents.
        """
        Returns True if 'extra' is in checkpoint_save_contents, indicating the extra state should be saved.
        """
        return "extra" in self.checkpoint_save_contents

    @property
    def should_save_hf_model(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "hf_model" is in save contents.
        """
        Returns True if 'hf_model' is in checkpoint_save_contents, indicating the model should be converted to hf
        model and saved.
        """
        return "hf_model" in self.checkpoint_save_contents

    @property
    def should_save_lora_only(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: returns config save_lora_only (dict lookup or attribute; default False); False with no config.
        if not self.checkpoint_config:
            return False
        if isinstance(self.checkpoint_config, dict):
            return self.checkpoint_config.get("save_lora_only", False)
        return getattr(self.checkpoint_config, "save_lora_only", False)

    @staticmethod
    def is_lora_only_state_dict(state_dict: dict) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if the state dict is non-empty and every key contains "lora_" or ".adapter_".
        # Inputs: state_dict - mapping. Outputs: bool.
        if not state_dict:
            return False
        return all("lora_" in k or ".adapter_" in k for k in state_dict)

    @property
    def should_load_model(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "model" is in load contents.
        """
        Returns True if 'model' is in checkpoint_load_contents, indicating the model state should be loaded.
        """
        return "model" in self.checkpoint_load_contents

    @property
    def should_load_optimizer(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "optimizer" is in load contents.
        """
        Returns True if 'optimizer' is in checkpoint_load_contents, indicating the optimizer state should be loaded.
        """
        return "optimizer" in self.checkpoint_load_contents

    @property
    def should_load_extra(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "extra" is in load contents.
        """
        Returns True if 'extra' is in checkpoint_load_contents, indicating the extra state should be loaded.
        """
        return "extra" in self.checkpoint_load_contents

    @property
    def should_load_hf_model(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if "hf_model" is in load contents.
        """
        Returns True if 'hf_model' is in checkpoint_load_contents, indicating an HF-format model
        checkpoint should be loaded (e.g. via a bridge).
        """
        return "hf_model" in self.checkpoint_load_contents

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load: bool = False):
        # [FUNCTION DOC]
        # Behavior: abstract; raises NotImplementedError. Inputs: local_path, hdfs_path, del_local_after_load.
        raise NotImplementedError

    def save_checkpoint(
        self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep: int = None
    ):
        # [FUNCTION DOC]
        # Behavior: abstract; raises NotImplementedError. Inputs: local_path, hdfs_path, global_step, max_ckpt_to_keep.
        raise NotImplementedError

    @staticmethod
    def checkpath(local_path: str, hdfs_path: str):
        # [FUNCTION DOC]
        # Behavior: returns (is local, path), preferring local_path over hdfs_path.
        # Exceptions: AssertionError if both are None.
        assert local_path is not None or hdfs_path is not None, "local_path and hdfs_path cannot be both None"
        return local_path is not None, local_path if local_path is not None else hdfs_path

    def remove_previous_save_local_path(self, path):
        # [FUNCTION DOC]
        # Behavior: for each given path (string or list), prints and recursively deletes it if it exists, ignoring errors.
        # Inputs: path - str or list of str. Outputs: none. Side effects: filesystem deletion.
        if isinstance(path, str):
            path = [path]
        for p in path:
            abs_path = os.path.abspath(p)
            print(f"Checkpoint manager remove previous save local path: {abs_path}")
            if not os.path.exists(abs_path):
                continue
            shutil.rmtree(abs_path, ignore_errors=True)

    def ensure_checkpoint_capacity(self, max_ckpt_to_keep: int):
        # [FUNCTION DOC]
        # Behavior: pre-save retention. Only for integer max_ckpt_to_keep > 1: if tracked paths already number at least the
        # limit, deletes the oldest so that limit - 1 remain. Inputs: max_ckpt_to_keep. Side effects: deletes dirs.
        """
        Remove old checkpoints to make room for a new one, keeping a safety buffer.

        With max_ckpt_to_keep=1, this does nothing - we keep the existing checkpoint
        until the new save completes successfully (handled by register_checkpoint).
        For max_ckpt_to_keep >= 2, we keep (max_ckpt_to_keep - 1) checkpoints before save.
        """
        if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 1):
            return
        if len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

    def register_checkpoint(self, new_path: str, max_ckpt_to_keep: int):
        # [FUNCTION DOC]
        # Behavior: appends new_path to tracked paths; for integer max_ckpt_to_keep > 0, deletes the oldest paths beyond the
        # limit. Inputs: new_path, max_ckpt_to_keep. Side effects: deletes dirs.
        """
        Register a successfully saved checkpoint and enforce retention limit.

        Adds the new checkpoint path to tracking and removes excess old
        checkpoints beyond max_ckpt_to_keep.
        """
        self.previous_saved_paths.append(new_path)
        if not (max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0):
            return
        if len(self.previous_saved_paths) > max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

    @staticmethod
    def get_rng_state():
        # [FUNCTION DOC]
        # Behavior: captures CPU torch, numpy and Python random RNG states, plus the accelerator RNG state keyed by device name
        # when a non-CPU device is actually available. Outputs: dict of states.
        rng_state = {
            "cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "random": random.getstate(),
        }

        # get_device_name() reports the platform's accelerator even on hosts that have
        # none, so the accelerator RNG state is only reachable when the device is
        # actually usable in this process.
        if get_device_name() != "cpu" and is_device_available():
            rng_state[get_device_name()] = get_torch_device().get_rng_state()

        return rng_state

    @staticmethod
    def load_rng_state(rng_state):
        # [FUNCTION DOC]
        # Behavior: restores CPU torch, numpy and random states, and the accelerator state if available and present.
        # Inputs: rng_state dict. Exceptions: KeyError if cpu/numpy/random entries missing.
        torch.set_rng_state(rng_state["cpu"])
        np.random.set_state(rng_state["numpy"])
        random.setstate(rng_state["random"])

        if is_device_available() and get_device_name() in rng_state:
            get_torch_device().set_rng_state(rng_state[get_device_name()])


def find_latest_ckpt_path(path, directory_format="global_step_{}"):
    # [FUNCTION DOC]
    # Behavior: returns None for None path or missing tracker (printing on rank 0 / non-distributed); reads the integer
    # step from the tracker, builds path/directory_format(step), returns None if that folder is missing.
    # Inputs: path, directory_format. Outputs: path or None. Exceptions: ValueError on non-integer tracker contents.
    # KNOWN ISSUE: print calls use logging-style "%s" formatting, so the message and path print separately unformatted.
    """
    Return the most recent checkpoint directory based on a tracker file.

    Args:
        path (str): Base directory containing the checkpoint tracker.
        directory_format (str): Template for checkpoint subfolders with one
            placeholder for the iteration number (default "global_step_{}").

    Returns:
        str or None: Full path to the latest checkpoint directory, or
        None if the tracker or checkpoint folder is missing.
    """
    if path is None:
        return None

    tracker_file = get_checkpoint_tracker_filename(path)
    if not os.path.exists(tracker_file):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Checkpoint tracker file does not exist: {tracker_file}")
        return None

    with open(tracker_file, "rb") as f:
        iteration = int(f.read().decode())
    ckpt_path = os.path.join(path, directory_format.format(iteration))
    if not os.path.exists(ckpt_path):
        print("Checkpoint does not exist: %s", ckpt_path)
        return None

    print("Found checkpoint: %s", ckpt_path)
    return ckpt_path


def get_checkpoint_tracker_filename(root_path: str):
    # [FUNCTION DOC]
    # Behavior: returns root_path joined with "latest_checkpointed_iteration.txt". Inputs: root_path. Outputs: str.
    """
    Tracker file rescords the latest chckpoint during training to restart from.
    """
    return os.path.join(root_path, "latest_checkpointed_iteration.txt")


def should_save_ckpt_esi(max_steps_duration: float, save_ckpt_duration: float = 60, redundant_time: float = 0) -> bool:
    # [FUNCTION DOC]
    # Behavior: if MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP is set (seconds), returns True when remaining time is
    # positive, max_steps_duration > 0 and remaining <= save + step + redundant durations (False on unparsable value).
    # Else if SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP is set, returns True when time until expiration is below
    # that total duration. Else False.
    # Inputs: max_steps_duration, save_ckpt_duration (default 60), redundant_time (default 0), all seconds. Outputs: bool.
    # Exceptions: ValueError if the AWS timestamp is not an integer.
    """
    Determine if checkpoint should be saved based on capacity esi expiration.

    Args:
        max_steps_duration: Max estimated time (seconds) required to complete one training step
        save_ckpt_duration: Estimated time (seconds) required to save checkpoint (default: 60)
        redundant_time: Additional buffer time (seconds) for unexpected delays (default: 0)
    """
    exp_ts_mlp = os.getenv("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # vemlp
    exp_ts_aws = os.getenv("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP")  # aws
    if exp_ts_mlp:
        try:
            import time

            remaining = float(exp_ts_mlp) - time.time()
        except ValueError:
            return False
        return (
            remaining > 0
            and max_steps_duration > 0
            and remaining <= save_ckpt_duration + max_steps_duration + redundant_time
        )
    elif exp_ts_aws:
        from datetime import datetime, timedelta

        expiration_time = datetime.fromtimestamp(int(exp_ts_aws))
        time_difference = expiration_time - datetime.now()
        threshold_minutes = (save_ckpt_duration + max_steps_duration + redundant_time) / 60
        return time_difference < timedelta(minutes=threshold_minutes)
    else:
        return False
