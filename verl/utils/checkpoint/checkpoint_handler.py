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


# TODO: add unit tests

# [MODULE DOC]
# Trainer-level checkpoint orchestration for engine-based trainers (e.g. SFT): saves/loads model state through the
# engine (or Ray worker group), stores StatefulDataLoader state per DP rank, maintains the latest-step tracker file,
# optional LoRA metadata, optional HDFS copy, and resolves the resume path from resume_mode.
# Checkpoint folders are named "global_step_<N>" under default_local_dir.

import json
import logging
import os
import re
from enum import Enum

import torch

import verl.utils.hdfs_io as hdfs_io
from verl.single_controller import WorkerGroup
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.logger import log_with_rank
from verl.workers.engine import BaseEngine


def extract_step(path):
    # [FUNCTION DOC]
    # HELPER: Extracts N from the first "global_step_<N>" substring in a path. Inputs: path string.
    # Outputs: int or None if not found.
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class OrchestrationMode(Enum):
    # [CLASS DOC]
    # How the trainer is orchestrated: SPMD (every process runs the trainer, uses torch.distributed) or RAY (single driver).
    SPMD = 0
    RAY = 1


class CheckpointHandler:
    # [CLASS DOC]
    # Checkpoint handler for one model. Holds directories, retention, resume settings, the engine/worker group, the train
    # dataloader, orchestration mode and optional LoRA metadata, plus rank info used to decide which process writes what.
    # KNOWN ISSUE: docstring says it is expected to be used SPMD style, but OrchestrationMode.RAY is also supported.
    """
    Checkpoint handler handles the path, global_step of a checkpoint folder.
    Currently, it only works with a single model.
    We can expand it to support multiple models. It is expected to be used with SPMD style (e.g., torchrun)
    """

    def __init__(
        self,
        engine: BaseEngine | WorkerGroup,
        train_dataloader,
        *,
        default_local_dir,
        max_ckpt_to_keep=None,
        default_hdfs_dir=None,
        resume_mode="auto",
        resume_from_path=None,
        mode=OrchestrationMode.SPMD,
        lora_train_meta=None,
    ):
        # [FUNCTION DOC]
        # Behavior: stores all settings. In SPMD mode, reads global rank from torch.distributed and the engine's
        # is-model-parallel-output-source flag and DP rank; in RAY mode uses rank 0, output source True and DP rank 0.
        # Inputs: engine - BaseEngine or WorkerGroup; train_dataloader - stateful loader; keyword-only default_local_dir,
        # max_ckpt_to_keep, default_hdfs_dir, resume_mode ("auto"/"disable"/"resume_path"), resume_from_path, mode,
        # lora_train_meta - dict or None. Outputs: none. Exceptions: ValueError for unknown mode.
        self.default_local_dir = default_local_dir
        self.max_ckpt_to_keep = max_ckpt_to_keep
        self.default_hdfs_dir = default_hdfs_dir
        self.resume_mode = resume_mode
        self.resume_from_path = resume_from_path
        self.engine = engine
        self.train_dataloader = train_dataloader
        self.mode = mode
        self.lora_train_meta = lora_train_meta

        if self.mode == OrchestrationMode.SPMD:
            self.rank = torch.distributed.get_rank()
            self.is_mp_src_rank_with_outputs = self.engine.is_mp_src_rank_with_outputs()
            self.dp_rank = self.engine.get_data_parallel_rank()
        elif self.mode == OrchestrationMode.RAY:
            self.rank = 0
            self.is_mp_src_rank_with_outputs = True
            self.dp_rank = 0
        else:
            raise ValueError(f"Unknown {self.mode=}")

    def save_checkpoint(self, step):
        # [FUNCTION DOC]
        # Behavior: saves step N to default_local_dir/global_step_N: engine saves model/optimizer with max_ckpt_to_keep;
        # rank 0 writes lora_train_meta.json if metadata given; each model-parallel output-source rank saves its dataloader
        # state to data_<dp_rank>.pt; rank 0 atomically updates the latest-step tracker (write temp file then rename) and copies
        # the folder to HDFS if configured; SPMD mode ends with a barrier.
        # Inputs: step - int. Outputs: none. Side effects: files, prints, HDFS, collective barrier.
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.default_local_dir, f"global_step_{step}")
        if self.rank == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = self.max_ckpt_to_keep

        # Use checkpoint manager to save
        self.engine.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state. Note that we only save the iterator in the train_dataloader.
        # So it's identical in each dp rank.
        if self.rank == 0 and self.lora_train_meta is not None:
            local_mkdir_safe(local_global_step_folder)
            lora_meta_path = os.path.join(local_global_step_folder, "lora_train_meta.json")
            with open(lora_meta_path, "w", encoding="utf-8") as f:
                json.dump(self.lora_train_meta, f, ensure_ascii=False, indent=4)
            print(f"Saved LoRA rank/alpha metadata to: {lora_meta_path}")

        if self.is_mp_src_rank_with_outputs:
            dp_rank = self.dp_rank
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, f"data_{dp_rank}.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

        if self.rank == 0:
            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.rank == 0 and self.default_hdfs_dir:
            hdfs_io.makedirs(self.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.default_hdfs_dir, dirs_exist_ok=True)

        if self.mode == OrchestrationMode.SPMD:
            torch.distributed.barrier()

    def load_checkpoint(self):
        # [FUNCTION DOC]
        # Behavior: resolves resume path; returns 0 if none. Extracts the step from the path (warns and returns 0 if absent),
        # records resume_global_step, loads engine checkpoint and dataloader state. Inputs: none. Outputs: int resume step.
        # Exceptions: propagates assertion/ValueError from path resolution and engine load errors.
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.engine.load_checkpoint(checkpoint_path)
        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        # [FUNCTION DOC]
        # HELPER: Loads data_<dp_rank>.pt from the checkpoint into the train dataloader if present (CPU map, full unpickling),
        # otherwise logs a warning. Inputs: checkpoint_path. Outputs: none.
        """Load dataloader state from checkpoint"""
        dp_rank = self.dp_rank
        dataloader_path = os.path.join(checkpoint_path, f"data_{dp_rank}.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.rank,
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        # [FUNCTION DOC]
        # HELPER: "disable" -> None; "auto" -> resume_from_path if given (asserted to exist and contain "global_step_"), else
        # the latest checkpoint in default_local_dir; "resume_path" -> resume_from_path with the same assertions.
        # Outputs: path or None. Exceptions: AssertionError on invalid path; ValueError for unknown mode
        # (TypeError if resume_path mode with no path, from os.path.exists(None)).
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = self.resume_mode
        resume_from_path = self.resume_from_path

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        # [FUNCTION DOC]
        # HELPER: Returns None if default_local_dir does not exist; else the latest checkpoint path via the tracker file
        # (rank 0 prints it). Outputs: path or None.
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.rank == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint
