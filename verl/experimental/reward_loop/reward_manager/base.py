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
# Abstract base for reward-loop reward managers, which score a single sample asynchronously.
# Defines the raw reward function type alias (any callable or none).
# Side effects on import: module logger with level from VERL_LOGGING_LEVEL (default WARN).

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


RawRewardFn = Callable[..., Any] | None


class RewardManagerBase(ABC):
    # [CLASS DOC]
    # Abstract base holding config, tokenizer, the raw compute_score callable and an event loop; has a per-class
    # one-time initialization hook and a default way of placing per-sample scores into an rm_scores tensor.
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer, compute_score: RawRewardFn):
        # [FUNCTION DOC]
        # Behavior: stores config, tokenizer and compute_score, obtains an event loop, and runs the class-level init hook.
        # Inputs: config; tokenizer; compute_score - callable or none.
        """Initialize reward manager.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.loop = get_event_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        # [FUNCTION DOC]
        # Behavior: sets the class's initialized flag on first call; later calls do nothing. Subclasses override to set up
        # shared state (e.g. rate limiters). Inputs: config, tokenizer (unused here).
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        # [FUNCTION DOC]
        # Behavior: abstract; subclasses compute and return a result mapping (reward_score, reward_extra_info) for one
        # sample. Exceptions: NotImplementedError if called on the base.
        raise NotImplementedError

    @classmethod
    def assemble_rm_scores(cls, data: DataProto, scores: list[float]) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: creates a float32 zero tensor shaped like responses and writes each sample's score at the index of its
        # last valid response token (response attention-mask sum minus one).
        # Inputs: data with prompts, responses, attention_mask; scores - one float per sample. Outputs: rm_scores tensor.
        """Assemble per-sample reward scores into the ``rm_scores`` tensor for a batch.

        Args:
            data: The concatenated batch passed to :meth:`run_single`.
                ``data.batch["prompts"]``, ``data.batch["responses"]`` and
                ``data.batch["attention_mask"]`` are expected to be present
                for the default implementation.
            scores: List of scalar reward scores, one per sample in ``data``.

        Returns:
            torch.Tensor: The ``rm_scores`` tensor with leading dimension equal to
            ``len(data)``.
        """
        prompt_length = data.batch["prompts"].size(1)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
        rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        rm_scores[torch.arange(rm_scores.size(0), device=rm_scores.device), valid_response_length - 1] = (
            rm_scores.new_tensor(scores)
        )
        return rm_scores
