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
# Naive reward manager (registered as "naive"): scores samples one by one in the driver process with
# an optional per-sample wall-clock timeout implemented with SIGALRM.

import signal
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@contextmanager
def _score_timeout(seconds: Optional[float]):
    # [FUNCTION DOC]
    # HELPER: Context manager bounding the wrapped block's run time.
    # Behavior: with no/non-positive seconds, just runs the block. Otherwise installs a SIGALRM handler
    # raising TimeoutError and starts a real-time interval timer; if installing fails (non-main thread)
    # runs without a timeout. On exit cancels the timer and restores the previous handler.
    # Inputs: seconds - optional float timeout. Exceptions: TimeoutError from inside the block.
    # Side effects: temporarily replaces the process SIGALRM handler and itimer.
    """Raise ``TimeoutError`` if the wrapped block runs longer than ``seconds``.

    A single ``compute_score`` call on a pathological model output (e.g. a long,
    degenerate RL rollout that triggers catastrophic regex backtracking or a slow
    symbolic comparison) can run for a very long time. Because ``NaiveRewardManager``
    scores samples serially in the driver process, one such call blocks the entire
    training loop. This guard bounds the wall-clock time of each call.

    Implemented with ``SIGALRM``, so it only takes effect on the main thread. If a
    handler cannot be installed (e.g. called from a non-main thread), it degrades to
    a no-op so reward computation still runs -- just without the timeout.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        # [FUNCTION DOC]
        # HELPER: Signal handler; always raises TimeoutError with the configured seconds.
        raise TimeoutError(f"compute_score timed out after {seconds}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
    except ValueError:
        # Not in the main thread -> SIGALRM is unavailable; run without a timeout.
        yield
        return

    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    # [CLASS DOC]
    # Per-sample reward manager storing tokenizer, examine count, scorer (default rule-based), data
    # source key and optional per-sample scoring timeout.
    """The reward manager."""

    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", compute_score_timeout=None
    ) -> None:
        # [FUNCTION DOC]
        # Behavior: stores arguments; scorer falls back to the default. Outputs: none.
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            compute_score_timeout: Optional per-sample timeout (in seconds) for `compute_score`. If set, a single
                scoring call that exceeds this limit is aborted, assigned a reward of 0.0, and training continues,
                instead of blocking the whole loop. Defaults to None (no timeout, unchanged behavior).
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.compute_score_timeout = compute_score_timeout  # per-sample timeout (seconds) for compute_score

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: returns precomputed rm_scores if present. Otherwise, per sample: computes valid prompt
        # (last N prompt tokens, N = prompt-region mask sum) and valid response (first M tokens), decodes
        # both; reads ground truth, data source, extra info (default empty) and adds "num_turns" (from
        # "__num_turns__", default None) and rollout reward scores into it. Calls the scorer under the
        # timeout guard; on TimeoutError prints a message and uses 0.0. Dict scores contribute "score" as
        # reward and all items to extra info lists; scalars used directly. Writes reward at the last valid
        # response token. Prints up to num_examine samples per data source.
        # Inputs: data; return_dict. Outputs: reward tensor or dict with reward_tensor and extra info.
        # Side effects: mutates per-sample extra_info; prints; uses SIGALRM.
        # KNOWN ISSUE: same zero-length edge cases as other managers (whole prompt chosen; index -1 written);
        # a timeout fallback of 0.0 among dict scores leaves extra info lists misaligned.
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            try:
                with _score_timeout(self.compute_score_timeout):
                    score = self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                    )
            except TimeoutError:
                print(
                    f"[NaiveRewardManager] compute_score exceeded "
                    f"{self.compute_score_timeout}s for data_source={data_source}; "
                    f"assigning reward 0.0 and continuing."
                )
                score = 0.0

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
