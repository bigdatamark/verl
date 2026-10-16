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
# Reward for the Geometry3K dataset combining a format check and boxed-answer accuracy graded by
# the mathruler library.
import re

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict_str: str) -> float:
    # [FUNCTION DOC]
    # Returns 1.0 if the whole string matches "<think>...</think>" followed somewhere by a
    # "\boxed{...}" (dot matches newlines), else 0.0.
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    # [FUNCTION DOC]
    # Returns 1.0 if the answer is graded equal to ground_truth by mathruler, else 0.0.
    # Inputs: predict_str; ground_truth; use_boxed - if true, the answer is the extracted boxed
    # content, otherwise the whole predict_str.
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    # [FUNCTION DOC]
    # Weighted sum: (1 - format_score) * accuracy reward + format_score * format reward.
    # Inputs: predict_str, ground_truth, use_boxed, format_score (weight, default 0.1).
    # Outputs: float in [0, 1].
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
