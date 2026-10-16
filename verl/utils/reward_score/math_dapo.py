# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
# [MODULE DOC]
# DAPO-style math reward (adapted from lm-eval-harness Hendrycks MATH utils). Extracts an answer
# from "Answer: ..." lines (Minerva style) or the last \boxed{...}, normalizes it with string
# substitutions/regexes, and compares it to the normalized ground truth. Reward is +1 / -1.
# Module constants SUBSTITUTIONS (ordered string replacements) and REMOVED_EXPRESSIONS (unit words
# and LaTeX fragments deleted) drive normalization.

import re
from typing import Optional


def last_boxed_only_string(string: str) -> Optional[str]:
    # [FUNCTION DOC]
    # HELPER: Finds the last occurrence of "\boxed{" and scans forward counting braces until the
    # matching close brace. Returns the substring from "\boxed{" through that brace, or None if no
    # "\boxed{" exists or braces never balance. Only the "\boxed{" form is recognized (not "\boxed ").
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Strips the leading "\boxed{" and trailing "}" from s and returns the inner content.
    # Exceptions: AssertionError if s does not start with "\boxed{" or does not end with "}".
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    # [FUNCTION DOC]
    # Normalizes an answer string for comparison. Steps in order: keep only text after the last "=";
    # apply each SUBSTITUTIONS replacement; delete each REMOVED_EXPRESSIONS entry; if a $...$ span
    # exists keep only its first inner content wrapped in $; unwrap \text{}, \textbf{}, \overline{}
    # (non-greedy) and \boxed{} (greedy); rewrite shorthand "fracab" to "frac{a}{b}" and "sqrta" to
    # "sqrt{a}"; delete "$"; if the result is digits after removing commas, drop the commas; strip
    # whitespace. Outputs: normalized string. No side effects.
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    # [FUNCTION DOC]
    # Extracts the last match of answer_pattern (default case-insensitive "Answer: <rest of line>"),
    # or "[INVALID]" if none, and normalizes it. The ground truth is normalized directly, or, if
    # gt_need_extract, after extracting its last boxed content.
    # Outputs: tuple (pred equals normalized gt, normalized pred).
    # Exceptions: if gt_need_extract and gt has no box, remove_boxed receives None and fails (TypeError).
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    # [FUNCTION DOC]
    # Checks only the tail of pred: the last 100 characters, or, when pause_tokens_index (exactly 4
    # indices) is given, from 100 characters before the last index onward. Extracts the last boxed
    # content and compares it to gt by exact string equality (no normalization).
    # Outputs: tuple (1 if equal else -1, extracted content or None).
    # Exceptions: AssertionError if pause_tokens_index length is not 4.
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    # [FUNCTION DOC]
    # Decides correctness. If strict_box_verify, uses is_correct_strict_box only. Otherwise tries
    # Minerva extraction; if it found an answer returns its result; else falls back to strict box
    # check and returns that if a box was found; else returns the Minerva result ("[INVALID]" pred).
    # Outputs: tuple (bool correct, extracted prediction string or None).
    # KNOWN ISSUE: annotated and documented as returning bool, but it returns a (bool, pred) tuple.
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    # Prefer Minerva (`Answer: ...`); fall back to `\boxed{}` when it fails.
    correct, pred = is_correct_minerva(solution_str, answer)
    if pred != "[INVALID]":
        return correct, pred

    box_correct, box_pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if box_pred is not None:
        return box_correct == 1, box_pred
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    # [FUNCTION DOC]
    # Truncates solution_str to its last 300 characters, runs verify, and builds the result.
    # Outputs: dict with "score" (1.0 if correct else -1.0), "acc" (bool) and "pred" (extracted answer).
    # KNOWN ISSUE: annotated/documented as returning float, but returns a dict.
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution (Minerva first, then `\boxed{}` fallback unless strict_box_verify)
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
