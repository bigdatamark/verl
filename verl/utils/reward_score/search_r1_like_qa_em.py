# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py
# [MODULE DOC]
# Search-R1 style QA reward (adapted from Search-R1 qa_em): extracts the last <answer>...</answer>
# span and scores exact match (or substring match) against a list of golden answers after SQuAD
# normalization. ground_truth is a dict whose "target" holds a string or list of golden answers.

import random
import re
import string


def normalize_answer(s):
    # [FUNCTION DOC]
    # HELPER: SQuAD-style normalization: lowercase, remove punctuation, replace articles a/an/the with
    # spaces, collapse whitespace. Outputs: normalized string.
    def remove_articles(text):
        # [FUNCTION DOC]
        # HELPER: Replaces whole-word "a", "an", "the" with a space.
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # [FUNCTION DOC]
        # HELPER: Collapses runs of whitespace into single spaces and trims ends.
        return " ".join(text.split())

    def remove_punc(text):
        # [FUNCTION DOC]
        # HELPER: Removes all ASCII punctuation characters.
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # [FUNCTION DOC]
        # HELPER: Lowercases text.
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    # [FUNCTION DOC]
    # Returns 1 if the normalized prediction equals any normalized golden answer (a single string is
    # treated as a one-element list), else 0.
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    # [FUNCTION DOC]
    # Returns 1 if any normalized golden answer is a substring of the normalized prediction, else 0.
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    # [FUNCTION DOC]
    # Returns the stripped content of the last <answer>...</answer> span (dot matches newlines), or None
    # if there is none.
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    # [FUNCTION DOC]
    # HELPER: Returns tuple (count of "<answer>", count of "</answer>") in text.
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # [FUNCTION DOC]
    # Exact-match score: 0 if no answer extracted; if EM matches ground_truth["target"], returns score
    # (divided by 4 if more than 10 opening or closing answer tags appear, to penalize tag spam);
    # otherwise format_score. method is unused. Randomly (1/64) prints debug info.
    # Exceptions: KeyError/TypeError if ground_truth lacks "target". Side effects: occasional prints.
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth["target"]):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                score = score / 4
                return score
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    # [FUNCTION DOC]
    # Substring-match variant: 0 if no answer; score if subem_check succeeds; else format_score.
    # No tag-count penalty; method unused; random 1/64 debug prints.
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
