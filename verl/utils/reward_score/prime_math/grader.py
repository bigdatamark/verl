# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# Copyright (c) 2023 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2021 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright 2024 PRIME team and/or its affiliates
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
# Numeric and symbolic answer-equivalence checker (from Hendrycks MATH / ToRA / CRITIC / PRM800K).
# math_equal tries string, numeric (with percentage variants), interval/tuple/matrix element-wise,
# and finally sympy symbolic comparison, each sympy step guarded by timeout_limit.
"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence), and borrowed from:
- https://github.com/microsoft/ToRA/blob/main/src/eval/grader.py
- https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- https://github.com/openai/prm800k
"""

import contextlib
import math
import re
from math import isclose

# sympy related
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

# verl related
from verl.utils.py_functional import timeout_limit


def is_digit(s):
    # [FUNCTION DOC]
    # HELPER: Tries to parse s as a float after removing "{,}" (LaTeX thousands separator) or plain
    # commas. Outputs: tuple (True, float) on success, (False, None) on ValueError.
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer, pi) -> str:
    # [FUNCTION DOC]
    # HELPER: If answer is a string matching "$<number>" (prefix match) returns it without the leading
    # "$"; if it is "<number>%" or "<number>\%" returns it with percent signs removed; otherwise
    # applies handle_base then handle_pi. Output may be a string, int or number (not always str).
    # checking if answer is $<number> and removing $ in that case to compare
    if isinstance(answer, str) and bool(re.match(r"\$\d+(\.\d+)?", answer)):
        return answer[1:]

    # checking if answer is <number>% or <number>\\% and removing %
    if isinstance(answer, str) and (
        bool(re.match(r"^\d+(\.\d+)?%$", answer)) or bool(re.match(r"^\d+(\.\d+)?\\%$", answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    # handle base
    answer = handle_base(answer)

    # handle pi
    answer = handle_pi(answer, pi)

    return answer


def handle_base(x) -> str:
    # [FUNCTION DOC]
    # HELPER: For strings containing "_" (base subscript like "101_2"), keeps the part before the first
    # underscore and converts it to int via float; other inputs returned unchanged.
    # Exceptions: ValueError if the prefix is not numeric (propagates to math_equal's caller).
    # KNOWN ISSUE: the base is discarded, so the digits are interpreted as decimal.
    if isinstance(x, str) and "_" in x:
        # Due to base
        x = x.split("_")[0]
        x = float(x)
        return int(x)
    return x


def handle_pi(string, pi):
    # [FUNCTION DOC]
    # HELPER: For strings containing "\pi", replaces each occurrence with "*<pi>" after a digit or
    # "1*<pi>" otherwise, then tries to evaluate the string as an expression (errors suppressed).
    # Outputs: evaluated number, or the rewritten/unchanged string.
    # Side effects: evaluates arbitrary model-provided text with the interpreter's eval.
    if isinstance(string, str) and "\\pi" in string:
        # Find the first occurrence of "\pi"
        idx = string.find("\\pi")

        # Iterate over the string and find all occurrences of "\pi" with a valid previous character
        while idx != -1:
            if idx > 0 and string[idx - 1].isdigit():
                # Replace "\pi" with "*math.pi" if the previous character is a digit
                string = string[:idx] + f"*{pi}" + string[idx + 3 :]
            else:
                # Replace "\pi" with "1*math.pi" if the previous character is not a digit
                string = string[:idx] + f"1*{pi}" + string[idx + 3 :]

            # Find the next occurrence of "\pi"
            idx = string.find("\\pi", idx + 1)

        # Evaluate the expression using eval() function
        with contextlib.suppress(Exception):
            string = eval(string)

    return string


def math_equal(
    prediction: bool | float | str,
    reference: float | str,
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    pi: float = math.pi,
) -> bool:
    # [FUNCTION DOC]
    # Decides whether prediction equals reference.
    # Behavior: normalize both (with given pi); truncate long string predictions to 1000 chars;
    # (0) case-insensitive stripped or whitespace-free string equality; (1) if both parse as numbers,
    # True if prediction is close (rel_tol=tolerance) to reference, reference/100 or reference*100
    # (only reference if include_percentage false), else returns False; empty non-zero prediction ->
    # False; (2) stringify, convert sympy Interval notation to brackets, strip outer brackets when
    # bracket types are compatible and remove braces/parens, compare; element-wise recursion for
    # same-bracket intervals, comma-separated lists (returns that result if same length), "Point(...)"
    # vs tuple, and pmatrix references vs Matrix(...) or nested list predictions; finally symbolic_equal.
    # Inputs: prediction, reference, include_percentage, tolerance, timeout (seconds for sympy steps),
    # pi (value substituted for \pi).
    # Outputs: bool. Exceptions: errors from normalize (e.g. handle_base) and eval of list-like
    # predictions in the matrix branch can propagate.
    # Side effects: eval of prediction text; prints sympy timeouts.
    # KNOWN ISSUE: recursive calls do not pass timeout or pi, so they use defaults.
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """

    prediction = normalize(prediction, pi)
    reference = normalize(reference, pi)

    if isinstance(prediction, str) and len(prediction) > 1000:  # handling weird corner-cases
        prediction = prediction[:1000]

    # 0. string comparison
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:  # 1. numerical equal
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]
            # number questions
            gt_result = [reference / 100, reference, reference * 100] if include_percentage else [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=True)
            ]
        ):
            return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            return bool(
                all(
                    [
                        math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance)
                        for i in range(len(pred_parts))
                    ]
                )
            )

    # if we have point == tuple of values
    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts) and all(
            [
                math_equal(pred_pt, ref_pt, include_percentage, tolerance)
                for pred_pt, ref_pt in zip(pred_parts, ref_parts, strict=False)
            ]
        ):
            return True

    # if reference is a matrix
    if r"\begin{pmatrix}" in reference and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items) and all(
                [
                    math_equal(pred, ref, include_percentage, tolerance)
                    for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
                ]
            ):
                return True
        except Exception:
            pass
    elif r"\begin{pmatrix}" in reference and prediction.startswith("[") and prediction.endswith("]"):
        if isinstance(eval(prediction), list):
            try:
                pred_matrix = eval(prediction)
                # ref_matrix_items = reference.split()[1:-1:2]
                ref_matrix_items = (
                    reference.removeprefix(r"\\begin{pmatrix}")
                    .removeprefix(r"\begin{pmatrix}")
                    .removesuffix(r"\\end{pmatrix}")
                    .removesuffix(r"\end{pmatrix}")
                )
                ref_matrix_items = ref_matrix_items.split("\\")
                ref_matrix_items = [row.split("&") if "&" in row else row for row in ref_matrix_items]
                if len(pred_matrix) == len(ref_matrix_items) and all(
                    [
                        math_equal(pred, ref, include_percentage, tolerance)
                        for ref, pred in zip(ref_matrix_items, pred_matrix, strict=False)
                    ]
                ):
                    return True
            except Exception:
                pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    # [FUNCTION DOC]
    # Parses a and b with sympy parse_expr, else parse_latex (each under timeout; failures fall back to
    # the raw string), then returns True if simplify(a - b) == 0, or if numeric evaluations are close
    # (rel_tol=tolerance); each step under timeout with timeouts printed and other errors ignored.
    # Outputs: bool.
    def _parse(s):
        # [FUNCTION DOC]
        # HELPER: Returns the first successful parse of s by parse_expr then parse_latex, or s unchanged.
        for f in [parse_expr, parse_latex]:
            try:
                with timeout_limit(seconds=timeout):
                    return f(s)
            except TimeoutError:
                print(f"Parsing timed out for {s}")
                continue
            except Exception:
                continue
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with timeout_limit(seconds=timeout):
            if simplify(a - b) == 0:
                return True
    except TimeoutError:
        print(f"Simplification timed out for {a} - {b}")
        pass
    except Exception:
        pass

    try:
        with timeout_limit(seconds=timeout):
            if isclose(N(a), N(b), rel_tol=tolerance):
                return True
    except TimeoutError:
        print(f"Numerical evaluation timed out for {a}, {b}")
        pass
    except Exception:
        pass
    return False


def format_intervals(prediction):
    # [FUNCTION DOC]
    # HELPER: Converts sympy-style "Interval(a, b)", "Interval.Ropen", "Interval.Lopen", "Interval.open"
    # strings into bracket notation "[a, b]", "[a, b)", "(a, b]", "(a, b)"; other strings unchanged.
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":  # Intarval(a, b) == [a, b]
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":  # Intarval.Ropen(a, b) == [a, b)
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":  # Intarval.Lopen(a, b) == (a, b]
                return f"({inner_content}]"
            elif key == "Interval.open(":  # Intarval.open(a, b) == (a, b)
                return f"({inner_content})"

    return prediction
