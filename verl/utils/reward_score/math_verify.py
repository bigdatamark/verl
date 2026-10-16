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
# Math-Verify based reward. Verification runs in a lazily-created, shared spawn-context process
# pool (4 workers) so math_verify's signal-based timeouts work and calls can be bounded with a
# timeout. If math_verify is not installed, a placeholder TimeoutException is defined and an
# install hint is printed at import time.

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        # [CLASS DOC]
        # HELPER: Fallback exception type used only when math_verify is not installed.
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    # [FUNCTION DOC]
    # HELPER: Returns the module-global ProcessPoolExecutor, creating it on first use with
    # double-checked locking (4 workers, spawn context). Side effects: may start worker processes.
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    # [FUNCTION DOC]
    # HELPER: Runs in a pool worker. Parses gold with LaTeX extraction and prediction with expression
    # plus LaTeX extraction; if both yield candidates returns 1.0 if any (gold, pred) pair verifies,
    # else 0.0. Outputs: float. Exceptions: propagates math_verify errors to the caller via the future.
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    # [FUNCTION DOC]
    # Wraps ground_truth in "\boxed{...}", submits verification to the pool and waits up to timeout
    # seconds. Returns the verification score; timeout_score on futures timeout or math_verify
    # TimeoutException; 0.0 on any other exception (printed).
    # Side effects: creates the process pool on first call; prints errors.
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except Exception as e:
        print(f"Error in math_verify compute_score: {e}")
    return ret_score
