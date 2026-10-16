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
# [PACKAGE DOC]
# PRIME code reward: executes a generated Python solution against APPS-style input/output test
# cases in local subprocesses (via utils.check_correctness / testing_util.run_test) and returns a
# pass/fail or fractional pass-rate score with per-test metadata.

import json
import traceback

from .utils import check_correctness as apps_check_correctness


def compute_score(completion, test_cases, continuous=False):
    # [FUNCTION DOC]
    # Scores a code completion against test cases.
    # Behavior: takes the text after the last "```python" fence up to the next "```" (whole text if no
    # fence). Parses test_cases from JSON if not a dict (parse errors are printed and ignored). First
    # runs all cases together (5s timeout); if every result is True returns (True, first metadata).
    # Otherwise splits into single-case tests; if continuous, runs up to the first 10 cases one by one
    # (10s timeout each), collecting metadata augmented with the case input, expected output and
    # result string, and computes the fraction of results that are exactly True.
    # Inputs: completion (model text), test_cases (dict or JSON string with "inputs"/"outputs" lists),
    # continuous - whether to compute a fractional score.
    # Outputs: tuple (success bool or float fraction, list of metadata dicts or None on error).
    # Exceptions: any exception in the per-case phase is caught, traceback printed, returns (False, None).
    # Side effects: spawns subprocesses executing untrusted code; prints errors.
    # KNOWN ISSUE: when continuous is False and the full run did not fully pass, metadata_list is never
    # assigned and the final return raises UnboundLocalError (success may also be unassigned if the
    # full run raised).
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = completion.split("```python")[-1].split("```")[0]
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x is True, res))
            if success:
                return success, metadata
        except Exception:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=10, debug=False)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= 9:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x is True, res_list)) / res_count
    except Exception:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list
