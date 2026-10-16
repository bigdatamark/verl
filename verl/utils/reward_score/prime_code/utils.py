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
# Process-isolated wrapper (borrowed from the codeparrot APPS metric) around testing_util.run_test
# that enforces a global timeout on top of run_test's per-case timeouts.

# Borrowed from: https://huggingface.co/spaces/codeparrot/apps_metric/blob/main/utils.py

import multiprocessing
import os
import sys
import traceback
from typing import Optional

from .testing_util import run_test


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    # [FUNCTION DOC]
    # HELPER: Target of the child process. Redirects stdout/stderr to devnull, runs run_test and
    # appends its result list and metadata to the shared lists. On exception prints a traceback (to
    # devnull) and appends a list of -1 per input plus an empty metadata dict.
    # Side effects: mutates the shared manager lists; replaces the process's sys.stdout/stderr.
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(in_outs: Optional[dict], generation, timeout=10, debug=True):
    # [FUNCTION DOC]
    # Runs _temp_run in a new process with manager-backed shared lists, waits timeout + 1 seconds and
    # kills the process if still alive. If no result was produced, treats every input as failed (-1)
    # and prints "global timeout" when debug.
    # Inputs: in_outs (dict with "inputs"/"outputs"), generation (code string), timeout (seconds per
    # test, also used for the global wait), debug.
    # Outputs: tuple (result list for the first run - True/False/-1/-2 per case, the shared metadata list).
    # Side effects: starts a multiprocessing Manager and a child process executing untrusted code.
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list
