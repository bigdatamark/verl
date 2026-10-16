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
# APPS-style test harness that executes generated Python code in-process (intended to run in a
# dedicated child process) against input/output cases. Supports two modes: call-based (a named
# function or Solution method called with JSON-decoded arguments) and standard-input (the program
# is wrapped into a function and fed stdin via mocks while stdout is captured). Timeouts use
# SIGALRM; destructive OS functions are disabled beforehand by reliability_guard.

import ast
import faulthandler
import json
import platform

# to run the solution files we're using a timing based approach
import signal
import sys
import traceback

# used for debugging to time steps
from datetime import datetime
from enum import Enum

# for capturing the stdout
from io import StringIO
from types import ModuleType

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import numpy as np


def module_from_string(name: str, code: str) -> ModuleType:
    # [FUNCTION DOC]
    # HELPER: Compiles code with filename "<string>", executes it in a fresh module namespace, registers
    # the module in the global module registry under name, and returns it.
    # Exceptions: propagates compile/runtime errors from the code. Side effects: executes arbitrary code.
    """Compile ``code`` into a new in-memory module registered under ``name``.

    Replaces ``pyext.RuntimeModule.from_string``, since ``pyext`` is unmaintained and cannot be
    installed on Python 3.12+. The ``<string>`` filename is what :func:`clean_traceback` looks for
    when trimming the traceback of a failing generated solution.
    """
    module = ModuleType(name)
    exec(compile(code, "<string>", "exec"), module.__dict__)
    sys.modules[name] = module
    return module


def truncatefn(s, length=300):
    # [FUNCTION DOC]
    # HELPER: Returns s unchanged if at most length chars; otherwise the first length/2 chars, the marker
    # "...(truncated) ..." and the last length/2 chars. Exceptions: AssertionError if s is not a str.
    assert isinstance(s, str)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    # [CLASS DOC]
    # Enumeration of test modes: call_based (0) and standard_input (1).
    call_based = 0
    standard_input = 1


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    # [CLASS DOC]
    # HELPER: Context manager that is also a list; captures everything written to stdout while active and
    # appends the captured text as one string element on exit.
    def __enter__(self):
        # [FUNCTION DOC]
        # Saves the current stdout, replaces it with an in-memory text buffer whose close is made a no-op,
        # and returns self.
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC]
        # Appends the buffer contents to self, deletes the buffer and restores the saved stdout.
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def only_int_check(val):
    # [FUNCTION DOC]
    # HELPER: True if val is an int.
    return isinstance(val, int)


def string_int_check(val):
    # [FUNCTION DOC]
    # HELPER: True if val is a str consisting only of digits.
    return isinstance(val, str) and val.isdigit()


def combined_int_check(val):
    # [FUNCTION DOC]
    # HELPER: True if val is an int or a digit-only string.
    return only_int_check(val) or string_int_check(val)


def clean_traceback(error_traceback):
    # [FUNCTION DOC]
    # HELPER: Trims a formatted traceback to start at the first 'File "<string>"' frame (generated code),
    # prefixed with a standard "Traceback (most recent call last):" header. If not found, the find index
    # is -1 and only the last character is kept after the header.
    file_start = error_traceback.find('File "<string>"')
    # print(file_start)
    error_traceback = "Traceback (most recent call last):\n  " + error_traceback[file_start:]
    return error_traceback


def run_test(in_outs, test=None, debug=False, timeout=15):
    # [FUNCTION DOC]
    # Executes generated code against all test cases and returns per-case results plus metadata of the
    # first failure.
    # Behavior:
    # - Calls reliability_guard first (irreversibly disables destructive functions in this process).
    # - Mode: standard_input if in_outs has no "fn_name", else call_based using that name.
    # - Prepends a fixed header of wildcard/standard imports and a raised recursion limit.
    # - Call-based: appends code, compiles under a timeout alarm; target is the module, or an instance of
    #   Solution when the code contains "class Solution". Standard-input: if the code ends with an
    #   if-name-equals-main block it is unwrapped; non-import lines are indented and wrapped into a
    #   function "code" (preceded by stdin/stdout aliases); compiled under alarm.
    # - Compile failures return ([-2], {error, traceback}); failure to find the method also returns -2.
    # - For each case: call-based decodes each input line as JSON and the expected output as JSON;
    #   string-keyed dicts are converted to int keys. The method is called with unpacked args; result
    #   (tuples turned into lists) is compared to expected, or expected[0], or tuple elements as lists.
    #   A mismatch returns immediately with {output, expected, inputs, error_message "Wrong Answer"};
    #   an exception/timeout appends -1 and returns {error, traceback}.
    # - Standard-input: joins list inputs/outputs with newlines, runs via call_method inside Capturing
    #   under alarm (exception -> -1 and return), splits captured stdout into lines and tries a cascade
    #   of comparisons: joined-string compare, line-list equality, stripped lines, re-splitting expected
    #   output into stripped non-empty lines, float closeness when not all integers, and finally
    #   whitespace-split set equality. A final mismatch returns "Wrong Answer" metadata.
    # Inputs: in_outs - dict with "inputs", "outputs", optional "fn_name" (mutated in place);
    # test - generated code string; debug - verbose prints; timeout - seconds per alarm.
    # Outputs: tuple (list of per-case results True/False/-1/-2, metadata dict; {} if all pass).
    # Exceptions: AssertionError if test is None.
    # Side effects: executes untrusted code, sets alarms, mutates in_outs, patches globals, prints.
    # KNOWN ISSUE: no SIGALRM handler is installed, so an expired alarm uses the default action and
    # terminates the whole process rather than raising; the outer process-level timeout handles it.
    # KNOWN ISSUE: if in_outs is falsy, which_type is never assigned and a NameError occurs.
    # KNOWN ISSUE: debug messages for call-based and standard-input runtime errors are swapped.
    # KNOWN ISSUE: the final set-based comparison converts output into sets but never re-compares, so
    # that last conversion has no effect on tmp_result.
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        raise AssertionError("should not happen: test code is none")
    elif test is not None:
        results = []
        sol = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"  # noqa: E501
        if debug:
            print(f"loading test code = {datetime.now().time()}")

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = module_from_string("tmp_sol", sol)
                tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                if debug:
                    print(f"type 0 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    # "error_code": -1,
                    # "error_message": "Compilation Error",
                    "traceback": clean_traceback(error_traceback),
                }
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it
            try:
                astree = ast.parse(test)
                last_block = astree.body[-1]
                if isinstance(last_block, ast.If):
                    condition = last_block.test
                    if ast.unparse(condition).strip() == "__name__ == '__main__'":
                        test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
            except Exception:
                pass

            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = module_from_string("tmp_sol", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                if debug:
                    print(f"type 1 compilation error = {e}")
                results.append(-2)
                return results, {
                    "error": repr(e),
                    # "error_code": -1,
                    # "error_message": "Compilation Error",
                    "traceback": clean_traceback(error_traceback),
                }
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except Exception:
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            error_info = sys.exc_info()
            print(f"unable to get function error = {error_info}")
            results.append(-2)
            return results, {
                "error": repr(error_info),
                # "error_code": -1,
                # "error_message": "Unable to extract code",
                "traceback": clean_traceback(error_traceback),
            }

        for index, inputs in enumerate(in_outs["inputs"]):
            raw_inputs = inputs
            raw_outputs = in_outs["outputs"][index]
            if which_type == CODE_TYPE.call_based:
                inputs = [json.loads(line) for line in inputs.split("\n")]
                in_outs["outputs"][index] = json.loads(in_outs["outputs"][index])

                truncate_line_size = 300 // (raw_inputs.count("\n") + 1)
                raw_inputs = "\n".join(
                    [truncatefn(line, truncate_line_size) for line in raw_inputs.strip().split("\n")]
                )
                raw_outputs = truncatefn(raw_outputs, 200)
            else:
                raw_inputs = truncatefn(raw_inputs)
                raw_outputs = truncatefn(raw_outputs, 200)
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
            except Exception:
                pass

            if debug:
                print(
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. "
                    f"type = {which_type}"
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)
                    raw_true_output = output

                    raw_true_output_copy = json.dumps(output)
                    raw_true_output_copy = truncatefn(raw_true_output_copy, 200)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    tmp_result = output == in_outs["outputs"][index]
                    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                    except Exception:
                        pass
                    results.append(tmp_result)
                    if tmp_result is not True:
                        return results, {
                            "output": raw_true_output_copy,
                            "expected": raw_outputs,
                            "inputs": raw_inputs,
                            # "error_code": -2,
                            "error_message": "Wrong Answer",
                        }
                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    error_traceback = traceback.format_exc()
                    faulthandler.disable()
                    if debug:
                        print(f"Standard input runtime error or time limit exceeded error = {e}")
                    results.append(-1)
                    return results, {
                        "error": repr(e),
                        "traceback": clean_traceback(error_traceback),
                    }
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, "
                        f"{type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )
            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                signal.alarm(timeout)
                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        error_traceback = traceback.format_exc()
                        print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                        results.append(-1)
                        return results, {
                            "error": repr(e),
                            "traceback": clean_traceback(error_traceback),
                        }
                    signal.alarm(0)
                raw_true_output = output[0]
                raw_true_output_copy = truncatefn(raw_true_output, 200)
                output = raw_true_output.splitlines()
                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, "
                                f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                                f"{output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, "
                                f"inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                    continue

                if passed and debug:
                    print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")
                    pass

                if tmp_result is True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                    in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")
                    pass

                if tmp_result is True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, "
                            f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                            f"{output == [in_outs['outputs'][index]]} {tmp_result=}"
                        )
                    else:
                        print(
                            f"@1 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, "
                            f"{type(inputs)}, {output == [in_outs['outputs'][index]]} {tmp_result=}"
                        )

                if debug:
                    print(f"{tmp_result=} @a")

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")
                    pass

                if debug:
                    print(f"{tmp_result=} @b")

                try:
                    all_ints = all(
                        combined_int_check(e1) and combined_int_check(e2)
                        for e1, e2 in zip(output, in_outs["outputs"][index], strict=True)
                    )
                    if not all_ints:
                        if debug:
                            print(
                                [
                                    combined_int_check(e1) and combined_int_check(e2)
                                    for e1, e2 in zip(output, in_outs["outputs"][index], strict=True)
                                ]
                            )
                        output_float = [float(e) for e in output]
                        gt_float = [float(e) for e in in_outs["outputs"][index]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                        )
                except Exception:
                    pass

                if debug:
                    print(f"{tmp_result=} @c")

                try:
                    if isinstance(output[0], list):
                        all_ints = all(
                            combined_int_check(e1) and combined_int_check(e2)
                            for e1, e2 in zip(output[0], in_outs["outputs"][index], strict=True)
                        )
                        if not all_ints:
                            output_float = [float(e) for e in output[0]]
                            gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                            tmp_result = tmp_result or (
                                (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                            )
                except Exception:
                    pass

                if tmp_result is True:
                    results.append(tmp_result)
                    continue

                if debug:
                    print(f"{tmp_result=} @d")
                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                if debug:
                    print(f"{tmp_result=} @e")

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")
                    continue

                if tmp_result is True:
                    results.append(tmp_result)
                    continue

                if debug:
                    print(f"{tmp_result=} @f")

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                if debug:
                    print(f"{tmp_result=} @g")

                if tmp_result is True and debug:
                    print("PASSED")

                results.append(tmp_result)
                if tmp_result is not True:
                    return results, {
                        "output": raw_true_output_copy,
                        "expected": raw_outputs,
                        "inputs": raw_inputs,
                        # "error_code": -2,
                        "error_message": "Wrong Answer",
                    }

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(
                            f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, "
                            f"inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, "
                            f"{output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(
                            f"@2 output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, "
                            f"{type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

                    print(f"results = {results}")

    return results, {}


def custom_compare_(output, ground_truth):
    # [FUNCTION DOC]
    # HELPER: When output is a list of lines, returns True if joining them with newlines (raw, or with
    # each line stripped) equals ground_truth after stripping both ends; otherwise False.
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    # [FUNCTION DOC]
    # HELPER: Equality of two strings after stripping leading/trailing whitespace.
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    # [FUNCTION DOC]
    # HELPER: Invokes a no-argument program function with stdin simulated from inputs (list joined by
    # newlines). Patches builtins.open (returns inputs), sys.stdin and its readline (successive lines),
    # readlines and read during the call; SystemExit is swallowed.
    # Outputs: the function's return value, or None on SystemExit.
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        # [FUNCTION DOC]
        # HELPER: Calls _method under the patches, returning its result or None if it raises SystemExit.
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    # [FUNCTION DOC]
    # Disables destructive functionality in the current process so generated code can't easily harm
    # the host (not a security sandbox). Optionally sets address-space, data (and on non-macOS, stack)
    # resource limits to maximum_memory_bytes. Disables faulthandler; sets OMP_NUM_THREADS=1; sets
    # exit/quit, many os functions (kill, system, remove, fork, chdir, getcwd, ...), shutil
    # rmtree/move/chown, subprocess.Popen and help to None; blocks importing ipdb, joblib, resource,
    # psutil, tkinter, subprocess and ctypes by setting their module registry entries to None.
    # Side effects: irreversible global mutation of the process.
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None  # 防止干扰repl评测
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None

    # Disable some built-in functions that can be destructive
    for mod in ["subprocess", "ctypes"]:
        sys.modules[mod] = None
