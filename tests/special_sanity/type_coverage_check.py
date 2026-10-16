# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# CLI tool measuring type-annotation coverage of function definitions on lines changed versus
# origin/main (or on all lines with --all-lines), failing when the annotated ratio is below a threshold.
# Constants CHECK_SUCCESS/CHECK_WARNING/CHECK_FAILURE (0/1/-1) are per-node check results.

"""Custom type annotation check tool.
To inspect the type annotation for functions in the entire codebase, please run:
find verl -type f -name "*.py" | xargs -n 1 python3 tests/special_sanity/type_coverage_check.py --all-lines
--debug --target-file
"""

import argparse
import ast
import linecache
import subprocess
from pathlib import Path


def get_changed_files() -> list[Path]:
    # [FUNCTION DOC]
    # HELPER: Runs git diff --name-only --diff-filter=AM origin/main...HEAD and returns Paths of added/modified .py files.
    # Side effects: spawns a git subprocess; git errors are not checked (returns empty list on failure output).
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", "origin/main...HEAD"], stdout=subprocess.PIPE, text=True
    )
    return [Path(f) for f in result.stdout.splitlines() if f.endswith(".py")]


def get_changed_lines(file_path: Path) -> set[int]:
    # [FUNCTION DOC]
    # HELPER: Runs git diff -U0 origin/main...HEAD for one file and parses each hunk header line ("@@ ...").
    # Behavior: for each whitespace-separated token starting with '+': "+start,count" adds the range
    # start..start+count-1, "+start" adds that single line; parse failures are silently ignored.
    # Inputs: file_path. Outputs: set of 1-based new-file line numbers. Side effects: git subprocess.
    # KNOWN ISSUE: tokens of the hunk's trailing context text that start with '+' are also parsed (mostly swallowed by the try/except, but numeric ones would add bogus lines).
    result = subprocess.run(
        ["git", "diff", "-U0", "origin/main...HEAD", "--", str(file_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    lines: set[int] = set()
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            for part in line.split():
                try:
                    if part.startswith("+") and "," in part:
                        start, count = map(int, part[1:].split(","))
                        lines.update(range(start, start + count))
                    elif part.startswith("+") and "," not in part:
                        lines.add(int(part[1:]))
                except Exception:
                    # (vermouth1992) There are many edge cases here because + can be in the changed program
                    pass
    return lines


CHECK_SUCCESS = 0
CHECK_WARNING = 1
CHECK_FAILURE = -1


def should_check_type(arg_name: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: Returns False for argument names self/cls or names starting with '*', else True.
    if arg_name in ("self", "cls"):
        return False
    if arg_name.startswith("*"):
        return False
    return True


def has_type_annotations(node: ast.AST, debug: bool = False) -> int:
    # [FUNCTION DOC]
    # HELPER: Classifies an AST node. For a (sync) FunctionDef: success if it has *args or **kwargs, if it is
    # private (name starts with '_'), or if every checked positional arg has an annotation and a return
    # annotation exists; otherwise failure (prints the node and arg annotations when debug). Any other node
    # type returns success.
    # Outputs: CHECK_SUCCESS or CHECK_FAILURE (CHECK_WARNING is never returned).
    # KNOWN ISSUE: keyword-only and positional-only args are not checked, and async functions are never checked.
    if isinstance(node, ast.FunctionDef):
        is_private = node.name.startswith("_")
        if node.args.vararg is not None or node.args.kwarg is not None:
            return CHECK_SUCCESS
        has_ann = (
            all(arg.annotation is not None for arg in node.args.args if should_check_type(arg.arg))
            and node.returns is not None
        )
        if has_ann or is_private:
            return CHECK_SUCCESS
        else:
            if debug:
                print(node, [(arg.annotation, arg.arg) for arg in node.args.args if should_check_type(arg.arg)])
            return CHECK_FAILURE
    return CHECK_SUCCESS


def check_file(
    file_path: Path, changed_lines: set[int], debug: bool = False
) -> tuple[int, int, list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    # [FUNCTION DOC]
    # HELPER: Parses a file and, for every FunctionDef/Assign/AnnAssign node whose start line is in
    # changed_lines, counts it in total and classifies it via has_type_annotations; success/warning count as
    # annotated (warnings recorded), failures recorded with their stripped source line.
    # Inputs: file_path; changed_lines - collection of line numbers; debug flag.
    # Outputs: (annotated, total, warning_lines, failure_lines), each line entry being (path, lineno, source text).
    # Exceptions: OSError on read, SyntaxError on unparsable source. Side effects: reads the file.
    # Note: Assign/AnnAssign nodes always count as annotated, inflating the ratio.
    with open(file_path) as f:
        source: str = f.read()
    tree = ast.parse(source, filename=str(file_path))
    annotated = 0
    total = 0
    warning_lines: list[tuple[Path, int, str]] = []
    failure_lines: list[tuple[Path, int, str]] = []

    for node in ast.walk(tree):
        if hasattr(node, "lineno") and node.lineno in changed_lines:
            if isinstance(node, ast.FunctionDef | ast.Assign | ast.AnnAssign):
                total += 1
                result = has_type_annotations(node, debug)
                if result == CHECK_SUCCESS or result == CHECK_WARNING:
                    annotated += 1
                    if result == CHECK_WARNING:
                        warning_lines.append(
                            (file_path, node.lineno, linecache.getline(str(file_path), node.lineno).strip())
                        )
                else:
                    source_line = linecache.getline(str(file_path), node.lineno).strip()
                    failure_lines.append((file_path, node.lineno, source_line))

    return annotated, total, warning_lines, failure_lines


def main() -> None:
    # [FUNCTION DOC]
    # CLI entry point. Options: --threshold (float, default 0.3), --target-file (single file, else changed
    # files from git), --all-lines (check every line), --debug.
    # Behavior: skip any path containing "tests/"; gather counts per file; ratio = annotated/total (1.0 if
    # nothing checked); print summary, warnings and failures; if ratio < threshold print exemption rules
    # and raise, otherwise print success.
    # Exceptions: raises Exception when coverage is below threshold. Side effects: stdout, git subprocesses, file reads.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Minimum ratio of annotated lines required (0.0 - 1.0)"
    )
    parser.add_argument("--target-file", type=str, default=None, help="Path to the Python source file to analyse")
    parser.add_argument(
        "--all-lines",
        action="store_true",
        help="Check all lines in the file instead of only changed lines based on git",
    )
    parser.add_argument("--debug", action="store_true", help="Add debugging logs")
    args = parser.parse_args()

    total_changed = 0
    total_annotated = 0
    all_warnings: list[tuple[Path, int, str]] = []
    all_failures: list[tuple[Path, int, str]] = []

    target_files = [args.target_file] if args.target_file is not None else get_changed_files()
    for fpath in target_files:
        if "tests/" in str(fpath):
            continue
        if args.all_lines:
            changed_lines = [i + 1 for i in range(len(open(fpath).readlines()))]
        else:
            changed_lines = get_changed_lines(fpath)
        annotated, total, warning_lines, failure_lines = check_file(fpath, changed_lines, args.debug)
        total_annotated += annotated
        total_changed += total
        all_warnings.extend(warning_lines)
        all_failures.extend(failure_lines)

    ratio = (total_annotated / total_changed) if total_changed else 1.0

    print(
        f"🔍 Type coverage on {'all' if args.all_lines else 'changed'} lines: "
        f"{total_annotated}/{total_changed} = {ratio:.2%}. Files inspected: {target_files}"
    )

    if all_warnings:
        print("\n⚠️ Suggest Improve: Lines missing type annotations for inputs and outputs:\n")
        for fname, lineno, line in all_warnings:
            print(f"{fname}:{lineno}: {line}")

    if all_failures:
        print("⚠️ [ERROR] Lines missing type annotations for inputs and outputs:\n")
        for fname, lineno, line in all_failures:
            print(f"{fname}:{lineno}: {line}")

    if ratio < args.threshold:
        print(
            f"Please add type annotations for inputs and outputs to meet threshold {args.threshold}. "
            f"Cases exempt from checking:"
        )
        print("1. Private methods.")
        print("2. Args with name in ('self', 'cls'), or *args / **kwargs")
        print("3. Files under tests/")
        raise Exception(f"\n❌ Type coverage below threshold ({args.threshold:.0%}).")
    else:
        if all_warnings or all_failures:
            print("")
        print("✅ Type annotation coverage acceptable.\n")


if __name__ == "__main__":
    main()
