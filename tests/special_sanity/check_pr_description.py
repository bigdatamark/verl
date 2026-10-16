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
# GitHub Actions sanity script: fails if the first NUM_LINES (5) lines of the PR body, after stripping, are identical to
# the first lines of .github/PULL_REQUEST_TEMPLATE.md (under GITHUB_WORKSPACE or "."), meaning the author left the
# "What does this PR do?" placeholder unchanged. Reads the event payload from GITHUB_EVENT_PATH.

#!/usr/bin/env python3
import json
import os

# Number of lines to check
NUM_LINES = 5


# Custom exception types for clear error handling
class TemplateFileError(Exception):
    # [CLASS DOC]
    # HELPER: Raised when the PR template cannot be read.
    pass


class PRBodyLoadError(Exception):
    # [CLASS DOC]
    # HELPER: Raised when the GitHub event payload cannot be read or parsed.
    pass


class PRDescriptionError(Exception):
    # [CLASS DOC]
    # HELPER: Raised when the PR body still starts with the unmodified template.
    pass


# Path to the PR template file
template_file = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), ".github", "PULL_REQUEST_TEMPLATE.md")


def load_template(path):
    # [FUNCTION DOC]
    # HELPER: Reads up to NUM_LINES lines from the template, stripping each, and returns them as a list.
    # Exceptions: TemplateFileError wrapping any read error.
    # KNOWN ISSUE: the docstring says lines are kept "without stripping any characters", but each line is stripped.
    """
    Load only the first NUM_LINES of the PR template file as a list of lines,
    without stripping any characters.
    """
    lines = []
    try:
        with open(path, encoding="utf-8") as f:
            for _ in range(NUM_LINES):
                line = f.readline()
                if not line:
                    break
                lines.append(line.strip())
        return lines
    except Exception as e:
        raise TemplateFileError(f"Failed to read PR template (first {NUM_LINES} lines) at {path}: {e}") from e


def load_pr_body(event_path):
    # [FUNCTION DOC]
    # HELPER: Loads the event JSON and returns pull_request.body, or "" when missing or null.
    # Exceptions: PRBodyLoadError wrapping any read/parse error.
    try:
        with open(event_path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("pull_request", {}).get("body", "") or ""
    except Exception as e:
        raise PRBodyLoadError(f"Failed to read PR body from {event_path}: {e}") from e


def check_pr_description(body, template_lines):
    # [FUNCTION DOC]
    # HELPER: Compares the stripped first NUM_LINES lines of the body with the template lines; raises PRDescriptionError if
    # equal, otherwise prints both lists for debugging.
    """
    Compare the first NUM_LINES lines of the PR body to the template lines.
    If they match exactly, the placeholder was not modified.
    """
    pr_lines = body.splitlines(keepends=True)
    pr_first = [x.strip() for x in pr_lines[:NUM_LINES]]
    if pr_first == template_lines:
        raise PRDescriptionError(
            "It looks like you haven't updated the '### What does this PR do?' section. Please replace "
            "the placeholder text with a concise description of what your PR does."
        )
    else:
        print(pr_first)
        print(template_lines)


def main():
    # [FUNCTION DOC]
    # Behavior: raises OSError if GITHUB_EVENT_PATH is unset; loads template and body, runs the comparison, prints success.
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if not event_path:
        raise OSError("GITHUB_EVENT_PATH is not set.")

    template_lines = load_template(template_file)
    pr_body = load_pr_body(event_path)
    check_pr_description(pr_body, template_lines)

    print("✅ '### What does this PR do?' section has been filled out.")


if __name__ == "__main__":
    main()
