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
# Sanity test enforcing the documentation layout of the core trainer YAML config files:
# every key line needs a comment line directly above it, no inline comments, and a blank line after it.

import re
from pathlib import Path


def validate_yaml_format(yaml_lines):
    # [FUNCTION DOC]
    # HELPER: Scans YAML lines and collects format violations.
    # Behavior: iterate line by line, skipping blank lines; for each line matching an (indented) identifier
    # key followed by a colon: (1) if it is the first line or the previous line is not a comment, record
    # "Missing comment above line N"; (2) if the line contains '#' after the first colon and the line is not
    # itself a comment, record "Inline comment found"; (3) if a next line exists and is non-blank, record
    # "Missing blank line after line N". Line numbers are 1-based.
    # Inputs: yaml_lines - list of raw lines (with newlines). Outputs: list of error strings.
    # Exceptions: none. Side effects: none.
    # KNOWN ISSUE: the comment says the blank-line rule is skipped when the next line is deeper-indented, but the code enforces it for any non-blank next line (nested keys must thus be preceded by blank+comment).
    # KNOWN ISSUE: '#' inside a quoted value is treated as an inline comment.
    errors = []
    i = 0

    while i < len(yaml_lines):
        line = yaml_lines[i]
        stripped = line.strip()

        # Skip empty lines
        if stripped == "":
            i += 1
            continue

        # Match YAML keys like "field:" or "field: value"
        key_match = re.match(r"^(\s*)([a-zA-Z0-9_]+):", line)
        if key_match:
            # Check if there's a comment above
            if i == 0 or not yaml_lines[i - 1].strip().startswith("#"):
                errors.append(f"Missing comment above line {i + 1}: {line.strip()}")

            # Check for inline comment
            if "#" in line and not stripped.startswith("#"):
                comment_index = line.index("#")
                colon_index = line.index(":")
                if comment_index > colon_index:
                    errors.append(f"Inline comment found on line {i + 1}: {line.strip()}")

            # Check for blank line after this key line (unless next is a deeper indent)
            if i + 1 < len(yaml_lines):
                next_line = yaml_lines[i + 1]
                next_stripped = next_line.strip()

                # If next is not empty and not a deeper nested line, enforce blank line
                if next_stripped != "":
                    errors.append(f"Missing blank line after line {i + 1}: {line.strip()}")

        i += 1

    return errors


def test_trainer_config_doc():
    # [FUNCTION DOC]
    # Runs validate_yaml_format over a fixed list of trainer YAML paths (ppo_trainer, actor, dp_actor, critic,
    # dp_critic, ref, dp_ref, rollout, transfer_queue; relative to cwd), printing every error per file.
    # Exceptions: raises a generic Exception if any file had errors; FileNotFoundError if not run from repo root.
    # Side effects: reads files, prints to stdout.
    yamls_to_inspect = [
        "verl/trainer/config/ppo_trainer.yaml",
        "verl/trainer/config/actor/actor.yaml",
        "verl/trainer/config/actor/dp_actor.yaml",
        "verl/trainer/config/critic/critic.yaml",
        "verl/trainer/config/critic/dp_critic.yaml",
        "verl/trainer/config/ref/ref.yaml",
        "verl/trainer/config/ref/dp_ref.yaml",
        "verl/trainer/config/rollout/rollout.yaml",
        "verl/trainer/config/transfer_queue/transfer_queue.yaml",
    ]
    success = True
    for yaml_to_inspect in yamls_to_inspect:
        yaml_path = Path(yaml_to_inspect)  # path to your YAML file
        with open(yaml_path) as f:
            lines = f.readlines()

        validation_errors = validate_yaml_format(lines)
        if validation_errors:
            success = False
            print("YAML documentation format check failed:")
            print(f"Please read the top block of {yaml_to_inspect} to see format rules:\n")
            for err in validation_errors:
                print(" -", err)

    if not success:
        raise Exception("Please fix documentation format.")
    else:
        print("YAML format check passed ✅")
