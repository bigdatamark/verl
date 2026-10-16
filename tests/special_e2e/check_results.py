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
# CLI checker used by e2e scripts: scans a training log for lines starting with "step", extracts the critic/rewards/mean
# metric from each, and asserts the best value exceeds --target (default 0.2). Prints the best reward and "Check passes".
# Exceptions: AssertionError (exit non-zero) when the target is not exceeded.

import argparse

import numpy as np


def extract_reward_from_line(line):
    # TODO: this function needs error handling
    # [FUNCTION DOC]
    # HELPER: Parses a metrics line formatted as "key:value" pairs separated by " - ".
    # Outputs: the float value of key "critic/rewards/mean", or negative infinity if absent or if any parsing error occurs
    # (e.g. a pair without exactly one colon, which aborts the whole line). Exceptions: none (all swallowed).
    try:
        key_vals = line.split(" - ")
        for key_val in key_vals:
            key, val = key_val.split(":")
            if key == "critic/rewards/mean":
                reward = float(val)
                return reward
        return -np.inf
    except Exception:
        return -np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--target", type=float, default=0.2, help="target reward score")

    args = parser.parse_args()

    with open(args.output_file) as f:
        output = f.read().split("\n")

    best_reward = -np.inf
    for line in output:
        if line.startswith("step"):
            reward = extract_reward_from_line(line)
            if reward > best_reward:
                best_reward = reward

    print(f"Best reward is {best_reward}")
    assert best_reward > args.target, f"Best reward must be greater than {args.target}. best_reward: {best_reward}"
    print("Check passes")
