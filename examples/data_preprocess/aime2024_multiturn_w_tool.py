# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# CLI script that converts the BytedTsinghua-SIA/AIME-2024 dataset (already in verl RL format)
# into the multi-turn tool-use format used by the ReTool recipe, by adding tool kwargs for a
# "code_interpreter" tool to each row's extra_info. The module docstring wrongly mentions DAPO.
# Args: --local_dataset_path (load from local instead of the hub, config "default"),
# --local_save_dir (default ~/data/retool_aime2024), deprecated --local_dir (overrides
# local_save_dir with a warning), --hdfs_dir (optional copy destination).
# Outputs: only the train split, written to <local_save_dir>/train.parquet; optionally copied to HDFS.
# KNOWN ISSUE: module docstring says it preprocesses DAPO-Math-17k but the script loads AIME-2024.
# KNOWN ISSUE: local_save_dir is not tilde-expanded or created before writing parquet.
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/retool_aime2024", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_path = "BytedTsinghua-SIA/AIME-2024"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        # [FUNCTION DOC]
        # Builds the per-row mapping function for the datasets map call (split arg is unused).
        # Inputs: split - split name (ignored). Outputs: returns process_fn.
        def process_fn(example, idx):
            # [FUNCTION DOC]
            # Rewrites one example in place: pops its existing extra_info dict, shallow-copies it, sets
            # need_tools_kwargs to true and tools_kwargs to a mapping for "code_interpreter" whose
            # create_kwargs carry the row's reward_model ground_truth, then reattaches it.
            # Inputs: example - row dict with "extra_info" and "reward_model"."ground_truth"; idx - row index (unused).
            # Outputs: returns the modified example. Exceptions: KeyError if required fields are missing.
            # Side effects: mutates the example dict.
            orig_extra_info = example.pop("extra_info")
            extra_info = orig_extra_info.copy()
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": example["reward_model"]["ground_truth"],
                    },
                },
            }
            example["extra_info"] = extra_info
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
