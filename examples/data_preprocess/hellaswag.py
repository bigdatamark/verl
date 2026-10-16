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
# CLI script converting Rowan/hellaswag (train/validation/test) to verl parquet for
# log-likelihood multiple-choice evaluation. Rows without a label are filtered out.
# Each record: user prompt = cleaned "activity_label: ctx_a Ctx_b", ability "nlp",
# reward_model {style "model", eval "multiple_choice", ground_truth int label, choices cleaned endings}.
# Args: --local_dataset_path, --local_save_dir (default ~/data/hellaswag), deprecated --local_dir, --hdfs_dir.
# KNOWN ISSUE: the `instruction` string is defined but never used.
# KNOWN ISSUE: the save dir is not tilde-expanded or created before writing.
"""
Preprocess Hellaswag dataset.

"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def preprocess(text):
    # [FUNCTION DOC]
    # HELPER: Cleans HellaSwag text: strips whitespace, replaces " [title]" with ". ", removes
    # any bracketed segments (non-greedy), then collapses double spaces into one (single pass,
    # so runs of 3+ spaces may remain). Inputs: text. Outputs: cleaned string.
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/hellaswag", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "Rowan/hellaswag"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    instruction = "Please complete the following sentence.\n"

    def make_map_fn(split):
        # [FUNCTION DOC]
        # Returns a process_fn closure bound to the split name.
        def process_fn(doc, idx):
            # [FUNCTION DOC]
            # Builds the query from activity_label and contexts (ctx_b capitalized), cleans each ending,
            # converts the label to int, and returns the record. Exceptions: ValueError if label is not numeric.
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
            query = preprocess(doc["activity_label"] + ": " + ctx)
            choices = [preprocess(ending) for ending in doc["endings"]]
            gold = int(doc["label"])

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": query}],
                "ability": "nlp",
                "reward_model": {
                    "style": "model",
                    "eval": "multiple_choice",  # using loglikelihood
                    "ground_truth": gold,
                    "choices": choices,
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    # filter data that doesn't have a label
    train_dataset = train_dataset.filter(lambda x: len(x["label"]) > 0)
    val_dataset = val_dataset.filter(lambda x: len(x["label"]) > 0)
    test_dataset = test_dataset.filter(lambda x: len(x["label"]) > 0)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("validation"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_save_dir, "validation.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
