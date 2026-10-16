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
# CPU tests for normalize_token_ids, which turns the many shapes of tokenizer
# output (plain lists/tuples, objects with tolist, numpy arrays, mappings or
# BatchEncoding-like objects with an input_ids field, a single-row batch, numpy
# scalars) into a flat list of Python ints, and raises TypeError for strings,
# mappings without input_ids, multi-row batches and non-integer elements.

import numpy as np
import pytest

from verl.utils.tokenizer import normalize_token_ids


class DummyBatchEncoding:
    # [CLASS DOC]
    # HELPER: Stand-in for a transformers BatchEncoding exposing only an input_ids attribute.
    def __init__(self, input_ids):
        # [FUNCTION DOC]
        # HELPER: Stores input_ids. Inputs: input_ids - any token-id container.
        self.input_ids = input_ids


class DummyToList:
    # [CLASS DOC]
    # HELPER: Stand-in for array/tensor-like objects exposing tolist.
    def __init__(self, data):
        # [FUNCTION DOC]
        # HELPER: Stores the data returned by tolist.
        self._data = data

    def tolist(self):
        # [FUNCTION DOC]
        # HELPER: Returns the stored data unchanged.
        return self._data


@pytest.mark.parametrize(
    ("tokenized_output", "expected"),
    [
        # transformers v4-style direct token ids
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        # common list-like outputs with tolist()/ndarray paths
        (DummyToList([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3], dtype=np.int64), [1, 2, 3]),
        # transformers v5-like mapping / BatchEncoding-style outputs
        ({"input_ids": [1, 2, 3]}, [1, 2, 3]),
        ({"input_ids": DummyToList([1, 2, 3])}, [1, 2, 3]),
        ({"input_ids": [[1, 2, 3]]}, [1, 2, 3]),
        (DummyBatchEncoding([1, 2, 3]), [1, 2, 3]),
        (DummyBatchEncoding(DummyToList([[1, 2, 3]])), [1, 2, 3]),
        # scalar item() support
        ([np.int64(1), np.int32(2), np.int16(3)], [1, 2, 3]),
    ],
)
def test_normalize_token_ids_valid_outputs(tokenized_output, expected):
    # [FUNCTION DOC]
    # Behavior: each parametrized valid tokenizer output normalizes to [1, 2, 3].
    # Inputs: tokenized_output - the raw output; expected - the flat int list.
    assert normalize_token_ids(tokenized_output) == expected


@pytest.mark.parametrize(
    "tokenized_output",
    [
        "not-token-ids",
        {"attention_mask": [1, 1, 1]},
        [[1, 2], [3, 4]],  # ambiguous batched ids should fail fast
        [1, object(), 3],
    ],
)
def test_normalize_token_ids_invalid_outputs(tokenized_output):
    # [FUNCTION DOC]
    # Behavior: each parametrized invalid output (a string, a mapping lacking
    # input_ids, a two-row batch, a list containing a non-integer object) raises
    # TypeError.
    with pytest.raises(TypeError):
        normalize_token_ids(tokenized_output)
