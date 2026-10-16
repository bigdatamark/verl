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
# Precision/dtype name helpers. Defines lookup lists of accepted aliases for fp16 (HALF_LIST),
# fp32 (FLOAT_LIST) and bf16 (BFLOAT_LIST), each mixing ints, strings and torch dtypes, and the
# PrecisionType class that classifies and converts between these aliases and torch dtypes.
"""
Adapted from Cruise.
"""

import torch

HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    # [CLASS DOC]
    # Namespace of precision-name constants (HALF="16", FLOAT="32", FULL="64", BFLOAT="bf16", MIXED="mixed")
    # plus static helpers for classifying precision aliases and converting them to/from torch dtypes.
    # KNOWN ISSUE: this is a plain class, not an Enum, so the docstring examples (HALF == 16) are False, and
    # supported_type / supported_types iterate over the class itself, which raises TypeError (a plain class is not iterable).
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: str | int) -> bool:
        # [FUNCTION DOC]
        # Intended to report whether precision equals any declared precision value.
        # Inputs: precision - string or int. Outputs: bool.
        # KNOWN ISSUE: iterating PrecisionType (not an Enum) raises TypeError at call time.
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        # [FUNCTION DOC]
        # Intended to return the list of declared precision values.
        # Outputs: list of strings.
        # KNOWN ISSUE: iterates the non-Enum class and accesses value, so it raises TypeError at call time.
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        # [FUNCTION DOC]
        # Returns True if precision is one of the fp16 aliases in HALF_LIST. No side effects.
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        # [FUNCTION DOC]
        # Returns True if precision is one of the fp32 aliases in FLOAT_LIST. No side effects.
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        # [FUNCTION DOC]
        # Returns True if precision is one of the bf16 aliases in BFLOAT_LIST. No side effects.
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision):
        # [FUNCTION DOC]
        # Converts a precision alias to the torch dtype: HALF_LIST -> float16, FLOAT_LIST -> float32, BFLOAT_LIST -> bfloat16.
        # Inputs: precision - int, string or torch dtype. Outputs: torch.dtype.
        # Exceptions: RuntimeError for unrecognized precision.
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def to_str(precision):
        # [FUNCTION DOC]
        # Converts a torch dtype into its short string name: float16 -> "fp16", float32 -> "fp32", bfloat16 -> "bf16".
        # Inputs: precision - torch dtype. Outputs: string.
        # Exceptions: RuntimeError for any other value.
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")
