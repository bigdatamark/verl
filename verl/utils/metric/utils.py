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
# Metric reduction helpers: a key-name-based reducer for metric dicts and a Metric accumulator
# class with explicit aggregation types, including aggregation across data-parallel ranks.
# NumericType is the tuple of accepted value types (int, float, Tensor, ndarray); Numeric is the
# matching union type alias.

"""
Metrics utils.
"""

from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch


def reduce_metrics(metrics: dict[str, Union["Metric", list[Any]]]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Behavior: for each entry: Metric values are replaced by their aggregate(); otherwise key
    # containing "max" -> np.max, containing "min" -> np.min (checked after "max"), else np.mean.
    # Inputs: metrics - mapping name to Metric or list of values.
    # Outputs: the same dict object with reduced values.
    # Side effects: mutates the input dict in place.
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, Metric):
            metrics[key] = val.aggregate()
        elif "max" in key:
            metrics[key] = np.max(val)
        elif "min" in key:
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


class AggregationType(Enum):
    # [CLASS DOC]
    # Enum of aggregation methods: MEAN, SUM, MIN, MAX (string values "mean", "sum", "min", "max").
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"


NumericType = int, float, torch.Tensor, np.ndarray
Numeric = int | float | torch.Tensor | np.ndarray


class Metric:
    # [CLASS DOC]
    # Accumulates numeric values with a fixed AggregationType.
    """
    A metric aggregator for collecting and aggregating numeric values.

    This class accumulates numeric values (int, float, or scalar tensors) and computes
    an aggregate statistic based on the specified aggregation type (MEAN, SUM, MIN, or MAX).

    Args:
        aggregation: The aggregation method to use. Can be a string ("mean", "sum", "min", "max")
            or an AggregationType enum value.
        value: Optional initial value(s) to add. Can be a single numeric value or a list of values.

    Example:
        >>> metric = Metric(aggregation="mean", value=1.0)
        >>> metric.append(2.0)
        >>> metric.append(3.0)
        >>> metric.aggregate()
        2.0
    """

    def __init__(self, aggregation: str | AggregationType, value: Optional[Numeric | list[Numeric]] = None) -> None:
        # [FUNCTION DOC]
        # Behavior: converts a string aggregation to AggregationType; stores an empty values list and
        # appends the initial value if given.
        # Inputs: aggregation - str or AggregationType; value - optional initial value.
        # Exceptions: ValueError for unknown aggregation strings or non-enum types; errors from append.
        # KNOWN ISSUE: the docstring says value may be a list, but append rejects lists (ValueError);
        # only scalars or a Metric work.
        if isinstance(aggregation, str):
            self.aggregation = AggregationType(aggregation)
        else:
            self.aggregation = aggregation
        if not isinstance(self.aggregation, AggregationType):
            raise ValueError(f"Unsupported aggregation type: {aggregation}")
        self.values = []
        if value is not None:
            self.append(value)

    def append(self, value: Union[Numeric, "Metric"]) -> None:
        # [FUNCTION DOC]
        # Behavior: a Metric is merged via extend; a tensor must have exactly one element and is converted
        # to a Python scalar; then the value must be int/float/Tensor/ndarray and is appended.
        # Inputs: value. Exceptions: ValueError for non-scalar tensors or unsupported types.
        # Side effects: mutates values.
        if isinstance(value, Metric):
            self.extend(value)
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Only scalar tensors can be converted to float")
            value = value.detach().item()
        if not isinstance(value, NumericType):
            raise ValueError(f"Unsupported value type: {type(value)}")
        self.values.append(value)

    def extend(self, values: Union["Metric", list[Numeric]]) -> None:
        # [FUNCTION DOC]
        # Behavior: if given a Metric, requires matching aggregation and uses its values; appends each value.
        # Inputs: values - Metric or iterable of numerics. Exceptions: ValueError on aggregation mismatch.
        if isinstance(values, Metric):
            if values.aggregation != self.aggregation:
                raise ValueError(f"Aggregation type mismatch: {self.aggregation} != {values.aggregation}")
            values = values.values
        for value in values:
            self.append(value)

    def aggregate(self) -> float:
        # [FUNCTION DOC]
        # Returns the aggregate of the stored values using this metric's aggregation (numpy scalar).
        return self._aggregate(self.values, self.aggregation)

    @classmethod
    def _aggregate(cls, values: list[Numeric], aggregation: AggregationType) -> float:
        # [FUNCTION DOC]
        # HELPER: Applies np.mean / np.sum / np.min / np.max to values according to aggregation.
        # Returns None for unmatched types.
        match aggregation:
            case AggregationType.MEAN:
                return np.mean(values)
            case AggregationType.SUM:
                return np.sum(values)
            case AggregationType.MIN:
                return np.min(values)
            case AggregationType.MAX:
                return np.max(values)

    @classmethod
    def aggregate_dp(cls, metric_lists: list["Metric"]) -> float:
        # [FUNCTION DOC]
        # Behavior: combines Metrics from DP ranks having equal value counts into a 2-D array
        # (ranks x values). SUM/MEAN: average across ranks per position, then apply the aggregation over
        # positions; MIN/MAX: apply over all values flattened. Uses the first metric's aggregation.
        # Inputs: metric_lists - non-empty list of Metric. Outputs: scalar.
        # Exceptions: ValueError for empty input or unequal lengths.
        if not metric_lists:
            raise ValueError("Cannot aggregate an empty list of metrics.")
        value_lists = [ml.values for ml in metric_lists]
        if not all(len(ls) == len(value_lists[0]) for ls in value_lists):
            raise ValueError(
                f"All Metric instances must have the same number of values "
                f"for dp aggregation: {[len(ls) for ls in value_lists]}"
            )
        value_arrays = np.array(value_lists)  # [num_dp, num_grad_accumulation]
        aggregation = metric_lists[0].aggregation
        match aggregation:
            case AggregationType.SUM | AggregationType.MEAN:
                return cls._aggregate(
                    values=np.mean(value_arrays, axis=0), aggregation=aggregation
                )  # mean over dp ranks
            case AggregationType.MIN | AggregationType.MAX:
                return cls._aggregate(values=value_arrays.flatten(), aggregation=aggregation)  # min/max over all values

    @classmethod
    def from_dict(cls, data: dict[str, Numeric], aggregation: str | AggregationType) -> dict[str, "Metric"]:
        # [FUNCTION DOC]
        # Behavior: returns a dict mapping each key to a new Metric with the given aggregation and value.
        return {key: cls(value=value, aggregation=aggregation) for key, value in data.items()}

    def init_list(self) -> "Metric":
        # [FUNCTION DOC]
        # Returns a new empty Metric with the same aggregation.
        return Metric(aggregation=self.aggregation)
