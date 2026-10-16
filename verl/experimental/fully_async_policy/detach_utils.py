# Copyright 2025 Meituan Ltd. and/or its affiliates
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
# Module detach_utils: utilities for fully async training: the RolloutSample record passed from rollouter to
# trainer, per-sample generation data preparation, assembly of a training batch (with timing, staleness and
# partial-rollout statistics) from many rollout samples, a MetricsAggregator that merges per-micro-step metrics over a
# weight-sync cycle, and helpers for creating asyncio tasks whose exceptions are reported.
import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class RolloutSample:
    # [CLASS DOC]
    # Dataclass for one generated sample group: full_batch (generated DataProto), sample_id, epoch and a rollout_status
    # dict of rollouter-side statistics.
    """Enhanced rollout sample containing both original batch info and AgentLoopOutput"""

    # Original batch information
    full_batch: Any

    # Metadata
    sample_id: str
    epoch: int

    # Processing metadata
    rollout_status: dict[str, Any]


def prepare_single_generation_data(batch_dict, config) -> DataProto:
    # [FUNCTION DOC]
    # Behavior: builds a DataProto from the batch dict; the pop lists are currently empty so nothing is removed; if
    # multi-turn is disabled sets every agent_name to "single_turn_agent"; repeats the batch rollout.n times interleaved.
    # Inputs: batch_dict - single-sample dataset dict; config - full config. Outputs: repeated DataProto.
    # KNOWN ISSUE: docstring says it returns a tuple (original batch, gen data) but a single DataProto is returned.
    """
    Similar to the logic of ray_trainer._prepare_generate_batch, but for a single sample.
    Separate the data used for generation from the original data.

    Returns:
        tuple: (original_batch_dict, gen_data_for_single_sample)
    """

    full_batch = DataProto.from_single_dict(batch_dict)

    batch_keys_to_pop = []
    non_tensor_batch_keys_to_pop = []

    existing_batch_keys = [k for k in batch_keys_to_pop if k in full_batch.batch.keys()]
    existing_non_tensor_keys = [k for k in non_tensor_batch_keys_to_pop if k in full_batch.non_tensor_batch.keys()]

    if existing_batch_keys or existing_non_tensor_keys:
        full_batch.pop(
            batch_keys=existing_batch_keys,
            non_tensor_batch_keys=existing_non_tensor_keys,
        )

    # Setting selected agent, that supports partial
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        full_batch.non_tensor_batch["agent_name"] = np.array(["single_turn_agent"] * len(full_batch), dtype=object)

    # Add global step count to generated data
    full_batch = full_batch.repeat(repeat_times=config.actor_rollout_ref.rollout.n, interleave=True)
    return full_batch


def addition_process(output: DataProto):
    # [FUNCTION DOC]
    # HELPER: Behavior: pops per-sample metrics from meta_info and stores lists of generate_sequences and tool_calls times
    # in non_tensor_batch under processing_times and tool_calls_times. Outputs: the same (mutated) DataProto.
    # Exceptions: KeyError if metrics missing.
    # KNOWN ISSUE: the stored values are plain Python lists rather than numpy arrays, unlike other non-tensor fields, which
    # may break DataProto operations that expect arrays.
    """collect metirics"""
    metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
    processing_times_list = [item["generate_sequences"] for item in metrics]
    tool_calls_times_list = [item["tool_calls"] for item in metrics]
    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list
    return output


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None
) -> DataProto:
    # [FUNCTION DOC]
    # Behavior: raises if empty; prefixes the first sample's rollout_status keys with "fully_async/"; post-processes each
    # sample's batch and concatenates them; computes response_mask if missing; optionally calls balance_batch; records per-row
    # token counts as global_token_num; computes processing-time stats (avg/max/min/p50/p95/p99, prefixed) and tool-call
    # time stats (max/min/mean); computes partial-rollout stats from min/max global steps per trajectory (count, ratio and
    # max span of parameter-version differences); updates meta_info with parameter version diversity, per-trajectory
    # versions and all stats. Prints progress and elapsed time.
    # Inputs: rollout_samples; tokenizer (unused); config (unused); balance_batch - optional callable taking (batch, metrics).
    # Outputs: assembled DataProto. Exceptions: ValueError on empty input; KeyError if required fields are missing.
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch)
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch


class MetricsAggregator:
    # [CLASS DOC]
    # Accumulates numeric metrics across steps and aggregates them per metric using explicit rules or name heuristics;
    # also derives throughput, idle ratio and resource-utilization metrics. Tracks hybrid/standalone GPU counts.
    """Metrics aggregator, used to combine metrics from multiple training steps"""

    def __init__(self, total_gpus: int, hybrid_gpus: int = 0, standalone_gpus: int = 0):
        # [FUNCTION DOC]
        # Behavior: initializes empty value lists, sample counts, timestamps, step count, GPU counts and aggregation rules.
        # Inputs: total_gpus; hybrid_gpus; standalone_gpus.
        # Store all values ​​for each metric
        self.metric_values: dict[str, list[float]] = defaultdict(list)
        # Store the number of samples at each step for weighted averaging
        self.sample_counts: list[int] = []
        # Store the timestamp of each step for time-related calculations
        self.timestamps: list[float] = []
        # Step Count
        self.step_count = 0
        # total num gpus used
        self.total_gpus = total_gpus
        # Number of GPUs that switch between rollout/train (hybrid, i.e. trainer-node GPUs
        # co-hosting rollout replicas under dynamic resource scheduling) and the number of
        # GPUs dedicated exclusively to rollout (standalone). Used to combine
        # dynamic_resource/{train,rollout}_resource_utilization into a single
        # dynamic_resource/resource_utilization metric — see
        # _special_metrics_aggergate() for the formula.
        self.hybrid_gpus = hybrid_gpus
        self.standalone_gpus = standalone_gpus

        # Metric aggregation rule configuration
        self.aggregation_rules = self._init_aggregation_rules()

    def _init_aggregation_rules(self) -> dict[str, dict[str, list[str]]]:
        # [FUNCTION DOC]
        # HELPER: Outputs: mapping aggregation type ("time_sum", "min", "avg", "max", "last", "sum") to explicit metric names.
        """Initialize metrics aggregation rules"""
        return {
            # Time-Based metrics, can add metrics here
            "time_sum": ["perf/time_per_step"],
            "min": ["timing_s/agent_loop/tool_calls/min"],
            "avg": ["timing_s/agent_loop/tool_calls/mean"],
            "max": ["timing_s/agent_loop/tool_calls/max"],
            "last": [
                "fully_async/count/total_generated_samples",
                "fully_async/count/stale_samples_processed",
                "fully_async/count/stale_trajectory_processed",
                "fully_async/count/current_param_version",
                "fully_async/count/dropped_stale_samples",
                "training/global_step",  # TODO change name to: total_step
                # End-of-sync-cycle snapshot, not a per-micro-step quantity to average.
                "dynamic_resource/mq_size",
            ],
            "sum": [
                # Raw numerator/denominator seconds for dynamic_resource/train_resource_utilization.
                # Summed across all micro-steps in a sync cycle; the ratio itself is computed once
                # from the summed totals in _special_metrics_aggergate(), not averaged per-micro-step.
                "dynamic_resource/train_compute_time_s",
                "dynamic_resource/train_allocated_time_s",
            ],
        }

    def add_step_metrics(self, metrics: dict[str, Any], sample_count: int, timestamp: float = None):
        # [FUNCTION DOC]
        # Behavior: records sample count and timestamp (now if None), increments step count, and appends float values of
        # numeric or tensor metrics (other types ignored). Side effects: mutates state.
        # Exceptions: tensor values with more than one element raise in item().
        """Adding a single-step metrics"""
        if timestamp is None:
            timestamp = time.time()

        self.sample_counts.append(sample_count)
        self.timestamps.append(timestamp)
        self.step_count += 1

        # Store all metrics values
        for key, value in metrics.items():
            if isinstance(value, int | float | np.number):
                self.metric_values[key].append(float(value))
            elif isinstance(value, torch.Tensor):
                self.metric_values[key].append(float(value.item()))

    def _get_aggregation_type(self, metric_name: str) -> str:
        # [FUNCTION DOC]
        # HELPER: Behavior: returns the explicit rule type if listed; else by lowercase name: contains "timing_s/" -> time_sum;
        # mean/avg/average -> avg; max/maximum -> max; min/minimum -> min; sum/total -> sum; weighted_avg -> weighted_avg;
        # default avg.
        # KNOWN ISSUE: the weighted_avg branch is unreachable because any name containing "weighted_avg" already matches "avg".
        """Determine the aggregation type based on the metric name"""
        for agg_type, metric_list in self.aggregation_rules.items():
            if metric_name in metric_list:
                return agg_type

        metric_lower = metric_name.lower()
        if any(keyword in metric_lower for keyword in ["timing_s/"]):
            return "time_sum"
        if any(keyword in metric_lower for keyword in ["mean", "avg", "average"]):
            return "avg"
        if any(keyword in metric_lower for keyword in ["max", "maximum"]):
            return "max"
        if any(keyword in metric_lower for keyword in ["min", "minimum"]):
            return "min"
        if any(keyword in metric_lower for keyword in ["sum", "total"]):
            return "sum"
        if any(keyword in metric_lower for keyword in ["weighted_avg"]):
            return "weighted_avg"

        return "avg"

    def _aggregate_single_metric(self, metric_name: str, values: list[float]) -> float:
        # [FUNCTION DOC]
        # HELPER: Behavior: empty -> 0.0; last -> last value; weighted_avg -> sample-count weighted mean (simple mean if lengths
        # differ or counts sum to 0); sum/time_sum -> sum; avg -> mean; max; min; otherwise mean. Outputs: float.
        """Aggregating a single metric"""
        if not values:
            return 0.0

        agg_type = self._get_aggregation_type(metric_name)

        if agg_type == "last":
            return values[-1]

        elif agg_type == "weighted_avg":
            # Weighted average
            if len(values) != len(self.sample_counts):
                # If the lengths do not match, use a simple average
                return sum(values) / len(values)

            total_samples = sum(self.sample_counts)
            if total_samples == 0:
                return sum(values) / len(values)

            weighted_sum = sum(v * c for v, c in zip(values, self.sample_counts, strict=False))
            return weighted_sum / total_samples

        elif agg_type == "sum" or agg_type == "time_sum":
            return sum(values)

        elif agg_type == "avg":
            return sum(values) / len(values)

        elif agg_type == "max":
            return max(values)

        elif agg_type == "min":
            return min(values)

        else:
            # Default average
            return sum(values) / len(values)

    def get_aggregated_metrics(self, rollout_resource_utilization: float | None = None) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: returns {} if no steps; aggregates every metric then applies special derived metrics; prints elapsed time.
        # Inputs: rollout_resource_utilization - optional value from the rollouter. Outputs: dict.
        """aggregated metrics

        Args:
            rollout_resource_utilization: The rollout-resource utilization for the sync
                cycle just finished (see FullyAsyncRollouter._compute_rollout_resource_utilization()).
                It is computed on the rollouter side and returned out-of-band via
                reset_staleness(), so it is injected here rather than flowing through
                add_step_metrics(). It is only surfaced in the returned dict (as
                "dynamic_resource/rollout_resource_utilization") once
                dynamic_resource/train_resource_utilization is also available, so it and
                "dynamic_resource/resource_utilization" start appearing together from the
                same sync cycle onward — see _special_metrics_aggergate().
        """
        t = time.time()
        if self.step_count == 0:
            return {}

        aggregated = {}

        # Aggregate all metrics
        for metric_name, values in self.metric_values.items():
            aggregated[metric_name] = self._aggregate_single_metric(metric_name, values)

        # Aggregate special metrics
        aggregated = self._special_metrics_aggergate(aggregated, rollout_resource_utilization)

        print(f"aggregated metrics done. cost {time.time() - t:.4f} seconds.")

        return aggregated

    def _special_metrics_aggergate(
        self, aggregated: dict[str, Any], rollout_resource_utilization: float | None = None
    ) -> dict[str, Any]:
        # [FUNCTION DOC]
        # HELPER: Behavior: recomputes global_seqlen/minmax_diff as max minus min when present; recomputes perf/throughput as total
        # tokens / (time per step * total_gpus) when all keys exist; sets idle ratio = timing_s/gen / timing_s/step; computes
        # train_resource_utilization from summed compute/allocated times; and, when that, the rollout utilization, a positive GPU
        # total and positive step time exist, adds rollout utilization and combined resource utilization using the weighted
        # formula in the comments (x = clipped wait_for_enough_samples / step time).
        # Inputs: aggregated dict (mutated); rollout_resource_utilization. Outputs: the same dict.
        # Exceptions: KeyError if global_seqlen/max or min missing when minmax_diff exists; ZeroDivisionError if timing_s/step is 0
        # while timing_s/gen exists.
        """calculate special metrics"""

        # global_seqlen/minmax_diff
        if "global_seqlen/minmax_diff" in aggregated.keys():
            aggregated["global_seqlen/minmax_diff"] = aggregated["global_seqlen/max"] - aggregated["global_seqlen/min"]

        # perf/throughput
        REQUIRED_PERF_KEYS = {"perf/throughput", "perf/total_num_tokens", "perf/time_per_step"}
        if REQUIRED_PERF_KEYS.issubset(aggregated):
            aggregated["perf/throughput"] = aggregated["perf/total_num_tokens"] / (
                aggregated["perf/time_per_step"] * self.total_gpus
            )

        # trainer/idle_ratio
        if "timing_s/gen" in aggregated.keys() and "timing_s/step" in aggregated.keys():
            aggregated["fully_async/trainer/idle_ratio"] = aggregated["timing_s/gen"] / aggregated["timing_s/step"]

        # dynamic_resource/train_resource_utilization: ratio computed once from the
        # sync-cycle-wide summed totals (NOT averaged per-micro-step), per the numerator/
        # denominator definitions documented in FullyAsyncTrainer._record_train_resource_utilization().
        REQUIRED_UTIL_KEYS = {"dynamic_resource/train_compute_time_s", "dynamic_resource/train_allocated_time_s"}
        if REQUIRED_UTIL_KEYS.issubset(aggregated) and aggregated["dynamic_resource/train_allocated_time_s"] > 0:
            aggregated["dynamic_resource/train_resource_utilization"] = (
                aggregated["dynamic_resource/train_compute_time_s"]
                / aggregated["dynamic_resource/train_allocated_time_s"]
            )

        # dynamic_resource/resource_utilization: cluster-wide utilization for this sync
        # cycle, combining the hybrid (trainer-node) GPUs' time-split between rollout and
        # train with the standalone (dedicated) rollout GPUs.
        #
        # Let a = self.hybrid_gpus, b = self.standalone_gpus, and x in [0, 1] be the
        # fraction of this cycle's wall-clock time that the hybrid GPUs spent doing
        # rollout (the rest, 1 - x, they spent training). x is estimated as the ratio of
        # summed "wait_for_enough_samples" time (hybrid-rollout wall-clock time within the
        # cycle) over the summed "step" time (total cycle wall-clock time):
        #
        #   x = timing_s/wait_for_enough_samples / timing_s/step
        #
        # "timing_s/wait_for_enough_samples" is only recorded when dynamic resource scheduling
        # is enabled (see FullyAsyncTrainer.fit_step()). When it's disabled, hybrid GPUs
        # never switch into rollout mode, i.e. x == 0 (100% of hybrid time is training) —
        # so it's treated as 0.0 rather than skipping the metric entirely.
        #
        # Then, weighting each utilization by the GPU-seconds it was measured over:
        #   - Hybrid GPUs spend (1 - x) * a GPU-time training at train_resource_utilization,
        #     and x * a GPU-time doing rollout at rollout_resource_utilization.
        #   - Standalone GPUs (b of them) spend all their time doing rollout at
        #     rollout_resource_utilization.
        #
        #   resource_utilization = ((1 - x) * a * train_resource_utilization
        #                            + (x * a + b) * rollout_resource_utilization) / (a + b)
        #
        # dynamic_resource/train_resource_utilization is only available once the aggregator
        # has collected at least one micro-step's dynamic_resource/train_{compute,allocated}_time_s
        # (see add_step_metrics()/_record_train_resource_utilization()) — this never holds on
        # the very first sync cycle, since the sync-triggering micro-step's own values are
        # only recorded *after* this method's caller (_fit_update_weights()) returns. To keep
        # "dynamic_resource/rollout_resource_utilization" and "dynamic_resource/resource_utilization"
        # appearing together (rather than the former appearing one cycle earlier), the former is
        # only added to the output once the latter can also be computed.
        total_gpus = self.hybrid_gpus + self.standalone_gpus
        if (
            "dynamic_resource/train_resource_utilization" in aggregated
            and rollout_resource_utilization is not None
            and total_gpus > 0
            and aggregated.get("timing_s/step", 0.0) > 0
        ):
            aggregated["dynamic_resource/rollout_resource_utilization"] = rollout_resource_utilization
            a = self.hybrid_gpus
            b = self.standalone_gpus
            wait_for_enough_samples_time = aggregated.get("timing_s/wait_for_enough_samples", 0.0)
            x = min(1.0, max(0.0, wait_for_enough_samples_time / aggregated["timing_s/step"]))
            train_util = aggregated["dynamic_resource/train_resource_utilization"]
            aggregated["dynamic_resource/resource_utilization"] = (
                (1 - x) * a * train_util + (x * a + b) * rollout_resource_utilization
            ) / total_gpus

        return aggregated

    def reset(self):
        # [FUNCTION DOC]
        # Behavior: clears all stored values, counts and timestamps and resets the step count.
        """Reset Aggregator"""
        self.metric_values.clear()
        self.sample_counts.clear()
        self.timestamps.clear()
        self.step_count = 0

    def get_current_stats(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: returns a debug dict with step count, number of metrics, total samples and metric names.
        """Get statistics about the current aggregation state (for debugging)"""
        return {
            "step_count": self.step_count,
            "metric_count": len(self.metric_values),
            "total_samples": sum(self.sample_counts),
            "metric_names": list(self.metric_values.keys()),
        }


def task_exception_handler(task: asyncio.Task):
    # [FUNCTION DOC]
    # HELPER: Behavior: done-callback that retrieves the task result; cancellation is ignored; other exceptions are printed
    # and re-raised (inside the event loop's callback handler, so they surface as loop errors).
    """Handle task exceptions and log them"""
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task was cancelled, this is expected
    except Exception as e:
        print(f"Task {task.get_name()} failed with exception: {e}")
        raise e


def safe_create_task(coro, name: str, task_set: set = None):
    # [FUNCTION DOC]
    # HELPER: Behavior: creates a named asyncio task, attaches task_exception_handler, and adds it to task_set if given (the
    # task is not removed when done). Outputs: the task. Exceptions: RuntimeError if no running loop.
    """Safely create a task with exception handling

    Args:
        coro: The coroutine to run
        name: Name for the task
        task_set: Optional set to add the task to

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(task_exception_handler)
    if task_set is not None:
        task_set.add(task)
    return task
