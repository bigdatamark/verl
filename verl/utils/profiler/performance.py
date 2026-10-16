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
# Lightweight performance helpers: device memory reporting/logging (function and decorator forms),
# caller-annotated printing, codetiming-based timers accumulating into a name->seconds mapping, and
# torch.distributed reductions/gathers of timing dictionaries.
import datetime
import inspect
import logging
from contextlib import contextmanager
from typing import Any, Optional

import torch
import torch.distributed as dist
from codetiming import Timer

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.logger import DecoratorLoggerBase


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    # [FUNCTION DOC]
    # HELPER: Reads current device memory stats as formatted strings.
    # Behavior: if the torch device module is torch.cpu, returns four "0.00" strings. Otherwise reads
    # allocated and reserved bytes from the caching allocator and free/total from mem_get_info (driver
    # level, so it sees vLLM sleep-mode memory), computes used = total - free, divides by 1024^3/^2/^1 for
    # GB/MB/KB and formats with the given decimal precision.
    # Outputs: tuple (allocated, reserved, used, total) strings.
    # KNOWN ISSUE: CPU branch always uses two decimals regardless of precision.
    # Exceptions: AssertionError for an unsupported unit. Side effects: none.
    """Get current memory usage.

    Note that CPU device memory info is always 0.

    Args:
        unit (str, optional): The unit of memory measurement. Defaults to "GB".
        precision (int, optional): The number of decimal places to round memory values. Defaults to 2.

    Returns:
        tuple[str]: A tuple containing memory allocated, memory reserved, memory used, and memory total
        in the specified unit.
    """
    assert unit in ["GB", "MB", "KB"]
    device = get_torch_device()
    # torch.cpu.memory_allocated() does not exist
    if device == torch.cpu:
        return "0.00", "0.00", "0.00", "0.00"

    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    # [FUNCTION DOC]
    # Behavior: if torch.distributed is not initialized, rank is None, or the current rank equals rank,
    # reads memory info (GB) and emits a single message prefixed by head via print (logger None) or
    # logger.log at level.
    # Outputs: None. Exceptions: none explicit. Side effects: prints/logs.
    """Log GPU memory usage information.

    Args:
        head (str): A descriptive header for the memory usage log message.
        logger (logging.Logger, optional): Logger instance to use for logging. If None, prints to stdout.
        level: Logging level to use. Defaults to logging.DEBUG.
        rank (int): The rank of the process to log memory for. Defaults to 0.
    """
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


class GPUMemoryLogger(DecoratorLoggerBase):
    # [CLASS DOC]
    # Decorator (via DecoratorLoggerBase) that logs device memory before and after each call of the wrapped
    # function, tagged with a role. Rank-filtering and the logging function come from the base class.
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.profiler.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        # [FUNCTION DOC]
        # Behavior: determines the current rank (distributed rank when initialized with world size > 1, else 0)
        # and initializes the base logger with role, logger, level, rank and log_only_rank_0.
        # Side effects: none beyond attribute setup.
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        # [FUNCTION DOC]
        # Behavior: returns a plain wrapper that routes calls through self.log.
        # KNOWN ISSUE: wrapper does not use functools.wraps, so name/docstring of the decorated function are lost.
        # Outputs: wrapper function.
        def f(*args, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Calls self.log with the decorated function and the call arguments; returns its result.
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        # [FUNCTION DOC]
        # Behavior: logs "Before <name>" memory message, calls func, logs "After <name>" message, returns output.
        # Inputs: func and its args/kwargs. Outputs: func's return value.
        # Exceptions: propagates func's exceptions (After message skipped). Side effects: logging.
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )
        self.logging_function(message)

        output = func(*args, **kwargs)

        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = (
            f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, "
            f"device memory used/total (GB): {mem_used}/{mem_total}"
        )

        self.logging_function(message)
        return output


def log_print(ctn: Any):
    # [FUNCTION DOC]
    # Behavior: prints "[<timestamp>-<file>:<line>:<function>]: <ctn>" using the caller's stack frame.
    # Inputs: ctn - any printable object. Outputs: None. Side effects: stdout.
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split("/")[-1]
    print(f"[{current_time}-{file_name}:{line_number}:{function_name}]: {ctn}")


def _timer(name: str, timing_raw: dict[str, float]):
    # [FUNCTION DOC]
    # HELPER: Generator shared by the timer context managers.
    # Behavior: runs a codetiming Timer around a single yield; after resuming, adds timer.last (seconds) to
    # timing_raw[name], initializing it to 0 if absent.
    # Inputs: name; timing_raw - mutable mapping. Outputs: generator yielding once.
    # Exceptions: if the body raises, the exception propagates and no time is recorded.
    # Side effects: mutates timing_raw.
    """Inner function that handles the core timing logic.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


@contextmanager
def simple_timer(name: str, timing_raw: dict[str, float]):
    # [FUNCTION DOC]
    # Behavior: context manager delegating to _timer (accumulates elapsed time). Side effects: mutates timing_raw.
    """Context manager for basic timing without NVTX markers.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


@contextmanager
def marked_timer(
    name: str,
    timing_raw: dict[str, float],
    color: str = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
):
    # [FUNCTION DOC]
    # Behavior: fallback with the same signature as the NVTX/MSTX versions; ignores color/domain/category and
    # only accumulates elapsed time via _timer. Side effects: mutates timing_raw.
    """Context manager for timing with platform markers.

    This utility function measures the execution time of code within its context,
    accumulates the timing information, and adds platform markers for profiling.
    This function is a default implementation when hardware profiler is not available.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.
        color (Optional[str]): Color for the marker. Defaults to None.
        domain (Optional[str]): Domain for the marker. Defaults to None.
        category (Optional[str]): Category for the marker. Defaults to None.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    yield from _timer(name, timing_raw)


def reduce_timing(
    timing_raw: dict[str, float], reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.AVG
) -> dict[str, float]:
    # [FUNCTION DOC]
    # Behavior: returns input unchanged if torch.distributed is not initialized. Else sorts keys, packs
    # values into a float32 tensor on the current device, all_reduces with reduce_op (default AVG, not sum
    # as the docstring says), and rebuilds a dict from keys to reduced Python floats.
    # Inputs: timing_raw - mapping (all ranks must have identical key sets); reduce_op.
    # Outputs: new dict. Exceptions: collective errors. Side effects: collective communication, device sync.
    """Reduce timing information across all processes.

    This function uses distributed communication to gather and sum the timing
    information from all processes in a distributed environment.

    Args:
        timing_raw (Dict[str, float]): Dictionary containing timing information.

    Returns:
        Dict[str, float]: Reduced timing information.
    """
    if not dist.is_initialized():
        return timing_raw

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])
    timing_list = torch.tensor(timing_list, dtype=torch.float32, device=get_device_id())
    torch.distributed.all_reduce(timing_list, op=reduce_op)
    timing_list = [tensor.item() for tensor in timing_list.to("cpu")]
    timing_generate = {key_list[i]: timing_list[i] for i in range(len(key_list))}
    return timing_generate


def topk_reduce_ratio_min_max(timing: float, k: int = 10) -> tuple[float, float, float]:
    """Calculate topk items take-up ratio, and min/max timing across all ranks."""
    # [FUNCTION DOC]
    # Behavior: returns (-1, -1, -1) when not distributed. Otherwise all_gathers each rank's scalar timing,
    # computes min and max, the (1 - k/100) quantile, and the fraction of ranks strictly above that quantile.
    # Inputs: timing - this rank's value; k - percentile size (top k%).
    # Outputs: tuple (tail_ratio, min, max).
    # KNOWN ISSUE: the local tensor is 0-dim while gather buffers have shape (1,); all_gather may reject the
    # shape mismatch depending on backend.
    # Side effects: collective communication.
    if not dist.is_initialized():
        return -1.0, -1.0, -1.0

    world_size = dist.get_world_size()
    timing_tensor = torch.tensor(timing, dtype=torch.float32, device=get_device_id())
    tensor_list = [torch.zeros(1, dtype=torch.float32, device=get_device_id()) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, timing_tensor)
    tensor_stack = torch.stack(tensor_list)
    timing_min = tensor_stack.min().cpu().item()
    timing_max = tensor_stack.max().cpu().item()
    top_k_percentile = torch.quantile(tensor_stack, 1 - k / 100)
    tail_ratio = torch.mean((tensor_stack > top_k_percentile).float()).cpu().item()
    return tail_ratio, timing_min, timing_max


def gather_timing(timing_raw: dict[str, float]) -> dict[str, list[float]]:
    # [FUNCTION DOC]
    # Behavior: if not distributed, wraps each value in a single-element list. Otherwise sorts keys, gathers
    # each rank's ordered value list via all_gather_object, and maps each key to the per-rank list of values.
    # Inputs: timing_raw - mapping with identical key sets across ranks.
    # Outputs: dict key -> list of floats indexed by rank. Side effects: collective communication.
    if not dist.is_initialized():
        return {k: [v] for k, v in timing_raw.items()}

    key_list, timing_list = [], []
    for key in sorted(timing_raw.keys()):
        key_list.append(key)
        timing_list.append(timing_raw[key])

    world_size = torch.distributed.get_world_size()

    object_gather_list = [None] * world_size

    torch.distributed.all_gather_object(object_gather_list, timing_list)

    timing_generate = {
        key_list[i]: [timing_list[i] for timing_list in object_gather_list] for i in range(len(key_list))
    }

    return timing_generate
