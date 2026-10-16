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
# Simple console / rank-aware logging helpers used by trainers and workers: a console metrics
# logger (LocalLogger), a base class for rank-filtered logging decorators, and free functions that
# print or log with rank prefixes and optional timestamps.

"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import logging
import numbers
import pprint

import torch


def concat_dict_to_str(dict: dict, step):
    # [FUNCTION DOC]
    # HELPER: Formats a metrics mapping as one line.
    # Behavior: starts with "step:<step>", appends "key:value" (value pretty-formatted) for each entry
    # whose value is a numbers.Number (non-numeric entries skipped), joins with " - ".
    # Inputs: dict - metric name to value; step - step label.
    # Outputs: the joined string. Exceptions/side effects: none.
    output = [f"step:{step}"]
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f"{k}:{pprint.pformat(v)}")
    output_str = " - ".join(output)
    return output_str


class LocalLogger:
    # [CLASS DOC]
    # Console tracking backend with the flush/log interface of other trackers.
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        # [FUNCTION DOC]
        # Inputs: print_to_console - whether log() prints. Side effects: stores the flag.
        self.print_to_console = print_to_console

    def flush(self):
        # [FUNCTION DOC]
        # No-op, provided for interface compatibility.
        pass

    def log(self, data, step):
        # [FUNCTION DOC]
        # Behavior: if printing enabled, prints concat_dict_to_str(data, step) with flush.
        # Inputs: data - metric mapping; step - step. Outputs: none. Side effects: stdout.
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    # [CLASS DOC]
    # Base for logging decorators; holds role/logger/level/rank settings and picks a logging function.
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        # [FUNCTION DOC]
        # Behavior: stores role, logger, level, rank, log_only_rank_0; sets logging_function to
        # log_by_logging, or to log_by_print when logger is None.
        # Inputs: see class docstring. Outputs: none. Side effects: attribute assignment.
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        # [FUNCTION DOC]
        # Behavior: prints "<role> <log_str>" (flushed) unless log_only_rank_0 is set and rank != 0.
        # Inputs: log_str. Side effects: stdout.
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        # [FUNCTION DOC]
        # Behavior: logs "<role> <log_str>" at the configured level via the logger, with the same rank filter.
        # Inputs: log_str. Exceptions: ValueError if no logger is set. Side effects: logging output.
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    # [FUNCTION DOC]
    # Behavior: prints (flushed) only on global rank 0 when torch.distributed is initialized,
    # otherwise always prints. Inputs: message. Outputs: none. Side effects: stdout.
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    # [FUNCTION DOC]
    # Behavior: prints "[Rank <rank>] <message>" unless log_only_rank_0 is True and rank != 0.
    # Inputs: message, rank, log_only_rank_0. Outputs: none. Side effects: stdout.
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    # [FUNCTION DOC]
    # Behavior: same as print_with_rank but prefixes the current local time as
    # "[YYYY-MM-DD HH:MM:SS]". Inputs/Outputs/Side effects: as print_with_rank.
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    # [FUNCTION DOC]
    # Behavior: logs "[Rank <rank>] <message>" at the given level via logger unless log_only_rank_0
    # is True and rank != 0. Inputs: message, rank, logger, level, log_only_rank_0.
    # Outputs: none. Side effects: logging output.
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
