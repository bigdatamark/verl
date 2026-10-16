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
# Process-level logging configuration helpers for verl.

import logging
import os
import sys

import torch


def configure_verl_logging() -> None:
    # [FUNCTION DOC]
    # Behavior: reads level from env VERL_LOGGING_LEVEL (default "INFO"), creates a stderr
    # StreamHandler with format "LEVEL:time:message", replaces all handlers of the "verl" logger with
    # it, sets the logger level and disables propagation to the root logger.
    # Inputs: none. Outputs: none. Exceptions: ValueError from logging if the level name is invalid.
    # Side effects: mutates global logging state for the "verl" logger.
    """Isolate verl logs from root logger changes in the current process."""
    level = os.getenv("VERL_LOGGING_LEVEL", "INFO")
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))

    verl_logger = logging.getLogger("verl")
    verl_logger.handlers = [handler]
    verl_logger.setLevel(level)
    verl_logger.propagate = False


def set_basic_config(level):
    # [FUNCTION DOC]
    # Behavior: calls logging.basicConfig with format "LEVEL:time:message" and the given level
    # (no effect if the root logger already has handlers).
    # Inputs: level. Side effects: global root logger configuration.
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)


def log_to_file(string):
    # [FUNCTION DOC]
    # Behavior: prints the string; if a "logs" directory exists in the CWD, appends the string plus
    # newline to logs/log_<global rank>.
    # Inputs: string. Exceptions: requires torch.distributed to be initialized when logs/ exists.
    # Side effects: stdout and file append.
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")
