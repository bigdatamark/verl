# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Command-line checker script (not a pytest test despite its name) that verifies profiler output exists for a set
# of training stages under a profiler directory, for GPU (torch profiler JSON traces) or NPU (Ascend profiling
# directories). Exits 0 on success and 1 on failure. Configures root logging at INFO on import.

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable

# Initialize logger
logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@dataclass
class DeviceCheckConfig:
    # [CLASS DOC]
    # Holds per-device validation callbacks: search_patterns maps a stage to glob patterns (relative to the profiler
    # dir); path_filter keeps matched paths of the right kind; count_validator decides whether the number of matches
    # for a stage is acceptable; prof_validator checks each matched path's content.
    """Device check configuration: encapsulates device-specific validation rules"""

    search_patterns: Callable[[str], list[str]]
    path_filter: Callable[[str], bool]
    count_validator: Callable[[str, list[str]], bool]
    prof_validator: Callable[[str], bool]


class ProfilerChecker:
    # [CLASS DOC]
    # Validates profiler deliverables for each configured stage pattern. Default stages are "actor_update",
    # "*rollout*" and "ref_*".
    """Unified Profiler checker supporting GPU/NPU devices"""

    TARGET_STAGES = ["actor_update", "*rollout*", "ref_*"]

    def __init__(self, device_type: str, profiler_dir: str, stages: list[str] | None = None):
        # [FUNCTION DOC]
        # Behavior: lowercases and stores device_type, stores profiler_dir, uses the given stages or the default list,
        # then builds the device config.
        # Inputs: device_type - "gpu" or "npu" (case-insensitive); profiler_dir - root path; stages - optional list of
        # glob stage patterns. Outputs: none.
        # Exceptions: ValueError for unsupported device type or an empty stages list.
        self.device_type = device_type.lower()
        self.profiler_dir = profiler_dir

        # Validate device type
        if self.device_type not in ["gpu", "npu"]:
            raise ValueError(f"Unsupported device type: {device_type}, only gpu/npu are supported")
        self.stages = stages if stages is not None else self.TARGET_STAGES
        if not self.stages:
            raise ValueError("At least one profiler stage is required")

        # Initialize device-specific configuration
        self._init_device_config()

    def _init_device_config(self):
        # [FUNCTION DOC]
        # HELPER: Sets self.config. GPU: patterns match any *.json* file recursively under a directory containing the
        # stage name, or a top-level file containing the stage name; keep only files ending .json or .json.gz; require at
        # least one; each must have non-zero size. NPU: pattern matches "*_ascend_*" directories directly under a stage
        # directory; keep only directories; require at least one; each must contain at least one "PROF_*" entry and the
        # first such entry must be a directory.
        """Initialize validation rules for different devices (core: device differences as config)"""
        if self.device_type == "gpu":
            self.config = DeviceCheckConfig(
                search_patterns=lambda stage: [os.path.join(f"*{stage}*", "**", "*.json*"), f"*{stage}*.json*"],
                path_filter=lambda p: os.path.isfile(p) and p.endswith((".json", ".json.gz")),
                count_validator=lambda stage, paths: len(paths) > 0,
                prof_validator=lambda p: os.path.getsize(p) > 0,
            )
        else:  # NPU
            self.config = DeviceCheckConfig(
                # NPU search pattern: match ascend subdirectory under stage
                search_patterns=lambda stage: [os.path.join(stage, "*_ascend_*")],
                path_filter=os.path.isdir,
                # Each stage needs output; multiple ranks or windows may produce multiple directories.
                count_validator=lambda stage, paths: len(paths) > 0,
                # NPU: PROF_* subdirectory must exist and be a valid directory
                prof_validator=lambda d: (
                    len(glob.glob(os.path.join(d, "PROF_*"))) > 0
                    and os.path.isdir(glob.glob(os.path.join(d, "PROF_*"))[0])
                ),
            )

    def _validate_stage(self, stage: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: Expands the stage's patterns under profiler_dir, globs recursively, filters, de-duplicates and sorts the
        # paths, logs the count and each path; returns False (logging an error) if the count check fails or any path fails
        # content validation, else True. Inputs: stage - pattern string. Outputs: bool. Side effects: logging.
        """Match, log and validate the stage's profiler output."""
        patterns = [os.path.join(self.profiler_dir, p) for p in self.config.search_patterns(stage)]
        paths = sorted(
            {
                path
                for pattern in patterns
                for path in glob.glob(pattern, recursive=True)
                if self.config.path_filter(path)
            }
        )
        logger.info(f"[{stage}] Found {len(paths)} profiler paths (patterns: {patterns})")
        for path in paths:
            logger.info(f"[{stage}] Found: {path}")

        if not self.config.count_validator(stage, paths):
            logger.error(f"[{stage}] Unexpected profiler output count: {len(paths)}")
            return False

        for path in paths:
            if not self.config.prof_validator(path):
                logger.error(f"[{stage}] Missing or empty profiler output: {path}")
                return False

        return True

    def check(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: logs start; returns False (with error log) if profiler_dir does not exist; validates each stage in
        # order, returning False at the first failure; logs success and returns True.
        # Outputs: bool. Side effects: logging.
        """Unified check entry point"""
        logger.info(f"Starting profiler deliverables check for {self.device_type.upper()}...")

        # Validate root directory exists
        if not os.path.exists(self.profiler_dir):
            logger.error(f"Profiler data directory not found: {self.profiler_dir}")
            return False

        # Run validation for all target stages
        for stage in self.stages:
            if not self._validate_stage(stage):
                return False

        logger.info(f"All {self.device_type.upper()} validation stages passed")
        return True


def parse_args():
    # [FUNCTION DOC]
    # Behavior: parses --device (required, gpu or npu), --profiler_dir (default ./profiler_data) and --stage (one or
    # more patterns, default None meaning the built-in stages) from the command line.
    # Outputs: argparse namespace. Exceptions: exits on invalid arguments.
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Check Profiler deliverables (support GPU/NPU)")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["gpu", "npu"],
        help="Device type, available values: gpu/npu (required)",
    )
    parser.add_argument(
        "--profiler_dir",
        type=str,
        default="./profiler_data",
        help="Path to profiler data directory (default: ./profiler_data)",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=None,
        help=f"Stage patterns to check (default: {ProfilerChecker.TARGET_STAGES})",
    )
    return parser.parse_args()


def main():
    # [FUNCTION DOC]
    # Behavior: parses arguments, runs the checker and exits with status 0 on success or 1 on failure; any exception
    # is logged with traceback and exits 1. Outputs: none (process exit). Side effects: logging, sys.exit.
    args = parse_args()

    try:
        checker = ProfilerChecker(device_type=args.device, profiler_dir=args.profiler_dir, stages=args.stage)
        if checker.check():
            logger.info(f"All {args.device.upper()} profiler deliverables check passed!")
            sys.exit(0)
        else:
            logger.error(f"{args.device.upper()} profiler check failed!")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Check failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
