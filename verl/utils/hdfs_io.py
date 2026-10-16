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
# Minimal HDFS-aware filesystem helpers (exists, makedirs, copy) that dispatch to the hdfs CLI ("<hdfs binary> dfs ...", located via PATH at import) for paths starting with "hdfs://" and to os/shutil for local paths. Logging level from VERL_SFT_LOGGING_LEVEL.

import logging
import os
import shutil

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

_HDFS_PREFIX = "hdfs://"

_HDFS_BIN_PATH = shutil.which("hdfs")


def exists(path: str, **kwargs) -> bool:
    # [FUNCTION DOC]
    # Returns hdfs "-test -e" result for HDFS paths, else os.path.exists.
    # KNOWN ISSUE: extra kwargs are forwarded to _exists, which accepts none, so passing any kwargs for an HDFS path raises TypeError.
    r"""Works like os.path.exists() but supports hdfs.

    Test whether a path exists. Returns False for broken symbolic links.

    Args:
        path (str): path to test

    Returns:
        bool: True if the path exists, False otherwise
    """
    if _is_non_local(path):
        return _exists(path, **kwargs)
    return os.path.exists(path)


def _exists(file_path: str):
    # [FUNCTION DOC]
    # HELPER: Runs "dfs -test -e" for paths starting with "hdfs" and returns True on exit code 0; otherwise os.path.exists.
    """hdfs capable to check whether a file_path is exists"""
    if file_path.startswith("hdfs"):
        return _run_cmd(_hdfs_cmd(f"-test -e {file_path}")) == 0
    return os.path.exists(file_path)


def makedirs(name, mode=0o777, exist_ok=False, **kwargs) -> None:
    # [FUNCTION DOC]
    # Behavior: HDFS paths run "dfs -mkdir -p" (mode and exist_ok ignored); local paths use os.makedirs with mode/exist_ok.
    # Exceptions: OSError locally when the dir exists and exist_ok is False.
    # KNOWN ISSUE: kwargs forwarded to _mkdir, which accepts none (TypeError if any given).
    r"""Works like os.makedirs() but supports hdfs.

    Super-mkdir; create a leaf directory and all intermediate ones.  Works like
    mkdir, except that any intermediate path segment (not just the rightmost)
    will be created if it does not exist. If the target directory already
    exists, raise an OSError if exist_ok is False. Otherwise no exception is
    raised.  This is recursive.

    Args:
        name (str): directory to create
        mode (int): file mode bits
        exist_ok (bool): if True, do not raise an exception if the directory already exists
        kwargs: keyword arguments for hdfs

    """
    if _is_non_local(name):
        # TODO(haibin.lin):
        # - handle OSError for hdfs(?)
        # - support exist_ok for hdfs(?)
        _mkdir(name, **kwargs)
    else:
        os.makedirs(name, mode=mode, exist_ok=exist_ok)


def _mkdir(file_path: str) -> bool:
    # [FUNCTION DOC]
    # HELPER: "dfs -mkdir -p" for hdfs paths (exit code ignored), else os.makedirs(exist_ok=True). Always returns True.
    """hdfs mkdir"""
    if file_path.startswith("hdfs"):
        _run_cmd(_hdfs_cmd(f"-mkdir -p {file_path}"))
    else:
        os.makedirs(file_path, exist_ok=True)
    return True


def copy(src: str, dst: str, **kwargs) -> bool:
    # [FUNCTION DOC]
    # Behavior: if either side is HDFS, delegates to _copy and returns a bool success; otherwise uses shutil.copytree for directories or shutil.copy for files (returning the destination path) with kwargs forwarded.
    # Exceptions: local shutil errors propagate.
    r"""Works like shutil.copy() for file, and shutil.copytree for dir, and supports hdfs.

    Copy data and mode bits ("cp src dst"). Return the file's destination.
    The destination may be a directory.
    If source and destination are the same file, a SameFileError will be
    raised.

    Arg:
        src (str): source file path
        dst (str): destination file path
        kwargs: keyword arguments for hdfs copy

    Returns:
        str: destination file path

    """
    if _is_non_local(src) or _is_non_local(dst):
        # TODO(haibin.lin):
        # - handle SameFileError for hdfs files(?)
        # - return file destination for hdfs files
        return _copy(src, dst)
    else:
        if os.path.isdir(src):
            return shutil.copytree(src, dst, **kwargs)
        else:
            return shutil.copy(src, dst, **kwargs)


def _copy(from_path: str, to_path: str, timeout: int = None) -> bool:
    # [FUNCTION DOC]
    # HELPER: to HDFS: "dfs -cp -f" from HDFS or "dfs -put -f" from local; from HDFS to local: "dfs -get"; local to local: shutil.copy, treating SameFileError as success and logging other errors as failure.
    # Outputs: True iff the return code is 0.
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(_hdfs_cmd(f"-cp -f {from_path} {to_path}"), timeout=timeout)
        else:
            returncode = _run_cmd(_hdfs_cmd(f"-put -f {from_path} {to_path}"), timeout=timeout)
    else:
        if from_path.startswith("hdfs"):
            returncode = _run_cmd(
                _hdfs_cmd(
                    f"-get \
                {from_path} {to_path}"
                ),
                timeout=timeout,
            )
        else:
            try:
                shutil.copy(from_path, to_path)
                returncode = 0
            except shutil.SameFileError:
                returncode = 0
            except Exception as e:
                logger.warning(f"copy {from_path} {to_path} failed: {e}")
                returncode = -1
    return returncode == 0


def _run_cmd(cmd: str, timeout=None):
    # [FUNCTION DOC]
    # HELPER: Runs the command string through the shell via os.system and returns its status; timeout is ignored.
    return os.system(cmd)


def _hdfs_cmd(cmd: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Prefixes a dfs subcommand with the hdfs binary path.
    return f"{_HDFS_BIN_PATH} dfs {cmd}"


def _is_non_local(path: str):
    # [FUNCTION DOC]
    # HELPER: True if path starts with "hdfs://".
    return path.startswith(_HDFS_PREFIX)
