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
# Ray-based trajectory tracker for dumping intermediate tensors to HDFS for offline comparison. Enabled by env VERL_ENABLE_TRACKER=1; destination from VERL_TRACKER_HDFS_DIR; verbosity from VERL_TRACKER_VERBOSE. A single detached named actor "global_tracker" queues uploads. Also defines remote_copy, a Ray remote wrapper around hdfs copy. The __main__ block is a manual smoke test.

"""
Trajectory tracker can be inserted into code to save the intermediate results.
The results will be dump to hdfs for offline comparison.
Each process will have a client that first move all the tensors to CPU
"""

import io
import os
import tempfile
from collections import deque

import ray
import torch

from verl.utils.hdfs_io import copy, makedirs

remote_copy = ray.remote(copy)


@ray.remote
def save_to_hdfs(data: io.BytesIO, name, hdfs_dir, verbose):
    # [FUNCTION DOC]
    # HELPER: Ray remote task that writes the buffer to a temp file name.pth and copies it into hdfs_dir.
    # Inputs: data - BytesIO buffer; name - file stem; hdfs_dir - destination; verbose - print before upload.
    # Outputs: none.
    # Exceptions: copy errors are caught and printed.
    # Side effects: temp file I/O, HDFS write, printing.
    filename = name + ".pth"
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_filepath = os.path.join(tmpdirname, filename)
        with open(local_filepath, "wb") as f:
            f.write(data.getbuffer())
        # upload to hdfs

        if verbose:
            print(f"Saving {local_filepath} to {hdfs_dir}")
        try:
            copy(local_filepath, hdfs_dir)
        except Exception as e:
            print(e)


@ray.remote
class TrajectoryTracker:
    # [CLASS DOC]
    # Ray actor holding a FIFO queue of pending save_to_hdfs task references.
    def __init__(self, hdfs_dir, verbose) -> None:
        # [FUNCTION DOC]
        # Stores hdfs_dir and verbose, creates the HDFS directory, initializes an empty deque of pending futures.
        # Side effects: HDFS mkdir.
        self.hdfs_dir = hdfs_dir
        makedirs(hdfs_dir)
        self.verbose = verbose

        self.handle = deque()

    def dump(self, data: io.BytesIO, name):
        # get a temp file and write to it
        # [FUNCTION DOC]
        # Launches a save_to_hdfs remote task for the buffer and appends its future to the queue (non-blocking).
        # Inputs: data - BytesIO; name - file stem.
        self.handle.append(save_to_hdfs.remote(data, name, self.hdfs_dir, self.verbose))

    def wait_for_hdfs(self):
        # [FUNCTION DOC]
        # Blocks until all queued upload futures complete, popping them in FIFO order.
        while len(self.handle) != 0:
            future = self.handle.popleft()
            ray.get(future)


def dump_data(data, name):
    # [FUNCTION DOC]
    # Behavior: no-op unless VERL_ENABLE_TRACKER is "1"; otherwise serializes data with torch.save into a buffer and sends it to the global tracker actor, waiting for the dump call to be enqueued.
    # Inputs: data - any torch-serializable object; name - file stem.
    # Side effects: creates/gets the tracker actor, triggers asynchronous HDFS upload.
    enable = os.getenv("VERL_ENABLE_TRACKER", "0") == "1"
    if not enable:
        return
    buffer = io.BytesIO()
    torch.save(data, buffer)
    tracker = get_trajectory_tracker()
    ray.get(tracker.dump.remote(buffer, name))


def get_trajectory_tracker():
    # [FUNCTION DOC]
    # Behavior: reads env config and gets-or-creates the detached named Ray actor "global_tracker".
    # Outputs: actor handle.
    # Exceptions: AssertionError if VERL_TRACKER_HDFS_DIR unset.
    hdfs_dir = os.getenv("VERL_TRACKER_HDFS_DIR", default=None)
    verbose = os.getenv("VERL_TRACKER_VERBOSE", default="0") == "1"
    assert hdfs_dir is not None
    tracker = TrajectoryTracker.options(name="global_tracker", get_if_exists=True, lifetime="detached").remote(
        hdfs_dir, verbose
    )
    return tracker


if __name__ == "__main__":
    # testing
    os.environ["VERL_ENABLE_TRACKER"] = "1"
    os.environ["VERL_TRACKER_HDFS_DIR"] = "~/debug/test"

    @ray.remote
    def process(iter):
        # [FUNCTION DOC]
        # HELPER: Test-only Ray task that dumps a random tensor dict named after the iteration.
        data = {"obs": torch.randn(10, 20)}
        dump_data(data, f"process_{iter}_obs")

    ray.init()

    output_lst = []

    for i in range(10):
        output_lst.append(process.remote(i))

    out = ray.get(output_lst)

    tracker = get_trajectory_tracker()
    ray.get(tracker.wait_for_hdfs.remote())
