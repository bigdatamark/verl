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
# Ray-actor-based bounded FIFO message queue connecting the rollouter (producer) and trainer (consumer), plus a
# client wrapper exposing async (and some blocking) access to it.

import asyncio
import logging
from collections import deque
from typing import Any

import ray
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    # [CLASS DOC]
    # Ray actor (2 CPUs, up to 20 concurrent calls). Holds a bounded deque of samples, a separate unbounded validation
    # deque, a running flag, an asyncio lock with a consumer condition, and produced/consumed/dropped counters.
    """
    Simplified Ray-based asynchronous message queue for communication between Rollouter and Trainer
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        # [FUNCTION DOC]
        # Behavior: stores config; raises ValueError if max_queue_size is none; creates the bounded deque and validation
        # deque, lock/condition, zeroed counters; logs.
        # Inputs: config; max_queue_size - int (default 1000). Exceptions: ValueError for none size.
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue = deque(maxlen=self.max_queue_size)

        self.val_queue = deque()

        # Asyncio for message handling
        self.running = True

        # async safe
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        # statistic message
        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        print(f"[MessageQueue] initialized with max_queue_size={max_queue_size}")

    async def put_sample(self, sample: Any) -> bool:
        # [FUNCTION DOC]
        # Behavior: under the lock, if the queue is full drops the oldest entry (counting it and warning), appends the new
        # sample, increments produced, notifies waiting consumers, logs every 100 produced.
        # Inputs: sample - any (usually serialized bytes, or none as termination marker).
        # Outputs: False if an old sample was dropped to make room, else True.
        """
        Put a batch sample into the queue

        Args:
            sample: Sample data

        Returns:
            bool: Whether the sample was successfully put into the queue
        """
        async with self._lock:
            # If queue is full, remove the oldest sample (rarely happens)
            is_drop = False
            if len(self.queue) >= self.max_queue_size:
                self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                logger.warning("Queue full, dropped sample")
            self.queue.append(sample)
            self.total_produced += 1

            # Notify waiting consumers
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            if is_drop:
                return False
            return True

    async def get_sample(self) -> Any | None:
        # [FUNCTION DOC]
        # Behavior: under the lock waits on the condition while the queue is empty and running; returns none if shut down
        # and empty; otherwise pops the oldest entry, increments consumed, and returns (sample, remaining length).
        # Outputs: a (sample, int) pair, or none.
        # KNOWN ISSUE: return type is inconsistent (pair vs bare none); callers that unpack a pair fail after shutdown.
        """
        Get a single sample from the queue, wait until one is available

        Returns:
            Any: Single sample data or None if queue is closed
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            # If queue is closed and empty, return None
            if not self.running and len(self.queue) == 0:
                return None

            # Get one sample
            data = self.queue.popleft()
            self.total_consumed += 1
            return data, len(self.queue)

    async def get_queue_size(self) -> int:
        # [FUNCTION DOC]
        # Behavior: returns current queue length under the lock.
        """Get current queue length"""
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: returns a mapping with queue_size, total_produced, total_consumed, dropped_samples, max_queue_size.
        """Get queue statistics"""
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
            }

    async def clear_queue(self):
        # [FUNCTION DOC]
        # Behavior: empties the queue under the lock and logs how many were removed. Counters unchanged.
        """Clear the queue"""
        async with self._lock:
            cleared_count = len(self.queue)
            self.queue.clear()
            logger.info(f"Cleared {cleared_count} samples from queue")

    async def shutdown(self):
        # [FUNCTION DOC]
        # Behavior: sets running false and wakes all waiting consumers; logs.
        """Shutdown the message queue"""
        async with self._lock:
            self.running = False
            # Notify all waiting coroutines so they can exit
            self._consumer_condition.notify_all()
        logger.info("MessageQueue shutdown")

    async def get_memory_usage(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: estimates memory: if non-empty, takes the shallow size of the first sample plus 1000 bytes per entry of
        # its original_batch_dict "batch" (if that attribute exists) plus 5000 if it has agent_loop_output, times the sample
        # count; falls back to 15000 bytes per sample on error. Returns sample count, bytes and megabytes.
        """Get memory usage statistics"""
        async with self._lock:
            # Estimate memory usage of samples in queue
            import sys

            total_size = 0
            sample_count = len(self.queue)

            if sample_count > 0:
                # Estimate size of a single sample (simplified estimation)
                sample = list(self.queue)[0]
                try:
                    sample_size = sys.getsizeof(sample)
                    # Since we now store RolloutSample directly, estimate based on its components
                    if hasattr(sample, "original_batch_dict") and sample.original_batch_dict:
                        # Estimate batch data size
                        batch_data = sample.original_batch_dict.get("batch", {})
                        sample_size += len(batch_data) * 1000  # Roughly estimate 1KB per batch entry
                    if hasattr(sample, "agent_loop_output"):
                        # Estimate AgentLoopOutput size
                        sample_size += 5000  # Roughly estimate 5KB for AgentLoopOutput
                    total_size = sample_size * sample_count
                except Exception:
                    total_size = sample_count * 15000  # Roughly estimate 15KB per RolloutSample

            return {
                "queue_samples": sample_count,
                "estimated_memory_bytes": total_size,
                "estimated_memory_mb": total_size / (1024 * 1024),
            }

    async def put_validate(self, data):
        # [FUNCTION DOC]
        # Behavior: appends data to the validation deque under the lock.
        async with self._lock:
            self.val_queue.append(data)

    async def get_validate(self):
        # [FUNCTION DOC]
        # Behavior: pops and returns the oldest validation entry, or none if empty.
        async with self._lock:
            if self.val_queue:
                return self.val_queue.popleft()
            else:
                return None


class MessageQueueClient:
    # [CLASS DOC]
    # Thin client wrapping a MessageQueue actor handle; async methods bridge Ray futures into asyncio, *_sync methods
    # block with ray.get. Can be passed between actors.
    """Asyncio-compatible MessageQueue client for communicating with MessageQueue Actor"""

    def __init__(self, queue_actor: Any):
        # [FUNCTION DOC]
        # Behavior: stores the actor handle.
        self.queue_actor = queue_actor

    async def put_sample(self, sample: Any) -> bool:
        # [FUNCTION DOC]
        # Behavior: forwards put_sample and awaits the result. Outputs: bool.
        """Put batch into queue (async)"""
        future = self.queue_actor.put_sample.remote(sample)
        return await asyncio.wrap_future(future.future())

    async def put_validate(self, data: Any) -> bool:
        # [FUNCTION DOC]
        # Behavior: forwards put_validate and awaits it. Outputs: the actor's result (none).
        future = self.queue_actor.put_validate.remote(data)
        return await asyncio.wrap_future(future.future())

    def get_validate_sync(self) -> Any | None:
        # [FUNCTION DOC]
        # Behavior: blocking get_validate. Outputs: entry or none.
        return ray.get(self.queue_actor.get_validate.remote())

    async def get_sample(self) -> Any | None:
        # [FUNCTION DOC]
        # Behavior: forwards get_sample and awaits it. Outputs: (sample, remaining length) or none.
        """Get single sample from queue, wait until one is available (async)"""
        future = self.queue_actor.get_sample.remote()
        return await asyncio.wrap_future(future.future())

    async def get_queue_size(self) -> int:
        # [FUNCTION DOC]
        # Behavior: forwards get_queue_size and awaits it. Outputs: int.
        """Get queue size (async)"""
        future = self.queue_actor.get_queue_size.remote()
        return await asyncio.wrap_future(future.future())

    async def get_statistics(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: forwards get_statistics and awaits it. Outputs: mapping.
        """Get statistics (async)"""
        future = self.queue_actor.get_statistics.remote()
        return await asyncio.wrap_future(future.future())

    async def clear_queue(self):
        # [FUNCTION DOC]
        # Behavior: forwards clear_queue and awaits completion.
        """Clear queue (async)"""
        future = self.queue_actor.clear_queue.remote()
        await asyncio.wrap_future(future.future())

    async def shutdown(self):
        # [FUNCTION DOC]
        # Behavior: forwards shutdown and awaits completion.
        """Shutdown queue (async)"""
        future = self.queue_actor.shutdown.remote()
        await asyncio.wrap_future(future.future())

    async def get_memory_usage(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: forwards get_memory_usage and awaits it. Outputs: mapping.
        """Get memory usage statistics (async)"""
        future = self.queue_actor.get_memory_usage.remote()
        return await asyncio.wrap_future(future.future())

    def get_sample_sync(self) -> Any | None:
        # [FUNCTION DOC]
        # Behavior: blocking get_sample (deprecated). Outputs: pair or none.
        """Get single sample from queue (sync - deprecated, use get_sample instead)"""
        return ray.get(self.queue_actor.get_sample.remote())

    def get_statistics_sync(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: blocking get_statistics (deprecated). Outputs: mapping.
        """Get statistics (sync - deprecated, use get_statistics instead)"""
        return ray.get(self.queue_actor.get_statistics.remote())

    def get_queue_size_sync(self) -> int:
        # [FUNCTION DOC]
        # Behavior: blocking get_queue_size for callers outside an event loop. Outputs: int.
        """Get current queue length (sync, blocking).

        Used by callers running outside an event loop (e.g. synchronous
        post-processing hooks) that need an immediate snapshot of the queue size.
        """
        return ray.get(self.queue_actor.get_queue_size.remote())
