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
# Self-contained asyncio demo of the streaming pattern used by the fully-async rollouter: a producer feeds items into
# a queue, a submitter launches bounded-concurrency processing tasks, and a consumer reads results. Runnable as a
# script; no external dependencies.

import asyncio
import random
import time


class SimpleStreamingSystem:
    # [CLASS DOC]
    # Demo system holding the concurrency limit, an input queue, a result queue and a consumed-result counter.
    """Simplified streaming system demonstration"""

    def __init__(self, max_concurrent_tasks: int = 4):
        # [FUNCTION DOC]
        # Behavior: stores the concurrency limit, creates both queues, sets counter to 0. Inputs: max_concurrent_tasks.
        self.max_concurrent_tasks = max_concurrent_tasks
        self.data_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.consumer_count = 0

    # Data stream coroutine
    async def data_stream(self):
        # [FUNCTION DOC]
        # Behavior: enqueues 8 test items (with spacing), waits 3 s, enqueues 5 more, waits 1 s, then enqueues the "DONE"
        # sentinel. Side effects: prints.
        # Add initial data
        # Prepare test data
        test_data = [{"id": f"task_{i}", "content": f"data_{i}"} for i in range(8)]
        await self.add_data_stream(test_data)

        # Simulate subsequent data stream
        await asyncio.sleep(3)
        print("\nAdding second batch of data...")
        extra_data = [{"id": f"extra_{i}", "content": f"extra_data_{i}"} for i in range(5)]
        await self.add_data_stream(extra_data)

        # Send termination signal
        await asyncio.sleep(1)
        await self.data_queue.put("DONE")
        print("Sending termination signal")

    async def add_data_stream(self, data_list: list[dict]):
        # [FUNCTION DOC]
        # Behavior: puts each item into the input queue, sleeping 0.8 s between items (not after the last). Prints progress.
        # Inputs: data_list - list of mappings with "id".
        """Simulate data stream"""
        print("Starting to add data stream...")

        for i, data_item in enumerate(data_list):
            await self.data_queue.put(data_item)
            print(f"Data {data_item['id']} added to pending queue")

            # Simulate interval between data streams
            if i < len(data_list) - 1:  # Don't wait after the last item
                await asyncio.sleep(0.8)

        print("Initial data stream added successfully")

    async def _process_data_async(self, data_item: dict):
        # [FUNCTION DOC]
        # HELPER: Sleeps a random 1-3 s then puts a result mapping (id, processed content, rounded time, completion timestamp)
        # into the result queue. Inputs: data_item with "id" and "content".
        """Asynchronously process a single data item"""
        data_id = data_item["id"]
        content = data_item["content"]

        # Simulate different processing times (1-3 seconds)
        processing_time = random.uniform(1, 3)

        print(f"    Starting to process {data_id}, estimated time {processing_time:.1f}s")

        # Asynchronously wait for processing completion
        await asyncio.sleep(processing_time)

        result = {
            "id": data_id,
            "processed_content": f"Processed {content}",
            "processing_time": round(processing_time, 2),
            "completed_at": time.time(),
        }

        # Immediately put into result queue
        await self.result_queue.put(result)
        print(f"    {data_id} processing completed! (took {processing_time:.1f}s) -> Added to result queue")

    async def _submit_worker(self):
        # [FUNCTION DOC]
        # HELPER: Loop: take next item; on "DONE" gather remaining tasks and exit. While at the concurrency limit, wait for at
        # least one task and await finished ones (logging failures). Then create a processing task and add it to the set.
        """Stream submission worker coroutine"""
        active_tasks = set()

        print("Stream submitter started...")

        while True:
            # Get data to process
            data_item = await self.data_queue.get()

            if data_item == "DONE":
                print("Received termination signal, waiting for remaining tasks to complete...")
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                break

            # Check concurrent limit
            while len(active_tasks) >= self.max_concurrent_tasks:
                print(f"Reached maximum concurrency {self.max_concurrent_tasks}, waiting for tasks to complete...")
                done_tasks, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

                # Clean up completed tasks
                for task in done_tasks:
                    try:
                        await task
                        print(f"Task completed {task}")
                    except Exception as e:
                        print(f"Task execution failed: {e}")

            # Immediately submit new task
            task = asyncio.create_task(self._process_data_async(data_item), name=f"active {data_item}")
            active_tasks.add(task)

            print(f"Submitted task {data_item['id']}, current concurrency: {len(active_tasks)}")

    async def _consumer_worker(self):
        # [FUNCTION DOC]
        # HELPER: Forever: wait up to 2 s for a result, increment counter and print; on timeout print and sleep 0.5 s. Runs
        # until cancelled.
        """Result consumer coroutine"""
        print("Consumer started...")

        while True:
            try:
                # Get processing result from result queue
                result = await asyncio.wait_for(self.result_queue.get(), timeout=2.0)

                self.consumer_count += 1

                print(
                    f"Consumed #{self.consumer_count}: {result['id']} "
                    f"(processing time {result['processing_time']}s) - {result['processed_content']}"
                )

            except asyncio.TimeoutError:
                print("    Consumer waiting...")
                await asyncio.sleep(0.5)

    async def run_demo(self):
        # [FUNCTION DOC]
        # Behavior: starts producer, submitter and consumer tasks; awaits producer then submitter; finally cancels submitter
        # and consumer and gathers them; prints the consumed count.
        """Run demonstration"""
        print("=" * 60)
        print(f"Maximum concurrency: {self.max_concurrent_tasks}")
        print("=" * 60)

        # Start core coroutines
        stream_task = asyncio.create_task(self.data_stream())
        submit_task = asyncio.create_task(self._submit_worker())
        consumer_task = asyncio.create_task(self._consumer_worker())

        try:
            # Wait for data stream to complete
            await stream_task
            print("Data stream completed")

            # Wait for processing to complete
            await submit_task
            print("All tasks processed")

        finally:
            # Cleanup
            submit_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(submit_task, consumer_task, return_exceptions=True)

        print(f"\nFinal statistics: Consumed {self.consumer_count} results")


async def main():
    # [FUNCTION DOC]
    # Behavior: builds a system with concurrency 3 and runs the demo.
    """Main function"""
    system = SimpleStreamingSystem(max_concurrent_tasks=3)
    await system.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
