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
# Tests for the private helper _settle_session_tasks in the v1 transfer-queue agent loop, which awaits a
# group of asyncio session tasks and collects their exceptions instead of propagating on first failure.

import asyncio

from verl.trainer.ppo.v1.agent_loop_tq import _settle_session_tasks


def test_settle_session_tasks_waits_for_siblings_after_failure():
    # [FUNCTION DOC]
    # Behavior: runs an event loop with one task that raises immediately and one that sleeps briefly then sets
    # an event; asserts the helper waited for the sibling (event set), all tasks are done, and exactly one
    # RuntimeError was returned in the error list.
    # Inputs: none. Outputs: none. Exceptions: AssertionError on mismatch. Side effects: runs asyncio.run.
    async def run():
        # [FUNCTION DOC]
        # HELPER: async body of the test; creates the two tasks, awaits the helper and performs the asserts.
        # Inputs: none. Outputs: none. Side effects: sets the shared event via the sibling task.
        settled = asyncio.Event()

        async def fail():
            # [FUNCTION DOC]
            # HELPER: coroutine that immediately raises RuntimeError("session failed").
            raise RuntimeError("session failed")

        async def finish_later():
            # [FUNCTION DOC]
            # HELPER: coroutine that sleeps 0.01s then sets the enclosing event to signal completion.
            await asyncio.sleep(0.01)
            settled.set()

        tasks = [asyncio.create_task(fail()), asyncio.create_task(finish_later())]
        errors = await _settle_session_tasks(tasks)

        assert settled.is_set()
        assert all(task.done() for task in tasks)
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)

    asyncio.run(run())
