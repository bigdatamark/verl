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
# Tests for the TensorRT-LLM rollout HTTP adapter (AsyncTRTLLMHttpAdapter, aiohttp mocked) and a GPU
# integration test that brings up a standalone TRTLLMReplica with TP 2 on a local Ray cluster.

import asyncio
import os
import subprocess
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import ray

from verl.workers.rollout.trtllm_rollout.trtllm_async_server import TRTLLMReplica
from verl.workers.rollout.trtllm_rollout.trtllm_rollout import AsyncTRTLLMHttpAdapter


class TestAsyncTRTLLMHttpAdapter:
    # [CLASS DOC]
    # Unit tests of the adapter's async request helper with mocked aiohttp sessions.
    def _build_async_session(
        self,
        *,
        adapter: AsyncTRTLLMHttpAdapter,
        method: str,
        response: AsyncMock | None = None,
    ) -> tuple[AsyncMock, AsyncMock]:
        # [FUNCTION DOC]
        # HELPER: builds a mock aiohttp session (spec ClientSession, not closed) and an async context manager yielding
        # it; if a response is given, the session's named method (get or post) returns an async context manager
        # yielding that response. Inputs: adapter (unused), method name, optional response mock.
        # Outputs: (session context manager mock, session mock).
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.closed = False

        if response is not None:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = response
            getattr(mock_session, method).return_value = mock_context_manager

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        return mock_session_cm, mock_session

    @pytest.mark.asyncio
    async def test_make_async_request_get_method(self):
        # [FUNCTION DOC]
        # Patches _get_session; asserts a GET returns the JSON body and session.get was called once with the
        # http://localhost:8000/test_endpoint URL and the adapter timeout.
        """Test HTTP GET request."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        get_response = AsyncMock()
        get_response.status = 200
        get_response.headers = {"Content-Type": "application/json"}
        get_response.raise_for_status = Mock()
        get_response.json = AsyncMock(return_value={"data": "test"})

        get_session_cm, get_session = self._build_async_session(
            adapter=adapter,
            method="get",
            response=get_response,
        )
        with patch.object(adapter, "_get_session", return_value=get_session_cm):
            get_result = await adapter._make_async_request("test_endpoint", method="GET")

        assert get_result == {"data": "test"}
        get_session.get.assert_called_once_with("http://localhost:8000/test_endpoint", timeout=adapter.timeout)

    @pytest.mark.asyncio
    async def test_make_async_request_post_method(self):
        # [FUNCTION DOC]
        # Same for POST with a JSON payload; asserts result and that post was called with URL, json and timeout.
        """Test HTTP POST request."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        post_response = AsyncMock()
        post_response.status = 200
        post_response.headers = {"Content-Type": "application/json"}
        post_response.raise_for_status = Mock()
        post_response.json = AsyncMock(return_value={"status": "ok"})

        post_session_cm, post_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=post_response,
        )
        with patch.object(adapter, "_get_session", return_value=post_session_cm):
            post_result = await adapter._make_async_request("test_endpoint", {"param": "value"})

        assert post_result == {"status": "ok"}
        post_session.post.assert_called_once_with(
            "http://localhost:8000/test_endpoint", json={"param": "value"}, timeout=adapter.timeout
        )

    @pytest.mark.asyncio
    async def test_make_async_request_http_error(self):
        # [FUNCTION DOC]
        # Response raise_for_status raises ClientResponseError (500); asserts the error propagates.
        """Test HTTP error handling."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.raise_for_status = Mock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(real_url="http://localhost:8000/test_endpoint"),
                history=(),
                status=500,
                message="server error",
            )
        )

        mock_session_cm, _mock_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=mock_response,
        )
        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            with pytest.raises(aiohttp.ClientResponseError):
                await adapter._make_async_request("test_endpoint", {"param": "value"})

    @pytest.mark.asyncio
    async def test_make_async_request_max_attempts_exceeded(self):
        # [FUNCTION DOC]
        # With max_attempts 1 and post raising asyncio TimeoutError, asserts RuntimeError "Failed to complete async request".
        """Test max retries exceeded."""
        adapter = AsyncTRTLLMHttpAdapter(host="localhost", port=8000, max_attempts=1)

        mock_session_cm, mock_session = self._build_async_session(
            adapter=adapter,
            method="post",
            response=None,
        )
        mock_session.post.side_effect = asyncio.TimeoutError()
        with patch.object(adapter, "_get_session", return_value=mock_session_cm):
            with pytest.raises(RuntimeError, match="Failed to complete async request"):
                await adapter._make_async_request("test_endpoint", {"param": "value"})


class TestTRTLLMServerAdapter:
    # [CLASS DOC]
    # GPU integration tests for TRTLLM replica initialization.
    def test_init_without_device_mesh(self):
        # [FUNCTION DOC]
        # Temporarily sets RANK=0, starts a local Ray cluster (TLLM_RAY_FORCE_LOCAL_CLUSTER defaulted to 1), composes
        # ppo_trainer (config dir verl/verl/trainer/config or verl/trainer/config relative to CWD), configures 2 GPUs,
        # Qwen2.5-1.5B-Instruct under TRTLLM_TEST_MODEL_PATH_ROOT (default ~/models), trtllm async rollout with TP 2 and
        # load_format auto. Runs init_standalone on replica 0 and asserts 2 workers, replica rank 0, worker 0 is leader
        # and worker 1 is not. Side effects: env var restore, ray.shutdown and "ray stop" subprocess in finally.
        """Test ServerAdapter init path without device mesh."""
        from hydra import compose, initialize_config_dir

        prev_rank = os.environ.get("RANK")
        os.environ["RANK"] = "0"

        try:
            os.environ.setdefault("TLLM_RAY_FORCE_LOCAL_CLUSTER", "1")
            ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

            config_dir = os.path.abspath("verl/verl/trainer/config")
            if not os.path.exists(config_dir):
                config_dir = os.path.abspath("verl/trainer/config")

            with initialize_config_dir(config_dir=config_dir, version_base=None):
                config = compose(config_name="ppo_trainer")

            config.trainer.n_gpus_per_node = 2
            config.trainer.nnodes = 1
            model_root = os.path.expanduser(os.getenv("TRTLLM_TEST_MODEL_PATH_ROOT", "~/models"))
            config.actor_rollout_ref.model.path = os.path.join(model_root, "Qwen/Qwen2.5-1.5B-Instruct")
            config.actor_rollout_ref.rollout.name = "trtllm"
            config.actor_rollout_ref.rollout.mode = "async"
            config.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
            # No trainer to sync weights from, so the server must load them from disk.
            config.actor_rollout_ref.rollout.load_format = "auto"

            rollout_config = config.actor_rollout_ref.rollout
            model_config = config.actor_rollout_ref.model

            replica = TRTLLMReplica(
                replica_rank=0,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=2,
            )

            asyncio.run(replica.init_standalone())

            assert len(replica.workers) == 2

            worker0 = replica.workers[0]
            worker1 = replica.workers[1]
            replica_rank = ray.get(worker0.get_replica_rank.remote())
            is_leader_rank_0 = ray.get(worker0.is_leader_rank.remote())
            is_leader_rank_1 = ray.get(worker1.is_leader_rank.remote())

            assert replica_rank == 0
            assert is_leader_rank_0 is True
            assert is_leader_rank_1 is False
        finally:
            if prev_rank is None:
                os.environ.pop("RANK", None)
            else:
                os.environ["RANK"] = prev_rank
            ray.shutdown()
            subprocess.run(["ray", "stop"], capture_output=True)
