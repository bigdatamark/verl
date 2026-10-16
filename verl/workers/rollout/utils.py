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
# Shared rollout-server utilities: max-context lookup from HF configs, running a FastAPI
# app on uvicorn with an OS-assigned port inside an asyncio loop, sync/async iterator
# adaptation, Qwen2.5-VL image-token dedup, vision placeholder token lookup, and
# Prometheus scrape-config distribution across Ray nodes. Module state: logger (level
# from VERL_LOGGING_LEVEL, default WARN).
import asyncio
import logging
import os

import numpy as np
import ray
import uvicorn
import yaml
from fastapi import FastAPI

from verl.utils.tokenizer import get_processor_token_id
from verl.workers.config.rollout import PrometheusConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_max_position_embeddings(hf_config) -> int:
    # [FUNCTION DOC]
    # Return max_position_embeddings from hf_config, falling back to hf_config.text_config
    # (multimodal models), as an int.
    # Exceptions: ValueError if neither defines it.
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):
    # [CLASS DOC]
    # HELPER: uvicorn.Server subclass that records the actually bound port (needed when
    # configured with port 0) and signals when startup has finished (success or failure).
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        # [FUNCTION DOC]
        # Initialize the server with config; actual_port None and an unset startup event.
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        # [FUNCTION DOC]
        # Run the normal startup; then set actual_port to the first socket's bound port when the
        # configured port is 0 and servers exist, otherwise to the configured port. Always sets
        # the startup event, even if startup raised (the exception propagates).
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        # [FUNCTION DOC]
        # Wait for startup to finish and return actual_port (None if startup failed before it was
        # assigned).
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    # [FUNCTION DOC]
    # Serve app on server_address with an ephemeral port in a background asyncio task.
    # Behavior: sets app.server_args, builds a uvicorn config (port 0, warning log level),
    # starts serve() as a task and awaits the reported port. If no port was reported, awaits
    # the task to surface the startup exception, else raises RuntimeError.
    # Outputs: (port, serve task).
    # Exceptions: whatever server startup raised; RuntimeError as above.
    # Side effects: binds a socket, mutates app, logs.
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    # [FUNCTION DOC]
    # Async generator yielding every item of an async iterable, or of a regular iterable.
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    # [FUNCTION DOC]
    # For processors whose image processor class name contains "Qwen2VLImageProcessor",
    # collapse each run of consecutive image/video pad tokens to a single token (keeps a
    # token unless both it and its predecessor are pad tokens); otherwise return the input
    # unchanged.
    # Outputs: list of ints when deduplicated, else the original prompt_ids object.
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.
    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    ):
        prompt_ids = np.array(prompt_ids)
        mask = np.ones(len(prompt_ids), dtype=bool)
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)
        mask[1:] &= ~(is_value[1:] & is_value[:-1])
        return prompt_ids[mask].tolist()
    else:
        return prompt_ids


def get_vision_placeholder_token_ids(processor) -> list[int]:
    # [FUNCTION DOC]
    # Return the processor's image and video placeholder token ids (each included only if
    # the processor defines it), e.g. for banning them during sampling; empty for text-only.
    """Vision placeholder token ids the policy must never sample.

    An `<|image_pad|>` or `<|video_pad|>` token only means something when a real image or video
    sits behind it: the k-th run of placeholders pairs with the k-th row of `image_grid_thw` /
    `video_grid_thw`. Nothing stops the policy from sampling one anyway -- it is an ordinary entry
    of the vocabulary -- and then that pairing silently breaks. Every downstream consumer trusts
    it, so each one has its own way of dying: `get_rope_index` walks off the end of the grid
    iterator, and `merge_multimodal_embeddings` scatters into a mask that no longer matches the
    embedding count. Both take the whole training job down with them.

    A tool that returns an image is unaffected: that image arrives in the next turn's prompt, not
    in what the model generates.

    Returns an empty list for text-only models, leaving sampling untouched.
    """
    token_ids = []
    for modality in ("image", "video"):
        token_id = get_processor_token_id(processor, modality)
        if token_id is not None:
            token_ids.append(token_id)
    return token_ids


def update_prometheus_config(config: PrometheusConfig, server_addresses: list[str], rollout_name: str | None = None):
    # [FUNCTION DOC]
    # Write a Prometheus config scraping the given rollout server addresses (plus Ray's
    # service-discovery file) to config.file on every alive Ray node, then ask Prometheus on
    # every node to reload via HTTP POST to /-/reload on config.port.
    # Behavior: warns and returns when no addresses. Uses zero-CPU Ray tasks pinned to each
    # node via node resources; prints progress. Any exception is caught and logged as error.
    # Inputs: config - PrometheusConfig (file, port); server_addresses - "host:port" list;
    # rollout_name - backend name used only in the printed message.
    # Side effects: writes files on all nodes, spawns curl subprocesses, network calls.
    """
    Update Prometheus configuration file with server addresses and reload on first node.

    server_addresses: vllm or sglang server addresses

    rollout_name: name of the rollout backend (e.g., "vllm", "sglang")
    """

    if not server_addresses:
        logger.warning("No server addresses available to update Prometheus config")
        return

    try:
        # Get Prometheus config file path from environment or use default
        prometheus_config_json = {
            "global": {"scrape_interval": "10s", "evaluation_interval": "10s"},
            "scrape_configs": [
                {
                    "job_name": "ray",
                    "file_sd_configs": [{"files": ["/tmp/ray/prom_metrics_service_discovery.json"]}],
                },
                {"job_name": "rollout", "static_configs": [{"targets": server_addresses}]},
            ],
        }

        # Write configuration file to all nodes
        @ray.remote(num_cpus=0)
        def write_config_file(config_data, config_path):
            # [FUNCTION DOC]
            # HELPER: Ray task that creates the parent directory and dumps config_data as YAML to
            # config_path; returns True.
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            return True

        # Reload Prometheus on all nodes. Only master node should succeed, skip errors on other nodes.
        @ray.remote(num_cpus=0)
        def reload_prometheus(port):
            # [FUNCTION DOC]
            # HELPER: Ray task that resolves this host's IP, runs curl POST against the Prometheus
            # reload endpoint on port (10 s timeout) and prints; all errors are ignored. Returns None.
            import socket
            import subprocess

            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)

            reload_url = f"http://{ip_address}:{port}/-/reload"

            try:
                subprocess.run(["curl", "-X", "POST", reload_url], capture_output=True, text=True, timeout=10)
                print(f"Reloading Prometheus on node: {reload_url}")
            except Exception:
                # Skip errors on non-master nodes
                pass

        # Get all available nodes and schedule tasks on each node
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node["Alive"]]

        # Write config files on all nodes
        write_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = write_config_file.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(prometheus_config_json, config.file)
            write_tasks.append(task)

        ray.get(write_tasks)

        server_type = rollout_name.upper() if rollout_name else "rollout"
        print(f"Updated Prometheus configuration at {config.file} with {len(server_addresses)} {server_type} servers")

        # Reload Prometheus on all nodes
        reload_tasks = []
        for node in alive_nodes:
            node_ip = node["NodeManagerAddress"]
            task = reload_prometheus.options(
                resources={"node:" + node_ip: 0.001}  # Schedule to specific node
            ).remote(config.port)
            reload_tasks.append(task)

        ray.get(reload_tasks)

    except Exception as e:
        logger.error(f"Failed to update Prometheus configuration: {e}")
