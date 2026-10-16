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
# Prefill/decode (PD) disaggregated SGLang replica: one prefill server plus N decode
# servers per replica, all on a single node, with possibly different TP sizes for
# prefill and decode. Module state: logger (INFO).
"""SGLang PD-disaggregated replica: 1 prefill + N decode servers per replica,
asymmetric TP supported. MVP: prefill_replicas=1, whole replica on one node."""

import asyncio
import logging
import os
from dataclasses import replace as _dc_replace
from typing import Optional

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from verl.utils.device import is_torch_npu_available
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.workers.config import RolloutConfig
from verl.workers.rollout.sglang_rollout.async_sglang_server import (
    SGLangReplica,
    visible_devices_keyword,
)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SGLangPDReplica(SGLangReplica):
    # [CLASS DOC]
    # SGLangReplica subclass that launches one prefill SGLangHttpServer and
    # disaggregation.decode_replicas decode servers, wires the prefill server to its decode
    # peers, and exposes the prefill server as the replica's entry address.
    """Replica that runs SGLang in prefill-decode disaggregated mode."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: DictConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
    ):
        # [FUNCTION DOC]
        # Behavior: initializes the base replica, then validates and records PD topology:
        # requires disaggregation.enabled, exactly one prefill replica, data_parallel_size 1,
        # and prefill_tp + n_decode * decode_tp <= gpus_per_node (decode_tp defaults to the
        # rollout TP). Overrides world_size to that PD total and recomputes
        # gpus_per_replica_node and nnodes; initializes empty server lists/addresses.
        # Inputs: replica_rank, config, model_config, gpus_per_node, reward/teacher flags.
        # Exceptions: AssertionError if disaggregation is disabled; NotImplementedError for
        # unsupported topologies.
        # KNOWN ISSUE: unlike SGLangReplica, does not accept or forward name_suffix.
        super().__init__(
            replica_rank,
            config,
            model_config,
            gpus_per_node,
            is_reward_model,
            is_teacher_model,
        )
        disagg = self.config.disaggregation
        assert disagg.enabled, "SGLangPDReplica requires rollout.disaggregation.enabled=True"

        if disagg.prefill_replicas != 1:
            raise NotImplementedError(f"prefill_replicas=1 only (got {disagg.prefill_replicas})")
        self._n_prefill = disagg.prefill_replicas
        self._n_decode = disagg.decode_replicas

        self._prefill_tp = self.config.tensor_model_parallel_size
        # Inline decode_tp default: OmegaConf/Ray serialization drops dataclass methods.
        self._decode_tp = (
            disagg.decode_tensor_model_parallel_size
            if disagg.decode_tensor_model_parallel_size is not None
            else self._prefill_tp
        )

        pd_world_size = self._prefill_tp + self._n_decode * self._decode_tp
        if pd_world_size > gpus_per_node:
            raise NotImplementedError(
                f"PD replica needs {pd_world_size} GPUs but gpus_per_node={gpus_per_node}; "
                f"use more replicas to span nodes"
            )

        if self.config.data_parallel_size != 1:
            raise NotImplementedError(f"data_parallel_size=1 only (got {self.config.data_parallel_size})")
        self.world_size = pd_world_size
        self.gpus_per_replica_node = min(self.gpus_per_node, self.world_size)
        assert self.world_size % self.gpus_per_replica_node == 0
        self.nnodes = self.world_size // self.gpus_per_replica_node

        self._prefill_servers: list[ActorHandle] = []
        self._decode_servers: list[ActorHandle] = []
        self._prefill_server_address: Optional[str] = None
        self._decode_server_addresses: list[str] = []
        self._bootstrap_port: Optional[int] = None

    async def launch_servers(self):
        # [FUNCTION DOC]
        # Launch the prefill server and then each decode server sequentially, then link them.
        # Behavior: collects (node id, visible devices) from every worker; uses the configured
        # bootstrap port or reserves a free one on this node's IP. The first prefill_tp workers
        # form the prefill server; consecutive decode_tp-sized slices form each decode server.
        # Each server is launched via _launch_one with its role and the shared bootstrap port.
        # Records formatted "host:port" addresses (IPv6 bracketed), sets the prefill server as
        # _server_handle/_server_address, sets servers to prefill + decode handles, calls
        # set_pd_peer on the prefill server with the decode handles and prefill host, and logs.
        # Exceptions: AssertionError if worker count mismatches world_size or on NPU.
        # Side effects: creates named Ray actors, binds/closes a socket.
        # KNOWN ISSUE: the code comment says the reserved bootstrap socket is held open until the
        # prefill server binds it, but it is closed before the prefill server is launched, so the
        # port-race window still exists.
        assert len(self.workers) == self.world_size
        assert not is_torch_npu_available(check_device=False), "PD on NPU not validated"

        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        os.environ[visible_devices_keyword],
                    )
                )
                for worker in self.workers
            ]
        )

        # Hold the bootstrap socket open until prefill binds it; closing earlier
        # opens a TOCTOU window where another process can grab the port.
        bootstrap_port = self.config.disaggregation.bootstrap_port
        self._bootstrap_sock = None
        if bootstrap_port is None:
            prefill_host_ip = ray.util.get_node_ip_address().strip("[]")
            bootstrap_port, self._bootstrap_sock = get_free_port(prefill_host_ip, with_alive_sock=True)
        self._bootstrap_port = bootstrap_port

        prefill_end = self._prefill_tp
        prefill_workers = self.workers[0:prefill_end]
        prefill_node_id = worker_infos[0][0]
        prefill_devs = self._collect_cuda_devices(worker_infos[0:prefill_end])

        if self._bootstrap_sock is not None:
            self._bootstrap_sock.close()
            self._bootstrap_sock = None

        [prefill_server] = await self._launch_one(
            role="prefill",
            workers=prefill_workers,
            node_id=prefill_node_id,
            cuda_visible_devices=prefill_devs,
            bootstrap_port=self._bootstrap_port,
            tp=self._prefill_tp,
            actor_name=f"sglang_server_{self.replica_rank}_0",
        )
        self._prefill_servers = [prefill_server]

        prefill_address, prefill_port = await prefill_server.get_server_address.remote()

        def _fmt(addr, port):
            # [FUNCTION DOC]
            # HELPER: Format address and port as "host:port", bracketing IPv6 hosts.
            return f"[{addr}]:{port}" if is_valid_ipv6_address(addr) else f"{addr}:{port}"

        self._prefill_server_address = _fmt(prefill_address, prefill_port)

        self._decode_servers = []
        self._decode_server_addresses = []
        for i in range(self._n_decode):
            start = self._prefill_tp + i * self._decode_tp
            end = start + self._decode_tp
            workers_i = self.workers[start:end]
            node_id_i = worker_infos[start][0]
            devs_i = self._collect_cuda_devices(worker_infos[start:end])

            [decode_server] = await self._launch_one(
                role="decode",
                workers=workers_i,
                node_id=node_id_i,
                cuda_visible_devices=devs_i,
                bootstrap_port=self._bootstrap_port,
                tp=self._decode_tp,
                actor_name=f"sglang_server_decode_{self.replica_rank}_{i}",
            )
            self._decode_servers.append(decode_server)

            d_addr, d_port = await decode_server.get_server_address.remote()
            self._decode_server_addresses.append(_fmt(d_addr, d_port))

        self._server_address = self._prefill_server_address
        self._server_handle = prefill_server
        self.servers = list(self._prefill_servers) + list(self._decode_servers)

        await prefill_server.set_pd_peer.remote(list(self._decode_servers), prefill_address)

        logger.info(
            f"SGLangPDReplica rank={self.replica_rank} launched: "
            f"prefill={self._prefill_server_address}, "
            f"decodes=[{', '.join(self._decode_server_addresses)}], "
            f"bootstrap_port={self._bootstrap_port}"
        )

    @staticmethod
    def _collect_cuda_devices(worker_infos) -> str:
        # [FUNCTION DOC]
        # HELPER: Union the comma-separated device ids of the given (node id, devices) pairs,
        # dropping blanks, and return them sorted and comma-joined.
        devs = set()
        for _, dev_str in worker_infos:
            for d in dev_str.split(","):
                if d.strip():
                    devs.add(int(d))
        return ",".join(str(d) for d in sorted(devs))

    async def _launch_one(
        self,
        role: str,
        workers: list[ActorHandle],
        node_id: str,
        cuda_visible_devices: str,
        bootstrap_port: int,
        tp: int,
        actor_name: str,
    ) -> list[ActorHandle]:
        # [FUNCTION DOC]
        # HELPER: Create and launch one single-node SGLangHttpServer actor for a PD role.
        # Behavior: base_gpu_id is 0 unless Ray's NOSET visible-devices env var is set, in which
        # case (replica_rank * world_size) mod gpus_per_node. Copies the rollout config with
        # tensor_model_parallel_size set to tp, creates the actor with hard node affinity, the
        # NOSET env var set to "1", the given name and max_concurrency, passing the role and
        # bootstrap port, and awaits launch_server.
        # Outputs: single-element list with the actor handle.
        # KNOWN ISSUE: with the NOSET env var set, every server of the replica (prefill and each
        # decode) gets the same base_gpu_id, which does not account for the slice offset of each
        # role's GPUs.
        base_gpu_id = 0
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            base_gpu_id = (0 + self.replica_rank * self.world_size) % self.gpus_per_node

        pool_config = _dc_replace(self.config, tensor_model_parallel_size=tp)

        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False
            ),
            runtime_env={"env_vars": {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}},
            name=actor_name,
            max_concurrency=self.max_concurrency,
        ).remote(
            config=pool_config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            nnodes=1,
            cuda_visible_devices=cuda_visible_devices,
            base_gpu_id=base_gpu_id,
            disaggregation_role=role,
            disaggregation_bootstrap_port=bootstrap_port,
        )
        await server.launch_server.remote(master_address=None, master_port=None)
        return [server]
