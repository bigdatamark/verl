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
# Ray-based rendezvous for creating NCCL-style communicators. Rank 0 generates a unique
# communicator id and publishes it through a named Ray actor; other ranks poll Ray for the
# named actor, fetch the id and build their communicator. The collective module (providing
# NcclCommunicator and get_unique_id) is obtained from the current verl platform plugin.

import logging
import time

import ray
from ray.util import list_named_actors

from verl.plugin.platform import get_platform


    # [FUNCTION DOC]
    # HELPER: Returns the platform's collective communication module.
    # Inputs: none. Outputs: the module object returned by the current platform.
    # Exceptions: RuntimeError if the platform returns no module (message names the device).
    # Side effects: none.
def _get_collective_module():
    mod = get_platform().get_collective_module()
    if mod is None:
        raise RuntimeError(
            f"Platform '{get_platform().device_name}' does not provide a collective communication module. "
            "Override get_collective_module() in your platform implementation."
        )
    return mod


@ray.remote
    # [CLASS DOC]
    # Ray remote actor that simply stores one NCCL unique id so other processes can retrieve it
    # by looking up the actor's name.
class NCCLIDStore:
        # [FUNCTION DOC]
        # Stores the given nccl_id on the actor. Inputs: nccl_id - opaque unique id object. Outputs: none.
    def __init__(self, nccl_id):
        self._nccl_id = nccl_id

        # [FUNCTION DOC]
        # Returns the stored nccl id. No side effects.
    def get(self):
        return self._nccl_id


    # [FUNCTION DOC]
    # HELPER: Looks up a named Ray actor across all namespaces.
    # Behavior: lists all named actors, filters by exact name match. If exactly one match, returns
    # a handle via ray.get_actor using the listing's fields (name, namespace). If multiple, logs a
    # warning; if none, logs info; both cases return None.
    # Inputs: name - actor name. Outputs: actor handle or None.
    # Side effects: Ray queries and logging.
def get_nccl_id_store_by_name(name):
    all_actors = list_named_actors(all_namespaces=True)
    matched_actors = [actor for actor in all_actors if actor.get("name", None) == name]
    if len(matched_actors) == 1:
        actor = matched_actors[0]
        return ray.get_actor(**actor)
    elif len(matched_actors) > 1:
        logging.warning("multiple actors with same name found: %s", matched_actors)
    elif len(matched_actors) == 0:
        logging.info("failed to get any actor named %s", name)
    return None


def create_nccl_communicator_in_ray(
    rank: int, world_size: int, group_name: str, max_retries: int = 100, interval_s: int = 5
    # [FUNCTION DOC]
    # Creates a communicator for the given rank, rendezvousing via a named Ray actor.
    # Behavior: rank 0 creates a unique id, launches an NCCLIDStore actor named group_name holding
    # it, asserts the actor returns the same id, then builds the communicator with rank 0. Other
    # ranks retry up to max_retries times: look up the actor by name, and if found fetch the id and
    # build the communicator with their rank; otherwise log and sleep interval_s seconds.
    # Inputs: rank, world_size, group_name (actor name used as rendezvous key), max_retries,
    # interval_s (seconds between retries).
    # Outputs: the communicator object; returns None (implicitly) if a non-zero rank exhausts retries.
    # Exceptions: RuntimeError from platform lookup; AssertionError if the stored id mismatches.
    # Side effects: creates a named Ray actor (rank 0, never cleaned up), sleeps, logs.
    # KNOWN ISSUE: non-zero ranks silently return None after exhausting retries instead of raising.
):
    collective = _get_collective_module()
    NcclCommunicator = collective.NcclCommunicator
    get_unique_id = collective.get_unique_id

    if rank == 0:
        nccl_id = get_unique_id()
        nccl_id_store = NCCLIDStore.options(name=group_name).remote(nccl_id)

        assert ray.get(nccl_id_store.get.remote()) == nccl_id
        communicator = NcclCommunicator(
            ndev=world_size,
            commId=nccl_id,
            rank=0,
        )
        return communicator
    else:
        for i in range(max_retries):
            nccl_id_store = get_nccl_id_store_by_name(group_name)
            if nccl_id_store is not None:
                logging.info("nccl_id_store %s got", group_name)
                nccl_id = ray.get(nccl_id_store.get.remote())
                logging.info("nccl id for %s got: %s", group_name, nccl_id)
                communicator = NcclCommunicator(
                    ndev=world_size,
                    commId=nccl_id,
                    rank=rank,
                )
                return communicator
            logging.info("failed to get nccl_id for %d time, sleep for %d seconds", i + 1, interval_s)
            time.sleep(interval_s)
