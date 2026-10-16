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
# FSDP backend for the model merger: loads per-rank FSDP checkpoint files (model_world_size_{W}_rank_{r}.pt),
# reassembles DTensor shards into full tensors, and saves or validates the merged HF model.

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.distributed._tensor import Placement, Shard

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from tqdm import tqdm

from .base_model_merger import BaseModelMerger


def merge_non_dtensor_shards(key: str, shards: list[torch.Tensor]) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: verifies that every rank's copy of a non-DTensor entry has the same shape and values as rank 0's copy.
    # Inputs: key - parameter name (for error messages); shards - per-rank tensors in rank order.
    # Outputs: rank 0's tensor.
    # Exceptions: ValueError naming the first rank that differs.
    """Return the single copy of a plain (non-DTensor) checkpoint entry.

    FSDP only produces DTensors for what it actually shards; plain tensors are buffers and
    unwrapped parameters, replicated in full on every rank (or the whole tensor when
    world_size == 1). Concatenating them would give a world_size-times oversized tensor
    and fails outright for 0-d tensors, so keep one copy and fail loudly if the ranks
    disagree, which would mean the checkpoint is malformed.
    """
    for rank, shard in enumerate(shards[1:], start=1):
        if shard.shape != shards[0].shape or not torch.equal(shard, shards[0]):
            raise ValueError(
                f"Non-DTensor entry {key!r} differs between rank 0 and rank {rank}; "
                "expected a replicated buffer/parameter identical on every rank"
            )
    return shards[0]


class FSDPModelMerger(BaseModelMerger):
    # [CLASS DOC]
    # Merger for FSDP checkpoints stored in config.local_dir. Supports 1-D "fsdp" and 2-D ("ddp", "fsdp") meshes;
    # tensor parallelism is not supported. All merged tensors are cast to bfloat16.
    """
    Model merger for FSDP (Fully Sharded Data Parallel) checkpoints.

    This class handles the conversion of FSDP distributed checkpoints into HuggingFace format.
    FSDP shards model parameters across multiple processes, and this merger reconstructs
    the full model by loading and concatenating the sharded parameters from all ranks.

    The merger supports various FSDP configurations including:
    - Pure FSDP (single dimension sharding)
    - FSDP + DDP (data parallel + fully sharded data parallel)
    - DTensor-based sharding with custom device meshes

    Key features:
    - Automatic detection of world size from checkpoint filenames
    - Support for DTensor and non-DTensor checkpoints
    - Parallel loading of checkpoint shards for efficiency
    - Validation against reference HuggingFace models

    Example:
        To merge FSDP checkpoints:
        ```python
        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir="path/to/fsdp/checkpoints",
            target_dir="path/to/output"
        )
        merger = FSDPModelMerger(config)
        merger.merge_and_save()
        ```
    """

    def _get_world_size(self) -> int:
        # [FUNCTION DOC]
        # HELPER: Reads world_size from local_dir/fsdp_config.json.
        # Outputs: integer world size.
        # Exceptions: FileNotFoundError if the file is missing; ValueError if world_size is absent.
        """_summary_
        From FSDP json config file, extract the world size.

        Returns:
            int: world size
        """
        config_path = Path(self.config.local_dir) / "fsdp_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist.")

        with open(config_path) as f:
            config = json.load(f)

        # Extract world size from the config
        world_size = config.get("world_size", None)
        if world_size is None:
            raise ValueError("World size not found in the config file.")

        return world_size

    def _load_rank_zero_state_dict(self, world_size: int) -> dict:
        # [FUNCTION DOC]
        # HELPER: Loads the rank 0 checkpoint file (full pickle load, on CPU) for the given world size.
        # Outputs: state dict of rank 0. Side effects: reads disk.
        return torch.load(
            Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_0.pt",
            map_location="cpu",
            weights_only=False,
        )

    def _extract_device_mesh_info(self, state_dict: dict, world_size: int) -> tuple[np.ndarray, tuple[str, ...]]:
        # [FUNCTION DOC]
        # HELPER: Inspects the alphabetically first key of the state dict; if it is a DTensor returns its device mesh array
        # and mesh dim names, otherwise returns a 1-D mesh array holding world_size and the names ("fsdp",).
        # KNOWN ISSUE: for non-DTensor checkpoints the returned mesh is an array containing the value world_size (shape [1]),
        # not a mesh of shape [world_size]; downstream uses mesh.shape[-1], giving 1 total shard regardless of world_size.
        # Outputs: tuple (mesh array, mesh dim names).
        """
        Retrieves sharding information (device_mesh, mesh_dim_names) from a DTensor in the state_dict.
        If no DTensor is found, infers a simple FSDP mesh based on world_size.
        """
        pivot_key = sorted(list(state_dict.keys()))[0]
        weight = state_dict[pivot_key]

        if isinstance(weight, DTensor):
            # get sharding info
            device_mesh = weight.device_mesh
            mesh = device_mesh.mesh
            mesh_dim_names = device_mesh.mesh_dim_names
        else:
            # for non-DTensor
            mesh = np.array([world_size], dtype=np.int64)
            mesh_dim_names = ("fsdp",)

        return mesh, mesh_dim_names

    def _calculate_shard_configuration(
        self, mesh: np.ndarray, mesh_dim_names: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        # [FUNCTION DOC]
        # HELPER: Asserts the mesh dim names are ("fsdp",) or ("ddp", "fsdp"). Total shards is the size of the last mesh
        # dim and mesh_shape is a 1-tuple of it (a "tp" branch exists but is unreachable due to the assert).
        # Outputs: tuple (total_shards, mesh_shape).
        # Exceptions: AssertionError for unsupported mesh names.
        """Calculates the total number of shards and the shape of the device mesh."""
        assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}"

        if "tp" in mesh_dim_names:
            # TODO: "tp" is not supported yet due to the above assert
            total_shards = mesh.shape[-1] * mesh.shape[-2]
            mesh_shape = (mesh.shape[-2], mesh.shape[-1])
        else:
            total_shards = mesh.shape[-1]
            mesh_shape = (mesh.shape[-1],)

        return total_shards, mesh_shape

    def _merge_by_placement(self, tensors: list[torch.Tensor], placement: Placement) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Combines per-rank local tensors by placement: replicate -> first tensor; shard -> concatenate along the
        # shard dim (made contiguous).
        # Exceptions: NotImplementedError for partial or unknown placements.
        """Merges a list of tensors based on their DTensor placement"""
        if placement.is_replicate():
            return tensors[0]
        elif placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        elif placement.is_shard():
            return torch.cat(tensors, dim=placement.dim).contiguous()

        raise NotImplementedError(f"Unsupported placement: {placement}")

    def _load_and_merge_state_dicts(
        self, world_size: int, total_shards: int, mesh_shape: tuple[int, ...], mesh_dim_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC]
        # HELPER: Loads total_shards rank files in parallel (thread pool up to 32 workers, tqdm progress) and merges them.
        # Behavior: for each key in rank 0's dict, pops each rank's tensor in order and collects bfloat16 copies (local
        # tensor for DTensors). For DTensors records placements (dropping the first placement when the first mesh dim is
        # "dp"/"ddp") and asserts they agree across ranks. Then for each key: DTensor keys with a 1-D mesh are merged via
        # the placement; a 2-D mesh raises; non-DTensor keys go through the replicated-equality check.
        # Inputs: world_size (file naming), total_shards, mesh_shape, mesh_dim_names.
        # Outputs: mapping name -> full bfloat16 tensor.
        # Exceptions: NotImplementedError for FSDP+TP or unsupported placements; AssertionError on inconsistent placements;
        # ValueError on mismatched replicated entries; KeyError if a rank lacks a key present on rank 0.
        # Side effects: reads checkpoint files; prints and shows progress.
        model_state_dict_lst = [None] * total_shards

        def process_one_shard(rank: int, model_state_dict_lst: list):
            # [FUNCTION DOC]
            # HELPER: Loads one rank's checkpoint file to CPU and stores it at index rank in the shared list.
            # Inputs: rank; model_state_dict_lst - shared list to fill. Outputs: the loaded state dict.
            # Side effects: reads disk; mutates the list.
            model_path = Path(self.config.local_dir) / f"model_world_size_{world_size}_rank_{rank}.pt"
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state_dict_lst[rank] = state_dict
            return state_dict

        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
            futures = [executor.submit(process_one_shard, rank, model_state_dict_lst) for rank in range(total_shards)]
            for future in tqdm(futures, desc=f"Loading {total_shards} FSDP shards", total=total_shards):
                future.result()

        # Merge state dicts from all shards
        state_dict = {}
        param_placements: dict[str, list] = {}

        for key in set(model_state_dict_lst[0].keys()):
            state_dict[key] = []
            for model_state_shard in model_state_dict_lst:
                # add tensor shard in order of rank to state_dict[key]
                tensor = model_state_shard.pop(key)
                if isinstance(tensor, DTensor):
                    state_dict[key].append(tensor._local_tensor.bfloat16())

                    placements = tuple(tensor.placements)
                    # replicated placement at dp dimension can be discarded
                    if mesh_dim_names[0] in ("dp", "ddp"):
                        placements = placements[1:]

                    if key not in param_placements:
                        param_placements[key] = placements
                    else:
                        assert param_placements[key] == placements
                else:
                    state_dict[key].append(tensor.bfloat16())

        del model_state_dict_lst

        # Merge tensors
        for key in sorted(state_dict):
            if not isinstance(state_dict[key], list):
                print(f"No need to merge key {key}")
                continue
            if key in param_placements:
                # merge shards
                placements: tuple[Shard] = param_placements[key]
                if len(mesh_shape) == 1:
                    # 1-D list, FSDP without TP
                    assert len(placements) == 1
                    shards = state_dict[key]
                    state_dict[key] = self._merge_by_placement(shards, placements[0])
                else:
                    # 2-D list, FSDP + TP
                    raise NotImplementedError("FSDP + TP is not supported yet")
            else:
                state_dict[key] = merge_non_dtensor_shards(key, state_dict[key])

        return state_dict

    def merge_and_save(self):
        # [FUNCTION DOC]
        # Behavior: determines world size, loads rank 0 to infer the mesh, computes shard configuration, loads and merges all
        # shards, then for "test" validates against test_hf_dir, or for "merge" saves the HF model/tokenizer and optionally
        # uploads to the Hub.
        # Exceptions: ValueError if test_hf_dir is missing for test or on unknown operation; propagates loader errors.
        # Side effects: reads checkpoints; writes output; prints progress; may upload.
        world_size = self._get_world_size()
        rank_zero_state_dict = self._load_rank_zero_state_dict(world_size)

        mesh, mesh_dim_names = self._extract_device_mesh_info(rank_zero_state_dict, world_size)
        print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

        total_shards, mesh_shape = self._calculate_shard_configuration(mesh, mesh_dim_names)
        print(f"Processing model shards with {total_shards} {mesh_shape} in total")

        merged_state_dict = self._load_and_merge_state_dicts(world_size, total_shards, mesh_shape, mesh_dim_names)

        if self.config.operation == "test":
            if not self.config.test_hf_dir:
                raise ValueError("test_hf_dir must be provided for test operation")
            self._validate_state_dict(merged_state_dict)
        elif self.config.operation == "merge":
            self.save_hf_model_and_tokenizer(merged_state_dict)
            if self.config.hf_upload:
                self.upload_to_huggingface()
        else:
            raise ValueError(f"Unknown operation: {self.config.operation}")

    def _validate_state_dict(self, state_dict: dict[str, torch.Tensor]):
        # [FUNCTION DOC]
        # HELPER: Loads the reference HF model from test_hf_dir in bfloat16 and asserts the merged state dict has exactly the
        # same keys, and per key the same shape, dtype, and values (atol/rtol 1e-6).
        # Exceptions: AssertionError on any mismatch.
        # Side effects: prints success message.
        auto_model_class = self.get_transformers_auto_model_class()

        hf_model = auto_model_class.from_pretrained(self.config.test_hf_dir, torch_dtype=torch.bfloat16)
        hf_state_dict = hf_model.state_dict()
        del hf_model

        hf_model_keys = set(hf_state_dict.keys())
        collected_keys = set(state_dict.keys())

        missing_keys = hf_model_keys - collected_keys
        assert len(missing_keys) == 0, f"Missing keys in collected state dict: {list(sorted(missing_keys))}"

        extra_keys = collected_keys - hf_model_keys
        assert len(extra_keys) == 0, f"Extra keys in collected state dict: {list(sorted(extra_keys))}"

        for key in hf_model_keys:
            hf_shape = hf_state_dict[key].shape
            collected_shape = state_dict[key].shape
            assert hf_shape == collected_shape, (
                f"Shape mismatch for key '{key}': original {hf_shape} vs collected {collected_shape}"
            )

            hf_dtype = hf_state_dict[key].dtype
            collected_dtype = state_dict[key].dtype
            assert hf_dtype == collected_dtype, (
                f"Dtype mismatch for key '{key}': original {hf_dtype} vs collected {collected_dtype}"
            )

            torch.testing.assert_close(hf_state_dict[key], state_dict[key], atol=1e-6, rtol=1e-6)

        print("FSDP checks passed: The merged state_dict matches the hf model saved by FSDPCheckpointManager.")

    def cleanup(self):
        # [FUNCTION DOC]
        # No-op: FSDP merging creates no temporary resources.
        """Cleanup temporary files if needed."""
        # FSDP merger does not create temporary files, so no cleanup is needed.
        pass
