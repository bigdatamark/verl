# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/cuda_accelerator.py
# [MODULE DOC]
# NVIDIA CUDA implementation of PlatformBase, registered in the platform registry under the name nvidia.

"""NVIDIA CUDA platform implementation."""

import os
import shutil
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch
import torch.cuda

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="nvidia")
class PlatformCUDA(PlatformBase):
    # [CLASS DOC]
    # Concrete platform for NVIDIA CUDA GPUs. Almost every method delegates directly to the torch.cuda namespace. Stateless.
    """Platform backend for NVIDIA CUDA GPUs and CUDA-compatible accelerators."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: cuda.
        return "cuda"

    @property
    def vendor_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: nvidia.
        return "nvidia"

    @property
    def device_module(self) -> ModuleType:
        # [FUNCTION DOC]
        # Outputs: the torch.cuda module.
        return torch.cuda

    def is_available(self) -> bool:
        # [FUNCTION DOC]
        # Outputs: torch.cuda.is_available().
        return torch.cuda.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        # [FUNCTION DOC]
        # Behavior: returns False if torch has no cuda attribute or if torch is a ROCm (HIP) build (ROCm is handled by the ROCm
        # platform). If use_smi_check, locate nvidia-smi on PATH or common absolute paths; if not found return False; if running
        # it succeeds return True. Otherwise (or if the SMI run fails) fall back to torch.cuda.is_available().
        # Inputs: use_smi_check - enable relaxed host-level check for CPU-only Ray actors.
        # Outputs: bool. Side effects: may spawn nvidia-smi.
        # KNOWN ISSUE: the inline comment says that if nvidia-smi is not found, a built torch.cuda is treated as sufficient, but
        # the code returns False in that case (not even falling back to torch.cuda.is_available()).
        if not hasattr(torch, "cuda"):
            return False
        # On ROCm, torch.cuda is present too; defer to PlatformROCm so that
        # auto-detection does not pick CUDA on AMD hardware.
        if torch.version.hip is not None:
            return False
        if use_smi_check:
            # In CPU-only Ray actors, torch.cuda.is_available() may return False
            # even though the cluster has GPUs. Fall back to nvidia-smi check,
            # and if that's also unavailable (e.g. not on PATH), treat
            # torch.cuda being built as sufficient evidence.
            cmd = "nvidia-smi"
            cmd_path = shutil.which(cmd)
            if cmd_path is None:
                # Fallback to common absolute paths if not found in PATH
                common_paths = [
                    f"/usr/bin/{cmd}",
                    f"/usr/local/bin/{cmd}",
                    f"/usr/local/cuda/bin/{cmd}",
                ]
                for path in common_paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        cmd_path = path
                        break
                if cmd_path is None:
                    return False
            if self.check_smi_command(cmd_path):
                return True
        return torch.cuda.is_available()

    def current_device(self) -> int:
        # [FUNCTION DOC]
        # Outputs: current CUDA device index.
        return torch.cuda.current_device()

    def device_count(self) -> int:
        # [FUNCTION DOC]
        # Outputs: number of visible CUDA devices.
        return torch.cuda.device_count()

    def set_device(self, device_index: int) -> None:
        # [FUNCTION DOC]
        # Inputs: device_index. Side effects: sets current CUDA device.
        torch.cuda.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        # [FUNCTION DOC]
        # Inputs: device_index - optional. Side effects: synchronizes the CUDA device.
        torch.cuda.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Inputs: seed. Side effects: seeds current CUDA device RNG.
        torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Inputs: seed. Side effects: seeds all CUDA device RNGs.
        torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # [FUNCTION DOC]
        # Inputs: settings - allocator config string. Side effects: calls the private torch CUDA allocator settings setter.
        torch.cuda.memory._set_allocator_settings(settings)

    def empty_cache(self) -> None:
        # [FUNCTION DOC]
        # Side effects: empties the CUDA caching allocator.
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # [FUNCTION DOC]
        # Inputs: device_index (default 0). Outputs: (major, minor) capability, or (None, None) if CUDA is unavailable.
        if not torch.cuda.is_available():
            return None, None
        return torch.cuda.get_device_capability(device_index)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: flagcx if env USE_FLAGCX is 1/true (case-insensitive), else nccl.
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "nccl"

    def visible_devices_envvar(self) -> str:
        # [FUNCTION DOC]
        # Outputs: CUDA_VISIBLE_DEVICES.
        return "CUDA_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: GPU.
        return "GPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Inputs: num_gpus. Outputs: mapping with num_gpus key set to the value.
        return {"num_gpus": num_gpus}

    def ray_noset_envvars(self) -> list[str]:
        # [FUNCTION DOC]
        # Outputs: list containing RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES.
        return ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # [FUNCTION DOC]
        # Outputs: True.
        return True

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Outputs: mapping NCCL_CUMEM_ENABLE to the current env value, defaulting to 0 (avoids hangs in weight sync between
        # actor and rollout in disaggregated mode).
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        return {"NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0")}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        # [FUNCTION DOC]
        # Outputs: the cupy nccl module if importable, else None.
        try:
            from cupy.cuda import nccl

            return nccl
        except (ImportError, ModuleNotFoundError):
            return None

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # [FUNCTION DOC]
        # Context manager: wraps the block in a torch CUDA NVTX range labelled msg.
        with torch.cuda.nvtx.range(msg):
            yield

    def profiler_start(self) -> None:
        # [FUNCTION DOC]
        # Side effects: starts the CUDA profiler.
        torch.cuda.profiler.start()

    def profiler_stop(self) -> None:
        # [FUNCTION DOC]
        # Side effects: stops the CUDA profiler.
        torch.cuda.profiler.stop()

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        # [FUNCTION DOC]
        # Outputs: torch CUDA runtime (cudart) object.
        return torch.cuda.cudart()
