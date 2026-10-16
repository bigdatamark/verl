# Copyright (c) 2026 BAAI. All rights reserved.
# [MODULE DOC]
# Huawei Ascend NPU implementation of PlatformBase, registered under the name huawei. At import time it attempts to
# import torch_npu (which attaches the npu namespace to torch) so later availability checks are cheap.

"""Huawei Ascend NPU platform implementation."""

import logging
import os
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from .platform_base import PlatformBase
from .platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


def _ensure_torch_npu() -> bool:
    # [FUNCTION DOC]
    # HELPER: Behavior: return True immediately if torch already has an npu attribute; otherwise try importing torch_npu and
    # return whether torch now has npu. Any import exception is logged at debug and yields False.
    # Outputs: bool. Side effects: may import torch_npu.
    """Try to import torch_npu so that torch.npu becomes available.

    Returns True if torch.npu is usable after the attempt.
    """
    if hasattr(torch, "npu"):
        return True
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu")
    except Exception as e:
        logger.debug("The current machine has no torch.npu, because: %s", e)
    return False


_ensure_torch_npu()  # Attempt to import torch_npu at module load time so that availability checks are faster later


@PlatformRegistry.register(platform="huawei")
class PlatformNPU(PlatformBase):
    # [CLASS DOC]
    # Concrete platform for Ascend NPUs; delegates to the torch.npu namespace. Profiling methods are no-ops; cudart is None.
    """Platform backend for Huawei Ascend NPU."""

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @property
    def device_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: npu.
        return "npu"

    @property
    def vendor_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: huawei.
        return "huawei"

    @property
    def device_module(self) -> ModuleType:
        # [FUNCTION DOC]
        # Outputs: the torch.npu module (AttributeError if torch_npu is not installed).
        return torch.npu

    def is_available(self) -> bool:
        # [FUNCTION DOC]
        # Outputs: torch.npu.is_available().
        return torch.npu.is_available()

    def is_platform_available(self, use_smi_check=False) -> bool:
        # [FUNCTION DOC]
        # Behavior: False if torch_npu cannot be loaded; True if use_smi_check (importability alone suffices); else
        # torch.npu.is_available().
        """Return True if this platform is available on this host.

        Used during auto-detection to determine if the environment targets
        this platform.  When ``use_smi_check=True``, only requires that
        torch_npu is importable (even if no devices are visible).
        """
        if not _ensure_torch_npu():
            return False
        if use_smi_check:
            # torch_npu imported successfully — NPU environment confirmed
            return True
        return torch.npu.is_available()

    def current_device(self) -> int:
        # [FUNCTION DOC]
        # Outputs: current NPU index.
        return torch.npu.current_device()

    def device_count(self) -> int:
        # [FUNCTION DOC]
        # Outputs: NPU count.
        return torch.npu.device_count()

    def set_device(self, device_index: int) -> None:
        # [FUNCTION DOC]
        # Side effects: sets current NPU.
        torch.npu.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        # [FUNCTION DOC]
        # Side effects: synchronizes the NPU (optional index).
        torch.npu.synchronize(device_index)

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    def manual_seed(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Side effects: seeds current NPU RNG.
        torch.npu.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Side effects: seeds all NPU RNGs.
        torch.npu.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def set_allocator_settings(self, settings: str) -> None:
        # [FUNCTION DOC]
        # Behavior: applies allocator settings via torch_npu's private setter; on any exception logs a warning asking to upgrade
        # torch-npu to 2.9.0+ instead of raising.
        try:
            torch.npu.memory._set_allocator_settings(settings)
        except Exception:
            logger.warning(
                "Current version of torch-npu does not support `_set_allocator_settings`, "
                "please upgrade torch-npu to 2.9.0 or later"
            )

    def empty_cache(self) -> None:
        # [FUNCTION DOC]
        # Side effects: empties the NPU cache.
        torch.npu.empty_cache()

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # [FUNCTION DOC]
        # Outputs: torch.npu capability if that function exists and returns non-None, else (None, None).
        if hasattr(torch.npu, "get_device_capability"):
            result = torch.npu.get_device_capability(device_index)
            # torch.npu.get_device_capability may return None instead of a tuple
            if result is None:
                return (None, None)
            return result
        return (None, None)

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def communication_backend_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: flagcx if USE_FLAGCX is 1/true, else hccl.
        return "flagcx" if os.getenv("USE_FLAGCX", "0").lower() in ["1", "true"] else "hccl"

    def visible_devices_envvar(self) -> str:
        # [FUNCTION DOC]
        # Outputs: ASCEND_RT_VISIBLE_DEVICES.
        return "ASCEND_RT_VISIBLE_DEVICES"

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    def ray_resource_name(self) -> str:
        # [FUNCTION DOC]
        # Outputs: NPU.
        return "NPU"

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Outputs: custom resources mapping with NPU set to num_gpus.
        return {"resources": {"NPU": num_gpus}}

    def ray_noset_envvars(self) -> list[str]:
        # [FUNCTION DOC]
        # Outputs: list with RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES.
        return ["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"]

    def rollout_env_vars(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Outputs: mapping with NCCL_CUMEM_ENABLE=0 and VLLM_ASCEND_AUTO_DETECT_QUANTIZATION=0; if VLLM_ASCEND_TASK_QUEUE_ENABLE
        # is set (non-empty), also TASK_QUEUE_ENABLE copied from it (lets rollout use a different task queue mode than training).
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        env_vars = {"NCCL_CUMEM_ENABLE": "0", "VLLM_ASCEND_AUTO_DETECT_QUANTIZATION": "0"}
        if os.environ.get("VLLM_ASCEND_TASK_QUEUE_ENABLE", None):
            # use VLLM_ASCEND_TASK_QUEUE_ENABLE to support different TASK_QUEUE_ENABLE mode for
            # train and rollout on Ascend NPU
            env_vars["TASK_QUEUE_ENABLE"] = os.environ["VLLM_ASCEND_TASK_QUEUE_ENABLE"]
        return env_vars

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    def is_ipc_supported(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: queries hardware, software and CANN versions via verl.utils.device.get_npu_versions (runs npu-smi). Returns
        # True for A5 hardware, otherwise defers to check_ipc_version_support on the software and CANN versions.
        # Outputs: bool.
        # Exceptions: RuntimeError wrapping a failed npu-smi call or any other error from the version query.
        # Side effects: spawns npu-smi via the helper.
        import subprocess

        from verl.utils.device import AscendHardwareVersion, check_ipc_version_support, get_npu_versions

        try:
            hardware_version, software_version, cann_version = get_npu_versions()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to execute npu-smi command: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error checking IPC support: {e}") from e

        if hardware_version == AscendHardwareVersion.A5:
            return True
        return check_ipc_version_support(software_version, cann_version)

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @contextmanager
    def nvtx_range(self, msg: str):
        # [FUNCTION DOC]
        # Context manager: no-op on NPU; logs msg at debug, then yields.
        # NPU does not have an NVTX equivalent, but we log for debugging
        logger.debug("NVTX range (no-op on NPU): %s", msg)
        yield

    def profiler_start(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        pass

    def profiler_stop(self) -> None:
        # [FUNCTION DOC]
        # No-op.
        pass

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    def cudart(self) -> Any:
        # [FUNCTION DOC]
        # Outputs: None (no CUDA runtime on NPU).
        return None
