# Copyright (c) 2026 BAAI. All rights reserved.
# Adopted from https://github.com/microsoft/DeepSpeed/blob/master/accelerator/abstract_accelerator.py
# [MODULE DOC]
# Defines PlatformBase, the abstract hardware-accelerator interface that concrete platforms (CUDA, ROCm, NPU, ...)
# implement so that the rest of verl stays device-agnostic.

"""Abstract base class defining the platform interface for device backends.

To add support for a new chip/accelerator, subclass ``PlatformBase`` and
implement all abstract methods.  Then register the platform name in
``platform_manager.py`` so that auto-detection or ``VERL_PLATFORM`` can
pick it up.
"""

import abc
import os
import shutil
import subprocess
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional


class PlatformBase(abc.ABC):
    # [CLASS DOC]
    # Abstract base class. Abstract members: device_name, vendor_name, device_module (properties), is_available,
    # current_device, device_count, set_device, synchronize, manual_seed, manual_seed_all, set_allocator_settings,
    # empty_cache, get_device_capability, communication_backend_name, visible_devices_envvar, nvtx_range, profiler_start,
    # profiler_stop, ray_resource_name, ray_noset_envvars, is_ipc_supported, cudart. Concrete defaults are provided for
    # check_smi_command, is_platform_available, get_device_uuid, apply_model_patches, ray_resource_options,
    # rollout_env_vars and get_collective_module. Holds no state.
    """Hardware-agnostic interface for accelerator backends.

    Every concrete platform (CUDA, NPU, CPU, XPU, …) must implement the
    methods below so that the rest of the verl codebase can remain
    device-agnostic.

    For profiling methods (``nvtx_range``, ``profiler_start``, ``profiler_stop``),
    platforms that do not support profiling should implement them as no-ops.
    """

    # ------------------------------------------------------------------
    # Core device management
    # ------------------------------------------------------------------

    @staticmethod
    def check_smi_command(cmd: str) -> bool:
        # [FUNCTION DOC]
        # Behavior: static. Resolves the command on PATH; if missing, tries /usr/bin, /usr/local/bin and /usr/local/cuda/bin
        # for an executable file of that name. Runs it with no arguments, output discarded, 10 second timeout.
        # Inputs: cmd - executable name (e.g. nvidia-smi).
        # Outputs: True if the process exits with code 0; False if not found, non-zero exit, timeout or OS error.
        # Exceptions: none (timeouts and OS errors swallowed).
        # Side effects: spawns a subprocess.
        """Run an SMI command (e.g. nvidia-smi, mx-smi) and return True if it exits successfully.

        Useful for CUDA-compatible hardware that needs to be distinguished
        from NVIDIA during auto-detection.
        """
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
        try:
            result = subprocess.run([cmd_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    @property
    @abc.abstractmethod
    def device_name(self) -> str:
        # [FUNCTION DOC]
        # Abstract property. Outputs: torch device type string such as cuda / npu.
        """Return the device type string (e.g. ``'cuda'``, ``'npu'``, ``'cpu'``)."""
        ...

    @property
    @abc.abstractmethod
    def vendor_name(self) -> str:
        # [FUNCTION DOC]
        # Abstract property. Outputs: hardware vendor name string such as nvidia / huawei.
        """Return the hardware vendor name (e.g. ``'nvidia'``, ``'metax'``, ``'huawei'``, ``'intel'``)."""
        ...

    @property
    @abc.abstractmethod
    def device_module(self) -> ModuleType:
        # [FUNCTION DOC]
        # Abstract property. Outputs: the torch per-device namespace module (e.g. torch.cuda).
        """Return the ``torch.<device>`` namespace module (e.g. ``torch.cuda``)."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        # [FUNCTION DOC]
        # Abstract. Outputs: True if the accelerator is visible and usable in the current process.
        """Return ``True`` if the accelerator device is visible and usable in this process."""
        ...

    def is_platform_available(self, use_smi_check=False) -> bool:
        # [FUNCTION DOC]
        # Behavior: auto-detection probe; default simply returns is_available(), ignoring use_smi_check.
        # Inputs: use_smi_check - when True subclasses may use relaxed host-level checks (package import, SMI commands) that
        # work in processes without device visibility.
        # Outputs: bool.
        """Return ``True`` if this platform is available on this host.

        Used during auto-detection to determine if the environment targets
        this platform.  When ``use_smi_check=True``, may use relaxed checks
        (e.g. package importability, system commands) that work even in
        processes without device visibility (CPU-only Ray actors).

        Default implementation delegates to ``is_available()``.  Subclasses
        can override to provide more sophisticated detection logic.
        """
        return self.is_available()

    @abc.abstractmethod
    def current_device(self) -> int:
        # [FUNCTION DOC]
        # Abstract. Outputs: index of the currently selected device.
        """Return the index of the currently selected device."""
        ...

    @abc.abstractmethod
    def device_count(self) -> int:
        # [FUNCTION DOC]
        # Abstract. Outputs: number of devices of this type.
        """Return the number of available devices of this type."""
        ...

    @abc.abstractmethod
    def set_device(self, device_index: int) -> None:
        # [FUNCTION DOC]
        # Abstract. Inputs: device_index - device to make current. Side effects: changes the process's current device.
        """Select the device at *device_index*."""
        ...

    @abc.abstractmethod
    def synchronize(self, device_index: Optional[int] = None) -> None:
        # [FUNCTION DOC]
        # Abstract. Inputs: device_index - optional device; None means current. Blocks until pending device work finishes.
        """Block until all pending work on the device completes."""
        ...

    # ------------------------------------------------------------------
    # Random number generator
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def manual_seed(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Abstract. Inputs: seed - integer. Side effects: seeds the current device's RNG.
        """Seed the current device's RNG."""
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed: int) -> None:
        # [FUNCTION DOC]
        # Abstract. Inputs: seed - integer. Side effects: seeds the RNG of all devices.
        """Seed **all** devices' RNG."""
        ...

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def set_allocator_settings(self, settings: str) -> None:
        # [FUNCTION DOC]
        # Abstract. Inputs: settings - allocator config string (e.g. expandable_segments:True). Side effects: configures the
        # device caching allocator.
        """Configure the memory allocator (e.g. expandable segments)."""
        ...

    @abc.abstractmethod
    def empty_cache(self) -> None:
        # [FUNCTION DOC]
        # Abstract. Side effects: releases unused cached device memory.
        """Release all unused cached memory."""
        ...

    # ------------------------------------------------------------------
    # Device properties
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        # [FUNCTION DOC]
        # Abstract. Inputs: device_index - device (default 0). Outputs: (major, minor) compute capability or (None, None).
        """Return ``(major, minor)`` compute capability, or ``(None, None)``."""
        ...

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def communication_backend_name(self) -> str:
        # [FUNCTION DOC]
        # Abstract. Outputs: default torch.distributed backend name (e.g. nccl, hccl).
        """Return the default collective-communication backend name (e.g. ``'nccl'``)."""
        ...

    @abc.abstractmethod
    def visible_devices_envvar(self) -> str:
        # [FUNCTION DOC]
        # Abstract. Outputs: name of the environment variable controlling visible devices (e.g. CUDA_VISIBLE_DEVICES).
        """Return the environment-variable name that controls visible devices."""
        ...

    # ------------------------------------------------------------------
    # Profiling helpers
    # ------------------------------------------------------------------

    @abc.abstractmethod
    @contextmanager
    def nvtx_range(self, msg: str):
        # [FUNCTION DOC]
        # Abstract context manager. Inputs: msg - range label. Behavior: wraps the block in a profiler range; no-op (just yield)
        # on platforms without profiling.
        """Context manager that wraps a block with an NVTX / profiler range.

        Platforms without profiling support should yield immediately (no-op).
        """
        ...

    @abc.abstractmethod
    def profiler_start(self) -> None:
        # [FUNCTION DOC]
        # Abstract. Side effects: starts the device profiler (no-op where unsupported).
        """Start the device profiler (no-op on unsupported platforms)."""
        ...

    @abc.abstractmethod
    def profiler_stop(self) -> None:
        # [FUNCTION DOC]
        # Abstract. Side effects: stops the device profiler (no-op where unsupported).
        """Stop the device profiler (no-op on unsupported platforms)."""
        ...

    # ------------------------------------------------------------------
    # vllm integration
    # ------------------------------------------------------------------
    def get_device_uuid(self, device_id):
        # [FUNCTION DOC]
        # Behavior: builds a device identifier string for vLLM integration. If the visible-devices env var is set, split it on
        # commas and return the Ray resource name concatenated directly with the visible device entry at device_id; otherwise
        # return the Ray resource name, a dash, and device_id.
        # Inputs: device_id - local integer device index.
        # Outputs: string identifier.
        # Exceptions: AssertionError if device_id is out of range of the visible device list.
        # KNOWN ISSUE: the env-var branch joins without a dash while the fallback uses a dash, so formats are inconsistent; the
        # name says UUID but no real device UUID is queried.
        if os.getenv(self.visible_devices_envvar()) is not None:
            visible_devices = os.environ[self.visible_devices_envvar()].split(",")
            assert device_id < len(visible_devices), f"device_id {device_id} must less than {visible_devices}"
            return self.ray_resource_name() + visible_devices[device_id]
        else:
            return f"{self.ray_resource_name()}-{device_id}"

    def apply_model_patches(self, model_type: str) -> None:
        # [FUNCTION DOC]
        # Behavior: default no-op hook for platform-specific model patches. Inputs: model_type - HF model type string.
        """Apply platform-specific model patches (e.g. replace ops unsupported on this device)."""
        return  # default no-op

    # ------------------------------------------------------------------
    # Ray integration
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def ray_resource_name(self) -> str:
        # [FUNCTION DOC]
        # Abstract. Outputs: Ray accelerator resource name (e.g. GPU, NPU).
        """Return the Ray accelerator resource name (e.g. ``'GPU'``, ``'NPU'``)."""
        ...

    @abc.abstractmethod
    def ray_noset_envvars(self) -> list[str]:
        # [FUNCTION DOC]
        # Abstract. Outputs: list of RAY_EXPERIMENTAL_NOSET_* env var names relevant to this platform.
        """Return ``RAY_EXPERIMENTAL_NOSET_*`` env var names for this platform."""
        ...

    def ray_resource_options(self, num_gpus: float) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Behavior: returns Ray actor options for requesting accelerators: num_gpus key when resource name is GPU, otherwise a
        # custom resources mapping from resource name to the amount.
        # Inputs: num_gpus - (fractional) number of accelerators. Outputs: dict of Ray actor options.
        """Return Ray actor resource options for allocating accelerators.

        CUDA uses ``{"num_gpus": N}`` while custom resources like NPU use
        ``{"resources": {"NPU": N}}``.  Subclasses may override for
        platform-specific behavior.
        """
        resource_name = self.ray_resource_name()
        if resource_name == "GPU":
            return {"num_gpus": num_gpus}
        return {"resources": {resource_name: num_gpus}}

    # ------------------------------------------------------------------
    # IPC support
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def is_ipc_supported(self) -> bool:
        # [FUNCTION DOC]
        # Abstract. Outputs: True if CUDA-style IPC tensor sharing between processes is supported.
        """Return ``True`` if the platform supports IPC for tensor sharing."""
        ...

    # ------------------------------------------------------------------
    # Rollout engine integration
    # ------------------------------------------------------------------

    def rollout_env_vars(self) -> dict[str, str]:
        # [FUNCTION DOC]
        # Behavior: default returns an empty mapping. Outputs: env vars to inject when launching rollout engines.
        """Return platform-specific env vars to inject when launching rollout engines."""
        return {}

    # ------------------------------------------------------------------
    # Collective communication
    # ------------------------------------------------------------------

    def get_collective_module(self) -> Any:
        # [FUNCTION DOC]
        # Behavior: default returns None. Outputs: collective communication module (e.g. cupy nccl) or None.
        """Return the collective communication module (e.g. ``cupy.cuda.nccl``).

        Returns ``None`` if not available. Subclasses should override.
        """
        return None

    # ------------------------------------------------------------------
    # Low-level runtime API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def cudart(self) -> Any:
        # [FUNCTION DOC]
        # Abstract. Outputs: CUDA runtime API object, or None if not applicable.
        """Return the CUDA runtime API object, or ``None`` if not applicable."""
        ...
