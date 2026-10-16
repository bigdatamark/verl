# Copyright (c) 2026 BAAI. All rights reserved.
# [MODULE DOC]
# Platform registry and process-wide platform singleton. Provides name-to-class registration (decorator), environment
# auto-detection, and get/set accessors. Module state: a module-level current-platform variable (initially unset) and a
# logger whose level comes from VERL_LOGGING_LEVEL (default INFO). Built-in CUDA, NPU and ROCm platforms are imported at
# the bottom of the module so their registration decorators run at import time.

"""Singleton platform manager with registry and auto-detection.

The platform is resolved **once** on first call to :func:`get_platform` and
cached for the rest of the process lifetime.

New hardware backends are added via :meth:`PlatformRegistry.register`::

    @PlatformRegistry.register(platform="metax")
    class PlatformMetaX(PlatformBase):
        ...

External plugins loaded by ``VERL_USE_EXTERNAL_MODULES`` or discovered via
setuptools entry_points can register their own platform classes without
modifying the verl source tree.
"""

import logging
import os

from .platform_base import PlatformBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_current_platform: PlatformBase | None = None


class PlatformRegistry:
    # [CLASS DOC]
    # Class-level registry (shared mapping from lowercase platform name to PlatformBase subclass); all methods are
    # classmethods, no instances needed. Registration order determines auto-detection probe order.
    """Registry that maps platform names to concrete ``PlatformBase`` subclasses.

    Built-in platforms (``cuda``, ``npu``) are registered at import time.
    External plugins can register additional platforms via the
    :meth:`register` decorator.

    Each platform name (e.g. ``cuda``, ``npu``, ``metax``, ``xpu``, ``mlu``)
    is already vendor-specific, so no separate vendor key is needed.
    """

    _platforms: dict[str, type[PlatformBase]] = {}

    @classmethod
    def register(cls, platform: str):
        # [FUNCTION DOC]
        # Behavior: returns a class decorator that registers the decorated class under the normalized (stripped, lowercased)
        # name, logging an info message when overriding an existing entry, and returns the class unchanged.
        # Inputs: platform - registration name.
        # Outputs: decorator callable.
        # Exceptions: decorator raises AssertionError if the class is not a PlatformBase subclass.
        # Side effects: mutates the class-level registry.
        """Class decorator that registers a ``PlatformBase`` subclass.

        Usage::

            @PlatformRegistry.register(platform="nvidia")
            class PlatformCUDA(PlatformBase):
                ...

            @PlatformRegistry.register(platform="metax")
            class PlatformMetaX(PlatformBase):
                ...
        """

        def decorator(platform_cls: type[PlatformBase]) -> type[PlatformBase]:
            # [FUNCTION DOC]
            # HELPER: the decorator returned by register; see register for behavior.
            assert issubclass(platform_cls, PlatformBase), f"{platform_cls.__name__} must be a subclass of PlatformBase"
            name = platform.strip().lower()
            if name in cls._platforms:
                logger.info(
                    "PlatformRegistry: overriding %s (%s -> %s)",
                    name,
                    cls._platforms[name].__name__,
                    platform_cls.__name__,
                )
            cls._platforms[name] = platform_cls
            return platform_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[PlatformBase] | None:
        # [FUNCTION DOC]
        # Inputs: name - platform name (normalized by strip+lowercase). Outputs: registered class or None.
        """Look up a registered platform class by name."""
        return cls._platforms.get(name.strip().lower())

    @classmethod
    def registered_names(cls) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # Outputs: tuple of registered names in registration order.
        """Return a tuple of all registered platform names."""
        return tuple(cls._platforms.keys())


def _detect_platform_name() -> str:
    # [FUNCTION DOC]
    # HELPER: Behavior: if VERL_PLATFORM is set (non-empty after strip/lowercase) return it unchecked. Otherwise iterate over
    # registered names, instantiate each class and return the first whose is_platform_available(use_smi_check=True) is True;
    # exceptions during a probe are logged at debug and skipped. If none match, log a warning and return nvidia.
    # Outputs: platform name string. Side effects: logging; instantiates platform classes.
    """Probe the environment and return the best platform name.

    Detection order:
    1. ``VERL_PLATFORM`` environment variable (explicit override)
    2. Probe registered platforms via ``is_platform_available(use_smi_check=True)``
    3. Fall back to ``'nvidia'``
    """

    env_platform = os.environ.get("VERL_PLATFORM", "").strip().lower()
    if env_platform:
        logger.info("Platform override from VERL_PLATFORM: %s", env_platform)
        return env_platform

    names = PlatformRegistry.registered_names()
    logger.debug("Registered platforms: %s", names)

    for name in names:
        platform_cls = PlatformRegistry.get(name)
        if platform_cls is None:
            continue
        try:
            instance = platform_cls()
            if instance.is_platform_available(use_smi_check=True):
                logger.debug("Auto-detected platform: %s", name)
                return name
        except Exception as e:
            logger.debug("Platform '%s' detection failed: %s", name, e)
            continue

    logger.warning(
        "No supported accelerator detected. Registered: %s. Falling back to 'nvidia'.",
        names,
    )
    return "nvidia"


def _create_platform(name: str) -> PlatformBase:
    # [FUNCTION DOC]
    # HELPER: Behavior: look up class by name and instantiate it; if the instance reports not available, log a warning
    # (may be a CPU-only Ray actor) but still return it.
    # Inputs: name - platform name. Outputs: platform instance.
    # Exceptions: ValueError if the name is not registered (e.g. a bad VERL_PLATFORM value).
    """Instantiate the concrete platform for *name*."""
    platform_cls = PlatformRegistry.get(name)
    if platform_cls is None:
        raise ValueError(
            f"Unknown platform '{name}'. "
            f"Registered: {PlatformRegistry.registered_names()}. "
            "Use @PlatformRegistry.register() to add a new platform."
        )
    platform = platform_cls()
    if not platform.is_available():
        logger.warning(
            "Platform '%s' (%s) is registered but not available. "
            "This may be due to this ray actor being a CPU-only actor.",
            name,
            platform_cls.__name__,
        )
    return platform


def get_platform() -> PlatformBase:
    # [FUNCTION DOC]
    # Behavior: lazily detects and creates the platform singleton on first call, caches it in module state, returns it.
    # Outputs: PlatformBase instance. Exceptions: propagates ValueError from creation. Side effects: sets module global.
    """Return the current platform singleton (auto-detected on first call)."""
    global _current_platform
    if _current_platform is None:
        name = _detect_platform_name()
        _current_platform = _create_platform(name)
        logger.debug("verl platform initialised: %s", _current_platform.device_name)
    return _current_platform


def set_platform(platform: PlatformBase) -> None:
    # [FUNCTION DOC]
    # Behavior: replaces the module singleton with the given instance, warning if one was already set.
    # Inputs: platform - PlatformBase instance. Side effects: mutates module global; logs.
    """Override the platform singleton with an already-instantiated platform.

    Must be called **before** any code calls :func:`get_platform`,
    otherwise a warning is emitted and the platform is replaced.
    """
    global _current_platform
    if _current_platform is not None:
        logger.warning(
            "Replacing already-initialised platform '%s' with '%s'",
            _current_platform.device_name,
            platform.device_name,
        )
    _current_platform = platform


# ---------------------------------------------------------------------------
# Register built-in platforms.  Imported here so that the @register decorator
# fires at module load time.  The imports are at the bottom to avoid circular
# references (platform_cuda/npu import PlatformBase from platform_base, not
# from this module).
# ---------------------------------------------------------------------------
from .platform_cuda import PlatformCUDA  # noqa: E402, F401
from .platform_npu import PlatformNPU  # noqa: E402, F401
from .platform_rocm import PlatformROCm  # noqa: E402, F401
