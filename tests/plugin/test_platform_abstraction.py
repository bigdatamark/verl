# Copyright (c) 2026 BAAI. All rights reserved.
# [MODULE DOC]
# CPU tests for verl.plugin.platform: VERL_PLATFORM-based detection, platform creation, the process-wide platform
# singleton with external injection, and PlatformRegistry registration/override. Tests reset the private
# platform_manager._current_platform to isolate singleton state.

"""Unit tests for the platform abstraction layer."""

import os
from contextlib import contextmanager
from unittest import mock

import pytest

from verl.plugin.platform import get_platform, set_platform
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import (
    PlatformRegistry,
    _create_platform,
    _detect_platform_name,
)


def _make_mock_platform(name="mock_xpu"):
    # [FUNCTION DOC]
    # HELPER: Returns a new concrete PlatformBase subclass (class name Mock_NAME) that implements every abstract member
    # with trivial values: device_name NAME, vendor "mock_NAME", torch as device module, always available, one device at
    # index 0, no-op seed/sync/cache/profiler/allocator calls, capability (None, None), backend "mock_ccl", visible
    # devices env var MOCK_VISIBLE_DEVICES, Ray resource "MOCK", no IPC, a no-op nvtx_range context, and no cudart.
    # Inputs: name - device name (default "mock_xpu").
    # Outputs: the class (not an instance).
    """Return a minimal concrete PlatformBase subclass for testing."""

    class _Mock(PlatformBase):
        # [CLASS DOC]
        # HELPER: The generated mock platform class; each member returns the constant described above.
        @property
        def device_name(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of device_name returning the fixed value described in the factory (or doing nothing).
            return name

        @property
        def vendor_name(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of vendor_name returning the fixed value described in the factory (or doing nothing).
            return f"mock_{name}"

        @property
        def device_module(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of device_module returning the fixed value described in the factory (or doing nothing).
            import torch

            return torch

        def is_available(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of is_available returning the fixed value described in the factory (or doing nothing).
            return True

        def is_platform_available(self, use_smi_check=False):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of is_platform_available returning the fixed value described in the factory (or doing nothing).
            return True

        def current_device(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of current_device returning the fixed value described in the factory (or doing nothing).
            return 0

        def device_count(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of device_count returning the fixed value described in the factory (or doing nothing).
            return 1

        def set_device(self, device_index):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of set_device returning the fixed value described in the factory (or doing nothing).
            pass

        def synchronize(self, device_index=None):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of synchronize returning the fixed value described in the factory (or doing nothing).
            pass

        def manual_seed(self, seed):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of manual_seed returning the fixed value described in the factory (or doing nothing).
            pass

        def manual_seed_all(self, seed):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of manual_seed_all returning the fixed value described in the factory (or doing nothing).
            pass

        def set_allocator_settings(self, settings):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of set_allocator_settings returning the fixed value described in the factory (or doing nothing).
            pass

        def empty_cache(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of empty_cache returning the fixed value described in the factory (or doing nothing).
            pass

        def get_device_capability(self, device_index=0):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of get_device_capability returning the fixed value described in the factory (or doing nothing).
            return (None, None)

        def communication_backend_name(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of communication_backend_name returning the fixed value described in the factory (or doing nothing).
            return "mock_ccl"

        def visible_devices_envvar(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of visible_devices_envvar returning the fixed value described in the factory (or doing nothing).
            return "MOCK_VISIBLE_DEVICES"

        def ray_resource_name(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of ray_resource_name returning the fixed value described in the factory (or doing nothing).
            return "MOCK"

        def ray_noset_envvars(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of ray_noset_envvars returning the fixed value described in the factory (or doing nothing).
            return ["RAY_EXPERIMENTAL_NOSET_MOCK_VISIBLE_DEVICES"]

        def is_ipc_supported(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of is_ipc_supported returning the fixed value described in the factory (or doing nothing).
            return False

        @contextmanager
        def nvtx_range(self, msg):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of nvtx_range returning the fixed value described in the factory (or doing nothing).
            yield

        def profiler_start(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of profiler_start returning the fixed value described in the factory (or doing nothing).
            pass

        def profiler_stop(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of profiler_stop returning the fixed value described in the factory (or doing nothing).
            pass

        def cudart(self):
            # [FUNCTION DOC]
            # HELPER: Mock implementation of cudart returning the fixed value described in the factory (or doing nothing).
            return None

    _Mock.__name__ = f"Mock_{name}"
    return _Mock


class TestPlatformDetection:
    # [CLASS DOC]
    # Tests _detect_platform_name with VERL_PLATFORM patched into the environment.
    """Test platform auto-detection logic."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Clears the cached platform singleton before each test.
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_env_override_nvidia(self):
        # [FUNCTION DOC]
        # Asserts VERL_PLATFORM=nvidia yields "nvidia".
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "nvidia"}):
            assert _detect_platform_name() == "nvidia"

    def test_env_override_huawei(self):
        # [FUNCTION DOC]
        # Asserts VERL_PLATFORM=huawei yields "huawei".
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "huawei"}):
            assert _detect_platform_name() == "huawei"

    def test_env_override_amd(self):
        # [FUNCTION DOC]
        # Asserts VERL_PLATFORM=amd yields "amd".
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "amd"}):
            assert _detect_platform_name() == "amd"

    def test_invalid_value_passes_through(self):
        # [FUNCTION DOC]
        # Asserts an unknown explicit value is returned unchanged (validation is deferred to creation).
        # When an explicit platform name is set, _detect_platform_name returns
        # it as-is (validation happens later in _create_platform).
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "invalid"}):
            assert _detect_platform_name() == "invalid"

    def test_case_insensitive(self):
        # [FUNCTION DOC]
        # Asserts the env value is lower-cased ("NVIDIA" -> "nvidia").
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": "NVIDIA"}):
            assert _detect_platform_name() == "nvidia"

    def test_empty_triggers_auto_detection(self):
        # [FUNCTION DOC]
        # Asserts an empty value triggers auto-detection returning one of nvidia, huawei, amd.
        with mock.patch.dict(os.environ, {"VERL_PLATFORM": ""}):
            assert _detect_platform_name() in ("nvidia", "huawei", "amd")


class TestPlatformCreation:
    # [CLASS DOC]
    # Tests _create_platform.
    """Test platform creation."""

    def test_cuda_warns_if_unavailable(self):
        # [FUNCTION DOC]
        # Asserts creating "nvidia" with CUDA reported unavailable logs exactly one warning and still returns a platform.
        with mock.patch("torch.cuda.is_available", return_value=False):
            with mock.patch("verl.plugin.platform.platform_manager.logger") as mock_logger:
                platform = _create_platform("nvidia")
                mock_logger.warning.assert_called_once()
                assert platform is not None

    def test_invalid_platform_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError for an unregistered name.
        with pytest.raises(ValueError):
            _create_platform("invalid_platform")


class TestPlatformSingleton:
    # [CLASS DOC]
    # Tests get_platform caching and set_platform injection.
    """Test singleton and external injection."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Clears the cached platform singleton before each test.
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_get_platform_returns_singleton(self):
        # [FUNCTION DOC]
        # Asserts two get_platform calls return the same object.
        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_set_platform_external_injection(self):
        # [FUNCTION DOC]
        # Asserts a platform injected via set_platform is returned by get_platform with its device name.
        # Side effects: leaves the mock installed as the global platform until the next reset.
        """Simulates an entry_points plugin calling set_platform()."""
        custom = _make_mock_platform("mock_xpu")()
        set_platform(custom)
        assert get_platform() is custom
        assert get_platform().device_name == "mock_xpu"


class TestPlatformRegistry:
    # [CLASS DOC]
    # Tests PlatformRegistry.
    """Test PlatformRegistry dynamic registration."""

    def setup_method(self):
        # [FUNCTION DOC]
        # HELPER: Clears the cached platform singleton before each test.
        import verl.plugin.platform.platform_manager as pm

        pm._current_platform = None

    def test_builtin_platforms_registered(self):
        # [FUNCTION DOC]
        # Asserts nvidia, huawei and amd are registered by default.
        names = PlatformRegistry.registered_names()
        assert "nvidia" in names
        assert "huawei" in names
        assert "amd" in names

    def test_register_custom_platform(self):
        # [FUNCTION DOC]
        # Registers a mock class under "mock_test" via the decorator form, asserts it is listed, retrievable and creatable,
        # then deletes it from the private registry mapping.
        # KNOWN ISSUE: cleanup is not in a finally block, so a failed assertion leaks the registration into later tests.
        """External plugin registers a new platform via @PlatformRegistry.register()."""
        MockCls = _make_mock_platform("mock_test")
        PlatformRegistry.register(platform="mock_test")(MockCls)

        assert "mock_test" in PlatformRegistry.registered_names()
        assert PlatformRegistry.get("mock_test") is MockCls
        assert _create_platform("mock_test").device_name == "mock_test"

        del PlatformRegistry._platforms["mock_test"]

    def test_register_override(self):
        # [FUNCTION DOC]
        # Asserts re-registering "nvidia" replaces the class (last writer wins), then restores the original entry.
        # KNOWN ISSUE: restoration is not in a finally block; a failure leaves the fake nvidia platform registered.
        """Re-registering the same name overrides the previous class (last writer wins)."""
        original = PlatformRegistry.get("nvidia")
        FakeCls = _make_mock_platform("nvidia")
        PlatformRegistry.register(platform="nvidia")(FakeCls)
        assert PlatformRegistry.get("nvidia") is FakeCls
        PlatformRegistry._platforms["nvidia"] = original

    def test_unregistered_platform_raises(self):
        # [FUNCTION DOC]
        # Asserts ValueError when creating an unregistered platform.
        with pytest.raises(ValueError):
            _create_platform("nonexistent_platform")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
