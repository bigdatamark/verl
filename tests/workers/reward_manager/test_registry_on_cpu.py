# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# CPU tests for the reward manager registry: lookup by name, case sensitivity, empty registry errors and
# the register decorator (registration, conflict detection, returning the original class).

import pytest

# Assuming REWARD_MANAGER_REGISTRY is defined somewhere in the module
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY, get_reward_manager_cls, register


@pytest.fixture
def setup():
    # [FUNCTION DOC]
    # HELPER: fixture that clears the global REWARD_MANAGER_REGISTRY and fills it with two name-to-string
    # entries, returning the registry. Side effects: mutates global registry.
    # KNOWN ISSUE: the registry is never restored after the test, so real reward manager registrations are lost
    # for any later tests in the same process.
    """Setup test cases with a mock registry."""
    REWARD_MANAGER_REGISTRY.clear()
    REWARD_MANAGER_REGISTRY.update({"manager1": "Manager1Class", "manager2": "Manager2Class"})
    return REWARD_MANAGER_REGISTRY


def test_get_existing_manager(setup):
    # [FUNCTION DOC]
    # Asserts get_reward_manager_cls returns the registered values for both names.
    """Test getting an existing reward manager class."""
    assert get_reward_manager_cls("manager1") == "Manager1Class"
    assert get_reward_manager_cls("manager2") == "Manager2Class"


def test_get_nonexistent_manager(setup):
    # [FUNCTION DOC]
    # Asserts an unknown name raises ValueError mentioning "Unknown reward manager: <name>".
    """Test getting a non-existent reward manager raises ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("unknown_manager")
    assert "Unknown reward manager: unknown_manager" in str(excinfo.value)


def test_case_sensitivity(setup):
    # [FUNCTION DOC]
    # Asserts differently-cased names raise ValueError.
    """Test that manager names are case-sensitive."""
    with pytest.raises(ValueError):
        get_reward_manager_cls("MANAGER1")
    with pytest.raises(ValueError):
        get_reward_manager_cls("Manager1")


def test_empty_registry(setup):
    # [FUNCTION DOC]
    # Clears the registry and asserts lookup raises ValueError with the unknown-manager message.
    """Test behavior when registry is empty."""
    REWARD_MANAGER_REGISTRY.clear()
    with pytest.raises(ValueError) as excinfo:
        get_reward_manager_cls("any_manager")
    assert "Unknown reward manager: any_manager" in str(excinfo.value)


def test_register_new_class(setup):
    # [FUNCTION DOC]
    # Registers a class via decorator and asserts it is stored under the name.
    """Test registering a new class with the decorator."""

    @register("test_manager")
    class TestManager:
        # [CLASS DOC]
        # HELPER: empty class used as a registration target.
        pass

    assert "test_manager" in REWARD_MANAGER_REGISTRY
    assert REWARD_MANAGER_REGISTRY["test_manager"] == TestManager


def test_register_different_classes_same_name(setup):
    # [FUNCTION DOC]
    # Asserts registering a different class under an already-used name raises ValueError and the original stays.
    """Test that registering different classes with same name raises ValueError."""

    @register("conflict_manager")
    class Manager1:
        # [CLASS DOC]
        # HELPER: empty class registered first.
        pass

    with pytest.raises(ValueError):

        @register("conflict_manager")
        class Manager2:
            # [CLASS DOC]
            # HELPER: empty class whose registration should conflict.
            pass

    assert REWARD_MANAGER_REGISTRY["conflict_manager"] == Manager1


def test_decorator_returns_original_class(setup):
    # [FUNCTION DOC]
    # Asserts the decorator returns the class unchanged (its method still works) and registers it.
    """Test that the decorator returns the original class unchanged."""

    @register("return_test")
    class OriginalClass:
        # [CLASS DOC]
        # HELPER: class with one method used to verify the decorator is transparent.
        def method(setup):
            # [FUNCTION DOC]
            # HELPER: returns 42 (its self parameter is oddly named setup).
            return 42

    assert OriginalClass().method() == 42
    assert REWARD_MANAGER_REGISTRY["return_test"] == OriginalClass
