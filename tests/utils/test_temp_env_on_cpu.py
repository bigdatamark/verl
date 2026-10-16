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
# CPU tests for the temp_env_var context manager, which sets an environment
# variable for the duration of a block and afterwards restores its previous value
# or removes it if it did not exist, including when the block raises.

import os

import pytest

from verl.utils.py_functional import temp_env_var


@pytest.fixture(autouse=True)
def clean_env():
    # [FUNCTION DOC]
    # HELPER: Autouse fixture snapshotting the whole environment, deleting TEST_VAR,
    # TEST_VAR_2 and EXISTING_VAR before each test, then after the test clearing the
    # environment and restoring the snapshot.
    # Side effects: mutates the process environment.
    """Fixture to clean up environment variables before and after each test."""
    # Store original environment state
    original_env = dict(os.environ)

    # Clean up any test variables that might exist
    test_vars = ["TEST_VAR", "TEST_VAR_2", "EXISTING_VAR"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

    # Yield control to the test function
    yield

    # Restore original environment state after test
    os.environ.clear()
    os.environ.update(original_env)


def test_set_new_env_var():
    # [FUNCTION DOC]
    # Behavior: a previously absent variable is set inside the block and absent after.
    """Test setting a new environment variable that didn't exist before."""
    # Ensure variable doesn't exist
    assert "TEST_VAR" not in os.environ

    with temp_env_var("TEST_VAR", "test_value"):
        # Variable should be set inside context
        assert os.environ["TEST_VAR"] == "test_value"
        assert "TEST_VAR" in os.environ

    # Variable should be removed after context
    assert "TEST_VAR" not in os.environ


def test_restore_existing_env_var():
    # [FUNCTION DOC]
    # Behavior: an existing variable takes the temporary value inside and its original
    # value after.
    """Test restoring an environment variable that already existed."""
    # Set up existing variable
    os.environ["EXISTING_VAR"] = "original_value"

    with temp_env_var("EXISTING_VAR", "temporary_value"):
        # Variable should be temporarily changed
        assert os.environ["EXISTING_VAR"] == "temporary_value"

    # Variable should be restored to original value
    assert os.environ["EXISTING_VAR"] == "original_value"


def test_env_var_restored_on_exception():
    # [FUNCTION DOC]
    # Behavior: a ValueError raised inside the block propagates and the original value
    # is still restored.
    """Test that environment variables are restored even when exceptions occur."""
    # Set up existing variable
    os.environ["EXISTING_VAR"] = "original_value"

    with pytest.raises(ValueError):
        with temp_env_var("EXISTING_VAR", "temporary_value"):
            # Verify variable is set
            assert os.environ["EXISTING_VAR"] == "temporary_value"
            # Raise exception
            raise ValueError("Test exception")

    # Variable should still be restored despite exception
    assert os.environ["EXISTING_VAR"] == "original_value"


def test_nested_context_managers():
    # [FUNCTION DOC]
    # Behavior: nested blocks on the same variable restore each outer level in turn
    # (level2 back to level1, then to original).
    """Test nested temp_env_var context managers."""
    # Set up original variable
    os.environ["TEST_VAR"] = "original"

    with temp_env_var("TEST_VAR", "level1"):
        assert os.environ["TEST_VAR"] == "level1"

        with temp_env_var("TEST_VAR", "level2"):
            assert os.environ["TEST_VAR"] == "level2"

        # Should restore to level1
        assert os.environ["TEST_VAR"] == "level1"

    # Should restore to original
    assert os.environ["TEST_VAR"] == "original"


def test_multiple_different_vars():
    # [FUNCTION DOC]
    # Behavior: nested blocks on two different variables restore the existing one and
    # remove the newly created one.
    """Test setting multiple different environment variables."""
    # Set up one existing variable
    os.environ["EXISTING_VAR"] = "existing_value"

    with temp_env_var("EXISTING_VAR", "modified"):
        with temp_env_var("TEST_VAR", "new_value"):
            assert os.environ["EXISTING_VAR"] == "modified"
            assert os.environ["TEST_VAR"] == "new_value"

    # Check restoration
    assert os.environ["EXISTING_VAR"] == "existing_value"
    assert "TEST_VAR" not in os.environ


def test_empty_string_value():
    # [FUNCTION DOC]
    # Behavior: an empty-string value is set (present but empty) and the variable is
    # removed afterwards.
    """Test setting environment variable to empty string."""
    with temp_env_var("TEST_VAR", ""):
        assert os.environ["TEST_VAR"] == ""
        assert "TEST_VAR" in os.environ

    # Should be removed after context
    assert "TEST_VAR" not in os.environ


def test_overwrite_with_empty_string():
    # [FUNCTION DOC]
    # Behavior: overwriting an existing value with an empty string is restored after.
    """Test overwriting existing variable with empty string."""
    os.environ["EXISTING_VAR"] = "original"

    with temp_env_var("EXISTING_VAR", ""):
        assert os.environ["EXISTING_VAR"] == ""

    # Should restore original value
    assert os.environ["EXISTING_VAR"] == "original"


def test_context_manager_returns_none():
    # [FUNCTION DOC]
    # Behavior: the context manager yields None while the variable is set.
    """Test that context manager yields None."""
    with temp_env_var("TEST_VAR", "value") as result:
        assert result is None
        assert os.environ["TEST_VAR"] == "value"
