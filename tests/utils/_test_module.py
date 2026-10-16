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


# Test module for import_utils.load_extern_object testing

# [MODULE DOC]
# HELPER: fixture module imported by tests of import_utils.load_extern_object. Exposes a class, a
# constant (TEST_CONSTANT) and a function so loading each kind of object by name can be verified.
class TestClass:
    # [CLASS DOC]
    # HELPER: trivial class holding a single value, used as a loadable object.
    """A test class to be imported by load_extern_object"""

    def __init__(self, value=None):
        # [FUNCTION DOC]
        # HELPER: stores value, falling back to "default" when value is falsy (None, empty string, 0, etc.).
        # Inputs: value - optional anything. Outputs: none. Side effects: sets self.value.
        self.value = value or "default"

    def get_value(self):
        # [FUNCTION DOC]
        # HELPER: returns the stored value. Inputs: none. Outputs: self.value.
        return self.value


TEST_CONSTANT = "test_constant_value"


def test_function():
    # [FUNCTION DOC]
    # HELPER: returns the constant string "test_function_result". Not a pytest test despite its name
    # (it takes no fixtures and always passes if collected).
    return "test_function_result"
