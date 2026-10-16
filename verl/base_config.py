# Copyright 2025 Bytedance Ltd. and/or its affiliates

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
# Defines BaseConfig, the common base for verl's dataclass configs: a frozen-by-default dataclass that also behaves as
# a read-only Mapping (dict-style access, iteration and length over its dataclass fields).

import collections
from dataclasses import FrozenInstanceError, dataclass, fields
from typing import Any


# BaseConfig class inherits from collections.abc.Mapping, which means it can act like a dictionary
@dataclass
class BaseConfig(collections.abc.Mapping):
    # [CLASS DOC]
    # Dataclass base implementing the Mapping interface. Fields become immutable once set (assignment to an existing
    # attribute raises FrozenInstanceError) unless their name is in the class-level _mutable_fields set. Field _target_
    # (default "") supports Hydra-style instantiation.
    # KNOWN ISSUE: __getitem__ raises AttributeError, not KeyError, for missing keys, so the Mapping-provided membership test
    # ("key in config") raises instead of returning False for absent keys.
    # KNOWN ISSUE: _mutable_fields is a shared class-level set; subclasses that mutate it instead of redefining it would
    # affect all configs.
    """The BaseConfig provides dict-like interface for a dataclass config.

    By default all fields in the config is not mutable, unless specified in
    "_mutable_fields". The BaseConfig class implements the Mapping Abstract Base Class.
    This allows instances of this class to be used like dictionaries.
    """

    _mutable_fields = set()
    _target_: str = ""

    def __setattr__(self, name: str, value):
        # [FUNCTION DOC]
        # Allows setting an attribute if it is not yet in the instance dict (e.g. during dataclass init) or if its name is in
        # _mutable_fields; otherwise raises FrozenInstanceError naming the field.
        # Inputs: name - attribute name; value - new value. Side effects: sets the attribute.
        """Set the value of an attribute. Check if the attr is mutable before setting the value."""
        # If the field already exists, it's considered frozen unless it's in _mutable_fields
        if name in self.__dict__ and name not in getattr(self, "_mutable_fields", set()):
            raise FrozenInstanceError(f"Field '{name}' is frozen and cannot be modified")
        super().__setattr__(name, value)

    def get(self, key: str, default: Any = None) -> Any:
        # [FUNCTION DOC]
        # Dict-style get: returns the attribute named key, or default when attribute lookup raises AttributeError.
        """Get the value associated with the given key. If the key does not exist, return the default value.

        Args:
            key (str): The attribute name to retrieve.
            default (Any, optional): The value to return if the attribute does not exist. Defaults to None.

        Returns:
            Any: The value of the attribute or the default value.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str):
        # [FUNCTION DOC]
        # Returns the attribute named key. Exceptions: AttributeError if missing; TypeError if key is not a string.
        """Implement the [] operator for the class. Allows accessing attributes like dictionary items.

        Args:
            key (str): The attribute name to retrieve.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute does not exist.
            TypeError: If the key type is not string
        """
        return getattr(self, key)

    def __iter__(self):
        # [FUNCTION DOC]
        # Yields the name of each dataclass field in declaration order (non-field class attributes are excluded).
        """Implement the iterator protocol. Allows iterating over the attribute names of the instance.

        Yields:
            str: The name of each field in the dataclass.
        """
        for f in fields(self):
            yield f.name

    def __len__(self):
        # [FUNCTION DOC]
        # Returns the number of dataclass fields.
        """
        Return the number of fields in the dataclass.

        Returns:
            int: The number of fields in the dataclass.
        """
        return len(fields(self))
