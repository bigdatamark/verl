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
# [PACKAGE DOC]
# single_controller package: the driver-side single-controller programming model (workers, worker groups, resource pools).
# Behavior at import: re-exports everything from the base subpackage, and reads the package version string from the file
# version/version located in the parent (verl) directory, stripping whitespace, into __version__.
# Exceptions: import fails with an OS error if the version file is missing.
# Side effects: file read at import time.
import os

from . import base
from .base import *

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# Note(haibin.lin): single_controller.__version__ is deprecated
with open(os.path.join(os.path.join(version_folder, os.pardir), "version/version")) as f:
    __version__ = f.read().strip()


__all__ = base.__all__
