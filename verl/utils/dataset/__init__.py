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
# HELPER: Dataset package. Re-exports RLHFDataset (prompt dataset for RL training) from rl_dataset and
#   RMDataset (reward-model pairwise dataset) from rm_dataset.
# Inputs: none. Outputs: public names RLHFDataset and RMDataset.
# Exceptions: import errors from the submodules propagate. Side effects: imports both submodules.

# [PACKAGE]
# Dataset package. Exposes RLHFDataset (prompt dataset for RL) and RMDataset (reward-model dataset).

from .rl_dataset import RLHFDataset
from .rm_dataset import RMDataset

__all__ = ["RLHFDataset", "RMDataset"]
