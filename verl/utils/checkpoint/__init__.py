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
# HELPER: Checkpoint utilities package. Re-exports CheckpointHandler (orchestrates saving/loading training
#   checkpoints) and OrchestrationMode (the enumeration of how checkpointing is coordinated) from checkpoint_handler.
# Inputs: none. Outputs: public names CheckpointHandler and OrchestrationMode.
# Exceptions: import errors from checkpoint_handler propagate. Side effects: imports checkpoint_handler.

# [PACKAGE]
# Checkpoint utilities package. Exposes CheckpointHandler and OrchestrationMode from checkpoint_handler; the
# FSDP/Megatron checkpoint managers live in submodules and are imported directly.

from .checkpoint_handler import CheckpointHandler, OrchestrationMode

__all__ = ["CheckpointHandler", "OrchestrationMode"]
