# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
# CPU tests asserting FSDP2 model offload/load helpers in verl.utils.fsdp_utils move the model with
# non-blocking copies, using Mock models.

from unittest.mock import Mock

from verl.utils import fsdp_utils


def test_offload_fsdp2_model_to_cpu_uses_non_blocking_copy():
    # [FUNCTION DOC]
    # Calls offload_fsdp2_model_to_cpu on a Mock with empty_cache disabled and asserts model.to was called once
    # with "cpu" and non_blocking=True.
    model = Mock()

    fsdp_utils.offload_fsdp2_model_to_cpu(model, empty_cache=False)

    model.to.assert_called_once_with("cpu", non_blocking=True)


def test_load_fsdp2_model_to_gpu_uses_non_blocking_copy(monkeypatch):
    # [FUNCTION DOC]
    # Patches get_device_id to return a sentinel device, calls load_fsdp2_model_to_gpu, and asserts model.to was
    # called once with that device and non_blocking=True.
    model = Mock()
    device = object()
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: device)

    fsdp_utils.load_fsdp2_model_to_gpu(model)

    model.to.assert_called_once_with(device, non_blocking=True)
