# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0(the "License");
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
# Tests the FSDP engine's temperature helpers: detection of a scalar temperature of exactly 1 (fast path that skips
# scaling) and logits scaling that must return the same tensor for unit temperature and an out-of-place division otherwise.

import pytest
import torch

from verl.workers.engine.fsdp.transformer_impl import (
    _is_scalar_unit_temperature,
    _scale_logits_by_temperature,
)


@pytest.mark.parametrize("temperature", [1, 1.0])
def test_scalar_unit_temperature_is_detected(temperature):
    # [FUNCTION DOC]
    # Integer 1 and float 1.0 are detected as scalar unit temperature.
    assert _is_scalar_unit_temperature(temperature) is True


@pytest.mark.parametrize("temperature", [0.7, 2.0, torch.tensor(1.0)])
def test_non_unit_or_tensor_temperature_uses_general_path(temperature):
    # [FUNCTION DOC]
    # Non-unit scalars and any tensor (even of value 1) are not treated as unit temperature.
    assert _is_scalar_unit_temperature(temperature) is False


def test_unit_temperature_preserves_logits_storage_and_gradient():
    # [FUNCTION DOC]
    # With is_unit_temperature True, the input view is returned as the same object and gradients flow to the base tensor.
    base = torch.randn(1, 4, 8, requires_grad=True)
    logits_view = base.squeeze(0)

    scaled = _scale_logits_by_temperature(
        logits_view,
        torch.ones(4, 1),
        is_unit_temperature=True,
    )

    assert scaled is logits_view
    scaled.sum().backward()
    torch.testing.assert_close(base.grad, torch.ones_like(base))


def test_non_unit_temperature_is_out_of_place_and_has_correct_gradient():
    # [FUNCTION DOC]
    # Temperature 0.5 yields a new tensor equal to logits x 2 with gradient 2 everywhere.
    logits = torch.randn(4, 8, requires_grad=True)
    temperature = torch.full((4, 1), 0.5)

    scaled = _scale_logits_by_temperature(
        logits,
        temperature,
        is_unit_temperature=False,
    )

    assert scaled is not logits
    torch.testing.assert_close(scaled, logits * 2)
    scaled.sum().backward()
    torch.testing.assert_close(logits.grad, torch.full_like(logits, 2))
