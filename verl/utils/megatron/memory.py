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
# Flat, contiguous device memory buffer from which shaped tensor views can be carved.

import torch

from verl.utils.device import get_device_id


class MemoryBuffer:
    # [CLASS DOC]
    # A 1-D zero-initialized tensor of numel_padded elements on the current device; numel is the
    # logical usable size.
    def __init__(self, numel, numel_padded, dtype):
        # [FUNCTION DOC]
        # Inputs: numel - usable element count; numel_padded - allocated element count; dtype.
        # Side effects: allocates a zeroed tensor (no grad) on get_device_id() and stores it as data.
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded, dtype=self.dtype, device=get_device_id(), requires_grad=False)

    def zero(self):
        # [FUNCTION DOC]
        # Behavior: zeroes the whole buffer in place. Outputs: none.
        """Reset the buffer to zero."""
        self.data.zero_()

    def get(self, shape, start_index):
        # [FUNCTION DOC]
        # Behavior: returns a view of data[start_index : start_index + shape.numel()] reshaped to shape.
        # Inputs: shape - torch.Size; start_index - element offset.
        # Outputs: tensor view sharing storage with the buffer.
        # Exceptions: AssertionError if the range exceeds numel (not numel_padded).
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor
