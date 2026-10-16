# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
# FP8 quantization helper specialization for the TensorRT-LLM rollout backend.

from verl.utils.fp8_utils import FP8QuantizerHelper


class TRTLLMFP8QuantizerHelper(FP8QuantizerHelper):
    # [CLASS DOC]
    # FP8 quantizer helper for TRT-LLM; currently identical to the generic FP8QuantizerHelper and exists
    # as a backend-specific extension point.
    def __init__(self, quant_config):
        # [FUNCTION DOC]
        # Behavior: delegates to the parent constructor with the given quant_config.
        # Inputs: quant_config - FP8 quantization config. Outputs: none. Side effects: parent init state.
        super().__init__(quant_config)
