# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Utilities for the MindSpeed engine backend, related to FP8 quantized weight reuse on NPU.


def reset_fp8_reuse_quantized_weight(engine, device: str, model: bool, optimizer: bool, grad: bool):
    # [FUNCTION DOC]
    # Resets MindSpeed's FP8 quantized-weight reuse cache when that feature is enabled.
    # Behavior: reads engine.engine_config.override_transformer_config; if it has fp8_reuse_quantized_weight truthy, clears the quantization reuse cache (releasing storage) and enables high-precision weight release only when the engine's mode attribute equals "train" (disabled otherwise, e.g. for ref models that must keep weights for offload).
    # Inputs: engine - engine with engine_config and optional mode; device, model, optimizer, grad - accepted for call-site compatibility but unused.
    # Outputs: none.
    # Side effects: mutates MindSpeed global FP8 reuse state; lazy imports mindspeed.
    override_config = getattr(engine.engine_config, "override_transformer_config", None)
    if override_config and override_config.get("fp8_reuse_quantized_weight", False):
        from mindspeed.te.pytorch.fp8.reuse import (
            clear_weight_quantization_reuse_cache,
            set_weight_release_enabled,
        )

        # clear quantized weights on NPU
        clear_weight_quantization_reuse_cache(release_storage=True)

        # enable release high-precision weights only when all modules are in training mode. For ref model,
        # we need to keep its high-precision weights for offloading. For actor_update model, the high-precision
        # weights will be released if possible, and then recovered before optimizer step
        set_weight_release_enabled(getattr(engine, "mode", None) == "train")
