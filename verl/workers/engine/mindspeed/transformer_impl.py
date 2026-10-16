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

# [MODULE DOC]
# MindSpeed (Ascend NPU) engines: Megatron LM-head and value-head engines registered for device "npu" that re-apply MindSpeed patches with verl's config before model-parallel init.
# Side effects: tries to import mindspeed.megatron_adaptor.repatch at import time (None if unavailable).

import logging
import os

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from verl.trainer.config import CheckpointConfig
from verl.workers.config import (
    HFModelConfig,
    McoreEngineConfig,
    McoreOptimizerConfig,
)

from ..base import EngineRegistry
from ..megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead
from .utils import (
    reset_fp8_reuse_quantized_weight,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _mindspeed_repatch(engine_config):
    # [FUNCTION DOC]
    # HELPER: Re-applies MindSpeed patches using the engine's override_transformer_config.
    # Behavior: no-op if repatch is unavailable; converts the override config's attention backend strings, sets use_flash_attn false when use_flash_attn_npu_batch_invariant is set, otherwise defaults use_flash_attn to true; injects context_parallel_size when greater than 1; calls repatch.
    # Inputs: engine_config - McoreEngineConfig.
    # Side effects: global monkey patching of Megatron by MindSpeed.
    if repatch is not None:
        from verl.utils.megatron_utils import mapping_string_to_attn_backend

        repatch_config = mapping_string_to_attn_backend(dict(engine_config.get("override_transformer_config", {})))
        # flash-attn-npu batch-invariant replaces DotProductAttention.forward; fusion attention
        # registers the same patch when use_flash_attn=True and causes "the patch of forward exist".
        if repatch_config.get("use_flash_attn_npu_batch_invariant"):
            repatch_config["use_flash_attn"] = False
        else:
            repatch_config.setdefault("use_flash_attn", True)
        if engine_config.context_parallel_size > 1:
            repatch_config["context_parallel_size"] = engine_config.context_parallel_size
        repatch(repatch_config)


@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    # [CLASS DOC]
    # MegatronEngineWithLMHead for NPU with MindSpeed repatching and FP8 quantized-weight cache reset on device moves.
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # [FUNCTION DOC]
        # Forwards all configs to MegatronEngineWithLMHead.
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # [FUNCTION DOC]
        # HELPER: Repatches MindSpeed with the engine config (so CP wrappers apply), then runs the parent device-mesh init.
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # [FUNCTION DOC]
        # Resets the FP8 quantized-weight reuse cache (when enabled) and then delegates to the parent to move model/optimizer/grad to device.
        # Inputs: device, model, optimizer, grad - as in BaseEngine.to.
        # Side effects: device transfers; MindSpeed cache mutation.
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        reset_fp8_reuse_quantized_weight(self, device, model, optimizer, grad)
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)


@EngineRegistry.register(model_type="value_model", backend="megatron", device="npu")
class MindspeedEngineWithValueHead(MegatronEngineWithValueHead):
    # [CLASS DOC]
    # MegatronEngineWithValueHead for NPU with MindSpeed repatching.
    # KNOWN ISSUE: unlike the LM-head variant, this class does not override to, so the FP8 quantized-weight reuse cache is not reset on device moves for value models.
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        # [FUNCTION DOC]
        # Forwards all configs to MegatronEngineWithValueHead.
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def _init_device_mesh(self):
        # [FUNCTION DOC]
        # HELPER: Repatches MindSpeed with the engine config, then runs the parent device-mesh init.
        # repatch must happen before initialize_model_parallel so that
        # initialize_model_parallel_cp_wrapper is in effect when the call is made.
        # The initial MindSpeed patch pass sees context_parallel_size=1 (default) because
        # verl passes CP size via hydra config rather than --context-parallel-size CLI arg,
        # so the CP ring-rank initialization wrapper is not registered on the first pass.
        _mindspeed_repatch(self.engine_config)
        super()._init_device_mesh()
