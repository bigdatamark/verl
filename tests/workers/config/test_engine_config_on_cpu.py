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
# CPU tests for engine config dataclasses McoreEngineConfig and FSDPEngineConfig: defaults, post-init
# auto-correction, frozen vs mutable fields and offload flags. Side effects: none.

import pytest

from verl.workers.config.engine import FSDPEngineConfig, McoreEngineConfig


class TestMcoreEngineConfig:
    # [CLASS DOC]
    # Tests for McoreEngineConfig.
    def test_default_values(self):
        # [FUNCTION DOC]
        # Asserts defaults: TP size 1, sequence_parallel False (auto-corrected because TP is 1), seed 42.
        config = McoreEngineConfig()
        assert config.tensor_model_parallel_size == 1
        assert config.sequence_parallel is False  # Should be auto-corrected
        assert config.seed == 42

    def test_post_init_validation(self):
        # [FUNCTION DOC]
        # Asserts sequence_parallel is forced False when TP size is 1 and becomes True when TP size is 2.
        # Test TP size 1 forces sequence_parallel=False
        config = McoreEngineConfig(tensor_model_parallel_size=1)
        assert config.sequence_parallel is False

        # Test TP >1 keeps sequence_parallel=True
        config = McoreEngineConfig(tensor_model_parallel_size=2)
        assert config.sequence_parallel is True

    def test_mutable_fields(self):
        # [FUNCTION DOC]
        # Asserts sequence_parallel can be reassigned but assigning tensor_model_parallel_size raises AttributeError.
        config = McoreEngineConfig()
        config.sequence_parallel = True  # Should be mutable
        with pytest.raises(AttributeError):
            config.tensor_model_parallel_size = 2  # Frozen field

    @pytest.mark.parametrize("offload_field", ["param_offload", "optimizer_offload"])
    def test_offload_flags(self, offload_field):
        # [FUNCTION DOC]
        # Parametrized over param_offload / optimizer_offload; constructs config with that flag True and asserts it is set.
        # Inputs: offload_field - name of the boolean field.
        config = McoreEngineConfig(**{offload_field: True})
        assert getattr(config, offload_field) is True


class TestFSDPEngineConfigCPU:
    # [CLASS DOC]
    # Tests for FSDPEngineConfig on CPU.
    def test_default_values(self):
        # [FUNCTION DOC]
        # Asserts defaults: both offloads False, fsdp_size -1, use_no_sync_for_gradient_accumulation True.
        config = FSDPEngineConfig()
        assert config.param_offload is False
        assert config.optimizer_offload is False
        assert config.fsdp_size == -1
        assert config.use_no_sync_for_gradient_accumulation is True

    def test_gradient_accumulation_sync_can_be_restored_per_micro_batch(self):
        # [FUNCTION DOC]
        # Asserts use_no_sync_for_gradient_accumulation can be set to False at construction.
        config = FSDPEngineConfig(use_no_sync_for_gradient_accumulation=False)
        assert config.use_no_sync_for_gradient_accumulation is False

    @pytest.mark.parametrize(
        "offload_params",
        [{"param_offload": True}, {"optimizer_offload": True}, {"param_offload": True, "optimizer_offload": True}],
    )
    def test_offload_combinations(self, offload_params):
        # [FUNCTION DOC]
        # Parametrized over dicts of offload flags; asserts each flag equals the given value or default False.
        # Inputs: offload_params - mapping of field name to bool.
        config = FSDPEngineConfig(**offload_params)
        assert config.param_offload == offload_params.get("param_offload", False)
        assert config.optimizer_offload == offload_params.get("optimizer_offload", False)

    def test_wrap_policy_configuration(self):
        # [FUNCTION DOC]
        # Asserts a provided wrap_policy mapping is stored unchanged.
        test_policy = {"layer_class": "TransformerBlock"}
        config = FSDPEngineConfig(wrap_policy=test_policy)
        assert config.wrap_policy == test_policy
