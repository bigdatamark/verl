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
# Registry lookups mapping HuggingFace architecture names to Megatron-core weight loader / saver functions.
# Imports of the mcore loader/saver modules are deferred into each function to avoid heavy imports at module load.


def get_weight_loader(arch: str):
    # [FUNCTION DOC]
    # Behavior: lazily imports the GPT-model Megatron loader and returns the state-dict loading function registered for the
    # architecture. Supported: LlamaForCausalLM, Qwen2ForCausalLM (both map to the generic GPT-model loader).
    # Inputs: arch - HF architecture class name string.
    # Outputs: callable that loads a HF state dict into a Megatron GPT model.
    # Exceptions: ValueError if arch is not registered (message lists supported keys).
    # Side effects: imports verl.models.mcore.loader.
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

    _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY = {
        "LlamaForCausalLM": load_state_dict_to_megatron_gptmodel,
        "Qwen2ForCausalLM": load_state_dict_to_megatron_gptmodel,
    }

    if arch in _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} loader are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_LOADER_REGISTRY.keys()}"
    )


def get_weight_saver(arch: str):
    # [FUNCTION DOC]
    # Behavior: lazily imports Megatron checkpoint merge functions and returns the saver registered for the architecture.
    # Llama/Qwen2/Qwen3 dense and token-classification variants use the generic GPT merger; Mixtral, Qwen2-MoE/Qwen3-MoE,
    # Qwen2.5-VL and DeepseekV3 use their specialized mergers (Qwen3Moe reuses the Qwen MoE merger).
    # Inputs: arch - HF architecture class name string.
    # Outputs: callable that merges a distributed Megatron checkpoint into a HF-style state dict.
    # Exceptions: ValueError if arch is not registered.
    # Side effects: imports verl.models.mcore.saver.
    from verl.models.mcore.saver import (
        merge_megatron_ckpt_gptmodel,
        merge_megatron_ckpt_gptmodel_dpskv3,
        merge_megatron_ckpt_gptmodel_mixtral,
        merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        merge_megatron_ckpt_gptmodel_qwen_moe,
    )

    _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY = {
        "LlamaForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen2ForCausalLM": merge_megatron_ckpt_gptmodel,
        "MixtralForCausalLM": merge_megatron_ckpt_gptmodel_mixtral,
        "Qwen2MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "Qwen2_5_VLForConditionalGeneration": merge_megatron_ckpt_gptmodel_qwen2_5_vl,
        "DeepseekV3ForCausalLM": merge_megatron_ckpt_gptmodel_dpskv3,
        "Qwen3ForCausalLM": merge_megatron_ckpt_gptmodel,
        "Qwen3ForTokenClassification": merge_megatron_ckpt_gptmodel,
        "Qwen3MoeForCausalLM": merge_megatron_ckpt_gptmodel_qwen_moe,
        "LlamaForTokenClassification": merge_megatron_ckpt_gptmodel,
    }
    if arch in _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY:
        return _MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {arch} saver are not supported for now. Supported architectures: "
        f"{_MODEL_WEIGHT_MEGATRON_SAVER_REGISTRY.keys()}"
    )
