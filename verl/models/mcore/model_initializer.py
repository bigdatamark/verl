# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# use mcore transformer config to initialize the model

# [MODULE DOC]
# Per-architecture initializers that build Megatron-Core GPTModel instances from an mcore TransformerConfig and the HF
# config: they choose the Transformer Engine decoder block spec, rope settings, optional MTP block, an optional scalar
# value head, and MoE router freezing defaults.

import inspect
from abc import ABC, abstractmethod

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from .config_converter import PretrainedConfig, TransformerConfig, get_hf_rope_theta


class BaseModelInitializer(ABC):
    # [CLASS DOC]
    # Abstract initializer storing tfconfig, hf_config and whether the installed Megatron's decoder spec function accepts a
    # vp_stage argument (has_vp_stage). Subclasses provide get_transformer_layer_spec.
    """Base class for model initializers."""

    def __init__(self, tfconfig: TransformerConfig, hf_config: PretrainedConfig):
        # [FUNCTION DOC]
        # Behavior: stores the configs and detects vp_stage support by inspecting get_gpt_decoder_block_spec's signature.
        self.tfconfig = tfconfig
        self.hf_config = hf_config
        self.has_vp_stage = inspect.signature(get_gpt_decoder_block_spec).parameters.get("vp_stage", None) is not None

    @abstractmethod
    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Abstract: returns the decoder block spec for the given virtual pipeline stage. Base implementation does nothing.
        """Get the transformer layer specification.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_layer_specs.py"""
        pass

    def get_rope_scaling_args(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: if the HF config has a non-null rope_scaling, returns its "factor" as seq_len_interpolation_factor;
        # otherwise an empty dict.
        # Exceptions: KeyError if rope_scaling lacks "factor".
        """Get rope scaling args."""
        rope_scaling_args = {}
        if "rope_scaling" in self.hf_config:
            if self.hf_config.rope_scaling is not None:
                # assert self.hf_config.rope_scaling["type"] == "linear", "only linear scaling is supported for now"
                rope_scaling_args["seq_len_interpolation_factor"] = self.hf_config.rope_scaling["factor"]
        return rope_scaling_args

    def initialize(
        self,
        pre_process: bool = True,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
        value: bool = False,
        **extra_kwargs,
    ) -> GPTModel:
        # [FUNCTION DOC]
        # Behavior: builds a GPTModel with the layer spec for extra_kwargs vp_stage, HF vocab size and max positions, rope
        # position embedding with HF rope theta and scaling args, optional mtp_block_spec from extra_kwargs, and vp_stage when
        # supported. If post_process and value, replaces output_layer with a bias-free hidden_size -> 1 LinearForLastLayer.
        # Inputs: pre_process, post_process, share_embeddings_and_output_weights, value; extra_kwargs may contain vp_stage and
        # mtp_block_spec (others ignored).
        # Outputs: GPTModel.
        """Initialize a GPT model with the given configuration.
        https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py

        Args:
            pre_process (bool): include embedding layer.
            post_process (bool): including an output layer.
            share_embeddings_and_output_weights (bool): input embeddings and output logit weights are shared.
            value (bool): add an extra linear layer for classification or regression.

        Returns:
            GPTModel: An initialized GPT model instance
        """
        vp_stage = extra_kwargs.get("vp_stage", None)
        transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
        rope_scaling_args = self.get_rope_scaling_args()
        mtp_block_spec = extra_kwargs.get("mtp_block_spec", None)
        model = GPTModel(
            config=self.tfconfig,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type="rope",
            rotary_base=get_hf_rope_theta(self.hf_config),
            **rope_scaling_args,
            mtp_block_spec=mtp_block_spec,
            **({} if not self.has_vp_stage else {"vp_stage": vp_stage}),
        )

        if post_process and value:
            from verl.models.mcore.bridge import LinearForLastLayer

            model.output_layer = LinearForLastLayer(
                input_size=self.tfconfig.hidden_size, output_size=1, sequence_parallel=self.tfconfig.sequence_parallel
            )

        return model


class DenseModel(BaseModelInitializer):
    # [CLASS DOC]
    # Initializer for dense models (Llama, Qwen2, Qwen3 dense).
    """Initializer for dense models like Llama and Qwen2."""

    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Behavior: asserts RMSNorm and returns the Transformer Engine GPT decoder block spec (with vp_stage when supported).
        # Exceptions: AssertionError for non-RMSNorm configs.
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        return get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)


class Qwen2MoEModel(BaseModelInitializer):
    # [CLASS DOC]
    # Initializer for Qwen2-MoE: enables gating on shared experts and freezes routers by default.
    """Initializer for Qwen2 MoE models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Behavior: asserts RMSNorm, gets the TE decoder block spec, and sets the shared_experts "gate" param to true on every
        # layer spec.
        # Exceptions: AssertionError for non-RMSNorm; AttributeError if a layer spec has no shared experts.
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)

        # Patch layer spec for shared experts
        for i in range(len(transformer_layer_spec.layer_specs)):
            transformer_layer_spec.layer_specs[i].submodules.mlp.submodules.shared_experts.params["gate"] = True

        return transformer_layer_spec

    def initialize(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: builds the model via the base class; unless freeze_moe_router is false (default true), disables gradients
        # on every decoder layer's router weight.
        # Inputs: kwargs - base initialize arguments plus optional freeze_moe_router.
        # Outputs: GPTModel.
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class MixtralModel(BaseModelInitializer):
    # [CLASS DOC]
    # Initializer for Mixtral; routers are trainable by default.
    """Initializer for Mixtral models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Behavior: asserts RMSNorm and returns the TE decoder block spec.
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: builds via the base class; if freeze_moe_router (default false) disables router weight gradients on every layer.
        # Outputs: GPTModel.
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", False)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class Qwen3MoEModel(BaseModelInitializer):
    # [CLASS DOC]
    # Initializer for Qwen3-MoE; routers frozen by default.
    """Initializer for Qwen3 MoE models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Behavior: asserts RMSNorm and returns the TE decoder block spec.
        assert self.tfconfig.normalization == "RMSNorm", "only RMSNorm is supported for now"
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def initialize(self, **kwargs):
        # [FUNCTION DOC]
        # Behavior: builds via the base class; if freeze_moe_router (default true) disables router weight gradients on every layer.
        # Outputs: GPTModel.
        # Qwen default freeze_moe_router: true
        model = super().initialize(**kwargs)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                layer.mlp.router.weight.requires_grad = False
        return model


class DeepseekV3Model(BaseModelInitializer):
    # [CLASS DOC]
    # Initializer for DeepseekV3 (MLA + MoE, optional MTP); rope scaling is handled in the MLA config, not here.
    """Initializer for DeepseekV3 models."""

    def get_transformer_layer_spec(self, vp_stage=None):
        # [FUNCTION DOC]
        # Behavior: returns the TE decoder block spec (no RMSNorm assertion).
        extra_kwargs = {} if not self.has_vp_stage else {"vp_stage": vp_stage}
        transformer_layer_spec = get_gpt_decoder_block_spec(self.tfconfig, use_transformer_engine=True, **extra_kwargs)
        return transformer_layer_spec

    def get_rope_scaling_args(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: returns an empty dict (rope scaling comes from the MLA transformer config).
        """Get rope scaling args."""
        rope_scaling_args = {}
        return rope_scaling_args

    def initialize(
        self,
        **kwargs,
    ):
        # [FUNCTION DOC]
        # Behavior: if freeze_moe_router (default true), sets the config's load balancing type to "none" before building and
        # afterwards disables router weight gradients on layers that have a router. When mtp_num_layers > 0, builds an MTP block
        # spec from the layer spec and passes it to the base initializer.
        # Inputs: kwargs - base initialize arguments plus optional vp_stage and freeze_moe_router.
        # Outputs: GPTModel.
        # Side effects: mutates self.tfconfig.moe_router_load_balancing_type.
        # KNOWN ISSUE: vp_stage is passed to get_gpt_mtp_block_spec unconditionally, even on Megatron versions where has_vp_stage is false.
        vp_stage = kwargs.get("vp_stage", None)
        freeze_moe_router = kwargs.get("freeze_moe_router", True)
        if freeze_moe_router:
            self.tfconfig.moe_router_load_balancing_type = "none"
        # MTP
        if self.tfconfig.mtp_num_layers is not None and self.tfconfig.mtp_num_layers > 0:
            transformer_layer_spec = self.get_transformer_layer_spec(vp_stage=vp_stage)
            mtp_block_spec = get_gpt_mtp_block_spec(
                self.tfconfig, transformer_layer_spec, use_transformer_engine=True, vp_stage=vp_stage
            )
            kwargs["mtp_block_spec"] = mtp_block_spec

        model = super().initialize(**kwargs)
        if freeze_moe_router:
            for layer in model.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    layer.mlp.router.weight.requires_grad = False
        return model
