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
# Import shim for Megatron-Bridge: re-exports AutoBridge plus LinearForLastLayer, freeze_moe_router and
# make_value_model. When the installed Megatron-Bridge (< v0.5.0) lacks the latter three, local fallback
# implementations defined here are used and LinearForLastLayer is registered with AutoMapping as a replicated module.
# Side effects on import: prints an install hint and re-raises if Megatron-Bridge is missing.

try:
    from megatron.bridge import AutoBridge
except ImportError:
    print("Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`")
    raise

# Megatron-Bridge >= v0.5.0 exposes these symbols in train_utils.
# Megatron-Bridge <  v0.5.0 does not, so we fall back to local implementations.
try:
    from megatron.bridge.training.utils.train_utils import LinearForLastLayer, freeze_moe_router, make_value_model
except ImportError:
    import torch
    from megatron.core import tensor_parallel

    def _ensure_model_list(model):
        # [FUNCTION DOC]
        # HELPER: Wraps a single model in a list; returns lists unchanged.
        return model if isinstance(model, list) else [model]

    class LinearForLastLayer(torch.nn.Linear):
        # [CLASS DOC]
        # Fallback bias-free linear output head (used for value models) returning float logits, optionally gathered from the
        # sequence-parallel region.
        def __init__(self, input_size, output_size, *, sequence_parallel: bool):
            # [FUNCTION DOC]
            # Behavior: creates a Linear(input_size -> output_size) without bias and records sequence_parallel; when enabled,
            # marks the weight as sequence-parallel so its gradients are all-reduced accordingly.
            # Side effects: allocates the weight parameter.
            super().__init__(in_features=input_size, out_features=output_size, bias=False)
            self.sequence_parallel = sequence_parallel
            if self.sequence_parallel:
                self.weight.sequence_parallel = True

        def forward(self, input_, weight=None, runtime_gather_output=None):
            # [FUNCTION DOC]
            # Behavior: applies the linear layer, casts to float32, and if sequence parallel gathers outputs across the sequence
            # parallel group (no output-grad reduction).
            # Inputs: input_ - hidden states; weight, runtime_gather_output - ignored (accepted for interface compatibility).
            # Outputs: tuple (logits, None) mirroring Megatron's output-layer return convention.
            logits = super().forward(input_)
            logits = logits.float()
            if self.sequence_parallel:
                logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
            return logits, None

    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("LinearForLastLayer", "replicated")

    def make_value_model(hidden_size, sequence_parallel):
        # [FUNCTION DOC]
        # Behavior: returns a post-creation hook that replaces output_layer with a LinearForLastLayer(hidden_size -> 1) on
        # every model chunk that is on the last pipeline stage (per virtual stage when PP>1 with virtual pipeline).
        # Inputs: hidden_size; sequence_parallel flag.
        # Outputs: the hook callable.
        from megatron.core import parallel_state

        def hook(model):
            # [FUNCTION DOC]
            # HELPER: The hook: computes per-chunk last-stage flags, asserts their count matches the model chunk list, and swaps
            # in the value head on last-stage chunks.
            # Inputs: model - module or list of chunks. Outputs: none.
            # Exceptions: AssertionError on length mismatch.
            # Side effects: mutates model chunks.
            model_post_process = []
            if (
                parallel_state.get_pipeline_model_parallel_world_size() > 1
                and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
            ):
                for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                    model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
            else:
                model_post_process.append(parallel_state.is_pipeline_last_stage())

            model_list = _ensure_model_list(model)
            assert len(model_post_process) == len(model_list), (
                "Model list length and post process list length must match."
            )

            for index, model_chunk in enumerate(model_list):
                if not model_post_process[index]:
                    continue
                model_chunk.output_layer = LinearForLastLayer(
                    input_size=hidden_size,
                    output_size=1,
                    sequence_parallel=sequence_parallel,
                )

        return hook

    def freeze_moe_router(model):
        # [FUNCTION DOC]
        # Behavior: for every decoder layer of each model chunk, disables gradients on MoE router weight/bias and on shared
        # expert gate weight/bias when present.
        # Inputs: model - module or list of chunks. Outputs: the same model.
        # Side effects: mutates requires_grad flags.
        for model_chunk in _ensure_model_list(model):
            if hasattr(model_chunk, "decoder") and hasattr(model_chunk.decoder, "layers"):
                for layer in model_chunk.decoder.layers:
                    if hasattr(layer.mlp, "router"):
                        if hasattr(layer.mlp.router, "weight"):
                            layer.mlp.router.weight.requires_grad = False
                        if hasattr(layer.mlp.router, "bias"):
                            layer.mlp.router.bias.requires_grad = False
                    if hasattr(layer.mlp, "shared_experts"):
                        if (
                            hasattr(layer.mlp.shared_experts, "gate_weight")
                            and layer.mlp.shared_experts.gate_weight is not None
                        ):
                            layer.mlp.shared_experts.gate_weight.requires_grad = False
                        if (
                            hasattr(layer.mlp.shared_experts, "gate_bias")
                            and layer.mlp.shared_experts.gate_bias is not None
                        ):
                            layer.mlp.shared_experts.gate_bias.requires_grad = False
        return model


__all__ = [
    "AutoBridge",
    "LinearForLastLayer",
    "freeze_moe_router",
    "make_value_model",
]
