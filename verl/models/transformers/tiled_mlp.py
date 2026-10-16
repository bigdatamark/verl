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
# Tiled (chunked) MLP for reduced activation memory: a custom autograd function that runs a SwiGLU MLP over sequence
# chunks without keeping intermediates, recomputing each chunk in backward, plus monkey-patch helpers that replace the
# forward of transformers' Llama/Qwen2/Qwen3 MLP classes. Module constant: mapping from model type to (module path, MLP
# class name); qwen2_5 reuses the Qwen2 MLP.

"""
FSDP2-compatible TiledMLP implementation for memory-efficient MLP computation.

This module provides a tiled MLP implementation that reduces peak memory usage
by processing the MLP forward/backward pass in chunks (tiles). This is particularly
useful for large models with FSDP2 training.
"""

import threading
from typing import Optional

import torch
import torch.nn as nn


class GradientAccumulator:
    # [CLASS DOC]
    # Accumulates parameter gradients across per-chunk backward passes. State: params list, total_shards, accumulation dtype
    # (default float32), a mapping param -> accumulated gradient tensor, the list of active hook handles, and a lock guarding
    # accumulation.
    """Gradient accumulator for TiledMLP (FSDP compatible).

    This class manages gradient accumulation across multiple shards during
    the backward pass of TiledMLP. It ensures correct gradient computation
    when processing input in chunks.
    """

    def __init__(self, params: list[torch.nn.Parameter], total_shards: int, dtype: torch.dtype = None):
        # [FUNCTION DOC]
        # Behavior: stores params and settings; for each param, moves any existing .grad into the accumulator (cast to the
        # accumulation dtype) and clears .grad, else starts the accumulator at zeros of the param's shape.
        # Inputs: params - parameters to track; total_shards - number of chunks; dtype - accumulation dtype or None for float32.
        # Side effects: clears param.grad of the given params.
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()

        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(self.grad_accumulation_dtype)
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(param, dtype=self.grad_accumulation_dtype)

    def install_hooks(self, is_last_shard: bool):
        # [FUNCTION DOC]
        # Behavior: removes previously installed hooks, then registers a gradient hook on every requires-grad param. Each hook,
        # under the lock, adds the incoming grad (cast to accumulation dtype) into the accumulator. For non-last shards it returns
        # None (gradient flows unchanged). On the last shard it clears param.grad and returns the full accumulated gradient
        # cast to the param dtype, so autograd writes the total into .grad exactly once.
        # Inputs: is_last_shard - whether the current chunk is the final one.
        # Side effects: registers hooks; mutates accumulator and param.grad.
        # KNOWN ISSUE: for non-last shards the hook returns None, so autograd also accumulates those chunk grads into param.grad;
        # correctness relies on the last hook clearing param.grad before the final value is written.
        """Install gradient hooks for the current shard."""
        self._remove_hooks()

        def create_hook(param):
            # [FUNCTION DOC]
            # HELPER: factory returning the per-parameter hook closure described in install_hooks.
            def hook(grad):
                # [FUNCTION DOC]
                # HELPER: the gradient hook; see install_hooks.
                with self.lock:
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    self.accumulated_grads[param] += grad_to_accum_dtype

                    if is_last_shard:
                        param.grad = None  # Critical: prevent double accumulation
                        final_grad = self.accumulated_grads[param].to(param.dtype)
                        return final_grad
                    return None

            return hook

        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def _remove_hooks(self):
        # [FUNCTION DOC]
        # HELPER: removes and forgets all registered hook handles.
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def cleanup(self):
        # [FUNCTION DOC]
        # Side effects: removes all hooks.
        """Cleanup hooks and resources."""
        self._remove_hooks()


class TiledMLP(torch.autograd.Function):
    # [CLASS DOC]
    # torch autograd Function implementing the tiled computation. Called as apply(fn, module, x, shards, compute_params).
    """TiledMLP implementation for memory-efficient MLP computation.

    This autograd function processes MLP forward/backward in tiles (chunks)
    to reduce peak memory usage. Compatible with FSDP2.
    """

    @staticmethod
    def forward(ctx, fn, module, x, shards, compute_params):
        # [FUNCTION DOC]
        # Behavior: saves fn, module, shard count and the requires-grad subset of compute_params on ctx, saves x for backward,
        # splits x into shards chunks along dim -2 (sequence), runs fn(module, chunk) for each chunk under no_grad and
        # concatenates outputs along dim -2.
        # Inputs: fn - callable(module, x) computing the MLP; module - MLP module; x - input of shape (..., seq, hidden);
        # shards - chunk count; compute_params - module params.
        # Outputs: tensor equal to fn(module, x).
        ctx.fn = fn
        ctx.module = module
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        # Split on dim=-2 (seqlen dimension) following Liger Kernel style
        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)
        return output_unsharded

    @staticmethod
    def backward(ctx, *grads):
        # [FUNCTION DOC]
        # Behavior: restores x, detaches it (preserving requires_grad), flattens x and the incoming gradient to
        # (tokens, hidden), preallocates a zero input-gradient buffer, and chunks tokens along dim 0. Creates a
        # GradientAccumulator with dtype of x. For each chunk: set the chunk's .grad to the matching narrowed slice of the buffer
        # (so input grads land in place), install hooks (last flag when index+1 equals shards), recompute fn with grad enabled,
        # and backprop the matching slice of the incoming gradient. Cleans up hooks afterwards.
        # Outputs: tuple (None, None, input gradient reshaped to original shape or None if x did not require grad, None, None).
        # Side effects: writes parameter .grad via hooks.
        # KNOWN ISSUE: the accumulator dtype is x.dtype (e.g. bf16) rather than the float32 default, reducing accumulation
        # precision. Also, chunking can produce fewer chunks than shards (short inputs), in which case the last-shard hook never
        # fires and a param's pre-existing gradient (moved into the accumulator) is dropped.
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        module = ctx.module
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        # Flatten to [bs*seqlen, hidden_size]
        hidden_size = x.shape[-1]
        x_shape_orig = x.shape
        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)

        # Pre-allocate input gradient
        x_grad = torch.zeros_like(x)

        # Split on dim=0
        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        grad_accumulator = GradientAccumulator(compute_params, shards, dtype=x.dtype)

        for i, x_shard in enumerate(x_shards):
            x_shard.requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            # narrow(0, ...) creates a contiguous view that can receive gradients
            x_shard.grad = x_grad.narrow(0, shard_offset, shard_step)
            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step)

            is_last_shard = i + 1 == shards
            grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                output = fn(module, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        grad_accumulator.cleanup()
        del grad_accumulator

        # Restore original shape
        x_grad = x_grad.view(x_shape_orig) if x_requires_grad else None
        return (None, None, x_grad, None, None)


def _mlp_forward_fn(module, x):
    # [FUNCTION DOC]
    # HELPER: SwiGLU MLP forward: down_proj(act_fn(gate_proj(x)) * up_proj(x)).
    """Forward function for LlamaMLP / Qwen2MLP / Qwen3MLP style."""
    return module.down_proj(module.act_fn(module.gate_proj(x)) * module.up_proj(x))


# ============================================================================
# Monkey Patch Functions
# ============================================================================

# Model type to MLP class mapping
_MODEL_TYPE_TO_MLP_CLASS = {
    "llama": ("transformers.models.llama.modeling_llama", "LlamaMLP"),
    "qwen2": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
    "qwen2_5": ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),  # Qwen2.5 uses Qwen2 MLP
    "qwen3": ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
}


def apply_tiled_mlp_monkey_patch(
    num_shards: int = 4,
    model_type: Optional[str] = None,
):
    # [FUNCTION DOC]
    # Behavior: chooses model types to patch (all supported if model_type is None, else just the given one). For each, imports
    # the transformers module, fetches the MLP class and patches its forward with the tiled version; Import/Attribute errors
    # print a warning and are skipped. Prints a summary of patched classes.
    # Inputs: num_shards - chunk count (default 4); model_type - optional model type key.
    # Outputs: list of unique patched class names.
    # Exceptions: ValueError for an unsupported model_type.
    # Side effects: globally replaces MLP class forward methods (must run before model instantiation); prints to stdout.
    """Apply TiledMLP monkey patch based on model_type.

    This function MUST be called BEFORE model instantiation to take effect.
    It patches the MLP classes in transformers library to use TiledMLP for
    memory-efficient computation during training.

    Args:
        num_shards: Number of shards to split the input into. Higher values
                   reduce peak memory but may slightly impact performance.
        model_type: The model type string (e.g., "llama", "qwen2", "qwen3").
                   If None, patches all supported model types.

    Returns:
        List of patched class names.
    """
    if model_type is None:
        types_to_patch = list(_MODEL_TYPE_TO_MLP_CLASS.keys())
    elif model_type in _MODEL_TYPE_TO_MLP_CLASS:
        types_to_patch = [model_type]
    else:
        raise ValueError(
            f"TiledMLP does not support model_type='{model_type}'. "
            f"Supported types: {list(_MODEL_TYPE_TO_MLP_CLASS.keys())}. "
            f"For SwiGLU-style MLPs, you can add support by extending _MODEL_TYPE_TO_MLP_CLASS "
            f"in verl/models/transformers/tiled_mlp.py"
        )

    patched_classes = []

    for mtype in types_to_patch:
        module_path, class_name = _MODEL_TYPE_TO_MLP_CLASS[mtype]
        try:
            import importlib

            module = importlib.import_module(module_path)
            mlp_class = getattr(module, class_name)
            _patch_mlp_class(mlp_class, _mlp_forward_fn, num_shards)
            if class_name not in patched_classes:
                patched_classes.append(class_name)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not patch {mtype} MLP: {e}")

    if patched_classes:
        print(f"TiledMLP monkey patch applied to: {', '.join(patched_classes)} (shards={num_shards})")

    return patched_classes


def _patch_mlp_class(mlp_class: type[nn.Module], forward_fn, num_shards: int):
    # [FUNCTION DOC]
    # HELPER: Behavior: replaces mlp_class.forward with a function that collects the module's requires-grad params and calls
    # TiledMLP.apply(forward_fn, self, x, num_shards, params).
    # Inputs: mlp_class - class to patch; forward_fn - callable(module, x); num_shards - chunk count.
    # Side effects: mutates the class.
    """Patch a single MLP class to use TiledMLP."""

    def tiled_forward(self, x):
        # [FUNCTION DOC]
        # HELPER: the patched forward; see _patch_mlp_class.
        compute_params = [p for p in self.parameters() if p.requires_grad]
        return TiledMLP.apply(forward_fn, self, x, num_shards, compute_params)

    mlp_class.forward = tiled_forward
