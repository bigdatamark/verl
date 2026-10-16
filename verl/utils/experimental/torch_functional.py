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
# Memory-efficient fused "LM head + log-prob + entropy" computation for PPO. Computes, per token, the log probability of a target id and the entropy of the temperature-scaled softmax without materializing full-vocab logits for the whole sequence, by processing tokens in chunks with a hand-written backward. Optionally uses Liger's fused linear scaled cross entropy kernel (_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY, None if liger_kernel is not installed; other import errors propagate) and flash-attn's Triton cross entropy (_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE) for the forward log-prob.

from typing import Optional

import torch

try:
    from liger_kernel.ops import (
        LigerFusedLinearScaledCrossEntropyFunction as _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY,
    )
except ModuleNotFoundError as exc:
    if exc.name != "liger_kernel":
        raise
    _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY = None

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False


def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # [FUNCTION DOC]
    # HELPER: Forward for one chunk. logits = hidden @ W^T / temperature, upcast to fp32; entropy = logsumexp(logits) - sum(softmax * logits); token log prob = negative flash-attn cross entropy if available, else gather from log_softmax.
    # Inputs: hidden_states (T, D); vocab_weights (V, D); input_ids (T); temperature.
    # Outputs: (fp32 token log probs (T), entropy (T) in the original logits dtype).
    # Exceptions: AssertionError if log probs are not fp32.
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    if _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        per_token_entropy_loss = cross_entropy_loss(logits, input_ids)[0]
        token_log_probs = -per_token_entropy_loss
    else:
        # Fallback to original PyTorch implementation
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    assert token_log_probs.dtype == torch.float32
    return token_log_probs, entropy.to(orig_dtype)


def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    # [FUNCTION DOC]
    # HELPER: Backward for one chunk; recomputes logits/probs in fp32. Gradient w.r.t. logits accumulates dlog_probs * (one_hot(target) - probs) and, for entropy, -dentropy * probs * (log_probs + entropy). The result is cast to the original dtype, divided by temperature, then dhidden = dlogits @ W and dW = dlogits^T @ hidden.
    # Inputs: dlog_probs, dentropy - optional upstream grads (T); hidden_states, vocab_weights, input_ids, temperature.
    # Outputs: (dhidden_states (T, D), dvocab_weights (V, D)).
    # Note: if both upstream grads are None, dlogits remains the integer 0 and the cast fails; the caller asserts at least one is given.
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    probs = logits.softmax(dim=-1)

    dlogits = 0

    # Gradient from log_probs
    if dlog_probs is not None:
        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

    # Gradient from entropy
    if dentropy is not None:
        log_probs = logits.log_softmax(dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    dlogits = dlogits.to(orig_dtype) / temperature

    dhidden_states = dlogits @ vocab_weights
    dvocab_weights = dlogits.t() @ hidden_states

    return dhidden_states, dvocab_weights


class FusedLinearForPPOFunction(torch.autograd.Function):
    # [CLASS DOC]
    # Custom autograd function wrapping the chunked forward/backward; grads are not materialized for unused outputs (None allowed in backward).
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # [FUNCTION DOC]
        # Behavior: accepts 2-D (T, D) or 3-D (B, S, D) hidden states (3-D flattened to 2-D, input_ids flattened accordingly, preserving requires_grad). Allocates fp32 log_probs and entropy (hidden dtype) outputs, computes each chunk_size slice with the forward helper, reshapes outputs back to (B, S) for 3-D inputs, and saves flattened hidden states, weights and ids plus shape/temperature/chunk_size on ctx.
        # Inputs: ctx; hidden_states; vocab_weights (V, D); input_ids; temperature; chunk_size.
        # Outputs: (log_probs, entropy).
        # Exceptions: AssertionError for invalid ranks/mismatched input_ids.
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

        orig_batch_size = -1
        if orig_ndim == 3:
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            orig_batch_size = hidden_states.shape[0]
            hidden_states_requires_grad = hidden_states.requires_grad
            hidden_states = hidden_states.flatten(0, 1)
            hidden_states.requires_grad_(hidden_states_requires_grad)
            input_ids = input_ids.flatten(0, 1)

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        # Logits are upcasted to fp32 before computing log_probs, which are also fp32
        log_probs = torch.zeros(T, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad)
        entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)

        # Perform forward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )
            log_probs[chunk_start:chunk_end] = chunk_log_probs
            entropy[chunk_start:chunk_end] = chunk_entropy

        # Cast the output back to the original input dimension
        if orig_ndim == 3:
            log_probs = log_probs.view(orig_batch_size, -1)
            entropy = entropy.view(orig_batch_size, -1)

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        # [FUNCTION DOC]
        # Behavior: flattens upstream grads for 3-D inputs, allocates zero gradients only for inputs requiring grad, iterates chunks calling the backward helper, writing chunk dhidden slices and summing dW across chunks, and reshapes dhidden back to 3-D when needed.
        # Outputs: (dhidden_states or None, dvocab_weights or None, None, None, None).
        # Exceptions: AssertionError if both upstream grads are None.
        assert dlog_probs is not None or dentropy is not None

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        orig_batch_size = ctx.orig_batch_size
        orig_ndim = ctx.orig_ndim
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size

        # Here orig_ndim refers to the orig_ndim of hidden_states
        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)

        # Perform backward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_dlog_probs = None
            if dlog_probs is not None:
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            chunk_dentropy = None
            if dentropy is not None:
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
            )

            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h
            if vocab_weights.requires_grad:
                dvocab_weights += v

        # Cast the output back to the original input dimension
        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
        )


class FusedLinearForPPO(torch.nn.Module):
    # [CLASS DOC]
    # nn.Module front end selecting between the torch chunked implementation and the Liger kernel.
    def __init__(self, chunk_size: int = 512, impl_backend: str = "torch"):
        # [FUNCTION DOC]
        # Stores chunk_size and impl_backend.
        # Exceptions: ValueError if impl_backend is not "torch" or "liger".
        super().__init__()

        if impl_backend not in ("torch", "liger"):
            raise ValueError(f"Unsupported FusedLinearForPPO backend: {impl_backend}. Choose 'torch' or 'liger'.")
        self.chunk_size = chunk_size
        self.impl_backend = impl_backend

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        # [FUNCTION DOC]
        # Behavior: casts input_ids to int64. Uses FusedLinearForPPOFunction when backend is "torch" or Liger is unavailable (silent fallback). Otherwise validates shapes, flattens to 2-D/1-D, calls the Liger function (ignore_index -100, scale 1, return entropy True) and returns negated NLL as log probs and entropy reshaped to input_ids' shape.
        # Inputs: hidden_states (T, D) or (B, S, D); vocab_weights; input_ids; temperature.
        # Outputs: (log_probs, entropy) shaped like input_ids.
        # Exceptions: ValueError for bad ranks or shape mismatch in the Liger path.
        input_ids = input_ids.to(torch.int64)
        if self.impl_backend == "torch" or _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is None:
            return FusedLinearForPPOFunction.apply(
                hidden_states,
                vocab_weights,
                input_ids,
                temperature,
                self.chunk_size,
            )

        if hidden_states.ndim not in (2, 3):
            raise ValueError(f"hidden_states must be 2D or 3D, got shape {tuple(hidden_states.shape)}")
        if input_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match hidden_states shape "
                f"{tuple(hidden_states.shape[:-1])}"
            )

        output_shape = input_ids.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        input_ids = input_ids.reshape(-1)

        nll, entropy = _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            -100,
            1,
            True,
        )
        log_probs = -nll

        return log_probs.reshape(output_shape), entropy.reshape(output_shape)
