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
# Fake-quantizing linear layer for NVFP4 QAT. Provides a Triton kernel that emulates NVFP4 quantization
# (FP4 E2M1 values with per-16-element FP8 E4M3 block scales and a per-tensor FP32 global scale) and
# dequantizes back to the input dtype, a straight-through-estimator autograd wrapper, the QATMode enum,
# and QATLinear, an nn.Linear subclass that fake-quantizes weights (and activations in W4A4 mode) in
# forward. Constants: FP4 max 6.0, FP8 E4M3 max 448.0.
"""QAT FakeQuantized Linear module for NVFP4 (W4A4/W4A16) with FSDP compatibility.

Includes Triton kernels for high-performance FP4 quantization.
"""

from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QATLinear", "QATMode"]


import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}
FP4_E2M1_MAX: float = 6.0
FP8_E4M3_MAX: float = 448.0


@triton.jit
def _fp4_fake_quant_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    global_scale_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    NUM_FP4_BLOCKS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # [FUNCTION DOC]
    # HELPER: Triton kernel; each program handles a TILE_M x TILE_N tile of a 2-D tensor.
    # Behavior: loads the tile (zero-padded at boundaries) as fp32 and reshapes into NUM_FP4_BLOCKS blocks of
    # BLOCK_SIZE per row. Per block: amax / (FP4_MAX * global_scale) (global scale guarded to >= 1e-12),
    # clipped to FP8_MAX, rounded through float8 e4m3, multiplied back by global_scale to get the effective
    # block scale (replaced with 1.0 if below 1e-5). Absolute values divided by that scale are rounded to the
    # nearest FP4 E2M1 magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6} using midpoint thresholds (ties at 0.25/1.25/
    # 2.5/5.0 round down, at 0.75/1.75/3.5 round up), rescaled, sign restored, cast to OUT_DTYPE and stored
    # with boundary checks.
    # Inputs: x/y pointers, dims M, N, pointer to global scale, strides, compile-time tile/block constants.
    # Outputs: writes y. Side effects: device memory writes.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    row_start = pid_m * TILE_M
    col_start = pid_n * TILE_N

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        strides=(stride_xm, stride_xn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        strides=(stride_ym, stride_yn),
        offsets=(row_start, col_start),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0),
    )

    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    global_scale_safe = tl.where(global_scale > 0.0, global_scale, 1e-12)

    tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    tile_reshaped = tl.reshape(tile, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    x_abs = tl.abs(tile_reshaped)

    block_max = tl.max(x_abs, axis=2, keep_dims=True)
    block_max_scaled = block_max / (FP4_MAX * global_scale_safe)
    block_max_scaled = tl.minimum(block_max_scaled, FP8_MAX)
    block_max_quant = block_max_scaled.to(tl.float8e4nv).to(tl.float32) * global_scale
    block_max_quant = tl.where(block_max_quant >= 1e-5, block_max_quant, 1.0)

    block_max_quant_broadcast = tl.broadcast_to(block_max_quant, (TILE_M, NUM_FP4_BLOCKS, BLOCK_SIZE))
    abs_scaled = x_abs / block_max_quant_broadcast

    q_val = tl.where(
        abs_scaled <= 0.25,
        0.0,
        tl.where(
            abs_scaled < 0.75,
            0.5,
            tl.where(
                abs_scaled <= 1.25,
                1.0,
                tl.where(
                    abs_scaled < 1.75,
                    1.5,
                    tl.where(
                        abs_scaled <= 2.5,
                        2.0,
                        tl.where(abs_scaled < 3.5, 3.0, tl.where(abs_scaled <= 5.0, 4.0, FP4_MAX)),
                    ),
                ),
            ),
        ),
    )

    x_rescaled = q_val * block_max_quant_broadcast
    x_rescaled = tl.where(tile_reshaped >= 0, x_rescaled, -x_rescaled)
    tile_quant = tl.reshape(x_rescaled, (TILE_M, TILE_N))

    tl.store(y_block_ptr, tile_quant.to(OUT_DTYPE), boundary_check=(0, 1))


def fp4_fake_quant_weight(
    weight: torch.Tensor,
    global_amax: torch.Tensor = None,
    block_size: int = 16,
    tile_rows: int = 16,
    tile_cols: int = 64,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: flattens weight to 2-D (last dim as columns), allocates output, aligns tile width up to a
    # multiple of block_size (at least block_size), computes global_amax (tensor abs-max if not given) and
    # global_scale = amax / (6 * 448), launches the kernel over a 2-D grid, and reshapes to the original shape.
    # Inputs: weight - fp32/fp16/bf16 CUDA tensor; global_amax - optional scalar tensor; block_size; tile sizes.
    # Outputs: fake-quantized tensor of same shape and dtype.
    # Exceptions: KeyError for unsupported dtypes; Triton errors for non-CUDA tensors.
    # Side effects: GPU kernel launch.
    """Apply FP4 fake quantization using Triton kernel."""
    x_shape = weight.shape
    x_dtype = weight.dtype
    x = weight.reshape(-1, x_shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty_like(x)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    tile_cols = max(tile_cols, block_size)
    tile_cols_aligned = ((tile_cols + block_size - 1) // block_size) * block_size
    num_fp4_blocks = tile_cols_aligned // block_size

    if global_amax is None:
        global_amax = weight.abs().max().to(torch.float32)
    global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

    grid = (triton.cdiv(M, tile_rows), triton.cdiv(N, tile_cols_aligned))

    _fp4_fake_quant_kernel[grid](
        x,
        y,
        M,
        N,
        global_scale,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_SIZE=block_size,
        TILE_M=tile_rows,
        TILE_N=tile_cols_aligned,
        NUM_FP4_BLOCKS=num_fp4_blocks,
        OUT_DTYPE=_TORCH_TO_TL_DTYPE[x_dtype],
        FP4_MAX=FP4_E2M1_MAX,
        FP8_MAX=FP8_E4M3_MAX,
    )
    return y.view(*x_shape)


class STEFP4QuantTriton(torch.autograd.Function):
    # [CLASS DOC]
    # Autograd function: forward applies fp4_fake_quant_weight; backward passes the gradient straight through
    # to x (no gradient for global_amax or block_size).
    """Straight-Through Estimator wrapper for Triton FP4 quantization kernel."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, global_amax: torch.Tensor, block_size: int) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: returns fp4_fake_quant_weight(x, global_amax, block_size). Side effects: kernel launch.
        return fp4_fake_quant_weight(x, global_amax=global_amax, block_size=block_size)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        # [FUNCTION DOC]
        # Outputs: (grad_output, None, None).
        return grad_output, None, None


class QATMode(str, Enum):
    # [CLASS DOC]
    # String enum of QAT modes: W4A4 ("w4a4", weights and activations) and W4A16 ("w4a16", weights only).
    """QAT quantization mode."""

    W4A4 = "w4a4"  # Weight 4-bit, Activation 4-bit (dynamic)
    W4A16 = "w4a16"  # Weight 4-bit, Activation 16-bit (weight only)


class QATLinear(nn.Linear):
    # [CLASS DOC]
    # nn.Linear that fake-quantizes in forward. Caches the weight amax (shared across fusion siblings),
    # weight global scale and blockwise scale (cleared by invalidate_all_scales). In W4A4 mode holds
    # persistent buffers input_global_scale and input_amax (initialized to the sentinel -1.0) updated by the
    # configured activation observer during training.
    """QAT FakeQuantized Linear layer with FSDP compatibility."""

    _UNINITIALIZED_SCALE = -1.0

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",  # Observer strategy for activation global_scale
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # [FUNCTION DOC]
        # Behavior: initializes the base Linear, stores mode/group_size/observer, initializes caches and sibling
        # refs to None, registers W4A4 buffers and EMA decay 0.01 when in W4A4 mode, and enables fake quant.
        # Side effects: parameter/buffer allocation.
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.mode = mode
        self.group_size = group_size
        self.activation_observer = activation_observer

        self._weight_blockwise_scale: Optional[torch.Tensor] = None
        self._weight_global_scale: Optional[torch.Tensor] = None
        self._cached_weight_amax: Optional[torch.Tensor] = None
        self._fusion_siblings_ref = None

        if mode == QATMode.W4A4:
            self.register_buffer(
                "input_global_scale", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self.register_buffer(
                "input_amax", torch.tensor([self._UNINITIALIZED_SCALE], dtype=torch.float32), persistent=True
            )

            self._ema_decay: float = 0.01

        self.fake_quant_enabled = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mode: QATMode = QATMode.W4A4,
        group_size: int = 16,
        activation_observer: str = "static_minmax",
    ) -> "QATLinear":
        # [FUNCTION DOC]
        # Behavior: creates a QATLinear with the same shape, bias presence, device and dtype; for non-meta
        # weights, copies (clones) weight and bias into new Parameters (meta weights stay uninitialized).
        # Outputs: new QATLinear. Side effects: memory allocation.
        """Create QATLinear from an existing nn.Linear."""
        has_bias = linear.bias is not None

        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=has_bias,
            mode=mode,
            group_size=group_size,
            activation_observer=activation_observer,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        if linear.weight.device != torch.device("meta"):
            new_linear.weight = nn.Parameter(linear.weight.clone())
            if has_bias:
                new_linear.bias = nn.Parameter(linear.bias.clone())

        return new_linear

    def _is_amax_initialized(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the input_amax buffer exists and differs from the -1.0 sentinel (device sync via item()).
        """Check if input_amax has been initialized."""
        if not hasattr(self, "input_amax"):
            return False
        return self.input_amax.item() != self._UNINITIALIZED_SCALE

    def _update_input_global_scale(self, x: torch.Tensor):
        # [FUNCTION DOC]
        # HELPER: Updates the W4A4 static input scale from a batch.
        # Behavior: computes abs-max of x (fp32, detached), MAX all-reduced across ranks when distributed.
        # "memoryless_minmax": scale = 2688 / (amax + 1e-12) from the current batch only. "static_minmax":
        # running max of amax. "minmax": EMA of amax with decay 0.01 (first batch initializes). For the latter two
        # scale = 2688 / (stored amax + 1e-12). Writes into input_global_scale (and input_amax).
        # Exceptions: AssertionError if not W4A4; ValueError for unknown observer.
        # Side effects: buffer mutation; collective communication.
        """Update static input_global_scale based on observer strategy."""
        assert self.mode == QATMode.W4A4, "_update_input_global_scale should only be called in W4A4 mode"

        current_amax = torch.amax(torch.abs(x)).detach().to(torch.float32)

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(current_amax, op=torch.distributed.ReduceOp.MAX)

        scale_factor = FP8_E4M3_MAX * FP4_E2M1_MAX

        if self.activation_observer == "memoryless_minmax":
            new_scale = (scale_factor / (current_amax + 1e-12)).view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "static_minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = torch.maximum(self.input_amax, current_amax.view(1).to(self.input_amax.device))
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        elif self.activation_observer == "minmax":
            if not self._is_amax_initialized():
                self.input_amax.copy_(current_amax.view(1).to(self.input_amax.device))
            else:
                new_amax = (1 - self._ema_decay) * self.input_amax + self._ema_decay * current_amax.view(1).to(
                    self.input_amax.device
                )
                self.input_amax.copy_(new_amax)
            amax_f32 = self.input_amax.to(torch.float32)
            new_scale = (scale_factor / (amax_f32 + 1e-12)).float().view(1)
            self.input_global_scale.copy_(new_scale.to(self.input_global_scale.device))

        else:
            raise ValueError(f"Unknown activation_observer: {self.activation_observer}")

    def _fake_quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Fake-quantizes the weight.
        # Behavior (no-grad for scale computation): uses cached amax if present; otherwise, with fusion siblings
        # (live, non-meta), reuses the first sibling's cached amax, or computes the max of all members' abs-max,
        # caching it on self and all siblings; without siblings uses this weight's abs-max. Sets the cached
        # global scale if absent. Applies the STE quantization with group_size.
        # Outputs: fake-quantized weight (gradient flows straight through).
        # Side effects: mutates caches on self and siblings.
        """Apply fake quantization to weight tensor using Triton kernel."""
        with torch.no_grad():
            if self._cached_weight_amax is not None:
                global_amax = self._cached_weight_amax
            else:
                siblings_ref = getattr(self, "_fusion_siblings_ref", None)

                if siblings_ref is not None:
                    siblings = [ref() for ref in siblings_ref if ref() is not None]
                    siblings = [s for s in siblings if s.weight.device != torch.device("meta")]

                    for sibling in siblings:
                        sibling_amax = getattr(sibling, "_cached_weight_amax", None)
                        if sibling_amax is not None:
                            global_amax = sibling_amax
                            self._cached_weight_amax = global_amax
                            break
                    else:
                        all_modules = [self] + siblings
                        amaxes = [m.weight.abs().max().to(torch.float32) for m in all_modules]
                        global_amax = torch.max(torch.stack(amaxes))

                        self._cached_weight_amax = global_amax
                        for sibling in siblings:
                            sibling._cached_weight_amax = global_amax
                else:
                    global_amax = weight.abs().max().to(torch.float32)
                    self._cached_weight_amax = global_amax

            if self._weight_global_scale is None:
                self._weight_global_scale = global_amax.float() / (FP4_E2M1_MAX * FP8_E4M3_MAX)

        result = STEFP4QuantTriton.apply(weight, global_amax, self.group_size)

        return result

    def _fake_quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # HELPER: Fake-quantizes activations in W4A4 mode.
        # Behavior: flattens 3-D input to 2-D; in training updates the input scale; raises if the scale is still
        # the sentinel; converts scale to amax = 2688 / scale and applies the STE kernel; restores shape.
        # Exceptions: RuntimeError when uninitialized. Side effects: buffer updates in training; device sync.
        """Apply fake quantization to activation tensor (W4A4 mode only)."""
        original_shape = x.shape

        if x.dim() == 3:
            x_2d = x.view(-1, x.shape[-1])
        else:
            x_2d = x

        if self.training:
            self._update_input_global_scale(x_2d)

        if self.input_global_scale.item() == self._UNINITIALIZED_SCALE:
            raise RuntimeError("W4A4 input_global_scale uninitialized. Load PTQ model first.")

        global_amax = (FP4_E2M1_MAX * FP8_E4M3_MAX) / self.input_global_scale.to(x.device)
        result = STEFP4QuantTriton.apply(x_2d, global_amax, self.group_size)
        return result.view(original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Behavior: plain linear if fake quant disabled; otherwise linear of (fake-quantized activation in W4A4,
        # else raw input) with fake-quantized weight and bias.
        # Outputs: output tensor.
        """Forward pass with fake quantization."""
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight, self.bias)

        weight_fq = self._fake_quantize_weight(self.weight)

        if self.mode == QATMode.W4A4:
            x_fq = self._fake_quantize_activation(x)
        else:
            x_fq = x

        return F.linear(x_fq, weight_fq, self.bias)

    def extra_repr(self) -> str:
        # [FUNCTION DOC]
        # Outputs: string with features, bias presence, mode, group_size and fake_quant_enabled.
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.value}, "
            f"group_size={self.group_size}, fake_quant_enabled={self.fake_quant_enabled}"
        )
