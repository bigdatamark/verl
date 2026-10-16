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
# Refit support for vLLM block-FP8 linear and MoE quant methods (see module docstring): preserves
# vLLM Parameter-subclass metadata across post-load parameter rebuilds, records each layer's
# checkpoint (pristine) FP8 layout at first post-processing, stages checkpoint-layout buffers before
# a reload, and folds re-derived kernel layouts back into the live (CUDA-graph-captured) storage.
# Provides version-specific replacement post-load hooks and builders for the mock patchers that
# install them. Layer attribute names used for stashing: pristine layout and live params.

"""Refit support for vLLM's ``Fp8LinearMethod`` and ``Fp8MoEMethod``.

Two problems have to be solved for an RL weight refit to land correctly on a
block-FP8 layer:

* vLLM's post-load hooks rebuild parameters as plain ``torch.nn.Parameter``,
  dropping the subclass metadata (``ModelWeightParameter`` and friends) that
  the weight loaders dispatch on. The patches here keep that metadata.
* Kernel post-processing rewrites weights and scales into an inference layout
  the checkpoint tensors no longer fit. The staging helpers hand
  ``load_weights`` a checkpoint-layout buffer and then fold the re-derived
  result back into the original storage, so pointers captured by a CUDA graph
  stay valid.

``verl/utils/vllm/vllm_quant_utils.py`` is the entry point that drives these.
"""

import inspect
import logging
import math
from unittest.mock import patch

import torch
from packaging import version

logger = logging.getLogger(__name__)


def _iter_transferable_param_attrs(src_param):
    # [FUNCTION DOC]
    # HELPER: Generator over (name, value) attributes of src_param worth copying: skips names present on
    # the base Parameter class, dunder names, attributes whose access raises AttributeError, and methods
    # bound to any tensor.
    """Yield the attributes worth moving onto a rebuilt parameter.

    Methods bound to a tensor are excluded, and that exclusion is the whole
    point of this helper. ``dir()`` reports vLLM's loader entry points
    (``load_merged_column_weight`` and friends) alongside the data attributes,
    and copying one lands a method whose ``self`` is some *other* tensor in the
    destination's instance dict, where it shadows the class. Every later load
    then writes into that stale tensor instead of the parameter it was handed --
    silently, because the shapes still agree.

    Note the test is "bound to a tensor", not "bound to the source": these
    methods are copied forward from rebuild to rebuild, so by the time one does
    damage its ``self`` is usually a temporary from several rebuilds back.
    ``weight_loader`` is bound to the layer, so it survives.
    """
    base_param_attrs = dir(torch.nn.Parameter)
    for attr in dir(src_param):
        if attr in base_param_attrs or attr.startswith("__"):
            continue
        try:
            value = getattr(src_param, attr)
        except AttributeError:
            continue
        if inspect.ismethod(value) and isinstance(value.__self__, torch.Tensor):
            continue
        yield attr, value


def _copy_param_subclass_attrs(dst_param, src_param):
    # [FUNCTION DOC]
    # HELPER: No-op if src_param is None. Copies transferable attributes onto dst_param (ignoring
    # AttributeError on set) and records subclass_type (the source's stored subclass_type, else its
    # type) on dst_param when that is not plain Parameter.
    # Side effects: mutates dst_param.
    """Carry vLLM's parameter metadata across a parameter rebuild.

    vLLM stores loader dispatch information (``output_dim``, ``weight_loader``,
    tp state, ...) as plain attributes on ``Parameter`` subclasses. Anything
    that re-wraps a tensor has to bring them along or the next refit will not
    know how to shard the incoming weight.
    """
    if src_param is None:
        return

    for attr, value in _iter_transferable_param_attrs(src_param):
        try:
            setattr(dst_param, attr, value)
        except AttributeError:
            pass

    subclass_type = getattr(src_param, "subclass_type", type(src_param))
    if subclass_type is not torch.nn.Parameter:
        dst_param.subclass_type = subclass_type


_FP8_PRISTINE_ATTR = "_verl_fp8_pristine"
_FP8_LIVE_ATTR = "_verl_fp8_live_params"

# ``Fp8LinearMethod`` owns the first group, ``Fp8MoEMethod`` the second. The MoE
# scale is named ``w13_weight_scale_inv`` under block quant and
# ``w13_weight_scale`` otherwise (``self.weight_scale_name``), so both spellings
# are listed; only the names a layer actually has get recorded.
_FP8_LINEAR_PARAM_NAMES = ("weight", "weight_scale_inv", "weight_scale")
_FP8_MOE_PARAM_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_weight_scale",
    "w2_weight_scale",
)
_FP8_REFIT_PARAM_NAMES = (*_FP8_LINEAR_PARAM_NAMES, *_FP8_MOE_PARAM_NAMES)


def _fold_into_live_param(layer, param_name, param, new_data):
    # [FUNCTION DOC]
    # HELPER: Unwraps a Parameter to its data, requires shape and dtype equal to the live param, copies
    # the data into the live storage unless it already is the same memory, and sets the live param back
    # on the layer.
    # Exceptions: RuntimeError on shape/dtype mismatch. Side effects: in-place copy; layer attribute set.
    """Move post-processing output into the storage the CUDA graph captured."""
    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    if new_data.shape != param.shape or new_data.dtype != param.dtype:
        raise RuntimeError(
            f"FP8 refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype}, "
            f"but the live parameter is {tuple(param.shape)}/{param.dtype}; "
            "its storage cannot be updated in place."
        )
    if new_data.data_ptr() != param.data_ptr():
        param.data.copy_(new_data)
    setattr(layer, param_name, param)


def replace_parameter_preserve_subclass(
    layer: torch.nn.Module, param_name: str, new_data: torch.Tensor | None, prefer_copy: bool = False
):
    # [FUNCTION DOC]
    # Behavior: if the layer has a stashed live param for param_name and new_data is not None, folds the
    # data into it and removes it from the stash. Else if new_data is None, sets the attribute to None.
    # Otherwise wraps new_data (unwrapped if a Parameter) in a fresh non-trainable Parameter, copies
    # metadata from the previous attribute, and sets it.
    # Inputs: layer; param_name; new_data - tensor, Parameter or None; prefer_copy - ignored.
    # Outputs: None. Exceptions: RuntimeError from folding. Side effects: mutates layer.
    """Stand-in for vLLM's ``replace_parameter`` inside the fp8 quant module.

    During a refit of a staged layer this folds the result into the live
    parameter and reinstates it immediately, rather than after
    ``process_weights_after_loading`` returns. That timing is required for MoE:
    the tail of ``Fp8MoEMethod._setup_kernel`` rebuilds ``moe_quant_config`` and
    ``moe_kernel`` from whatever is on the layer at that moment, and those cache
    tensor references, so a later swap would leave the kernel pointing at the
    staging buffers.
    """
    live = getattr(layer, _FP8_LIVE_ATTR, None) or {}
    param = live.get(param_name)
    if param is not None and new_data is not None:
        _fold_into_live_param(layer, param_name, param, new_data)
        del live[param_name]
        return

    if new_data is None:
        setattr(layer, param_name, None)
        return

    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    old_param = getattr(layer, param_name, None)
    param = torch.nn.Parameter(new_data, requires_grad=False)
    _copy_param_subclass_attrs(param, old_param)
    setattr(layer, param_name, param)


def _restore_layer_param_subclass_attrs(layer: torch.nn.Module, old_params: dict[str, torch.nn.Parameter]):
    # [FUNCTION DOC]
    # HELPER: For each name in old_params whose current attribute on the layer is a Parameter, copies
    # the old param's metadata onto it.
    for name, old_param in old_params.items():
        new_param = getattr(layer, name, None)
        if isinstance(new_param, torch.nn.Parameter):
            _copy_param_subclass_attrs(new_param, old_param)


def _record_pristine_fp8_layout(layer, params):
    # [FUNCTION DOC]
    # HELPER: If the layer has no pristine record yet, records (shape, dtype) for each known FP8 refit
    # param name found as a Parameter in params, and stores the mapping on the layer if non-empty.
    # Side effects: sets a layer attribute.
    """Remember the checkpoint layout of a layer's FP8 weights and scales.

    Kernel post-processing rewrites these into inference layouts that the
    checkpoint tensors no longer fit. DeepGEMM, for instance, packs a
    ``(N/128, K/128)`` block scale into an ``(N, K/512)`` int32 UE8M0 tensor,
    so a refit feeding checkpoint-layout scales hits a shape assert. Keep
    enough state to rebuild the original buffers before each reload.

    Only the layout is recorded, never a tensor. This runs during model load,
    inside the memory pool vLLM tags as ``weights``, so any buffer kept here
    would be subject to that pool's discard-and-rezero across a sleep and would
    not survive to the next refit anyway.
    """
    if hasattr(layer, _FP8_PRISTINE_ATTR):
        return

    pristine = {}
    for name in _FP8_REFIT_PARAM_NAMES:
        param = params.get(name)
        if not isinstance(param, torch.nn.Parameter):
            continue
        pristine[name] = (tuple(param.shape), param.dtype)

    if pristine:
        setattr(layer, _FP8_PRISTINE_ATTR, pristine)


def _layer_needs_fp8_staging(layer, pristine) -> bool:
    # [FUNCTION DOC]
    # HELPER: True if any recorded param that is still a Parameter has a different shape or dtype than
    # recorded, or is flagged is_shuffled (ROCm AITER); else False.
    for name, (shape, dtype) in pristine.items():
        param = getattr(layer, name, None)
        if not isinstance(param, torch.nn.Parameter):
            continue
        if tuple(param.shape) != shape or param.dtype != dtype:
            return True
        # ROCm's AITER MoE backend permutes the expert weights into its MFMA
        # layout while leaving shape and dtype untouched, so the comparison
        # above cannot see it. ``is_shuffled``, which that backend sets on the
        # repacked parameter, is the only signal that the live buffer is no
        # longer in checkpoint layout.
        if getattr(param, "is_shuffled", False):
            return True
    return False


def _fp8_staging_data(param, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: If dtype matches and element counts match, returns a reshape of the live data (shared
    # storage); otherwise allocates a new buffer filled with all-ones bytes (NaN sentinel).
    # Inputs: param; shape; dtype. Outputs: tensor.
    """Checkpoint-layout tensor that ``load_weights`` can write into.

    Reuses the live storage whenever the kernel transform only reshaped it, so
    those writes land straight in the tensor the CUDA graph points at.
    Otherwise the buffer is allocated fresh for this refit and dropped with it,
    which keeps nothing alive across the engine's sleep.

    An all-ones fill is NaN in every dtype staged here, so a buffer the incoming
    stream fails to write blows up on the first forward instead of being repacked
    into the live weights. Uninitialized memory would carry whatever the caching
    allocator last held, which for a scale is most likely a plausible-looking
    value from a previous refit.
    """
    if param.dtype == dtype and param.numel() == math.prod(shape):
        return param.data.reshape(shape)
    staging = torch.empty(shape, dtype=dtype, device=param.device)
    staging.view(torch.uint8).fill_(0xFF)
    return staging


def stage_fp8_params_for_loading(model):
    # [FUNCTION DOC]
    # Behavior: for each module with a pristine record that needs staging, replaces each recorded param
    # with a non-trainable staging Parameter in checkpoint layout (metadata copied), stashing the live
    # params on the layer. Logs the number of staged layers.
    # Inputs: model. Outputs: list of staged layers. Side effects: mutates layers; logs.
    """Expose checkpoint-layout buffers so ``load_weights`` can write into them.

    Covers both quant methods: ``Fp8LinearMethod`` layers keyed on ``weight`` /
    ``weight_scale*`` and ``Fp8MoEMethod`` expert modules keyed on ``w13_*`` /
    ``w2_*``. The live parameters are set aside and reinstated by
    ``process_fp8_weights_after_loading``, which puts the re-derived inference
    layout back into their original storage. No storage the CUDA graph captured
    is reallocated -- only tensor contents change -- and no forward pass runs
    between the two calls.
    """
    staged_layers = []
    for layer in model.modules():
        pristine = getattr(layer, _FP8_PRISTINE_ATTR, None)
        # A layer left in checkpoint layout (its kernel did not repack, e.g. a
        # Triton or vLLM-CUTLASS MoE backend) already loads as-is.
        if not pristine or not _layer_needs_fp8_staging(layer, pristine):
            continue

        live = {}
        for name, (shape, dtype) in pristine.items():
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            live[name] = param
            staged = torch.nn.Parameter(_fp8_staging_data(param, shape, dtype), requires_grad=False)
            _copy_param_subclass_attrs(staged, param)
            setattr(layer, name, staged)

        setattr(layer, _FP8_LIVE_ATTR, live)
        staged_layers.append(layer)

    # A zero here means no layer ever recorded a pristine layout, so the
    # post-processing hook never ran and the refit is unprotected.
    logger.info("Staged %d FP8 layers for refit", len(staged_layers))
    return staged_layers


def process_fp8_weights_after_loading(layers):
    # [FUNCTION DOC]
    # Behavior: for each staged layer, calls its quant_method.process_weights_after_loading if present
    # (which, under the patches, folds params via the subclass-preserving replace), then folds any
    # remaining stashed live params from the layer's current attribute, and deletes the stash.
    # Inputs: layers. Outputs: None. Exceptions: RuntimeError from folding (e.g. shape mismatch, or
    # attribute is None). Side effects: in-place parameter updates.
    """Re-derive the inference layout in place and reinstate the live params."""
    for layer in layers:
        live = getattr(layer, _FP8_LIVE_ATTR, None) or {}

        quant_method = getattr(layer, "quant_method", None)
        process = getattr(quant_method, "process_weights_after_loading", None)
        if process is not None:
            process(layer)

        # Anything routed through the patched ``replace_parameter`` has already
        # been folded and dropped from ``live``; what remains was rewritten by a
        # kernel that imported ``replace_parameter`` into its own module, out of
        # reach of the patch. The block-scaled linear kernels all do that.
        for name, param in list(live.items()):
            _fold_into_live_param(layer, name, param, getattr(layer, name))

        if hasattr(layer, _FP8_LIVE_ATTR):
            delattr(layer, _FP8_LIVE_ATTR)


def _make_process_weights_after_loading_for_vllm20(original_fn):
    # [FUNCTION DOC]
    # HELPER: Builds a wrapper for vLLM 0.20+ post-load hooks.
    # Inputs: original_fn - original unbound process_weights_after_loading. Outputs: wrapped method.
    def _patched_process_weights_after_loading(self, layer) -> None:
        # [FUNCTION DOC]
        # HELPER: Snapshots the layer's direct parameters, records the pristine layout, runs the original
        # hook with the fp8 module's replace_parameter patched to the subclass-preserving version, then
        # restores metadata onto rebuilt params. Side effects: mutates layer.
        old_params = dict(layer.named_parameters(recurse=False))
        _record_pristine_fp8_layout(layer, old_params)
        with patch(
            "vllm.model_executor.layers.quantization.fp8.replace_parameter", replace_parameter_preserve_subclass
        ):
            original_fn(self, layer)
        _restore_layer_param_subclass_attrs(layer, old_params)

    return _patched_process_weights_after_loading


def process_weights_after_loading_for_vllm14(self, layer) -> None:
    # [FUNCTION DOC]
    # Behavior: replacement Fp8LinearMethod post-load hook for older vLLM. Requires block quant with an
    # FP8-serialized checkpoint and dynamic activation. Runs vLLM's block-strategy processing on weight
    # and inverse scale, then rebuilds weight as a ModelWeightParameter and weight_scale_inv as a
    # BlockQuantScaleParameter (output dim 0, input dim 1, original weight loaders), each converted to a
    # plain Parameter with subclass metadata copied. Ensures input_scale exists (None) and runs vLLM's
    # optional block post-processing.
    # Inputs: self - quant method; layer. Outputs: None.
    # Exceptions: AssertionError on unsupported configs. Side effects: mutates layer.
    # KNOWN ISSUE: docstring says vLLM v0.18-v0.19 while the name says v14; it is used for every vLLM
    # below 0.20.
    """This function is used to process the weights after loading for a Linear layer, it is used for vllm v0.18-v0.19.

    Compared to the original process_weights_after_loading in vllm, we just avoid creation of
    new torch.nn.Parameter objects, because that removes the weight_loader attribute which we need for refit.
    """
    from torch.nn import Parameter
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        maybe_post_process_fp8_weight_block,
        process_fp8_weight_block_strategy,
    )
    from vllm.model_executor.parameter import (
        BlockQuantScaleParameter,
        ModelWeightParameter,
    )

    assert self.block_quant and self.quant_config.is_checkpoint_fp8_serialized
    assert self.quant_config.activation_scheme == "dynamic"

    def _create_param_from_subclass_attributes(custom_param):
        # [FUNCTION DOC]
        # HELPER: Wraps the custom parameter's data in a non-trainable Parameter carrying its metadata.
        param = Parameter(custom_param.data, requires_grad=False)
        _copy_param_subclass_attrs(param, custom_param)
        return param

    weight, weight_scale_inv = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)

    layer.weight = _create_param_from_subclass_attributes(
        ModelWeightParameter(
            data=weight.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight.weight_loader,
        )
    )
    layer.weight_scale_inv = _create_param_from_subclass_attributes(
        BlockQuantScaleParameter(
            data=weight_scale_inv.data,
            output_dim=0,
            input_dim=1,
            weight_loader=layer.weight_scale_inv.weight_loader,
        )
    )

    # vLLM v0.17 removed the `else: register_parameter("input_scale", None)` from
    # create_weights() for dynamic activation, but apply() still accesses layer.input_scale.
    # Since block_quant always uses dynamic activation, ensure the attribute exists.
    if not hasattr(layer, "input_scale"):
        layer.input_scale = None

    maybe_post_process_fp8_weight_block(layer)


def process_weights_after_loading_moe_for_vllm14(self, layer) -> None:
    # [FUNCTION DOC]
    # Behavior: replacement Fp8MoEMethod post-load hook for vLLM below 0.20 (without the reentrancy
    # guard so it can run on every refit). Converts w13/w2 weights and scales (scale name from
    # weight_scale_name) to kernel format, then rebuilds w13_weight, w2_weight, w13_weight_scale_inv and
    # w2_weight_scale_inv as plain Parameters with transferable attributes copied. Rebuilds the MoE quant
    # config and, if set, the MoE kernel, using the newer factory signature (with routing tables and
    # shared experts) when available, else the older one returning (kernel, use_inplace).
    # Inputs: self - quant method; layer. Outputs: None.
    # Exceptions: AssertionError if experts_cls is None when a quant config exists.
    # Side effects: mutates layer and quant method.
    # KNOWN ISSUE: scales are always written to the *_weight_scale_inv names and read from them as the
    # copy source, so a non-block layer (weight_scale_name "weight_scale") would fail on attribute lookup.
    # removed the reentrancy guard here for refit
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        convert_to_fp8_moe_kernel_format,
        make_fp8_moe_kernel,
    )

    # Allow for accessing weights and scales in standard way.
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
    w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
    w13_input_scale = layer.w13_input_scale
    w2_input_scale = layer.w2_input_scale

    # Shuffle weights to runtime format and setup kernel.
    w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
        fp8_backend=self.fp8_backend,
        layer=layer,
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        w13_input_scale=w13_input_scale,
        w2_input_scale=w2_input_scale,
    )
    from torch.nn import Parameter

    def _create_param_from_subclass_attributes(custom_data, custom_weight):
        # [FUNCTION DOC]
        # HELPER: Non-trainable Parameter from custom_data with transferable attributes of custom_weight set
        # on it (without AttributeError protection).
        param = Parameter(custom_data, requires_grad=False)
        for attr, value in _iter_transferable_param_attrs(custom_weight):
            setattr(param, attr, value)
        return param

    # Replace parameters with updated versions. Note that this helper
    # function ensures the replacement is compatible with RL weight reloads.
    layer.w13_weight = _create_param_from_subclass_attributes(w13, layer.w13_weight)
    layer.w2_weight = _create_param_from_subclass_attributes(w2, layer.w2_weight)
    layer.w13_weight_scale_inv = _create_param_from_subclass_attributes(w13_scale, layer.w13_weight_scale_inv)
    layer.w2_weight_scale_inv = _create_param_from_subclass_attributes(w2_scale, layer.w2_weight_scale_inv)

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    if self.moe_quant_config:
        assert self.experts_cls is not None

        # Check for the new API by inspecting the function signature, which is more
        # robust than version string comparison, especially for dev/pre-release versions.
        sig = inspect.signature(make_fp8_moe_kernel)
        if "routing_tables" in sig.parameters:
            # vLLM >= 0.16+: routing_tables/shared_experts added, returns kernel directly
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )
        else:
            # vLLM 0.14/0.15: routing_tables/shared_experts not supported, returns (kernel, use_inplace)
            self.kernel, self.use_inplace = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
            )


def build_fp8_method_patchers(vllm_version):
    # [FUNCTION DOC]
    # Behavior: returns unstarted mock patchers for Fp8LinearMethod and Fp8MoEMethod
    # process_weights_after_loading: on vLLM 0.20 or newer, wrappers around the original methods;
    # otherwise the replacement functions in this module.
    # Inputs: vllm_version - parsed packaging Version. Outputs: list of two patchers.
    """Patchers that make the FP8 quant methods survive a refit, not yet started.

    The caller owns starting and tracking them so all patch lifetime lives in
    one place.
    """
    linear_path = "vllm.model_executor.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading"
    moe_path = "vllm.model_executor.layers.quantization.fp8.Fp8MoEMethod.process_weights_after_loading"

    # vLLM 0.20 refactored FP8 post-load handling and removed
    # maybe_post_process_fp8_weight_block. Keep its native transformation logic,
    # but preserve parameter subclass metadata for RL weight reloads.
    if vllm_version >= version.parse("0.20.0"):
        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8LinearMethod,
            Fp8MoEMethod,
        )

        wrap = _make_process_weights_after_loading_for_vllm20
        return [
            patch(linear_path, wrap(Fp8LinearMethod.process_weights_after_loading)),
            patch(moe_path, wrap(Fp8MoEMethod.process_weights_after_loading)),
        ]

    return [
        patch(linear_path, process_weights_after_loading_for_vllm14),
        patch(moe_path, process_weights_after_loading_moe_for_vllm14),
    ]
