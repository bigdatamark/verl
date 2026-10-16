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
# Patches vLLM's ModelOpt NVFP4 quantization methods (dense linear, fused MoE, KV-cache scales) so that
# QAT-trained weights in ModelOpt HF format can be reloaded repeatedly at runtime using the FP4 Marlin
# backend. The approach mirrors verl.utils.qat.vllm_patch: on the first post-processing call, HF param
# metadata and weight loaders are saved and references to the Marlin-format tensors are kept; later calls
# convert newly loaded HF data and copy it into those references (stable addresses for CUDA graphs);
# helper functions rebuild HF-shaped params before a reload and re-run post-processing after it. The
# patches also force the Marlin backend at method construction time.
"""vLLM ModelOpt NVFP4 patches for dynamic weight updates (Marlin backend)."""

from typing import Optional

import torch
from torch.nn import Parameter

from verl.utils.device import get_device_name


def _save_param_meta(layer: torch.nn.Module, param_name: str):
    # [FUNCTION DOC]
    # HELPER: Ensures layer._hf_param_meta exists and, if the named param exists, records its shape, dtype,
    # device string, param class and optional _input_dim/_output_dim. Side effects: mutates layer.
    if not hasattr(layer, "_hf_param_meta"):
        layer._hf_param_meta = {}

    param = getattr(layer, param_name, None)
    if param is None:
        return

    meta = {
        "shape": tuple(param.shape),
        "dtype": param.dtype,
        "device": str(param.device),
        "param_class": type(param),
    }

    if hasattr(param, "_input_dim"):
        meta["input_dim"] = param._input_dim
    if hasattr(param, "_output_dim"):
        meta["output_dim"] = param._output_dim

    layer._hf_param_meta[param_name] = meta


def _create_param_from_meta(
    module: torch.nn.Module,
    param_name: str,
    meta: dict,
    device: Optional[torch.device] = None,
) -> Parameter:
    # [FUNCTION DOC]
    # HELPER: Allocates an empty tensor from metadata (device arg, else recorded device, else default device
    # name) and wraps it: a vLLM parameter subclass constructed with data, weight_loader and optional
    # input/output dims when the class is not plain Parameter and a loader was saved; otherwise a non-grad
    # Parameter with weight_loader attached if available.
    # Exceptions: errors from the param class constructor propagate (no fallback, unlike the qat variant).
    # Outputs: new Parameter.
    shape = meta["shape"]
    dtype = meta["dtype"]
    dev = device or meta.get("device", get_device_name())
    param_class = meta.get("param_class", Parameter)

    weight_loaders = getattr(module, "_weight_loaders", {})
    weight_loader = weight_loaders.get(param_name)

    data = torch.empty(shape, dtype=dtype, device=dev)

    if param_class is not Parameter and weight_loader is not None:
        kwargs = {"data": data, "weight_loader": weight_loader}
        if "input_dim" in meta:
            kwargs["input_dim"] = meta["input_dim"]
        if "output_dim" in meta:
            kwargs["output_dim"] = meta["output_dim"]
        new_param = param_class(**kwargs)
    else:
        new_param = Parameter(data, requires_grad=False)
        if weight_loader is not None:
            new_param.weight_loader = weight_loader

    return new_param


def _check_first_call(layer: torch.nn.Module) -> bool:
    # [FUNCTION DOC]
    # HELPER: Increments layer._process_weights_call_count; True if this was the first call.
    count = getattr(layer, "_process_weights_call_count", 0)
    layer._process_weights_call_count = count + 1
    return count == 0


def _save_weight_loaders(layer: torch.nn.Module, param_names: list[str]):
    # [FUNCTION DOC]
    # HELPER: Stores each named param's weight_loader (when present) into layer._weight_loaders.
    if not hasattr(layer, "_weight_loaders"):
        layer._weight_loaders = {}
    for pname in param_names:
        param = getattr(layer, pname, None)
        if param is not None and hasattr(param, "weight_loader"):
            layer._weight_loaders[pname] = param.weight_loader


def _update_ref_or_create(layer, ref_name, new_data):
    # [FUNCTION DOC]
    # HELPER: If a saved Marlin tensor ref exists for ref_name, copies new_data into it in place and sets the
    # layer attribute to a Parameter wrapping the ref; otherwise sets a Parameter of a clone (or tensor of a
    # scalar). Side effects: mutates layer.
    refs = getattr(layer, "_marlin_tensor_refs", {})
    ref = refs.get(ref_name)
    if ref is not None:
        ref.copy_(new_data)
        setattr(layer, ref_name, Parameter(ref, requires_grad=False))
    else:
        t = new_data.clone() if isinstance(new_data, torch.Tensor) else torch.tensor(new_data)
        setattr(layer, ref_name, Parameter(t, requires_grad=False))


def _unwrap_marlin_scale(value):
    # [FUNCTION DOC]
    # HELPER: Returns the first element if value is a tuple (newer vLLM returns (scales, factor)), else value.
    return value[0] if isinstance(value, tuple) else value


def _split_marlin_scale(value):
    # [FUNCTION DOC]
    # HELPER: Normalizes to a (scales, scale_factor) pair, using factor 1.0 when value is not a tuple.
    return value if isinstance(value, tuple) else (value, 1.0)


class ModelOptParamMetaDict(dict):
    # [CLASS DOC]
    # Name -> Parameter dict for a vLLM model with lazy rebuild of deleted HF-format params from saved
    # _hf_param_meta and a list of tensor-swap entries: every metadata param that also has a saved Marlin ref.
    """Dict-like parameter store with metadata-based rebuild and tensor swap."""

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        # [FUNCTION DOC]
        # Behavior: unwraps model.model if present, builds mappings, and fills with current named parameters.
        super().__init__()
        self.device = device

        actual_model = model
        if hasattr(model, "model"):
            actual_model = model.model
        self._model = actual_model

        self._layer_meta_cache: dict[str, dict] = {}
        self._tensor_swap_layers: dict[str, dict] = {}

        self._build_mappings()

        for name, param in actual_model.named_parameters():
            self[name] = param

    def _build_mappings(self):
        # [FUNCTION DOC]
        # HELPER: Caches module+meta for every module with _hf_param_meta and registers a swap entry keyed
        # "<layer>.<param>" for each metadata param present in _marlin_tensor_refs.
        for layer_name, module in self._model.named_modules():
            if not hasattr(module, "_hf_param_meta"):
                continue

            self._layer_meta_cache[layer_name] = {
                "module": module,
                "meta": module._hf_param_meta,
            }

            marlin_refs = getattr(module, "_marlin_tensor_refs", {})
            for param_name, meta in module._hf_param_meta.items():
                if param_name in marlin_refs:
                    key = f"{layer_name}.{param_name}" if layer_name else param_name
                    self._tensor_swap_layers[key] = {
                        "module": module,
                        "param_name": param_name,
                        "marlin_ref": marlin_refs[param_name],
                        "hf_meta": meta,
                    }

    def _try_rebuild(self, key: str) -> Optional[Parameter]:
        # [FUNCTION DOC]
        # HELPER: For "<layer>.<param>" keys with metadata, returns the existing non-None attribute or creates and
        # registers a new param from metadata; None otherwise. Side effects: may register a parameter.
        parts = key.rsplit(".", 1)
        if len(parts) != 2:
            return None
        layer_name, param_name = parts
        if layer_name not in self._layer_meta_cache:
            return None
        cache_entry = self._layer_meta_cache[layer_name]
        module = cache_entry["module"]
        meta = cache_entry["meta"]
        if param_name not in meta:
            return None
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            if param is not None:
                return param
        new_param = _create_param_from_meta(module, param_name, meta[param_name], self.device)
        module.register_parameter(param_name, new_param)
        return new_param

    def prepare_for_reload(self) -> None:
        # [FUNCTION DOC]
        # Behavior: for each swap entry whose module has the param, replaces it with a fresh HF-shape param.
        # Side effects: replaces module attributes (Marlin refs remain saved).
        """Replace kernel-format tensors with HF-shape tensors for reload."""
        for _key, swap_info in self._tensor_swap_layers.items():
            module = swap_info["module"]
            param_name = swap_info["param_name"]
            hf_meta = swap_info["hf_meta"]
            if hasattr(module, param_name):
                new_param = _create_param_from_meta(module, param_name, hf_meta, self.device)
                setattr(module, param_name, new_param)

    def __getitem__(self, key: str) -> Parameter:
        # [FUNCTION DOC]
        # Behavior: stored lookup, else rebuild-and-cache. Exceptions: KeyError if not found.
        if key in dict.keys(self):
            return super().__getitem__(key)
        param = self._try_rebuild(key)
        if param is not None:
            self[key] = param
            return param
        raise KeyError(f"Parameter not found: {key}")

    def __contains__(self, key: str) -> bool:
        # [FUNCTION DOC]
        # Behavior: True if stored or rebuildable from layer metadata.
        if super().__contains__(key):
            return True
        parts = key.rsplit(".", 1)
        if len(parts) == 2:
            layer_name, param_name = parts
            if layer_name in self._layer_meta_cache:
                if param_name in self._layer_meta_cache[layer_name]["meta"]:
                    return True
        return False

    def get(self, key: str, default=None):
        # [FUNCTION DOC]
        # Behavior: self[key] or default on KeyError.
        try:
            return self[key]
        except KeyError:
            return default


_DENSE_HF_PARAMS = ["weight", "weight_scale", "input_scale", "weight_scale_2"]


def _modelopt_dense_process_weights(self, layer: torch.nn.Module) -> None:
    # [FUNCTION DOC]
    # Behavior: first call saves metadata and loaders for weight, weight_scale, input_scale, weight_scale_2.
    # Asserts the block scale is FP8 E4M3. Repacks the packed weight (int32 view, transposed) to Marlin;
    # transposes the scale to params dtype, permutes it for Marlin and processes it (splitting off an optional
    # scale factor); global scale = processed max(weight_scale_2) divided by that factor. First call registers
    # weight, weight_scale, weight_global_scale and weight_scale_2 (sharing the global scale tensor) and saves
    # refs; later calls update refs in place and re-point weight_global_scale to the weight_scale_2 ref. If the
    # layer has input_scale, sets input_global_scale (max), alpha (input scale * max weight_scale_2) and
    # input_global_scale_inv, then deletes input_scale.
    # Exceptions: AssertionError for non-FP8 scales. Side effects: mutates layer.
    """
    Replacement for ModelOptNvFp4LinearMethod.process_weights_after_loading.

    First call:  save metadata + weight_loaders, convert HF→Marlin format,
                 save _marlin_tensor_refs for CUDA Graph stability.
    Subsequent:  read reloaded HF data, convert, copy_ into saved refs.
    """
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    is_first_call = _check_first_call(layer)

    if is_first_call:
        for pname in _DENSE_HF_PARAMS:
            _save_param_meta(layer, pname)
        _save_weight_loaders(layer, _DENSE_HF_PARAMS)

    weight_data = layer.weight.data
    weight_scale_data = layer.weight_scale.data
    weight_scale_2_data = layer.weight_scale_2.data

    assert weight_scale_data.dtype == torch.float8_e4m3fn

    device = weight_data.device
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    param_dtype = layer.params_dtype
    group_size = 16
    weight_scale_2_max = weight_scale_2_data.max().to(torch.float32)

    if is_first_call:
        layer.workspace = marlin_make_workspace_new(device)

    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = weight_data.view(torch.int32).T.contiguous()
    marlin_weight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=4,
    )

    weight_scale = weight_scale_data.T.contiguous().to(param_dtype)
    weight_scale = marlin_permute_scales(
        s=weight_scale,
        size_k=part_size_k,
        size_n=part_size_n,
        group_size=group_size,
    )
    marlin_weight_scale, scale_factor = _split_marlin_scale(
        nvfp4_marlin_process_scales(weight_scale, a_dtype=param_dtype)
    )
    marlin_weight_global_scale = (
        nvfp4_marlin_process_global_scale(weight_scale_2_max.to(torch.float32), param_dtype) / scale_factor
    )

    if is_first_call:
        layer.weight = Parameter(marlin_weight, requires_grad=False)
        layer.weight_scale = Parameter(marlin_weight_scale, requires_grad=False)
        layer.weight_global_scale = Parameter(marlin_weight_global_scale, requires_grad=False)
        layer.weight_scale_2 = Parameter(layer.weight_global_scale.data, requires_grad=False)
        layer._marlin_tensor_refs = {
            "weight": layer.weight.data,
            "weight_scale": layer.weight_scale.data,
            "weight_scale_2": layer.weight_global_scale.data,
        }
    else:
        _update_ref_or_create(layer, "weight", marlin_weight)
        _update_ref_or_create(layer, "weight_scale", marlin_weight_scale)
        _update_ref_or_create(layer, "weight_scale_2", marlin_weight_global_scale)
        layer.weight_global_scale = Parameter(layer._marlin_tensor_refs["weight_scale_2"], requires_grad=False)

    if hasattr(layer, "input_scale"):
        input_global_scale = layer.input_scale.data.max().to(torch.float32)
        layer.input_global_scale = Parameter(input_global_scale, requires_grad=False)
        layer.alpha = Parameter(input_global_scale * weight_scale_2_max, requires_grad=False)
        layer.input_global_scale_inv = Parameter((1.0 / input_global_scale).to(torch.float32), requires_grad=False)

    for attr in ["input_scale"]:
        if hasattr(layer, attr):
            delattr(layer, attr)


def _marlin_repack_experts(packed, perm, size_k, size_n, num_experts):
    # [FUNCTION DOC]
    # HELPER: Per-expert Marlin repack of packed weights (int32 view transposed); returns stacked tensor.
    import vllm._custom_ops as ops

    result = []
    for i in range(num_experts):
        qweight = packed[i].view(torch.int32).T.contiguous()
        result.append(
            ops.gptq_marlin_repack(
                b_q_weight=qweight,
                perm=perm,
                size_k=size_k,
                size_n=size_n,
                num_bits=4,
            )
        )
    return torch.stack(result)


def _marlin_process_scales_experts(scale_hf, param_dtype, size_k, size_n, group_size, num_experts):
    # [FUNCTION DOC]
    # HELPER: Per-expert Marlin permutation and NVFP4 processing of block scales (cast to params dtype,
    # transposed), dropping any returned scale factor; returns stacked tensor.
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import nvfp4_marlin_process_scales

    result = []
    scales = scale_hf.to(param_dtype)
    for i in range(num_experts):
        s = marlin_permute_scales(s=scales[i].T, size_k=size_k, size_n=size_n, group_size=group_size)
        result.append(_unwrap_marlin_scale(nvfp4_marlin_process_scales(s)))
    return torch.stack(result)


_MOE_HF_PARAMS = [
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
]


def _modelopt_moe_marlin_convert(self, layer: torch.nn.Module, is_first_call: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: Converts MoE weights to Marlin using vLLM's internal prepare_nvfp4_moe_layer_for_marlin (fetched
    # from the globals of convert_to_nvfp4_moe_kernel_format, a private dependency). First call registers the
    # six Marlin params and saves refs; later calls update refs in place. Clears w13/w2 input scales.
    # Side effects: mutates layer.
    from vllm.model_executor.layers.quantization import modelopt as vllm_modelopt

    prepare_for_marlin = vllm_modelopt.convert_to_nvfp4_moe_kernel_format.__globals__[
        "prepare_nvfp4_moe_layer_for_marlin"
    ]
    (
        w13_weight_marlin,
        w13_weight_scale_marlin,
        w13_scale_2_processed,
        w2_weight_marlin,
        w2_weight_scale_marlin,
        w2_scale_2_processed,
    ) = prepare_for_marlin(
        layer=layer,
        w13=layer.w13_weight.data,
        w13_scale=layer.w13_weight_scale.data,
        w13_scale_2=layer.w13_weight_scale_2.data,
        w2=layer.w2_weight.data,
        w2_scale=layer.w2_weight_scale.data,
        w2_scale_2=layer.w2_weight_scale_2.data,
        is_act_and_mul=self.moe.is_act_and_mul,
    )

    if is_first_call:
        layer.w13_weight = Parameter(w13_weight_marlin, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight_marlin, requires_grad=False)
        layer.w13_weight_scale = Parameter(w13_weight_scale_marlin, requires_grad=False)
        layer.w2_weight_scale = Parameter(w2_weight_scale_marlin, requires_grad=False)
        layer.w13_weight_scale_2 = Parameter(w13_scale_2_processed, requires_grad=False)
        layer.w2_weight_scale_2 = Parameter(w2_scale_2_processed, requires_grad=False)
        if not hasattr(layer, "_marlin_tensor_refs"):
            layer._marlin_tensor_refs = {}
        for rn in [
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
        ]:
            layer._marlin_tensor_refs[rn] = getattr(layer, rn).data
    else:
        for rn, nd in [
            ("w13_weight", w13_weight_marlin),
            ("w2_weight", w2_weight_marlin),
            ("w13_weight_scale", w13_weight_scale_marlin),
            ("w2_weight_scale", w2_weight_scale_marlin),
            ("w13_weight_scale_2", w13_scale_2_processed),
            ("w2_weight_scale_2", w2_scale_2_processed),
        ]:
            _update_ref_or_create(layer, rn, nd)

    layer.w13_input_scale = None
    layer.w2_input_scale = None


def _modelopt_moe_process_weights(self, layer: torch.nn.Module) -> None:
    # [FUNCTION DOC]
    # Behavior: first call saves metadata/loaders for the eight MoE HF params. Reduces a 2-D
    # w13_weight_scale_2 to its first column. Converts to Marlin, rebuilds the fused MoE quant config and
    # kernel (with shared experts and routing tables if available), and runs the kernel's experts
    # post-processing on the layer.
    # Exceptions: AssertionError if experts_cls is unset. Side effects: mutates layer and self.
    """
    Replacement for ModelOptNvFp4FusedMoE.process_weights_after_loading (Marlin).

    First call:  save metadata + weight_loaders, convert HF→Marlin format,
                 save _marlin_tensor_refs for CUDA Graph stability.
    Subsequent:  read reloaded HF data, convert, copy_ into saved refs.
    """
    is_first_call = _check_first_call(layer)

    if is_first_call:
        for pname in _MOE_HF_PARAMS:
            _save_param_meta(layer, pname)
        _save_weight_loaders(layer, _MOE_HF_PARAMS)

    w13_weight_scale_2 = layer.w13_weight_scale_2.data
    if w13_weight_scale_2.dim() == 2:
        w13_weight_scale_2 = w13_weight_scale_2[:, 0]
    layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

    _modelopt_moe_marlin_convert(self, layer, is_first_call)

    from vllm.model_executor.layers.quantization.modelopt import make_nvfp4_moe_kernel

    self.moe_quant_config = self.get_fused_moe_quant_config(layer)
    assert self.experts_cls is not None
    routing_tables = (
        layer._maybe_init_expert_routing_tables() if hasattr(layer, "_maybe_init_expert_routing_tables") else None
    )
    self.moe_kernel = make_nvfp4_moe_kernel(
        moe_quant_config=self.moe_quant_config,
        moe_config=self.moe,
        experts_cls=self.experts_cls,
        shared_experts=getattr(layer, "shared_experts", None),
        routing_tables=routing_tables,
    )
    self.moe_kernel.fused_experts.process_weights_after_loading(layer)


def _modelopt_kv_process_weights(self, layer) -> None:
    # [FUNCTION DOC]
    # Replacement for BaseKVCacheMethod.process_weights_after_loading that keeps the scale params.
    # Behavior: for non-auto KV dtype without dynamic scale calculation: both k/v scales positive -> use them;
    # both negative -> 1.0; otherwise duplicate the max into both (asserting k positive). Values doubled on
    # FP8-FNUZ platforms. Requires float per-tensor values; copies into _k_scale/_v_scale (and _q_scale from k
    # when q unset) and stores float copies. Then q_scale (doubled on FNUZ; disables calculate_kv_scales) or
    # 1.0, prob_scale likewise, validated as singleton floats and copied into _q_scale/_prob_scale.
    # Exceptions: ValueError for non per-tensor scales; AssertionError in the mixed-sign case.
    # Side effects: mutates layer buffers/attributes.
    """
    Replacement for BaseKVCacheMethod.process_weights_after_loading.
    Doesn't delete k_scale, v_scale, q_scale, prob_scale to allow
    for dynamic updates during refit.
    """
    from vllm.platforms import current_platform

    if layer.kv_cache_dtype != "auto" and not layer.calculate_kv_scales:
        if layer.k_scale > 0.0 and layer.v_scale > 0.0:
            k_scale = layer.k_scale.to("cpu").tolist()
            v_scale = layer.v_scale.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2
        elif layer.k_scale < 0.0 and layer.v_scale < 0.0:
            k_scale = 1.0
            v_scale = 1.0
        else:
            assert layer.k_scale > 0.0
            scale_to_duplicate = max(layer.k_scale, layer.v_scale)
            k_scale = scale_to_duplicate.to("cpu").tolist()
            v_scale = scale_to_duplicate.to("cpu").tolist()
            if current_platform.is_fp8_fnuz():
                k_scale *= 2
                v_scale *= 2

        if not isinstance(k_scale, float) or not isinstance(v_scale, float):
            raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

        if layer.q_scale < 0.0:
            layer._q_scale.copy_(k_scale)
            layer._q_scale_float = k_scale

        layer._k_scale.copy_(k_scale)
        layer._v_scale.copy_(v_scale)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale

    if layer.q_scale > 0.0:
        q_scale = layer.q_scale
        if current_platform.is_fp8_fnuz():
            q_scale *= 2
        layer.calculate_kv_scales = False
    else:
        q_scale = 1.0
    if layer.prob_scale > 0.0:
        prob_scale = layer.prob_scale
        if current_platform.is_fp8_fnuz():
            prob_scale *= 2
    else:
        prob_scale = 1.0

    is_singleton_float = (
        lambda x: isinstance(x, float) or isinstance(x, torch.Tensor) and x.numel() == 1 and x.is_floating_point()
    )
    if not is_singleton_float(q_scale) or not is_singleton_float(prob_scale):
        raise ValueError("Only support per-tensor scaling factor for fp8-quantized Q/prob")

    layer._q_scale.copy_(q_scale)
    layer._q_scale_float = q_scale.item() if isinstance(q_scale, torch.Tensor) else q_scale
    layer._prob_scale.copy_(prob_scale)


_patched = False
_original_dense_init = None
_original_moe_init = None


def _require_fp4_marlin_supported() -> None:
    # [FUNCTION DOC]
    # HELPER: Raises RuntimeError if vLLM reports FP4 Marlin unsupported on this platform.
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import is_fp4_marlin_supported

    if not is_fp4_marlin_supported():
        raise RuntimeError("ModelOpt NVFP4 QAT weight reload requires vLLM FP4 Marlin support.")


def _modelopt_dense_init_marlin(self, quant_config) -> None:
    # [FUNCTION DOC]
    # HELPER: Patched ModelOptNvFp4LinearMethod.__init__: runs the original, requires Marlin support, and
    # forces backend "marlin".
    _original_dense_init(self, quant_config)
    _require_fp4_marlin_supported()
    self.backend = "marlin"


def _modelopt_moe_init_marlin(self, quant_config, moe_config) -> None:
    # [FUNCTION DOC]
    # HELPER: Patched ModelOptNvFp4FusedMoE.__init__: runs the original, requires Marlin support, and forces
    # the Marlin MoE backend (disables flashinfer/cutlass, sets nvfp4_backend MARLIN, experts class
    # MarlinExperts, use_global_sf False).
    _original_moe_init(self, quant_config, moe_config)
    _require_fp4_marlin_supported()
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend

    self.allow_flashinfer = False
    self.cutlass_nvfp4_supported = False
    self.flashinfer_moe_backend = None
    self.use_marlin = True
    self.backend = "marlin"
    self.nvfp4_backend = NvFp4MoeBackend.MARLIN
    self.experts_cls = MarlinExperts
    self.use_global_sf = False


def prepare_modelopt_for_weight_reload(model, device=None):
    # [FUNCTION DOC]
    # Behavior: unwraps model.model, builds ModelOptParamMetaDict, swaps Marlin-format params to HF shape,
    # then re-creates every recorded param unless an existing one already matches HF shape/dtype and has a
    # weight_loader. Stores the dict on the inner model as _param_meta_for_restore.
    # Outputs: the dict.
    # KNOWN ISSUE: the model is unwrapped here and again inside ModelOptParamMetaDict (double .model unwrap).
    # Side effects: registers/replaces params.
    """Prepare ModelOpt model for weight reloading. Call ONCE before each reload cycle."""
    inner_model = model
    if hasattr(model, "model"):
        inner_model = model.model

    param_meta = ModelOptParamMetaDict(inner_model, device=device)

    param_meta.prepare_for_reload()

    for layer_name, cache_entry in param_meta._layer_meta_cache.items():
        module = cache_entry["module"]
        for param_name, pm in cache_entry["meta"].items():
            existing = getattr(module, param_name, None)
            if existing is not None:
                hf_shape = tuple(pm["shape"])
                hf_dtype = pm["dtype"]
                if (
                    tuple(existing.shape) == hf_shape
                    and existing.dtype == hf_dtype
                    and hasattr(existing, "weight_loader")
                ):
                    continue
            new_param = _create_param_from_meta(module, param_name, pm, device)
            module.register_parameter(param_name, new_param)

    inner_model._param_meta_for_restore = param_meta
    return param_meta


def modelopt_process_weights_after_loading(model):
    # [FUNCTION DOC]
    # Behavior: unwraps model.model; calls scheme.process_weights_after_loading for modules with a scheme, and
    # quant_method.process_weights_after_loading for other modules having one, skipping KV-cache methods.
    # KNOWN ISSUE: because KV-cache methods are skipped, the patched KV scale processing is not re-run after a
    # reload, so refitted k/v/q scales are not applied.
    # Outputs: number of modules processed. Side effects: runs post-processing.
    """Trigger weight post-processing for all quantized layers after load_weights."""
    dense_count = 0
    moe_count = 0

    actual_model = model
    if hasattr(model, "model"):
        actual_model = model.model

    for module in actual_model.modules():
        if hasattr(module, "scheme"):
            module.scheme.process_weights_after_loading(module)
            dense_count += 1

        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None and not hasattr(module, "scheme"):
            if hasattr(quant_method, "process_weights_after_loading"):
                if "KVCache" in quant_method.__class__.__name__:
                    continue
                quant_method.process_weights_after_loading(module)
                moe_count += 1

    return dense_count + moe_count


def apply_modelopt_nvfp4_patches():
    # [FUNCTION DOC]
    # Behavior: once per process, saves the original dense/MoE __init__ methods and replaces them with the
    # Marlin-forcing versions, and replaces process_weights_after_loading on the dense, MoE and KV-cache
    # method classes. Idempotent via a module flag.
    # Exceptions: ImportError if vLLM modules are missing. Side effects: process-global monkey-patching.
    """Apply ModelOpt NVFP4 patches to support dynamic weight updates. Call before model loading."""
    global _patched
    global _original_dense_init
    global _original_moe_init

    if _patched:
        return

    from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod

    _original_dense_init = ModelOptNvFp4LinearMethod.__init__
    _original_moe_init = ModelOptNvFp4FusedMoE.__init__

    ModelOptNvFp4LinearMethod.__init__ = _modelopt_dense_init_marlin
    ModelOptNvFp4FusedMoE.__init__ = _modelopt_moe_init_marlin
    ModelOptNvFp4LinearMethod.process_weights_after_loading = _modelopt_dense_process_weights
    ModelOptNvFp4FusedMoE.process_weights_after_loading = _modelopt_moe_process_weights
    BaseKVCacheMethod.process_weights_after_loading = _modelopt_kv_process_weights

    _patched = True
