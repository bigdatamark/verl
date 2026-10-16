# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# TRT-LLM Ray worker extensions loaded by class path inside TRT-LLM GPU workers to support
# verl weight sync. TRT-LLM imports are guarded: on CPU-only actors the names fall back to
# None / identity decorator / object base so the module can still be imported.
import base64
import inspect
from typing import Optional

# Defer tensorrt_llm imports to avoid FlashInfer's check_cuda_arch() crash
# when this module is loaded on CPU-only Ray actors. The module is normally
# loaded only on GPU workers via string path in trtllm_async_server.py, but
# guard defensively in case of transitive imports.
try:
    from tensorrt_llm import serialization
    from tensorrt_llm._ray_utils import control_action_decorator
    from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import MoeLoadBalancer
    from tensorrt_llm._torch.utils import get_device_uuid
    from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension as TrtllmWorkerExtension
    from tensorrt_llm.logger import logger
except (ImportError, RuntimeError):
    # On CPU actors without CUDA, these imports may fail.
    # The class below won't be usable, but the module can be imported safely.
    serialization = None
    control_action_decorator = lambda f: f  # noqa: E731 — identity fallback
    MoeLoadBalancer = None
    get_device_uuid = None
    TrtllmWorkerExtension = object
    logger = None


class WorkerExtension(TrtllmWorkerExtension):
    # [CLASS DOC]
    # Worker extension (used for VLM models) adding partial-loading detection and an
    # update_weights RPC that rebuilds tensors from CUDA IPC handles and reloads them into
    # the TRT-LLM model, plus prefix-cache reset. Relies on the TRT-LLM worker providing
    # self.engine and self.device_id.
    def __init__(self):
        # [FUNCTION DOC]
        # No-op constructor; intentionally does not call the base initializer.
        pass

    @control_action_decorator
    def supports_partial_loading(self) -> bool:
        # [FUNCTION DOC]
        # Control RPC: True if the model's load_weights accepts an allow_partial_loading
        # argument; any error is logged as a warning and yields False.
        """Check if the model supports partial weight loading."""
        try:
            model = self.engine.model_engine.model
            load_weights_args = inspect.getfullargspec(model.load_weights).args
            return "allow_partial_loading" in load_weights_args
        except Exception as e:
            logger.warning(f"Failed to check partial loading support: {e}")
            return False

    @control_action_decorator
    def update_weights(self, ipc_handles: Optional[dict] = None):
        # [FUNCTION DOC]
        # Control RPC applying one weight-update step.
        # Behavior: on the first call of an update cycle (no first_pre_reload_weights marker on the
        # model), calls pre_reload_weights on every module that has it and has not had weights
        # removed, then sets the marker. If ipc_handles is given: looks up this device's UUID
        # entry; if it is a string, base64-decodes it and loads it with TRT-LLM's restricted
        # unpickler allowing only basic builtins and torch tensor/storage reconstruction names,
        # requiring a list; otherwise uses it as-is. For each (name, (rebuild func, args)) replaces
        # argument index 6 (device index) with this worker's device id and rebuilds the tensor,
        # then reloads the mapping into the model with allow_partial_loading set according to
        # model support. If ipc_handles is None (finalize): calls process_weights_after_loading and
        # post_load_weights on eligible modules, finalizes the MoE load balancer if present,
        # resets the prefix cache and removes the first_pre_reload_weights marker.
        # Exceptions: ValueError if this device's UUID is missing or decoded data is not a list;
        # any error is logged and re-raised.
        # Side effects: mutates model weights and module state.
        try:
            if not hasattr(self.engine.model_engine.model, "first_pre_reload_weights"):
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(module, "_weights_removed", False):
                        module.pre_reload_weights()
                self.engine.model_engine.model.first_pre_reload_weights = True

            if ipc_handles is not None:
                logger.info("Update weights from IPC handles")
                device_uuid = get_device_uuid(self.device_id)

                if device_uuid not in ipc_handles:
                    raise ValueError(f"Device UUID {device_uuid} not found in ipc_handles")

                weights = {}

                serialized_handles = ipc_handles[device_uuid]
                if isinstance(serialized_handles, str):
                    # Data is base64-encoded pickled bytes - deserialize it
                    # using restricted unpickler from tensorrt_llm.serialization
                    logger.info("Deserializing base64-encoded weight handles")
                    decoded_data = base64.b64decode(serialized_handles)
                    # Allow basic builtins and torch tensor reconstruction classes
                    approved_imports = {
                        "builtins": [
                            "list",
                            "tuple",
                            "str",
                            "int",
                            "float",
                            "bool",
                            "bytes",
                            "dict",
                            "NoneType",
                            "type",
                        ],
                        "torch": [
                            "Tensor",
                            "FloatTensor",
                            "DoubleTensor",
                            "HalfTensor",
                            "BFloat16Tensor",
                            "IntTensor",
                            "LongTensor",
                            "ShortTensor",
                            "CharTensor",
                            "ByteTensor",
                            "BoolTensor",
                            "Size",
                            "dtype",
                            "device",
                            "float32",
                            "float16",
                            "bfloat16",
                            "int32",
                            "int64",
                            "int16",
                            "int8",
                            "uint8",
                            "bool",
                        ],
                        "torch.multiprocessing.reductions": [
                            "rebuild_cuda_tensor",
                            "rebuild_tensor",
                        ],
                        "torch._utils": [
                            "_rebuild_tensor_v2",
                        ],
                        "torch.storage": [
                            "_load_from_bytes",
                            "_TypedStorage",
                            "UntypedStorage",
                            "TypedStorage",
                        ],
                    }
                    all_handles = serialization.loads(
                        decoded_data,
                        approved_imports=approved_imports,
                    )

                    # Verify the result is a list as expected
                    if not isinstance(all_handles, list):
                        raise ValueError(f"Deserialized data must be a list, got {type(all_handles).__name__} instead")
                else:
                    # Data is already in the correct format (backward compatibility)
                    all_handles = serialized_handles

                for param_name, tensor_handle in all_handles:
                    func, args = tensor_handle
                    list_args = list(args)
                    list_args[6] = self.device_id
                    tensor = func(*list_args)
                    weights[param_name] = tensor

                logger.info(f"weights key size: {len(weights.keys())}")

                # Check if model supports partial loading and use appropriate strategy
                model = self.engine.model_engine.model
                load_weights_args = inspect.getfullargspec(model.load_weights).args
                supports_partial_loading = "allow_partial_loading" in load_weights_args

                if supports_partial_loading:
                    self.engine.model_engine.model_loader.reload(model, weights, allow_partial_loading=True)
                else:
                    self.engine.model_engine.model_loader.reload(model, weights, allow_partial_loading=False)
            else:
                logger.info("Finalize update weights")
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "process_weights_after_loading") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.process_weights_after_loading()
                    if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
                        module.post_load_weights()
                moe_load_balancer = getattr(self.engine.model_engine, "moe_load_balancer", None)
                if isinstance(moe_load_balancer, MoeLoadBalancer):
                    moe_load_balancer.register_weight_slots_after_to_cuda()
                    logger.info("moe_load_balancer finalizing model...")
                    moe_load_balancer.finalize_model()
                    logger.info("moe_load_balancer finalize model done")
                self.engine.reset_prefix_cache()
                delattr(self.engine.model_engine.model, "first_pre_reload_weights")

        except Exception as e:
            logger.error("Encountered an error in update_weights")
            raise e

    def reset_prefix_cache(self) -> None:
        # [FUNCTION DOC]
        # Reset the engine's KV prefix cache (after weight updates).
        """Invalidate the KV cache prefix reuse state after weight updates."""
        self.engine.reset_prefix_cache()


# TODO: remove this class and revert the non-VLM path in trtllm_async_server.py
# to use "tensorrt_llm.llmapi.rlhf_utils.WorkerExtension" once verl's TRT-LLM version
# is bumped to include https://github.com/NVIDIA/TensorRT-LLM/pull/13784.
class RlhfWorkerExtension(TrtllmWorkerExtension):
    # [CLASS DOC]
    # Thin extension of TRT-LLM's RLHF worker extension used for non-VLM models, adding the
    # wait_for_engine_idle RPC verl calls after pausing generation.
    """Minimal extension of TRT-LLM's WorkerExtension for non-VLM RLHF models."""

    @control_action_decorator
    def wait_for_engine_idle(self) -> None:
        # [FUNCTION DOC]
        # Control RPC intended to block until the engine is idle.
        # KNOWN ISSUE: the body is empty, so it returns immediately despite the docstring; it only
        # exists so the collective RPC name resolves.
        """Block until the engine has no active or queued requests."""
        pass
