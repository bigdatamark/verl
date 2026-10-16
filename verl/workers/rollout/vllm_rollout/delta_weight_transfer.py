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
# vLLM worker-side weight-transfer engine for verl's delta checkpoint engine. Registers a
# vLLM WeightTransferEngine backend ("verl_delta_ipc") that receives each DeltaFlush
# payload over the bucketed ZMQ/CUDA-IPC channel, decodes it into vLLM
# CheckpointWeightPatch objects and applies them with vLLM's checkpoint patch loader.
# Import-time requirement: raises RuntimeError if vLLM lacks the weight_transfer modules.
# Module state: VERL_DELTA_WEIGHT_TRANSFER_BACKEND name, _REGISTERED flag.
"""Receive ``DeltaFlush`` payloads inside vLLM workers.

VERL sends each payload through its existing ZMQ/CUDA-IPC channel. This
adapter decodes it and calls vLLM's checkpoint patch loader, which handles
checkpoint names, packed weights, and TP slicing.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import vllm

from verl.utils.device import get_device_name, is_cuda_available

try:
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        WeightTransferEngine,
        WeightTransferInitInfo,
        WeightTransferUpdateInfo,
    )
except ImportError as exc:
    raise RuntimeError(
        "vLLM delta_sharded support requires the vllm.config.weight_transfer "
        "and vllm.distributed.weight_transfer modules"
    ) from exc

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.checkpoint_weight_patch import (
        CheckpointWeightPatch,
    )

VERL_DELTA_WEIGHT_TRANSFER_BACKEND = "verl_delta_ipc"
_REGISTERED = False


def _checkpoint_patch_api():
    # [FUNCTION DOC]
    # HELPER: Import and return (CheckpointWeightPatch, load_checkpoint_weight_patches) from
    # vLLM's checkpoint_weight_patch module.
    # Exceptions: RuntimeError with upgrade guidance if that module is missing; other import
    # errors propagate.
    try:
        from vllm.model_executor.model_loader.checkpoint_weight_patch import (
            CheckpointWeightPatch,
            load_checkpoint_weight_patches,
        )
    except ImportError as exc:
        if exc.name != "vllm.model_executor.model_loader.checkpoint_weight_patch":
            raise
        raise RuntimeError(
            "checkpoint_engine.backend='delta_sharded' with vLLM requires the checkpoint patch API from "
            f"vLLM #50723; installed vLLM {getattr(vllm, '__version__', 'unknown')} does not provide it. "
            "Install a compatible vLLM build or select a different checkpoint engine backend."
        ) from exc
    return CheckpointWeightPatch, load_checkpoint_weight_patches


def require_vllm_delta_support() -> None:
    # [FUNCTION DOC]
    # Check that WeightTransferEngine.__init__ accepts config, vllm_config, device and model
    # and that the checkpoint patch API exists.
    # Exceptions: RuntimeError when either is unavailable.
    """Verify that vLLM provides the required weight transfer and checkpoint patch APIs."""

    init_parameters = inspect.signature(WeightTransferEngine.__init__).parameters
    if not {"config", "vllm_config", "device", "model"}.issubset(init_parameters):
        raise RuntimeError(
            "checkpoint_engine.backend='delta_sharded' with vLLM requires the engine-owned WeightTransferEngine "
            "API from vLLM #44353 and the checkpoint patch API from vLLM #50723; installed vLLM "
            f"{getattr(vllm, '__version__', 'unknown')} does not provide the required WeightTransferEngine API. "
            "Install a compatible vLLM build or select a different checkpoint engine backend."
        )
    _checkpoint_patch_api()


def is_moe_model(hf_config) -> bool:
    # [FUNCTION DOC]
    # Return whether vLLM would treat hf_config as MoE: uses vLLM's model-arch config
    # convertor for the model type to get the expert count (> 0). On any failure falls back to
    # inspecting the text config for num_experts / moe_num_experts / n_routed_experts /
    # num_local_experts (first element if a list).
    """Whether vLLM will treat this HF config as a MoE model.

    Delta updates require the Triton MoE backend, so this check must agree with
    the engine's own MoE detection: expert counts may live under a different
    attribute (Dbrx ``moe_num_experts``, DeepSeek ``n_routed_experts``, Mixtral
    ``num_local_experts``) or on a nested text config (multimodal MoE). The
    fallback mirrors vLLM's attribute list for configs its convertor cannot
    process.
    """
    try:
        from vllm.transformers_utils.config import get_hf_text_config
        from vllm.transformers_utils.model_arch_config_convertor import (
            MODEL_ARCH_CONFIG_CONVERTORS,
            ModelArchConfigConvertorBase,
        )

        convertor_cls = MODEL_ARCH_CONFIG_CONVERTORS.get(
            getattr(hf_config, "model_type", ""), ModelArchConfigConvertorBase
        )
        return int(convertor_cls(hf_config, get_hf_text_config(hf_config)).get_num_experts()) > 0
    except Exception:
        text_config = hf_config.get_text_config() if hasattr(hf_config, "get_text_config") else hf_config
        for name in ("num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"):
            num_experts = getattr(text_config, name, 0)
            if isinstance(num_experts, list):
                num_experts = num_experts[0] if num_experts else 0
            if num_experts and int(num_experts) > 0:
                return True
        return False


@dataclass
class VerlDeltaIPCInitInfo(WeightTransferInitInfo):
    # [CLASS DOC]
    # Empty init-info dataclass for this transport (no setup data needed).
    """No initialization data is needed for the local IPC transport."""


@dataclass
class VerlDeltaIPCUpdateInfo(WeightTransferUpdateInfo):
    # [CLASS DOC]
    # Per-flush update info carrying the worker-local ZMQ endpoint (zmq_handle).
    """Connection details for one flush."""

    zmq_handle: str | None = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC]
        # Validate zmq_handle. Exceptions: ValueError if it is not a non-empty string.
        if not isinstance(self.zmq_handle, str) or not self.zmq_handle:
            raise ValueError("VERL delta IPC updates require a worker-local zmq_handle")


def decode_delta_payload(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[str, list[CheckpointWeightPatch]]:
    # [FUNCTION DOC]
    # Validate and decode one DeltaFlush into vLLM patches.
    # Behavior: builds a mapping from the named tensors; parses "__delta_spec__" bytes as JSON;
    # requires "__values__"; encoding must be "dense" or "indices". Positions default to an
    # empty uint8 tensor; dense flushes must have no positions and indices flushes must carry
    # "__positions__". Verifies the checksum over positions and values. For each param spec
    # builds a CheckpointWeightPatch with name, shape, dtype, the values slice and, for
    # indices, the positions slice viewed as int32 (requires pos_width 4 and 4 bytes per
    # value); dense patches have no indices.
    # Outputs: (encoding, list of patches).
    # Exceptions: ValueError for missing tensors, invalid JSON, unsupported encoding or
    # inconsistent slices; RuntimeError on checksum mismatch or missing vLLM API.
    """Validate and decode one DeltaFlush into vLLM checkpoint patches."""

    from verl.checkpoint_engine.delta_sync.encode import checksum

    CheckpointWeightPatch, _ = _checkpoint_patch_api()
    tensors = dict(named_tensors)
    try:
        spec_tensor = tensors["__delta_spec__"]
        values = tensors["__values__"]
        spec = json.loads(bytes(spec_tensor.detach().cpu().numpy().tobytes()).decode())
    except KeyError as exc:
        raise ValueError(f"DeltaFlush is missing {exc.args[0]}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("DeltaFlush contains an invalid JSON spec") from exc

    encoding = spec["encoding"]
    if encoding not in {"dense", "indices"}:
        raise ValueError(f"Unsupported DeltaFlush encoding: {encoding!r}")
    positions = tensors.get("__positions__")
    if positions is None:
        positions = torch.empty(0, dtype=torch.uint8, device=values.device)
    if encoding == "dense" and positions.numel():
        raise ValueError("Dense DeltaFlush must not contain positions")
    if encoding == "indices" and "__positions__" not in tensors:
        raise ValueError("Indices DeltaFlush requires __positions__")

    got_checksum = checksum(positions, values)
    expected_checksum = int(spec["checksum"])
    if got_checksum != expected_checksum:
        raise RuntimeError(
            f"Delta checksum mismatch in vLLM consumer: got {got_checksum}, expected {expected_checksum}"
        )

    patches: list[CheckpointWeightPatch] = []
    for param_spec in spec["params"]:
        name = param_spec["name"]
        dtype = getattr(torch, param_spec["dtype"])
        patch_values = values[param_spec["val_start"] : param_spec["val_end"]]
        if encoding == "dense":
            patch_indices = None
        else:
            if param_spec["pos_width"] != 4:
                raise ValueError(f"{name}: indices encoding requires pos_width=4")
            pos_bytes = positions[param_spec["pos_start"] : param_spec["pos_end"]]
            if pos_bytes.numel() != patch_values.numel() * 4:
                raise ValueError(f"{name}: position and value slices have inconsistent lengths")
            patch_indices = pos_bytes.view(torch.int32)

        patches.append(
            CheckpointWeightPatch(
                name=name,
                shape=tuple(param_spec["shape"]),
                dtype=dtype,
                values=patch_values,
                indices=patch_indices,
            )
        )
    return encoding, patches


class VerlDeltaIPCWeightTransferEngine(WeightTransferEngine):
    # [CLASS DOC]
    # vLLM WeightTransferEngine consuming verl DeltaFlush payloads. Tracks the encoding of the
    # current update session (all flushes of one update must share it) and a sticky failure
    # flag that blocks further updates after any partial failure. Does not support draft
    # (speculative) weight updates; sending is not supported.
    """Consume VERL DeltaFlush payloads over the colocated CUDA-IPC channel."""

    init_info_cls = VerlDeltaIPCInitInfo
    update_info_cls = VerlDeltaIPCUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        # [FUNCTION DOC]
        # Initialize the vLLM base engine and reset session encoding and failure flag.
        super().__init__(config, vllm_config, device, model)
        self._session_encoding: str | None = None
        self._update_failed = False

    def _validate_configuration(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Ensure the environment is supported: delta API present, CUDA device, data and
        # pipeline parallel sizes 1, BF16 model dtype, no quantization, no speculative decoding.
        # Exceptions: RuntimeError or NotImplementedError describing the unsupported condition.
        require_vllm_delta_support()
        if not is_cuda_available or self.device.type != get_device_name():
            raise NotImplementedError("VERL delta weight transfer requires CUDA")
        if self.parallel_config.data_parallel_size != 1:
            raise NotImplementedError("VERL delta weight transfer requires data_parallel_size=1")
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("VERL delta weight transfer requires pipeline_parallel_size=1")
        if self.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("VERL delta weight transfer supports only BF16 vLLM models")
        if getattr(self.vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("VERL delta weight transfer does not support quantized vLLM models")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("VERL delta weight transfer does not support speculative decoding")

    def init_transfer_engine(self, init_info: VerlDeltaIPCInitInfo) -> None:
        # [FUNCTION DOC]
        # Validate configuration; no other initialization is needed (init_info unused).
        self._validate_configuration()

    def start_weight_update(self) -> None:
        # [FUNCTION DOC]
        # Begin an update session by clearing the session encoding.
        # Exceptions: RuntimeError if a previous update failed (the job must be restarted).
        if self._update_failed:
            raise RuntimeError(
                "A previous delta update did not complete; restart the full job so every rollout worker receives a "
                "fresh dense seed. Restarting only the vLLM workers is unsafe because the producer snapshot may "
                "already have advanced."
            )
        self._session_encoding = None

    def _receive_payload(self, *, zmq_handle: str) -> list[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC]
        # HELPER: Receive one flush through a BucketedWeightReceiver (CUDA IPC, no shm) on this
        # device, cloning each received tensor because the IPC buffer is reused. Errors raised in
        # the callback are deferred until the receive handshake completes, then re-raised.
        # Outputs: list of (name, tensor) pairs.
        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightReceiver,
        )

        receiver = BucketedWeightReceiver(
            zmq_handle=zmq_handle,
            device=self.device,
            use_shm=False,
        )
        payload: list[tuple[str, torch.Tensor]] = []
        callback_error: BaseException | None = None

        def retain_bucket(weights: list[tuple[str, torch.Tensor]], _is_last: bool) -> None:
            # [FUNCTION DOC]
            # HELPER: Bucket callback that appends clones of the bucket's tensors to the payload,
            # recording the first error and ignoring later buckets once one occurred.
            nonlocal callback_error
            if callback_error is not None:
                return
            try:
                # The receiver reuses its IPC buffer for the next bucket, so
                # retain a private copy until this flush is decoded and applied.
                payload.extend((name, tensor.clone()) for name, tensor in weights)
            except BaseException as exc:
                # Finish the receiver handshake before reporting a callback error.
                callback_error = exc

        receiver.receive_weights(on_bucket_received=retain_bucket)
        if callback_error is not None:
            raise callback_error
        return payload

    def receive_weights(self, update_info: VerlDeltaIPCUpdateInfo) -> None:
        # [FUNCTION DOC]
        # Receive, decode and apply one flush. The first flush fixes the session encoding (dense
        # initializes vLLM's layerwise reload on the model); later flushes must match it. Applies
        # patches with duplicate-index validation disabled.
        # Exceptions: ValueError on mixed encodings; any error sets the sticky failure flag and is
        # re-raised. Side effects: mutates model weights.
        assert update_info.zmq_handle is not None
        try:
            payload = self._receive_payload(zmq_handle=update_info.zmq_handle)
            encoding, patches = decode_delta_payload(payload)

            if self._session_encoding is None:
                self._session_encoding = encoding
                if encoding == "dense":
                    from vllm.model_executor.model_loader.reload import (
                        initialize_layerwise_reload,
                    )

                    initialize_layerwise_reload(self.model)
            elif encoding != self._session_encoding:
                raise ValueError(
                    "One weight update cannot mix dense and sparse DeltaFlush payloads "
                    f"({self._session_encoding!r} then {encoding!r})"
                )

            _, load_checkpoint_weight_patches = _checkpoint_patch_api()
            # The trusted producer emits every changed position at most once;
            # skip vLLM's sort-based duplicate check on the rollout GPU.
            load_checkpoint_weight_patches(
                self.model,
                patches,
                validate_unique_indices=False,
            )
        except BaseException:
            # An earlier flush may already have changed runtime weights, so a
            # later update would start from an unknown partial state.
            self._update_failed = True
            raise

    def finish_weight_update(self) -> None:
        # [FUNCTION DOC]
        # End the session: for dense sessions finalize the layerwise reload (failure sets the
        # failure flag and re-raises); then clear the session encoding.
        if self._session_encoding == "dense":
            # Dense updates use vLLM's layerwise reload lifecycle. Finalize it
            # once after the last flush to run model-specific post-load work.
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            try:
                finalize_layerwise_reload(self.model, self.model_config)
            except BaseException:
                self._update_failed = True
                raise
        # Sparse updates write to initialized runtime tensors, so there is
        # nothing to rebuild after the last flush.
        self._session_encoding = None

    def shutdown(self) -> None:
        # [FUNCTION DOC]
        # Clear the session encoding.
        self._session_encoding = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[Any],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        # [FUNCTION DOC]
        # Not supported on this receive-only adapter. Exceptions: always NotImplementedError.
        raise NotImplementedError("VERL sends DeltaFlush payloads; this adapter only receives them")


def register_verl_delta_weight_transfer_engine() -> None:
    # [FUNCTION DOC]
    # Register VerlDeltaIPCWeightTransferEngine with vLLM's WeightTransferEngineFactory under
    # the verl backend name, once per process (idempotent via the module flag).
    """Register the adapter before vLLM creates its worker-side engine."""

    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    WeightTransferEngineFactory.register_engine(
        VERL_DELTA_WEIGHT_TRANSFER_BACKEND,
        VerlDeltaIPCWeightTransferEngine,
    )
    _REGISTERED = True
