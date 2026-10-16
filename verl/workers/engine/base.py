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
# Defines the abstract training-engine interface (BaseEngine), the train/eval context manager base (BaseEngineCtx) handling automatic parameter/optimizer offload, and EngineRegistry, which maps (model_type, backend, device, vendor) to concrete engine classes.
"""
The abstract base class defining the interface for model training engines.
"""

import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name, get_vendor
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids


class BaseEngine:
    # [CLASS DOC]
    # Abstract engine interface implemented by every backend (FSDP, Megatron, Automodel, TorchTitan, MindSpeed...).
    # Most methods raise NotImplementedError; train_batch, infer_batch, prime_delta_snapshots, to (argument check only) and disable_adapter have concrete default implementations.
    # Class attribute delta_pin_snapshots (default True) controls whether delta diff snapshots use pinned host memory on CUDA.
    """
    Abstract base class defining the interface for model training engines. Interface is subject to
    change before release.

    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.
    """

    def initialize(self):
        # [FUNCTION DOC]
        # Abstract: build/load model, optimizer and LR scheduler. Raises NotImplementedError.
        """
        Instantiate or load the model, optimizer, and learning rate scheduler.

        Should prepare all components necessary for training or evaluation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Abstract property: whether model parameters are offloaded to CPU between uses. Raises NotImplementedError.
        """Whether parameter offloading is enabled."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        # [FUNCTION DOC]
        # Abstract property: whether optimizer states are offloaded to CPU between uses. Raises NotImplementedError.
        """Whether optimizer offloading is enabled."""
        raise NotImplementedError

    def train_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Abstract: returns a context manager that puts engine/model in training mode (kwargs forwarded to the context, e.g. disable_auto_offload). Raises NotImplementedError.
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        raise NotImplementedError

    def eval_mode(self, **kwargs):
        # [FUNCTION DOC]
        # Abstract: returns a context manager that puts engine/model in evaluation mode. Raises NotImplementedError.
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        # [FUNCTION DOC]
        # Abstract: zero optimizer gradients. Raises NotImplementedError.
        """
        Zero the gradients of the optimizer.
        """
        raise NotImplementedError

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Abstract: perform one optimizer step; implementations return the gradient norm. Raises NotImplementedError.
        """
        Perform an optimization step using the optimizer.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        # [FUNCTION DOC]
        # Abstract: advance the LR scheduler and return the current learning rate(s). Raises NotImplementedError.
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        # [FUNCTION DOC]
        # Abstract: run forward (and backward unless forward_only) over a batch, possibly with micro-batching, using loss_function to compute losses/metrics. Returns a backend-specific output dict (typically containing "metrics", and "model_output" on output ranks). Raises NotImplementedError.
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        raise NotImplementedError

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        # [FUNCTION DOC]
        # Runs a full training step on one batch.
        # Behavior: fixes 3D position ids in data in place, zeroes gradients, runs forward_backward_batch with backward, runs optimizer_step to get grad_norm, and on the model-parallel source rank with outputs inserts grad_norm into outputs["metrics"].
        # Inputs: data - TensorDict batch; loss_function - loss callable.
        # Outputs: the dict returned by forward_backward_batch (with grad_norm added on the output rank).
        # Exceptions: AssertionError if "grad_norm" already exists in metrics.
        # Side effects: updates model weights and optimizer state; mutates data.
        """
        Perform a training step on a batch of data.

        Args:
            data: The input data for training, typically containing tensors and metadata.
            loss_function: A function that computes the loss and metrics given a batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the batch.
        """
        maybe_fix_3d_position_ids(data)

        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs["metrics"]
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
        # [FUNCTION DOC]
        # Runs a forward-only pass under no-grad.
        # Behavior: fixes 3D position ids in data, then calls forward_backward_batch with forward_only true inside torch.no_grad.
        # Inputs: data - TensorDict batch; loss_function - optional loss callable (may be None).
        # Outputs: whatever forward_backward_batch returns.
        # Side effects: mutates data's position ids.
        """
        Perform inference on a batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.

        Returns:
            Any: The output of the inference, which can be used for predictions or other purposes.
        """
        # see comments from train_batch
        maybe_fix_3d_position_ids(data)

        with torch.no_grad():
            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
        return outputs

    def get_per_tensor_param(self) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        # [FUNCTION DOC]
        # Abstract: return a generator of (name, full tensor) in HF naming plus an optional PEFT config dict. Raises NotImplementedError.
        """
        Get a generator that yields per-tensor parameters and optional peft config.

        Returns:
            Generator[tuple[str, torch.Tensor]]: A generator that yields tuples of parameter names and tensors.
            Optional[dict]: None without an adapter, else a PEFT config carrying peft_type,
                task_type, r, lora_alpha, target_modules, lora_dropout and bias -- the first
                two as enum members, not strings.
        """
        raise NotImplementedError

    def get_per_tensor_param_shard(self, **kwargs) -> tuple[Generator, Optional[dict]]:
        # [FUNCTION DOC]
        # Abstract: like get_per_tensor_param but yields (name, local_shard, ShardSpec) without gathering. Raises NotImplementedError.
        """
        Like :meth:`get_per_tensor_param`, but yields each rank's *local* parameter shard
        instead of all-gathering full tensors.

        Used by checkpoint engines that operate on shards (e.g. the ``delta_sharded``
        backends, which byte-diff each rank's shard locally and only gather the changed
        elements), so the export never materializes full tensors on any rank.

        Implementations yield ``(name, local_shard, ShardSpec)`` -- the spec (see
        :mod:`verl.workers.engine.spec`) carries all placement knowledge
        (offset translation or a dense rebuild callable), so the consuming checkpoint
        engine stays trainer-agnostic.

        Returns:
            Generator: A generator that yields per-parameter local shards with placement metadata.
            Optional[dict]: Optional peft config.
        """
        raise NotImplementedError

    # Host-memory policy for the delta diff base: pinned (cudaHostAlloc) gives a
    # faster H2D on the diff read-back but competes with every other pinned pool
    # on the node; backends whose memory profile makes that competition
    # dangerous override this to False (see MegatronEngine).
    delta_pin_snapshots: bool = True

    def prime_delta_snapshots(self) -> None:
        # [FUNCTION DOC]
        # Records the current local shards as the base for later delta exports.
        # Behavior: lazily creates the self._delta_shard_snap mapping (keeps an existing one), obtains the shard generator from get_per_tensor_param_shard and passes it to the shared prime_delta_snapshots utility, pinning host memory only when CUDA is available and delta_pin_snapshots is true.
        # Inputs: none.
        # Outputs: none.
        # Side effects: allocates host-memory snapshots stored on self._delta_shard_snap.
        """
        Snapshot this rank's CURRENT shards as the delta diff base. Called right after the
        seed sync (which streams :meth:`get_per_tensor_param`'s full HF export):
        weights do not move during the sync, so the snapshots equal exactly what the
        rollout side received and the first
        :meth:`get_per_tensor_param_delta_shard` diff is correct.

        Concrete here: it only consumes :meth:`get_per_tensor_param_shard`, so any
        engine that implements the shard export gets it for free.
        """
        from verl.utils.device import is_cuda_available
        from verl.workers.engine.utils import prime_delta_snapshots

        self._delta_shard_snap = getattr(self, "_delta_shard_snap", {})
        gen, _ = self.get_per_tensor_param_shard()
        prime_delta_snapshots(gen, self._delta_shard_snap, pin=is_cuda_available and self.delta_pin_snapshots)

    def get_per_tensor_param_delta_shard(self, **kwargs) -> tuple[Generator, Optional[dict]]:
        # [FUNCTION DOC]
        # Abstract: yield per-parameter HF-coordinate delta payloads (slots, dtype_str, counts, hf_idx, hf_val, gather_group) in identical order on all ranks, plus optional PEFT config. Raises NotImplementedError.
        """
        Yield the delta engine's per-parameter payloads in FINAL HF coordinates:
        ``(slots, dtype_str, counts, hf_idx, hf_val, gather_group)``, in an order
        identical on every rank (non-contributing replicas yield zero-count entries
        but stay in the lockstep sequence). ``slots`` enumerates the HF tensors this
        parameter maps to (identity params: itself; fused params: their hf_slots),
        and ``hf_idx``/``hf_val`` are the changed elements since the previous export,
        already converted to HF coordinates.

        Everything backend-specific lives behind this call: the weight->HF naming,
        the to-HF conversion, the diff and its base. A backend that already keeps the
        previous step's weights (e.g. Decoupled PPO) can diff against that instead of
        a dedicated snapshot; :func:`verl.workers.engine.utils.hf_delta_export`
        implements the default pinned-CPU-snapshot strategy (primed by
        :meth:`prime_delta_snapshots`). The consuming delta engine only batches,
        gathers and ships.

        Returns:
            Generator: A generator that yields per-parameter HF-coordinate delta entries.
            Optional[dict]: Optional peft config.
        """
        raise NotImplementedError

    def get_data_parallel_size(self):
        # [FUNCTION DOC]
        # Abstract: return the data-parallel world size. Raises NotImplementedError.
        raise NotImplementedError

    def get_data_parallel_rank(self):
        # [FUNCTION DOC]
        # Abstract: return this rank's data-parallel rank. Raises NotImplementedError.
        raise NotImplementedError

    def get_data_parallel_group(self):
        # [FUNCTION DOC]
        # Abstract: return the data-parallel process group. Raises NotImplementedError.
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        # [FUNCTION DOC]
        # Base device-move hook; subclasses perform the actual moves and should call this for validation.
        # Behavior: only validates that grad implies model.
        # Inputs: device - target device; model, optimizer, grad - which components to move.
        # Outputs: none.
        # Exceptions: AssertionError when grad is true but model is false.
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
            grad: If True, move the gradient buffer.
        """
        if grad:
            assert model, "Gradient buffers must be moved to device along with model parameters"

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC]
        # Abstract: save model/optimizer/scheduler state to local_path (optionally upload to hdfs_path), keeping at most max_ckpt_to_keep checkpoints. Raises NotImplementedError.
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        # [FUNCTION DOC]
        # Abstract: load model/optimizer/scheduler state from local_path (or hdfs_path), optionally deleting the local copy afterwards. Raises NotImplementedError.
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        # [FUNCTION DOC]
        # Abstract: whether this rank is the first rank of the model-parallel group that holds model outputs (used to decide who reports metrics). Raises NotImplementedError.
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        raise NotImplementedError

    def disable_adapter(self) -> ContextManager:
        # [FUNCTION DOC]
        # Returns a context manager that temporarily disables LoRA adapters; default is a no-op null context.
        """
        Disable all adapters temporarily under the context in the model for LoRA
        """
        return nullcontext()


class BaseEngineCtx:
    # [CLASS DOC]
    # Base context manager used by engines' train_mode/eval_mode: sets engine.mode, loads offloaded state onto the device on entry and offloads to CPU on exit.
    def __init__(self, engine: BaseEngine, mode, **kwargs):
        # [FUNCTION DOC]
        # Stores the engine and mode and pops context options.
        # Inputs: engine - BaseEngine; mode - "train" or "eval"; kwargs - disable_auto_offload (default False) and zero_grad_on_exit (default True) are popped; other kwargs ignored.
        # Exceptions: AssertionError if mode is not "train" or "eval".
        """Base Engine context that handles load and offload

        Args:
            engine:
            **kwargs:
        """
        self.engine = engine
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)
        self.zero_grad_on_exit = kwargs.pop("zero_grad_on_exit", True)

    def _context_switch(self, device):
        # [FUNCTION DOC]
        # HELPER: Moves engine state to the given device according to mode and offload settings.
        # Behavior: no-op if auto offload is disabled; for a non-CPU target also no-op when neither param nor optimizer offload is enabled. In eval mode moves only the model (if param offload enabled), never optimizer or grads; in train mode moves model and grads per param offload and optimizer per optimizer offload.
        # Inputs: device - target device string.
        # Side effects: calls engine.to, moving tensors between host and device.
        if self.disable_auto_offload:
            return
        if device != "cpu":
            if not self.engine.is_param_offload_enabled and not self.engine.is_optimizer_offload_enabled:
                return
        if self.mode == "eval":
            self.engine.to(device=device, model=self.engine.is_param_offload_enabled, optimizer=False, grad=False)
        elif self.mode == "train":
            self.engine.to(
                device=device,
                model=self.engine.is_param_offload_enabled,
                optimizer=self.engine.is_optimizer_offload_enabled,
                grad=self.engine.is_param_offload_enabled,
            )

    def __enter__(self):
        # [FUNCTION DOC]
        # Sets engine.mode to this context's mode and loads offloaded state onto the accelerator device. Returns None (no bound value).
        self.engine.mode = self.mode
        self._context_switch(get_device_name())

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC]
        # Offloads state to CPU (per offload settings) and resets engine.mode to None. Does not suppress exceptions.
        self._context_switch("cpu")
        self.engine.mode = None


class EngineRegistry:
    # [CLASS DOC]
    # Global registry of engine classes keyed by model_type, then backend, then a device key (device string or (device, vendor) tuple). Stored in the class-level _engines mapping.
    """
    A registry for managing and instantiating different types of training engines.

    This class uses a dictionary to store engine classes, mapping a string key to each class.
    It provides a decorator `register` to add new engines to the registry and a `new` method
    to create an instance of a registered engine.
    """

    _engines = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        backend: list[str] | str,
        device: list[str] | str = "cuda",
        vendor: list[str] | str | None = None,
    ):
        # [FUNCTION DOC]
        # Class-method decorator factory registering an engine class.
        # Behavior: the returned decorator asserts the class subclasses BaseEngine, normalizes backend/device/vendor to lists (vendor None or empty becomes a single None), and for each backend x device x vendor stores the class under key device (no vendor) or (device, vendor).
        # Inputs: model_type; backend - str or list; device - str or list (default "cuda"); vendor - optional str or list.
        # Outputs: the decorator, which returns the class unchanged.
        # Exceptions: AssertionError on non-BaseEngine class or duplicate key.
        # Side effects: mutates the global registry.
        """
        A class method decorator that registers an engine class with a given key.

        This allows for dynamic instantiation of engine classes by their registered key.

        Args:
            model_type (str): The type of the model
            backend (list[str] | str): The backend to use for the model type
            device (list[str] | str): The device type (e.g., "cuda", "npu", "cpu") this engine supports,
                default is "cuda"
            vendor (list[str] | str | None): The hardware vendor (e.g., "nvidia", "metax") this engine
                supports. If None, the engine is registered as the default for the device type.

        Returns:
            A decorator function that takes an engine class and registers it.
        """

        def decorator(engine_class):
            # [FUNCTION DOC]
            # HELPER: Registers engine_class under all requested keys and returns it (see register).
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            vendors = vendor if isinstance(vendor, list) else ([vendor] if vendor else [None])
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    for current_vendor in vendors:
                        key = (current_device, current_vendor) if current_vendor else current_device
                        assert key not in cls._engines[model_type][current_backend], (
                            f"The key(device-vendor: {key}) has been already registed!"
                        )
                        cls._engines[model_type][current_backend][key] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        # [FUNCTION DOC]
        # Resolves the engine class for the current hardware.
        # Behavior: detects device name and vendor, overridable by env vars VERL_ENGINE_DEVICE and VERL_ENGINE_VENDOR; looks up (device, vendor), then device alone, then for cuda with a non-nvidia vendor falls back to (cuda, "nvidia").
        # Inputs: model_type, backend - registry keys.
        # Outputs: the registered engine class.
        # Exceptions: AssertionError for unknown model_type or backend; ValueError when no device/vendor match.
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        vendor = get_vendor()
        # Allow environment variables to override detected device and vendor for engine selection, if set
        if os.getenv("VERL_ENGINE_DEVICE"):
            device = os.getenv("VERL_ENGINE_DEVICE")
        if os.getenv("VERL_ENGINE_VENDOR"):
            vendor = os.getenv("VERL_ENGINE_VENDOR")
        registry = cls._engines[model_type][backend]

        # Try vendor-specific lookup: (device, vendor)
        vendor_key = (device, vendor)
        if vendor_key in registry:
            return registry[vendor_key]

        # Fallback to device-only key (registered without vendor)
        if device in registry:
            return registry[device]

        # For cuda-compatible vendors without a specific registration, try nvidia
        if device == "cuda" and vendor != "nvidia":
            nvidia_key = (device, "nvidia")
            if nvidia_key in registry:
                return registry[nvidia_key]

        raise ValueError(
            f"No engine registered for device={device!r}, vendor={vendor!r}, "
            f"model_type={model_type!r}, backend={backend!r}"
        )

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        # [FUNCTION DOC]
        # Instantiates the engine class resolved by get_engine_cls with the given args/kwargs and returns the instance.
        # KNOWN ISSUE: docstring mentions a "key" config argument and NotImplementedError, but the method takes model_type/backend and raises AssertionError/ValueError on lookup failure.
        """
        Function to create a new training engine instance based on the provided config.
        Args:
            key: A configuration object containing the engine key and other settings.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            engine: An instance of the training engine corresponding to the config.
        Raises:
            NotImplementedError: If the engine key in the config does not match any known engines.
        """
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)
