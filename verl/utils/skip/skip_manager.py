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
# Process-global SkipManager: builds skip instances from config for every registered skip, tracks the
# current training step, and provides decorators that wrap rollout methods so configured steps are
# served from cache (or dumped to cache on a miss). Validation calls always bypass skipping.

import functools
import inspect
from typing import Callable

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.skip.base_skip import SKIP_REGISTRY
from verl.utils.skip.config import SkipManagerConfig


class SkipManager:
    # [CLASS DOC]
    # Class-level state only (never instantiated): config (SkipManagerConfig or None), step (current
    # trainer step shared by skips without online step extraction, default -1) and skip_instances
    # (role name to skip instance). Decorators are no-ops until init is called.
    """SkipManager is a manager for skip.

    Class attributes default here so code paths (e.g. tests or modules that only import
    ``@SkipManager.annotate``) work **before** ``SkipManager.init(config)`` runs — decorators
    then no-op until the trainer initializes skip state.
    """

    config: SkipManagerConfig | None = None
    step: int = -1
    # This step is shared across all skip_instances
    # Different enabled skip_instances (no online step acquisition) shall use the same step definition.
    skip_instances: dict = {}  # noqa: RUF012 — intentionally mutable class defaults, reset in ``init``

    @classmethod
    def init(cls, config):
        # [FUNCTION DOC]
        # Converts config.skip to SkipManagerConfig, resets step to -1 and instances, then for each registry
        # entry whose name is a field of the skip config, instantiates the skip with (sub-config, full config).
        # Exceptions: config conversion errors; ValueError from skips with unsupported actions.
        cls.config = omega_conf_to_dataclass(config.skip, dataclass_type=SkipManagerConfig)
        cls.step = -1
        cls.skip_instances = {}
        for name, skip_cls in SKIP_REGISTRY.items():
            local_cfg = getattr(cls.config, name, None)
            if local_cfg is None:
                continue
            instance = skip_cls(local_cfg, config)
            cls.skip_instances[name] = instance

    @classmethod
    def set_step(cls, step: int):
        # [FUNCTION DOC]
        # Sets the shared current step.
        cls.step = step

    @staticmethod
    def _get_prompts_batch(args, kwargs):
        # [FUNCTION DOC]
        # HELPER: Finds the prompts batch in a decorated call: the "prompts" keyword, else the second
        # positional argument (after self), else a sole positional argument that has meta_info; else None.
        """Resolve ``DataProto`` from decorated ``generate_sequences(self, prompts, ...)`` calls."""
        prompts = kwargs.get("prompts")
        if prompts is not None:
            return prompts
        if len(args) > 1:
            return args[1]
        if len(args) == 1 and hasattr(args[0], "meta_info"):
            return args[0]
        return None

    @classmethod
    def _should_bypass_for_validation(cls, args, kwargs) -> bool:
        # [FUNCTION DOC]
        # HELPER: True if the call is a validation call. Uses the TensorDict check when the prompts object has
        # non_tensor_data (or when not found but more than one positional arg); otherwise reads
        # meta_info["validate"] (default False). No prompts -> False.
        prompts = cls._get_prompts_batch(args, kwargs)
        if prompts is None and len(args) > 1:
            return cls._should_bypass_for_validation_tensordict(args, kwargs)
        if prompts is None:
            return False
        if hasattr(prompts, "non_tensor_data"):
            return cls._should_bypass_for_validation_tensordict(args, kwargs)
        meta_info = getattr(prompts, "meta_info", None) or {}
        return bool(meta_info.get("validate", False))

    @classmethod
    def annotate(cls, role: str, **kwargs_outer) -> Callable:
        # [FUNCTION DOC]
        # Returns a decorator for a rollout method associated with role (extra keyword args ignored). The
        # wrapper (async for coroutine functions, otherwise sync) calls the original function directly when:
        # validation call, no enabled skip instance for role, or the step (extracted online if supported,
        # else the shared step) is not in the skip's configured steps. Otherwise, if meet_precondition holds,
        # returns warp_function's cached result without calling the function; else calls the function, passes
        # its result to prepare_data (dump), and returns it.
        def decorator(func: Callable) -> Callable:
            # [FUNCTION DOC]
            # HELPER: Wraps func with the async or sync skip wrapper described in annotate.
            if inspect.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs_inner):
                    # [FUNCTION DOC]
                    # HELPER: Async skip wrapper; awaits the original function when not skipping.
                    if cls._should_bypass_for_validation(args, kwargs_inner):
                        return await func(*args, **kwargs_inner)
                    skip_instance = cls.skip_instances.get(role)
                    if skip_instance is None or not skip_instance.is_enabled():
                        return await func(*args, **kwargs_inner)
                    if skip_instance.support_online_step:
                        step = skip_instance.extract_step(*args, **kwargs_inner)
                    else:
                        step = cls.step
                    if step not in skip_instance.steps:
                        return await func(*args, **kwargs_inner)
                    if skip_instance.meet_precondition(step, func, *args, **kwargs_inner):
                        return skip_instance.warp_function(step, func, *args, **kwargs_inner)
                    result = await func(*args, **kwargs_inner)
                    skip_instance.prepare_data(step, result, *args, **kwargs_inner)
                    return result

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs_inner):
                # [FUNCTION DOC]
                # HELPER: Synchronous skip wrapper.
                if cls._should_bypass_for_validation(args, kwargs_inner):
                    return func(*args, **kwargs_inner)
                skip_instance = cls.skip_instances.get(role)
                if skip_instance is None or not skip_instance.is_enabled():
                    return func(*args, **kwargs_inner)
                if skip_instance.support_online_step:
                    step = skip_instance.extract_step(*args, **kwargs_inner)
                else:
                    step = cls.step
                if step not in skip_instance.steps:
                    return func(*args, **kwargs_inner)
                if skip_instance.meet_precondition(step, func, *args, **kwargs_inner):
                    return skip_instance.warp_function(step, func, *args, **kwargs_inner)
                result = func(*args, **kwargs_inner)
                skip_instance.prepare_data(step, result, *args, **kwargs_inner)
                return result

            return sync_wrapper

        return decorator

    @classmethod
    def annotate_tq(cls, role: str, phase: str):
        # [FUNCTION DOC]
        # Returns a decorator for the V1 TransferQueue trainer methods. If no enabled instance for role or
        # the shared step is not configured, calls through. phase "submit": ignores the decorated function,
        # calls self._next_train_batch(); if the skip injects cached data for the batch uids, returns None;
        # otherwise calls self._submit_batch_to_rollout(batch) and returns None. Any other phase ("sample"):
        # calls the function, takes the first element if the result is a tuple, and if should_save for its
        # partition, dumps it via prepare_data(global_steps = step); returns the original result.
        # Inputs: role - skip registry name; phase - "submit" or "sample".
        """V1 TransferQueue-based skip decorator, unified across both phases.

        V1's split architecture separates "submit prompts" (``_add_batch_to_generate``)
        from "sample trajectories" (``ReplayBuffer.sample``).  This decorator handles
        both, selected by *phase*:

        - ``phase="submit"``: decorate ``_add_batch_to_generate``.  The method must be
          split into ``_next_train_batch`` (dataloader + uid) and ``_submit_batch_to_rollout``
          (tag registration + generate_sequences).  The decorator always calls
          ``_next_train_batch`` first (keeping the dataloader aligned even on cache-hit
          steps), then checks ``meet_precondition``.  On cache-hit it injects cached data
          into the TQ and returns; on cache-miss it calls ``_submit_batch_to_rollout``.

        - ``phase="sample"``: decorate ``ReplayBuffer.sample``.  After the original
          ``sample`` runs, if the skip instance says to save, persist the result to disk
          (phase one / cache-miss).  When ``parameter_sync_step > 1`` (separate async),
          ``sample`` is called multiple times per step; each call saves its mini-batch to
          a separate inner sub-directory, and the full set is merged on load.
        """

        def decorator(func: Callable) -> Callable:
            # [FUNCTION DOC]
            # HELPER: Wraps a method with the phase-specific TransferQueue skip wrapper.
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # [FUNCTION DOC]
                # HELPER: Method wrapper implementing the submit/sample phase logic described in annotate_tq.
                skip_instance = cls.skip_instances.get(role)
                if skip_instance is None or not skip_instance.is_enabled():
                    return func(self, *args, **kwargs)
                step = cls.step
                if step not in skip_instance.steps:
                    return func(self, *args, **kwargs)

                if phase == "submit":
                    # Always prepare batch (keeps dataloader aligned on cache-hit steps)
                    batch = self._next_train_batch()
                    if skip_instance.maybe_load_and_inject(step, list(batch["uid"])):
                        return
                    self._submit_batch_to_rollout(batch)
                    return

                # phase == "sample": post-save after original sample runs
                result = func(self, *args, **kwargs)
                # ReplayBuffer.sample returns (batch, off_policy_metrics)
                batch = result[0] if isinstance(result, tuple) else result
                if skip_instance.should_save(step, batch.partition_id):
                    skip_instance.prepare_data(step=step, batch=batch, global_steps=step)
                return result

            return wrapper

        return decorator

    @staticmethod
    def _should_bypass_for_validation_tensordict(args, kwargs) -> bool:
        # [FUNCTION DOC]
        # HELPER: Reads non_tensor_data["validate"] from the "prompts" keyword or second positional argument;
        # False if absent or on any error.
        """Check ``validate`` flag in TensorDict's non_tensor_data for V1 path."""
        prompts = kwargs.get("prompts", None)
        if prompts is None and len(args) > 1:
            prompts = args[1]
        if prompts is None:
            return False
        try:
            return bool(getattr(prompts, "non_tensor_data", {}).get("validate", False))
        except Exception:
            return False
