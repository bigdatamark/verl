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
# Integration with Huawei msprobe's PrecisionDebugger for dumping model tensors per training stage.
# Maps stage names (actor_update, actor_compute_log_prob, ref_compute_log_prob, compute_values,
# critic_update, compute_rm_score) to roles and, per role, to candidate dotted attribute paths on a
# worker where the model can be found. "rollout_generate" is always skipped. Dumps land under
# <save_path>/step_<n>/<stage>.
import functools
import logging
import os
from dataclasses import asdict
from typing import Optional

from verl.utils.import_utils import is_msprobe_available
from verl.utils.profiler.config import PrecisionDebuggerToolConfig

logger = logging.getLogger(__name__)

_STAGE_TO_ROLE = {
    "actor_update": "actor",
    "actor_compute_log_prob": "actor",
    "ref_compute_log_prob": "ref",
    "compute_values": "critic",
    "critic_update": "critic",
    "compute_rm_score": "reward_model",
}

_MODEL_ATTRS_BY_ROLE = {
    "actor": (
        "actor.engine.module",
        "actor.actor_module",
        "actor.actor_module_fsdp",
        "actor_module_fsdp",
        "actor_module",
    ),
    "ref": (
        "ref.engine.module",
        "ref.actor_module",
        "ref_policy.actor_module",
        "ref_module_fsdp",
        "ref_module",
        "ref_policy.ref_module",
    ),
    "critic": (
        "critic.engine.module",
        "critic.critic_module",
        "critic_module_fsdp",
        "critic_module",
    ),
    "reward_model": (
        "reward_model.engine.module",
        "reward_model_module_fsdp",
        "reward_model_module",
        "rm.reward_model_module",
    ),
}

_SKIP_STAGES = {"rollout_generate"}


class PrecisionDebuggerProfiler:
    # [CLASS DOC]
    # Wraps a single lazily-created msprobe PrecisionDebugger. Provides start/stop around a stage and an
    # annotate decorator for worker methods that resolves the model and global step from the worker
    # instance/arguments. Non-strict mode swallows failures (returns False); strict mode raises.
    """Minimal msprobe PrecisionDebugger integration."""

    def __init__(self, precision_cfg, rank: Optional[int] = None, save_path: Optional[str] = None):
        # [FUNCTION DOC]
        # Behavior: normalizes precision_cfg into a PrecisionDebuggerToolConfig, records msprobe availability,
        # normalizes the stage filter, sets dump root (save_path or "outputs/profile"), and warns if the
        # deprecated steps field is set.
        # Inputs: precision_cfg - None, config dataclass, DictConfig, dict, or other dataclass; rank; save_path.
        # Exceptions: ValueError from stage normalization in strict mode with unknown stages.
        # Side effects: may log a warning.
        self.rank = rank
        self.precision_cfg = self._normalize_config(precision_cfg)
        self._available = is_msprobe_available()
        self._debugger = None
        self._stages = self._normalize_stages(self.precision_cfg.stages)
        self._current_global_step = None
        self._dump_root = save_path or "outputs/profile"
        if self.precision_cfg.steps is not None:
            logger.warning(
                "`precision_debugger.steps` is deprecated and ignored. "
                "Use `global_profiler.steps` to control profiling steps."
            )

    @staticmethod
    def _normalize_config(precision_cfg) -> PrecisionDebuggerToolConfig:
        # [FUNCTION DOC]
        # HELPER: Converts input to PrecisionDebuggerToolConfig: None -> defaults; same type returned as-is;
        # objects with to_container (OmegaConf) resolved to a container first; dicts expanded as kwargs; else
        # treated as a dataclass via asdict.
        # Exceptions: TypeError for unexpected keys or non-dataclass input.
        if precision_cfg is None:
            return PrecisionDebuggerToolConfig()
        if isinstance(precision_cfg, PrecisionDebuggerToolConfig):
            return precision_cfg
        if hasattr(precision_cfg, "to_container"):
            precision_cfg = precision_cfg.to_container(resolve=True)
        if isinstance(precision_cfg, dict):
            return PrecisionDebuggerToolConfig(**precision_cfg)
        return PrecisionDebuggerToolConfig(**asdict(precision_cfg))

    @staticmethod
    def _normalize_stage(stage: Optional[str]) -> Optional[str]:
        # [FUNCTION DOC]
        # HELPER: Identity hook for stage-name normalization; returns stage unchanged.
        return stage

    def _normalize_stages(self, stages: Optional[list[str]]) -> Optional[set[str]]:
        # [FUNCTION DOC]
        # HELPER: Builds the stage filter set.
        # Behavior: None -> None (all stages). Otherwise normalizes, warns and drops skipped stages, warns about
        # (or raises on, if strict) unknown stages, and returns the intersection with known stages.
        # Exceptions: ValueError for unknown stages when strict. Side effects: logging.
        if stages is None:
            return None
        normalized = {self._normalize_stage(stage) for stage in stages}
        if _SKIP_STAGES & normalized:
            logger.warning("Ignoring precision_debugger stages: %s", sorted(_SKIP_STAGES & normalized))
        normalized = normalized - _SKIP_STAGES
        unknown = normalized - set(_STAGE_TO_ROLE.keys())
        if unknown:
            msg = f"Unknown precision_debugger stages: {sorted(unknown)}"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
        return normalized & set(_STAGE_TO_ROLE.keys())

    @staticmethod
    def _resolve_attr(obj, attr_path: str):
        # [FUNCTION DOC]
        # HELPER: Follows a dotted attribute path from obj, returning None as soon as any segment is missing
        # or None.
        current = obj
        for part in attr_path.split("."):
            current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _is_valid_model(model) -> bool:
        # [FUNCTION DOC]
        # HELPER: True when model is not None and has a callable forward attribute.
        return model is not None and callable(getattr(model, "forward", None))

    def _get_candidate_attrs(self, stage: str) -> tuple[str, ...]:
        # [FUNCTION DOC]
        # HELPER: Returns the candidate model attribute paths for the stage's role, or empty tuple if unknown.
        role = _STAGE_TO_ROLE.get(stage)
        if role is None:
            return ()
        return _MODEL_ATTRS_BY_ROLE.get(role, ())

    def _resolve_model(self, self_instance, stage: str):
        # [FUNCTION DOC]
        # HELPER: Locates the model for a stage on a worker instance.
        # Behavior: for each candidate path, returns the resolved value if it is a valid model; if it is a
        # list/tuple (e.g. Megatron chunks) returns the first valid element, warning when several exist. Falls
        # back to self_instance.module if valid, else None.
        # Side effects: may log a warning.
        for attr in self._get_candidate_attrs(stage):
            value = self._resolve_attr(self_instance, attr)
            if self._is_valid_model(value):
                return value

            # Megatron stores model chunks in ``engine.module``. msprobe's
            # PrecisionDebugger accepts one module, so bind the first chunk
            # that can be called directly.
            if isinstance(value, list | tuple):
                models = [model for model in value if self._is_valid_model(model)]
                if models:
                    if len(models) > 1:
                        logger.warning(
                            "PrecisionDebugger only binds the first of %d model chunks for stage '%s'",
                            len(models),
                            stage,
                        )
                    return models[0]
        fallback = getattr(self_instance, "module", None)
        return fallback if self._is_valid_model(fallback) else None

    def _resolve_global_step(self, self_instance, args, kwargs):
        # [FUNCTION DOC]
        # HELPER: Determines the global step for a call.
        # Behavior: scans positional then keyword argument values: an object whose meta_info dict has
        # "global_steps", or a dict with "global_steps", wins. Then checks instance attributes global_step and
        # _global_step. Falls back to the last step recorded via start().
        # Outputs: step value or None.
        for val in list(args) + list(kwargs.values()):
            if hasattr(val, "meta_info"):
                meta = val.meta_info
                if isinstance(meta, dict) and "global_steps" in meta:
                    return meta.get("global_steps")
            if isinstance(val, dict) and "global_steps" in val:
                return val.get("global_steps")
        for attr in ("global_step", "_global_step"):
            if hasattr(self_instance, attr):
                return getattr(self_instance, attr)
        return self._current_global_step

    def _should_collect(self, stage: str, global_step: Optional[int]) -> bool:
        # [FUNCTION DOC]
        # HELPER: Decides whether a stage should be collected (global_step currently unused).
        # Behavior: False for skipped stages; unknown stages warn (or raise if strict) and return False; False if
        # a stage filter exists and excludes the stage; else True.
        # Exceptions: ValueError for unknown stage in strict mode.
        if stage in _SKIP_STAGES:
            return False
        if stage not in _STAGE_TO_ROLE:
            msg = f"Unknown precision_debugger stage: {stage}"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False
        if self._stages is not None and stage not in self._stages:
            return False
        return True

    def start(self, stage: Optional[str] = None, global_step: Optional[int] = None, model=None, **kwargs) -> bool:
        # [FUNCTION DOC]
        # Behavior: records global_step/profile_step kwargs as the current step; returns False if stage is None,
        # not collectable, msprobe unavailable (ImportError if strict), config_path or dump root missing, or the
        # model invalid (ValueError if strict, else warning). Otherwise creates dump_path
        # <root>/step_<n or unknown>/<stage>, lazily creates the debugger with config_path and dump_path,
        # updates the debugger service's dump_path for subsequent stages, and starts it on the model.
        # Outputs: True if collection started, else False.
        # Exceptions: in strict mode the above errors and any exception during setup propagate; otherwise
        # exceptions are swallowed and False returned.
        # Side effects: directory creation, debugger state, logging.
        profile_step = kwargs.get("global_step", kwargs.get("profile_step"))
        if profile_step is not None:
            self._current_global_step = profile_step
        stage = self._normalize_stage(stage)
        if stage is None:
            return False
        if global_step is None:
            global_step = self._current_global_step
        if not self._should_collect(stage, global_step):
            return False
        if not self._available:
            if self.precision_cfg.strict:
                raise ImportError("msprobe is not available but precision_debugger.strict is True")
            return False
        if not self.precision_cfg.config_path or not self._dump_root:
            return False
        if not self._is_valid_model(model):
            msg = f"PrecisionDebugger model not resolved for stage '{stage}'"
            if self.precision_cfg.strict:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        try:
            from msprobe.pytorch import PrecisionDebugger

            step_tag = f"step_{global_step}" if global_step is not None else "step_unknown"
            dump_path = os.path.join(self._dump_root, step_tag, stage)
            os.makedirs(dump_path, exist_ok=True)

            if self._debugger is None:
                self._debugger = PrecisionDebugger(config_path=self.precision_cfg.config_path, dump_path=dump_path)
                if self._debugger is None:
                    if self.precision_cfg.strict:
                        raise RuntimeError("Failed to create PrecisionDebugger instance")
                    return False
            if hasattr(self._debugger, "service") and hasattr(self._debugger.service, "config"):
                self._debugger.service.config.dump_path = dump_path
            self._debugger.start(model)
            return True
        except Exception:
            if self.precision_cfg.strict:
                raise
            return False

    def stop(self, started: bool = False) -> None:
        # [FUNCTION DOC]
        # Behavior: if started, msprobe available, and a debugger exists, stops it and resets its service status.
        # Inputs: started - value returned by start. Side effects: debugger state.
        if not started:
            return
        if not self._available:
            return
        if self._debugger is None:
            return
        self._debugger.stop()
        self._reset_debugger_status()

    def annotate(
        self,
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs_outer,
    ):
        # [FUNCTION DOC]
        # Behavior: message/color/domain/category are ignored. Stage is taken from the "role" kwarg; if absent,
        # returns an identity decorator. Otherwise returns a decorator whose wrapper (for worker methods)
        # resolves the global step and model, starts collection, calls the function, and always stops in finally.
        # Outputs: decorator.
        _ = (message, color, domain, category)
        stage = self._normalize_stage(kwargs_outer.get("role"))
        if stage is None:
            return lambda func: func

        def decorator(func):
            # [FUNCTION DOC]
            # HELPER: Builds the functools.wraps wrapper around func.
            @functools.wraps(func)
            def wrapper(self_instance, *args, **kwargs_inner):
                # [FUNCTION DOC]
                # HELPER: Worker-method wrapper; start before, stop after (finally); returns func's result.
                global_step = self._resolve_global_step(self_instance, args, kwargs_inner)
                model = self._resolve_model(self_instance, stage)
                started = self.start(stage=stage, global_step=global_step, model=model)
                try:
                    return func(self_instance, *args, **kwargs_inner)
                finally:
                    self.stop(started=started)

            return wrapper

        return decorator

    def _reset_debugger_status(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls the debugger service's reset_status, or else _reset_status, when present and callable;
        # no-op if there is no service.
        service = getattr(self._debugger, "service", None)
        if service is None:
            return

        reset_status = getattr(service, "reset_status", None)
        if callable(reset_status):
            reset_status()
            return

        reset_status = getattr(service, "_reset_status", None)
        if callable(reset_status):
            reset_status()
