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
# Tinker-style training workers: extend TrainingWorker and ActorRolloutRefWorker with split
# training primitives (zero grad, forward/backward only, optimizer step with optional global
# param-group overrides) so a caller can control each phase explicitly.
from typing import TypedDict

from codetiming import Timer
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.profiler import DistProfiler
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker


class OptimStepParams(TypedDict, total=False):
    # [CLASS DOC]
    # Optional-key typed mapping of per-step optimizer overrides: lr, eps, betas, weight_decay.
    """Runtime param-group override for Tinker-style optimizer steps.

    This payload is only consumed by ``TinkerTrainingWorker.optimizer_step``. It is
    not part of the generic engine API, does not participate in optimizer
    construction, and does not configure LR scheduler behavior. Tinker callers
    that own scheduling should pass the resulting optimizer values here.

    The current implementation applies each provided key to all optimizer param
    groups before one explicit optimizer step. For VeOmni MultiOptimizer, the
    Tinker worker flattens the wrapped optimizers' param groups and applies the
    same global override. Optimizer-specific or group-specific overrides require a
    different payload shape.
    """

    lr: float
    eps: float
    betas: tuple[float, float]
    weight_decay: float


def _iter_optimizer_param_groups(optimizer):
    # [FUNCTION DOC]
    # HELPER: Collects param groups from the optimizer, or from each child of a VeOmni multi-optimizer
    # (flag _is_multi_optimizer, children in optimizers_dict), into one flat list (same group objects).
    # Exceptions: NotImplementedError when an optimizer lacks param_groups.
    """Return a flat list of param groups, including VeOmni MultiOptimizer children."""
    if getattr(optimizer, "_is_multi_optimizer", False):
        optimizers = optimizer.optimizers_dict.values()
    else:
        optimizers = [optimizer]

    param_groups = []
    for opt in optimizers:
        opt_param_groups = getattr(opt, "param_groups", None)
        if opt_param_groups is None:
            raise NotImplementedError(
                f"{type(opt).__name__} does not expose param_groups for per-step optimizer params"
            )
        param_groups.extend(opt_param_groups)
    return param_groups


def _apply_optim_step_params(optimizer, optim_step_params: OptimStepParams | None) -> None:
    # [FUNCTION DOC]
    # HELPER: Applies non-None override values to every param group.
    # Behavior: no-op for None; converts objects with to_dict; requires a dict; drops None values and
    # returns if nothing remains. Gets flat param groups (must be non-empty). For each key, requires the
    # key in the first group, takes that value's type, requires the override to be an instance of it,
    # and requires every group to have the key with the same type. Then updates every group with all
    # overrides.
    # Inputs: optimizer; optim_step_params - mapping or None.
    # Exceptions: TypeError on non-dict input or type mismatch; ValueError on missing/unsupported keys.
    # Side effects: mutates optimizer param groups.
    # KNOWN ISSUE: strict type check rejects an int lr against a float group value, or a list betas
    # against a tuple (common after serialization).
    """Apply a Tinker step-time override to every optimizer param group.

    The override is intentionally global: every provided key must exist with the
    same value type on all param groups. This keeps mixed optimizers such as
    VeOmni Muon+AdamW fail-fast for optimizer-specific keys like ``betas`` while
    still allowing shared keys such as ``lr``.
    """
    if optim_step_params is None:
        return

    if hasattr(optim_step_params, "to_dict"):
        optim_step_params = optim_step_params.to_dict()
    if not isinstance(optim_step_params, dict):
        raise TypeError(f"optim_step_params must be a dict, got {type(optim_step_params)}")

    normalized_params = {key: value for key, value in optim_step_params.items() if value is not None}
    if not normalized_params:
        return

    param_groups = _iter_optimizer_param_groups(optimizer)
    if not param_groups:
        raise ValueError(f"{type(optimizer).__name__} does not have param_groups")

    for key, value in normalized_params.items():
        if key not in param_groups[0]:
            raise ValueError(f"{type(optimizer).__name__} does not support optim_step_params key: {key!r}")

        expected_type = type(param_groups[0][key])
        if not isinstance(value, expected_type):
            raise TypeError(
                f"optim_step_params type mismatch for {type(optimizer).__name__}: "
                f"{key!r} got {type(value).__name__}, expected {expected_type.__name__}"
            )

        for param_group in param_groups:
            if key not in param_group:
                raise ValueError(f"{type(optimizer).__name__} has inconsistent param_group key: {key!r}")
            if not isinstance(param_group[key], expected_type):
                raise TypeError(f"{type(optimizer).__name__} has inconsistent param_group type for {key!r}")

    for param_group in param_groups:
        param_group.update(normalized_params)


class TinkerTrainingWorker(TrainingWorker):
    # [CLASS DOC]
    # TrainingWorker subclass exposing optimizer_zero_grad, forward_backward and optimizer_step.
    """
    Training worker exposing Tinker-style split training primitives.

    Unlike TrainingWorker.train_batch(), these APIs let a caller explicitly separate gradient
    clearing, forward/backward, and optimizer stepping.
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_zero_grad(self) -> None:
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) within train mode (no zero grad on exit) zeroes engine grads.
        with self.engine.train_mode(zero_grad_on_exit=False):
            self.engine.optimizer_zero_grad()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="forward_backward")
    def forward_backward(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, train-mesh, non-blocking, profiled) asserts a loss fn exists and engine is not
        # forward-only; reads token counts, disable_auto_offload, images_seqlens; fills missing batching
        # defaults (as TrainingWorker.train_batch); sets return_model_output true; fixes 3-D position ids;
        # runs engine.forward_backward_batch (not forward-only) in train mode without zeroing grads on exit,
        # timed. Output-holder ranks return the postprocessed CPU TensorDict (includes model outputs);
        # others None. Does not step the optimizer.
        # Exceptions: AssertionError on preconditions.
        # Side effects: accumulates gradients; mutates data.
        assert self.loss_fn is not None, "loss function can't be None when calling forward_backward"
        assert not self.engine_config.forward_only, (
            "Can't run `forward_backward` when forward_only is in the engine config."
        )
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # Tinker returns per-token log-probs to the client after backward.
        # FSDP normally discards model_output to reduce peak memory, so
        # explicitly opt this split-training path into retaining it.
        tu.assign_non_tensor(data, return_model_output=True)

        maybe_fix_3d_position_ids(data)

        with (
            self.engine.train_mode(
                disable_auto_offload=disable_auto_offload,
                zero_grad_on_exit=False,
            ),
            Timer(name="forward_backward", logger=None) as timer,
        ):
            output = self.engine.forward_backward_batch(data, loss_function=self.loss_fn, forward_only=False)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, optim_step_params: OptimStepParams | None = None) -> dict:
        # [FUNCTION DOC]
        # Behavior: (remote, one-to-all) in train mode with zero grad on exit, applies overrides to the
        # engine optimizer and steps it; returns a dict with grad_norm on output-holder ranks when the
        # step returned one, else empty.
        # Exceptions: those of the override application.
        """Run one Tinker optimizer step with an optional global param-group override."""
        with self.engine.train_mode(zero_grad_on_exit=True):
            _apply_optim_step_params(self.engine.optimizer, optim_step_params)
            grad_norm = self.engine.optimizer_step()

        metrics = {}
        if grad_norm is not None and self.engine.is_mp_src_rank_with_outputs():
            metrics["grad_norm"] = grad_norm
        return metrics


class TinkerActorRolloutRefWorker(ActorRolloutRefWorker):
    # [CLASS DOC]
    # Hybrid worker using TinkerTrainingWorker as actor and forwarding split primitives to it.
    """Actor-rollout-ref worker exposing Tinker-style split training primitives for the actor."""

    actor_worker_cls = TinkerTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_zero_grad(self) -> None:
        # [FUNCTION DOC]
        # Behavior: (remote) asserts actor role and forwards zero grad to the actor.
        assert "actor" in self.role, "optimizer_zero_grad only support actor role"
        return self.actor.optimizer_zero_grad()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"), blocking=False)
    def forward_backward(self, data: TensorDict) -> TensorDict:
        # [FUNCTION DOC]
        # Behavior: (remote, actor mesh, non-blocking) asserts actor role, runs actor forward_backward,
        # returns CPU output or None.
        assert "actor" in self.role, "forward_backward only support actor role"
        output = self.actor.forward_backward(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def optimizer_step(self, optim_step_params: OptimStepParams | None = None) -> dict:
        # [FUNCTION DOC]
        # Behavior: (remote) asserts actor role and forwards optimizer step, returning its metrics.
        assert "actor" in self.role, "optimizer_step only support actor role"
        return self.actor.optimizer_step(optim_step_params=optim_step_params)
