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
# Registers the "separate_async" PPO trainer: training GPUs ("hybrid" replicas, colocated with the actor) and a
# separate standalone rollout pool. Rollout is always served by the standalone pool; when idle the hybrid engine can
# be lent to generation and switched back at step boundaries. Contains an adaptive cost/benefit policy deciding when
# to switch the hybrid engine to rollout mode, and Decoupled-PPO old-log-prob handling via CPU weight snapshots.

import logging
import os
import time
from collections import deque
from dataclasses import replace
from enum import Enum

import ray
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.separation.engine_workers import DetachActorWorker
from verl.trainer.config import HybridRolloutSwitchConfig
from verl.trainer.ppo.utils import Role, need_reward_model
from verl.trainer.ppo.v1.trainer_base import PPOTrainer, register_trainer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerManager
from verl.workers.rollout.utils import update_prometheus_config

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HybridEngineMode(Enum):
    # [CLASS DOC]
    # Enum of the hybrid engine's current role: TRAINER (GPUs used for training) or ROLLOUT (GPUs serving generation).
    TRAINER = "trainer"
    ROLLOUT = "rollout"


@register_trainer("separate_async")
class PPOTrainerSeparateAsync(PPOTrainer):
    # [CLASS DOC]
    # Separate-async PPO trainer. Key state: current_mode (HybridEngineMode), standalone_server_manager and
    # standalone_checkpoint_manager for the standalone pool, hybrid_rollout_config (HybridRolloutSwitchConfig), and
    # switch-policy counters set by _init_hybrid_rollout_state. Class attribute _enable_hybrid_replicas defaults True for
    # instances that skip __init__.
    """Asynchronous PPO trainer
    1. Trainer and rollout are separate, trainer may switch to rollout if idle.
    2. Partial rollout is enabled.
    """

    # Default for instances that bypass __init__ (e.g. test stubs): hybrid
    # replicas are enabled unless actor_rollout_ref.hybrid_engine=False.
    _enable_hybrid_replicas = True

    def __init__(self, config: DictConfig):
        # [FUNCTION DOC]
        # Behavior: validates config then initializes the base trainer and switch config.
        # Validation (assertions): train_batch_size equals parameter_sync_step * ppo_mini_batch_size; rollout nnodes > 0 and
        # n_gpus_per_node > 0; checkpoint_engine.backend is not "naive"; if a reward model is needed, it must use its own
        # resource pool. Sets _enable_hybrid_replicas from actor_rollout_ref.hybrid_engine (default True) before calling the
        # base constructor (base _setup reads it). Converts trainer.v1.separate_async.hybrid_rollout to a dataclass; if hybrid
        # replicas are disabled but enable_switch is set, warns and forces enable_switch False. When switching is enabled:
        # rejects rollout disaggregation, requires the replay buffer to have wait_for_sampleable and get_sampleable_count,
        # then initializes switch-policy state.
        # Inputs: config - full DictConfig. Outputs: none.
        # Exceptions: AssertionError on invalid config; ValueError if disaggregation enabled with switching; TypeError if the
        # replay buffer lacks required methods.
        train_batch_size = config.data.train_batch_size
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        parameter_sync_step = config.trainer.v1.separate_async.parameter_sync_step
        assert train_batch_size == parameter_sync_step * ppo_mini_batch_size, (
            f"train_batch_size must equal parameter_sync_step * ppo_mini_batch_size in separate async "
            f"training, but got train_batch_size={train_batch_size}, "
            f"parameter_sync_step={parameter_sync_step}, ppo_mini_batch_size={ppo_mini_batch_size}"
        )
        assert config.actor_rollout_ref.rollout.nnodes > 0, "nnodes must be > 0 in separate async training"
        assert config.actor_rollout_ref.rollout.n_gpus_per_node > 0, (
            "n_gpus_per_node must be > 0 in separate async training"
        )
        assert config.actor_rollout_ref.rollout.checkpoint_engine.backend != "naive", (
            "please use nccl/nixl/mooncake, etc. backend for separate async training"
        )
        if need_reward_model(config):
            assert config.reward.reward_model.enable_resource_pool, (
                "Colocate reward model (reward.reward_model.enable_resource_pool=False) is not supported "
                "in separate async mode, because the standalone rollout never pauses to free GPU memory. "
                "Use standalone mode (reward.reward_model.enable_resource_pool=True) instead."
            )

        # actor_rollout_ref.hybrid_engine=False disables the colocated (hybrid)
        # rollout replicas on the training GPUs: rollout is then served
        # exclusively by the standalone rollout pool (v0 fully-async semantics).
        # Read by PPOTrainer._setup() (triggered by init()).
        self._enable_hybrid_replicas = bool(config.actor_rollout_ref.get("hybrid_engine", True))

        super().__init__(config)

        self.hybrid_rollout_config: HybridRolloutSwitchConfig = omega_conf_to_dataclass(
            self.config.trainer.v1.separate_async.hybrid_rollout
        )
        if not self._enable_hybrid_replicas and self.hybrid_rollout_config.enable_switch:
            logger.warning(
                "trainer.v1.separate_async.hybrid_rollout.enable_switch is ignored because "
                "actor_rollout_ref.hybrid_engine=False; disabling hybrid switching"
            )
            self.hybrid_rollout_config = replace(self.hybrid_rollout_config, enable_switch=False)
        if self.hybrid_rollout_config.enable_switch:
            # No support for PD disaggregation for switching
            rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
            disaggregation_cfg = rollout_cfg.get("disaggregation", {})
            if bool(disaggregation_cfg.get("enabled", False)):
                raise ValueError(
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch does not support rollout disaggregation: "
                    "step-boundary redistribution relies on paused replicas queueing new requests"
                )
            # Custom samplers own their polling. One that cannot wait on buffer depth would hand the
            # GPUs back before the step has anything to train on, which is worse than not switching.
            required_methods = ("wait_for_sampleable", "get_sampleable_count")
            if any(not hasattr(self.replay_buffer, method) for method in required_methods):
                raise TypeError(
                    f"{type(self.replay_buffer).__name__} must implement {required_methods} when "
                    "trainer.v1.separate_async.hybrid_rollout.enable_switch=True"
                )
            self._init_hybrid_rollout_state()

    def _init_resource_pool_mgr(self):
        # [FUNCTION DOC]
        # HELPER: Calls base resource-pool init, then replaces the actor worker class with a ray-remote DetachActorWorker
        # (which supports CPU save/restore of weights): for the Actor role (and RefPolicy if present), else ActorRolloutRef,
        # else ActorRollout. Inputs/Outputs: none. Side effects: mutates role_worker_mapping.
        super()._init_resource_pool_mgr()
        # Replace ActorRolloutRefWorker with DetachActorWorker to get CPU save/restore
        # capability needed for Decoupled PPO when parameter_sync_step > 1.
        # The base class uses a pure Actor role when hybrid replicas are disabled.
        if Role.Actor in self.role_worker_mapping:
            self.role_worker_mapping[Role.Actor] = ray.remote(DetachActorWorker)
            if Role.RefPolicy in self.role_worker_mapping:
                self.role_worker_mapping[Role.RefPolicy] = ray.remote(DetachActorWorker)
        elif Role.ActorRolloutRef in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(DetachActorWorker)
        elif Role.ActorRollout in self.role_worker_mapping:
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(DetachActorWorker)

    def _init_hybrid_rollout_state(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Initializes adaptive switch-policy state: current threshold ratio from config, idle/calm step counters,
        # per-step and cumulative wait time/sample counters, sample start timestamp, step threshold, bounded deques (size
        # switch_cost_window_size) of measured to-rollout and to-trainer switch costs, and a scaling factor equal to
        # (trainer GPUs + standalone rollout GPUs) / standalone rollout GPUs. Inputs/Outputs: none. Side effects: sets attributes.
        # Exceptions: division by zero if standalone GPU count is 0 (prevented by __init__ assertions).
        config = self.hybrid_rollout_config
        self._switch_threshold_ratio = config.switch_threshold_ratio
        self._idle_steps = 0
        self._calm_steps = 0
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._sample_start = time.perf_counter()
        self._step_threshold = 0
        self._wait_seconds = 0.0
        self._wait_samples = 0
        self._to_rollout_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        self._to_trainer_costs: deque[float] = deque(maxlen=config.switch_cost_window_size)
        rollout_cfg = self.config.get("actor_rollout_ref", {}).get("rollout", {})
        trainer_cfg = self.config.trainer
        hybrid_gpus = trainer_cfg.nnodes * trainer_cfg.n_gpus_per_node
        standalone_gpus = rollout_cfg.nnodes * rollout_cfg.n_gpus_per_node
        # TODO: Use a more accurate or dynamic scaling factor.
        self._scaling_factor = (hybrid_gpus + standalone_gpus) / standalone_gpus

    def _setup(self):
        # [FUNCTION DOC]
        # HELPER: Extends base setup: creates the standalone LLMServerManager with ranks starting after the hybrid replicas;
        # updates Prometheus config with both pools' server addresses if enabled; builds a CheckpointEngineManager linking the
        # actor worker group to standalone replicas. If hybrid replicas are enabled sets mode ROLLOUT and registers hybrid
        # servers in the standalone load balancer; otherwise sets mode TRAINER and logs.
        # Inputs/Outputs: none. Side effects: spawns rollout servers, sets attributes.
        # KNOWN ISSUE: the logger.info call passes flush=True, which standard logging does not accept and raises TypeError.
        super()._setup()

        # initialize standalone rollout
        # TODO: make initialization parallel with super().init()
        hybrid_num_replicas = len(self.llm_server_manager.rollout_replicas)
        self.standalone_server_manager: LLMServerManager = LLMServerManager.create(
            config=self.config, start_rank=hybrid_num_replicas
        )
        rollout_config = self.config.actor_rollout_ref.rollout
        if rollout_config.prometheus.enable:
            server_addresses = (
                self.llm_server_manager.server_addresses + self.standalone_server_manager.server_addresses
            )
            update_prometheus_config(rollout_config.prometheus, server_addresses, rollout_config.name)

        # create checkpoint engine manager for trainer and standalone rollout
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )

        # hybrid engine is in rollout mode after initialization
        if self._enable_hybrid_replicas:
            self.current_mode = HybridEngineMode.ROLLOUT
            self.add_replicas_to_balancer()
        else:
            # No colocated replicas exist on the training GPUs (hybrid_engine=False):
            # rollout is served exclusively by the standalone pool, so never enter
            # ROLLOUT mode; the mode-switch hooks and balancer updates become no-ops.
            self.current_mode = HybridEngineMode.TRAINER
            logger.info(
                "[V1SepAsync] hybrid replicas disabled (actor_rollout_ref.hybrid_engine=False): "
                f"rollout served by {len(self.standalone_server_manager.get_replicas())} standalone replicas only",
                flush=True,
            )

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        # [FUNCTION DOC]
        # HELPER: Overrides old-log-prob computation. If rollout_correction.bypass_mode is set, delegates to base. Otherwise
        # uses CPU snapshots keyed by local_trigger_step: at step 0 saves current weights as snapshot 0 (pi_old) and delegates;
        # at later steps saves current weights under the step key, restores snapshot 0, computes via base, restores current
        # weights, and clears the step snapshot.
        # Inputs: batch - KVBatchMeta; metrics - dict passed through. Outputs: KVBatchMeta from base computation.
        # Side effects: remote weight save/restore on actor workers.
        """Version-aware old_log_probs computation for Decoupled PPO.

        In bypass mode, delegates to the base class (copies rollout_log_probs directly).
        In Decoupled mode, uses save_model_to_cpu / restore_model_from_cpu to ensure
        all mini-batches within a parameter_sync_step cycle use the same stable π_old.

        - local_trigger_step == 0: Current weights are π_old → save to CPU, compute directly.
        - local_trigger_step >= 1: Save current weights → restore π_old → compute → restore current.
        """
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:
            return super()._compute_old_log_prob(batch, metrics)

        if self.local_trigger_step == 0:
            self.actor_rollout_wg.save_model_to_cpu(0)
            return super()._compute_old_log_prob(batch, metrics)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(0)
            result = super()._compute_old_log_prob(batch, metrics)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
            return result

    def get_llm_client(self):
        # [FUNCTION DOC]
        # Behavior: returns a FullyAsyncLLMServerClient from the standalone server manager (all generation routes there).
        # Inputs: none. Outputs: client.
        # get server client from standalone rollout
        return self.standalone_server_manager.get_client(client_cls=FullyAsyncLLMServerClient)

    def _rollout_server_managers(self) -> list:
        # [FUNCTION DOC]
        # HELPER: Returns the base list of rollout server managers plus the standalone manager if it exists.
        # Inputs: none. Outputs: list of managers.
        managers = super()._rollout_server_managers()
        standalone = getattr(self, "standalone_server_manager", None)
        if standalone is not None:
            managers.append(standalone)
        return managers

    def on_init_end(self):
        # [FUNCTION DOC]
        # Behavior: after init, syncs weights at global_steps to standalone replicas, and to hybrid replicas if enabled.
        # Inputs/Outputs: none. Side effects: weight sync.
        # update weights after loading checkpoint
        self.standalone_checkpoint_manager.update_weights(self.global_steps)
        if self._enable_hybrid_replicas:
            self.checkpoint_manager.update_weights(self.global_steps)

    def on_train_begin(self):
        # [FUNCTION DOC]
        # Behavior: schedules separate_async.num_warmup_batches async warmup batches. Inputs/Outputs: none.
        self._add_async_warmup_batches(self.config.trainer.v1.separate_async.num_warmup_batches)

    def on_validate_begin(self):
        # [FUNCTION DOC]
        # Behavior: before validation, if hybrid engine is in TRAINER mode, switches it to rollout. Inputs/Outputs: none.
        if self.current_mode == HybridEngineMode.TRAINER:
            logger.info("Switching hybrid engine to rollout mode for validation")
            self.switch_to_rollout()

    def on_step_begin(self):
        # [FUNCTION DOC]
        # Behavior: resets per-step wait counters and step threshold; zeroes timing "switch_wait" when switching is enabled.
        # If in ROLLOUT mode: without switching, always switches to trainer (timed); with switching, computes the step
        # threshold and switches to trainer immediately if the replay buffer already has at least that many sampleable
        # groups for this step (otherwise prepare_step waits). Inputs/Outputs: none. Side effects: mode change, timing.
        self._step_sample_wait_seconds = 0.0
        self._step_wait_samples = 0
        self._step_threshold = 0
        if self.hybrid_rollout_config.enable_switch:
            self.timing_raw["switch_wait"] = 0.0
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return
        if not self.hybrid_rollout_config.enable_switch:
            self._timed_switch_to_trainer()
            return

        self._step_threshold = self._switch_threshold()
        sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
        if sampleable_count >= self._step_threshold:
            self._timed_switch_to_trainer()

    def _timed_switch_to_trainer(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls switch_to_trainer inside a "switch_to_trainer" timer and, if switching is enabled, records elapsed
        # wall-clock time in the to-trainer cost window. Inputs/Outputs: none.
        """Switch Hybrid back to training and record the full remove/abort/sleep cost."""
        switch_start = time.perf_counter()
        with marked_timer("switch_to_trainer", self.timing_raw, color="cyan"):
            self.switch_to_trainer()
        if self.hybrid_rollout_config.enable_switch:
            self._to_trainer_costs.append(time.perf_counter() - switch_start)

    def on_sample_begin(self):
        # [FUNCTION DOC]
        # Behavior: when switching enabled, adds the shortfall (mini-batch size = train_batch_size / parameter_sync_step minus
        # currently sampleable count, floored at 0) to per-step wait samples; records sample start time.
        # Inputs/Outputs: none.
        if self.hybrid_rollout_config.enable_switch:
            sampleable = self.replay_buffer.get_sampleable_count(self.global_steps, "train")
            mini_batch_size = self.config.data.train_batch_size // self.parameter_sync_step
            self._step_wait_samples += max(0, mini_batch_size - sampleable)
        self._sample_start = time.perf_counter()

    def on_sample_end(self):
        # [FUNCTION DOC]
        # Behavior: adds elapsed time since sample start to per-step sample wait seconds. Inputs/Outputs: none.
        self._step_sample_wait_seconds += time.perf_counter() - self._sample_start

    def prepare_step(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: base prepare_step, then merges metrics from waiting for sampleable data and switching to trainer.
        # Inputs: none. Outputs: metrics dict.
        metrics = super().prepare_step()
        metrics.update(self._wait_for_sampleable_and_switch())
        return metrics

    def _wait_for_sampleable_and_switch(self) -> dict:
        # [FUNCTION DOC]
        # HELPER: If not in ROLLOUT mode returns empty dict. Otherwise logs, blocks (timed as "switch_wait") until the replay
        # buffer has step_threshold sampleable groups for the current step, then switches to trainer.
        # Inputs: none. Outputs: eviction metrics dict returned by the replay buffer.
        if self.current_mode != HybridEngineMode.ROLLOUT:
            return {}

        logger.info(f"Lending hybrid engine to generation until {self._step_threshold} groups are sampleable")
        with marked_timer("switch_wait", self.timing_raw, color="yellow"):
            _, eviction_metrics = self.replay_buffer.wait_for_sampleable(
                self.global_steps, "train", self._step_threshold
            )
        self._timed_switch_to_trainer()
        return eviction_metrics

    def _switch_threshold(self) -> int:
        # [FUNCTION DOC]
        # HELPER: Computes switch threshold as round(ratio * train_batch_size) clamped to [mini_batch_size, train_batch_size].
        # Inputs: none. Outputs: int.
        """Sampleable prompts before switching to trainer, floored at one mini-batch."""
        train_batch_size = self.config.data.train_batch_size
        mini_batch_size = train_batch_size // self.parameter_sync_step
        target = round(self._switch_threshold_ratio * train_batch_size)
        return min(max(target, mini_batch_size), train_batch_size)

    def _step_had_idle(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: Returns True if this step's accumulated sample wait exceeds the replay buffer's poll_interval (default 2.0s).
        """Whether waiting for sampleable prompts."""
        poll_interval = getattr(self.replay_buffer, "poll_interval", 2.0)
        return self._step_sample_wait_seconds > poll_interval

    def _adapt_switch_threshold(self, had_idle: bool) -> None:
        # [FUNCTION DOC]
        # HELPER: Adaptive hysteresis on threshold ratio. If idle: reset calm counter, increment idle counter (capped at
        # release_steps); once it reaches switch_threshold_release_steps, raise ratio by step_up (max 1.0). If not idle: reset
        # idle counter, increment calm counter (capped); once it reaches release_steps, lower ratio by step_down (min
        # 1/parameter_sync_step). Counters are not reset after adjustment, so adjustments repeat each subsequent step.
        # Inputs: had_idle - bool. Outputs: none. Side effects: mutates ratio and counters.
        config = self.hybrid_rollout_config
        if had_idle:
            self._calm_steps = 0
            self._idle_steps = min(self._idle_steps + 1, config.switch_threshold_release_steps)
            if self._idle_steps < config.switch_threshold_release_steps:
                return
            self._switch_threshold_ratio = min(1.0, self._switch_threshold_ratio + config.switch_threshold_step_up)
            return

        self._idle_steps = 0
        self._calm_steps = min(self._calm_steps + 1, config.switch_threshold_release_steps)
        if self._calm_steps < config.switch_threshold_release_steps:
            return
        min_ratio = 1.0 / self.parameter_sync_step
        self._switch_threshold_ratio = max(min_ratio, self._switch_threshold_ratio - config.switch_threshold_step_down)

    def _effective_switch_cost(self) -> float | None:
        # [FUNCTION DOC]
        # HELPER: Returns None if either cost window is empty, else mean to-rollout cost plus mean to-trainer cost (seconds).
        if not self._to_rollout_costs or not self._to_trainer_costs:
            return None
        return sum(self._to_rollout_costs) / len(self._to_rollout_costs) + sum(self._to_trainer_costs) / len(
            self._to_trainer_costs
        )

    def on_step_end(self):
        # [FUNCTION DOC]
        # Behavior: end-of-step hook. Stops rollout profiling if the previous step was profiled. If switching enabled:
        # records idle flag, accumulates per-step wait time/samples into cumulative totals (only when both > 0), optionally
        # adapts threshold, computes next-step threshold and remaining = threshold minus sampleable count for global_steps+1,
        # per-sample time = cumulative wait seconds / samples, benefit = remaining * per_sample_time * (1 - 1/scaling_factor);
        # decides to switch to rollout when not at the final step, remaining > 0, and benefit exceeds effective switch cost
        # (or either is unknown). Builds decision metrics. If switching, pre-registers hybrid servers in the balancer and
        # clears its sticky cache (timed). Then syncs weights to standalone replicas (timed "update_weights"), storing the
        # result in _pending_sync_metrics, merges decision metrics, and if switching performs switch_to_rollout with
        # already_registered and records the total switch cost.
        # Inputs/Outputs: none. Side effects: timing, metrics, mode change, remote calls.
        # _stop_profiling() already moved this step's flag to prev_step_profile.
        if self.prev_step_profile:
            self._stop_rollout_profiling()

        config = self.hybrid_rollout_config
        should_switch = False
        prepare_seconds = 0.0
        decision_metrics: dict[str, float] = {}
        if config.enable_switch:
            ratio_used = self._switch_threshold_ratio
            had_idle = self._step_had_idle()
            if self._step_wait_samples > 0 and self._step_sample_wait_seconds > 0:
                self._wait_seconds += self._step_sample_wait_seconds
                self._wait_samples += self._step_wait_samples
            if config.adaptive_switch_threshold:
                self._adapt_switch_threshold(had_idle)

            decision_threshold = self._switch_threshold()
            sampleable_count = self.replay_buffer.get_sampleable_count(self.global_steps + 1, "train")
            remaining = max(0, decision_threshold - sampleable_count)
            per_sample_time = self._wait_seconds / self._wait_samples if self._wait_samples > 0 else None
            effective_switch_cost = self._effective_switch_cost()
            benefit = (
                remaining * per_sample_time * (1.0 - 1.0 / self._scaling_factor)
                if per_sample_time is not None
                else None
            )
            should_switch = (
                self.global_steps < self.total_training_steps
                and remaining > 0
                and (benefit is None or effective_switch_cost is None or benefit > effective_switch_cost)
            )
            decision_metrics = {
                "separate_async/switch/threshold_ratio": ratio_used,
                "separate_async/switch/wait_samples": float(self._step_wait_samples),
                "separate_async/switch/idle": float(had_idle),
                "separate_async/decision/sampleable_count": float(sampleable_count),
                "separate_async/decision/remaining": float(remaining),
                "separate_async/decision/should_switch_to_rollout": float(should_switch),
            }
            if per_sample_time is not None:
                decision_metrics["separate_async/decision/per_sample_time_seconds"] = per_sample_time
            if effective_switch_cost is not None:
                decision_metrics["separate_async/decision/effective_switch_cost_seconds"] = effective_switch_cost

        if should_switch:
            prepare_start = time.perf_counter()
            # Accumulate preparation and wake-up under one user-facing transition metric.
            with marked_timer("switch_to_rollout", self.timing_raw, color="cyan"):
                self.add_replicas_to_balancer()
                self.clear_sticky_cache()
            prepare_seconds = time.perf_counter() - prepare_start

        with marked_timer("update_weights", self.timing_raw, color="red"):
            self._pending_sync_metrics = dict(
                self.standalone_checkpoint_manager.update_weights(self.global_steps) or {}
            )

        if config.enable_switch:
            self._pending_sync_metrics.update(decision_metrics)

            if should_switch:
                switch_start = time.perf_counter()
                with marked_timer("switch_to_rollout", self.timing_raw, color="cyan"):
                    logger.info("Switching hybrid engine to rollout mode for the next step")
                    self.switch_to_rollout(already_registered=True)
                self._to_rollout_costs.append(prepare_seconds + time.perf_counter() - switch_start)

    def _get_n_gpus_for_throughput(self) -> int:
        # [FUNCTION DOC]
        # HELPER: Returns trainer pool GPU count plus standalone rollout GPUs (n_gpus_per_node * nnodes) for throughput metrics.
        """Include standalone rollout GPUs in the throughput denominator."""
        trainer_gpus = self.resource_pool_manager.get_n_gpus()
        rollout_gpus = (
            self.config.actor_rollout_ref.rollout.n_gpus_per_node * self.config.actor_rollout_ref.rollout.nnodes
        )
        return trainer_gpus + rollout_gpus

    def switch_to_rollout(self, *, already_registered: bool = False):
        # [FUNCTION DOC]
        # Behavior: no-op if hybrid replicas disabled. Syncs weights to hybrid replicas, resumes generation, registers them
        # in the load balancer unless already_registered, and sets mode ROLLOUT.
        # Inputs: already_registered - keyword-only bool. Outputs: none.
        """Install committed weights and make Hybrid replicas available for generation."""
        if not self._enable_hybrid_replicas:
            return
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager.resume_generation_replicas()
        if not already_registered:
            self.add_replicas_to_balancer()
        self.current_mode = HybridEngineMode.ROLLOUT

    def switch_to_trainer(self):
        # [FUNCTION DOC]
        # Behavior: no-op if hybrid replicas disabled. Removes hybrid servers from the balancer, aborts in-flight requests
        # (rejecting new ones), sleeps replicas to free GPU memory, sets mode TRAINER. Inputs/Outputs: none.
        """Stop routing to Hybrid, abort partial requests, and return its GPU memory to training."""
        if not self._enable_hybrid_replicas:
            return
        self.remove_replicas_from_balancer()
        self.checkpoint_manager.abort_replicas(reject_request=True)
        self.checkpoint_manager.sleep_replicas()
        self.current_mode = HybridEngineMode.TRAINER

    def add_replicas_to_balancer(self):
        # [FUNCTION DOC]
        # Behavior: no-op if hybrid replicas disabled; registers hybrid servers (address-to-handle mapping) with the
        # standalone global load balancer and waits for completion. Exceptions: ValueError if address/handle counts differ.
        if not self._enable_hybrid_replicas:
            return
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        servers = dict(
            zip(self.llm_server_manager.server_addresses, self.llm_server_manager.server_handles, strict=True)
        )
        ray.get(global_load_balancer.add_servers.remote(servers))

    def remove_replicas_from_balancer(self):
        # [FUNCTION DOC]
        # Behavior: no-op if hybrid replicas disabled; removes hybrid server addresses from the global load balancer (blocking).
        if not self._enable_hybrid_replicas:
            return
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        ray.get(global_load_balancer.remove_servers.remote(self.llm_server_manager.server_addresses))

    def clear_sticky_cache(self) -> dict:
        # [FUNCTION DOC]
        # Behavior: clears the global load balancer's sticky routing cache and returns its result (blocking remote call).
        # Outputs: whatever the balancer returns (annotated dict).
        global_load_balancer = self.standalone_server_manager.global_load_balancer
        return ray.get(global_load_balancer.clear_sticky_cache.remote())
