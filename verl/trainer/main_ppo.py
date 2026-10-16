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
# Hydra entry point (config "ppo_trainer") for PPO-family training. Validates config, then runs either the V1
# TransferQueue-based trainer (TaskRunnerV1) or the deprecated legacy TaskRunner from main_ppo_v0 inside Ray.
# Logger level from VERL_LOGGING_LEVEL (default INFO).

import logging
import os
from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class) -> None:
    # [FUNCTION DOC]
    # Behavior: if rollout full_determinism (or reward-model rollout full_determinism with RM enabled) sets
    # VERL_FULL_DETERMINISM=1, VLLM_BATCH_INVARIANT=1 and PYTHONHASHSEED to the rollout seed in the environment; sets
    # VERL_RL_INSIGHT_ENABLE=1 if "rl_insight" is among trainer loggers (string or list). If Ray is not initialized,
    # merges the default PPO runtime env (adding TRANSFER_QUEUE_ENABLE=1 when transfer_queue.enable) with
    # ray_kwargs.ray_init.runtime_env (user values win), prints and calls ray.init. Creates the task runner actor, with
    # nsight runtime options when on CUDA with the nsys profiler and non-empty profiler steps (asserting nvtx is
    # installed), runs runner.run(config) to completion, and optionally dumps a Ray timeline to
    # ray_kwargs.timeline_json_file.
    # Inputs: config; task_runner_class - Ray actor class with a run method. Outputs: None.
    # Exceptions: AssertionError if nvtx missing; anything raised by the remote run. Side effects: env mutation, Ray init, files.
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Propagate determinism env vars from config before ray.init() so
    # get_ppo_ray_runtime_env() forwards them to all Ray actors.
    rollout_cfg = config.actor_rollout_ref.rollout
    rm_rollout_cfg = config.reward.reward_model.rollout
    if rollout_cfg.full_determinism or (config.reward.reward_model.enable and rm_rollout_cfg.full_determinism):
        os.environ["VERL_FULL_DETERMINISM"] = "1"
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["PYTHONHASHSEED"] = str(rollout_cfg.seed)

    trainer_logger = config.trainer.get("logger", [])
    if "rl_insight" in ([trainer_logger] if isinstance(trainer_logger, str) else trainer_logger or []):
        os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"

    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env(config)
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Set this on the defaults, not on ray_init.runtime_env: the latter is a struct
            # DictConfig, so adding an `env_vars` key to it raises ConfigKeyError.
            default_runtime_env.setdefault("env_vars", {})["TRANSFER_QUEUE_ENABLE"] = "1"

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote
class TaskRunnerV1:
    # [CLASS DOC]
    # Ray actor that drives the V1 PPO trainer: holds config, trainer and agent loop manager.
    """V1 TaskRunner for PPO training."""

    def __init__(self):
        # [FUNCTION DOC]
        # Initializes config, trainer and agent_loop_manager attributes to None.
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        # [FUNCTION DOC]
        # Behavior: picks the agent loop manager class from rollout.agent.agent_loop_manager_class (loaded by fully
        # qualified name) or defaults to the TransferQueue manager; creates it via its create factory with config and the
        # trainer's LLM client, teacher client and reward loop worker handles; stores it on self.
        # Exceptions: import/load errors for a bad class name. Requires self.trainer to be set.
        """Initialize the agent loop manager to generate sequences.

        NOTE: User can customize their own agent loop manager, the only requirement is:
        1. implement `generate_sequences` method
        2. put agent loop outputs into TransferQueue
        """
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        # [FUNCTION DOC]
        # Behavior: configures verl logging; resolves the trainer class from trainer.v1.trainer_mode; forces
        # transfer_queue.enable true; prints and resolves config; initializes the transfer queue; constructs and inits the
        # trainer, creates the agent loop manager and runs fit. Finally, finishes the trainer's tracking logger (if any)
        # with exit code 0 on success or 1 on failure, and always closes the transfer queue.
        # Inputs: config DictConfig (mutated). Outputs: None. Exceptions: propagates training errors after cleanup.
        """Run the PPO training process."""
        configure_verl_logging()

        import transfer_queue as tq

        from verl.trainer.ppo.v1 import get_trainer_cls

        trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        # initialize transfer queue
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # [FUNCTION DOC]
    # Behavior: auto-sets the device (e.g. NPU), validates config with flags for reference policy and critic needs, then
    # runs run_ppo with TaskRunnerV1 if trainer.use_v1, else logs a deprecation warning and uses the legacy TaskRunner.
    # Inputs: Hydra config. Outputs: None.
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        logger.warning(
            "Legacy trainer `main_ppo_v0.py` is deprecated, and wil be removed in v0.9.0."
            "Please set `trainer.use_v1=True` in config to use V1 trainer."
        )
        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
