# Copyright 2025 Bytedance Ltd. and/or its affiliates

# [MODULE DOC]
# Debug script: composes the Hydra ppo_trainer config from verl/trainer/config,
# prints it, then converts critic.profiler into its dataclass and prints that.
# Exits with ImportError (instructing to install hydra-core) if hydra is missing.

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
try:
    import hydra
except ImportError as e:
    raise ImportError("Please install hydra-core via 'pip install hydra-core' and retry.") from e


@hydra.main(config_path="../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    # [FUNCTION DOC]
    # Behavior: Hydra entry point; prints the full composed config, then the
    # critic profiler section converted via omega_conf_to_dataclass.
    # Inputs: config - OmegaConf DictConfig injected by Hydra (CLI overrides apply).
    # Outputs: none. Exceptions: conversion errors if the profiler section is invalid.
    # Side effects: prints to stdout.
    # KNOWN ISSUE: docstring names the argument "config_dict" but the parameter is "config"; also it only prints config rather than running PPO training as the docstring implies.
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    print(config)
    from verl.utils.config import omega_conf_to_dataclass

    profiler_config = omega_conf_to_dataclass(config.critic.profiler)
    print(profiler_config)


if __name__ == "__main__":
    main()
