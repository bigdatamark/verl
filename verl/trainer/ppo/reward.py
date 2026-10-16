# Copyright 2025 Individual Contributor: Thibaut Barroyer
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
# Reward-related helpers for the PPO trainer: loading a user-provided custom reward
# (scoring) function from a file, resolving/instantiating the reward manager class from
# config (registry or importlib), and extracting reward tensors from a batch.
from __future__ import annotations

import inspect
import multiprocessing
from functools import partial
from typing import TYPE_CHECKING, Any, Optional, cast

from verl import DataProto
from verl.utils.reward_score import get_default_compute_score

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from verl.experimental.reward_loop.reward_manager.base import RawRewardFn, RewardManagerBase
    from verl.trainer.config.config import ModuleConfig
    from verl.workers.config.reward import RewardManagerConfig


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Calls raw_fn with positional args and the call-time kwargs merged with
    # extra_kwargs (extra_kwargs win on key conflicts). Outputs: raw_fn's return value.
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Async variant of _call_with_kwargs; awaits raw_fn with merged kwargs
    # (extra_kwargs take precedence). Outputs: awaited result.
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    # [FUNCTION DOC]
    # Behavior: reads config.reward.custom_reward_function (may be missing). If no path, returns
    # None. Otherwise requires a name, loads that object from the external module path, and
    # returns a partial binding reward_kwargs (from config, default empty) via the sync or async
    # wrapper depending on whether the loaded function is a coroutine function.
    # Inputs: config - full trainer config.
    # Outputs: callable or None.
    # Exceptions: AssertionError when name is missing; loader errors (file not found, import,
    # missing attribute) propagate from load_extern_object.
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def resolve_reward_manager_cls(config: DictConfig) -> type[RewardManagerBase]:
    # [FUNCTION DOC]
    # Behavior: reads config.reward.reward_manager. source "register" -> look up the class by
    # name in the reward-loop registry; source "importlib" -> requires module.path and loads the
    # named object from that path.
    # Inputs: config. Outputs: reward manager class (not instantiated).
    # Exceptions: AssertionError for missing module path; ValueError for unknown source.
    """Resolve the reward manager class from ``config`` without instantiating it."""
    reward_manager_cfg: RewardManagerConfig = config.reward.reward_manager
    if reward_manager_cfg.source == "register":
        from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls

        return get_reward_manager_cls(reward_manager_cfg.name)
    elif reward_manager_cfg.source == "importlib":
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_manager_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_manager_cfg.source=}, but got {module_cfg=}"
        )
        return cast(
            "type[RewardManagerBase]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_manager_cfg.name),
        )
    else:
        raise ValueError(f"Unknown reward manager source: {reward_manager_cfg.source}")


def load_reward_manager(config: DictConfig, tokenizer: Any, **reward_kwargs: Any) -> RewardManagerBase:
    # [FUNCTION DOC]
    # Behavior: obtains an optional custom compute_score function; resolves the manager class;
    # gets the default compute_score for the manager name. If no custom function: when
    # reward.sandbox_fusion.url is set, creates a multiprocessing Manager semaphore
    # (max_concurrent default 64) and binds sandbox url, semaphore and memory_limit_mb (default
    # 1024) into the default scorer; else uses the default scorer as-is. Instantiates the
    # manager with config, tokenizer, compute_score and extra kwargs.
    # Inputs: config; tokenizer; reward_kwargs - forwarded to the manager constructor.
    # Outputs: reward manager instance.
    # Side effects: may spawn a multiprocessing manager process.
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # Try to get a custom reward function based on the configuration
    # user defined reward manager can be registered in custom_reward_fn
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    reward_manager_cfg: RewardManagerConfig = config.reward.reward_manager
    reward_manager_cls = resolve_reward_manager_cls(config)

    default_compute_score_ = get_default_compute_score(reward_manager_cfg.name)

    if compute_score is None:
        sandbox_config = config.reward.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score_,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score_

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        config=config,
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        **reward_kwargs,
    )


def extract_reward(batch: DataProto):
    # [FUNCTION DOC]
    # Behavior: returns the "rm_scores" tensor from batch.batch and a dict mapping each key in
    # meta_info "reward_extra_keys" (default empty) to its non-tensor array.
    # Inputs: batch - DataProto. Outputs: (reward_tensor, extra_info_dict).
    # Exceptions: KeyError if rm_scores or a listed extra key is absent.
    """
    Extract reward tensor and extra info from batch data.
    """
    reward_tensor = batch.batch["rm_scores"]
    reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
    reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
    return reward_tensor, reward_extra_infos_dict
