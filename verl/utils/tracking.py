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
# Unified experiment-tracking facade. Tracking fans metric logging out to backends (wandb, mlflow, swanlab,
# vemlp_wandb, tensorboard, console, clearml, trackio, a JSONL file logger and rl-insight); backend adapters live in
# this module, along with helpers that flatten configs into MLflow params, a validation-generations logger that writes
# sample tables/text to several backends, and a wandb-only DAPO filtered-reward table logger. MLflow init/log retries
# are bounded by MLFLOW_MAX_ATTEMPTS (3) with MLFLOW_SLEEP_SECONDS (5) between init attempts.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import logging
import os
from contextlib import contextmanager
from enum import Enum
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import orjson
from packaging.version import Version

logger = logging.getLogger(__name__)

MLFLOW_MAX_ATTEMPTS = 3
MLFLOW_SLEEP_SECONDS = 5


class Tracking:
    # [CLASS DOC]
    # Facade that owns one logger object per enabled backend (mapping backend name to object exposing log and finish) and
    # dispatches log/finish calls to them. supported_backend lists accepted backend names.
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
        "rl_insight",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        # [FUNCTION DOC]
        # Initializes the selected backends.
        # Behavior: accepts a single name or list; "tracking" is accepted with a DeprecationWarning and treated as wandb; any
        # other unsupported name fails an assertion. Then per backend: wandb - init with project/name, WANDB_ENTITY env, config,
        # optional https proxy from config trainer.wandb_proxy; trackio - init only if no current run, wrapped in an adapter;
        # mlflow - up to 3 attempts setting tracking URI (MLFLOW_TRACKING_URI, default a sqlite file under /tmp), attaching to
        # MLFLOW_RUN_ID if set or else setting experiment = project_name and starting a run named experiment_name, logging
        # flattened config params; failures are logged and retried after a sleep, and after the last failure mlflow is skipped;
        # swanlab - optional login with SWANLAB_API_KEY, init with logdir/mode from env and config plus FRAMEWORK=verl;
        # vemlp_wandb - init the Volcengine platform from VOLC_ACCESS_KEY_ID / VOLC_SECRET_ACCESS_KEY / MLP_TRACKING_REGION env
        # vars then its wandb with tensorboard sync; tensorboard, console (LocalLogger printing to console, also stored as
        # console_logger), clearml, file and rl_insight via their adapter classes.
        # Inputs: project_name, experiment_name, default_backend (default "console"), config (mapping, may be None).
        # Exceptions: AssertionError for unsupported backends; KeyError for missing Volcengine env vars; ImportError for
        # missing backend packages; backend init errors (except mlflow) propagate.
        # Side effects: network/file I/O, global backend state, logging, sleeping on mlflow retries.
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}
        self._finished = False

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb.init(project=project_name, name=experiment_name, entity=entity, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio
            from trackio import context_vars

            if context_vars.current_run.get() is None:
                trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = _TrackioLoggingAdapter(trackio)

        if "mlflow" in default_backend:
            import os
            import time

            import mlflow

            for _mlflow_attempt in range(1, MLFLOW_MAX_ATTEMPTS + 1):
                try:
                    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
                    logger.info("Using MLFlow tracking URI: %s", MLFLOW_TRACKING_URI)
                    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

                    # Some cloud providers like Azure ML or Databricks automatically set MLFLOW_RUN_ID
                    # If set, attach to the existing run instead of creating a new one
                    run_id = os.environ.get("MLFLOW_RUN_ID")
                    if run_id:
                        mlflow.start_run(run_id=run_id)
                    else:
                        # Project_name is actually experiment_name in MLFlow
                        # If experiment does not exist, will create a new experiment
                        experiment = mlflow.set_experiment(project_name)
                        mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

                    mlflow.log_params(_compute_mlflow_params_from_objects(config))
                    self.logger["mlflow"] = _MlflowLoggingAdapter()
                    break  # Success
                except Exception as e:
                    logger.warning(
                        "MLflow initialization attempt %d/%d failed: %s", _mlflow_attempt, MLFLOW_MAX_ATTEMPTS, e
                    )
                    if _mlflow_attempt < MLFLOW_MAX_ATTEMPTS:
                        time.sleep(MLFLOW_SLEEP_SECONDS)
                    else:
                        logger.warning("All MLflow initialization attempts failed. Proceeding without MLflow tracking.")

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

        if "rl_insight" in default_backend:
            self.logger["rl_insight"] = RLInsightLogger(project_name, experiment_name, config)

    def log(self, data, step, backend=None):
        # [FUNCTION DOC]
        # Calls log(data, step) on every backend logger, or only those whose name is contained in backend when given.
        # KNOWN ISSUE: membership test against backend means a string backend argument matches by substring.
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def finish(self, exit_code: int = 0):
        # [FUNCTION DOC]
        # Idempotently finalizes all backends: on first call marks finished, then calls finish on each present backend
        # (wandb and vemlp_wandb with exit_code). Safe on partially initialized instances.
        """Flush and finalize every configured backend exactly once."""
        if getattr(self, "_finished", False):
            return
        self._finished = True
        loggers = getattr(self, "logger", {})

        if "wandb" in loggers:
            loggers["wandb"].finish(exit_code=exit_code)
        if "swanlab" in loggers:
            loggers["swanlab"].finish()
        if "vemlp_wandb" in loggers:
            loggers["vemlp_wandb"].finish(exit_code=exit_code)
        if "tensorboard" in loggers:
            loggers["tensorboard"].finish()
        if "clearml" in loggers:
            loggers["clearml"].finish()
        if "mlflow" in loggers:
            loggers["mlflow"].finish()
        if "trackio" in loggers:
            loggers["trackio"].finish()
        if "file" in loggers:
            loggers["file"].finish()
        if "rl_insight" in loggers:
            loggers["rl_insight"].finish()

    def __del__(self):
        # [FUNCTION DOC]
        # Destructor that calls finish so backends are flushed when the object is garbage collected.
        self.finish()


class RLInsightLogger:
    # [CLASS DOC]
    # rl-insight backend implemented entirely with class-level state (enabled only when env var VERL_RL_INSIGHT_ENABLE is
    # "1"): lazily imported module, init flag, set of registered Prometheus scrape targets and set of already-warned
    # unsupported features. Exposes metric logging, state/span tracing, agent-loop sessions and metrics-endpoint
    # registration; features needing rl-insight 0.3.0+ degrade with a one-time warning.
    """Logger backend that exports scalar metrics and rl-insight runtime signals."""

    ENABLE_ENV = "VERL_RL_INSIGHT_ENABLE"
    MINIMUM_RL_INSIGHT_VERSION = Version("0.3.0")
    _init_done = False
    _rl_insight_module = None
    _registered_metrics: set[tuple[str | None, tuple[str, ...], str | None]] = set()
    _warned_unsupported_features: set[str] = set()

    def __init__(self, project_name, experiment_name, config=None):
        # [FUNCTION DOC]
        # Delegates to the class-level init with the given names and config.
        self.init(project_name=project_name, experiment_name=experiment_name, config=config)

    @classmethod
    def _get_rl_insight(cls):
        # [FUNCTION DOC]
        # HELPER: Imports and caches the rl_insight module on the class; returns it. Exceptions: ImportError if not installed.
        if cls._rl_insight_module is None:
            import rl_insight

            cls._rl_insight_module = rl_insight
        return cls._rl_insight_module

    @classmethod
    def enabled(cls) -> bool:
        # [FUNCTION DOC]
        # Returns True iff the enable env var equals "1".
        """Return whether rl-insight is globally enabled in this process."""
        return os.getenv(cls.ENABLE_ENV) == "1"

    @classmethod
    def _warn_unsupported_rl_insight(cls, feature: str) -> None:
        # [FUNCTION DOC]
        # HELPER: Logs a warning once per feature name that the installed rl-insight is too old; records the feature.
        """Warn once when an optional RL-Insight feature is unavailable."""
        if feature in cls._warned_unsupported_features:
            return
        cls._warned_unsupported_features.add(feature)
        logger.warning(
            "RL-Insight does not support %s (requires >= %s); monitoring is disabled for this feature",
            feature,
            cls.MINIMUM_RL_INSIGHT_VERSION,
        )

    @classmethod
    def _rl_insight_version(cls) -> Version:
        # [FUNCTION DOC]
        # HELPER: Returns the installed rl-insight version parsed from its __version__ (missing -> "0").
        module = cls._get_rl_insight()
        return Version(getattr(module, "__version__", "0"))

    @classmethod
    def init(cls, project_name=None, experiment_name=None, config=None):
        # [FUNCTION DOC]
        # One-time initialization: no-op when disabled or already initialized. Reads trainer.rl_insight from config (empty on
        # missing/malformed config), calls rl_insight init with project, experiment and that config, marks initialized and,
        # if a config is given, registers the transfer-queue metrics endpoint.
        if not cls.enabled() or cls._init_done:
            return

        rl_insight_config = {}
        if config is not None:
            try:
                rl_insight_config = config.get("trainer", {}).get("rl_insight", {}) or {}
            except (AttributeError, KeyError, TypeError):
                pass
        cls._get_rl_insight().init(project=project_name, experiment_name=experiment_name, config=rl_insight_config)
        cls._init_done = True
        if config is not None:
            cls.register_transfer_queue_metrics(config)

    @classmethod
    def log(cls, data, step):
        # [FUNCTION DOC]
        # When enabled, ensures init and reports each value convertible to float as a gauge, with "/" in the key replaced by
        # "_"; non-numeric values are skipped.
        if not cls.enabled():
            return
        cls._ensure_rl_insight_init()
        metric_gauge = cls._get_rl_insight().metric_gauge

        for key, value in data.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            metric_gauge(str(key).replace("/", "_"), scalar)

    @classmethod
    def finish(cls):
        # [FUNCTION DOC]
        # If initialized, calls rl_insight finish, clears the init flag and the registered metric targets.
        if not cls._init_done:
            return

        cls._get_rl_insight().finish()
        cls._init_done = False
        cls._registered_metrics.clear()

    @classmethod
    @contextmanager
    def trace_state(
        cls,
        state_name: str,
        *,
        state_lane_id: str | int | None = None,
        **labels: Any,
    ):
        # [FUNCTION DOC]
        # Context manager tracing a named state: yields directly when disabled; otherwise ensures init and yields inside
        # rl_insight's trace_state with the lane id and labels.
        if not cls.enabled():
            yield
            return

        cls._ensure_rl_insight_init()
        with cls._get_rl_insight().trace_state(state_name, state_lane_id=state_lane_id, **labels):
            yield

    @classmethod
    def trace_span(
        cls,
        name: str,
        *,
        start_time_ns: int,
        end_time_ns: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Reports one completed span (name, start/end ns, attributes copied to a dict) when enabled; warns once and returns if
        # the version is below the minimum or trace_span is missing. Ensures init before reporting.
        """Report one completed span through rl-insight's direct trace API."""
        if not cls.enabled():
            return

        module = cls._get_rl_insight()
        if cls._rl_insight_version() < cls.MINIMUM_RL_INSIGHT_VERSION or not callable(
            getattr(module, "trace_span", None)
        ):
            cls._warn_unsupported_rl_insight("trace_span")
            return
        cls._ensure_rl_insight_init()
        cls._get_rl_insight().trace_span(
            name=name,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            attributes=dict(attributes or {}),
        )

    @classmethod
    def _ensure_rl_insight_init(cls) -> None:
        # [FUNCTION DOC]
        # HELPER: Calls rl_insight init with no arguments and marks initialized if not already.
        if not cls._init_done:
            cls._get_rl_insight().init()
            cls._init_done = True

    @classmethod
    def agent_loop_session(
        cls,
        *,
        experiment_name: Any | None = None,
        sample: Any,
        session: Any,
        traj: Any = 0,
        uid: Any = None,
        global_steps: Any = None,
        session_id: Any = None,
    ):
        # [FUNCTION DOC]
        # Returns an agent-loop session object for tracing.
        # Behavior: takes project and experiment names from the rollout trace config singleton (experiment defaults to
        # "default" when absent). When enabled but the version is too old, warns and returns a fallback; when enabled and
        # uninitialized, initializes. Tries to import rl_insight's agent_loop_session; on ImportError warns and returns the
        # fallback; otherwise returns its result for the given identifiers. The fallback is a simple namespace with an identity
        # mapping of string fields (project, experiment, sample, session, traj, a composite state_lane_id, uid, global_steps,
        # session_id) and a no-op finish.
        """Return the shared agent-loop session trace state."""
        from verl.utils.rollout_trace import RolloutTraceConfig

        rollout_config = RolloutTraceConfig.get_instance()
        project_name = rollout_config.project_name
        if experiment_name is None:
            experiment_name = rollout_config.experiment_name or "default"

        def fallback_session():
            # [FUNCTION DOC]
            # HELPER: Builds the no-op fallback session namespace described above.
            return SimpleNamespace(
                identity={
                    "project": project_name or "default",
                    "experiment_name": experiment_name,
                    "sample": str(sample),
                    "session": str(session),
                    "traj": str(traj),
                    "state_lane_id": f"experiment={experiment_name}/sample={sample}/session={session}/traj={traj}",
                    "uid": uid or "",
                    "global_steps": global_steps if global_steps is not None else "",
                    "session_id": session_id or "",
                },
                finish=lambda **kwargs: None,
            )

        if cls.enabled() and cls._rl_insight_version() < cls.MINIMUM_RL_INSIGHT_VERSION:
            cls._warn_unsupported_rl_insight("agent_loop_session")
            return fallback_session()
        if cls.enabled() and not cls._init_done:
            cls.init(project_name=project_name, experiment_name=experiment_name)

        try:
            from rl_insight.agent_loop import agent_loop_session
        except ImportError:
            cls._warn_unsupported_rl_insight("agent_loop_session")
            return fallback_session()

        return agent_loop_session(
            project=project_name,
            experiment_name=experiment_name,
            sample=sample,
            session=session,
            traj=traj,
            uid=uid,
            global_steps=global_steps,
            session_id=session_id,
        )

    @classmethod
    def register_rollout_metrics(
        cls,
        server_addresses: list[str],
        rollout_name: str | None,
        labels: list[dict[str, Any] | None] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Registers rollout server addresses as a scrape job named rollout_name (delegates to register_metrics).
        cls.register_metrics(server_addresses, rollout_name, labels)

    @classmethod
    def register_transfer_queue_metrics(cls, config) -> None:
        # [FUNCTION DOC]
        # If config transfer_queue.metrics.enabled is true, gets the transfer_queue metrics endpoint and registers it as the
        # "transfer_queue" job; any failure obtaining it is logged with traceback and ignored.
        if not (config or {}).get("transfer_queue", {}).get("metrics", {}).get("enabled", False):
            return

        try:
            import transfer_queue as tq

            endpoint = tq.get_metrics_endpoint()
        except Exception:
            logger.exception("[rl-insight] Failed to get transfer_queue metrics endpoint")
            return

        if endpoint:
            cls.register_metrics([endpoint], "transfer_queue", None)

    @classmethod
    def register_metrics(
        cls,
        server_addresses: list[str],
        job_name: str | None = None,
        labels: list[dict[str, Any] | None] | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # When enabled, registers a Prometheus scrape config once per (job, addresses, labels repr) key via rl_insight; errors
        # are logged with traceback and the key is not recorded.
        if not cls.enabled():
            return

        metric_key = (job_name, tuple(server_addresses), repr(labels))
        if metric_key in cls._registered_metrics:
            return

        try:
            cls._get_rl_insight().update_prometheus_config(server_addresses, job_name, labels)
            cls._registered_metrics.add(metric_key)
        except Exception:
            logger.exception("[rl-insight] Failed to register metrics endpoint")


class ClearMLLogger:
    # [CLASS DOC]
    # ClearML backend: opens (or continues) a ClearML task for project/experiment and reports scalars and tables.
    def __init__(self, project_name: str, experiment_name: str, config):
        # [FUNCTION DOC]
        # Stores names, initializes a ClearML task (continue last task, no output upload) and connects config as
        # "Hyperparameters". Side effects: network I/O.
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        # [FUNCTION DOC]
        # HELPER: Returns the task's ClearML logger.
        return self._task.get_logger()

    def log(self, data, step):
        # [FUNCTION DOC]
        # Splits each key at the first "/" into title and series; reports numeric values as scalars and DataFrames as tables at
        # iteration step; other types trigger a warning.
        # KNOWN ISSUE: keys without "/" raise ValueError on unpacking; the local name logger shadows the module logger, so the warning goes through the ClearML logger object.
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        # [FUNCTION DOC]
        # Closes the ClearML task.
        self._task.close()


class _TrackioLoggingAdapter:
    # [CLASS DOC]
    # HELPER: Adapter giving trackio a log(data, step)/finish interface.
    def __init__(self, trackio):
        # [FUNCTION DOC]
        # Stores the trackio module.
        self.trackio = trackio

    def log(self, data, step):
        # [FUNCTION DOC]
        # Logs data at step via trackio.
        self.trackio.log(data, step=step)

    def finish(self):
        # [FUNCTION DOC]
        # Finishes trackio only if a run is currently active.
        from trackio import context_vars

        if context_vars.current_run.get() is not None:
            self.trackio.finish()


class FileLogger:
    # [CLASS DOC]
    # JSONL file backend: appends one line per log call containing the step and data.
    def __init__(self, project_name: str, experiment_name: str):
        # [FUNCTION DOC]
        # Determines the file path: VERL_FILE_LOGGER_PATH if set, else <VERL_FILE_LOGGER_ROOT or ".">/<project>/<experiment>.jsonl
        # (creating the directory). Prints the path and opens the file for unbuffered binary writing (truncating).
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
        print(f"Creating file logger at {os.path.abspath(self.filepath)}")
        self.fp = open(self.filepath, "wb", buffering=0)

    def log(self, data, step):
        # [FUNCTION DOC]
        # Writes a JSON line of step and data, serializing numpy values. Exceptions: orjson errors for unserializable data.
        data = {"step": step, "data": data}
        self.fp.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")

    def finish(self):
        # [FUNCTION DOC]
        # Closes the file.
        self.fp.close()


class _TensorboardAdapter:
    # [CLASS DOC]
    # HELPER: TensorBoard backend writing scalars with a SummaryWriter.
    def __init__(self, project_name, experiment_name):
        # [FUNCTION DOC]
        # Uses TENSORBOARD_DIR env or tensorboard_log/<project>/<experiment>, creates it, prints it and opens a SummaryWriter.
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        # [FUNCTION DOC]
        # Adds each data entry as a scalar at step. Exceptions: writer errors for non-scalar values.
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        # [FUNCTION DOC]
        # Closes the writer.
        self.writer.close()


class _MlflowLoggingAdapter:
    # [CLASS DOC]
    # HELPER: MLflow backend: sanitizes metric keys to MLflow's allowed charset and logs metrics with retries.
    def __init__(self):
        # [FUNCTION DOC]
        # Sets up a logger, lowers botocore credential log noise to WARNING, compiles patterns for invalid key characters and
        # repeated slashes, and creates a sanitization cache.
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # Suppress noisy "Found credentials from IAM Role" on every MLflow request
        logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")
        self._sanitized_key_cache = {}

    def _sanitize_key(self, key):
        # [FUNCTION DOC]
        # HELPER: Sanitizes a metric key (cached): "@" -> "_at_", collapse repeated slashes, replace disallowed characters with
        # "_". Logs a warning the first time a key changes. Outputs: sanitized key (the original if unchanged).
        if key in self._sanitized_key_cache:
            return self._sanitized_key_cache[key] or key
        # First replace @ with _at_ for backward compatibility
        sanitized = key.replace("@", "_at_")
        # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
        sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
        # Then replace any other invalid characters with _
        sanitized = self._invalid_chars_pattern.sub("_", sanitized)
        if sanitized == key:
            self._sanitized_key_cache[key] = None
        else:
            self.logger.warning("[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized)
            self._sanitized_key_cache[key] = sanitized
        return sanitized

    def log(self, data, step):
        # [FUNCTION DOC]
        # Logs sanitized metrics at step, retrying up to the max attempts without sleeping; failures are logged (info, then
        # warning on the last) and the data is dropped after all attempts.
        import mlflow

        results = {self._sanitize_key(k): v for k, v in data.items()}
        for _attempt in range(MLFLOW_MAX_ATTEMPTS):
            try:
                mlflow.log_metrics(metrics=results, step=step)
                return
            except Exception as error:
                # No sleep between retries — this runs per training step, so we avoid blocking.
                msg = "mlflow.log_metrics failed (attempt %d/%d): %s"
                args = (_attempt + 1, MLFLOW_MAX_ATTEMPTS, error)
                if _attempt < MLFLOW_MAX_ATTEMPTS - 1:
                    self.logger.info(msg, *args)
                else:
                    self.logger.warning(msg, *args)

    def finish(self):
        # [FUNCTION DOC]
        # Ends the active MLflow run if any.
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run()


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Converts a config object into a flat dict of MLflow params ("/"-separated keys, lists expanded into indexed
    # entries plus list_len). None -> empty dict.
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    # [FUNCTION DOC]
    # HELPER: Recursively converts dataclasses (via asdict), dicts, lists (either into an indexed mapping with list_len or
    # a list), Paths (to string) and Enums (to value); other values unchanged.
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Flattens a nested dict with pandas json_normalize using sep, returning the single record as a dict.
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    # [CLASS DOC]
    # Dataclass that logs validation generation samples (each sample a sequence of input, output, score) to the backends
    # named in the given loggers collection. Lazily keeps backend state (wandb table, tensorboard writer) on the instance.
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        # [FUNCTION DOC]
        # Dispatches to the per-backend generation loggers for wandb, swanlab, mlflow, trackio, clearml, tensorboard and
        # vemlp_wandb when present in loggers.
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)
        if "trackio" in loggers:
            self.log_generations_to_trackio(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        # [FUNCTION DOC]
        # Logs the generations table through the Volcengine wandb module.
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        # [FUNCTION DOC]
        # Logs the generations table through wandb.
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        # [FUNCTION DOC]
        # HELPER: Appends one wide row (step followed by input/output/score of every sample) to a cumulative wandb table,
        # recreating it from previous data each call (wandb workaround), logs it as val/generations when a run is active, and
        # keeps the new table. Columns depend on the sample count of the first call.
        # KNOWN ISSUE: if later calls have a different number of samples, the row length no longer matches the column list.
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        if wandb.run is not None:
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        # [FUNCTION DOC]
        # Logs an echarts table with columns step/input/output/score (one row per sample) to swanlab as val/generations.
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        # Create column names
        headers = ["step", "input", "output", "score"]

        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        # [FUNCTION DOC]
        # Writes the samples as a JSON list of input/output/score objects to a temp file val_step<step>.json and logs it as an
        # MLflow artifact; any error is printed as a warning.
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_trackio(self, samples, step):
        # [FUNCTION DOC]
        # Builds one trackio Trace per sample (user=input, assistant=output, metadata with source, index and score; samples
        # shorter than 3 are treated as input only) and logs them as val/generations if any.
        """Log validation generations to trackio as traces."""
        import trackio

        traces = []
        for sample_index, sample in enumerate(samples):
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]
            else:
                input_text, output_text, score = sample, "", None

            traces.append(
                trackio.Trace(
                    messages=[
                        {"role": "user", "content": str(input_text)},
                        {"role": "assistant", "content": str(output_text)},
                    ],
                    metadata={
                        "source": "validation_generations",
                        "sample_index": sample_index,
                        "score": score,
                    },
                )
            )

        if traces:
            trackio.log({"val/generations": traces}, step=step)

    def log_generations_to_clearml(self, samples, step):
        # [FUNCTION DOC]
        # If a current ClearML task exists, reports a table (step, input, output, score per sample) titled "Validation".
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        # [FUNCTION DOC]
        # Lazily creates a SummaryWriter (TENSORBOARD_DIR or tensorboard_log[/project/experiment]); formats all samples as one
        # markdown text block (input, output, score, or raw data for short samples) and writes it as val/generations, then flushes.
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()


@dataclasses.dataclass
class DapoFilteredRewardTableLogger:
    # [CLASS DOC]
    # Dataclass logging per-step DAPO filtered-group reward counts to a wandb table (incremental mode on wandb 0.20+,
    # otherwise rebuilt from accumulated rows). Other backends are ignored.
    """Wandb table of DAPO-filtered (no-signal) group counts per reward value.

    Each training step adds one row containing compact ``reward:count`` pairs. Wandb 0.20+
    uploads rows incrementally; older versions rebuild the full table for compatibility.

    Intentionally wandb-only: this "value distribution over time" view is a table, which other
    tracking backends do not render usefully. Non-wandb backends are silently skipped.
    """

    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, reward_counts: dict, step: int):
        # [FUNCTION DOC]
        # Forwards to the wandb implementation when "wandb" is in loggers.
        """reward_counts maps metric value -> count for this step (already merged across mini-batches)."""
        if "wandb" in loggers:
            self._log_to_wandb(reward_counts, step)

    def _log_to_wandb(self, reward_counts: dict, step: int):
        # [FUNCTION DOC]
        # HELPER: When a wandb run exists, formats counts as "value:count" pairs sorted by value, lazily decides incremental
        # vs. full-rebuild mode by wandb version (warning on old versions), adds a row (step, text) and logs the table under
        # training/filter_groups/filtered_reward_counts.
        import wandb

        if wandb.run is None:
            return

        row = {float(value): int(count) for value, count in reward_counts.items()}
        counts_text = ", ".join(f"{value:g}:{row[value]}" for value in sorted(row))
        columns = ["step", "reward_counts"]

        if not hasattr(self, "_use_incremental_table"):
            self._use_incremental_table = Version(wandb.__version__) >= Version("0.20.0")
            if self._use_incremental_table:
                self._table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
            else:
                self._rows = []
                logger.warning(
                    "wandb<0.20.0 does not support incremental tables; "
                    "the DAPO filtered-reward table will re-upload its full history each step."
                )

        if self._use_incremental_table:
            self._table.add_data(step, counts_text)
            table = self._table
        else:
            self._rows.append([step, counts_text])
            table = wandb.Table(columns=columns, data=list(self._rows))

        wandb.log({"training/filter_groups/filtered_reward_counts": table}, step=step)
