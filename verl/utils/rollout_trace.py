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
# Rollout tracing utilities supporting Weave, MLflow and Trackio backends. A process-wide singleton
# config selects the backend; a context manager attaches per-sample attributes (sample index, step,
# rollout number, validate flag, experiment name) to traces; and a decorator traces sync or async
# methods, optionally decoding prompt/response token ids into text. Two context variables hold
# whether tracing is enabled (default true) and the current attribute dict (used by non-weave backends).

import contextlib
import functools
import inspect
import json
import os
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
_trace_attributes: ContextVar[dict | None] = ContextVar("_trace_attributes", default=None)


class RolloutTraceConfig:
    # [CLASS DOC]
    # Singleton holding tracing settings as class-level defaults overridden on the single instance:
    # backend name, backend client, token2text flag, project/experiment names,
    # max_samples_per_step_per_worker (stored only; sampling is enforced elsewhere) and an initialized flag.
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
            tracing backends like Weave, MLflow, and Trackio.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: str | None = None
    client: object | None = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: int | None = None

    def __new__(cls, *args, **kwargs):
        # [FUNCTION DOC]
        # Returns the single shared instance, creating it (with initialized False) on first call.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        # [FUNCTION DOC]
        # Returns the singleton instance, creating it if needed.
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: int | None = None,
    ):
        # [FUNCTION DOC]
        # Initializes the singleton once (later calls are no-ops). Stores the settings, then by backend:
        # "weave" -> client from weave.init(project_name); "mlflow" -> enables async logging, sets the
        # tracking URI from env MLFLOW_TRACKING_URI (default a sqlite file under /tmp), sets experiment to
        # project_name, client is the mlflow module; "trackio" -> calls trackio.init(project, name, config
        # {"framework": "verl"}) only if no current run exists, client is the trackio module; otherwise
        # client None. Marks initialized.
        # Side effects: imports and initializes the backend library; may create runs/experiments.
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "trackio":
            import trackio
            from trackio import context_vars

            if context_vars.current_run.get() is None:
                trackio.init(project=project_name, name=experiment_name, config={"framework": "verl"})
            config.client = trackio
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> str | None:
        # [FUNCTION DOC]
        # Returns the configured backend name or None.
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> object | None:
        # [FUNCTION DOC]
        # Returns the configured backend client or None.
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> bool | None:
        # [FUNCTION DOC]
        # Returns whether token ids should be decoded to text in traces.
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        # [FUNCTION DOC]
        # Discards the singleton so the next access creates a fresh, uninitialized instance.
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    # [FUNCTION DOC]
    # Context manager scoping trace attributes. If a backend is set and trace is False, disables tracing
    # (context var) for the body. If no backend, yields without changes. Otherwise builds attributes
    # (sample_index, step, rollout_n only if not None; validate; experiment_name), stores them in the
    # attribute context var for the body, and additionally: weave -> wraps the body in weave.attributes;
    # mlflow -> opens a span named name and sets each attribute as a trace tag; others (trackio) -> only
    # the context var. Context vars are restored on exit.
    # Side effects: backend span/attribute creation.
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    token = _trace_attributes.set(attributes)
    if backend == "weave":
        import weave

        try:
            with weave.attributes(attributes):
                yield
        finally:
            _trace_attributes.reset(token)
    elif backend == "mlflow":
        import mlflow

        try:
            with mlflow.start_span(name=name) as span:
                trace_id = span.trace_id
                for key, value in attributes.items():
                    mlflow.set_trace_tag(trace_id, str(key), str(value))
                yield
        finally:
            _trace_attributes.reset(token)
    else:
        try:
            yield
        finally:
            _trace_attributes.reset(token)


def _json_trace_content(value):
    # [FUNCTION DOC]
    # HELPER: JSON-serializes value (pydantic models dumped first; non-serializable values stringified;
    # non-ASCII preserved).
    if isinstance(value, BaseModel):
        value = value.model_dump()
    return json.dumps(value, default=str, ensure_ascii=False)


def _json_trace_metadata(value):
    # [FUNCTION DOC]
    # HELPER: Recursively converts value into JSON-safe metadata: pydantic models dumped; None and
    # primitives kept; dict keys stringified; lists/tuples become lists; anything else stringified.
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _json_trace_metadata(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_trace_metadata(v) for v in value]
    return str(value)


def _trackio_message_dict(message):
    # [FUNCTION DOC]
    # HELPER: Returns a shallow copy of message if it is a dict with a string "role", else None.
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str):
        return None
    return dict(message)


def _trackio_output_dict(output):
    # [FUNCTION DOC]
    # HELPER: Converts output to a dict (pydantic dump, dict as-is, or the object's attribute dict), else None.
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    return None


def _trackio_trace_key(op_name):
    # [FUNCTION DOC]
    # HELPER: Builds the log key "rollout_trace/<op_name>" with characters other than alphanumerics,
    # ".", "_" and "-" replaced by "_".
    return "rollout_trace/" + "".join(char if char.isalnum() or char in "._-" else "_" for char in op_name)


def _trackio_trace_step(attributes):
    # [FUNCTION DOC]
    # HELPER: Returns attributes["step"] as int, or None if missing or not convertible.
    step = attributes.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _log_trackio_trace(op_name, inputs, output=None, exception=None):
    # [FUNCTION DOC]
    # HELPER: Logs one operation as a trackio Trace.
    # Behavior: metadata includes op name, backend, experiment name, JSON-safe inputs (excluding
    # "messages"), current trace attributes, and either status "error" with exception type or status
    # "success" with output. Messages: valid role dicts from inputs["messages"] if present, else a system
    # message naming the op and a user message with JSON inputs; then an assistant message containing
    # the exception, the output's "response_text", its "answer", or the JSON output (first that applies).
    # Logged under the op-specific key at the attribute step.
    # Side effects: calls the trackio client's log.
    trackio = RolloutTraceConfig.get_client()
    attributes = _current_trace_attributes()
    metadata_inputs = {key: value for key, value in inputs.items() if key != "messages"}
    output_dict = _trackio_output_dict(output)
    metadata = {
        "op": op_name,
        "backend": "trackio",
        "experiment_name": RolloutTraceConfig.get_instance().experiment_name,
        "inputs": _json_trace_metadata(metadata_inputs),
        **{key: _json_trace_metadata(value) for key, value in attributes.items()},
    }
    if exception is not None:
        metadata["status"] = "error"
        metadata["exception_type"] = type(exception).__name__
    else:
        metadata["status"] = "success"
        metadata["output"] = _json_trace_metadata(output_dict if output_dict is not None else output)

    messages = []
    input_messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if isinstance(input_messages, list):
        messages = [
            message for message in (_trackio_message_dict(message) for message in input_messages) if message is not None
        ]

    if not messages:
        messages = [
            {"role": "system", "content": f"verl rollout trace operation: {op_name}"},
            {"role": "user", "content": _json_trace_content({"inputs": inputs})},
        ]

    if exception is not None:
        messages.append(
            {
                "role": "assistant",
                "content": _json_trace_content(
                    {
                        "exception_type": type(exception).__name__,
                        "exception": str(exception),
                    }
                ),
            }
        )
    elif output_dict is not None and output_dict.get("response_text"):
        messages.append({"role": "assistant", "content": str(output_dict["response_text"])})
    elif output_dict is not None and output_dict.get("answer"):
        messages.append({"role": "assistant", "content": str(output_dict["answer"])})
    else:
        messages.append({"role": "assistant", "content": _json_trace_content({"output": output})})

    trackio.log(
        {_trackio_trace_key(op_name): trackio.Trace(messages=messages, metadata=metadata)},
        step=_trackio_trace_step(attributes),
    )


def _current_trace_attributes():
    # [FUNCTION DOC]
    # HELPER: Returns a copy of current attributes: from weave's call context when backend is weave,
    # otherwise from the attribute context var (empty dict if unset).
    backend = RolloutTraceConfig.get_backend()
    if backend == "weave":
        from weave.trace.context import call_context

        return {**call_context.call_attributes.get()}
    return {**(_trace_attributes.get() or {})}


def _trace_field(value, field_name):
    # [FUNCTION DOC]
    # HELPER: Reads field_name from a dict key or object attribute; None if absent.
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _trace_output_copy(output):
    # [FUNCTION DOC]
    # HELPER: Returns a mutable dict copy of output (pydantic dump, dict copy, attribute dict) or None.
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return dict(output)
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    return None


async def _add_token2text(instance, inputs, output):
    # [FUNCTION DOC]
    # HELPER: Async. Adds decoded text to a traced output using instance.tokenizer.decode if available.
    # Prompt ids come from output "prompt_ids" or inputs["prompt_ids"]; response ids from output
    # "response_ids" or "token_ids". If no decoder, no ids, or output can't be copied to a dict, returns
    # output unchanged. Otherwise decodes in the event loop's default executor and returns a copy with
    # "prompt_text" and/or "response_text" added. The original output is not modified.
    tokenizer = getattr(instance, "tokenizer", None)
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return output

    # Agent-loop outputs contain prompt_ids/response_ids, while per-turn
    # LLMServerClient.generate calls receive prompt_ids and return token_ids.
    prompt_ids = _trace_field(output, "prompt_ids")
    if prompt_ids is None:
        prompt_ids = inputs.get("prompt_ids")
    response_ids = _trace_field(output, "response_ids")
    if response_ids is None:
        response_ids = _trace_field(output, "token_ids")

    if prompt_ids is None and response_ids is None:
        return output

    output_copy = _trace_output_copy(output)
    if output_copy is None:
        return output

    loop = get_event_loop()
    if prompt_ids is not None:
        output_copy["prompt_text"] = await loop.run_in_executor(None, decode, prompt_ids)
    if response_ids is not None:
        output_copy["response_text"] = await loop.run_in_executor(None, decode, response_ids)
    return output_copy


def rollout_trace_op(func):
    # [FUNCTION DOC]
    # Decorator for methods (first arg self) that records each call with the configured backend.
    # Returns an async wrapper for coroutine functions, else a sync wrapper. Inputs recorded are the
    # bound arguments with defaults applied, excluding self.
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Async tracing wrapper. Calls through directly if tracing is disabled or no backend.
        # weave: create a call (op = qualified name, current attributes), finish with output (token-decoded
        # if enabled) or exception (re-raised). mlflow: run inside a span named after the function, setting
        # inputs and outputs (decoded if enabled). trackio: log via _log_trackio_trace with output or
        # exception (re-raised). Always returns the original, undecoded result.
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await _add_token2text(self, inputs, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                span.set_inputs(inputs)
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await _add_token2text(self, inputs, result)
                    span.set_outputs(_result)
                else:
                    span.set_outputs(result)

            return result
        elif backend == "trackio":
            try:
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await _add_token2text(self, inputs, result)
                    _log_trackio_trace(func.__qualname__, inputs, output=_result)
                else:
                    _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Sync tracing wrapper: same as the async one without token decoding; for mlflow delegates
        # to mlflow.trace applied to the function.
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "trackio":
            try:
                result = func(self, *args, **kwargs)
                _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
