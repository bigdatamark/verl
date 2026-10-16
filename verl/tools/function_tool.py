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
# Decorator-based registration of plain Python functions as agent tools.
# Globals: FUNCTION_TOOL_REGISTRY maps tool name to FunctionTool; _LOADED_FUNCTION_TOOL_PATHS caches, per absolute
# file path, the tools that loading that file registered (so a file is executed only once).
# Side effects: module logger level from VERL_LOGGING_LEVEL (default WARN).

"""
Lightweight function-based tool registration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from transformers.utils import get_json_schema

from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# global registry
FUNCTION_TOOL_REGISTRY: dict[str, FunctionTool] = {}

_LOADED_FUNCTION_TOOL_PATHS: dict[str, list[FunctionTool]] = {}


@dataclass
class FunctionTool:
    # [CLASS DOC]
    # Dataclass record for a registered function tool: name, fn (callable), tool_schema, is_async (true when fn is a coroutine function).
    """Carrier object stored in :data:`FUNCTION_TOOL_REGISTRY`.

    Exposes the minimal interface that the agent loop relies on:

    - ``name``: tool name (matches ``tool_schema.function.name``)
    - ``tool_schema``: ``OpenAIFunctionToolSchema`` for prompt assembly
    - ``fn``: the underlying callable
    """

    name: str
    fn: Callable[..., Any]
    tool_schema: OpenAIFunctionToolSchema
    is_async: bool = False

    async def call(self, parameters: dict[str, Any]) -> Any:
        # [FUNCTION DOC]
        # Behavior: calls fn with parameters as keyword arguments; awaits directly if async, otherwise runs it in a worker
        # thread via asyncio.to_thread. Inputs: parameters - mapping of arg name to value. Outputs: fn's raw return value.
        # Exceptions: propagates anything fn raises (including TypeError for bad arguments).
        """Invoke the underlying function with the LLM-supplied parameters."""
        if self.is_async:
            return await self.fn(**parameters)
        return await asyncio.to_thread(self.fn, **parameters)


def function_tool(
    name: Optional[str | Callable] = None,
    *,
    schema: Optional[OpenAIFunctionToolSchema | dict] = None,
):
    # [FUNCTION DOC]
    # Behavior: decorator factory usable bare or called. If name is callable and no schema is given, treats it as the
    # function and registers immediately; otherwise returns a decorator using name as override.
    # Inputs: name - tool name, or the function itself (bare form); schema - optional prebuilt schema or dict.
    # Outputs: the original function (bare) or a decorator returning the original function.
    # Exceptions: see inner decorator. Side effects: mutates FUNCTION_TOOL_REGISTRY.
    # Note: bare usage combined with schema is impossible since schema is keyword-only; calling with a callable name AND
    # a schema would treat the callable as a name override (likely a misuse).
    """Register a Python function as a verl tool.

    The OpenAI tool schema is inferred from the function via
    :func:`transformers.utils.get_json_schema`, so the function **must** carry:

    - a Google-style docstring summarising the tool;
    - a ``Args:`` block describing every parameter;
    - a type hint on every parameter.

    If any of those are missing, ``transformers`` raises
    ``DocstringParsingException`` / ``TypeHintParsingException`` at
    registration time.

    Supports both decorator forms::

        @function_tool                          # bare, name = fn.__name__
        def web_search(...): ...

        @function_tool("web_search")            # rename the tool
        def search(...): ...

    Args:
        name: Tool name exposed to the LLM. Defaults to the function name.
            When used as a bare ``@function_tool`` (no parentheses), this
            position receives the function being decorated.
        schema: Skip schema inference entirely and use the supplied
            ``OpenAIFunctionToolSchema`` (or a dict matching that shape) as-is.
            Use this only if your function's signature can't be expressed in
            JSON Schema -- the normal path is to fix the function.
    """

    def _make_decorator(tool_name_override: Optional[str]):
        # [FUNCTION DOC]
        # HELPER: builds the actual decorator closed over a name override. Inputs: tool_name_override - name or None.
        # Outputs: decorator function.
        def decorator(fn: Callable):
            # [FUNCTION DOC]
            # HELPER: Behavior: resolves the tool name (override or fn's name); builds schema from: a given schema object as-is,
            # a dict validated into OpenAIFunctionToolSchema, or inferred from fn. Creates a FunctionTool entry. If a different
            # function is already registered under that name raises ValueError; re-registering the same function is allowed
            # (overwrites entry). Stores entry in the registry, logs at info, returns fn unchanged.
            # Exceptions: ValueError on name conflict; pydantic validation errors; schema inference errors.
            tool_name = tool_name_override or fn.__name__

            if isinstance(schema, OpenAIFunctionToolSchema):
                built_schema = schema
            elif isinstance(schema, dict):
                built_schema = OpenAIFunctionToolSchema.model_validate(schema)
            else:
                built_schema = _build_schema_from_fn(fn, tool_name)

            entry = FunctionTool(
                name=tool_name,
                fn=fn,
                tool_schema=built_schema,
                is_async=inspect.iscoroutinefunction(fn),
            )

            existing = FUNCTION_TOOL_REGISTRY.get(tool_name)
            if existing is not None and existing.fn is not fn:
                raise ValueError(
                    f"Function tool '{tool_name}' is already registered to "
                    f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                    f"refusing to overwrite with {fn.__module__}.{fn.__qualname__}."
                )
            FUNCTION_TOOL_REGISTRY[tool_name] = entry
            logger.info("Registered function tool '%s' from %s.%s", tool_name, fn.__module__, fn.__qualname__)
            return fn

        return decorator

    if callable(name) and schema is None:
        fn = name
        return _make_decorator(None)(fn)

    return _make_decorator(name)


def get_function_tool(name: str) -> FunctionTool:
    # [FUNCTION DOC]
    # Behavior: returns the registry entry for name. Exceptions: KeyError with guidance message if not registered.
    """Look up a registered function tool by name. Raises ``KeyError`` if absent."""
    if name not in FUNCTION_TOOL_REGISTRY:
        raise KeyError(
            f"Function tool '{name}' not found in registry. Make sure its defining "
            f"file is referenced via the rollout `function_tool_path` config."
        )
    return FUNCTION_TOOL_REGISTRY[name]


def load_function_tools_from_path(path: str) -> list[FunctionTool]:
    # [FUNCTION DOC]
    # Behavior: resolves path to absolute; raises FileNotFoundError if not a file; returns the cached tool list if
    # already loaded. Otherwise snapshots registry keys, executes the file as a synthetic module (name derived from the
    # path with separators and dots replaced by underscores) registered in sys.modules, then collects newly-added
    # registry names (sorted). Warns if none were added, else logs info. Caches and returns the list of new FunctionTools.
    # Exceptions: FileNotFoundError; ImportError if no import spec can be created; anything raised by the executed file.
    # Side effects: executes arbitrary code, mutates sys.modules, the registry and the path cache.
    # Note: tools in the file whose names were already registered before loading are not included in the result.
    """Execute a Python file at ``path`` and return its registered function tools."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"function_tool_path does not exist: {path}")

    if abs_path in _LOADED_FUNCTION_TOOL_PATHS:
        return _LOADED_FUNCTION_TOOL_PATHS[abs_path]

    before = set(FUNCTION_TOOL_REGISTRY)

    # Use a path-derived synthetic module name so the imported file can
    # ``from X import Y`` its siblings via ``sys.modules``.
    module_name = "_verl_function_tools_" + abs_path.replace(os.sep, "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for function_tool_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    new_names = sorted(set(FUNCTION_TOOL_REGISTRY) - before)
    if not new_names:
        logger.warning(
            "function_tool_path '%s' loaded but no @function_tool decorators found; "
            "did you forget to apply the decorator?",
            path,
        )
    else:
        logger.info("Loaded %d function tool(s) from %s: %s", len(new_names), path, new_names)

    tools = [FUNCTION_TOOL_REGISTRY[name] for name in new_names]
    _LOADED_FUNCTION_TOOL_PATHS[abs_path] = tools
    return tools


def _build_schema_from_fn(fn: Callable, tool_name: str) -> OpenAIFunctionToolSchema:
    # [FUNCTION DOC]
    # HELPER: Behavior: rejects functions with *args/**kwargs-style parameters, then uses transformers get_json_schema
    # to build the schema from the docstring and type hints, overrides the function name with tool_name, validates
    # into OpenAIFunctionToolSchema. Exceptions: ValueError for variadic params; transformers docstring/type-hint
    # parsing exceptions; pydantic validation errors.
    """Infer the OpenAI tool schema for ``fn`` via transformers.

    The heavy lifting (signature inspection + Google-style docstring parsing
    + JSON-Schema type mapping) is delegated to
    :func:`transformers.utils.get_json_schema`.
    """
    sig = inspect.signature(fn)
    variadic = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        raise ValueError(
            f"@function_tool '{tool_name}' ({fn.__module__}.{fn.__qualname__}) "
            f"declares variadic parameter(s) {variadic}, which can't be "
            f"expressed in an OpenAI tool schema. Replace them with explicit "
            f"named parameters."
        )

    raw = get_json_schema(fn)
    raw["function"]["name"] = tool_name
    return OpenAIFunctionToolSchema.model_validate(raw)


def normalize_function_tool_return(ret: Any) -> tuple[ToolResponse, float, dict]:
    # [FUNCTION DOC]
    # Behavior: normalizes a tool function's return value to (ToolResponse, reward float, metrics dict):
    # ToolResponse as-is; str as text; dict as JSON text (non-ASCII kept); tuple of length 1-3 where element 0 is coerced
    # to a response, element 1 (None means 0.0) is cast to float, element 2 (None means empty) is cast to dict;
    # anything else becomes its string form. Missing reward/metrics default to 0.0 / empty.
    # Exceptions: TypeError for tuples of length 0 or more than 3; ValueError/TypeError from float/dict casts.
    """Coerce a function's return value into the ``(ToolResponse, reward, metrics)`` triple.

    Accepted shapes:

    - ``ToolResponse``  -> as-is, reward 0.0, metrics {}
    - ``str``           -> ``ToolResponse(text=ret)``
    - ``dict``          -> ``ToolResponse(text=json.dumps(ret))``
    - ``(response,)`` / ``(response, reward)`` / ``(response, reward, metrics)``
      -- ``reward`` may be ``None`` (treated as ``0.0``); ``metrics`` may be
      ``None`` (treated as ``{}``).
    - anything else     -> ``ToolResponse(text=str(ret))``

    Tuples of length 0 or >= 4 raise ``TypeError`` rather than being silently
    stringified, since they almost always signal a tool authoring bug.
    ``None`` is detected via ``is None`` rather than truthiness, so a
    legitimate ``0`` / ``0.0`` / ``False`` reward is preserved.
    """
    if isinstance(ret, ToolResponse):
        return ret, 0.0, {}
    if isinstance(ret, str):
        return ToolResponse(text=ret), 0.0, {}
    if isinstance(ret, dict):
        return ToolResponse(text=json.dumps(ret, ensure_ascii=False)), 0.0, {}
    if isinstance(ret, tuple):
        if not 1 <= len(ret) <= 3:
            raise TypeError(
                f"@function_tool return tuple must have length 1, 2, or 3 "
                f"(got length {len(ret)}: {ret!r}). Use (response,), "
                f"(response, reward), or (response, reward, metrics)."
            )
        response = _coerce_response(ret[0])
        reward = 0.0 if len(ret) < 2 or ret[1] is None else float(ret[1])
        metrics = {} if len(ret) < 3 or ret[2] is None else dict(ret[2])
        return response, reward, metrics
    return ToolResponse(text=str(ret)), 0.0, {}


def _coerce_response(value: Any) -> ToolResponse:
    # [FUNCTION DOC]
    # HELPER: Converts a single value to ToolResponse: ToolResponse unchanged, str as text, dict as JSON text, else string form.
    if isinstance(value, ToolResponse):
        return value
    if isinstance(value, str):
        return ToolResponse(text=value)
    if isinstance(value, dict):
        return ToolResponse(text=json.dumps(value, ensure_ascii=False))
    return ToolResponse(text=str(value))
