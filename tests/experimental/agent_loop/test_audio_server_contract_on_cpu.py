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
# Static source-contract tests (no imports of the code under test) verifying
# that audio data and mm_processor_kwargs are plumbed through the LLM server
# client, the single-turn/tool agent loops, and the vLLM async server. Uses AST
# inspection or substring checks on source files resolved relative to repo root.

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LLM_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/llm_server.py"
SINGLE_TURN_AGENT_LOOP_SOURCE = REPO_ROOT / "verl/experimental/agent_loop/single_turn_agent_loop.py"
TOOL_AGENT_LOOP_SOURCE = REPO_ROOT / "verl/experimental/agent_loop/tool_agent_loop.py"
VLLM_SERVER_SOURCE = REPO_ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"


def _load_module_ast(path: Path) -> ast.Module:
    # [FUNCTION DOC]
    # HELPER: reads a source file as UTF-8 and returns its parsed AST module.
    # Exceptions: OSError if missing, SyntaxError if unparsable.
    return ast.parse(path.read_text(encoding="utf-8"))


def test_async_server_manager_generate_accepts_audio_and_mm_kwargs() -> None:
    # [FUNCTION DOC]
    # Behavior: finds async method generate on top-level class LLMServerClient in
    # llm_server.py and asserts its keyword-only args include audio_data and
    # mm_processor_kwargs. Fails if the class/method is not found.
    module = _load_module_ast(LLM_SERVER_SOURCE)

    generate_fn = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMServerClient":
            for inner in node.body:
                if isinstance(inner, ast.AsyncFunctionDef) and inner.name == "generate":
                    generate_fn = inner
                    break

    assert generate_fn is not None
    arg_names = [arg.arg for arg in generate_fn.args.kwonlyargs]
    assert "audio_data" in arg_names
    assert "mm_processor_kwargs" in arg_names


def test_async_server_manager_generate_forwards_audio_and_mm_kwargs() -> None:
    # [FUNCTION DOC]
    # Behavior: asserts llm_server.py source literally assigns audio_data and
    # mm_processor_kwargs into the multimodal_kwargs mapping.
    source = LLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'multimodal_kwargs["audio_data"] = audio_data' in source
    assert 'multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs' in source


def test_fully_async_server_manager_generate_forwards_audio_and_mm_kwargs() -> None:
    # [FUNCTION DOC]
    # Behavior: asserts the single-turn agent loop passes audio_data=audios and
    # mm_processor_kwargs, and the tool agent loop passes them from agent_data.
    # KNOWN ISSUE: the name mentions a "fully async server manager" but it actually checks the agent loop sources.
    single_turn_source = SINGLE_TURN_AGENT_LOOP_SOURCE.read_text(encoding="utf-8")
    assert "audio_data=audios" in single_turn_source
    assert "mm_processor_kwargs=mm_processor_kwargs" in single_turn_source

    tool_source = TOOL_AGENT_LOOP_SOURCE.read_text(encoding="utf-8")
    assert "audio_data=agent_data.audio_data" in tool_source
    assert "mm_processor_kwargs=agent_data.mm_processor_kwargs" in tool_source


def test_vllm_generate_includes_audio_and_mm_processor_kwargs() -> None:
    # [FUNCTION DOC]
    # Behavior: asserts vllm_async_server.py puts audio into multi_modal_data["audio"]
    # and mm_processor_kwargs into prompt_kwargs.
    source = VLLM_SERVER_SOURCE.read_text(encoding="utf-8")
    assert 'multi_modal_data["audio"] = audio_data' in source
    assert 'prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs' in source
