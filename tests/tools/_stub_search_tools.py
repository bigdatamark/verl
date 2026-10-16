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
# HELPER: Stub BaseTool implementations (search and crawl) used by tool tests. Each records the
# parameters of every call and returns deterministic text with zero reward.

"""
Stub native tools for testing
"""

from __future__ import annotations

from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class StubSearchTool(BaseTool):
    # [CLASS DOC]
    # HELPER: Fake search tool. Attribute calls - list of parameter mappings passed to execute, in order.
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [FUNCTION DOC]
        # Delegates to BaseTool init with config and schema, then initialises an empty calls list.
        super().__init__(config, tool_schema)
        self.calls: list[dict[str, Any]] = []

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # [FUNCTION DOC]
        # Async. Appends parameters to calls; reads "query_list" (missing/empty -> empty list); returns a
        # ToolResponse whose text is "hits-for:<q>" per query joined by "; " (or "hits-for:<empty>" when none),
        # reward 0.0 and metrics {"num_queries": count}. instance_id and kwargs are ignored.
        self.calls.append(parameters)
        queries = parameters.get("query_list") or []
        text = "; ".join(f"hits-for:{q}" for q in queries) or "hits-for:<empty>"
        return ToolResponse(text=text), 0.0, {"num_queries": len(queries)}


class StubCrawlTool(BaseTool):
    # [CLASS DOC]
    # HELPER: Fake crawl tool with the same calls recording as StubSearchTool.
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [FUNCTION DOC]
        # Delegates to BaseTool init and initialises an empty calls list.
        super().__init__(config, tool_schema)
        self.calls: list[dict[str, Any]] = []

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # [FUNCTION DOC]
        # Async. Appends parameters to calls; reads "url_list" (default empty); returns ToolResponse text
        # "crawled:" followed by the URLs joined by commas, reward 0.0 and metrics {"num_urls": count}.
        self.calls.append(parameters)
        urls = parameters.get("url_list") or []
        text = "crawled:" + ",".join(urls)
        return ToolResponse(text=text), 0.0, {"num_urls": len(urls)}
