# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
# Defines BaseTool, the abstract-ish interface for stateful agent-loop tools (create / execute / calc_reward / release).

import json
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .schemas import OpenAIFunctionToolSchema, ToolResponse


class BaseTool:
    # [CLASS DOC]
    # Base class that concrete tools subclass. Holds the raw config dict, the OpenAI tool schema and the tool name.
    # Default implementations are no-ops so subclasses only override what they need. All lifecycle methods are async.
    """Base class for tools.

    A tool should support the following methods:

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [FUNCTION DOC]
        # Behavior: stores config; uses the passed tool_schema or falls back to get_openai_tool_schema(); asserts a schema
        # exists; sets name from schema.function.name; prints the schema as indented JSON (unset/None fields excluded).
        # Inputs: config - tool config dict; tool_schema - OpenAIFunctionToolSchema or None.
        # Exceptions: AssertionError if no schema.
        # Side effects: prints to stdout.
        # KNOWN ISSUE: the fallback calls the base get_openai_tool_schema, which reads self.tool_schema before it is assigned,
        # so passing None raises AttributeError unless a subclass overrides get_openai_tool_schema.
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # [FUNCTION DOC]
        # Returns the stored tool schema. Subclasses may override to build a schema dynamically.
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        # [FUNCTION DOC]
        # Behavior: returns (instance_id, empty ToolResponse); generates a uuid4 string when instance_id is None.
        # Inputs: instance_id - optional id; kwargs ignored. Outputs: tuple(str id, ToolResponse). Side effects: none.
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            return str(uuid4()), ToolResponse()
        else:
            return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        # [FUNCTION DOC]
        # Behavior: default execution, traced via rollout_trace_op; returns a fixed text response "Updated the tool state.",
        # reward 0.0 and empty metrics. Inputs: instance_id, parameters (mapping of args), kwargs ignored.
        # Outputs: (ToolResponse, float reward, dict metrics). Side effects: trace recording by decorator.
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The json string of the parameters of the tool.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The ToolResponse object containing text, image, and/or video content.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        return ToolResponse(text="Updated the tool state."), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # [FUNCTION DOC]
        # Default reward calculation; always returns 0.0. Inputs: instance_id, kwargs ignored.
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # [FUNCTION DOC]
        # Default release; does nothing. Inputs: instance_id, kwargs ignored. Outputs: None.
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass
