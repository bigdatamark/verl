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
# Pydantic models describing OpenAI-format tool schemas, parsed tool calls and tool responses.

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OpenAIFunctionPropertySchema(BaseModel):
    # [CLASS DOC]
    # Schema of a single function parameter: type (string or list of strings for unions), optional description, optional enum values.
    """The schema of a parameter in OpenAI format."""

    # Union's type is list[str], e.g. ["integer", "number"] for int | float unions.
    type: str | list[str]
    description: str | None = None
    # JSON Schema's ``enum`` accepts any JSON value, not just strings, so
    # ``Literal[1, 2, 3]`` -> ``enum: [1, 2, 3]`` is a valid schema.
    enum: list[Any] | None = None


class OpenAIFunctionParametersSchema(BaseModel):
    # [CLASS DOC]
    # Parameters object of a function schema: type (normally "object"), properties mapping name to property schema, required names (default empty).
    """The schema of parameters in OpenAI format."""

    type: str
    properties: dict[str, OpenAIFunctionPropertySchema]
    # ``required`` can be omitted when no parameter is required.
    required: list[str] = Field(default_factory=list)


class OpenAIFunctionSchema(BaseModel):
    # [CLASS DOC]
    # Function schema: name, description, parameters (default: empty object schema), strict flag (default false).
    """The schema of a function in OpenAI format."""

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(type="object", properties={}, required=[])
    )
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    # [CLASS DOC]
    # Top-level tool schema: type (e.g. "function") plus function schema.
    """The schema of a tool in OpenAI format."""

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    # [CLASS DOC]
    # Raw tool call as parsed from model output: name and arguments as a JSON string.
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    # [CLASS DOC]
    # Decoded tool call: name and arguments as a mapping.
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        # [FUNCTION DOC]
        # Behavior: JSON-decodes parsed_schema.arguments; on decode failure or a non-mapping result, uses an empty mapping
        # and flags an error. Inputs: parsed_schema - OpenAIFunctionParsedSchema. Outputs: (OpenAIFunctionCallSchema,
        # bool has_decode_error). Exceptions: none raised for bad JSON.
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    # [CLASS DOC]
    # OpenAI tool call record: id, type fixed to "function", decoded function call.
    """The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


class ToolResponse(BaseModel):
    # [CLASS DOC]
    # Multimodal tool output: optional text, optional list of images, optional list of videos.
    """The response from a tool execution."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        # [FUNCTION DOC]
        # Behavior: pre-validation hook; ensures image and video, when present in the raw input mapping, are lists.
        # Inputs: values - raw input mapping. Outputs: values unchanged. Exceptions: ValueError with guidance if not a list.
        # Note: a present-but-None image/video also fails the list check, so explicitly passing None raises.
        if "image" in values and not isinstance(values["image"], list):
            raise ValueError(
                f"Image must be a list, but got {type(values['image'])}. Please check the tool.execute(). "
                f"For single images, wrap in a list: [image]. "
                f"Example: {{'image': [img1]}} or {{'image': [img1, img2, ...]}}."
            )
        if "video" in values and not isinstance(values["video"], list):
            raise ValueError(
                f"Video must be a list, but got {type(values['video'])}. Please check the tool.execute(). "
                f"For single videos, wrap in a list: [video]. "
                f"Example: {{'video': [video1]}} or {{'video': [video1, video2, ...]}}."
            )

        return values

    def is_empty(self) -> bool:
        # [FUNCTION DOC]
        # Returns True when text, image and video are all falsy.
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        # [FUNCTION DOC]
        # Returns truthy when text is non-empty and there are no images or videos.
        # KNOWN ISSUE: annotated as bool but returns the text string itself (or a falsy value) rather than a strict bool.
        return self.text and not self.image and not self.video
