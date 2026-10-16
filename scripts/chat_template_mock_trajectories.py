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
# Deterministic, message-level mock conversations used by the chat-template checker and related
# tests. Text part: four deterministic FunctionTools (get_weather, estimate_travel, reserve_room,
# lookup_incident) in TOOLS; one single-turn trajectory (singleturnchat) and four tool-agent
# trajectories (multiturnsingletool, multiturnmultitool, multiturnretryuser, multiturnretrysystem)
# collected in TRAJECTORIES plus by-name lookup tables. VL part: lazily built (Pillow imported only
# when building) trajectories with solid-color images in prompts and tool responses
# (vl_singleturnchat, vl_multiturnsingletool, vl_multiturnmultitool), whose assistant tool-call
# turns embed parser-specific raw tool-call text as content.
"""Structured mock trajectories for the chat-template checker."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from verl.tools.function_tool import FunctionTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


@dataclass(frozen=True)
class TrajectoryStep:
    # [CLASS DOC]
    # Immutable record of one assistant message (OpenAI-style dict, may include tool_calls) and the
    # tuple of environment messages (tool/user/system) appended after it.
    """One assistant generation followed by optional environment messages."""

    assistant: dict[str, Any]
    appended_messages: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SingleTurnTrajectory:
    # [CLASS DOC]
    # Immutable single-turn case: name, description, raw_prompt messages, the assistant response text,
    # and the expected number of turns (default 2: prompt + response).
    """A single prompt/assistant-response trajectory for SingleTurnAgentLoop."""

    name: str
    description: str
    raw_prompt: tuple[dict[str, Any], ...]
    assistant_response: str
    expected_num_turns: int = 2


@dataclass(frozen=True)
class ToolAgentTrajectory:
    # [CLASS DOC]
    # Immutable multi-turn tool-agent case: name, description, raw_prompt, ordered steps, the
    # FunctionTools available, and the maximum number of parallel tool calls (default 1).
    """A messages-level trajectory independent of any rollout server."""

    name: str
    description: str
    raw_prompt: tuple[dict[str, Any], ...]
    steps: tuple[TrajectoryStep, ...]
    tools: tuple[FunctionTool, ...]
    max_parallel_calls: int = 1

    @property
    def expected_generation_turns(self) -> int:
        # [FUNCTION DOC]
        # Number of assistant generations, i.e. the number of steps.
        return len(self.steps)

    @property
    def expected_num_turns(self) -> int:
        # [FUNCTION DOC]
        # 1 (prompt) + number of steps + number of steps that append environment messages.
        environment_turns = sum(1 for step in self.steps if step.appended_messages)
        return 1 + self.expected_generation_turns + environment_turns

    def tool_schemas(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns each tool's OpenAI schema dumped to a dict, omitting unset and None fields.
        return [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in self.tools]

    def full_messages(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Returns the whole conversation as a new list: shallow copies of the prompt messages, then for each
        # step a deep (JSON) copy of the assistant message followed by copies of its appended messages.
        messages = [dict(message) for message in self.raw_prompt]
        for step in self.steps:
            messages.append(_clone_message(step.assistant))
            messages.extend(_clone_message(message) for message in step.appended_messages)
        return messages


MockTrajectory = SingleTurnTrajectory | ToolAgentTrajectory


def _clone_message(message: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Deep-copies a JSON-compatible message via a JSON round trip.
    return json.loads(json.dumps(message, ensure_ascii=False))


def _schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> OpenAIFunctionToolSchema:
    # [FUNCTION DOC]
    # HELPER: Builds and validates an OpenAIFunctionToolSchema of type function with the given name,
    # description, object parameters with the given properties, and required list.
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    )


def _assistant(content: str, tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds an assistant message with content, adding tool_calls only when a non-empty list is given.
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds an OpenAI-style tool call dict with id, type "function", and function name plus
    # arguments (kept as a dict, not a JSON string).
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _tool_message(call_id: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC]
    # HELPER: Builds a tool-role message with tool_call_id, name, and the payload serialized as compact
    # JSON (no spaces, non-ASCII preserved) as content.
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
    }


def get_weather(city: str, day: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Deterministic mock tool: looks up a fixed table keyed by (city, day) and returns city and day merged
    # with the entry, or condition "unknown" and high_c None when absent.
    """Return a deterministic weather snapshot for a city and day."""
    table = {
        ("Seattle", "today"): {"condition": "rain", "high_c": 12, "wind_kph": 22},
        ("Portland", "today"): {"condition": "cloudy", "high_c": 16, "wind_kph": 11},
        ("Denver", "tomorrow"): {"condition": "clear", "high_c": 24, "wind_kph": 18},
        ("Boulder", "tomorrow"): {"condition": "storms", "high_c": 21, "wind_kph": 31},
    }
    return {"city": city, "day": day, **table.get((city, day), {"condition": "unknown", "high_c": None})}


def estimate_travel(origin: str, destination: str, depart_window: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Deterministic mock tool: returns route "origin->destination", depart_window, merged with the fixed
    # table entry (minutes, risk) for the triple, or nothing extra when absent.
    """Estimate deterministic travel time for a simple route."""
    route = f"{origin}->{destination}"
    table = {
        ("Depot A", "West Clinic", "morning"): {"minutes": 32, "risk": "low"},
        ("Depot A", "East Clinic", "morning"): {"minutes": 51, "risk": "medium"},
        ("Depot B", "West Clinic", "afternoon"): {"minutes": 44, "risk": "medium"},
        ("Depot B", "East Clinic", "afternoon"): {"minutes": 28, "risk": "low"},
    }
    return {"route": route, "depart_window": depart_window, **table.get((origin, destination, depart_window), {})}


def reserve_room(room: str, start_time: str, end_time: str | None = None) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Deterministic mock tool: returns a failure dict (ok false, error, room, start_time) if end_time is
    # missing or empty; otherwise a success dict with the fixed confirmation "room-4821".
    """Reserve a room if the request has a complete time range."""
    if not end_time:
        return {"ok": False, "error": "end_time is required", "room": room, "start_time": start_time}
    return {"ok": True, "room": room, "start_time": start_time, "end_time": end_time, "confirmation": "room-4821"}


def lookup_incident(incident_id: str) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Deterministic mock tool: only "INC-1042" succeeds (status mitigated, owner edge-oncall); any other id
    # returns ok false with a format error and the received id.
    """Look up a deterministic support incident."""
    if incident_id != "INC-1042":
        return {"ok": False, "error": "incident_id must use the INC-#### format", "received": incident_id}
    return {"ok": True, "incident_id": incident_id, "status": "mitigated", "owner": "edge-oncall"}


def build_common_tools() -> tuple[FunctionTool, ...]:
    # [FUNCTION DOC]
    # HELPER: Returns a tuple of the four FunctionTools wrapping the mock tool functions with their schemas
    # (reserve_room's end_time allows string or null and is optional).
    return (
        FunctionTool(
            name="get_weather",
            fn=get_weather,
            tool_schema=_schema(
                "get_weather",
                "Return a deterministic weather snapshot.",
                {
                    "city": {"type": "string", "description": "City to check."},
                    "day": {"type": "string", "description": "Day label such as today or tomorrow."},
                },
                ["city", "day"],
            ),
        ),
        FunctionTool(
            name="estimate_travel",
            fn=estimate_travel,
            tool_schema=_schema(
                "estimate_travel",
                "Estimate deterministic travel time for a route.",
                {
                    "origin": {"type": "string", "description": "Starting location."},
                    "destination": {"type": "string", "description": "Destination location."},
                    "depart_window": {"type": "string", "description": "Departure window label."},
                },
                ["origin", "destination", "depart_window"],
            ),
        ),
        FunctionTool(
            name="reserve_room",
            fn=reserve_room,
            tool_schema=_schema(
                "reserve_room",
                "Reserve a meeting room with an explicit time range.",
                {
                    "room": {"type": "string", "description": "Room name."},
                    "start_time": {"type": "string", "description": "Local start time."},
                    "end_time": {"type": ["string", "null"], "description": "Local end time."},
                },
                ["room", "start_time"],
            ),
        ),
        FunctionTool(
            name="lookup_incident",
            fn=lookup_incident,
            tool_schema=_schema(
                "lookup_incident",
                "Look up an incident status.",
                {
                    "incident_id": {"type": "string", "description": "Incident id in INC-#### format."},
                },
                ["incident_id"],
            ),
        ),
    )


TOOLS = build_common_tools()


# system + user -> assistant
SINGLE_TURN_CHAT = SingleTurnTrajectory(
    name="singleturnchat",
    description="One user request followed by one plain assistant response.",
    raw_prompt=(
        {"role": "system", "content": "You answer scheduling questions in one concise sentence."},
        {"role": "user", "content": "Can I move the Friday demo to 15:30 if the room is free?"},
    ),
    assistant_response="Yes, move the Friday demo to 15:30 if the room remains free and notify the attendees.",
)


# system + user
# -> assistant(tool_call: get_weather)
# -> tool(get_weather)
# -> assistant(tool_call: get_weather)
# -> tool(get_weather)
# -> assistant
MULTITURN_SINGLE_TOOL = ToolAgentTrajectory(
    name="multiturnsingletool",
    description="Two sequential assistant tool calls, one tool response per turn, then a final answer.",
    raw_prompt=(
        {"role": "system", "content": "You are a concise trip-planning assistant. Use tools for live facts."},
        {
            "role": "user",
            "content": "Pick a better city for an outdoor team lunch today: Seattle or Portland. Check both first.",
        },
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will check Seattle first, then compare it with Portland.",
                [_tool_call("call_weather_seattle", "get_weather", {"city": "Seattle", "day": "today"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_weather_seattle",
                    "get_weather",
                    {"city": "Seattle", "day": "today", "condition": "rain", "high_c": 12, "wind_kph": 22},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Seattle looks wet, so I will check Portland before deciding.",
                [_tool_call("call_weather_portland", "get_weather", {"city": "Portland", "day": "today"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_weather_portland",
                    "get_weather",
                    {"city": "Portland", "day": "today", "condition": "cloudy", "high_c": 16, "wind_kph": 11},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Choose Portland. It is cloudy rather than rainy, warmer, and less windy; "
                "keep Seattle as the indoor fallback."
            )
        ),
    ),
)


# system + user
# -> assistant(two parallel tool_calls: estimate_travel + get_weather)
# -> tool(estimate_travel) + tool(get_weather)
# -> assistant(two parallel tool_calls: estimate_travel + get_weather)
# -> tool(estimate_travel) + tool(get_weather)
# -> assistant
MULTITURN_MULTI_TOOL = ToolAgentTrajectory(
    name="multiturnmultitool",
    description="Two assistant turns with parallel tool calls, followed by a final comparison.",
    raw_prompt=(
        {"role": "system", "content": "You coordinate clinic deliveries and summarize tradeoffs briefly."},
        {
            "role": "user",
            "content": (
                "Compare morning deliveries from Depot A to the West and East clinics, then compare the afternoon "
                "Depot B options. Use weather and travel estimates before recommending a route."
            ),
        },
    ),
    tools=TOOLS,
    max_parallel_calls=2,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will check the morning route and destination weather together.",
                [
                    _tool_call(
                        "call_travel_west",
                        "estimate_travel",
                        {"origin": "Depot A", "destination": "West Clinic", "depart_window": "morning"},
                    ),
                    _tool_call("call_weather_denver", "get_weather", {"city": "Denver", "day": "tomorrow"}),
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_travel_west",
                    "estimate_travel",
                    {"route": "Depot A->West Clinic", "depart_window": "morning", "minutes": 32, "risk": "low"},
                ),
                _tool_message(
                    "call_weather_denver",
                    "get_weather",
                    {"city": "Denver", "day": "tomorrow", "condition": "clear", "high_c": 24, "wind_kph": 18},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Now I will check the afternoon alternative and the other destination weather.",
                [
                    _tool_call(
                        "call_travel_east",
                        "estimate_travel",
                        {"origin": "Depot B", "destination": "East Clinic", "depart_window": "afternoon"},
                    ),
                    _tool_call("call_weather_boulder", "get_weather", {"city": "Boulder", "day": "tomorrow"}),
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_travel_east",
                    "estimate_travel",
                    {"route": "Depot B->East Clinic", "depart_window": "afternoon", "minutes": 28, "risk": "low"},
                ),
                _tool_message(
                    "call_weather_boulder",
                    "get_weather",
                    {"city": "Boulder", "day": "tomorrow", "condition": "storms", "high_c": 21, "wind_kph": 31},
                ),
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "Use Depot A to West Clinic in the morning. It is slightly longer than the best afternoon estimate, "
                "but the weather is clear and the travel risk is low; avoid the Boulder storm window."
            )
        ),
    ),
)


# system + user
# -> assistant(tool_call: reserve_room, invalid arguments)
# -> user(retry instruction)
# -> assistant(tool_call: reserve_room, corrected arguments)
# -> tool(reserve_room)
# -> assistant
MULTITURN_RETRY_USER = ToolAgentTrajectory(
    name="multiturnretryuser",
    description="A failed tool call is followed by a user retry instruction, then the assistant retries correctly.",
    raw_prompt=(
        {"role": "system", "content": "You reserve rooms. If a request is incomplete, recover with one concise retry."},
        {"role": "user", "content": "Reserve Maple Room tomorrow from 10:00 to 10:30 and confirm the booking."},
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will try to reserve the room now.",
                [_tool_call("call_room_incomplete", "reserve_room", {"room": "Maple Room", "start_time": "10:00"})],
            ),
            appended_messages=(
                {
                    "role": "user",
                    "content": (
                        "The room reservation tool rejected the previous call because end_time was missing. "
                        "Retry with room=Maple Room, start_time=10:00, end_time=10:30."
                    ),
                },
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "I missed the end time. I will retry with the complete time range.",
                [
                    _tool_call(
                        "call_room_retry",
                        "reserve_room",
                        {"room": "Maple Room", "start_time": "10:00", "end_time": "10:30"},
                    )
                ],
            ),
            appended_messages=(
                _tool_message(
                    "call_room_retry",
                    "reserve_room",
                    {
                        "ok": True,
                        "room": "Maple Room",
                        "start_time": "10:00",
                        "end_time": "10:30",
                        "confirmation": "room-4821",
                    },
                ),
            ),
        ),
        TrajectoryStep(assistant=_assistant("Maple Room is reserved from 10:00 to 10:30. Confirmation: room-4821.")),
    ),
)


# system + user
# -> assistant(tool_call: lookup_incident, invalid arguments)
# -> system(retry directive)
# -> assistant(tool_call: lookup_incident, corrected arguments)
# -> tool(lookup_incident)
# -> assistant
MULTITURN_RETRY_SYSTEM = ToolAgentTrajectory(
    name="multiturnretrysystem",
    description="A failed tool call is followed by a system retry directive, then the assistant retries correctly.",
    raw_prompt=(
        {"role": "system", "content": "You are an incident assistant. Respect system retry directives exactly."},
        {"role": "user", "content": "Check the mitigation status for incident 1042 and summarize the owner."},
    ),
    tools=TOOLS,
    steps=(
        TrajectoryStep(
            assistant=_assistant(
                "I will look up the incident status.",
                [_tool_call("call_incident_bad", "lookup_incident", {"incident_id": "1042"})],
            ),
            appended_messages=(
                {
                    "role": "system",
                    "content": (
                        "Tool validation failed: incident_id must use the INC-#### format. "
                        "Retry with incident_id=INC-1042 before answering."
                    ),
                },
            ),
        ),
        TrajectoryStep(
            assistant=_assistant(
                "I will retry with the required incident id format.",
                [_tool_call("call_incident_retry", "lookup_incident", {"incident_id": "INC-1042"})],
            ),
            appended_messages=(
                _tool_message(
                    "call_incident_retry",
                    "lookup_incident",
                    {"ok": True, "incident_id": "INC-1042", "status": "mitigated", "owner": "edge-oncall"},
                ),
            ),
        ),
        TrajectoryStep(assistant=_assistant("Incident INC-1042 is mitigated, and the current owner is edge-oncall.")),
    ),
)


SINGLE_TURN_TRAJECTORIES: tuple[SingleTurnTrajectory, ...] = (SINGLE_TURN_CHAT,)


TOOL_AGENT_TRAJECTORIES: tuple[ToolAgentTrajectory, ...] = (
    MULTITURN_SINGLE_TOOL,
    MULTITURN_MULTI_TOOL,
    MULTITURN_RETRY_USER,
    MULTITURN_RETRY_SYSTEM,
)


TRAJECTORIES: tuple[MockTrajectory, ...] = SINGLE_TURN_TRAJECTORIES + TOOL_AGENT_TRAJECTORIES


TRAJECTORY_BY_NAME: dict[str, MockTrajectory] = {trajectory.name: trajectory for trajectory in TRAJECTORIES}
TOOL_AGENT_TRAJECTORY_BY_NAME = {trajectory.name: trajectory for trajectory in TOOL_AGENT_TRAJECTORIES}


def get_trajectory(name: str) -> MockTrajectory:
    # [FUNCTION DOC]
    # Looks up a text trajectory by name. Exceptions: ValueError listing valid names if unknown.
    try:
        return TRAJECTORY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unknown trajectory {name!r}. Choices: {sorted(TRAJECTORY_BY_NAME)}") from exc


def select_trajectories(names: list[str] | None = None) -> list[MockTrajectory]:
    # [FUNCTION DOC]
    # Returns all text trajectories when names is empty/None, else those named (in the given order).
    # Exceptions: ValueError for unknown names.
    if not names:
        return list(TRAJECTORIES)
    return [get_trajectory(name) for name in names]


def get_tool_agent_trajectory(name: str) -> ToolAgentTrajectory:
    # [FUNCTION DOC]
    # Looks up a tool-agent trajectory by name. Exceptions: ValueError listing valid names if unknown.
    try:
        return TOOL_AGENT_TRAJECTORY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown tool-agent trajectory {name!r}. Choices: {sorted(TOOL_AGENT_TRAJECTORY_BY_NAME)}"
        ) from exc


def select_tool_agent_trajectories(names: list[str] | None = None) -> list[ToolAgentTrajectory]:
    # [FUNCTION DOC]
    # Returns all tool-agent trajectories when names is empty/None, else those named.
    if not names:
        return list(TOOL_AGENT_TRAJECTORIES)
    return [get_tool_agent_trajectory(name) for name in names]


# =============================================================================
# Vision-language (VL) trajectories
#
# These embed real images in both the initial user prompt and inside tool
# responses.
#
# They are intentionally kept OUT of the text ``TRAJECTORIES`` tuple: the text
# chat-template checker renders trajectories with a bare tokenizer and cannot
# expand image content. Images are generated lazily in ``build_vl_trajectories``
# so importing this module for text-only use never requires Pillow.
#
# VL chat templates render assistant ``content`` verbatim but drop a structured
# ``tool_calls`` field, so a tool-call turn embeds the parser's raw tool-call
# text as content (matching how a VL model emits tool calls at rollout time),
# and the real tool parser can then extract it.
# =============================================================================


def _format_tool_call_text(tool_parser: str, name: str, arguments: dict[str, Any]) -> str:
    # [FUNCTION DOC]
    # HELPER: Renders one tool call as raw model-generated text for a parser format: hermes/qwen3_coder
    # wrap a JSON object of name and arguments in <tool_call> tags on separate lines; glm uses
    # <tool_call>name followed by <arg_key>/<arg_value> pairs (values stringified); kimi uses
    # <|tool_call_begin|>name<|tool_call_argument_begin|>JSON<|tool_call_end|>; gemma4 uses
    # <|tool_call>call:name{key:value,...}<tool_call|> with strings wrapped in <|"|>, booleans lowercase,
    # None as null, others bare. Exceptions: ValueError for unsupported parsers.
    """Render a single tool call as the raw text the model would generate for
    the given tool-parser format."""
    args_json = json.dumps(arguments, ensure_ascii=False)
    if tool_parser in ("hermes", "qwen3_coder"):
        return f'<tool_call>\n{{"name": "{name}", "arguments": {args_json}}}\n</tool_call>'
    if tool_parser == "glm":
        arg_pairs = "".join(
            f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>" for key, value in arguments.items()
        )
        return f"<tool_call>{name}{arg_pairs}</tool_call>"
    if tool_parser == "kimi":
        return f"<|tool_call_begin|>{name}<|tool_call_argument_begin|>{args_json}<|tool_call_end|>"
    if tool_parser == "gemma4":
        # Gemma-4 format: string args wrapped in <|"|>...<|"|>, other scalars bare.
        arg_parts = []
        for key, value in arguments.items():
            if isinstance(value, str):
                arg_parts.append(f'{key}:<|"|>{value}<|"|>')
            elif isinstance(value, bool):
                arg_parts.append(f"{key}:{str(value).lower()}")
            elif value is None:
                arg_parts.append(f"{key}:null")
            else:
                arg_parts.append(f"{key}:{value}")
        return f"<|tool_call>call:{name}{{{','.join(arg_parts)}}}<tool_call|>"
    raise ValueError(f"Unsupported tool parser for mock tool-call rendering: {tool_parser!r}")


def _format_assistant_content(tool_parser: str, reasoning: str, calls: list[tuple[str, dict[str, Any]]]) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns just the reasoning if there are no calls. Otherwise renders each call; for kimi
    # concatenates them inside a tool_calls_section begin/end wrapper, for others joins them with
    # newlines; prefixes reasoning plus a newline when reasoning is non-empty.
    """Build the assistant turn's content (reasoning + raw tool-call text)."""
    if not calls:
        return reasoning
    call_texts = [_format_tool_call_text(tool_parser, name, arguments) for name, arguments in calls]
    if tool_parser == "kimi":
        # Kimi wraps all calls in a single section.
        body = "".join(call_texts)
        tool_block = f"<|tool_calls_section_begin|>{body}<|tool_calls_section_end|>"
    else:
        tool_block = "\n".join(call_texts)
    if reasoning:
        return f"{reasoning}\n{tool_block}"
    return tool_block


@dataclass
class VLToolResponseSpec:
    # [CLASS DOC]
    # A canned tool response (tool name, text, optional PIL image) used both as the trajectory's tool
    # message and as the runtime tool's return value. tool_name is informational only.
    """A single tool response used both to build the mock trajectory prefix and
    to drive the deterministic tool at runtime."""

    tool_name: str
    text: str
    image: PILImage | None = None

    def to_message(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Builds a tool-role message whose content is a list: an image block first if an image exists, then a
        # text block. No tool_call_id or name is included.
        content: list[dict[str, Any]] = []
        if self.image is not None:
            content.append({"type": "image", "image": self.image})
        content.append({"type": "text", "text": self.text})
        return {"role": "tool", "content": content}

    def to_tool_response(self) -> ToolResponse:
        # [FUNCTION DOC]
        # Returns a ToolResponse with the text, plus a one-image list when an image exists.
        if self.image is not None:
            return ToolResponse(text=self.text, image=[self.image])
        return ToolResponse(text=self.text)


@dataclass
class VLToolStep:
    # [CLASS DOC]
    # One VL assistant turn: reasoning text, list of (tool name, arguments) calls, and the responses
    # appended afterwards (see docstring).
    """One assistant generation.

    ``calls`` is the list of ``(tool_name, arguments)`` the assistant emits this
    turn; when non-empty, ``responses`` holds the tool answers that get appended
    before the next turn. ``reasoning`` is the assistant text before/around the
    tool call (or the final answer for the terminating turn).

    VL chat templates typically render assistant ``content`` verbatim but drop a
    structured ``tool_calls`` field, so the mock assistant "generation" for a
    tool-call turn is the parser's raw tool-call text embedded as content. This
    matches how a VL model actually emits tool calls at rollout time and lets the
    tool parser extract them.
    """

    reasoning: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    responses: list[VLToolResponseSpec] = field(default_factory=list)

    def assistant_message(self, tool_parser: str) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns an assistant message whose content is the reasoning plus raw tool-call text for the parser.
        content = _format_assistant_content(tool_parser, self.reasoning, self.calls)
        return {"role": "assistant", "content": content}

    @property
    def appended_messages(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Property returning the tool messages built from each response, in order.
        return [response.to_message() for response in self.responses]


@dataclass
class VLToolTrajectory:
    # [CLASS DOC]
    # Mutable VL tool-agent case: name, description, raw_prompt (may contain images), steps, FunctionTools
    # and max parallel calls (default 1).
    name: str
    description: str
    raw_prompt: list[dict[str, Any]]
    steps: list[VLToolStep]
    tools: list[FunctionTool]
    max_parallel_calls: int = 1

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        # [FUNCTION DOC]
        # Property returning each tool's schema dict (unset and None fields omitted). Note: a property here,
        # unlike the method on ToolAgentTrajectory.
        return [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in self.tools]

    @property
    def expected_generation_turns(self) -> int:
        # [FUNCTION DOC]
        # Number of steps.
        return len(self.steps)

    @property
    def expected_num_turns(self) -> int:
        # [FUNCTION DOC]
        # Steps + steps with responses + 1 for the prompt.
        tool_rounds = sum(1 for step in self.steps if step.responses)
        return self.expected_generation_turns + tool_rounds + 1


@dataclass
class VLSingleTurnTrajectory:
    # [CLASS DOC]
    # VL single-turn case: name, description, raw_prompt with an image, assistant response text, and
    # fixed expected turn counts (1 generation, 2 total).
    name: str
    description: str
    raw_prompt: list[dict[str, Any]]
    assistant_response: str

    expected_generation_turns: int = 1
    expected_num_turns: int = 2

    def assistant_message(self) -> dict[str, Any]:
        # [FUNCTION DOC]
        # Returns the assistant message with the response text as content.
        return {"role": "assistant", "content": self.assistant_response}


VLMockTrajectory = VLSingleTurnTrajectory | VLToolTrajectory


def _solid_image(size: tuple[int, int], color: str) -> PILImage:
    # [FUNCTION DOC]
    # HELPER: Imports Pillow lazily and returns a new RGB image of the given size filled with the color.
    from PIL import Image

    return Image.new("RGB", size, color=color)


def _make_query_tool(name: str, description: str, response: VLToolResponseSpec) -> FunctionTool:
    # [FUNCTION DOC]
    # HELPER: Builds a FunctionTool named name whose schema has one required string argument "query" and
    # whose function ignores the query and returns the given response's ToolResponse.
    """Build a deterministic function tool with a single ``query`` argument.

    The tool ignores the query and always returns ``response``; the query only
    exists so the model-rendered ``tool_calls`` carry an argument that survives
    the chat template round-trip and the tool-parser extraction.
    """

    schema = OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=name,
            description=description,
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "query": OpenAIFunctionPropertySchema(type="string", description="What to inspect."),
                },
                required=["query"],
            ),
        ),
    )

    def _fn(query: str = "") -> ToolResponse:
        # [FUNCTION DOC]
        # HELPER: Tool body: discards the query and returns the canned ToolResponse.
        del query
        return response.to_tool_response()

    return FunctionTool(name=name, fn=_fn, tool_schema=schema)


def build_vl_trajectories() -> dict[str, VLMockTrajectory]:
    # [FUNCTION DOC]
    # Creates fresh images (84x84 teal prompt image, 56x56 orange and purple tool images) and returns a
    # mapping from name to the three VL trajectories: a single-turn description, a single zoom_tool call
    # returning the orange image followed by an answer, and two parallel calls (left_tool, right_tool)
    # returning the two images followed by a summary (max_parallel_calls 2).
    # Exceptions: ImportError if Pillow is missing.
    """Construct the inline VL trajectories.

    Uses small solid-color images so token counts stay small and image
    processing stays fast on CPU. Sizes are multiples of 28 to satisfy the
    Qwen-family patch grid. Fresh images are created on each call.
    """

    prompt_image = _solid_image((84, 84), "teal")
    tool_image_a = _solid_image((56, 56), "orange")
    tool_image_b = _solid_image((56, 56), "purple")

    single_turn = VLSingleTurnTrajectory(
        name="vl_singleturnchat",
        description="One user request with an image, followed by one plain assistant response.",
        raw_prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prompt_image},
                    {"type": "text", "text": "Describe the image in one short sentence."},
                ],
            }
        ],
        assistant_response="The image is a solid teal square.",
    )

    single_tool_response = VLToolResponseSpec(
        tool_name="zoom_tool",
        text="Zoomed crop attached.",
        image=tool_image_a,
    )
    single_tool = VLToolTrajectory(
        name="vl_multiturnsingletool",
        description="Image prompt, one tool call returning an image, then a final answer.",
        raw_prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prompt_image},
                    {"type": "text", "text": "Use zoom_tool to inspect the image, then describe what you see."},
                ],
            }
        ],
        steps=[
            VLToolStep(
                reasoning="I'll zoom in to inspect the image.",
                calls=[("zoom_tool", {"query": "center crop"})],
                responses=[single_tool_response],
            ),
            VLToolStep(
                reasoning="The zoomed crop shows an orange region over the teal background.",
            ),
        ],
        tools=[
            _make_query_tool("zoom_tool", "Return a zoomed crop of the current image.", single_tool_response),
        ],
        max_parallel_calls=1,
    )

    left_response = VLToolResponseSpec(tool_name="left_tool", text="Left half crop.", image=tool_image_a)
    right_response = VLToolResponseSpec(tool_name="right_tool", text="Right half crop.", image=tool_image_b)
    multi_tool = VLToolTrajectory(
        name="vl_multiturnmultitool",
        description="Image prompt, two parallel tool calls each returning an image, then a summary.",
        raw_prompt=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prompt_image},
                    {"type": "text", "text": "Inspect the left and right halves with the tools, then summarize."},
                ],
            }
        ],
        steps=[
            VLToolStep(
                reasoning="Let me inspect both halves.",
                calls=[
                    ("left_tool", {"query": "left half"}),
                    ("right_tool", {"query": "right half"}),
                ],
                responses=[left_response, right_response],
            ),
            VLToolStep(
                reasoning="Both halves are crops of the same teal image, tinted orange and purple.",
            ),
        ],
        tools=[
            _make_query_tool("left_tool", "Return the left-half crop of the current image.", left_response),
            _make_query_tool("right_tool", "Return the right-half crop of the current image.", right_response),
        ],
        max_parallel_calls=2,
    )

    return {
        single_turn.name: single_turn,
        single_tool.name: single_tool,
        multi_tool.name: multi_tool,
    }


VL_SINGLE_TURN_TRAJECTORY_NAMES: tuple[str, ...] = ("vl_singleturnchat",)
VL_TOOL_AGENT_TRAJECTORY_NAMES: tuple[str, ...] = ("vl_multiturnsingletool", "vl_multiturnmultitool")
VL_TRAJECTORY_NAMES: tuple[str, ...] = VL_SINGLE_TURN_TRAJECTORY_NAMES + VL_TOOL_AGENT_TRAJECTORY_NAMES


def get_vl_trajectory(name: str) -> VLMockTrajectory:
    # [FUNCTION DOC]
    # Builds the VL trajectories and returns the named one. Exceptions: ValueError if unknown.
    trajectories = build_vl_trajectories()
    try:
        return trajectories[name]
    except KeyError as exc:
        raise ValueError(f"Unknown VL trajectory {name!r}. Choices: {sorted(trajectories)}") from exc


def select_vl_trajectories(names: list[str] | None = None) -> list[VLMockTrajectory]:
    # [FUNCTION DOC]
    # Builds the VL trajectories and returns all (when names empty/None) or the named ones in order.
    # Exceptions: ValueError listing any unknown names.
    trajectories = build_vl_trajectories()
    if not names:
        return list(trajectories.values())
    missing = [name for name in names if name not in trajectories]
    if missing:
        raise ValueError(f"Unknown VL trajectories {missing!r}. Choices: {sorted(trajectories)}")
    return [trajectories[name] for name in names]
