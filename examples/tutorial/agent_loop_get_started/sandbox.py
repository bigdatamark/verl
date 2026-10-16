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
# Tutorial tool implementation: a code-interpreter tool that sends Python code to a Sandbox Fusion
# HTTP service and returns its stdout+stderr to the agent loop.
import re

import aiohttp
from transformers.utils import get_json_schema

from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse


class SandboxTool(BaseTool):
    # [CLASS DOC]
    # verl BaseTool subclass exposing a single "code_interpreter" function. The config must contain
    # "sandbox_fusion_url". The OpenAI tool schema is derived from the code_interpreter signature and
    # docstring rather than from the provided tool_schema.
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # [FUNCTION DOC]
        # Calls the BaseTool initializer and compiles a regex that captures the content of a fenced
        # code block opened with "```py" (non-greedy, dot matches newlines; also matches "```python",
        # leaving "thon" at the start of the capture).
        # KNOWN ISSUE: for "```python" fences the captured code starts with "thon", which will break execution.
        super().__init__(config, tool_schema)
        # Different model may use different code pattern, e.g. python, py, etc.
        self.code_pattern = re.compile(r"```py(.*?)```", re.DOTALL)

    async def code_interpreter(self, code: str) -> str:
        # [FUNCTION DOC]
        # Opens an aiohttp session and POSTs JSON {"code": code} to the configured sandbox URL; raises
        # for non-2xx status, parses the JSON body and returns run_result.stdout concatenated with
        # run_result.stderr. Inputs: code string. Outputs: combined output string.
        # Exceptions: aiohttp HTTP/connection errors, KeyError if the response lacks run_result fields.
        # Side effects: network request.
        """Execute the code in the sandbox.

        Args:
            code: The code to be executed.

        Returns:
            str: The output of the code execution.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.get("sandbox_fusion_url"),
                json={"code": code},
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                stdout, stderr = result["run_result"]["stdout"], result["run_result"]["stderr"]
                return stdout + stderr

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # [FUNCTION DOC]
        # Generates a JSON schema from the code_interpreter method via transformers' get_json_schema and
        # wraps it in OpenAIFunctionToolSchema. Outputs: the schema object.
        schema = get_json_schema(self.code_interpreter)
        return OpenAIFunctionToolSchema(**schema)

    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[str, float, dict]:
        # [FUNCTION DOC]
        # Agent-loop entry point. Takes parameters["code"]; if it contains a ```py fenced block, uses
        # the first block's stripped content. Then finds the last non-empty line and, if it does not start
        # with "print", wraps it as a print call of that line so expression results are shown. Runs the
        # code via code_interpreter.
        # Inputs: instance_id (unused); parameters dict with "code"; kwargs ignored.
        # Outputs: tuple of (ToolResponse with the output text, reward 0.0, empty metrics dict).
        # Exceptions: KeyError if "code" missing; network errors propagate.
        # KNOWN ISSUE: return annotation says the first element is str but a ToolResponse is returned.
        # KNOWN ISSUE: wrapping the last line in print breaks when it is a statement (assignment, indented
        # block line, import, etc.); lines containing only whitespace are not skipped.
        code = parameters["code"]
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: Some script may not explicitly print result, we need to add a print statement to the end of the script.
        # More better way is to SFT the model to make it print result by default, we skip SFT stage in this tutorial.
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        result = await self.code_interpreter(code)
        return ToolResponse(text=result), 0.0, {}
