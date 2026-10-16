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
# Custom reward functions used by reward-loop tests: an async generative-RM (GenRM) scorer for GSM8K that
# queries a reward model router over the OpenAI chat API, and a rule-based math_verify scorer.
# GRM_PROMPT_TEMPLATE asks the judge to output a single integer score 1-10 for a problem/solution pair.

import json
import os

import aiohttp
from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizer

GRM_PROMPT_TEMPLATE = """
You are given a problem and a proposed solution.

Problem:
{problem}

Solution:
{solution}

Please evaluate how well the solution addresses the problem. 
Give a score from 1 to 10, where:
- 1 means the solution is completely irrelevant or incorrect.
- 5 means the solution is partially correct but incomplete or not well reasoned.
- 10 means the solution is fully correct, well-reasoned, and directly solves the problem.

Only output the score as a single number (integer).
""".strip()


async def chat_complete(router_address: str, chat_complete_request: dict):
    # [FUNCTION DOC]
    # HELPER: Async POST of a chat completion request to http://ROUTER/v1/chat/completions using a fresh aiohttp
    # session with no timeout; parses the response text as JSON into a ChatCompletion.
    # Inputs: router_address - host:port; chat_complete_request - request body mapping.
    # Outputs: ChatCompletion.
    # Exceptions: re-raises any network/JSON/validation error.
    # Side effects: network I/O; the session is always closed.
    # KNOWN ISSUE: if creating the session itself raises, the finally clause references an unbound session variable
    # and raises UnboundLocalError, masking the original error. The try/except re-raise is a no-op.
    url = f"http://{router_address}/v1/chat/completions"
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
            output = json.loads(output)
            return ChatCompletion(**output)
    except Exception as e:
        raise e
    finally:
        await session.close()


async def compute_score_gsm8k(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    # [FUNCTION DOC]
    # Behavior: formats the GRM prompt with extra_info["question"] and the solution, sends it as one user message
    # (temperature 0.7, top_p 0.8, max_tokens 4096) to the router using model name from env GENRM_MODEL_NAME
    # (default ~/models/Qwen/Qwen2.5-1.5B-Instruct expanded), and parses the integer after the last blank-line
    # separator in the reply; any parse failure yields score 0.
    # Inputs: data_source, ground_truth, reward_model_tokenizer - unused; solution_str; extra_info - must hold
    # "question"; reward_router_address - host:port.
    # Outputs: dict with score (int), acc (score == 10), genrm_response (raw text).
    # Exceptions: KeyError if question missing; network errors propagate.
    # Side effects: network call.
    """Compute the reward score."""

    grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=extra_info["question"], solution=solution_str)
    messages = [{"role": "user", "content": grm_prompt}]
    sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
    model_name = os.environ.get("GENRM_MODEL_NAME", os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct"))
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **sampling_params,
    }
    result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=chat_complete_request,
    )
    grm_response = result.choices[0].message.content
    try:
        score = int(grm_response.split("\n\n")[-1].strip())
    except Exception:
        score = 0
    return {"score": score, "acc": score == 10, "genrm_response": grm_response}


def compute_score_math_verify(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    # [FUNCTION DOC]
    # Behavior: delegates to verl.utils.reward_score.math_verify.compute_score with the solution and ground truth.
    # Inputs: solution_str, ground_truth; data_source, extra_info, extra kwargs ignored.
    # Outputs: whatever compute_score returns (a float score).
    """Compute the reward score."""
    from verl.utils.reward_score.math_verify import compute_score

    return compute_score(
        model_output=solution_str,
        ground_truth=ground_truth,
    )
