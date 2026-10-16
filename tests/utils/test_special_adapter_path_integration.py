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
# GPU integration tests (skipped when sglang is not importable) exercising a single
# module-scoped sglang Engine with LoRA enabled for Qwen2.5-0.5B-Instruct. Test classes
# run in file order and share engine state: base weight sync, adapter load from
# tensors, unload/reload, then memory release/resume cycles. Generation checks only
# assert non-empty greedy output text. Downloads the model from the HF hub.

"""Integration test for sglang LoRA adapter path.

Tests the two-phase weight sync (base weights then adapter deltas) that
engine_workers.update_weights() performs when lora.merge=False.

Requires 1 GPU with sglang installed.
"""

from dataclasses import asdict
from importlib.util import find_spec

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = pytest.mark.skipif(find_spec("sglang") is None, reason="sglang not installed")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_NAME = "verl_lora_adapter"


@pytest.fixture(scope="module")
def tokenizer():
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture returning the HF tokenizer for MODEL_ID.
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="module")
def lora_config():
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture returning a peft LoraConfig: rank 8, alpha 16,
    # targets q_proj and v_proj, no dropout, no bias, causal LM task.
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )


@pytest.fixture(scope="module")
def peft_model(lora_config):
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture loading MODEL_ID on CPU in bfloat16 and wrapping it
    # with peft using the lora_config fixture. Returns the peft model.
    """Create a peft-wrapped model (CPU, for extracting params)."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    return get_peft_model(model, lora_config)


@pytest.fixture(scope="module")
def engine():
    # [FUNCTION DOC]
    # HELPER: Module-scoped fixture launching an in-process sglang Engine (bf16, 50%
    # static memory, TP 1, LoRA enabled with up to 4 adapters per batch, max rank 16,
    # targets q_proj/v_proj). Yields it and shuts it down at module teardown.
    # Side effects: allocates GPU memory and spawns sglang worker processes.
    """Launch sglang engine with LoRA support."""
    import sglang as sgl

    eng = sgl.Engine(
        model_path=MODEL_ID,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        tp_size=1,
        enable_lora=True,
        max_loras_per_batch=4,
        max_lora_rank=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    yield eng
    eng.shutdown()


def _make_prompt(tokenizer, text="What is 2+2?"):
    # [FUNCTION DOC]
    # HELPER: Builds a chat-formatted prompt string (not tokenized) for one user
    # message with the generation prompt appended.
    # Inputs: tokenizer; text - user message, default "What is 2+2?". Outputs: string.
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )


def _extract_base_params(peft_model):
    # [FUNCTION DOC]
    # HELPER: Collects non-LoRA parameters of the peft model as a list of
    # (name, detached clone) pairs, with peft prefixes "base_model.model." and
    # ".base_layer" stripped to recover plain HF names.
    # Inputs: peft_model. Outputs: list of (str, tensor).
    """Extract base model params with clean HF key names.

    Mimics the base_sync_done=False path in engine_workers — sends full
    base model weights (no LoRA deltas) to the rollout engine.
    """
    params = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            continue
        clean = name.replace("base_model.model.", "").replace(".base_layer", "")
        params.append((clean, param.detach().clone()))
    return params


def _extract_lora_tensors(peft_model):
    # [FUNCTION DOC]
    # HELPER: Returns the peft adapter state dict as a list of (key, detached clone)
    # pairs, as sent when loading an adapter from tensors.
    # Inputs: peft_model. Outputs: list of (str, tensor).
    """Extract LoRA adapter tensors as a list of (name, tensor) tuples.

    Mimics the base_sync_done=True path — sends adapter deltas via
    LoadLoRAAdapterFromTensorsReqInput.
    """
    from peft import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(peft_model)
    return [(k, v.detach().clone()) for k, v in state_dict.items()]


class TestBaseWeightSync:
    # [CLASS DOC]
    # Phase 1 tests: pushing full base weights into the engine.
    """Phase 1: sync base weights (base_sync_done=False path)."""

    def test_update_weights_succeeds(self, engine, peft_model):
        # [FUNCTION DOC]
        # Behavior: sends the extracted base params via update_weights_from_tensor; passes
        # if no exception is raised.
        base_params = _extract_base_params(peft_model)
        engine.update_weights_from_tensor(named_tensors=base_params)

    def test_generation_after_base_sync(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: greedy-generates up to 32 tokens and asserts non-empty text.
        # Side effects: prints generated text.
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 32, "temperature": 0.0})
        text = output["text"]
        print(f"[base sync] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after base weight sync"


class TestAdapterLoading:
    # [CLASS DOC]
    # Phase 2 tests: loading the LoRA adapter from in-memory tensors.
    """Phase 2: load adapter deltas (base_sync_done=True path)."""

    def test_load_adapter_from_tensors(self, engine, peft_model, lora_config):
        # [FUNCTION DOC]
        # Behavior: loads the adapter under ADAPTER_NAME with the peft adapter tensors and
        # the LoraConfig converted to a dict with None-valued fields dropped; passes if no
        # exception.
        lora_tensors = _extract_lora_tensors(peft_model)
        config_dict = {k: v for k, v in asdict(lora_config).items() if v is not None}
        engine.load_lora_adapter_from_tensors(
            lora_name=ADAPTER_NAME,
            tensors=lora_tensors,
            config_dict=config_dict,
        )

    @pytest.mark.xfail(
        reason="sglang load_lora_adapter_from_tensors doesn't populate lora_ref_cache, "
        "so _resolve_lora_path validation fails. Adapter IS loaded in TP workers. "
        "This works in verl's actual flow because verl uses the HTTP server adapter "
        "which bypasses this validation.",
        raises=Exception,
    )
    def test_generation_with_adapter(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: generates with lora_path set to ADAPTER_NAME and asserts non-empty
        # text. Marked expected-failure because sglang's tensor-based adapter loading does
        # not register the adapter in its reference cache, so path validation fails.
        prompt = _make_prompt(tokenizer)
        output = engine.generate(
            prompt,
            {"max_new_tokens": 32, "temperature": 0.0},
            lora_path=ADAPTER_NAME,
        )
        text = output["text"]
        print(f"[adapter gen] Generated: {text!r}")
        assert len(text) > 0, "Empty generation with adapter"


class TestAdapterLifecycle:
    # [CLASS DOC]
    # Adapter unload/reload cycle, simulating subsequent training iterations.
    """Test unload + reload cycle (simulates subsequent training iterations)."""

    def test_unload_adapter(self, engine):
        # [FUNCTION DOC]
        # Behavior: unloads ADAPTER_NAME; passes if no exception.
        engine.unload_lora_adapter(ADAPTER_NAME)

    def test_generation_without_adapter(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: after unload, greedy generation without adapter returns non-empty text.
        """After unload, base model should still generate."""
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 32, "temperature": 0.0})
        text = output["text"]
        print(f"[after unload] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after adapter unload"

    def test_reload_adapter(self, engine, peft_model, lora_config):
        # [FUNCTION DOC]
        # Behavior: reloads the adapter from tensors exactly as in the load test.
        lora_tensors = _extract_lora_tensors(peft_model)
        config_dict = {k: v for k, v in asdict(lora_config).items() if v is not None}
        engine.load_lora_adapter_from_tensors(
            lora_name=ADAPTER_NAME,
            tensors=lora_tensors,
            config_dict=config_dict,
        )

    @pytest.mark.xfail(
        reason="sglang load_lora_adapter_from_tensors doesn't populate lora_ref_cache",
        raises=Exception,
    )
    def test_generation_after_reload(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: generation with the reloaded adapter; expected failure for the same
        # reference-cache reason as the first adapter generation test.
        prompt = _make_prompt(tokenizer)
        output = engine.generate(
            prompt,
            {"max_new_tokens": 32, "temperature": 0.0},
            lora_path=ADAPTER_NAME,
        )
        text = output["text"]
        print(f"[after reload] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after adapter reload"


class TestSleepWakeCycle:
    # [CLASS DOC]
    # Release/resume of engine GPU memory by tag.
    """Test release/resume with adapter-aware tags."""

    def test_release_kv_only_keeps_weights(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: releases and resumes only the kv_cache tag, then asserts a 16-token
        # greedy generation is non-empty (weights retained).
        """Adapter mode: release only kv_cache, keep base weights."""
        engine.release_memory_occupation(tags=["kv_cache"])
        engine.resume_memory_occupation(tags=["kv_cache"])

        # Should still generate after kv-only release/resume
        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 16, "temperature": 0.0})
        text = output["text"]
        print(f"[kv-only cycle] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after kv-only release/resume"

    def test_full_release_and_resume(self, engine, tokenizer):
        # [FUNCTION DOC]
        # Behavior: releases and resumes both kv_cache and weights, then asserts a 16-token
        # generation is non-empty.
        # KNOWN ISSUE: weights are not re-synced after resuming the weights tag, so the
        # model may hold undefined values; the test only checks that some text is produced,
        # not that it is meaningful.
        """Merge/no-LoRA mode: release everything, resume everything."""
        engine.release_memory_occupation(tags=["kv_cache", "weights"])
        engine.resume_memory_occupation(tags=["kv_cache", "weights"])

        prompt = _make_prompt(tokenizer)
        output = engine.generate(prompt, {"max_new_tokens": 16, "temperature": 0.0})
        text = output["text"]
        print(f"[full cycle] Generated: {text!r}")
        assert len(text) > 0, "Empty generation after full release/resume"
