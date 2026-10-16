# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Tests that MegatronEngineWithValueHead.forward_step requests a contiguous context-parallel layout and value-model mode
# from the mcore forward function for DeepSeek-V4 hybrid attention models.

from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithValueHead


def test_dsv4_value_head_forwards_contiguous_cp_layout():
    # [FUNCTION DOC]
    # Builds a value-head engine without __init__ (remove-padding on, no pad_to_length, input prep passing input_ids),
    # a model whose config's attention variant is "dsv4_hybrid", patches device id and the mcore forward-fn factory with a
    # kwargs-capturing fake, runs forward_step on a one-batch iterator, and asserts value_model True and cp_layout "contiguous".
    engine = object.__new__(MegatronEngineWithValueHead)
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )
    engine.engine_config = SimpleNamespace(use_remove_padding=True, pad_to_length=False)
    engine.prepare_model_inputs = lambda batch: {
        "input_ids": batch["input_ids"],
        "multi_modal_inputs": None,
    }

    batch = TensorDict({"input_ids": torch.tensor([[1, 2, 3]])}, batch_size=[1])
    model = SimpleNamespace(config=SimpleNamespace(experimental_attention_variant="dsv4_hybrid"))
    captured = {}

    def forward_fn(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Fake forward; records keyword arguments and returns a zero tensor.
        captured.update(kwargs)
        return torch.zeros(1)

    with (
        patch("verl.workers.engine.megatron.transformer_impl.get_device_id", return_value="cpu"),
        patch("verl.models.mcore.get_mcore_engine_forward_fn", return_value=forward_fn),
    ):
        engine.forward_step(iter([batch]), model, None, lambda: None)

    assert captured["value_model"] is True
    assert captured["cp_layout"] == "contiguous"
