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
# CPU unit test for AgentLoopWorker._compute_position_ids when the multimodal processor exposes a
# get_rope_index_kwargs hook (e.g. audio models needing extra rope arguments such as audio_seqlens).

import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


def test_compute_position_ids_accepts_processor_rope_kwargs_hook():
    # [FUNCTION DOC]
    # Behavior: builds a fake processor whose get_rope_index_kwargs derives audio_seqlens by summing the
    # feature_attention_mask, and whose get_rope_index asserts it received audio_seqlens equal to [2] and returns
    # all-zero 3-axis rope position ids. Calls the unbound _compute_position_ids with a minimal stand-in worker
    # object (only a processor attribute) on a 1x3 input.
    # Inputs: none (pytest test).
    # Outputs: asserts result shape is (batch=1, 4, seq=3), i.e. a text position row prepended to the 3 rope rows,
    # and that channel 0 equals the plain sequential positions [0, 1, 2].
    # Exceptions: AssertionError on mismatch.
    # Side effects: none.
    class Processor:
        # [CLASS DOC]
        # HELPER: Minimal fake multimodal processor exposing the rope-index hook API.
        def get_rope_index_kwargs(self, multi_modal_inputs):
            # [FUNCTION DOC]
            # HELPER: Returns a mapping with key audio_seqlens = row-wise sum of multi_modal_inputs feature_attention_mask.
            return {"audio_seqlens": multi_modal_inputs["feature_attention_mask"].sum(-1)}

        def get_rope_index(
            self,
            *,
            input_ids,
            attention_mask,
            image_grid_thw,
            video_grid_thw,
            audio_seqlens,
        ):
            # [FUNCTION DOC]
            # HELPER: Keyword-only fake rope index. Ignores attention_mask and grid args, asserts audio_seqlens equals
            # tensor [2], and returns a tuple (zeros long tensor of shape (3, 1, seq_len), None).
            # Exceptions: AssertionError if audio_seqlens was not forwarded correctly.
            del attention_mask, image_grid_thw, video_grid_thw
            torch.testing.assert_close(audio_seqlens, torch.tensor([2]))
            return torch.zeros((3, 1, input_ids.shape[-1]), dtype=torch.long), None

    worker = type("Worker", (), {"processor": Processor()})()
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {"feature_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long)}

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids,
        attention_mask,
        multi_modal_inputs,
    )

    assert position_ids.shape == (1, 4, 3)
    torch.testing.assert_close(position_ids[:, 0], torch.tensor([[0, 1, 2]]))
