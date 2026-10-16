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
# CPU tests for audio support in multimodal input handling: audio placeholder expansion in RLHFDataset message
# building, audio kwargs in build_multimodal_processor_inputs, and padding-merge of variable-length audio features
# in extract_multi_modal_inputs.

import types

import pytest
import torch

from verl.utils.model import extract_multi_modal_inputs
from verl.utils.tokenizer import build_multimodal_processor_inputs


def test_build_messages_replaces_audio_placeholder() -> None:
    # [FUNCTION DOC]
    # Behavior: skips if the datasets package is missing; builds an RLHFDataset without its constructor, setting
    # prompt/image/video/audio keys and a placeholder processor, and asserts a prompt "Listen to this: <audio> and
    # answer." with one audio path becomes text, audio (path) and text parts in order. Inputs: none. Outputs: none.
    pytest.importorskip("datasets")
    from verl.utils.dataset.rl_dataset import RLHFDataset

    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = object()

    example = {
        "prompt": [
            {"role": "user", "content": "Listen to this: <audio> and answer."},
        ],
        "audios": ["/tmp/example.wav"],
    }

    messages = dataset._build_messages(example, key=dataset.prompt_key)
    content = messages[0]["content"]
    assert content == [
        {"type": "text", "text": "Listen to this: "},
        {"type": "audio", "audio": "/tmp/example.wav"},
        {"type": "text", "text": " and answer."},
    ]


def test_build_multimodal_processor_inputs_includes_audio_sampling_rate() -> None:
    # [FUNCTION DOC]
    # Behavior: uses a fake processor exposing feature_extractor.sampling_rate 16000 that records call kwargs and
    # returns fixed input_ids; asserts build_multimodal_processor_inputs returns the processor output and passes
    # audio, sampling_rate 16000 (from the feature extractor) and the extra mm kwarg use_audio_in_video True.
    # Inputs: none. Outputs: none.
    captured = {}

    class AudioProcessor:
        # [CLASS DOC]
        # HELPER: Fake audio-capable processor.
        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets feature_extractor to a namespace with sampling_rate 16000.
            self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

        def __call__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs into captured and returns input_ids tensor [[1, 2, 3]].
            captured.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    processor = AudioProcessor()
    output = build_multimodal_processor_inputs(
        processor,
        text=["hello"],
        images=None,
        videos=None,
        audio=["waveform"],
        mm_processor_kwargs={"use_audio_in_video": True},
    )

    assert torch.equal(output["input_ids"], torch.tensor([[1, 2, 3]]))
    assert captured["audio"] == ["waveform"]
    assert captured["sampling_rate"] == 16000
    assert captured["use_audio_in_video"] is True


def test_build_multimodal_processor_inputs_skips_video_kwargs_when_no_videos() -> None:
    # [FUNCTION DOC]
    # Behavior: with a text-only fake processor and no media, asserts no video-specific kwargs (do_sample_frames,
    # video_metadata) are passed. Inputs: none. Outputs: none.
    captured = {}

    class TextOnlyProcessor:
        # [CLASS DOC]
        # HELPER: Fake processor with no feature extractor.
        def __call__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs into captured and returns input_ids [[1, 2, 3]].
            captured.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]])}

    build_multimodal_processor_inputs(
        TextOnlyProcessor(),
        text=["hello"],
        images=None,
        videos=None,
        audio=None,
    )
    assert "do_sample_frames" not in captured
    assert "video_metadata" not in captured


def test_extract_multi_modal_inputs_merges_variable_audio_fields() -> None:
    # [FUNCTION DOC]
    # Behavior: merges two samples whose input_features have last dims 3 and 5 and attention masks of length 3 and 5;
    # asserts features are right-padded with zeros to (2, 2, 5) and concatenated, masks are zero-padded to length 5
    # and stacked, and the 1-D video_second_per_grid values are concatenated. Inputs: none. Outputs: none.
    first_features = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
    second_features = torch.arange(10, dtype=torch.float32).view(1, 2, 5)

    merged = extract_multi_modal_inputs(
        [
            {
                "input_features": first_features,
                "feature_attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                "video_second_per_grid": torch.tensor([0.5]),
            },
            {
                "input_features": second_features,
                "feature_attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
                "video_second_per_grid": torch.tensor([1.0]),
            },
        ]
    )

    assert merged["input_features"].shape == (2, 2, 5)
    assert torch.equal(merged["input_features"][0, :, :3], first_features.squeeze(0))
    assert torch.equal(merged["input_features"][0, :, 3:], torch.zeros(2, 2))
    assert torch.equal(merged["input_features"][1], second_features.squeeze(0))
    assert torch.equal(merged["feature_attention_mask"], torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]))
    assert torch.equal(merged["video_second_per_grid"], torch.tensor([0.5, 1.0]))


def test_extract_multi_modal_inputs_rejects_unknown_ragged_fields() -> None:
    # [FUNCTION DOC]
    # Behavior: asserts merging two samples with an unrecognized field of mismatched shapes raises RuntimeError
    # mentioning the field name. Inputs: none. Outputs: none.
    with pytest.raises(RuntimeError, match="extra_audio_field"):
        extract_multi_modal_inputs(
            [
                {"extra_audio_field": torch.ones(1, 2, 3)},
                {"extra_audio_field": torch.ones(1, 2, 5)},
            ]
        )
