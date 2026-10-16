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
# Tests for RLHFDataset (verl.utils.dataset.rl_dataset): loading gsm8k / geo3k / video data into batches via
# collate_fn and DataProto, max_samples truncation, multimodal message building from image/video paths,
# overlong-prompt filtering with images, and non-blocking async multimodal processing.
# Several tests require local models under ~/models and data under ~/data (not truly CPU-only/self-contained).
import asyncio
import json
import os
import threading
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from verl import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _mock_rlhf_dataset():
    # [FUNCTION DOC]
    # HELPER: Creates an RLHFDataset instance without running its constructor (bypasses __init__) and sets only
    # the attributes needed by _build_messages: prompt_key "prompt", image_key "images", video_key "videos",
    # audio_key "audios", and a placeholder non-None processor (a bare object) so multimodal paths are taken.
    # Inputs: none. Outputs: the partially-initialized dataset. Side effects: none.
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.prompt_key = "prompt"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.audio_key = "audios"
    dataset.processor = object()
    return dataset


def get_gsm8k_data():
    # [FUNCTION DOC]
    # HELPER: Returns the expected path ~/data/gsm8k/train.parquet (user-expanded), creating the folder if missing.
    # Does not download the data; the file must already exist for callers to succeed.
    # Inputs: none. Outputs: path string. Side effects: creates directory ~/data/gsm8k/ if absent.
    # prepare test dataset
    local_folder = os.path.expanduser("~/data/gsm8k/")
    local_path = os.path.join(local_folder, "train.parquet")
    os.makedirs(local_folder, exist_ok=True)
    return local_path


def test_rl_dataset():
    # [FUNCTION DOC]
    # Behavior: loads the deepseek-coder-1.3b-instruct tokenizer from ~/models, builds an RLHFDataset over gsm8k
    # train parquet with prompt_key "prompt", max_prompt_length 256 and overlong filtering (2 workers); takes one
    # shuffled batch of 16 via DataLoader with collate_fn; splits the batch into tensor and non-tensor dicts,
    # builds a DataProto and asserts its length is 16 and "raw_prompt" is in non_tensor_batch.
    # Inputs: none. Outputs: none.
    # Exceptions: fails if model/data are missing. Side effects: reads local files, creates ~/data/gsm8k dir.
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/deepseek-coder-1.3b-instruct"))
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config)

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert len(data_proto) == 16
    assert "raw_prompt" in data_proto.non_tensor_batch


def test_rl_dataset_with_max_samples():
    # [FUNCTION DOC]
    # Behavior: same dataset setup as test_rl_dataset but with max_samples 5 (both in config and as a constructor
    # argument); asserts the dataset length is exactly 5.
    # Inputs: none. Outputs: none. Side effects: reads local model/data files.
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/deepseek-ai/deepseek-coder-1.3b-instruct"))
    local_path = get_gsm8k_data()
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 256,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 2,
            "max_samples": 5,
        }
    )
    dataset = RLHFDataset(data_files=local_path, tokenizer=tokenizer, config=config, max_samples=5)
    assert len(dataset) == 5


def test_build_messages_accepts_image_path():
    # [FUNCTION DOC]
    # Behavior: with a mock dataset, builds messages for a prompt "Describe <image>" with one image path string
    # (file URI); asserts the user content is split into a text part "Describe " followed by an image part whose
    # "image" value is the original path string (paths are passed through, not loaded).
    # Inputs: none. Outputs: none. Side effects: none.
    dataset = _mock_rlhf_dataset()
    image_path = "file:///tmp/image.jpg"
    example = {
        "prompt": [{"role": "user", "content": "Describe <image>"}],
        "images": [image_path],
    }

    messages = dataset._build_messages(example, key=dataset.prompt_key)

    assert messages[0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "image", "image": image_path},
    ]


@pytest.mark.parametrize(
    ("videos", "expected_video"),
    [
        (["file:///tmp/video.mp4"], "file:///tmp/video.mp4"),
        ([["file:///tmp/frame1.jpg", "file:///tmp/frame2.jpg"]], ["file:///tmp/frame1.jpg", "file:///tmp/frame2.jpg"]),
        ([[Path("/tmp/frame1.jpg"), Path("/tmp/frame2.jpg")]], ["/tmp/frame1.jpg", "/tmp/frame2.jpg"]),
    ],
)
def test_build_messages_accepts_video_path_or_frame_list(videos, expected_video):
    # [FUNCTION DOC]
    # Behavior: parametrized; builds messages for "Describe <video>" with the given videos field and asserts the
    # content becomes a text part "Describe " followed by a video part whose "video" equals expected_video.
    # Cases: a single video path string passes through; a list of frame path strings passes through as a list;
    # a list of pathlib Path frames is converted to a list of plain strings.
    # Inputs: videos - the "videos" field value; expected_video - expected video entry value.
    # Outputs: none. Side effects: none.
    dataset = _mock_rlhf_dataset()
    example = {
        "prompt": [{"role": "user", "content": "Describe <video>"}],
        "videos": videos,
    }

    messages = dataset._build_messages(example, key=dataset.prompt_key)

    assert messages[0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "video", "video": expected_video},
    ]


def test_maybe_filter_out_long_prompts_accepts_image_path(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: verifies maybe_filter_out_long_prompts works when images are given as paths. Defines a fake
    # dataframe (a list with a filter method keeping docs where the predicate is true), a fake processor whose chat
    # template returns a fixed string and whose call asserts images equal ["processed-image"] and returns 3 token ids;
    # monkeypatches RLHFDataset._process_multi_modal_info with a classmethod that records a deep copy of the
    # messages and returns ["processed-image"] with no videos/audios. Configures the mock dataset (no tokenizer,
    # filtering on, max_prompt_length 10, no workers etc.), filters a one-row dataframe, and asserts the row is kept
    # (3 tokens <= 10) and the captured messages had content split into a text part and an image-path part.
    # Inputs: monkeypatch - pytest fixture. Outputs: none. Side effects: monkeypatches RLHFDataset for the test.
    class FakeDataFrame(list):
        # [CLASS DOC]
        # HELPER: Minimal list subclass emulating a HuggingFace datasets object's filter method.
        def filter(self, fn, num_proc=None, desc=None):
            # [FUNCTION DOC]
            # HELPER: Returns a new FakeDataFrame of the elements for which fn returns truthy; num_proc and desc ignored.
            return FakeDataFrame([doc for doc in self if fn(doc)])

    class FakeProcessor:
        # [CLASS DOC]
        # HELPER: Fake multimodal processor used by the filtering test.
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Ignores all arguments and returns the fixed string "formatted prompt".
            return "formatted prompt"

        def __call__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Asserts kwargs "images" equals ["processed-image"] and returns a mapping with input_ids [[1, 2, 3]].
            # Exceptions: AssertionError if images differ.
            assert kwargs["images"] == ["processed-image"]
            return {"input_ids": [[1, 2, 3]]}

    captured = {}

    def fake_process_multi_modal_info(cls, messages, image_patch_size, config):
        # [FUNCTION DOC]
        # HELPER: Replacement for _process_multi_modal_info; stores a deep copy of messages in captured["messages"]
        # and returns (["processed-image"], None, None) for images, videos, audios. Side effects: mutates captured.
        captured["messages"] = deepcopy(messages)
        return ["processed-image"], None, None

    monkeypatch.setattr(
        RLHFDataset,
        "_process_multi_modal_info",
        classmethod(fake_process_multi_modal_info),
    )

    dataset = _mock_rlhf_dataset()
    dataset.processor = FakeProcessor()
    dataset.tokenizer = None
    dataset.filter_overlong_prompts = True
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.image_patch_size = 14
    dataset.config = OmegaConf.create({})
    dataset.max_prompt_length = 10
    dataset.mm_processor_kwargs = {}
    dataset.num_workers = None

    image_path = "file:///tmp/image.jpg"
    dataframe = FakeDataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "Describe <image>"}],
                "images": [image_path],
            }
        ]
    )

    filtered = dataset.maybe_filter_out_long_prompts(dataframe)

    assert len(filtered) == 1
    assert captured["messages"][0]["content"] == [
        {"type": "text", "text": "Describe "},
        {"type": "image", "image": image_path},
    ]


def test_process_multi_modal_info_keeps_event_loop_responsive(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: checks that the async RLHFDataset.process_multi_modal_info offloads the (blocking) synchronous
    # _process_multi_modal_info so the event loop keeps running. Monkeypatches the sync method to set a
    # "processing_started" event then block up to 0.2 s waiting for an "observer_ran" event. Inside an asyncio
    # run, schedules an observer task that yields once then sets observer_ran, awaits the async processing call,
    # records whether observer_ran was set by then, awaits the observer, and returns that flag. Asserts the flag is
    # true (event loop progressed during processing) and that processing actually started.
    # Inputs: monkeypatch fixture. Outputs: none. Side effects: patches RLHFDataset; uses threads/events.
    processing_started = threading.Event()
    observer_ran = threading.Event()

    def blocking_process_multi_modal_info(cls, messages, image_patch_size, config):
        # [FUNCTION DOC]
        # HELPER: Blocking stand-in; sets processing_started, waits up to 0.2 s for observer_ran, returns (None, None, None).
        processing_started.set()
        observer_ran.wait(timeout=0.2)
        return None, None, None

    monkeypatch.setattr(
        RLHFDataset,
        "_process_multi_modal_info",
        classmethod(blocking_process_multi_modal_info),
    )

    async def run():
        # [FUNCTION DOC]
        # HELPER: Coroutine driving the test; returns True if the observer task ran while processing was in progress.
        async def observe_event_loop():
            # [FUNCTION DOC]
            # HELPER: Yields control to the event loop once, then sets observer_ran.
            await asyncio.sleep(0)
            observer_ran.set()

        observer = asyncio.create_task(observe_event_loop())
        await RLHFDataset.process_multi_modal_info([], image_patch_size=14, config=OmegaConf.create({}))
        observer_ran_while_processing = observer_ran.is_set()
        await observer
        return observer_ran_while_processing

    assert asyncio.run(run())
    assert processing_started.is_set()


def test_image_rl_data():
    # [FUNCTION DOC]
    # Behavior: loads the Qwen2-VL-2B-Instruct tokenizer and processor from ~/models, builds an RLHFDataset on
    # ~/data/geo3k/train.parquet (max_prompt_length 1024, filtering on with no workers to avoid a CI hang), takes
    # one batch of 16, builds a DataProto and asserts length 16, no "images" key in non_tensor_batch, and that every
    # raw_prompt is a single user message with two content parts: an image part holding a PIL Image and then a text
    # part holding a string. Prints the first raw_prompt.
    # Inputs: none. Outputs: none. Side effects: reads local model/data files; prints to stdout.
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,  # num_workers=1 hang in ci
        }
    )
    dataset = RLHFDataset(
        data_files=os.path.expanduser("~/data/geo3k/train.parquet"),
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)

    a = next(iter(dataloader))

    tensors = {}
    non_tensors = {}

    for key, val in a.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert len(data_proto) == 16
    assert "images" not in data_proto.non_tensor_batch

    for prompt in data_proto.non_tensor_batch["raw_prompt"]:
        assert len(prompt) == 1
        prompt = prompt[0]
        role, content = prompt["role"], prompt["content"]
        assert role == "user"
        assert len(content) == 2
        assert content[0]["type"] == "image" and isinstance(content[0]["image"], Image.Image)
        assert content[1]["type"] == "text" and isinstance(content[1]["text"], str)

    print("raw_prompt", data_proto.non_tensor_batch["raw_prompt"][0])


@pytest.fixture
def video_data_file():
    # [FUNCTION DOC]
    # HELPER: pytest fixture writing 30 copies of one multiple-choice video QA record (prompt content with a video
    # path from LLaVA-Video-178K and a question text) as indented JSON to test_data/test_video.json.
    # Inputs: none. Outputs: the relative file path string.
    # Side effects: creates directory test_data and writes the JSON file in the current working directory (not cleaned up).
    data = [
        {
            "problem_id": 17,
            "problem": "How does the crowd's excitement change as the match progresses?",
            "data_type": "video",
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "LLaVA-Video-178K/academic_source/activitynet/v_2g9GrshWQrU.mp4"},
                        {
                            "type": "text",
                            "text": "How does the crowd's excitement change as the match progresses? "
                            "A. It fluctuates; B. It decreases; C. It builds up; D. It remains the same. "
                            "Put your answer in <answer></answer>",
                        },
                    ],
                }
            ],
            "problem_type": "multiple choice",
            "solution": "C",
            "data_source": "LLaVA-Video-178K/2_3_m_academic_v0_1",
        }
    ] * 30

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_video.json"
    with open(test_file, "w") as f:
        json.dump(data, f, indent=2)

    return test_file


def test_video_rl_data(video_data_file):
    # [FUNCTION DOC]
    # Behavior: loads the Qwen2-VL-2B-Instruct tokenizer/processor, builds an RLHFDataset over the video JSON file
    # with filtering disabled, takes one batch of 16, builds a DataProto and asserts length 16 and that "images" is
    # not in non_tensor_batch; prints the first raw_prompt.
    # Inputs: video_data_file - path from the fixture. Outputs: none.
    # Exceptions: fails if model or referenced video files are unavailable. Side effects: file reads, stdout print.
    tokenizer = hf_tokenizer(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    processor = hf_processor(os.path.expanduser("~/models/Qwen/Qwen2-VL-2B-Instruct"))
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
        }
    )
    dataset = RLHFDataset(
        data_files=video_data_file,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
    )

    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, drop_last=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    tensors = {}
    non_tensors = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        else:
            non_tensors[key] = val

    data_proto = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
    assert len(data_proto) == 16
    assert "images" not in data_proto.non_tensor_batch

    print("raw_prompt", data_proto.non_tensor_batch["raw_prompt"][0])
