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
# Vision/video preprocessing helpers wrapping qwen_vl_utils (image and video fetching) plus a MiniCPM-o specific multimodal input reshaper. Defines VIDEO_FORMAT_HELP, a help message describing supported video dict formats.

from io import BytesIO
from typing import Optional

import torch
from PIL import Image


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    # [FUNCTION DOC]
    # Behavior: if given a PIL image, returns it converted to RGB. Otherwise treats image as a dict; if it has raw "bytes" it decodes them into a PIL image under the "image" key, then calls qwen_vl_utils fetch_image with image_patch_size, falling back to calling it without that argument on any exception (older library versions).
    # Inputs: image - PIL image or dict in qwen-vl format; image_patch_size - patch size used for resizing.
    # Outputs: a PIL image.
    # Exceptions: AssertionError if dict has both "bytes" and "image"; errors from fetch_image fallback propagate.
    # Side effects: mutates the input dict by adding "image" when "bytes" present.
    from qwen_vl_utils import fetch_image

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    try:
        ans = fetch_image(image, image_patch_size=image_patch_size)
    except Exception:
        ans = fetch_image(image)
    return ans


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor:
    # [FUNCTION DOC]
    # Behavior: validates the video dict, shallow-copies it, and if it has no own sampling rules ("nframes" or "fps") injects nframes, or fps with optional min_frames/max_frames; then delegates to qwen_vl_utils fetch_video.
    # Inputs: video - dict with "video" key; image_patch_size; nframes / fps - mutually exclusive sampling controls; fps_min_frames/fps_max_frames - bounds when using fps; return_video_sample_fps / return_video_metadata - forwarded flags.
    # Outputs: whatever fetch_video returns (frame tensor, optionally with sample fps and/or metadata).
    # Exceptions: NotImplementedError with VIDEO_FORMAT_HELP if not a dict with "video"; AssertionError if both nframes and fps are given.
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """
    from qwen_vl_utils import fetch_video

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(
        video,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )


def process_multi_modal_inputs_for_minicpmo(input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    # [FUNCTION DOC]
    # Behavior: rewrites MiniCPM-o multimodal inputs for packed (remove-padding) sequences: shifts each sample's image_bound by subtracting its left-padding length (index of first nonzero attention) and adding its cumulative sequence offset; flattens per-sample pixel_values lists into one list; vertically stacks image bounds and tgt_sizes; attaches input_ids, attention_mask and position_ids.
    # Inputs: input_ids, attention_mask (batch x seq), position_ids, cu_seqlens - cumulative sequence start offsets; multi_modal_inputs - dict with image_bound, pixel_values, tgt_sizes lists per sample.
    # Outputs: dict with single key "data" holding the modified multi_modal_inputs.
    # Side effects: mutates multi_modal_inputs in place.
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device) - left_padding_length[i] + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}
