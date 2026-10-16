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
# CPU tests for megatron_utils.get_hf_config_attr and get_hf_rope_theta, which look up attributes on HF configs
# that may be nested (text_config, sub_configs, get_text_config accessor, transformers v5 rope_parameters),
# avoid cycles, prefer text configs, never fall back to unrelated multimodal siblings, and preserve falsy values.

from types import SimpleNamespace

import pytest

from verl.utils.megatron_utils import get_hf_config_attr, get_hf_rope_theta


def test_get_hf_rope_theta_from_nested_omni_text_config():
    # [FUNCTION DOC]
    # An Omni-style config whose text config (reached through get_text_config) has rope_theta 1e6 and a sibling
    # code2wav config with 1e4: asserts the text value 1e6 is returned.
    text_config = SimpleNamespace(rope_theta=1_000_000.0)

    class OmniConfig:
        # [CLASS DOC]
        # HELPER: Fake composite Omni config declaring thinker and code2wav sub-configs.
        sub_configs = {"thinker_config": object, "code2wav_config": object}

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets thinker_config (wrapping the shared text config) and code2wav_config with its own rope_theta.
            self.thinker_config = SimpleNamespace(text_config=text_config)
            self.code2wav_config = SimpleNamespace(rope_theta=10_000.0)

        def get_text_config(self):
            # [FUNCTION DOC]
            # HELPER: Returns the shared text config.
            return text_config

    assert get_hf_rope_theta(OmniConfig()) == 1_000_000.0


def test_get_hf_rope_theta_from_nested_transformers_v5_parameters():
    # [FUNCTION DOC]
    # A doubly nested config whose language config carries rope_parameters with a full_attention rope_theta of 1e4:
    # asserts that value is found.
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            language_config=SimpleNamespace(rope_parameters={"full_attention": {"rope_theta": 10_000.0}}),
        )
    )

    assert get_hf_rope_theta(config) == 10_000.0


def test_get_hf_config_attr_handles_nested_configs_and_cycles():
    # [FUNCTION DOC]
    # A config whose model_config points back to itself (a cycle) and whose thinker text config has hidden_size 4096:
    # asserts the lookup terminates and returns 4096.
    config = SimpleNamespace()
    config.thinker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4096))
    config.model_config = config

    assert get_hf_config_attr(config, "hidden_size") == 4096


def test_get_hf_config_attr_uses_standard_sub_configs_and_text_accessor():
    # [FUNCTION DOC]
    # When the root has hidden_size 1024, a decoder sub-config 2048 and get_text_config returns 8192, asserts 8192 wins.
    text_config = SimpleNamespace(hidden_size=8192)

    class CompositeConfig:
        # [CLASS DOC]
        # HELPER: Fake composite config with a root hidden_size, a decoder sub-config, and a text-config accessor.
        sub_configs = {"decoder_config": object}

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets root hidden_size 1024 and decoder_config hidden_size 2048.
            self.hidden_size = 1024
            self.decoder_config = SimpleNamespace(hidden_size=2048)

        def get_text_config(self):
            # [FUNCTION DOC]
            # HELPER: Returns the outer text config (hidden_size 8192).
            return text_config

    assert get_hf_config_attr(CompositeConfig(), "hidden_size") == 8192


def test_get_hf_config_attr_prefers_direct_text_config_over_root():
    # [FUNCTION DOC]
    # A direct text_config value (4096) is preferred over the root value (1024).
    config = SimpleNamespace(hidden_size=1024, text_config=SimpleNamespace(hidden_size=4096))

    assert get_hf_config_attr(config, "hidden_size") == 4096


def test_nested_config_lookup_does_not_fall_back_to_multimodal_sibling():
    # [FUNCTION DOC]
    # With only audio and code2wav sub-configs, looking up hidden_size raises AttributeError ("has no nested
    # hidden_size") and rope_theta raises AttributeError ("has no rope_theta"), i.e. no fallback to siblings.
    class CompositeConfig:
        # [CLASS DOC]
        # HELPER: Fake composite config with only multimodal (audio, code2wav) sub-configs.
        sub_configs = {"audio_config": object, "code2wav_config": object}

        def __init__(self):
            # [FUNCTION DOC]
            # HELPER: Sets audio_config and code2wav_config namespaces.
            self.audio_config = SimpleNamespace(hidden_size=1024)
            self.code2wav_config = SimpleNamespace(rope_theta=10_000.0)

    with pytest.raises(AttributeError, match="has no nested hidden_size"):
        get_hf_config_attr(CompositeConfig(), "hidden_size")
    with pytest.raises(AttributeError, match="has no rope_theta"):
        get_hf_rope_theta(CompositeConfig())


def test_get_hf_config_attr_preserves_falsey_text_values():
    # [FUNCTION DOC]
    # False and 0 values in text_config are returned as-is rather than treated as missing.
    config = SimpleNamespace(text_config=SimpleNamespace(flag=False, count=0))

    assert get_hf_config_attr(config, "flag") is False
    assert get_hf_config_attr(config, "count") == 0


def test_nested_config_lookup_rejects_missing_attributes():
    # [FUNCTION DOC]
    # An empty text_config with no root attribute raises AttributeError ("has no nested hidden_size").
    config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(AttributeError, match="has no nested hidden_size"):
        get_hf_config_attr(config, "hidden_size")
