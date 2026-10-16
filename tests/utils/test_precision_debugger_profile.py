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
# CPU test for PrecisionDebuggerProfiler._resolve_model with Megatron-style engines whose module is a list of
# pipeline model chunks.

import logging
import types

from verl.utils.profiler.config import PrecisionDebuggerToolConfig
from verl.utils.profiler.precision_debugger_profile import PrecisionDebuggerProfiler


class _FakeModel:
    # [CLASS DOC]
    # HELPER: Minimal model type with a forward method (the property used to recognize a valid chunk).
    def forward(self):
        # [FUNCTION DOC]
        # HELPER: No-op forward.
        pass


def test_resolve_megatron_model_chunks_uses_first_valid_chunk(caplog):
    # [FUNCTION DOC]
    # A worker whose actor engine module list is [plain object, model A, model B]: asserts _resolve_model returns model A
    # (first chunk with forward) and logs a warning that only the first of 2 model chunks is bound.
    """Megatron's engine module may contain pipeline model chunks."""
    first_model = _FakeModel()
    second_model = _FakeModel()
    worker = types.SimpleNamespace(
        actor=types.SimpleNamespace(
            engine=types.SimpleNamespace(module=[object(), first_model, second_model]),
        )
    )
    profiler = PrecisionDebuggerProfiler(PrecisionDebuggerToolConfig())

    with caplog.at_level(logging.WARNING):
        model = profiler._resolve_model(worker, "actor_compute_log_prob")

    assert model is first_model
    assert "only binds the first of 2 model chunks" in caplog.text
