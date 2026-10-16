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
# Tests for the example-script naming checker (tests/special_sanity/check_example_naming.py).
# Canonical form under examples/ is run_<model>_<train-backend>.sh where the train backend is
# one of ALLOWED_BACKENDS and must be the final token; tokens in FORBIDDEN_TOKENS (engine,
# platform, quantization, machine names) are rejected anywhere in the name.

"""Unit-style smoke tests for ``check_example_naming``."""

from __future__ import annotations

from pathlib import Path

from tests.special_sanity.check_example_naming import (
    ALLOWED_BACKENDS,
    FORBIDDEN_TOKENS,
    check_filename,
    main,
)


def _violations(name: str) -> list[str]:
    # [FUNCTION DOC]
    # HELPER: Runs check_filename on a synthetic path examples/grpo_trainer/<name>.
    # Inputs: name - bare script filename. Outputs: list of violation message strings (empty = valid).
    # Side effects: none (path need not exist).
    return check_filename(Path(f"examples/grpo_trainer/{name}"))


def test_canonical_name_passes():
    # [FUNCTION DOC]
    # Asserts run_qwen3_8b_fsdp.sh produces no violations.
    assert _violations("run_qwen3_8b_fsdp.sh") == []


def test_pre_backend_suffix_passes():
    # [FUNCTION DOC]
    # Asserts extra model tokens before the backend (from_adapter, merge) are allowed.
    # Tokens before the train-backend are part of <model>; only the trailing
    # token is constrained.
    assert _violations("run_qwen3_8b_from_adapter_fsdp.sh") == []
    assert _violations("run_qwen3_8b_merge_fsdp.sh") == []


def test_all_train_backends_accepted():
    # [FUNCTION DOC]
    # Iterates every allowed backend and asserts run_qwen3_8b_<backend>.sh is valid.
    for backend in ALLOWED_BACKENDS:
        assert _violations(f"run_qwen3_8b_{backend}.sh") == [], backend


def test_multi_token_train_backend_accepted():
    # [FUNCTION DOC]
    # Asserts a backend containing an underscore (megatron_lite) is recognized as the trailing backend.
    assert _violations("run_deepseek_v4_megatron_lite.sh") == []


def test_forbidden_engine_token_rejected():
    # [FUNCTION DOC]
    # Asserts a name containing "vllm" yields at least one error mentioning vllm.
    errs = _violations("run_qwen3_8b_vllm_fsdp.sh")
    # `vllm` is both a forbidden token AND occupies the last-token slot
    # (`fsdp` is no longer last). Ensure at least one error mentions vllm.
    assert errs and any("vllm" in e for e in errs)


def test_forbidden_platform_token_rejected():
    # [FUNCTION DOC]
    # Asserts the trailing "npu" token produces an error mentioning npu.
    errs = _violations("run_qwen3_8b_fsdp_npu.sh")
    assert errs and any("npu" in e for e in errs)


def test_forbidden_quantization_token_rejected():
    # [FUNCTION DOC]
    # Asserts the trailing "fp8" token produces an error mentioning fp8.
    errs = _violations("run_qwen3_30b_a3b_megatron_fp8.sh")
    assert errs and any("fp8" in e for e in errs)


def test_forbidden_machine_token_rejected():
    # [FUNCTION DOC]
    # Asserts the trailing "gb200" token produces an error mentioning gb200.
    errs = _violations("run_qwen3_8b_fsdp_gb200.sh")
    assert errs and any("gb200" in e for e in errs)


def test_trailing_suffix_after_backend_rejected():
    # [FUNCTION DOC]
    # Asserts a non-forbidden suffix after the backend still errors with the
    # "must end with '_<train-backend>.sh'" message.
    # Even if no token is in FORBIDDEN_TOKENS, anything after the train-
    # backend is a violation: the train-backend MUST be the last token.
    errs = _violations("run_qwen3_8b_fsdp_extra.sh")
    assert errs
    assert any("must end with '_<train-backend>.sh'" in e for e in errs)


def test_missing_train_backend_rejected():
    # [FUNCTION DOC]
    # Asserts a name lacking a backend errors and the first error lists an allowed backend name.
    errs = _violations("run_qwen3_8b.sh")
    assert errs and any(b in errs[0] for b in ALLOWED_BACKENDS)


def test_non_run_prefix_rejected():
    # [FUNCTION DOC]
    # Asserts a name not starting with run_ errors, the first error mentioning "run_".
    errs = _violations("badname.sh")
    assert errs and "run_" in errs[0]


def test_forbidden_tokens_kept_in_sync_with_legacy_pattern():
    # [FUNCTION DOC]
    # Asserts each legacy token (vllm, sglang, trtllm, npu, gb200, fp8) is in FORBIDDEN_TOKENS.
    # If we ever forget to forbid one of the deprecated tokens, this test
    # would also start failing because at least one filename in the pre-
    # refactor era used each of these.
    for tok in ("vllm", "sglang", "trtllm", "npu", "gb200", "fp8"):
        assert tok in FORBIDDEN_TOKENS, tok


def test_repo_tree_passes(tmp_path):
    # [FUNCTION DOC]
    # Runs main against the real examples/ tree (repo root = cwd) and asserts exit code 0.
    # Inputs: tmp_path fixture (unused). Side effects: reads the repository filesystem; must run from repo root.
    # Run the entry point against the actual ``examples/`` tree to mirror
    # what pre-commit will do. ``main`` returns 0 on success.
    assert main(["--root", "examples", "--repo-root", "."]) == 0


def test_synthetic_violation_fails(tmp_path):
    # [FUNCTION DOC]
    # Creates a temp examples/grpo_trainer dir with one bad (vllm) and one good script, runs main with
    # that root, empty ignore-dir/ignore-file lists, and asserts exit code 1.
    # Side effects: writes files under tmp_path.
    fake = tmp_path / "examples" / "grpo_trainer"
    fake.mkdir(parents=True)
    (fake / "run_qwen3_8b_vllm_fsdp.sh").write_text("#!/bin/bash\n")
    (fake / "run_ok_fsdp.sh").write_text("#!/bin/bash\n")

    rc = main(
        [
            "--root",
            str(tmp_path / "examples"),
            "--repo-root",
            str(tmp_path),
            "--ignore-dirs",
            "--ignore-files",
        ]
    )
    assert rc == 1
