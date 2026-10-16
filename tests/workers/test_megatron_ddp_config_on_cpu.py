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
# Tests MegatronEngine._resolve_override_ddp_config: grad_reduce_in_fp32 is derived from the optimizer's effective main
# gradient dtype unless explicitly overridden, and nothing is injected when there is no optimizer.

from types import SimpleNamespace

import pytest

from verl.workers.engine.megatron.transformer_impl import MegatronEngine


class _Config(SimpleNamespace):
    # [CLASS DOC]
    # HELPER: SimpleNamespace that also supports a dict-like get(key, default).
    def get(self, key, default=None):
        # [FUNCTION DOC]
        # HELPER: Returns the attribute named key, or default if absent.
        return getattr(self, key, default)


def _resolve_ddp_config(
    *,
    use_precision_aware_optimizer=False,
    main_grads_dtype="fp32",
    override_ddp_config=None,
    with_optimizer=True,
):
    # [FUNCTION DOC]
    # HELPER: Builds a MegatronEngine without __init__ with an engine config holding override_ddp_config (default empty)
    # and, if with_optimizer, an adam optimizer config with the given precision-aware flag and main grads dtype (else None);
    # returns the engine's resolved DDP override dict.
    engine = object.__new__(MegatronEngine)
    engine.engine_config = _Config(override_ddp_config=override_ddp_config or {})
    engine.optimizer_config = (
        _Config(
            optimizer="adam",
            use_precision_aware_optimizer=use_precision_aware_optimizer,
            main_grads_dtype=main_grads_dtype,
        )
        if with_optimizer
        else None
    )
    return engine._resolve_override_ddp_config()


@pytest.mark.parametrize(
    ("use_precision_aware_optimizer", "main_grads_dtype", "expected"),
    [
        (False, "fp32", True),
        (False, "bf16", True),
        (True, "fp32", True),
        (True, "bf16", False),
    ],
)
def test_ddp_grad_dtype_follows_effective_main_grad_dtype(use_precision_aware_optimizer, main_grads_dtype, expected):
    # [FUNCTION DOC]
    # grad_reduce_in_fp32 is True unless the precision-aware optimizer is used with bf16 main grads.
    resolved = _resolve_ddp_config(
        use_precision_aware_optimizer=use_precision_aware_optimizer,
        main_grads_dtype=main_grads_dtype,
    )

    assert resolved["grad_reduce_in_fp32"] is expected


@pytest.mark.parametrize("explicit_value", [False, True])
def test_explicit_ddp_grad_dtype_override_wins(explicit_value):
    # [FUNCTION DOC]
    # An explicit grad_reduce_in_fp32 override is preserved regardless of the optimizer settings.
    resolved = _resolve_ddp_config(
        use_precision_aware_optimizer=True,
        main_grads_dtype="bf16",
        override_ddp_config={"grad_reduce_in_fp32": explicit_value},
    )

    assert resolved["grad_reduce_in_fp32"] is explicit_value


def test_optimizerless_engine_does_not_inject_grad_dtype():
    # [FUNCTION DOC]
    # Without an optimizer config the resolved overrides are empty.
    assert _resolve_ddp_config(with_optimizer=False) == {}
