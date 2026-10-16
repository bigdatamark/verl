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
# GPU-free tests for prefill/decode (PD) disaggregation: DisaggregationConfig defaults and validation,
# RolloutConfig backend compatibility, replica registry dispatch, and the rollout-rank to PD role mapping.

"""GPU-free unit tests for PD disaggregation config + replica plumbing."""

from __future__ import annotations

import pytest

from verl.workers.config import DisaggregationConfig, RolloutConfig


def test_disaggregation_defaults_disabled_and_valid():
    # [FUNCTION DOC]
    # Asserts defaults: disabled, 1 prefill and 1 decode replica, nixl backend, no bootstrap port, no IB device.
    cfg = DisaggregationConfig()
    assert cfg.enabled is False
    assert cfg.prefill_replicas == 1
    assert cfg.decode_replicas == 1
    assert cfg.transfer_backend == "nixl"
    assert cfg.bootstrap_port is None
    assert cfg.ib_device is None


def test_disaggregation_enabled_nixl_accepted():
    # [FUNCTION DOC]
    # Asserts enabling with nixl backend is accepted.
    cfg = DisaggregationConfig(enabled=True, transfer_backend="nixl")
    assert cfg.enabled is True and cfg.transfer_backend == "nixl"


def test_disaggregation_enabled_mooncake_accepted():
    # [FUNCTION DOC]
    # Asserts enabling with mooncake backend and an IB device is accepted.
    cfg = DisaggregationConfig(enabled=True, transfer_backend="mooncake", ib_device="mlx5_roce0")
    assert cfg.transfer_backend == "mooncake"


def test_disaggregation_unknown_backend_rejected():
    # [FUNCTION DOC]
    # Asserts an unknown transfer backend raises ValueError mentioning transfer_backend.
    with pytest.raises(ValueError, match="transfer_backend"):
        DisaggregationConfig(enabled=True, transfer_backend="bogus")


def test_disaggregation_zero_replicas_rejected():
    # [FUNCTION DOC]
    # Asserts zero prefill or zero decode replicas raise ValueError naming the field.
    with pytest.raises(ValueError, match="prefill_replicas"):
        DisaggregationConfig(enabled=True, prefill_replicas=0)
    with pytest.raises(ValueError, match="decode_replicas"):
        DisaggregationConfig(enabled=True, decode_replicas=0)


def test_disaggregation_bad_bootstrap_port_rejected():
    # [FUNCTION DOC]
    # Asserts an out-of-range bootstrap port (70000) raises ValueError.
    with pytest.raises(ValueError, match="bootstrap_port"):
        DisaggregationConfig(enabled=True, bootstrap_port=70000)


def test_disaggregation_disabled_skips_validation():
    # [FUNCTION DOC]
    # Asserts invalid values are tolerated when disabled.
    """When enabled=False, even bad values are tolerated (matches YAML defaults)."""
    cfg = DisaggregationConfig(enabled=False, transfer_backend="bogus", bootstrap_port=70000)
    assert cfg.enabled is False


def test_rollout_config_sglang_with_disagg_enabled_ok():
    # [FUNCTION DOC]
    # Asserts an sglang RolloutConfig accepts enabled disaggregation.
    cfg = RolloutConfig(
        name="sglang",
        disaggregation=DisaggregationConfig(enabled=True),
    )
    assert cfg.disaggregation.enabled is True


@pytest.mark.parametrize("name", ["trtllm"])
def test_rollout_config_non_pd_backend_rejects_pd(name):
    # [FUNCTION DOC]
    # Parametrized (trtllm); asserts enabling disaggregation on a non-PD backend raises ValueError.
    with pytest.raises(ValueError, match="disaggregation.enabled=True"):
        RolloutConfig(
            name=name,
            disaggregation=DisaggregationConfig(enabled=True),
        )


def test_rollout_config_disabled_pd_works_on_any_backend():
    # [FUNCTION DOC]
    # Asserts sglang, vllm and trtllm configs default to disabled disaggregation.
    for name in ("sglang", "vllm", "trtllm"):
        cfg = RolloutConfig(name=name)
        assert cfg.disaggregation.enabled is False


def test_effective_decode_tp_defaults_to_prefill_tp():
    # [FUNCTION DOC]
    # Asserts decode TP falls back to the prefill TP when not overridden.
    cfg = DisaggregationConfig(enabled=True)
    assert cfg.effective_decode_tp(prefill_tp=4) == 4


def test_effective_decode_tp_respects_override():
    # [FUNCTION DOC]
    # Asserts an explicit decode TP override is returned.
    cfg = DisaggregationConfig(enabled=True, decode_tensor_model_parallel_size=2)
    assert cfg.effective_decode_tp(prefill_tp=4) == 2


def test_registry_no_pd_aliases():
    # [FUNCTION DOC]
    # Asserts the replica registry has sglang and vllm entries but no sglang_pd / vllm_pd aliases.
    """``sglang_pd``/``vllm_pd`` were dropped: PD is selected via the disaggregation flag."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    assert "sglang" in RolloutReplicaRegistry._registry
    assert "vllm" in RolloutReplicaRegistry._registry
    assert "sglang_pd" not in RolloutReplicaRegistry._registry
    assert "vllm_pd" not in RolloutReplicaRegistry._registry


def _sglang_available() -> bool:
    # [FUNCTION DOC]
    # HELPER: returns whether the sglang package spec can be found.
    import importlib.util

    return importlib.util.find_spec("sglang") is not None


@pytest.mark.skipif(not _sglang_available(), reason="sglang not installed")
def test_dispatch_sglang_returns_pd_replica_when_flag_set():
    # [FUNCTION DOC]
    # Skipped without sglang; asserts the replica class lookup returns SGLangReplica normally and its subclass
    # SGLangPDReplica when disaggregation is enabled.
    from verl.workers.rollout.replica import get_rollout_replica_class

    plain_cls = get_rollout_replica_class("sglang", disaggregation_enabled=False)
    pd_cls = get_rollout_replica_class("sglang", disaggregation_enabled=True)
    assert plain_cls.__name__ == "SGLangReplica"
    assert pd_cls.__name__ == "SGLangPDReplica"
    assert issubclass(pd_cls, plain_cls)


def test_dispatch_non_pd_backend_with_flag_raises():
    # [FUNCTION DOC]
    # Asserts requesting a trtllm replica with disaggregation enabled raises NotImplementedError.
    from verl.workers.rollout.replica import get_rollout_replica_class

    with pytest.raises(NotImplementedError, match="PD disaggregation"):
        get_rollout_replica_class("trtllm", disaggregation_enabled=True)


def _assign_pd_role(rollout_rank: int, prefill_tp: int, decode_replicas: int, decode_tp: int):
    # [FUNCTION DOC]
    # HELPER: mirrors ServerAdapter's role assignment: ranks below prefill_tp are prefill server 0 with tp rank equal
    # to the rank; the next decode_replicas times decode_tp ranks are decode with server index offset // decode_tp and
    # tp rank offset % decode_tp; beyond that returns (None, None, None).
    # KNOWN ISSUE: this is a copy of production logic, so the tests below do not exercise the real ServerAdapter.
    """Mirror of ServerAdapter.__init__'s role-assignment block."""
    if rollout_rank < prefill_tp:
        return "prefill", 0, rollout_rank
    off = rollout_rank - prefill_tp
    if off < decode_replicas * decode_tp:
        return "decode", off // decode_tp, off % decode_tp
    return None, None, None


@pytest.mark.parametrize(
    "prefill_tp,decode_replicas,decode_tp,rollout_rank,expected",
    [
        (1, 3, 1, 0, ("prefill", 0, 0)),
        (1, 3, 1, 1, ("decode", 0, 0)),
        (1, 3, 1, 2, ("decode", 1, 0)),
        (1, 3, 1, 3, ("decode", 2, 0)),
        (1, 7, 1, 0, ("prefill", 0, 0)),
        (1, 7, 1, 7, ("decode", 6, 0)),
        (2, 3, 2, 0, ("prefill", 0, 0)),
        (2, 3, 2, 1, ("prefill", 0, 1)),
        (2, 3, 2, 2, ("decode", 0, 0)),
        (2, 3, 2, 3, ("decode", 0, 1)),
        (2, 3, 2, 6, ("decode", 2, 0)),
        (2, 3, 2, 7, ("decode", 2, 1)),
    ],
)
def test_pd_role_assignment(prefill_tp, decode_replicas, decode_tp, rollout_rank, expected):
    # [FUNCTION DOC]
    # Parametrized table of (prefill TP, decode replicas, decode TP, rank) to expected role tuples; asserts each.
    assert _assign_pd_role(rollout_rank, prefill_tp, decode_replicas, decode_tp) == expected


@pytest.mark.parametrize("prefill_tp,decode_replicas,decode_tp", [(1, 3, 1), (1, 7, 1), (2, 3, 2), (1, 1, 4)])
def test_pd_role_covers_every_rank_exactly_once(prefill_tp, decode_replicas, decode_tp):
    # [FUNCTION DOC]
    # Parametrized layouts; asserts every rank in the world gets a role and all role tuples are distinct.
    world = prefill_tp + decode_replicas * decode_tp
    seen: set[tuple[str, int, int]] = set()
    for rr in range(world):
        role, srv, tp_rank = _assign_pd_role(rr, prefill_tp, decode_replicas, decode_tp)
        assert role is not None, f"rollout_rank={rr} got no role"
        seen.add((role, srv, tp_rank))
    assert len(seen) == world, "each rank must map to a distinct (role, server_index, tp_local_rank) triple"
