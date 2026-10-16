# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# CPU tests for verl.trainer.ppo.core_algos: the advantage-estimator registry, multi-turn GAE masking,
# equivalence of loop vs vectorized RLOO/GRPO advantage implementations, and the straight-through "+"
# KL penalty variants. Runnable directly via unittest.

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    kl_penalty,
    register_adv_est,
)


def mock_test_fn():
    # [FUNCTION DOC]
    # HELPER: No-op function used as a registration target in duplicate-registration tests.
    pass


class TestRegisterAdvEst(unittest.TestCase):
    # [CLASS DOC]
    # unittest case for register_adv_est / get_adv_estimator_fn. Each test runs against a fresh registry
    # containing only two fake estimators: "gae" (doubles input) and "vtrace" (adds one).
    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Clears the real module-level ADV_ESTIMATOR_REGISTRY, then rebinds the module attribute to a new
        # mapping with the fake "gae"/"vtrace" lambdas and keeps a reference on self.
        # Side effects: mutates global state of core_algos.
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        # [FUNCTION DOC]
        # HELPER: Clears the (test) registry and calls the base tearDown.
        # KNOWN ISSUE: the original registry contents are never restored, so after this class runs the core_algos registry is empty for any later test in the same process.
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        # [FUNCTION DOC]
        # Registers a nested function under a string name and asserts the registry maps the name to it.
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            # [FUNCTION DOC]
            # HELPER: No-op function registered as "test_estimator".
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        # [FUNCTION DOC]
        # Registers via a locally defined Enum member and asserts the registry key is the enum's string value.
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            # [CLASS DOC]
            # HELPER: Local Enum with a single member TEST = "test_enum_estimator" (shadows, not the real AdvantageEstimator).
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            # [FUNCTION DOC]
            # HELPER: No-op function registered under the enum member.
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        # [FUNCTION DOC]
        # Registers the same function twice under one name and asserts no error and the mapping is kept.
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        # [FUNCTION DOC]
        # Asserts registering a second, different function under an existing name raises ValueError.
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            # [FUNCTION DOC]
            # HELPER: No-op function registered first as "conflict_test".
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                # [FUNCTION DOC]
                # HELPER: No-op function whose registration under "conflict_test" is expected to fail.
                pass

    def test_decorator_preserves_function(self):
        # [FUNCTION DOC]
        # Asserts the decorator returns the original callable (calling it still returns "original").
        """Test that the decorator returns the original function"""

        def test_fn():
            # [FUNCTION DOC]
            # HELPER: Returns the string "original".
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        # [FUNCTION DOC]
        # Registers two functions and asserts the registry grew by two and maps each name correctly.
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            # [FUNCTION DOC]
            # HELPER: No-op registered as "estimator1".
            pass

        @register_adv_est("estimator2")
        def fn2():
            # [FUNCTION DOC]
            # HELPER: No-op registered as "estimator2".
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        # [FUNCTION DOC]
        # Asserts lookup of "gae"/"vtrace" returns the fake lambdas (5 -> 10 and 5 -> 6).
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        # [FUNCTION DOC]
        # Asserts an unknown name raises ValueError with message "Unknown advantage estimator simply: invalid_name".
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        # [FUNCTION DOC]
        # Asserts lookup is case-sensitive ("GAE" raises ValueError).
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    # [FUNCTION DOC]
    # With random gamma/lam, computes GAE twice on an 11-token sequence where two value tensors differ only at
    # masked (response_mask 0) positions; asserts advantages are identical and returns are identical after
    # masking, i.e. values of observation tokens are ignored. Prints results.
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: Builds a shuffled int64 numpy array of group ids of length batch_size with num_groups groups,
    # each of size at least 2 (extra samples assigned to random groups). Uses the global random module.
    # Exceptions: AssertionError if batch_size < 2 * num_groups.
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Returns a random 0/1 float mask of shape (batch_size, seq_len), forcing the last column to 1 in any all-zero row.
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # [FUNCTION DOC]
    # Parametrized over batch size / sequence length / group count / seed: seeds torch, random and numpy,
    # builds groups, mask and masked random rewards, and asserts the loop and vectorized RLOO implementations
    # yield identical-shaped advantages and returns within rtol 1e-5 / atol 1e-6. Prints max differences.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_grpo_vectorized_matches_original_for_low_variance_rewards():
    # [FUNCTION DOC]
    # Asserts loop and vectorized GRPO agree for two groups whose rewards differ by only 1e-5 (near-zero std edge case), with string group ids.
    token_level_rewards = torch.tensor([[1.0], [1.00001], [2.0], [2.00001]], dtype=torch.float32)
    response_mask = torch.ones_like(token_level_rewards)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)

    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # [FUNCTION DOC]
    # Same randomized equivalence check as the RLOO test, for compute_grpo_outcome_advantage versus its vectorized variant.
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "name,base",
    [
        ("k1+", "k1"),
        ("kl+", "kl"),
        ("abs+", "abs"),
        ("k3+", "k3"),
        ("low_var_kl+", "low_var_kl"),
    ],
)
def test_kl_penalty_straight_through_value_matches_base(name, base):
    # [FUNCTION DOC]
    # Parametrized over k1+/kl+/abs+/k3+/low_var_kl+: asserts the forward value equals the base estimator's value.
    """The ``+`` suffix is a straight-through trick that swaps in the k2
    gradient while keeping the base estimator's value. Therefore the forward
    value of e.g. ``k3+`` must match the value of plain ``k3``.

    Regression test for the bug where ``kl_penalty(..., "k3+")`` raised
    ``NotImplementedError`` because the wrapper forwarded the ``+`` suffix to
    ``kl_penalty_forward`` without stripping it.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    plus_value = kl_penalty(logprob, ref_logprob, name)
    base_value = kl_penalty(logprob, ref_logprob, base)
    assert torch.allclose(plus_value, base_value)


def test_kl_penalty_k3_plus_uses_k2_gradient():
    # [FUNCTION DOC]
    # Asserts the gradient of the summed "k3+" penalty with respect to logprob equals that of the "k2" penalty.
    """With ``k3+`` the gradient w.r.t. ``logprob`` should equal the gradient
    obtained from the ``k2`` (``0.5 * log_ratio**2``) estimator, since the
    straight-through trick routes the backward pass through ``k2``.
    """
    torch.manual_seed(0)
    logprob = torch.randn(4, 8, requires_grad=True)
    ref_logprob = torch.randn(4, 8)

    out_plus = kl_penalty(logprob, ref_logprob, "k3+").sum()
    (grad_plus,) = torch.autograd.grad(out_plus, logprob)

    logprob_k2 = logprob.detach().clone().requires_grad_(True)
    out_k2 = kl_penalty(logprob_k2, ref_logprob, "k2").sum()
    (grad_k2,) = torch.autograd.grad(out_k2, logprob_k2)

    assert torch.allclose(grad_plus, grad_k2)


if __name__ == "__main__":
    unittest.main()
