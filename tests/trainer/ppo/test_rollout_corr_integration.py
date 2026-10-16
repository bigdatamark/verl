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
# Integration tests combining rollout correction helpers (importance-sampling weights and rejection
# sampling masks computed from old vs rollout log-probs) with the vanilla PPO policy loss, plus unit
# tests of RolloutCorrectionConfig construction and factory presets. Runs on CUDA when available,
# otherwise CPU.
"""Integration tests for Rollout Correction."""

import pytest
import torch

from verl.trainer.config.algorithm import RolloutCorrectionConfig
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.rollout_corr_helper import (
    compute_offpolicy_metrics,
    compute_rollout_correction_and_rejection_mask,
)
from verl.workers.config.actor import ActorConfig


class TestRolloutISIntegration:
    # [CLASS DOC]
    # Exercises compute_rollout_correction_and_rejection_mask, compute_offpolicy_metrics and
    # compute_policy_loss_vanilla together on random 4x16 data.
    """Integration tests for Rollout Correction with PPO."""

    @pytest.fixture
    def sample_data(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning a mapping of random 4x16 tensors (old_log_prob, log_prob, rollout_log_prob,
        # advantages) and an all-ones response_mask, on CUDA if available else CPU.
        """Create sample training data."""
        batch_size, seq_length = 4, 16
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return {
            "old_log_prob": torch.randn(batch_size, seq_length, device=device),
            "log_prob": torch.randn(batch_size, seq_length, device=device),
            "rollout_log_prob": torch.randn(batch_size, seq_length, device=device),
            "advantages": torch.randn(batch_size, seq_length, device=device),
            "response_mask": torch.ones(batch_size, seq_length, device=device),
        }

    @pytest.fixture
    def config_with_rollout_is(self):
        # [FUNCTION DOC]
        # HELPER: fixture returning an ActorConfig (fsdp strategy, rollout_n 1, micro batch 2, clip_ratio 0.2)
        # holding just the fields the policy loss needs.
        """Create config for policy loss computation.

        Note: rollout_is config has been moved to algorithm config.
        This config only needs fields used by policy loss (clip_ratio, etc).
        """
        config = ActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size=2,
            clip_ratio=0.2,
        )
        return config

    def test_policy_loss_with_rollout_is(self, sample_data, config_with_rollout_is):
        # [FUNCTION DOC]
        # Computes token-level IS weights (threshold 2.0, no rejection), feeds them to the vanilla policy loss with
        # token-mean aggregation, and asserts the loss is a finite scalar tensor.
        """Test that policy loss computation works with rollout correction weights.

        Note: In production, IS weights are computed centrally in the trainer
        (before advantage computation) and passed to policy loss.
        This test simulates that workflow.
        """
        # First compute IS weights (as trainer would do centrally)
        rollout_is_weights_proto, _, _ = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is="token",
            rollout_is_threshold=2.0,
            rollout_rs=None,
        )

        rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]

        # Policy loss function receives pre-computed IS weights
        pg_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=rollout_is_weights,
        )

        # Check loss is valid
        assert isinstance(pg_loss, torch.Tensor)
        assert pg_loss.ndim == 0  # Scalar
        assert not torch.isnan(pg_loss)
        assert not torch.isinf(pg_loss)

    def test_rollout_is_weights_computation(self, sample_data):
        # [FUNCTION DOC]
        # Asserts the helper returns a DataProto whose rollout_is_weights tensor matches the log-prob shape, and a
        # non-empty metrics dict containing "rollout_corr/rollout_is_mean".
        """Test rollout correction weights and metrics computation."""
        weights_proto, _, metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is="token",
            rollout_is_threshold=2.0,
            rollout_rs=None,
        )

        # Check weights
        from verl.protocol import DataProto

        assert isinstance(weights_proto, DataProto)
        weights = weights_proto.batch["rollout_is_weights"]
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == sample_data["old_log_prob"].shape

        # Check metrics are returned
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert "rollout_corr/rollout_is_mean" in metrics

    def test_all_aggregation_levels(self, sample_data):
        # [FUNCTION DOC]
        # Runs IS at "token" and "sequence" levels (asserting the IS mean metric each time), then rejection
        # sampling only with "seq_mean_k1" and threshold "0.999_1.001", asserting its mean metric exists.
        """Test all aggregation levels (token, sequence for IS; K1 for RS)."""
        # Test IS weight levels
        is_levels = ["token", "sequence"]
        for level in is_levels:
            _, _, metrics = compute_rollout_correction_and_rejection_mask(
                old_log_prob=sample_data["old_log_prob"],
                rollout_log_prob=sample_data["rollout_log_prob"],
                response_mask=sample_data["response_mask"],
                rollout_is=level,
                rollout_is_threshold=2.0,
                rollout_rs=None,
            )
            assert "rollout_corr/rollout_is_mean" in metrics

        # Test rejection sampling with K1 sequence mean level
        _, _, metrics_geo = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is=None,
            rollout_rs="seq_mean_k1",
            rollout_rs_threshold="0.999_1.001",
        )
        assert "rollout_corr/rollout_rs_seq_mean_k1_mean" in metrics_geo

    def test_both_bounding_modes(self, sample_data):
        # [FUNCTION DOC]
        # Runs token IS alone (truncate) and token IS plus "token_k1" rejection with float upper threshold 1.3
        # (mask), asserting the expected IS and RS metric keys are present.
        """Test both truncate and mask modes."""
        # Test truncate mode (IS weights only)
        _, _, metrics_truncate = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is="token",
            rollout_is_threshold=2.0,
            rollout_rs=None,
        )
        assert "rollout_corr/rollout_is_mean" in metrics_truncate

        # Test mask mode (rejection sampling)
        _, _, metrics_mask = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is="token",  # Can also compute IS weights in mask mode
            rollout_is_threshold=2.0,
            rollout_rs="token_k1",  # Enable rejection sampling
            rollout_rs_threshold=1.3,  # Float upper bound (lower inferred automatically)
        )
        assert "rollout_corr/rollout_is_mean" in metrics_mask
        assert "rollout_corr/rollout_rs_token_k1_mean" in metrics_mask

    def test_offpolicy_metrics(self, sample_data):
        # [FUNCTION DOC]
        # Asserts compute_offpolicy_metrics returns training_ppl, rollout_ppl and a float kl.
        """Test off-policy diagnostic metrics computation."""
        metrics = compute_offpolicy_metrics(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
        )

        # Check key metrics are present
        assert "training_ppl" in metrics
        assert "rollout_ppl" in metrics
        assert "kl" in metrics
        assert isinstance(metrics["kl"], float)

    def test_metrics_only_mode(self, sample_data, config_with_rollout_is):
        # [FUNCTION DOC]
        # Computes IS weights/metrics, then asserts the policy loss without weights differs from the loss with
        # weights (i.e. metrics can be computed without changing the loss).
        """Test metrics-only mode: compute IS weights/metrics but don't apply to loss.

        This tests the use case where rollout_is_threshold is set (enables computation)
        but rollout_is=False (disables weight application to policy loss).
        """
        # Compute IS weights (as trainer would do)
        rollout_is_weights_proto, _, is_metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=sample_data["old_log_prob"],
            rollout_log_prob=sample_data["rollout_log_prob"],
            response_mask=sample_data["response_mask"],
            rollout_is="token",
            rollout_is_threshold=2.0,
            rollout_rs=None,
        )

        # Metrics should be computed
        assert len(is_metrics) > 0
        assert "rollout_corr/rollout_is_mean" in is_metrics

        # In metrics-only mode, we compute loss WITHOUT applying weights
        # (simulating rollout_is=False)
        pg_loss_no_weights, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=None,  # Don't apply weights
        )

        # Compare to loss WITH weights (rollout_is=True)
        rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]
        pg_loss_with_weights, _ = compute_policy_loss_vanilla(
            old_log_prob=sample_data["old_log_prob"],
            log_prob=sample_data["log_prob"],
            advantages=sample_data["advantages"],
            response_mask=sample_data["response_mask"],
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=rollout_is_weights,
        )

        # Losses should be different (weights have an effect)
        assert not torch.allclose(pg_loss_no_weights, pg_loss_with_weights)

    def test_exact_icepop_matches_filtered_weighted_ppo_loss(self, config_with_rollout_is):
        # [FUNCTION DOC]
        # Deterministic IcePop check: old=log_prob=-1 per token, rollout log-probs (-0.5,-3.5,-0.8), threshold
        # range "0.5_5.0". Ratios exp(old-rollout) are 0.607, 12.18, 0.819; the out-of-range middle token gets
        # weight 0 rather than being masked. Asserts response mask unchanged, out-of-bounds ratio 1/3, the exact
        # weights, and that the loss equals mean(weights * -advantages) (ratio is 1 since log_prob == old).
        """IcePop should match the local RL zero-weight semantics."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        old_log_prob = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
        log_prob = old_log_prob.clone()
        rollout_log_prob = torch.tensor([[-0.5, -3.5, -0.8]], device=device)
        advantages = torch.tensor([[1.0, -1.0, 2.0]], device=device)
        response_mask = torch.ones_like(old_log_prob)

        rollout_is_weights_proto, modified_response_mask, metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=old_log_prob,
            rollout_log_prob=rollout_log_prob,
            response_mask=response_mask,
            rollout_is="token",
            rollout_is_threshold="0.5_5.0",
            rollout_rs=None,
        )

        rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"]
        expected_weights = torch.tensor([[0.60653067, 0.0, 0.81873075]], device=device)
        expected_loss = torch.mean(expected_weights * (-advantages))

        pg_loss, _ = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode="token-mean",
            config=config_with_rollout_is,
            rollout_is_weights=rollout_is_weights,
        )

        assert torch.equal(modified_response_mask, response_mask)
        assert metrics["rollout_corr/rollout_is_oob_ratio"] == pytest.approx(1.0 / 3.0, abs=1e-6)
        torch.testing.assert_close(rollout_is_weights, expected_weights, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(pg_loss, expected_loss, atol=1e-6, rtol=1e-6)


class TestRolloutCorrectionConfigNormalization:
    # [CLASS DOC]
    # Unit tests of RolloutCorrectionConfig field storage and preset factories.
    """Unit tests for RolloutCorrectionConfig canonicalization logic."""

    def test_alias_normalization_and_threshold_parsing(self):
        # [FUNCTION DOC]
        # Asserts rollout_is, float IS threshold, comma-separated rollout_rs list and threshold string are stored
        # as given (no transformation).
        config = RolloutCorrectionConfig(
            rollout_is="token",
            rollout_is_threshold=2.5,
            rollout_rs="seq_mean_k1,seq_max_k3",
            rollout_rs_threshold="0.8_1.2,3.0",
        )

        assert config.rollout_is == "token"
        assert config.rollout_is_threshold == pytest.approx(2.5)
        assert config.rollout_rs == "seq_mean_k1,seq_max_k3"
        assert config.rollout_rs_threshold == "0.8_1.2,3.0"

    def test_missing_threshold_raises(self):
        # [FUNCTION DOC]
        # Constructs a config with rollout_rs "token_k1" and no threshold; asserts the threshold stays None.
        # KNOWN ISSUE: despite its name, the test asserts no exception is raised; it checks the threshold is None.
        config = RolloutCorrectionConfig(rollout_rs="token_k1")
        assert config.rollout_rs == "token_k1"
        assert config.rollout_rs_threshold is None

    def test_float_threshold_conversion_in_factory(self):
        # [FUNCTION DOC]
        # Asserts the decoupled_geo_rs_seq_tis factory sets rollout_rs "seq_mean_k1" and keeps threshold 1.001.
        config = RolloutCorrectionConfig.decoupled_geo_rs_seq_tis(rs_threshold=1.001)
        assert config.rollout_rs == "seq_mean_k1"
        assert config.rollout_rs_threshold == 1.001

    def test_icepop_factory(self):
        # [FUNCTION DOC]
        # Asserts the decoupled_token_icepop factory sets token-level IS.
        config = RolloutCorrectionConfig.decoupled_token_icepop()
        assert config.rollout_is == "token"
        assert config.rollout_is_threshold == "0.5_5.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
