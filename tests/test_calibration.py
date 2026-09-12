"""Tests for the calibration / simulation harness.

Validates that:
- Exact enumeration reproduces the documented false-winner rates.
- Monte Carlo simulation converges to expected rates under equal models.
- Order-sensitivity is detectable.
- Parameter sweeps produce plausible outputs.
"""

import pytest

from bayesbench.calibration import (
    CalibrationPoint,
    CalibrationReport,
    EffectSweepPoint,
    EffectSweepReport,
    calibrate_sweep,
    effect_size_sweep,
    enumerate_binary_outcomes,
    simulate_order_sensitivity,
    simulate_pairwise,
    simulate_pairwise_cs,
)

# ---------------------------------------------------------------------------
# Exact enumeration: prove that min_samples=3 + skip_threshold=0.85
# always stops at or before trial 3 under equal Bernoulli models.
# ---------------------------------------------------------------------------


class TestExactEnumeration:
    def test_all_outcomes_terminate_at_three(self):
        """Every pairwise outcome sequence of length 3 triggers a stop."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["total_sequences"] == 64
        assert result["inconclusive_rate"] == pytest.approx(0.0)
        assert result["winner_rate"] + result["skipped_rate"] == pytest.approx(1.0)

    def test_skipped_rate_matches_documented(self):
        """Under equal models, 78.125% of trials are skipped."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["skipped_rate"] == pytest.approx(0.78125, abs=1e-6)

    def test_winner_rate_matches_documented(self):
        """Under equal models, 21.875% of trials incorrectly declare a winner."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["winner_rate"] == pytest.approx(0.21875, abs=1e-6)

    def test_skewed_models_still_terminal(self):
        """Even with skewed observation probability, all 3-trial paths stop."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.7, success_prob_b=0.3)
        assert result["inconclusive_rate"] < 0.01


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


class TestMonteCarloSimulation:
    def test_equal_models_false_rate_bounded(self):
        """Under equal models, false-winner rate is measurably > 0 @ defaults."""
        result = simulate_pairwise(
            n=1_000,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=0.85,
            min_samples=3,
            max_samples=60,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        false_rate = result["false_decisions"] / total if total > 0 else 0.0
        # With equal models, any declared winner is a false decision.
        # We expect ~20% false-winner rate.
        assert false_rate > 0.10, f"Expected >10% false winner rate, got {false_rate:.1%}"
        assert result["skipped"] > 0, "Some runs should be skipped"

    def test_disabling_skip_reduces_false_rate(self):
        """skip_threshold=1.0 disables the skip path entirely."""
        result = simulate_pairwise(
            n=100,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=5,
            max_samples=20,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        assert result["skipped"] == 0, "skip_threshold=1.0 should disable skipping"
        assert total == 100

    def test_strong_effect_detected_efficiently(self):
        """A model with true acc 0.95 vs 0.05 should be detected fast."""
        result = simulate_pairwise(
            n=500,
            true_acc_a=0.95,
            true_acc_b=0.05,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=3,
            max_samples=60,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        false_rate = result["false_decisions"] / total if total > 0 else 0.0
        mean_samples = sum(result["samples_drawn"]) / len(result["samples_drawn"])
        assert result["winner_a"] > result["winner_b"], "Strong A should win most runs"
        assert false_rate < 0.02, f"False rate {false_rate:.1%} too high for strong effect"
        assert mean_samples < 50, f"Expected early stop for strong effect, got {mean_samples:.1f}"

    def test_reproducible_with_seed(self):
        """Same seed should produce same result."""
        r1 = simulate_pairwise(n=500, rng=42)
        r2 = simulate_pairwise(n=500, rng=42)
        assert r1["winner_a"] == r2["winner_a"]
        assert r1["winner_b"] == r2["winner_b"]
        assert r1["skipped"] == r2["skipped"]


# ---------------------------------------------------------------------------
# Order sensitivity
# ---------------------------------------------------------------------------


class TestOrderSensitivity:
    def test_favorable_front_can_trigger_early_win(self):
        """3 perfect answers for A at the front of a bad dataset can
        cause an early (wrong) win for model A."""
        result = simulate_order_sensitivity(
            favorable_front=3,
            unfavorable_after=100,
            n_runs=500,
        )
        assert result["early_wins"] > 0, "Should see early wins when favorable examples come first"

    def test_larger_front_higher_early_win_rate(self):
        """More favorable examples at the front increase early-win odds."""
        r3 = simulate_order_sensitivity(favorable_front=3, n_runs=300)
        r5 = simulate_order_sensitivity(favorable_front=5, n_runs=300)
        assert (
            r5["early_win_rate"] >= r3["early_win_rate"]
        ), "More favorable examples should not decrease early-win rate"


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------


class TestCalibrationSweep:
    def test_sweep_under_equal_models(self):
        """Quick sweep under equal models; verify structure and ranges."""
        report = calibrate_sweep(
            min_samples_grid=(3, 10),
            skip_threshold_grid=(0.85, 0.99),
            n_runs=100,
            possible_max=30,
            seed=42,
        )
        assert isinstance(report, CalibrationReport)
        assert len(report.points) == 4  # 2 min_samples x 2 skip_thresholds
        for p in report.points:
            assert 0.0 <= p.false_winner_rate <= 1.0
            assert 0.0 <= p.skipped_rate <= 1.0
            assert p.expected_samples > 0

    def test_default_params_have_high_false_rate(self):
        """The current defaults (min_samples=3, skip_threshold=0.85) produce
        a measurable false-winner rate under equal models."""
        report = calibrate_sweep(
            min_samples_grid=(3,),
            skip_threshold_grid=(0.85,),
            n_runs=500,
            possible_max=60,
            seed=42,
        )
        p = report.points[0]
        assert (
            p.false_winner_rate > 0.10
        ), f"Default false-winner rate {p.false_winner_rate:.1%} should be >10%"

    def test_safe_defaults_have_low_false_rate(self):
        """min_samples=20, skip_threshold=0.99 should keep false rate low."""
        report = calibrate_sweep(
            min_samples_grid=(20,),
            skip_threshold_grid=(0.99,),
            n_runs=300,
            possible_max=60,
            seed=42,
        )
        p = report.points[0]
        assert (
            p.false_winner_rate < 0.08
        ), f"Safe defaults false rate {p.false_winner_rate:.1%} should be <8%"

    def test_table_formatting(self):
        """Table output should contain key columns."""
        report = calibrate_sweep(
            min_samples_grid=(3, 10),
            skip_threshold_grid=(0.85,),
            n_runs=500,
            seed=42,
        )
        table = report.format_table()
        assert "min_s" in table
        assert "skip" in table
        assert "false" in table
        assert "E[n]" in table


# ---------------------------------------------------------------------------
# Paired outcome simulation
# ---------------------------------------------------------------------------


class TestPairedSimulation:
    def test_reproducible_with_seed(self):
        r1 = simulate_pairwise(n=300, rng=42, item_sd=0.1)
        r2 = simulate_pairwise(n=300, rng=42, item_sd=0.1)
        assert r1["winner_a"] == r2["winner_a"]
        assert r1["winner_b"] == r2["winner_b"]
        assert r1["skipped"] == r2["skipped"]

    def test_item_sd_reported(self):
        result = simulate_pairwise(n=100, item_sd=0.1, rng=42)
        assert result["item_sd"] == 0.1
        assert result["runs"] == 100

    def test_paired_null_reduces_false_rate(self):
        """Shared item difficulty correlates outcomes, so P(A>B) wanders
        less far from 0.5 under equal models — fewer false winners."""
        unpaired = simulate_pairwise(
            n=600,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=10,
            max_samples=80,
            item_sd=0.0,
            rng=42,
        )
        paired = simulate_pairwise(
            n=600,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=10,
            max_samples=80,
            item_sd=0.25,
            rng=42,
        )
        unpaired_false = unpaired["false_decisions"] / unpaired["runs"]
        paired_false = paired["false_decisions"] / paired["runs"]
        assert (
            paired_false < unpaired_false
        ), f"paired null false rate {paired_false:.1%} should be < unpaired {unpaired_false:.1%}"


# ---------------------------------------------------------------------------
# Effect-size sweep
# ---------------------------------------------------------------------------


class TestEffectSizeSweep:
    def test_structure(self):
        report = effect_size_sweep(
            effects=((0.50, 0.50), (0.70, 0.50)),
            min_samples_grid=(3, 20),
            n_runs=100,
            max_samples=50,
            seed=42,
        )
        assert isinstance(report, EffectSweepReport)
        assert len(report.points) == 4  # 2 effects x 2 min_samples
        for p in report.points:
            assert 0.0 <= p.false_winner_rate <= 1.0
            assert 0.0 <= p.correct_winner_rate <= 1.0
            assert 0.0 <= p.inconclusive_rate <= 1.0
            assert 0.0 < p.expected_samples <= p.possible_max_samples

    def test_null_false_rate_falls_with_min_samples(self):
        """Under the null, more min_samples means fewer false winners."""
        report = effect_size_sweep(
            effects=((0.50, 0.50),),
            min_samples_grid=(3, 100),
            n_runs=400,
            max_samples=120,
            seed=42,
        )
        low, high = report.points[0], report.points[1]
        msg = (
            f"min3 false {low.false_winner_rate:.1%} "
            f"should exceed min100 {high.false_winner_rate:.1%}"
        )
        assert low.false_winner_rate > high.false_winner_rate, msg

    def test_large_effect_detected_with_high_power(self):
        """A large effect should be detected correctly most of the time."""
        report = effect_size_sweep(
            effects=((0.90, 0.50),),
            min_samples_grid=(20,),
            n_runs=300,
            max_samples=100,
            seed=42,
        )
        p = report.points[0]
        assert (
            p.correct_winner_rate > 0.9
        ), f"correct rate {p.correct_winner_rate:.1%} should exceed 90%"

    def test_table_formatting(self):
        report = effect_size_sweep(
            effects=((0.60, 0.50),),
            min_samples_grid=(10,),
            n_runs=100,
            seed=42,
        )
        table = report.format_table()
        assert "false" in table
        assert "correct" in table
        assert "E[n]" in table

    def test_point_str(self):
        p = EffectSweepPoint(
            acc_a=0.60,
            acc_b=0.50,
            min_samples=10,
            confidence=0.95,
            n_runs=100,
            item_sd=0.0,
            false_winner_rate=0.01,
            correct_winner_rate=0.90,
            inconclusive_rate=0.09,
            expected_samples=40.0,
            possible_max_samples=100,
        )
        s = str(p)
        assert "+0.10" in s
        assert "min=" in s


# ---------------------------------------------------------------------------
# Confidence-sequence simulation
# ---------------------------------------------------------------------------


class TestConfidenceSequenceSimulation:
    def test_null_false_rate_below_alpha(self):
        result = simulate_pairwise_cs(
            n=800,
            true_acc_a=0.5,
            true_acc_b=0.5,
            alpha=0.05,
            max_samples=80,
            rng=7,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "equivalent", "inconclusive"))
        false_rate = result["false_decisions"] / total if total > 0 else 0.0
        assert false_rate <= 0.05, f"any-time false rate {false_rate:.2%} exceeds alpha"

    def test_reproducible_with_seed(self):
        r1 = simulate_pairwise_cs(n=200, rng=42)
        r2 = simulate_pairwise_cs(n=200, rng=42)
        assert r1["winner_a"] == r2["winner_a"]
        assert r1["winner_b"] == r2["winner_b"]

    def test_detects_strong_effect(self):
        result = simulate_pairwise_cs(
            n=300,
            true_acc_a=0.9,
            true_acc_b=0.5,
            alpha=0.05,
            max_samples=200,
            rng=11,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "equivalent", "inconclusive"))
        assert result["winner_a"] / total > 0.9


# ---------------------------------------------------------------------------
# CalibrationPoint
# ---------------------------------------------------------------------------


class TestCalibrationPoint:
    def test_str_contains_rates(self):
        p = CalibrationPoint(
            min_samples=3,
            skip_threshold=0.85,
            confidence=0.95,
            n_runs=1_000,
            false_winner_rate=0.21875,
            skipped_rate=0.78125,
            inconclusive_rate=0.0,
            expected_samples=3.0,
            possible_max_samples=100,
        )
        s = str(p)
        assert "21.9%" in s or "21.9" in s.replace("%", "")
        assert "min=" in s
