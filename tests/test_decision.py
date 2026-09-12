"""Tests for decision rules: statuses, ROPE equivalence, confidence sequences.

Validates:
- DecisionStatus semantics and legacy skip behaviour.
- PosteriorThresholdRule winner / equivalence / inconclusive paths.
- Ville-validity of ConfidenceSequenceRule (exact e-value property + any-time
  false-decision rate at or below alpha in simulation).
- PairedDifferenceRule for binary and continuous paired evaluation.
- prob_beats_value closed forms.
"""

import math

import numpy as np
import pytest

from bayesbench.decision import (
    ConfidenceSequenceRule,
    DecisionStatus,
    PairedDifferenceRule,
    PosteriorThresholdRule,
    SequentialDecision,
    _cs_bounds,
    _log_e_value,
    make_decision_rule,
)
from bayesbench.posteriors import BetaPosterior, NormalPosterior

# ---------------------------------------------------------------------------
# Status / dataclass basics
# ---------------------------------------------------------------------------


class TestSequentialDecision:
    def test_terminal_statuses(self):
        assert SequentialDecision(DecisionStatus.INCONCLUSIVE, 0.5).terminal is False
        assert SequentialDecision(DecisionStatus.WINNER_A, 0.97).terminal is True
        assert SequentialDecision(DecisionStatus.WINNER_B, 0.03).terminal is True
        assert SequentialDecision(DecisionStatus.EQUIVALENT, 0.5).terminal is True

    def test_str(self):
        s = str(SequentialDecision(DecisionStatus.WINNER_A, 0.96))
        assert "winner_a" in s and "0.96" in s


# ---------------------------------------------------------------------------
# Posterior threshold rule
# ---------------------------------------------------------------------------


class TestPosteriorThresholdRule:
    def test_winner_a_on_strong_stream(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=3)
        for _ in range(10):
            decision = rule.observe(True, False)
        assert decision.status is DecisionStatus.WINNER_A
        assert decision.p_a_beats_b >= 0.95

    def test_winner_b_on_strong_stream(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=3)
        for _ in range(10):
            decision = rule.observe(False, True)
        assert decision.status is DecisionStatus.WINNER_B

    def test_min_samples_gates_decisions(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=5)
        for _ in range(4):
            decision = rule.observe(True, False)
            assert decision.status is DecisionStatus.INCONCLUSIVE
        decision = rule.observe(True, False)
        assert decision.status is DecisionStatus.WINNER_A

    def test_legacy_skip_is_equivalent(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=3, skip_threshold=0.85)
        decision = rule.observe(True, False)
        decision = rule.observe(False, True)
        decision = rule.observe(True, False)
        assert decision.status is DecisionStatus.EQUIVALENT
        assert "skip" in decision.details

    def test_skip_disabled_by_default(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=3)
        rule.observe(True, False)
        rule.observe(False, True)
        decision = rule.observe(True, False)
        assert decision.status is not DecisionStatus.EQUIVALENT

    def test_rope_equivalence_for_equal_models(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=10, equivalence_margin=0.1)
        for _ in range(60):
            decision = rule.observe(True, True)
            if decision.terminal:
                break
        assert decision.status is DecisionStatus.EQUIVALENT

    def test_rope_does_not_fire_with_large_gap(self):
        rule = PosteriorThresholdRule(confidence=0.95, min_samples=10, equivalence_margin=0.05)
        for _ in range(60):
            decision = rule.observe(True, False)
            if decision.terminal:
                break
        assert decision.status is DecisionStatus.WINNER_A

    def test_rope_mass_is_plausible(self):
        rule = PosteriorThresholdRule(min_samples=1)
        rule.observe(True, True)
        mass = rule._rope_mass(0.5)
        assert 0.0 < mass <= 1.0

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            PosteriorThresholdRule(confidence=0.5)

    def test_invalid_skip_threshold(self):
        with pytest.raises(ValueError):
            PosteriorThresholdRule(skip_threshold=0.5)

    def test_last_decision_tracked(self):
        rule = PosteriorThresholdRule(min_samples=2)
        rule.observe(True, False)
        assert rule.last_decision is not None
        assert rule.last_decision.status is DecisionStatus.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Confidence sequence rule
# ---------------------------------------------------------------------------


class TestConfidenceSequence:
    def test_e_value_has_expectation_one_exactly(self):
        """E_theta[E_t(theta)] = sum_s Binom(t, theta)(s) * E_t(s) == 1."""
        t, theta = 40, 0.4
        total = sum(
            math.comb(t, s)
            * (theta**s)
            * ((1 - theta) ** (t - s))
            * math.exp(_log_e_value(s, t, theta))
            for s in range(t + 1)
        )
        assert total == pytest.approx(1.0, rel=1e-9)

    def test_cs_bounds_contain_mle(self):
        lo, hi = _cs_bounds(70, 100, 0.025)
        assert lo <= 0.7 <= hi
        assert lo > 0.5  # 70/100 excludes small thetas at this level

    def test_cs_bounds_boundary_cases(self):
        lo, hi = _cs_bounds(0, 50, 0.025)
        assert lo == 0.0 and hi < 1.0
        lo, hi = _cs_bounds(50, 50, 0.025)
        assert lo > 0.0 and hi == 1.0
        lo, hi = _cs_bounds(0, 0, 0.025)
        assert (lo, hi) == (0.0, 1.0)

    def test_cs_bounds_nested_in_posterior_scale(self):
        """Bounds at higher confidence (smaller alpha) are wider."""
        lo_tight, hi_tight = _cs_bounds(30, 60, 0.05)
        lo_wide, hi_wide = _cs_bounds(30, 60, 0.005)
        assert lo_wide <= lo_tight and hi_wide >= hi_tight

    def test_any_time_false_rate_below_alpha_under_null(self):
        """Under equal Bernoulli models the CS rule must declare a wrong
        winner in at most alpha of runs at ANY stopping time (Ville)."""
        rng = np.random.default_rng(7)
        n_runs, max_t = 800, 60
        false = 0
        for _ in range(n_runs):
            rule = ConfidenceSequenceRule(alpha=0.05)
            for _ in range(max_t):
                decision = rule.observe(bool(rng.random() < 0.5), bool(rng.random() < 0.5))
                if decision.terminal:
                    false += 1
                    break
        assert false / n_runs <= 0.05, f"any-time false rate {false / n_runs:.2%}"

    def test_detects_large_effect(self):
        rng = np.random.default_rng(11)
        n_runs, max_t = 200, 200
        wins = 0
        for _ in range(n_runs):
            rule = ConfidenceSequenceRule(alpha=0.05)
            for _ in range(max_t):
                decision = rule.observe(bool(rng.random() < 0.9), bool(rng.random() < 0.5))
                if decision.terminal:
                    wins += decision.status is DecisionStatus.WINNER_A
                    break
        assert wins / n_runs > 0.9, f"power {wins / n_runs:.1%} too low for d=0.4"

    def test_rejects_continuous_values(self):
        rule = ConfidenceSequenceRule(alpha=0.05)
        with pytest.raises(TypeError):
            rule.observe(0.7, 0.3)

    def test_rope_equivalence(self):
        rule = ConfidenceSequenceRule(alpha=0.05, equivalence_margin=0.1)
        for _ in range(100):
            decision = rule.observe(True, True)
            if decision.terminal:
                break
        assert decision.status is DecisionStatus.EQUIVALENT

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ConfidenceSequenceRule(alpha=1.5)


# ---------------------------------------------------------------------------
# Paired difference rule
# ---------------------------------------------------------------------------


class TestPairedDifferenceRule:
    def test_binary_discordant_pairs_drive_decision(self):
        rule = PairedDifferenceRule(confidence=0.95, min_samples=3)
        for _ in range(10):
            decision = rule.observe(True, False)
        assert decision.status is DecisionStatus.WINNER_A
        assert decision.p_a_beats_b >= 0.95

    def test_binary_ties_do_not_update_posterior(self):
        rule = PairedDifferenceRule(confidence=0.95, min_samples=3)
        for _ in range(50):
            decision = rule.observe(True, True)
        assert decision.status is DecisionStatus.INCONCLUSIVE
        assert rule.post.n == 0

    def test_continuous_difference_wins(self):
        rule = PairedDifferenceRule(
            confidence=0.95, min_samples=5, posterior_factory=NormalPosterior
        )
        for _ in range(30):
            decision = rule.observe(0.8, 0.3)
        assert decision.status is DecisionStatus.WINNER_A

    def test_continuous_equal_scores_equivalent(self):
        rule = PairedDifferenceRule(
            confidence=0.95,
            min_samples=10,
            equivalence_margin=0.1,
            posterior_factory=NormalPosterior,
        )
        for _ in range(40):
            decision = rule.observe(0.55, 0.55)
            if decision.terminal:
                break
        assert decision.status is DecisionStatus.EQUIVALENT

    def test_binary_equal_solvers_equivalent_via_rope(self):
        """Binary paired ROPE is measured around the 0.5 discordant center."""
        rule = PairedDifferenceRule(confidence=0.95, min_samples=50, equivalence_margin=0.1)
        rng = np.random.default_rng(0)
        for _ in range(300):
            a = bool(rng.random() < 0.55)
            b = bool(rng.random() < 0.55)
            decision = rule.observe(a, b)
            if decision.terminal:
                break
        assert decision.status is DecisionStatus.EQUIVALENT


# ---------------------------------------------------------------------------
# prob_beats_value closed forms
# ---------------------------------------------------------------------------


class TestProbBeatsValue:
    def test_beta_closed_form(self):
        from scipy import stats

        p = BetaPosterior()
        for _ in range(10):
            p.observe_one(True)
        expected = 1.0 - stats.beta.cdf(0.5, p.alpha, p.beta)
        assert p.prob_beats_value(0.5) == pytest.approx(expected, abs=1e-9)

    def test_beta_matches_cdf(self):
        from scipy import stats

        p = BetaPosterior(3.0, 2.0)
        assert p.prob_beats_value(0.4) == pytest.approx(1.0 - stats.beta.cdf(0.4, 3.0, 2.0))

    def test_normal_closed_form(self):
        p = NormalPosterior()
        for _ in range(20):
            p.observe_one(0.7)
        prob = p.prob_beats_value(0.5)
        assert prob > 0.95

    def test_base_default_mc(self):
        from bayesbench.posteriors.base import Posterior

        class Minimal(Posterior):
            def __init__(self) -> None:
                self._rng = np.random.default_rng(0)

            def observe_one(self, value: float | bool) -> None:
                return None

            def prob_beats(self, other: Posterior, n_samples: int = 10_000) -> float:
                return 0.5

            def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
                return (0.0, 1.0)

            @property
            def mean(self) -> float:
                return 0.5

            def sample(self, n: int = 1) -> np.ndarray:
                return self._rng.uniform(0.0, 1.0, size=n)

        p = Minimal()
        assert p.prob_beats_value(0.5) == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestMakeDecisionRule:
    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            make_decision_rule("bogus")

    def test_factory_builds_each_kind(self):
        posterior = make_decision_rule("posterior")
        assert isinstance(posterior, PosteriorThresholdRule)
        cs = make_decision_rule("confidence_sequence")
        assert isinstance(cs, ConfidenceSequenceRule)
        paired = make_decision_rule("posterior", paired=True)
        assert isinstance(paired, PairedDifferenceRule)
