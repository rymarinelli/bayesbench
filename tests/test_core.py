"""Tests for bayesbench.core."""

import pytest

from bayesbench.core import BetaPosterior, is_non_discriminating, prob_a_beats_b


class TestBetaPosterior:
    def test_jeffreys_prior_defaults(self):
        p = BetaPosterior()
        assert p.alpha == 0.5
        assert p.beta == 0.5

    def test_mean_at_prior(self):
        p = BetaPosterior()
        assert p.mean == pytest.approx(0.5)

    def test_observe_one_success(self):
        p = BetaPosterior()
        p.observe_one(True)
        assert p.alpha == 1.5
        assert p.beta == 0.5

    def test_observe_one_failure(self):
        p = BetaPosterior()
        p.observe_one(False)
        assert p.alpha == 0.5
        assert p.beta == 1.5

    def test_observe_returns_new_instance(self):
        p = BetaPosterior()
        p2 = p.observe(True)
        assert p.alpha == 0.5  # original unchanged
        assert p2.alpha == 1.5

    def test_observe_batch(self):
        p = BetaPosterior()
        p.observe_batch(successes=7, total=10)
        assert p.alpha == pytest.approx(7.5)
        assert p.beta == pytest.approx(3.5)

    def test_mean_after_observations(self):
        p = BetaPosterior(alpha=10.5, beta=0.5)
        assert p.mean == pytest.approx(10.5 / 11.0)

    def test_credible_interval_width(self):
        p = BetaPosterior()
        lo, hi = p.credible_interval(0.95)
        assert lo < hi
        assert 0.0 <= lo < 0.5
        assert 0.5 < hi <= 1.0

    def test_credible_interval_narrows_with_data(self):
        few = BetaPosterior()
        few.observe_batch(5, 10)
        many = BetaPosterior()
        many.observe_batch(50, 100)
        lo_f, hi_f = few.credible_interval()
        lo_m, hi_m = many.credible_interval()
        assert (hi_f - lo_f) > (hi_m - lo_m)

    def test_n_property(self):
        p = BetaPosterior()
        p.observe_batch(3, 5)
        assert p.n == pytest.approx(5.0)


class TestProbABeatsB:
    def test_symmetric_at_equal_posteriors(self):
        p = BetaPosterior(alpha=10, beta=5)
        assert prob_a_beats_b(p, p) == pytest.approx(0.5, abs=0.01)

    def test_strong_model_a(self):
        strong = BetaPosterior(alpha=50, beta=5)
        weak = BetaPosterior(alpha=5, beta=50)
        p = prob_a_beats_b(strong, weak)
        assert p > 0.99

    def test_strong_model_b(self):
        weak = BetaPosterior(alpha=5, beta=50)
        strong = BetaPosterior(alpha=50, beta=5)
        p = prob_a_beats_b(weak, strong)
        assert p < 0.01

    def test_result_in_unit_interval(self):
        a = BetaPosterior(alpha=3, beta=7)
        b = BetaPosterior(alpha=7, beta=3)
        p = prob_a_beats_b(a, b)
        assert 0.0 <= p <= 1.0

    def test_complementary(self):
        a = BetaPosterior(alpha=20, beta=10)
        b = BetaPosterior(alpha=10, beta=20)
        p_ab = prob_a_beats_b(a, b)
        p_ba = prob_a_beats_b(b, a)
        # P(A>B) + P(B>A) ≈ 1 (ignoring ties)
        assert p_ab + p_ba == pytest.approx(1.0, abs=0.02)


class TestIsNonDiscriminating:
    def test_equal_models_are_non_discriminating(self):
        p = BetaPosterior(alpha=10, beta=5)
        assert is_non_discriminating(p, p, threshold=0.85)

    def test_very_different_models_are_discriminating(self):
        strong = BetaPosterior(alpha=50, beta=2)
        weak = BetaPosterior(alpha=2, beta=50)
        assert not is_non_discriminating(strong, weak, threshold=0.85)

    def test_threshold_respected(self):
        a = BetaPosterior(alpha=12, beta=8)
        b = BetaPosterior(alpha=8, beta=12)
        # With a tight threshold, these may look discriminating
        assert not is_non_discriminating(a, b, threshold=0.6)
