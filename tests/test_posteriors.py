"""Tests for baysbench.posteriors."""

import numpy as np
import pytest

from baysbench.posteriors import BetaPosterior, NormalPosterior, Posterior

# ---------------------------------------------------------------------------
# Posterior protocol conformance
# ---------------------------------------------------------------------------


class TestPosteriorProtocol:
    @pytest.mark.parametrize("cls", [BetaPosterior, NormalPosterior])
    def test_is_posterior_subclass(self, cls):
        assert issubclass(cls, Posterior)

    @pytest.mark.parametrize("cls", [BetaPosterior, NormalPosterior])
    def test_has_required_methods(self, cls):
        p = cls()
        assert hasattr(p, "observe_one")
        assert hasattr(p, "prob_beats")
        assert hasattr(p, "credible_interval")
        assert hasattr(p, "mean")


# ---------------------------------------------------------------------------
# BetaPosterior
# ---------------------------------------------------------------------------


class TestBetaPosterior:
    def test_jeffreys_defaults(self):
        p = BetaPosterior()
        assert p.alpha == 0.5
        assert p.beta == 0.5

    def test_observe_one_true(self):
        p = BetaPosterior()
        p.observe_one(True)
        assert p.alpha == 1.5

    def test_observe_one_false(self):
        p = BetaPosterior()
        p.observe_one(False)
        assert p.beta == 1.5

    def test_observe_immutable(self):
        p = BetaPosterior()
        p2 = p.observe(True)
        assert p.alpha == 0.5
        assert p2.alpha == 1.5

    def test_observe_batch(self):
        p = BetaPosterior()
        p.observe_batch(7, 10)
        assert p.alpha == pytest.approx(7.5)
        assert p.beta == pytest.approx(3.5)

    def test_mean_at_prior(self):
        assert BetaPosterior().mean == pytest.approx(0.5)

    def test_mean_tracks_data(self):
        p = BetaPosterior()
        p.observe_batch(90, 100)
        assert p.mean > 0.88

    def test_credible_interval_valid(self):
        p = BetaPosterior()
        lo, hi = p.credible_interval()
        assert 0.0 <= lo < hi <= 1.0

    def test_ci_narrows_with_more_data(self):
        few = BetaPosterior()
        few.observe_batch(5, 10)
        many = BetaPosterior()
        many.observe_batch(50, 100)
        w_few = few.credible_interval()[1] - few.credible_interval()[0]
        w_many = many.credible_interval()[1] - many.credible_interval()[0]
        assert w_few > w_many

    def test_prob_beats_symmetric(self):
        p = BetaPosterior(10, 5)
        assert p.prob_beats(p) == pytest.approx(0.5, abs=0.02)

    def test_prob_beats_strong(self):
        strong = BetaPosterior(50, 5)
        weak = BetaPosterior(5, 50)
        assert strong.prob_beats(weak) > 0.99

    def test_sample_shape(self):
        p = BetaPosterior(5, 5)
        s = p.sample(100)
        assert s.shape == (100,)
        assert np.all((s >= 0) & (s <= 1))


# ---------------------------------------------------------------------------
# NormalPosterior
# ---------------------------------------------------------------------------


class TestNormalPosterior:
    def test_prior_mean(self):
        p = NormalPosterior(mu_0=0.5)
        assert p.mean == pytest.approx(0.5)

    def test_observe_one_updates_mean(self):
        p = NormalPosterior(mu_0=0.5)
        for _ in range(20):
            p.observe_one(0.9)
        assert p.mean > 0.7

    def test_observe_one_float(self):
        p = NormalPosterior()
        p.observe_one(0.42)
        assert p._n == 1

    def test_observe_batch(self):
        p = NormalPosterior()
        p.observe_batch([0.8, 0.7, 0.9])
        assert p._n == 3

    def test_credible_interval_valid(self):
        p = NormalPosterior()
        p.observe_one(0.5)
        lo, hi = p.credible_interval()
        assert lo < hi

    def test_ci_narrows_with_more_data(self):
        few = NormalPosterior()
        for _ in range(5):
            few.observe_one(0.7)
        many = NormalPosterior()
        for _ in range(50):
            many.observe_one(0.7)
        w_few = few.credible_interval()[1] - few.credible_interval()[0]
        w_many = many.credible_interval()[1] - many.credible_interval()[0]
        assert w_few > w_many

    def test_prob_beats_near_equal(self):
        a = NormalPosterior(mu_0=0.5)
        for _ in range(30):
            a.observe_one(0.7)
        b = NormalPosterior(mu_0=0.5)
        for _ in range(30):
            b.observe_one(0.7)
        assert a.prob_beats(b) == pytest.approx(0.5, abs=0.15)

    def test_prob_beats_clearly_better(self):
        np.random.seed(0)
        good = NormalPosterior(mu_0=0.5)
        for _ in range(50):
            good.observe_one(0.9)
        bad = NormalPosterior(mu_0=0.5)
        for _ in range(50):
            bad.observe_one(0.4)
        assert good.prob_beats(bad) > 0.95

    def test_n_property(self):
        p = NormalPosterior()
        p.observe_batch([0.5, 0.6, 0.7])
        assert p.n == 3.0

    def test_sample_shape(self):
        p = NormalPosterior()
        [p.observe_one(0.6) for _ in range(10)]
        s = p.sample(200)
        assert s.shape == (200,)

    def test_custom_prior_mu(self):
        p = NormalPosterior(mu_0=0.3)
        assert p.mean == pytest.approx(0.3)
