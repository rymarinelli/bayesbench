"""Tests for DirichletPosterior, GammaPosterior, and the convenience API."""
import pytest

from baysbench.posteriors import DirichletPosterior, GammaPosterior, Posterior


# ---------------------------------------------------------------------------
# DirichletPosterior
# ---------------------------------------------------------------------------


class TestDirichletPosterior:
    def test_is_posterior_subclass(self):
        assert issubclass(DirichletPosterior, Posterior)

    def test_has_required_methods(self):
        p = DirichletPosterior()
        assert hasattr(p, "observe_one")
        assert hasattr(p, "prob_beats")
        assert hasattr(p, "credible_interval")
        assert hasattr(p, "mean")

    def test_defaults_k2(self):
        p = DirichletPosterior()
        assert p._k == 2
        assert p._alpha_0 == 0.5
        assert p._target_class == 0

    def test_observe_bool_true_increments_class0(self):
        p = DirichletPosterior(k=2)
        p.observe_one(True)
        assert p._counts[0] == 1
        assert p._counts[1] == 0

    def test_observe_bool_false_increments_class1(self):
        p = DirichletPosterior(k=2)
        p.observe_one(False)
        assert p._counts[0] == 0
        assert p._counts[1] == 1

    def test_observe_int_category(self):
        p = DirichletPosterior(k=4)
        p.observe_one(2)
        assert p._counts[2] == 1

    def test_observe_out_of_range_raises(self):
        p = DirichletPosterior(k=3)
        with pytest.raises(ValueError, match="out of range"):
            p.observe_one(5)

    def test_mean_increases_with_correct_observations(self):
        p = DirichletPosterior(k=4)
        prior_mean = p.mean
        for _ in range(20):
            p.observe_one(0)  # always correct (category 0)
        assert p.mean > prior_mean

    def test_mean_at_prior(self):
        # With k=4 and symmetric prior, mean of each class = 1/4 = 0.25
        p = DirichletPosterior(k=4, alpha_0=1.0)
        assert p.mean == pytest.approx(0.25)

    def test_mean_at_prior_k2(self):
        # k=2, symmetric: mean = 0.5
        p = DirichletPosterior(k=2, alpha_0=0.5)
        assert p.mean == pytest.approx(0.5)

    def test_prob_beats_symmetric_is_near_half(self):
        p = DirichletPosterior(k=2)
        q = DirichletPosterior(k=2)
        assert 0.4 <= p.prob_beats(q) <= 0.6

    def test_prob_beats_better_model_wins(self):
        p = DirichletPosterior(k=2)
        q = DirichletPosterior(k=2)
        for _ in range(50):
            p.observe_one(True)   # 50 correct
        for _ in range(50):
            q.observe_one(False)  # 50 wrong
        assert p.prob_beats(q) > 0.99

    def test_prob_beats_wrong_type_raises(self):
        from baysbench.posteriors import BetaPosterior

        p = DirichletPosterior(k=2)
        with pytest.raises(TypeError):
            p.prob_beats(BetaPosterior())

    def test_credible_interval_contains_mean(self):
        p = DirichletPosterior(k=2)
        for _ in range(10):
            p.observe_one(True)
        lo, hi = p.credible_interval()
        assert lo < p.mean < hi

    def test_credible_interval_width_shrinks_with_data(self):
        p_few = DirichletPosterior(k=2)
        p_many = DirichletPosterior(k=2)
        for _ in range(5):
            p_few.observe_one(True)
        for _ in range(100):
            p_many.observe_one(True)
        width_few = p_few.credible_interval()[1] - p_few.credible_interval()[0]
        width_many = p_many.credible_interval()[1] - p_many.credible_interval()[0]
        assert width_many < width_few

    def test_n_tracks_observations(self):
        p = DirichletPosterior(k=3)
        assert p.n == 0
        p.observe_one(1)
        p.observe_one(2)
        assert p.n == 2

    def test_sample_shape(self):
        p = DirichletPosterior(k=4)
        samples = p.sample(n=100)
        assert samples.shape == (100, 4)

    def test_sample_rows_sum_to_one(self):
        import numpy as np

        p = DirichletPosterior(k=3)
        samples = p.sample(n=50)
        assert all(abs(row.sum() - 1.0) < 1e-10 for row in samples)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be >= 2"):
            DirichletPosterior(k=1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_0 must be positive"):
            DirichletPosterior(alpha_0=0)

    def test_invalid_target_class_raises(self):
        with pytest.raises(ValueError, match="target_class"):
            DirichletPosterior(k=3, target_class=5)

    def test_target_class_parameter(self):
        p = DirichletPosterior(k=3, target_class=2)
        for _ in range(20):
            p.observe_one(2)  # always class 2
        assert p.mean > 0.5  # class 2 dominates

    def test_k2_equivalent_behavior_to_beta(self):
        """k=2 DirichletPosterior should behave like BetaPosterior(0.5, 0.5)."""
        from baysbench.posteriors import BetaPosterior

        dp = DirichletPosterior(k=2, alpha_0=0.5)
        bp = BetaPosterior(alpha=0.5, beta=0.5)
        obs = [True, False, True, True, False]
        for v in obs:
            dp.observe_one(v)
            bp.observe_one(v)
        assert dp.mean == pytest.approx(bp.mean, abs=1e-10)


# ---------------------------------------------------------------------------
# GammaPosterior
# ---------------------------------------------------------------------------


class TestGammaPosterior:
    def test_is_posterior_subclass(self):
        assert issubclass(GammaPosterior, Posterior)

    def test_has_required_methods(self):
        p = GammaPosterior()
        assert hasattr(p, "observe_one")
        assert hasattr(p, "prob_beats")
        assert hasattr(p, "credible_interval")
        assert hasattr(p, "mean")

    def test_defaults(self):
        p = GammaPosterior()
        assert p._alpha_0 == 1.0
        assert p._beta_0 == 1.0
        assert p.higher_is_better is True

    def test_mean_at_prior(self):
        p = GammaPosterior(alpha_0=2.0, beta_0=4.0)
        assert p.mean == pytest.approx(0.5)

    def test_observe_updates_mean(self):
        p = GammaPosterior(alpha_0=1.0, beta_0=1.0)
        p.observe_one(10)
        # alpha = 1 + 10 = 11, beta = 1 + 1 = 2, mean = 5.5
        assert p.mean == pytest.approx(5.5)

    def test_observe_multiple(self):
        p = GammaPosterior(alpha_0=1.0, beta_0=1.0)
        p.observe_one(4)
        p.observe_one(6)
        # alpha = 1 + 4 + 6 = 11, beta = 1 + 2 = 3, mean ≈ 3.67
        assert p.mean == pytest.approx(11.0 / 3.0)

    def test_negative_observation_raises(self):
        p = GammaPosterior()
        with pytest.raises(ValueError, match="non-negative"):
            p.observe_one(-1)

    def test_n_tracks_observations(self):
        p = GammaPosterior()
        assert p.n == 0
        p.observe_one(5)
        p.observe_one(3)
        assert p.n == 2

    def test_prob_beats_higher_is_better(self):
        # Model A has higher rate
        a = GammaPosterior(higher_is_better=True)
        b = GammaPosterior(higher_is_better=True)
        for _ in range(30):
            a.observe_one(100)
        for _ in range(30):
            b.observe_one(10)
        assert a.prob_beats(b) > 0.99

    def test_prob_beats_lower_is_better(self):
        # Model A has lower rate (lower latency = better)
        a = GammaPosterior(higher_is_better=False)
        b = GammaPosterior(higher_is_better=False)
        for _ in range(30):
            a.observe_one(10)   # fast
        for _ in range(30):
            b.observe_one(100)  # slow
        assert a.prob_beats(b) > 0.99

    def test_prob_beats_symmetric_near_half(self):
        a = GammaPosterior()
        b = GammaPosterior()
        assert 0.3 <= a.prob_beats(b) <= 0.7

    def test_prob_beats_wrong_type_raises(self):
        from baysbench.posteriors import BetaPosterior

        p = GammaPosterior()
        with pytest.raises(TypeError):
            p.prob_beats(BetaPosterior())

    def test_credible_interval_contains_mean(self):
        p = GammaPosterior()
        for i in range(10):
            p.observe_one(float(i + 1))
        lo, hi = p.credible_interval()
        assert lo < p.mean < hi

    def test_credible_interval_shrinks_with_data(self):
        p_few = GammaPosterior()
        p_many = GammaPosterior()
        for _ in range(3):
            p_few.observe_one(5.0)
        for _ in range(100):
            p_many.observe_one(5.0)
        w_few = p_few.credible_interval()[1] - p_few.credible_interval()[0]
        w_many = p_many.credible_interval()[1] - p_many.credible_interval()[0]
        assert w_many < w_few

    def test_sample_shape(self):
        p = GammaPosterior()
        for _ in range(5):
            p.observe_one(3.0)
        samples = p.sample(n=50)
        assert samples.shape == (50,)

    def test_sample_positive(self):
        p = GammaPosterior()
        for _ in range(5):
            p.observe_one(5.0)
        samples = p.sample(n=100)
        assert all(s > 0 for s in samples)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha_0 must be positive"):
            GammaPosterior(alpha_0=0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta_0 must be positive"):
            GammaPosterior(beta_0=-1)

    def test_bool_observation_coerced(self):
        p = GammaPosterior()
        p.observe_one(True)   # 1.0
        p.observe_one(False)  # 0.0
        assert p.n == 2


# ---------------------------------------------------------------------------
# Top-level convenience API: baysbench.compare and baysbench.rank
# ---------------------------------------------------------------------------


PROBLEMS = [{"q": str(i), "a": str(i)} for i in range(50)]


def perfect_model(problem):
    return problem["a"]


def wrong_model(problem):
    return "WRONG"


def score(problem, response):
    return response == problem["a"]


class TestConvenienceCompare:
    def test_compare_returns_task_result(self):
        import baysbench
        from baysbench.benchmark import TaskResult

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert isinstance(result, TaskResult)

    def test_compare_declares_winner(self):
        import baysbench

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert result.winner == "model_a"

    def test_compare_stops_early(self):
        import baysbench

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert result.problems_tested < len(PROBLEMS)

    def test_compare_custom_confidence(self):
        import baysbench

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
            confidence=0.99,
        )
        assert result.winner == "model_a"

    def test_compare_name_propagated(self):
        import baysbench

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
            name="my_task",
        )
        assert result.name == "my_task"

    def test_compare_to_dict(self):
        import baysbench

        result = baysbench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
        )
        d = result.to_dict()
        assert "winner" in d
        assert "p_a_beats_b" in d
        assert "mean_a" in d
        assert "mean_b" in d
        assert "ci_a_lo" in d
        assert "efficiency" in d

    def test_compare_with_normal_posterior(self):
        import baysbench
        from baysbench.posteriors import NormalPosterior

        problems = [{"a": 0.8} for _ in range(100)]

        def good_model(p):
            return 0.85

        def bad_model(p):
            return 0.05

        def score_fn(p, r):
            return r

        result = baysbench.compare(
            model_a=good_model,
            model_b=bad_model,
            score_fn=score_fn,
            dataset=problems,
            posterior_factory=NormalPosterior,
            skip_threshold=0.99,  # disable skipping for this test
        )
        assert result.winner == "model_a"


class TestConvenienceRank:
    def test_rank_returns_ranking_result(self):
        import baysbench
        from baysbench.ranking import RankingResult

        result = baysbench.rank(
            models={"perfect": perfect_model, "wrong": wrong_model},
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert isinstance(result, RankingResult)

    def test_rank_dict_input(self):
        import baysbench

        result = baysbench.rank(
            models={"perfect": perfect_model, "wrong": wrong_model},
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert result.best.name == "perfect"

    def test_rank_list_input(self):
        import baysbench

        result = baysbench.rank(
            models=[("perfect", perfect_model), ("wrong", wrong_model)],
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert result.best.name == "perfect"

    def test_rank_to_dict(self):
        import baysbench

        result = baysbench.rank(
            models={"perfect": perfect_model, "wrong": wrong_model},
            score_fn=score,
            dataset=PROBLEMS,
        )
        d = result.to_dict()
        assert "rankings" in d
        assert "efficiency" in d
        assert "converged" in d

    def test_rank_model_ranking_to_dict(self):
        import baysbench

        result = baysbench.rank(
            models={"perfect": perfect_model, "wrong": wrong_model},
            score_fn=score,
            dataset=PROBLEMS,
        )
        row = result.rankings[0].to_dict()
        assert "rank" in row
        assert "name" in row
        assert "mean" in row
        assert "ci_lo" in row
        assert "ci_hi" in row

    def test_rank_three_models(self):
        import baysbench

        def medium_model(problem):
            i = int(problem["q"])
            return problem["a"] if i % 2 == 0 else "WRONG"

        result = baysbench.rank(
            models={
                "perfect": perfect_model,
                "medium": medium_model,
                "wrong": wrong_model,
            },
            score_fn=score,
            dataset=PROBLEMS,
        )
        assert len(result.rankings) == 3
        assert result.rankings[0].name == "perfect"


# ---------------------------------------------------------------------------
# BenchmarkReport serialization
# ---------------------------------------------------------------------------


class TestBenchmarkReportSerialization:
    def test_to_dict_structure(self):
        from baysbench import BayesianBenchmark

        bench = BayesianBenchmark(confidence=0.95, min_samples=3)
        report = bench.compare(
            model_a=perfect_model,
            model_b=wrong_model,
            score_fn=score,
            dataset=PROBLEMS,
        )
        # Use the TaskResult.to_dict directly
        d = report.to_dict()
        assert "name" in d
        assert "winner" in d
        assert "efficiency" in d

    def test_benchmark_report_to_dict(self):
        from baysbench import BayesianBenchmark

        bench = BayesianBenchmark(confidence=0.95, min_samples=3)

        @bench.task(dataset=PROBLEMS)
        def task1(problem):
            return score(problem, perfect_model(problem)), score(problem, wrong_model(problem))

        report = bench.run()
        d = report.to_dict()
        assert "tasks" in d
        assert len(d["tasks"]) == 1
        assert "overall_efficiency" in d
        assert "winners" in d
