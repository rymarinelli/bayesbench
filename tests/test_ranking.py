"""Tests for baysbench.ranking."""
import asyncio

import numpy as np
import pytest

from baysbench import BayesianRanker, RankingResult
from baysbench.posteriors import NormalPosterior
from baysbench.ranking import ModelRanking

PROBLEMS = [{"q": str(i), "a": str(i)} for i in range(200)]


def perfect_model(problem):
    return problem["a"]


def weak_model(problem):
    # 60% accuracy
    return problem["a"] if int(problem["a"]) % 5 != 0 else "WRONG"


def terrible_model(problem):
    return "WRONG"


def score(problem, response):
    return response == problem["a"]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_add_model_chainable(self):
        ranker = BayesianRanker()
        result = ranker.add_model("a", perfect_model)
        assert result is ranker

    def test_add_multiple_models(self):
        ranker = BayesianRanker()
        ranker.add_model("a", perfect_model).add_model("b", weak_model)
        assert len(ranker._models) == 2

    def test_evaluate_decorator_stores_fn(self):
        ranker = BayesianRanker()

        @ranker.evaluate
        def my_score(problem, response):
            return response == problem["a"]

        # _score_fn holds the original (unwrapped) function
        assert ranker._score_fn is not None
        assert callable(ranker._score_fn)
        # The decorated name is preserved via functools.wraps
        assert my_score.__name__ == "my_score"

    def test_evaluate_preserves_function(self):
        ranker = BayesianRanker()

        @ranker.evaluate
        def fn(problem, response):
            return True

        assert fn({"a": "x"}, "x") is True

    def test_rank_no_models_raises(self):
        ranker = BayesianRanker()
        with pytest.raises(ValueError, match="No models"):
            ranker.rank(PROBLEMS, score_fn=score)

    def test_rank_no_score_fn_raises(self):
        ranker = BayesianRanker()
        ranker.add_model("a", perfect_model)
        with pytest.raises(ValueError, match="No score_fn"):
            ranker.rank(PROBLEMS)


# ---------------------------------------------------------------------------
# Ranking correctness
# ---------------------------------------------------------------------------


class TestRankingCorrectness:
    def test_returns_ranking_result(self):
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        assert isinstance(result, RankingResult)

    def test_best_model_ranked_first(self):
        np.random.seed(42)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        assert result.rankings[0].name == "perfect"

    def test_three_model_order(self):
        np.random.seed(0)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("weak", weak_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        names = [r.name for r in result.rankings]
        assert names.index("perfect") < names.index("weak")
        assert names.index("weak") < names.index("terrible")

    def test_early_stopping_saves_budget(self):
        np.random.seed(1)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        assert result.problems_tested < len(PROBLEMS)
        assert result.efficiency > 0.0

    def test_converged_flag(self):
        np.random.seed(2)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        assert result.converged is True

    def test_best_property(self):
        np.random.seed(3)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        assert result.best.name == "perfect"

    def test_summary_contains_all_models(self):
        np.random.seed(4)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        summary = result.summary()
        assert "perfect" in summary
        assert "terrible" in summary

    def test_p_beats_next_populated(self):
        np.random.seed(5)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = ranker.rank(PROBLEMS, score_fn=score)
        # First-ranked should have p_beats_next
        assert result.rankings[0].p_beats_next is not None
        # Last-ranked has no p_beats_next
        assert result.rankings[-1].p_beats_next is None


# ---------------------------------------------------------------------------
# Custom posterior
# ---------------------------------------------------------------------------


class TestRankingWithNormalPosterior:
    def test_continuous_scores_no_models_raises(self):
        ranker = BayesianRanker(
            confidence=0.95,
            min_samples=5,
            posterior_factory=NormalPosterior,
        )
        with pytest.raises(ValueError, match="No models"):
            ranker.rank(PROBLEMS, score_fn=lambda p, r: float(r))

    def test_normal_posterior_ranking(self):
        np.random.seed(11)
        ranker = BayesianRanker(
            confidence=0.95,
            min_samples=5,
            posterior_factory=NormalPosterior,
        )

        def high_score_model(problem):
            return "0.9"

        def low_score_model(problem):
            return "0.3"

        ranker.add_model("high", high_score_model)
        ranker.add_model("low", low_score_model)
        result = ranker.rank(
            PROBLEMS,
            score_fn=lambda p, r: float(r),
        )
        assert result.rankings[0].name == "high"


# ---------------------------------------------------------------------------
# evaluate decorator + rank
# ---------------------------------------------------------------------------


class TestEvaluateDecorator:
    def test_evaluate_used_by_rank(self):
        np.random.seed(20)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)

        @ranker.evaluate
        def exact(problem, response):
            return response == problem["a"]

        result = ranker.rank(PROBLEMS)  # no score_fn kwarg
        assert result.rankings[0].name == "perfect"


# ---------------------------------------------------------------------------
# Async ranking
# ---------------------------------------------------------------------------


class TestAsyncRanking:
    @pytest.mark.asyncio
    async def test_async_rank(self):
        np.random.seed(30)
        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", perfect_model)
        ranker.add_model("terrible", terrible_model)
        result = await ranker.rank_async(PROBLEMS, score_fn=score)
        assert result.rankings[0].name == "perfect"

    @pytest.mark.asyncio
    async def test_async_rank_async_models(self):
        np.random.seed(31)

        async def async_perfect(problem):
            return problem["a"]

        async def async_terrible(problem):
            return "WRONG"

        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("perfect", async_perfect)
        ranker.add_model("terrible", async_terrible)
        result = await ranker.rank_async(PROBLEMS, score_fn=score)
        assert result.rankings[0].name == "perfect"
