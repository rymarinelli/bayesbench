"""BayesianRanker: rank N models simultaneously with early stopping.

Instead of running O(N²) pairwise comparisons, the ranker evaluates all
models on every problem *together*, maintains one posterior per model, and
stops as soon as the ranking order is statistically clear: every consecutive
ranked pair (rank k, rank k+1) satisfies P(k > k+1) ≥ confidence.

Usage::

    from bayesbench import BayesianRanker

    ranker = BayesianRanker(confidence=0.95, min_samples=5)
    ranker.add_model("gpt-4o",       gpt4_fn)
    ranker.add_model("llama-3-8b",   llama_fn)
    ranker.add_model("gpt-4o-mini",  mini_fn)

    # Option A: pass score_fn separately
    result = ranker.rank(
        dataset=problems,
        score_fn=lambda p, r: r.strip() == p["answer"],
    )
    print(result.summary())

    # Option B: decorate the scoring function
    @ranker.evaluate
    def exact_match(problem, response):
        return response.strip() == problem["answer"]

    result = ranker.rank(dataset=problems)
    print(result.summary())

    # Async
    result = await ranker.rank_async(dataset=problems, score_fn=score)
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .posteriors.base import Posterior
from .posteriors.beta import BetaPosterior

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ModelRanking:
    """Statistics for a single model within a :class:`RankingResult`."""

    rank: int
    name: str
    posterior: Posterior
    p_beats_next: float | None  # P(this > next ranked model), None if last

    @property
    def mean(self) -> float:
        return self.posterior.mean

    @property
    def credible_interval(self) -> tuple[float, float]:
        return self.posterior.credible_interval()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (no posterior object).

        Returns rank, name, posterior mean, 95% credible interval, and
        the probability of beating the next-ranked model.
        """
        lo, hi = self.credible_interval
        return {
            "rank": self.rank,
            "name": self.name,
            "mean": self.mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "p_beats_next": self.p_beats_next,
            "next_model": self._next_name if self.p_beats_next is not None else None,
        }

    def __str__(self) -> str:
        lo, hi = self.credible_interval
        p_str = (
            f"  P(>{self._next_name})={self.p_beats_next:.3f}"
            if self.p_beats_next is not None
            else ""
        )
        return (
            f"#{self.rank} {self.name:20s}  "
            f"score={self.mean:.3f}  "
            f"95%CI=[{lo:.3f},{hi:.3f}]{p_str}"
        )

    # Set by RankingResult after construction
    _next_name: str = field(default="next", repr=False)


@dataclass
class RankingResult:
    """Result of a multi-model ranking run."""

    rankings: list[ModelRanking]
    problems_tested: int
    total_problems: int
    converged: bool  # True if stopping criterion was met before exhausting data

    @property
    def efficiency(self) -> float:
        if self.total_problems == 0:
            return 0.0
        return 1.0 - self.problems_tested / self.total_problems

    @property
    def best(self) -> ModelRanking:
        """The top-ranked model."""
        return self.rankings[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ranking result to a plain dict.

        Example::

            import json
            print(json.dumps(result.to_dict(), indent=2))
        """
        return {
            "rankings": [r.to_dict() for r in self.rankings],
            "problems_tested": self.problems_tested,
            "total_problems": self.total_problems,
            "efficiency": self.efficiency,
            "converged": self.converged,
        }

    def to_dataframe(self) -> Any:
        """Return a :class:`pandas.DataFrame` with one row per model.

        Requires ``pandas`` (``pip install pandas``).

        Columns: ``rank``, ``name``, ``mean``, ``ci_lo``, ``ci_hi``,
        ``p_beats_next``, ``next_model``.

        Example::

            df = ranker.rank(dataset=problems, score_fn=score).to_dataframe()
            print(df.to_string())
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            ) from exc
        return pd.DataFrame([r.to_dict() for r in self.rankings])

    def summary(self) -> str:
        lines = ["=== Bayesian Ranking Result ==="]
        for i, r in enumerate(self.rankings):
            r._next_name = self.rankings[i + 1].name if i + 1 < len(self.rankings) else ""
            lines.append(f"  {r}")
        status = "CONVERGED" if self.converged else "EXHAUSTED"
        lines.append(
            f"\n{status}: tested {self.problems_tested}/{self.total_problems} "
            f"problems ({self.efficiency:.1%} cost reduction)"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class BayesianRanker:
    """Rank N models simultaneously using Bayesian sequential testing.

    Each model accumulates its own posterior over performance.  After each
    problem the models are sorted by posterior mean and the ranker checks
    whether every consecutive ranked pair (k, k+1) satisfies
    P(k > k+1) ≥ confidence.  When all pairs are resolved the ranker stops
    early, saving evaluation budget.

    Args:
        confidence: Minimum P(rank_k > rank_{k+1}) required to lock in the
                    ordering of every consecutive pair.
        skip_threshold: Declare two models indistinguishable (tie) when
                        P(k > k+1) ∈ (1-skip_threshold, skip_threshold)
                        and they have both seen ≥ min_samples observations.
                        Set to 1.0 to disable tie detection.
        min_samples: Minimum evaluations per model before stopping.
        posterior_factory: Zero-arg callable returning a fresh
                           :class:`~bayesbench.posteriors.Posterior`.

    Example::

        ranker = BayesianRanker(confidence=0.95)
        ranker.add_model("big",   big_model)
        ranker.add_model("small", small_model)
        result = ranker.rank(dataset=problems, score_fn=lambda p, r: r == p["a"])
    """

    def __init__(
        self,
        confidence: float = 0.95,
        skip_threshold: float = 0.85,
        min_samples: int = 5,
        posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
    ) -> None:
        if not (0.5 < confidence <= 1.0):
            raise ValueError("confidence must be in (0.5, 1.0]")
        self.confidence = confidence
        self.skip_threshold = skip_threshold
        self.min_samples = min_samples
        self._posterior_factory: Callable[[], Posterior] = (
            posterior_factory if posterior_factory is not None else BetaPosterior
        )
        self._models: list[tuple[str, Callable]] = []  # (name, fn)
        self._score_fn: Callable | None = None

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def add_model(self, name: str, fn: Callable[[Any], Any]) -> BayesianRanker:
        """Register a model to include in the ranking.

        Args:
            name: Human-readable model name shown in the report.
            fn: ``callable(problem) -> response``

        Returns:
            ``self`` for chaining.
        """
        self._models.append((name, fn))
        return self

    def evaluate(self, score_fn: Callable[[Any, Any], Any]) -> Callable:
        """Decorator to set the scoring function for this ranker.

        The decorated function must have signature
        ``score_fn(problem, response) -> value`` where the value type
        matches the posterior (bool for BetaPosterior, float for
        NormalPosterior).

        Example::

            @ranker.evaluate
            def exact_match(problem, response):
                return response.strip() == problem["answer"]
        """
        self._score_fn = score_fn  # store original so .rank() can use it

        @functools.wraps(score_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return score_fn(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(
        self,
        dataset: Iterable[Any],
        score_fn: Callable[[Any, Any], Any] | None = None,
        verbose: bool = False,
    ) -> RankingResult:
        """Run all registered models on ``dataset`` and return a ranking.

        Args:
            dataset: Iterable of problems.
            score_fn: ``score_fn(problem, response) -> value``.
                      If omitted, uses the function set via ``@ranker.evaluate``.
            verbose: Show a tqdm progress bar (requires ``tqdm``).

        Returns:
            :class:`RankingResult` with ordered models, posteriors, and
            efficiency statistics.
        """
        fn = score_fn or self._score_fn
        if fn is None:
            raise ValueError(
                "No score_fn provided. Pass score_fn= to .rank() or use @ranker.evaluate."
            )
        if not self._models:
            raise ValueError("No models registered. Call ranker.add_model() first.")

        problems = list(dataset)
        posteriors = {name: self._posterior_factory() for name, _ in self._models}
        model_names = [name for name, _ in self._models]

        _log.info("rank: starting  models=%s  n=%d", model_names, len(problems))

        problem_iter: Iterable[Any] = enumerate(problems)
        if verbose and _HAS_TQDM:
            problem_iter = enumerate(_tqdm(problems, desc="Ranking", unit="problem"))

        for i, problem in problem_iter:
            for name, model_fn in self._models:
                response = model_fn(problem)
                posteriors[name].observe_one(fn(problem, response))

            tested = i + 1
            if tested < self.min_samples:
                continue

            if self._ranking_converged(posteriors):
                ordered = self._build_rankings(posteriors)
                _log.info(
                    "rank: CONVERGED at %d/%d  best=%s",
                    tested,
                    len(problems),
                    ordered[0].name,
                )
                return RankingResult(ordered, tested, len(problems), converged=True)

        ordered = self._build_rankings(posteriors)
        _log.info(
            "rank: EXHAUSTED %d problems  best=%s",
            len(problems),
            ordered[0].name if ordered else "N/A",
        )
        return RankingResult(ordered, len(problems), len(problems), converged=False)

    async def rank_async(
        self,
        dataset: Iterable[Any],
        score_fn: Callable[[Any, Any], Any] | None = None,
    ) -> RankingResult:
        """Async version of :meth:`rank`.

        All models are queried concurrently per problem via
        ``asyncio.gather``.
        """
        fn = score_fn or self._score_fn
        if fn is None:
            raise ValueError("No score_fn provided.")
        if not self._models:
            raise ValueError("No models registered.")

        async def call(model_fn: Callable, problem: Any) -> Any:
            if asyncio.iscoroutinefunction(model_fn):
                return await model_fn(problem)
            return model_fn(problem)

        problems = list(dataset)
        posteriors = {name: self._posterior_factory() for name, _ in self._models}

        _log.info("rank_async: starting  models=%d  n=%d", len(self._models), len(problems))

        for i, problem in enumerate(problems):
            responses = await asyncio.gather(
                *[call(model_fn, problem) for _, model_fn in self._models]
            )
            for (name, _), response in zip(self._models, responses):
                posteriors[name].observe_one(fn(problem, response))

            tested = i + 1
            if tested < self.min_samples:
                continue

            if self._ranking_converged(posteriors):
                ordered = self._build_rankings(posteriors)
                _log.info("rank_async: CONVERGED at %d/%d", tested, len(problems))
                return RankingResult(ordered, tested, len(problems), converged=True)

        ordered = self._build_rankings(posteriors)
        _log.info("rank_async: EXHAUSTED %d problems", len(problems))
        return RankingResult(ordered, len(problems), len(problems), converged=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sorted_names(self, posteriors: dict[str, Posterior]) -> list[str]:
        """Return model names sorted by descending posterior mean."""
        return sorted(posteriors, key=lambda n: posteriors[n].mean, reverse=True)

    def _ranking_converged(self, posteriors: dict[str, Posterior]) -> bool:
        """Return True when all consecutive ranked pairs are decided."""
        names = self._sorted_names(posteriors)
        for k in range(len(names) - 1):
            p = posteriors[names[k]].prob_beats(posteriors[names[k + 1]])
            if p < self.confidence:
                return False
        return True

    def _build_rankings(self, posteriors: dict[str, Posterior]) -> list[ModelRanking]:
        names = self._sorted_names(posteriors)
        result = []
        for k, name in enumerate(names):
            p_next = None
            next_name = ""
            if k + 1 < len(names):
                p_next = posteriors[name].prob_beats(posteriors[names[k + 1]])
                next_name = names[k + 1]
            mr = ModelRanking(
                rank=k + 1,
                name=name,
                posterior=posteriors[name],
                p_beats_next=p_next,
            )
            mr._next_name = next_name
            result.append(mr)
        return result
