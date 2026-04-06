"""BayesianBenchmark: posterior-generic sequential testing engine."""

from __future__ import annotations

import asyncio
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
class TaskResult:
    """Result of running a single benchmark task."""

    name: str
    problems_tested: int
    total_problems: int
    posterior_a: Posterior
    posterior_b: Posterior
    p_a_beats_b: float
    skipped: bool = False

    @property
    def winner(self) -> str | None:
        """'model_a', 'model_b', or None (inconclusive).

        Uses the confidence threshold baked into p_a_beats_b; a result is
        declared only when p ≥ 0.95 or p ≤ 0.05.
        """
        if self.p_a_beats_b >= 0.95:
            return "model_a"
        if self.p_a_beats_b <= 0.05:
            return "model_b"
        return None

    @property
    def efficiency(self) -> float:
        """Fraction of problems *not* tested (0 = no savings, 1 = all skipped)."""
        if self.total_problems == 0:
            return 0.0
        return 1.0 - self.problems_tested / self.total_problems

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe, no posterior objects).

        Returns a dict with task metadata, posterior means, credible intervals,
        and efficiency statistics — suitable for JSON export or DataFrame rows.

        Example::

            import json
            print(json.dumps(result.to_dict(), indent=2))
        """
        lo_a, hi_a = self.posterior_a.credible_interval()
        lo_b, hi_b = self.posterior_b.credible_interval()
        return {
            "name": self.name,
            "problems_tested": self.problems_tested,
            "total_problems": self.total_problems,
            "efficiency": self.efficiency,
            "p_a_beats_b": self.p_a_beats_b,
            "winner": self.winner,
            "skipped": self.skipped,
            "mean_a": self.posterior_a.mean,
            "mean_b": self.posterior_b.mean,
            "ci_a_lo": lo_a,
            "ci_a_hi": hi_a,
            "ci_b_lo": lo_b,
            "ci_b_hi": hi_b,
        }

    def __str__(self) -> str:
        status = "SKIPPED" if self.skipped else (self.winner or "INCONCLUSIVE")
        return (
            f"{self.name}: {status} | "
            f"tested {self.problems_tested}/{self.total_problems} "
            f"({self.efficiency:.1%} saved) | "
            f"P(A>B)={self.p_a_beats_b:.3f} | "
            f"A={self.posterior_a.mean:.3f}, B={self.posterior_b.mean:.3f}"
        )


@dataclass
class BenchmarkReport:
    """Aggregate results across all tasks in a benchmark run."""

    task_results: list[TaskResult] = field(default_factory=list)

    @property
    def total_problems_tested(self) -> int:
        return sum(r.problems_tested for r in self.task_results)

    @property
    def total_problems_available(self) -> int:
        return sum(r.total_problems for r in self.task_results)

    @property
    def overall_efficiency(self) -> float:
        if self.total_problems_available == 0:
            return 0.0
        return 1.0 - self.total_problems_tested / self.total_problems_available

    @property
    def winners(self) -> dict[str, str | None]:
        """Map of task name → winner ('model_a', 'model_b', or None)."""
        return {r.name: r.winner for r in self.task_results}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report to a nested plain dict.

        Example::

            import json
            with open("report.json", "w") as f:
                json.dump(bench.run().to_dict(), f, indent=2)
        """
        return {
            "tasks": [r.to_dict() for r in self.task_results],
            "total_problems_tested": self.total_problems_tested,
            "total_problems_available": self.total_problems_available,
            "overall_efficiency": self.overall_efficiency,
            "winners": self.winners,
        }

    def to_dataframe(self) -> Any:
        """Return a :class:`pandas.DataFrame` with one row per task.

        Requires ``pandas`` to be installed (``pip install pandas``).

        Columns: ``name``, ``problems_tested``, ``total_problems``,
        ``efficiency``, ``p_a_beats_b``, ``winner``, ``skipped``,
        ``mean_a``, ``mean_b``, ``ci_a_lo``, ``ci_a_hi``,
        ``ci_b_lo``, ``ci_b_hi``.

        Example::

            df = bench.run().to_dataframe()
            print(df[["name", "winner", "efficiency"]].to_string())
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). Install with: pip install pandas"
            ) from exc
        return pd.DataFrame([r.to_dict() for r in self.task_results])

    def summary(self) -> str:
        lines = ["=== Bayesian Benchmark Report ==="]
        for r in self.task_results:
            lines.append(f"  {r}")
        lines.append(
            f"\nTotal: tested {self.total_problems_tested}/{self.total_problems_available} "
            f"problems ({self.overall_efficiency:.1%} cost reduction)"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class BayesianBenchmark:
    """Sequential Bayesian benchmark for comparing two models (A vs B).

    Stops evaluating a task early once the posterior evidence crosses a
    confidence threshold, dramatically reducing evaluation cost.

    Works with *any* posterior type via the ``posterior_factory`` parameter —
    swap in :class:`~bayesbench.posteriors.NormalPosterior` for continuous
    scores, or supply your own :class:`~bayesbench.posteriors.Posterior`
    subclass for custom Bayesian models.

    Args:
        confidence: Stopping threshold. Declare a winner when
                    P(A>B) ≥ confidence or ≤ (1-confidence).
        skip_threshold: Skip a task as non-discriminating when
                        P(A>B) ∈ (1-skip_threshold, skip_threshold).
        min_samples: Minimum evaluations before any early stopping.
        posterior_factory: Zero-argument callable that returns a fresh
                           :class:`~bayesbench.posteriors.Posterior`.
                           Defaults to :class:`~bayesbench.posteriors.BetaPosterior`
                           (binary outcomes). Pass ``NormalPosterior`` for
                           continuous score tasks.

    Examples::

        # Binary outcomes (default)
        bench = BayesianBenchmark(confidence=0.95)

        # Continuous scores (BLEU, ROUGE, LLM-judge 0-1)
        from bayesbench.posteriors import NormalPosterior
        bench = BayesianBenchmark(posterior_factory=NormalPosterior)

        # Custom prior: expect ~30% BLEU baseline
        bench = BayesianBenchmark(
            posterior_factory=lambda: NormalPosterior(mu_0=0.30)
        )
    """

    def __init__(
        self,
        confidence: float = 0.95,
        skip_threshold: float = 0.85,
        min_samples: int = 3,
        posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
    ) -> None:
        if not (0.5 < confidence <= 1.0):
            raise ValueError("confidence must be in (0.5, 1.0]")
        if not (0.5 < skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in (0.5, 1.0]")
        self.confidence = confidence
        self.skip_threshold = skip_threshold
        self.min_samples = min_samples
        self._posterior_factory: Callable[[], Posterior] = (
            posterior_factory if posterior_factory is not None else BetaPosterior
        )
        self._tasks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_posterior(self) -> Posterior:
        return self._posterior_factory()

    def _is_non_discriminating(self, pa: Posterior, pb: Posterior) -> bool:
        p = pa.prob_beats(pb)
        return (1.0 - self.skip_threshold) < p < self.skip_threshold

    def _stopping(self, pa: Posterior, pb: Posterior) -> tuple[bool, float]:
        """Return (should_stop, p_a_beats_b)."""
        p = pa.prob_beats(pb)
        return (p >= self.confidence or p <= (1.0 - self.confidence)), p

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def task(
        self,
        name: str | None = None,
        dataset: Iterable[Any] | None = None,
        posterior_factory: Callable[[], Posterior] | None = None,
    ) -> Callable:
        """Decorator to register an evaluation function as a benchmark task.

        The decorated function receives one problem at a time and must return
        a tuple ``(value_a, value_b)`` where the value type matches the
        posterior: ``bool`` for :class:`~bayesbench.posteriors.BetaPosterior`,
        ``float`` for :class:`~bayesbench.posteriors.NormalPosterior`.

        Args:
            name: Task name (defaults to function name).
            dataset: Iterable of problems.
            posterior_factory: Override the benchmark-level posterior for
                               this specific task.

        Example::

            from bayesbench.posteriors import NormalPosterior

            bench = BayesianBenchmark()   # default: BetaPosterior

            # Override a single task to use continuous scores
            @bench.task(dataset=problems, posterior_factory=NormalPosterior)
            def bleu_task(problem):
                score_a = compute_bleu(model_a(problem), problem["ref"])
                score_b = compute_bleu(model_b(problem), problem["ref"])
                return score_a, score_b
        """

        def decorator(fn: Callable) -> Callable:
            task_name = name or fn.__name__
            entry: dict[str, Any] = {
                "name": task_name,
                "fn": fn,
                "dataset": dataset,
                "posterior_factory": posterior_factory,
            }
            self._tasks.append(entry)
            fn._bayesbench_task = entry  # type: ignore[attr-defined]
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Direct comparison (no decorator)
    # ------------------------------------------------------------------

    def compare(
        self,
        model_a: Callable[[Any], Any],
        model_b: Callable[[Any], Any],
        score_fn: Callable[[Any, Any], Any],
        dataset: Iterable[Any],
        name: str = "task",
        posterior_factory: Callable[[], Posterior] | None = None,
        verbose: bool = False,
    ) -> TaskResult:
        """Directly compare two models on a dataset using a scoring function.

        Args:
            model_a: ``model_a(problem) -> response``
            model_b: ``model_b(problem) -> response``
            score_fn: ``score_fn(problem, response) -> value`` where value is
                      ``bool`` for binary posteriors or ``float`` for
                      continuous posteriors.
            dataset: Iterable of problems.
            name: Task name.
            posterior_factory: Override the benchmark-level posterior.
            verbose: Show a tqdm progress bar (requires ``tqdm``).

        Returns:
            :class:`TaskResult`

        Example::

            # Binary
            result = bench.compare(gpt4, gpt35, lambda p, r: r == p["a"], problems)

            # Continuous (BLEU)
            from bayesbench.posteriors import NormalPosterior
            result = bench.compare(
                gpt4, gpt35,
                lambda p, r: bleu(r, p["ref"]),
                problems,
                posterior_factory=NormalPosterior,
            )
        """
        factory = posterior_factory or self._posterior_factory
        problems = list(dataset)
        post_a = factory()
        post_b = factory()

        iterator: Iterable[Any] = enumerate(problems)
        if verbose and _HAS_TQDM:
            iterator = enumerate(_tqdm(problems, desc=name, unit="problem"))

        _log.debug("compare: task=%s  n=%d", name, len(problems))
        for i, problem in iterator:
            post_a.observe_one(score_fn(problem, model_a(problem)))
            post_b.observe_one(score_fn(problem, model_b(problem)))
            tested = i + 1

            if tested < self.min_samples:
                continue

            if self._is_non_discriminating(post_a, post_b):
                p = post_a.prob_beats(post_b)
                _log.info(
                    "compare: task=%s SKIPPED at %d/%d  P(A>B)=%.3f",
                    name,
                    tested,
                    len(problems),
                    p,
                )
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            stop, p = self._stopping(post_a, post_b)
            if stop:
                _log.info(
                    "compare: task=%s STOPPED at %d/%d  P(A>B)=%.3f  winner=%s",
                    name,
                    tested,
                    len(problems),
                    p,
                    "model_a" if p >= self.confidence else "model_b",
                )
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = post_a.prob_beats(post_b)
        _log.info(
            "compare: task=%s EXHAUSTED %d problems  P(A>B)=%.3f",
            name,
            len(problems),
            p,
        )
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    async def compare_async(
        self,
        model_a: Callable[[Any], Any],
        model_b: Callable[[Any], Any],
        score_fn: Callable[[Any, Any], Any],
        dataset: Iterable[Any],
        name: str = "task",
        posterior_factory: Callable[[], Posterior] | None = None,
    ) -> TaskResult:
        """Async version of :meth:`compare` — accepts async model callables.

        Both models are called concurrently per problem using
        ``asyncio.gather``.
        """

        async def call(fn: Callable, problem: Any) -> Any:
            if asyncio.iscoroutinefunction(fn):
                return await fn(problem)
            return fn(problem)

        factory = posterior_factory or self._posterior_factory
        problems = list(dataset)
        post_a = factory()
        post_b = factory()

        _log.debug("compare_async: task=%s  n=%d", name, len(problems))
        for i, problem in enumerate(problems):
            output_a, output_b = await asyncio.gather(
                call(model_a, problem),
                call(model_b, problem),
            )
            post_a.observe_one(score_fn(problem, output_a))
            post_b.observe_one(score_fn(problem, output_b))
            tested = i + 1

            if tested < self.min_samples:
                continue

            if self._is_non_discriminating(post_a, post_b):
                p = post_a.prob_beats(post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            stop, p = self._stopping(post_a, post_b)
            if stop:
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = post_a.prob_beats(post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    # ------------------------------------------------------------------
    # Run all registered tasks
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> BenchmarkReport:
        """Run all tasks registered via :meth:`task` and return a report.

        Args:
            verbose: Show tqdm progress bars at the task and problem level.
                     Requires ``tqdm`` (``pip install tqdm``).
        """
        report = BenchmarkReport()
        tasks = self._tasks
        task_iter: Iterable[dict[str, Any]] = tasks
        if verbose and _HAS_TQDM:
            task_iter = _tqdm(tasks, desc="Benchmark", unit="task")
        _log.info("run: starting benchmark  tasks=%d", len(tasks))
        for task_def in task_iter:
            result = self._run_task(task_def, verbose=verbose)
            _log.info(
                "run: task=%s  winner=%s  P(A>B)=%.3f  tested=%d/%d",
                result.name,
                result.winner or "INCONCLUSIVE",
                result.p_a_beats_b,
                result.problems_tested,
                result.total_problems,
            )
            report.task_results.append(result)
        _log.info(
            "run: done  overall_efficiency=%.1f%%",
            report.overall_efficiency * 100,
        )
        return report

    async def run_async(self, verbose: bool = False) -> BenchmarkReport:
        """Async version of :meth:`run`.

        Args:
            verbose: Ignored in async mode (tqdm is not async-safe here).
        """
        report = BenchmarkReport()
        _log.info("run_async: starting benchmark  tasks=%d", len(self._tasks))
        results = await asyncio.gather(*[self._run_task_async(t) for t in self._tasks])
        report.task_results.extend(results)
        _log.info(
            "run_async: done  overall_efficiency=%.1f%%",
            report.overall_efficiency * 100,
        )
        return report

    # ------------------------------------------------------------------
    # Internal task runners
    # ------------------------------------------------------------------

    def _run_task(self, task_def: dict[str, Any], verbose: bool = False) -> TaskResult:
        fn = task_def["fn"]
        name = task_def["name"]
        dataset = task_def.get("dataset") or getattr(fn, "dataset", None)
        if dataset is None:
            raise ValueError(f"Task '{name}' has no dataset. Pass dataset= to @bench.task().")
        factory = task_def.get("posterior_factory") or self._posterior_factory
        problems = list(dataset)
        post_a = factory()
        post_b = factory()

        problem_iter: Iterable[Any] = enumerate(problems)
        if verbose and _HAS_TQDM:
            problem_iter = enumerate(_tqdm(problems, desc=name, unit="problem", leave=False))

        for i, problem in problem_iter:
            val_a, val_b = fn(problem)
            post_a.observe_one(val_a)
            post_b.observe_one(val_b)
            tested = i + 1

            if tested < self.min_samples:
                continue

            if self._is_non_discriminating(post_a, post_b):
                p = post_a.prob_beats(post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            stop, p = self._stopping(post_a, post_b)
            if stop:
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = post_a.prob_beats(post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    async def _run_task_async(self, task_def: dict[str, Any]) -> TaskResult:
        fn = task_def["fn"]
        name = task_def["name"]
        dataset = task_def.get("dataset") or getattr(fn, "dataset", None)
        if dataset is None:
            raise ValueError(f"Task '{name}' has no dataset.")
        factory = task_def.get("posterior_factory") or self._posterior_factory
        problems = list(dataset)
        post_a = factory()
        post_b = factory()

        for i, problem in enumerate(problems):
            if asyncio.iscoroutinefunction(fn):
                result = await fn(problem)
            else:
                result = fn(problem)
            val_a, val_b = result
            post_a.observe_one(val_a)
            post_b.observe_one(val_b)
            tested = i + 1

            if tested < self.min_samples:
                continue

            if self._is_non_discriminating(post_a, post_b):
                p = post_a.prob_beats(post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            stop, p = self._stopping(post_a, post_b)
            if stop:
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = post_a.prob_beats(post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)
