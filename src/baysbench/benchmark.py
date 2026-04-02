"""BayesianBenchmark: sequential testing engine for comparing two LLMs/agents."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .core import BetaPosterior, is_non_discriminating, prob_a_beats_b


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""

    name: str
    problems_tested: int
    total_problems: int
    posterior_a: BetaPosterior
    posterior_b: BetaPosterior
    p_a_beats_b: float
    skipped: bool = False

    @property
    def winner(self) -> str | None:
        """'model_a', 'model_b', or None (inconclusive)."""
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

    Stops evaluating a task early as soon as enough statistical evidence
    accumulates, dramatically reducing the number of problems that need
    to be evaluated.

    Args:
        confidence: Posterior probability threshold to declare a winner.
                    Stops when P(A>B) ≥ confidence or ≤ (1-confidence).
        skip_threshold: Treats a task as non-discriminating and skips it
                        when P(A>B) ∈ (1-skip_threshold, skip_threshold).
        min_samples: Minimum problems to evaluate before any early stopping.

    Example::

        bench = BayesianBenchmark(confidence=0.95)

        @bench.task(dataset=math_problems)
        def math(problem):
            a_correct = model_a(problem["question"]) == problem["answer"]
            b_correct = model_b(problem["question"]) == problem["answer"]
            return a_correct, b_correct

        report = bench.run()
        print(report.summary())
    """

    def __init__(
        self,
        confidence: float = 0.95,
        skip_threshold: float = 0.85,
        min_samples: int = 3,
    ) -> None:
        if not (0.5 < confidence <= 1.0):
            raise ValueError("confidence must be in (0.5, 1.0]")
        if not (0.5 < skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in (0.5, 1.0]")
        self.confidence = confidence
        self.skip_threshold = skip_threshold
        self.min_samples = min_samples
        self._tasks: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

    def task(
        self,
        name: str | None = None,
        dataset: Iterable[Any] | None = None,
    ) -> Callable:
        """Decorator to register an evaluation function as a benchmark task.

        The decorated function receives one problem at a time and must return
        a ``(bool, bool)`` tuple: ``(model_a_correct, model_b_correct)``.

        Args:
            name: Human-readable task name. Defaults to the function name.
            dataset: Iterable of problems to evaluate. Can also be set later
                     by assigning to ``fn.dataset``.

        Example::

            @bench.task(dataset=problems, name="arithmetic")
            def arithmetic(problem):
                a_correct = model_a(problem["q"]) == problem["a"]
                b_correct = model_b(problem["q"]) == problem["a"]
                return a_correct, b_correct
        """

        def decorator(fn: Callable) -> Callable:
            task_name = name or fn.__name__
            entry: dict[str, Any] = {"name": task_name, "fn": fn, "dataset": dataset}
            self._tasks.append(entry)
            # Attach metadata for introspection
            fn._baysbench_task = entry  # type: ignore[attr-defined]
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Direct comparison (no decorator)
    # ------------------------------------------------------------------

    def compare(
        self,
        model_a: Callable[[Any], Any],
        model_b: Callable[[Any], Any],
        score_fn: Callable[[Any, Any], bool],
        dataset: Iterable[Any],
        name: str = "task",
    ) -> TaskResult:
        """Directly compare two models on a dataset using a scoring function.

        Args:
            model_a: Callable that takes a problem and returns a response.
            model_b: Callable that takes a problem and returns a response.
            score_fn: ``score_fn(problem, response) -> bool`` — True if correct.
            dataset: Iterable of problems.
            name: Name for this comparison task.

        Returns:
            :class:`TaskResult` with posteriors, winner, and efficiency stats.

        Example::

            result = bench.compare(
                model_a=gpt4,
                model_b=gpt35,
                score_fn=lambda p, r: r.strip() == p["answer"],
                dataset=problems,
                name="gsm8k",
            )
        """
        problems = list(dataset)
        post_a = BetaPosterior()
        post_b = BetaPosterior()

        for i, problem in enumerate(problems):
            output_a = model_a(problem)
            output_b = model_b(problem)
            post_a.observe_one(score_fn(problem, output_a))
            post_b.observe_one(score_fn(problem, output_b))
            tested = i + 1

            if tested < self.min_samples:
                continue

            if is_non_discriminating(post_a, post_b, self.skip_threshold):
                p = prob_a_beats_b(post_a, post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            p = prob_a_beats_b(post_a, post_b)
            if p >= self.confidence or p <= (1.0 - self.confidence):
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = prob_a_beats_b(post_a, post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    async def compare_async(
        self,
        model_a: Callable[[Any], Any],
        model_b: Callable[[Any], Any],
        score_fn: Callable[[Any, Any], bool],
        dataset: Iterable[Any],
        name: str = "task",
        concurrency: int = 1,
    ) -> TaskResult:
        """Async version of :meth:`compare` supporting async model callables.

        When ``concurrency > 1``, both models are called concurrently for the
        same problem using ``asyncio.gather``.

        Args:
            model_a: Sync or async callable taking a problem → response.
            model_b: Sync or async callable taking a problem → response.
            score_fn: ``score_fn(problem, response) -> bool``.
            dataset: Iterable of problems.
            name: Task name.
            concurrency: Number of problems to evaluate in parallel (default 1
                         preserves sequential stopping semantics exactly).

        Returns:
            :class:`TaskResult`.
        """

        async def call(fn: Callable, problem: Any) -> Any:
            if asyncio.iscoroutinefunction(fn):
                return await fn(problem)
            return fn(problem)

        problems = list(dataset)
        post_a = BetaPosterior()
        post_b = BetaPosterior()

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

            if is_non_discriminating(post_a, post_b, self.skip_threshold):
                p = prob_a_beats_b(post_a, post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            p = prob_a_beats_b(post_a, post_b)
            if p >= self.confidence or p <= (1.0 - self.confidence):
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = prob_a_beats_b(post_a, post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    # ------------------------------------------------------------------
    # Run all registered tasks
    # ------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Run all tasks registered via :meth:`task` and return a report.

        Returns:
            :class:`BenchmarkReport` aggregating all task results.
        """
        report = BenchmarkReport()
        for task_def in self._tasks:
            result = self._run_task(task_def)
            report.task_results.append(result)
        return report

    async def run_async(self) -> BenchmarkReport:
        """Async version of :meth:`run` for tasks whose functions are async."""
        report = BenchmarkReport()
        results = await asyncio.gather(*[self._run_task_async(t) for t in self._tasks])
        report.task_results.extend(results)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_task(self, task_def: dict[str, Any]) -> TaskResult:
        fn = task_def["fn"]
        name = task_def["name"]
        dataset = task_def.get("dataset") or getattr(fn, "dataset", None)
        if dataset is None:
            raise ValueError(f"Task '{name}' has no dataset. Pass dataset= to @bench.task().")
        problems = list(dataset)
        post_a = BetaPosterior()
        post_b = BetaPosterior()

        for i, problem in enumerate(problems):
            correct_a, correct_b = fn(problem)
            post_a.observe_one(correct_a)
            post_b.observe_one(correct_b)
            tested = i + 1

            if tested < self.min_samples:
                continue

            if is_non_discriminating(post_a, post_b, self.skip_threshold):
                p = prob_a_beats_b(post_a, post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            p = prob_a_beats_b(post_a, post_b)
            if p >= self.confidence or p <= (1.0 - self.confidence):
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = prob_a_beats_b(post_a, post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)

    async def _run_task_async(self, task_def: dict[str, Any]) -> TaskResult:
        fn = task_def["fn"]
        name = task_def["name"]
        dataset = task_def.get("dataset") or getattr(fn, "dataset", None)
        if dataset is None:
            raise ValueError(f"Task '{name}' has no dataset.")
        problems = list(dataset)
        post_a = BetaPosterior()
        post_b = BetaPosterior()

        for i, problem in enumerate(problems):
            if asyncio.iscoroutinefunction(fn):
                result = await fn(problem)
            else:
                result = fn(problem)
            correct_a, correct_b = result
            post_a.observe_one(correct_a)
            post_b.observe_one(correct_b)
            tested = i + 1

            if tested < self.min_samples:
                continue

            if is_non_discriminating(post_a, post_b, self.skip_threshold):
                p = prob_a_beats_b(post_a, post_b)
                return TaskResult(name, tested, len(problems), post_a, post_b, p, skipped=True)

            p = prob_a_beats_b(post_a, post_b)
            if p >= self.confidence or p <= (1.0 - self.confidence):
                return TaskResult(name, tested, len(problems), post_a, post_b, p)

        p = prob_a_beats_b(post_a, post_b)
        return TaskResult(name, len(problems), len(problems), post_a, post_b, p)
