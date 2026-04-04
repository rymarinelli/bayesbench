"""Standalone decorator API for Bayesian benchmarking.

Provides framework-agnostic decorators that work with any callable —
plain functions, async functions, HuggingFace adapters, or agent wrappers.

All three styles accept a ``posterior_factory`` argument to swap in a
different Bayesian model (e.g. :class:`~baysbench.posteriors.NormalPosterior`
for continuous scores).

Styles
------

**@benchmark** — one-shot comparison with a score function::

    from baysbench import benchmark
    from baysbench.posteriors import NormalPosterior

    @benchmark(
        model_a=gpt4_fn,
        model_b=gpt35_fn,
        dataset=problems,
        confidence=0.95,
    )
    def exact_match(problem, response):
        return response.strip() == problem["answer"]

    result = exact_match.run()      # TaskResult
    print(result.winner)

    # Continuous scores
    @benchmark(
        model_a=gpt4_fn,
        model_b=gpt35_fn,
        dataset=problems,
        posterior_factory=NormalPosterior,
    )
    def bleu(problem, response):
        return compute_bleu(response, problem["reference"])   # float

    result = bleu.run()

**BayesianBenchmark + @bench.task** — multi-task suite::

    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=problems, name="math")
    def math_task(problem):
        return (
            model_a(problem["q"]) == problem["a"],
            model_b(problem["q"]) == problem["a"],
        )

    report = bench.run()

**@suite** — class-based multi-task suite::

    @suite(confidence=0.95)
    class MathBenchmark:
        dataset = problems

        @staticmethod
        def task_arithmetic(problem):
            return model_a(problem["q"]) == problem["a"], \\
                   model_b(problem["q"]) == problem["a"]

    report = MathBenchmark.run()
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

from .benchmark import BayesianBenchmark, BenchmarkReport, TaskResult
from .posteriors.base import Posterior
from .posteriors.beta import BetaPosterior


def benchmark(
    model_a: Callable[[Any], Any],
    model_b: Callable[[Any], Any],
    dataset: Iterable[Any],
    name: str | None = None,
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
    posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
) -> Callable:
    """Decorate a scoring function to create a runnable benchmark task.

    The decorated function keeps its original behaviour and gains a
    ``.run()`` method that executes the sequential benchmark.

    Args:
        model_a: ``model_a(problem) -> response``
        model_b: ``model_b(problem) -> response``
        dataset: Iterable of problems.
        name: Task name (defaults to function name).
        confidence: Stopping threshold.
        skip_threshold: Non-discriminating skip threshold.
        min_samples: Minimum evaluations before early stopping.
        posterior_factory: Override the Bayesian model.
                           :class:`~baysbench.posteriors.BetaPosterior` (default)
                           for binary outcomes; pass
                           :class:`~baysbench.posteriors.NormalPosterior` for
                           continuous scores.

    Returns:
        The original function augmented with a ``.run() -> TaskResult`` method
        and a ``.run_async() -> TaskResult`` coroutine method.

    Examples::

        # Binary exact-match
        @benchmark(model_a=gpt4, model_b=gpt35, dataset=problems)
        def exact_match(problem, response):
            return response.strip() == problem["answer"]

        result = exact_match.run()

        # Continuous BLEU score
        from baysbench.posteriors import NormalPosterior

        @benchmark(
            model_a=gpt4,
            model_b=gpt35,
            dataset=problems,
            posterior_factory=NormalPosterior,
        )
        def bleu_score(problem, response):
            return compute_bleu(response, problem["reference"])

        result = bleu_score.run()
    """

    def decorator(score_fn: Callable) -> Callable:
        task_name = name or score_fn.__name__
        bench = BayesianBenchmark(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
            posterior_factory=posterior_factory,
        )

        @functools.wraps(score_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return score_fn(*args, **kwargs)

        def run() -> TaskResult:
            return bench.compare(
                model_a=model_a,
                model_b=model_b,
                score_fn=score_fn,
                dataset=dataset,
                name=task_name,
            )

        async def run_async() -> TaskResult:
            return await bench.compare_async(
                model_a=model_a,
                model_b=model_b,
                score_fn=score_fn,
                dataset=dataset,
                name=task_name,
            )

        wrapper.run = run  # type: ignore[attr-defined]
        wrapper.run_async = run_async  # type: ignore[attr-defined]
        wrapper._baysbench_config = {  # type: ignore[attr-defined]
            "confidence": confidence,
            "skip_threshold": skip_threshold,
            "min_samples": min_samples,
            "posterior_factory": posterior_factory or BetaPosterior,
        }
        return wrapper

    return decorator


def suite(
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
    posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
) -> Callable:
    """Class decorator that turns a class into a multi-task benchmark suite.

    Methods whose names start with ``task_`` are automatically registered.
    The class-level ``dataset`` attribute is used as the default dataset;
    individual tasks can override it via a ``dataset_<task_name>`` attribute.

    Args:
        confidence: Stopping confidence threshold.
        skip_threshold: Non-discriminating skip threshold.
        min_samples: Minimum samples before early stopping.
        posterior_factory: Override the Bayesian model for all tasks.
                           Individual tasks can further override via
                           ``posterior_<task_name>`` class attributes.

    Returns:
        The decorated class, augmented with a ``run() -> BenchmarkReport``
        classmethod and a ``run_async()`` async classmethod.

    Example::

        from baysbench.posteriors import NormalPosterior

        @suite(confidence=0.95, posterior_factory=NormalPosterior)
        class QualityBenchmark:
            dataset = problems

            @staticmethod
            def task_bleu(problem):
                a = compute_bleu(model_a(problem["src"]), problem["ref"])
                b = compute_bleu(model_b(problem["src"]), problem["ref"])
                return a, b

        report = QualityBenchmark.run()
    """

    def decorator(cls: type) -> type:
        bench = BayesianBenchmark(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
            posterior_factory=posterior_factory,
        )
        default_dataset = getattr(cls, "dataset", None)

        for attr_name in sorted(dir(cls)):
            if not attr_name.startswith("task_"):
                continue
            fn = getattr(cls, attr_name)
            if not callable(fn):
                continue
            task_name = attr_name[len("task_") :]
            task_dataset = getattr(cls, f"dataset_{task_name}", default_dataset)
            task_posterior = getattr(cls, f"posterior_{task_name}", None)
            bench._tasks.append(
                {
                    "name": task_name,
                    "fn": fn,
                    "dataset": task_dataset,
                    "posterior_factory": task_posterior,
                }
            )

        cls._bench = bench  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def run(klass: type) -> BenchmarkReport:
            return bench.run()

        @classmethod  # type: ignore[misc]
        async def run_async(klass: type) -> BenchmarkReport:
            return await bench.run_async()

        cls.run = run  # type: ignore[attr-defined]
        cls.run_async = run_async  # type: ignore[attr-defined]
        return cls

    return decorator
