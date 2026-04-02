"""Standalone decorator API for Bayesian benchmarking.

Provides framework-agnostic decorators that work with any callable —
plain functions, class methods, async functions, or agent wrappers.

Typical usage::

    from baysbench import benchmark, BayesianBenchmark

    # ── Approach 1: standalone @benchmark decorator ──────────────────────
    @benchmark(
        model_a=gpt4_fn,
        model_b=gpt35_fn,
        dataset=problems,
        confidence=0.95,
    )
    def gsm8k(problem, response):
        return response.strip() == problem["answer"]

    result = gsm8k.run()          # TaskResult
    print(result)

    # ── Approach 2: BayesianBenchmark instance + @bench.task ─────────────
    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=problems, name="math")
    def math_task(problem):
        a_correct = gpt4_fn(problem["question"]) == problem["answer"]
        b_correct = gpt35_fn(problem["question"]) == problem["answer"]
        return a_correct, b_correct

    report = bench.run()          # BenchmarkReport
    print(report.summary())

    # ── Approach 3: class-based suite with @suite ─────────────────────────
    @suite(confidence=0.95)
    class MathBenchmark:
        dataset = problems

        @staticmethod
        def task_arithmetic(problem):
            return (
                gpt4_fn(problem["q"]) == problem["a"],
                gpt35_fn(problem["q"]) == problem["a"],
            )

        @staticmethod
        def task_algebra(problem):
            return (
                gpt4_fn(problem["q"]) == problem["a"],
                gpt35_fn(problem["q"]) == problem["a"],
            )

    report = MathBenchmark.run()
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Iterable

from .benchmark import BayesianBenchmark, BenchmarkReport, TaskResult


def benchmark(
    model_a: Callable[[Any], Any],
    model_b: Callable[[Any], Any],
    dataset: Iterable[Any],
    name: str | None = None,
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
) -> Callable:
    """Decorate a scoring function to create a runnable benchmark task.

    The decorated function acts as a normal scoring function but gains a
    ``.run()`` method that executes the full sequential benchmark and returns
    a :class:`~baysbench.benchmark.TaskResult`.

    Args:
        model_a: Callable ``model_a(problem) -> response`` for model A.
        model_b: Callable ``model_b(problem) -> response`` for model B.
        dataset: Iterable of problems fed to both models.
        name: Optional task name (defaults to the function name).
        confidence: Stopping threshold — declare winner when P(A>B) ≥ this.
        skip_threshold: Skip a non-discriminating task when P(A>B) is inside
                        ``(1-skip_threshold, skip_threshold)``.
        min_samples: Minimum evaluations before early stopping is considered.

    Returns:
        The original scoring function, augmented with a ``.run()`` method.

    Example::

        @benchmark(
            model_a=lambda p: big_model(p["question"]),
            model_b=lambda p: small_model(p["question"]),
            dataset=gsm8k_problems,
            confidence=0.95,
        )
        def exact_match(problem, response):
            return response.strip() == problem["answer"]

        result = exact_match.run()
        print(result.winner)        # "model_a", "model_b", or None
        print(result.efficiency)    # fraction of problems saved
    """

    def decorator(score_fn: Callable[[Any, Any], bool]) -> Callable:
        task_name = name or score_fn.__name__
        bench = BayesianBenchmark(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
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

        wrapper.run = run  # type: ignore[attr-defined]
        wrapper._baysbench_config = {  # type: ignore[attr-defined]
            "confidence": confidence,
            "skip_threshold": skip_threshold,
            "min_samples": min_samples,
        }
        return wrapper

    return decorator


def suite(
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
) -> Callable:
    """Class decorator that turns a class into a multi-task benchmark suite.

    Any ``staticmethod`` or regular method whose name starts with ``task_``
    is automatically registered as a benchmark task.  The class-level
    ``dataset`` attribute is used as the default dataset for all tasks
    (individual tasks can override it by using a ``dataset_<name>`` attribute).

    Args:
        confidence: Stopping confidence threshold.
        skip_threshold: Non-discriminating skip threshold.
        min_samples: Minimum samples before early stopping.

    Returns:
        The decorated class, augmented with a ``run()`` classmethod.

    Example::

        @suite(confidence=0.95)
        class Eval:
            dataset = all_problems

            @staticmethod
            def task_math(problem):
                return model_a(problem["q"]) == problem["a"], \\
                       model_b(problem["q"]) == problem["a"]

            @staticmethod
            def task_code(problem):
                return run_code(model_a, problem), run_code(model_b, problem)

        report = Eval.run()
    """

    def decorator(cls: type) -> type:
        bench = BayesianBenchmark(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
        )
        default_dataset = getattr(cls, "dataset", None)

        for attr_name in sorted(dir(cls)):
            if not attr_name.startswith("task_"):
                continue
            fn = getattr(cls, attr_name)
            if not callable(fn):
                continue
            task_name = attr_name[len("task_"):]
            # Allow per-task dataset via dataset_<name> class attribute
            task_dataset = getattr(cls, f"dataset_{task_name}", default_dataset)
            bench._tasks.append({"name": task_name, "fn": fn, "dataset": task_dataset})

        cls._bench = bench  # type: ignore[attr-defined]

        @classmethod  # type: ignore[misc]
        def run(klass: type) -> BenchmarkReport:
            return bench.run()

        cls.run = run  # type: ignore[attr-defined]
        return cls

    return decorator
