"""baysbench — Bayesian sequential benchmarking for LLMs and agents.

Quick start::

    from baysbench import BayesianBenchmark, benchmark, suite

    # ── Option 1: instance + @bench.task ─────────────────────────────────
    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=problems)
    def math(problem):
        a = model_a(problem["question"]) == problem["answer"]
        b = model_b(problem["question"]) == problem["answer"]
        return a, b

    report = bench.run()
    print(report.summary())

    # ── Option 2: standalone @benchmark decorator ─────────────────────────
    from baysbench import benchmark

    @benchmark(model_a=model_a, model_b=model_b, dataset=problems)
    def exact_match(problem, response):
        return response.strip() == problem["answer"]

    result = exact_match.run()

    # ── Option 3: class-based @suite ──────────────────────────────────────
    from baysbench import suite

    @suite(confidence=0.95)
    class MyEval:
        dataset = problems

        @staticmethod
        def task_math(problem):
            return model_a(problem["q"]) == problem["a"], \\
                   model_b(problem["q"]) == problem["a"]

    report = MyEval.run()
"""

from .benchmark import BayesianBenchmark, BenchmarkReport, TaskResult
from .core import BetaPosterior, is_non_discriminating, prob_a_beats_b
from .decorators import benchmark, suite

__all__ = [
    # High-level API
    "BayesianBenchmark",
    "BenchmarkReport",
    "TaskResult",
    # Decorators
    "benchmark",
    "suite",
    # Core primitives (exposed for custom workflows)
    "BetaPosterior",
    "prob_a_beats_b",
    "is_non_discriminating",
]
