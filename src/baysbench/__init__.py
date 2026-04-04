"""baysbench — Bayesian sequential benchmarking for LLMs and agents.

Features
--------
- **Early stopping**: Halts evaluation once posterior evidence crosses a
  confidence threshold — reducing cost by up to 99% vs full-dataset runs.
- **Pluggable posteriors**: Beta-Bernoulli for binary outcomes, Normal-Inverse-
  Gamma for continuous scores, Dirichlet-Multinomial for categorical outcomes,
  Gamma-Poisson for counts/latency, or bring your own.
- **Framework adapters**: HuggingFace Inference API, OpenAI-compatible APIs
  (Groq, Together AI, vLLM, Ollama, …), and Anthropic — all via the same
  decorator API.
- **Multi-model ranking**: Rank N models simultaneously with Bayesian early
  stopping, without running O(N²) pairwise comparisons.
- **Three decorator styles**: ``@benchmark``, ``@bench.task``, ``@suite``.
- **Convenience functions**: ``baysbench.compare()`` and ``baysbench.rank()``
  for quick one-liner evaluations — no class instantiation required.
- **Async-first**: All operations support ``async/await``.
- **Export**: ``to_dict()`` and ``to_dataframe()`` on all result types.

Quick start::

    import baysbench

    # ── One-liner compare ────────────────────────────────────────────────
    result = baysbench.compare(
        model_a=big_llm,
        model_b=small_llm,
        score_fn=lambda p, r: r.strip() == p["answer"],
        dataset=problems,
    )
    print(result.winner, f"{result.efficiency:.1%} cost saved")

    # ── One-liner rank ───────────────────────────────────────────────────
    result = baysbench.rank(
        models={"gpt-4o": gpt4_fn, "llama-3": llama_fn, "mistral": mistral_fn},
        score_fn=lambda p, r: r.strip() == p["answer"],
        dataset=problems,
    )
    print(result.summary())

    # ── @benchmark — one-shot ────────────────────────────────────────────
    from baysbench import benchmark

    @benchmark(model_a=big_llm, model_b=small_llm, dataset=problems)
    def exact_match(problem, response):
        return response.strip() == problem["answer"]

    result = exact_match.run()

    # ── @bench.task — multi-task ─────────────────────────────────────────
    from baysbench import BayesianBenchmark

    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=problems, name="gsm8k")
    def gsm8k(problem):
        return big_llm(problem["q"]) == problem["a"], \\
               small_llm(problem["q"]) == problem["a"]

    report = bench.run(verbose=True)  # tqdm progress bars
    print(report.summary())
    df = report.to_dataframe()        # pandas DataFrame

    # ── @suite — class-based ─────────────────────────────────────────────
    from baysbench import suite

    @suite(confidence=0.95)
    class MyEval:
        dataset = problems
        def task_math(problem): ...

    print(MyEval.run().summary())

    # ── BayesianRanker — rank N models ───────────────────────────────────
    from baysbench import BayesianRanker

    ranker = BayesianRanker(confidence=0.95)
    ranker.add_model("gpt-4o",   gpt4_fn)
    ranker.add_model("llama-3",  llama_fn)
    ranker.add_model("mistral",  mistral_fn)

    result = ranker.rank(dataset=problems, score_fn=lambda p, r: r == p["a"])
    print(result.summary())
    df = result.to_dataframe()

    # ── Posterior selection guide ────────────────────────────────────────
    from baysbench.posteriors import (
        BetaPosterior,       # bool / 0-1 int outcomes
        NormalPosterior,     # float in [0,1] (BLEU, ROUGE, LLM-judge)
        DirichletPosterior,  # int category 0..K-1 (MCQ, classification)
        GammaPosterior,      # non-neg count/latency (tokens, ms)
    )

    # 4-choice MCQ — category 0 = correct answer
    bench = BayesianBenchmark(posterior_factory=lambda: DirichletPosterior(k=4))

    # Latency benchmark — lower is better
    bench = BayesianBenchmark(
        posterior_factory=lambda: GammaPosterior(higher_is_better=False)
    )

    # ── Framework adapters ───────────────────────────────────────────────
    from baysbench.adapters.huggingface    import hf_model, hf_dataset
    from baysbench.adapters.openai_compat  import openai_model
    from baysbench.adapters.anthropic_adapter import anthropic_model
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .benchmark import BayesianBenchmark, BenchmarkReport, TaskResult
from .core import BetaPosterior, is_non_discriminating, prob_a_beats_b
from .decorators import benchmark, suite
from .posteriors import DirichletPosterior, GammaPosterior, NormalPosterior, Posterior
from .ranking import BayesianRanker, RankingResult

__version__ = "0.4.0"

__all__ = [
    # High-level API
    "BayesianBenchmark",
    "BenchmarkReport",
    "TaskResult",
    # Decorators
    "benchmark",
    "suite",
    # Ranking
    "BayesianRanker",
    "RankingResult",
    # Posteriors
    "Posterior",
    "BetaPosterior",
    "NormalPosterior",
    "DirichletPosterior",
    "GammaPosterior",
    # Convenience functions
    "compare",
    "rank",
    # Core functions (backward compat)
    "prob_a_beats_b",
    "is_non_discriminating",
    # Version
    "__version__",
]


# ---------------------------------------------------------------------------
# Convenience functions — one-liner API
# ---------------------------------------------------------------------------


def compare(
    model_a: Callable[[Any], Any],
    model_b: Callable[[Any], Any],
    score_fn: Callable[[Any, Any], Any],
    dataset: Iterable[Any],
    *,
    name: str = "task",
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
    posterior_factory: Callable | None = None,
    verbose: bool = False,
) -> TaskResult:
    """Compare two models in one line — no class instantiation required.

    Creates a :class:`BayesianBenchmark` with the given settings, runs a
    single pairwise comparison, and returns the :class:`TaskResult`.

    Args:
        model_a: ``model_a(problem) -> response``
        model_b: ``model_b(problem) -> response``
        score_fn: ``score_fn(problem, response) -> value``
                  (``bool`` for binary, ``float`` for continuous)
        dataset: Iterable of problems.
        name: Task name for the result.
        confidence: Stopping threshold (default 0.95).
        skip_threshold: Non-discriminating threshold (default 0.85).
        min_samples: Minimum evaluations before early stopping (default 3).
        posterior_factory: Override the posterior (e.g. ``NormalPosterior``).
        verbose: Show a tqdm progress bar.

    Returns:
        :class:`TaskResult`

    Example::

        import baysbench

        result = baysbench.compare(
            model_a=gpt4_fn,
            model_b=llama_fn,
            score_fn=lambda p, r: r.strip() == p["answer"],
            dataset=problems,
        )
        print(result.winner, f"P(A>B)={result.p_a_beats_b:.3f}")
        print(result.to_dict())
    """
    bench = BayesianBenchmark(
        confidence=confidence,
        skip_threshold=skip_threshold,
        min_samples=min_samples,
        posterior_factory=posterior_factory,
    )
    return bench.compare(
        model_a,
        model_b,
        score_fn,
        dataset,
        name=name,
        verbose=verbose,
    )


def rank(
    models: dict[str, Callable[[Any], Any]] | list[tuple[str, Callable[[Any], Any]]],
    score_fn: Callable[[Any, Any], Any],
    dataset: Iterable[Any],
    *,
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 5,
    posterior_factory: Callable | None = None,
    verbose: bool = False,
) -> RankingResult:
    """Rank multiple models in one line — no class instantiation required.

    Creates a :class:`BayesianRanker` with the given settings, registers all
    models, and returns the :class:`RankingResult`.

    Args:
        models: Either a ``dict`` mapping name → callable, or a list of
                ``(name, callable)`` pairs.  Dict order is preserved (Python
                3.7+).
        score_fn: ``score_fn(problem, response) -> value``
        dataset: Iterable of problems.
        confidence: Stopping threshold (default 0.95).
        skip_threshold: Non-discriminating threshold (default 0.85).
        min_samples: Minimum evaluations before early stopping (default 5).
        posterior_factory: Override the posterior.
        verbose: Show a tqdm progress bar.

    Returns:
        :class:`RankingResult`

    Example::

        import baysbench

        result = baysbench.rank(
            models={
                "gpt-4o":    gpt4_fn,
                "llama-3":   llama_fn,
                "mistral":   mistral_fn,
            },
            score_fn=lambda p, r: r.strip() == p["answer"],
            dataset=problems,
        )
        print(result.summary())
        df = result.to_dataframe()
    """
    ranker = BayesianRanker(
        confidence=confidence,
        skip_threshold=skip_threshold,
        min_samples=min_samples,
        posterior_factory=posterior_factory,
    )
    model_list = list(models.items()) if isinstance(models, dict) else list(models)
    for model_name, fn in model_list:
        ranker.add_model(model_name, fn)
    return ranker.rank(dataset, score_fn, verbose=verbose)
