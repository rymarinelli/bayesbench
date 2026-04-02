"""baysbench — Bayesian sequential benchmarking for LLMs and agents.

Features
--------
- **Early stopping**: Halts evaluation once posterior evidence crosses a
  confidence threshold — reducing cost by up to 99% vs full-dataset runs.
- **Pluggable posteriors**: Beta-Bernoulli for binary outcomes, Normal-Inverse-
  Gamma for continuous scores (BLEU, ROUGE, LLM-judge), or bring your own.
- **Framework adapters**: HuggingFace Inference API, OpenAI-compatible APIs
  (Groq, Together AI, vLLM, Ollama, …), and Anthropic — all via the same
  decorator API.
- **Multi-model ranking**: Rank N models simultaneously with Bayesian early
  stopping, without running O(N²) pairwise comparisons.
- **Three decorator styles**: ``@benchmark``, ``@bench.task``, ``@suite``.
- **Async-first**: All operations support ``async/await``.

Quick start::

    from baysbench import BayesianBenchmark, benchmark, suite, BayesianRanker

    # ── @benchmark — one-shot ────────────────────────────────────────────
    @benchmark(model_a=big_llm, model_b=small_llm, dataset=problems)
    def exact_match(problem, response):
        return response.strip() == problem["answer"]

    result = exact_match.run()
    print(result.winner, result.efficiency)

    # ── @bench.task — multi-task ─────────────────────────────────────────
    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=problems, name="gsm8k")
    def gsm8k(problem):
        return big_llm(problem["q"]) == problem["a"], \\
               small_llm(problem["q"]) == problem["a"]

    print(bench.run().summary())

    # ── @suite — class-based ─────────────────────────────────────────────
    @suite(confidence=0.95)
    class MyEval:
        dataset = problems
        def task_math(problem): ...

    print(MyEval.run().summary())

    # ── BayesianRanker — rank N models ───────────────────────────────────
    ranker = BayesianRanker(confidence=0.95)
    ranker.add_model("gpt-4o",   gpt4_fn)
    ranker.add_model("llama-3",  llama_fn)
    ranker.add_model("mistral",  mistral_fn)

    result = ranker.rank(dataset=problems, score_fn=lambda p, r: r == p["a"])
    print(result.summary())

    # ── Continuous scores ────────────────────────────────────────────────
    from baysbench.posteriors import NormalPosterior

    @benchmark(
        model_a=big_llm, model_b=small_llm, dataset=problems,
        posterior_factory=NormalPosterior,
    )
    def bleu(problem, response):
        return compute_bleu(response, problem["reference"])   # float

    # ── Framework adapters ───────────────────────────────────────────────
    from baysbench.adapters.huggingface  import hf_model, hf_dataset
    from baysbench.adapters.openai_compat import openai_model
    from baysbench.adapters.anthropic_adapter import anthropic_model
"""

from .benchmark import BayesianBenchmark, BenchmarkReport, TaskResult
from .core import BetaPosterior, is_non_discriminating, prob_a_beats_b
from .decorators import benchmark, suite
from .posteriors import NormalPosterior, Posterior
from .ranking import BayesianRanker, RankingResult

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
    # Core functions (backward compat)
    "prob_a_beats_b",
    "is_non_discriminating",
]
