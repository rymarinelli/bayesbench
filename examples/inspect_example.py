"""inspect_example.py — Bayesian benchmarking with AISI's Inspect framework.

Demonstrates how to use baysbench alongside inspect_ai to compare LLMs with
Bayesian sequential testing, cutting evaluation cost by up to 99%.

Requires:
    pip install baysbench[inspect]
    # Plus an API key for whichever models you want to test

Run with:
    python examples/inspect_example.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Stub dataset (replace with hf_dataset / csv_dataset / json_dataset calls)
# ---------------------------------------------------------------------------

class _Sample:
    """Minimal stub that matches inspect_ai.dataset.Sample's interface."""
    def __init__(self, input, target, choices=None, id=None, metadata=None):
        self.input = input
        self.target = target
        self.choices = choices
        self.id = id
        self.metadata = metadata or {}

STUB_SAMPLES = [
    _Sample("What is the capital of France?",   "Paris"),
    _Sample("What is 12 × 12?",                "144"),
    _Sample("Who wrote Hamlet?",               "Shakespeare"),
    _Sample("What element has symbol Au?",     "gold"),
    _Sample("What is the boiling point of water in Celsius?", "100"),
] * 40  # repeat to give the sequential test enough data


# ---------------------------------------------------------------------------
# Import adapter pieces
# ---------------------------------------------------------------------------

from baysbench import BayesianBenchmark, BayesianRanker, benchmark, suite
from baysbench.adapters.inspect_ai import (
    any_target_score,
    choice_score,
    exact_match_score,
    from_inspect_dataset,
    includes_score,
    inspect_model,
    inspect_model_async,
    pattern_score,
)

# Convert stub samples → baysbench problem dicts
problems = from_inspect_dataset(STUB_SAMPLES)
print(f"Loaded {len(problems)} problems from Inspect dataset\n")
print("Example problem:", problems[0], "\n")


# ===========================================================================
# Score functions (no API calls needed)
# ===========================================================================

print("=== Score function demos ===")
p = {"input": "Capital of France?", "target": "Paris",
     "all_targets": ["Paris", "paris"], "choices": None, "id": None, "metadata": {}}

print("exact_match_score:", exact_match_score(p, "paris"))          # True (case-insensitive)
print("includes_score:   ", includes_score(p, "The answer is Paris!"))  # True
print("any_target_score: ", any_target_score(p, "paris france"))    # True
print("pattern_score:    ", pattern_score(r"\bParis\b")(p, "Paris"))  # True
print()


# ===========================================================================
# Style 1 — @benchmark decorator with Inspect models
# ===========================================================================

print("=== Style 1: @benchmark decorator with inspect_model ===")
print("(skipping real API call — showing structure only)\n")

# Uncomment to run with real models:
# model_a = inspect_model("openai/gpt-4o",     system_prompt="Answer in one word.")
# model_b = inspect_model("openai/gpt-4o-mini", system_prompt="Answer in one word.")
#
# @benchmark(
#     model_a=model_a,
#     model_b=model_b,
#     dataset=problems,
#     name="general_knowledge",
#     confidence=0.95,
# )
# def score(problem, response):
#     return includes_score(problem, response)
#
# result = score.run()
# print(result)


# ===========================================================================
# Style 2 — BayesianBenchmark + @bench.task
# ===========================================================================

print("=== Style 2: BayesianBenchmark + @bench.task ===")
print("(skipping real API call — showing structure only)\n")

# bench = BayesianBenchmark(confidence=0.95)
#
# model_a = inspect_model("anthropic/claude-haiku-4-5")
# model_b = inspect_model("openai/gpt-4o-mini")
#
# @bench.task(dataset=problems, name="knowledge_qa")
# def knowledge_qa(problem):
#     return (
#         exact_match_score(problem, model_a(problem)),
#         exact_match_score(problem, model_b(problem)),
#     )
#
# report = bench.run()
# print(report.summary())


# ===========================================================================
# Style 3 — BayesianRanker to rank N Inspect models
# ===========================================================================

print("=== Style 3: BayesianRanker across N Inspect models ===")
print("(skipping real API call — showing structure only)\n")

# ranker = BayesianRanker(confidence=0.95, min_samples=10)
#
# for model_id in [
#     "openai/gpt-4o",
#     "openai/gpt-4o-mini",
#     "anthropic/claude-opus-4-6",
#     "anthropic/claude-haiku-4-5",
# ]:
#     ranker.add_model(model_id, inspect_model(model_id))
#
# result = ranker.rank(dataset=problems, score_fn=includes_score)
# print(result.summary())


# ===========================================================================
# Style 4 — Async with inspect_model_async
# ===========================================================================

print("=== Style 4: async via inspect_model_async ===")
print("(skipping real API call — showing structure only)\n")

# import asyncio
#
# async def main():
#     bench = BayesianBenchmark(confidence=0.95)
#     async_a = inspect_model_async("openai/gpt-4o")
#     async_b = inspect_model_async("openai/gpt-4o-mini")
#
#     result = await bench.compare_async(
#         model_a=async_a,
#         model_b=async_b,
#         score_fn=includes_score,
#         dataset=problems,
#         name="async_comparison",
#     )
#     print(result)
#
# asyncio.run(main())


# ===========================================================================
# Style 5 — Multiple-choice tasks with choice_score
# ===========================================================================

print("=== Style 5: multiple-choice with choice_score ===")

class _MCQSample:
    def __init__(self, question, choices, answer):
        self.input = question
        self.choices = choices
        self.target = answer
        self.id = None
        self.metadata = {}

MCQ_SAMPLES = [
    _MCQSample(
        "What is the capital of France?",
        ["Berlin", "Paris", "Rome", "Madrid"],
        "B",
    ),
    _MCQSample(
        "What is 2 + 2?",
        ["3", "4", "5", "6"],
        "B",
    ),
] * 40

mcq_problems = from_inspect_dataset(MCQ_SAMPLES)
print(f"MCQ problems loaded: {len(mcq_problems)}")
print("Choices field:", mcq_problems[0]["choices"])
print("Target (correct letter):", mcq_problems[0]["target"])

# Deterministic mock models
def model_correct(problem):
    return problem["target"]  # always picks the correct letter

def model_random(problem):
    return "A"  # always picks A

from baysbench import BayesianBenchmark
bench_mcq = BayesianBenchmark(confidence=0.95, min_samples=5)

@bench_mcq.task(dataset=mcq_problems, name="mcq")
def mcq_task(problem):
    return (
        choice_score(problem, model_correct(problem)),
        choice_score(problem, model_random(problem)),
    )

report = bench_mcq.run()
print("\n" + report.summary())
print("\nDone! Replace stub models with real inspect_model() calls.")
