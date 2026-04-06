"""llm_comparison.py — async example for comparing two LLM API endpoints.

Demonstrates how to use bayesbench with async model callables (e.g. the
Anthropic or OpenAI Python SDKs).

This file shows the *structure* — replace the stub calls with real SDK calls.

Run with:
    python examples/llm_comparison.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bayesbench import BayesianBenchmark

# ---------------------------------------------------------------------------
# Dataset (replace with a real JSONL / HuggingFace dataset loader)
# ---------------------------------------------------------------------------
MMLU_SAMPLE = [
    {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": "4"},
    {"question": "Capital of France?", "choices": ["Berlin", "Paris", "Rome", "Madrid"], "answer": "Paris"},
    # ... load your full dataset here
]


# ---------------------------------------------------------------------------
# Async model stubs — replace with real SDK calls
# ---------------------------------------------------------------------------

async def call_model_a(problem: dict) -> str:
    """Stub: call your larger / more capable model."""
    await asyncio.sleep(0)          # Replace with: await anthropic_client.messages.create(...)
    return problem["answer"]        # perfect for demo purposes


async def call_model_b(problem: dict) -> str:
    """Stub: call your smaller / cheaper model."""
    await asyncio.sleep(0)
    # Simulate 70% accuracy
    import random
    return problem["answer"] if random.random() < 0.70 else problem["choices"][0]


def score(problem: dict, response: str) -> bool:
    return response.strip() == problem["answer"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main():
    bench = BayesianBenchmark(confidence=0.95, min_samples=3)

    result = await bench.compare_async(
        model_a=call_model_a,
        model_b=call_model_b,
        score_fn=score,
        dataset=MMLU_SAMPLE * 50,   # expand sample for demo
        name="mmlu_mc",
    )

    print(result)
    print(f"\nWinner: {result.winner}")
    print(f"Problems evaluated: {result.problems_tested}/{result.total_problems}")
    print(f"Cost reduction: {result.efficiency:.1%}")
    print(f"Model A accuracy: {result.posterior_a.mean:.2%}")
    print(f"Model B accuracy: {result.posterior_b.mean:.2%}")
    lo_a, hi_a = result.posterior_a.credible_interval()
    lo_b, hi_b = result.posterior_b.credible_interval()
    print(f"Model A 95% CI: [{lo_a:.2%}, {hi_a:.2%}]")
    print(f"Model B 95% CI: [{lo_b:.2%}, {hi_b:.2%}]")


if __name__ == "__main__":
    asyncio.run(main())
