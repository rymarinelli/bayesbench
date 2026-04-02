"""multi_model_ranking.py — rank N models simultaneously with early stopping.

Demonstrates BayesianRanker with:
  - add_model() API
  - @ranker.evaluate decorator
  - Continuous scores via NormalPosterior
  - Async ranking

Run with:
    python examples/multi_model_ranking.py
"""
import asyncio
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baysbench import BayesianRanker
from baysbench.posteriors import NormalPosterior

random.seed(42)

PROBLEMS = [{"question": f"{a}+{b}", "answer": str(a + b)}
            for a, b in [(random.randint(1, 100), random.randint(1, 100))
                         for _ in range(300)]]


# ---------------------------------------------------------------------------
# Mock models with different accuracy levels
# ---------------------------------------------------------------------------

def model_90pct(problem):
    return problem["answer"] if random.random() < 0.90 else "WRONG"

def model_75pct(problem):
    return problem["answer"] if random.random() < 0.75 else "WRONG"

def model_55pct(problem):
    return problem["answer"] if random.random() < 0.55 else "WRONG"

def model_30pct(problem):
    return problem["answer"] if random.random() < 0.30 else "WRONG"


# ===========================================================================
# Style 1 — add_model() + score_fn kwarg
# ===========================================================================
print("=" * 60)
print("Style 1: add_model() + rank(score_fn=...)")
print("=" * 60)

ranker = BayesianRanker(confidence=0.95, min_samples=5)
ranker.add_model("model-90", model_90pct)
ranker.add_model("model-75", model_75pct)
ranker.add_model("model-55", model_55pct)
ranker.add_model("model-30", model_30pct)

result = ranker.rank(
    dataset=PROBLEMS,
    score_fn=lambda p, r: r == p["answer"],
)
print(result.summary())
print()


# ===========================================================================
# Style 2 — @ranker.evaluate decorator
# ===========================================================================
print("=" * 60)
print("Style 2: @ranker.evaluate decorator")
print("=" * 60)

ranker2 = BayesianRanker(confidence=0.95, min_samples=5)
ranker2.add_model("model-90", model_90pct)
ranker2.add_model("model-75", model_75pct)
ranker2.add_model("model-55", model_55pct)


@ranker2.evaluate
def exact_match(problem, response):
    return response == problem["answer"]


result2 = ranker2.rank(dataset=PROBLEMS)
print(result2.summary())
print()


# ===========================================================================
# Style 3 — Continuous scores (BLEU-like) with NormalPosterior
# ===========================================================================
print("=" * 60)
print("Style 3: NormalPosterior for continuous scores")
print("=" * 60)


def bleu_like_score(problem, response):
    """Fake BLEU score: 1.0 if exact, else Levenshtein-ish fraction."""
    if response == problem["answer"]:
        return 1.0
    # Simulate partial credit: longer shared prefix = higher score
    expected = problem["answer"]
    shared = sum(a == b for a, b in zip(response, expected))
    return shared / max(len(expected), 1)


def model_high_bleu(problem):
    """Mostly correct, rarely drops a digit."""
    ans = problem["answer"]
    if random.random() < 0.85:
        return ans
    # Flip last digit
    return ans[:-1] + str((int(ans[-1]) + 1) % 10) if ans else "0"


def model_low_bleu(problem):
    return problem["answer"] if random.random() < 0.40 else "0"


ranker3 = BayesianRanker(
    confidence=0.95,
    min_samples=5,
    posterior_factory=NormalPosterior,
)
ranker3.add_model("high-bleu", model_high_bleu)
ranker3.add_model("low-bleu", model_low_bleu)

result3 = ranker3.rank(dataset=PROBLEMS, score_fn=bleu_like_score)
print(result3.summary())
print()


# ===========================================================================
# Style 4 — Async ranking with async model callables
# ===========================================================================
print("=" * 60)
print("Style 4: async ranking")
print("=" * 60)


async def async_model_90(problem):
    return problem["answer"] if random.random() < 0.90 else "WRONG"


async def async_model_60(problem):
    return problem["answer"] if random.random() < 0.60 else "WRONG"


async def main():
    ranker4 = BayesianRanker(confidence=0.95, min_samples=5)
    ranker4.add_model("async-90", async_model_90)
    ranker4.add_model("async-60", async_model_60)

    result4 = await ranker4.rank_async(
        dataset=PROBLEMS,
        score_fn=lambda p, r: r == p["answer"],
    )
    print(result4.summary())


asyncio.run(main())
