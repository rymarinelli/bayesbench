"""quickstart.py — minimal end-to-end example using all three API styles.

Run with:
    python examples/quickstart.py
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baysbench import BayesianBenchmark, benchmark, suite

# ---------------------------------------------------------------------------
# Toy dataset: 200 arithmetic problems  {question: str, answer: str}
# ---------------------------------------------------------------------------
random.seed(42)
PROBLEMS = [
    {"question": f"{a} + {b}", "answer": str(a + b)}
    for a, b in [(random.randint(1, 100), random.randint(1, 100)) for _ in range(200)]
]


# ---------------------------------------------------------------------------
# Mock models — replace these with real LLM calls
# ---------------------------------------------------------------------------

def strong_model(problem: dict) -> str:
    """Gets arithmetic right 90% of the time."""
    correct = problem["answer"]
    return correct if random.random() < 0.90 else str(int(correct) + 1)


def weak_model(problem: dict) -> str:
    """Gets arithmetic right 55% of the time."""
    correct = problem["answer"]
    return correct if random.random() < 0.55 else str(int(correct) + 1)


def score(problem: dict, response: str) -> bool:
    return response.strip() == problem["answer"]


# ===========================================================================
# Style 1 — standalone @benchmark decorator
# ===========================================================================
print("=" * 60)
print("Style 1: standalone @benchmark decorator")
print("=" * 60)

@benchmark(
    model_a=strong_model,
    model_b=weak_model,
    dataset=PROBLEMS,
    confidence=0.95,
    min_samples=5,
    name="arithmetic",
)
def exact_match(problem: dict, response: str) -> bool:
    return response.strip() == problem["answer"]

result = exact_match.run()
print(result)
print()


# ===========================================================================
# Style 2 — BayesianBenchmark instance + @bench.task
# ===========================================================================
print("=" * 60)
print("Style 2: BayesianBenchmark instance + @bench.task")
print("=" * 60)

bench = BayesianBenchmark(confidence=0.95, skip_threshold=0.85, min_samples=5)


@bench.task(dataset=PROBLEMS, name="arithmetic")
def compare_arithmetic(problem: dict):
    a_correct = strong_model(problem) == problem["answer"]
    b_correct = weak_model(problem) == problem["answer"]
    return a_correct, b_correct


@bench.task(dataset=PROBLEMS[:50], name="easy_subset")
def compare_easy(problem: dict):
    # Identical models → expect non-discriminating / skip
    a_correct = strong_model(problem) == problem["answer"]
    b_correct = strong_model(problem) == problem["answer"]
    return a_correct, b_correct


report = bench.run()
print(report.summary())
print()


# ===========================================================================
# Style 3 — class-based @suite decorator
# ===========================================================================
print("=" * 60)
print("Style 3: class-based @suite decorator")
print("=" * 60)


@suite(confidence=0.95, min_samples=5)
class ArithmeticBenchmark:
    """Benchmark suite comparing strong vs. weak model on arithmetic tasks."""

    dataset = PROBLEMS

    @staticmethod
    def task_addition(problem: dict):
        a_correct = strong_model(problem) == problem["answer"]
        b_correct = weak_model(problem) == problem["answer"]
        return a_correct, b_correct

    @staticmethod
    def task_hard_cases(problem: dict):
        # Subset: only large numbers
        if int(problem["answer"]) < 100:
            return True, True  # trivially skip small problems
        a_correct = strong_model(problem) == problem["answer"]
        b_correct = weak_model(problem) == problem["answer"]
        return a_correct, b_correct


suite_report = ArithmeticBenchmark.run()
print(suite_report.summary())
