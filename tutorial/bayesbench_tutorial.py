"""Synthetic solvers and datasets for the BayesBench tutorial.

Everything here is pure Python and deterministic (seeded / hash-based), so
the tutorial runs on CPU with no API keys and no model downloads.

The point of the synthetic solvers is didactic: they let us *know* the
ground truth (which solver is really better) so we can judge the stopping
rule's decisions against reality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Synthetic dataset: two-digit arithmetic
# ---------------------------------------------------------------------------


def make_arithmetic_dataset(
    n: int = 400,
    seed: int = 0,
    filename: str | None = None,
) -> list[dict[str, str]]:
    """Generate *n* two-digit addition/subtraction problems.

    Each problem is a dict with a question, its correct answer, and a
    per-problem id used by the hash-based solvers.
    """
    rng_hash = hashlib.sha256(f"{seed}".encode())
    problems: list[dict[str, str]] = []
    for i in range(n):
        h = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
        a = int(h[0:2], 16) % 90 + 10
        b = int(h[2:4], 16) % 90 + 10
        if int(h[4], 16) % 2 == 0:
            q, ans = f"{a} + {b}", str(a + b)
        else:
            q, ans = f"{a} - {b}", str(a - b)
        problems.append({"id": i, "q": q, "a": ans})
    if filename is not None:
        path = DATA_DIR / filename
        path.write_text("\n".join(json.dumps(p) for p in problems))
    return problems


# ---------------------------------------------------------------------------
# Solvers (models)
# ---------------------------------------------------------------------------


def _p_for(problem: dict, target: float, seed: str) -> bool:
    """Deterministic pseudo-random success for a given problem + seed."""
    h = int(hashlib.sha256(f"{seed}:{problem['id']}".encode()).hexdigest(), 16)
    return (h % 1000) / 1000.0 < target


def perfect_solver(problem: dict) -> str:
    """Always returns the correct answer."""
    return problem["a"]


def noisy_solver(problem: dict, accuracy: float = 0.55, variant: str = "") -> str:
    """Returns the correct answer with probability ``accuracy``.

    Deterministic per problem (hash-based), so reruns are identical. Pass
    distinct ``variant`` values to build two *different* solvers with the
    same accuracy (needed to compare equal models that are not literally
    the same function).
    """
    if _p_for(problem, accuracy, seed=f"noisy-{accuracy}-{variant}"):
        return problem["a"]
    return "0"


def flaky_front_solver(problem: dict, good_first_n: int = 3) -> str:
    """Perfect on the first ``good_first_n`` problems, always wrong after.

    Demonstrates the order-sensitivity trap: a model that looks excellent
    when the dataset happens to start with problems it can solve.
    """
    if problem["id"] < good_first_n:
        return problem["a"]
    return "0"


def reverse_front_solver(problem: dict, wrong_first_n: int = 3) -> str:
    """Wrong on the first ``wrong_first_n`` problems, perfect after.

    The mirror image of :func:`flaky_front_solver`: a strong model that
    looks terrible when the dataset starts with problems it cannot solve.
    """
    if problem["id"] < wrong_first_n:
        return "0"
    return problem["a"]


def exact_match(problem: dict, response: str) -> bool:
    """Binary score: did the model produce the exact answer?"""
    return str(response).strip() == problem["a"]


# ---------------------------------------------------------------------------
# Continuous metric: simulated latency (lower is better)
# ---------------------------------------------------------------------------


def latency_solver(name: str, base: float, jitter: float = 0.02) -> callable:
    """Return a callable mapping a problem to a deterministic fake latency.

    ``base`` is the mean latency in seconds; jitter is hash-based noise.
    """

    def solve(problem: dict) -> float:
        h = int(hashlib.sha256(f"{name}:{problem['id']}".encode()).hexdigest(), 16)
        noise = ((h % 1000) / 1000.0 - 0.5) * 2.0 * jitter
        return round(base + noise, 4)

    return solve


def latency_score(problem: dict, response: float) -> float:
    """Pass-through score for a latency value (used with lower_is_better)."""
    return float(response)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_dataset(filename: str) -> list[dict[str, str]]:
    path = DATA_DIR / filename
    return [json.loads(line) for line in path.read_text().splitlines()]


def ensure_datasets() -> tuple[list[dict], list[dict]]:
    """Create (or load) the standard datasets used by the notebook."""
    small = DATA_DIR / "arithmetic_400.jsonl"
    big = DATA_DIR / "arithmetic_2000.jsonl"
    if not small.exists():
        make_arithmetic_dataset(400, seed=0, filename=small.name)
    if not big.exists():
        make_arithmetic_dataset(2000, seed=1, filename=big.name)
    return load_dataset(small.name), load_dataset(big.name)
