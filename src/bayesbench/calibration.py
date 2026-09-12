"""Simulation harness for calibrating stopping-policy decision quality.

Evaluates false-decision rates, order sensitivity, and expected sample counts
across synthetic effect sizes. Use this to find safe defaults before teaching
or publishing performance claims.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .decision import (
    ConfidenceSequenceRule,
    DecisionStatus,
    PosteriorThresholdRule,
)
from .posteriors.base import Posterior
from .posteriors.beta import BetaPosterior

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CalibrationPoint:
    """Single row in a calibration sweep."""

    min_samples: int
    skip_threshold: float
    confidence: float
    n_runs: int
    false_winner_rate: float
    skipped_rate: float
    inconclusive_rate: float
    expected_samples: float
    possible_max_samples: int

    def __str__(self) -> str:
        return (
            f"min={self.min_samples} skip={self.skip_threshold:.2f} "
            f"conf={self.confidence:.2f}  "
            f"false_winner={self.false_winner_rate:.1%}  "
            f"skipped={self.skipped_rate:.1%}  "
            f"inconclusive={self.inconclusive_rate:.1%}  "
            f"E[samples]={self.expected_samples:.1f}/{self.possible_max_samples}"
        )


@dataclass
class CalibrationReport:
    """Results of a calibration sweep."""

    params_label: str
    points: list[CalibrationPoint]

    def format_table(self) -> str:
        header = (
            f"{'min_s':>6} {'skip':>6} {'conf':>6} "
            f"{'false':>7} {'skip%':>7} {'incon':>7} {'E[n]':>7}"
        )
        rows = [header, "-" * len(header)]
        for p in self.points:
            rows.append(
                f"{p.min_samples:>6} {p.skip_threshold:>6.2f} {p.confidence:>6.2f} "
                f"{p.false_winner_rate:>7.1%} {p.skipped_rate:>7.1%} "
                f"{p.inconclusive_rate:>7.1%} {p.expected_samples:>7.1f}"
            )
        return "\n".join(rows)


@dataclass
class EffectSweepPoint:
    """One row of an effect-size sweep: decision quality vs effect size."""

    acc_a: float
    acc_b: float
    min_samples: int
    confidence: float
    n_runs: int
    item_sd: float
    false_winner_rate: float
    correct_winner_rate: float
    inconclusive_rate: float
    expected_samples: float
    possible_max_samples: int

    def __str__(self) -> str:
        return (
            f"d={self.acc_a - self.acc_b:+.2f} min={self.min_samples:>3} "
            f"conf={self.confidence:.2f}  false={self.false_winner_rate:.2%}  "
            f"correct={self.correct_winner_rate:.2%}  "
            f"incon={self.inconclusive_rate:.2%}  "
            f"E[n]={self.expected_samples:.1f}/{self.possible_max_samples}"
        )


@dataclass
class EffectSweepReport:
    """Results of an effect-size sweep."""

    params_label: str
    points: list[EffectSweepPoint]

    def format_table(self) -> str:
        header = f"{'d':>6} {'min_s':>6} {'false':>7} {'correct':>8} " f"{'incon':>7} {'E[n]':>7}"
        rows = [header, "-" * len(header)]
        for p in self.points:
            rows.append(
                f"{p.acc_a - p.acc_b:>+6.2f} {p.min_samples:>6} "
                f"{p.false_winner_rate:>7.1%} {p.correct_winner_rate:>8.1%} "
                f"{p.inconclusive_rate:>7.1%} {p.expected_samples:>7.1f}"
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Exact enumeration
# ---------------------------------------------------------------------------


def enumerate_binary_outcomes(
    n: int,
    success_prob_a: float = 0.5,
    success_prob_b: float = 0.5,
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
) -> dict[str, Any]:
    """Exact enumeration of all 4^n pairwise binary outcome sequences.

    Each trial produces four possible pairs: (A=T,B=T), (A=T,B=F),
    (A=F,B=T), (A=F,B=F). With n=3 this gives 4^3 = 64 equally likely
    sequences under equal models. The function classifies how each
    sequence is handled by the stopping rules.
    """
    outcomes = [(True, True), (True, False), (False, True), (False, False)]
    sequences = list(itertools.product(outcomes, repeat=n))
    decisions: list[str] = []
    probabilities: list[float] = []

    for seq_pairs in sequences:
        rule = PosteriorThresholdRule(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
            posterior_factory=BetaPosterior,
        )
        for val_a, val_b in seq_pairs:
            dec = rule.observe(val_a, val_b)
            if dec.status is DecisionStatus.INCONCLUSIVE:
                continue
            if dec.status is DecisionStatus.EQUIVALENT:
                decisions.append("skipped")
            elif dec.status is DecisionStatus.WINNER_A:
                decisions.append("winner_a")
            else:
                decisions.append("winner_b")
            break
        else:
            decisions.append("inconclusive")

        seq_prob = 1.0
        for val_a, val_b in seq_pairs:
            pa = success_prob_a if val_a else (1.0 - success_prob_a)
            pb = success_prob_b if val_b else (1.0 - success_prob_b)
            seq_prob *= pa * pb
        probabilities.append(seq_prob)

    total_prob = sum(probabilities)
    skipped_prob = sum(p for p, d in zip(probabilities, decisions) if d == "skipped")
    winner_prob = sum(p for p, d in zip(probabilities, decisions) if d in ("winner_a", "winner_b"))
    incon_prob = sum(p for p, d in zip(probabilities, decisions) if d == "inconclusive")

    return {
        "sequences": list(zip(sequences, decisions, probabilities)),
        "n": n,
        "success_prob_a": success_prob_a,
        "success_prob_b": success_prob_b,
        "skipped_rate": skipped_prob / total_prob if total_prob > 0 else 0.0,
        "winner_rate": winner_prob / total_prob if total_prob > 0 else 0.0,
        "inconclusive_rate": incon_prob / total_prob if total_prob > 0 else 0.0,
        "total_sequences": len(sequences),
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


def simulate_pairwise(
    n: int = 10_000,
    true_acc_a: float = 0.5,
    true_acc_b: float = 0.5,
    rng: np.random.Generator | int | None = None,
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
    max_samples: int = 500,
    posterior_factory: Callable[[], Posterior] | None = None,
    item_sd: float = 0.0,
) -> dict[str, Any]:
    """Monte Carlo simulation of the pairwise stopping policy.

    Simulates *n* independent runs of the binary stopping loop. Each run draws
    Bernoulli outcomes from the true per-model accuracies. Reports empirical
    decision rates and expected sample counts.

    Args:
        n: Number of simulation runs.
        true_acc_a: True Bernoulli parameter for model A.
        true_acc_b: True Bernoulli parameter for model B.
        rng: Seed or pre-instantiated Generator.
        confidence: Stopping confidence threshold.
        skip_threshold: Non-discrimination skip window.
        min_samples: Minimum evaluations before any early stopping.
        max_samples: Budget cap per run; runs past it are inconclusive.
        posterior_factory: Posterior factory (default BetaPosterior).
        item_sd: Standard deviation of shared item difficulty. When > 0, each
                 step draws a shared difficulty z ~ N(0, item_sd) and both
                 models answer item i with P(correct) = clip(acc + z). This
                 induces positive correlation between the two outcome streams,
                 modeling paired evaluation on shared items. The marginal
                 expectation of each stream approximates ``true_acc`` (exact
                 when clipping never binds).

    Returns:
        Dict with decision counts, rates, and per-run samples drawn.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    winner_a = 0
    winner_b = 0
    skipped = 0
    inconclusive = 0
    false_decisions = 0
    samples_drawn: list[int] = []

    factory: Callable[[], Posterior] = posterior_factory or BetaPosterior

    for _ in range(n):
        rule = PosteriorThresholdRule(
            confidence=confidence,
            skip_threshold=skip_threshold,
            min_samples=min_samples,
            posterior_factory=factory,
        )

        for j in range(max_samples):
            if item_sd > 0.0:
                z = rng.normal(0.0, item_sd)
                p_a = float(np.clip(true_acc_a + z, 1e-4, 1.0 - 1e-4))
                p_b = float(np.clip(true_acc_b + z, 1e-4, 1.0 - 1e-4))
            else:
                p_a, p_b = true_acc_a, true_acc_b
            val_a = rng.random() < p_a
            val_b = rng.random() < p_b
            dec = rule.observe(val_a, val_b)
            tested = j + 1

            if dec.status is DecisionStatus.EQUIVALENT:
                skipped += 1
                samples_drawn.append(tested)
                break
            if dec.status is DecisionStatus.WINNER_A:
                winner_a += 1
                if true_acc_a <= true_acc_b:
                    false_decisions += 1
                samples_drawn.append(tested)
                break
            if dec.status is DecisionStatus.WINNER_B:
                winner_b += 1
                if true_acc_b <= true_acc_a:
                    false_decisions += 1
                samples_drawn.append(tested)
                break
        else:
            inconclusive += 1
            samples_drawn.append(max_samples)

    return {
        "winner_a": winner_a,
        "winner_b": winner_b,
        "skipped": skipped,
        "inconclusive": inconclusive,
        "samples_drawn": samples_drawn,
        "false_decisions": false_decisions,
        "runs": n,
        "true_acc_a": true_acc_a,
        "true_acc_b": true_acc_b,
        "item_sd": item_sd,
        "params": {
            "confidence": confidence,
            "skip_threshold": skip_threshold,
            "min_samples": min_samples,
        },
    }


# ---------------------------------------------------------------------------
# Order-sensitivity test
# ---------------------------------------------------------------------------


def simulate_pairwise_cs(
    n: int = 10_000,
    true_acc_a: float = 0.5,
    true_acc_b: float = 0.5,
    alpha: float = 0.05,
    min_samples: int = 1,
    max_samples: int = 500,
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """Monte Carlo simulation of the confidence-sequence stopping rule.

    Same protocol as :func:`simulate_pairwise` but driven by
    :class:`~bayesbench.decision.ConfidenceSequenceRule`, which guarantees
    P(wrong winner at any stopping time) <= alpha by Ville's inequality.
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif rng is None:
        rng = np.random.default_rng()

    winner_a = 0
    winner_b = 0
    equivalent = 0
    inconclusive = 0
    false_decisions = 0
    samples_drawn: list[int] = []

    for _ in range(n):
        rule = ConfidenceSequenceRule(alpha=alpha, min_samples=min_samples)
        for j in range(max_samples):
            val_a = rng.random() < true_acc_a
            val_b = rng.random() < true_acc_b
            dec = rule.observe(val_a, val_b)
            tested = j + 1

            if dec.status is DecisionStatus.INCONCLUSIVE:
                continue
            if dec.status is DecisionStatus.EQUIVALENT:
                equivalent += 1
            elif dec.status is DecisionStatus.WINNER_A:
                winner_a += 1
                if true_acc_a <= true_acc_b:
                    false_decisions += 1
            else:
                winner_b += 1
                if true_acc_b <= true_acc_a:
                    false_decisions += 1
            samples_drawn.append(tested)
            break
        else:
            inconclusive += 1
            samples_drawn.append(max_samples)

    return {
        "winner_a": winner_a,
        "winner_b": winner_b,
        "equivalent": equivalent,
        "inconclusive": inconclusive,
        "samples_drawn": samples_drawn,
        "false_decisions": false_decisions,
        "runs": n,
        "true_acc_a": true_acc_a,
        "true_acc_b": true_acc_b,
        "params": {
            "alpha": alpha,
            "min_samples": min_samples,
            "max_samples": max_samples,
        },
    }


# ---------------------------------------------------------------------------
# Order-sensitivity test
# ---------------------------------------------------------------------------


def simulate_order_sensitivity(
    favorable_front: int = 3,
    unfavorable_after: int = 100,
    n_runs: int = 1_000,
) -> dict[str, Any]:
    """Test whether dataset order can flip the decision.

    Simulates a dataset where the first *favorable_front* problems are easy
    for model A (always correct) and the rest are hard (always wrong). If
    the stopping rule fires after the favorable front, model A wins despite
    being wrong on the majority of the dataset.
    """
    factory: Callable[[], Posterior] = BetaPosterior

    early_wins = 0
    total_samples: list[int] = []

    for _ in range(n_runs):
        rule = PosteriorThresholdRule(
            confidence=0.95,
            skip_threshold=0.85,
            min_samples=3,
            posterior_factory=factory,
        )

        for _ in range(favorable_front):
            rule.observe(True, False)

        last = rule.last_decision
        if last is not None and last.status is DecisionStatus.WINNER_A:
            early_wins += 1
            total_samples.append(favorable_front)
            continue

        samples = favorable_front
        for _ in range(unfavorable_after):
            last = rule.observe(False, True)
            samples += 1
            if last.terminal:
                break
        total_samples.append(samples)

    return {
        "favorable_front": favorable_front,
        "unfavorable_after": unfavorable_after,
        "n_runs": n_runs,
        "early_wins": early_wins,
        "early_win_rate": early_wins / n_runs,
        "mean_samples": float(np.mean(total_samples)),
    }


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------


def calibrate_sweep(
    min_samples_grid: Sequence[int] = (3, 5, 10, 20, 50),
    skip_threshold_grid: Sequence[float] = (0.85, 0.95, 0.99, 1.0),
    confidence: float = 0.95,
    true_acc_a: float = 0.5,
    true_acc_b: float = 0.5,
    n_runs: int = 5_000,
    possible_max: int = 100,
    seed: int = 42,
) -> CalibrationReport:
    """Run a full calibration sweep across parameter grids.

    Returns a CalibrationReport with one CalibrationPoint per grid cell.
    """
    rng = np.random.default_rng(seed)
    points: list[CalibrationPoint] = []

    for min_s in min_samples_grid:
        for skip in skip_threshold_grid:
            result = simulate_pairwise(
                n=n_runs,
                true_acc_a=true_acc_a,
                true_acc_b=true_acc_b,
                confidence=confidence,
                skip_threshold=skip,
                min_samples=min_s,
                max_samples=possible_max,
                rng=rng,
            )

            total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
            false_rate = result["false_decisions"] / total if total > 0 else 0.0
            skip_rate = result["skipped"] / total if total > 0 else 0.0
            incon_rate = result["inconclusive"] / total if total > 0 else 0.0

            points.append(
                CalibrationPoint(
                    min_samples=min_s,
                    skip_threshold=skip,
                    confidence=confidence,
                    n_runs=n_runs,
                    false_winner_rate=false_rate,
                    skipped_rate=skip_rate,
                    inconclusive_rate=incon_rate,
                    expected_samples=(
                        float(np.mean(result["samples_drawn"])) if result["samples_drawn"] else 0.0
                    ),
                    possible_max_samples=possible_max,
                )
            )

    label = f"true_A={true_acc_a:.2f} true_B={true_acc_b:.2f} " f"conf={confidence:.2f} n={n_runs}"
    return CalibrationReport(params_label=label, points=points)


def effect_size_sweep(
    effects: Sequence[tuple[float, float]] = (
        (0.50, 0.50),
        (0.55, 0.50),
        (0.60, 0.50),
        (0.70, 0.50),
        (0.90, 0.50),
    ),
    min_samples_grid: Sequence[int] = (3, 5, 10, 20, 30, 50, 100),
    confidence: float = 0.95,
    skip_threshold: float = 1.0,
    n_runs: int = 2_000,
    max_samples: int = 200,
    item_sd: float = 0.0,
    seed: int = 42,
) -> EffectSweepReport:
    """Sweep effect sizes and min_samples: decision quality vs cost.

    For each (acc_a, acc_b) effect size and each min_samples, runs
    :func:`simulate_pairwise` and reports:

    - false_winner_rate: winner opposite the true ordering (under the null,
      any winner is false),
    - correct_winner_rate: winner matching the true ordering,
    - inconclusive_rate: budget exhausted without a decision,
    - expected_samples: mean samples consumed.

    Set ``skip_threshold=1.0`` to disable the skip heuristic, isolating the
    confidence-based stopping rule.
    """
    rng = np.random.default_rng(seed)
    points: list[EffectSweepPoint] = []

    for acc_a, acc_b in effects:
        for min_s in min_samples_grid:
            result = simulate_pairwise(
                n=n_runs,
                true_acc_a=acc_a,
                true_acc_b=acc_b,
                confidence=confidence,
                skip_threshold=skip_threshold,
                min_samples=min_s,
                max_samples=max_samples,
                item_sd=item_sd,
                rng=rng,
            )

            total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
            false_rate = result["false_decisions"] / total if total > 0 else 0.0
            correct_rate = 0.0
            if acc_a > acc_b:
                correct_rate = result["winner_a"] / total if total > 0 else 0.0
            elif acc_b > acc_a:
                correct_rate = result["winner_b"] / total if total > 0 else 0.0
            incon_rate = result["inconclusive"] / total if total > 0 else 0.0

            points.append(
                EffectSweepPoint(
                    acc_a=acc_a,
                    acc_b=acc_b,
                    min_samples=min_s,
                    confidence=confidence,
                    n_runs=n_runs,
                    item_sd=item_sd,
                    false_winner_rate=false_rate,
                    correct_winner_rate=correct_rate,
                    inconclusive_rate=incon_rate,
                    expected_samples=(
                        float(np.mean(result["samples_drawn"])) if result["samples_drawn"] else 0.0
                    ),
                    possible_max_samples=max_samples,
                )
            )

    label = (
        f"effects={effects} conf={confidence:.2f} n={n_runs} "
        f"item_sd={item_sd} skip_threshold={skip_threshold}"
    )
    return EffectSweepReport(params_label=label, points=points)
