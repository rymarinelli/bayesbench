"""Decision rules for sequential benchmarking.

A decision rule consumes paired per-problem observations one at a time and
returns a :class:`SequentialDecision`. Rules are stateful: the benchmark
engine creates one fresh rule instance per task run.

Two rules ship out of the box:

- :class:`PosteriorThresholdRule` — posterior probability P(A > B) crossing
  a confidence threshold. Powerful and cheap, but the false-decision rate
  depends on the stopping horizon (see the calibration harness). Treat the
  threshold as an evidence weight, not a frequentist error rate, unless you
  calibrate it for your dataset size.

- :class:`ConfidenceSequenceRule` — Ville-valid confidence sequences
  (mixture likelihood ratios, Waudby-Smith & Ramdas style). Guarantees
  P(wrong winner at *any* stopping time) <= alpha by construction, with no
  calibration and no horizon assumptions, at the cost of requiring more
  samples. Binary outcomes only.

- :class:`PairedDifferenceRule` — for paired evaluations where both models
  answer the same problems: models the per-item score *difference*, so
  shared item difficulty does not get treated as independent evidence.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

from .posteriors.base import Posterior
from .posteriors.beta import BetaPosterior

_EPS = 1e-12


class DecisionStatus(str, Enum):
    """Terminal and non-terminal states of a sequential comparison."""

    WINNER_A = "winner_a"
    WINNER_B = "winner_b"
    EQUIVALENT = "equivalent"
    INCONCLUSIVE = "inconclusive"


@dataclass
class SequentialDecision:
    """Outcome of feeding one paired observation into a decision rule."""

    status: DecisionStatus
    p_a_beats_b: float
    details: str = ""

    @property
    def terminal(self) -> bool:
        """True when the run should stop after this observation."""
        return self.status is not DecisionStatus.INCONCLUSIVE

    def __str__(self) -> str:
        return f"{self.status.value} (p={self.p_a_beats_b:.4f})"


@dataclass
class StepTrace:
    """One update step of a sequential comparison (for live UI / notebooks).

    Records the per-problem scores, the current evidence, and whether the
    run ended at this step.
    """

    step: int
    score_a: Any
    score_b: Any
    p_a_beats_b: float
    status: DecisionStatus
    is_terminal: bool = False
    terminal_reason: str = ""


class DecisionRule(ABC):
    """Abstract base for sequential decision rules.

    Subclasses hold per-run state. ``observe`` feeds one paired observation
    and returns the current decision. Concrete rules expose ``post_a`` and
    ``post_b`` (the posterior pair backing the decision, used for reporting);
    the paired rule exposes the difference posterior as ``post``.
    """

    post_a: Posterior
    post_b: Posterior

    @abstractmethod
    def observe(self, value_a: Any, value_b: Any) -> SequentialDecision:
        """Update the rule state with one paired observation.

        Args:
            value_a: Score of model A on this problem.
            value_b: Score of model B on this problem.
        """


class PosteriorThresholdRule(DecisionRule):
    """Posterior P(A > B) threshold rule with optional ROPE equivalence.

    Args:
        confidence: Declare a winner when P(A>B) >= confidence (A) or
                    <= 1-confidence (B).
        min_samples: Minimum paired observations before any terminal decision.
        skip_threshold: Legacy P(A>B)-window "skip" heuristic. Disabled by
                        default (None). When set below 1.0, a P(A>B) inside
                        (1-threshold, threshold) is reported as EQUIVALENT.
                        Deprecated: it converts insufficient early evidence
                        into premature terminal decisions.
        equivalence_margin: Optional ROPE half-width. When set, the rule
                            declares EQUIVALENT once the posterior mass of
                            |theta_a - theta_b| <= margin reaches confidence
                            (estimated by seeded Monte Carlo).
        posterior_factory: Factory for fresh posteriors (default Beta).
        rng: Seed or Generator for equivalence Monte Carlo estimation.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        min_samples: int = 30,
        skip_threshold: float | None = None,
        equivalence_margin: float | None = None,
        posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
        rng: np.random.Generator | int | None = None,
        equivalence_samples: int = 4_000,
    ) -> None:
        if not (0.5 < confidence <= 1.0):
            raise ValueError("confidence must be in (0.5, 1.0]")
        if skip_threshold is not None and not (0.5 < skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in (0.5, 1.0]")
        if equivalence_margin is not None and equivalence_margin < 0:
            raise ValueError("equivalence_margin must be non-negative")
        self.confidence = confidence
        self.min_samples = min_samples
        self.skip_threshold = skip_threshold
        self.equivalence_margin = equivalence_margin
        self.equivalence_samples = equivalence_samples
        self.factory: Callable[[], Posterior] = posterior_factory or BetaPosterior
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng or np.random.default_rng()
        self.post_a: Posterior = self.factory()
        self.post_b: Posterior = self.factory()
        self.tested = 0
        self.last_decision: SequentialDecision | None = None

    def observe(self, value_a: Any, value_b: Any) -> SequentialDecision:
        self.post_a.observe_one(value_a)
        self.post_b.observe_one(value_b)
        self.tested += 1

        if self.tested < self.min_samples:
            decision = SequentialDecision(DecisionStatus.INCONCLUSIVE, 0.5)
            self.last_decision = decision
            return decision

        p = self.post_a.prob_beats(self.post_b, rng=self.rng)

        if self.skip_threshold is not None and self.skip_threshold < 1.0:
            if (1.0 - self.skip_threshold) < p < self.skip_threshold:
                decision = SequentialDecision(
                    DecisionStatus.EQUIVALENT,
                    p,
                    "legacy skip heuristic: P(A>B) inside the skip window",
                )
                self.last_decision = decision
                return decision

        if p >= self.confidence:
            decision = SequentialDecision(DecisionStatus.WINNER_A, p)
            self.last_decision = decision
            return decision
        if p <= 1.0 - self.confidence:
            decision = SequentialDecision(DecisionStatus.WINNER_B, p)
            self.last_decision = decision
            return decision

        if self.equivalence_margin is not None:
            if self._rope_mass(self.equivalence_margin) >= self.confidence:
                decision = SequentialDecision(
                    DecisionStatus.EQUIVALENT,
                    p,
                    f"posterior mass inside |theta_a - theta_b| <= "
                    f"{self.equivalence_margin} reached {self.confidence}",
                )
                self.last_decision = decision
                return decision

        decision = SequentialDecision(DecisionStatus.INCONCLUSIVE, p)
        self.last_decision = decision
        return decision

    def _rope_mass(self, margin: float) -> float:
        """Estimate P(|theta_a - theta_b| <= margin) by seeded Monte Carlo."""
        samples_a = np.asarray(self.post_a.sample(self.equivalence_samples))
        samples_b = np.asarray(self.post_b.sample(self.equivalence_samples))
        return float(np.mean(np.abs(samples_a - samples_b) <= margin))


def _log_mix_marginal(s: int, t: int) -> float:
    """Log of the Beta-Binomial marginal of a uniform-prior mixture.

    f_mix(sequence with s successes in t trials) = 1 / ((t + 1) * C(t, s)).
    """
    if t == 0:
        return 0.0
    return -(math.log(t + 1) + gammaln(t + 1) - gammaln(s + 1) - gammaln(t - s + 1))


def _log_e_value(s: int, t: int, theta: float) -> float:
    """log E_t(theta) for a Bernoulli stream with s successes in t trials.

    E_t(theta) = f_mix(x_1..x_t) / f_theta(x_1..x_t) is an e-value for the
    point null theta: E_theta[E_t] = 1 for every t, so by Ville's inequality
    P(sup_t E_t >= 1/alpha) <= alpha.
    """
    if t == 0:
        return 0.0
    return _log_mix_marginal(s, t) - (s * math.log(theta) + (t - s) * math.log1p(-theta))


def _cs_bounds(s: int, t: int, alpha_side: float) -> tuple[float, float]:
    """Confidence sequence bounds for a Bernoulli mean at level alpha_side.

    The CS is {theta : E_t(theta) < 1/alpha_side}; because log E_t is convex
    in logit(theta) the set is an interval whose endpoints are found by
    root-finding the two crossings of log E_t(theta) = -log(alpha_side).
    """
    if t == 0:
        return 0.0, 1.0
    log_c = -math.log(alpha_side)

    def g(theta: float) -> float:
        return _log_e_value(s, t, theta) - log_c

    lo, hi = 0.0, 1.0
    if s > 0 and g(_EPS) > 0:
        lo = brentq(g, _EPS, s / t if s < t else 1.0 - _EPS)
    if s < t and g(1.0 - _EPS) > 0:
        hi = brentq(g, s / t if s > 0 else _EPS, 1.0 - _EPS)
    return lo, hi


class ConfidenceSequenceRule(DecisionRule):
    """Ville-valid any-time stopping rule for binary outcomes.

    Maintains one confidence sequence per model (uniform-prior mixture
    likelihood ratio, Waudby-Smith & Ramdas 2023) and stops when the bounds
    separate: CS_lo(A) > CS_hi(B) or CS_lo(B) > CS_hi(A).

    Guarantee: P(declare the wrong winner at *any* stopping time) <= alpha,
    regardless of horizon, data ordering, or optional stopping.

    Args:
        alpha: Total any-time false-decision probability (default 0.05).
               Split alpha/2 per side.
        min_samples: Minimum paired observations before any terminal decision.
        equivalence_margin: Optional ROPE half-width: declare EQUIVALENT when
                            CS_hi(B) - CS_lo(A) <= margin and
                            CS_hi(A) - CS_lo(B) <= margin. This event also has
                            any-time error control at level alpha.

    Only boolean (correct/incorrect) observations are supported.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_samples: int = 1,
        equivalence_margin: float | None = None,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        if equivalence_margin is not None and equivalence_margin < 0:
            raise ValueError("equivalence_margin must be non-negative")
        self.alpha = alpha
        self.alpha_side = alpha / 2.0
        self.min_samples = min_samples
        self.equivalence_margin = equivalence_margin
        self.s_a = 0
        self.t_a = 0
        self.s_b = 0
        self.t_b = 0
        self.tested = 0
        self.lo_a, self.hi_a = 0.0, 1.0
        self.lo_b, self.hi_b = 0.0, 1.0
        # Descriptive posterior (Jeffreys) for reporting P(A>B) only; the
        # decision authority stays with the confidence sequences.
        self.report_a = BetaPosterior()
        self.report_b = BetaPosterior()
        self.last_decision: SequentialDecision | None = None
        # Same objects, exposed under the DecisionRule attribute names so the
        # benchmark engine can build TaskResults without special-casing.
        self.post_a: Posterior = self.report_a
        self.post_b: Posterior = self.report_b

    def observe(self, value_a: Any, value_b: Any) -> SequentialDecision:
        if not isinstance(value_a, bool) or not isinstance(value_b, bool):
            raise TypeError(
                "ConfidenceSequenceRule supports boolean outcomes only; "
                "use PosteriorThresholdRule with NormalPosterior for "
                "continuous scores."
            )
        self.s_a += int(value_a)
        self.t_a += 1
        self.s_b += int(value_b)
        self.t_b += 1
        self.tested += 1
        self.report_a.observe_one(value_a)
        self.report_b.observe_one(value_b)

        self.lo_a, self.hi_a = _cs_bounds(self.s_a, self.t_a, self.alpha_side)
        self.lo_b, self.hi_b = _cs_bounds(self.s_b, self.t_b, self.alpha_side)

        if self.tested < self.min_samples:
            decision = SequentialDecision(DecisionStatus.INCONCLUSIVE, 0.5)
            self.last_decision = decision
            return decision

        if self.lo_a > self.hi_b:
            decision = SequentialDecision(
                DecisionStatus.WINNER_A,
                self.report_a.prob_beats(self.report_b),
                f"CS_lo(A)={self.lo_a:.4f} > CS_hi(B)={self.hi_b:.4f}",
            )
            self.last_decision = decision
            return decision
        if self.lo_b > self.hi_a:
            decision = SequentialDecision(
                DecisionStatus.WINNER_B,
                self.report_a.prob_beats(self.report_b),
                f"CS_lo(B)={self.lo_b:.4f} > CS_hi(A)={self.hi_a:.4f}",
            )
            self.last_decision = decision
            return decision

        if self.equivalence_margin is not None:
            margin = self.equivalence_margin
            if self.hi_b - self.lo_a <= margin and self.hi_a - self.lo_b <= margin:
                decision = SequentialDecision(
                    DecisionStatus.EQUIVALENT,
                    self.report_a.prob_beats(self.report_b),
                    f"CS difference inside ROPE margin {margin}",
                )
                self.last_decision = decision
                return decision

        decision = SequentialDecision(
            DecisionStatus.INCONCLUSIVE, self.report_a.prob_beats(self.report_b)
        )
        self.last_decision = decision
        return decision


class PairedDifferenceRule(DecisionRule):
    """Rule for paired evaluations: evidence is the per-item difference.

    Instead of two independent posterior streams, the rule models one
    posterior over per-item differences d_i = score_a - score_b (continuous)
    or over discordant pairs (binary), and decides via
    P(difference > 0) >= confidence, i.e. posterior mass above the
    no-difference point.

    Shared item difficulty affects both models together and therefore does
    not masquerade as independent confirming evidence.

    Args:
        confidence: Decision threshold on P(diff > center).
        min_samples: Minimum problems evaluated before any terminal decision.
        equivalence_margin: Optional ROPE half-width for EQUIVALENT.
        posterior_factory: Posterior family (Normal for continuous,
                           Beta for binary).
        rng: Seed or Generator for equivalence Monte Carlo estimation.
        equivalence_samples: MC samples for ROPE mass estimation.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        min_samples: int = 30,
        equivalence_margin: float | None = None,
        posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
        rng: np.random.Generator | int | None = None,
        equivalence_samples: int = 4_000,
    ) -> None:
        if not (0.5 < confidence <= 1.0):
            raise ValueError("confidence must be in (0.5, 1.0]")
        self.confidence = confidence
        self.min_samples = min_samples
        self.equivalence_margin = equivalence_margin
        self.equivalence_samples = equivalence_samples
        self.factory: Callable[[], Posterior] = posterior_factory or BetaPosterior
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)
        self.rng = rng or np.random.default_rng()
        self.post: Posterior = self.factory()
        self.tested = 0
        self._binary = isinstance(self.post, BetaPosterior)
        # Shape-compatible posteriors for TaskResult: the difference posterior
        # plus a zero-observation posterior of the same family.
        self.post_a: Posterior = self.post
        self.post_b: Posterior = self.factory()
        self.last_decision: SequentialDecision | None = None

    def observe(self, value_a: Any, value_b: Any) -> SequentialDecision:
        self.tested += 1
        if self._binary:
            if bool(value_a) and not bool(value_b):
                self.post.observe_one(True)
            elif bool(value_b) and not bool(value_a):
                self.post.observe_one(False)
        else:
            self.post.observe_one(float(value_a) - float(value_b))

        if self.tested < self.min_samples:
            decision = SequentialDecision(DecisionStatus.INCONCLUSIVE, 0.5)
            self.last_decision = decision
            return decision

        center = 0.5 if self._binary else 0.0
        p = self.post.prob_beats_value(center)

        if p >= self.confidence:
            decision = SequentialDecision(DecisionStatus.WINNER_A, p)
            self.last_decision = decision
            return decision
        if p <= 1.0 - self.confidence:
            decision = SequentialDecision(DecisionStatus.WINNER_B, p)
            self.last_decision = decision
            return decision

        if self.equivalence_margin is not None:
            mass = self._rope_mass(self.equivalence_margin)
            if mass >= self.confidence:
                decision = SequentialDecision(
                    DecisionStatus.EQUIVALENT,
                    p,
                    f"difference posterior mass inside ROPE "
                    f"{self.equivalence_margin} reached {self.confidence}",
                )
                self.last_decision = decision
                return decision

        decision = SequentialDecision(DecisionStatus.INCONCLUSIVE, p)
        self.last_decision = decision
        return decision

    def _rope_mass(self, margin: float) -> float:
        samples = np.asarray(self.post.sample(self.equivalence_samples))
        center = 0.5 if self._binary else 0.0
        return float(np.mean(np.abs(samples - center) <= margin))


def make_decision_rule(
    name: str,
    confidence: float = 0.95,
    min_samples: int = 30,
    skip_threshold: float | None = None,
    equivalence_margin: float | None = None,
    posterior_factory: Callable[[], Posterior] | type[Posterior] | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    rng: np.random.Generator | int | None = None,
) -> DecisionRule:
    """Build a fresh decision rule from a name and shared parameters.

    Args:
        name: "posterior" or "confidence_sequence".
        paired: Wrap the rule in paired-difference semantics.
    """
    if name == "posterior":
        rule: DecisionRule = PosteriorThresholdRule(
            confidence=confidence,
            min_samples=min_samples,
            skip_threshold=skip_threshold,
            equivalence_margin=equivalence_margin,
            posterior_factory=posterior_factory,
            rng=rng,
        )
    elif name == "confidence_sequence":
        rule = ConfidenceSequenceRule(
            alpha=alpha,
            min_samples=min_samples,
            equivalence_margin=equivalence_margin,
        )
    else:
        raise ValueError(f"Unknown decision rule: {name!r}")

    if paired:
        return PairedDifferenceRule(
            confidence=confidence,
            min_samples=min_samples,
            equivalence_margin=equivalence_margin,
            posterior_factory=posterior_factory,
            rng=rng,
        )
    return rule
