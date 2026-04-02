"""Core Bayesian statistical primitives for sequential LLM benchmarking.

Based on the Beta-Bernoulli conjugate framework from:
"Bayesian Sequential Testing for Efficient LLM Benchmarking"
(40th International Workshop on Statistical Modelling, Oslo 2026)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, stats


@dataclass
class BetaPosterior:
    """Beta-Bernoulli conjugate posterior for tracking model accuracy.

    Uses Jeffreys prior Beta(0.5, 0.5) by default, which is non-informative
    and invariant under reparameterization.

    Args:
        alpha: Successes + prior alpha (default 0.5 for Jeffreys).
        beta: Failures + prior beta (default 0.5 for Jeffreys).
    """

    alpha: float = 0.5
    beta: float = 0.5

    def observe(self, success: bool) -> "BetaPosterior":
        """Return a new updated posterior after observing one outcome."""
        if success:
            return BetaPosterior(self.alpha + 1, self.beta)
        return BetaPosterior(self.alpha, self.beta + 1)

    def observe_one(self, success: bool) -> None:
        """Update this posterior in-place after observing one outcome."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def observe_batch(self, successes: int, total: int) -> None:
        """Update in-place from a batch of observations."""
        self.alpha += successes
        self.beta += total - successes

    @property
    def mean(self) -> float:
        """Expected accuracy (posterior mean)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        """Effective number of observations (excluding prior)."""
        return self.alpha + self.beta - 1.0  # subtract Jeffreys prior mass

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Compute a highest-density credible interval.

        Args:
            ci: Credible mass, e.g. 0.95 for a 95% CI.

        Returns:
            (lower, upper) bounds.
        """
        dist = stats.beta(self.alpha, self.beta)
        tail = (1 - ci) / 2
        return float(dist.ppf(tail)), float(dist.ppf(1 - tail))

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"BetaPosterior(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"mean={self.mean:.3f}, 95%CI=[{lo:.3f}, {hi:.3f}])"
        )


def prob_a_beats_b(posterior_a: BetaPosterior, posterior_b: BetaPosterior) -> float:
    """Compute P(accuracy_A > accuracy_B) via numerical integration.

    Uses the identity:
        P(X > Y) = ∫₀¹ f_A(x) · F_B(x) dx
    where f_A is the Beta PDF of model A and F_B is the Beta CDF of model B.

    Args:
        posterior_a: Posterior for model A.
        posterior_b: Posterior for model B.

    Returns:
        Probability in [0, 1] that model A has higher true accuracy than model B.
    """
    def integrand(x: float) -> float:
        return stats.beta.pdf(x, posterior_a.alpha, posterior_a.beta) * stats.beta.cdf(
            x, posterior_b.alpha, posterior_b.beta
        )

    result, _ = integrate.quad(integrand, 0.0, 1.0, limit=100)
    return float(np.clip(result, 0.0, 1.0))


def is_non_discriminating(
    posterior_a: BetaPosterior,
    posterior_b: BetaPosterior,
    threshold: float = 0.85,
) -> bool:
    """Return True when a task cannot distinguish between the two models.

    A task is non-discriminating when neither model has a strong probability
    advantage, i.e. P(A > B) is close to 0.5. We skip such tasks early to
    save evaluation budget.

    Args:
        posterior_a: Posterior for model A.
        posterior_b: Posterior for model B.
        threshold: If P(A>B) ∈ (1-threshold, threshold) the task is skipped.
                   Default 0.85 means we skip when 0.15 < P < 0.85.

    Returns:
        True if the task should be skipped.
    """
    p = prob_a_beats_b(posterior_a, posterior_b)
    return (1.0 - threshold) < p < threshold
