"""Beta-Bernoulli conjugate posterior for binary outcomes.

Use this when each evaluation produces a binary correct/incorrect outcome,
e.g. exact-match, pass/fail unit tests, multiple-choice accuracy.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate, stats

from .base import Posterior


class BetaPosterior(Posterior):
    """Beta-Bernoulli conjugate posterior for tracking binary accuracy.

    Uses Jeffreys prior Beta(0.5, 0.5) by default — non-informative and
    invariant under reparameterisation.

    Args:
        alpha: Prior + observed successes (default 0.5 for Jeffreys prior).
        beta:  Prior + observed failures  (default 0.5 for Jeffreys prior).

    Example::

        p = BetaPosterior()
        p.observe_one(True)   # model answered correctly
        p.observe_one(False)  # model answered incorrectly
        print(p.mean)         # 0.6
        print(p.credible_interval())  # (lo, hi)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        self.alpha = alpha
        self.beta = beta

    def observe_one(self, value: bool | float) -> None:  # noqa: FBT001
        """Update in-place. ``value`` is truthy = success, falsy = failure."""
        if value:
            self.alpha += 1
        else:
            self.beta += 1

    def observe(self, success: bool) -> "BetaPosterior":
        """Return a *new* updated posterior (immutable style)."""
        if success:
            return BetaPosterior(self.alpha + 1, self.beta)
        return BetaPosterior(self.alpha, self.beta + 1)

    def observe_batch(self, successes: int, total: int) -> None:
        """Update in-place from a batch of observations."""
        self.alpha += successes
        self.beta += total - successes

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        """Effective observations (subtracts Jeffreys prior mass of 1)."""
        return max(0.0, self.alpha + self.beta - 1.0)

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        dist = stats.beta(self.alpha, self.beta)
        tail = (1 - ci) / 2
        return float(dist.ppf(tail)), float(dist.ppf(1 - tail))

    def prob_beats(self, other: "BetaPosterior", n_samples: int = 10_000) -> float:  # noqa: ARG002
        """Compute P(self accuracy > other accuracy) via numerical integration.

        Uses the closed-form identity:
            P(X > Y) = ∫₀¹ f_A(x) · F_B(x) dx

        The ``n_samples`` argument is accepted for API compatibility but
        ignored — the integral is computed analytically.
        """

        def integrand(x: float) -> float:
            return stats.beta.pdf(x, self.alpha, self.beta) * stats.beta.cdf(
                x, other.alpha, other.beta
            )

        result, _ = integrate.quad(integrand, 0.0, 1.0, limit=100)
        return float(np.clip(result, 0.0, 1.0))

    def sample(self, n: int = 1) -> np.ndarray:
        """Draw ``n`` samples from the posterior."""
        return np.random.beta(self.alpha, self.beta, size=n)

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"BetaPosterior(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"mean={self.mean:.3f}, 95%CI=[{lo:.3f}, {hi:.3f}])"
        )
