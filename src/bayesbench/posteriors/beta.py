"""Beta-Bernoulli conjugate posterior for binary outcomes.

Use this when each evaluation produces a binary correct/incorrect outcome,
e.g. exact-match, pass/fail unit tests, multiple-choice accuracy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special, stats

from .base import Posterior

# 48-point Gauss-Legendre nodes mapped to [0, 1]. P(A > B) is computed with a
# split quadrature: the left half uses x = u²/2 and the right half uses
# x = 1 - v²/2, which removes the x^{a-1} / (1-x)^{b-1} endpoint singularities
# of the Beta density and lets fixed quadrature reach ~1e-8 accuracy in ~30µs
# (about 280x faster than adaptive scipy.integrate.quad).
_GL_X, _GL_W = leggauss(48)
_GL_U = 0.5 * (_GL_X + 1.0)
_GL_WU = 0.5 * _GL_W


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

    def observe(self, success: bool) -> BetaPosterior:
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

    def prob_beats(
        self,
        other: Posterior,
        n_samples: int = 10_000,
        rng: Any = None,  # noqa: ANN401, ARG002
    ) -> float:
        """Compute P(self accuracy > other accuracy) via numerical integration.

        Uses the closed-form identity:
            P(X > Y) = ∫₀¹ f_A(x) · F_B(x) dx

        Evaluated with fixed Gauss-Legendre quadrature after an endpoint-
        singularity-removing substitution (see module docstring); accurate to
        ~1e-8 and deterministic (no Monte Carlo, no RNG).

        The ``n_samples`` and ``rng`` arguments are accepted for API
        compatibility but ignored.
        """
        if not isinstance(other, BetaPosterior):
            raise TypeError("BetaPosterior.prob_beats expects another BetaPosterior")

        ln_b = special.betaln(self.alpha, self.beta)
        a2, b2 = other.alpha, other.beta

        x_left = 0.5 * _GL_U * _GL_U
        pdf_left = np.exp(
            (self.alpha - 1.0) * np.log(x_left) + (self.beta - 1.0) * np.log1p(-x_left) - ln_b
        )
        s_left = float(np.sum(_GL_WU * pdf_left * special.betainc(a2, b2, x_left) * _GL_U))

        x_right = 1.0 - 0.5 * _GL_U * _GL_U
        pdf_right = np.exp(
            (self.alpha - 1.0) * np.log(x_right) + (self.beta - 1.0) * np.log1p(-x_right) - ln_b
        )
        s_right = float(np.sum(_GL_WU * pdf_right * special.betainc(a2, b2, x_right) * _GL_U))

        return float(np.clip(s_left + s_right, 0.0, 1.0))

    def prob_beats_value(self, value: float, n_samples: int = 10_000) -> float:  # noqa: ARG002
        """Compute P(self accuracy > ``value``) exactly via the Beta CDF.

        ``n_samples`` is accepted for API compatibility and ignored.
        """
        value = float(np.clip(value, 0.0, 1.0))
        return float(1.0 - special.betainc(self.alpha, self.beta, value))

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` samples from the posterior.

        Args:
            n: Number of samples.
            rng: Optional Generator for reproducible draws.
        """
        if rng is None:
            return np.random.beta(self.alpha, self.beta, size=n)
        return rng.beta(self.alpha, self.beta, size=n)

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"BetaPosterior(alpha={self.alpha:.2f}, beta={self.beta:.2f}, "
            f"mean={self.mean:.3f}, 95%CI=[{lo:.3f}, {hi:.3f}])"
        )
