"""Gamma-Poisson conjugate posterior for count and rate data.

Use this when observations are non-negative counts or durations:
- Token count per response ("Model A generates more concise answers")
- Response latency in milliseconds ("Model A is faster")
- Word count, character count, number of API calls
- Any non-negative, approximately Poisson-distributed quantity
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Posterior


class GammaPosterior(Posterior):
    """Gamma-Poisson conjugate model for non-negative count or rate observations.

    Places a Gamma prior on the Poisson rate ``λ``.  The posterior after
    observing counts ``x₁, …, xₙ`` is:

        λ | data ~ Gamma(α₀ + Σxᵢ, β₀ + n)

    For comparing models use ``higher_is_better``:

    - ``True`` (default): higher rate = better.
      ``prob_beats`` returns P(λ_A > λ_B) — useful for throughput, recall.
    - ``False``: lower rate = better.
      ``prob_beats`` returns P(λ_A < λ_B) — useful for latency, token cost.

    Args:
        alpha_0: Gamma prior shape (> 0).  Defaults to ``1.0`` (weak prior).
        beta_0: Gamma prior rate (> 0).  Defaults to ``1.0``.
        higher_is_better: Direction for :meth:`prob_beats`.

    Usage::

        from baysbench.posteriors import GammaPosterior

        # Latency benchmark: lower is better
        latency_post = GammaPosterior(higher_is_better=False)
        for ms in [120, 85, 200, 95]:
            latency_post.observe_one(ms)

        # Token-count benchmark: lower is better (more concise)
        token_post = GammaPosterior(higher_is_better=False)
        for count in [42, 55, 38]:
            token_post.observe_one(count)

        # Rate benchmark: higher is better
        rate_post = GammaPosterior(higher_is_better=True)
        rate_post.observe_one(150)

        print(f"Mean rate: {latency_post.mean:.1f} ms")
        lo, hi = latency_post.credible_interval()
        print(f"95% CI: [{lo:.1f}, {hi:.1f}]")
    """

    def __init__(
        self,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        higher_is_better: bool = True,
    ) -> None:
        if alpha_0 <= 0:
            raise ValueError("alpha_0 must be positive")
        if beta_0 <= 0:
            raise ValueError("beta_0 must be positive")
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._alpha = alpha_0
        self._beta = beta_0
        self._n = 0
        self.higher_is_better = higher_is_better

    @property
    def n(self) -> float:
        return float(self._n)

    def observe_one(self, value: float | bool) -> None:
        """Observe a count or duration measurement.

        Args:
            value: Non-negative numeric observation (token count, latency ms, …).
                   ``bool`` is coerced to 0/1.
        """
        x = float(value)
        if x < 0:
            raise ValueError(f"GammaPosterior requires non-negative observations, got {x}")
        self._alpha += x
        self._beta += 1.0
        self._n += 1

    @property
    def mean(self) -> float:
        """Posterior mean rate E[λ] = α / β."""
        return self._alpha / self._beta

    def prob_beats(self, other: Posterior, n_samples: int = 10_000) -> float:
        """Probability that this model's rate is "better" than ``other``.

        If ``higher_is_better`` is ``True``: returns P(λ_self > λ_other).
        If ``higher_is_better`` is ``False``: returns P(λ_self < λ_other).

        Args:
            other: Another :class:`GammaPosterior`.
            n_samples: Number of Monte Carlo samples.
        """
        if not isinstance(other, GammaPosterior):
            raise TypeError("GammaPosterior.prob_beats requires another GammaPosterior")
        rng = np.random.default_rng(42)
        samples_a = rng.gamma(self._alpha, 1.0 / self._beta, size=n_samples)
        samples_b = rng.gamma(other._alpha, 1.0 / other._beta, size=n_samples)
        if self.higher_is_better:
            return float((samples_a > samples_b).mean())
        return float((samples_a < samples_b).mean())

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Central credible interval for the Poisson rate ``λ``."""
        lo = (1.0 - ci) / 2.0
        dist = stats.gamma(self._alpha, scale=1.0 / self._beta)
        return float(dist.ppf(lo)), float(dist.ppf(1.0 - lo))

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` rate samples from the Gamma posterior.

        Returns:
            Array of shape ``(n,)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.gamma(self._alpha, 1.0 / self._beta, size=n)
