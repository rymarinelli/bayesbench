"""Normal-Inverse-Gamma conjugate posterior for continuous scores.

Use this when each evaluation produces a continuous score in any range,
e.g. BLEU, ROUGE-L, semantic similarity (0–1), normalised perplexity,
Likert ratings divided by max-scale, LLM-judge scores.

Statistical model
-----------------
We use the Normal-Inverse-Gamma (NIG) conjugate prior:

    sigma² ~ InvGamma(alpha, beta)
    mu | sigma² ~ Normal(mu_0, sigma² / kappa)

Conjugate updates after observing x_1, ..., x_n::

    kappa_n  = kappa_0 + n
    mu_n     = (kappa_0 * mu_0 + n * x_bar) / kappa_n
    alpha_n  = alpha_0 + n / 2
    beta_n   = beta_0 + 0.5 * sum(x_i - x_bar)^2
               + (n * kappa_0 / (2 * kappa_n)) * (x_bar - mu_0)^2

The marginal posterior for mu is a Student-t distribution:
    mu | data ~ t(2 * alpha_n, mu_n, beta_n / (alpha_n * kappa_n))

P(mu_A > mu_B) is estimated via Monte Carlo sampling from both posteriors.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Posterior


class NormalPosterior(Posterior):
    """Normal-Inverse-Gamma conjugate posterior for continuous scores.

    Args:
        mu_0:    Prior mean (default 0.5 — centre of a [0, 1] score range).
        kappa_0: Prior pseudo-observations for the mean (strength of prior).
                 Increase to make the prior more informative.
        alpha_0: Shape of the inverse-gamma prior on variance (> 0).
        beta_0:  Scale of the inverse-gamma prior on variance (> 0).

    Example (BLEU scores)::

        p = NormalPosterior(mu_0=0.3)   # expect ~30% BLEU baseline
        p.observe_one(0.42)
        p.observe_one(0.38)
        print(p.mean)                   # ~0.39
        print(p.credible_interval())    # (lo, hi)

    Example (0–5 Likert, normalised)::

        p = NormalPosterior(mu_0=0.6)   # expect 3/5 baseline
        p.observe_one(4 / 5)
        p.observe_one(3 / 5)
    """

    def __init__(
        self,
        mu_0: float = 0.5,
        kappa_0: float = 1.0,
        alpha_0: float = 2.0,
        beta_0: float = 0.5,
    ) -> None:
        # Current NIG parameters (initialised to prior)
        self.mu_n = mu_0
        self.kappa_n = kappa_0
        self.alpha_n = alpha_0
        self.beta_n = beta_0
        # Store prior for reference / reset
        self._mu_0 = mu_0
        self._kappa_0 = kappa_0
        self._alpha_0 = alpha_0
        self._beta_0 = beta_0
        self._n = 0  # raw observation count

    def observe_one(self, value: float | bool) -> None:
        """Conjugate update from a single score observation."""
        x = float(value)
        kappa_old = self.kappa_n
        mu_old = self.mu_n

        self.kappa_n += 1
        self.mu_n = (kappa_old * mu_old + x) / self.kappa_n
        self.alpha_n += 0.5
        self.beta_n += (
            (kappa_old / (2.0 * self.kappa_n)) * (x - mu_old) ** 2
        )
        self._n += 1

    def observe_batch(self, values: list[float]) -> None:
        """Update in-place from a list of scores."""
        for v in values:
            self.observe_one(v)

    @property
    def mean(self) -> float:
        """Posterior mean of the true score."""
        return float(self.mu_n)

    @property
    def n(self) -> float:
        return float(self._n)

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Marginal posterior credible interval for mu (Student-t distribution)."""
        df = 2.0 * self.alpha_n
        scale = float(np.sqrt(self.beta_n / (self.alpha_n * self.kappa_n)))
        dist = stats.t(df=df, loc=self.mu_n, scale=scale)
        tail = (1 - ci) / 2
        return float(dist.ppf(tail)), float(dist.ppf(1 - tail))

    def sample(self, n: int = 10_000) -> np.ndarray:
        """Draw ``n`` samples of mu from the marginal posterior (Student-t)."""
        df = 2.0 * self.alpha_n
        scale = float(np.sqrt(self.beta_n / (self.alpha_n * self.kappa_n)))
        return stats.t.rvs(df=df, loc=self.mu_n, scale=scale, size=n)

    def prob_beats(self, other: "NormalPosterior", n_samples: int = 10_000) -> float:
        """Estimate P(self mean score > other mean score) via Monte Carlo.

        Args:
            other: Another :class:`NormalPosterior`.
            n_samples: Monte Carlo samples (default 10,000).

        Returns:
            Probability in [0, 1].
        """
        samples_self = self.sample(n_samples)
        samples_other = other.sample(n_samples)
        return float(np.mean(samples_self > samples_other))

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"NormalPosterior(mu={self.mu_n:.3f}, n={self._n}, "
            f"95%CI=[{lo:.3f}, {hi:.3f}])"
        )
