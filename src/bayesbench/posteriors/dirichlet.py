"""Dirichlet-Multinomial conjugate posterior for categorical outcomes.

Use this when outcomes are drawn from a fixed set of K categories:
- Multiple-choice accuracy (A/B/C/D) where category 0 = correct
- Topic or sentiment classification
- Any K-way categorical label

When K=2, this is equivalent to :class:`~bayesbench.posteriors.BetaPosterior`
with a symmetric prior.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .base import Posterior


class DirichletPosterior(Posterior):
    """Dirichlet-Multinomial conjugate model for K-category outcomes.

    Maintains a Dirichlet posterior over category probabilities.  The scalar
    performance metric is the posterior mean probability of ``target_class``
    (default 0), so for accuracy tasks set category 0 = "correct answer".

    Args:
        k: Number of categories (≥ 2).
        alpha_0: Symmetric Dirichlet concentration (prior pseudo-count per
                 category).  ``0.5`` gives a sparse, Jeffreys-like prior.
        target_class: The category index whose probability is used as the
                      scalar performance metric for ``prob_beats`` and
                      ``credible_interval``.  Defaults to ``0``.

    Usage::

        # 4-choice MCQ — category 0 = correct answer
        p = DirichletPosterior(k=4)
        p.observe_one(0)   # model answered correctly
        p.observe_one(2)   # model picked option C (wrong)
        print(p.mean)      # P(correct) posterior mean

        # Binary — equivalent to BetaPosterior(0.5, 0.5)
        p = DirichletPosterior(k=2)
        p.observe_one(True)  # True → category 0 (correct)

    Observing with a ``bool`` maps ``True`` → category 0, ``False`` → category 1,
    so ``DirichletPosterior(k=2)`` is a drop-in replacement for
    :class:`~bayesbench.posteriors.BetaPosterior` in binary benchmarks.
    """

    def __init__(
        self,
        k: int = 2,
        alpha_0: float = 0.5,
        target_class: int = 0,
    ) -> None:
        if k < 2:
            raise ValueError("k must be >= 2")
        if alpha_0 <= 0:
            raise ValueError("alpha_0 must be positive")
        if not (0 <= target_class < k):
            raise ValueError(f"target_class must be in [0, {k - 1}]")
        self._k = k
        self._alpha_0 = alpha_0
        self._target_class = target_class
        self._counts: np.ndarray = np.zeros(k)

    @property
    def n(self) -> float:
        return float(self._counts.sum())

    @property
    def alpha(self) -> np.ndarray:
        """Posterior Dirichlet concentration parameters."""
        return self._counts + self._alpha_0

    def observe_one(self, value: float | bool) -> None:
        """Observe one categorical outcome.

        Args:
            value: Integer category index ``0..K-1``, or a ``bool``
                   (``True`` → category 0, ``False`` → category 1).
        """
        if isinstance(value, bool):
            idx = 0 if value else 1
        else:
            idx = int(value)
        if not (0 <= idx < self._k):
            raise ValueError(f"Category index {idx} out of range [0, {self._k - 1}]")
        self._counts[idx] += 1

    @property
    def mean(self) -> float:
        """Posterior mean probability of the target class."""
        a = self.alpha
        return float(a[self._target_class] / a.sum())

    def prob_beats(self, other: Posterior, n_samples: int = 10_000) -> float:
        """P(target-class proportion A > target-class proportion B) via Monte Carlo.

        Args:
            other: Another :class:`DirichletPosterior` (must have same ``k``).
            n_samples: Number of MC samples.

        Returns:
            Probability in [0, 1].
        """
        if not isinstance(other, DirichletPosterior):
            raise TypeError("DirichletPosterior.prob_beats requires another DirichletPosterior")
        rng = np.random.default_rng(42)
        samples_a = rng.dirichlet(self.alpha, size=n_samples)[:, self._target_class]
        samples_b = rng.dirichlet(other.alpha, size=n_samples)[:, other._target_class]
        return float((samples_a > samples_b).mean())

    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Central credible interval for the target-class proportion.

        Uses the Beta marginal of the Dirichlet for the target class.
        """
        a = self.alpha
        alpha_t = float(a[self._target_class])
        beta_t = float(a.sum() - alpha_t)
        lo = (1.0 - ci) / 2.0
        dist = stats.beta(alpha_t, beta_t)
        return float(dist.ppf(lo)), float(dist.ppf(1.0 - lo))

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` probability vectors from the Dirichlet posterior.

        Returns:
            Array of shape ``(n, k)``.
        """
        if rng is None:
            rng = np.random.default_rng()
        return rng.dirichlet(self.alpha, size=n)
