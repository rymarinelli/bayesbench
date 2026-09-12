"""Abstract Posterior protocol.

All posterior types must satisfy this interface so the benchmark engine
remains agnostic to the choice of Bayesian model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Posterior(ABC):
    """Abstract base class for conjugate posterior distributions.

    A ``Posterior`` tracks beliefs about a model's performance metric
    and supports Bayesian sequential testing via ``prob_beats``.

    Subclass this to plug in any Bayesian model:
    - :class:`~bayesbench.posteriors.BetaPosterior` — binary correct/incorrect
    - :class:`~bayesbench.posteriors.NormalPosterior` — continuous scores
    - Your own class for custom distributions
    """

    @abstractmethod
    def observe_one(self, value: float | bool) -> None:
        """Update the posterior in-place from a single observation.

        Args:
            value: A scalar observation. For binary posteriors pass ``bool``;
                   for continuous posteriors pass a ``float`` score.
        """

    @abstractmethod
    def prob_beats(self, other: Posterior, n_samples: int = 10_000, rng: Any = None) -> float:
        """Compute P(self's metric > other's metric).

        Args:
            other: Another posterior of the *same type*.
            n_samples: Number of Monte Carlo samples for numerical estimation
                       (used by implementations that lack closed-form P(A>B)).
            rng: Optional seed or ``numpy.random.Generator`` for reproducible
                 Monte Carlo estimation. Implementations with closed forms
                 may ignore it.

        Returns:
            Probability in [0, 1].
        """

    @abstractmethod
    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Return a central credible interval of mass ``ci``.

        Returns:
            (lower, upper) tuple.
        """

    def sample(self, n: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n`` samples from the posterior.

        Concrete subclasses should override with an efficient sampler; the
        default raises :class:`NotImplementedError`.

        Args:
            n: Number of samples.
            rng: Optional Generator for reproducible draws.

        Returns:
            Array of samples.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sample()")

    def prob_beats_value(self, value: float, n_samples: int = 10_000) -> float:
        """Compute P(self's metric > ``value``).

        Default implementation draws Monte Carlo samples; subclasses with
        closed forms should override.

        Args:
            value: A fixed scalar to compare against.
            n_samples: Monte Carlo samples for the default implementation.

        Returns:
            Probability in [0, 1].
        """
        samples = np.asarray(self.sample(n_samples))
        return float(np.mean(samples > value))

    @property
    @abstractmethod
    def mean(self) -> float:
        """Posterior mean of the performance metric."""

    @property
    def n(self) -> float:
        """Effective number of observations (excluding prior mass).

        Default implementation returns 0; override in subclasses.
        """
        return 0.0
