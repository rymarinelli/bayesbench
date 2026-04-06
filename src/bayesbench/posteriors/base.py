"""Abstract Posterior protocol.

All posterior types must satisfy this interface so the benchmark engine
remains agnostic to the choice of Bayesian model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


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
    def prob_beats(self, other: Posterior, n_samples: int = 10_000) -> float:
        """Compute P(self's metric > other's metric).

        Args:
            other: Another posterior of the *same type*.
            n_samples: Number of Monte Carlo samples for numerical estimation
                       (used by implementations that lack closed-form P(A>B)).

        Returns:
            Probability in [0, 1].
        """

    @abstractmethod
    def credible_interval(self, ci: float = 0.95) -> tuple[float, float]:
        """Return a central credible interval of mass ``ci``.

        Returns:
            (lower, upper) tuple.
        """

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
