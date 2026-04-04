"""Core Bayesian statistical primitives — backward-compatible re-exports.

The canonical implementations now live in :mod:`baysbench.posteriors`.
This module keeps existing imports working.
"""

from __future__ import annotations

from .posteriors.base import Posterior

# Re-export for backward compatibility
from .posteriors.beta import BetaPosterior


def prob_a_beats_b(posterior_a: BetaPosterior, posterior_b: BetaPosterior) -> float:
    """Compute P(accuracy_A > accuracy_B).

    Thin wrapper around :meth:`~baysbench.posteriors.BetaPosterior.prob_beats`
    kept for backward compatibility.
    """
    return posterior_a.prob_beats(posterior_b)


def is_non_discriminating(
    posterior_a: BetaPosterior,
    posterior_b: BetaPosterior,
    threshold: float = 0.85,
) -> bool:
    """Return True when a task cannot distinguish between the two models.

    Thin wrapper kept for backward compatibility.
    """
    p = posterior_a.prob_beats(posterior_b)
    return (1.0 - threshold) < p < threshold


__all__ = ["BetaPosterior", "Posterior", "prob_a_beats_b", "is_non_discriminating"]
