"""Bayesian posterior distributions for benchmarking.

Available posteriors
--------------------
:class:`BetaPosterior`
    Beta-Bernoulli model for **binary** outcomes (exact match, pass/fail).
    Closed-form P(A>B) via numerical integration of the Beta distribution.

:class:`NormalPosterior`
    Normal-Inverse-Gamma model for **continuous** scores (BLEU, ROUGE,
    semantic similarity, LLM-judge 0-1 scores, normalised Likert ratings).
    P(A>B) estimated via Monte Carlo sampling from the Student-t marginal.

:class:`Posterior`
    Abstract base class — subclass this to add custom distributions.

Usage::

    from baysbench.posteriors import BetaPosterior, NormalPosterior

    # Binary (exact-match accuracy)
    p = BetaPosterior()
    p.observe_one(True)

    # Continuous (BLEU score)
    p = NormalPosterior(mu_0=0.3)
    p.observe_one(0.42)
"""

from .base import Posterior
from .beta import BetaPosterior
from .normal import NormalPosterior

__all__ = ["Posterior", "BetaPosterior", "NormalPosterior"]
