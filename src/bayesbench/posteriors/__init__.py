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

:class:`DirichletPosterior`
    Dirichlet-Multinomial model for **categorical** outcomes (multiple-choice
    accuracy, topic labels, K-class classification).  ``target_class=0``
    (default) treats category 0 as "correct".

:class:`GammaPosterior`
    Gamma-Poisson model for **count/rate** data (token counts, response
    latency, API call cost).  Set ``higher_is_better=False`` for latency/cost.

:class:`Posterior`
    Abstract base class — subclass this to add custom distributions.

Choosing a posterior
--------------------

================  ================  ========================================
Score type        Posterior         Example
================  ================  ========================================
bool / 0-1 int    BetaPosterior     Exact match, PASS/FAIL
float in [0,1]    NormalPosterior   BLEU, ROUGE, cosine sim, LLM-judge
int category      DirichletPosterior  4-choice MCQ (k=4), sentiment (k=3)
non-neg count     GammaPosterior    Token count, latency (ms)
================  ================  ========================================

Usage::

    from bayesbench.posteriors import BetaPosterior, NormalPosterior
    from bayesbench.posteriors import DirichletPosterior, GammaPosterior

    # Binary (exact-match accuracy)
    p = BetaPosterior()
    p.observe_one(True)

    # Continuous (BLEU score)
    p = NormalPosterior(mu_0=0.3)
    p.observe_one(0.42)

    # 4-choice MCQ — category 0 = correct
    p = DirichletPosterior(k=4)
    p.observe_one(0)   # correct

    # Latency — lower is better
    p = GammaPosterior(higher_is_better=False)
    p.observe_one(120)  # 120 ms
"""

from .base import Posterior
from .beta import BetaPosterior
from .dirichlet import DirichletPosterior
from .gamma import GammaPosterior
from .normal import NormalPosterior

__all__ = [
    "Posterior",
    "BetaPosterior",
    "NormalPosterior",
    "DirichletPosterior",
    "GammaPosterior",
]
