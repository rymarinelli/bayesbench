# Core concepts and tuning

## Sequential Bayesian benchmarking

`bayesbench` updates posteriors after each example and continuously checks whether
one model is likely better than another.

Every run ends in one of four explicit decision states:

- **winner_a / winner_b**: posterior evidence for superiority crossed the threshold.
- **equivalent**: posterior mass inside a ROPE (`equivalence_margin`) reached the threshold.
- **inconclusive**: the budget ran out without enough evidence.

Read the decision from `result.decision`. The `winner` property remains for
backward compatibility.

## Decision rules

`BayesianBenchmark(decision_rule=...)` selects the stopping rule:

### `"posterior"` (default)

- Stop when `P(A > B) >= confidence` (A) or `<= 1 - confidence` (B).
- Fast and powerful, but a **heuristic**: under equal models P(A>B) is a random
  walk that eventually crosses any fixed threshold, so the false-winner rate
  grows with the number of problems evaluated (the optional-stopping trap).
  Pair it with a calibration sweep (`bayesbench.calibration`) and report the
  false-decision rate alongside any savings claim.

### `"confidence_sequence"`

- Ville-valid any-time inference (mixture likelihood ratios,
  Waudby-Smith & Ramdas 2023).
- Guarantees P(wrong winner at **any** stopping time) <= `alpha` by
  construction — no calibration, no horizon assumptions, no order sensitivity.
- Binary outcomes only; needs more samples than the threshold rule.
  Use it for public performance claims.

## Main knobs

### `confidence`

- Default: `0.95`
- Meaning: required posterior certainty before stopping.
- Increase to `0.99` for higher-stakes choices.
- With `decision_rule="confidence_sequence"`, `alpha = 1 - confidence` is the
  any-time error budget.

### `min_samples`

- Minimum evaluations before any terminal decision.
- Default: `30` (calibrated against null false-decision rates; the old default
  of 3 terminated every binary comparison at example 3).

### `equivalence_margin` (ROPE)

- Optional half-width of the region of practical equivalence.
- When set, the run reports `equivalent` once posterior mass inside
  |theta_a - theta_b| <= margin reaches `confidence`.
- P(A>B) near 0.5 is *never* treated as equivalence on its own — early 50/50
  evidence is insufficient information.

### `skip_threshold` (deprecated)

- Legacy P(A>B)-window skip heuristic, disabled by default. It converted
  insufficient early evidence into premature terminal decisions.
- `result.skipped` is now a deprecated alias for `decision == equivalent`.

### `max_samples`

- Optional hard cap on problems evaluated per task; the run reports
  `inconclusive` when the cap is hit.

## Paired evaluation

When both models answer the same items, use `paired=True`: the rule models
per-item score differences instead of two independent streams, so shared item
difficulty is not double-counted as independent evidence.

## Choosing a posterior

### Binary outcomes → `BetaPosterior`

Use for exact match, pass/fail, rubric pass/fail, and multiple-choice correctness.

### Continuous outcomes → `NormalPosterior`

Use for BLEU/ROUGE-like scores, cosine similarity, judge scores, and other real-valued metrics.

## Interpreting results

A task result gives:

- `decision`: winner_a / winner_b / equivalent / inconclusive
- `winner`: which model won (`model_a`, `model_b`, or `None`) — legacy view
- `p_a_beats_b`: posterior probability that A outperforms B
- `efficiency`: fraction of evaluations saved by early stopping (actual calls
  — see `iter_compare` / `trace`)
- `trace`: per-step evidence trail (scores, P(A>B), status, terminal reason)
- credible intervals for each model's latent quality

Do not treat the posterior probability as a universal truth; it is conditional on:

- your dataset,
- your scoring function,
- your prior choice,
- and any filtering/skipping rules.

## Practical defaults

- Start: `confidence=0.95`, `min_samples=30`, skip heuristic off
- Public claims: `decision_rule="confidence_sequence"` (≤5% any-time error)
- Equivalence claims: set `equivalence_margin` and say what margin you used
- Noisy judge scores: increase `min_samples` and prefer larger validation sets
- Highly heterogeneous tasks: report per-task outcomes, not only aggregate winner
- Always report savings as prospective and next to calibration numbers
