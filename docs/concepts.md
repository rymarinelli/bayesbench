# Core concepts and tuning

## Sequential Bayesian benchmarking

`bayesbench` updates posteriors after each example and continuously checks whether
one model is likely better than another.

A typical decision rule is:

- Declare **model A wins** if `P(A > B) >= confidence`.
- Declare **model B wins** if `P(A > B) <= (1 - confidence)`.
- Otherwise continue until the dataset is exhausted (or evidence stabilizes).

## Main knobs

### `confidence`

- Default: `0.95`
- Meaning: required posterior certainty before stopping.
- Increase to `0.99` for higher-stakes choices.

### `min_samples`

- Guards against very early conclusions from tiny sample counts.
- Start with `5` for noisy tasks.

### `skip_threshold`

- Skips tasks that appear non-discriminating between models.
- Helps avoid spending budget on tasks unlikely to change decisions.

## Choosing a posterior

### Binary outcomes → `BetaPosterior`

Use for exact match, pass/fail, rubric pass/fail, and multiple-choice correctness.

### Continuous outcomes → `NormalPosterior`

Use for BLEU/ROUGE-like scores, cosine similarity, judge scores, and other real-valued metrics.

## Interpreting results

A task result usually gives:

- `winner`: which model won (`model_a`, `model_b`, or `None`)
- `p_a_beats_b`: posterior probability that A outperforms B
- `efficiency`: fraction of evaluations saved by early stopping
- credible intervals for each model's latent quality

Do not treat the posterior probability as a universal truth; it is conditional on:

- your dataset,
- your scoring function,
- your prior choice,
- and any filtering/skipping rules.

## Practical defaults

- Start: `confidence=0.95`, `min_samples=5`
- Final release decisions: `confidence=0.99`
- Noisy judge scores: increase `min_samples` and prefer larger validation sets
- Highly heterogeneous tasks: report per-task outcomes, not only aggregate winner
