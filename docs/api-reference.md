# API reference

## Core classes

### `BayesianBenchmark`

```python
BayesianBenchmark(
    confidence: float = 0.95,
    skip_threshold: float | None = None,   # deprecated legacy skip heuristic
    min_samples: int = 30,
    posterior_factory: Callable = BetaPosterior,
    decision_rule: str = "posterior",      # or "confidence_sequence"
    equivalence_margin: float | None = None,
    max_samples: int | None = None,
    paired: bool = False,
    alpha: float = 0.05,
    rng: np.random.Generator | int | None = None,
    on_step: Callable[[StepTrace], None] | None = None,
)
```

| Method | Returns | Description |
|---|---|---|
| `.task(name, dataset, posterior_factory)` | decorator | Register a named evaluation task |
| `.compare(model_a, model_b, score_fn, dataset)` | `TaskResult` | Direct pairwise comparison |
| `.compare_async(...)` | `TaskResult` | Async pairwise comparison |
| `.iter_compare(...)` | `Iterable[LiveUpdate]` | Streaming generator; last update carries the final `TaskResult`; models are invoked lazily so reported counts equal actual calls |
| `.run(verbose=False)` | `BenchmarkReport` | Run all registered tasks |
| `.run_async()` | `BenchmarkReport` | Async run over registered tasks |

### `BayesianRanker`

```python
BayesianRanker(
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 5,
    posterior_factory: Callable = BetaPosterior,
    lower_is_better: bool = False,
)
```

| Method | Returns | Description |
|---|---|---|
| `.add_model(name, fn)` | `self` | Register a model callable |
| `.evaluate` | decorator | Register scoring function |
| `.rank(dataset, score_fn, verbose=False)` | `RankingResult` | Rank models with Bayesian comparisons |
| `.rank_async(dataset, score_fn)` | `RankingResult` | Async ranking |

## Decision rules (`bayesbench.decision`)

| Class | Description |
|---|---|
| `DecisionStatus` | `winner_a`, `winner_b`, `equivalent`, `inconclusive` |
| `StepTrace` | One update step: scores, P(A>B), status, terminal reason |
| `PosteriorThresholdRule` | Confidence threshold + optional ROPE `equivalence_margin` |
| `ConfidenceSequenceRule` | Ville-valid any-time rule: P(wrong winner at any stopping time) <= `alpha` (binary outcomes) |
| `PairedDifferenceRule` | Per-item score differences for `paired=True` |

## Result objects

### `TaskResult`

| Attribute | Description |
|---|---|
| `decision` | `DecisionStatus`: winner_a / winner_b / equivalent / inconclusive |
| `winner` | `"model_a"`, `"model_b"`, or `None` (legacy threshold view) |
| `p_a_beats_b` | Posterior probability that A beats B |
| `confidence` | Confidence threshold used to declare a winner |
| `efficiency` | Fraction of evaluations saved (actual calls) |
| `problems_tested` | Number of evaluated problems |
| `total_problems` | Total dataset size |
| `posterior_a`, `posterior_b` | Final posterior objects |
| `terminal_reason` | Human-readable stopping reason |
| `trace` | `list[StepTrace]` — full decision trail |
| `skipped` | Deprecated alias for `decision == equivalent` |

### `BenchmarkReport`

| Attribute / Method | Description |
|---|---|
| `task_results` | List of task-level outcomes |
| `overall_efficiency` | Aggregate fraction of evaluations saved |
| `winners` | Mapping from task name to winner |
| `summary()` | Text summary for quick review |
| `to_dict()` | Serialize report to a plain dictionary |
| `to_dataframe()` | Export to pandas DataFrame (if pandas is installed) |

## Calibration harness (`bayesbench.calibration`)

| Function | Description |
|---|---|
| `enumerate_binary_outcomes(...)` | Exact enumeration of all outcome sequences |
| `simulate_pairwise(...)` | Monte Carlo decision-quality sims (incl. shared item difficulty via `item_sd`) |
| `simulate_pairwise_cs(...)` | Same protocol for the confidence-sequence rule |
| `simulate_order_sensitivity(...)` | Order-dependence of the stopping rule |
| `effect_size_sweep(...)` | False/correct/inconclusive rates vs effect size |
| `calibrate_sweep(...)` | Parameter-grid sweep (legacy rule) |

## CLI

```bash
bayesbench my_benchmark.py
bayesbench my_benchmark.py --confidence 0.99 --min-samples 10 --skip-threshold 0.90
bayesbench --version
```

Benchmark files should expose either:

- `bench = BayesianBenchmark(...)`, or
- a `@suite`-decorated class.
