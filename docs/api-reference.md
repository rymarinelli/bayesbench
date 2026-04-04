# API reference

## Core classes

### `BayesianBenchmark`

```python
BayesianBenchmark(
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 3,
    posterior_factory: Callable = BetaPosterior,
)
```

| Method | Returns | Description |
|---|---|---|
| `.task(name, dataset, posterior_factory)` | decorator | Register a named evaluation task |
| `.compare(model_a, model_b, score_fn, dataset)` | `TaskResult` | Direct pairwise comparison |
| `.compare_async(...)` | `TaskResult` | Async pairwise comparison |
| `.run(verbose=False)` | `BenchmarkReport` | Run all registered tasks |
| `.run_async()` | `BenchmarkReport` | Async run over registered tasks |

### `BayesianRanker`

```python
BayesianRanker(
    confidence: float = 0.95,
    skip_threshold: float = 0.85,
    min_samples: int = 5,
    posterior_factory: Callable = BetaPosterior,
)
```

| Method | Returns | Description |
|---|---|---|
| `.add_model(name, fn)` | `self` | Register a model callable |
| `.evaluate` | decorator | Register scoring function |
| `.rank(dataset, score_fn, verbose=False)` | `RankingResult` | Rank models with Bayesian comparisons |
| `.rank_async(dataset, score_fn)` | `RankingResult` | Async ranking |

## Result objects

### `TaskResult`

| Attribute | Description |
|---|---|
| `winner` | `"model_a"`, `"model_b"`, or `None` |
| `p_a_beats_b` | Posterior probability that A beats B |
| `efficiency` | Fraction of evaluations saved |
| `problems_tested` | Number of evaluated problems |
| `total_problems` | Total dataset size |
| `posterior_a`, `posterior_b` | Final posterior objects |
| `skipped` | Whether task was skipped as non-discriminating |

### `BenchmarkReport`

| Attribute / Method | Description |
|---|---|
| `task_results` | List of task-level outcomes |
| `overall_efficiency` | Aggregate fraction of evaluations saved |
| `winners` | Mapping from task name to winner |
| `summary()` | Text summary for quick review |
| `to_dict()` | Serialize report to a plain dictionary |
| `to_dataframe()` | Export to pandas DataFrame (if pandas is installed) |

## CLI

```bash
bayesbench my_benchmark.py
bayesbench my_benchmark.py --confidence 0.99 --min-samples 10 --skip-threshold 0.90
bayesbench --version
```

Benchmark files should expose either:

- `bench = BayesianBenchmark(...)`, or
- a `@suite`-decorated class.
