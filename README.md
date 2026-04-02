# baysbench

**Bayesian sequential benchmarking for LLMs and agents.**

Stop testing when you have enough evidence — not when you run out of problems.
Based on the Beta-Bernoulli conjugate framework from
*"Bayesian Sequential Testing for Efficient LLM Benchmarking"*
(40th International Workshop on Statistical Modelling, Oslo 2026),
which achieved a **98.7% cost reduction** on NorEval by testing only 410 of 31,800 problems.

---

## Installation

```bash
pip install baysbench
```

## How it works

Instead of evaluating every problem in a dataset, baysbench:

1. Maintains a **Beta posterior** over each model's true accuracy.
2. After each problem, computes **P(Model A accuracy > Model B accuracy)** via numerical integration.
3. **Stops early** the moment that probability crosses a confidence threshold (default 0.95).
4. **Skips non-discriminating tasks** where both models perform similarly (P close to 0.5).

The result: dramatic cost savings with statistically rigorous conclusions.

---

## Quick start

### Style 1 — standalone `@benchmark` decorator

```python
from baysbench import benchmark

@benchmark(
    model_a=lambda p: big_llm(p["question"]),
    model_b=lambda p: small_llm(p["question"]),
    dataset=problems,          # any iterable
    confidence=0.95,
)
def exact_match(problem, response):
    return response.strip() == problem["answer"]

result = exact_match.run()
print(result.winner)      # "model_a", "model_b", or None
print(result.efficiency)  # fraction of problems saved
```

### Style 2 — `BayesianBenchmark` instance + `@bench.task`

```python
from baysbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95)

@bench.task(dataset=problems, name="gsm8k")
def gsm8k(problem):
    a_correct = big_llm(problem["question"]) == problem["answer"]
    b_correct = small_llm(problem["question"]) == problem["answer"]
    return a_correct, b_correct

report = bench.run()
print(report.summary())
```

### Style 3 — class-based `@suite`

```python
from baysbench import suite

@suite(confidence=0.95)
class MathBenchmark:
    dataset = problems

    @staticmethod
    def task_arithmetic(problem):
        return big_llm(problem["q"]) == problem["a"], \
               small_llm(problem["q"]) == problem["a"]

    @staticmethod
    def task_algebra(problem):
        return big_llm(problem["q"]) == problem["a"], \
               small_llm(problem["q"]) == problem["a"]

report = MathBenchmark.run()
print(report.summary())
```

### Async models

```python
import asyncio
from baysbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95)

async def main():
    result = await bench.compare_async(
        model_a=async_big_llm,    # async def model(problem) -> str
        model_b=async_small_llm,
        score_fn=lambda p, r: r == p["answer"],
        dataset=problems,
    )
    print(result)

asyncio.run(main())
```

---

## CLI

```bash
# Run all tasks defined in a benchmark file
baysbench my_benchmark.py

# Override stopping thresholds
baysbench my_benchmark.py --confidence 0.99 --min-samples 10
```

The file must define a `bench = BayesianBenchmark()` instance, or use `@suite`.

---

## API reference

### `BetaPosterior`

| Attribute / Method | Description |
|---|---|
| `.alpha`, `.beta` | Beta distribution parameters |
| `.mean` | Posterior mean (expected accuracy) |
| `.observe_one(success)` | In-place update from one outcome |
| `.observe(success)` | Returns a new updated posterior |
| `.observe_batch(successes, total)` | In-place batch update |
| `.credible_interval(ci=0.95)` | Returns `(lower, upper)` |

### `BayesianBenchmark(confidence, skip_threshold, min_samples)`

| Method | Description |
|---|---|
| `.task(name, dataset)` | Decorator to register a task function |
| `.compare(model_a, model_b, score_fn, dataset)` | Direct comparison |
| `.compare_async(...)` | Async version with `await` |
| `.run()` → `BenchmarkReport` | Run all registered tasks |
| `.run_async()` | Async version |

### `TaskResult`

| Attribute | Description |
|---|---|
| `.winner` | `"model_a"`, `"model_b"`, or `None` |
| `.efficiency` | Fraction of problems not tested |
| `.problems_tested` / `.total_problems` | Sample counts |
| `.posterior_a` / `.posterior_b` | Final Beta posteriors |
| `.p_a_beats_b` | Final P(A > B) |
| `.skipped` | True if task was non-discriminating |

---

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `confidence` | 0.95 | Stop when P(A>B) ≥ this or ≤ 1-this |
| `skip_threshold` | 0.85 | Skip when P(A>B) ∈ (0.15, 0.85) |
| `min_samples` | 3 | Minimum problems before early stopping |

---

## License

MIT
