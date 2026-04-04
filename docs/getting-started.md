# Getting started

## Installation

```bash
pip install bayesbench
```

Optional integrations:

```bash
pip install "bayesbench[openai]"
pip install "bayesbench[anthropic]"
pip install "bayesbench[huggingface]"
pip install "bayesbench[inspect]"
pip install "bayesbench[mteb]"
pip install "bayesbench[openclaw]"
pip install "bayesbench[all]"
```

## Minimal pairwise benchmark

```python
from bayesbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95)

result = bench.compare(
    model_a=lambda p: big_model(p["question"]),
    model_b=lambda p: small_model(p["question"]),
    dataset=problems,
    score_fn=lambda p, r: int(r.strip() == p["answer"]),
    name="quickstart_exact_match",
)

print(result.winner)
print(result.p_a_beats_b)
print(result.efficiency)
```

## Register multiple tasks

```python
from bayesbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95, min_samples=5)

@bench.task(dataset=gsm8k, name="gsm8k")
def math_task(problem):
    return model_a(problem["q"]) == problem["a"], model_b(problem["q"]) == problem["a"]

@bench.task(dataset=mmlu, name="mmlu")
def reasoning_task(problem):
    return model_a(problem["q"]) == problem["a"], model_b(problem["q"]) == problem["a"]

report = bench.run(verbose=True)
print(report.summary())
```

## Use continuous metrics

```python
from bayesbench import BayesianBenchmark
from bayesbench.posteriors import NormalPosterior

bench = BayesianBenchmark(confidence=0.95, posterior_factory=NormalPosterior)

result = bench.compare(
    model_a=translation_model_a,
    model_b=translation_model_b,
    dataset=translation_set,
    score_fn=lambda p, r: compute_bleu(r, p["reference"]),
    name="bleu_eval",
)
```

## CLI usage

```bash
bayesbench my_benchmark.py
bayesbench my_benchmark.py --confidence 0.99 --min-samples 10 --skip-threshold 0.90
bayesbench --version
```

Your benchmark file should expose either:

- `bench = BayesianBenchmark(...)`, or
- a `@suite`-decorated class.

## What to read next

- **[Workflows](workflows.md)** for complete templates.
- **[Concepts](concepts.md)** for tuning confidence and stopping behavior.
- **[Examples gallery](examples.md)** for scripts in `examples/`.
