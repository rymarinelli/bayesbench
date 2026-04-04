# bayesbench

**Bayesian sequential benchmarking for LLMs and agents.**

`bayesbench` helps you stop evaluations as soon as posterior evidence is strong enough,
instead of evaluating every model on every example.

## Why use bayesbench?

- **Lower eval cost:** stop early when confidence is reached.
- **Statistically principled:** Bayesian posteriors and credible intervals.
- **Flexible inputs:** binary or continuous scores.
- **Practical integrations:** OpenAI-compatible APIs, Anthropic, Hugging Face, Inspect, MTEB, and OpenClaw.

## Install

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

## Quick start

### Pairwise comparison

```python
from bayesbench import benchmark

@benchmark(
    model_a=lambda p: big_model(p["question"]),
    model_b=lambda p: small_model(p["question"]),
    dataset=problems,
    confidence=0.95,
)
def exact_match(problem, response):
    return response.strip() == problem["answer"]

result = exact_match.run()
print(result.winner)
print(result.efficiency)
print(result.p_a_beats_b)
```

### Rank multiple models

```python
from bayesbench import BayesianRanker

ranker = BayesianRanker(confidence=0.95)
ranker.add_model("large", large_model)
ranker.add_model("medium", medium_model)
ranker.add_model("small", small_model)

result = ranker.rank(dataset=problems, score_fn=lambda p, r: r == p["answer"])
print(result.summary())
```

## Core concepts

- **Confidence threshold (`confidence`)**
  - The stopping rule, e.g. `0.95` means stop when posterior win probability reaches 95%.
- **Minimum samples (`min_samples`)**
  - Prevents stopping too early from tiny sample sizes.
- **Skip threshold (`skip_threshold`)**
  - Skips non-discriminating examples where both models behave similarly.
- **Posterior family**
  - Use `BetaPosterior` for binary outcomes and `NormalPosterior` for continuous scores.

## Continuous-score benchmarking

```python
from bayesbench import BayesianBenchmark
from bayesbench.posteriors import NormalPosterior

bench = BayesianBenchmark(confidence=0.95, posterior_factory=NormalPosterior)

result = bench.compare(
    model_a=large_model,
    model_b=small_model,
    score_fn=lambda p, r: bleu(r, p["reference"]),
    dataset=translation_examples,
)
```

## Adapter example: OpenClaw

```python
from bayesbench.adapters.openclaw import openclaw_agent

agent = openclaw_agent(my_openclaw_agent)
response = agent({"input": "Solve 17 * 19"})
```

## CLI

```bash
bayesbench my_benchmark.py
bayesbench my_benchmark.py --confidence 0.99 --min-samples 10 --skip-threshold 0.90
bayesbench --version
```

Your benchmark file should expose either:

- `bench = BayesianBenchmark(...)`, or
- a `@suite`-decorated class.

## Docs development

```bash
pip install -e ".[docs]"
mkdocs serve
mkdocs build --strict
```

## Next steps

- Browse source examples in `examples/`.
- See `README.md` for a broader project overview.
- Contribute improvements via `CONTRIBUTING.md`.
