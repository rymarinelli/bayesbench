<div align="center">

# bayesbench

**Bayesian sequential benchmarking for LLMs and agents.**

[![CI](https://github.com/rymarinelli/baysbench/actions/workflows/ci.yml/badge.svg)](https://github.com/rymarinelli/baysbench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/baysbench)](https://pypi.org/project/baysbench/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![codecov](https://codecov.io/gh/rymarinelli/baysbench/branch/main/graph/badge.svg)](https://codecov.io/gh/rymarinelli/baysbench)

Stop evaluating when you have enough evidence — not when you run out of problems.

</div>

---

baysbench applies **Bayesian sequential testing** to LLM evaluation. Instead of running every model on every problem, it stops as soon as posterior evidence crosses a statistical confidence threshold — delivering the same rigorous conclusions at a fraction of the cost.

> Based on *"Bayesian Sequential Testing for Efficient LLM Benchmarking"*, submitted to the **40th International Workshop on Statistical Modelling, Oslo 2026**.  
> Demonstrated a **98.7% cost reduction** on NorEval — 410 problems evaluated out of 31,800.

## How it works

```
Problem 1 → update posteriors → P(A>B) = 0.61  (inconclusive, continue)
Problem 2 → update posteriors → P(A>B) = 0.74  (inconclusive, continue)
Problem 3 → update posteriors → P(A>B) = 0.96  ✓ STOP — Model A wins
```

1. Maintains a **conjugate posterior** over each model's true performance metric.
2. After every problem, computes **P(Model A beats Model B)** analytically or via Monte Carlo.
3. **Stops early** the moment that probability crosses the confidence threshold (default 0.95).
4. **Skips non-discriminating tasks** automatically when both models perform indistinguishably.

## Installation

```bash
pip install baysbench
```

**Optional framework integrations:**

```bash
pip install bayesbench[openai]        # OpenAI, Groq, Together AI, Ollama, vLLM, …
pip install bayesbench[anthropic]     # Anthropic (Claude)
pip install bayesbench[huggingface]   # HuggingFace Inference API + datasets
pip install bayesbench[inspect]       # AISI Inspect eval framework
pip install bayesbench[mteb]          # MTEB embedding benchmark
pip install baysbench[all]           # everything above
```

## Quick start

### Pairwise comparison

```python
from bayesbench import benchmark

@benchmark(
    model_a=lambda p: big_llm(p["question"]),
    model_b=lambda p: small_llm(p["question"]),
    dataset=problems,
    confidence=0.95,
)
def exact_match(problem, response):
    return response.strip() == problem["answer"]

result = exact_match.run()
print(result.winner)      # "model_a", "model_b", or None
print(result.efficiency)  # e.g. 0.87 → 87% of problems saved
```

### Multi-task suite

```python
from bayesbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95)

@bench.task(dataset=gsm8k,  name="gsm8k")
def math(problem):
    return model_a(problem["q"]) == problem["a"], \
           model_b(problem["q"]) == problem["a"]

@bench.task(dataset=mmlu,   name="mmlu")
def science(problem):
    return model_a(problem["q"]) == problem["a"], \
           model_b(problem["q"]) == problem["a"]

report = bench.run(verbose=True)   # tqdm progress bar
print(report.summary())
report.to_dataframe().to_csv("results.csv")
```

### Class-based suite

```python
from bayesbench import suite

@suite(confidence=0.95)
class EvalSuite:
    dataset = problems

    @staticmethod
    def task_reasoning(problem):
        return model_a(problem["q"]) == problem["a"], \
               model_b(problem["q"]) == problem["a"]

    @staticmethod
    def task_coding(problem):
        return run_tests(model_a, problem), run_tests(model_b, problem)

report = EvalSuite.run()
```

### Rank N models

```python
from bayesbench import BayesianRanker

ranker = BayesianRanker(confidence=0.95)
ranker.add_model("gpt-4o",       gpt4_fn)
ranker.add_model("gpt-4o-mini",  mini_fn)
ranker.add_model("llama-3-70b",  llama_fn)
ranker.add_model("mistral-large", mistral_fn)

result = ranker.rank(
    dataset=problems,
    score_fn=lambda p, r: r.strip() == p["answer"],
    verbose=True,
)
print(result.summary())
# Rank 1: gpt-4o         score=0.912  95%CI=[0.881, 0.943]  P(>gpt-4o-mini)=0.981
# Rank 2: llama-3-70b    score=0.884  95%CI=[0.850, 0.918]  P(>gpt-4o-mini)=0.963
# Rank 3: gpt-4o-mini    score=0.851  95%CI=[0.814, 0.888]  P(>mistral-large)=0.971
# Rank 4: mistral-large  score=0.803  95%CI=[0.762, 0.844]
```

### Continuous scores (BLEU, ROUGE, LLM-judge)

```python
from bayesbench import BayesianBenchmark
from bayesbench.posteriors import NormalPosterior

bench = BayesianBenchmark(
    confidence=0.95,
    posterior_factory=NormalPosterior,   # Normal-Inverse-Gamma conjugate model
)

result = bench.compare(
    model_a=big_llm,
    model_b=small_llm,
    score_fn=lambda p, r: compute_bleu(r, p["reference"]),  # returns float
    dataset=translation_problems,
)
```

### Async models

```python
result = await bench.compare_async(
    model_a=async_big_llm,
    model_b=async_small_llm,
    score_fn=lambda p, r: r == p["answer"],
    dataset=problems,
)
```

## Framework adapters

All adapters return a plain `callable(problem) -> str` that plugs into any baysbench API.

| Adapter | Import | Works with |
|---|---|---|
| **OpenAI-compatible** | `from baysbench.adapters.openai_compat import openai_model` | OpenAI, Groq, Together AI, Fireworks, Ollama, vLLM, Azure OpenAI |
| **Anthropic** | `from baysbench.adapters.anthropic_adapter import anthropic_model` | Claude (all versions) |
| **HuggingFace** | `from baysbench.adapters.huggingface import hf_model, hf_dataset` | Any HF Inference API endpoint |
| **Inspect AI** | `from baysbench.adapters.inspect_ai import inspect_model, from_inspect_dataset` | AISI Inspect `Dataset`, `Task`, `Scorer` |
| **MTEB** | `from baysbench.adapters.mteb import st_model, mteb_sts_dataset` | SentenceTransformers, MTEB STS + Classification |

```python
from bayesbench import BayesianRanker
from bayesbench.adapters.openai_compat import openai_model
from bayesbench.adapters.anthropic_adapter import anthropic_model

ranker = BayesianRanker(confidence=0.95)
ranker.add_model("gpt-4o",          openai_model("gpt-4o"))
ranker.add_model("claude-opus-4-6", anthropic_model("claude-opus-4-6"))
ranker.add_model("llama-3-groq",    openai_model("llama-3.1-70b-versatile",
                                        base_url="https://api.groq.com/openai/v1"))

result = ranker.rank(dataset=problems, score_fn=score)
```

## Posteriors

Swap the Bayesian model to match your metric type:

| Posterior | Use when | Import |
|---|---|---|
| `BetaPosterior` | Binary outcomes: exact match, pass/fail, multiple choice | `from baysbench.posteriors import BetaPosterior` |
| `NormalPosterior` | Continuous scores: BLEU, ROUGE, cosine similarity, LLM-judge (0–1) | `from baysbench.posteriors import NormalPosterior` |
| Custom | Any distribution — subclass `Posterior` | `from baysbench.posteriors import Posterior` |

```python
# Custom prior: expect ~30% BLEU baseline
from bayesbench.posteriors import NormalPosterior
bench = BayesianBenchmark(posterior_factory=lambda: NormalPosterior(mu_0=0.30))

# Per-task posterior override
@bench.task(dataset=problems, posterior_factory=NormalPosterior)
def bleu_task(problem):
    return compute_bleu(model_a(problem), problem["ref"]), \
           compute_bleu(model_b(problem), problem["ref"])
```

## Results & export

```python
report = bench.run()

# Text summary
print(report.summary())

# Serialise to dict / JSON
import json
print(json.dumps(report.to_dict(), indent=2))

# Pandas DataFrame (requires pandas)
df = report.to_dataframe()
df.to_csv("results.csv", index=False)

# Individual task result
result = report.task_results[0]
print(result.winner)            # "model_a" | "model_b" | None
print(result.efficiency)        # 0.0 – 1.0
print(result.p_a_beats_b)       # posterior probability
lo, hi = result.posterior_a.credible_interval()
```

## CLI

```bash
# Run all tasks in a benchmark file
bayesbench my_benchmark.py

# Override stopping thresholds
bayesbench my_benchmark.py --confidence 0.99 --min-samples 10 --skip-threshold 0.90

# Print version
bayesbench --version
```

The benchmark file must expose a `bench = BayesianBenchmark(...)` instance or a `@suite`-decorated class.

## API reference

### `BayesianBenchmark`

```python
BayesianBenchmark(
    confidence: float = 0.95,           # P(A>B) threshold to declare winner
    skip_threshold: float = 0.85,       # skip non-discriminating tasks
    min_samples: int = 3,               # minimum evaluations before stopping
    posterior_factory: Callable = BetaPosterior,
)
```

| Method | Returns | Description |
|---|---|---|
| `.task(name, dataset, posterior_factory)` | decorator | Register an evaluation function |
| `.compare(model_a, model_b, score_fn, dataset)` | `TaskResult` | Direct pairwise comparison |
| `.compare_async(...)` | `TaskResult` | Async pairwise comparison |
| `.run(verbose=False)` | `BenchmarkReport` | Run all registered tasks |
| `.run_async()` | `BenchmarkReport` | Async version |

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
| `.add_model(name, fn)` | `self` | Register a model (chainable) |
| `.evaluate` | decorator | Set the scoring function |
| `.rank(dataset, score_fn, verbose=False)` | `RankingResult` | Rank all models |
| `.rank_async(dataset, score_fn)` | `RankingResult` | Async version |

### `TaskResult`

| Attribute | Type | Description |
|---|---|---|
| `.winner` | `str \| None` | `"model_a"`, `"model_b"`, or `None` |
| `.efficiency` | `float` | Fraction of problems not evaluated |
| `.problems_tested` | `int` | Problems evaluated before stopping |
| `.total_problems` | `int` | Dataset size |
| `.p_a_beats_b` | `float` | Final P(A > B) |
| `.posterior_a`, `.posterior_b` | `Posterior` | Final posteriors |
| `.skipped` | `bool` | True if task was non-discriminating |
| `.to_dict()` | `dict` | Serialise to plain dict |

### `BenchmarkReport`

| Attribute / Method | Description |
|---|---|
| `.task_results` | List of `TaskResult` objects |
| `.overall_efficiency` | Aggregate fraction of problems saved |
| `.winners` | `{task_name: winner}` dict |
| `.summary()` | Formatted text report |
| `.to_dict()` | Serialise to plain dict |
| `.to_dataframe()` | Returns a `pandas.DataFrame` (requires pandas) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short:

```bash
git clone https://github.com/rymarinelli/baysbench
cd bayesbench
pip install -e ".[dev]"
pytest          # run tests
ruff check .    # lint
```

## Citation

```bibtex
@inproceedings{marinelli2026bayesian,
  title     = {Bayesian Sequential Testing for Efficient {LLM} Benchmarking},
  author    = {Marinelli, Ryan},
  booktitle = {Proceedings of the 40th International Workshop on Statistical Modelling},
  year      = {2026},
  address   = {Oslo, Norway},
}
```

## License

MIT — see [LICENSE](LICENSE).
