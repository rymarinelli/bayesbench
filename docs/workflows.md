# Workflow playbooks

This page provides practical, end-to-end workflows for common benchmarking setups.
Each workflow includes when to use it, a reference implementation, and expected outputs.

## Workflow 1: LLM A/B with OpenAI-compatible APIs

**Use when:** comparing two chat/completion models (OpenAI, Groq, Together, Ollama, vLLM).

```python
from bayesbench import BayesianBenchmark
from bayesbench.adapters.openai_compat import openai_model

bench = BayesianBenchmark(confidence=0.95, min_samples=5)

model_a = openai_model("gpt-4o")
model_b = openai_model("llama-3.1-70b-versatile", base_url="https://api.groq.com/openai/v1")

result = bench.compare(
    model_a=model_a,
    model_b=model_b,
    dataset=problems,
    score_fn=lambda p, r: int(r.strip() == p["answer"]),
    name="openai_compat_ab",
)

print(result.winner, result.p_a_beats_b, result.efficiency)
```

**Typical output interpretation:**

- `winner` tells you which model has stronger posterior support.
- `p_a_beats_b` indicates certainty.
- `efficiency` measures cost saved from stopping early.

---

## Workflow 2: Multi-task suite for release gating

**Use when:** you need one report across multiple benchmarks (math, reasoning, coding, etc.).

```python
from bayesbench import BayesianBenchmark

bench = BayesianBenchmark(confidence=0.95)

@bench.task(dataset=math_problems, name="math")
def math_task(problem):
    return model_a(problem) == problem["answer"], model_b(problem) == problem["answer"]

@bench.task(dataset=reasoning_problems, name="reasoning")
def reasoning_task(problem):
    return score_reasoning(problem, model_a(problem)), score_reasoning(problem, model_b(problem))

report = bench.run(verbose=True)
print(report.summary())
```

**Why teams use this:**

- One unified report for go/no-go decisions.
- Early stopping can reduce cost on each task independently.

---

## Workflow 3: Inspect-native dataset and model pipeline

**Use when:** your existing evaluation stack is built around AISI Inspect.

```python
from inspect_ai.dataset import hf_dataset, FieldSpec
from bayesbench import BayesianBenchmark
from bayesbench.adapters.inspect_ai import from_inspect_dataset, inspect_model, exact_match_score

bench = BayesianBenchmark(confidence=0.95)

problems = from_inspect_dataset(
    hf_dataset("openai/gsm8k", split="test", sample_fields=FieldSpec(input="question", target="answer"))
)

model_a = inspect_model("openai/gpt-4o")
model_b = inspect_model("openai/gpt-4o-mini")

@bench.task(dataset=problems, name="gsm8k")
def gsm8k(problem):
    return exact_match_score(problem, model_a(problem)), exact_match_score(problem, model_b(problem))

print(bench.run().summary())
```

---

## Workflow 4: Embedding benchmarking with MTEB

**Use when:** comparing embedding models on STS/semantic similarity tasks.

```python
from bayesbench import BayesianBenchmark
from bayesbench.posteriors import NormalPosterior
from bayesbench.adapters.mteb import mteb_sts_dataset, st_model, sts_score_fn

bench = BayesianBenchmark(confidence=0.95, posterior_factory=NormalPosterior)

result = bench.compare(
    model_a=st_model("sentence-transformers/all-mpnet-base-v2"),
    model_b=st_model("sentence-transformers/all-MiniLM-L6-v2"),
    score_fn=sts_score_fn,
    dataset=mteb_sts_dataset("STSBenchmark", max_samples=500),
    name="mteb_sts",
)

print(result.summary())
```

---

## Workflow 5: Agent-vs-agent with OpenClaw

**Use when:** evaluating full agent loops (planning, tools, retries) instead of raw completions.

```python
from bayesbench import BayesianBenchmark
from bayesbench.adapters.openclaw import openclaw_agent

bench = BayesianBenchmark(confidence=0.95)

agent_a = openclaw_agent(react_agent)
agent_b = openclaw_agent(planner_executor_agent)

result = bench.compare(
    model_a=agent_a,
    model_b=agent_b,
    dataset=tasks,
    score_fn=lambda p, out: int(out.strip() == p["expected"]),
    name="openclaw_agent_match",
)

print(result.summary())
```

## Workflow tuning checklist

- Binary metric? Use default posterior (`BetaPosterior`).
- Continuous metric? Use `NormalPosterior`.
- High-stakes decisions? Increase `confidence` to `0.99`.
- Noisy tasks? Increase `min_samples`.
- Reporting to stakeholders? Include both winner and efficiency.
