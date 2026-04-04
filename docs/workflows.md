# Workflows: LLM and agentic benchmarking

This page shows practical workflows for benchmarking both **LLM responses** and
**agent behaviour** with `bayesbench`.

## LLM benchmarking workflows

### 1) AISI Inspect workflow (task/eval datasets)

Use this when your eval pipeline already uses Inspect datasets and model routing.

```python
from inspect_ai.dataset import hf_dataset, FieldSpec
from bayesbench import BayesianBenchmark
from bayesbench.adapters.inspect_ai import (
    from_inspect_dataset,
    inspect_model,
    exact_match_score,
)

bench = BayesianBenchmark(confidence=0.95)

problems = from_inspect_dataset(
    hf_dataset(
        "openai/gsm8k",
        split="test",
        sample_fields=FieldSpec(input="question", target="answer"),
    )
)

model_a = inspect_model("openai/gpt-4o")
model_b = inspect_model("openai/gpt-4o-mini")

@bench.task(dataset=problems, name="gsm8k")
def gsm8k(problem):
    return (
        exact_match_score(problem, model_a(problem)),
        exact_match_score(problem, model_b(problem)),
    )

print(bench.run().summary())
```

Why this is useful:
- Reuses Inspect datasets/models directly.
- Gives Bayesian early stopping without rewriting your eval stack.

---

### 2) MTEB workflow (embedding model benchmarking)

Use this to compare embedding models on STS tasks with continuous scores.

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
    name="sts_benchmark",
)

print(result.winner, result.p_a_beats_b, result.efficiency)
```

Why this is useful:
- Uses the right posterior family (`NormalPosterior`) for continuous metrics.
- Often reaches stable conclusions before full-dataset evaluation.

---

### 3) OpenAI-compatible provider workflow

Use this for fast A/B tests across OpenAI, Groq, Together, Ollama, and other
OpenAI-compatible providers.

```python
from bayesbench import BayesianBenchmark
from bayesbench.adapters.openai_compat import openai_model

bench = BayesianBenchmark(confidence=0.95)

model_a = openai_model("gpt-4o")
model_b = openai_model(
    "llama-3.1-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
)

result = bench.compare(
    model_a=model_a,
    model_b=model_b,
    dataset=problems,
    score_fn=lambda p, r: r.strip() == p["answer"],
    name="ab_eval",
)

print(result.summary())
```

## Agentic benchmarking workflows

### 1) OpenClaw agent-vs-agent benchmarking

Use this when evaluating complete agent loops (tool use, planning, retries),
not just single-shot text completion.

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
    score_fn=lambda p, output: int(output.strip() == p["expected"]),
    name="agent_exact_match",
)

print(result.summary())
```

---

### 2) Mixed workflow: LLM baseline vs agent system

Compare a pure LLM baseline against a tool-using agent on the same benchmark.

```python
from bayesbench import BayesianBenchmark
from bayesbench.adapters.openai_compat import openai_model
from bayesbench.adapters.openclaw import openclaw_agent

bench = BayesianBenchmark(confidence=0.95)

llm_baseline = openai_model("gpt-4o-mini")
agent_system = openclaw_agent(my_openclaw_agent)

result = bench.compare(
    model_a=agent_system,
    model_b=llm_baseline,
    dataset=tool_use_tasks,
    score_fn=lambda p, r: float(p["grader"](r)),
    name="agent_vs_llm",
)

print(result.winner, result.credible_interval_a, result.credible_interval_b)
```

## Suggested benchmarking checklist

- Use **binary scores** (`0/1`) for exact match, pass/fail, rubric pass.
- Use **continuous scores** (`0.0-1.0`) for semantic/judge-based quality.
- Start with `confidence=0.95`; raise to `0.99` for higher-stakes decisions.
- Set `min_samples` for noisy tasks to avoid premature stopping.
- Report both winner and **efficiency** (fraction of evaluations saved).
