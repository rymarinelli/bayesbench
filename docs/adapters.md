# Adapter guide

Adapters convert external model/framework APIs into plain callables that work with `bayesbench`.

## OpenAI-compatible adapter

```python
from bayesbench.adapters.openai_compat import openai_model

model = openai_model("gpt-4o")
groq_model = openai_model(
    "llama-3.1-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
)
```

Use for OpenAI and OpenAI-compatible providers (Groq, Together, Ollama, vLLM, etc.).

## Anthropic adapter

```python
from bayesbench.adapters.anthropic_adapter import anthropic_model

model = anthropic_model("claude-opus-4-6")
```

Use for Claude model families through Anthropic APIs.

## Hugging Face adapter

```python
from bayesbench.adapters.huggingface import hf_model, hf_dataset

model = hf_model("meta-llama/Llama-3.1-8B-Instruct")
dataset = hf_dataset("openai/gsm8k", split="test")
```

Use when your workloads are centered on Hugging Face-hosted models/datasets.

## Inspect adapter

```python
from bayesbench.adapters.inspect_ai import from_inspect_dataset, inspect_model
```

Use for AISI Inspect datasets, model wrappers, and scorers without rewriting task logic.

## MTEB adapter

```python
from bayesbench.adapters.mteb import mteb_sts_dataset, st_model, sts_score_fn
```

Use for embedding benchmarking and STS-style continuous metrics.

## OpenClaw adapter

```python
from bayesbench.adapters.openclaw import openclaw_agent

agent = openclaw_agent(my_agent)
```

Use for agent-vs-agent or agent-vs-LLM comparisons.

## Adapter selection checklist

- Need text-generation A/B across providers → **OpenAI-compatible** or **Anthropic**
- Need existing Inspect pipeline reuse → **Inspect**
- Need embedding STS benchmarks → **MTEB**
- Need full agent-loop benchmarking → **OpenClaw**
- Need HF-centric model/data loading → **Hugging Face**
