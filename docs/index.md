# baysbench

Bayesian sequential benchmarking for LLMs and agents.

## Highlights

- Early stopping with posterior confidence thresholds.
- Adapters for OpenAI-compatible APIs, Anthropic, Inspect AI, MTEB, and OpenClaw agents.
- Pairwise comparison and N-model ranking.

## Install

```bash
pip install baysbench
pip install "baysbench[openclaw]"  # Optional OpenClaw adapter support
```

## OpenClaw adapter

```python
from baysbench.adapters.openclaw import openclaw_agent

agent = openclaw_agent(my_openclaw_agent)
response = agent({"input": "Solve 17 * 19"})
```
