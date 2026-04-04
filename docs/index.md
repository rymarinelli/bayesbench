# bayesbench documentation

**Bayesian sequential benchmarking for LLMs and agents.**

`bayesbench` helps you stop evaluations as soon as posterior evidence is strong enough,
instead of evaluating every model on every example.

## What you'll find in these docs

- **[Getting started](getting-started.md):** installation, first benchmark, CLI usage.
- **[Workflows](workflows.md):** end-to-end templates for LLM and agentic evaluations.
- **[Examples gallery](examples.md):** copy-pasteable snippets mapped to the `examples/` folder.
- **[Concepts](concepts.md):** confidence, early stopping, posterior choices, and tuning tips.
- **[Adapters](adapters.md):** provider/framework integrations and when to use each.
- **[API reference](api-reference.md):** core classes and methods.

## Why bayesbench

- **Lower evaluation cost:** stop when evidence is sufficient.
- **Statistically principled:** Bayesian posteriors and credible intervals.
- **Flexible metrics:** binary and continuous scoring.
- **Practical integrations:** OpenAI-compatible APIs, Anthropic, Hugging Face, Inspect, MTEB, and OpenClaw.

## Typical evaluation journey

1. Pick a workflow that matches your stack (Inspect, MTEB, OpenAI-compatible, OpenClaw).
2. Start with `confidence=0.95` and a small `min_samples` (for fast iteration).
3. Run benchmarks and inspect **winner**, **P(A > B)**, and **efficiency**.
4. Raise confidence to `0.99` for higher-stakes final runs.
5. Export reports to CSV/JSON for tracking over time.

## Need a fast start?

Go to **[Getting started](getting-started.md)** for a minimal script you can run in minutes.
