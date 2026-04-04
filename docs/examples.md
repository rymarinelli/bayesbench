# Examples gallery

The repository ships runnable examples under `examples/`.
Use these as practical templates and adapt to your own tasks.

## 1) `quickstart.py` — smallest useful benchmark

Best when you want to verify setup quickly before integrating providers.

- Focus: basic A/B comparison and early stopping.
- Start here if you're new to `bayesbench`.

## 2) `llm_comparison.py` — LLM-vs-LLM workflow

Best when comparing two text-generation models using exact match or rubric scoring.

- Focus: model callable shape and binary scoring.
- Good baseline for task-oriented LLM evals.

## 3) `multi_model_ranking.py` — ranking many models

Best when you need a leaderboard rather than only pairwise A/B.

- Focus: `BayesianRanker`, pairwise posterior comparisons, rank summary.
- Useful for release gates and regression tracking.

## 4) `framework_adapters.py` — integration patterns

Best when your models and datasets come from external frameworks.

- Focus: adapters that normalize framework-specific APIs into benchmark callables.
- Good for teams with existing evaluation stacks.

## 5) `inspect_example.py` — AISI Inspect integration

Best when you already use Inspect datasets/tasks and want Bayesian stopping.

- Focus: `from_inspect_dataset`, `inspect_model`, and inspect-native wiring.
- Lets you keep Inspect pipelines while reducing evaluation cost.

## 6) `mteb_example.py` — embedding model comparisons

Best when evaluating embedding quality on STS-style tasks.

- Focus: continuous scores + `NormalPosterior`.
- Recommended for semantic similarity benchmarks.

## Suggested progression

1. Run `quickstart.py`.
2. Choose either `llm_comparison.py` or `mteb_example.py` based on your metric type.
3. Move to `multi_model_ranking.py` when comparing 3+ models.
4. Adopt adapter examples to connect your production providers.
