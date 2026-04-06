# Changelog

All notable changes to bayesbench are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [0.4.0] — 2026-04-02

### Added
- **`DirichletPosterior`** (`bayesbench.posteriors.DirichletPosterior`)
  - Dirichlet-Multinomial conjugate for K-class categorical outcomes
  - Handles multiple-choice accuracy (set `k=4` for A/B/C/D), sentiment labels, or any K-way classification
  - `target_class` parameter selects which category counts as "correct"
  - `k=2` is a drop-in replacement for `BetaPosterior(0.5, 0.5)` for binary tasks
- **`GammaPosterior`** (`bayesbench.posteriors.GammaPosterior`)
  - Gamma-Poisson conjugate for non-negative count/rate observations
  - Ideal for token count, latency (ms), API cost benchmarking
  - `higher_is_better=False` flips `prob_beats` direction for "lower is better" metrics
- **`bayesbench.compare()`** — one-liner pairwise comparison with no class instantiation
- **`bayesbench.rank()`** — one-liner multi-model ranking; accepts `dict` or `list[tuple]` of models
- **`TaskResult.to_dict()`** — JSON-serializable dict with means, credible intervals, efficiency
- **`BenchmarkReport.to_dict()`** and **`BenchmarkReport.to_dataframe()`** — report export
- **`RankingResult.to_dict()`** and **`RankingResult.to_dataframe()`** — ranking export
- **`ModelRanking.to_dict()`** — per-model statistics as plain dict
- **`verbose=True`** on `BayesianBenchmark.run()`, `BayesianBenchmark.compare()`, and `BayesianRanker.rank()` — tqdm progress bars at task and problem level
- **Structured logging** via `logging.getLogger("bayesbench.benchmark")` and `"bayesbench.ranking"` — INFO-level task summaries, DEBUG-level per-problem details
- **`tqdm>=4.64`** added as a core dependency
- **`src/bayesbench/py.typed`** — PEP 561 typed-package marker
- **`[tool.mypy]`** and **`[tool.coverage.*]`** sections in `pyproject.toml`
- 57 new tests for `DirichletPosterior`, `GammaPosterior`, and the convenience API

### Changed
- `pyproject.toml` version bumped to `0.4.0`; `Development Status` → `4 - Beta`
- `dev` extra now includes `pytest-cov` and `mypy`
- `posteriors/__init__.py` updated with a posterior-selection table in the module docstring

---

## [0.3.0] — 2026-04-02

### Added
- **AISI Inspect adapter** (`bayesbench.adapters.inspect_ai`)
  - `from_inspect_dataset()` — converts any Inspect `Dataset` to bayesbench problem dicts
  - `inspect_model()` / `inspect_model_async()` — synchronous and async wrappers around `inspect_ai.get_model()`
  - Score functions: `exact_match_score`, `includes_score`, `any_target_score`, `pattern_score`, `choice_score`
- **MTEB adapter** (`bayesbench.adapters.mteb`)
  - `mteb_sts_dataset()` — loads MTEB STS tasks as sentence-pair dicts with normalised gold scores
  - `mteb_classification_dataset()` — loads train + test splits for classification tasks
  - `st_model()` / `mteb_classification_model()` — SentenceTransformer wrappers
  - `sts_score_fn()` — agreement score for continuous STS metrics with `NormalPosterior`
  - `make_classification_score_fn()` — k-NN classifier pre-trained on the MTEB training split
  - `mteb_task_info()` — metadata lookup without loading data
- Optional extras: `bayesbench[inspect]`, `bayesbench[mteb]`
- 48 new tests for Inspect and MTEB adapters (all mock-based, no API calls)

---

## [0.2.0] — 2026-04-02

### Added
- **Pluggable posterior distributions** (`bayesbench.posteriors`)
  - Abstract `Posterior` base class — subclass to add custom Bayesian models
  - `BetaPosterior` refactored to implement the `Posterior` protocol
  - `NormalPosterior` — Normal-Inverse-Gamma conjugate model for continuous scores (BLEU, ROUGE, cosine similarity, LLM-judge)
- **`BayesianBenchmark.posterior_factory`** parameter — swap posteriors globally or per task
- **`BayesianRanker`** (`bayesbench.ranking`) — rank N models simultaneously
  - Individual posteriors per model; stops when all consecutive ranked pairs are statistically decided
  - `add_model()` chaining, `@ranker.evaluate` decorator, `rank_async()`
- **Framework adapters** (`bayesbench.adapters`)
  - `openai_compat`: `openai_model()` / `openai_model_async()` for OpenAI, Groq, Together AI, Fireworks, Ollama, vLLM, Azure
  - `anthropic_adapter`: `anthropic_model()` / `anthropic_model_async()`
  - `huggingface`: `hf_model()`, `hf_model_async()`, `hf_dataset()`
- Optional extras: `bayesbench[openai]`, `bayesbench[anthropic]`, `bayesbench[huggingface]`, `bayesbench[all]`
- `@suite` decorator gains `posterior_factory` and per-task `posterior_<name>` class attributes

### Changed
- `benchmark.py` engine is now fully posterior-generic; `BetaPosterior` remains the default
- `core.py` retains `prob_a_beats_b` / `is_non_discriminating` as thin backward-compatible shims

---

## [0.1.0] — 2026-04-02

### Added
- Initial production release
- `BayesianBenchmark` with sequential Beta-Bernoulli stopping
- Three decorator styles: `@benchmark`, `@bench.task`, `@suite`
- `BetaPosterior` with Jeffreys prior, `prob_a_beats_b` (closed-form integration), `is_non_discriminating`
- `compare()` and `compare_async()` for direct pairwise comparison
- `BenchmarkReport` with `summary()` and aggregate efficiency statistics
- `bayesbench` CLI with `--confidence`, `--skip-threshold`, `--min-samples` flags
- 42 tests covering all statistical primitives and decorator styles

[0.3.0]: https://github.com/rymarinelli/bayesbench/releases/tag/v0.3.0
[0.2.0]: https://github.com/rymarinelli/bayesbench/releases/tag/v0.2.0
[0.1.0]: https://github.com/rymarinelli/bayesbench/releases/tag/v0.1.0
