# Contributing to baysbench

Thank you for your interest in contributing! This document covers everything
you need to get started.

## Development setup

```bash
git clone https://github.com/rymarinelli/baysbench
cd baysbench
pip install -e ".[dev]"
```

This installs baysbench in editable mode with `pytest`, `pytest-asyncio`,
`pytest-cov`, `ruff`, and `mypy`.

## Running tests

```bash
pytest                          # run the full suite
pytest -q                       # quiet output
pytest tests/test_core.py       # single file
pytest --cov=baysbench          # with coverage
```

All 148+ tests are mock-based and run without any API keys or external services.

## Linting and type checking

```bash
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
mypy src/baysbench              # type check
```

CI enforces all three on every push.

## Adding a new adapter

1. Create `src/baysbench/adapters/<framework>.py`.
2. Use `from .base import _require` for lazy imports — adapters must not fail
   at import time if the underlying library is absent.
3. Add the library as an optional extra in `pyproject.toml`.
4. Add mock-based tests in `tests/test_<framework>_adapter.py`.
5. Document the adapter in `src/baysbench/adapters/__init__.py` and `README.md`.

## Adding a new posterior

1. Create `src/baysbench/posteriors/<name>.py`.
2. Subclass `Posterior` from `baysbench.posteriors.base` and implement all
   abstract methods: `observe_one`, `prob_beats`, `credible_interval`, `mean`.
3. Export from `src/baysbench/posteriors/__init__.py`.
4. Add tests in `tests/test_posteriors.py`.

## Pull request checklist

- [ ] Tests pass locally (`pytest`)
- [ ] No lint errors (`ruff check`)
- [ ] No type errors (`mypy src/baysbench --ignore-missing-imports`)
- [ ] New features have tests
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Commit style

Use conventional commits where practical:

```
feat: add DirichletPosterior for multi-class outcomes
fix: correct NormalPosterior credible interval at n=1
docs: add Groq example to README
test: add ranking convergence edge-case tests
```

## Reporting issues

Please open a GitHub issue with:
- baysbench version (`baysbench --version`)
- Python version
- Minimal reproducible example
- Expected vs actual behaviour
