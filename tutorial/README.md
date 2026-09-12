# Stop When the Evidence Is Enough — tutorial package

Sequential Bayesian evaluation for LLMs and agents, taught with
**BayesBench**. Everything runs on CPU with no API keys and no model
downloads — the solvers are synthetic, which is the point: we *know*
the ground truth, so we can judge the stopping rules against reality.

## Contents

| Path | What it is |
|---|---|
| `stop_when_the_evidence_is_enough.ipynb` | The core notebook, 6 sections, executed with outputs |
| `exercises/exercises.ipynb` | 6 exercises |
| `exercises/solutions.ipynb` | Worked solutions (executed) |
| `slides/slides.md` + `slides.pdf` + `slides.html` | Presentation deck (Marp) |
| `bayesbench_tutorial.py` | Shared helpers: synthetic datasets + solvers |
| `data/` | Generated synthetic datasets (JSONL) |
| `results/` | Exported decision traces (JSON/CSV) from Section 5 |

## Environment

Python 3.11 or 3.12, CPU only:

```bash
pip install -r requirements.txt
```

This installs `bayesbench` plus numpy, scipy, matplotlib, pandas, and
jupyter. No API keys are required anywhere in the tutorial.

## Running

```bash
jupyter notebook stop_when_the_evidence_is_enough.ipynb
```

The notebook regenerates its synthetic datasets on first run (pure
Python, deterministic) and writes exported traces to `results/`.

## Regenerating everything from scratch

```bash
python -m pip install -r requirements.txt
jupyter nbconvert --execute --inplace stop_when_the_evidence_is_enough.ipynb
jupyter nbconvert --execute --inplace exercises/solutions.ipynb
```

Slides (requires Node):

```bash
npx @marp-team/marp-cli slides/slides.md --pdf --html --allow-local-files
```

## Learning objectives

1. Quantify the cost of fixed-budget evaluation and when early stopping
   can avoid it.
2. Explain Bayesian posterior updating for binary and continuous scores.
3. Distinguish superiority, equivalence (ROPE), and insufficient
   evidence — and never confuse P(A>B) ≈ 0.5 with equivalence.
4. Calibrate a stopping rule across effect sizes and seeds, and explain
   the optional-stopping trap.
5. Use confidence sequences for any-time error control (≤ 5% false
   decisions at any stopping time).
6. Run streaming, traced, paired, and lower-is-better evaluations and
   export reproducible evidence.

## Audience and prerequisites

ML practitioners and advanced undergraduates. Basic Python; probability
up to Bayes' rule and conjugate updating. No Bayesian statistics beyond
that is assumed — the posterior math is done by the package.

## References

- tinyBenchmarks — Maia Polo et al., ICML 2024
- MixEval — Ni et al., NeurIPS 2024
- How Benchmark Prediction from Fewer Data Misses the Mark — NeurIPS 2025
- Waudby-Smith & Ramdas, *Estimating means of bounded random variables
  by betting*, JRSS-B 2023
- BayesBench: https://github.com/rymarinelli/bayesbench
