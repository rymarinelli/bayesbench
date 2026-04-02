"""mteb_example.py — Bayesian benchmarking of embedding models with MTEB.

Demonstrates how to use baysbench to compare sentence-transformer embedding
models on MTEB tasks with Bayesian sequential testing, using NormalPosterior
for continuous STS scores and BetaPosterior for classification accuracy.

Requires:
    pip install baysbench[mteb]

Run with:
    python examples/mteb_example.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baysbench import BayesianBenchmark, BayesianRanker
from baysbench.adapters.mteb import (
    _KNNClassifier,
    make_classification_score_fn,
    sts_score_fn,
)
from baysbench.posteriors import NormalPosterior

# ---------------------------------------------------------------------------
# Synthetic STS dataset (replace with mteb_sts_dataset("STSBenchmark"))
# ---------------------------------------------------------------------------

def make_synthetic_sts(n: int, seed: int = 42) -> list[dict]:
    """Sentence pairs with gold cosine-similarity scores in [0, 1]."""
    rng = np.random.default_rng(seed)
    return [
        {
            "sentence1": f"sentence_{i}_a",
            "sentence2": f"sentence_{i}_b",
            "gold_score": float(rng.uniform(0, 1)),
            "raw_score": float(rng.uniform(0, 5)),
            "task": "synthetic",
        }
        for i in range(n)
    ]

# ---------------------------------------------------------------------------
# Synthetic embedding models
# ---------------------------------------------------------------------------

DIM = 32
rng_global = np.random.default_rng(0)

def _make_model(accuracy: float, seed: int):
    """An embedding model that tracks gold similarity with given accuracy."""
    rng = np.random.default_rng(seed)

    def encode(problem: dict) -> np.ndarray:
        gold_cos = problem["gold_score"] * 2 - 1  # [0,1] → [-1,1]
        # Perturb: higher accuracy → less noise
        noise = rng.normal(0, 1 - accuracy + 0.05)
        cos = float(np.clip(gold_cos + noise, -1, 1))
        e1 = np.zeros(DIM); e1[0] = 1.0
        e2 = np.zeros(DIM)
        e2[0] = cos
        e2[1] = np.sqrt(max(0.0, 1.0 - cos ** 2))
        return np.stack([e1, e2])

    encode.__baysbench_model__ = f"model_acc{accuracy:.0%}"
    return encode

model_strong = _make_model(0.90, seed=1)   # tracks gold closely
model_medium = _make_model(0.70, seed=2)
model_weak   = _make_model(0.40, seed=3)


# ===========================================================================
# Example 1 — STS pairwise comparison with @benchmark decorator
# ===========================================================================
print("=" * 60)
print("Example 1: STS pairwise comparison")
print("=" * 60)

from baysbench import benchmark

@benchmark(
    model_a=model_strong,
    model_b=model_weak,
    dataset=make_synthetic_sts(300),
    posterior_factory=NormalPosterior,
    confidence=0.95,
    min_samples=10,
    skip_threshold=0.51,   # disable non-discriminating skip for this demo
    name="sts_strong_vs_weak",
)
def sts_score(problem, embeddings):
    return sts_score_fn(problem, embeddings)

result = sts_score.run()
print(result)
print(f"Model A mean: {result.posterior_a.mean:.3f}")
print(f"Model B mean: {result.posterior_b.mean:.3f}")
print()


# ===========================================================================
# Example 2 — STS multi-task suite with @suite decorator
# ===========================================================================
print("=" * 60)
print("Example 2: STS multi-task @suite")
print("=" * 60)

from baysbench import suite

@suite(confidence=0.95, posterior_factory=NormalPosterior, min_samples=10)
class STSBenchmark:
    dataset = make_synthetic_sts(200)

    @staticmethod
    def task_short_sentences(problem):
        a = sts_score_fn(problem, model_strong(problem))
        b = sts_score_fn(problem, model_medium(problem))
        return a, b

    @staticmethod
    def task_long_sentences(problem):
        a = sts_score_fn(problem, model_strong(problem))
        b = sts_score_fn(problem, model_weak(problem))
        return a, b

report = STSBenchmark.run()
print(report.summary())
print()


# ===========================================================================
# Example 3 — Rank N embedding models simultaneously
# ===========================================================================
print("=" * 60)
print("Example 3: Rank N embedding models")
print("=" * 60)

ranker = BayesianRanker(
    confidence=0.95,
    min_samples=10,
    skip_threshold=0.51,
    posterior_factory=NormalPosterior,
)
ranker.add_model("strong-90%", model_strong)
ranker.add_model("medium-70%", model_medium)
ranker.add_model("weak-40%",   model_weak)

ranking = ranker.rank(
    dataset=make_synthetic_sts(400),
    score_fn=sts_score_fn,
)
print(ranking.summary())
print()


# ===========================================================================
# Example 4 — Classification with k-NN (binary BetaPosterior)
# ===========================================================================
print("=" * 60)
print("Example 4: Classification with k-NN")
print("=" * 60)

# Synthetic 2-class dataset: embeddings centred at [±1, 0, ..., 0]
np.random.seed(7)
CDIM = 16

def _make_classification_data(n_train=100, n_test=50):
    rng = np.random.default_rng(55)
    A_train = rng.normal([1] * CDIM, 0.3, (n_train // 2, CDIM))
    B_train = rng.normal([-1] * CDIM, 0.3, (n_train // 2, CDIM))
    A_test  = rng.normal([1] * CDIM, 0.3, (n_test // 2,  CDIM))
    B_test  = rng.normal([-1] * CDIM, 0.3, (n_test // 2,  CDIM))
    # Normalise
    for arr in [A_train, B_train, A_test, B_test]:
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

    train = (
        [{"text": f"a{i}", "label": "A"} for i in range(n_train // 2)]
        + [{"text": f"b{i}", "label": "B"} for i in range(n_train // 2)]
    )
    test = (
        [{"text": f"ta{i}", "label": "A"} for i in range(n_test // 2)]
        + [{"text": f"tb{i}", "label": "B"} for i in range(n_test // 2)]
    )
    embed_map = (
        {f"a{i}": A_train[i] for i in range(n_train // 2)}
        | {f"b{i}": B_train[i] for i in range(n_train // 2)}
        | {f"ta{i}": A_test[i] for i in range(n_test // 2)}
        | {f"tb{i}": B_test[i] for i in range(n_test // 2)}
    )
    return train, test, embed_map

train_data, test_data, embed_map = _make_classification_data()

# Strong classifier: embeddings cluster perfectly
def cls_model_strong(problem): return embed_map[problem["text"]]

# Weak classifier: add lots of noise
def cls_model_weak(problem):
    e = embed_map[problem["text"]] + np.random.normal(0, 2, CDIM)
    return e / np.linalg.norm(e)

# Build score fns — use our KNN directly with pre-computed train embeddings
clf_strong = _KNNClassifier(k=3)
clf_strong.fit(
    np.stack([embed_map[d["text"]] for d in train_data]),
    [d["label"] for d in train_data],
)

np.random.seed(42)
weak_train_embs = np.stack([
    embed_map[d["text"]] + np.random.normal(0, 2, CDIM)
    for d in train_data
])
weak_train_embs /= np.linalg.norm(weak_train_embs, axis=1, keepdims=True)
clf_weak = _KNNClassifier(k=3)
clf_weak.fit(weak_train_embs, [d["label"] for d in train_data])

bench_cls = BayesianBenchmark(confidence=0.95, min_samples=5)

@bench_cls.task(dataset=test_data, name="classification")
def cls_task(problem):
    return (
        clf_strong.predict(cls_model_strong(problem)) == problem["label"],
        clf_weak.predict(cls_model_weak(problem))   == problem["label"],
    )

cls_report = bench_cls.run()
print(cls_report.summary())


# ===========================================================================
# Real MTEB usage (commented out — requires pip install baysbench[mteb])
# ===========================================================================
print()
print("=" * 60)
print("Real MTEB usage (uncomment to run with real models):")
print("=" * 60)
print("""
from baysbench.adapters.mteb import mteb_sts_dataset, st_model

# STS comparison
bench = BayesianBenchmark(confidence=0.95, posterior_factory=NormalPosterior)
problems = mteb_sts_dataset("STSBenchmark", max_samples=500)
model_a = st_model("sentence-transformers/all-mpnet-base-v2")
model_b = st_model("sentence-transformers/all-MiniLM-L6-v2")
result = bench.compare(model_a, model_b, sts_score_fn, problems)
print(result)

# Ranking
from baysbench.adapters.mteb import st_model
ranker = BayesianRanker(confidence=0.95, posterior_factory=NormalPosterior)
for name in [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]:
    ranker.add_model(name.split("/")[-1], st_model(name))
result = ranker.rank(dataset=mteb_sts_dataset("STSBenchmark"), score_fn=sts_score_fn)
print(result.summary())

# Classification
from baysbench.adapters.mteb import (
    mteb_classification_dataset, mteb_classification_model, make_classification_score_fn
)
data = mteb_classification_dataset("Banking77Classification.v2")
model_a = mteb_classification_model("all-mpnet-base-v2")
model_b = mteb_classification_model("all-MiniLM-L6-v2")
score_a = make_classification_score_fn(data["train"], model_a)
score_b = make_classification_score_fn(data["train"], model_b)
bench = BayesianBenchmark(confidence=0.95)
@bench.task(dataset=data["test"], name="banking77")
def banking(problem):
    return score_a(problem, model_a(problem)), score_b(problem, model_b(problem))
print(bench.run().summary())
""")
