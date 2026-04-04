"""Integration with MTEB (Massive Text Embedding Benchmark).

MTEB is the standard benchmark for text embedding models covering
Classification, Clustering, Retrieval, Reranking, STS, Summarization,
and more across 100+ datasets and 50+ languages.

This adapter lets you use baysbench's Bayesian sequential testing to compare
embedding models on MTEB tasks with dramatic cost reductions vs full runs.

Install dependencies::

    pip install baysbench[mteb]

Supported task types
--------------------
- **STS** (Semantic Textual Similarity) — compare sentence-pair cosine
  similarities against gold scores.  Use :class:`~baysbench.posteriors.NormalPosterior`.
- **Classification** — embed texts, classify with k-NN, compare accuracy.
  Use the default :class:`~baysbench.posteriors.BetaPosterior`.

Usage — STS comparison::

    from baysbench import BayesianBenchmark
    from baysbench.posteriors import NormalPosterior
    from baysbench.adapters.mteb import (
        mteb_sts_dataset,
        st_model,
        sts_score_fn,
    )

    bench = BayesianBenchmark(confidence=0.95, posterior_factory=NormalPosterior)

    model_a = st_model("sentence-transformers/all-mpnet-base-v2")
    model_b = st_model("sentence-transformers/all-MiniLM-L6-v2")

    result = bench.compare(
        model_a=model_a,
        model_b=model_b,
        score_fn=sts_score_fn,
        dataset=mteb_sts_dataset("STSBenchmark"),
        name="sts_benchmark",
    )
    print(result)

Usage — Classification comparison::

    from baysbench import BayesianBenchmark
    from baysbench.adapters.mteb import mteb_classification_dataset, st_model

    data = mteb_classification_dataset("Banking77Classification.v2")
    model_a = st_model("sentence-transformers/all-mpnet-base-v2")
    model_b = st_model("sentence-transformers/all-MiniLM-L6-v2")

    score_a = make_classification_score_fn(data["train"], model_a)
    score_b = make_classification_score_fn(data["train"], model_b)
    callable_a = mteb_classification_model(model_a)
    callable_b = mteb_classification_model(model_b)

    bench = BayesianBenchmark(confidence=0.95)

    @bench.task(dataset=data["test"], name="banking77")
    def banking77(problem):
        return (
            score_a(problem, callable_a(problem)),
            score_b(problem, callable_b(problem)),
        )

    print(bench.run().summary())

Usage — rank N embedding models::

    from baysbench import BayesianRanker
    from baysbench.posteriors import NormalPosterior
    from baysbench.adapters.mteb import mteb_sts_dataset, st_model, sts_score_fn

    ranker = BayesianRanker(confidence=0.95, posterior_factory=NormalPosterior)
    for name in [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ]:
        ranker.add_model(name.split("/")[-1], st_model(name))

    result = ranker.rank(
        dataset=mteb_sts_dataset("STSBenchmark", max_samples=500),
        score_fn=sts_score_fn,
    )
    print(result.summary())
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import _require

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def mteb_sts_dataset(
    task_name: str,
    *,
    split: str = "test",
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Load an MTEB STS task as a list of sentence-pair problem dicts.

    Each dict contains:
    - ``"sentence1"``, ``"sentence2"`` — the sentence pair
    - ``"gold_score"`` — normalised similarity in [0, 1] (raw MTEB scores
      are divided by their task maximum, typically 5.0)
    - ``"raw_score"`` — original score before normalisation
    - ``"task"`` — task name for reference

    Args:
        task_name: MTEB task name, e.g. ``"STSBenchmark"``,
                   ``"STS12"``, ``"SICK-R"``.
        split: Dataset split (``"test"``, ``"dev"``, ``"train"``).
        max_samples: Truncate after loading (applied after optional shuffle).
        shuffle: Randomly shuffle before truncation.
        seed: Random seed for shuffling.

    Returns:
        List of problem dicts.

    Example::

        from baysbench.adapters.mteb import mteb_sts_dataset
        problems = mteb_sts_dataset("STSBenchmark", max_samples=300)
    """
    _require("mteb", "mteb")
    import mteb

    task = mteb.get_task(task_name)
    task.load_data(eval_splits=[split])
    raw = task.dataset[split]

    # Determine score scale — MTEB STS typically uses 0-5 but some tasks use 0-1
    scores = [float(row.get("score", row.get("similarity_score", 0))) for row in raw]
    scale = max(scores) if scores else 5.0
    if scale <= 1.0:
        scale = 1.0  # already normalised

    problems: list[dict] = []
    for row in raw:
        raw_score = float(row.get("score", row.get("similarity_score", 0)))
        problems.append(
            {
                "sentence1": row.get("sentence1", row.get("text1", "")),
                "sentence2": row.get("sentence2", row.get("text2", "")),
                "gold_score": raw_score / scale,
                "raw_score": raw_score,
                "task": task_name,
            }
        )

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(problems))
        problems = [problems[i] for i in idx]

    if max_samples is not None:
        problems = problems[:max_samples]

    return problems


def mteb_classification_dataset(
    task_name: str,
    *,
    test_split: str = "test",
    train_split: str = "train",
    max_test_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Load an MTEB classification task as ``{"train": [...], "test": [...]}``.

    Each element is a dict with ``"text"`` and ``"label"`` keys.

    Args:
        task_name: MTEB task name, e.g. ``"Banking77Classification.v2"``,
                   ``"EmotionClassification"``, ``"AmazonReviewsClassification"``.
        test_split: Name of the evaluation split.
        train_split: Name of the training split (used to fit the k-NN
                     classifier in :func:`make_classification_score_fn`).
        max_test_samples: Truncate the test set (applied after optional shuffle).
        shuffle: Shuffle the test set before truncation.
        seed: Random seed for shuffling.

    Returns:
        ``{"train": list[dict], "test": list[dict]}``

    Example::

        from baysbench.adapters.mteb import mteb_classification_dataset
        data = mteb_classification_dataset("Banking77Classification.v2")
        # data["train"] → [{text: ..., label: ...}, ...]
        # data["test"]  → [{text: ..., label: ...}, ...]
    """
    _require("mteb", "mteb")
    import mteb

    task = mteb.get_task(task_name)
    splits_to_load = list({test_split, train_split})
    task.load_data(eval_splits=splits_to_load)

    def _extract(split_name: str) -> list[dict]:
        split_data = task.dataset[split_name]
        rows = []
        for row in split_data:
            # MTEB classification tasks use "text"/"label" or "sentence"/"label"
            text = row.get("text", row.get("sentence", row.get("input", "")))
            label = row.get("label", row.get("labels", row.get("class", None)))
            rows.append({"text": text, "label": label, "task": task_name})
        return rows

    train_data = _extract(train_split)
    test_data = _extract(test_split)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(test_data))
        test_data = [test_data[i] for i in idx]

    if max_test_samples is not None:
        test_data = test_data[:max_test_samples]

    return {"train": train_data, "test": test_data}


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


def st_model(
    model_name_or_model: str | Any,
    *,
    device: str | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> Callable[[dict], np.ndarray]:
    """Create a model callable from a SentenceTransformer model.

    For **STS tasks** the callable returns a ``(2, dim)`` array of normalised
    embeddings for the sentence pair (``sentence1``, ``sentence2``).

    Pass this callable as ``model_a`` / ``model_b`` to
    :meth:`~baysbench.BayesianBenchmark.compare` together with
    :func:`sts_score_fn`.

    Args:
        model_name_or_model: HuggingFace model ID string *or* an already-loaded
                              ``SentenceTransformer`` instance.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``).  ``None``
                lets sentence-transformers auto-detect.
        batch_size: Encoding batch size.
        normalize: Normalise embeddings to unit length (required for cosine
                   similarity via dot product).

    Returns:
        ``callable(problem: dict) -> np.ndarray`` where the array has shape
        ``(2, embedding_dim)`` for STS problems.

    Example::

        from baysbench.adapters.mteb import st_model, sts_score_fn

        model = st_model("sentence-transformers/all-MiniLM-L6-v2")
        problem = {"sentence1": "A dog runs.", "sentence2": "A puppy jogs.", "gold_score": 0.8}
        embeddings = model(problem)   # shape (2, 384)
        score = sts_score_fn(problem, embeddings)
    """
    _require("sentence_transformers", "mteb")
    from sentence_transformers import SentenceTransformer

    if isinstance(model_name_or_model, str):
        _model = SentenceTransformer(model_name_or_model, device=device)
        _name = model_name_or_model
    else:
        _model = model_name_or_model
        _name = getattr(_model, "name_or_path", repr(_model))

    def encode_sts(problem: dict) -> np.ndarray:
        sentences = [problem["sentence1"], problem["sentence2"]]
        embeddings = _model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings  # shape (2, dim)

    encode_sts.__baysbench_model__ = _name  # type: ignore[attr-defined]
    return encode_sts


def mteb_classification_model(
    model_name_or_model: str | Any,
    *,
    device: str | None = None,
    batch_size: int = 32,
    normalize: bool = True,
) -> Callable[[dict], np.ndarray]:
    """Create a model callable for **classification** tasks.

    Returns a ``(embedding_dim,)`` array for a single text, to be used
    with :func:`make_classification_score_fn`.

    Args:
        model_name_or_model: HuggingFace model ID or SentenceTransformer instance.
        device: Device string.
        batch_size: Encoding batch size.
        normalize: Normalise embeddings.

    Returns:
        ``callable(problem: dict) -> np.ndarray`` of shape ``(dim,)``.
    """
    _require("sentence_transformers", "mteb")
    from sentence_transformers import SentenceTransformer

    if isinstance(model_name_or_model, str):
        _model = SentenceTransformer(model_name_or_model, device=device)
        _name = model_name_or_model
    else:
        _model = model_name_or_model
        _name = getattr(_model, "name_or_path", repr(_model))

    def encode_single(problem: dict) -> np.ndarray:
        embeddings = _model.encode(
            [problem["text"]],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings[0]  # shape (dim,)

    encode_single.__baysbench_model__ = _name  # type: ignore[attr-defined]
    return encode_single


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def sts_score_fn(problem: dict, embeddings: np.ndarray) -> float:
    """Score an STS problem: agreement between predicted and gold similarity.

    Given normalised embeddings (shape ``(2, dim)``), computes cosine
    similarity (dot product), maps to ``[0, 1]``, then returns
    ``1 - |predicted - gold|`` so *higher = better agreement*.

    Use with :class:`~baysbench.posteriors.NormalPosterior` since the score
    is continuous.

    Args:
        problem: Problem dict with ``"gold_score"`` in ``[0, 1]``.
        embeddings: ``np.ndarray`` of shape ``(2, dim)`` from :func:`st_model`.

    Returns:
        Agreement score in ``[0, 1]``.
    """
    emb1, emb2 = embeddings[0], embeddings[1]
    cos_sim = float(np.dot(emb1, emb2))
    # Normalised embeddings → cosine similarity ∈ [-1, 1]; map to [0, 1]
    pred = (cos_sim + 1.0) / 2.0
    gold = float(problem["gold_score"])
    return 1.0 - abs(pred - gold)


# ---------------------------------------------------------------------------
# k-NN classifier for classification tasks
# ---------------------------------------------------------------------------


class _KNNClassifier:
    """Minimal k-nearest-neighbour classifier for embedding-based tasks."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._embeddings: np.ndarray | None = None
        self._labels: list = []

    def fit(self, embeddings: np.ndarray, labels: list) -> None:
        self._embeddings = embeddings
        self._labels = labels

    def predict(self, query: np.ndarray) -> Any:
        assert self._embeddings is not None, "Call fit() first."
        # Cosine similarity (embeddings assumed normalised)
        sims = self._embeddings @ query
        top_k = np.argsort(sims)[-self.k :]
        votes = [self._labels[i] for i in top_k]
        return max(set(votes), key=votes.count)


def make_classification_score_fn(
    train_data: list[dict],
    model_callable: Callable[[dict], np.ndarray],
    *,
    k: int = 5,
    batch_size: int = 256,
) -> Callable[[dict, np.ndarray], bool]:
    """Build a classification score function using k-NN on training embeddings.

    Pre-computes embeddings for all training examples once, then for each
    test problem checks whether a k-NN lookup returns the correct label.

    Args:
        train_data: List of ``{"text": ..., "label": ...}`` dicts (the
                    training split from :func:`mteb_classification_dataset`).
        model_callable: A callable returned by :func:`mteb_classification_model`.
        k: Number of nearest neighbours.
        batch_size: Batch size for pre-computing train embeddings.

    Returns:
        ``score_fn(problem: dict, response: np.ndarray) -> bool``

    Example::

        from baysbench.adapters.mteb import (
            mteb_classification_dataset, mteb_classification_model,
            make_classification_score_fn,
        )

        data = mteb_classification_dataset("Banking77Classification.v2")
        model = mteb_classification_model("sentence-transformers/all-MiniLM-L6-v2")
        score_fn = make_classification_score_fn(data["train"], model)

        # Use in bench.compare() or @bench.task
    """
    _require("sentence_transformers", "mteb")

    # Pre-compute training embeddings in batches
    train_texts = [d["text"] for d in train_data]
    train_labels = [d["label"] for d in train_data]

    # Use the underlying SentenceTransformer if available for batched encoding
    _inner_model = getattr(model_callable, "__self__", None)
    if _inner_model is None:
        # Fall back to calling the callable one-by-one in batches
        all_embeddings = []
        for i in range(0, len(train_texts), batch_size):
            batch = train_texts[i : i + batch_size]
            for text in batch:
                all_embeddings.append(model_callable({"text": text}))
        train_embeddings = np.stack(all_embeddings)
    else:
        train_embeddings = _inner_model.encode(
            train_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    clf = _KNNClassifier(k=k)
    clf.fit(train_embeddings, train_labels)

    def score_fn(problem: dict, response: np.ndarray) -> bool:
        predicted = clf.predict(response)
        return predicted == problem["label"]

    return score_fn


# ---------------------------------------------------------------------------
# Convenience: load any MTEB task's raw dataset
# ---------------------------------------------------------------------------


def mteb_task_info(task_name: str) -> dict:
    """Return metadata about an MTEB task without loading its data.

    Useful for inspecting task type, languages, and domains before deciding
    which score function to use.

    Args:
        task_name: MTEB task identifier.

    Returns:
        Dict with keys ``name``, ``type``, ``languages``, ``domains``.
    """
    _require("mteb", "mteb")
    import mteb

    task = mteb.get_task(task_name)
    meta = task.metadata
    return {
        "name": task_name,
        "type": getattr(meta, "type", getattr(meta, "task_type", "unknown")),
        "languages": getattr(meta, "eval_langs", []),
        "domains": getattr(meta, "domains", []),
        "description": getattr(meta, "description", ""),
    }
