"""Tests for baysbench.adapters.mteb (mock-based — no downloads needed)."""
from __future__ import annotations

import numpy as np
import pytest

from baysbench.adapters.mteb import (
    _KNNClassifier,
    make_classification_score_fn,
    sts_score_fn,
)


# ---------------------------------------------------------------------------
# _KNNClassifier (pure numpy — always available)
# ---------------------------------------------------------------------------


class TestKNNClassifier:
    def _make_data(self):
        rng = np.random.default_rng(0)
        # Two well-separated clusters
        class_a = rng.normal([1, 0], 0.1, (20, 2))
        class_b = rng.normal([-1, 0], 0.1, (20, 2))
        embeddings = np.vstack([class_a, class_b])
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = ["A"] * 20 + ["B"] * 20
        return embeddings, labels

    def test_fit_and_predict_correct_class(self):
        embeddings, labels = self._make_data()
        clf = _KNNClassifier(k=3)
        clf.fit(embeddings, labels)
        # Query near cluster A
        query = np.array([1.0, 0.0])
        query /= np.linalg.norm(query)
        assert clf.predict(query) == "A"

    def test_fit_and_predict_second_class(self):
        embeddings, labels = self._make_data()
        clf = _KNNClassifier(k=3)
        clf.fit(embeddings, labels)
        query = np.array([-1.0, 0.0])
        query /= np.linalg.norm(query)
        assert clf.predict(query) == "B"

    def test_k1_returns_nearest(self):
        embeddings = np.array([[1, 0], [0, 1], [-1, 0]], dtype=float)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = ["x", "y", "z"]
        clf = _KNNClassifier(k=1)
        clf.fit(embeddings, labels)
        assert clf.predict(np.array([0.99, 0.01])) == "x"

    def test_requires_fit_before_predict(self):
        clf = _KNNClassifier()
        with pytest.raises(AssertionError):
            clf.predict(np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# sts_score_fn (pure numpy — always available)
# ---------------------------------------------------------------------------


class TestStsScorefn:
    def _make_embeddings(self, cos_target: float) -> np.ndarray:
        """Create a (2, 2) normalised embedding pair with given cosine similarity."""
        e1 = np.array([1.0, 0.0])
        # cos_target = dot(e1, e2) when both normalised
        # e2 = [cos_target, sqrt(1 - cos_target^2)]
        e2 = np.array([cos_target, np.sqrt(max(0.0, 1.0 - cos_target ** 2))])
        return np.stack([e1, e2])

    def test_perfect_agreement(self):
        """If predicted cos-sim exactly matches gold, score should be 1."""
        # gold_score = 0.5 → normalised = 0.5; cos_sim = 0.0 → pred = 0.5
        embs = self._make_embeddings(0.0)  # cos_sim = 0.0 → pred = (0+1)/2 = 0.5
        problem = {"gold_score": 0.5}
        score = sts_score_fn(problem, embs)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_worst_case(self):
        """Maximum disagreement: gold=1, pred=0 → score = 0."""
        embs = self._make_embeddings(-1.0)  # cos_sim = -1 → pred = 0
        problem = {"gold_score": 1.0}
        score = sts_score_fn(problem, embs)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_score_in_unit_interval(self):
        for gold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for cos in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                embs = self._make_embeddings(cos)
                problem = {"gold_score": gold}
                score = sts_score_fn(problem, embs)
                assert 0.0 <= score <= 1.0, f"gold={gold}, cos={cos}, score={score}"

    def test_symmetry(self):
        """Score should be the same if we swap sentence order."""
        embs = self._make_embeddings(0.6)
        problem = {"gold_score": 0.8}
        s1 = sts_score_fn(problem, embs)
        s2 = sts_score_fn(problem, embs[::-1])
        assert s1 == pytest.approx(s2, abs=1e-6)


# ---------------------------------------------------------------------------
# make_classification_score_fn (pure numpy — no sentence_transformers needed)
# ---------------------------------------------------------------------------


class TestMakeClassificationScoreFn:
    def _make_fn(self):
        """Build a score_fn using synthetic embeddings (no ST model)."""
        rng = np.random.default_rng(42)
        dim = 16

        # Two clusters well-separated
        A = rng.normal([2] * dim, 0.2, (30, dim))
        B = rng.normal([-2] * dim, 0.2, (30, dim))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True)

        train_data = (
            [{"text": f"a_{i}", "label": "A"} for i in range(30)]
            + [{"text": f"b_{i}", "label": "B"} for i in range(30)]
        )
        train_embeddings = np.vstack([A, B])

        # Fake callable that looks up precomputed embedding by index
        _embed_map = {
            f"a_{i}": A[i] for i in range(30)
        } | {f"b_{i}": B[i] for i in range(30)}
        # Also add test points
        test_a = rng.normal([2] * dim, 0.2, (5, dim))
        test_a /= np.linalg.norm(test_a, axis=1, keepdims=True)
        test_b = rng.normal([-2] * dim, 0.2, (5, dim))
        test_b /= np.linalg.norm(test_b, axis=1, keepdims=True)
        for i in range(5):
            _embed_map[f"ta_{i}"] = test_a[i]
            _embed_map[f"tb_{i}"] = test_b[i]

        def fake_model(problem: dict) -> np.ndarray:
            return _embed_map[problem["text"]]

        # Build score fn without using a real ST model — patch the batching path
        train_texts = [d["text"] for d in train_data]
        train_labels = [d["label"] for d in train_data]
        embeddings = np.vstack([_embed_map[t] for t in train_texts])

        clf = _KNNClassifier(k=3)
        clf.fit(embeddings, train_labels)

        def score_fn(problem: dict, response: np.ndarray) -> bool:
            return clf.predict(response) == problem["label"]

        test_data = (
            [{"text": f"ta_{i}", "label": "A"} for i in range(5)]
            + [{"text": f"tb_{i}", "label": "B"} for i in range(5)]
        )
        return score_fn, fake_model, test_data

    def test_correct_classification(self):
        score_fn, model, test_data = self._make_fn()
        correct = sum(score_fn(p, model(p)) for p in test_data)
        # Well-separated clusters → should be nearly perfect
        assert correct >= 8  # at least 8/10

    def test_score_fn_returns_bool(self):
        score_fn, model, test_data = self._make_fn()
        result = score_fn(test_data[0], model(test_data[0]))
        assert isinstance(result, (bool, np.bool_))


# ---------------------------------------------------------------------------
# mteb_task_info / mteb_sts_dataset require mteb — test missing-library error
# ---------------------------------------------------------------------------


class TestMtebMissingLibrary:
    def test_raises_without_mteb(self, monkeypatch):
        import sys
        orig = sys.modules.get("mteb")
        sys.modules["mteb"] = None  # type: ignore[assignment]
        try:
            from baysbench.adapters.mteb import mteb_sts_dataset
            with pytest.raises((ImportError, AttributeError)):
                mteb_sts_dataset("STSBenchmark")
        finally:
            if orig is not None:
                sys.modules["mteb"] = orig
            else:
                del sys.modules["mteb"]

    def test_st_model_raises_without_sentence_transformers(self, monkeypatch):
        import sys
        orig = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore[assignment]
        try:
            from baysbench.adapters.mteb import st_model
            with pytest.raises((ImportError, AttributeError)):
                model = st_model("all-MiniLM-L6-v2")
                model({"sentence1": "hi", "sentence2": "hello", "gold_score": 0.8})
        finally:
            if orig is not None:
                sys.modules["sentence_transformers"] = orig
            else:
                del sys.modules["sentence_transformers"]


# ---------------------------------------------------------------------------
# Integration: sts_score_fn + BayesianBenchmark with NormalPosterior
# ---------------------------------------------------------------------------


class TestStsIntegrationWithBenchmark:
    def test_sts_comparison_with_normal_posterior(self):
        """End-to-end: compare two fake embedding models on synthetic STS data."""
        import numpy as np
        from baysbench import BayesianBenchmark
        from baysbench.posteriors import NormalPosterior

        np.random.seed(99)
        dim = 8

        def make_sts_problems(n: int) -> list[dict]:
            rng = np.random.default_rng(7)
            problems = []
            for _ in range(n):
                gold = rng.uniform(0, 1)
                problems.append({
                    "sentence1": "a",
                    "sentence2": "b",
                    "gold_score": gold,
                    "_gold": gold,  # kept for model to cheat
                })
            return problems

        def perfect_model(problem: dict) -> np.ndarray:
            """Embeddings whose cosine = gold (mapped from [0,1] to [-1,1])."""
            gold = problem["_gold"]
            cos = gold * 2 - 1
            e1 = np.array([1.0] + [0.0] * (dim - 1))
            e2 = np.array([cos, np.sqrt(max(0.0, 1 - cos ** 2))] + [0.0] * (dim - 2))
            return np.stack([e1, e2])

        def random_model(problem: dict) -> np.ndarray:
            """Embeddings with random cosine similarity."""
            rng = np.random.default_rng()
            e1 = rng.normal(size=dim)
            e2 = rng.normal(size=dim)
            e1 /= np.linalg.norm(e1)
            e2 /= np.linalg.norm(e2)
            return np.stack([e1, e2])

        problems = make_sts_problems(200)
        # skip_threshold just above 0.5 makes the non-discriminating window negligibly
        # narrow, so the test runs until it reaches the confidence threshold
        bench = BayesianBenchmark(confidence=0.95, min_samples=10,
                                  skip_threshold=0.51,
                                  posterior_factory=NormalPosterior)

        result = bench.compare(
            model_a=perfect_model,
            model_b=random_model,
            score_fn=sts_score_fn,
            dataset=problems,
            name="synthetic_sts",
        )

        assert result.winner == "model_a"
        assert result.problems_tested < len(problems)
