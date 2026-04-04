"""Tests for baysbench.adapters.inspect_ai (mock-based — no real API calls)."""

from __future__ import annotations

import pytest

from baysbench.adapters.inspect_ai import (
    any_target_score,
    choice_score,
    exact_match_score,
    from_inspect_dataset,
    includes_score,
    pattern_score,
)

# ---------------------------------------------------------------------------
# Minimal Inspect Sample stub (no inspect_ai installation needed)
# ---------------------------------------------------------------------------


class _FakeSample:
    def __init__(self, input, target, choices=None, id=None, metadata=None):
        self.input = input
        self.target = target
        self.choices = choices
        self.id = id
        self.metadata = metadata or {}


class _FakeChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


# ---------------------------------------------------------------------------
# from_inspect_dataset
# ---------------------------------------------------------------------------


class TestFromInspectDataset:
    def test_string_input(self):
        samples = [_FakeSample(input="What is 2+2?", target="4")]
        problems = from_inspect_dataset(samples)
        assert problems[0]["input"] == "What is 2+2?"
        assert problems[0]["target"] == "4"

    def test_list_input_extracts_last_user_message(self):
        msgs = [
            _FakeChatMessage("system", "You are helpful."),
            _FakeChatMessage("user", "What is 3+3?"),
        ]
        samples = [_FakeSample(input=msgs, target="6")]
        problems = from_inspect_dataset(samples)
        assert problems[0]["input"] == "What is 3+3?"

    def test_list_target_uses_first(self):
        samples = [_FakeSample(input="Q?", target=["answer1", "answer2"])]
        problems = from_inspect_dataset(samples)
        assert problems[0]["target"] == "answer1"
        assert "answer2" in problems[0]["all_targets"]

    def test_all_targets_populated(self):
        samples = [_FakeSample(input="Q?", target=["a", "b", "c"])]
        problems = from_inspect_dataset(samples)
        assert problems[0]["all_targets"] == ["a", "b", "c"]

    def test_single_target_in_all_targets(self):
        samples = [_FakeSample(input="Q?", target="ans")]
        problems = from_inspect_dataset(samples)
        assert problems[0]["all_targets"] == ["ans"]

    def test_metadata_preserved(self):
        samples = [_FakeSample(input="Q?", target="A", metadata={"domain": "math"})]
        problems = from_inspect_dataset(samples)
        assert problems[0]["metadata"]["domain"] == "math"

    def test_choices_preserved(self):
        samples = [_FakeSample(input="Q?", target="B", choices=["A", "B", "C", "D"])]
        problems = from_inspect_dataset(samples)
        assert problems[0]["choices"] == ["A", "B", "C", "D"]

    def test_id_preserved(self):
        samples = [_FakeSample(input="Q?", target="A", id="sample_001")]
        problems = from_inspect_dataset(samples)
        assert problems[0]["id"] == "sample_001"

    def test_multiple_samples(self):
        samples = [_FakeSample(input=f"Q{i}?", target=str(i)) for i in range(5)]
        problems = from_inspect_dataset(samples)
        assert len(problems) == 5

    def test_none_metadata_becomes_empty_dict(self):
        samples = [_FakeSample(input="Q?", target="A", metadata=None)]
        problems = from_inspect_dataset(samples)
        assert problems[0]["metadata"] == {}

    def test_empty_dataset(self):
        assert from_inspect_dataset([]) == []

    def test_non_string_input_converted(self):
        samples = [_FakeSample(input=42, target="A")]
        problems = from_inspect_dataset(samples)
        assert isinstance(problems[0]["input"], str)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


class TestExactMatchScore:
    def test_exact_match(self):
        p = {"target": "Paris"}
        assert exact_match_score(p, "Paris") is True

    def test_case_insensitive(self):
        p = {"target": "Paris"}
        assert exact_match_score(p, "paris") is True
        assert exact_match_score(p, "PARIS") is True

    def test_whitespace_stripped(self):
        p = {"target": "Paris"}
        assert exact_match_score(p, "  Paris  ") is True

    def test_wrong_answer(self):
        p = {"target": "Paris"}
        assert exact_match_score(p, "London") is False

    def test_partial_match_fails(self):
        p = {"target": "Paris"}
        assert exact_match_score(p, "Paris is nice") is False

    def test_empty_target(self):
        p = {"target": ""}
        assert exact_match_score(p, "") is True
        assert exact_match_score(p, "something") is False


class TestIncludesScore:
    def test_exact_inclusion(self):
        p = {"target": "42"}
        assert includes_score(p, "The answer is 42.") is True

    def test_case_insensitive(self):
        p = {"target": "Paris"}
        assert includes_score(p, "the capital is PARIS!") is True

    def test_not_included(self):
        p = {"target": "42"}
        assert includes_score(p, "The answer is 43") is False

    def test_empty_target_always_matches(self):
        p = {"target": ""}
        assert includes_score(p, "anything") is True


class TestAnyTargetScore:
    def test_matches_first(self):
        p = {"target": "a", "all_targets": ["a", "b", "c"]}
        assert any_target_score(p, "I choose a.") is True

    def test_matches_second(self):
        p = {"target": "a", "all_targets": ["a", "b"]}
        assert any_target_score(p, "My answer is b.") is True

    def test_no_match(self):
        p = {"target": "cat", "all_targets": ["cat", "dog"]}
        assert any_target_score(p, "none of the above") is False

    def test_falls_back_to_target_if_no_all_targets(self):
        p = {"target": "yes"}
        assert any_target_score(p, "yes I agree") is True


class TestPatternScore:
    def test_simple_match(self):
        score = pattern_score(r"\d+")
        assert score({}, "The answer is 42.") is True
        assert score({}, "no digits here") is False

    def test_case_insensitive_by_default(self):
        score = pattern_score(r"yes|no")
        assert score({}, "YES") is True
        assert score({}, "No") is True

    def test_case_sensitive(self):
        score = pattern_score(r"YES", case_sensitive=True)
        assert score({}, "YES") is True
        assert score({}, "yes") is False

    def test_anchored_pattern(self):
        score = pattern_score(r"^[A-D]$")
        assert score({}, "B") is True
        assert score({}, "AB") is False


class TestChoiceScore:
    def test_correct_letter(self):
        p = {"target": "B", "choices": ["Paris", "London", "Rome", "Berlin"]}
        assert choice_score(p, "B") is True

    def test_wrong_letter(self):
        p = {"target": "B", "choices": ["Paris", "London", "Rome", "Berlin"]}
        assert choice_score(p, "C") is False

    def test_letter_in_response(self):
        p = {"target": "A", "choices": []}
        assert choice_score(p, "The answer is A.") is True

    def test_case_insensitive(self):
        p = {"target": "C", "choices": []}
        assert choice_score(p, "c is correct") is True


# ---------------------------------------------------------------------------
# inspect_model requires inspect_ai — test that missing library gives clear error
# ---------------------------------------------------------------------------


class TestInspectModelMissingLibrary:
    def test_raises_import_error_without_inspect_ai(self, monkeypatch):
        import sys

        orig = sys.modules.get("inspect_ai")
        sys.modules["inspect_ai"] = None  # type: ignore[assignment]
        try:
            from baysbench.adapters.inspect_ai import inspect_model

            with pytest.raises((ImportError, AttributeError)):
                model = inspect_model("openai/gpt-4o")
                model({"input": "hi", "target": "hello"})
        finally:
            if orig is not None:
                sys.modules["inspect_ai"] = orig
            else:
                del sys.modules["inspect_ai"]
