"""Integration with AISI's Inspect evaluation framework.

Inspect (inspect_ai) is the UK AI Safety Institute's framework for LLM
evaluation. This adapter bridges Inspect's rich dataset ecosystem and model
infrastructure with bayesbench's Bayesian sequential testing.

Install dependencies::

    pip install bayesbench[inspect]

Usage — convert an Inspect dataset and compare two models::

    from inspect_ai.dataset import hf_dataset
    from bayesbench import BayesianBenchmark
    from bayesbench.adapters.inspect_ai import (
        from_inspect_dataset,
        inspect_model,
        exact_match_score,
        includes_score,
    )

    bench = BayesianBenchmark(confidence=0.95)

    problems = from_inspect_dataset(
        hf_dataset("openai/gsm8k", split="test", sample_fields=FieldSpec(
            input="question", target="answer"
        ))
    )

    model_a = inspect_model("openai/gpt-4o")
    model_b = inspect_model("openai/gpt-4o-mini")

    @bench.task(dataset=problems, name="gsm8k")
    def gsm8k(problem):
        return (
            exact_match_score(problem, model_a(problem)),
            exact_match_score(problem, model_b(problem)),
        )

    print(bench.run().summary())

Usage — rank N models with BayesianRanker::

    from bayesbench import BayesianRanker
    from bayesbench.adapters.inspect_ai import inspect_model, includes_score

    ranker = BayesianRanker(confidence=0.95)
    for name in ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"]:
        ranker.add_model(name, inspect_model(name))

    result = ranker.rank(dataset=problems, score_fn=includes_score)
    print(result.summary())

Usage — async (preserves Inspect's native async model interface)::

    from bayesbench.adapters.inspect_ai import inspect_model_async

    async_a = inspect_model_async("openai/gpt-4o")
    result = await bench.compare_async(async_a, async_b, includes_score, problems)
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from .base import _require

# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


def from_inspect_dataset(dataset: Any) -> list[dict]:
    """Convert an Inspect ``Dataset`` to a list of bayesbench problem dicts.

    Each problem dict contains:
    - ``"input"`` — the prompt string (last user message if multi-turn)
    - ``"target"`` — expected answer (first element if list)
    - ``"choices"`` — multiple-choice options, or ``None``
    - ``"id"`` — sample ID
    - ``"metadata"`` — sample metadata dict

    Args:
        dataset: An ``inspect_ai.dataset.Dataset`` instance, or any iterable
                 of objects with ``input``, ``target``, ``id``, and ``metadata``
                 attributes.

    Returns:
        List of problem dicts compatible with all bayesbench APIs.

    Example::

        from inspect_ai.dataset import hf_dataset, FieldSpec
        from bayesbench.adapters.inspect_ai import from_inspect_dataset

        problems = from_inspect_dataset(
            hf_dataset(
                "openai/gsm8k",
                split="test",
                sample_fields=FieldSpec(input="question", target="answer"),
            )
        )
    """
    problems = []
    for sample in dataset:
        # Resolve input: str or list[ChatMessage]
        raw_input = sample.input
        if isinstance(raw_input, str):
            prompt = raw_input
        elif isinstance(raw_input, list):
            # Extract last user-role message content
            prompt = ""
            for msg in reversed(raw_input):
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", "")
                if role in ("user", None) or role != "system":
                    prompt = content if isinstance(content, str) else str(content)
                    break
            if not prompt and raw_input:
                last = raw_input[-1]
                prompt = getattr(last, "content", str(last))
        else:
            prompt = str(raw_input)

        # Resolve target: str or list[str]
        raw_target = sample.target
        if isinstance(raw_target, list):
            target = raw_target[0] if raw_target else ""
            all_targets = raw_target
        else:
            target = raw_target or ""
            all_targets = [target] if target else []

        problems.append(
            {
                "input": prompt,
                "target": target,
                "all_targets": all_targets,
                "choices": getattr(sample, "choices", None),
                "id": sample.id,
                "metadata": sample.metadata or {},
            }
        )
    return problems


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------


def inspect_model(
    model_id: str,
    *,
    system_prompt: str | None = None,
    config: Any | None = None,
) -> Callable[[dict], str]:
    """Create a **synchronous** model callable backed by Inspect's model layer.

    Wraps ``inspect_ai.model.get_model`` so bayesbench can call it like any
    other ``callable(problem) -> str``.  Uses ``asyncio.run()`` internally;
    do not use inside an already-running event loop (use
    :func:`inspect_model_async` instead).

    Args:
        model_id: Inspect model string, e.g. ``"openai/gpt-4o"`` or
                  ``"anthropic/claude-haiku-4-5"``.
        system_prompt: Optional system message prepended to every call.
        config: ``inspect_ai.model.GenerateConfig`` instance for temperature,
                max_tokens, etc.

    Returns:
        A synchronous ``callable(problem: dict) -> str``.

    Example::

        model = inspect_model("openai/gpt-4o", system_prompt="Be concise.")
        response = model({"input": "What is 2+2?", "target": "4"})
    """
    _require("inspect_ai", "inspect")
    from inspect_ai.model import get_model

    _model = get_model(model_id)

    def call(problem: dict) -> str:
        messages: list[Any] = []
        if system_prompt:
            try:
                from inspect_ai.model import ChatMessageSystem

                messages.append(ChatMessageSystem(content=system_prompt))
            except ImportError:
                pass
        try:
            from inspect_ai.model import ChatMessageUser

            messages.append(ChatMessageUser(content=problem["input"]))
        except ImportError:
            messages.append({"role": "user", "content": problem["input"]})

        output = asyncio.run(_model.generate(messages, config=config))
        return output.completion.strip()

    call.__bayesbench_model__ = model_id  # type: ignore[attr-defined]
    return call


def inspect_model_async(
    model_id: str,
    *,
    system_prompt: str | None = None,
    config: Any | None = None,
) -> Callable[[dict], Any]:
    """Create an **async** model callable backed by Inspect's model layer.

    Use this with :meth:`~bayesbench.BayesianBenchmark.compare_async` or
    :meth:`~bayesbench.BayesianRanker.rank_async` to avoid creating a new
    event loop per call.

    Args:
        model_id: Inspect model string, e.g. ``"openai/gpt-4o"``.
        system_prompt: Optional system message.
        config: ``GenerateConfig`` for temperature, max_tokens, etc.

    Returns:
        An ``async callable(problem: dict) -> str``.
    """
    _require("inspect_ai", "inspect")
    from inspect_ai.model import get_model

    _model = get_model(model_id)

    async def call(problem: dict) -> str:
        messages: list[Any] = []
        if system_prompt:
            try:
                from inspect_ai.model import ChatMessageSystem

                messages.append(ChatMessageSystem(content=system_prompt))
            except ImportError:
                pass
        try:
            from inspect_ai.model import ChatMessageUser

            messages.append(ChatMessageUser(content=problem["input"]))
        except ImportError:
            messages.append({"role": "user", "content": problem["input"]})

        output = await _model.generate(messages, config=config)
        return output.completion.strip()

    call.__bayesbench_model__ = model_id  # type: ignore[attr-defined]
    return call


# ---------------------------------------------------------------------------
# Score functions  (mirror Inspect's built-in scorers as plain callables)
# ---------------------------------------------------------------------------


def exact_match_score(problem: dict, response: str) -> bool:
    """Case-insensitive exact match against ``problem["target"]``.

    Mirrors Inspect's ``exact()`` scorer.
    """
    target = str(problem.get("target", "")).strip().lower()
    return response.strip().lower() == target


def includes_score(problem: dict, response: str) -> bool:
    """Return True if the response *contains* ``problem["target"]``.

    Mirrors Inspect's ``includes()`` scorer.
    """
    target = str(problem.get("target", "")).strip().lower()
    return target in response.strip().lower()


def any_target_score(problem: dict, response: str) -> bool:
    """Return True if the response matches *any* of ``problem["all_targets"]``.

    Useful when a task has multiple valid answers.
    """
    all_targets = problem.get("all_targets") or [problem.get("target", "")]
    resp = response.strip().lower()
    return any(str(t).strip().lower() in resp for t in all_targets)


def pattern_score(pat: str, *, case_sensitive: bool = False) -> Callable[[dict, str], bool]:
    """Build a score_fn that matches a regex ``pat`` against the response.

    Mirrors Inspect's ``pattern()`` scorer.

    Args:
        pat: Regular expression pattern to search for in the response.
        case_sensitive: Whether the match is case-sensitive.

    Returns:
        A ``score_fn(problem, response) -> bool``.

    Example::

        score = pattern_score(r"\\b(yes|no)\\b")
        result = bench.compare(model_a, model_b, score, problems)
    """
    flags = 0 if case_sensitive else re.IGNORECASE

    def _score(problem: dict, response: str) -> bool:  # noqa: ARG001
        return bool(re.search(pat, response, flags))

    return _score


def choice_score(problem: dict, response: str) -> bool:
    """Match a multiple-choice letter (A/B/C/D) in the response.

    Expects ``problem["target"]`` to be the correct letter and
    ``problem["choices"]`` to be the list of options.

    Mirrors Inspect's ``choice()`` scorer.
    """
    target = str(problem.get("target", "")).strip().upper()
    # Accept bare letter or letter followed by punctuation
    found = re.search(r"\b([A-D])\b", response.strip().upper())
    if found:
        return found.group(1) == target
    return response.strip().upper().startswith(target)
