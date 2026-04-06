"""Anthropic API adapter for bayesbench.

Wraps the Anthropic Python SDK so Claude models plug directly into the
bayesbench decorator API.

Install dependencies::

    pip install bayesbench[anthropic]

Usage::

    import os
    from bayesbench import BayesianBenchmark
    from bayesbench.adapters.anthropic_adapter import anthropic_model

    bench = BayesianBenchmark(confidence=0.95)

    model_a = anthropic_model(
        "claude-opus-4-6",
        system_prompt="Answer the question in one sentence.",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    model_b = anthropic_model("claude-haiku-4-5-20251001")

    @bench.task(dataset=problems)
    def task(problem):
        return (
            model_a(problem) == problem["answer"],
            model_b(problem) == problem["answer"],
        )

    print(bench.run().summary())
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from .base import _require


def anthropic_model(
    model: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **message_kwargs: Any,
) -> Callable[[Any], str]:
    """Create a synchronous model callable using the Anthropic client.

    Args:
        model: Anthropic model ID, e.g. ``"claude-opus-4-6"`` or
               ``"claude-haiku-4-5-20251001"``.
        prompt_fn: Converts a problem dict to a prompt string.
                   Defaults to ``str(problem)``.
        system_prompt: System prompt passed to the Messages API.
        api_key: Anthropic API key. Falls back to ``ANTHROPIC_API_KEY``
                 env var.
        max_tokens: Maximum tokens in the completion (required by Anthropic).
        temperature: Sampling temperature (0 = deterministic).
        **message_kwargs: Additional kwargs forwarded to
                          ``client.messages.create``.

    Returns:
        A synchronous ``callable(problem) -> str``.
    """
    _require("anthropic", "anthropic")

    import anthropic as _anthropic

    client = _anthropic.Anthropic(api_key=api_key)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    @functools.wraps(anthropic_model)
    def call(problem: Any) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": _prompt_fn(problem)}],
            **message_kwargs,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()

    call.__bayesbench_model__ = model  # type: ignore[attr-defined]
    return call


def anthropic_model_async(
    model: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **message_kwargs: Any,
) -> Callable[[Any], Any]:
    """Async version of :func:`anthropic_model`.

    Returns an ``async callable(problem) -> str`` for use with
    :meth:`~bayesbench.BayesianBenchmark.compare_async`.
    """
    _require("anthropic", "anthropic")

    import anthropic as _anthropic

    client = _anthropic.AsyncAnthropic(api_key=api_key)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    async def call(problem: Any) -> str:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": _prompt_fn(problem)}],
            **message_kwargs,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = await client.messages.create(**kwargs)
        return response.content[0].text.strip()

    call.__bayesbench_model__ = model  # type: ignore[attr-defined]
    return call
