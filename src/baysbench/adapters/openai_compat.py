"""OpenAI-compatible API adapter for baysbench.

Works with any provider that exposes an OpenAI-compatible chat completions
endpoint: OpenAI, Azure OpenAI, Groq, Together AI, Fireworks AI,
Ollama (local), vLLM, LiteLLM, and more.

Install dependencies::

    pip install baysbench[openai]

Usage (OpenAI)::

    import os
    from baysbench import BayesianBenchmark
    from baysbench.adapters.openai_compat import openai_model

    bench = BayesianBenchmark(confidence=0.95)

    model_a = openai_model("gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    model_b = openai_model("gpt-4o-mini")

    @bench.task(dataset=problems)
    def task(problem):
        return (
            model_a(problem) == problem["answer"],
            model_b(problem) == problem["answer"],
        )

Usage (Groq)::

    model = openai_model(
        "llama-3.1-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
    )

Usage (local Ollama)::

    model = openai_model(
        "llama3",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from .base import _require


def openai_model(
    model: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **completion_kwargs: Any,
) -> Callable[[Any], str]:
    """Create a synchronous model callable using the OpenAI client.

    Args:
        model: Model name (e.g. ``"gpt-4o"``, ``"llama-3.1-70b-versatile"``).
        prompt_fn: Converts a problem dict to a prompt string.
                   Defaults to ``str(problem)``.
        system_prompt: System message prepended to every request.
        api_key: API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Override the API base URL for non-OpenAI providers.
        organization: OpenAI organisation ID (optional).
        max_tokens: Maximum completion tokens.
        temperature: Sampling temperature (0 = deterministic).
        **completion_kwargs: Additional kwargs forwarded to
                              ``client.chat.completions.create``.

    Returns:
        A synchronous ``callable(problem) -> str``.
    """
    _require("openai", "openai")

    from openai import OpenAI  # type: ignore[import]

    client = OpenAI(api_key=api_key, base_url=base_url, organization=organization)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    @functools.wraps(openai_model)
    def call(problem: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _prompt_fn(problem)})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **completion_kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    call.__baysbench_model__ = model  # type: ignore[attr-defined]
    return call


def openai_model_async(
    model: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    **completion_kwargs: Any,
) -> Callable[[Any], Any]:
    """Async version of :func:`openai_model`.

    Returns an ``async callable(problem) -> str``.
    """
    _require("openai", "openai")

    from openai import AsyncOpenAI  # type: ignore[import]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, organization=organization)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    async def call(problem: Any) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _prompt_fn(problem)})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **completion_kwargs,
        )
        return (response.choices[0].message.content or "").strip()

    call.__baysbench_model__ = model  # type: ignore[attr-defined]
    return call
