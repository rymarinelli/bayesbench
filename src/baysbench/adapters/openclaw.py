"""OpenClaw adapter for baysbench.

This wrapper lets you benchmark OpenClaw agents with baysbench's sequential
Bayesian evaluation loop.

Install dependencies::

    pip install baysbench[openclaw]

Usage::

    from baysbench.adapters.openclaw import openclaw_agent

    agent_a = openclaw_agent(my_openclaw_agent)
    response = agent_a({"input": "Solve 2+2"})
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    for attr in ("response", "output", "text", "content", "answer"):
        value = getattr(response, attr, None)
        if isinstance(value, str):
            return value.strip()

    if isinstance(response, dict):
        for key in ("response", "output", "text", "content", "answer"):
            value = response.get(key)
            if isinstance(value, str):
                return value.strip()

    return str(response).strip()


def openclaw_agent(
    agent: Any,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
) -> Callable[[Any], str]:
    """Wrap an OpenClaw agent as a ``callable(problem) -> str``.

    The wrapper is intentionally duck-typed so it works across OpenClaw
    versions. It tries ``agent.run(prompt)`` first, then ``agent(prompt)``.
    """
    _prompt_fn = prompt_fn or (lambda p: p.get("input", str(p)) if isinstance(p, dict) else str(p))

    def call(problem: Any) -> str:
        prompt = _prompt_fn(problem)
        if hasattr(agent, "run") and callable(agent.run):
            response = agent.run(prompt)
        elif callable(agent):
            response = agent(prompt)
        else:
            raise TypeError(
                "OpenClaw adapter expected an agent with run(prompt) " "or __call__(prompt)."
            )
        return _extract_text(response)

    call.__baysbench_model__ = getattr(agent, "name", "openclaw-agent")  # type: ignore[attr-defined]
    return call
