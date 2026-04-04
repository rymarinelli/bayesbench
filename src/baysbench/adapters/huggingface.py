"""HuggingFace adapters for baysbench.

Provides thin wrappers around the HuggingFace Inference API and the
``datasets`` library so they plug directly into the baysbench decorator API.

Install dependencies::

    pip install baysbench[huggingface]

Usage::

    import os
    from baysbench import BayesianBenchmark
    from baysbench.adapters.huggingface import hf_model, hf_dataset

    bench = BayesianBenchmark(confidence=0.95)

    model_a = hf_model(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        prompt_fn=lambda p: p["question"],
        api_key=os.getenv("HF_TOKEN"),
        max_new_tokens=64,
    )
    model_b = hf_model(
        "mistralai/Mistral-7B-Instruct-v0.3",
        prompt_fn=lambda p: p["question"],
        api_key=os.getenv("HF_TOKEN"),
    )
    dataset = hf_dataset("gsm8k", "main", split="test[:500]")

    @bench.task(dataset=dataset)
    def gsm8k(problem):
        a = model_a(problem) == problem["answer"]
        b = model_b(problem) == problem["answer"]
        return a, b

    print(bench.run().summary())
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from .base import _require


def hf_model(
    model_id: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    **generation_kwargs: Any,
) -> Callable[[Any], str]:
    """Create a callable that queries a HuggingFace Inference API model.

    Uses the ``huggingface_hub`` ``InferenceClient`` under the hood.

    Args:
        model_id: HuggingFace model repo ID, e.g.
                  ``"meta-llama/Meta-Llama-3-8B-Instruct"``.
        prompt_fn: Converts a problem dict to a prompt string.
                   Defaults to ``str(problem)``.
        system_prompt: Optional system message prepended to every call.
        api_key: HuggingFace API token (or set ``HF_TOKEN`` env var).
        base_url: Override the endpoint URL (for private / dedicated
                  endpoints).
        max_new_tokens: Maximum tokens in the completion.
        temperature: Sampling temperature (0 = greedy).
        **generation_kwargs: Extra kwargs forwarded to the client.

    Returns:
        A synchronous ``callable(problem) -> str``.
    """
    _require("huggingface_hub", "huggingface")

    from huggingface_hub import InferenceClient

    client = InferenceClient(model=model_id, token=api_key, base_url=base_url)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    @functools.wraps(hf_model)
    def call(problem: Any) -> str:
        prompt = _prompt_fn(problem)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **generation_kwargs,
        )
        return response.choices[0].message.content.strip()

    call.__baysbench_model__ = model_id  # type: ignore[attr-defined]
    return call


def hf_model_async(
    model_id: str,
    *,
    prompt_fn: Callable[[Any], str] | None = None,
    system_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    **generation_kwargs: Any,
) -> Callable[[Any], Any]:
    """Async version of :func:`hf_model`.

    Returns an ``async callable(problem) -> str`` for use with
    :meth:`~baysbench.BayesianBenchmark.compare_async`.
    """
    _require("huggingface_hub", "huggingface")

    from huggingface_hub import AsyncInferenceClient

    client = AsyncInferenceClient(model=model_id, token=api_key, base_url=base_url)
    _prompt_fn = prompt_fn or (lambda p: str(p))

    async def call(problem: Any) -> str:
        prompt = _prompt_fn(problem)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            **generation_kwargs,
        )
        return response.choices[0].message.content.strip()

    call.__baysbench_model__ = model_id  # type: ignore[attr-defined]
    return call


def hf_dataset(
    path: str,
    name: str | None = None,
    *,
    split: str = "test",
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Load a HuggingFace dataset as a plain list of dicts.

    Args:
        path: Dataset path on the Hub, e.g. ``"gsm8k"`` or
              ``"allenai/ai2_arc"``.
        name: Dataset config name (e.g. ``"main"`` for gsm8k).
        split: Dataset split (e.g. ``"test"``, ``"test[:500]"``).
        max_samples: Truncate to this many examples after loading.
        shuffle: Shuffle the dataset before truncation.
        seed: Random seed for shuffling.

    Returns:
        A list of dicts (one per example).

    Example::

        from baysbench.adapters.huggingface import hf_dataset

        problems = hf_dataset("gsm8k", "main", split="test", max_samples=200)
        # Each element is a dict like {"question": ..., "answer": ...}
    """
    _require("datasets", "huggingface")

    from datasets import load_dataset

    ds = load_dataset(path, name, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return list(ds)
