"""Framework adapters for baysbench.

Each adapter converts a framework's model API into a simple
``callable(problem) -> str`` that plugs into the baysbench decorator API
without any changes to the benchmarking logic.

Available adapters
------------------
:mod:`~baysbench.adapters.huggingface`
    ``hf_model``, ``hf_model_async``, ``hf_dataset``
    — HuggingFace Inference API + Hub datasets.
    Requires: ``pip install baysbench[huggingface]``

:mod:`~baysbench.adapters.openai_compat`
    ``openai_model``, ``openai_model_async``
    — OpenAI-compatible chat completions (OpenAI, Groq, Together AI,
      Fireworks, Ollama, vLLM, LiteLLM, Azure OpenAI, …).
    Requires: ``pip install baysbench[openai]``

:mod:`~baysbench.adapters.anthropic_adapter`
    ``anthropic_model``, ``anthropic_model_async``
    — Anthropic Messages API (Claude models).
    Requires: ``pip install baysbench[anthropic]``

Quick example::

    from baysbench.adapters.openai_compat import openai_model
    from baysbench.adapters.anthropic_adapter import anthropic_model
    from baysbench.adapters.huggingface import hf_model, hf_dataset
"""

from .base import ModelAdapter

__all__ = ["ModelAdapter"]
