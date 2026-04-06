"""Framework adapters for bayesbench.

Each adapter converts a framework's model API into a simple
``callable(problem) -> str`` that plugs into the bayesbench decorator API
without any changes to the benchmarking logic.

Available adapters
------------------
:mod:`~bayesbench.adapters.huggingface`
    ``hf_model``, ``hf_model_async``, ``hf_dataset``
    — HuggingFace Inference API + Hub datasets.
    Requires: ``pip install bayesbench[huggingface]``

:mod:`~bayesbench.adapters.openai_compat`
    ``openai_model``, ``openai_model_async``
    — OpenAI-compatible chat completions (OpenAI, Groq, Together AI,
      Fireworks, Ollama, vLLM, LiteLLM, Azure OpenAI, …).
    Requires: ``pip install bayesbench[openai]``

:mod:`~bayesbench.adapters.anthropic_adapter`
    ``anthropic_model``, ``anthropic_model_async``
    — Anthropic Messages API (Claude models).
    Requires: ``pip install bayesbench[anthropic]``

:mod:`~bayesbench.adapters.inspect_ai`
    ``from_inspect_dataset``, ``inspect_model``, ``inspect_model_async``,
    ``exact_match_score``, ``includes_score``, ``pattern_score``,
    ``choice_score``, ``any_target_score``
    — AISI Inspect framework: dataset conversion + model/scorer wrappers.
    Requires: ``pip install bayesbench[inspect]``

:mod:`~bayesbench.adapters.mteb`
    ``mteb_sts_dataset``, ``mteb_classification_dataset``, ``st_model``,
    ``mteb_classification_model``, ``sts_score_fn``,
    ``make_classification_score_fn``, ``mteb_task_info``
    — MTEB benchmark: STS + classification datasets and embedding model adapters.
    Requires: ``pip install bayesbench[mteb]``

:mod:`~bayesbench.adapters.openclaw`
    ``openclaw_agent``
    — OpenClaw agent wrapper for bayesbench callables.
    Requires: ``pip install bayesbench[openclaw]``

Quick example::

    from bayesbench.adapters.openai_compat import openai_model
    from bayesbench.adapters.anthropic_adapter import anthropic_model
    from bayesbench.adapters.huggingface import hf_model, hf_dataset
"""

from .base import ModelAdapter

__all__ = ["ModelAdapter"]
