"""framework_adapters.py — how to use baysbench with HuggingFace, OpenAI,
Anthropic, and any OpenAI-compatible endpoint.

This file shows the *structure* — replace stub calls with real API calls.
Real dependencies are optional extras:
    pip install baysbench[huggingface]
    pip install baysbench[openai]
    pip install baysbench[anthropic]

Run with:
    python examples/framework_adapters.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Sample dataset (normally loaded from HuggingFace or a file)
# ---------------------------------------------------------------------------

PROBLEMS = [
    {"question": "What is 7 × 8?", "answer": "56"},
    {"question": "Capital of Norway?", "answer": "Oslo"},
    {"question": "What colour is chlorophyll?", "answer": "green"},
] * 40   # repeat to give the sequential test enough data


# ===========================================================================
# HuggingFace Inference API
# ===========================================================================

def demo_huggingface():
    """Compare two HuggingFace models via the Inference API."""
    print("=== HuggingFace adapter demo ===")
    print("(requires: pip install baysbench[huggingface])")
    print("(requires: HF_TOKEN env var)\n")

    try:
        from baysbench import BayesianBenchmark
        from baysbench.adapters.huggingface import hf_dataset, hf_model

        bench = BayesianBenchmark(confidence=0.95)

        model_a = hf_model(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("HF_TOKEN"),
            max_new_tokens=32,
            system_prompt="Answer in one word or number.",
        )
        model_b = hf_model(
            "mistralai/Mistral-7B-Instruct-v0.3",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("HF_TOKEN"),
            max_new_tokens=32,
        )

        # Load a real dataset from the Hub
        # dataset = hf_dataset("gsm8k", "main", split="test", max_samples=500)
        # Using local stub dataset instead:
        dataset = PROBLEMS

        @bench.task(dataset=dataset, name="qa")
        def qa_task(problem):
            a = model_a(problem).lower() == problem["answer"].lower()
            b = model_b(problem).lower() == problem["answer"].lower()
            return a, b

        print("  Would run bench.run() — skipping to avoid real API call")
        # report = bench.run()
        # print(report.summary())

    except ImportError as e:
        print(f"  Skipped: {e}\n")


# ===========================================================================
# OpenAI-compatible API (OpenAI, Groq, Together AI, Ollama, vLLM, …)
# ===========================================================================

def demo_openai():
    """Compare two OpenAI models."""
    print("=== OpenAI-compatible adapter demo ===")
    print("(requires: pip install baysbench[openai])")
    print("(requires: OPENAI_API_KEY env var)\n")

    try:
        from baysbench import benchmark
        from baysbench.adapters.openai_compat import openai_model

        model_a = openai_model(
            "gpt-4o",
            prompt_fn=lambda p: p["question"],
            system_prompt="Answer in one word or number only.",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=16,
            temperature=0.0,
        )
        model_b = openai_model(
            "gpt-4o-mini",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=16,
        )

        @benchmark(
            model_a=model_a,
            model_b=model_b,
            dataset=PROBLEMS,
            name="gpt4o_vs_mini",
            confidence=0.95,
        )
        def exact_match(problem, response):
            return response.strip().lower() == problem["answer"].lower()

        print("  Would run exact_match.run() — skipping to avoid real API call")
        # result = exact_match.run()
        # print(result)

    except ImportError as e:
        print(f"  Skipped: {e}\n")


def demo_groq():
    """Same decorator, different base_url → Groq."""
    print("=== Groq (OpenAI-compatible) adapter demo ===")
    print("(requires: pip install baysbench[openai])")
    print("(requires: GROQ_API_KEY env var)\n")

    try:
        from baysbench.adapters.openai_compat import openai_model

        model = openai_model(
            "llama-3.1-70b-versatile",
            prompt_fn=lambda p: p["question"],
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=32,
        )
        print(f"  Groq model ready: {model.__baysbench_model__}")  # type: ignore
    except ImportError as e:
        print(f"  Skipped: {e}\n")


def demo_ollama():
    """Same decorator, local Ollama endpoint."""
    print("=== Ollama (local) adapter demo ===")
    print("(requires: pip install baysbench[openai])")
    print("(requires: Ollama running on localhost:11434)\n")

    try:
        from baysbench.adapters.openai_compat import openai_model

        model = openai_model(
            "llama3",
            prompt_fn=lambda p: p["question"],
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Ollama ignores the key
            max_tokens=64,
        )
        print(f"  Ollama model ready: {model.__baysbench_model__}")  # type: ignore
    except ImportError as e:
        print(f"  Skipped: {e}\n")


# ===========================================================================
# Anthropic
# ===========================================================================

def demo_anthropic():
    """Compare two Claude models."""
    print("=== Anthropic adapter demo ===")
    print("(requires: pip install baysbench[anthropic])")
    print("(requires: ANTHROPIC_API_KEY env var)\n")

    try:
        from baysbench import BayesianRanker
        from baysbench.adapters.anthropic_adapter import anthropic_model

        opus = anthropic_model(
            "claude-opus-4-6",
            prompt_fn=lambda p: p["question"],
            system_prompt="Answer in one word or number.",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=16,
            temperature=0.0,
        )
        haiku = anthropic_model(
            "claude-haiku-4-5-20251001",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=16,
        )
        sonnet = anthropic_model(
            "claude-sonnet-4-6",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=16,
        )

        ranker = BayesianRanker(confidence=0.95, min_samples=5)
        ranker.add_model("claude-opus-4-6", opus)
        ranker.add_model("claude-haiku-4-5", haiku)
        ranker.add_model("claude-sonnet-4-6", sonnet)

        @ranker.evaluate
        def score(problem, response):
            return response.strip().lower() == problem["answer"].lower()

        print("  Would run ranker.rank(dataset=PROBLEMS) — skipping to avoid real API call")
        # result = ranker.rank(dataset=PROBLEMS)
        # print(result.summary())

    except ImportError as e:
        print(f"  Skipped: {e}\n")


# ===========================================================================
# Async cross-framework comparison
# ===========================================================================

async def demo_async_cross_framework():
    """Mix an async OpenAI model with an async HuggingFace model."""
    print("=== Async cross-framework demo ===")
    print("(requires: pip install baysbench[openai,huggingface])\n")

    try:
        from baysbench import BayesianBenchmark
        from baysbench.adapters.huggingface import hf_model_async
        from baysbench.adapters.openai_compat import openai_model_async

        bench = BayesianBenchmark(confidence=0.95)

        async_openai = openai_model_async(
            "gpt-4o-mini",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        async_hf = hf_model_async(
            "mistralai/Mistral-7B-Instruct-v0.3",
            prompt_fn=lambda p: p["question"],
            api_key=os.getenv("HF_TOKEN"),
        )

        def score(problem, response):
            return response.strip().lower() == problem["answer"].lower()

        print("  Would run bench.compare_async(...) — skipping to avoid real API call")
        # result = await bench.compare_async(async_openai, async_hf, score, PROBLEMS)
        # print(result)

    except ImportError as e:
        print(f"  Skipped: {e}\n")


if __name__ == "__main__":
    demo_huggingface()
    demo_openai()
    demo_groq()
    demo_ollama()
    demo_anthropic()
    asyncio.run(demo_async_cross_framework())
    print("Done. Replace stubs with real API keys and uncomment .run() calls.")
