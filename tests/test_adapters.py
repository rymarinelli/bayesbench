"""Tests for bayesbench.adapters (mock-based — no real API calls)."""

import pytest

from bayesbench.adapters.base import ModelAdapter, _require

# ---------------------------------------------------------------------------
# ModelAdapter protocol
# ---------------------------------------------------------------------------


class TestModelAdapterProtocol:
    def test_plain_function_satisfies_protocol(self):
        def my_model(problem):
            return "answer"

        assert isinstance(my_model, ModelAdapter)

    def test_callable_class_satisfies_protocol(self):
        class MyModel:
            def __call__(self, problem):
                return "answer"

        assert isinstance(MyModel(), ModelAdapter)

    def test_non_callable_does_not_satisfy(self):
        assert not isinstance("not_callable", ModelAdapter)
        assert not isinstance(42, ModelAdapter)


# ---------------------------------------------------------------------------
# _require helper
# ---------------------------------------------------------------------------


class TestRequireHelper:
    def test_no_error_when_installed(self):
        # numpy is always installed in this project
        _require("numpy", "test")  # should not raise

    def test_raises_import_error_for_missing(self):
        with pytest.raises(ImportError, match="pip install bayesbench"):
            _require("_bayesbench_nonexistent_pkg_xyz", "some-extra")


# ---------------------------------------------------------------------------
# HuggingFace adapter (mock)
# ---------------------------------------------------------------------------


class TestHuggingFaceAdapterMocked:
    def test_hf_model_raises_without_library(self, monkeypatch):
        """If huggingface_hub is absent, hf_model() raises ImportError at call time."""
        import sys

        # Temporarily hide the module if present
        orig = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            from bayesbench.adapters.huggingface import hf_model

            with pytest.raises((ImportError, AttributeError)):
                model = hf_model("test/model", api_key="fake")
                model({"question": "hi"})
        finally:
            if orig is not None:
                sys.modules["huggingface_hub"] = orig
            else:
                del sys.modules["huggingface_hub"]

    def test_hf_model_prompt_fn_default(self, monkeypatch):
        """Default prompt_fn converts the problem to str."""
        import sys
        from types import ModuleType

        class FakeMessage:
            content = "  mocked answer  "

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            def chat_completion(self, messages, **kw):
                return FakeResponse()

        mock_hub = ModuleType("huggingface_hub")
        mock_hub.InferenceClient = FakeClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hub)

        import importlib

        import bayesbench.adapters.huggingface as hf_mod

        importlib.reload(hf_mod)

        model = hf_mod.hf_model("test/model", api_key="fake")
        result = model({"question": "What is 2+2?"})
        assert result == "mocked answer"


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter (mock)
# ---------------------------------------------------------------------------


class TestOpenAIAdapterMocked:
    def test_openai_model_missing_raises(self, monkeypatch):
        import sys

        orig = sys.modules.get("openai")
        sys.modules["openai"] = None  # type: ignore[assignment]
        try:
            from bayesbench.adapters.openai_compat import openai_model

            with pytest.raises((ImportError, AttributeError)):
                model = openai_model("gpt-4o", api_key="fake")
                model({"question": "hi"})
        finally:
            if orig is not None:
                sys.modules["openai"] = orig
            else:
                del sys.modules["openai"]

    def test_openai_model_returns_callable(self, monkeypatch):
        import sys
        from types import ModuleType

        class FakeMsg:
            content = "  mocked  "

        class FakeChoice:
            message = FakeMsg()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kw):
                return FakeCompletion()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *a, **kw):
                self.chat = FakeChat()

        mock_openai = ModuleType("openai")
        mock_openai.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        import importlib

        import bayesbench.adapters.openai_compat as oa_mod

        importlib.reload(oa_mod)

        model = oa_mod.openai_model("gpt-4o", api_key="fake")
        result = model({"question": "hi"})
        assert result == "mocked"

    def test_system_prompt_forwarded(self, monkeypatch):
        import sys
        from types import ModuleType

        captured = {}

        class FakeMsg:
            content = "answer"

        class FakeChoice:
            message = FakeMsg()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, messages, **kw):
                captured["messages"] = messages
                return FakeCompletion()

        class FakeChat:
            completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, *a, **kw):
                self.chat = FakeChat()

        mock_openai = ModuleType("openai")
        mock_openai.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        import importlib

        import bayesbench.adapters.openai_compat as oa_mod

        importlib.reload(oa_mod)

        model = oa_mod.openai_model(
            "gpt-4o",
            system_prompt="Be concise.",
            api_key="fake",
        )
        model({"question": "hi"})
        assert captured["messages"][0]["role"] == "system"
        assert "Be concise." in captured["messages"][0]["content"]

    def test_openai_async_forwards_kwargs_and_metadata(self, monkeypatch):
        import asyncio
        import sys
        from types import ModuleType

        captured = {}

        class FakeMsg:
            content = "  async mocked  "

        class FakeChoice:
            message = FakeMsg()

        class FakeCompletion:
            choices = [FakeChoice()]

        class FakeCompletions:
            async def create(self, **kw):
                captured.update(kw)
                return FakeCompletion()

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, *a, **kw):
                self.chat = FakeChat()

        mock_openai = ModuleType("openai")
        mock_openai.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", mock_openai)

        import importlib

        import bayesbench.adapters.openai_compat as oa_mod

        importlib.reload(oa_mod)

        model = oa_mod.openai_model_async(
            "gpt-4o-mini",
            prompt_fn=lambda p: p["question"],
            system_prompt="system",
            api_key="fake",
            max_tokens=64,
            temperature=0.2,
            top_p=0.7,
        )
        assert model.__bayesbench_model__ == "gpt-4o-mini"
        result = asyncio.run(model({"question": "hi"}))
        assert result == "async mocked"
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][1]["content"] == "hi"
        assert captured["max_tokens"] == 64
        assert captured["temperature"] == 0.2
        assert captured["top_p"] == 0.7


# ---------------------------------------------------------------------------
# Anthropic adapter (mock)
# ---------------------------------------------------------------------------


class TestAnthropicAdapterMocked:
    def test_anthropic_model_missing_raises(self, monkeypatch):
        import sys

        orig = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            from bayesbench.adapters.anthropic_adapter import anthropic_model

            with pytest.raises((ImportError, AttributeError)):
                model = anthropic_model("claude-opus-4-6", api_key="fake")
                model({"question": "hi"})
        finally:
            if orig is not None:
                sys.modules["anthropic"] = orig
            else:
                del sys.modules["anthropic"]

    def test_anthropic_model_returns_callable(self, monkeypatch):
        import sys
        from types import ModuleType

        class FakeContent:
            text = "  mocked  "

        class FakeResponse:
            content = [FakeContent()]

        class FakeMessages:
            def create(self, **kw):
                return FakeResponse()

        class FakeAnthropic:
            def __init__(self, *a, **kw):
                self.messages = FakeMessages()

        mock_anthropic = ModuleType("anthropic")
        mock_anthropic.Anthropic = FakeAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        import importlib

        import bayesbench.adapters.anthropic_adapter as ant_mod

        importlib.reload(ant_mod)

        model = ant_mod.anthropic_model("claude-opus-4-6", api_key="fake")
        result = model({"question": "hi"})
        assert result == "mocked"

    def test_anthropic_async_forwards_system_and_kwargs(self, monkeypatch):
        import asyncio
        import sys
        from types import ModuleType

        captured = {}

        class FakeContent:
            text = "  async anthropic  "

        class FakeResponse:
            content = [FakeContent()]

        class FakeMessages:
            async def create(self, **kw):
                captured.update(kw)
                return FakeResponse()

        class FakeAsyncAnthropic:
            def __init__(self, *a, **kw):
                self.messages = FakeMessages()

        mock_anthropic = ModuleType("anthropic")
        mock_anthropic.AsyncAnthropic = FakeAsyncAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        import importlib

        import bayesbench.adapters.anthropic_adapter as ant_mod

        importlib.reload(ant_mod)

        model = ant_mod.anthropic_model_async(
            "claude-haiku-test",
            prompt_fn=lambda p: p["question"],
            system_prompt="helpful",
            api_key="fake",
            max_tokens=32,
            temperature=0.3,
            top_p=0.9,
        )
        assert model.__bayesbench_model__ == "claude-haiku-test"
        result = asyncio.run(model({"question": "hello"}))
        assert result == "async anthropic"
        assert captured["messages"][0]["content"] == "hello"
        assert captured["system"] == "helpful"
        assert captured["top_p"] == 0.9


class TestHuggingFaceAdapterAsyncMocked:
    def test_hf_async_prompt_and_kwargs(self, monkeypatch):
        import asyncio
        import sys
        from types import ModuleType

        captured = {}

        class FakeMessage:
            content = "  hf async  "

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeAsyncClient:
            def __init__(self, *a, **kw):
                pass

            async def chat_completion(self, messages, **kw):
                captured["messages"] = messages
                captured.update(kw)
                return FakeResponse()

        mock_hub = ModuleType("huggingface_hub")
        mock_hub.AsyncInferenceClient = FakeAsyncClient  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", mock_hub)

        import importlib

        import bayesbench.adapters.huggingface as hf_mod

        importlib.reload(hf_mod)

        model = hf_mod.hf_model_async(
            "test/model",
            prompt_fn=lambda p: p["prompt"],
            system_prompt="be brief",
            api_key="fake",
            max_new_tokens=17,
            temperature=0.6,
            top_p=0.8,
        )
        assert model.__bayesbench_model__ == "test/model"
        result = asyncio.run(model({"prompt": "Ping?"}))
        assert result == "hf async"
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][1]["content"] == "Ping?"
        assert captured["max_tokens"] == 17
        assert captured["temperature"] == 0.6
        assert captured["top_p"] == 0.8
