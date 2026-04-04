"""Tests for baysbench.adapters.openclaw."""

import pytest

from baysbench.adapters.openclaw import openclaw_agent


class _RunAgent:
    def run(self, prompt):
        return {"output": f"  {prompt}  "}


class _CallableAgent:
    def __call__(self, prompt):
        return type("Resp", (), {"text": f"{prompt} done"})()


class TestOpenClawAdapter:
    def test_run_method(self):
        model = openclaw_agent(_RunAgent())
        assert model({"input": "hello"}) == "hello"

    def test_callable_agent(self):
        model = openclaw_agent(_CallableAgent())
        assert model({"input": "task"}) == "task done"

    def test_custom_prompt_fn(self):
        model = openclaw_agent(_RunAgent(), prompt_fn=lambda p: p["question"])
        assert model({"question": "custom"}) == "custom"

    def test_invalid_agent_raises(self):
        with pytest.raises(TypeError):
            openclaw_agent(object())({"input": "x"})
