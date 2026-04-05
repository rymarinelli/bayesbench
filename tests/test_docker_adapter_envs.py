"""Optional Docker-based adapter smoke tests.

These tests validate that selected extras can be installed in a clean
container and that each adapter can be exercised without real API calls.

They are disabled by default and run only when:
- Docker is available on the host, and
- BAYESBENCH_RUN_DOCKER_TESTS=1 is set.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _docker_is_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return result.returncode == 0


def _run_in_docker(
    extra: str,
    python_snippet: str,
    additional_install: str = "",
) -> subprocess.CompletedProcess[str]:
    quoted_script = shlex.quote(textwrap.dedent(python_snippet).strip())
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPO_ROOT}:/work",
        "-w",
        "/work",
        "python:3.11-slim",
        "bash",
        "-lc",
        (
            "python -m pip install --upgrade pip >/dev/null "
            f"&& python -m pip install --no-cache-dir '.[dev,{extra}]' >/dev/null "
            f"&& {additional_install} "
            f"&& python -c {quoted_script}"
        ),
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.docker
@pytest.mark.parametrize(
    ("extra", "additional_install", "snippet"),
    [
        (
            "openai",
            "true",
            """
from unittest.mock import patch

try:
    from bayesbench.adapters.openai_compat import openai_model
except ImportError:
    from baysbench.adapters.openai_compat import openai_model

class _Response:
    choices = [type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()]

class _Completions:
    def create(self, **kwargs):
        return _Response()

class _Client:
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": _Completions()})()

with patch("openai.OpenAI", _Client):
    model = openai_model("gpt-4o-mini", api_key="fake")
    assert model({"input": "hello"}) == "ok"
""",
        ),
        (
            "anthropic",
            "true",
            """
from unittest.mock import patch

try:
    from bayesbench.adapters.anthropic_adapter import anthropic_model
except ImportError:
    from baysbench.adapters.anthropic_adapter import anthropic_model

class _Client:
    def __init__(self, **kwargs):
        self.messages = self

    def create(self, **kwargs):
        return type("Response", (), {"content": [type("Part", (), {"text": "ok"})()]})()

with patch("anthropic.Anthropic", _Client):
    model = anthropic_model("claude-3-5-haiku-latest", api_key="fake")
    assert model({"input": "hello"}) == "ok"
""",
        ),
        (
            "openclaw",
            "python -m pip install --no-cache-dir tenacity >/dev/null",
            """

import openclaw
assert openclaw is not None

try:
    from bayesbench.adapters.openclaw import openclaw_agent
except ImportError:
    from baysbench.adapters.openclaw import openclaw_agent

class _Agent:
    def run(self, prompt):
        return {"output": f"  {prompt}  "}

model = openclaw_agent(_Agent())
assert model({"input": "hello"}) == "hello"
""",
        ),
        (
            "inspect",
            "true",
            """

from inspect_ai.dataset import Sample

try:
    from bayesbench.adapters.inspect_ai import from_inspect_dataset, exact_match_score
except ImportError:
    from baysbench.adapters.inspect_ai import from_inspect_dataset, exact_match_score

samples = [Sample(input="What is 2+2?", target="4", id="s1")]
problems = from_inspect_dataset(samples)
assert problems[0]["input"] == "What is 2+2?"
assert problems[0]["target"] == "4"
assert problems[0]["id"] == "s1"
assert exact_match_score(problems[0], "4") is True
""",
        ),
    ],
)
def test_adapter_smoke_in_clean_docker_env(extra: str, additional_install: str, snippet: str):
    if os.getenv("BAYESBENCH_RUN_DOCKER_TESTS") != "1":
        pytest.skip("Set BAYESBENCH_RUN_DOCKER_TESTS=1 to run Docker integration tests")

    if not _docker_is_available():
        pytest.skip("Docker is not available in this environment")

    result = _run_in_docker(
        extra=extra,
        python_snippet=snippet,
        additional_install=additional_install,
    )
    assert result.returncode == 0, (
        f"Docker smoke test failed for extra '{extra}'.\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )
