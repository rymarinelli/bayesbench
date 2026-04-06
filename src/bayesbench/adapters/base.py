"""ModelAdapter protocol and shared utilities.

An adapter is simply a callable ``(problem: Any) -> str``.  The protocol
makes static type-checking work and serves as documentation.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for a model callable used in bayesbench.

    Any object that implements ``__call__(problem) -> str`` satisfies this
    protocol — plain functions, async functions, and framework-specific
    wrappers all qualify.
    """

    def __call__(self, problem: Any) -> str:
        """Call the model on a problem dict and return a string response."""
        ...


def _require(package: str, extra: str) -> None:
    """Raise a helpful ImportError if ``package`` is not installed."""
    try:
        __import__(package)
    except ImportError as exc:
        raise ImportError(
            f"Package '{package}' is required for this adapter. "
            f"Install it with: pip install bayesbench[{extra}]"
        ) from exc
