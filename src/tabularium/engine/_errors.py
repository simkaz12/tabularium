"""Engine error types, kept free of imports from ``core`` so any engine module can raise them."""

from __future__ import annotations


class ToolError(Exception):
    """A wrong call or a missing target; surfaces as ``isError=true`` in the MCP layer.

    Parks and bounces are envelope outcomes (``isError=false``), not ToolErrors.
    """


class SupersessionWriteFailed(ToolError):
    """A storage failure inside a save's supersession write; ``__cause__`` is the storage error.

    Message is ``supersedes: <cause>``. Raised only by the Fact and Decision save arms.
    """


def require_text(
    value: str | None,
    message: str,
    *,
    error: type[Exception] = ToolError,
) -> None:
    """Raise ``error(message)`` when ``value`` is ``None``, empty or only whitespace."""
    if not value or not value.strip():
        raise error(message)
