"""Token-counter seam: ``TokenCounter`` and its ``litellm`` adapter.

``TokenCounter`` is the synchronous protocol that ``engine/context.py`` uses for budgeting and
``engine/judge.py`` uses to count each judge prompt's tokens. ``LitellmTokenCounter`` binds a
model name to ``litellm.token_counter``.
Tests inject a counter from ``tests/support/doubles.py``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import litellm

# ---------------------------------------------------------------------------
# Protocol (seam)
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenCounter(Protocol):
    """Synchronous token-counter seam.

    ``count`` returns the same number for the same text on repeated calls and has no side
    effects. Implementors do not raise for a well-formed ``str``.
    """

    def count(self, text: str) -> int:
        """Return the token cost of ``text`` for this counter's model."""
        ...


# ---------------------------------------------------------------------------
# litellm adapter
# ---------------------------------------------------------------------------


class LitellmTokenCounter:
    """Counter backed by ``litellm.token_counter`` for the bound ``model``.

    The default model is ``gpt-4``. It adds no error handling, so whatever ``litellm`` raises
    propagates.
    """

    _DEFAULT_MODEL = "gpt-4"

    def __init__(self, *, model: str = _DEFAULT_MODEL) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def count(self, text: str) -> int:
        result: int = litellm.token_counter(model=self._model, text=text)
        return result
