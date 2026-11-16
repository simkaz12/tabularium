"""Embedder seam and its ``litellm`` adapter.

``Embedder`` is the async protocol the engine depends on. ``LitellmEmbedder`` implements it with
``litellm.aembedding``; tests inject a double from ``tests/support/doubles.py``.

``embed`` raises ``EmbedderUnavailable`` for any exception from the ``litellm`` call and for a
vector whose length differs from the configured ``dim``. The save paths catch it and continue
without a vector, recall continues without the vector leg and reports the degraded reason, and
the backfill pass ends on it. There is no pre-flight probe. Retries are ``litellm``'s own
``num_retries``.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import litellm

from tabularium.config import (
    DEFAULT_EMBEDDER_API_BASE,
    DEFAULT_EMBEDDER_DIM,
    DEFAULT_EMBEDDER_MODEL,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class EmbedderUnavailable(Exception):
    """Raised when the embedder cannot produce a vector; ``reason`` is the message."""

    def __init__(self, reason: str, *, cause: BaseException | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cause = cause


# ---------------------------------------------------------------------------
# Protocol (seam)
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Async embedder seam.

    Implementors embed one string and return a float list of length ``dim``. A failure raises
    ``EmbedderUnavailable`` rather than a provider error.
    """

    @property
    def dim(self) -> int:
        """Vector dimensionality produced by this embedder."""
        ...

    @property
    def model(self) -> str:
        """Embedder model identifier (e.g. ``'ollama/nomic-embed-text'``)."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed ``text`` and return a float list of length ``self.dim``.

        Raises:
            EmbedderUnavailable: on any provider or network error.
        """
        ...


# ---------------------------------------------------------------------------
# litellm adapter
# ---------------------------------------------------------------------------


class LitellmEmbedder:
    """Embedder backed by ``litellm.aembedding``.

    Defaults: model ``ollama/nomic-embed-text``, ``dim`` 768, ``api_base``
    ``http://localhost:11434``, a 10 second timeout per attempt and 2 ``litellm`` retries.
    """

    _DEFAULT_TIMEOUT = 10.0
    _DEFAULT_NUM_RETRIES = 2

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDER_MODEL,
        dim: int = DEFAULT_EMBEDDER_DIM,
        api_base: str = DEFAULT_EMBEDDER_API_BASE,
        timeout: float = _DEFAULT_TIMEOUT,
        num_retries: int = _DEFAULT_NUM_RETRIES,
    ) -> None:
        self._model = model
        self._dim = dim
        self._api_base = api_base
        self._timeout = timeout
        self._num_retries = num_retries

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Embed ``text`` and return the vector.

        Raises:
            EmbedderUnavailable: if the ``litellm`` call raises, or the vector length is not
                ``dim``.
        """
        try:
            response = await litellm.aembedding(
                model=self._model,
                input=[text],
                api_base=self._api_base,
                timeout=self._timeout,
                num_retries=self._num_retries,
            )
            vector: list[float] = response.data[0]["embedding"]
        except Exception as exc:
            _log.warning("Embedder unavailable: %s", exc)
            raise EmbedderUnavailable(str(exc), cause=exc) from exc

        if len(vector) != self._dim:
            msg = (
                f"Embedder returned {len(vector)} dimensions but configured dim={self._dim}; "
                "check model/dim config alignment"
            )
            raise EmbedderUnavailable(msg)

        return vector
