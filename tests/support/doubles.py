"""Test doubles for the clock, embedder, token counter and judge seams, plus two helpers."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import pathlib

from tabularium.domain import Fact, FactLifecycle, HypomnemaId
from tabularium.embedding import EmbedderUnavailable
from tabularium.engine.judge import _FINGERPRINT_NOT_APPLICABLE, JudgeUnavailable
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import StorageAdapter

_ALL_FACT_LIFECYCLES: frozenset[FactLifecycle] = frozenset(
    {"active", "hot", "ember", "archived", "shelved"}
)
_FROZEN_FIXTURE = (
    pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "frozen_vectors.json"
)


class TestClock:
    """Deterministic clock with a fixed start time and manual advance."""

    __test__ = False

    def __init__(self, start: datetime.datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("TestClock start time must be tz-aware UTC")
        self._current = start

    def now(self) -> datetime.datetime:
        return self._current

    def advance(self, delta: datetime.timedelta) -> None:
        self._current = self._current + delta

    def set(self, t: datetime.datetime) -> None:
        if t.tzinfo is None:
            raise ValueError("TestClock.set requires a tz-aware datetime")
        self._current = t


class FakeEmbedder:
    """Single-hot unit vector at ``sha256(text) % dim``; ``should_fail`` simulates an outage.

    Identical text gives identical vectors; texts whose digests land in different slots are
    orthogonal. ``peek(text)`` returns the vector ``embed(text)`` would produce.
    """

    def __init__(
        self,
        *,
        dim: int = 768,
        model: str = "fake/nomic-embed-text",
        should_fail: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim
        self._model = model
        self.should_fail = should_fail

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        slot = int.from_bytes(digest[:4], "big") % self._dim
        v = [0.0] * self._dim
        v[slot] = 1.0
        return v

    def peek(self, text: str) -> list[float]:
        return self._vector(text)

    async def embed(self, text: str) -> list[float]:
        if self.should_fail:
            raise EmbedderUnavailable("FakeEmbedder: simulated outage")
        return self._vector(text)


class FrozenEmbedder:
    """Dict-backed embedder over ``tests/fixtures/frozen_vectors.json``.

    ``embed`` raises ``EmbedderUnavailable`` for a text missing from the fixture.
    """

    def __init__(self, fixture_path: pathlib.Path | None = None) -> None:
        path = fixture_path or _FROZEN_FIXTURE
        if not path.exists():
            raise FileNotFoundError(
                f"Frozen-vector fixture not found at {path}. "
                "Run scripts/freeze_vectors.py to generate it."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._model: str = raw["embedder_model"]
        self._dim: int = raw["dim"]
        self._vectors: dict[str, list[float]] = raw["vectors"]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        vec = self._vectors.get(text)
        if vec is None:
            raise EmbedderUnavailable(
                f"FrozenEmbedder: unknown text not in fixture: {text[:80]!r}. "
                "Add it to freeze_vectors.py and regenerate."
            )
        return list(vec)


class FakeTokenCounter:
    """Deterministic counter: ``len(text) // 4``."""

    def count(self, text: str) -> int:
        return len(text) // 4


class FakeJudgeAdapter:
    """Canned-response judge adapter.

    ``should_fail`` raises ``JudgeUnavailable``; ``raise_exception`` raises any other exception;
    ``delay`` sleeps inside ``call`` before returning or raising.
    """

    def __init__(
        self,
        *,
        identity: str = "fake:judge",
        response: str = "",
        should_fail: bool = False,
        fail_reason: str = "FakeJudgeAdapter: simulated outage",
        raise_exception: BaseException | None = None,
        delay: float = 0.0,
        instruction_fingerprint: str = _FINGERPRINT_NOT_APPLICABLE,
    ) -> None:
        self._identity = identity
        self._response = response
        self._should_fail = should_fail
        self._fail_reason = fail_reason
        self._raise_exception = raise_exception
        self._delay = delay
        self._instruction_fingerprint = instruction_fingerprint
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return self._instruction_fingerprint

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise_exception is not None:
            raise self._raise_exception
        if self._should_fail:
            raise JudgeUnavailable(self._fail_reason)
        return self._response


async def find_superseded_by(id: HypomnemaId, *, adapter: StorageAdapter) -> list[Fact]:
    """Every Fact whose ``superseded_by == id``, at any Fact lifecycle."""
    results: list[Fact] = []
    async for rec in await adapter.scan_records(lifecycle_in=_ALL_FACT_LIFECYCLES):
        if isinstance(rec, Fact) and rec.superseded_by == id:
            results.append(rec)
    return results


def reopen_memory_at(existing: MemoryAdapter, *, embedding_model: str) -> MemoryAdapter:
    """A second adapter view over ``existing``'s rows and events, configured at ``embedding_model``.

    Shares ``_rows`` and ``_events`` only; the insert-order counter is not shared, so writing new
    records through the reopened view is unsupported.
    """
    reopened = MemoryAdapter(
        embedding_model=embedding_model,
        dim=existing._dim,
        clock=existing._clock,
    )
    reopened._rows = existing._rows
    reopened._events = existing._events
    return reopened
