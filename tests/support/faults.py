"""Fault-injection wrappers for the admission case-set harness.

Three named hooks, none of them in ``src/`` — the engine must never know an injector exists:

1. ``FaultInjectingAdapter`` / ``FaultInjectingEmbedder`` delegate every call to a real adapter or
   embedder and raise, BEFORE delegation, on the Nth call of one named method — proving the
   pre-write error path. Counting starts only after ``arm()``, so seeding writes never consume the
   injected fault.
2. ``read_only_table_dir`` makes a LanceDB table's on-disk directory tree read-only so a real
   ``merge_insert`` fails inside its own commit — the in-batch storage fault.
3. The process-kill atomicity check lives in ``tests/test_admission_process_kill.py``, a standalone
   integration test rather than anything this module offers — it proves a storage property, not an
   admission outcome.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from tabularium.embedding import Embedder, EmbedderUnavailable
from tabularium.storage.types import StorageAdapter

_FAULT_INJECTED_REASON = "fault-injected"


class FaultInjectingAdapter:
    """Delegates every attribute to ``inner`` via ``__getattr__``, except that the ONE method
    named by ``fault()`` raises before delegation on its Nth call, counted from the most recent
    ``arm()``.

    Construct with a real ``StorageAdapter`` (``MemoryAdapter`` or ``LanceDBAdapter``); pass the
    wrapper to ``Tabularium`` via ``as_adapter()`` so mypy stays honest about the structural gap
    ``__getattr__`` delegation leaves.
    """

    def __init__(self, inner: StorageAdapter) -> None:
        self._inner = inner
        self._armed = False
        self._fault_on: str | None = None
        self._fault_call: int | None = None
        self._fault_exc: BaseException | None = None
        self._call_counts: dict[str, int] = {}

    def fault(self, *, on: str, call: int, exc: BaseException) -> None:
        """Configure the Nth call of method ``on`` to raise ``exc`` before delegation."""
        self._fault_on = on
        self._fault_call = call
        self._fault_exc = exc

    def arm(self) -> None:
        """Start counting calls to the faulted method — calls before this are free."""
        self._armed = True
        self._call_counts.clear()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._fault_on or not callable(attr):
            return attr

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if self._armed:
                count = self._call_counts.get(name, 0) + 1
                self._call_counts[name] = count
                if count == self._fault_call:
                    assert self._fault_exc is not None
                    raise self._fault_exc
            return await attr(*args, **kwargs)

        return _wrapped


def as_adapter(wrapper: FaultInjectingAdapter) -> StorageAdapter:
    """Cast a ``FaultInjectingAdapter`` to ``StorageAdapter`` at a construction site.

    The wrapper satisfies ``StorageAdapter`` only by runtime delegation through
    ``__getattr__`` — structurally invisible to mypy — so callers that hand the wrapper to
    ``Tabularium`` go through this cast rather than silencing the error inline.
    """
    return cast(StorageAdapter, wrapper)


class FaultInjectingEmbedder:
    """Delegates ``embed`` to ``inner``, with two fault modes:

    - Plain ``arm()`` (no ``fault()`` call): every ``embed`` raises
      ``EmbedderUnavailable("fault-injected")`` until ``lift()``.
    - ``fault(on="embed", call=N, exc=...)`` then ``arm()``: only the Nth call after arming
      raises ``exc``; every other call delegates normally.
    """

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._armed = False
        self._fault_call: int | None = None
        self._fault_exc: BaseException | None = None
        self._embed_calls = 0

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model(self) -> str:
        return self._inner.model

    def fault(self, *, on: str, call: int, exc: BaseException) -> None:
        if on != "embed":
            raise ValueError(f"FaultInjectingEmbedder only injects 'embed', got {on!r}")
        self._fault_call = call
        self._fault_exc = exc

    def arm(self) -> None:
        self._armed = True
        self._embed_calls = 0

    def lift(self) -> None:
        """Clear the armed fault — a case can bring the embedder back mid-run."""
        self._armed = False
        self._fault_call = None
        self._fault_exc = None
        self._embed_calls = 0

    async def embed(self, text: str) -> list[float]:
        if self._armed:
            if self._fault_call is None:
                raise EmbedderUnavailable(_FAULT_INJECTED_REASON)
            self._embed_calls += 1
            if self._embed_calls == self._fault_call:
                assert self._fault_exc is not None
                raise self._fault_exc
        return await self._inner.embed(text)


def as_embedder(wrapper: FaultInjectingEmbedder) -> Embedder:
    """Cast a ``FaultInjectingEmbedder`` to ``Embedder`` at a construction site."""
    return cast(Embedder, wrapper)


@contextmanager
def read_only_table_dir(store_path: Path, table: str) -> Iterator[None]:
    """Make ``<store_path>/<table>.lance`` and every directory under it read-only for the
    duration of the ``with`` block, then restore the original modes.

    Raises ``RuntimeError`` when running as root — root ignores the read-only bit, so the fault
    this context manager exists to inject would silently not fire; the caller's own test skips on
    that.
    """
    if os.geteuid() == 0:
        raise RuntimeError("running as root, read-only bit has no effect")

    table_dir = store_path / f"{table}.lance"
    directories = sorted((p for p in table_dir.rglob("*") if p.is_dir()), reverse=True)
    directories.insert(0, table_dir)
    original_modes = [(d, stat.S_IMODE(d.stat().st_mode)) for d in directories]

    try:
        for directory, _ in original_modes:
            os.chmod(directory, 0o555)
        yield
    finally:
        for directory, mode in reversed(original_modes):
            os.chmod(directory, mode)
