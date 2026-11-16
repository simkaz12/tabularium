"""Single-flight re-embed of vector-less and stale-model rows.

``Backfiller`` is the single-flight scheduler the engine holds one of. ``task`` is the in-flight
``asyncio.Task`` or ``None``; ``rerun_pending`` is set by a trigger that lands while a pass runs;
``dirty`` gates whether a trigger schedules a pass at all, so a save or recall on a clean store
never pays for a table scan. Writers see it through ``BackfillSignals``: ``embedded`` after a
successful embed and ``written`` after a vector-less row commits. ``NoBackfill`` is the signal
sink for callers that must not schedule a pass.

``run_pass`` embeds outside the write lock, one row at a time, and takes the caller-supplied lock
only for the guarded two-column write. ``EmbedderUnavailable`` ends the pass and counts every
remaining target as failed. An exception from the guarded write is logged at WARNING, counted as
failed, and the pass moves on; any other exception from ``embed`` propagates.

``Backfiller.start`` starts a task when idle. A trigger while a pass runs sets ``rerun_pending``
and the same task loops once more before going idle, so two passes never run at once.
``Backfiller.pending`` awaits the current task; on success that task has absorbed every pending
rerun.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from pydantic import BaseModel, Field

from tabularium.embedding import Embedder, EmbedderUnavailable
from tabularium.storage.types import StorageAdapter

_log = logging.getLogger("tabularium")


class BackfillReport(BaseModel):
    """Counts for one backfill pass."""

    model_config = {"frozen": True}

    targets: int = Field(ge=0)
    filled: int = Field(ge=0)
    skipped_hash: int = Field(ge=0)
    failed: int = Field(ge=0)


async def run_pass(
    adapter: StorageAdapter,
    embedder: Embedder,
    lock: asyncio.Lock,
    *,
    embedding_model: str,
    trigger: str,
) -> BackfillReport:
    """Run one backfill pass and return its report.

    Reads targets and embeds with no lock held; only the guarded write takes ``lock``, so a pass
    never holds it across an embed.
    """
    targets = await adapter.backfill_targets()
    filled = 0
    skipped_hash = 0
    failed = 0

    for index, target in enumerate(targets):
        try:
            vector = await embedder.embed(target.content)
        except EmbedderUnavailable:
            failed += len(targets) - index
            break

        try:
            async with lock:
                ok = await adapter.backfill(
                    target.id,
                    vector,
                    embedding_model=embedding_model,
                    expected_hash=target.content_hash,
                )
        except Exception as exc:
            _log.warning(
                "backfill row failed id=%s: %s", target.id, exc, exc_info=exc
            )
            failed += 1
            continue

        if ok:
            filled += 1
        else:
            skipped_hash += 1

    report = BackfillReport(
        targets=len(targets), filled=filled, skipped_hash=skipped_hash, failed=failed
    )
    _log.info(
        "backfill_pass trigger=%s targets=%d filled=%d skipped_hash=%d failed=%d",
        trigger,
        report.targets,
        report.filled,
        report.skipped_hash,
        report.failed,
        extra={
            "trigger": trigger,
            "targets": report.targets,
            "filled": report.filled,
            "skipped_hash": report.skipped_hash,
            "failed": report.failed,
        },
    )
    return report


class BackfillSignals(Protocol):
    """What a writer tells the backfiller: a successful embed, or a row written without a vector."""

    def embedded(self) -> None: ...

    def written(self) -> None: ...


class NoBackfill:
    """Signal sink that schedules nothing and records nothing."""

    def embedded(self) -> None:
        return None

    def written(self) -> None:
        return None


class Backfiller:
    """Single-flight backfill scheduler: one instance per engine."""

    def __init__(self, adapter: StorageAdapter, embedder: Embedder, lock: asyncio.Lock) -> None:
        self._adapter = adapter
        self._embedder = embedder
        self._lock = lock
        self.task: asyncio.Task[None] | None = None
        self.rerun_pending: bool = False
        self.dirty: bool = False

    def embedded(self) -> None:
        """Record a successful embed: schedule a pass with trigger ``embed``."""
        self.schedule(trigger="embed")

    def written(self) -> None:
        """Record a row written without a vector: mark the store dirty."""
        self.dirty = True

    def schedule(self, *, trigger: str) -> None:
        """Schedule a backfill pass when the store is dirty or a pass is already running.

        A no-op only when both are false, so a save or recall on a clean, idle store never pays
        for a table scan. The running-pass check is separate from ``dirty`` because a pass clears
        ``dirty`` for its whole run, and a trigger arriving then must still reach ``start()``
        to set ``rerun_pending``.

        The pass clears ``dirty`` before its scan. A vector-less write sets ``dirty`` after its
        row commits, with no ``await`` between, so its mark either lands before the clear (the
        scan sees the row) or after it (nothing clears it again). At the end the pass only sets
        ``dirty`` back to True: for leftover backlog, or on exception or cancellation, where the
        ``BaseException`` catch restores it before re-raising. It never clears a concurrent
        write's mark.
        """
        pass_is_running = self.task is not None and not self.task.done()
        if not pass_is_running and not self.dirty:
            return

        async def _pass() -> BackfillReport:
            self.dirty = False
            try:
                report = await run_pass(
                    self._adapter,
                    self._embedder,
                    self._lock,
                    embedding_model=self._embedder.model,
                    trigger=trigger,
                )
            except BaseException:
                self.dirty = True
                raise
            if report.skipped_hash > 0 or report.failed > 0:
                self.dirty = True
            return report

        self.start(_pass)

    def start(self, coro_factory: Callable[[], Awaitable[BackfillReport]]) -> None:
        """Start a pass if idle, otherwise mark a rerun for the in-flight pass.

        Never starts a second pass while one is running.
        """
        if self.task is None or self.task.done():
            self.rerun_pending = False
            self.task = asyncio.ensure_future(self._run_and_absorb_reruns(coro_factory))
        else:
            self.rerun_pending = True

    async def pending(self) -> None:
        """Resolve once no pass is running and no rerun is pending."""
        task = self.task
        if task is not None:
            await task

    async def _run_and_absorb_reruns(
        self, coro_factory: Callable[[], Awaitable[BackfillReport]]
    ) -> None:
        """Run ``coro_factory`` until no rerun is pending, then clear ``task``.

        ``task`` is reset to ``None`` on success and failure alike, so the next ``start`` begins
        a fresh task. A caller already awaiting the failed task sees its exception; a later
        ``pending`` call sees ``None`` and returns cleanly. A rerun requested while a failing
        pass was in flight is rescheduled rather than dropped, because the row it names landed
        after that pass's scan.
        """
        try:
            while True:
                await coro_factory()
                if not self.rerun_pending:
                    break
                self.rerun_pending = False
        finally:
            self.task = None
            if self.rerun_pending:
                self.rerun_pending = False
                self.start(coro_factory)


__all__ = [
    "BackfillReport",
    "BackfillSignals",
    "Backfiller",
    "NoBackfill",
    "run_pass",
]
