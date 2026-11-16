"""Startup passes ``Tabularium.initialize()`` runs before serving starts.

Order is fixed:

1. fill ``content_key``/``content_hash`` on pre-existing rows;
2. assign a fresh edge id to every supersession edge that lacks one;
3. finish any undo a kill interrupted (``repair_unsupersedes``), which needs every edge id
   populated to compare against, and which therefore runs here rather than in ``open()``;
4. resolve every leftover ``pending`` verdict row to ``acted`` or ``failed``.
"""

from __future__ import annotations

import asyncio
import logging
import time

from tabularium.clock import Clock
from tabularium.engine.verdicts import reconcile_pending_verdicts
from tabularium.storage.types import StorageAdapter


async def run_startup_passes(
    adapter: StorageAdapter,
    log: logging.Logger,
    *,
    lock: asyncio.Lock,
    clock: Clock,
) -> None:
    """Run the four passes in order, each timed and logged as one INFO line.

    Holds ``lock``, the engine's single global write lock, for the whole call.
    """
    async with lock:
        for name, populate in (
            ("content_key", adapter.populate_content_keys),
            ("edge_id", adapter.populate_edge_ids),
        ):
            start = time.monotonic()
            written = await populate()
            elapsed = time.monotonic() - start
            log.info("startup pass %s: %d rows, %.3fs", name, written, elapsed)

        start = time.monotonic()
        report = await adapter.repair_unsupersedes()
        elapsed = time.monotonic() - start
        log.info(
            "unsupersede_repair completed=%d skipped=%d refused=%d (%.3fs)",
            report.completed,
            report.skipped,
            report.refused,
            elapsed,
        )

        start = time.monotonic()
        acted, failed = await reconcile_pending_verdicts(adapter, now=clock.now())
        elapsed = time.monotonic() - start
        log.info(
            "verdict_reconcile acted=%d failed=%d (%.3fs)",
            acted,
            failed,
            elapsed,
        )
