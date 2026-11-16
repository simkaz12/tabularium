"""Read-event logging: ``log_read_events`` appends one ``ReadEvent`` per record a read delivers.

Callers pass ``source`` themselves: ``recall`` uses ``"recall"``, the decision view uses
``"decision-view"``. ``context()`` logs nothing. ``"grounding"`` and ``"console"`` stay in
``ReadSource`` so stored rows keep parsing.

Logging never fails the read it is attached to: an ``append_read_event`` failure is caught and
logged at WARNING. Appends are plain inserts, one per id, so one id's failure does not lose the
others. This module never takes the engine write lock, so a read cannot block or be blocked by a
save or disposition.

NOTE: do not call ``maybe_upkeep`` from here. Upkeep at read frequency accumulates dataset
versions faster than ``reclaim`` prunes them and grows latency and memory without bound.

NOTE: the read-events table has no retention policy and grows without bound.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tabularium.clock import Clock
from tabularium.domain import HypomnemaId
from tabularium.storage.types import ReadEvent, ReadSource, StorageAdapter

_log = logging.getLogger("tabularium")


async def log_read_events(
    ids: Sequence[HypomnemaId],
    source: ReadSource,
    *,
    adapter: StorageAdapter,
    clock: Clock,
) -> None:
    """Append one ``ReadEvent(source=source)`` per id, all sharing one ``clock.now()`` sample.

    A no-op for an empty ``ids``. An ``append_read_event`` failure is logged at WARNING and
    swallowed, so a read does not fail because its audit trail could not be written. Does not call
    ``maybe_upkeep``; see the module NOTE.
    """
    if not ids:
        return
    now = clock.now()
    for target_id in ids:
        try:
            await adapter.append_read_event(
                ReadEvent(target_id=target_id, at=now, source=source)
            )
        except Exception:
            _log.warning(
                "failed to log read event (source=%r, target_id=%s)",
                source,
                target_id,
                exc_info=True,
            )

