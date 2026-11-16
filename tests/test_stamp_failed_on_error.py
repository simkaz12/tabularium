from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from tabularium.domain import new_id
from tabularium.engine.assessor import RelationVerdict, Trace
from tabularium.engine.verdicts import StampFailedOnError, build_pending_row
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import VerdictRow
from tests.support.doubles import TestClock

_EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


class _RecordingAdapter(MemoryAdapter):
    def __init__(self) -> None:
        super().__init__(embedding_model="test/stamp", dim=4, clock=TestClock(_EPOCH))
        self.updates: list[list[VerdictRow]] = []

    async def update_verdicts(self, rows: Sequence[VerdictRow]) -> None:
        self.updates.append(list(rows))


class _CountingClock:
    def __init__(self) -> None:
        self.reads = 0

    def now(self) -> datetime:
        self.reads += 1
        return _EPOCH + timedelta(seconds=self.reads)


def _row() -> VerdictRow:
    verdict = RelationVerdict(
        method="relation",
        pick="replacement",
        odds={"replacement": 1.0},
        confidence=1.0,
        trace=Trace(
            model="test",
            wording_hash="test",
            latency_ms=0,
            input_tokens=None,
            output_tokens=None,
            request_id=None,
        ),
        reason_class="correction",
    )
    return build_pending_row(
        candidate_id=new_id(),
        similarity=0.9,
        verdict=verdict,
        action="supersede",
        effect_ref="edge",
        new_record_id=str(uuid.uuid4()),
        now=_EPOCH,
    )


async def test_exception_stamps_each_group_in_order_and_reraises_the_same_object() -> None:
    adapter = _RecordingAdapter()
    clock = _CountingClock()
    first, second = _row(), _row()
    boom = RuntimeError("boom")

    with pytest.raises(RuntimeError) as caught:
        async with StampFailedOnError(adapter, clock.now, [first], [second]):
            raise boom

    assert caught.value is boom
    assert [[r.status for r in rows] for rows in adapter.updates] == [["failed"], ["failed"]]
    assert [rows[0].candidate_id for rows in adapter.updates] == [
        first.candidate_id,
        second.candidate_id,
    ]
    assert [rows[0].stamped_at for rows in adapter.updates] == [
        _EPOCH + timedelta(seconds=1),
        _EPOCH + timedelta(seconds=2),
    ]
    assert clock.reads == 2


async def test_clean_exit_reads_no_clock_and_stamps_nothing() -> None:
    adapter = _RecordingAdapter()
    clock = _CountingClock()

    async with StampFailedOnError(adapter, clock.now, [_row()]):
        pass

    assert adapter.updates == []
    assert clock.reads == 0


async def test_cancellation_is_not_stamped() -> None:
    adapter = _RecordingAdapter()
    clock = _CountingClock()

    with pytest.raises(asyncio.CancelledError):
        async with StampFailedOnError(adapter, clock.now, [_row()]):
            raise asyncio.CancelledError

    assert adapter.updates == []
    assert clock.reads == 0


async def test_none_group_is_skipped_and_empty_group_reads_the_clock_without_a_write() -> None:
    adapter = _RecordingAdapter()
    clock = _CountingClock()

    with pytest.raises(ValueError):
        async with StampFailedOnError(adapter, clock.now, None, [], [_row()]):
            raise ValueError("x")

    assert len(adapter.updates) == 1
    assert clock.reads == 2
