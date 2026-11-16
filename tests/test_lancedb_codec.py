"""LanceDB codec: timestamp decoding and the row pairs for tombstones, events and pairs."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tabularium.domain import HypomnemaId, Scope
from tabularium.storage.lancedb_codec import (
    distinct_pair_to_row,
    event_to_row,
    from_ts,
    read_event_to_row,
    row_to_distinct_pair,
    row_to_event,
    row_to_row_meta,
    row_to_tombstone,
    tombstone_to_row,
)
from tabularium.storage.types import (
    DistinctPair,
    ReadEvent,
    ReinforcementEvent,
    Tombstone,
)

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _hid(n: int) -> HypomnemaId:
    return HypomnemaId(uuid.UUID(int=n))


class TestFromTs:
    def test_none_reads_as_none(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="tabularium"):
            assert from_ts(None) is None

        assert caplog.records == []

    def test_naive_datetime_gains_utc(self) -> None:
        naive = EPOCH.replace(tzinfo=None)

        assert from_ts(naive) == EPOCH

    def test_aware_datetime_is_returned_unchanged(self) -> None:
        aware = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone(timedelta(hours=2)))

        assert from_ts(aware) is aware

    def test_integer_microseconds_convert(self) -> None:
        assert from_ts(1_000_000) == datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("value", "type_name"),
        [
            ("not-a-number", "str"),
            (object(), "object"),
            (10**30, "int"),
            (float("inf"), "float"),
            (float("nan"), "float"),
        ],
    )
    def test_malformed_value_reads_as_none_and_logs_a_warning(
        self, value: object, type_name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="tabularium"):
            result = from_ts(value)

        assert result is None
        assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
            (
                logging.WARNING,
                f"Unconvertible timestamp value of type {type_name} decoded as None",
            )
        ]

    def test_an_unexpected_exception_propagates(self) -> None:
        class Exploding:
            def __int__(self) -> int:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            from_ts(Exploding())


class TestTombstoneRow:
    def test_ordinary_burial_round_trips(self) -> None:
        tombstone = Tombstone(
            id=uuid.UUID(int=1),
            buried_id=_hid(2),
            type="fact",
            reason="gone",
            timestamp=EPOCH,
            model_generated=True,
        )

        row = tombstone_to_row(tombstone)

        assert row["restores"] is None
        assert row["restores_content_hash"] is None
        assert row_to_tombstone(row) == tombstone

    def test_undo_marker_round_trips(self) -> None:
        tombstone = Tombstone(
            id=uuid.UUID(int=1),
            buried_id=_hid(2),
            type="decision",
            reason="unsupersede: why",
            timestamp=EPOCH,
            restores=_hid(3),
            restores_content_hash="abc",
        )

        assert row_to_tombstone(tombstone_to_row(tombstone)) == tombstone

    def test_row_without_undo_columns_decodes_as_ordinary(self) -> None:
        row = {
            "tombstone_id": str(uuid.UUID(int=1)),
            "buried_id": str(_hid(2)),
            "type": "fact",
            "reason": "gone",
            "timestamp": EPOCH,
            "model_generated": False,
        }

        tombstone = row_to_tombstone(row)

        assert tombstone.restores is None
        assert tombstone.restores_content_hash is None


class TestEventRows:
    def test_full_event_round_trips(self) -> None:
        event = ReinforcementEvent(
            target_id=_hid(1),
            at=EPOCH,
            source="feedback",
            matched_from_id=_hid(2),
            value=0.5,
            reason="because",
            similarity=0.75,
        )

        assert row_to_event(event_to_row(event)) == event

    def test_sparse_event_decodes_with_none_defaults(self) -> None:
        row = {"target_id": str(_hid(1)), "at": EPOCH, "source": "dedup"}

        event = row_to_event(row)

        assert event.matched_from_id is None
        assert event.value is None
        assert event.reason is None
        assert event.similarity is None

    def test_read_event_row_has_the_three_columns(self) -> None:
        event = ReadEvent(target_id=_hid(1), at=EPOCH, source="recall")

        assert read_event_to_row(event) == {
            "target_id": str(_hid(1)),
            "at": EPOCH,
            "source": "recall",
        }


class TestDistinctPairRow:
    @pytest.mark.parametrize("similarity", [0.8, None])
    def test_round_trips(self, similarity: float | None) -> None:
        pair = DistinctPair(
            lower_id=_hid(1),
            higher_id=_hid(2),
            source="save",
            judged_at=EPOCH,
            similarity=similarity,
        )

        assert row_to_distinct_pair(distinct_pair_to_row(pair)) == pair


class TestRowMeta:
    def test_missing_optional_columns_take_their_defaults(self) -> None:
        row = {
            "id": str(_hid(1)),
            "type": "decision",
            "scope": "proj-a",
            "lifecycle": "in_force",
            "validity": "valid",
            "recall_count": 2,
        }

        meta = row_to_row_meta(row)

        assert meta.scope == Scope("proj-a")
        assert meta.recall_count == 2
        assert meta.feedback_score == 0.0
        assert meta.last_referenced_at is None
        assert meta.decay_due_at is None
        assert meta.embedding_model is None
