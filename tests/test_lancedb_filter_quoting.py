"""LanceDB adapter: every query that carries a caller value survives a single quote.

Each formerly interpolated path is driven with a value containing ``'``, once as a
value that must match a stored row and once as an injection-shaped value that must
match nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tabularium.domain import Scope, Session, new_id
from tabularium.storage.content_key import content_hash, content_key
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import CleanerPassRecord, CleanerState, RecordReview
from tests.conformance.test_conformance import _fact, _vec, _verdict

EPOCH = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
QUOTED = "o'brien's \"x\""
INJECTION = "x' OR '1'='1"
LONE = "'"


@pytest.fixture
async def adapter(tmp_path: Path) -> LanceDBAdapter:
    return await LanceDBAdapter.open(str(tmp_path / "store"), embedding_model="test-model", dim=4)


async def test_scope_paths_through_the_adapter_api(adapter: LanceDBAdapter) -> None:
    fact = _fact(scope="proj-a")
    await adapter.upsert(fact, _vec())
    since = EPOCH - timedelta(days=1)
    before = EPOCH + timedelta(days=1)

    for probe in (QUOTED, INJECTION, LONE):
        assert await adapter.list_born_since(scope=Scope(probe), since=since, limit=10) == []
        assert (
            await adapter.list_due_for_review(
                scope=Scope(probe), now=EPOCH, created_before=before, limit=10
            )
            == []
        )
        assert await adapter.get_cleaner_state(scope=Scope(probe)) is None
        assert await adapter.list_cleaner_passes(scope=Scope(probe), limit=5) == []

    born = await adapter.list_born_since(scope=Scope("proj-a"), since=since, limit=10)
    assert [r.id for r in born] == [fact.id]

    state = CleanerState(watermark_at=EPOCH, last_pass_at=EPOCH, report=None)
    await adapter.set_cleaner_state(scope=Scope(QUOTED), state=state)
    assert await adapter.get_cleaner_state(scope=Scope(QUOTED)) == state
    assert await adapter.get_cleaner_state(scope=Scope(INJECTION)) is None

    record = CleanerPassRecord(
        scope=Scope(QUOTED), pass_at=EPOCH, outcome="completed", dry_run=False, payload="{}"
    )
    await adapter.append_cleaner_pass(record=record)
    assert await adapter.list_cleaner_passes(scope=Scope(QUOTED), limit=5) == [record]
    assert await adapter.list_cleaner_passes(scope=Scope(INJECTION), limit=5) == []

    review = RecordReview(
        target_id=fact.id,
        scope=Scope(QUOTED),
        reviewed_at=EPOCH,
        next_review_due_at=EPOCH + timedelta(days=1),
        outcome="judged",
        band="keep",
        grades=("a",),
        rubric_version="v1",
    )
    await adapter.append_review(review)
    assert await adapter.list_reviews_for(target_id=fact.id) == [review]


async def test_session_paths(adapter: LanceDBAdapter) -> None:
    session = Session(
        id=uuid.uuid4(),
        scope=QUOTED,
        started_at=EPOCH,
        ended_at=EPOCH + timedelta(hours=1),
        content="it's a session",
    )
    await adapter.append_session(session)

    assert await adapter.latest_session_end(scope=QUOTED) == session.ended_at
    assert await adapter.latest_session_end(scope=INJECTION) is None
    found = await adapter.search_sessions(
        scopes=[QUOTED, INJECTION],
        query=None,
        since=EPOCH,
        until=EPOCH + timedelta(days=1),
        limit=5,
    )
    assert [s.id for s in found] == [session.id]
    assert (
        await adapter.search_sessions(
            scopes=[INJECTION, LONE], query=None, since=None, until=None, limit=5
        )
        == []
    )
    assert (
        await adapter.search_sessions(
            scopes=[INJECTION], query="session", since=None, until=None, limit=5
        )
        == []
    )


async def test_record_filters_with_quoted_scopes(adapter: LanceDBAdapter) -> None:
    from tabularium.storage.types import Filters

    fact = _fact(scope="proj-a", content="needle text")
    await adapter.upsert(fact, _vec())
    filters = Filters(
        scopes=[Scope("proj-a"), Scope(QUOTED)],
        validity=frozenset({"valid"}),
        lifecycle_in=frozenset({"active"}),
    )
    miss = Filters(
        scopes=[Scope(INJECTION), Scope(LONE)],
        validity=frozenset({"valid"}),
        lifecycle_in=frozenset({"active"}),
    )

    hits = await adapter.vector_search(_vec(), filters=filters, k=5)
    assert [h.record.id for h in hits] == [fact.id]
    assert await adapter.vector_search(_vec(), filters=miss, k=5) == []
    text_hits = await adapter.text_search("needle", filters=filters, k=5)
    assert [h.record.id for h in text_hits] == [fact.id]
    assert await adapter.text_search("needle", filters=miss, k=5) == []

    key = content_key("needle text")
    found = await adapter.find_by_content_key(key, [Scope("proj-a"), Scope(QUOTED)])
    assert [r.id for r in found] == [fact.id]
    assert await adapter.find_by_content_key(key, [Scope(INJECTION)]) == []
    assert await adapter.find_by_content_key(QUOTED, [Scope("proj-a")]) == []
    assert await adapter.find_by_content_key(INJECTION, [Scope("proj-a")]) == []


async def test_event_source_edge_ref_and_registry_paths(adapter: LanceDBAdapter) -> None:
    assert await adapter.query_events(source=QUOTED) == []
    assert await adapter.query_events(source=INJECTION) == []
    assert await adapter.get_unsupersede(QUOTED) is None
    assert await adapter.get_unsupersede(INJECTION) is None
    assert await adapter.verdicts_by_effect(QUOTED) == []
    assert await adapter.verdicts_by_effect(INJECTION) == []
    assert await adapter.verdicts_by_bounce(INJECTION) == []

    row = _verdict(candidate_id=None, created_at=EPOCH, bounce_id=QUOTED, effect_ref=QUOTED)
    await adapter.append_verdicts([row])
    assert [v.id for v in await adapter.verdicts_by_bounce(QUOTED)] == [row.id]
    assert [v.id for v in await adapter.verdicts_by_effect(QUOTED)] == [row.id]
    assert await adapter.verdicts_by_effect(INJECTION) == []

    await adapter.create_role(QUOTED)
    assert QUOTED in await adapter.list_roles()
    await adapter.delete_role(QUOTED)
    assert QUOTED not in await adapter.list_roles()


async def test_backfill_with_quoted_model_and_hash(tmp_path: Path) -> None:
    store = str(tmp_path / "bf")
    first = await LanceDBAdapter.open(store, embedding_model=QUOTED, dim=4)
    fact = _fact(content="backfill me")
    await first.upsert(fact, _vec())
    assert await first.backfill_targets() == []

    second = await LanceDBAdapter.open(store, embedding_model="o'other", dim=4)
    targets = await second.backfill_targets()
    assert [t.id for t in targets] == [fact.id]

    good = content_hash("backfill me")
    assert (
        await second.backfill(fact.id, _vec(), embedding_model="o'other", expected_hash=INJECTION)
        is False
    )
    assert (
        await second.backfill(fact.id, _vec(), embedding_model="o'other", expected_hash=QUOTED)
        is False
    )
    assert await second.backfill_targets() != []
    assert (
        await second.backfill(fact.id, _vec(), embedding_model="o'other", expected_hash=good)
        is True
    )
    assert await second.backfill_targets() == []


async def test_burial_and_restore_paths_still_work(tmp_path: Path) -> None:
    store = str(tmp_path / "burial")
    adapter = await LanceDBAdapter.open(store, embedding_model=QUOTED, dim=4)
    fact = _fact()
    await adapter.upsert(fact, _vec())
    await adapter.bury(fact.id, "gone")
    assert await adapter.get_tombstone(fact.id) is not None
    assert await adapter.verdicts_for_record(new_id()) == []

    reopened = await LanceDBAdapter.open(store, embedding_model=QUOTED, dim=4)
    assert fact.id in reopened._buried_ids


async def test_find_by_content_key_with_no_scopes_is_empty_on_both_adapters(
    adapter: LanceDBAdapter,
) -> None:
    fact = _fact(content="needle text")
    memory = MemoryAdapter(embedding_model="test-model", dim=4)
    for target in (adapter, memory):
        await target.upsert(fact, _vec())

    key = content_key("needle text")
    assert await adapter.find_by_content_key(key, []) == []
    assert await memory.find_by_content_key(key, []) == []
