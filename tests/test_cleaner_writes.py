"""``engine/cleaner_writes.py``: the Cleaner's own save of a trimmed or merged text.

Hermetic: ``MemoryAdapter``, ``FakeEmbedder`` (a distinct text is an orthogonal vector, the same
text is the same vector) and a counting Assessor. Nothing here reaches a network or a real judge.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine.cleaner_writes import (
    apply_merge,
    apply_trim,
    merge_carry,
    save_cleaner_fact,
    trim_after_shelve,
)
from tabularium.engine.dispositions import set_shelved, supersede, unsupersede
from tabularium.engine.facts import save_internal as _real_save_internal
from tabularium.results import SaveResult
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import VerdictRow
from tests.support.doubles import FakeEmbedder, TestClock

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_DIM = 16


def _adapter() -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic-embed-text", dim=_DIM)


def _fact(content: str, *, scope: str = "proj-a", purpose: str | None = None) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle="active",
        validity="valid",
        purpose=purpose,
        created_at=_NOW,
        updated_at=_NOW,
    )


async def _store(adapter: MemoryAdapter, embedder: FakeEmbedder, fact: Fact) -> Fact:
    await adapter.create_scope(fact.scope)
    await adapter.upsert(fact, await embedder.embed(fact.content))
    return fact


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=_DIM)


class TestTrimAfterShelve:
    async def test_keep_saved_after_the_source_is_shelved(self, embedder: FakeEmbedder) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(
            adapter, embedder, _fact("stale rambling plus one rule", purpose="gotcha")
        )
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        result = await trim_after_shelve(
            record=source,
            keep="the one rule",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "trimmed"
        assert result.new_id is not None
        saved = await adapter.get_by_id(result.new_id)
        assert isinstance(saved, Fact)
        assert saved.content == "the one rule"
        assert saved.scope == source.scope
        assert saved.purpose == "gotcha"
        assert saved.lifecycle == "active"
        assert saved.source == f"cleaner:trim:{source.id}"

    async def test_keep_saved_while_the_source_is_active_bounces_against_it(
        self, embedder: FakeEmbedder
    ) -> None:
        adapter = _adapter()
        source = await _store(adapter, embedder, _fact("the one rule"))

        result = await save_cleaner_fact(
            content="the one rule",
            scope=source.scope,
            purpose=None,
            source="cleaner:trim:x",
            adapter=adapter,
            embedder=embedder,
            clock=TestClock(_NOW),
            cfg=EngineConfig(),
            lock=asyncio.Lock(),
        )

        assert result.outcome == "already_saved"

    async def test_keep_identical_to_another_record_is_covered(
        self, embedder: FakeEmbedder
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("long stale record"))
        await _store(adapter, embedder, _fact("the one rule"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        result = await trim_after_shelve(
            record=source,
            keep="the one rule",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "covered"
        assert result.new_id is None
        stored = await adapter.get_by_id(source.id)
        assert stored.lifecycle == "shelved"

    async def test_a_raising_save_unshelves_the_source(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("long stale record"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        async def boom(**_: Any) -> Any:
            raise RuntimeError("disk full")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)

        result = await trim_after_shelve(
            record=source,
            keep="the one rule",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        assert result.error == "disk full"
        stored = await adapter.get_by_id(source.id)
        assert stored.lifecycle == "active"


def _verdict_row(
    *, candidate_id: Any, action: str, new_record_id: str | None, effect_ref: str | None
) -> VerdictRow:
    return VerdictRow(
        id=str(new_id()),
        created_at=_NOW,
        caller="cleaner_prepass",
        method="relevance",
        bounce_id=None,
        candidate_id=str(candidate_id),
        probe_rank=None,
        similarity=None,
        pick=None,
        odds_json=None,
        confidence=None,
        reason_class=None,
        unavailable_reason=None,
        unavailable_detail=None,
        model=None,
        wording_hash=None,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        request_id=None,
        status="acted",
        action=action,
        actor="panel",
        matched=True,
        effect_ref=effect_ref,
        new_record_id=new_record_id,
        incoming_text=None,
        override=None,
        override_at=None,
        stamped_at=_NOW,
    )


class TestTrimBounceAndRevertHygiene:
    async def test_a_bounced_keep_reverts_the_shelve(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("long stale record"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        async def bounced(**_: Any) -> Any:
            return SaveResult(outcome="bounced", reason="near a distinct record")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", bounced)

        result = await trim_after_shelve(
            record=source,
            keep="the one rule",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        assert result.error == "near a distinct record"
        assert (await adapter.get_by_id(source.id)).lifecycle == "active"

    async def test_a_reverted_trim_of_an_archived_record_restores_archived_and_stamps_failed(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        archived = _fact("long stale record").model_copy(update={"lifecycle": "archived"})
        source = await _store(adapter, embedder, archived)
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )
        row = _verdict_row(
            candidate_id=source.id,
            action="panel_shelve",
            new_record_id=str(source.id),
            effect_ref=None,
        )
        await adapter.append_verdicts([row])

        async def boom(**_: Any) -> Any:
            raise RuntimeError("disk full")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)

        result = await trim_after_shelve(
            record=source,
            keep="the one rule",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        assert (await adapter.get_by_id(source.id)).lifecycle == "archived"
        events = await adapter.query_events(target_id=source.id)
        assert [e.source for e in events] == ["shelved", "shelve_reverted"]
        stored_row = (await adapter.verdicts_for_record(source.id))[0]
        assert stored_row.status == "failed"
        assert stored_row.override is None

    async def test_a_merge_revert_stamps_the_edge_row_failed_with_no_override(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )
        edge_id = (await adapter.get_by_id(older.id)).supersession_edge_id
        assert edge_id is not None
        row = _verdict_row(
            candidate_id=older.id,
            action="supersede",
            new_record_id=str(newer.id),
            effect_ref=edge_id,
        )
        await adapter.append_verdicts([row])

        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="the part O still holds",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        stored_row = (await adapter.verdicts_for_record(older.id))[0]
        assert stored_row.status == "failed"
        assert stored_row.override is None
        all_rows = list(adapter._verdicts.values())
        live_rows = [row for row in all_rows if row.status != "failed"]
        assert [row for row in live_rows if row.override is not None] == []


class TestFailedUndoReportsRevertFailed:
    async def test_a_storage_failure_during_the_undo_is_not_reported_as_reverted(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )

        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        async def failing_restore(*_: Any, **__: Any) -> None:
            raise RuntimeError("restore failed")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        monkeypatch.setattr(adapter, "restore", failing_restore)

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="the part O still holds",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "revert_failed"
        assert result.error is not None
        assert "revert failed" in result.error
        assert (await adapter.get_by_id(older.id)).superseded_by == newer.id


class TestFailedRevertTransitionLeavesNoRevertEvent:
    async def test_a_failed_transition_keeps_the_owner_round_trip_protected(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tabularium.engine.revisit_pool import is_protected

        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("long stale record"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        async def boom(**_: Any) -> Any:
            raise RuntimeError("disk full")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        real_transition = adapter.transition
        calls = {"n": 0}

        async def flaky_transition(*args: Any, **kwargs: Any) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transition failed")
            await real_transition(*args, **kwargs)

        monkeypatch.setattr(adapter, "transition", flaky_transition)

        with pytest.raises(RuntimeError, match="transition failed"):
            await trim_after_shelve(
                record=source,
                keep="the one rule",
                adapter=adapter,
                embedder=embedder,
                clock=clock,
                cfg=EngineConfig(),
                lock=lock,
            )

        events = await adapter.query_events(target_id=source.id)
        assert [e.source for e in events] == ["shelved"]
        await set_shelved(
            id=source.id, shelved=False, reason="owner", adapter=adapter, clock=clock, lock=lock
        )
        stored = await adapter.get_by_id(source.id)
        assert isinstance(stored, Fact)
        assert is_protected(stored, await adapter.query_events(target_id=source.id))


class TestCancellationRollsBack:
    async def test_cancelled_trim_save_unshelves_the_source_then_propagates(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("long stale record"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        async def cancelled(**_: Any) -> Any:
            raise asyncio.CancelledError

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", cancelled)

        with pytest.raises(asyncio.CancelledError):
            await trim_after_shelve(
                record=source,
                keep="the one rule",
                adapter=adapter,
                embedder=embedder,
                clock=clock,
                cfg=EngineConfig(),
                lock=lock,
            )

        assert (await adapter.get_by_id(source.id)).lifecycle == "active"

    async def test_cancelled_merge_save_undoes_the_edge_then_propagates(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )

        async def cancelled(**_: Any) -> Any:
            raise asyncio.CancelledError

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", cancelled)

        with pytest.raises(asyncio.CancelledError):
            await merge_carry(
                newer=newer,
                older=older,
                carry="the part O still holds",
                adapter=adapter,
                embedder=embedder,
                clock=clock,
                cfg=EngineConfig(),
                lock=lock,
            )

        stored = await adapter.get_by_id(older.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None


class TestSaveCallShape:
    async def test_the_save_passes_no_assessor_and_never_holds_the_lock(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        lock = asyncio.Lock()
        seen: list[dict[str, Any]] = []
        real = _real_save_internal

        async def spy(**kwargs: Any) -> Any:
            seen.append({"assessor": kwargs["assessor"], "locked": kwargs["lock"].locked()})
            return await real(**kwargs)

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", spy)
        await adapter.create_scope(Scope("proj-a"))

        await save_cleaner_fact(
            content="a rule",
            scope=Scope("proj-a"),
            purpose=None,
            source="cleaner:trim:x",
            adapter=adapter,
            embedder=embedder,
            clock=TestClock(_NOW),
            cfg=EngineConfig(),
            lock=lock,
        )

        assert seen == [{"assessor": None, "locked": False}]


class TestMergeCarry:
    async def test_merged_record_supersedes_the_newer_one(self, embedder: FakeEmbedder) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record", purpose="gotcha"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="Still true: keep the lid on.",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "merged"
        assert result.new_id is not None
        merged = await adapter.get_by_id(result.new_id)
        assert isinstance(merged, Fact)
        assert merged.content == "newer record\n\nStill true: keep the lid on."
        assert merged.source == f"cleaner:merge:{older.id}"
        assert merged.purpose == "gotcha"
        stored_newer = await adapter.get_by_id(newer.id)
        assert stored_newer.superseded_by == merged.id
        assert stored_newer.supersession_reason == "elaborated"
        assert stored_newer.supersession_note == f"merged: carries what {older.id} still held"
        stored_older = await adapter.get_by_id(older.id)
        assert stored_older.superseded_by == newer.id


class _Voter:
    def __init__(self, keep: str | None, votes_shelve: bool = True) -> None:
        self.keep = keep
        self.votes_shelve = votes_shelve


class TestApplyResultsCarryTheEntries:
    async def test_apply_trim_returns_the_pair_and_leaves_the_refusal_unset(
        self, embedder: FakeEmbedder
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        source = await _store(adapter, embedder, _fact("stale rambling plus one rule"))
        await set_shelved(
            id=source.id, shelved=True, reason="r", adapter=adapter, clock=clock, lock=lock
        )

        applied = await apply_trim(
            record=source,
            verdicts=[_Voter("the one rule")],
            embedder=embedder,
            require_scope=None,
            adapter=adapter,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
            dry_run=False,
        )

        assert applied.refusal is None
        assert applied.pair is not None
        assert applied.pair.old_id == source.id
        assert applied.pair.new_id is not None

    async def test_apply_trim_dry_run_returns_the_pair_without_a_new_id(
        self, embedder: FakeEmbedder
    ) -> None:
        source = _fact("stale rambling plus one rule")

        applied = await apply_trim(
            record=source,
            verdicts=[_Voter("the one rule")],
            embedder=None,
            require_scope=None,
            adapter=_adapter(),
            clock=TestClock(_NOW),
            cfg=EngineConfig(),
            lock=asyncio.Lock(),
            dry_run=True,
        )

        assert applied.refusal is None
        assert applied.pair is not None
        assert applied.pair.new_id is None

    async def test_apply_trim_without_a_keep_text_records_nothing(
        self, embedder: FakeEmbedder
    ) -> None:
        applied = await apply_trim(
            record=_fact("x"),
            verdicts=[_Voter(None)],
            embedder=embedder,
            require_scope=None,
            adapter=_adapter(),
            clock=TestClock(_NOW),
            cfg=EngineConfig(),
            lock=asyncio.Lock(),
            dry_run=False,
        )

        assert applied.pair is None
        assert applied.refusal is None

    async def test_apply_merge_dry_run_returns_the_triple_and_the_edge_stands(
        self, embedder: FakeEmbedder
    ) -> None:
        older = _fact("older record")
        newer = _fact("newer record")

        applied = await apply_merge(
            newer=newer,
            older=older,
            carry="c",
            embedder=embedder,
            require_scope=None,
            adapter=_adapter(),
            clock=TestClock(_NOW),
            cfg=EngineConfig(),
            lock=asyncio.Lock(),
            dry_run=True,
        )

        assert applied.merged is not None
        assert applied.merged.merged_id is None
        assert applied.reverted is None
        assert applied.revert_failed is None
        assert applied.edge_stands is True

    async def test_apply_merge_reports_a_reverted_edge_as_not_standing(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )

        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)

        applied = await apply_merge(
            newer=newer,
            older=older,
            carry="c",
            embedder=embedder,
            require_scope=None,
            adapter=adapter,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
            dry_run=False,
        )

        assert applied.merged is None
        assert applied.revert_failed is None
        assert applied.reverted is not None
        assert applied.reverted.error == "no disk"
        assert applied.edge_stands is False

    async def test_apply_merge_reports_a_failed_undo_as_standing(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )

        async def boom(**_: Any) -> Any:
            raise RuntimeError("no disk")

        async def failing_restore(*_: Any, **__: Any) -> None:
            raise RuntimeError("restore failed")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", boom)
        monkeypatch.setattr(adapter, "restore", failing_restore)

        applied = await apply_merge(
            newer=newer,
            older=older,
            carry="c",
            embedder=embedder,
            require_scope=None,
            adapter=adapter,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
            dry_run=False,
        )

        assert applied.reverted is None
        assert applied.revert_failed is not None
        assert "revert failed" in applied.revert_failed.error
        assert applied.edge_stands is True


class TestMergeRollbackOnlyUndoesItsOwnEdge:
    async def _drawn_edge(
        self, adapter: MemoryAdapter, embedder: FakeEmbedder, clock: TestClock, lock: asyncio.Lock
    ) -> tuple[Fact, Fact]:
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )
        return older, newer

    async def test_a_replacement_drawn_while_the_save_ran_is_left_alone(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older, newer = await self._drawn_edge(adapter, embedder, clock, lock)
        other = await _store(adapter, embedder, _fact("a different replacement"))

        async def interleaved(**_: Any) -> Any:
            await unsupersede(
                id=older.id, reason="owner undo", adapter=adapter, clock=clock, lock=lock
            )
            await supersede(
                successor_id=other.id,
                predecessor_ids=[older.id],
                reason="correction",
                note=None,
                adapter=adapter,
                clock=clock,
                lock=lock,
            )
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", interleaved)

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="the part O still holds",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        assert result.error is not None
        assert "nothing to undo" in result.error
        stored = await adapter.get_by_id(older.id)
        assert stored.superseded_by == other.id

    async def test_an_edge_undone_while_the_save_ran_is_not_an_error(
        self, embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter()
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older, newer = await self._drawn_edge(adapter, embedder, clock, lock)

        async def interleaved(**_: Any) -> Any:
            await unsupersede(
                id=older.id, reason="owner undo", adapter=adapter, clock=clock, lock=lock
            )
            raise RuntimeError("no disk")

        monkeypatch.setattr("tabularium.engine.cleaner_writes.save_internal", interleaved)

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="the part O still holds",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        stored = await adapter.get_by_id(older.id)
        assert stored.superseded_by is None


class _FailsOneRead(MemoryAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_next_read_of: Any = None

    async def get_by_id(self, id: Any) -> Any:
        if self.fail_next_read_of == id:
            self.fail_next_read_of = None
            raise RuntimeError("storage hiccup")
        return await super().get_by_id(id)


class TestFailureBeforeTheSaveStillRollsBack:
    async def test_a_failed_edge_read_undoes_the_edge(self, embedder: FakeEmbedder) -> None:
        adapter = _FailsOneRead(embedding_model="fake/nomic-embed-text", dim=_DIM)
        clock = TestClock(_NOW)
        lock = asyncio.Lock()
        older = await _store(adapter, embedder, _fact("older record"))
        newer = await _store(adapter, embedder, _fact("newer record"))
        await supersede(
            successor_id=newer.id,
            predecessor_ids=[older.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=clock,
            lock=lock,
        )
        adapter.fail_next_read_of = older.id

        result = await merge_carry(
            newer=newer,
            older=older,
            carry="the part O still holds",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=EngineConfig(),
            lock=lock,
        )

        assert result.status == "reverted"
        assert result.error == "storage hiccup"
        stored = await adapter.get_by_id(older.id)
        assert stored.validity == "valid"
        assert stored.superseded_by is None
