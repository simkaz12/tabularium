"""Tests for scripts/eval_v1v2/pmv2/v1_lane.py.

The hermetic classes use a fake engine; ``TestLiveStore`` needs Ollama and is integration-marked.
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import pytest

from tabularium.domain import HypomnemaId, Kind
from tabularium.results import HypomnemaRef, RecallResult, SaveResult

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from eval_v1v2.pmv2.rows import HitRow, Note, PersonaQuestion, WriteRow
from eval_v1v2.pmv2.v1_lane import (
    group_by_persona,
    note_content,
    persona_scope,
    recall_questions,
    record_id_of,
    run_persona,
    write_notes,
)
from eval_v1v2.v1_engine import assert_temp_path, open_v1
from eval_v1v2.v1_search import DegradedRecall


def make_notes() -> list[Note]:
    texts = [
        ("n1", "Likes hiking", "The user hikes every weekend in the Alps."),
        ("n2", "Weekend hiker", "The user goes hiking each weekend in the Alps."),
        ("n3", "Works in finance", "The user works as an accountant at a bank."),
        ("n4", "Forget request", "The user asked to forget the detail about their commute."),
    ]

    return [
        Note(note_id=note_id, persona_id=7, chunk_id="7-0", headline=headline, claim=claim)
        for note_id, headline, claim in texts
    ]


def make_questions() -> list[PersonaQuestion]:
    return [
        PersonaQuestion(
            question_id=f"7-{index}",
            persona_id=7,
            query=query,
            choices=["a", "b", "c", "d"],
            gold="A",
            forget=False,
            pref_type="ask_to_forget",
        )
        for index, query in enumerate(["Where does the user hike?", "What is their job?"])
    ]


class FakeEngine:
    def __init__(self, legs: list[str] | None = None) -> None:
        self.saved: list[tuple[str, str]] = []
        self.recalled: list[tuple[str, str, int]] = []
        self.legs = legs if legs is not None else ["vector", "text"]

    async def save(self, content: str, *, scope: str, kind: Kind = "fact") -> SaveResult:
        self.saved.append((scope, content))
        ref = HypomnemaRef(id=HypomnemaId(uuid.uuid4()), kind="fact")

        return SaveResult(outcome="saved", hypomnema=ref)

    async def recall(self, query: str, *, scope: str, limit: int) -> RecallResult:
        self.recalled.append((query, scope, limit))

        return RecallResult(hits=[], legs_used=self.legs)


class TestWrite:
    async def test_every_note_gets_a_write_row_in_order(self) -> None:
        engine = FakeEngine()
        notes = make_notes()

        rows = await write_notes(engine, 7, notes)

        assert [row.note_id for row in rows] == ["n1", "n2", "n3", "n4"]
        assert {row.status for row in rows} == {"saved"}
        assert all(row.system == "v1" and row.record_id for row in rows)
        assert [scope for scope, _ in engine.saved] == ["pmv2-7"] * 4

    def test_content_is_headline_newline_claim(self) -> None:
        note = make_notes()[0]

        assert note_content(note) == "Likes hiking\nThe user hikes every weekend in the Alps."

    def test_scope_names_the_persona(self) -> None:
        assert persona_scope(39) == "pmv2-39"

    def test_bounce_has_no_record_id(self) -> None:
        result = SaveResult.model_construct(outcome="bounced", hypomnema=None, reinforced_id=None)

        assert record_id_of(result) is None

    def test_reinforced_points_at_the_candidate(self) -> None:
        target = uuid.uuid4()
        result = SaveResult.model_construct(
            outcome="reinforced", hypomnema=None, reinforced_id=target
        )

        assert record_id_of(result) == str(target)

    def test_group_by_persona_keeps_order(self) -> None:
        grouped = group_by_persona(make_notes())

        assert list(grouped) == [7]
        assert [note.note_id for note in grouped[7]] == ["n1", "n2", "n3", "n4"]


class TestRecall:
    async def test_recall_uses_limit_ten_and_the_persona_scope(self) -> None:
        engine = FakeEngine()

        await recall_questions(engine, 7, make_questions())

        assert engine.recalled == [
            ("Where does the user hike?", "pmv2-7", 10),
            ("What is their job?", "pmv2-7", 10),
        ]

    async def test_degraded_recall_stops_the_lane(self) -> None:
        engine = FakeEngine(legs=["text"])

        with pytest.raises(DegradedRecall):
            await recall_questions(engine, 7, make_questions())


@pytest.mark.integration
class TestLiveStore:
    async def test_rows_for_every_note_and_hits_stay_in_scope(self) -> None:
        harness = await open_v1(use_assessor=False)

        assert assert_temp_path(harness.dir) == harness.dir

        other = Note(
            note_id="x1",
            persona_id=8,
            chunk_id="8-0",
            headline="Other persona",
            claim="Persona eight hikes in the Alps every weekend.",
        )
        await run_persona(harness, 8, [other], [])
        writes, hits = await run_persona(harness, 7, make_notes(), make_questions())

        assert [row.note_id for row in writes] == ["n1", "n2", "n3", "n4"]
        assert all(isinstance(row, WriteRow) for row in writes)
        assert hits
        assert all(isinstance(row, HitRow) for row in hits)

        scoped = await harness.engine.recall("hiking in the Alps", scope="pmv2-7", limit=10)

        assert scoped.hits
        assert all(hit.scope == "pmv2-7" for hit in scoped.hits)
        assert "Persona eight" not in " ".join(hit.content for hit in scoped.hits)
