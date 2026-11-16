"""Tests for scripts/eval_v1v2/v1_engine.py, v1_save.py and v1_search.py.

Falsifiability: swapping the ``"flagged"`` and ``"wrong_target"`` branches of the bounce bucket in
``map_result`` turned ``test_bounce_on_existing_is_flagged`` and
``test_bounce_on_other_record_is_wrong_target`` red; dropping the ``bury`` of the pair's rows in
``replay_pair`` turned the integration test's base-count assertion red. Both reverted.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tabularium.domain import Decision, Fact, HypomnemaId
from tabularium.embedding import EmbedderUnavailable
from tabularium.results import (
    BounceExit,
    BounceNeighbor,
    FactBounce,
    HypomnemaRef,
    JudgedUnavailable,
    JudgedVerdict,
    RecallResult,
    SaveResult,
    WithheldReason,
)

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import v1_search  # noqa: E402
from eval_v1v2.rows import (  # noqa: E402
    CorpusRow,
    NoiseRow,
    PairRow,
    QuestionRow,
    SearchRow,
    write_jsonl,
)
from eval_v1v2.v1_engine import (  # noqa: E402
    EVAL_SCOPE,
    assert_temp_path,
    count_records,
    load_corpus,
    load_noise,
    open_v1,
    record_from_corpus,
    record_from_noise,
    snapshot_clock,
)
from eval_v1v2.v1_save import WARM_UP_TEXT, map_result, replay_pair, run_save  # noqa: E402
from eval_v1v2.v1_search import (  # noqa: E402
    DegradedRecall,
    require_both_legs,
    run_search,
    sample_known_items,
    self_check,
    zero_hit_calls,
)

sys.path.pop(0)

EXISTING = str(uuid.uuid4())
OTHER = str(uuid.uuid4())


def _hid(raw: str) -> HypomnemaId:
    return HypomnemaId(uuid.UUID(raw))


EXIT = BounceExit(action="reinforce", how="save(...)", when="same thing")


def _verdict(pick: str = "restatement") -> dict[str, JudgedVerdict | JudgedUnavailable]:
    return {"relation": JudgedVerdict(pick=pick, odds={pick: 1.0}, confidence=0.9)}


def _bounce(
    candidate: str,
    *,
    judged: dict[str, JudgedVerdict | JudgedUnavailable] | None = None,
    others: list[BounceNeighbor] | None = None,
    withheld: WithheldReason | None = None,
) -> SaveResult:
    bounce = FactBounce(
        candidate_id=_hid(candidate),
        candidate_content="x",
        candidate_scope=EVAL_SCOPE,
        similarity=0.9,
        verdict="near_dup",
        other_candidates=others or [],
        judged=judged,
        withheld=withheld,
        bounce_id="b1",
        exits=(EXIT,),
    )

    return SaveResult(outcome="bounced", bounce=bounce)


def _ref(record_id: str) -> HypomnemaRef:
    return HypomnemaRef(id=_hid(record_id), kind="fact")


class TestOutcomeMapping:
    def test_saved_is_new(self) -> None:
        mapped = map_result(SaveResult(outcome="saved", hypomnema=_ref(OTHER)), EXISTING)

        assert (mapped.bucket, mapped.found_existing, mapped.raw_outcome) == ("new", False, "saved")

    def test_reconciled_is_new(self) -> None:
        mapped = map_result(SaveResult(outcome="reconciled", hypomnema=_ref(OTHER)), EXISTING)

        assert mapped.bucket == "new"

    def test_reinforced_existing_is_caught(self) -> None:
        mapped = map_result(
            SaveResult(outcome="reinforced", reinforced_id=_hid(EXISTING)), EXISTING
        )

        assert (mapped.bucket, mapped.found_existing) == ("caught", True)

    def test_merge_outcomes_carry_the_target_id(self) -> None:
        reinforced = map_result(
            SaveResult(outcome="reinforced", reinforced_id=_hid(OTHER)), EXISTING
        )
        revived = map_result(SaveResult(outcome="revived", reinforced_id=_hid(OTHER)), EXISTING)
        saved = map_result(SaveResult(outcome="already_saved", hypomnema=_ref(OTHER)), EXISTING)

        assert reinforced.raw_outcome == f"reinforced:{OTHER}"
        assert revived.raw_outcome == f"revived:{OTHER}"
        assert saved.raw_outcome == f"already_saved:{OTHER}"

    def test_reinforced_other_is_wrong_target(self) -> None:
        mapped = map_result(SaveResult(outcome="reinforced", reinforced_id=_hid(OTHER)), EXISTING)

        assert (mapped.bucket, mapped.found_existing) == ("wrong_target", False)

    def test_revived_existing_is_caught(self) -> None:
        mapped = map_result(SaveResult(outcome="revived", reinforced_id=_hid(EXISTING)), EXISTING)

        assert mapped.bucket == "caught"

    def test_already_saved_existing_is_caught(self) -> None:
        mapped = map_result(SaveResult(outcome="already_saved", hypomnema=_ref(EXISTING)), EXISTING)

        assert (mapped.bucket, mapped.found_existing) == ("caught", True)

    def test_already_saved_other_is_wrong_target(self) -> None:
        mapped = map_result(SaveResult(outcome="already_saved", hypomnema=_ref(OTHER)), EXISTING)

        assert mapped.bucket == "wrong_target"

    def test_bounce_on_existing_is_flagged(self) -> None:
        mapped = map_result(_bounce(EXISTING, judged=_verdict("restatement")), EXISTING)

        assert mapped.bucket == "flagged"
        assert mapped.found_existing is True
        assert mapped.relation_pick == "restatement"
        assert mapped.degraded is None
        assert mapped.raw_outcome == f"bounced:near_dup:{EXISTING}"

    def test_bounce_on_other_record_is_wrong_target(self) -> None:
        mapped = map_result(_bounce(OTHER, judged=_verdict()), EXISTING)

        assert (mapped.bucket, mapped.found_existing, mapped.relation_pick) == (
            "wrong_target",
            False,
            None,
        )

    def test_existing_among_other_candidates_is_found_but_wrong_target(self) -> None:
        neighbour = BounceNeighbor(
            candidate_id=_hid(EXISTING), similarity=0.85, judged=_verdict("replacement")
        )
        mapped = map_result(_bounce(OTHER, judged=_verdict(), others=[neighbour]), EXISTING)

        assert mapped.bucket == "wrong_target"
        assert mapped.found_existing is True
        assert mapped.relation_pick == "replacement"

    def test_unavailable_assessor_is_degraded(self) -> None:
        judged: dict[str, JudgedVerdict | JudgedUnavailable] = {
            "relation": JudgedUnavailable(reason="timeout")
        }
        mapped = map_result(_bounce(EXISTING, judged=judged), EXISTING)

        assert mapped.bucket == "flagged"
        assert mapped.relation_pick is None
        assert mapped.degraded == "assessor_unavailable:timeout"

    def test_unavailable_assessor_on_a_secondary_candidate_is_degraded(self) -> None:
        neighbour = BounceNeighbor(
            candidate_id=_hid(OTHER),
            similarity=0.8,
            judged={"relation": JudgedUnavailable(reason="timeout")},
        )
        mapped = map_result(_bounce(EXISTING, judged=_verdict(), others=[neighbour]), EXISTING)

        assert mapped.degraded == "assessor_unavailable:timeout"

    def test_unjudged_secondary_candidate_is_not_degraded(self) -> None:
        neighbour = BounceNeighbor(candidate_id=_hid(OTHER), similarity=0.8)
        mapped = map_result(_bounce(EXISTING, judged=_verdict(), others=[neighbour]), EXISTING)

        assert mapped.degraded is None

    def test_absent_assessor_is_degraded(self) -> None:
        mapped = map_result(_bounce(EXISTING), EXISTING)

        assert mapped.degraded == "assessor_absent"

    def test_switch_off_withheld_is_not_degraded(self) -> None:
        mapped = map_result(_bounce(EXISTING, judged=_verdict(), withheld="switch_off"), EXISTING)

        assert mapped.degraded is None

    def test_candidate_changed_is_degraded(self) -> None:
        mapped = map_result(
            _bounce(EXISTING, judged=_verdict(), withheld="candidate_changed"), EXISTING
        )

        assert mapped.degraded == "withheld:candidate_changed"

    def test_embedder_down_save_is_degraded(self) -> None:
        mapped = map_result(
            SaveResult(outcome="saved", hypomnema=_ref(OTHER), embedded=False), EXISTING
        )

        assert mapped.degraded == "embedder_unavailable"


class TestStorePath:
    def test_temp_path_accepted(self, tmp_path: Path) -> None:
        assert assert_temp_path(tmp_path / "store") == (tmp_path / "store").resolve()

    def test_media_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="never under"):
            assert_temp_path("/var/lib/apps/tabularium")

    def test_home_path_rejected(self) -> None:
        with pytest.raises(ValueError):
            assert_temp_path(Path.home() / "store")


class TestRowBuilding:
    def test_noise_fact_keeps_counters_and_scope(self) -> None:
        row = NoiseRow(
            id=str(uuid.uuid4()),
            kind="fact",
            content="n",
            vector=[0.1] * 768,
            recall_count=4,
            created_at="2026-07-31T13:54:10.708933+00:00",
            lifecycle="hot",
        )
        record = record_from_noise(row)

        assert record.type == "fact"
        assert record.lifecycle == "hot"
        assert record.recall_count == 4
        assert record.scope == EVAL_SCOPE

    def test_noise_decision_is_in_force(self) -> None:
        row = NoiseRow(
            id=str(uuid.uuid4()),
            kind="decision",
            content="n",
            vector=[0.1] * 768,
            recall_count=0,
            created_at="2026-07-31T13:54:10",
            lifecycle="superseded",
        )
        record = record_from_noise(row)

        assert record.lifecycle == "in_force"
        assert record.created_at.tzinfo is not None

    def test_corpus_row_keeps_original_scope(self) -> None:
        row = CorpusRow(
            id=str(uuid.uuid4()),
            kind="fact",
            content="c",
            scope="tabularium",
            lifecycle="active",
            vector=[0.1] * 768,
            recall_count=2,
            created_at="2026-07-31T13:54:10+00:00",
        )

        assert record_from_corpus(row).scope == "tabularium"


def _corpus_row(**extra: object) -> CorpusRow:
    return CorpusRow(
        id=str(uuid.uuid4()),
        kind="fact",
        content="c",
        scope="tabularium",
        lifecycle="active",
        vector=[0.1] * 768,
        recall_count=2,
        created_at="2026-07-31T13:54:10+00:00",
        **extra,
    )


class TestCarriedFields:
    def test_fact_keeps_feedback_and_last_reference(self) -> None:
        row = _corpus_row(
            feedback_score=0.75,
            last_referenced_at="2026-08-05T09:00:00+00:00",
            updated_at="2026-08-06T09:00:00+00:00",
            hot_rejection_count=2,
            embedding_model="other/model",
        )
        record = record_from_corpus(row)

        assert isinstance(record, Fact)
        assert record.feedback_score == 0.75
        assert record.last_referenced_at == datetime(2026, 8, 5, 9, tzinfo=UTC)
        assert record.updated_at == datetime(2026, 8, 6, 9, tzinfo=UTC)
        assert record.hot_rejection_count == 2
        assert record.embedding_model == "other/model"

    def test_row_without_extras_keeps_defaults(self) -> None:
        record = record_from_corpus(_corpus_row())

        assert isinstance(record, Fact)
        assert record.feedback_score == 0.0
        assert record.last_referenced_at is None
        assert record.updated_at == record.created_at

    def test_null_extras_fall_back_to_defaults(self) -> None:
        record = record_from_corpus(_corpus_row(last_referenced_at=None, supersession_chain=None))

        assert record.last_referenced_at is None

    def test_decision_drops_fact_only_extras(self) -> None:
        row = CorpusRow(
            id=str(uuid.uuid4()),
            kind="decision",
            content="d",
            scope="tabularium",
            lifecycle="in_force",
            vector=[0.1] * 768,
            recall_count=1,
            created_at="2026-07-31T13:54:10+00:00",
            **{
                "feedback_score": 0.5,
                "hot_rejection_count": 1,
                "source": "agent",
                "updated_at": "2026-08-01T00:00:00+00:00",
            },
        )
        record = record_from_corpus(row)

        assert isinstance(record, Decision)
        assert record.source == "agent"
        assert record.updated_at == datetime(2026, 8, 1, tzinfo=UTC)


class TestSnapshotClock:
    def test_pins_to_latest_timestamp_plus_one_second(self) -> None:
        rows = [
            _corpus_row(updated_at="2026-08-01T00:00:00+00:00"),
            _corpus_row(
                updated_at="2026-08-02T00:00:00+00:00",
                last_referenced_at="2026-09-01T12:00:00+00:00",
            ),
        ]

        assert snapshot_clock(rows).now() == datetime(2026, 9, 1, 12, 0, 1, tzinfo=UTC)

    def test_created_at_counts(self) -> None:
        assert snapshot_clock([_corpus_row()]).now() == datetime(
            2026, 7, 31, 13, 54, 11, tzinfo=UTC
        )

    def test_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no timestamps"):
            snapshot_clock([])


@pytest.mark.integration
class TestLiveEngine:
    async def test_identical_is_not_new_unrelated_is_new_and_base_restored(
        self, tmp_path: Path
    ) -> None:
        assert os.environ.get("TABULARIUM_READ_ONLY") is None

        harness = await open_v1(tmp_path / "store", use_assessor=False)
        noise = [
            NoiseRow(
                id=str(uuid.uuid4()),
                kind="fact",
                content=text,
                vector=await harness.embedder.embed(text),
                recall_count=0,
                created_at="2026-07-31T13:54:10+00:00",
                lifecycle="active",
            )
            for text in (
                "Postgres vacuum runs nightly on the reporting replica.",
                "The Nuxt dev server listens on port 3000 by default.",
                "Release tags follow semantic versioning with a v prefix.",
            )
        ]

        await load_noise(harness, noise, lambda _: None)

        base = await count_records(harness)
        pairs = [
            PairRow(
                pair_id="same",
                source="golden",
                kind="fact",
                existing="The staging database password rotates every ninety days.",
                incoming="The staging database password rotates every ninety days.",
                similarity=None,
                old_label=None,
            ),
            PairRow(
                pair_id="far",
                source="golden",
                kind="fact",
                existing="The staging database password rotates every ninety days.",
                incoming="Sourdough starters need feeding twice a day in summer heat.",
                similarity=None,
                old_label=None,
            ),
        ]
        out = tmp_path / "save-v1.jsonl"
        rows = await run_save(harness, pairs, out)
        by_id = {row.pair_id: row for row in rows}

        assert by_id["same"].bucket != "new"
        assert by_id["same"].found_existing is True
        assert by_id["far"].bucket == "new"
        assert await count_records(harness) == base
        assert len(out.read_text().splitlines()) == 2

        resumed = await run_save(harness, pairs, out, frozenset(row.pair_id for row in rows))

        assert resumed == []

    async def test_run_save_warms_the_save_path_before_the_first_pair(self, tmp_path: Path) -> None:
        harness = await open_v1(tmp_path / "store", use_assessor=False)

        await load_noise(harness, [], lambda _: None)

        base = await count_records(harness)
        seen: list[str] = []
        real_save = harness.engine.save

        async def spy(text: str, **kwargs: Any) -> Any:
            seen.append(text)

            return await real_save(text, **kwargs)

        harness.engine.save = spy  # type: ignore[method-assign, assignment]
        pair = PairRow(
            pair_id="p",
            source="golden",
            kind="fact",
            existing="Alpha text about caches.",
            incoming="Beta text about sourdough.",
            similarity=None,
            old_label=None,
        )

        await run_save(harness, [pair], tmp_path / "out.jsonl")

        assert seen == [WARM_UP_TEXT, pair.incoming]
        assert await count_records(harness) == base

    async def test_replay_refuses_a_dirty_base(self, tmp_path: Path) -> None:
        harness = await open_v1(tmp_path / "store", use_assessor=False)
        pair = PairRow(
            pair_id="p",
            source="golden",
            kind="fact",
            existing="Alpha text.",
            incoming="Beta text.",
            similarity=None,
            old_label=None,
        )

        with pytest.raises(AssertionError, match="base is 5"):
            await replay_pair(harness, pair, 5)

    async def test_search_ranks_and_times_each_question(self, tmp_path: Path) -> None:
        harness = await open_v1(tmp_path / "store", use_assessor=False)
        texts = [
            "The deploy script restarts the nginx container after each release.",
            "Sourdough starters need feeding twice a day in summer heat.",
        ]
        corpus = [
            CorpusRow(
                id=str(uuid.uuid4()),
                kind="fact",
                content=text,
                scope="global",
                lifecycle="active",
                vector=await harness.embedder.embed(text),
                recall_count=0,
                created_at="2026-07-31T13:54:10+00:00",
            )
            for text in texts
        ]

        await load_corpus(harness, corpus, lambda _: None)

        questions = [
            QuestionRow(
                question_id="q1",
                query="how do I feed sourdough starter",
                scope="global",
                origin="mined",
            )
        ]
        rows = await run_search(harness, questions)

        assert rows[0].record_id == corpus[1].id
        assert [row.rank for row in rows] == [1, 2]
        assert len({row.elapsed_ms for row in rows}) == 1
        assert json.loads(rows[0].model_dump_json())["arm"] == "v1"


def _recall_result(**fields: Any) -> RecallResult:
    return RecallResult(hits=[], **fields)


class TestRecallLegs:
    def test_both_legs_pass(self) -> None:
        require_both_legs(_recall_result(legs_used=["vector", "text"]), "q")

    def test_degraded_recall_is_refused(self) -> None:
        degraded = _recall_result(degraded=True, reason="ollama down", legs_used=["text"])

        with pytest.raises(DegradedRecall, match="ollama down"):
            require_both_legs(degraded, "q")

    def test_one_leg_without_the_flag_is_refused(self) -> None:
        with pytest.raises(DegradedRecall):
            require_both_legs(_recall_result(legs_used=["vector"]), "q")

    async def test_run_search_stops_on_a_degraded_timed_recall(self) -> None:
        calls: list[str] = []

        async def recall(query: str, **_: Any) -> RecallResult:
            calls.append(query)

            if len(calls) > 1:
                return _recall_result(degraded=True, reason="down", legs_used=["text"])

            return _recall_result(legs_used=["vector", "text"])

        harness: Any = type("Stub", (), {"engine": type("E", (), {"recall": staticmethod(recall)})})
        question = QuestionRow(question_id="q1", query="a", scope="global", origin="mined")

        with pytest.raises(DegradedRecall):
            await run_search(harness, [question])


class TestSelfCheckSample:
    def test_sample_is_seeded_facts_only_and_skips_duplicates(self) -> None:
        rows = [
            _corpus_row().model_copy(update={"content": f"fact number {n}"}) for n in range(30)
        ] + [
            CorpusRow(
                id=str(uuid.uuid4()),
                kind="decision",
                content="a decision",
                scope="tabularium",
                lifecycle="in_force",
                vector=[0.1] * 768,
                recall_count=0,
                created_at="2026-07-31T13:54:10+00:00",
            ),
            _corpus_row().model_copy(update={"content": "fact number 0"}),
        ]
        first = sample_known_items(rows, 10, 5)

        assert [row.id for row in first] == [row.id for row in sample_known_items(rows, 10, 5)]
        assert len(first) == 10
        assert all(row.kind == "fact" for row in first)
        assert all(row.content != "fact number 0" for row in first)


@pytest.mark.integration
class TestSearchIntegrity:
    async def test_exact_content_ranks_first_after_bulk_load_and_upkeep(
        self, tmp_path: Path
    ) -> None:
        harness = await open_v1(tmp_path / "store", use_assessor=False)
        subjects = ["tides", "kilns", "ferries", "bees", "glaciers", "looms", "comets", "taxes"]
        texts = [
            f"Note {n}: the {subjects[n % 8]} register number {n} is audited on day {n}."
            for n in range(64)
        ]
        corpus = [
            CorpusRow(
                id=str(uuid.uuid4()),
                kind="fact",
                content=text,
                scope="global",
                lifecycle="active",
                vector=await harness.embedder.embed(text),
                recall_count=0,
                created_at="2026-07-31T13:54:10+00:00",
            )
            for text in texts
        ]

        await load_corpus(harness, corpus, lambda _: None)
        await harness.adapter.upkeep()

        debt = await harness.adapter.maintenance_debt()
        check = await self_check(harness, sample_known_items(corpus, 20, 3))

        assert debt.num_unindexed_rows == 0
        assert check.sampled == 20
        assert check.passed
        assert check.rank_one >= 18

    async def test_driver_writes_nothing_when_recall_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rows = [
            CorpusRow(
                id=str(uuid.uuid4()),
                kind="fact",
                content=f"Row {n} about caches.",
                scope="global",
                lifecycle="active",
                vector=[0.1] * 768,
                recall_count=0,
                created_at="2026-07-31T13:54:10+00:00",
            )
            for n in range(4)
        ]
        question = QuestionRow(question_id="q1", query="caches", scope="global", origin="mined")
        write_jsonl(tmp_path / "corpus.jsonl", rows)
        write_jsonl(tmp_path / "questions.jsonl", [question])
        real_load = v1_search.load_corpus  # type: ignore[attr-defined]

        async def load_then_break(harness: Any, loaded: Any, log: Any) -> None:
            await real_load(harness, loaded, log)

            async def down(_: str) -> list[float]:
                raise EmbedderUnavailable("ollama down")

            harness.embedder.embed = down

        monkeypatch.setattr(v1_search, "load_corpus", load_then_break)

        out = tmp_path / "search-v1.jsonl"
        code = await v1_search._main(
            [
                "--corpus",
                str(tmp_path / "corpus.jsonl"),
                "--questions",
                str(tmp_path / "questions.jsonl"),
                "--out",
                str(out),
                "--store",
                str(tmp_path / "store"),
            ]
        )

        assert code == 1
        assert not out.exists()


def test_zero_hit_calls_counts_questions_without_rows() -> None:
    questions = [
        QuestionRow(question_id=q, query=q, scope="global", origin="mined") for q in ("a", "b", "c")
    ]
    rows = [
        SearchRow(question_id=q, arm="v1", rank=1, record_id="r", score=1.0, elapsed_ms=1.0)
        for q in ("a", "a", "c")
    ]

    assert zero_hit_calls(questions, rows) == 1
