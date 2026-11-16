"""Tests for scripts/calibrate_relevance_rubric.py -- the relevance-rubric calibration harness
(see the script's own module docstring for the full brief).

Hermetic lane only -- ``MemoryAdapter`` + ``FakeJudgeAdapter``, no network, no real store, no
credential (same convention as tests/test_relevance_axis.py).

Falsifiability (per this repo's own bar -- recorded here, not just asserted):

1. Scoring maths (``TestScoreAggregates``): temporarily inverted ``score_aggregates``'s
   ``a.majority_band == "keep"`` comparison to ``!=`` (production code,
   ``scripts/calibrate_relevance_rubric.py``) and re-ran
   ``test_recovery_and_retention_are_independent_fractions``. It went red:
   ``AssertionError: assert 0.25 == 0.75`` (the pinned ``false_positive_recovery`` value: 1 of the
   4 keep-labelled records is a genuine "shelve" majority, so 3 should count as recovered under
   the correct ``==`` comparison but only 1 does under the inverted ``!=`` one), because inverting
   the comparison flips which keep-labelled records count as "recovered". Reverted after
   confirming red.
2. No-write guarantee (``TestNoWriteGuarantee``): temporarily changed the ``dry_run=True`` literal
   in ``run_calibration``'s call to ``run_relevance_axis`` to ``dry_run=False`` (production code)
   and re-ran ``test_run_calibration_never_writes_even_when_every_verdict_would_shelve``. It went
   red with ``RuntimeError: calibrate_relevance_rubric: refused adapter.append_event() -- this
   harness only ever calls get_by_id; every other adapter call is structurally refused`` -- raised
   from ``_NoWriteGuard.__getattr__``'s stub (layer 2 fired first, before the write ever reached
   ``_WriteRaisingAdapter``'s own layer-1 stub), via ``set_shelved``'s "event before transition"
   ordering calling ``adapter.append_event(...)``. Proves BOTH the test's assertion of "this never
   happens" is load-bearing rather than vacuously true, AND that ``_NoWriteGuard`` (not just
   ``run_relevance_axis``'s own ``dry_run`` gate) is independently sufficient to catch an
   accidental live write. Reverted after confirming red. (Both
   fixture records in that test use ``lifecycle="active"`` specifically so this break is
   observable -- an already-``shelved`` Fact's ``set_shelved(shelved=True)`` is its own idempotent
   no-op with no adapter write at all, which would have made the break invisible.)
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.domain import Decision, Fact, HypomnemaId, Scope, new_id
from tabularium.engine.rubric import load_rubric
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeJudgeAdapter

# Insert the scripts dir temporarily so calibrate_relevance_rubric.py is importable without
# making it a package (the same import convention as the other script tests). E402 is
# suppressed: the path manipulation is intentional and must precede these imports.
_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import calibrate_relevance_rubric as script  # noqa: E402
from calibrate_relevance_rubric import (  # noqa: E402
    CalibrationResult,
    LabelledId,
    RecordRunVerdict,
    aggregate_record,
    load_ids_file,
    majority_band,
    run_calibration,
    score_aggregates,
    write_results,
)

sys.path.pop(0)

_DIM = 4
_NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _make_adapter() -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)


def _vec(seed: float = 1.0) -> list[float]:
    return [seed] * _DIM


def _fact(*, content: str, lifecycle: str = "active", now: datetime = _NOW) -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope("proj-a"),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity="valid",
        created_at=now,
        updated_at=now,
    )


def _decision(*, content: str, now: datetime = _NOW) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content=content,
        scope=Scope("proj-a"),
        lifecycle="in_force",
        validity="valid",
        created_at=now,
        updated_at=now,
    )


def _v(index: int, grade: str, *, unclear: bool = False, why: str = "because") -> dict[str, Any]:
    return {"index": index, "grade": grade, "unclear": unclear, "why": why}


def _payload(items: list[dict[str, Any]]) -> str:
    return json.dumps({"verdicts": items})


class _SequencedJudgeAdapter:
    """Returns a different canned response on each successive call, in order -- duplicated locally
    from tests/test_relevance_axis.py's own fixture of the same name (that file's own precedent:
    a private per-test-module fake, not a shared import)."""

    def __init__(self, *, identity: str, responses: list[str]) -> None:
        self._identity = identity
        self._responses = responses
        self.calls: list[str] = []

    @property
    def identity(self) -> str:
        return self._identity

    def instruction_fingerprint(self) -> str:
        return "not_applicable"

    async def call(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._responses[len(self.calls) - 1]


# ===========================================================================
# load_ids_file
# ===========================================================================


class TestLoadIdsFile:
    def test_round_trips_a_well_formed_file(self, tmp_path: pathlib.Path) -> None:
        id_a = new_id()
        id_b = new_id()
        path = tmp_path / "ids.json"
        path.write_text(
            json.dumps([{"id": str(id_a), "label": "keep"}, {"id": str(id_b), "label": "shelve"}])
        )

        entries = load_ids_file(path)

        assert entries == [
            LabelledId(id=id_a, label="keep"),
            LabelledId(id=id_b, label="shelve"),
        ]

    def test_not_a_json_array_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps({"id": "x", "label": "keep"}))

        with pytest.raises(ValueError, match="expected a JSON array"):
            load_ids_file(path)

    def test_invalid_label_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps([{"id": str(new_id()), "label": "discard"}]))

        with pytest.raises(ValueError, match="must be 'keep' or 'shelve'"):
            load_ids_file(path)

    def test_missing_field_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps([{"id": str(new_id())}]))

        with pytest.raises(ValueError, match="must have both 'id' and 'label'"):
            load_ids_file(path)

    def test_malformed_uuid_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "ids.json"
        path.write_text(json.dumps([{"id": "not-a-uuid", "label": "keep"}]))

        with pytest.raises(ValueError, match="invalid id"):
            load_ids_file(path)

    def test_duplicate_id_raises_even_with_the_same_label(self, tmp_path: pathlib.Path) -> None:
        """A duplicate id must fail loud, not
        silently double-judge a record and skew a scoring denominator."""
        dupe = new_id()
        path = tmp_path / "ids.json"
        path.write_text(
            json.dumps([{"id": str(dupe), "label": "keep"}, {"id": str(dupe), "label": "keep"}])
        )

        with pytest.raises(ValueError, match="duplicate id"):
            load_ids_file(path)

    def test_duplicate_id_with_conflicting_labels_also_raises(
        self, tmp_path: pathlib.Path
    ) -> None:
        dupe = new_id()
        path = tmp_path / "ids.json"
        path.write_text(
            json.dumps([{"id": str(dupe), "label": "keep"}, {"id": str(dupe), "label": "shelve"}])
        )

        with pytest.raises(ValueError, match="duplicate id"):
            load_ids_file(path)


# ===========================================================================
# Scoring maths -- pure functions, no adapter/judge involved
# ===========================================================================


class TestMajorityBand:
    def test_empty_bands_is_none_none(self) -> None:
        assert majority_band([]) == (None, None)

    def test_unanimous_bands_have_full_stability(self) -> None:
        winner, stability = majority_band(["keep", "keep", "keep"])
        assert winner == "keep"
        assert stability == 1.0

    def test_majority_wins_and_stability_is_the_agreeing_fraction(self) -> None:
        winner, stability = majority_band(["shelve", "keep", "keep"])
        assert winner == "keep"
        assert stability == pytest.approx(2 / 3)

    def test_a_genuine_tie_reports_no_majority(self) -> None:
        """A tie must never be silently
        resolved to a "winning" band -- ``score_aggregates`` would otherwise count an unstable
        coin-flip as a real recovery/retention."""
        assert majority_band(["shelve", "keep"]) == (None, None)

    def test_three_way_tie_also_reports_no_majority(self) -> None:
        assert majority_band(["shelve", "keep", "unclear"]) == (None, None)


class TestAggregateRecord:
    def test_batch_failed_runs_excluded_from_bands_but_counted_separately(self) -> None:
        record_id = new_id()
        runs = [
            RecordRunVerdict(record_id=record_id, run_index=0, band="shelve", rubric_version="v1"),
            RecordRunVerdict(
                record_id=record_id, run_index=1, band="batch_failed", rubric_version="v1"
            ),
            RecordRunVerdict(record_id=record_id, run_index=2, band="shelve", rubric_version="v1"),
        ]

        agg = aggregate_record(record_id, "shelve", runs)

        assert agg.bands == ("shelve", "shelve")
        assert agg.batch_failed_runs == 1
        assert agg.majority_band == "shelve"
        assert agg.stability == 1.0

    def test_every_run_failed_yields_no_majority(self) -> None:
        record_id = new_id()
        runs = [
            RecordRunVerdict(
                record_id=record_id, run_index=0, band="batch_failed", rubric_version="v1"
            ),
        ]

        agg = aggregate_record(record_id, "keep", runs)

        assert agg.bands == ()
        assert agg.batch_failed_runs == 1
        assert agg.majority_band is None
        assert agg.stability is None

    def test_a_tied_record_reports_no_majority_and_is_excluded_from_scoring(self) -> None:
        """A
        record that flips 1-1 between two runs must not silently count as "recovered" under
        either label."""
        record_id = new_id()
        runs = [
            RecordRunVerdict(record_id=record_id, run_index=0, band="shelve", rubric_version="v1"),
            RecordRunVerdict(record_id=record_id, run_index=1, band="keep", rubric_version="v1"),
        ]

        agg = aggregate_record(record_id, "keep", runs)

        assert agg.bands == ("shelve", "keep")
        assert agg.majority_band is None
        assert agg.stability is None

        score = score_aggregates([agg])
        assert score.false_positive_total == 0
        assert score.false_positive_recovery is None


class TestScoreAggregates:
    def test_recovery_and_retention_are_independent_fractions(self) -> None:
        """4 keep-labelled records, 3 recovered (majority band "keep"); 4 shelve-labelled
        records, 4 retained (majority band "shelve") -- the two fractions must differ, proving
        neither metric is silently derived from the other (module brief: a uniform leniency shift
        would move only one of them)."""
        keep_aggs = [
            aggregate_record(
                new_id(),
                "keep",
                [RecordRunVerdict(record_id=new_id(), run_index=0, band=b, rubric_version="v1")],
            )
            for b in ("keep", "keep", "keep", "shelve")
        ]
        shelve_aggs = [
            aggregate_record(
                new_id(),
                "shelve",
                [RecordRunVerdict(record_id=new_id(), run_index=0, band=b, rubric_version="v1")],
            )
            for b in ("shelve", "shelve", "shelve", "shelve")
        ]

        score = score_aggregates(keep_aggs + shelve_aggs)

        assert score.false_positive_recovery == pytest.approx(0.75)
        assert score.false_positive_recovered == 3
        assert score.false_positive_total == 4
        assert score.control_retention == 1.0
        assert score.control_retained == 4
        assert score.control_total == 4

    def test_unclear_and_contested_bands_count_against_both_metrics(self) -> None:
        """A keep-labelled record whose majority band is "unclear"/"contested" is neither
        recovered nor a crash -- it lands in the denominator (it DID score a majority) but not
        the numerator (its majority band is not literally "keep")."""
        agg = aggregate_record(
            new_id(),
            "keep",
            [
                RecordRunVerdict(
                    record_id=new_id(), run_index=0, band="unclear", rubric_version="v1"
                )
            ],
        )

        score = score_aggregates([agg])

        assert score.false_positive_total == 1
        assert score.false_positive_recovered == 0
        assert score.false_positive_recovery == 0.0

    def test_empty_label_group_reports_none_not_zero(self) -> None:
        shelve_agg = aggregate_record(
            new_id(),
            "shelve",
            [RecordRunVerdict(record_id=new_id(), run_index=0, band="shelve", rubric_version="v1")],
        )

        score = score_aggregates([shelve_agg])

        assert score.false_positive_recovery is None
        assert score.false_positive_total == 0
        assert score.control_retention == 1.0

    def test_records_with_no_majority_are_excluded_from_both_denominators(self) -> None:
        never_scored = aggregate_record(
            new_id(),
            "keep",
            [
                RecordRunVerdict(
                    record_id=new_id(), run_index=0, band="batch_failed", rubric_version="v1"
                )
            ],
        )

        score = score_aggregates([never_scored])

        assert score.false_positive_total == 0
        assert score.false_positive_recovery is None


# ===========================================================================
# run_calibration -- end to end against MemoryAdapter + FakeJudgeAdapter
# ===========================================================================


class TestRunCalibrationMultiRunAggregation:
    async def test_band_flip_across_runs_is_captured_by_majority_and_stability(self) -> None:
        adapter = _make_adapter()
        record = _fact(content="borderline record, flips band across runs", lifecycle="shelved")
        await adapter.upsert(record, _vec())

        judge_a = _SequencedJudgeAdapter(
            identity="a",
            responses=[
                _payload([_v(1, "shelve-clear")]),
                _payload([_v(1, "keep-clear")]),
                _payload([_v(1, "keep-clear")]),
            ],
        )
        judge_b = _SequencedJudgeAdapter(
            identity="b",
            responses=[
                _payload([_v(1, "shelve-clear")]),
                _payload([_v(1, "keep-clear")]),
                _payload([_v(1, "keep-clear")]),
            ],
        )

        result = await run_calibration(
            adapter,
            scope=Scope("proj-a"),
            labelled_ids=[LabelledId(id=record.id, label="keep")],
            runs=3,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(result.per_run_verdicts) == 3
        assert [v.band for v in result.per_run_verdicts] == ["shelve", "keep", "keep"]
        assert [v.run_index for v in result.per_run_verdicts] == [0, 1, 2]
        assert all(v.judge_a is not None and v.judge_b is not None for v in result.per_run_verdicts)

        [agg] = result.aggregates
        assert agg.record_id == record.id
        assert agg.label == "keep"
        assert agg.bands == ("shelve", "keep", "keep")
        assert agg.majority_band == "keep"
        assert agg.stability == pytest.approx(2 / 3)
        assert agg.batch_failed_runs == 0

        assert result.score.false_positive_recovery == 1.0
        assert result.score.control_retention is None  # no shelve-labelled records in this batch

    async def test_both_seats_grade_and_why_are_captured_per_run(self) -> None:
        adapter = _make_adapter()
        record = _fact(content="a record with a clear gate answer")
        await adapter.upsert(record, _vec())

        judge_a = FakeJudgeAdapter(
            identity="claude:sonnet",
            response=_payload([_v(1, "keep-clear", why="A's reason")]),
        )
        judge_b = FakeJudgeAdapter(
            identity="codex:gpt", response=_payload([_v(1, "keep-lean", why="B's reason")])
        )

        result = await run_calibration(
            adapter,
            scope=Scope("proj-a"),
            labelled_ids=[LabelledId(id=record.id, label="keep")],
            runs=1,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        [verdict] = result.per_run_verdicts
        assert verdict.band == "keep"
        assert verdict.judge_a == script.JudgeGrade(
            identity="claude:sonnet", grade="keep-clear", why="A's reason", unclear=False
        )
        assert verdict.judge_b == script.JudgeGrade(
            identity="codex:gpt", grade="keep-lean", why="B's reason", unclear=False
        )
        assert verdict.rubric_version == load_rubric().version


class TestMissingAndNonFactIds:
    async def test_missing_id_is_reported_not_crashed_on(self) -> None:
        adapter = _make_adapter()
        present = _fact(content="present in the store")
        await adapter.upsert(present, _vec())
        missing_id = new_id()

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([_v(1, "keep-clear")]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([_v(1, "keep-clear")]))

        result = await run_calibration(
            adapter,
            scope=Scope("proj-a"),
            labelled_ids=[
                LabelledId(id=present.id, label="keep"),
                LabelledId(id=missing_id, label="shelve"),
            ],
            runs=1,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert result.missing_ids == (missing_id,)
        assert len(result.aggregates) == 1
        assert result.aggregates[0].record_id == present.id
        # the missing id never reaches a judge -- only one record in the batch
        assert len(judge_a.calls) == 1

    async def test_decision_id_is_skipped_not_crashed_on(self) -> None:
        adapter = _make_adapter()
        decision = _decision(content="an in-force decision, not a Fact")
        await adapter.upsert(decision, _vec())

        judge_a = FakeJudgeAdapter(identity="a", response=_payload([]))
        judge_b = FakeJudgeAdapter(identity="b", response=_payload([]))

        result = await run_calibration(
            adapter,
            scope=Scope("proj-a"),
            labelled_ids=[LabelledId(id=decision.id, label="keep")],
            runs=1,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert result.skipped_non_fact_ids == (decision.id,)
        assert result.aggregates == ()
        assert judge_a.calls == []


# ===========================================================================
# No-write guarantee -- structurally incapable of writing to the store
# ===========================================================================


class _WriteRaisingAdapter:
    """A ``StorageAdapter`` double whose every method except ``get_by_id`` raises
    ``AssertionError`` if called -- the no-write guarantee's actual falsifiable test surface (see
    the module docstring's falsifiability note 2)."""

    def __init__(self, records: dict[HypomnemaId, Fact]) -> None:
        self._records = records

    async def get_by_id(self, id: HypomnemaId) -> Fact | Decision:
        if id not in self._records:
            raise KeyError(id)
        return self._records[id]

    def __getattr__(self, name: str) -> Any:
        async def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"no-write guarantee violated: adapter.{name}() was called")

        return _boom


class TestNoWriteGuarantee:
    async def test_run_calibration_never_writes_even_when_every_verdict_would_shelve(self) -> None:
        """Two ``active`` Facts (deliberately not already ``shelved`` -- see the module
        docstring's falsifiability note 2 for why), both unanimously graded ``shelve-clear`` by
        both seats across 2 runs. If ``run_relevance_axis`` (or this harness) ever attempted the
        write a live pass would make here, ``_WriteRaisingAdapter``'s ``transition`` stub would
        raise -- this test's only assertion is that the whole run completes without that ever
        happening."""
        keep_record = _fact(
            content="would have been a false positive if written", lifecycle="active"
        )
        shelve_record = _fact(content="correctly shelved if written", lifecycle="active")
        records = {keep_record.id: keep_record, shelve_record.id: shelve_record}
        adapter = _WriteRaisingAdapter(records)

        judge_a = FakeJudgeAdapter(
            identity="a", response=_payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        )
        judge_b = FakeJudgeAdapter(
            identity="b", response=_payload([_v(1, "shelve-clear"), _v(2, "shelve-clear")])
        )

        result = await run_calibration(
            adapter,
            scope=Scope("proj-a"),
            labelled_ids=[
                LabelledId(id=keep_record.id, label="keep"),
                LabelledId(id=shelve_record.id, label="shelve"),
            ],
            runs=2,
            rubric=load_rubric(),
            judge_a=judge_a,
            judge_b=judge_b,
        )

        assert len(result.per_run_verdicts) == 4  # 2 records x 2 runs
        assert all(v.band == "shelve" for v in result.per_run_verdicts)
        # Both records "recovered"/"retained" as shelve here is the WRONG outcome by ground
        # truth (the panel agreed with the old rubric, not the labels) -- that is fine; this
        # test's only job is proving nothing was ever written while producing that verdict.
        assert result.score.control_retention == 1.0


# ===========================================================================
# write_results -- durable JSON artifact, atomic write
# ===========================================================================


class TestWriteResults:
    def test_round_trips_a_result_to_disk(self, tmp_path: pathlib.Path) -> None:
        record_id = new_id()
        result = CalibrationResult(
            scope="proj-a",
            runs=1,
            rubric_version="cleaner-rubric-v1-deadbeef",
            missing_ids=(),
            skipped_non_fact_ids=(),
            per_run_verdicts=(
                RecordRunVerdict(
                    record_id=record_id,
                    run_index=0,
                    band="keep",
                    rubric_version="cleaner-rubric-v1-deadbeef",
                ),
            ),
            aggregates=(
                aggregate_record(
                    record_id,
                    "keep",
                    [
                        RecordRunVerdict(
                            record_id=record_id,
                            run_index=0,
                            band="keep",
                            rubric_version="cleaner-rubric-v1-deadbeef",
                        )
                    ],
                ),
            ),
            score=score_aggregates(
                [
                    aggregate_record(
                        record_id,
                        "keep",
                        [
                            RecordRunVerdict(
                                record_id=record_id,
                                run_index=0,
                                band="keep",
                                rubric_version="cleaner-rubric-v1-deadbeef",
                            )
                        ],
                    )
                ]
            ),
        )
        out = tmp_path / "results.json"

        write_results(out, result)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["scope"] == "proj-a"
        assert data["score"]["false_positive_recovery"] == 1.0
        assert data["per_run_verdicts"][0]["record_id"] == str(record_id)
