from __future__ import annotations

import asyncio
import json
import pathlib
import sys

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.covers as covers  # noqa: E402
import eval_v1v2.fold as fold  # noqa: E402
import eval_v1v2.judges as judges  # noqa: E402
from eval_v1v2.rows import LabelRow, PairRow  # noqa: E402
from tabularium.engine.judge import JudgeUnavailable  # noqa: E402

sys.path.pop(0)


def answer(probability: float | None) -> covers.Answer:
    if probability is None:
        return covers.Answer(answer=None, p_yes=None, probabilities=None, error="down")

    return covers.Answer(
        answer="yes" if probability >= 0.5 else "no",
        p_yes=probability,
        probabilities={"yes": probability, "no": 1 - probability},
        error=None,
    )


def case(
    pair_id: str,
    pick: str | None,
    *,
    similarity: float | None = 0.8,
    new_covers_old: float = 0.0,
    enhances: float = 0.0,
) -> covers.Case:
    return covers.Case(
        pair_id=pair_id,
        pick=pick,
        similarity=similarity,
        in_band=pick == "follow_up",
        covers=covers.CoversSignal(
            pair_id=pair_id,
            run=1,
            new_covers_old=answer(new_covers_old),
            old_covers_new=answer(0.0),
            enhances=answer(enhances),
            elapsed_ms=1.0,
            error=None,
        ),
        sonnet=None,
    )


def grade(**answers: bool) -> fold.FoldGrade:
    return fold.FoldGrade(answers=answers)


GOOD = grade(nothing_lost=True, nothing_invented=True, newer_wins=True, one_record=True)
LOST = grade(nothing_lost=False, nothing_invented=True, newer_wins=True, one_record=True)
UNSURE = grade(nothing_lost=True, nothing_invented=True, newer_wins=True)


def pair(pair_id: str, existing: str = "old text", incoming: str = "new text") -> PairRow:
    return PairRow(
        pair_id=pair_id,
        source="golden",
        kind="fact",
        existing=existing,
        incoming=incoming,
        similarity=None,
        old_label=None,
    )


def band_row(pair_id: str, existing: str = "old text", incoming: str = "new text") -> fold.BandRow:
    return fold.BandRow(
        pair_id=pair_id,
        source="golden",
        relation=None,
        similarity=0.8,
        pick="follow_up",
        enhances=0.0,
        existing=existing,
        incoming=incoming,
    )


def fold_row(
    pair_id: str, writer: str = "haiku", text: str | None = "folded", error: str | None = None
) -> fold.FoldRow:
    return fold.FoldRow(
        pair_id=pair_id, writer=writer, text=text, elapsed_ms=5.0, attempts=1, error=error
    )


class FakeAdapter:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.prompts: list[str] = []
        self._replies = replies

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        return reply


class Scripted:
    def __init__(self, flip: set[str]) -> None:
        self._flip = flip

    async def call(self, prompt: str) -> str:
        verdicts = []

        for position, block in enumerate(prompt.split("\n\n[")[1:], start=1):
            question = block.split("QUESTION: ")[1].split("\n")[0]
            label = "no" if question in self._flip else "yes"
            verdicts.append({"index": position, "label": label, "reason": "r"})

        return json.dumps({"verdicts": verdicts})


class TestBand:
    def test_band_keeps_only_saves_the_containment_rule_would_make_as_new(self) -> None:
        cases = {
            "reinforced": case("reinforced", "restatement"),
            "superseded": case("superseded", "follow_up", new_covers_old=0.95),
            "flagged": case("flagged", "replacement"),
            "near": case("near", "follow_up", similarity=0.76),
            "far": case("far", "follow_up", similarity=0.74),
            "alone": case("alone", None, similarity=None),
        }
        pairs = {pid: pair(pid) for pid in cases}

        band = fold.build_band(cases, {"near": "follow_up"}, pairs)

        assert [row.pair_id for row in band] == ["near"]
        assert band[0].relation == "follow_up"

    def test_band_row_carries_the_texts_and_the_enhances_reading(self) -> None:
        cases = {"p": case("p", "follow_up", enhances=0.77)}

        row = fold.build_band(cases, {}, {"p": pair("p", "older", "newer")})[0]

        assert (row.existing, row.incoming, row.enhances, row.relation) == (
            "older",
            "newer",
            0.77,
            None,
        )


class TestRules:
    def test_rule_family_is_thirty_six_rules(self) -> None:
        assert len(fold.all_fold_rules()) == 36

    def test_containment_decides_first_so_a_restatement_pick_never_folds(self) -> None:
        rule = fold.fold_rule(0.75, None, "any non-distinct")

        assert rule.decide(case("p", "restatement", similarity=0.99)) == "reinforce"
        assert rule.decide(case("p", "follow_up", new_covers_old=0.95)) == "supersede"
        assert rule.decide(case("p", "replacement")) == "flag"

    def test_similarity_floor_gates_the_fold(self) -> None:
        rule = fold.fold_rule(0.85, None, "follow_up only")

        assert rule.decide(case("p", "follow_up", similarity=0.84)) == "new"
        assert rule.decide(case("p", "follow_up", similarity=0.85)) == "fold"

    def test_enhances_floor_gates_the_fold_and_a_missing_reading_never_fires(self) -> None:
        rule = fold.fold_rule(0.75, 0.80, "follow_up only")

        assert rule.decide(case("p", "follow_up", enhances=0.79)) == "new"
        assert rule.decide(case("p", "follow_up", enhances=0.80)) == "fold"

        bare = covers.Case(
            pair_id="p", pick="follow_up", similarity=0.9, in_band=True, covers=None, sonnet=None
        )

        assert rule.decide(bare) == "new"

    def test_pick_set_excludes_distinct_from_every_rule(self) -> None:
        for rule in fold.all_fold_rules():
            assert rule.decide(case("p", "distinct")) == "new"

    def test_unclear_pick_folds_only_when_the_pick_set_allows_it(self) -> None:
        narrow = fold.fold_rule(0.75, None, "follow_up only")
        wide = fold.fold_rule(0.75, None, "follow_up or unclear")

        assert narrow.decide(case("p", "unclear")) == "new"
        assert wide.decide(case("p", "unclear")) == "fold"


class TestGrades:
    def test_one_keyed_no_makes_the_fold_bad(self) -> None:
        assert LOST.status == "bad"
        assert LOST.failing == ["nothing_lost"]

    def test_all_four_keyed_yes_is_good_and_a_missing_answer_is_disputed(self) -> None:
        assert GOOD.status == "good"
        assert UNSURE.status == "disputed"
        assert fold.FoldGrade().status == "disputed"

    def test_a_no_outranks_a_missing_answer(self) -> None:
        assert grade(nothing_lost=False).status == "bad"

    def test_key_rows_become_grades_per_pair_and_writer(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "key-fold.jsonl"
        rows = [
            ("p1~haiku#nothing_lost", "yes"),
            ("p1~haiku#one_record", "no"),
            ("p1~sonnet#newer_wins", "yes"),
        ]
        path.write_text(
            "".join(
                json.dumps({"item_id": item, "label": label, "source": "agreed"}) + "\n"
                for item, label in rows
            ),
            encoding="utf-8",
        )

        grades = fold.load_fold_grades(path)

        assert grades[("p1", "haiku")].answers == {"nothing_lost": True, "one_record": False}
        assert grades[("p1", "sonnet")].answers == {"newer_wins": True}
        assert fold.split_fold_item("p~1~haiku#one_record") == ("p~1", "haiku", "one_record")


class TestScoring:
    def folded(self, ids: dict[str, str]) -> dict[str, covers.Case]:
        return {pid: case(pid, "follow_up") for pid in ids}

    def test_only_a_good_fold_resolves_a_restatement(self) -> None:
        relation = {"a": "restatement", "b": "restatement", "c": "restatement"}
        cases = self.folded(relation)
        grades = {"a": GOOD, "b": LOST, "c": UNSURE}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, grades)

        assert score.resolved_restatement == 1
        assert (len(score.good), len(score.bad), len(score.disputed)) == (1, 1, 1)
        assert score.bad_by_answer == {"nothing_lost": 1}
        assert score.bad_by_label == {"restatement": 1}

    def test_a_replacement_counts_on_newer_wins_and_nothing_invented_alone(self) -> None:
        relation = {"a": "replacement", "b": "replacement", "c": "replacement"}
        cases = self.folded(relation)
        grades = {
            "a": grade(nothing_lost=False, nothing_invented=True, newer_wins=True, one_record=True),
            "b": grade(nothing_lost=True, nothing_invented=False, newer_wins=True, one_record=True),
            "c": grade(nothing_lost=True, nothing_invented=True, newer_wins=False, one_record=True),
        }
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, grades)

        assert score.caught_replacement == 1
        assert score.folded_good_replacement == 1

    def test_loose_counts_a_disputed_fold_and_strict_charges_it_as_an_error(self) -> None:
        relation = {"a": "restatement", "b": "restatement", "c": "replacement"}
        cases = self.folded(relation)
        grades = {"a": UNSURE, "b": LOST, "c": grade(newer_wins=True)}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, grades)

        assert score.resolved_restatement == 0
        assert score.loose_restatement == 1
        assert score.caught_replacement == 0
        assert score.loose_replacement == 1
        assert score.errors == 1
        assert score.strict_errors == 3

    def test_flags_and_supersedes_still_count_beside_folds(self) -> None:
        cases = {
            "flag": case("flag", "replacement"),
            "sup": case("sup", "follow_up", new_covers_old=0.95),
            "fold": case("fold", "follow_up"),
        }
        relation = {"flag": "replacement", "sup": "replacement", "fold": "replacement"}
        cover = {"sup": {"new_covers_old": True}}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, cover, {"fold": GOOD})

        assert score.caught_replacement == 3
        assert score.base.flagged_replacement == 1
        assert score.base.superseded_replacement == 1

    def test_follow_up_folds_split_by_one_record(self) -> None:
        relation = {"a": "follow_up", "b": "follow_up", "c": "follow_up"}
        cases = self.folded(relation)
        glued = grade(nothing_lost=True, nothing_invented=True, newer_wins=True, one_record=False)
        grades = {"a": GOOD, "b": glued, "c": UNSURE}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, grades)

        assert score.follow_up_folds == 3
        assert score.follow_up_one_record_yes == 1
        assert score.follow_up_one_record_no == 1
        assert score.bad_by_answer == {"one_record": 1}

    def test_distinct_folds_are_listed_and_flag_a_missing_one_record(self) -> None:
        cases = {"a": case("a", "follow_up"), "b": case("b", "follow_up")}
        relation = {"a": "distinct", "b": "distinct"}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, {"a": GOOD, "b": UNSURE})

        assert score.distinct_folds == ["a", "b"]
        assert score.distinct_not_one_record == ["b"]

    def test_wrong_supersedes_and_bad_folds_both_break_the_bars(self) -> None:
        cases = {"x": case("x", "follow_up", new_covers_old=0.95)}
        relation = {"x": "follow_up"}
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {"x": {"new_covers_old": False}}, {})

        assert score.base.wrong == 1
        assert score.errors == 1
        assert not score.meets_bars

    def test_only_restricts_scoring_to_the_named_pairs(self) -> None:
        relation = {"a": "restatement", "b": "restatement"}
        cases = self.folded(relation)
        rule = fold.fold_rule(0.75, None, "follow_up only")

        score = fold.score_fold(rule, cases, relation, {}, {"a": GOOD, "b": GOOD}, frozenset({"b"}))

        assert score.triggered == ["b"]
        assert score.base.restatements == 1

    def test_the_best_rule_has_no_errors_before_it_has_more_resolved(self) -> None:
        relation = {"a": "restatement", "b": "restatement"}
        cases = {
            "a": case("a", "follow_up", similarity=0.9),
            "b": case("b", "follow_up", similarity=0.76),
        }
        grades = {"a": GOOD, "b": LOST}
        rules = [
            fold.fold_rule(0.75, None, "follow_up only"),
            fold.fold_rule(0.85, None, "follow_up only"),
        ]

        best = fold.best_fold_rule(rules, cases, relation, {}, grades)

        assert best.floor == 0.85


class TestLeaveOneSourceOut:
    def test_a_pair_without_a_relation_key_counts_in_the_held_out_bad_folds(
        self, tmp_path: pathlib.Path
    ) -> None:
        cases = {"a": case("a", "follow_up"), "n": case("n", "follow_up")}
        inputs = fold.Inputs(
            pairs={
                "a": pair("a"),
                "n": pair("n").model_copy(update={"source": "bounce_log"}),
            },
            cases=cases,
            relation={"a": "restatement"},
            cover={},
            band=[],
            folds={},
            grades={("a", "haiku"): GOOD, ("n", "haiku"): LOST},
            labels=[],
            published_v1=[],
            out_dir=tmp_path,
        )
        rules = [fold.fold_rule(0.75, None, "follow_up only")]

        table, pooled = fold.leave_one_source_out(rules, inputs, "haiku")

        assert pooled["bad"] == 1
        assert pooled["restatements"] == 1
        assert "| bounce_log |" in table


class TestWriters:
    def test_prompt_demands_the_four_rules_and_marks_the_newer_note(self) -> None:
        prompt = fold.render_writer_prompt("the older text", "the newer text")

        assert "OLDER:\nthe older text" in prompt
        assert "NEWER:\nthe newer text" in prompt
        assert "NEWER arrives\nnow and is current" in prompt
        assert "Keep every concrete detail from both notes" in prompt
        assert "Add nothing that is in neither note" in prompt
        assert "NEWER wins" in prompt
        assert "No preamble" in prompt

    def test_the_reply_is_stripped_and_timed(self) -> None:
        adapter = FakeAdapter(["  merged note \n"])

        result = asyncio.run(fold.write_fold(adapter, "haiku", band_row("p")))  # type: ignore[arg-type]

        assert result.text == "merged note"
        assert result.answered
        assert result.elapsed_ms >= 0
        assert "old text" in adapter.prompts[0]

    def test_an_empty_reply_is_retried_then_recorded_as_an_error(self) -> None:
        adapter = FakeAdapter(["  ", ""])

        result = asyncio.run(fold.write_fold(adapter, "haiku", band_row("p")))  # type: ignore[arg-type]

        assert not result.answered
        assert result.error == "EmptyFold"
        assert result.attempts == 2

    def test_an_unavailable_judge_is_recorded_not_raised(self) -> None:
        adapter = FakeAdapter([JudgeUnavailable("limit"), "second try"])

        result = asyncio.run(fold.write_fold(adapter, "sonnet", band_row("p")))  # type: ignore[arg-type]

        assert result.text == "second try"
        assert result.attempts == 2

    def test_resume_skips_answered_folds_and_retries_failed_ones(self) -> None:
        band = [band_row("a"), band_row("b"), band_row("c")]
        done = {
            ("a", "haiku"): fold_row("a"),
            ("b", "haiku"): fold_row("b", text=None, error="JudgeUnavailable: limit"),
            ("a", "sonnet"): fold_row("a", "sonnet"),
        }

        assert [row.pair_id for row in fold.pending_band(band, done, "haiku")] == ["b", "c"]
        assert [row.pair_id for row in fold.pending_band(band, done, "sonnet")] == ["b", "c"]

    def test_the_log_keeps_the_latest_row_per_pair_and_writer(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "folds.jsonl"
        rows = [fold_row("a", text=None, error="x"), fold_row("a", text="later")]
        path.write_text(
            "".join(json.dumps(row.model_dump(mode="json")) + "\n" for row in rows),
            encoding="utf-8",
        )

        assert fold.load_folds(path)[("a", "haiku")].text == "later"

    def test_a_run_of_failures_stops_the_writer(self) -> None:
        adapter = FakeAdapter([JudgeUnavailable("limit")] * 40)
        band = [band_row(f"p{n}") for n in range(20)]
        seen: list[fold.FoldRow] = []

        asyncio.run(fold.write_all(adapter, "haiku", band, seen.append))  # type: ignore[arg-type]

        assert len(seen) < len(band)


class TestJudging:
    def folds(self) -> dict[tuple[str, str], fold.FoldRow]:
        return {
            ("a", "haiku"): fold_row("a", "haiku", "first fold text"),
            ("a", "sonnet"): fold_row("a", "sonnet", "second fold text"),
            ("b", "haiku"): fold_row("b", "haiku", None, "EmptyFold"),
        }

    def test_each_answered_fold_becomes_four_items_and_failed_folds_none(self) -> None:
        items = fold.fold_items([band_row("a"), band_row("b")], self.folds())

        assert len(items) == 8
        assert items[0].item_id == "a~haiku#nothing_lost"
        assert {item.fields["fold"] for item in items} == {"first fold text", "second fold text"}

    def test_the_judge_prompt_never_names_the_writer(self) -> None:
        items = fold.fold_items([band_row("a")], self.folds())

        prompt = judges.build_prompt(judges.TASKS["fold"], items)

        assert "haiku" not in prompt
        assert "sonnet" not in prompt
        assert (
            "QUESTION: nothing_lost\nEXISTING: old text\nINCOMING: new text\nFOLD: first fold text"
            in prompt
        )
        assert "The label is yes or no." in prompt
        for name in fold.QUESTIONS:
            assert f"- {name}:" in prompt

    def test_run_task_keys_only_the_answers_both_judges_give_alike(
        self, tmp_path: pathlib.Path
    ) -> None:
        items = fold.fold_items([band_row("a")], {("a", "haiku"): fold_row("a", "haiku", "f")})

        def seat(judge: str, flip: set[str]) -> judges.Seat:
            return judges.Seat(judge=judge, model="m", adapter=Scripted(flip))  # type: ignore[arg-type]

        outcome = judges.run_task(
            "fold",
            items,
            seats=[seat("claude", {"one_record"}), seat("codex", {"one_record", "newer_wins"})],
            out_dir=tmp_path,
        )

        keyed = {row.item_id: row.label for row in outcome.key}

        assert keyed == {
            "a~haiku#nothing_lost": "yes",
            "a~haiku#nothing_invented": "yes",
            "a~haiku#one_record": "no",
        }

    def test_agreement_is_reported_per_answer(self) -> None:
        def label(item: str, judge: str, value: str) -> LabelRow:
            return LabelRow.model_validate(
                {"item_id": item, "judge": judge, "model": "m", "label": value, "reason": "r"}
            )

        rows = [
            label("a~haiku#one_record", "claude", "yes"),
            label("a~haiku#one_record", "codex", "no"),
            label("a~haiku#newer_wins", "claude", "yes"),
            label("a~haiku#newer_wins", "codex", "yes"),
        ]

        text = fold.agreement_table(rows)

        assert "| one_record | 1 | 0 | 0.0% |" in text
        assert "| newer_wins | 1 | 1 | 100.0% |" in text


class TestReports:
    def inputs(self, tmp_path: pathlib.Path) -> fold.Inputs:
        cases = {"a": case("a", "follow_up"), "b": case("b", "follow_up")}
        band = [band_row("a", "old a", "new a"), band_row("b", "old b", "new b")]

        return fold.Inputs(
            pairs={pid: pair(pid) for pid in cases},
            cases=cases,
            relation={"a": "restatement", "b": "follow_up"},
            cover={},
            band=band,
            folds={
                ("a", "haiku"): fold_row("a", "haiku", "fold of a"),
                ("b", "haiku"): fold_row("b", "haiku", "fold of b"),
            },
            grades={("a", "haiku"): GOOD, ("b", "haiku"): LOST},
            labels=[],
            published_v1=[],
            out_dir=tmp_path,
        )

    def test_bad_folds_list_pair_sources_fold_and_failing_answer(
        self, tmp_path: pathlib.Path
    ) -> None:
        inputs = self.inputs(tmp_path)
        rule = fold.fold_rule(0.75, None, "follow_up only")

        text = fold.bad_folds_text(inputs, {"haiku": rule, "sonnet": rule})

        assert "1 bad folds." in text
        assert "### b (haiku, relation key: disputed)" in text
        assert "Failing: nothing_lost." in text
        assert "- existing: old b" in text
        assert "- incoming: new b" in text
        assert "- fold: fold of b" in text
        assert "fold of a" not in text

    def test_unjudged_answers_are_counted_when_a_judge_is_missing(
        self, tmp_path: pathlib.Path
    ) -> None:
        inputs = self.inputs(tmp_path)

        assert fold.unjudged_count(inputs) == 8

    def test_length_ratio_is_fold_over_the_longer_source(self) -> None:
        row = band_row("a", "x" * 10, "y" * 20)

        assert fold.length_ratio(fold_row("a", text="z" * 30), row) == 1.5
