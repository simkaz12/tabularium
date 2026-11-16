from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace
from typing import Any

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.covers as covers  # noqa: E402
import eval_v1v2.fold as fold  # noqa: E402
import eval_v1v2.fold_guard as guard  # noqa: E402
import eval_v1v2.fold_guard_recount as recount  # noqa: E402
from eval_v1v2.rows import PairRow, Source  # noqa: E402

sys.path.pop(0)

ALL_YES = {"nothing_lost": True, "nothing_invented": True, "newer_wins": True, "one_record": True}


def answer(probability: float | None, elapsed: float = 10.0) -> guard.TimedAnswer:
    if probability is None:
        return guard.TimedAnswer(
            answer=None, p_yes=None, probabilities=None, error="down", elapsed_ms=elapsed
        )

    return guard.TimedAnswer(
        answer="yes" if probability >= 0.5 else "no",
        p_yes=probability,
        probabilities={"yes": probability, "no": 1 - probability},
        error=None,
        elapsed_ms=elapsed,
    )


def case(
    pair_id: str,
    pick: str | None = "follow_up",
    *,
    similarity: float | None = 0.82,
    new_covers_old: float = 0.0,
    enhances: float = 0.0,
) -> covers.Case:
    def reading(probability: float) -> covers.Answer:
        return covers.Answer(
            answer="yes" if probability >= 0.5 else "no",
            p_yes=probability,
            probabilities=None,
            error=None,
        )

    return covers.Case(
        pair_id=pair_id,
        pick=pick,
        similarity=similarity,
        in_band=pick == "follow_up",
        covers=covers.CoversSignal(
            pair_id=pair_id,
            run=1,
            new_covers_old=reading(new_covers_old),
            old_covers_new=reading(0.0),
            enhances=reading(enhances),
            elapsed_ms=1.0,
            error=None,
        ),
        sonnet=None,
    )


def signal(
    pair_id: str,
    writer: str = "sonnet",
    *,
    old: float | None = 0.95,
    new: float | None = 0.95,
    sources: float | None = 0.95,
    single: float | None = 0.95,
) -> guard.GuardSignal:
    return guard.GuardSignal(
        pair_id=pair_id,
        writer=writer,
        run=1,
        fold_covers_old=answer(old),
        fold_covers_new=answer(new),
        sources_cover_fold=answer(sources),
        fold_single=answer(single),
        error=None,
    )


def subject(pair_id: str, probability: float) -> guard.BandSignal:
    return guard.BandSignal(pair_id=pair_id, run=1, same_subject=answer(probability), error=None)


def fold_row(pair_id: str, writer: str = "sonnet", text: str | None = "merged") -> fold.FoldRow:
    return fold.FoldRow(
        pair_id=pair_id,
        writer=writer,
        text=text,
        elapsed_ms=5.0,
        attempts=1,
        error=None if text else "EmptyFold",
    )


def pair_row(pair_id: str, source: Source = "golden") -> PairRow:
    return PairRow(
        pair_id=pair_id, source=source, kind="fact", existing="old", incoming="new",
        similarity=None, old_label=None,
    )  # fmt: skip


def band_row(pair_id: str, relation: str | None = None) -> fold.BandRow:
    return fold.BandRow(
        pair_id=pair_id, source="golden", relation=relation, similarity=0.82, pick="follow_up",
        enhances=0.0, existing="old", incoming="new",
    )  # fmt: skip


def data_for(
    cases: dict[str, covers.Case],
    *,
    relation: dict[str, str] | None = None,
    grades: dict[tuple[str, str], dict[str, bool]] | None = None,
    guards: list[guard.GuardSignal] | None = None,
    subjects: list[guard.BandSignal] | None = None,
    replacement: dict[str, float] | None = None,
    folds: list[fold.FoldRow] | None = None,
    out_dir: pathlib.Path | None = None,
) -> guard.GuardData:
    labels = relation or {}
    made = folds if folds is not None else [fold_row(pid) for pid in cases]
    inputs = fold.Inputs(
        pairs={pid: pair_row(pid) for pid in cases},
        cases=cases,
        relation=labels,
        cover={},
        band=[band_row(pid, labels.get(pid)) for pid in cases],
        folds={(row.pair_id, row.writer): row for row in made},
        grades={key: fold.FoldGrade(answers=value) for key, value in (grades or {}).items()},
        labels=[],
        published_v1=[],
        out_dir=out_dir or pathlib.Path("."),
    )

    return guard.GuardData(
        inputs=inputs,
        guards={(row.pair_id, row.writer): row for row in guards or []},
        subjects={row.pair_id: row for row in subjects or []},
        p_replacement=replacement or {},
        out_dir=out_dir or pathlib.Path("."),
    )


def rule(
    floor: float = 0.75,
    enhances: float | None = None,
    same_subject: float | None = None,
    replacement: float | None = None,
    kind: str = "off",
    threshold: float | None = None,
) -> guard.GuardRule:
    return guard.GuardRule(floor, enhances, same_subject, replacement, kind, threshold)


def action(r: guard.GuardRule, data: guard.GuardData, pair_id: str, writer: str = "sonnet") -> str:
    return r.bind(writer, data).decide(data.inputs.cases[pair_id])


class TestRuleSpace:
    def test_sweep_has_540_distinct_rules_and_names(self) -> None:
        rules = guard.all_guard_rules()

        assert len(rules) == 540
        assert len({r.name for r in rules}) == 540

    def test_knobs_count_the_settings_that_are_on(self) -> None:
        assert rule().knobs == 1
        assert rule(enhances=0.7, replacement=0.25, kind="all", threshold=0.8).knobs == 4
        assert rule(same_subject=0.5).knobs == 2

    def test_a_rule_never_sets_both_gates(self) -> None:
        for r in guard.all_guard_rules():
            assert r.enhances is None or r.subject is None


class TestTrigger:
    def test_containment_decides_first(self) -> None:
        data = data_for(
            {
                "r": case("r", "restatement", similarity=0.99),
                "s": case("s", "follow_up", new_covers_old=0.95),
                "f": case("f", "replacement"),
            }
        )

        assert action(rule(), data, "r") == "reinforce"
        assert action(rule(), data, "s") == "supersede"
        assert action(rule(), data, "f") == "flag"

    def test_floor_and_enhances_gate_the_fold(self) -> None:
        data = data_for({"a": case("a", similarity=0.84, enhances=0.75)})

        assert action(rule(floor=0.85), data, "a") == "new"
        assert action(rule(floor=0.80, enhances=0.70), data, "a") == "fold"
        assert action(rule(floor=0.80, enhances=0.80), data, "a") == "new"

    def test_same_subject_replaces_enhances_as_the_gate(self) -> None:
        data = data_for(
            {"a": case("a", enhances=0.0), "b": case("b", enhances=0.0)},
            subjects=[subject("a", 0.72)],
        )

        assert action(rule(same_subject=0.70), data, "a") == "fold"
        assert action(rule(same_subject=0.90), data, "a") == "new"
        assert action(rule(same_subject=0.50), data, "b") == "new"

    def test_replacement_trigger_ignores_floor_and_gate_but_not_the_band(self) -> None:
        data = data_for(
            {
                "near": case("near", similarity=0.76, enhances=0.0),
                "far": case("far", similarity=0.70),
                "distinct": case("distinct", "distinct", similarity=0.90),
            },
            replacement={"near": 0.30, "far": 0.9, "distinct": 0.9},
        )
        r = rule(floor=0.85, enhances=0.90, replacement=0.25)

        assert action(r, data, "near") == "fold"
        assert action(r, data, "far") == "new"
        assert action(r, data, "distinct") == "new"
        assert action(rule(floor=0.85, enhances=0.90, replacement=0.35), data, "near") == "new"

    def test_no_answered_fold_means_save_as_new(self) -> None:
        data = data_for({"a": case("a")}, folds=[fold_row("a", text=None)])

        assert action(rule(), data, "a") == "new"

        other = data_for({"a": case("a")}, folds=[fold_row("a", writer="haiku")])

        assert action(rule(), other, "a") == "new"


class TestGuards:
    def test_coverage_guards_need_both_readings_at_the_threshold(self) -> None:
        data = data_for(
            {"a": case("a"), "b": case("b"), "c": case("c")},
            guards=[
                signal("a", old=0.85, new=0.85),
                signal("b", old=0.85, new=0.65),
                signal("c", old=None),
            ],
        )
        r = rule(kind="coverage", threshold=0.80)

        assert [action(r, data, pid) for pid in "abc"] == ["fold", "new", "new"]

    def test_each_kind_checks_exactly_its_own_questions(self) -> None:
        data = data_for(
            {"a": case("a"), "b": case("b")},
            guards=[signal("a", sources=0.1), signal("b", single=0.1)],
        )

        assert action(rule(kind="coverage", threshold=0.8), data, "a") == "fold"
        assert action(rule(kind="coverage+sources", threshold=0.8), data, "a") == "new"
        assert action(rule(kind="coverage+sources", threshold=0.8), data, "b") == "fold"
        assert action(rule(kind="all", threshold=0.8), data, "b") == "new"

    def test_a_missing_guard_row_rejects_when_guards_are_on_and_passes_when_off(self) -> None:
        data = data_for({"a": case("a")})

        assert action(rule(kind="coverage", threshold=0.7), data, "a") == "new"
        assert action(rule(), data, "a") == "fold"


class TestScoring:
    def test_rejected_folds_split_by_grade_and_leave_the_save_as_new(self) -> None:
        cases = {pid: case(pid) for pid in ("g", "b", "d", "ok")}
        relation = dict.fromkeys(cases, "restatement")
        grades = {
            ("g", "sonnet"): ALL_YES,
            ("b", "sonnet"): {**ALL_YES, "one_record": False},
            ("d", "sonnet"): {"nothing_lost": True},
            ("ok", "sonnet"): ALL_YES,
        }
        guards = [
            signal("g", single=0.1),
            signal("b", single=0.1),
            signal("d", single=0.1),
            signal("ok"),
        ]
        data = data_for(cases, relation=relation, grades=grades, guards=guards)

        score = guard.score_guard(rule(kind="all", threshold=0.8), data, "sonnet")

        assert score.fired == ["g", "b", "d", "ok"]
        assert score.result.triggered == ["ok"]
        assert score.rejected_good == ["g"]
        assert score.rejected_bad == ["b"]
        assert score.rejected_disputed == ["d"]
        assert score.resolved("strict") == 1
        assert score.errors("strict") == 0

    def test_guards_off_matches_the_experiment_four_fold_rule(self) -> None:
        cases = {
            "a": case("a", similarity=0.9, enhances=0.95),
            "b": case("b", similarity=0.82, enhances=0.75),
            "c": case("c", similarity=0.78, enhances=0.99),
        }
        relation = {"a": "restatement", "b": "restatement", "c": "replacement"}
        grades = {
            ("a", "sonnet"): ALL_YES,
            ("b", "sonnet"): {**ALL_YES, "nothing_lost": False},
            ("c", "sonnet"): ALL_YES,
        }
        data = data_for(cases, relation=relation, grades=grades)
        old = fold.score_fold(
            fold.fold_rule(0.80, 0.70, "follow_up only"),
            cases,
            relation,
            {},
            fold.writer_grades(data.inputs.grades, "sonnet"),
        )

        new = guard.score_guard(rule(floor=0.80, enhances=0.70), data, "sonnet")

        assert new.result.triggered == old.triggered == ["a", "b"]
        assert new.result.bad == old.bad == ["b"]
        assert new.resolved("strict") == old.resolved_restatement == 1

    def test_a_fold_with_no_keyed_answer_is_disputed_not_a_crash(self) -> None:
        cases = {"a": case("a")}
        data = data_for(cases, relation={"a": "restatement"})

        score = guard.score_guard(rule(), data, "sonnet")

        assert score.result.disputed == ["a"]
        assert score.errors("strict") == 1
        assert score.errors("loose") == 0
        assert score.resolved("loose") == 1

    def test_only_restricts_scoring_and_the_fired_list(self) -> None:
        cases = {"a": case("a"), "b": case("b")}
        data = data_for(cases, relation={"a": "restatement", "b": "restatement"})

        score = guard.score_guard(rule(), data, "sonnet", frozenset({"b"}))

        assert score.fired == ["b"]


class TestLeaveOneSourceOut:
    def test_a_pair_without_a_relation_key_counts_in_the_held_out_bad_folds(self) -> None:
        cases = {"a": case("a"), "n": case("n")}
        grades = {
            ("a", "sonnet"): ALL_YES,
            ("n", "sonnet"): {**ALL_YES, "nothing_lost": False},
        }
        data = data_for(cases, relation={"a": "restatement"}, grades=grades)
        pairs = {"a": pair_row("a"), "n": pair_row("n", "bounce_log")}
        data = replace(data, inputs=replace(data.inputs, pairs=pairs))

        table, pooled = guard.loso_table([rule()], data, "sonnet", "strict")

        assert pooled["bad"] == 1
        assert pooled["restatements"] == 1
        assert "| bounce_log |" in table


class TestRanking:
    def make(
        self, restatements: int, replacements: int, bad: int, accepted: int = 0
    ) -> guard.GuardScore:
        result = fold.FoldScore(base=covers.Score(restatements=40, replacements=45))
        result.base.reinforced_restatement = restatements
        result.base.flagged_replacement = replacements
        result.bad = [f"b{i}" for i in range(bad)]
        result.triggered = [f"t{i}" for i in range(accepted)]

        return guard.GuardScore(result=result)

    def test_closeness_to_the_bars_outranks_bad_folds(self) -> None:
        close_but_bad = self.make(35, 42, 5)
        far_but_clean = self.make(30, 40, 0)

        assert guard.rank_reading(close_but_bad, "loose") < guard.rank_reading(
            far_but_clean, "loose"
        )

    def test_equal_shortfall_falls_back_to_fewest_bad_folds(self) -> None:
        assert guard.rank_reading(self.make(33, 42, 1), "strict") < guard.rank_reading(
            self.make(33, 42, 3), "strict"
        )

    def test_shortfall_ignores_surplus(self) -> None:
        assert guard.shortfall(40, 45) == 0
        assert guard.shortfall(30, 40) == 7

    def test_clean_rule_has_zero_errors_and_most_restatements(self) -> None:
        a, b, c = rule(), rule(floor=0.8), rule(floor=0.85)
        scored = [(a, self.make(30, 40, 0)), (b, self.make(34, 40, 2)), (c, self.make(32, 40, 0))]

        pick = guard.best_clean_rule(scored, "strict")

        assert pick is not None
        assert pick[0] is c
        assert guard.best_clean_rule([(b, self.make(34, 40, 2))], "strict") is None

    def test_best_rule_search_respects_only(self) -> None:
        cases = {"a": case("a", similarity=0.9), "b": case("b", similarity=0.76)}
        relation = {"a": "restatement", "b": "restatement"}
        grades = {("a", "sonnet"): ALL_YES, ("b", "sonnet"): {**ALL_YES, "one_record": False}}
        data = data_for(cases, relation=relation, grades=grades)
        rules = [rule(floor=0.75), rule(floor=0.85)]

        assert guard.best_guard_rule(rules, data, "sonnet", "strict").floor == 0.85
        assert (
            guard.best_guard_rule(rules, data, "sonnet", "strict", frozenset({"b"})).floor == 0.85
        )


class TestAccuracy:
    def test_each_guard_is_scored_against_its_matching_judge_answer(self) -> None:
        cases = {"a": case("a"), "b": case("b")}
        grades = {
            ("a", "sonnet"): {"nothing_lost": True, "nothing_invented": False, "one_record": True},
            ("b", "sonnet"): {"nothing_lost": False, "nothing_invented": True, "one_record": False},
        }
        guards = [
            signal("a", old=0.9, sources=0.9, single=0.9),
            signal("b", old=0.2, sources=0.9, single=0.2),
        ]
        data = data_for(cases, grades=grades, guards=guards)

        assert sorted(guard.guard_confusions(data, "fold_covers_old", None)) == [
            (False, 0.2),
            (True, 0.9),
        ]
        assert sorted(guard.guard_confusions(data, "sources_cover_fold", None)) == [
            (False, 0.9),
            (True, 0.9),
        ]
        assert guard.guard_confusions(data, "fold_single", "haiku") == []
        assert "fold_single | one_record | both | 2" in guard.guard_accuracy_table(data)

    def test_repeat_rows_count_flips_gaps_and_crossings(self) -> None:
        first = {"a": answer(0.9), "b": answer(0.6), "c": answer(None)}
        second = {"a": answer(0.65), "b": answer(0.62), "c": answer(0.5), "z": answer(0.1)}

        count, flips, mean, largest, crossings = guard.repeat_rows(first, second)

        assert (count, flips, crossings) == (2, 0, 1)
        assert abs(mean - 0.135) < 1e-9
        assert abs(largest - 0.25) < 1e-9


class TestRows:
    def test_signal_rows_round_trip_and_report_latency(self, tmp_path: pathlib.Path) -> None:
        row = signal("a")
        path = tmp_path / "guard-signals.jsonl"
        path.write_text(json.dumps(row.model_dump(mode="json")) + "\n", encoding="utf-8")

        loaded = guard.load_guard_signals(path)

        assert loaded[("a", "sonnet")].sequential_ms == 40.0
        assert loaded[("a", "sonnet")].parallel_ms == 10.0
        assert guard.load_guard_signals(tmp_path / "missing.jsonl") == {}

    def test_score_refuses_to_run_on_incomplete_signals(self, tmp_path: pathlib.Path) -> None:
        data = data_for({"a": case("a")}, out_dir=tmp_path)

        assert guard.unfinished(data) == (1, 1)

        done = data_for(
            {"a": case("a")},
            guards=[signal("a")],
            subjects=[subject("a", 0.9)],
            out_dir=tmp_path,
        )

        assert guard.unfinished(done) == (0, 0)


class TestRecount:
    def test_recount_from_raw_rows_matches_the_scorer(self, tmp_path: pathlib.Path) -> None:
        base = tmp_path
        (base / "experiments" / "covers").mkdir(parents=True)
        (base / "experiments" / "dup-rules").mkdir(parents=True)
        (base / "experiments" / "fold").mkdir(parents=True)
        (base / "experiments" / "fold-guard").mkdir(parents=True)

        def write(path: str, rows: list[dict[str, Any]]) -> None:
            (base / path).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

        specs = {"a": (0.9, "restatement"), "b": (0.82, "restatement"), "c": (0.78, "replacement")}
        write(
            "experiments/dup-rules/signals.jsonl",
            [
                {
                    "pair_id": pid,
                    "existing": {
                        "pick": "follow_up",
                        "similarity": sim,
                        "probabilities": {"replacement": 0.4 if pid == "c" else 0.0},
                    },
                }
                for pid, (sim, _) in specs.items()
            ],
        )
        write(
            "experiments/covers/signals.jsonl",
            [
                {
                    "pair_id": pid,
                    "new_covers_old": {"p_yes": 0.1},
                    "enhances": {"p_yes": 0.95 if pid == "a" else 0.75},
                }
                for pid in specs
            ],
        )
        write(
            "key-relation.jsonl",
            [{"item_id": pid, "label": label} for pid, (_, label) in specs.items()],
        )
        write("experiments/covers/key-covers.jsonl", [])
        write(
            "experiments/fold/folds.jsonl",
            [{"pair_id": pid, "writer": "sonnet", "text": "x"} for pid in specs],
        )
        fold_key = [
            {
                "item_id": f"{pid}~sonnet#{q}",
                "label": "no" if (pid, q) == ("b", "one_record") else "yes",
            }
            for pid in specs
            for q in recount.QUESTIONS
        ]
        write("experiments/fold/key-fold.jsonl", fold_key)
        write(
            "experiments/fold-guard/guard-signals.jsonl",
            [
                {
                    "pair_id": pid,
                    "writer": "sonnet",
                    **{
                        q: {"p_yes": 0.5 if (pid, q) == ("a", "fold_single") else 0.95}
                        for q in recount.GUARDS
                    },
                }
                for pid in specs
            ],
        )
        write(
            "experiments/fold-guard/band-signals.jsonl",
            [{"pair_id": pid, "same_subject": {"p_yes": 0.8}} for pid in specs],
        )

        line = recount.recount(base, "sonnet", 0.80, 0.70, None, 0.25, "all", 0.80)

        counts = dict(token.split("=") for token in line.split(": ")[1].split())

        assert counts["restatements"] == "0"
        assert counts["replacements"] == "1"
        assert counts["fired"] == "3"
        assert counts["accepted"] == "2"
        assert counts["bad"] == "1"
        assert counts["good_rejected"] == "1"
