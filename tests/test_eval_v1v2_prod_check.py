from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import eval_v1v2.covers as covers_mod
import eval_v1v2.fold as fold_mod
import eval_v1v2.prod_check as mod
import eval_v1v2.rows as rows_mod
from eval_v1v2.agent_fold_ask import AskRow
from eval_v1v2.rows import SaveRow


def save_row(pair_id: str, bucket: str, answer: dict[str, object] | None = None) -> SaveRow:
    return SaveRow.model_validate(
        {
            **({} if answer is None else {"answer": answer}),
            "pair_id": pair_id,
            "system": "v2",
            "bucket": bucket,
            "found_existing": True,
            "relation_pick": "follow_up",
            "raw_outcome": bucket,
            "elapsed_ms": 1.0,
            "degraded": None,
        }
    )


def answer_row(pair_id: str, replay: str, end_state: str, asked: bool = True) -> mod.AnswerRow:
    return mod.AnswerRow.model_validate(
        {
            "pair_id": pair_id,
            "asked": asked,
            "replay": replay,
            "status": "approved",
            "ask_answer": None,
            "end_state": end_state,
            "existing_live": True,
            "existing_superseded_by_fold": False,
            "fold_live": False,
            "held_live": True,
            "fold_covers_old": None,
            "fold_covers_new": None,
            "error": None,
        }
    )


def ask_row(pair_id: str, action: str | None, error: str | None = None) -> AskRow:
    return AskRow.model_validate(
        {
            "pair_id": pair_id,
            "arm": mod.REPLAY_ARM,
            "model": "opus",
            "wording": "fold_default",
            "framing": "in_task",
            "action": action,
            "note": "n" if action == "fold" else None,
            "reason": "r" if action == "keep_both" else None,
            "raw": None,
            "elapsed_ms": 1.0,
            "attempts": 1,
            "error": error,
        }
    )


def asks_of(*rows: AskRow) -> dict[tuple[str, str], AskRow]:
    return {(row.pair_id, row.arm): row for row in rows}


def bench(reinforce: int, covers: int, ask: int) -> tuple[dict[str, str], set[str]]:
    actions = {f"r{i}": "reinforce" for i in range(reinforce)}
    actions |= {f"s{i}": "supersede" for i in range(covers)}
    actions |= {f"a{i}": "new" for i in range(ask)}

    return actions, {f"s{i}" for i in range(covers)} | {f"a{i}" for i in range(ask)}


def covers_save(pair_id: str, answer: dict[str, object] | None = None) -> SaveRow:
    return save_row(pair_id, "asked", answer).model_copy(
        update={"raw_outcome": f"asked:covers:{pair_id}-existing"}
    )


def matching_save(reinforce: int = 24, covers: int = 9, ask: int = 38) -> list[SaveRow]:
    return (
        [save_row(f"r{i}", "caught") for i in range(reinforce)]
        + [covers_save(f"s{i}") for i in range(covers)]
        + [save_row(f"a{i}", "asked") for i in range(ask)]
    )


def matching_answers(
    ask: int = 38, covers: int = 9
) -> tuple[list[mod.AnswerRow], dict[tuple[str, str], AskRow]]:
    rows: list[mod.AnswerRow] = []
    asks: list[AskRow] = []
    pairs = [f"s{i}" for i in range(covers)] + [f"a{i}" for i in range(ask)]

    for i, pair_id in enumerate(pairs):
        action = ("fold", "keep_both", "continue")[i % 3]
        end = "folded" if action == "fold" else "both_live"

        rows.append(answer_row(pair_id, action, end))
        asks.append(ask_row(pair_id, action))

    return rows, asks_of(*asks)


def test_matching_replay_has_no_findings() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks)

    assert check.findings == []
    assert check.ok
    assert check.end_states == {"folded": 16, "both_live": 31}
    assert [bound.ok for bound in check.bounds] == [True, True, True]
    assert check.bounds[2].expected == 47
    assert check.covers_asked == sorted(f"s{i}" for i in range(9))


def test_counts_inside_the_tolerance_pass_and_outside_fail() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers(36)
    inside = mod.run_check(matching_save(25, 9, 36), answers, actions, bench_ask, asks)
    outside = mod.run_check(
        matching_save(24, 9, 35), matching_answers(35)[0], actions, bench_ask, asks
    )

    assert all(bound.ok for bound in inside.bounds)
    assert [finding for finding in outside.findings if finding.startswith("asked is 44")]
    assert not outside.ok


def test_any_supersede_is_a_finding_and_breaks_the_zero_bound() -> None:
    actions, bench_ask = bench(24, 9, 0)
    one = [*matching_save(24, 9, 0), save_row("x1", "superseded")]
    check = mod.run_check(one, [], actions, bench_ask, {})

    assert check.superseded == ["x1"]
    assert check.bounds[1].expected == 0
    assert check.bounds[1].tolerance == 0
    assert not check.bounds[1].ok
    assert [f for f in check.findings if "superseded without a fold answer" in f]
    assert not check.ok


def test_a_supersede_near_a_threshold_is_still_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 0)
    probs = seed_probs(x1=(0.94, None))
    check = mod.run_check(
        [*matching_save(24, 9, 0), save_row("x1", "superseded")],
        [],
        actions,
        bench_ask,
        {},
        probs=probs,
    )

    assert check.superseded == ["x1"]
    assert not check.ok


def test_a_missing_bench_ask_is_listed_for_both_sets() -> None:
    actions, bench_ask = bench(24, 9, 38)
    save = [row for row in matching_save() if row.pair_id not in ("s0", "a0")]
    check = mod.run_check(save, [], actions, bench_ask, {})

    assert check.asked_missing == ["a0", "s0"]


def test_a_covers_pair_asked_through_the_trigger_is_not_counted_as_a_covers_ask() -> None:
    actions, bench_ask = bench(24, 9, 38)
    save = [save_row("s0", "asked") if row.pair_id == "s0" else row for row in matching_save()]
    check = mod.run_check(save, [], actions, bench_ask, {})

    assert "s0" not in check.covers_asked
    assert "s0" in check.asked


def with_answer(answers: list[mod.AnswerRow], row: mod.AnswerRow) -> list[mod.AnswerRow]:
    return [row if existing.pair_id == row.pair_id else existing for existing in answers]


def test_end_state_that_differs_from_the_recorded_reply_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    answers = with_answer(answers, answer_row("a0", "fold", "both_live"))
    answers = with_answer(answers, answer_row("a1", "keep_both", "both_live", asked=False))
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks)

    assert len(check.end_state_mismatches) == 2
    assert check.end_state_mismatches[0].startswith("a0: both_live")
    assert [f for f in check.findings if "wrong end state" in f]


def test_an_asked_pair_without_an_answer_row_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    answers = [row for row in answers if row.pair_id != "a0"]
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks)

    assert check.asked_without_answer_row == ["a0"]
    assert not check.ok


def test_an_errored_reply_expects_no_reply() -> None:
    asks = asks_of(ask_row("a0", None, error="down"), ask_row("a1", "continue"))

    assert mod.replay_of(asks[("a0", mod.REPLAY_ARM)]) == "none"
    assert mod.replay_of(asks[("a1", mod.REPLAY_ARM)]) == "continue"
    assert mod.replay_of(None) == "none"
    assert mod.EXPECTED_END["none"] == "no_reply"


def noise_row(pair_id: str, outcome: str) -> SaveRow:
    return save_row(pair_id, "wrong_target").model_copy(update={"raw_outcome": outcome})


def judged_row(pair_id: str, event: str, noise_id: str, **answers: str | None) -> mod.JudgedRow:
    return mod.JudgedRow.model_validate(
        {
            "pair_id": pair_id,
            "event": event,
            "noise_id": noise_id,
            "new_covers_old": answers.get("new_covers_old"),
            "one_record": answers.get("one_record"),
        }
    )


def test_events_on_noise_rows_are_unkeyed_with_their_verdicts_and_not_findings() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [
        *matching_save(),
        noise_row("w", "approved:supersede:noise-1"),
        noise_row("c", "asked:covers:noise-2"),
        noise_row("k", "asked:noise-3"),
        noise_row("i", "asked:covers:noise-4"),
    ]
    judged = [
        judged_row("c", "asked", "noise-2", one_record="yes"),
        judged_row("k", "asked", "noise-3", new_covers_old="no", one_record="no"),
        judged_row("i", "asked", "noise-4", one_record=None),
    ]
    check = mod.run_check(save, answers, actions, bench_ask, asks, judged=judged)
    by_pair = {event.pair_id: event for event in check.unkeyed}

    assert check.ok
    assert check.wrong_target == ["c", "i", "k", "w"]
    assert sorted(by_pair) == ["c", "i", "k", "w"]
    assert by_pair["w"].kind == "relation suggestion"
    assert by_pair["w"].verdict == "not judged"
    assert by_pair["c"].kind == "ask"
    assert by_pair["c"].verdict == "one_record key yes"
    assert by_pair["k"].verdict == "new_covers_old key no, one_record key no"
    assert by_pair["i"].verdict == "one_record key no agreement"
    assert "## Events on noise rows: 4" in mod.render(check)
    assert "- c: ask on noise-2; one_record key yes" in mod.render(check)


def shortfall_check(events: list[SaveRow], judged: list[mod.JudgedRow]) -> mod.Check:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [row for row in matching_save() if row.pair_id not in ("s0", "s1", "s2")]

    return mod.run_check([*save, *events], answers, actions, bench_ask, asks, judged=judged)


def test_a_noise_only_ask_keyed_yes_lowers_expected_and_never_counts_as_asked() -> None:
    judged = [
        judged_row("s0", "asked", "n1", one_record="yes"),
        judged_row("s1", "asked", "n2", one_record="yes"),
    ]
    plain = shortfall_check([], judged)
    one = shortfall_check([noise_row("s0", "asked:covers:n1")], judged)
    two = shortfall_check(
        [noise_row("s0", "asked:covers:n1"), noise_row("s1", "asked:covers:n2")], judged
    )

    assert [check.bounds[2].actual for check in (plain, one, two)] == [44, 44, 44]
    assert [check.bounds[2].expected for check in (plain, one, two)] == [47, 46, 45]
    assert [check.bounds[2].ok for check in (plain, one, two)] == [False, True, True]
    assert two.asked_on_noise == ["s0", "s1"]
    assert two.noise_only_asks == ["s0: one_record key yes", "s1: one_record key yes"]


def test_a_noise_only_ask_keyed_no_leaves_the_shortfall_and_is_listed_with_its_verdict() -> None:
    judged = [judged_row("s0", "asked", "n1", one_record="no")]
    check = shortfall_check([noise_row("s0", "asked:covers:n1")], judged)

    assert check.bounds[2].actual == 44
    assert check.bounds[2].expected == 47
    assert not check.bounds[2].ok
    assert check.asked_on_noise == []
    assert check.noise_only_asks == ["s0: one_record key no"]
    assert "s0: one_record key no" in mod.render(check)


@pytest.mark.parametrize(
    "judged",
    [
        [],
        [judged_row("s0", "asked", "n1", one_record=None)],
        [judged_row("s0", "asked", "n1", one_record="no")],
        [judged_row("s0", "contained", "n1", new_covers_old="yes")],
    ],
)
def test_an_ask_on_noise_without_a_one_record_yes_excuses_no_shortfall(
    judged: list[mod.JudgedRow],
) -> None:
    check = shortfall_check([noise_row("s0", "asked:covers:n1")], judged)

    assert check.bounds[2].adjustment == 0


def test_a_relation_suggestion_on_noise_excuses_no_shortfall() -> None:
    check = shortfall_check([noise_row("s0", "approved:supersede:n1")], [])

    assert check.bounds[2].adjustment == 0


def test_an_ask_on_a_noise_row_is_listed_for_a_bench_pair_only_and_is_not_counted() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [row for row in matching_save() if row.pair_id not in ("a0", "s0")]
    events = [noise_row("a0", "asked:n1"), noise_row("outside", "asked:n2")]
    check = mod.run_check([*save, *events], answers, actions, bench_ask, asks)

    assert check.bounds[2].actual == 45
    assert check.noise_only_asks == ["a0: not judged"]
    assert check.asked_on_noise == []


def test_a_suppressed_near_threshold_pair_is_not_credited_a_second_time() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    suppressed = ("a0", "a1", "a2")
    missing = ("a3", "a4", "a5")
    save = [row for row in matching_save() if row.pair_id not in (*suppressed, *missing)]
    events = [noise_row(pair, "approved:supersede:n1") for pair in suppressed]
    probs = seed_probs(**dict.fromkeys(suppressed, (0.96, None)))
    check = mod.run_check([*save, *events], answers, actions, bench_ask, asks, probs=probs)

    assert check.ask_suppressed == list(suppressed)
    assert check.near_flips == []
    assert check.bounds[2].actual == 41
    assert check.bounds[2].expected == 44
    assert check.bounds[2].adjustment == 0
    assert not check.bounds[2].ok


def test_a_replacement_suggestion_on_noise_lowers_the_asked_bench_and_is_named() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    gone = ("a0", "a1", "a2")
    save = [row for row in matching_save() if row.pair_id not in gone]
    events = [noise_row(pair, "approved:supersede:n1") for pair in gone]
    check = mod.run_check([*save, *events], answers, actions, bench_ask, asks)

    assert check.bounds[2].actual == 44
    assert check.bounds[2].expected == 44
    assert check.ask_suppressed == ["a0", "a1", "a2"]
    assert check.bounds[2].ok
    assert "bench asks suppressed by a replacement suggestion" in mod.render(check)
    assert "(a0, a1, a2)" in mod.render(check)


def test_a_reinforce_suggestion_on_noise_does_not_suppress_an_ask() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [row for row in matching_save() if row.pair_id != "a0"]
    check = mod.run_check(
        [*save, noise_row("a0", "reinforced:reinforce:n1")], answers, actions, bench_ask, asks
    )

    assert check.ask_suppressed == []
    assert check.bounds[2].expected == 47


def test_the_finding_prints_the_adjusted_count_the_bound_compares() -> None:
    bound = mod.Bound(name="asked", expected=44, tolerance=2, actual=49, adjustment=-2)
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks)
    findings = mod.findings_for(check.model_copy(update={"bounds": [bound]}))

    assert bound.judged == 47
    assert findings == ["asked is 49, 47 after the -2 explained-pair adjustment, bench 44 +-2"]


def test_a_containment_supersede_on_a_noise_row_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [*matching_save(), noise_row("b", "approved:contained:noise-9")]
    check = mod.run_check(save, answers, actions, bench_ask, asks)

    assert check.contained_on_noise == ["b on noise-9"]
    assert not check.ok
    assert any("containment superseded a noise row" in finding for finding in check.findings)
    assert "containment superseded a noise row: 1" in mod.render(check)


def seed_probs(**values: tuple[float | None, float | None]) -> dict[str, mod.SeedProbs]:
    return {
        pair: mod.SeedProbs(new_covers_old=cover, same_subject=subject)
        for pair, (cover, subject) in values.items()
    }


def test_near_threshold_is_within_the_margin_of_either_floor() -> None:
    probs = seed_probs(
        a=(0.95, None),
        b=(0.89, None),
        c=(0.88, None),
        d=(None, 0.76),
        e=(None, 0.77),
        f=(0.80, None),
        g=(0.99, None),
    )

    assert mod.near_threshold(probs, 0.95) == {"a", "b", "d", "g"}
    assert mod.near_threshold(probs, 0.90) == {"a", "b", "c", "d"}


def test_a_flip_near_a_threshold_is_reported_and_is_not_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [row for row in matching_save() if row.pair_id not in ("s0", "a36")]
    save += [save_row("s0", "caught"), save_row("a38", "asked")]
    answers = [*answers, answer_row("a38", "keep_both", "both_live")]
    probs = seed_probs(s0=(0.96, 0.2), a36=(0.1, 0.72), s5=(0.99, None))
    check = mod.run_check(
        save,
        answers,
        actions,
        bench_ask,
        asks_of(*asks.values(), ask_row("a38", "keep_both")),
        probs=probs,
    )

    assert check.near_threshold == ["a36", "s0", "s5"]
    assert check.near_flips == ["a36", "s0"]
    assert check.near_flip_notes[0].startswith(
        "a36 (not asked): bench new_covers_old 0.10, same_subject 0.72"
    )
    assert check.bounds[2].adjustment == 2
    assert check.ok


def test_a_flip_far_from_every_threshold_still_counts() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    extra = [save_row(f"x{i}", "asked") for i in range(1, 5)]
    near = seed_probs(x1=(0.90, None), x2=(0.5, None), x3=(0.5, None), x4=(0.5, None))
    check = mod.run_check([*matching_save(), *extra], answers, actions, bench_ask, asks, probs=near)

    assert check.near_flips == ["x1"]
    assert check.bounds[2].actual == 51
    assert check.bounds[2].adjustment == -1
    assert not check.bounds[2].ok
    assert not check.ok


def test_the_live_seed_answers_are_shown_when_the_row_carries_them() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    seeded = save_row("s0", "caught").model_copy(
        update={
            "neighbours": [
                rows_mod.NeighbourRow(
                    id="e",
                    source="v2",
                    similarity=0.9,
                    relation_pick="follow_up",
                    relation_confidence=0.8,
                    new_covers_old=0.91,
                    same_subject=0.2,
                    is_seed=True,
                )
            ]
        }
    )
    save = [row for row in matching_save() if row.pair_id != "s0"]
    check = mod.run_check(
        [*save, seeded],
        answers,
        actions,
        bench_ask,
        asks,
        probs=seed_probs(s0=(0.96, 0.2)),
    )

    assert "live new_covers_old 0.91, same_subject 0.20" in check.near_flip_notes[0]


def case(pair_id: str, p_yes: float, pick: str = "follow_up") -> covers_mod.Case:
    answer = {"answer": "yes", "p_yes": p_yes, "probabilities": None, "error": None}
    no = {"answer": "no", "p_yes": 0.0, "probabilities": None, "error": None}

    return covers_mod.Case(
        pair_id=pair_id,
        pick=pick,
        similarity=0.9,
        in_band=True,
        covers=covers_mod.CoversSignal.model_validate(
            {
                "pair_id": pair_id,
                "run": 1,
                "new_covers_old": answer,
                "old_covers_new": no,
                "enhances": no,
                "elapsed_ms": 1.0,
                "error": None,
            }
        ),
        sonnet=None,
    )


def test_the_bench_covers_set_follows_the_floor() -> None:
    cases = {
        "p93": case("p93", 0.93),
        "p95": case("p95", 0.95),
        "r": case("r", 0.99, "restatement"),
    }

    assert mod.decide_all(cases, 0.95) == {"p93": "new", "p95": "supersede", "r": "reinforce"}
    assert mod.decide_all(cases, 0.90)["p93"] == "supersede"
    assert mod.covers_set(mod.decide_all(cases, 0.95)) == {"p95"}
    assert mod.covers_set(mod.decide_all(cases, 0.90)) == {"p93", "p95"}
    assert mod.CONTAINMENT_FLOOR == 0.95


def test_the_reinforce_bound_is_24_plus_or_minus_2() -> None:
    actions, bench_ask = bench(24, 9, 0)

    for count, within in ((22, True), (26, True), (21, False), (27, False)):
        check = mod.run_check(matching_save(count, 9, 0), [], actions, bench_ask, {})

        assert check.bounds[0].ok is within


def test_render_lists_counts_and_findings() -> None:
    actions, bench_ask = bench(24, 9, 38)
    check = mod.run_check(matching_save(24, 9, 30), [], actions, bench_ask, {})
    text = mod.render(check)

    assert "| asked | 39 | 47 | 2 | +0 | 39 | NO |" in text
    assert "asked is 39, bench 47 +-2" in text
    assert "- asked through covers: 9" in text


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_band_fixture(tmp_path: Path, similarity: float, p_yes: float, pick: str) -> None:
    write_jsonl(
        tmp_path / "experiments" / "fold" / "band.jsonl",
        [
            {
                "pair_id": "p",
                "source": "jev_pairs",
                "relation": None,
                "similarity": similarity,
                "pick": pick,
                "enhances": None,
                "existing": "e",
                "incoming": "i",
            }
        ],
    )
    timed = {
        "answer": "yes",
        "p_yes": p_yes,
        "probabilities": {"yes": p_yes, "no": 1 - p_yes},
        "error": None,
        "elapsed_ms": 1.0,
    }
    write_jsonl(
        tmp_path / "experiments" / "fold-guard" / "band-signals.jsonl",
        [{"pair_id": "p", "run": 1, "same_subject": timed, "error": None}],
    )


@pytest.mark.parametrize(
    ("similarity", "p_yes", "pick", "action", "asked"),
    [
        (0.80, 0.70, "follow_up", "new", True),
        (0.79, 0.90, "follow_up", "new", False),
        (0.90, 0.69, "follow_up", "new", False),
        (0.90, 0.90, "distinct", "new", False),
        (0.90, 0.90, "follow_up", "supersede", False),
        (0.90, 0.90, "follow_up", "flag", False),
    ],
)
def test_trigger_asked_applies_the_trigger(
    tmp_path: Path, similarity: float, p_yes: float, pick: str, action: str, asked: bool
) -> None:
    write_band_fixture(tmp_path, similarity, p_yes, pick)

    assert mod.trigger_asked(tmp_path, {"p": action}) == ({"p"} if asked else set())


def test_the_bench_ask_set_is_the_trigger_set_plus_the_covers_set(tmp_path: Path) -> None:
    write_band_fixture(tmp_path, 0.90, 0.90, "follow_up")
    actions = {"p": "new", "outside": "supersede", "other": "reinforce"}

    assert mod.bench_asked(tmp_path, actions) == {"p", "outside"}
    assert mod.bench_asked(tmp_path, {"p": "supersede", "x": "new"}) == {"p"}


def test_answer_rows_with_and_without_the_raw_fields_both_load() -> None:
    old = answer_row("a0", "fold", "folded")
    raw = mod.AnswerRow.model_validate(
        {
            **old.model_dump(),
            "neighbours": [],
            "ask": {
                "existing_id": "e",
                "similarity": 0.9,
                "same_subject": 0.8,
                "new_covers_old": 0.1,
                "is_seed": True,
            },
        }
    )

    assert old.ask is None
    assert raw.ask is not None
    assert raw.ask.is_seed


def test_an_explained_extra_ask_lowers_the_count_it_is_judged_on() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    extra = [save_row("x1", "asked"), save_row("x2", "asked")]
    probs = seed_probs(x1=(0.93, None), x2=(0.92, None))
    check = mod.run_check(
        [*matching_save(), *extra], answers, actions, bench_ask, asks, probs=probs
    )

    assert check.bounds[2].actual == 49
    assert check.bounds[2].adjustment == -2
    assert check.bounds[2].ok


def converted_to_new(count: int, covers: int = 9) -> list[SaveRow]:
    return [save_row(f"s{i}", "new") if i < count else covers_save(f"s{i}") for i in range(covers)]


def test_every_covers_pair_saved_as_new_is_a_finding_and_exits_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers(38, 0)
    save = [*matching_save(24, 0, 38), *converted_to_new(9)]
    probs = seed_probs(**{f"s{i}": (0.96, None) for i in range(9)})
    check = mod.run_check(save, answers, actions, bench_ask, asks, probs=probs)

    assert check.covers_asked == []
    assert check.asked_missing == [f"s{i}" for i in range(9)]
    assert [f for f in check.findings if f.startswith("asked is 38")]
    assert not check.ok

    monkeypatch.setattr(mod, "run", lambda *args, **kwargs: check)
    monkeypatch.setattr(sys, "argv", ["prod_check", "--replay", "/tmp"])
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: 0)

    with pytest.raises(SystemExit) as exited:
        mod.main()

    assert exited.value.code == 1


def test_near_excuses_cover_at_most_half_of_the_bench_covers_set() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers(38, 2)
    probs = seed_probs(**{f"s{i}": (0.96, None) for i in range(9)})
    save = [*matching_save(24, 0, 38), *converted_to_new(7)]
    check = mod.run_check(save, answers, actions, bench_ask, asks, probs=probs)

    assert len(check.asked) == 40
    assert len(check.near_flips) == 4
    assert check.bounds[2].adjustment == 4
    assert not check.bounds[2].ok
    assert not check.ok


def test_a_near_excuse_needs_the_bench_probability_on_the_opposite_side() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    extra = [save_row("x1", "asked"), save_row("x2", "asked")]
    probs = seed_probs(x1=(0.96, None), x2=(0.94, None))
    check = mod.run_check(
        [*matching_save(), *extra], answers, actions, bench_ask, asks, probs=probs
    )

    assert check.near_flips == ["x2"]
    assert check.asked_outside_bench == ["x1", "x2"]
    assert check.bounds[2].adjustment == -1


def test_a_failed_answer_pass_on_a_near_threshold_pair_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    answers = with_answer(answers, answer_row("a0", "fold", "failed"))
    probs = seed_probs(a0=(0.2, 0.72))
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks, probs=probs)

    assert "a0" in check.near_threshold
    assert check.end_state_mismatches[0].startswith("a0: failed")
    assert not check.ok


def test_a_missing_answer_row_on_a_near_threshold_pair_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    answers = [row for row in answers if row.pair_id != "a0"]
    probs = seed_probs(a0=(0.2, 0.72))
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks, probs=probs)

    assert "a0" in check.near_threshold
    assert check.asked_without_answer_row == ["a0"]
    assert not check.ok


def outcome(replay: str, end_state: str, held_live: bool = True) -> dict[str, object]:
    return {
        "replay": replay,
        "status": "approved",
        "ask_answer": None,
        "end_state": end_state,
        "existing_live": True,
        "existing_superseded_by_fold": False,
        "fold_live": False,
        "held_live": held_live,
        "fold_covers_old": None,
        "fold_covers_new": None,
        "error": None,
    }


def one_pass_save(states: dict[str, tuple[str, str]]) -> list[SaveRow]:
    rows = [row for row in matching_save() if row.bucket == "caught"]

    return rows + [
        covers_save(pair, outcome(*state))
        if pair.startswith("s")
        else save_row(pair, "asked", outcome(*state))
        for pair, state in states.items()
    ]


def one_pass_states() -> dict[str, tuple[str, str]]:
    states: dict[str, tuple[str, str]] = {}
    pairs = [f"s{i}" for i in range(9)] + [f"a{i}" for i in range(38)]

    for i, pair in enumerate(pairs):
        action = ("fold", "keep_both", "continue")[i % 3]
        states[pair] = (action, "folded" if action == "fold" else "both_live")

    return states


def one_pass_asks() -> dict[tuple[str, str], AskRow]:
    pairs = [f"s{i}" for i in range(9)] + [f"a{i}" for i in range(38)]

    return asks_of(
        *[ask_row(pair, ("fold", "keep_both", "continue")[i % 3]) for i, pair in enumerate(pairs)]
    )


def test_end_states_carried_on_the_save_rows_need_no_answer_file() -> None:
    actions, bench_ask = bench(24, 9, 38)
    save = one_pass_save(one_pass_states())
    check = mod.run_check(save, [], actions, bench_ask, one_pass_asks())

    assert check.findings == []
    assert check.end_states == {"folded": 16, "both_live": 31}
    assert check.no_reply == []


def test_an_asked_save_row_without_an_answer_and_without_an_answer_file_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    save = [row for row in one_pass_save(one_pass_states()) if row.pair_id != "a0"]
    save.append(save_row("a0", "asked"))
    check = mod.run_check(save, [], actions, bench_ask, one_pass_asks())

    assert check.asked_without_answer_row == ["a0"]
    assert not check.ok


def test_no_reply_on_a_pair_the_bench_never_asked_is_reported_and_is_not_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    states = one_pass_states() | {"a0": ("none", "no_reply"), "a5": ("none", "no_reply")}
    asks = one_pass_asks()
    del asks[("a0", mod.REPLAY_ARM)]
    asks[("a5", mod.REPLAY_ARM)] = ask_row("a5", None, error="down")
    check = mod.run_check(one_pass_save(states), [], actions, bench_ask, asks)

    assert check.no_reply == ["a0", "a5"]
    assert check.findings == []
    assert check.ok
    assert (
        "## Asked with no recorded reply: 2, expired and saved as new, not findings"
        in mod.render(check)
    )


@pytest.mark.parametrize(
    ("reply", "end_state"),
    [("fold", "no_reply"), ("keep_both", "no_reply"), ("continue", "no_reply")],
)
def test_no_reply_on_a_pair_with_a_recorded_reply_is_a_finding(reply: str, end_state: str) -> None:
    actions, bench_ask = bench(24, 9, 38)
    states = one_pass_states() | {"a0": ("none", end_state)}
    asks = one_pass_asks() | {("a0", mod.REPLAY_ARM): ask_row("a0", reply)}
    check = mod.run_check(one_pass_save(states), [], actions, bench_ask, asks)

    assert check.no_reply == []
    assert check.end_state_mismatches == [
        "a0: no_reply (asked=True), expected " + mod.EXPECTED_END[reply]
    ]
    assert not check.ok


def test_a_rejected_answer_with_a_recorded_reply_is_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    states = one_pass_states() | {"a0": ("fold", "failed")}
    check = mod.run_check(one_pass_save(states), [], actions, bench_ask, one_pass_asks())

    assert check.end_state_mismatches == ["a0: failed (asked=True), expected folded"]
    assert not check.ok


@pytest.mark.parametrize(
    ("reply", "end_state"),
    [("fold", "both_live"), ("keep_both", "folded"), ("continue", "folded")],
)
def test_an_end_state_that_does_not_match_the_reply_is_a_finding(
    reply: str, end_state: str
) -> None:
    actions, bench_ask = bench(24, 9, 38)
    states = one_pass_states() | {"a0": (reply, end_state)}
    asks = one_pass_asks() | {("a0", mod.REPLAY_ARM): ask_row("a0", reply)}
    check = mod.run_check(one_pass_save(states), [], actions, bench_ask, asks)

    assert len(check.end_state_mismatches) == 1
    assert not check.ok


def test_old_answer_rows_still_count_a_not_asked_pair_as_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    answers = with_answer(answers, answer_row("a0", "none", "not_asked", asked=False))
    check = mod.run_check(matching_save(), answers, actions, bench_ask, asks)

    assert check.end_state_mismatches[0].startswith("a0: not_asked")
    assert not check.ok


def test_no_reply_is_accepted_only_when_the_held_note_is_live() -> None:
    actions, bench_ask = bench(24, 9, 38)
    asks = one_pass_asks()
    del asks[("a0", mod.REPLAY_ARM)]
    live = one_pass_states() | {"a0": ("none", "no_reply")}
    dead_states = {pair: outcome(*state, held_live=pair != "a0") for pair, state in live.items()}
    dead = [row for row in one_pass_save(live) if row.bucket == "caught"] + [
        covers_save(pair, state) if pair.startswith("s") else save_row(pair, "asked", state)
        for pair, state in dead_states.items()
    ]
    accepted = mod.run_check(one_pass_save(live), [], actions, bench_ask, asks)
    rejected = mod.run_check(dead, [], actions, bench_ask, asks)

    assert accepted.ok
    assert rejected.end_state_mismatches == [
        "a0: no_reply (asked=True, held note not live), expected no_reply"
    ]
    assert not rejected.ok


def test_a_current_inline_answer_wins_over_a_legacy_answer_row_for_the_same_pair() -> None:
    save = [save_row("a0", "asked", outcome("fold", "failed"))]
    legacy = [answer_row("a0", "fold", "folded")]

    merged = mod.answers_of(save, legacy)

    assert merged["a0"].end_state == "failed"
    assert mod.answers_of([save_row("a1", "asked")], legacy)["a0"].end_state == "folded"


def graded(**answers: bool) -> fold_mod.FoldGrade:
    return fold_mod.FoldGrade(answers=answers)


GOOD = {"nothing_lost": True, "nothing_invented": True, "newer_wins": True, "one_record": True}


def test_fold_grades_count_the_folded_pairs_and_list_the_bad_ones() -> None:
    answers = {
        "g": mod.AnswerRow.model_validate(
            {"pair_id": "g", "asked": True, **outcome("fold", "folded")}
        ),
        "b": mod.AnswerRow.model_validate(
            {"pair_id": "b", "asked": True, **outcome("fold", "folded")}
        ),
        "d": mod.AnswerRow.model_validate(
            {"pair_id": "d", "asked": True, **outcome("fold", "folded")}
        ),
        "k": mod.AnswerRow.model_validate(
            {"pair_id": "k", "asked": True, **outcome("keep_both", "both_live")}
        ),
    }
    grades = {
        "g": graded(**GOOD),
        "b": graded(**{**GOOD, "nothing_lost": False}),
        "d": graded(nothing_lost=True),
        "k": graded(**{**GOOD, "one_record": False}),
    }

    counts, bad, ungraded, screened = mod.fold_grade_counts(answers, grades)

    assert counts == {"good": 1, "bad": 1, "disputed": 1}
    assert bad == ["b"]
    assert ungraded == []
    assert screened == []


def test_a_folded_pair_with_no_grade_is_a_finding_only_when_grades_are_supplied() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    every = {row.pair_id: graded(**GOOD) for row in answers if row.end_state == "folded"}
    missing = {pair: grade for pair, grade in every.items() if pair != "a0"}
    without = mod.run_check(matching_save(), answers, actions, bench_ask, asks)
    complete = mod.run_check(matching_save(), answers, actions, bench_ask, asks, grades=every)
    partial = mod.run_check(matching_save(), answers, actions, bench_ask, asks, grades=missing)

    assert without.ok
    assert complete.ok
    assert complete.fold_grades == {"good": 16}
    assert partial.ungraded_folds == ["a0"]
    assert not partial.ok


def test_grades_come_from_both_keys_and_replies_from_both_ask_files(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    arm = mod.REPLAY_ARM

    def key_rows(pair_id: str, label: str) -> list[dict[str, object]]:
        return [
            {"item_id": f"{pair_id}~{arm}#{question}", "label": label, "source": "agreed"}
            for question in ("nothing_lost", "nothing_invented", "newer_wins", "one_record")
        ]

    write_jsonl(experiments / "agent-fold" / "key-fold.jsonl", key_rows("band", "yes"))
    write_jsonl(
        experiments / "prod-replay" / mod.COVERS_FOLD_DIR / "key-fold.jsonl",
        key_rows("outside", "no"),
    )

    grades = mod.load_grades(tmp_path, arm)

    assert grades["band"].status == "good"
    assert grades["outside"].status == "bad"

    reply = {
        "model": "opus",
        "wording": "fold_default",
        "framing": "in_task",
        "action": "fold",
        "note": "n",
        "reason": None,
        "raw": None,
        "elapsed_ms": 1.0,
        "attempts": 1,
        "error": None,
        "arm": arm,
    }
    write_jsonl(experiments / "agent-fold" / "asks.jsonl", [{**reply, "pair_id": "band"}])
    write_jsonl(
        experiments / "prod-replay" / mod.COVERS_ASKS_FILE,
        [{**reply, "pair_id": "outside"}, {**reply, "pair_id": "band", "action": "keep_both"}],
    )

    asks = mod.load_all_asks(tmp_path)

    assert sorted(pair for pair, _ in asks) == ["band", "outside"]
    assert asks[("band", arm)].action == "fold"
    assert asks[("outside", arm)].action == "fold"


def test_pairs_needing_replies_are_the_covers_set_and_the_replayed_supersedes_without_a_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mod,
        "bench_actions",
        lambda base, floor: {"c1": "supersede", "c2": "supersede", "n": "new", "r": "reinforce"},
    )
    replay = tmp_path / "experiments" / "prod-replay"

    def write_save(path: Path, *rows: SaveRow) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")

    write_save(replay / "save-v2.jsonl", save_row("first", "superseded"))
    write_save(
        replay / "run-1" / "save-v2.jsonl", save_row("late", "superseded"), save_row("x", "new")
    )
    write_save(replay / "run-9" / "save-v2.jsonl", save_row("c1", "asked"))
    reply = ask_row("c2", "fold").model_dump(mode="json")
    write_jsonl(tmp_path / "experiments" / "agent-fold" / "asks.jsonl", [reply])

    assert mod.superseded_in_runs(replay) == {"first", "late"}
    assert mod.pairs_needing_replies(tmp_path, replay) == ["c1", "first", "late"]

    reply["action"] = None
    reply["error"] = "down"
    write_jsonl(tmp_path / "experiments" / "agent-fold" / "asks.jsonl", [reply])

    assert "c2" in mod.pairs_needing_replies(tmp_path, replay)


def folded_with_grades(skip: str) -> tuple[list[mod.AnswerRow], dict[str, fold_mod.FoldGrade]]:
    answers, _ = matching_answers()
    every = {row.pair_id: graded(**GOOD) for row in answers if row.end_state == "folded"}

    return answers, {pair: grade for pair, grade in every.items() if pair != skip}


def test_a_fold_the_client_screen_kept_out_is_reported_apart_and_is_not_a_finding() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, grades = folded_with_grades("a0")
    asks = matching_answers()[1]
    check = mod.run_check(
        matching_save(), answers, actions, bench_ask, asks, grades=grades, screened={"a0"}
    )

    assert check.screened_folds == ["a0"]
    assert check.ungraded_folds == []
    assert check.ok
    assert "not gradable (client-name screen): 1 (a0)" in mod.render(check)


def test_a_fold_with_no_grade_for_another_reason_stays_a_finding_beside_a_screened_one() -> None:
    actions, bench_ask = bench(24, 9, 38)
    answers, grades = folded_with_grades("a0")
    grades = {pair: grade for pair, grade in grades.items() if pair != "a3"}
    asks = matching_answers()[1]
    check = mod.run_check(
        matching_save(), answers, actions, bench_ask, asks, grades=grades, screened={"a0"}
    )

    assert check.screened_folds == ["a0"]
    assert check.ungraded_folds == ["a3"]
    assert not check.ok


def test_the_screen_left_out_list_is_read_from_the_covers_fold_judge_dir(tmp_path: Path) -> None:
    assert mod.load_screened(tmp_path, "arm-a") == set()

    directory = tmp_path / "experiments" / "prod-replay" / mod.COVERS_FOLD_DIR
    directory.mkdir(parents=True)
    (directory / mod.left_out_file("arm-a")).write_text("x\ny\n", encoding="utf-8")

    assert mod.load_screened(tmp_path, "arm-a") == {"x", "y"}


def test_a_left_out_list_written_for_another_arm_does_not_exempt_a_fold(tmp_path: Path) -> None:
    directory = tmp_path / "experiments" / "prod-replay" / mod.COVERS_FOLD_DIR
    directory.mkdir(parents=True)
    (directory / mod.left_out_file("arm-a")).write_text("a0\n", encoding="utf-8")
    screened = mod.load_screened(tmp_path, "arm-b")
    actions, bench_ask = bench(24, 9, 38)
    answers, grades = folded_with_grades("a0")
    grades = {pair: grade for pair, grade in grades.items() if pair != "a0"}
    asks = matching_answers()[1]
    check = mod.run_check(
        matching_save(), answers, actions, bench_ask, asks, grades=grades, screened=screened
    )

    assert screened == set()
    assert check.ungraded_folds == ["a0"]
    assert not check.ok


@pytest.mark.parametrize(
    "judged",
    [[], [judged_row(pair, "asked", f"n{pair}", one_record="no") for pair in ("s0", "s1", "s2")]],
)
def test_noise_only_asks_not_keyed_yes_are_not_near_flips_or_explained(
    judged: list[mod.JudgedRow],
) -> None:
    events = [noise_row(pair, f"asked:covers:n{pair}") for pair in ("s0", "s1", "s2")]
    actions, bench_ask = bench(24, 9, 38)
    answers, asks = matching_answers()
    save = [row for row in matching_save() if row.pair_id not in ("s0", "s1", "s2")]
    probs = seed_probs(s0=(0.1, 0.72), s1=(0.1, 0.72), s2=(0.1, 0.72))
    check = mod.run_check(
        [*save, *events], answers, actions, bench_ask, asks, probs=probs, judged=judged
    )

    assert check.bounds[2].adjustment == 0
    assert not check.bounds[2].ok
    assert check.near_flips == []
