from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from collections.abc import Callable

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.covers as covers  # noqa: E402
import eval_v1v2.covers_calls as covers_calls  # noqa: E402
import eval_v1v2.judges as judges  # noqa: E402
from eval_v1v2.dup_rules import SignalRow  # noqa: E402
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


def signal(
    pair_id: str,
    *,
    new: float | None = 0.0,
    old: float | None = 0.0,
    enhances: float | None = 0.0,
    run: int = 1,
) -> covers.CoversSignal:
    return covers.CoversSignal(
        pair_id=pair_id,
        run=run,
        new_covers_old=answer(new),
        old_covers_new=answer(old),
        enhances=answer(enhances),
        elapsed_ms=10.0,
        error=None,
    )


def sonnet_call(pair_id: str, new: str | None, old: str | None) -> covers_calls.SonnetCall:
    return covers_calls.SonnetCall.model_validate(
        {
            "pair_id": pair_id,
            "new_covers_old": new,
            "old_covers_new": old,
            "reason": "",
            "elapsed_ms": 1.0,
            "attempts": 1,
            "error": None if new is not None else "JudgeUnavailable: limit",
        }
    )


def case(
    pair_id: str,
    pick: str | None,
    *,
    similarity: float | None = 0.9,
    cover: covers.CoversSignal | None = None,
    sonnet: covers_calls.SonnetCall | None = None,
) -> covers.Case:
    return covers.Case(
        pair_id=pair_id,
        pick=pick,
        similarity=similarity,
        in_band=pick == "follow_up",
        covers=cover,
        sonnet=sonnet,
    )


def by_id(*rows: covers.Case) -> dict[str, covers.Case]:
    return {row.pair_id: row for row in rows}


class TestRuleDecisions:
    def test_a_restatement_pick_reinforces_whatever_containment_says(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, None)

        assert rule.decide(case("a", "restatement", cover=signal("a", new=0.95))) == "reinforce"

    def test_containment_at_the_threshold_supersedes_and_below_it_does_not(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, None)

        assert rule.decide(case("a", "follow_up", cover=signal("a", new=0.7))) == "supersede"
        assert rule.decide(case("b", "follow_up", cover=signal("b", new=0.69))) == "new"

    def test_containment_outranks_a_replacement_pick_which_otherwise_flags(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, None)

        assert rule.decide(case("a", "replacement", cover=signal("a", new=0.9))) == "supersede"
        assert rule.decide(case("b", "replacement", cover=signal("b", new=0.1))) == "flag"

    def test_similarity_floor_blocks_a_supersede(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, 0.85)
        high = case("a", "follow_up", similarity=0.85, cover=signal("a", new=0.9))
        low = case("b", "follow_up", similarity=0.84, cover=signal("b", new=0.9))

        assert rule.decide(high) == "supersede"
        assert rule.decide(low) == "new"

    def test_no_neighbour_means_nothing_to_supersede(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.5, None)

        assert rule.decide(case("a", None, similarity=None, cover=signal("a", new=1.0))) == "new"

    def test_a_missing_signal_never_supersedes(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.5, None)

        assert rule.decide(case("a", "follow_up", cover=signal("a", new=None))) == "new"
        assert rule.decide(case("b", "follow_up", cover=None)) == "new"

    def test_the_enhances_variant_reads_the_enhances_probability(self) -> None:
        rule = covers.jev_rule("enhances", False, 0.7, None)
        row = case("a", "follow_up", cover=signal("a", new=0.1, enhances=0.8))

        assert rule.decide(row) == "supersede"
        assert covers.jev_rule("new_covers_old", False, 0.7, None).decide(row) == "new"

    def test_subset_variant_reinforces_when_old_covers_new_but_not_the_reverse(self) -> None:
        plain = covers.jev_rule("new_covers_old", False, 0.7, None)
        subset = covers.jev_rule("new_covers_old", True, 0.7, None)
        inside = case("a", "follow_up", cover=signal("a", new=0.2, old=0.9))
        mutual = case("b", "follow_up", cover=signal("b", new=0.9, old=0.9))

        assert plain.decide(inside) == "new"
        assert subset.decide(inside) == "reinforce"
        assert subset.decide(mutual) == "supersede"

    def test_sonnet_rule_acts_only_inside_the_follow_up_band(self) -> None:
        rule = covers.sonnet_rule(False, None)
        yes = sonnet_call("a", "yes", "no")

        assert rule.decide(case("a", "follow_up", sonnet=yes)) == "supersede"
        assert rule.decide(case("b", "distinct", sonnet=yes)) == "new"
        assert rule.decide(case("c", "replacement", sonnet=yes)) == "flag"
        assert rule.decide(case("d", "follow_up", sonnet=None)) == "new"

    def test_sonnet_subset_reinforces_an_incoming_note_the_old_one_contains(self) -> None:
        row = case("a", "follow_up", sonnet=sonnet_call("a", "no", "yes"))

        assert covers.sonnet_rule(False, None).decide(row) == "new"
        assert covers.sonnet_rule(True, None).decide(row) == "reinforce"

    def test_rule_family_stays_small(self) -> None:
        assert len(covers.THRESHOLDS) == 9
        assert covers.THRESHOLDS[0] == 0.5
        assert covers.THRESHOLDS[-1] == 0.9
        assert len(covers.jev_rules()) == 2 * 2 * 3 * 9
        assert len(covers.sonnet_rules()) == 6


class TestScoring:
    def test_supersedes_are_judged_against_the_containment_key(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, None)
        rows = by_id(
            *(
                case(pid, "follow_up", cover=signal(pid, new=0.9))
                for pid in ("bad", "good-unrelated", "good-restatement", "nokey")
            )
        )
        relation = {
            "bad": "follow_up",
            "good-unrelated": "follow_up",
            "good-restatement": "restatement",
        }
        cover = {
            "bad": {"new_covers_old": False},
            "good-unrelated": {"new_covers_old": True},
            "good-restatement": {"new_covers_old": True},
        }

        score = covers.score_rule(rule, rows, relation, cover)

        assert score.superseded == ["bad", "good-unrelated", "good-restatement", "nokey"]
        assert score.wrong_supersedes == ["bad"]
        assert score.covered_unrelated == ["good-unrelated"]
        assert score.unverified_supersedes == ["nokey"]
        assert score.superseded_restatement == 1
        assert score.resolved_restatement == 1

    def test_reinforcing_a_keyed_follow_up_or_replacement_is_a_lost_record_or_update(self) -> None:
        rows = by_id(
            case("fu", "restatement"), case("rp", "restatement"), case("rs", "restatement")
        )
        relation = {"fu": "follow_up", "rp": "replacement", "rs": "restatement"}

        score = covers.score_rule(covers.BASELINE, rows, relation, {})

        assert (score.lost_record, score.lost_update) == (1, 1)
        assert score.reinforced_restatement == 1

    def test_reinforcing_when_the_key_says_old_does_not_cover_new_drops_detail(self) -> None:
        rows = by_id(case("a", "restatement"), case("b", "restatement"))
        cover = {"a": {"old_covers_new": False}, "b": {"old_covers_new": True}}

        score = covers.score_rule(covers.BASELINE, rows, {}, cover)

        assert score.reinforce_drops == ["a"]

    def test_replacements_count_when_superseded_or_flagged(self) -> None:
        rule = covers.jev_rule("new_covers_old", False, 0.7, None)
        rows = by_id(
            case("flagged", "replacement", cover=signal("flagged")),
            case("superseded", "replacement", cover=signal("superseded", new=0.9)),
            case("missed", "follow_up", cover=signal("missed")),
        )
        relation = {"flagged": "replacement", "superseded": "replacement", "missed": "replacement"}

        score = covers.score_rule(rule, rows, relation, {})

        assert score.replacements == 3
        assert score.handled_replacement == 2
        assert (score.superseded_replacement, score.flagged_replacement) == (1, 1)

    def test_only_restricts_scoring_to_the_named_pairs(self) -> None:
        rows = by_id(case("a", "restatement"), case("b", "restatement"))

        score = covers.score_rule(
            covers.BASELINE, rows, {"a": "restatement", "b": "restatement"}, {}, frozenset({"a"})
        )

        assert score.restatements == 1

    def test_both_bars_need_zero_wrong_and_zero_lost(self) -> None:
        clean = covers.Score(
            restatements=39,
            reinforced_restatement=35,
            replacements=42,
            flagged_replacement=42,
        )
        wrong = clean.model_copy(update={"wrong_supersedes": ["x"]})
        lost = clean.model_copy(update={"lost_record": 1})
        short = clean.model_copy(update={"reinforced_restatement": 34})

        assert clean.beats_bars
        assert not wrong.beats_bars
        assert not lost.beats_bars
        assert not short.beats_bars

    def test_ranking_puts_wrong_supersedes_before_everything_else(self) -> None:
        safe = covers.Score(reinforced_restatement=1)
        greedy = covers.Score(reinforced_restatement=30, wrong_supersedes=["x"])

        assert covers.rank_key(safe) < covers.rank_key(greedy)

    def test_leave_one_source_out_scores_on_the_held_out_source_only(self) -> None:
        rows = by_id(
            case("a1", "follow_up", cover=signal("a1", new=0.9)),
            case("b1", "follow_up", cover=signal("b1", new=0.9)),
        )
        relation = {"a1": "restatement", "b1": "restatement"}
        cover = {"a1": {"new_covers_old": True}, "b1": {"new_covers_old": True}}
        rules = [covers.BASELINE, covers.jev_rule("new_covers_old", False, 0.7, None)]

        text = covers.leave_one_source_out(rules, rows, relation, cover, {"a1": "a", "b1": "b"})
        pooled = next(line for line in text.splitlines() if line.startswith("| pooled"))

        assert "| 2/2 |" in pooled
        assert text.count("new_covers_old t>=0.70") == 2


class TestSignalAccuracy:
    def test_best_threshold_picks_the_most_accurate_cut(self) -> None:
        data = [(True, 0.95), (True, 0.8), (False, 0.7), (False, 0.1)]

        threshold, table = covers.best_threshold(data)

        assert threshold == 0.75
        assert table.accuracy == 1.0
        assert (table.true_yes_pred_yes, table.true_no_pred_no) == (2, 2)

    def test_enhances_is_compared_with_new_covers_old_and_not_old_covers_new(self) -> None:
        cover = {
            "a": {"new_covers_old": True, "old_covers_new": False},
            "b": {"new_covers_old": True, "old_covers_new": True},
            "c": {"new_covers_old": True},
        }
        signals = {pid: signal(pid, enhances=0.9) for pid in cover}

        pairs = covers.signal_pairs("enhances", signals, cover)

        assert sorted(pairs) == [(False, 0.9), (True, 0.9)]

    def test_a_pair_without_a_signal_is_left_out(self) -> None:
        cover = {"a": {"new_covers_old": True}, "b": {"new_covers_old": False}}

        pairs = covers.signal_pairs("new_covers_old", {"a": signal("a", new=0.8)}, cover)

        assert pairs == [(True, 0.8)]

    def test_sonnet_confusion_counts_only_answered_keyed_pairs(self) -> None:
        calls = {
            "a": sonnet_call("a", "yes", "no"),
            "b": sonnet_call("b", None, None),
            "c": sonnet_call("c", "yes", "yes"),
        }
        cover = {
            "a": {"new_covers_old": True, "old_covers_new": True},
            "b": {"new_covers_old": True},
        }

        tables = covers.sonnet_confusions(calls, cover)

        assert tables["new_covers_old"].total == 1
        assert tables["old_covers_new"].true_yes_pred_no == 1


class TestRepeatability:
    def test_counts_flips_and_threshold_crossings(self) -> None:
        first = {"a": signal("a", new=0.9), "b": signal("b", new=0.6)}
        second = {"a": signal("a", new=0.4, run=2), "b": signal("b", new=0.6, run=2)}

        text = covers.repeatability_table(first, second)
        row = next(line for line in text.splitlines() if line.startswith("| new_covers_old"))

        assert row == "| new_covers_old | 2 | 1 | 0.250 | 0.500 | 1 |"


class TestKeyAndAgreement:
    def test_cover_key_splits_item_ids_per_question(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "key.jsonl"
        path.write_text(
            json.dumps({"item_id": "p1#new_covers_old", "label": "yes", "source": "agreed"})
            + "\n"
            + json.dumps({"item_id": "p1#old_covers_new", "label": "no", "source": "agreed"})
            + "\n",
            encoding="utf-8",
        )

        assert covers.load_cover_key(path) == {
            "p1": {"new_covers_old": True, "old_covers_new": False}
        }

    def test_agreement_is_reported_per_question(self) -> None:
        def label(item: str, judge: str, value: str) -> LabelRow:
            return LabelRow.model_validate(
                {"item_id": item, "judge": judge, "model": "m", "label": value, "reason": "r"}
            )

        rows = [
            label("a#new_covers_old", "claude", "yes"),
            label("a#new_covers_old", "codex", "yes"),
            label("b#new_covers_old", "claude", "yes"),
            label("b#new_covers_old", "codex", "no"),
            label("a#old_covers_new", "claude", "no"),
            label("a#old_covers_new", "codex", "no"),
        ]

        text = covers.agreement_table(rows)

        assert "| new_covers_old | 2 | 1 | 50.0% |" in text
        assert "| old_covers_new | 1 | 1 | 100.0% |" in text


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


class TestContainmentJudges:
    def test_each_pair_becomes_one_item_per_question(self) -> None:
        items = covers_calls.cover_items([pair("p1"), pair("p2")])

        assert [item.item_id for item in items] == [
            "p1#new_covers_old",
            "p1#old_covers_new",
            "p2#new_covers_old",
            "p2#old_covers_new",
        ]
        assert items[0].fields["question"] == "new_covers_old"
        assert covers_calls.split_item_id("p1#old_covers_new") == ("p1", "old_covers_new")

    def test_prompt_carries_the_question_and_asks_for_yes_or_no(self) -> None:
        prompt = judges.build_prompt(judges.TASKS["covers"], covers_calls.cover_items([pair("p1")]))

        assert "QUESTION: new_covers_old\nEXISTING: old text\nINCOMING: new text" in prompt
        assert "The label is yes or no." in prompt
        assert "restatement" not in prompt.split("There are")[1]

    def test_run_task_writes_a_key_where_both_judges_agree(self, tmp_path: pathlib.Path) -> None:
        items = covers_calls.cover_items([pair("p1"), pair("p2")])

        def reply(labels: dict[str, str]) -> Callable[[str], str]:
            def respond(prompt: str) -> str:
                verdicts = []

                for position, block in enumerate(prompt.split("\n\n[")[1:], start=1):
                    question = block.split("QUESTION: ")[1].split("\n")[0]
                    marker = "p2" if "p2" in block else "p1"
                    verdicts.append(
                        {
                            "index": position,
                            "label": labels[f"{marker}#{question}"],
                            "reason": "r",
                        }
                    )

                return json.dumps({"verdicts": verdicts})

            return respond

        class Fake:
            def __init__(self, responder: Callable[[str], str]) -> None:
                self.responder = responder

            async def call(self, prompt: str) -> str:
                return self.responder(prompt)

        answers = {
            "p1#new_covers_old": "yes",
            "p1#old_covers_new": "no",
            "p2#new_covers_old": "yes",
            "p2#old_covers_new": "no",
        }
        split = {**answers, "p2#new_covers_old": "no"}
        seats = [
            judges.Seat("claude", "m", Fake(reply(answers))),
            judges.Seat("codex", "m", Fake(reply(split))),
        ]
        items_with_markers = [
            judges.Item(item.item_id, {**item.fields, "existing": item.item_id.split("#")[0]})
            for item in items
        ]

        outcome = judges.run_task("covers", items_with_markers, seats=seats, out_dir=tmp_path)
        key = covers.load_cover_key(tmp_path / "key-covers.jsonl")

        assert len(outcome.labels) == 8
        assert key == {
            "p1": {"new_covers_old": True, "old_covers_new": False},
            "p2": {"old_covers_new": False},
        }


class StubAdapter:
    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = replies
        self.calls = 0

    async def call(self, _prompt: str) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1

        if isinstance(reply, Exception):
            raise reply

        return reply


class TestSonnetArm:
    def test_parses_both_answers(self) -> None:
        parsed = covers_calls.parse_sonnet(
            'noise {"new_covers_old": "yes", "old_covers_new": "no", "reason": "x"} noise'
        )

        assert (parsed.new_covers_old, parsed.old_covers_new) == ("yes", "no")

    def test_rejects_an_answer_that_is_not_yes_or_no(self) -> None:
        with pytest.raises(covers_calls.UnparseableAnswer):
            covers_calls.parse_sonnet('{"new_covers_old": "maybe", "old_covers_new": "no"}')

        with pytest.raises(covers_calls.UnparseableAnswer):
            covers_calls.parse_sonnet("no json here")

    def test_resume_skips_answered_pairs_and_retries_failed_ones(self) -> None:
        def row(pair_id: str, pick: str) -> SignalRow:
            return SignalRow.model_validate(
                {
                    "pair_id": pair_id,
                    "bucket": "new",
                    "raw_outcome": "approved",
                    "found_existing": True,
                    "relation_pick": pick,
                    "degraded": None,
                    "existing": {
                        "id": pair_id,
                        "pick": pick,
                        "confidence": 0.9,
                        "probabilities": None,
                        "reason_class": None,
                        "similarity": 0.9,
                    },
                    "top_other": None,
                    "adds_information": {
                        "answer": None,
                        "probability": None,
                        "probabilities": None,
                        "elapsed_ms": 1.0,
                        "error": None,
                    },
                    "propose_ms": 1.0,
                }
            )

        signals = {
            "done": row("done", "follow_up"),
            "failed": row("failed", "follow_up"),
            "fresh": row("fresh", "follow_up"),
            "other": row("other", "distinct"),
        }
        pairs = {pid: pair(pid) for pid in signals}
        done = {
            "done": sonnet_call("done", "yes", "yes"),
            "failed": sonnet_call("failed", None, None),
        }

        todo = covers_calls.pending_pairs(pairs, signals, done)

        assert [item.pair_id for item in todo] == ["failed", "fresh"]

    def test_a_run_of_failures_stops_the_arm_and_keeps_what_landed(self) -> None:
        good = '{"new_covers_old": "yes", "old_covers_new": "no", "reason": ""}'
        adapter = StubAdapter([good, *[JudgeUnavailable("usage limit")] * 50])
        landed: list[covers_calls.SonnetCall] = []
        pairs = [pair(f"p{index}") for index in range(40)]

        results = asyncio.run(covers_calls.ask_all(adapter, pairs, landed.append))  # type: ignore[arg-type]

        assert sum(1 for call in results if call.answered) == 1
        assert len(results) < len(pairs)
        assert len(landed) == len(results)
