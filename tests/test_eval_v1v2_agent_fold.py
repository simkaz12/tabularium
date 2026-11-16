from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import sys
from collections.abc import Callable, Sequence
from typing import Any, Literal

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.agent_fold as mod  # noqa: E402
import eval_v1v2.agent_fold_ask as ask  # noqa: E402
import eval_v1v2.agent_fold_recount as recount  # noqa: E402
import eval_v1v2.covers as covers  # noqa: E402
import eval_v1v2.fold as fold  # noqa: E402
import eval_v1v2.fold_guard as guard  # noqa: E402
import eval_v1v2.second_judge as second_judge  # noqa: E402
from eval_v1v2.rows import LabelRow, PairRow, Source  # noqa: E402
from tabularium.engine.judge import JudgeUnavailable  # noqa: E402

sys.path.pop(0)

ALL_YES = {"nothing_lost": True, "nothing_invented": True, "newer_wins": True, "one_record": True}
SONNET_CHOICE = "sonnet-choice-cold"
SONNET_TASK = "sonnet-fold_default-in_task"
CODEX_YES_CLAUDE_NO: tuple[tuple[Literal["codex", "claude"], Literal["yes", "no"]], ...] = (
    ("codex", "yes"),
    ("claude", "no"),
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


def band_row(pair_id: str, relation: str | None = None) -> fold.BandRow:
    return fold.BandRow(
        pair_id=pair_id,
        source="golden",
        relation=relation,
        similarity=0.82,
        pick="follow_up",
        enhances=0.0,
        existing="old note",
        incoming="new note",
    )


def ask_row(
    pair_id: str,
    arm: str = SONNET_CHOICE,
    action: str | None = "fold",
    note: str | None = "merged",
    error: str | None = None,
    elapsed: float = 100.0,
) -> ask.AskRow:
    model, wording, framing = arm.split("-")

    return ask.AskRow.model_validate(
        {
            "pair_id": pair_id,
            "arm": arm,
            "model": model,
            "wording": wording,
            "framing": framing,
            "action": action,
            "note": note if action == "fold" else None,
            "reason": None,
            "raw": None,
            "elapsed_ms": elapsed,
            "attempts": 1,
            "error": error,
        }
    )


def reply_json(**fields: Any) -> str:
    return json.dumps(fields)


class TestPrompts:
    def test_both_notes_land_once_and_no_token_is_left(self) -> None:
        for wording, framing in ask.SHAPES:
            text = ask.render_prompt(wording, framing, "EXISTING-TEXT", "INCOMING-TEXT")

            assert text.count("EXISTING-TEXT") == 1
            assert text.count("INCOMING-TEXT") == 1
            assert "<<" not in text

    def test_a_note_that_contains_a_token_is_not_substituted_again(self) -> None:
        text = ask.render_prompt("choice", "cold", "see <<INCOMING>> here", "plain")

        assert "see <<INCOMING>> here" in text
        assert text.count("plain") == 1

    def test_only_the_in_task_framing_offers_continue_and_carries_the_task(self) -> None:
        cold = ask.render_prompt("fold_default", "cold", "a", "b")
        task = ask.render_prompt("fold_default", "in_task", "a", "b")

        assert '"continue"' not in cold
        assert '"continue"' in task
        assert ask.IN_TASK_TASK in task
        assert ask.IN_TASK_TASK not in cold

    def test_the_two_wordings_differ_in_their_exit_clause_and_share_the_fold_rules(self) -> None:
        choice = ask.prompt_template("choice", "cold")
        default = ask.prompt_template("fold_default", "cold")

        assert ask.FOLD_RULES in choice
        assert ask.FOLD_RULES in default
        assert "names the" in default
        assert "names the" not in choice
        assert "separate facts" in choice

    def test_fold_rules_keep_what_the_writer_prompt_keeps(self) -> None:
        for phrase in (
            "every concrete detail",
            "rejected options",
            "Add nothing that is in neither note",
            "newer one and wins",
            "one fact or one decision",
        ):
            assert phrase in " ".join(ask.FOLD_RULES.split())

    def test_prompts_are_written_per_wording_and_framing(self, tmp_path: pathlib.Path) -> None:
        ask.write_prompts(tmp_path)

        names = sorted(p.name for p in (tmp_path / "ask-prompts").iterdir())

        assert names == [
            "choice-cold.txt",
            "choice-in_task.txt",
            "fold_default-cold.txt",
            "fold_default-in_task.txt",
        ]


class TestParse:
    def test_fold_needs_a_note(self) -> None:
        assert ask.parse_reply(reply_json(action="fold", note="x"), "choice", "cold").note == "x"

        for bad in (reply_json(action="fold"), reply_json(action="fold", note="  ")):
            try:
                ask.parse_reply(bad, "choice", "cold")
            except ask.UnparseableReply:
                continue

            raise AssertionError("a fold without a note was accepted")

    def test_keep_both_needs_a_reason_only_under_fold_default(self) -> None:
        bare = reply_json(action="keep_both")

        assert ask.parse_reply(bare, "choice", "cold").action == "keep_both"

        try:
            ask.parse_reply(bare, "fold_default", "cold")
        except ask.UnparseableReply:
            return

        raise AssertionError("fold_default accepted keep_both without a reason")

    def test_continue_is_refused_in_the_cold_framing(self) -> None:
        text = reply_json(action="continue")

        assert ask.parse_reply(text, "fold_default", "in_task").action == "continue"

        try:
            ask.parse_reply(text, "fold_default", "cold")
        except ask.UnparseableReply:
            return

        raise AssertionError("cold accepted continue")

    def test_json_inside_prose_and_fences_parses_and_braces_in_a_note_survive(self) -> None:
        note = "set {a: 1} in config"
        wrapped = "Here:\n```json\n" + reply_json(action="fold", note=note) + "\n```"

        assert ask.parse_reply(wrapped, "choice", "cold").note == note

    def test_unknown_action_is_unparseable(self) -> None:
        try:
            ask.parse_reply(reply_json(action="merge", note="x"), "choice", "cold")
        except ask.UnparseableReply:
            return

        raise AssertionError("unknown action accepted")


class TestAsking:
    def arm(self) -> ask.Arm:
        return ask.Arm("sonnet", "fold_default", "in_task")

    def test_a_good_reply_is_recorded_with_its_arm(self) -> None:
        adapter = FakeAdapter([reply_json(action="fold", note="merged")])

        row = asyncio.run(ask.ask_pair(adapter, self.arm(), band_row("p")))  # type: ignore[arg-type]

        assert (row.action, row.note, row.arm, row.attempts) == (
            "fold",
            "merged",
            SONNET_TASK,
            1,
        )
        assert row.answered and row.folded

    def test_an_unparseable_reply_is_retried_then_kept_as_invalid_not_as_an_error(self) -> None:
        adapter = FakeAdapter(["sure, here you go", "still prose"])

        row = asyncio.run(ask.ask_pair(adapter, self.arm(), band_row("p")))  # type: ignore[arg-type]

        assert row.action == "invalid"
        assert row.error is None
        assert row.answered and not row.folded
        assert row.raw == "still prose"
        assert row.attempts == second_judge.MAX_ATTEMPTS

    def test_an_unavailable_model_leaves_the_pair_pending(self) -> None:
        adapter = FakeAdapter([JudgeUnavailable("limit")] * second_judge.MAX_ATTEMPTS)

        row = asyncio.run(ask.ask_pair(adapter, self.arm(), band_row("p")))  # type: ignore[arg-type]

        assert row.action is None
        assert row.error is not None and "limit" in row.error
        assert not row.answered

    def test_a_retry_after_an_outage_can_succeed(self) -> None:
        adapter = FakeAdapter([JudgeUnavailable("limit"), reply_json(action="continue")])

        row = asyncio.run(ask.ask_pair(adapter, self.arm(), band_row("p")))  # type: ignore[arg-type]

        assert row.action == "continue"
        assert row.error is None
        assert row.attempts == 2

    def test_the_prompt_sent_carries_the_pair_notes(self) -> None:
        adapter = FakeAdapter([reply_json(action="continue")])

        asyncio.run(ask.ask_pair(adapter, self.arm(), band_row("p")))  # type: ignore[arg-type]

        assert "old note" in adapter.prompts[0]
        assert "new note" in adapter.prompts[0]

    def test_a_run_of_failures_stops_the_arm_and_keeps_what_landed(self) -> None:
        band = [band_row(f"p{i}") for i in range(30)]
        adapter = FakeAdapter([JudgeUnavailable("limit")] * 200)
        seen: list[ask.AskRow] = []

        asyncio.run(ask.ask_all(adapter, self.arm(), band, seen.append))  # type: ignore[arg-type]

        assert 0 < len(seen) < len(band)
        assert all(not row.answered for row in seen)


class TestResume:
    def test_load_asks_keeps_the_latest_row_per_pair_and_arm(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "asks.jsonl"
        rows = [
            ask_row("p", action=None, error="down"),
            ask_row("p", action="keep_both"),
            ask_row("q", arm=SONNET_TASK),
        ]
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")

        loaded = ask.load_asks(path)

        assert loaded[("p", SONNET_CHOICE)].action == "keep_both"
        assert set(loaded) == {("p", SONNET_CHOICE), ("q", SONNET_TASK)}

    def test_only_unanswered_pairs_are_pending(self) -> None:
        band = [band_row(p) for p in ("a", "b", "c", "d")]
        asks = {
            ("a", SONNET_CHOICE): ask_row("a"),
            ("b", SONNET_CHOICE): ask_row("b", action=None, error="down"),
            ("c", SONNET_CHOICE): ask_row("c", action="invalid"),
            ("d", SONNET_TASK): ask_row("d", arm=SONNET_TASK),
        }

        pending = ask.pending_pairs(band, asks, ask.Arm("sonnet", "choice", "cold"))

        assert [row.pair_id for row in pending] == ["b", "d"]

    def test_there_are_six_arms_sonnet_first(self) -> None:
        arms = ask.all_arms()

        assert len(arms) == 6
        assert [arm.model for arm in arms] == ["sonnet"] * 3 + ["opus"] * 3
        assert len({arm.id for arm in arms}) == 6


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


def timed(probability: float | None) -> guard.TimedAnswer:
    if probability is None:
        return guard.TimedAnswer(
            answer=None, p_yes=None, probabilities=None, error="down", elapsed_ms=1.0
        )

    return guard.TimedAnswer(
        answer="yes" if probability >= 0.5 else "no",
        p_yes=probability,
        probabilities=None,
        error=None,
        elapsed_ms=1.0,
    )


def signal(
    pair_id: str, arm: str = SONNET_CHOICE, old: float | None = 0.95, new: float | None = 0.95
) -> guard.GuardSignal:
    return guard.GuardSignal(
        pair_id=pair_id,
        writer=arm,
        run=1,
        fold_covers_old=timed(old),
        fold_covers_new=timed(new),
        sources_cover_fold=timed(0.9),
        fold_single=timed(0.9),
        error=None,
    )


def pair_row(pair_id: str, source: Source = "golden") -> PairRow:
    return PairRow(
        pair_id=pair_id, source=source, kind="fact", existing="old", incoming="new",
        similarity=None, old_label=None,
    )  # fmt: skip


def data_for(
    cases: dict[str, covers.Case],
    *,
    relation: dict[str, str] | None = None,
    asks: list[ask.AskRow] | None = None,
    grades: dict[tuple[str, str], dict[str, bool]] | None = None,
    signals: list[guard.GuardSignal] | None = None,
    subjects: dict[str, float] | None = None,
    sources: dict[str, Source] | None = None,
) -> mod.AgentData:
    labels = relation or {}
    inputs = fold.Inputs(
        pairs={pid: pair_row(pid, (sources or {}).get(pid, "golden")) for pid in cases},
        cases=cases,
        relation=labels,
        cover={},
        band=[band_row(pid, labels.get(pid)) for pid in cases],
        folds={},
        grades={},
        labels=[],
        published_v1=[],
        out_dir=pathlib.Path("."),
    )
    server = guard.GuardData(
        inputs=inputs,
        guards={},
        subjects={
            pid: guard.BandSignal(pair_id=pid, run=1, same_subject=timed(value), error=None)
            for pid, value in (subjects or {}).items()
        },
        p_replacement={},
        out_dir=pathlib.Path("."),
    )

    return mod.AgentData(
        guard=server,
        asks={(row.pair_id, row.arm): row for row in asks or []},
        grades={key: fold.FoldGrade(answers=value) for key, value in (grades or {}).items()},
        signals={(row.pair_id, row.writer): row for row in signals or []},
        labels=[],
        out_dir=pathlib.Path("."),
    )


def agent_rule(
    data: mod.AgentData,
    *,
    arm: str = SONNET_CHOICE,
    floor: float = 0.75,
    enhances: float | None = None,
    same_subject: float | None = None,
    kind: str = "off",
    threshold: float | None = None,
) -> mod.AgentRule:
    trigger = guard.GuardRule(floor, enhances, same_subject, None, "off", None)

    return mod.AgentRule(arm, trigger, kind, threshold, data)


def decide(rule: mod.AgentRule, pair_id: str) -> str:
    return rule.decide(rule.data.inputs.cases[pair_id])


class TestRule:
    def test_containment_decides_before_any_ask(self) -> None:
        data = data_for(
            {
                "r": case("r", "restatement", similarity=0.99),
                "s": case("s", new_covers_old=0.95),
                "f": case("f", "replacement"),
            },
            asks=[ask_row(pid) for pid in "rsf"],
        )
        rule = agent_rule(data)

        assert [decide(rule, pid) for pid in "rsf"] == ["reinforce", "supersede", "flag"]
        assert rule.asked(data.inputs.cases["r"]) is False

    def test_only_a_fold_reply_folds_and_keep_both_continue_invalid_save_as_new(self) -> None:
        cases = {pid: case(pid) for pid in ("fold", "keep", "cont", "bad", "gone")}
        data = data_for(
            cases,
            asks=[
                ask_row("fold"),
                ask_row("keep", action="keep_both"),
                ask_row("cont", action="continue"),
                ask_row("bad", action="invalid"),
                ask_row("gone", action=None, error="down"),
            ],
        )
        rule = agent_rule(data)

        assert [decide(rule, pid) for pid in cases] == ["fold", "new", "new", "new", "new"]

    def test_a_pair_the_trigger_does_not_fire_on_is_never_folded(self) -> None:
        data = data_for(
            {"low": case("low", similarity=0.78), "high": case("high", similarity=0.9)},
            asks=[ask_row("low"), ask_row("high")],
        )
        rule = agent_rule(data, floor=0.80)

        assert decide(rule, "low") == "new"
        assert decide(rule, "high") == "fold"

    def test_same_subject_and_enhances_gate_the_ask(self) -> None:
        data = data_for(
            {"a": case("a", enhances=0.8), "b": case("b", enhances=0.8)},
            asks=[ask_row("a"), ask_row("b")],
            subjects={"a": 0.75, "b": 0.2},
        )

        assert decide(agent_rule(data, floor=0.8, same_subject=0.7), "a") == "fold"
        assert decide(agent_rule(data, floor=0.8, same_subject=0.7), "b") == "new"
        assert decide(agent_rule(data, floor=0.8, enhances=0.9), "a") == "new"

    def test_the_coverage_guard_needs_both_readings_and_a_signal(self) -> None:
        cases = {pid: case(pid) for pid in ("ok", "weak", "none", "nosignal")}
        data = data_for(
            cases,
            asks=[ask_row(pid) for pid in cases],
            signals=[signal("ok"), signal("weak", new=0.6), signal("none", old=None)],
        )
        rule = agent_rule(data, kind="coverage", threshold=0.70)

        assert [decide(rule, pid) for pid in cases] == ["fold", "new", "new", "new"]
        assert decide(agent_rule(data, kind="coverage", threshold=0.5), "weak") == "fold"

    def test_a_guard_rule_never_reads_another_arms_signals(self) -> None:
        data = data_for(
            {"a": case("a")},
            asks=[ask_row("a", arm=SONNET_TASK)],
            signals=[signal("a", arm=SONNET_CHOICE)],
        )

        assert (
            decide(agent_rule(data, arm=SONNET_TASK, kind="coverage", threshold=0.7), "a") == "new"
        )
        assert decide(agent_rule(data, arm=SONNET_TASK), "a") == "fold"


class TestScoring:
    def build(self) -> mod.AgentData:
        cases = {pid: case(pid) for pid in ("good", "bad", "disp", "keep", "rep")}
        relation = {
            "good": "restatement",
            "bad": "restatement",
            "disp": "restatement",
            "keep": "restatement",
            "rep": "replacement",
        }
        asks = [
            ask_row("good"),
            ask_row("bad"),
            ask_row("disp"),
            ask_row("keep", action="keep_both"),
            ask_row("rep"),
        ]
        grades = {
            ("good", SONNET_CHOICE): ALL_YES,
            ("bad", SONNET_CHOICE): {**ALL_YES, "nothing_lost": False},
            ("disp", SONNET_CHOICE): {"nothing_lost": True},
            ("rep", SONNET_CHOICE): ALL_YES,
        }
        signals = [signal("good"), signal("bad", old=0.3), signal("disp"), signal("rep")]

        return data_for(cases, relation=relation, asks=asks, grades=grades, signals=signals)

    def test_strict_and_loose_readings_differ_on_the_disputed_fold(self) -> None:
        score = mod.score_agent(agent_rule(self.build()))

        assert score.good == ["good", "rep"]
        assert score.bad == ["bad"]
        assert score.disputed == ["disp"]
        assert mod.resolved(score, "strict") == 1
        assert mod.resolved(score, "loose") == 2
        assert mod.caught(score, "strict") == 1

    def test_a_disputed_replacement_fold_earns_loose_credit_but_not_strict(self) -> None:
        cases = {"disp": case("disp"), "good": case("good")}
        data = data_for(
            cases,
            relation={"disp": "replacement", "good": "replacement"},
            asks=[ask_row("disp"), ask_row("good")],
            grades={
                ("disp", SONNET_CHOICE): {
                    "nothing_invented": True,
                    "newer_wins": True,
                    "one_record": True,
                },
                ("good", SONNET_CHOICE): ALL_YES,
            },
        )

        score = mod.score_agent(agent_rule(data))

        assert score.disputed == ["disp"]
        assert mod.caught(score, "strict") == 1
        assert mod.caught(score, "loose") == 2

    def test_a_bad_replacement_fold_earns_no_loose_credit_even_with_the_two_answers_yes(
        self,
    ) -> None:
        cases = {"bad": case("bad")}
        data = data_for(
            cases,
            relation={"bad": "replacement"},
            asks=[ask_row("bad")],
            grades={("bad", SONNET_CHOICE): {**ALL_YES, "nothing_lost": False}},
        )

        score = mod.score_agent(agent_rule(data))

        assert mod.caught(score, "strict") == 0
        assert mod.caught(score, "loose") == 0

    def test_the_guard_rejects_the_bad_fold_and_counts_it_as_rejected_not_accepted(self) -> None:
        rule = agent_rule(self.build(), kind="coverage", threshold=0.7)
        score = mod.score_agent(rule)
        stats, latencies = mod.ask_stats(rule)

        assert score.bad == []
        assert stats.asked == 5
        assert stats.replied_fold == 4
        assert stats.keep_both == 1
        assert stats.accepted == 3
        assert (stats.rejected_good, stats.rejected_bad, stats.rejected_disputed) == (0, 1, 0)
        assert len(latencies) == 5

    def test_an_unanswered_ask_counts_as_unanswered_and_takes_no_latency(self) -> None:
        data = data_for(
            {"a": case("a"), "b": case("b")},
            asks=[ask_row("a"), ask_row("b", action=None, error="down")],
        )

        stats, latencies = mod.ask_stats(agent_rule(data))

        assert (stats.asked, stats.unanswered, stats.replied_fold) == (2, 1, 1)
        assert latencies == [100.0]

    def test_codex_alone_grades_ignore_the_other_judge(self) -> None:
        labels = [
            LabelRow(
                item_id=fold.fold_item_id("p", SONNET_CHOICE, q),
                judge=judge,
                model="m",
                label=label,
                reason="r",
            )
            for q in fold.QUESTIONS
            for judge, label in CODEX_YES_CLAUDE_NO
        ]

        grades = mod.codex_grades(labels)

        assert grades[("p", SONNET_CHOICE)].status == "good"


class TestChoice:
    def data(self) -> mod.AgentData:
        cases = {f"r{i}": case(f"r{i}") for i in range(4)}
        cases |= {f"d{i}": case(f"d{i}") for i in range(2)}
        relation = {pid: "restatement" if pid[0] == "r" else "distinct" for pid in cases}
        asks = [
            ask_row("r0", arm=SONNET_CHOICE),
            ask_row("r1", arm=SONNET_CHOICE, action="keep_both"),
            ask_row("r2", arm=SONNET_CHOICE),
            ask_row("r3", arm=SONNET_CHOICE),
            ask_row("d0", arm=SONNET_CHOICE),
            ask_row("d1", arm=SONNET_CHOICE, action="keep_both"),
            *[ask_row(pid, arm=SONNET_TASK, action="continue") for pid in cases if pid[0] == "r"],
            *[ask_row(pid, arm=SONNET_TASK, action="keep_both") for pid in cases if pid[0] == "d"],
        ]

        return data_for(cases, relation=relation, asks=asks)

    def test_easy_exit_counts_true_restatements_not_folded_and_wrong_fold_counts_distinct(
        self,
    ) -> None:
        data = self.data()

        assert mod.easy_exit_rate(data, SONNET_CHOICE) == 0.25
        assert mod.wrong_fold_rate(data, SONNET_CHOICE) == 0.5
        assert mod.easy_exit_rate(data, SONNET_TASK) == 1.0
        assert mod.wrong_fold_rate(data, SONNET_TASK) == 0.0

    def test_selection_takes_the_lowest_easy_exit_and_adds_the_in_task_arm(self) -> None:
        picks = mod.select_arms(self.data())

        assert picks == {"sonnet": [SONNET_CHOICE, SONNET_TASK]}

    def test_a_tie_on_easy_exit_goes_to_the_lower_wrong_fold_rate_then_arm_order(self) -> None:
        cold = "sonnet-fold_default-cold"
        data = data_for(
            {"r0": case("r0"), "d0": case("d0")},
            relation={"r0": "restatement", "d0": "distinct"},
            asks=[
                ask_row("r0", arm=SONNET_CHOICE),
                ask_row("d0", arm=SONNET_CHOICE),
                ask_row("r0", arm=cold),
                ask_row("d0", arm=cold, action="keep_both"),
            ],
        )

        assert mod.select_arms(data) == {"sonnet": [cold]}

    def test_an_arm_with_no_answers_is_not_selected(self) -> None:
        data = data_for(
            {"r0": case("r0")},
            relation={"r0": "restatement"},
            asks=[ask_row("r0", action=None, error="down")],
        )

        assert mod.select_arms(data) == {}

    def test_overview_reports_rates_and_pending(self) -> None:
        text = mod.choice_overview(self.data(), [SONNET_CHOICE])

        assert "| sonnet-choice-cold | 6 | 4 | 2 | 0 | 0 | 0 | 67% | 33% | 0% | 25% (4 res" in text

    def test_judged_arms_keep_any_arm_that_already_has_labels(self) -> None:
        data = self.data()
        label = LabelRow(
            item_id=fold.fold_item_id("r0", "opus-choice-cold", "nothing_lost"),
            judge="codex",
            model="m",
            label="yes",
            reason="r",
        )
        kept = mod.AgentData(
            guard=data.guard,
            asks=data.asks,
            grades=data.grades,
            signals=data.signals,
            labels=[label],
            out_dir=data.out_dir,
        )

        assert mod.judged_arms(kept) == [SONNET_CHOICE, SONNET_TASK, "opus-choice-cold"]

    def test_fold_items_carry_the_agents_note_for_four_questions(self) -> None:
        items = mod.fold_items(self.data(), [SONNET_CHOICE])

        assert len(items) == 4 * 4
        assert {item.fields["fold"] for item in items} == {"merged"}
        assert {i.item_id.split("#")[1] for i in items} == set(fold.QUESTIONS)
        assert all(
            i.item_id.startswith(("r", "d")) and "~sonnet-choice-cold#" in i.item_id for i in items
        )


class TestLazyGuard:
    def test_a_lazy_resend_fails_the_guard_when_its_own_coverage_of_the_old_note_is_low(
        self,
    ) -> None:
        cases = {
            "low": case("low", new_covers_old=0.30),
            "mid": case("mid", new_covers_old=0.60),
            "high": case("high", new_covers_old=0.85),
        }
        data = data_for(cases)

        lines = mod.lazy_resend_table(data).splitlines()

        assert lines[2].startswith("| 0.50 | 3 | 1 ")
        assert lines[3].startswith("| 0.70 | 3 | 2 ")
        assert lines[4].startswith("| 0.90 | 3 | 3 ")

    def test_a_near_copy_of_the_agents_own_note_is_detected(self) -> None:
        assert mod.near_copy("the quick brown fox jumps", "the quick brown fox jumps")
        assert not mod.near_copy("the quick brown fox jumps", "an entirely different thing here")

    def test_the_minimum_coverage_keeps_a_zero_reading(self) -> None:
        data = data_for(
            {"p": case("p"), "q": case("q"), "r": case("r")},
            signals=[signal("p", old=0.0, new=0.9), signal("q", old=0.0, new=0.0)],
        )

        assert mod.min_coverage(data, "p", SONNET_CHOICE) == 0.0
        assert mod.min_coverage(data, "q", SONNET_CHOICE) == 0.0
        assert mod.min_coverage(data, "r", SONNET_CHOICE) is None

    def test_the_minimum_coverage_is_missing_when_either_reading_is_missing(self) -> None:
        data = data_for({"p": case("p")}, signals=[signal("p", old=None, new=0.9)])

        assert mod.min_coverage(data, "p", SONNET_CHOICE) is None

    def test_agent_guard_table_counts_good_and_bad_folds_it_rejects(self) -> None:
        cases = {pid: case(pid) for pid in ("g1", "g2", "b1")}
        data = data_for(
            cases,
            asks=[ask_row(pid) for pid in cases],
            grades={
                ("g1", SONNET_CHOICE): ALL_YES,
                ("g2", SONNET_CHOICE): ALL_YES,
                ("b1", SONNET_CHOICE): {**ALL_YES, "nothing_lost": False},
            },
            signals=[signal("g1"), signal("g2", old=0.6), signal("b1", new=0.4)],
        )

        row = next(
            line
            for line in mod.agent_guard_table(data, [SONNET_CHOICE]).splitlines()
            if line.startswith("| sonnet-choice-cold | 0.70")
        )

        assert row.split(" | ")[2:9] == ["3", "2", "1", "1", "1", "0", "0"]


class TestLeaveOneSourceOut:
    def competing(self) -> mod.AgentData:
        ids = ("a1", "a2", "b1", "b2")
        similarity = {"a1": 0.90, "a2": 0.78, "b1": 0.90, "b2": 0.78}

        return data_for(
            {pid: case(pid, similarity=similarity[pid]) for pid in ids},
            relation=dict.fromkeys(ids, "restatement"),
            asks=[ask_row(pid) for pid in ids],
            grades={
                ("a1", SONNET_CHOICE): ALL_YES,
                ("a2", SONNET_CHOICE): ALL_YES,
                ("b1", SONNET_CHOICE): ALL_YES,
                ("b2", SONNET_CHOICE): {**ALL_YES, "nothing_lost": False},
            },
            sources={"a1": "golden", "a2": "golden", "b1": "bounce_log", "b2": "bounce_log"},
        )

    def test_the_full_data_winner_differs_from_the_winner_trained_without_a_source(self) -> None:
        data = self.competing()
        rules = mod.rules_for(data, [SONNET_CHOICE])
        everyone = frozenset(data.inputs.cases)
        without_golden = frozenset({"b1", "b2"})

        assert mod.best_rule(rules, "strict", everyone).trigger.floor == 0.75
        assert mod.best_rule(rules, "strict", without_golden).trigger.floor == 0.80

    def test_each_held_out_source_uses_the_rule_chosen_on_the_others(self) -> None:
        data = self.competing()

        table, pooled = mod.loso_table(mod.rules_for(data, [SONNET_CHOICE]), data, "strict")

        golden = next(line for line in table.splitlines() if line.startswith("| golden |"))
        bounce = next(line for line in table.splitlines() if line.startswith("| bounce_log |"))
        assert golden.split(" | ")[1] == f"{SONNET_CHOICE}, sim>=0.80"
        assert golden.split(" | ")[2] == "1/2"
        assert bounce.split(" | ")[1] == f"{SONNET_CHOICE}, sim>=0.75"
        assert bounce.split(" | ")[2] == "1/2"
        assert pooled["restatements"] == 2
        assert pooled["bad"] == 1
        assert "| pooled held-out | | 2/4 |" in table

    def test_a_tie_on_the_other_sources_is_listed_and_not_pooled(self) -> None:
        cases = {"a": case("a", similarity=0.9), "b": case("b", similarity=0.78)}
        data = data_for(
            cases,
            relation={"a": "restatement", "b": "restatement"},
            asks=[ask_row("a"), ask_row("b")],
            grades={("a", SONNET_CHOICE): ALL_YES, ("b", SONNET_CHOICE): ALL_YES},
            sources={"a": "golden", "b": "bounce_log"},
        )

        table, pooled = mod.loso_table(mod.rules_for(data, [SONNET_CHOICE]), data, "strict")

        assert pooled["restatements"] == 2
        assert pooled["bad"] == 0
        assert "| pooled held-out | | 2/2 |" in table
        assert "tied on the other sources, not pooled" in table


class TestRecount:
    def write(self, path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_the_independent_recount_counts_a_small_fixture_by_hand(
        self, tmp_path: pathlib.Path
    ) -> None:
        base = tmp_path / "base"
        out = base / "experiments" / "agent-fold"
        ids = ["a", "b", "c"]
        neighbour = {"pick": "follow_up", "similarity": 0.85, "probabilities": {}}
        self.write(
            base / "experiments" / "dup-rules" / "signals.jsonl",
            [{"pair_id": pid, "existing": neighbour} for pid in ids],
        )
        self.write(
            base / "experiments" / "covers" / "signals.jsonl",
            [
                {
                    "pair_id": pid,
                    "new_covers_old": {"p_yes": 0.1},
                    "enhances": {"p_yes": 0.8},
                }
                for pid in ids
            ],
        )
        self.write(
            base / "key-relation.jsonl",
            [
                {"item_id": "a", "label": "restatement"},
                {"item_id": "b", "label": "replacement"},
                {"item_id": "c", "label": "follow_up"},
            ],
        )
        self.write(base / "experiments" / "covers" / "key-covers.jsonl", [])
        self.write(base / "experiments" / "fold-guard" / "band-signals.jsonl", [])
        self.write(
            out / "asks.jsonl",
            [json.loads(ask_row(pid).model_dump_json()) for pid in ids],
        )
        self.write(
            out / "key-fold.jsonl",
            [
                {"item_id": f"{pid}~{SONNET_CHOICE}#{q}", "label": "yes"}
                for pid in ("a", "b")
                for q in fold.QUESTIONS
            ]
            + [
                {"item_id": f"c~{SONNET_CHOICE}#nothing_lost", "label": "no"},
            ],
        )
        self.write(out / "guard-signals.jsonl", [])

        counted = recount.recount(base, out, SONNET_CHOICE, 0.80, 0.70, None, "off", None)

        assert "restatements=1 replacements=1" in counted
        assert "loose_replacements=1" in counted
        assert "asked=3 replied_fold=3 accepted=3 bad=1" in counted


def test_a_relative_log_is_read_back_from_the_directory_the_replies_were_written_to(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [band_row("p1"), band_row("p2")]
    arm = ask.Arm("sonnet", "choice", "cold")

    async def fake_ask_all(
        adapter: object,
        chosen: ask.Arm,
        todo: Sequence[fold.BandRow],
        sink: Callable[[ask.AskRow], None],
    ) -> list[ask.AskRow]:
        written = [ask_row(row.pair_id, chosen.id) for row in todo]

        for row in written:
            sink(row)

        return written

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(ask, "ask_all", fake_ask_all)

    complete = ask.ask_arms(pathlib.Path("out") / "asks.jsonl", rows, [arm])

    assert complete
    assert len(ask.load_asks(tmp_path / "out" / "asks.jsonl")) == 2
