from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import ClassVar

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.fold as fold_mod  # noqa: E402
import eval_v1v2.judges as judges  # noqa: E402
import eval_v1v2.noise_relation as mod  # noqa: E402
import eval_v1v2.noise_report as noise_report  # noqa: E402
from eval_v1v2.prod_check import JudgedRow  # noqa: E402
from eval_v1v2.rows import NeighbourRow, SaveRow, read_jsonl  # noqa: E402

sys.path.pop(0)


def save_row(pair_id: str, bucket: str, raw_outcome: str) -> SaveRow:
    return SaveRow.model_validate(
        {
            "pair_id": pair_id,
            "system": "v2",
            "bucket": bucket,
            "found_existing": True,
            "relation_pick": None,
            "raw_outcome": raw_outcome,
            "elapsed_ms": 1.0,
            "degraded": None,
        }
    )


def judged_row(pair_id: str, noise_id: str, label: str | None) -> mod.RelationJudgedRow:
    return mod.RelationJudgedRow(
        pair_id=pair_id, noise_id=noise_id, actions={"run-1": "reinforce"}, label=label, labels={}
    )


class FakeAdapter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        match = re.search(r"There are (\d+) items", prompt)
        assert match is not None
        count = int(match.group(1))
        verdicts = [
            {"index": index, "label": self.label, "reason": "r"} for index in range(1, count + 1)
        ]

        return json.dumps({"verdicts": verdicts})


def fake_seats(label: str) -> list[judges.Seat]:
    return [
        judges.Seat(judge="claude", model="opus", adapter=FakeAdapter(label)),
        judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=FakeAdapter(label)),
    ]


class TestEvents:
    def test_collects_relation_events_on_noise_only(self) -> None:
        runs = {
            "run-1": [
                save_row("a", "wrong_target", "approved:supersede:n1"),
                save_row("b", "wrong_target", "reinforced:reinforce:n2"),
                save_row("c", "wrong_target", "asked:n3"),
                save_row("d", "caught", "reinforced:reinforce:seed"),
            ],
            "run-2": [save_row("a", "wrong_target", "approved:supersede:n1")],
        }

        events = mod.relation_events(runs)

        assert [(e.pair_id, e.noise_id, e.actions) for e in events] == [
            ("a", "n1", {"run-1": "supersede", "run-2": "supersede"}),
            ("b", "n2", {"run-1": "reinforce"}),
        ]

    def test_sets_client_texts_aside(self) -> None:
        runs = {
            "run-1": [
                save_row("a", "wrong_target", "approved:supersede:n1"),
                save_row("b", "wrong_target", "reinforced:reinforce:n2"),
                save_row("c", "wrong_target", "reinforced:reinforce:n3"),
            ]
        }
        noise = {"n1": "plain note", "n2": "scope project-a-api peers", "n3": "plain"}
        incoming = {"a": "x", "b": "y", "c": "uses project-a"}

        sendable, left = mod.split_sendable(mod.relation_events(runs), noise, incoming)

        assert [e.pair_id for e in sendable] == ["a"]
        assert [e.pair_id for e in left] == ["b", "c"]


class TestClientNames:
    @pytest.mark.parametrize(
        "text",
        [
            "peers like project-a-api",
            "the Project-a repo",
            "scope project-f-web",
            "scope 'project-k' is peer",
        ],
    )
    def test_flags_client_mentions(self, text: str) -> None:
        assert mod.mentions_client(text)

    @pytest.mark.parametrize(
        "text",
        [
            "auto-derives api_mode",
            "GPT auto-review comments",
            "no-auto-commit rule",
            "a mobile phone",
        ],
    )
    def test_keeps_ordinary_words(self, text: str) -> None:
        assert not mod.mentions_client(text)


class TestVerdicts:
    @pytest.mark.parametrize(
        ("action", "label", "expected"),
        [
            ("reinforce", "restatement", "right"),
            ("supersede", "replacement", "right"),
            ("reinforce", "follow_up", "wrong"),
            ("supersede", "distinct", "wrong"),
            ("reinforce", "replacement", "wrong"),
            ("supersede", "restatement", "wrong"),
            ("reinforce", None, "unkeyed"),
        ],
    )
    def test_relation_verdict(self, action: str, label: str | None, expected: str) -> None:
        assert mod.relation_verdict(action, label) == expected

    def test_tallies_count_wrong_reinforces(self) -> None:
        save = [
            save_row("a", "wrong_target", "reinforced:reinforce:n1"),
            save_row("b", "wrong_target", "approved:supersede:n2"),
            save_row("c", "wrong_target", "reinforced:reinforce:n3"),
        ]
        judged = mod.NoiseJudgments(
            earlier={},
            relation={
                ("a", "n1"): judged_row("a", "n1", "distinct"),
                ("b", "n2"): judged_row("b", "n2", "replacement"),
                ("c", "n3"): judged_row("c", "n3", None),
            },
        )

        relation = mod.noise_tallies(save, judged)["relation suggestion"]

        assert dict(relation.counts) == {"wrong": 1, "right": 1, "unkeyed": 1}
        assert relation.reinforced_wrong == [("a", "n1")]

    def test_containment_and_ask_use_the_earlier_judgments(self) -> None:
        save = [
            save_row("a", "wrong_target", "approved:contained:n1"),
            save_row("b", "wrong_target", "asked:n2"),
            save_row("c", "wrong_target", "asked:n3"),
        ]
        earlier = {
            ("a", "n1"): JudgedRow(
                pair_id="a", event="contained", noise_id="n1", new_covers_old="yes"
            ),
        }
        asked = {
            ("b", "n2"): JudgedRow(
                pair_id="b", event="asked", noise_id="n2", new_covers_old="no", one_record="no"
            ),
        }

        tallies = mod.noise_tallies(
            save, mod.NoiseJudgments(earlier=earlier, relation={}, asked=asked)
        )

        assert dict(tallies["containment"].counts) == {"right": 1}
        assert dict(tallies["ask"].counts) == {"wrong": 1, "left_out": 1}

    def test_a_covers_ask_on_a_noise_row_is_an_ask_event_judged_by_one_record(self) -> None:
        save = [
            save_row("a", "wrong_target", "asked:covers:n1"),
            save_row("b", "wrong_target", "asked:covers:n2"),
        ]
        asked = {
            ("a", "n1"): JudgedRow(
                pair_id="a", event="asked", noise_id="n1", new_covers_old=None, one_record="yes"
            ),
            ("b", "n2"): JudgedRow(
                pair_id="b", event="asked", noise_id="n2", new_covers_old=None, one_record="no"
            ),
        }

        tallies = mod.noise_tallies(save, mod.NoiseJudgments(earlier={}, relation={}, asked=asked))

        assert dict(tallies["ask"].counts) == {"right": 1, "wrong": 1}
        assert dict(tallies["containment"].counts) == {}


class TestExtras:
    def test_counts_noise_reinforce_on_restatement_and_supersede_on_replacement(self) -> None:
        save = [
            save_row("a", "wrong_target", "reinforced:reinforce:n1"),
            save_row("b", "wrong_target", "approved:supersede:n2"),
            save_row("c", "wrong_target", "reinforced:reinforce:n3"),
            save_row("d", "wrong_target", "approved:supersede:n4"),
            save_row("e", "caught", "reinforced:reinforce:seed"),
        ]
        relation = {"a": "restatement", "b": "replacement", "c": "follow_up", "d": "restatement"}
        judged = {
            ("a", "n1"): judged_row("a", "n1", "restatement"),
            ("b", "n2"): judged_row("b", "n2", "replacement"),
            ("c", "n3"): judged_row("c", "n3", "restatement"),
            ("d", "n4"): judged_row("d", "n4", "replacement"),
        }

        resolved, caught = mod.noise_extras(save, relation, judged, frozenset())

        assert resolved == ["a"]
        assert caught == ["b"]

    def test_skips_pairs_the_replay_folded(self) -> None:
        save = [save_row("a", "wrong_target", "reinforced:reinforce:n1")]
        judged = {("a", "n1"): judged_row("a", "n1", "restatement")}

        resolved, _ = mod.noise_extras(save, {"a": "restatement"}, judged, frozenset({"a"}))

        assert resolved == []


class TestInterruptions:
    def test_counts_asks_flags_and_folds_on_new_pairs(self) -> None:
        save = [
            save_row("a", "asked", "asked:n1"),
            save_row("b", "wrong_target", "asked:n2"),
            save_row("c", "flagged", "approved:supersede:s"),
            save_row("d", "new", "approved:"),
            save_row("e", "asked", "asked:n5"),
            save_row("f", "asked", "asked:covers:n6"),
        ]
        relation = {
            "a": "follow_up",
            "b": "distinct",
            "c": "distinct",
            "d": "follow_up",
            "e": "restatement",
            "f": "follow_up",
        }

        result = mod.interruptions(save, relation, frozenset({"a", "d", "e"}))

        assert result.model_dump() == {
            "keyed_new": 5,
            "asked": 3,
            "asked_seed": 2,
            "flagged": 1,
            "folded": 2,
            "covers_asked": 1,
        }


class TestJudging:
    def test_judges_keys_and_resumes(self, tmp_path: pathlib.Path) -> None:
        events = [
            mod.RelationEvent(pair_id="a", noise_id="n1", actions={"run-1": "reinforce"}),
            mod.RelationEvent(pair_id="b", noise_id="n2", actions={"run-1": "supersede"}),
        ]
        noise = {"n1": "old one", "n2": "old two"}
        incoming = {"a": "new one", "b": "new two"}
        seats = fake_seats("restatement")

        rows = mod.judge_events(events, noise, incoming, tmp_path, seats)

        assert [row.label for row in rows] == ["restatement", "restatement"]
        assert all(row.labels == {"claude": "restatement", "codex": "restatement"} for row in rows)
        calls = [len(seat.adapter.prompts) for seat in seats]  # type: ignore[attr-defined]

        again = mod.judge_events(events, noise, incoming, tmp_path, fake_seats("distinct"))

        assert [row.label for row in again] == ["restatement", "restatement"]
        assert calls == [1, 1]

    def test_split_judges_leave_the_event_unkeyed(self, tmp_path: pathlib.Path) -> None:
        events = [mod.RelationEvent(pair_id="a", noise_id="n1", actions={"run-1": "reinforce"})]
        seats = [
            judges.Seat(judge="claude", model="opus", adapter=FakeAdapter("follow_up")),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=FakeAdapter("restatement")),
        ]

        rows = mod.judge_events(events, {"n1": "o"}, {"a": "i"}, tmp_path, seats)

        assert rows[0].label is None
        assert rows[0].labels == {"claude": "follow_up", "codex": "restatement"}


def test_action_of() -> None:
    assert mod.action_of("reinforced:reinforce:n") == "reinforce"
    assert mod.action_of("approved:supersede:n") == "supersede"


class TestReport:
    def test_published_rows_read_both_systems(self, tmp_path: pathlib.Path) -> None:
        report = tmp_path / "report.md"
        report.write_text(
            "| Restatement caught | 1 of 39, 3% | 24 of 39, 62% |\n"
            "| Restatement noticed (caught or flagged) | 34 of 39, 87% | 26 of 39, 67% |\n"
            "| Replacement flagged | 41 of 42, 98% | 36 of 42, 86% |\n"
            "| Follow-up or distinct flagged (false alarm) | 89 of 121, 74% | 4 of 121, 3% |\n",
            encoding="utf-8",
        )

        rows = noise_report.published_rows(report)

        assert rows == {
            "caught": ("1 of 39", "24 of 39"),
            "noticed": ("34 of 39", "26 of 39"),
            "flagged": ("41 of 42", "36 of 42"),
            "alarm": ("89 of 121", "4 of 121"),
        }

    def test_missing_row_raises(self, tmp_path: pathlib.Path) -> None:
        report = tmp_path / "report.md"
        report.write_text("nothing", encoding="utf-8")

        with pytest.raises(ValueError, match="not found"):
            noise_report.published_rows(report)


def write_save(directory: pathlib.Path, rows: list[SaveRow]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "save-v2.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )


def answered(pair_id: str, end_state: str) -> SaveRow:
    row = save_row(pair_id, "asked", "asked:seed")
    outcome = {
        "replay": "fold",
        "status": "folded",
        "ask_answer": "fold",
        "end_state": end_state,
        "existing_live": False,
        "existing_superseded_by_fold": True,
        "fold_live": True,
        "held_live": False,
        "fold_covers_old": None,
        "fold_covers_new": None,
        "error": None,
    }

    return SaveRow.model_validate({**row.model_dump(), "answer": outcome})


class TestRuns:
    def test_discovers_every_run_directory_in_numeric_order(self, tmp_path: pathlib.Path) -> None:
        for name in ("run-2", "run-10", "run-1"):
            write_save(tmp_path / name, [])
        (tmp_path / "run-3").mkdir()

        assert mod.discover_runs(tmp_path) == ["run-1", "run-2", "run-10"]

    def test_explicit_runs_are_kept_and_a_missing_one_raises(self, tmp_path: pathlib.Path) -> None:
        write_save(tmp_path / "run-5", [])

        assert mod.resolve_runs(tmp_path, ["run-5"]) == ["run-5"]

        with pytest.raises(ValueError, match="run-6"):
            mod.resolve_runs(tmp_path, ["run-5", "run-6"])

    def test_no_runs_at_all_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="no run directories"):
            mod.resolve_runs(tmp_path, [])

    def test_folded_pairs_come_from_the_save_rows_and_the_old_answer_file(
        self, tmp_path: pathlib.Path
    ) -> None:
        write_save(
            tmp_path,
            [answered("a", "folded"), answered("b", "both_live"), answered("c", "no_reply")],
        )
        (tmp_path / "answer-v2.jsonl").write_text(
            json.dumps({"pair_id": "d", "end_state": "folded"})
            + "\n"
            + json.dumps({"pair_id": "e", "end_state": "both_live"})
            + "\n",
            encoding="utf-8",
        )

        assert mod.folded_pairs(tmp_path) == frozenset({"a", "d"})


def result_for(name: str, right: int, wrong: int) -> noise_report.RunResult:
    tallies = {kind: mod.Tally() for kind in mod.KINDS}

    for kind in mod.KINDS:
        tallies[kind].counts.update({"right": right, "wrong": wrong})

    readings = mod.Readings(
        restatements=39,
        replacements=42,
        seed_resolved=30 + right,
        seed_caught=37,
        extra_resolved=["x"] * right,
        extra_caught=[],
        loose_resolved=0,
        loose_caught=0,
    )
    interrupted = mod.Interruptions(keyed_new=121, asked=3, asked_seed=3, flagged=1, folded=2)

    return noise_report.RunResult(name, readings, interrupted, tallies)


class TestSummary:
    def test_reading_lines_cover_any_number_of_runs(self) -> None:
        results = [result_for(f"run-{index}", index, 0) for index in range(1, 7)]

        lines = noise_report.reading_lines(results, 0.47, 16, 25, 3, 0)

        assert "run-6 6 right, 0 wrong" in lines[0]
        assert "run-1 1 right" in lines[1]
        assert sum("run-3 " in line for line in lines) == 4

    def test_the_summary_states_the_rules_both_readings_and_the_comparison_rows(self) -> None:
        results = [
            noise_report.RunResult(
                "first run", result_for("x", 0, 0).readings, result_for("x", 0, 0).interrupted, None
            ),
            *[result_for(f"run-{index}", index, 0) for index in (5, 6)],
        ]
        published = {
            "noticed": ("34 of 39", "26 of 39"),
            "flagged": ("41 of 42", "36 of 42"),
            "alarm": ("89 of 121", "4 of 121"),
            "caught": ("1 of 39", "24 of 39"),
        }
        text = noise_report.summary_text(
            results,
            published,
            {("p", "n"): {"run-5": "reinforce"}},
            0,
            {"cohens_kappa": 0.5, "items_agreed": 10},
            12,
        )

        assert "Restatement rule" in text
        assert "graded good" in text
        assert "Reading `no new duplicate`" in text
        assert "Reading `seed only`" in text
        assert "| v1 published |" in text
        assert "| v2 published" in text
        assert "| Bench Opus agent-fold |" in text
        assert "| replay run-6, no new duplicate |" in text
        assert "| run-5 | seed only |" in text


class TestContainmentJudging:
    def test_new_events_are_judged_and_earlier_labels_are_kept(
        self, tmp_path: pathlib.Path
    ) -> None:
        earlier = [JudgedRow(pair_id="a", event="contained", noise_id="n1", new_covers_old="yes")]
        noise = {"n1": "old one", "n2": "old two", "n3": "Project-a note"}
        incoming = {"a": "new one", "b": "new two", "c": "new three"}
        yes = [FakeAdapter("yes"), FakeAdapter("yes")]
        seats = [
            judges.Seat(judge="claude", model="opus", adapter=yes[0]),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=yes[1]),
        ]
        first = mod.judge_containment([("a", "n1")], earlier, noise, incoming, tmp_path, seats)
        no = [FakeAdapter("no"), FakeAdapter("no")]
        again_seats = [
            judges.Seat(judge="claude", model="opus", adapter=no[0]),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=no[1]),
        ]
        rows = mod.judge_containment(
            [("a", "n1"), ("b", "n2"), ("c", "n3")], earlier, noise, incoming, tmp_path, again_seats
        )

        assert [row.new_covers_old for row in first] == ["yes"]
        assert [(row.pair_id, row.new_covers_old) for row in rows] == [("a", "yes"), ("b", "no")]
        prompts = [prompt for adapter in no for prompt in adapter.prompts]
        assert len(prompts) == 2
        assert all("old two" in prompt and "old one" not in prompt for prompt in prompts)

    def test_containment_events_are_collected_from_every_run(self) -> None:
        runs = {
            "run-1": [save_row("a", "wrong_target", "approved:contained:n1")],
            "run-2": [
                save_row("a", "wrong_target", "approved:contained:n1"),
                save_row("b", "wrong_target", "asked:n2"),
            ],
        }

        assert mod.containment_pairs(runs) == [("a", "n1")]


class TaskAwareAdapter:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        match = re.search(r"There are (\d+) items", prompt)
        assert match is not None
        label = "restatement" if "restatement" in prompt else "yes"
        verdicts = [
            {"index": index, "label": label, "reason": "r"}
            for index in range(1, int(match.group(1)) + 1)
        ]

        return json.dumps({"verdicts": verdicts})


def task_seats() -> list[judges.Seat]:
    return [
        judges.Seat(judge="claude", model="opus", adapter=TaskAwareAdapter()),
        judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=TaskAwareAdapter()),
    ]


def pair_line(pair_id: str, incoming: str) -> str:
    return json.dumps(
        {
            "pair_id": pair_id,
            "source": "golden",
            "kind": "fact",
            "existing": "seed",
            "incoming": incoming,
            "similarity": None,
            "old_label": None,
        }
    )


class TestContainmentIdentity:
    def test_a_rerun_over_its_own_output_keeps_ids_and_does_not_rejudge(
        self, tmp_path: pathlib.Path
    ) -> None:
        noise = {"n1": "old one", "n2": "old two"}
        incoming = {"a": "new one"}
        events = [("a", "n1"), ("a", "n2")]
        first = mod.judge_containment(events, [], noise, incoming, tmp_path, fake_seats("yes"))
        adapters = [FakeAdapter("no"), FakeAdapter("no")]
        no = [
            judges.Seat(judge="claude", model="opus", adapter=adapters[0]),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=adapters[1]),
        ]

        again = mod.judge_containment(events, first, noise, incoming, tmp_path, no)

        assert [(row.pair_id, row.noise_id, row.new_covers_old) for row in again] == [
            ("a", "n1", "yes"),
            ("a", "n2", "yes"),
        ]
        assert [row.item_id for row in again] == [row.item_id for row in first]
        assert all(adapter.prompts == [] for adapter in adapters)

    def test_a_legacy_row_without_an_item_id_keeps_its_pair_only_id(
        self, tmp_path: pathlib.Path
    ) -> None:
        legacy = [JudgedRow(pair_id="a", event="contained", noise_id="n1", new_covers_old="yes")]

        rows = mod.judge_containment(
            [("a", "n1")], legacy, {"n1": "old"}, {"a": "new"}, tmp_path, fake_seats("yes")
        )
        again = mod.judge_containment(
            [("a", "n1")], rows, {"n1": "old"}, {"a": "new"}, tmp_path, fake_seats("no")
        )

        assert rows[0].item_id == "a#new_covers_old"
        assert again[0].item_id == "a#new_covers_old"
        assert again[0].new_covers_old == "yes"


class TestAskJudging:
    NOISE: ClassVar[dict[str, str]] = {
        "n1": "old one",
        "n2": "old two",
        "n3": "Project-a old three",
    }
    INCOMING: ClassVar[dict[str, str]] = {"a": "new one", "b": "new two", "c": "new three"}
    NOTES: ClassVar[dict[str, str]] = {"a": "fold one", "b": "fold two", "c": "fold three"}

    def test_each_ask_event_is_judged_on_one_record_with_the_pair_fold(
        self, tmp_path: pathlib.Path
    ) -> None:
        adapters = [FakeAdapter("yes"), FakeAdapter("yes")]
        seats = [
            judges.Seat(judge="claude", model="opus", adapter=adapters[0]),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=adapters[1]),
        ]

        rows = mod.judge_asks(
            [("a", "n1")], [], self.NOISE, self.INCOMING, self.NOTES, tmp_path, seats
        )

        assert [(row.pair_id, row.event, row.noise_id, row.one_record) for row in rows] == [
            ("a", "asked", "n1", "yes")
        ]
        assert rows[0].new_covers_old is None
        assert rows[0].item_id == "a|n1#one_record"
        assert rows[0].one_record_labels == {"claude": "yes", "codex": "yes"}
        prompt = adapters[0].prompts[0]
        assert "QUESTION: one_record" in prompt
        assert "EXISTING: old one" in prompt
        assert "INCOMING: new one" in prompt
        assert "FOLD: fold one" in prompt

    def test_an_event_without_a_pair_fold_or_with_a_client_name_is_left_out(
        self, tmp_path: pathlib.Path
    ) -> None:
        notes = {"a": "fold one", "c": "fold three"}

        rows = mod.judge_asks(
            [("a", "n1"), ("b", "n2"), ("c", "n3")],
            [],
            self.NOISE,
            self.INCOMING,
            notes,
            tmp_path,
            fake_seats("yes"),
        )

        assert [(row.pair_id, row.noise_id) for row in rows] == [("a", "n1")]

    def test_a_recorded_row_keeps_its_id_its_covers_answer_and_its_labels(
        self, tmp_path: pathlib.Path
    ) -> None:
        legacy = [
            JudgedRow(
                pair_id="a",
                event="asked",
                noise_id="n1",
                new_covers_old="no",
                one_record="no",
            )
        ]
        first = mod.judge_asks(
            [("a", "n1")], legacy, self.NOISE, self.INCOMING, self.NOTES, tmp_path, fake_seats("no")
        )
        adapters = [FakeAdapter("yes"), FakeAdapter("yes")]
        seats = [
            judges.Seat(judge="claude", model="opus", adapter=adapters[0]),
            judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=adapters[1]),
        ]

        again = mod.judge_asks(
            [("a", "n1"), ("b", "n2")],
            first,
            self.NOISE,
            self.INCOMING,
            self.NOTES,
            tmp_path,
            seats,
        )

        assert first[0].item_id == "a#one_record"
        assert [
            (row.pair_id, row.item_id, row.new_covers_old, row.one_record) for row in again
        ] == [
            ("a", "a#one_record", "no", "no"),
            ("b", "b|n2#one_record", None, "yes"),
        ]
        prompts = [prompt for adapter in adapters for prompt in adapter.prompts]
        assert len(prompts) == 2
        assert all("old two" in prompt and "old one" not in prompt for prompt in prompts)

    def test_a_recorded_row_without_a_fold_note_is_kept_as_it_was(
        self, tmp_path: pathlib.Path
    ) -> None:
        legacy = [
            JudgedRow(
                pair_id="a", event="asked", noise_id="n1", new_covers_old=None, one_record="no"
            )
        ]

        rows = mod.judge_asks(
            [], legacy, self.NOISE, self.INCOMING, {}, tmp_path, fake_seats("yes")
        )

        assert [(row.pair_id, row.one_record) for row in rows] == [("a", "no")]

    def test_ask_events_come_from_every_run_including_covers_asks(self) -> None:
        runs = {
            "run-1": [save_row("a", "wrong_target", "asked:covers:n1")],
            "run-2": [
                save_row("a", "wrong_target", "asked:covers:n1"),
                save_row("b", "wrong_target", "asked:n2"),
                save_row("c", "wrong_target", "approved:contained:n3"),
            ],
        }

        assert mod.ask_pairs(runs) == [("a", "n1"), ("b", "n2")]

    def test_reply_notes_read_both_ask_files_and_only_folds_of_the_replay_arm(
        self, tmp_path: pathlib.Path
    ) -> None:
        experiments = tmp_path / "experiments"
        reply = {
            "model": "opus",
            "wording": "fold_default",
            "framing": "in_task",
            "reason": None,
            "raw": None,
            "elapsed_ms": 1.0,
            "attempts": 1,
            "error": None,
        }

        def line(pair_id: str, arm: str, action: str, note: str | None) -> str:
            return json.dumps(
                {**reply, "pair_id": pair_id, "arm": arm, "action": action, "note": note}
            )

        (experiments / "agent-fold").mkdir(parents=True)
        (experiments / "prod-replay").mkdir(parents=True)
        (experiments / "agent-fold" / "asks.jsonl").write_text(
            line("band", mod.REPLAY_ARM, "fold", "band note")
            + "\n"
            + line("kept", mod.REPLAY_ARM, "keep_both", None)
            + "\n"
            + line("other", "sonnet-fold_default-in_task", "fold", "sonnet note")
            + "\n",
            encoding="utf-8",
        )
        (experiments / "prod-replay" / "asks-covers.jsonl").write_text(
            line("outside", mod.REPLAY_ARM, "fold", "outside note") + "\n", encoding="utf-8"
        )

        assert mod.reply_notes(tmp_path) == {"band": "band note", "outside": "outside note"}

    def test_the_covers_fold_key_is_read_from_the_replay_directory(
        self, tmp_path: pathlib.Path
    ) -> None:
        folder = tmp_path / "covers-fold-judge"
        folder.mkdir()
        (folder / "key-fold.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "item_id": f"p~{mod.REPLAY_ARM}#{question}",
                        "label": "yes",
                        "source": "agreed",
                    }
                )
                + "\n"
                for question in fold_mod.QUESTIONS
            ),
            encoding="utf-8",
        )

        grades = mod.covers_grades(tmp_path)

        assert list(grades) == [("p", mod.REPLAY_ARM)]
        assert grades[("p", mod.REPLAY_ARM)].status == "good"
        assert mod.covers_grades(tmp_path / "missing") == {}


class TestJudgeCommand:
    def build(self, tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        base = tmp_path / "base"
        replay = base / "experiments" / "prod-replay"
        replay.mkdir(parents=True)
        (base / "noise.jsonl").write_text(
            json.dumps({"id": "n1", "content": "old one"})
            + "\n"
            + json.dumps({"id": "n2", "content": "old two"})
            + "\n",
            encoding="utf-8",
        )
        (base / "pairs.jsonl").write_text(
            pair_line("a", "new one") + "\n" + pair_line("b", "new two") + "\n", encoding="utf-8"
        )
        write_save(replay, [])
        write_save(replay / "run-1", [save_row("a", "wrong_target", "reinforced:reinforce:n1")])
        write_save(replay / "run-2", [save_row("b", "wrong_target", "approved:supersede:n2")])
        (replay / "noise-judged.jsonl").write_text(
            JudgedRow(
                pair_id="a", event="contained", noise_id="n1", new_covers_old="yes"
            ).model_dump_json()
            + "\n",
            encoding="utf-8",
        )

        return base, replay

    def relation_rows(self, replay: pathlib.Path) -> list[tuple[str, str, str | None]]:
        lines = (replay / "noise-relation-judged.jsonl").read_text(encoding="utf-8").splitlines()
        rows = [mod.RelationJudgedRow.model_validate_json(line) for line in lines]

        return [(row.pair_id, row.noise_id, row.label) for row in rows]

    def test_a_relation_event_on_a_containment_row_is_still_judged(
        self, tmp_path: pathlib.Path
    ) -> None:
        base, replay = self.build(tmp_path)

        mod.judge_command(base, replay, ["run-1"], task_seats())

        assert self.relation_rows(replay) == [("a", "n1", "restatement")]

    def test_a_later_run_over_a_subset_keeps_the_earlier_relation_labels(
        self, tmp_path: pathlib.Path
    ) -> None:
        base, replay = self.build(tmp_path)

        mod.judge_command(base, replay, ["run-1"], task_seats())
        mod.judge_command(base, replay, ["run-2"], task_seats())

        assert self.relation_rows(replay) == [
            ("a", "n1", "restatement"),
            ("b", "n2", "restatement"),
        ]
        kept = read_jsonl(replay / "noise-judged.jsonl", JudgedRow)
        assert [(row.pair_id, row.noise_id, row.new_covers_old) for row in kept] == [
            ("a", "n1", "yes")
        ]


class TestWrongReinforceText:
    def test_the_reinforced_target_is_listed_not_the_first_judged_target_for_the_pair(self) -> None:
        run_1 = result_for("run-1", 0, 1)
        run_2 = result_for("run-2", 0, 1)

        for run in (run_1, run_2):
            assert run.tallies is not None
            run.tallies["relation suggestion"].reinforced_wrong.append(("a", "n2"))

        judged = mod.NoiseJudgments(
            earlier={},
            relation={
                ("a", "n1"): judged_row("a", "n1", "restatement"),
                ("a", "n2"): judged_row("a", "n2", "distinct"),
            },
        )
        noise = {"n1": "first target text", "n2": "second target text"}

        text = noise_report.wrong_reinforce_text([run_1, run_2], judged, noise, {"a": "new"}, {})

        assert "- Noise id: n2" in text
        assert "- Existing (noise note): second target text" in text
        assert "- Key: distinct" in text
        assert "first target text" not in text.split("## Reinforces with no key")[0]
        assert "- Runs: run-1, run-2" in text


def neighbour_row(noise_id: str, probability: float) -> NeighbourRow:
    return NeighbourRow.model_validate(
        {
            "id": noise_id,
            "source": "records",
            "similarity": 0.9,
            "relation_pick": None,
            "relation_confidence": None,
            "new_covers_old": probability,
            "same_subject": None,
            "is_seed": False,
        }
    )


class TestWrongContainments:
    def build(self) -> tuple[dict[str, list[SaveRow]], mod.NoiseJudgments]:
        hit = save_row("a", "wrong_target", "approved:contained:n1").model_copy(
            update={"neighbours": [neighbour_row("n1", 0.96)]}
        )
        asked = save_row("a", "wrong_target", "asked:n1")
        right = save_row("b", "wrong_target", "approved:contained:n2")
        runs = {"run-1": [hit, right], "run-2": [asked, right], "run-3": [hit, right]}
        earlier = {
            ("a", "n1"): JudgedRow(
                pair_id="a", event="contained", noise_id="n1", new_covers_old="no"
            ),
            ("b", "n2"): JudgedRow(
                pair_id="b", event="contained", noise_id="n2", new_covers_old="yes"
            ),
        }

        return runs, mod.NoiseJudgments(earlier=earlier, relation={})

    def test_counts_runs_with_a_keyed_wrong_containment_and_their_probabilities(self) -> None:
        runs, judged = self.build()

        found = noise_report.wrong_containments(runs, judged)

        assert found == {("a", "n1"): {"run-1": 0.96, "run-3": 0.96}}

    def test_the_section_quotes_both_notes_in_full(self) -> None:
        runs, judged = self.build()
        incoming = {"a": "first line\n\nsecond line", "b": "other"}
        noise = {"n1": "noise text", "n2": "other noise"}

        text = "\n".join(noise_report.wrong_containment_lines(runs, judged, noise, incoming))

        assert "2 of 3 runs contain a wrong noise containment (run-1, run-3)" in text
        assert "- a on noise row n1: run-1 0.96, run-3 0.96." in text
        assert "  > first line\n  >\n  > second line" in text
        assert "  > noise text" in text
        assert "other noise" not in text
