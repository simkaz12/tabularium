from __future__ import annotations

import json
import pathlib
import re
import sys
from collections.abc import Callable

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import judges  # noqa: E402
from eval_v1v2.rows import KeyRow, LabelRow, read_jsonl, write_jsonl  # noqa: E402
from tabularium.engine.judge import JudgeUnavailable  # noqa: E402

sys.path.pop(0)

Responder = Callable[[int, str], str]


class FakeAdapter:
    def __init__(self, responder: Responder) -> None:
        self.prompts: list[str] = []
        self._responder = responder

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responder(len(self.prompts), prompt)


def _items(count: int) -> list[judges.Item]:
    return [
        judges.Item(f"p{i}", {"existing": f"old note {i}", "incoming": f"new note {i}"})
        for i in range(count)
    ]


def _expected_by_note(answers: dict[str, str | int]) -> Responder:
    def respond(_call: int, prompt: str) -> str:
        found = re.findall(r"\[(\d+)\]\nEXISTING: old note \d+\nINCOMING: new note (\d+)", prompt)
        verdicts = [
            {"index": int(position), "label": answers[f"p{marker}"], "reason": "because"}
            for position, marker in found
        ]
        return json.dumps({"verdicts": verdicts})

    return respond


def _seat(judge: str, adapter: FakeAdapter) -> judges.Seat:
    return judges.Seat(judge=judge, model=f"{judge}-fake", adapter=adapter)  # type: ignore[arg-type]


def _label(item_id: str, judge: str, label: str | int) -> LabelRow:
    return LabelRow.model_validate(
        {"item_id": item_id, "judge": judge, "model": "m", "label": label, "reason": "r"}
    )


class TestAgreement:
    def test_known_answer(self) -> None:
        pairs = {
            "a": ("restatement", "restatement"),
            "b": ("restatement", "restatement"),
            "c": ("distinct", "distinct"),
            "d": ("distinct", "restatement"),
        }
        rows = [
            _label(item, judge, labels[index])
            for item, labels in pairs.items()
            for index, judge in enumerate(("claude", "codex"))
        ]

        stats = judges.agreement_stats(rows)

        assert stats["items_labelled_by_both"] == 4
        assert stats["items_agreed"] == 3
        assert stats["raw_agreement"] == 0.75
        assert stats["cohens_kappa"] == pytest.approx(0.5)
        assert stats["confusion"] == {
            "restatement": {"restatement": 2},
            "distinct": {"distinct": 1, "restatement": 1},
        }

    def test_key_keeps_only_agreed_and_preserves_owner_rows(self) -> None:
        rows = [
            _label("a", "claude", "restatement"),
            _label("a", "codex", "restatement"),
            _label("b", "claude", "distinct"),
            _label("b", "codex", "replacement"),
            _label("c", "claude", "distinct"),
            _label("c", "codex", "replacement"),
        ]
        prior = [KeyRow(item_id="c", label="follow_up", source="owner")]

        key = {row.item_id: row for row in judges.agreed_key(rows, prior)}

        assert set(key) == {"a", "c"}
        assert key["a"].source == "agreed"
        assert key["c"].label == "follow_up"
        assert key["c"].source == "owner"

    def test_all_one_label_does_not_divide_by_zero(self) -> None:
        rows = [_label("a", j, "distinct") for j in ("claude", "codex")]

        assert judges.agreement_stats(rows)["cohens_kappa"] == 1.0


class TestRunTask:
    def test_labels_key_and_agreement_written(self, tmp_path: pathlib.Path) -> None:
        answers: dict[str, str | int] = {"p0": "restatement", "p1": "distinct", "p2": "replacement"}
        claude = FakeAdapter(_expected_by_note(answers))
        codex = FakeAdapter(_expected_by_note({**answers, "p2": "follow_up"}))

        outcome = judges.run_task(
            "relation",
            _items(3),
            seats=[_seat("claude", claude), _seat("codex", codex)],
            out_dir=tmp_path,
        )

        labels = read_jsonl(tmp_path / "labels-relation.jsonl", LabelRow)
        key = read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)
        agreement = json.loads((tmp_path / "agreement-relation.json").read_text())
        assert len(labels) == 6
        assert {row.model for row in labels} == {"claude-fake", "codex-fake"}
        assert {(row.item_id, row.label) for row in key} == {
            ("p0", "restatement"),
            ("p1", "distinct"),
        }
        assert agreement["items_agreed"] == 2
        assert outcome.unlabelled == {"claude": 0, "codex": 0}

    def test_resume_skips_labelled_items(self, tmp_path: pathlib.Path) -> None:
        write_jsonl(
            tmp_path / "labels-relation.jsonl",
            [_label("p0", "claude", "restatement"), _label("p1", "claude", "distinct")],
        )
        answers: dict[str, str | int] = {"p0": "restatement", "p1": "distinct", "p2": "distinct"}
        claude = FakeAdapter(_expected_by_note(answers))
        codex = FakeAdapter(_expected_by_note(answers))

        judges.run_task(
            "relation",
            _items(3),
            seats=[_seat("claude", claude), _seat("codex", codex)],
            out_dir=tmp_path,
        )

        assert len(claude.prompts) == 1
        assert "new note 2" in claude.prompts[0]
        assert "new note 0" not in claude.prompts[0]
        assert "new note 1" not in claude.prompts[0]
        assert all(f"new note {i}" in codex.prompts[0] for i in range(3))
        assert len(read_jsonl(tmp_path / "labels-relation.jsonl", LabelRow)) == 6

    def test_malformed_reply_retries_once_then_leaves_items_unlabelled(
        self, tmp_path: pathlib.Path
    ) -> None:
        good = FakeAdapter(
            _expected_by_note({"p0": "distinct", "p1": "distinct", "p2": "distinct"})
        )
        bad = FakeAdapter(lambda _call, _prompt: "not json at all")

        outcome = judges.run_task(
            "relation",
            _items(3),
            seats=[_seat("claude", bad), _seat("codex", good)],
            out_dir=tmp_path,
        )

        labels = read_jsonl(tmp_path / "labels-relation.jsonl", LabelRow)
        assert len(bad.prompts) == 2
        assert {row.judge for row in labels} == {"codex"}
        assert outcome.unlabelled == {"claude": 3, "codex": 0}
        assert read_jsonl(tmp_path / "key-relation.jsonl", KeyRow) == []

    def test_unavailable_then_good_reply_succeeds_on_retry(self, tmp_path: pathlib.Path) -> None:
        answers: dict[str, str | int] = {"p0": "distinct", "p1": "distinct"}
        inner = _expected_by_note(answers)

        def flaky(call: int, prompt: str) -> str:
            if call == 1:
                raise JudgeUnavailable("boom")
            return inner(call, prompt)

        outcome = judges.run_task(
            "relation",
            _items(2),
            seats=[_seat("claude", FakeAdapter(flaky)), _seat("codex", FakeAdapter(inner))],
            out_dir=tmp_path,
        )

        assert outcome.unlabelled == {"claude": 0, "codex": 0}
        assert len(outcome.key) == 2

    def test_invalid_relation_label_is_rejected(self, tmp_path: pathlib.Path) -> None:
        bad = FakeAdapter(_expected_by_note({"p0": "same", "p1": "same"}))
        good = FakeAdapter(_expected_by_note({"p0": "distinct", "p1": "distinct"}))

        outcome = judges.run_task(
            "relation",
            _items(2),
            seats=[_seat("claude", bad), _seat("codex", good)],
            out_dir=tmp_path,
        )

        assert outcome.unlabelled["claude"] == 2

    def test_batches_of_twenty_and_shuffle_is_seeded(self, tmp_path: pathlib.Path) -> None:
        answers: dict[str, str | int] = {f"p{i}": "distinct" for i in range(45)}
        first = FakeAdapter(_expected_by_note(answers))
        second = FakeAdapter(_expected_by_note(answers))
        items = _items(45)

        judges.run_task(
            "relation", items, seats=[_seat("claude", first)], out_dir=tmp_path / "a", seed=7
        )
        judges.run_task(
            "relation", items, seats=[_seat("claude", second)], out_dir=tmp_path / "b", seed=7
        )

        assert [prompt.count("EXISTING:") for prompt in first.prompts] == [20, 20, 5]
        assert first.prompts == second.prompts
        in_order = [f"new note {i}" for i in range(20)]
        assert not all(text in first.prompts[0] for text in in_order)

    def test_duplicate_item_ids_are_refused(self, tmp_path: pathlib.Path) -> None:
        items = [_items(1)[0], _items(1)[0]]

        with pytest.raises(ValueError, match="duplicate"):
            judges.run_task("relation", items, seats=[], out_dir=tmp_path)


class TestPrompts:
    def test_relation_prompt_is_blind_and_defines_each_label(self) -> None:
        prompt = judges.build_prompt(judges.TASKS["relation"], _items(2))

        for label in judges.RELATION_LABELS:
            assert f"- {label}:" in prompt
        for banned in ("old_label", "store_label", "v1", "v2", "Jev", "claude", "codex"):
            assert banned not in prompt

    def test_relevance_prompt_states_scarcity_of_grade_two(self) -> None:
        item = judges.Item("q1:r1", {"question": "how do I deploy", "record": "deploy notes"})

        prompt = judges.build_prompt(judges.TASKS["relevance"], [item])

        assert "Grade 2 is rare" in prompt
        assert "about one in ten grade 2" in prompt
        assert "QUESTION: how do I deploy" in prompt
        assert "RECORD: deploy notes" in prompt

    def test_missing_fields_are_refused(self, tmp_path: pathlib.Path) -> None:
        item = judges.Item("x", {"question": "q"})

        with pytest.raises(ValueError, match="lacks fields"):
            judges.run_task("relevance", [item], seats=[], out_dir=tmp_path)


class TestRelevanceTask:
    def _items(self) -> list[judges.Item]:
        return [
            judges.Item(f"p{i}", {"question": f"q{i}", "record": f"{i} text"}) for i in range(2)
        ]

    def test_integer_grades_accepted_and_keyed(self, tmp_path: pathlib.Path) -> None:
        def respond(_call: int, prompt: str) -> str:
            count = prompt.count("QUESTION:")
            return json.dumps(
                {"verdicts": [{"index": i, "label": 1, "reason": "r"} for i in range(1, count + 1)]}
            )

        outcome = judges.run_task(
            "relevance",
            self._items(),
            seats=[_seat("claude", FakeAdapter(respond)), _seat("codex", FakeAdapter(respond))],
            out_dir=tmp_path,
        )

        assert [(row.item_id, row.label) for row in outcome.key] == [("p0", 1), ("p1", 1)]
        raw = (tmp_path / "labels-relevance.jsonl").read_text()
        assert '"label": 1' in raw

    def test_string_grade_is_rejected(self, tmp_path: pathlib.Path) -> None:
        def respond(_call: int, prompt: str) -> str:
            count = prompt.count("QUESTION:")
            return json.dumps(
                {
                    "verdicts": [
                        {"index": i, "label": "1", "reason": "r"} for i in range(1, count + 1)
                    ]
                }
            )

        outcome = judges.run_task(
            "relevance",
            self._items(),
            seats=[_seat("claude", FakeAdapter(respond))],
            out_dir=tmp_path,
        )

        assert outcome.unlabelled == {"claude": 2}
