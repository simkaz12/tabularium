from __future__ import annotations

import json
import pathlib
import re
import sys

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
import eval_v1v2.covers_fold_judge as mod  # noqa: E402
import eval_v1v2.judges as judges  # noqa: E402
from eval_v1v2.rows import PairRow  # noqa: E402

sys.path.pop(0)

ARM = "opus-fold_default-in_task"


class FakeAdapter:
    def __init__(self, label: str) -> None:
        self.label = label
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        match = re.search(r"There are (\d+) items", prompt)
        assert match is not None
        count = int(match.group(1))

        return json.dumps(
            {
                "verdicts": [
                    {"index": index, "label": self.label, "reason": "r"}
                    for index in range(1, count + 1)
                ]
            }
        )


def pair(pair_id: str, existing: str, incoming: str) -> PairRow:
    return PairRow(
        pair_id=pair_id,
        source="golden",
        kind="fact",
        existing=existing,
        incoming=incoming,
        similarity=0.8,
        old_label=None,
    )


def ask_line(pair_id: str, action: str, note: str | None) -> str:
    return json.dumps(
        {
            "pair_id": pair_id,
            "arm": ARM,
            "model": "opus",
            "wording": "fold_default",
            "framing": "in_task",
            "action": action,
            "note": note,
            "reason": None,
            "raw": None,
            "elapsed_ms": 1.0,
            "attempts": 1,
            "error": None,
        }
    )


def test_fold_notes_keeps_folded_replies_only(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "asks.jsonl"
    path.write_text(
        "\n".join([ask_line("a", "fold", "merged"), ask_line("b", "continue", None)]) + "\n",
        encoding="utf-8",
    )

    assert mod.fold_notes(path, ARM) == {"a": "merged"}


def test_fold_items_screens_client_names_and_asks_four_questions() -> None:
    pairs = {
        "a": pair("a", "old a", "new a"),
        "b": pair("b", "old project-a note", "new b"),
        "c": pair("c", "old c", "new c"),
    }

    items, left_out = mod.fold_items(
        {"a": "fold a", "b": "fold b", "c": "fold of project-f-api"}, pairs, ARM
    )

    assert left_out == ["b", "c"]
    assert [item.item_id for item in items] == [
        f"a~{ARM}#nothing_lost",
        f"a~{ARM}#nothing_invented",
        f"a~{ARM}#newer_wins",
        f"a~{ARM}#one_record",
    ]
    assert all(item.fields["fold"] == "fold a" for item in items)


def test_judge_command_writes_a_key_for_sendable_folds(tmp_path: pathlib.Path) -> None:
    base = tmp_path
    replay = base / "experiments" / "prod-replay"
    replay.mkdir(parents=True)
    (base / "pairs.jsonl").write_text(
        pair("a", "old a", "new a").model_dump_json() + "\n", encoding="utf-8"
    )
    (replay / "asks-covers.jsonl").write_text(
        ask_line("a", "fold", "merged") + "\n", encoding="utf-8"
    )
    seats = [
        judges.Seat(judge="claude", model="opus", adapter=FakeAdapter("yes")),
        judges.Seat(judge="codex", model="gpt-5.6-sol", adapter=FakeAdapter("yes")),
    ]

    keyed, left_out = mod.judge_command(base, ARM, seats)

    assert (keyed, left_out) == (4, [])
    key = (replay / "covers-fold-judge" / "key-fold.jsonl").read_text(encoding="utf-8")
    assert f"a~{ARM}#one_record" in key


def test_screen_only_persists_the_left_out_pairs_for_the_check(tmp_path: pathlib.Path) -> None:
    replay = tmp_path / "experiments" / "prod-replay"
    replay.mkdir(parents=True)
    (tmp_path / "pairs.jsonl").write_text(
        pair("a", "old a", "new a").model_dump_json()
        + "\n"
        + pair("b", "old project-a note", "new b").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (replay / "asks-covers.jsonl").write_text(
        ask_line("a", "fold", "merged") + "\n" + ask_line("b", "fold", "merged b") + "\n",
        encoding="utf-8",
    )

    assert mod.main(["--base", str(tmp_path), "--screen-only"]) == 0
    assert (replay / "covers-fold-judge" / f"left-out-fold-{ARM}.txt").read_text() == "b\n"
