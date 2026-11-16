from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval_v1v2.pmv2.prep import pmv2_dir
from eval_v1v2.pmv2.read import ANSWERS_NAME, ARMS, load_all
from eval_v1v2.pmv2.rows import AnswerRow, PersonaQuestion, WriteRow
from eval_v1v2.score_search import bootstrap

SEED = 20261116
RESAMPLES = 10_000
BASELINE = "v1"
COMPARED = ("v2_rerank", "v2_plain")
REPORT_NAME = "report.md"
ASKS_NAME = "asks-v2.jsonl"


def correct_by_question(answers: Sequence[AnswerRow], arm: str) -> dict[str, bool]:
    return {row.question_id: row.correct for row in answers if row.arm == arm}


def accuracy_cell(correct: Mapping[str, bool], ids: Sequence[str]) -> str:
    if not ids:
        return "n/a"
    hits = sum(1 for question_id in ids if correct.get(question_id, False))

    return f"{hits / len(ids):.1%} ({hits}/{len(ids)})"


def split_ids(questions: Sequence[PersonaQuestion]) -> dict[str, list[str]]:
    return {
        "all": [question.question_id for question in questions],
        "forget": [question.question_id for question in questions if question.forget],
        "other": [question.question_id for question in questions if not question.forget],
    }


def accuracy_table(questions: Sequence[PersonaQuestion], answers: Sequence[AnswerRow]) -> str:
    ids = split_ids(questions)
    lines = [
        "| arm | all | forget | other | unanswered |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        correct = correct_by_question(answers, arm)
        unanswered = sum(1 for question_id in ids["all"] if question_id not in correct)
        cells = [accuracy_cell(correct, ids[part]) for part in ("all", "forget", "other")]
        lines.append(f"| {arm} | {' | '.join(cells)} | {unanswered} |")

    return "\n".join(lines)


def paired_diffs(
    arm_correct: Mapping[str, bool], base_correct: Mapping[str, bool], ids: Sequence[str]
) -> list[float]:
    return [
        float(arm_correct.get(question_id, False)) - float(base_correct.get(question_id, False))
        for question_id in ids
    ]


def difference_table(questions: Sequence[PersonaQuestion], answers: Sequence[AnswerRow]) -> str:
    ids = split_ids(questions)["all"]
    base = correct_by_question(answers, BASELINE)
    lines = [
        f"| arm minus {BASELINE} | mean difference | 95% interval |",
        "|---|---|---|",
    ]
    for arm in COMPARED:
        diffs = paired_diffs(correct_by_question(answers, arm), base, ids)
        interval = bootstrap(diffs, seed=SEED, resamples=RESAMPLES)
        if interval is None:
            lines.append(f"| {arm} | n/a | n/a |")
            continue
        mean = sum(diffs) / len(diffs)
        lines.append(f"| {arm} | {mean:+.3f} | [{interval[0]:+.3f}, {interval[1]:+.3f}] |")

    return "\n".join(lines)


def ask_record_ids(asks: Sequence[Mapping[str, object]]) -> set[str]:
    return {str(row["record_id"]) for row in asks if row.get("record_id") is not None}


def write_table(writes: Sequence[WriteRow], asks: Sequence[Mapping[str, object]] = ()) -> str:
    systems = sorted({row.system for row in writes})
    statuses = sorted({row.status for row in writes})
    header = ["system", "notes in", "records stored", *statuses]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for system in systems:
        rows = [row for row in writes if row.system == system]
        counts = Counter(row.status for row in rows)
        stored = {row.record_id for row in rows if row.record_id is not None}
        if system == "v2":
            stored |= ask_record_ids(asks)
        records = len(stored)
        cells = [system, str(len(rows)), str(records), *(str(counts[s]) for s in statuses)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def ask_table(asks: Sequence[Mapping[str, object]]) -> str:
    stored = sum(1 for row in asks if row.get("record_id") is not None)
    answers = Counter(str(row.get("ask_answer")) for row in asks)
    by_answer = ", ".join(f"{key}: {value}" for key, value in sorted(answers.items())) or "none"
    lines = [
        "| v2 asks raised | ended stored | by ask_answer |",
        "|---|---|---|",
        f"| {len(asks)} | {stored} | {by_answer} |",
    ]

    return "\n".join(lines)


def load_asks(directory: Path, persona: int | None) -> list[dict[str, object]]:
    path = directory / ASKS_NAME
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    return [row for row in rows if persona is None or str(row.get("persona_id")) == str(persona)]


def load_writes(directory: Path) -> list[WriteRow]:
    rows: list[WriteRow] = []
    for path in sorted(directory.glob("writes-*.jsonl")):
        rows.extend(load_all(path, WriteRow))

    return rows


def render(
    questions: Sequence[PersonaQuestion],
    answers: Sequence[AnswerRow],
    writes: Sequence[WriteRow],
    asks: Sequence[Mapping[str, object]] = (),
) -> str:
    forget = sum(1 for question in questions if question.forget)

    return (
        "\n\n".join(
            [
                "# PersonaMem-v2 lane",
                f"{len(questions)} questions, {forget} forget. A missing answer scores 0.",
                "## Accuracy\n\n" + accuracy_table(questions, answers),
                "## Paired difference\n\n" + difference_table(questions, answers),
                "## Write side\n\n" + write_table(writes, asks),
                "## v2 asks\n\n" + ask_table(asks),
            ]
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_v1v2.pmv2.report")
    parser.add_argument("--persona", type=int)
    args = parser.parse_args(argv)

    directory = pmv2_dir()
    questions = load_all(directory / "questions.jsonl", PersonaQuestion)
    if args.persona is not None:
        questions = [q for q in questions if q.persona_id == args.persona]
    answers = load_all(directory / ANSWERS_NAME, AnswerRow)
    writes = load_writes(directory)
    if args.persona is not None:
        writes = [row for row in writes if row.persona_id == args.persona]

    target = directory / REPORT_NAME
    target.write_text(
        render(questions, answers, writes, load_asks(directory, args.persona)), encoding="utf-8"
    )
    print(f"wrote {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
