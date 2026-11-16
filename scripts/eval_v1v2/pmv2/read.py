from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, get_args

from pydantic import BaseModel

from eval_v1v2.pmv2.prep import LETTERS, pmv2_dir
from eval_v1v2.pmv2.rows import AnswerRow, Arm, Chunk, HitRow, PersonaQuestion
from tabularium.engine.judge import (
    IndexedVerdictBase,
    JudgeInvalidOutput,
    JudgeUnavailable,
    SubprocessJudgeAdapter,
    parse_batch_verdicts,
)

ARMS: tuple[str, ...] = get_args(Arm)
READER_MODEL = "opus"
BATCH_SIZE = 20
CALL_TIMEOUT = 900.0
ATTEMPTS = 2
ANSWERS_NAME = "answers.jsonl"

PREAMBLE = (
    "You answer multiple-choice questions about one user. Each question has four options, "
    "A to D, and exactly one is right for this user. Use only the material given with the "
    "question. If it does not settle the answer, pick the option you think is most likely."
)


class Caller(Protocol):
    async def call(self, prompt: str) -> str: ...


class ReaderVerdict(IndexedVerdictBase):
    answer: str


def parse_letter(raw: str) -> str | None:
    letter = raw.strip().strip(".)(:").upper()

    return letter if letter in LETTERS else None


def render_choices(question: PersonaQuestion) -> str:
    return "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(question.choices))


def render_notes(hits: Sequence[HitRow]) -> str:
    if not hits:
        return "(no notes)"

    ordered = sorted(hits, key=lambda hit: hit.rank)

    return "\n\n".join(f"{position}. {hit.text}" for position, hit in enumerate(ordered, start=1))


def render_question(question: PersonaQuestion, notes: str | None) -> str:
    parts = [f"Question: {question.query}"]
    if notes is not None:
        parts.append(f"Notes saved about the user, best match first:\n{notes}")
    parts.append(f"Options:\n{render_choices(question)}")

    return "\n".join(parts)


def render_history(chunks: Sequence[Chunk]) -> str:
    ordered = sorted(chunks, key=lambda chunk: chunk.index)
    lines = [
        f"{message.role}: {message.content}" for chunk in ordered for message in chunk.messages
    ]

    return "\n\n".join(lines)


def build_prompt(entries: Sequence[str], history: str | None = None) -> str:
    count = len(entries)
    numbered = "\n\n".join(f"[{position}]\n{entry}" for position, entry in enumerate(entries, 1))
    head = f"{PREAMBLE}\n\n"
    if history is not None:
        head += f"Conversation history with the user:\n\n{history}\n\n"

    return (
        f"{head}"
        f"There are {count} questions below, numbered 1 to {count}.\n\n"
        f"{numbered}\n\n"
        "Reply with one JSON object and nothing else: "
        '{"verdicts": [{"index": 1, "answer": "A"}, ...]}. '
        f"The answer is one letter from A to D. Give each index from 1 to {count} exactly once. "
        "The first character of your reply is `{`. No code fences."
    )


def batched[T](items: Sequence[T], size: int = BATCH_SIZE) -> list[list[T]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def group_hits(hits: Sequence[HitRow], arm: str) -> dict[str, list[HitRow]]:
    grouped: dict[str, list[HitRow]] = defaultdict(list)
    for hit in hits:
        if hit.arm == arm:
            grouped[hit.question_id].append(hit)

    return grouped


def group_chunks(chunks: Sequence[Chunk]) -> dict[int, list[Chunk]]:
    grouped: dict[int, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.persona_id].append(chunk)

    return grouped


async def ask(adapter: Caller, prompt: str, size: int) -> list[str | None] | None:
    for _attempt in range(ATTEMPTS):
        try:
            raw = await adapter.call(prompt)
            verdicts = parse_batch_verdicts(raw, item_model=ReaderVerdict, batch_size=size)
        except (JudgeUnavailable, JudgeInvalidOutput):
            continue

        return [parse_letter(verdict.answer) for verdict in verdicts]

    return None


def score(arm: str, question: PersonaQuestion, answer: str | None) -> AnswerRow:
    return AnswerRow.model_validate(
        {
            "arm": arm,
            "question_id": question.question_id,
            "answer": answer,
            "correct": answer == question.gold,
        }
    )


async def read_arm(
    arm: str,
    questions: Sequence[PersonaQuestion],
    hits: Sequence[HitRow],
    chunks: Sequence[Chunk],
    adapter: Caller,
    done: set[str],
) -> tuple[list[AnswerRow], int]:
    pending = [question for question in questions if question.question_id not in done]
    answers: list[AnswerRow] = []
    failed = 0

    for batch, history in _plan(arm, pending, chunks):
        entries = [
            render_question(question, notes) for question, notes in _with_notes(arm, batch, hits)
        ]
        letters = await ask(adapter, build_prompt(entries, history), len(batch))
        if letters is None:
            failed += len(batch)
            continue
        answers.extend(
            score(arm, question, letter) for question, letter in zip(batch, letters, strict=True)
        )

    return answers, failed


def _plan(
    arm: str, pending: Sequence[PersonaQuestion], chunks: Sequence[Chunk]
) -> list[tuple[list[PersonaQuestion], str | None]]:
    if arm != "full_history":
        return [(batch, None) for batch in batched(pending)]

    by_persona: dict[int, list[PersonaQuestion]] = defaultdict(list)
    for question in pending:
        by_persona[question.persona_id].append(question)
    histories = group_chunks(chunks)

    return [
        (batch, render_history(histories.get(persona, [])))
        for persona, persona_questions in by_persona.items()
        for batch in batched(persona_questions)
    ]


def _with_notes(
    arm: str, batch: Sequence[PersonaQuestion], hits: Sequence[HitRow]
) -> list[tuple[PersonaQuestion, str | None]]:
    if arm in ("no_memory", "full_history"):
        return [(question, None) for question in batch]

    grouped = group_hits(hits, arm)

    return [(question, render_notes(grouped.get(question.question_id, []))) for question in batch]


def load_all[RowT: BaseModel](path: Path, model: type[RowT]) -> list[RowT]:
    if not path.exists():
        return []

    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_hits(directory: Path) -> list[HitRow]:
    rows: list[HitRow] = []
    for path in sorted(directory.glob("hits-*.jsonl")):
        rows.extend(load_all(path, HitRow))

    return rows


def append_answers(path: Path, rows: Sequence[AnswerRow]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def build_adapter(model: str = READER_MODEL) -> SubprocessJudgeAdapter:
    binary = shutil.which("claude")
    if binary is None:
        raise JudgeUnavailable("claude binary not found on PATH")

    return SubprocessJudgeAdapter(cli="claude", binary=binary, model=model, timeout=CALL_TIMEOUT)


async def run(
    directory: Path,
    adapter: Caller,
    arms: Sequence[str],
    limit: int | None = None,
    persona: int | None = None,
) -> Mapping[str, tuple[int, int]]:
    questions = load_all(directory / "questions.jsonl", PersonaQuestion)
    if persona is not None:
        questions = [question for question in questions if question.persona_id == persona]
    if limit is not None:
        questions = questions[:limit]
    hits = load_hits(directory)
    chunks = load_all(directory / "chunks.jsonl", Chunk)
    answers_path = directory / ANSWERS_NAME
    summary: dict[str, tuple[int, int]] = {}

    for arm in arms:
        existing = load_all(answers_path, AnswerRow)
        done = {row.question_id for row in existing if row.arm == arm}
        rows, failed = await read_arm(arm, questions, hits, chunks, adapter, done)
        append_answers(answers_path, rows)
        summary[arm] = (len(rows), failed)

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_v1v2.pmv2.read")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--persona", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=READER_MODEL)
    args = parser.parse_args(argv)

    arms = [args.arm] if args.arm else list(ARMS)
    summary = asyncio.run(
        run(pmv2_dir(), build_adapter(args.model), arms, limit=args.limit, persona=args.persona)
    )
    for arm, (written, failed) in summary.items():
        print(f"{arm}: {written} answers written, {failed} unanswered", file=sys.stderr)

    return 1 if any(failed for _, failed in summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
