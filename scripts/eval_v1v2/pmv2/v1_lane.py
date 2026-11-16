from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

from eval_v1v2.pmv2.prep import pmv2_dir
from eval_v1v2.pmv2.rows import HitRow, Note, PersonaQuestion, WriteRow
from eval_v1v2.rows import iter_jsonl, read_jsonl
from eval_v1v2.v1_engine import V1Harness, open_v1, register_scopes, require_assessor_key
from eval_v1v2.v1_search import RECALL_LIMIT, require_both_legs
from tabularium.domain import Kind
from tabularium.results import RecallResult, SaveResult

WRITES_NAME = "writes-v1.jsonl"
HITS_NAME = "hits-v1.jsonl"


class LaneEngine(Protocol):
    async def save(self, content: str, *, scope: str, kind: Kind = "fact") -> SaveResult: ...

    async def recall(self, query: str, *, scope: str, limit: int) -> RecallResult: ...


def persona_scope(persona_id: int) -> str:
    return f"pmv2-{persona_id}"


def note_content(note: Note) -> str:
    return f"{note.headline}\n{note.claim}"


def record_id_of(result: SaveResult) -> str | None:
    if result.hypomnema is not None:
        return str(result.hypomnema.id)

    if result.reinforced_id is not None:
        return str(result.reinforced_id)

    return None


async def write_notes(engine: LaneEngine, persona_id: int, notes: list[Note]) -> list[WriteRow]:
    scope = persona_scope(persona_id)
    rows: list[WriteRow] = []

    for note in notes:
        started = time.perf_counter()
        result = await engine.save(note_content(note), scope=scope, kind="fact")
        elapsed = round((time.perf_counter() - started) * 1000)

        rows.append(
            WriteRow(
                system="v1",
                note_id=note.note_id,
                persona_id=persona_id,
                status=result.outcome,
                record_id=record_id_of(result),
                elapsed_ms=elapsed,
            )
        )

    return rows


async def recall_questions(
    engine: LaneEngine, persona_id: int, questions: list[PersonaQuestion]
) -> list[HitRow]:
    scope = persona_scope(persona_id)
    rows: list[HitRow] = []

    for question in questions:
        text_query = question.query.replace('"', " ")
        result = await engine.recall(text_query, scope=scope, limit=RECALL_LIMIT)

        require_both_legs(result, question.query)

        for rank, hit in enumerate(result.hits, start=1):
            rows.append(
                HitRow(
                    arm="v1",
                    question_id=question.question_id,
                    rank=rank,
                    record_id=str(hit.id),
                    text=hit.content,
                )
            )

    return rows


async def run_persona(
    harness: V1Harness,
    persona_id: int,
    notes: list[Note],
    questions: list[PersonaQuestion],
) -> tuple[list[WriteRow], list[HitRow]]:
    await register_scopes(harness, [persona_scope(persona_id)])

    writes = await write_notes(harness.engine, persona_id, notes)
    hits = await recall_questions(harness.engine, persona_id, questions)

    return writes, hits


def append_rows(path: Path, rows: list[WriteRow] | list[HitRow]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")


def group_by_persona[RowT: (Note, PersonaQuestion)](rows: list[RowT]) -> dict[int, list[RowT]]:
    grouped: dict[int, list[RowT]] = defaultdict(list)

    for row in rows:
        grouped[row.persona_id].append(row)

    return grouped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PersonaMem-v2 notes through the v1 lane")
    parser.add_argument("--persona", type=int, default=None)
    parser.add_argument("--notes", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)

    return parser


async def _main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)

    require_assessor_key()

    base = pmv2_dir()
    out = args.out or base
    notes = group_by_persona(list(iter_jsonl(args.notes or base / "notes.jsonl", Note)))
    questions = group_by_persona(
        read_jsonl(args.questions or base / "questions.jsonl", PersonaQuestion)
    )
    personas = [args.persona] if args.persona is not None else sorted(questions)

    out.mkdir(parents=True, exist_ok=True)
    (out / WRITES_NAME).write_text("", encoding="utf-8")
    (out / HITS_NAME).write_text("", encoding="utf-8")

    for persona_id in personas:
        harness = await open_v1(use_assessor=True)
        writes, hits = await run_persona(
            harness, persona_id, notes.get(persona_id, []), questions.get(persona_id, [])
        )

        append_rows(out / WRITES_NAME, writes)
        append_rows(out / HITS_NAME, hits)

        statuses: dict[str, int] = defaultdict(int)
        for row in writes:
            statuses[row.status] += 1

        print(f"persona={persona_id} notes={len(writes)} hits={len(hits)} {dict(statuses)}")

    return 0


def main() -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
