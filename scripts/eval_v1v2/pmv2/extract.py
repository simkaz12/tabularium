from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eval_v1v2.pmv2.prep import pmv2_dir
from eval_v1v2.pmv2.rows import Chunk, Note
from eval_v1v2.rows import Row, read_jsonl
from tabularium.engine.judge import JudgeInvalidOutput, JudgeUnavailable, SubprocessJudgeAdapter

MODEL = "opus"
HEADLINE_LIMIT = 100
ATTEMPTS = 2
STATUS_NAME = "extract-status.jsonl"
SETTLED = ("ok", "empty")

PROMPT = """\
You are the memory of an assistant that works with one user over many sessions. Below is one \
stretch of the conversation between the user and the assistant. Write down the notes the \
assistant should save about the user so that it can help them better later.

Rules:
- One fact per note.
- Each note has a "headline" (a single sentence of fewer than 100 characters) and a "claim" \
(the fact in full, with the detail needed to act on it later).
- Save what the user said or showed about themselves: preferences, habits, circumstances, \
plans, constraints, relationships, work, opinions. Skip pleasantries and one-off task content \
with no bearing on the user.
- If the user asks the assistant to forget or stop using something, write that request down as \
its own note, worded as the user said it.
- Write nothing the user did not say or clearly show.
- If nothing in this stretch is worth saving, return an empty list.

Reply with JSON only, no other text, in exactly this shape:
{{"notes": [{{"headline": "...", "claim": "..."}}]}}

Conversation:
{conversation}
"""


class Extracted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=HEADLINE_LIMIT - 1)
    claim: str = Field(min_length=1)


class ExtractedNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[Extracted]


class Caller(Protocol):
    async def call(self, prompt: str) -> str: ...


class ChunkStatus(Row):
    chunk_id: str
    status: str


@dataclass
class Tally:
    done: int = 0
    empty: int = 0
    skipped_done: int = 0
    failed: int = 0
    notes: int = 0

    @property
    def failed_rate(self) -> float:
        attempted = self.done + self.failed

        return self.failed / attempted if attempted else 0.0


def render_conversation(chunk: Chunk) -> str:
    return "\n\n".join(f"{message.role.upper()}: {message.content}" for message in chunk.messages)


def build_prompt(chunk: Chunk) -> str:
    return PROMPT.format(conversation=render_conversation(chunk))


def parse_notes(raw: str) -> list[Extracted]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return ExtractedNotes.model_validate_json(text).notes
    except ValidationError as exc:
        raise JudgeInvalidOutput(f"extractor output failed validation: {exc}") from exc


async def extract_chunk(adapter: Caller, chunk: Chunk) -> list[Extracted] | None:
    prompt = build_prompt(chunk)
    for _attempt in range(ATTEMPTS):
        try:
            raw = await adapter.call(prompt)
            return parse_notes(raw)
        except (JudgeUnavailable, JudgeInvalidOutput):
            continue

    return None


def to_notes(chunk: Chunk, extracted: Sequence[Extracted]) -> list[Note]:
    return [
        Note(
            note_id=f"{chunk.chunk_id}-n{position}",
            persona_id=chunk.persona_id,
            chunk_id=chunk.chunk_id,
            headline=item.headline,
            claim=item.claim,
        )
        for position, item in enumerate(extracted)
    ]


def status_path(out_path: Path) -> Path:
    return out_path.with_name(STATUS_NAME)


def settled_chunk_ids(out_path: Path) -> set[str]:
    settled: set[str] = set()
    if out_path.exists():
        settled.update(note.chunk_id for note in read_jsonl(out_path, Note))

    path = status_path(out_path)
    if path.exists():
        latest = {row.chunk_id: row.status for row in read_jsonl(path, ChunkStatus)}
        settled.update(chunk_id for chunk_id, status in latest.items() if status in SETTLED)
        settled.difference_update(
            chunk_id for chunk_id, status in latest.items() if status not in SETTLED
        )

    return settled


def append_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")


def sort_notes_in_chunk_order(out_path: Path, chunks: Sequence[Chunk]) -> None:
    if not out_path.exists():
        return

    index = {chunk.chunk_id: chunk.index for chunk in chunks}
    notes = read_jsonl(out_path, Note)
    ordered = sorted(
        notes,
        key=lambda note: (note.persona_id, index.get(note.chunk_id, sys.maxsize)),
    )
    staging = out_path.with_name(out_path.name + ".tmp")
    staging.write_text("", encoding="utf-8")
    append_jsonl(staging, ordered)
    staging.replace(out_path)


def select_chunks(chunks: Sequence[Chunk], persona: int | None, limit: int | None) -> list[Chunk]:
    chosen = [chunk for chunk in chunks if persona is None or chunk.persona_id == persona]
    if limit is not None:
        chosen = chosen[:limit]

    return chosen


async def run(
    adapter: Caller,
    chunks: Sequence[Chunk],
    out_path: Path,
    *,
    persona: int | None = None,
    limit: int | None = None,
) -> Tally:
    tally = Tally()
    finished = settled_chunk_ids(out_path)
    statuses = status_path(out_path)

    for chunk in select_chunks(chunks, persona, limit):
        if chunk.chunk_id in finished:
            tally.skipped_done += 1
            continue

        extracted = await extract_chunk(adapter, chunk)
        if extracted is None:
            tally.failed += 1
            append_jsonl(statuses, [ChunkStatus(chunk_id=chunk.chunk_id, status="failed")])
            print(f"failed chunk {chunk.chunk_id}", file=sys.stderr)
            continue

        notes = to_notes(chunk, extracted)
        append_jsonl(out_path, notes)
        outcome = "ok" if notes else "empty"
        append_jsonl(statuses, [ChunkStatus(chunk_id=chunk.chunk_id, status=outcome)])
        tally.done += 1
        tally.empty += 0 if notes else 1
        tally.notes += len(notes)

    sort_notes_in_chunk_order(out_path, chunks)

    return tally


def build_adapter() -> SubprocessJudgeAdapter:
    binary = shutil.which("claude")
    if binary is None:
        raise JudgeUnavailable("claude binary not found on PATH")

    return SubprocessJudgeAdapter(cli="claude", binary=binary, model=MODEL)


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_v1v2.pmv2.extract")
    parser.add_argument("--persona", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    directory = pmv2_dir()
    chunks = read_jsonl(directory / "chunks.jsonl", Chunk)
    tally = asyncio.run(
        run(
            build_adapter(),
            chunks,
            directory / "notes.jsonl",
            persona=args.persona,
            limit=args.limit,
        )
    )
    print(
        f"chunks_done={tally.done} skipped_done={tally.skipped_done} "
        f"empty={tally.empty} failed={tally.failed} failed_rate={tally.failed_rate:.1%} "
        f"notes={tally.notes}"
    )


if __name__ == "__main__":
    main()
