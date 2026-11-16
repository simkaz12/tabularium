from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from eval_v1v2.pmv2.extract import STATUS_NAME, ChunkStatus, build_prompt, parse_notes, run
from eval_v1v2.pmv2.rows import Chunk, Message, Note
from eval_v1v2.rows import read_jsonl
from tabularium.engine.judge import JudgeInvalidOutput

VALID = json.dumps(
    {"notes": [{"headline": "Prefers vegetarian food", "claim": "The user eats vegetarian."}]}
)


class FakeAdapter:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def chunk(index: int = 0, persona_id: int = 39) -> Chunk:
    return Chunk(
        chunk_id=f"{persona_id}-c{index}",
        persona_id=persona_id,
        index=index,
        messages=[
            Message(role="user", content="Please forget my old address."),
            Message(role="assistant", content="Done."),
        ],
    )


def test_valid_output_parses() -> None:
    notes = parse_notes(VALID)

    assert [note.headline for note in notes] == ["Prefers vegetarian food"]


def test_fenced_output_parses() -> None:
    assert len(parse_notes(f"```json\n{VALID}\n```")) == 1


def test_overlong_headline_is_invalid() -> None:
    raw = json.dumps({"notes": [{"headline": "x" * 100, "claim": "c"}]})

    with pytest.raises(JudgeInvalidOutput):
        parse_notes(raw)


def test_prompt_names_no_benchmark() -> None:
    prompt = build_prompt(chunk()).lower()

    for word in ("benchmark", "personamem", "question", "multiple"):
        assert word not in prompt


async def test_invalid_then_valid_retries_once(tmp_path: pathlib.Path) -> None:
    adapter = FakeAdapter(["not json", VALID])
    out = tmp_path / "notes.jsonl"

    tally = await run(adapter, [chunk()], out)

    assert len(adapter.prompts) == 2
    assert tally.done == 1
    assert tally.failed == 0
    assert [note.note_id for note in read_jsonl(out, Note)] == ["39-c0-n0"]


async def test_invalid_twice_is_skipped_and_counted(tmp_path: pathlib.Path) -> None:
    adapter = FakeAdapter(["nope", "still nope", VALID])
    out = tmp_path / "notes.jsonl"

    tally = await run(adapter, [chunk(0), chunk(1)], out)

    assert tally.failed == 1
    assert tally.done == 1
    assert [note.chunk_id for note in read_jsonl(out, Note)] == ["39-c1"]


async def test_rerun_skips_done_chunks(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "notes.jsonl"
    await run(FakeAdapter([VALID, VALID]), [chunk(0), chunk(1)], out)
    adapter = FakeAdapter([])

    tally = await run(adapter, [chunk(0), chunk(1)], out)

    assert adapter.prompts == []
    assert tally.skipped_done == 2
    assert len(read_jsonl(out, Note)) == 2


async def test_persona_and_limit_filter(tmp_path: pathlib.Path) -> None:
    adapter = FakeAdapter([VALID])
    chunks = [chunk(0, 39), chunk(1, 39), chunk(0, 55)]

    tally = await run(adapter, chunks, tmp_path / "notes.jsonl", persona=39, limit=1)

    assert tally.done == 1
    assert len(adapter.prompts) == 1


EMPTY = json.dumps({"notes": []})


async def test_out_of_order_resume_yields_chunk_ordered_notes(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "notes.jsonl"
    chunks = [chunk(0), chunk(1), chunk(2)]
    await run(FakeAdapter([VALID]), [chunk(2)], out)
    await run(FakeAdapter([VALID, VALID]), chunks, out)

    assert [note.chunk_id for note in read_jsonl(out, Note)] == ["39-c0", "39-c1", "39-c2"]


async def test_empty_chunk_is_not_reextracted(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "notes.jsonl"
    first = await run(FakeAdapter([EMPTY]), [chunk(0)], out)
    adapter = FakeAdapter([])

    second = await run(adapter, [chunk(0)], out)

    assert first.empty == 1
    assert adapter.prompts == []
    assert second.skipped_done == 1
    rows = read_jsonl(tmp_path / STATUS_NAME, ChunkStatus)
    assert [(row.chunk_id, row.status) for row in rows] == [("39-c0", "empty")]


async def test_failed_chunk_is_retried_and_counted(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "notes.jsonl"
    first = await run(FakeAdapter(["nope", "nope"]), [chunk(0)], out)
    second = await run(FakeAdapter([VALID]), [chunk(0)], out)

    assert first.failed == 1
    assert first.failed_rate == 1.0
    assert second.failed == 0
    assert second.done == 1
    assert [note.chunk_id for note in read_jsonl(out, Note)] == ["39-c0"]
