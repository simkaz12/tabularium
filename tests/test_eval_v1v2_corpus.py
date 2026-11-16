from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from typing import Any

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import corpus, questions  # noqa: E402
from eval_v1v2.rows import CorpusRow, QuestionRow, read_jsonl  # noqa: E402

sys.path.pop(0)

VECTOR = [0.5] * 768


def pool_record(record_id: str, scope: str, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": record_id,
        "type": "fact",
        "content": f"content of {record_id}",
        "scope": scope,
        "purpose": "reference",
        "validity": "valid",
        "recall_count": 3,
        "created_at": "2026-07-31T13:54:10+00:00",
        "updated_at": "2026-08-01T13:54:10+00:00",
        "lifecycle": "active",
        "feedback_score": 0.5,
        "vector": VECTOR,
    }
    record.update(overrides)
    return record


def test_corpus_excludes_client_scopes_and_dead_records(tmp_path: pathlib.Path) -> None:
    pool = [
        pool_record("keep-project-b", "project-b"),
        pool_record("keep-global", "global", type="decision", lifecycle="in_force"),
        pool_record("drop-project-a", "project-a"),
        pool_record("drop-project-i", "project-i"),
        pool_record("drop-invalid", "project-e", validity="invalid"),
        pool_record("drop-superseded", "project-e", lifecycle="superseded"),
    ]
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool))
    out = tmp_path / "corpus.jsonl"

    corpus.build(pool_path, out)
    rows = read_jsonl(out, CorpusRow)

    assert [row.id for row in rows] == ["keep-project-b", "keep-global"]
    assert {row.scope for row in rows}.isdisjoint({"project-a", "project-i"})
    assert rows[1].kind == "decision"


def test_corpus_carries_pool_fields_as_extras(tmp_path: pathlib.Path) -> None:
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps([pool_record("a", "project-e")]))
    out = tmp_path / "corpus.jsonl"

    corpus.build(pool_path, out)
    row = read_jsonl(out, CorpusRow)[0]

    assert row.model_extra is not None
    assert row.model_extra["purpose"] == "reference"
    assert row.model_extra["feedback_score"] == 0.5
    assert row.model_extra["updated_at"] == "2026-08-01T13:54:10+00:00"
    assert row.recall_count == 3


def write_transcript(path: pathlib.Path, calls: list[tuple[str, dict[str, Any], str]]) -> None:
    lines = []
    for name, arguments, cwd in calls:
        block = {"type": "tool_use", "name": name, "input": arguments}
        lines.append(json.dumps({"cwd": cwd, "message": {"content": [block]}}))
    lines.append(json.dumps({"cwd": "/x", "message": {"content": "recall plain text"}}))
    lines.append("not json recall")
    path.write_text("\n".join(lines))


def test_transcript_yields_queries_and_scopes(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "Projects"
    project = tmp_path / "projects" / "-p"
    project.mkdir(parents=True)
    write_transcript(
        project / "s.jsonl",
        [
            (
                "mcp__tabularium-mcp__recall",
                {"query": "how does the dedup gate work", "scope": "tabularium"},
                f"{root}/Tabularium",
            ),
            (
                "mcp__t3-code__memory_recall",
                {"query": "project-b deploy ssh config notes"},
                f"{root}/project-b/app",
            ),
            (
                "mcp__t3-code__memory_recall",
                {"query": "project-a deploy pipeline steps"},
                f"{root}/project-a-web",
            ),
            ("mcp__t3-code__memory_recall", {"query": "short one"}, f"{root}/project-c"),
            ("mcp__other__tool", {"query": "not a recall call at all"}, f"{root}/project-c"),
        ],
    )

    found = questions.mine_all(tmp_path / "projects", root)

    assert [(c.query, c.scope) for c in found] == [
        ("how does the dedup gate work", "tabularium"),
        ("project-b deploy ssh config notes", "project-b"),
        ("project-a deploy pipeline steps", "project-a"),
    ]


def test_exact_duplicates_collapse(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "Projects"
    project = tmp_path / "projects" / "-p"
    project.mkdir(parents=True)
    cwd = f"{root}/project-e"
    call = ("mcp__t3-code__memory_recall", {"query": "Same Query About Deploys"}, cwd)
    again = ("mcp__t3-code__memory_recall", {"query": "same query about deploys"}, cwd)
    write_transcript(project / "a.jsonl", [call, again])
    write_transcript(project / "b.jsonl", [call])

    assert len(questions.mine_all(tmp_path / "projects", root)) == 1


def test_near_duplicates_collapse_by_cosine() -> None:
    candidates = [questions.Candidate(f"query number {i}", "project-e") for i in range(3)]
    vectors = [[1.0, 0.0], [0.99, 0.05], [0.0, 1.0]]

    kept = questions.collapse_near_duplicates(candidates, vectors)

    assert [c.query for c in kept] == ["query number 0", "query number 2"]


def test_probes_are_dropped() -> None:
    assert questions.is_probe("")
    assert questions.is_probe("two words")
    assert questions.is_probe("9864911c-5e7c-40f8-b14f-22d46ee35c99")
    assert questions.is_probe("9864911c5e7c40f8")
    assert not questions.is_probe("how does recall rank results")


def test_sample_spreads_across_scopes_and_is_seeded() -> None:
    candidates = [questions.Candidate(f"big scope query {i}", "big") for i in range(30)]
    candidates += [questions.Candidate(f"small scope query {i}", "small") for i in range(3)]

    first = questions.spread_sample(candidates, 10, seed=1)
    second = questions.spread_sample(candidates, 10, seed=1)

    assert first == second
    assert sum(c.scope == "small" for c in first) == 3
    assert len(first) == 10


def test_mine_questions_end_to_end(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "Projects"
    project = tmp_path / "projects" / "-p"
    project.mkdir(parents=True)
    write_transcript(
        project / "s.jsonl",
        [
            (
                "mcp__t3-code__memory_recall",
                {"query": "alpha topic about deploys"},
                f"{root}/project-e",
            ),
            (
                "mcp__t3-code__memory_recall",
                {"query": "beta topic about deploys"},
                f"{root}/project-e",
            ),
            (
                "mcp__t3-code__memory_recall",
                {"query": "client work notes project-a"},
                f"{root}/project-a-api",
            ),
        ],
    )

    async def embed(text: str) -> list[float]:
        return [1.0, 0.0] if text.startswith(("alpha", "beta")) else [0.0, 1.0]

    rows = questions.mine_questions({"project-e"}, embed, tmp_path / "projects", root)

    assert [(r.query, r.scope, r.origin) for r in rows] == [
        ("alpha topic about deploys", "project-e", "mined")
    ]


def corpus_row(record_id: str, scope: str) -> CorpusRow:
    return corpus.to_row(pool_record(record_id, scope))


def test_top_up_generates_only_the_shortfall_and_marks_origin() -> None:
    records = [corpus_row(f"r{i}", "project-e") for i in range(10)]
    existing = [
        QuestionRow(
            question_id="q-1",
            query="existing mined question here",
            scope="project-e",
            origin="mined",
        )
    ]
    prompts: list[str] = []

    async def call(prompt: str) -> str:
        prompts.append(prompt)
        queries = [{"index": i, "query": f"generated question number {i}"} for i in range(1, 6)]
        return "noise " + json.dumps({"queries": queries})

    async def embed(text: str) -> list[float]:
        digits = [int(ch) for ch in text if ch.isdigit()]
        vector = [0.0] * 8
        vector[digits[0] if digits else 7] = 1.0
        return vector

    rows = asyncio.run(questions.generate_top_up(records, existing, embed, call, target=4))

    assert len(rows) == 3
    assert {row.origin for row in rows} == {"generated"}
    assert {row.scope for row in rows} == {"project-e"}
    assert len(prompts) == 1


def test_top_up_does_nothing_when_target_met() -> None:
    existing = [
        QuestionRow(
            question_id=f"q-{i}",
            query=f"mined question {i} text",
            scope="project-e",
            origin="mined",
        )
        for i in range(4)
    ]

    async def call(prompt: str) -> str:
        raise AssertionError("judge must not be called")

    async def embed(text: str) -> list[float]:
        return [1.0]

    assert asyncio.run(questions.generate_top_up([], existing, embed, call, target=4)) == []
