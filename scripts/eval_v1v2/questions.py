from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_v1v2.paths import out_path
from eval_v1v2.rows import CorpusRow, Origin, QuestionRow, read_jsonl, write_jsonl

RECALL_TOOLS = frozenset({"mcp__tabularium-mcp__recall", "mcp__t3-code__memory_recall"})
TRANSCRIPT_ROOT = Path(os.environ.get("TABULARIUM_EVAL_TRANSCRIPTS", "transcripts"))
PROJECTS_ROOT = Path(os.environ.get("TABULARIUM_EVAL_PROJECTS", "projects"))
SEED = 20261112
SAMPLE_SIZE = 60
MIN_QUESTIONS = 40
NEAR_DUPLICATE_COSINE = 0.90
MIN_WORDS = 3
QUESTIONS_NAME = "questions.jsonl"
CORPUS_NAME = "corpus.jsonl"

PATH_SCOPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("project-a-api", "project-a-web"), "project-a"),
    (("project-i-api", "project-i-app"), "project-i"),
    (("project-b",), "project-b"),
    (("project-c",), "project-c"),
    (("project-d",), "project-d"),
    (("project-e",), "project-e"),
    (("project-f-api", "project-f-web"), "project-f"),
    (("Tabularium", "tabularium"), "tabularium"),
    (("project-h",), "project-h"),
    (("project-j",), "project-j"),
    (("project-g",), "project-g"),
)

UUID_PATTERN = re.compile(r"^[0-9a-f]{8}(-?[0-9a-f]{4}){3}-?[0-9a-f]{12}$", re.IGNORECASE)
HEX_ID_PATTERN = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)

Embed = Callable[[str], Awaitable[list[float]]]


@dataclass(frozen=True)
class Candidate:
    query: str
    scope: str


def scope_for_cwd(cwd: str | None, root: Path = PROJECTS_ROOT) -> str | None:
    if not cwd:
        return None
    prefix = str(root) + "/"
    if not cwd.startswith(prefix):
        return None
    top = cwd[len(prefix) :].split("/", 1)[0]
    for names, scope in PATH_SCOPES:
        if any(top.startswith(name) for name in names):
            return scope
    return None


def is_probe(query: str) -> bool:
    text = query.strip()
    if len(text.split()) < MIN_WORDS:
        return True
    return bool(UUID_PATTERN.match(text) or HEX_ID_PATTERN.match(text))


def recall_calls(path: Path) -> Iterator[tuple[dict[str, Any], str | None]]:
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "recall" not in line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = entry.get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use" or block.get("name") not in RECALL_TOOLS:
                    continue
                arguments = block.get("input")
                if isinstance(arguments, dict):
                    yield arguments, entry.get("cwd")


def mine_transcript(path: Path, root: Path = PROJECTS_ROOT) -> Iterator[Candidate]:
    last_cwd: str | None = None
    for arguments, cwd in recall_calls(path):
        last_cwd = cwd or last_cwd
        query = arguments.get("query")
        if not isinstance(query, str) or is_probe(query):
            continue
        scope = arguments.get("scope")
        if not isinstance(scope, str) or not scope:
            scope = scope_for_cwd(cwd or last_cwd, root)
        if scope:
            yield Candidate(query=query.strip(), scope=scope)


def mine_all(
    transcript_root: Path = TRANSCRIPT_ROOT, root: Path = PROJECTS_ROOT
) -> list[Candidate]:
    seen: set[tuple[str, str]] = set()
    mined: list[Candidate] = []
    for path in sorted(transcript_root.glob("*/*.jsonl")):
        for candidate in mine_transcript(path, root):
            key = (candidate.query.casefold(), candidate.scope)
            if key not in seen:
                seen.add(key)
                mined.append(candidate)
    return mined


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


def collapse_near_duplicates(
    candidates: list[Candidate],
    vectors: list[list[float]],
    threshold: float = NEAR_DUPLICATE_COSINE,
) -> list[Candidate]:
    kept: list[Candidate] = []
    kept_vectors: list[list[float]] = []
    for candidate, vector in zip(candidates, vectors, strict=True):
        if any(cosine(vector, other) >= threshold for other in kept_vectors):
            continue
        kept.append(candidate)
        kept_vectors.append(vector)
    return kept


def spread_sample(candidates: list[Candidate], size: int, seed: int = SEED) -> list[Candidate]:
    rng = random.Random(seed)
    by_scope: dict[str, list[Candidate]] = {}
    for candidate in sorted(candidates, key=lambda c: (c.scope, c.query)):
        by_scope.setdefault(candidate.scope, []).append(candidate)
    for group in by_scope.values():
        rng.shuffle(group)
    scopes = sorted(by_scope)
    picked: list[Candidate] = []
    while len(picked) < size and any(by_scope.values()):
        for scope in scopes:
            if by_scope[scope] and len(picked) < size:
                picked.append(by_scope[scope].pop())
    return picked


def question_id(query: str, scope: str, origin: Origin) -> str:
    digest = hashlib.sha1(f"{origin}\0{scope}\0{query}".encode()).hexdigest()
    return f"q-{digest[:12]}"


def to_rows(candidates: list[Candidate], origin: Origin) -> list[QuestionRow]:
    return [
        QuestionRow(
            question_id=question_id(c.query, c.scope, origin),
            query=c.query,
            scope=c.scope,
            origin=origin,
        )
        for c in candidates
    ]


async def embed_all(texts: list[str], embed: Embed) -> list[list[float]]:
    return [await embed(text) for text in texts]


def mine_questions(
    corpus_scopes: set[str],
    embed: Embed,
    transcript_root: Path = TRANSCRIPT_ROOT,
    root: Path = PROJECTS_ROOT,
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> list[QuestionRow]:
    mined = [c for c in mine_all(transcript_root, root) if c.scope in corpus_scopes]
    vectors = asyncio.run(embed_all([c.query for c in mined], embed))
    distinct = collapse_near_duplicates(mined, vectors)
    return to_rows(spread_sample(distinct, size, seed), "mined")


TOP_UP_PROMPT = """\
Below are numbered memory records from a developer's long-term memory store. For each record,
write one short search query (4 to 12 words) that the developer might type when they need that
record's knowledge, without copying its wording. Reply with one JSON object and nothing else:
{{"queries": [{{"index": 1, "query": "..."}}]}}. The first character of your reply is `{{`.

{records}
"""


def build_top_up_prompt(records: list[CorpusRow]) -> str:
    rendered = "\n\n".join(
        f"[{index}] scope={row.scope}\n{row.content[:600]}"
        for index, row in enumerate(records, start=1)
    )
    return TOP_UP_PROMPT.format(records=rendered)


def parse_top_up(raw: str, records: list[CorpusRow]) -> list[Candidate]:
    payload = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    found: list[Candidate] = []
    for entry in payload["queries"]:
        index = int(entry["index"])
        query = str(entry["query"]).strip()
        if 1 <= index <= len(records) and not is_probe(query):
            found.append(Candidate(query=query, scope=records[index - 1].scope))
    return found


def sample_records(corpus: list[CorpusRow], size: int, seed: int = SEED) -> list[CorpusRow]:
    rng = random.Random(f"{seed}:top-up")
    pool = sorted(corpus, key=lambda row: row.id)
    rng.shuffle(pool)
    return pool[:size]


async def generate_top_up(
    corpus: list[CorpusRow],
    existing: list[QuestionRow],
    embed: Embed,
    call: Callable[[str], Awaitable[str]],
    target: int = MIN_QUESTIONS,
    seed: int = SEED,
) -> list[QuestionRow]:
    need = target - len(existing)
    if need <= 0:
        return []
    records = sample_records(corpus, need * 3, seed)
    generated = parse_top_up(await call(build_top_up_prompt(records)), records)
    pool = [Candidate(row.query, row.scope) for row in existing] + generated
    vectors = await embed_all([c.query for c in pool], embed)
    distinct = collapse_near_duplicates(pool, vectors)
    fresh = distinct[len(existing) :][:need]
    return to_rows(fresh, "generated")


def default_embed() -> Embed:
    from tabularium.embedding import LitellmEmbedder

    return LitellmEmbedder().embed


def top_up_from_judge(corpus: list[CorpusRow], existing: list[QuestionRow]) -> list[QuestionRow]:
    from eval_v1v2.judges import build_seats

    seat = build_seats()[0]
    return asyncio.run(generate_top_up(corpus, existing, default_embed(), seat.adapter.call))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine search questions from recall transcripts.")
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--top-up", action="store_true")
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPT_ROOT)
    parser.add_argument("--projects", type=Path, default=PROJECTS_ROOT)
    args = parser.parse_args()
    corpus = read_jsonl(out_path(CORPUS_NAME), CorpusRow)
    scopes = {row.scope for row in corpus}
    rows = mine_questions(
        scopes,
        default_embed(),
        args.transcripts,
        args.projects,
        size=args.size,
    )
    if len(rows) < MIN_QUESTIONS and args.top_up:
        rows += top_up_from_judge(corpus, rows)
    write_jsonl(out_path(QUESTIONS_NAME), rows)
    print(f"questions: {len(rows)}")
    for origin, count in sorted(Counter(r.origin for r in rows).items()):
        print(f"  origin {origin}: {count}")
    for scope, count in sorted(Counter(r.scope for r in rows).items()):
        print(f"  scope {scope}: {count}")
    if len(rows) < MIN_QUESTIONS:
        print(f"SHORTFALL: {MIN_QUESTIONS - len(rows)} below {MIN_QUESTIONS}")


if __name__ == "__main__":
    main()
