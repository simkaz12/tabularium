from __future__ import annotations

import ast
import csv
import json
import random
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

from eval_v1v2 import paths
from eval_v1v2.pmv2.rows import Chunk, Letter, Message, PersonaQuestion
from eval_v1v2.rows import write_jsonl

PERSONAS = (39, 327, 55, 321, 309)
CHUNK_PAIRS = 4
BASE_URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main/"
BENCHMARK_PATH = "benchmark/text/benchmark.csv"
LETTERS: tuple[str, ...] = get_args(Letter)


def pmv2_dir() -> Path:
    directory = paths.OUT_DIR.parent / "pmv2"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def raw_dir() -> Path:
    directory = pmv2_dir() / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def fetch(relative: str, raw: Path) -> Path:
    target = raw / relative
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BASE_URL + relative, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)

    return target


def parse_query(raw_query: str) -> str:
    parsed = ast.literal_eval(raw_query)
    if isinstance(parsed, dict):
        return str(parsed["content"])

    return str(parsed)


def build_choices(question_id: str, gold: str, distractors: list[str]) -> tuple[list[str], Letter]:
    options = [gold, *distractors]
    random.Random(question_id).shuffle(options)
    letter: Letter = LETTERS[options.index(gold)]  # type: ignore[assignment]

    return options, letter


def build_question(row: dict[str, str], row_index: int) -> PersonaQuestion:
    persona_id = int(row["persona_id"])
    question_id = f"{persona_id}-{row_index}"
    distractors = [str(item) for item in json.loads(row["incorrect_answers"])]
    choices, gold = build_choices(question_id, row["correct_answer"], distractors)

    return PersonaQuestion(
        question_id=question_id,
        persona_id=persona_id,
        query=parse_query(row["user_query"]),
        choices=choices,
        gold=gold,
        forget=row["updated"].strip().lower() == "true",
        pref_type=row["pref_type"],
    )


def build_questions(
    rows: Iterable[dict[str, str]], personas: Iterable[int]
) -> list[PersonaQuestion]:
    wanted = set(personas)
    questions: list[PersonaQuestion] = []
    for row_index, row in enumerate(rows):
        if int(row["persona_id"]) not in wanted:
            continue
        questions.append(build_question(row, row_index))

    return questions


def drop_system(messages: list[dict[str, Any]]) -> list[Message]:
    kept = messages[1:] if messages and messages[0].get("role") == "system" else messages

    return [Message(role=str(item["role"]), content=str(item["content"])) for item in kept]


def build_chunks(persona_id: int, history: dict[str, Any]) -> list[Chunk]:
    messages = drop_system(history["chat_history"])
    size = CHUNK_PAIRS * 2
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(messages), size)):
        chunks.append(
            Chunk(
                chunk_id=f"{persona_id}-c{index}",
                persona_id=persona_id,
                index=index,
                messages=messages[start : start + size],
            )
        )

    return chunks


def read_benchmark(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def history_links(rows: Iterable[dict[str, str]], personas: Iterable[int]) -> dict[int, str]:
    wanted = set(personas)
    links: dict[int, str] = {}
    for row in rows:
        persona_id = int(row["persona_id"])
        if persona_id in wanted:
            links.setdefault(persona_id, row["chat_history_32k_link"])

    return links


def main() -> None:
    raw = raw_dir()
    rows = read_benchmark(fetch(BENCHMARK_PATH, raw))
    questions = build_questions(rows, PERSONAS)

    chunks: list[Chunk] = []
    for persona_id, link in history_links(rows, PERSONAS).items():
        history = json.loads(fetch(link, raw).read_text(encoding="utf-8"))
        chunks.extend(build_chunks(persona_id, history))

    out = pmv2_dir()
    write_jsonl(out / "questions.jsonl", questions)
    write_jsonl(out / "chunks.jsonl", chunks)
    forget = sum(1 for question in questions if question.forget)
    print(f"questions={len(questions)} forget={forget} chunks={len(chunks)}")


if __name__ == "__main__":
    main()
