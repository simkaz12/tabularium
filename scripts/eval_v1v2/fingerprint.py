from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

INPUTS_NAME = "inputs.json"
PAIRS = "pairs.jsonl"
NOISE = "noise.jsonl"
CORPUS = "corpus.jsonl"
QUESTIONS = "questions.jsonl"
SEARCH_OUTPUTS = (
    "search-v1.jsonl",
    "search-v2_rerank.jsonl",
    "search-v2_plain.jsonl",
    "search-v2_rerank_facts.jsonl",
    "search-v2_plain_facts.jsonl",
)

DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "labels-relation.jsonl": (PAIRS,),
    "save-v1.jsonl": (PAIRS, NOISE),
    "save-v2.jsonl": (PAIRS, NOISE),
    "targets/labels-relation.jsonl": (PAIRS, NOISE),
    "labels-relevance.jsonl": (CORPUS, QUESTIONS),
    **dict.fromkeys(SEARCH_OUTPUTS, (CORPUS, QUESTIONS)),
}

STAGE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "judge-relation": ("labels-relation.jsonl",),
    "save-v1": ("save-v1.jsonl",),
    "save-v2": ("save-v2.jsonl",),
    "judge-targets": ("targets/labels-relation.jsonl",),
    "search-v1": ("search-v1.jsonl",),
    "search-v2": SEARCH_OUTPUTS[1:],
    "judge-relevance": ("labels-relevance.jsonl",),
}

STAGE_READS: dict[str, tuple[str, ...]] = {
    "disputes": ("labels-relation.jsonl",),
    "score-save": (
        "labels-relation.jsonl",
        "save-v1.jsonl",
        "save-v2.jsonl",
        "targets/labels-relation.jsonl",
    ),
    "score-search": (*SEARCH_OUTPUTS, "labels-relevance.jsonl"),
    "report": tuple(DEPENDS_ON),
}


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def current(base: Path, output: str) -> dict[str, str]:
    return {name: hash_file(base / name) for name in DEPENDS_ON[output]}


def _load(base: Path) -> dict[str, dict[str, str]]:
    path = base / INPUTS_NAME

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(base: Path, recorded: dict[str, dict[str, str]]) -> None:
    (base / INPUTS_NAME).write_text(
        json.dumps(recorded, indent=2, sort_keys=True), encoding="utf-8"
    )


def adopt(base: Path) -> None:
    recorded = _load(base)
    added = {
        output: current(base, output)
        for output in DEPENDS_ON
        if (base / output).exists() and output not in recorded
    }

    if added:
        _save(base, {**recorded, **added})


def record(base: Path, outputs: Iterable[str]) -> None:
    recorded = _load(base)

    for output in outputs:
        if (base / output).exists():
            recorded[output] = current(base, output)

    _save(base, recorded)


def stale(base: Path, outputs: Iterable[str]) -> list[str]:
    recorded = _load(base)

    return [
        output
        for output in outputs
        if (base / output).exists()
        and output in recorded
        and recorded[output] != current(base, output)
    ]
