from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

VECTOR_DIM = 768

Source = Literal["jev_pairs", "bounce_log", "golden", "admission"]
Kind = Literal["fact", "decision"]
Judge = Literal["claude", "codex"]
Bucket = Literal["caught", "flagged", "new", "wrong_target"]
ReplayBucket = Literal["superseded", "asked"]
System = Literal["v1", "v2"]
Arm = Literal["v1", "v2_rerank", "v2_plain", "v2_rerank_facts", "v2_plain_facts"]
Origin = Literal["mined", "generated"]
KeySource = Literal["agreed", "owner"]
EndState = Literal["folded", "both_live", "not_asked", "unreplayed", "no_reply", "failed"]
Replay = Literal["fold", "keep_both", "continue", "none"]
Label = Literal["restatement", "replacement", "follow_up", "distinct", "yes", "no", 0, 1, 2]

Vector = Annotated[list[float], Field(min_length=VECTOR_DIM, max_length=VECTOR_DIM)]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PairRow(Row):
    pair_id: str
    source: Source
    kind: Kind
    existing: str
    incoming: str
    similarity: float | None
    old_label: str | None


class NoiseRow(Row):
    id: str
    kind: Kind
    content: str
    vector: Vector
    recall_count: int
    created_at: str
    lifecycle: str


class LabelRow(Row):
    item_id: str
    judge: Judge
    model: str
    label: Label
    reason: str


class KeyRow(Row):
    item_id: str
    label: Label
    source: KeySource


class NeighbourRow(Row):
    id: str
    source: str
    similarity: float
    relation_pick: str | None
    relation_confidence: float | None
    new_covers_old: float | None
    same_subject: float | None
    is_seed: bool


class AnswerOutcome(Row):
    replay: Replay
    status: str
    ask_answer: str | None
    end_state: EndState
    existing_live: bool
    existing_superseded_by_fold: bool
    fold_live: bool
    held_live: bool
    fold_covers_old: float | None
    fold_covers_new: float | None
    error: str | None


class SaveRow(Row):
    pair_id: str
    system: System
    bucket: Bucket | ReplayBucket
    found_existing: bool
    relation_pick: str | None
    raw_outcome: str
    elapsed_ms: float
    degraded: str | None
    neighbours: list[NeighbourRow] | None = None
    answer: AnswerOutcome | None = None


class CorpusRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    kind: Kind
    content: str
    scope: str
    lifecycle: str
    vector: Vector
    recall_count: int
    created_at: str


class QuestionRow(Row):
    question_id: str
    query: str
    scope: str
    origin: Origin


class SearchRow(Row):
    question_id: str
    arm: Arm
    rank: int
    record_id: str
    score: float
    elapsed_ms: float


def read_jsonl[RowT: BaseModel](path: Path, model: type[RowT]) -> list[RowT]:
    return list(iter_jsonl(path, model))


def iter_jsonl[RowT: BaseModel](path: Path, model: type[RowT]) -> Iterator[RowT]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield model.model_validate_json(line)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count
