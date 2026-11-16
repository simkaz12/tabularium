from __future__ import annotations

from typing import Literal

from eval_v1v2.rows import Row

Arm = Literal["v1", "v2_rerank", "v2_plain", "no_memory", "full_history"]
System = Literal["v1", "v2"]
Letter = Literal["A", "B", "C", "D"]


class Message(Row):
    role: str
    content: str


class PersonaQuestion(Row):
    question_id: str
    persona_id: int
    query: str
    choices: list[str]
    gold: Letter
    forget: bool
    pref_type: str


class Chunk(Row):
    chunk_id: str
    persona_id: int
    index: int
    messages: list[Message]


class Note(Row):
    note_id: str
    persona_id: int
    chunk_id: str
    headline: str
    claim: str


class WriteRow(Row):
    system: System
    note_id: str
    persona_id: int
    status: str
    record_id: str | None
    elapsed_ms: int


class HitRow(Row):
    arm: Arm
    question_id: str
    rank: int
    record_id: str
    text: str


class AnswerRow(Row):
    arm: Arm
    question_id: str
    answer: Letter | None
    correct: bool
