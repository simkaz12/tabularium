from __future__ import annotations

import json
import pathlib
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from eval_v1v2.pmv2.prep import build_chunks, build_question, build_questions
from eval_v1v2.pmv2.rows import PersonaQuestion


def csv_row(persona_id: int = 39, updated: str = "True") -> dict[str, str]:
    return {
        "persona_id": str(persona_id),
        "user_query": repr({"role": "user", "content": "What should I eat tonight?"}),
        "correct_answer": "Something vegetarian",
        "incorrect_answers": json.dumps(["Steak", "Burger", "Sushi"]),
        "updated": updated,
        "pref_type": "ask_to_forget",
        "chat_history_32k_link": "data/chat_history_32k/x.json",
    }


def history(pairs: int, system: bool = True) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": '{"traits": "secret"}'})
    for index in range(pairs):
        messages.append({"role": "user", "content": f"u{index}"})
        messages.append({"role": "assistant", "content": f"a{index}"})

    return {"metadata": {}, "chat_history": messages}


def test_parsed_row() -> None:
    question = build_question(csv_row(), 7)

    assert question.question_id == "39-7"
    assert question.query == "What should I eat tonight?"
    assert sorted(question.choices) == ["Burger", "Something vegetarian", "Steak", "Sushi"]
    assert question.choices["ABCD".index(question.gold)] == "Something vegetarian"
    assert question.forget is True
    assert question.pref_type == "ask_to_forget"


def test_gold_letter_stable_across_runs() -> None:
    first = build_question(csv_row(), 7)
    second = build_question(csv_row(), 7)

    assert first.choices == second.choices
    assert first.gold == second.gold


def test_forget_false_and_persona_filter() -> None:
    rows = [csv_row(39, "False"), csv_row(1, "True")]

    questions = build_questions(rows, [39])

    assert [question.question_id for question in questions] == ["39-0"]
    assert questions[0].forget is False


def test_system_message_dropped() -> None:
    chunks = build_chunks(39, history(2))

    roles = [message.role for chunk in chunks for message in chunk.messages]

    assert "system" not in roles
    assert roles == ["user", "assistant"] * 2


def test_nine_pairs_chunk_as_four_four_one() -> None:
    chunks = build_chunks(39, history(9))

    assert [len(chunk.messages) for chunk in chunks] == [8, 8, 2]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].chunk_id == "39-c0"


def test_history_without_system_keeps_first_message() -> None:
    chunks = build_chunks(39, history(1, system=False))

    assert chunks[0].messages[0].content == "u0"


def test_rows_forbid_extra_fields() -> None:
    payload = build_question(csv_row(), 1).model_dump()
    payload["extra"] = 1

    with pytest.raises(ValidationError):
        PersonaQuestion(**payload)
