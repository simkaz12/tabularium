from __future__ import annotations

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from eval_v1v2.pmv2.read import (
    ANSWERS_NAME,
    build_prompt,
    parse_letter,
    read_arm,
    render_choices,
    render_notes,
    run,
)
from eval_v1v2.pmv2.report import (
    RESAMPLES,
    SEED,
    accuracy_table,
    difference_table,
    render,
    split_ids,
    write_table,
)
from eval_v1v2.pmv2.rows import AnswerRow, Chunk, HitRow, Message, PersonaQuestion, WriteRow
from eval_v1v2.score_search import bootstrap


class FakeAdapter:
    def __init__(self, answers: list[str] | None = None, raw: list[str] | None = None) -> None:
        self.answers = answers
        self.raw = raw or []
        self.prompts: list[str] = []

    async def call(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.raw:
            return self.raw.pop(0)
        count = int(prompt.split("There are ")[-1].split(" questions")[0])
        letters = self.answers or ["A"] * count
        verdicts = [{"index": i + 1, "answer": letters[i]} for i in range(count)]

        return json.dumps({"verdicts": verdicts})


def question(
    index: int, forget: bool = False, gold: str = "A", persona: int = 1
) -> PersonaQuestion:
    return PersonaQuestion(
        question_id=f"{persona}-{index}",
        persona_id=persona,
        query=f"query {index}",
        choices=["one", "two", "three", "four"],
        gold=gold,  # type: ignore[arg-type]
        forget=forget,
        pref_type="x",
    )


def hit(question_id: str, rank: int, text: str, arm: str = "v1") -> HitRow:
    return HitRow.model_validate(
        {
            "arm": arm,
            "question_id": question_id,
            "rank": rank,
            "record_id": f"r{rank}",
            "text": text,
        }
    )


def test_notes_appear_in_rank_order_and_choices_use_letters() -> None:
    hits = [hit("1-0", 2, "second"), hit("1-0", 1, "first")]
    rendered = render_notes(hits)

    assert rendered.index("first") < rendered.index("second")
    assert render_choices(question(0)).splitlines()[2] == "C. three"

    prompt = build_prompt(["x"])
    assert "A to D" in prompt


def test_hit_arm_prompt_carries_only_its_arm_notes() -> None:
    hits = [hit("1-0", 1, "v1 note", "v1"), hit("1-0", 1, "plain note", "v2_plain")]
    adapter = FakeAdapter()

    asyncio.run(read_arm("v1", [question(0)], hits, [], adapter, set()))

    assert "v1 note" in adapter.prompts[0]
    assert "plain note" not in adapter.prompts[0]


def test_no_memory_has_no_notes_and_full_history_shares_one_history() -> None:
    chunks = [
        Chunk(
            chunk_id="1-c0",
            persona_id=1,
            index=0,
            messages=[Message(role="user", content="I like tea")],
        )
    ]
    questions = [question(0), question(1)]
    adapter = FakeAdapter()

    asyncio.run(read_arm("no_memory", questions, [], chunks, adapter, set()))
    assert "Notes saved" not in adapter.prompts[0]
    assert "I like tea" not in adapter.prompts[0]

    full = FakeAdapter()
    asyncio.run(read_arm("full_history", questions, [], chunks, full, set()))
    assert len(full.prompts) == 1
    assert full.prompts[0].count("I like tea") == 1


def test_batches_cap_at_twenty() -> None:
    questions = [question(i) for i in range(45)]
    adapter = FakeAdapter()

    rows, failed = asyncio.run(read_arm("no_memory", questions, [], [], adapter, set()))

    assert len(adapter.prompts) == 3
    assert len(rows) == 45
    assert failed == 0


def test_unparseable_answer_scores_zero() -> None:
    assert parse_letter("E") is None
    assert parse_letter(" b. ") == "B"

    adapter = FakeAdapter(answers=["A", "Z"])
    rows, _ = asyncio.run(read_arm("no_memory", [question(0), question(1)], [], [], adapter, set()))

    assert [row.correct for row in rows] == [True, False]
    assert rows[1].answer is None


def test_invalid_output_retries_once_then_leaves_question_unanswered() -> None:
    adapter = FakeAdapter(raw=["nope", "still nope"])
    rows, failed = asyncio.run(read_arm("no_memory", [question(0)], [], [], adapter, set()))

    assert rows == []
    assert failed == 1
    assert len(adapter.prompts) == 2


def test_run_is_resumable_per_arm_and_question(tmp_path: pathlib.Path) -> None:
    questions = [question(0), question(1)]
    (tmp_path / "questions.jsonl").write_text(
        "\n".join(q.model_dump_json() for q in questions) + "\n", encoding="utf-8"
    )

    first = FakeAdapter()
    asyncio.run(run(tmp_path, first, ["no_memory"], limit=1))
    second = FakeAdapter()
    asyncio.run(run(tmp_path, second, ["no_memory"]))

    lines = (tmp_path / ANSWERS_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert len(second.prompts) == 1
    assert "query 0" not in second.prompts[0]


def answer(arm: str, question_id: str, correct: bool) -> AnswerRow:
    return AnswerRow.model_validate(
        {"arm": arm, "question_id": question_id, "answer": "A", "correct": correct}
    )


def test_split_sums_and_missing_answers_count_as_wrong() -> None:
    questions = [question(0, forget=True), question(1), question(2)]
    ids = split_ids(questions)

    assert len(ids["forget"]) + len(ids["other"]) == len(ids["all"])

    answers = [answer("v1", "1-0", True), answer("v1", "1-1", True)]
    table = accuracy_table(questions, answers)
    v1_line = next(line for line in table.splitlines() if line.startswith("| v1 "))

    assert "66.7% (2/3)" in v1_line
    assert "100.0% (1/1)" in v1_line
    assert "| 1 |" in v1_line


def test_bootstrap_is_deterministic_under_fixed_seed() -> None:
    diffs = [1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    first = bootstrap(diffs, seed=SEED, resamples=RESAMPLES)
    second = bootstrap(diffs, seed=SEED, resamples=RESAMPLES)

    assert first == second
    assert first is not None

    questions = [question(i) for i in range(8)]
    answers = [answer("v1", q.question_id, i % 2 == 0) for i, q in enumerate(questions)]
    answers += [answer("v2_plain", q.question_id, True) for q in questions]

    assert difference_table(questions, answers) == difference_table(questions, answers)


def test_write_table_counts_statuses_and_records() -> None:
    def write(system: str, note: str, status: str, record: str | None) -> WriteRow:
        return WriteRow.model_validate(
            {
                "system": system,
                "note_id": note,
                "persona_id": 1,
                "status": status,
                "record_id": record,
                "elapsed_ms": 1,
            }
        )

    writes = [
        write("v1", "n1", "saved", "a"),
        write("v1", "n2", "bounced", None),
        write("v2", "n1", "saved", "b"),
    ]
    table = write_table(writes)
    v1_line = next(line for line in table.splitlines() if line.startswith("| v1 "))

    assert v1_line.startswith("| v1 | 2 | 1 |")

    report = render([question(0)], [answer("v1", "1-0", True)], writes)
    assert "## Accuracy" in report
    assert "## Write side" in report


def test_ask_table_counts_stored_and_answers() -> None:
    from eval_v1v2.pmv2.report import ask_table

    table = ask_table(
        [
            {"note_id": "1-n0", "status": "approved", "ask_answer": "no_reply", "record_id": "r1"},
            {"note_id": "1-n1", "status": "held", "ask_answer": "no_reply", "record_id": None},
        ]
    )

    assert "| 2 | 1 | no_reply: 2 |" in table


def test_v2_records_stored_includes_expired_ask_records() -> None:
    def write(note: str, record: str | None) -> WriteRow:
        return WriteRow.model_validate(
            {
                "system": "v2",
                "note_id": note,
                "persona_id": 1,
                "status": "saved" if record else "asked",
                "record_id": record,
                "elapsed_ms": 1,
            }
        )

    asks = [{"note_id": "n2", "ask_answer": "unanswered", "record_id": "b"}]
    table = write_table([write("n1", "a"), write("n2", None)], asks)
    v2_line = next(line for line in table.splitlines() if line.startswith("| v2 "))

    assert v2_line.startswith("| v2 | 2 | 2 |")
