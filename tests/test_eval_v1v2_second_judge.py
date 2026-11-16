from __future__ import annotations

import json
import pathlib
import sys

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2.dup_rules import BASELINE, SignalRow, score_rule  # noqa: E402
from eval_v1v2.second_judge import (  # noqa: E402
    SecondCall,
    UnparseableAnswer,
    band_pair_ids,
    load_codex_key,
    parse_answer,
    render_prompt,
    second_rule,
)

sys.path.pop(0)


def signal(
    pair_id: str, pick: str | None, *, similarity: float = 0.9, found: bool = True
) -> SignalRow:
    existing = None

    if found:
        existing = {
            "id": f"{pair_id}-record",
            "pick": pick,
            "confidence": 0.9,
            "probabilities": None,
            "reason_class": None,
            "similarity": similarity,
        }

    return SignalRow.model_validate(
        {
            "pair_id": pair_id,
            "bucket": "new",
            "raw_outcome": "approved",
            "found_existing": found,
            "relation_pick": pick if found else None,
            "degraded": None,
            "existing": existing,
            "top_other": None,
            "adds_information": {
                "answer": None,
                "probability": None,
                "probabilities": None,
                "elapsed_ms": 1.0,
                "error": None,
            },
            "propose_ms": 1.0,
        }
    )


def call(pair_id: str, model: str, label: str, confidence: float | None = 0.95) -> SecondCall:
    return SecondCall(
        pair_id=pair_id,
        model=model,
        label=label,
        confidence=confidence,
        new_information=None,
        elapsed_ms=1.0,
        attempts=1,
        error=None,
    )


def calls_of(*rows: SecondCall) -> dict[str, dict[str, SecondCall]]:
    table: dict[str, dict[str, SecondCall]] = {"haiku": {}, "sonnet": {}}

    for row in rows:
        table[row.model][row.pair_id] = row

    return table


def by_id(*rows: SignalRow) -> dict[str, SignalRow]:
    return {row.pair_id: row for row in rows}


def test_override_applies_only_inside_the_band() -> None:
    calls = calls_of(
        call("in", "haiku", "restatement"),
        call("jev-merge", "haiku", "follow_up"),
        call("jev-flag", "haiku", "follow_up"),
        call("jev-distinct", "haiku", "restatement"),
        call("missing", "haiku", "restatement"),
    )
    rule = second_rule(("haiku",), calls, None, None)

    assert rule.decide(signal("in", "follow_up")) == "merge"
    assert rule.decide(signal("jev-merge", "restatement")) == "merge"
    assert rule.decide(signal("jev-flag", "replacement")) == "flag_replace"
    assert rule.decide(signal("jev-distinct", "distinct")) == "new"
    assert rule.decide(signal("missing", "restatement", found=False)) == "new"


def test_similarity_floor_bounds_the_band() -> None:
    rows = by_id(signal("low", "follow_up", similarity=0.79), signal("high", "follow_up"))
    calls = calls_of(call("low", "haiku", "restatement"), call("high", "haiku", "restatement"))
    rule = second_rule(("haiku",), calls, 0.80, None)

    assert band_pair_ids(rows, 0.80) == ["high"]
    assert band_pair_ids(rows, None) == ["low", "high"]
    assert rule.decide(rows["low"]) == "new"
    assert rule.decide(rows["high"]) == "merge"


def test_second_judge_label_maps_to_a_decision() -> None:
    calls = calls_of(
        call("a", "sonnet", "replacement"),
        call("b", "sonnet", "distinct"),
        call("c", "sonnet", "follow_up"),
    )
    rule = second_rule(("sonnet",), calls, None, None)

    assert rule.decide(signal("a", "follow_up")) == "flag_replace"
    assert rule.decide(signal("b", "follow_up")) == "new"
    assert rule.decide(signal("c", "follow_up")) == "new"


def test_a_merged_keyed_follow_up_is_a_lost_record_and_a_merged_replacement_a_lost_update() -> None:
    rows = by_id(signal("fu", "follow_up"), signal("rp", "follow_up"), signal("rs", "follow_up"))
    key = {"fu": "follow_up", "rp": "replacement", "rs": "restatement"}
    calls = calls_of(
        call("fu", "haiku", "restatement"),
        call("rp", "haiku", "restatement"),
        call("rs", "haiku", "restatement"),
    )
    score = score_rule(second_rule(("haiku",), calls, None, None), rows, key)
    base = score_rule(BASELINE, rows, key)

    assert (score.lost_record, score.lost_update, score.merged_restatement) == (1, 1, 1)
    assert (base.lost_record, base.lost_update, base.merged_restatement) == (0, 0, 0)


def test_both_agree_acts_only_on_equal_labels() -> None:
    calls = calls_of(
        call("same", "haiku", "restatement"),
        call("same", "sonnet", "restatement"),
        call("split", "haiku", "restatement"),
        call("split", "sonnet", "follow_up"),
        call("solo", "haiku", "restatement"),
    )
    rule = second_rule(("haiku", "sonnet"), calls, None, None)

    assert rule.decide(signal("same", "follow_up")) == "merge"
    assert rule.decide(signal("split", "follow_up")) == "new"
    assert rule.decide(signal("solo", "follow_up")) == "new"


def test_none_answer_falls_back_to_the_jev_pick() -> None:
    calls = calls_of(call("a", "haiku", "none"), call("b", "haiku", "none"))
    rule = second_rule(("haiku",), calls, None, None)

    assert rule.decide(signal("a", "follow_up")) == "new"
    assert rule.decide(signal("b", "restatement")) == "merge"

    both = second_rule(("haiku", "sonnet"), calls_of(call("a", "haiku", "restatement")), None, None)

    assert both.decide(signal("a", "follow_up")) == "new"


def test_confidence_floor_is_inclusive_and_falls_back_below_it() -> None:
    calls = calls_of(
        call("at", "haiku", "restatement", 0.8), call("under", "haiku", "restatement", 0.79)
    )
    rule = second_rule(("haiku",), calls, None, 0.8)

    assert rule.decide(signal("at", "follow_up")) == "merge"
    assert rule.decide(signal("under", "follow_up")) == "new"


def test_codex_alone_key_scores_every_pair(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "labels.jsonl"
    lines = []

    for index in range(242):
        for judge in ("claude", "codex"):
            label = "restatement" if judge == "codex" else "distinct"
            lines.append(
                json.dumps(
                    {
                        "item_id": f"p{index}",
                        "judge": judge,
                        "model": "m",
                        "label": label,
                        "reason": "r",
                    }
                )
            )

    path.write_text("\n".join(lines), encoding="utf-8")
    key = load_codex_key(path)
    rows = by_id(*(signal(f"p{index}", "follow_up") for index in range(242)))
    calls = calls_of(*(call(f"p{index}", "haiku", "restatement") for index in range(242)))
    score = score_rule(second_rule(("haiku",), calls, None, None), rows, key)

    assert len(key) == 242
    assert set(key.values()) == {"restatement"}
    assert score.restatements == 242
    assert score.merged_restatement == 242
    assert sum(score.disputed.values()) == 0


def test_prompt_carries_both_texts_and_keeps_braces() -> None:
    text = render_prompt("old {x}", "new {y}")

    assert "old {x}" in text
    assert "new {y}" in text
    assert "<<EXISTING>>" not in text


def test_parse_answer_accepts_fenced_json_and_rejects_bad_confidence() -> None:
    answer = parse_answer('```json\n{"label": "follow_up", "confidence": 0.6}\n```')

    assert (answer.label, answer.confidence) == ("follow_up", 0.6)

    for bad in (
        "no json here",
        '{"label": "maybe", "confidence": 0.5}',
        '{"label": "restatement", "confidence": 1.5}',
    ):
        with pytest.raises(UnparseableAnswer):
            parse_answer(bad)
