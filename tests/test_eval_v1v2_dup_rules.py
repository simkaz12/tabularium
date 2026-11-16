from __future__ import annotations

import pathlib
import sys
from typing import Any

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import dup_rules  # noqa: E402
from eval_v1v2.dup_rules import (  # noqa: E402
    BASELINE,
    Score,
    SignalRow,
    compose,
    merge_rule,
    replacement_rule,
    score_rule,
    select_for,
    stability,
)
from eval_v1v2.rows import SaveRow  # noqa: E402

sys.path.pop(0)


def signal(
    pair_id: str,
    pick: str | None,
    *,
    restatement: float = 0.0,
    replacement: float = 0.0,
    similarity: float = 0.9,
    adds: str | None = "yes",
    adds_p: float = 0.95,
    found: bool = True,
) -> SignalRow:
    existing: dict[str, Any] | None = None

    if found:
        existing = {
            "id": f"{pair_id}-record",
            "pick": pick,
            "confidence": 0.9,
            "probabilities": {
                "restatement": restatement,
                "replacement": replacement,
                "follow_up": 0.5,
                "distinct": 0.1,
                "unclear": 0.0,
            },
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
                "answer": adds,
                "probability": adds_p if adds is not None else None,
                "probabilities": None,
                "elapsed_ms": 200.0,
                "error": None,
            },
            "propose_ms": 500.0,
        }
    )


def by_id(*rows: SignalRow) -> dict[str, SignalRow]:
    return {row.pair_id: row for row in rows}


def test_baseline_follows_the_pick() -> None:
    assert BASELINE.decide(signal("a", "restatement")) == "merge"
    assert BASELINE.decide(signal("b", "replacement")) == "flag_replace"
    assert BASELINE.decide(signal("c", "follow_up")) == "new"
    assert BASELINE.decide(signal("d", "distinct")) == "new"
    assert BASELINE.decide(signal("e", "restatement", found=False)) == "new"


def test_p_restatement_threshold_is_inclusive_and_overrides_the_pick() -> None:
    rule = merge_rule("p_restatement", 0.3, None, None)

    assert rule.decide(signal("a", "follow_up", restatement=0.30)) == "merge"
    assert rule.decide(signal("b", "follow_up", restatement=0.29)) == "new"
    assert rule.decide(signal("c", "replacement", restatement=0.35)) == "merge"
    assert rule.decide(signal("d", "replacement", restatement=0.1)) == "flag_replace"


def test_no_new_info_needs_a_follow_up_pick_and_a_no_answer() -> None:
    rule = merge_rule("no_new_info", None, None, None)

    assert rule.decide(signal("a", "follow_up", adds="no")) == "merge"
    assert rule.decide(signal("b", "follow_up", adds="yes")) == "new"
    assert rule.decide(signal("c", "follow_up", adds=None)) == "new"
    assert rule.decide(signal("d", "distinct", adds="no")) == "new"
    assert rule.decide(signal("e", "replacement", adds="no")) == "flag_replace"


def test_no_new_info_probability_floor() -> None:
    rule = merge_rule("no_new_info", None, 0.8, None)

    assert rule.decide(signal("a", "follow_up", adds="no", adds_p=0.80)) == "merge"
    assert rule.decide(signal("b", "follow_up", adds="no", adds_p=0.79)) == "new"


def test_both_requires_each_signal_and_either_requires_one() -> None:
    both = merge_rule("both", 0.3, None, None)
    either = merge_rule("either", 0.3, None, None)
    only_probability = signal("a", "follow_up", restatement=0.4, adds="yes")
    only_question = signal("b", "follow_up", restatement=0.1, adds="no")
    neither = signal("c", "follow_up", restatement=0.1, adds="yes")
    together = signal("d", "follow_up", restatement=0.4, adds="no")

    assert [both.decide(row) for row in (only_probability, only_question, neither, together)] == [
        "new",
        "new",
        "new",
        "merge",
    ]
    assert [either.decide(row) for row in (only_probability, only_question, neither, together)] == [
        "merge",
        "merge",
        "new",
        "merge",
    ]


def test_similarity_floor_gates_the_rule_and_falls_back_to_baseline() -> None:
    rule = merge_rule("p_restatement", 0.3, None, 0.85)

    assert rule.decide(signal("a", "follow_up", restatement=0.5, similarity=0.85)) == "merge"
    assert rule.decide(signal("b", "follow_up", restatement=0.5, similarity=0.849)) == "new"
    assert rule.decide(signal("c", "restatement", restatement=0.9, similarity=0.5)) == "merge"


def test_replacement_rule_flags_only_follow_up_picks() -> None:
    rule = replacement_rule(0.25, None)

    assert rule.decide(signal("a", "follow_up", replacement=0.25)) == "flag_replace"
    assert rule.decide(signal("b", "follow_up", replacement=0.24)) == "new"
    assert rule.decide(signal("c", "distinct", replacement=0.9)) == "new"
    assert rule.decide(signal("d", "restatement", replacement=0.9)) == "merge"


def test_compose_lets_the_merge_win_and_otherwise_flags() -> None:
    rule = compose(merge_rule("p_restatement", 0.3, None, None), replacement_rule(0.3, None))

    assert rule.decide(signal("a", "follow_up", restatement=0.4, replacement=0.4)) == "merge"
    assert rule.decide(signal("b", "follow_up", restatement=0.1, replacement=0.4)) == "flag_replace"
    assert rule.decide(signal("c", "follow_up", restatement=0.1, replacement=0.1)) == "new"


def test_a_merged_keyed_follow_up_is_a_lost_record_and_a_merged_replacement_a_lost_update() -> None:
    rule = merge_rule("p_restatement", 0.3, None, None)
    signals = by_id(
        signal("keep-fu", "follow_up", restatement=0.4),
        signal("keep-di", "distinct", restatement=0.4),
        signal("update", "follow_up", restatement=0.4),
        signal("dup", "follow_up", restatement=0.4),
        signal("dup-missed", "follow_up", restatement=0.1),
    )
    key = {
        "keep-fu": "follow_up",
        "keep-di": "distinct",
        "update": "replacement",
        "dup": "restatement",
        "dup-missed": "restatement",
    }
    score = score_rule(rule, signals, key)

    assert score.lost_record == 2
    assert score.lost_update == 1
    assert score.merged_restatement == 1
    assert (score.restatements, score.replacements, score.unrelated) == (2, 1, 2)
    assert score_rule(BASELINE, signals, key).lost == 0


def test_disputed_pairs_never_enter_the_scored_counts() -> None:
    rule = merge_rule("p_restatement", 0.3, None, None)
    signals = by_id(
        signal("keyed", "follow_up", restatement=0.4),
        signal("disputed-a", "follow_up", restatement=0.4),
        signal("disputed-b", "replacement", restatement=0.0),
        signal("disputed-c", "distinct", restatement=0.0),
    )
    score = score_rule(rule, signals, {"keyed": "follow_up"})

    assert score.unrelated == 1
    assert score.lost_record == 1
    assert score.restatements == score.replacements == 0
    assert score.merged_restatement == score.flagged_replacement == score.false_flag == 0
    assert score.disputed == {"merge": 1, "flag_replace": 1, "new": 1}


def test_false_flag_counts_flags_on_unrelated_pairs_and_flagged_replacement_the_right_ones() -> (
    None
):
    rule = replacement_rule(0.3, None)
    signals = by_id(
        signal("update", "follow_up", replacement=0.5),
        signal("keep", "follow_up", replacement=0.5),
    )
    score = score_rule(rule, signals, {"update": "replacement", "keep": "follow_up"})

    assert score.flagged_replacement == 1
    assert score.false_flag == 1
    assert score.lost == 0


@pytest.mark.parametrize(
    ("merged", "flagged", "lost_record", "lost_update", "expected"),
    [
        (35, 41, 0, 0, True),
        (34, 41, 0, 0, False),
        (35, 40, 0, 0, False),
        (35, 41, 1, 0, False),
        (35, 41, 0, 1, False),
    ],
)
def test_beats_v1_needs_every_condition(
    merged: int, flagged: int, lost_record: int, lost_update: int, expected: bool
) -> None:
    score = Score(
        merged_restatement=merged,
        flagged_replacement=flagged,
        lost_record=lost_record,
        lost_update=lost_update,
    )

    assert score.beats_v1 is expected


def test_the_grid_has_the_documented_size() -> None:
    assert len(list(dup_rules.merge_rules())) == 4 * (7 + 5 + 35 + 35)
    assert len(list(dup_rules.replacement_rules())) == 4 * 7
    assert len(list(dup_rules.composite_rules())) == 4 * (7 + 5 + 35 + 35) * 7


def test_selection_sees_only_the_training_sources() -> None:
    loose = merge_rule("p_restatement", 0.15, None, None)
    strict = merge_rule("p_restatement", 0.5, None, None)
    signals = by_id(
        signal("train-dup", "follow_up", restatement=0.2),
        signal("train-fu", "follow_up", restatement=0.1),
        signal("test-fu", "follow_up", restatement=0.2),
    )
    key = {"train-dup": "restatement", "train-fu": "follow_up", "test-fu": "follow_up"}
    train = frozenset({"train-dup", "train-fu"})

    chosen = select_for([strict, loose], signals, key, train)

    assert chosen is loose
    assert score_rule(chosen, signals, key, frozenset({"test-fu"})).lost_record == 1
    assert select_for([strict, loose], signals, key, frozenset({"test-fu"})) is strict


def test_stability_counts_changed_picks_including_to_and_from_none() -> None:
    published = [
        SaveRow.model_validate(
            {
                "pair_id": pid,
                "system": "v2",
                "bucket": "new",
                "found_existing": pick is not None,
                "relation_pick": pick,
                "raw_outcome": "approved",
                "elapsed_ms": 1.0,
                "degraded": None,
            }
        )
        for pid, pick in [("a", "follow_up"), ("b", "restatement"), ("c", None), ("d", "distinct")]
    ]
    now = [
        signal("a", "follow_up"),
        signal("b", "follow_up"),
        signal("c", "distinct"),
        signal("d", "distinct", found=False),
    ]
    changed, moves = stability(now, published)

    assert changed == 3
    assert moves[("restatement", "follow_up")] == 1
    assert moves[("none", "distinct")] == 1
    assert moves[("distinct", "none")] == 1
