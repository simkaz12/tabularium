"""Envelope tests: ``compute_confidence``, the ``Verdict``/``RelationVerdict`` validators,
``wording_hash``, and the bounce's keyed-by-method ``JudgedBlock`` (Assessor).
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from tabularium.engine.assessor import (
    RELATION_QUESTION,
    QuestionSpec,
    RelationVerdict,
    Trace,
    Verdict,
    compute_confidence,
    wording_hash,
)
from tabularium.results import JudgedBlock, JudgedUnavailable, JudgedVerdict

_PINNED_RELATION_WORDING_HASH = "ef1ac415b3f156f0"


def _trace() -> Trace:
    return Trace(
        model="test", wording_hash="0" * 16, latency_ms=1,
        input_tokens=None, output_tokens=None, request_id=None,
    )


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("odds", "expected"),
    [
        ({"a": 0.7, "b": 0.2, "c": 0.1}, 0.55),
        ({"a": 0.9, "b": 0.1}, 0.8),
        ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 0.0),
        ({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, 0.0),
        ({"a": 1.0, "b": 0.0}, 1.0),
        ({"a": 1.0}, 1.0),
    ],
)
def test_confidence_table(odds: dict[str, float], expected: float) -> None:
    assert compute_confidence(odds) == pytest.approx(expected, abs=1e-9)


def test_confidence_clamps_below_zero() -> None:
    assert compute_confidence({"a": 0.1, "b": 0.1}) == 0.0


def test_confidence_clamps_above_one() -> None:
    assert compute_confidence({"a": 1.5, "b": 0.0}) == 1.0


# ---------------------------------------------------------------------------
# Verdict validator
# ---------------------------------------------------------------------------


def test_odds_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        Verdict(method="worth", pick="keep", odds={"keep": 0.9}, confidence=1.0, trace=_trace())


def test_pick_must_be_in_odds() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            method="worth", pick="noise", odds={"keep": 1.0}, confidence=1.0, trace=_trace()
        )


def test_odds_values_must_be_non_negative() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            method="worth", pick="keep", odds={"keep": 1.2, "noise": -0.2}, confidence=1.0,
            trace=_trace(),
        )


def test_confidence_must_match_compute_confidence() -> None:
    with pytest.raises(ValidationError):
        Verdict(
            method="worth", pick="keep", odds={"keep": 0.6, "noise": 0.4}, confidence=1.0,
            trace=_trace(),
        )


def test_moved_refused_by_type() -> None:
    with pytest.raises(ValidationError):
        RelationVerdict(
            method="relation", pick="replacement", odds={"replacement": 1.0}, confidence=1.0,
            trace=_trace(), reason_class="moved",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# wording_hash
# ---------------------------------------------------------------------------


def test_relation_wording_hash_is_pinned() -> None:
    assert wording_hash(RELATION_QUESTION) == _PINNED_RELATION_WORDING_HASH


def test_wording_hash_changes_with_a_one_character_edit() -> None:
    edited = QuestionSpec(
        id=RELATION_QUESTION.id,
        type=RELATION_QUESTION.type,
        instructions=RELATION_QUESTION.instructions + ".",
        criteria=RELATION_QUESTION.criteria,
    )
    assert wording_hash(edited) != wording_hash(RELATION_QUESTION)


# ---------------------------------------------------------------------------
# JudgedBlock round-trip
# ---------------------------------------------------------------------------


def test_judged_block_round_trips_through_json() -> None:
    block: JudgedBlock = {
        "relation": JudgedVerdict(
            pick="restatement", odds={"restatement": 1.0}, confidence=1.0, reason_class=None,
        ),
        "worth": JudgedUnavailable(reason="timeout", detail=None),
    }
    adapter: TypeAdapter[JudgedBlock] = TypeAdapter(JudgedBlock)
    raw = adapter.dump_json(block)
    restored = adapter.validate_json(raw)

    assert isinstance(restored["relation"], JudgedVerdict)
    assert restored["relation"].pick == "restatement"
    assert isinstance(restored["worth"], JudgedUnavailable)
    assert restored["worth"].reason == "timeout"
