"""``compute_projections``: one derivation for reinforce, feedback and the projection rebuild."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine.projections import (
    Projections,
    compute_decay_due_at,
    compute_ewma,
    compute_last_referenced_at,
    compute_projections,
    compute_recall_count,
)
from tabularium.storage.types import ReinforcementEvent

_NOW = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)


def _fact() -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content="a fact",
        scope=Scope("proj-a"),
        lifecycle="active",
        validity="valid",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _decision() -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content="a decision",
        scope=Scope("proj-a"),
        lifecycle="in_force",
        validity="valid",
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_fact_with_events_uses_the_four_pure_helpers() -> None:
    cfg = EngineConfig()
    fact = _fact()
    events = [
        ReinforcementEvent(target_id=fact.id, at=_NOW, source="dedup"),
        ReinforcementEvent(
            target_id=fact.id,
            at=_NOW + timedelta(days=1),
            source="feedback",
            value=0.8,
        ),
    ]

    result = compute_projections(fact, events, cfg)

    recall_count = compute_recall_count(events)
    last_referenced_at = compute_last_referenced_at(events)
    feedback_score = compute_ewma(events)
    assert result == Projections(
        recall_count=recall_count,
        last_referenced_at=last_referenced_at,
        feedback_score=feedback_score,
        decay_due_at=compute_decay_due_at(
            recall_count=recall_count,
            feedback_score=feedback_score,
            last_referenced_at=last_referenced_at,
            created_at=fact.created_at,
            cfg=cfg,
        ),
    )


def test_fact_with_zero_events_has_no_decay_due_at() -> None:
    result = compute_projections(_fact(), [], EngineConfig())

    assert result == Projections(
        recall_count=0,
        last_referenced_at=None,
        feedback_score=0.0,
        decay_due_at=None,
    )


def test_decision_gets_neutral_feedback_and_no_decay() -> None:
    decision = _decision()
    events = [
        ReinforcementEvent(target_id=decision.id, at=_NOW, source="revival"),
        ReinforcementEvent(
            target_id=decision.id,
            at=_NOW + timedelta(days=1),
            source="feedback",
            value=0.9,
        ),
    ]

    result = compute_projections(decision, events, EngineConfig())

    assert result == Projections(
        recall_count=1,
        last_referenced_at=_NOW,
        feedback_score=0.0,
        decay_due_at=None,
    )


def test_projections_are_frozen() -> None:
    result = compute_projections(_fact(), [], EngineConfig())

    with pytest.raises(AttributeError):
        result.recall_count = 1  # type: ignore[misc]
