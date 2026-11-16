"""Tests for domain.py -- construction, arm separation, and Scope validation.

Covers SPEC §1.1 canonical field inventory and §2 Scope pattern.
"""

from __future__ import annotations

import datetime
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from tabularium.domain import (
    BaseHypomnema,
    Decision,
    Fact,
    Hypomnema,
    HypomnemaId,
    PromotedFromSnapshot,
    Session,
    SupersedesSnapshot,
    new_id,
    parse_scope,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = datetime.UTC


def _now() -> datetime.datetime:
    return datetime.datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)


def _id() -> HypomnemaId:
    return new_id()


def _base_fact(**kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": _id(),
        "type": "fact",
        "content": "Tabularium uses LanceDB as its default vector store.",
        "scope": "tabularium",
        "lifecycle": "active",
        "created_at": _now(),
        "updated_at": _now(),
    }
    defaults.update(kw)
    return defaults


def _base_decision(**kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": _id(),
        "type": "decision",
        "content": "All time flows through the Clock seam.",
        "scope": "tabularium",
        "lifecycle": "in_force",
        "created_at": _now(),
        "updated_at": _now(),
    }
    defaults.update(kw)
    return defaults


# ---------------------------------------------------------------------------
# A1 -- Legal construction
# ---------------------------------------------------------------------------


class TestFactConstruction:
    def test_minimal_fact_constructs(self) -> None:
        f = Fact(**_base_fact())
        assert f.type == "fact"
        assert f.lifecycle == "active"
        assert f.validity == "valid"
        assert f.valence is None
        assert f.feedback_score == 0.0

    def test_fact_with_all_optional_fields(self) -> None:
        f = Fact(
            **_base_fact(
                purpose="architecture",
                audience_hint=["implementer"],
                source="agent-a",
                valence="guideline",
                feedback_score=0.75,
                hot_rejection_count=2,
                decay_due_at=_now(),
                lifecycle="ember",
            )
        )
        assert f.valence == "guideline"
        assert f.feedback_score == 0.75

    def test_feedback_score_boundary_zero(self) -> None:
        f = Fact(**_base_fact(feedback_score=0.0))
        assert f.feedback_score == 0.0

    def test_feedback_score_boundary_one(self) -> None:
        f = Fact(**_base_fact(feedback_score=1.0))
        assert f.feedback_score == 1.0

    def test_feedback_score_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(feedback_score=1.5))

    def test_fact_is_frozen(self) -> None:
        f = Fact(**_base_fact())
        with pytest.raises((ValidationError, AttributeError)):
            f.content = "mutated"  # pydantic frozen raises ValidationError at runtime


class TestDecisionConstruction:
    def test_minimal_decision_constructs(self) -> None:
        d = Decision(**_base_decision())
        assert d.type == "decision"
        assert d.lifecycle == "in_force"
        assert d.superseded_by is None
        assert d.supersession_chain == ()
        assert d.promoted_from == ()

    def test_superseded_decision_requires_forward_pointer(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(lifecycle="superseded", superseded_by=None))

    def test_superseded_by_only_when_superseded(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(lifecycle="in_force", superseded_by=_id()))

    def test_valid_superseded_decision(self) -> None:
        fwd_id = _id()
        snap = SupersedesSnapshot(id=_id(), title="Prior decision", created_at=_now())
        d = Decision(
            **_base_decision(
                lifecycle="superseded",
                superseded_by=fwd_id,
                supersession_chain=[snap],
            )
        )
        assert d.superseded_by == fwd_id
        assert len(d.supersession_chain) == 1

    def test_decision_is_frozen(self) -> None:
        d = Decision(**_base_decision())
        with pytest.raises((ValidationError, AttributeError)):
            d.content = "mutated"  # pydantic frozen raises ValidationError at runtime


# ---------------------------------------------------------------------------
# A2 -- Wrong-arm field rejection (extra="forbid")
# ---------------------------------------------------------------------------


class TestArmSeparation:
    def test_superseded_by_on_fact_rejected(self) -> None:
        """``superseded_by`` is a Decision-only field; Fact must reject it."""
        with pytest.raises(ValidationError):
            Fact(**_base_fact(superseded_by=_id()))

    def test_supersession_chain_on_fact_rejected(self) -> None:
        snap = SupersedesSnapshot(id=_id(), title="x", created_at=_now())
        with pytest.raises(ValidationError):
            Fact(**_base_fact(supersession_chain=[snap]))

    def test_promoted_from_on_fact_rejected(self) -> None:
        snap = PromotedFromSnapshot(
            id=_id(), content="x", valence="guideline", recurrence_count=1
        )
        with pytest.raises(ValidationError):
            Fact(**_base_fact(promoted_from=[snap]))

    def test_decay_due_at_on_decision_rejected(self) -> None:
        """``decay_due_at`` is Fact-only; Decision must reject it."""
        with pytest.raises(ValidationError):
            Decision(**_base_decision(decay_due_at=_now()))

    def test_valence_on_decision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(valence="correction"))

    def test_feedback_score_on_decision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(feedback_score=0.5))

    def test_hot_rejection_count_on_decision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(hot_rejection_count=1))


# ---------------------------------------------------------------------------
# A3 -- Scope validation -- property-based
# ---------------------------------------------------------------------------

# Valid scope strategy: [a-z][a-z0-9-]{0,31}  total length 1-32
_VALID_SCOPE_STRATEGY = st.from_regex(r"^[a-z][a-z0-9-]{0,31}$", fullmatch=True)

# Invalid strategies -- each triggers rejection
_UPPERCASE_STRATEGY = st.from_regex(r"^[A-Z][a-z0-9-]{0,31}$", fullmatch=True)
_LEADING_DIGIT_STRATEGY = st.from_regex(r"^[0-9][a-z0-9-]{0,30}$", fullmatch=True)
_UNDERSCORE_STRATEGY = st.from_regex(r"^[a-z][a-z0-9_-]{0,30}_[a-z0-9-]{0,10}$", fullmatch=True)
_TOOLONG_STRATEGY = st.from_regex(r"^[a-z][a-z0-9-]{32,50}$", fullmatch=True)
_EMPTY_STRATEGY = st.just("")


class TestScopeValidation:
    @given(_VALID_SCOPE_STRATEGY)
    @settings(max_examples=200)
    def test_valid_scopes_pass(self, s: str) -> None:
        scope = parse_scope(s)
        assert scope == s

    @given(_UPPERCASE_STRATEGY)
    @settings(max_examples=50)
    def test_uppercase_initial_rejected(self, s: str) -> None:
        with pytest.raises(ValueError):
            parse_scope(s)

    @given(_LEADING_DIGIT_STRATEGY)
    @settings(max_examples=50)
    def test_leading_digit_rejected(self, s: str) -> None:
        with pytest.raises(ValueError):
            parse_scope(s)

    @given(_UNDERSCORE_STRATEGY)
    @settings(max_examples=50)
    def test_underscore_rejected(self, s: str) -> None:
        with pytest.raises(ValueError):
            parse_scope(s)

    @given(_TOOLONG_STRATEGY)
    @settings(max_examples=50)
    def test_too_long_rejected(self, s: str) -> None:
        with pytest.raises(ValueError):
            parse_scope(s)

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_scope("")

    def test_scope_embedded_in_fact_validated(self) -> None:
        """The _ScopeStr annotation must fire inside a model, not just standalone."""
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope="BadScope"))

    def test_scope_with_leading_digit_in_fact(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope="1bad"))

    def test_scope_with_underscore_in_fact(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope="bad_scope"))

    def test_scope_global_is_valid(self) -> None:
        scope = parse_scope("global")
        assert scope == "global"

    def test_scope_max_length_valid(self) -> None:
        # 32 chars: 1 leading + 31 rest
        s = "a" + "b" * 31
        assert len(s) == 32
        scope = parse_scope(s)
        assert scope == s

    def test_scope_33_chars_rejected(self) -> None:
        s = "a" + "b" * 32
        assert len(s) == 33
        with pytest.raises(ValueError):
            parse_scope(s)


# ---------------------------------------------------------------------------
# A4 -- Discriminated-union round-trip
# ---------------------------------------------------------------------------

_hypomnema_adapter: TypeAdapter[Hypomnema] = TypeAdapter(Hypomnema)


class TestDiscriminatedUnionRoundTrip:
    def test_fact_round_trip(self) -> None:
        original = Fact(**_base_fact())
        dumped = original.model_dump()
        restored = _hypomnema_adapter.validate_python(dumped)
        assert isinstance(restored, Fact)
        assert restored == original

    def test_decision_round_trip(self) -> None:
        original = Decision(**_base_decision())
        dumped = original.model_dump()
        restored = _hypomnema_adapter.validate_python(dumped)
        assert isinstance(restored, Decision)
        assert restored == original

    def test_fact_round_trip_json(self) -> None:
        original = Fact(**_base_fact(valence="correction", lifecycle="ember"))
        json_str = original.model_dump_json()
        restored = _hypomnema_adapter.validate_json(json_str)
        assert isinstance(restored, Fact)
        assert restored.valence == "correction"

    def test_decision_round_trip_with_chain(self) -> None:
        snap = SupersedesSnapshot(id=_id(), title="Old", created_at=_now())
        fwd = _id()
        original = Decision(
            **_base_decision(
                lifecycle="superseded",
                superseded_by=fwd,
                supersession_chain=[snap],
            )
        )
        dumped = original.model_dump()
        restored = _hypomnema_adapter.validate_python(dumped)
        assert isinstance(restored, Decision)
        assert restored.lifecycle == "superseded"
        assert len(restored.supersession_chain) == 1

    def test_unknown_type_rejected(self) -> None:
        data = {**_base_fact(), "type": "unknown"}
        with pytest.raises((ValidationError, Exception)):
            _hypomnema_adapter.validate_python(data)

    def test_naive_datetime_rejected(self) -> None:
        """Naive datetimes must be rejected on all fields (SPEC §1.1 + §8.7)."""
        with pytest.raises(ValidationError):
            Fact(**_base_fact(created_at=datetime.datetime(2025, 1, 1)))


# ---------------------------------------------------------------------------
# A5 -- Regression: birth-frozen lineage cannot be mutated in place
# ---------------------------------------------------------------------------


class TestBirthFrozenLineage:
    def test_supersession_chain_is_immutable_collection(self) -> None:
        """supersession_chain on a frozen Decision must not allow in-place append."""
        snap = SupersedesSnapshot(id=_id(), title="Prior", created_at=_now())
        fwd = _id()
        d = Decision(
            **_base_decision(
                lifecycle="superseded",
                superseded_by=fwd,
                supersession_chain=[snap],
            )
        )
        # The field is a tuple — no .append method exists.
        assert not hasattr(d.supersession_chain, "append"), (
            "supersession_chain must be a tuple, not a list — "
            "mutable lists allow silent in-place mutation of birth-frozen lineage"
        )

    def test_promoted_from_is_immutable_collection(self) -> None:
        """promoted_from on a frozen Decision must not allow in-place append."""
        d = Decision(**_base_decision())
        assert not hasattr(d.promoted_from, "append"), (
            "promoted_from must be a tuple, not a list"
        )


# ---------------------------------------------------------------------------
# A6 -- Regression: non-string scope raises ValidationError, not TypeError
# ---------------------------------------------------------------------------


class TestScopeNonStringInput:
    def test_integer_scope_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope=123))

    def test_none_scope_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope=None))

    def test_bytes_scope_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope=b"tabularium"))

    def test_trailing_newline_scope_rejected(self) -> None:
        """Trailing newline rejected (fullmatch not match -- Python $ matches before newline)."""
        with pytest.raises(ValueError):
            parse_scope("tabularium\n")

    def test_trailing_space_scope_rejected(self) -> None:
        with pytest.raises(ValueError):
            parse_scope("tabularium ")

    def test_trailing_newline_in_model_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(scope="tabularium\n"))


# ---------------------------------------------------------------------------
# A7 -- Regression: count fields reject negative values
# ---------------------------------------------------------------------------


class TestCountFieldConstraints:
    def test_recall_count_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(recall_count=-1))

    def test_hot_rejection_count_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Fact(**_base_fact(hot_rejection_count=-1))

    def test_recall_count_zero_valid(self) -> None:
        f = Fact(**_base_fact(recall_count=0))
        assert f.recall_count == 0

    def test_recall_count_on_decision_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Decision(**_base_decision(recall_count=-1))

    def test_recurrence_count_negative_on_snapshot_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PromotedFromSnapshot(
                id=_id(), content="x", valence="guideline", recurrence_count=-1
            )

    def test_recurrence_count_zero_on_snapshot_valid(self) -> None:
        snap = PromotedFromSnapshot(
            id=_id(), content="x", valence="guideline", recurrence_count=0
        )
        assert snap.recurrence_count == 0


# ---------------------------------------------------------------------------
# A8 -- Session: deliberately NOT a hypomnema (a locked design decision)
# ---------------------------------------------------------------------------


def _base_session(**kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": _id(),
        "scope": "tabularium",
        "started_at": _now(),
        "ended_at": _now(),
        "content": "Landed the Session domain model.",
    }
    defaults.update(kw)
    return defaults


class TestSessionArmSeparation:
    def test_session_is_not_a_hypomnema(self) -> None:
        """The separation is pinned by a test, not left to convention."""
        assert not issubclass(Session, BaseHypomnema)

    def test_session_field_set_is_exact(self) -> None:
        """A future field addition has to argue with this test, not drift in."""
        assert set(Session.model_fields.keys()) == {
            "id",
            "scope",
            "started_at",
            "ended_at",
            "content",
            "source",
        }


class TestSessionConstruction:
    def test_minimal_session_constructs(self) -> None:
        s = Session(**_base_session())
        assert s.source is None

    def test_session_with_source(self) -> None:
        s = Session(**_base_session(source="digest"))
        assert s.source == "digest"


class TestSessionValidators:
    def test_started_after_ended_rejected(self) -> None:
        t = _now()
        with pytest.raises(ValidationError):
            Session(
                **_base_session(
                    started_at=t + datetime.timedelta(hours=1),
                    ended_at=t,
                )
            )

    def test_started_equals_ended_allowed(self) -> None:
        t = _now()
        s = Session(**_base_session(started_at=t, ended_at=t))
        assert s.started_at == s.ended_at

    def test_started_before_ended_allowed(self) -> None:
        t = _now()
        s = Session(
            **_base_session(started_at=t, ended_at=t + datetime.timedelta(hours=1))
        )
        assert s.started_at < s.ended_at

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Session(**_base_session(content=""))

    def test_whitespace_only_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Session(**_base_session(content="   \n\t "))
