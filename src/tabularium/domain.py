"""Domain models: ``Fact``, ``Decision`` and the ``Hypomnema`` union, plus ``Session``.

``Fact``, ``Decision`` and their snapshots are frozen and reject unknown fields, so a ``Fact``
cannot carry ``supersession_chain`` or ``promoted_from`` and a ``Decision`` cannot carry
``decay_due_at`` or ``valence``. ``Hypomnema`` is the union discriminated on ``type``.

To change a record, rebuild it through validation::

    updated = type(record).model_validate({**record.model_dump(mode="python"), **updates})

``model_copy(update=...)`` skips validators. ``Decision.supersession_chain`` and
``promoted_from`` are tuples so that frozen also blocks mutating the collection.

Identifiers are UUIDs. A ``Scope`` matches ``^[a-z][a-z0-9-]{0,31}$``.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, NewType

from pydantic import AfterValidator, BeforeValidator, Field, field_validator, model_validator
from pydantic import BaseModel as _BaseModel

# ---------------------------------------------------------------------------
# Identifier types
# ---------------------------------------------------------------------------

HypomnemaId = NewType("HypomnemaId", uuid.UUID)


def new_id() -> HypomnemaId:
    """Return a fresh random ``HypomnemaId``."""
    return HypomnemaId(uuid.uuid4())


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

Scope = NewType("Scope", str)

_SCOPE_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


def parse_scope(raw: Any) -> Scope:
    """Validate raw input as a ``Scope`` and return the typed value.

    Pattern: ``^[a-z][a-z0-9-]{0,31}$``: lowercase initial, up to 32 characters, hyphens
    allowed, no underscores or uppercase. The value is not normalised.

    Non-string input raises ``ValueError``, which pydantic reports as a ``ValidationError``.
    ``fullmatch`` rejects a trailing newline, which ``$`` would let through.
    """
    if not isinstance(raw, str):
        raise ValueError(
            f"scope must be a str, got {type(raw).__name__!r}: {raw!r}"
        )
    if not _SCOPE_RE.fullmatch(raw):
        raise ValueError(
            f"Invalid scope {raw!r}: must match ^[a-z][a-z0-9-]{{0,31}}$"
        )
    return Scope(raw)


# BeforeValidator runs the pattern check on every model that declares ``scope: _ScopeStr``.
_ScopeStr = Annotated[Scope, BeforeValidator(parse_scope)]


def _require_utc(value: datetime) -> datetime:
    """Reject a naive datetime; ``ValueError`` surfaces as a pydantic ``ValidationError``."""
    if value.tzinfo is None:
        raise ValueError("datetime must be tz-aware UTC")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_require_utc)]


# ---------------------------------------------------------------------------
# Named closed value sets
# ---------------------------------------------------------------------------

SupersessionReason = Literal["correction", "changed", "elaborated", "moved"]

Kind = Literal["fact", "decision"]
Validity = Literal["valid", "invalid"]
Valence = Literal["correction", "guideline"]
InvalidationReason = Literal["correction", "changed"]
Asserter = Literal["owner", "agent"]
BatchOutcome = Literal["ok", "invalid_output", "unavailable"]
FactLifecycle = Literal["active", "hot", "ember", "archived", "shelved"]
DecisionLifecycle = Literal["in_force", "superseded", "revoked"]
SaveLifecycle = Literal["active", "ember"]
VerdictActor = Literal["panel", "engine"]
AssessorPick = Literal["coexist", "replace"]
ReviewBand = Literal["shelve", "keep", "unclear", "contested"]
AdmissionAction = Literal["reinforce", "confirm_distinct", "supersede", "promoted_from"]
AdmissionStep = Literal["resolve", "survivors"]
UnsupersedeShape = Literal["fact_both_live", "decision_restored"]
StoreBackend = Literal["memory", "lancedb"]
CreatedAtSort = Literal["created_at_asc", "created_at_desc"]


# ---------------------------------------------------------------------------
# Shared base config
# ---------------------------------------------------------------------------


class _FrozenForbid(_BaseModel):
    """Shared base: frozen + extra='forbid'."""

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }


# ---------------------------------------------------------------------------
# Snapshot helpers (frozen history bags)
# ---------------------------------------------------------------------------


class SupersedesSnapshot(_FrozenForbid):
    """Frozen snapshot of a superseded predecessor captured at transition time.

    ``title`` is derived from the predecessor's content at snapshot time (first sentence,
    truncated to 80 characters); it is not a record column.
    """

    id: HypomnemaId
    title: str
    created_at: UtcDatetime


class PromotedFromSnapshot(_FrozenForbid):
    """Frozen snapshot of a Fact captured at promotion time.

    ``valence`` is ``None`` when the promoted Fact never had one (only ember-born Facts do).
    """

    id: HypomnemaId
    content: str
    valence: Valence | None
    recurrence_count: int = Field(ge=0)


# ---------------------------------------------------------------------------
# BaseHypomnema
# ---------------------------------------------------------------------------


class BaseHypomnema(_FrozenForbid):
    """Fields shared by both arms of the discriminated union."""

    id: HypomnemaId
    type: Kind  # discriminator
    content: str
    scope: _ScopeStr
    purpose: str | None = None
    audience_hint: tuple[str, ...] | None = None
    source: str | None = None
    validity: Validity = "valid"
    reconciles: HypomnemaId | None = None
    recall_count: int = Field(default=0, ge=0)
    last_referenced_at: UtcDatetime | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    embedding_model: str | None = None


# ---------------------------------------------------------------------------
# Fact arm
# ---------------------------------------------------------------------------


class Fact(BaseHypomnema):
    """A Fact: decaying, feedback-tracked, lifecycle-active record."""

    type: Literal["fact"] = "fact"
    lifecycle: FactLifecycle
    # No validator ties valence to lifecycle; it is set by the ember save path.
    valence: Valence | None = None
    feedback_score: float = 0.0
    hot_rejection_count: int = Field(default=0, ge=0)
    decay_due_at: UtcDatetime | None = None
    # Ember recurrence counters, split by who asserted the recurrence.
    owner_recurrence_count: int = Field(default=0, ge=0)
    agent_recurrence_count: int = Field(default=0, ge=0)
    # The escalation suggestion fires when owner_recurrence_count crosses
    # escalation_baseline + escalation_threshold. set_hot(hot=False) resets the baseline to
    # the current owner_recurrence_count.
    escalation_baseline: int = Field(default=0, ge=0)
    # True marks an ember whose audience was defaulted to every registered role and is awaiting
    # an owner ruling. Persisted because an all-roles hint cannot tell that apart from an
    # owner-chosen one. Promotion and set_audience clear it.
    provisional: bool = Field(default=False)
    # A superseded Fact is validity="invalid" plus this pointer, set in one transition (see the
    # validator below). The converse does not hold: set_validity(valid=False) invalidates
    # without naming a replacement.
    superseded_by: HypomnemaId | None = None
    # content_key and content_hash are derived in the storage layer and are not fields here.
    previous_content: str | None = None
    supersession_reason: SupersessionReason | None = None
    supersession_note: str | None = None
    supersession_edge_id: str | None = None

    @field_validator("feedback_score")
    @classmethod
    def _feedback_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"feedback_score must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _check_supersession_consistency(self) -> Fact:
        if self.superseded_by is not None and self.validity != "invalid":
            raise ValueError(
                "Fact.superseded_by may only be set when validity='invalid' "
                "-- the pointer and the validity flip land in one transition"
            )
        return self


# ---------------------------------------------------------------------------
# Decision arm
# ---------------------------------------------------------------------------


class Decision(BaseHypomnema):
    """A Decision: immutable content, mutable lifecycle, no decay machinery.

    ``decay_due_at``, ``valence``, ``feedback_score`` and ``hot_rejection_count`` are Fact-only
    and rejected at construction. ``superseded_by`` is set exactly when ``lifecycle`` is
    ``superseded``. ``supersession_chain`` and ``promoted_from`` are tuples; pydantic accepts
    ``list`` input and serializes back to ``list``.
    """

    type: Literal["decision"] = "decision"
    content: str
    lifecycle: DecisionLifecycle
    superseded_by: HypomnemaId | None = None
    supersession_chain: tuple[SupersedesSnapshot, ...] = Field(default_factory=tuple)
    promoted_from: tuple[PromotedFromSnapshot, ...] = Field(default_factory=tuple)
    previous_content: str | None = None
    supersession_reason: SupersessionReason | None = None
    supersession_note: str | None = None
    supersession_edge_id: str | None = None
    invalidated_by: str | None = None
    invalidation_reason: InvalidationReason | None = None

    @model_validator(mode="after")
    def _check_supersession_consistency(self) -> Decision:
        if self.lifecycle == "superseded" and self.superseded_by is None:
            raise ValueError(
                "A superseded Decision must carry a superseded_by forward pointer"
            )
        if self.lifecycle != "superseded" and self.superseded_by is not None:
            raise ValueError(
                "superseded_by may only be set when lifecycle='superseded'"
            )
        return self


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


# Session does not subclass BaseHypomnema, is not an arm of the Hypomnema union, and has no
# embedding, lifecycle or validity field. tests/test_domain.py pins the subclass rule.
class Session(_BaseModel):
    """An append-only work-session digest — never a Fact, never a Decision."""

    id: uuid.UUID
    scope: str
    started_at: datetime
    # No created_at: ended_at is the creation moment.
    ended_at: datetime
    content: str
    source: str | None = None

    @model_validator(mode="after")
    def _check_time_range(self) -> Session:
        if self.started_at > self.ended_at:
            raise ValueError("started_at must be <= ended_at")
        return self

    @field_validator("content")
    @classmethod
    def _require_nonblank_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

Hypomnema = Annotated[Fact | Decision, Field(discriminator="type")]
