"""LanceDB codec: PyArrow schemas, timestamp helpers, and row conversion.

Pure data mapping between domain objects and flat LanceDB row dicts; no I/O.

- Schemas: ``make_records_schema(dim)`` (vector width is fixed at adapter-open
  time) and the static ``*_SCHEMA`` constants for the events, read-events,
  pairs, tombstones, registry, sessions, cleaner-state, pass-history, reviews,
  record-judgments, manual-contests, unsupersedes, revalidations and assessor-verdicts tables.
- Timestamps: ``to_utc`` makes a datetime tz-aware; ``from_ts`` reads a
  LanceDB timestamp as ``datetime | None`` (an unconvertible value logs a
  WARNING and reads as ``None``); ``req_ts`` is the non-nullable variant.
- Rows: a ``*_to_row`` / ``row_to_*`` pair per table that is both written and
  read. ``record_to_row`` and ``row_to_record`` convert ``Fact | Decision``;
  ``read_events`` and ``record_judgments`` have a writer only, since nothing
  reads them back. The
  remaining ``row_*`` helpers read one column or one search hit, so the
  adapter never indexes a raw row itself.
- Queries: ``where_clause`` (record filters) and ``sessions_where_clause``, both
  built through ``lance_filters``.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa

from tabularium.domain import (
    Decision,
    Fact,
    HypomnemaId,
    PromotedFromSnapshot,
    Scope,
    Session,
    SupersedesSnapshot,
)
from tabularium.storage.content_key import content_hash, content_key
from tabularium.storage.lance_filters import and_, in_, ts_cmp
from tabularium.storage.types import (
    BackfillTarget,
    CleanerPassRecord,
    CleanerState,
    DistinctPair,
    Filters,
    ReadEvent,
    RecordJudgment,
    RecordReview,
    ReinforcementEvent,
    RevalidationEvent,
    RowMeta,
    Tombstone,
    UnsupersedeEvent,
    VerdictRow,
)

_log = logging.getLogger("tabularium")

# ---------------------------------------------------------------------------
# PyArrow schemas
# ---------------------------------------------------------------------------

# ``make_records_schema`` is built at adapter-open time with the store's fixed vector dimension.

def make_records_schema(dim: int) -> pa.Schema:
    """Build the records table schema with a fixed-size vector column."""
    return pa.schema(
        [
            pa.field("id", pa.utf8(), nullable=False),
            pa.field("type", pa.utf8(), nullable=False),  # "fact" | "decision"
            pa.field("content", pa.utf8(), nullable=False),
            pa.field("scope", pa.utf8(), nullable=False),
            pa.field("purpose", pa.utf8(), nullable=True),
            pa.field("audience_hint", pa.utf8(), nullable=True),  # JSON list | NULL
            pa.field("source", pa.utf8(), nullable=True),
            pa.field("validity", pa.utf8(), nullable=False),
            pa.field("reconciles", pa.utf8(), nullable=True),
            pa.field("recall_count", pa.int64(), nullable=False),
            pa.field("last_referenced_at", pa.timestamp("us", tz="UTC"), nullable=True),
            pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("updated_at", pa.timestamp("us", tz="UTC"), nullable=False),
            # Fact-only fields
            pa.field("lifecycle", pa.utf8(), nullable=False),
            pa.field("valence", pa.utf8(), nullable=True),
            pa.field("feedback_score", pa.float64(), nullable=False),
            pa.field("hot_rejection_count", pa.int64(), nullable=False),
            pa.field("decay_due_at", pa.timestamp("us", tz="UTC"), nullable=True),
            # Ember recurrence-ladder counters
            pa.field("owner_recurrence_count", pa.int64(), nullable=False),
            pa.field("agent_recurrence_count", pa.int64(), nullable=False),
            pa.field("escalation_baseline", pa.int64(), nullable=False),
            # Owner-verdict-outstanding marker; Fact-only, defaults False on Decision rows.
            pa.field("provisional", pa.bool_(), nullable=False),
            # superseded_by is shared by Fact and Decision; supersession_chain and
            # promoted_from are Decision-only (NULL for Facts).
            pa.field("superseded_by", pa.utf8(), nullable=True),
            pa.field("supersession_chain", pa.utf8(), nullable=True),  # JSON
            pa.field("promoted_from", pa.utf8(), nullable=True),  # JSON
            # Embedding provenance
            pa.field("embedding_model", pa.utf8(), nullable=True),
            # Fixed-size list for vector: backs the brute-force cosine search
            pa.field("vector", pa.list_(pa.float32(), dim), nullable=True),
            # content_key and content_hash are derived by the storage layer on every row
            # write and never carried on the domain object; the other six round-trip
            # verbatim.
            pa.field("content_key", pa.utf8(), nullable=True),
            pa.field("content_hash", pa.utf8(), nullable=True),
            pa.field("previous_content", pa.utf8(), nullable=True),
            pa.field("supersession_reason", pa.utf8(), nullable=True),
            pa.field("supersession_note", pa.utf8(), nullable=True),
            pa.field("supersession_edge_id", pa.utf8(), nullable=True),
            pa.field("invalidated_by", pa.utf8(), nullable=True),
            pa.field("invalidation_reason", pa.utf8(), nullable=True),
        ]
    )

EVENTS_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.utf8(), nullable=False),
        pa.field("at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.utf8(), nullable=False),
        pa.field("matched_from_id", pa.utf8(), nullable=True),
        pa.field("value", pa.float64(), nullable=True),
        # Caller-supplied audit string: set on feedback events (NULL when omitted) and
        # on "shelved" events (both directions of set_shelved). NULL for every other
        # source.
        pa.field("reason", pa.utf8(), nullable=True),
        pa.field("similarity", pa.float64(), nullable=True),
    ]
)

# Separate from EVENTS_SCHEMA: the same shape minus the fields only a
# reinforcement event carries (matched_from_id, value, reason). See
# ``storage.types.ReadEvent``.
READ_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.utf8(), nullable=False),
        pa.field("at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("source", pa.utf8(), nullable=False),
    ]
)

PAIRS_SCHEMA = pa.schema(
    [
        pa.field("lower_id", pa.utf8(), nullable=False),
        pa.field("higher_id", pa.utf8(), nullable=False),
        pa.field("source", pa.utf8(), nullable=False),
        pa.field("judged_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("similarity", pa.float64(), nullable=True),
    ]
)

TOMBSTONES_SCHEMA = pa.schema(
    [
        pa.field("tombstone_id", pa.utf8(), nullable=False),
        pa.field("buried_id", pa.utf8(), nullable=False),
        pa.field("type", pa.utf8(), nullable=False),
        pa.field("reason", pa.utf8(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("model_generated", pa.bool_(), nullable=False),
        # Undo marker: set by a pending unsupersede, NULL on ordinary burials.
        pa.field("restores", pa.utf8(), nullable=True),
        pa.field("restores_content_hash", pa.utf8(), nullable=True),
    ]
)

REGISTRY_SCHEMA = pa.schema(
    [
        pa.field("name", pa.utf8(), nullable=False),
    ]
)

# No vector column: a Session is never embedded, deduped or decayed.
SESSIONS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8(), nullable=False),
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("ended_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("content", pa.utf8(), nullable=False),
        pa.field("source", pa.utf8(), nullable=True),
    ]
)

# One row per scope, upserted on ``scope`` via merge_insert. ``last_pass_at``
# and ``report`` are still written on every pass but have no production reader:
# the newest PASS_HISTORY_SCHEMA row answers "when" and "what".
CLEANER_STATE_SCHEMA = pa.schema(
    [
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("watermark_at", pa.timestamp("us", tz="UTC"), nullable=False),
        # Written on every pass; unread (see the pass history).
        pa.field("last_pass_at", pa.timestamp("us", tz="UTC"), nullable=False),
        # Written on every completed or blocked pass; unread (see the pass history).
        pa.field("report", pa.utf8(), nullable=True),
    ]
)

# Append-only, one row per Cleaner pass (separate from cleaner_state, which is
# one row per scope). ``scope``, ``pass_at``, ``outcome`` and ``dry_run`` are
# columns because list queries filter and order on them; the rest of a pass is
# opaque JSON in ``payload``.
PASS_HISTORY_SCHEMA = pa.schema(
    [
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("pass_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("outcome", pa.utf8(), nullable=False),
        pa.field("dry_run", pa.bool_(), nullable=False),
        pa.field("payload", pa.utf8(), nullable=False),
    ]
)

# Append-only, one row per record per revisit-axis pass (judged, skipped or
# protected). ``list_due_for_review`` takes the latest row per record at read
# time; the table never upserts.
REVIEWS_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.utf8(), nullable=False),
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("reviewed_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("next_review_due_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("outcome", pa.utf8(), nullable=False),
        pa.field("band", pa.utf8(), nullable=True),
        pa.field("grades", pa.utf8(), nullable=True),  # JSON-encoded list[str] | NULL
        pa.field("rubric_version", pa.utf8(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# record_judgments: append-only, one row per contest (many per target_id).
# Not a join key to a specific ``record_reviews`` row.
# ---------------------------------------------------------------------------

RECORD_JUDGMENTS_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.utf8(), nullable=False),
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("judged_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("rubric_version", pa.utf8(), nullable=False),
        pa.field("judge_a_identity", pa.utf8(), nullable=False),
        pa.field("judge_a_grade", pa.utf8(), nullable=False),
        pa.field("judge_a_why", pa.utf8(), nullable=False),
        pa.field("judge_b_identity", pa.utf8(), nullable=False),
        pa.field("judge_b_grade", pa.utf8(), nullable=False),
        pa.field("judge_b_why", pa.utf8(), nullable=False),
        pa.field("paired_with", pa.utf8(), nullable=True),  # falsity-axis contests only
        pa.field("similarity", pa.float64(), nullable=True),  # falsity-axis contests only
    ]
)

# ---------------------------------------------------------------------------
# manual_contests: one row per contest, many per target_id. The table is opened
# and maintained but nothing writes or reads it.
# ---------------------------------------------------------------------------

MANUAL_CONTESTS_SCHEMA = pa.schema(
    [
        pa.field("target_id", pa.utf8(), nullable=False),
        pa.field("scope", pa.utf8(), nullable=False),
        pa.field("contested_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("reason", pa.utf8(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# unsupersedes: append-only undo events, idempotent on ``edge_id`` at the call
# level (``append_unsupersede`` checks before writing; LanceDB has no unique
# constraint). Every column except the six structural ones is nullable.
# ---------------------------------------------------------------------------

UNSUPERSEDES_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8(), nullable=False),
        pa.field("edge_id", pa.utf8(), nullable=False),
        pa.field("predecessor_id", pa.utf8(), nullable=False),
        pa.field("successor_id", pa.utf8(), nullable=False),
        pa.field("arm", pa.utf8(), nullable=False),
        pa.field("at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("withdrawn_reason", pa.utf8(), nullable=True),
        pa.field("withdrawn_note", pa.utf8(), nullable=True),
        pa.field("reason", pa.utf8(), nullable=True),
        pa.field("refused_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)

# Content-free revalidation events: ids, the withdrawn invalidation's cause and
# class, and the caller's reason.
REVALIDATIONS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8(), nullable=False),
        pa.field("decision_id", pa.utf8(), nullable=False),
        pa.field("withdrawn_cause", pa.utf8(), nullable=False),
        pa.field("withdrawn_reason", pa.utf8(), nullable=False),
        pa.field("reason", pa.utf8(), nullable=False),
        pa.field("at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# Updated in place by ``id`` (``merge_insert("id").when_matched_update_all()``).
# Every column except ``id``, ``created_at``, ``caller``, ``method`` and
# ``status`` is nullable.
ASSESSOR_VERDICTS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("caller", pa.utf8(), nullable=False),
        pa.field("method", pa.utf8(), nullable=False),
        pa.field("bounce_id", pa.utf8(), nullable=True),
        pa.field("candidate_id", pa.utf8(), nullable=True),
        pa.field("probe_rank", pa.int64(), nullable=True),
        pa.field("similarity", pa.float64(), nullable=True),
        pa.field("pick", pa.utf8(), nullable=True),
        pa.field("odds_json", pa.utf8(), nullable=True),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("reason_class", pa.utf8(), nullable=True),
        pa.field("unavailable_reason", pa.utf8(), nullable=True),
        pa.field("unavailable_detail", pa.utf8(), nullable=True),
        pa.field("model", pa.utf8(), nullable=True),
        pa.field("wording_hash", pa.utf8(), nullable=True),
        pa.field("latency_ms", pa.int64(), nullable=True),
        pa.field("input_tokens", pa.int64(), nullable=True),
        pa.field("output_tokens", pa.int64(), nullable=True),
        pa.field("request_id", pa.utf8(), nullable=True),
        pa.field("status", pa.utf8(), nullable=False),
        pa.field("action", pa.utf8(), nullable=True),
        pa.field("actor", pa.utf8(), nullable=True),
        pa.field("matched", pa.bool_(), nullable=True),
        pa.field("effect_ref", pa.utf8(), nullable=True),
        pa.field("new_record_id", pa.utf8(), nullable=True),
        pa.field("incoming_text", pa.utf8(), nullable=True),
        pa.field("override", pa.utf8(), nullable=True),
        pa.field("override_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("stamped_at", pa.timestamp("us", tz="UTC"), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Timestamp helpers — LanceDB stores UTC, Python expects tz-aware datetimes
# ---------------------------------------------------------------------------


def to_utc(dt: datetime) -> datetime:
    """Ensure the datetime is UTC-aware (LanceDB requires tz-aware timestamps)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def from_ts(val: Any) -> datetime | None:
    """Convert a PyArrow/LanceDB timestamp value to a tz-aware datetime or None.

    A non-datetime value is read as integer microseconds since the epoch. One that
    cannot be converted (``TypeError``, ``ValueError`` or ``OverflowError``) logs a
    WARNING naming the value's type and reads as ``None``; any other exception
    propagates.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=UTC)
        return val
    try:
        return datetime.fromtimestamp(int(val) / 1_000_000, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        _log.warning(
            "Unconvertible timestamp value of type %s decoded as None", type(val).__name__
        )
        return None


def req_ts(val: Any, field: str) -> datetime:
    """Non-nullable timestamp — raises if None."""
    result = from_ts(val)
    if result is None:
        raise ValueError(f"Required timestamp field {field!r} is NULL in LanceDB row")
    return result


# ---------------------------------------------------------------------------
# Record ↔ row conversion helpers
# ---------------------------------------------------------------------------


def _encode_json(obj: Any) -> str | None:
    """Encode a Python value to a JSON string, or None when it is None."""
    if obj is None:
        return None
    return json.dumps(obj)


def record_to_row(
    record: Fact | Decision, vector: list[float] | None, embedding_model: str | None
) -> dict[str, Any]:
    """Convert a domain record and vector to a flat LanceDB row dict.

    ``embedding_model`` is written only when ``vector`` is present, so a
    caller-passed stamp is ignored on a vector-less write and the two columns
    cannot disagree about whether the row is embedded.
    """
    base: dict[str, Any] = {
        "id": str(record.id),
        "type": record.type,
        "content": record.content,
        "scope": str(record.scope),
        "purpose": record.purpose,
        "audience_hint": _encode_json(list(record.audience_hint) if record.audience_hint else None),
        "source": record.source,
        "validity": record.validity,
        "reconciles": str(record.reconciles) if record.reconciles is not None else None,
        "recall_count": record.recall_count,
        "last_referenced_at": (
            to_utc(record.last_referenced_at) if record.last_referenced_at else None
        ),
        "created_at": to_utc(record.created_at),
        "updated_at": to_utc(record.updated_at),
        "lifecycle": record.lifecycle,
        "embedding_model": embedding_model if vector else None,
        # float32-compatible vector
        "vector": [float(x) for x in vector] if vector else None,
        # Fact-only defaults
        "valence": None,
        "feedback_score": 0.0,
        "hot_rejection_count": 0,
        "decay_due_at": None,
        "owner_recurrence_count": 0,
        "agent_recurrence_count": 0,
        "escalation_baseline": 0,
        "provisional": False,
        # superseded_by is shared (Fact + Decision); supersession_chain and
        # promoted_from stay Decision-only.
        "superseded_by": None,
        "supersession_chain": None,
        "promoted_from": None,
        # content_key and content_hash are derived here from the record's content on
        # every write; the other six ride the domain object verbatim.
        "content_key": content_key(record.content),
        "content_hash": content_hash(record.content),
        "previous_content": record.previous_content,
        "supersession_reason": record.supersession_reason,
        "supersession_note": record.supersession_note,
        "supersession_edge_id": record.supersession_edge_id,
        "invalidated_by": record.invalidated_by if isinstance(record, Decision) else None,
        "invalidation_reason": (
            record.invalidation_reason if isinstance(record, Decision) else None
        ),
    }

    if isinstance(record, Fact):
        base["valence"] = record.valence
        base["feedback_score"] = float(record.feedback_score)
        base["hot_rejection_count"] = record.hot_rejection_count
        base["decay_due_at"] = to_utc(record.decay_due_at) if record.decay_due_at else None
        base["owner_recurrence_count"] = record.owner_recurrence_count
        base["agent_recurrence_count"] = record.agent_recurrence_count
        base["escalation_baseline"] = record.escalation_baseline
        base["provisional"] = record.provisional
        base["superseded_by"] = str(record.superseded_by) if record.superseded_by else None
    elif isinstance(record, Decision):
        base["superseded_by"] = str(record.superseded_by) if record.superseded_by else None
        # Serialize tuple/list of SupersedesSnapshot
        chain = record.supersession_chain
        if chain:
            base["supersession_chain"] = json.dumps(
                [
                    {
                        "id": str(s.id),
                        "title": s.title,
                        "created_at": to_utc(s.created_at).isoformat(),
                    }
                    for s in chain
                ]
            )
        prom = record.promoted_from
        if prom:
            base["promoted_from"] = json.dumps(
                [
                    {
                        "id": str(p.id),
                        "content": p.content,
                        "valence": p.valence,
                        "recurrence_count": p.recurrence_count,
                    }
                    for p in prom
                ]
            )

    return base


def row_to_record(row: dict[str, Any]) -> Fact | Decision:
    """Reconstruct a Fact or Decision from a flat LanceDB row dict."""
    rec_type: str = row["type"]

    # Shared fields
    common: dict[str, Any] = {
        "id": HypomnemaId(uuid.UUID(row["id"])),
        "type": rec_type,
        "content": row["content"],
        "scope": Scope(row["scope"]),
        "purpose": row.get("purpose"),
        "audience_hint": json.loads(row["audience_hint"]) if row.get("audience_hint") else None,
        "source": row.get("source"),
        "validity": row["validity"],
        "reconciles": HypomnemaId(uuid.UUID(row["reconciles"])) if row.get("reconciles") else None,
        "recall_count": int(row["recall_count"]),
        "last_referenced_at": from_ts(row.get("last_referenced_at")),
        "created_at": req_ts(row.get("created_at"), "created_at"),
        "updated_at": req_ts(row.get("updated_at"), "updated_at"),
        "lifecycle": row["lifecycle"],
        "embedding_model": row.get("embedding_model"),
        # content_key and content_hash are storage-only and not read back onto the
        # domain object; the other six round-trip.
        "previous_content": row.get("previous_content"),
        "supersession_reason": row.get("supersession_reason"),
        "supersession_note": row.get("supersession_note"),
        "supersession_edge_id": row.get("supersession_edge_id"),
    }

    if rec_type == "fact":
        common["valence"] = row.get("valence")
        common["feedback_score"] = float(row.get("feedback_score") or 0.0)
        common["hot_rejection_count"] = int(row.get("hot_rejection_count") or 0)
        common["decay_due_at"] = from_ts(row.get("decay_due_at"))
        common["owner_recurrence_count"] = int(row.get("owner_recurrence_count") or 0)
        common["agent_recurrence_count"] = int(row.get("agent_recurrence_count") or 0)
        common["escalation_baseline"] = int(row.get("escalation_baseline") or 0)
        common["provisional"] = bool(row.get("provisional") or False)
        common["superseded_by"] = (
            HypomnemaId(uuid.UUID(row["superseded_by"])) if row.get("superseded_by") else None
        )
        return Fact.model_validate(common)

    # Decision
    chain_raw = row.get("supersession_chain")
    chain: list[SupersedesSnapshot] = []
    if chain_raw:
        for item in json.loads(chain_raw):
            chain.append(
                SupersedesSnapshot(
                    id=HypomnemaId(uuid.UUID(item["id"])),
                    title=item["title"],
                    created_at=req_ts(
                        datetime.fromisoformat(item["created_at"]), "chain.created_at"
                    ),
                )
            )

    prom_raw = row.get("promoted_from")
    prom: list[PromotedFromSnapshot] = []
    if prom_raw:
        for item in json.loads(prom_raw):
            prom.append(
                PromotedFromSnapshot(
                    id=HypomnemaId(uuid.UUID(item["id"])),
                    content=item["content"],
                    valence=item["valence"],
                    recurrence_count=item["recurrence_count"],
                )
            )

    common["superseded_by"] = (
        HypomnemaId(uuid.UUID(row["superseded_by"])) if row.get("superseded_by") else None
    )
    common["supersession_chain"] = tuple(chain)
    common["promoted_from"] = tuple(prom)
    common["invalidated_by"] = row.get("invalidated_by")
    common["invalidation_reason"] = row.get("invalidation_reason")
    return Decision.model_validate(common)


# ---------------------------------------------------------------------------
# Filter → WHERE clause builder
# ---------------------------------------------------------------------------


def where_clause(filters: Filters) -> str:
    """Build a LanceDB SQL WHERE clause from a Filters object.

    Produces a string like:
        scope IN ('proj-a', 'global') AND validity IN ('valid') AND lifecycle IN ('active', 'hot')
    """
    return and_(
        in_("scope", filters.scopes),
        in_("validity", filters.validity),
        in_("lifecycle", filters.lifecycle_in),
    )


# ---------------------------------------------------------------------------
# Session <-> row conversion and WHERE-clause builder
# ---------------------------------------------------------------------------


def session_to_row(session: Session) -> dict[str, Any]:
    """Convert a domain ``Session`` to a flat LanceDB row dict (no vector)."""
    return {
        "id": str(session.id),
        "scope": session.scope,
        "started_at": to_utc(session.started_at),
        "ended_at": to_utc(session.ended_at),
        "content": session.content,
        "source": session.source,
    }


def row_to_session(row: dict[str, Any]) -> Session:
    """Reconstruct a domain ``Session`` from a flat LanceDB row dict."""
    return Session(
        id=uuid.UUID(row["id"]),
        scope=row["scope"],
        started_at=req_ts(row.get("started_at"), "started_at"),
        ended_at=req_ts(row.get("ended_at"), "ended_at"),
        content=row["content"],
        source=row.get("source"),
    )


def sessions_where_clause(
    scopes: Sequence[str], *, since: datetime | None, until: datetime | None
) -> str:
    """Build a LanceDB SQL WHERE clause for ``search_sessions``: the scope
    IN-list plus an inclusive ``ended_at`` range predicate on either end.

    Produces a string like:
        scope IN ('proj-a', 'global') AND ended_at >= timestamp '...'
        AND ended_at <= timestamp '...'
    """
    clauses = [in_("scope", scopes)]
    if since is not None:
        clauses.append(ts_cmp("ended_at", ">=", to_utc(since)))
    if until is not None:
        clauses.append(ts_cmp("ended_at", "<=", to_utc(until)))
    return and_(*clauses)


# ---------------------------------------------------------------------------
# CleanerState <-> row conversion
# ---------------------------------------------------------------------------


def cleaner_state_to_row(scope: Scope, state: CleanerState) -> dict[str, Any]:
    """Convert a per-scope ``CleanerState`` to a flat LanceDB row dict."""
    return {
        "scope": str(scope),
        "watermark_at": to_utc(state.watermark_at),
        "last_pass_at": to_utc(state.last_pass_at),
        "report": state.report,
    }


def row_to_cleaner_state(row: dict[str, Any]) -> CleanerState:
    """Reconstruct a ``CleanerState`` from a flat LanceDB row dict.

    ``scope`` is the table's key column, not a ``CleanerState`` field — the
    caller already knows which scope it queried for.
    """
    return CleanerState(
        watermark_at=req_ts(row.get("watermark_at"), "watermark_at"),
        last_pass_at=req_ts(row.get("last_pass_at"), "last_pass_at"),
        report=row.get("report"),
    )


# ---------------------------------------------------------------------------
# CleanerPassRecord <-> row conversion
# ---------------------------------------------------------------------------


def cleaner_pass_to_row(record: CleanerPassRecord) -> dict[str, Any]:
    """Convert a ``CleanerPassRecord`` to a flat LanceDB row dict.

    Unlike ``cleaner_state_to_row``, ``scope`` here is a plain column, not a
    merge_insert key — this table has no key column at all (append-only).
    """
    return {
        "scope": str(record.scope),
        "pass_at": to_utc(record.pass_at),
        "outcome": record.outcome,
        "dry_run": record.dry_run,
        "payload": record.payload,
    }


def row_to_cleaner_pass(row: dict[str, Any]) -> CleanerPassRecord:
    """Reconstruct a ``CleanerPassRecord`` from a flat LanceDB row dict."""
    return CleanerPassRecord(
        scope=Scope(row["scope"]),
        pass_at=req_ts(row.get("pass_at"), "pass_at"),
        outcome=row["outcome"],
        dry_run=bool(row["dry_run"]),
        payload=row["payload"],
    )


# ---------------------------------------------------------------------------
# RecordReview <-> row conversion
# ---------------------------------------------------------------------------


def record_review_to_row(review: RecordReview) -> dict[str, Any]:
    """Convert a ``RecordReview`` to a flat LanceDB row dict.

    ``grades`` follows the same "nullable JSON column, ``None`` when empty" convention
    ``supersession_chain``/``promoted_from`` already use on the ``records`` table.
    """
    return {
        "target_id": str(review.target_id),
        "scope": str(review.scope),
        "reviewed_at": to_utc(review.reviewed_at),
        "next_review_due_at": to_utc(review.next_review_due_at),
        "outcome": review.outcome,
        "band": review.band,
        "grades": _encode_json(list(review.grades)) if review.grades else None,
        "rubric_version": review.rubric_version,
    }


def row_to_record_review(row: dict[str, Any]) -> RecordReview:
    """Reconstruct a ``RecordReview`` from a flat LanceDB row dict."""
    return RecordReview(
        target_id=HypomnemaId(uuid.UUID(row["target_id"])),
        scope=Scope(row["scope"]),
        reviewed_at=req_ts(row.get("reviewed_at"), "reviewed_at"),
        next_review_due_at=req_ts(row.get("next_review_due_at"), "next_review_due_at"),
        outcome=row["outcome"],
        band=row.get("band"),
        grades=tuple(json.loads(row["grades"])) if row.get("grades") else (),
        rubric_version=row["rubric_version"],
    )


def record_judgment_to_row(judgment: RecordJudgment) -> dict[str, Any]:
    """Convert a ``RecordJudgment`` to a flat LanceDB row dict."""
    return {
        "target_id": str(judgment.target_id),
        "scope": str(judgment.scope),
        "judged_at": to_utc(judgment.judged_at),
        "rubric_version": judgment.rubric_version,
        "judge_a_identity": judgment.judge_a_identity,
        "judge_a_grade": judgment.judge_a_grade,
        "judge_a_why": judgment.judge_a_why,
        "judge_b_identity": judgment.judge_b_identity,
        "judge_b_grade": judgment.judge_b_grade,
        "judge_b_why": judgment.judge_b_why,
        "paired_with": str(judgment.paired_with) if judgment.paired_with is not None else None,
        "similarity": judgment.similarity,
    }


# ---------------------------------------------------------------------------
# UnsupersedeEvent <-> row conversion
# ---------------------------------------------------------------------------


def unsupersede_event_to_row(event: UnsupersedeEvent) -> dict[str, Any]:
    """Convert an ``UnsupersedeEvent`` to a flat LanceDB row dict."""
    return {
        "id": str(event.id),
        "edge_id": event.edge_id,
        "predecessor_id": str(event.predecessor_id),
        "successor_id": str(event.successor_id),
        "arm": event.arm,
        "at": to_utc(event.at),
        "withdrawn_reason": event.withdrawn_reason,
        "withdrawn_note": event.withdrawn_note,
        "reason": event.reason,
        "refused_at": to_utc(event.refused_at) if event.refused_at else None,
    }


def row_to_unsupersede_event(row: dict[str, Any]) -> UnsupersedeEvent:
    """Reconstruct an ``UnsupersedeEvent`` from a flat LanceDB row dict."""
    return UnsupersedeEvent(
        id=uuid.UUID(row["id"]),
        edge_id=row["edge_id"],
        predecessor_id=HypomnemaId(uuid.UUID(row["predecessor_id"])),
        successor_id=HypomnemaId(uuid.UUID(row["successor_id"])),
        arm=row["arm"],
        at=req_ts(row.get("at"), "at"),
        withdrawn_reason=row["withdrawn_reason"],
        withdrawn_note=row.get("withdrawn_note"),
        reason=row["reason"],
        refused_at=from_ts(row.get("refused_at")),
    )


def revalidation_event_to_row(event: RevalidationEvent) -> dict[str, Any]:
    """Convert a ``RevalidationEvent`` to a flat LanceDB row dict."""
    return {
        "id": str(event.id),
        "decision_id": str(event.decision_id),
        "withdrawn_cause": str(event.withdrawn_cause),
        "withdrawn_reason": event.withdrawn_reason,
        "reason": event.reason,
        "at": to_utc(event.at),
    }


def row_to_revalidation_event(row: dict[str, Any]) -> RevalidationEvent:
    """Reconstruct a ``RevalidationEvent`` from a flat LanceDB row dict."""
    return RevalidationEvent(
        id=uuid.UUID(row["id"]),
        decision_id=HypomnemaId(uuid.UUID(row["decision_id"])),
        withdrawn_cause=HypomnemaId(uuid.UUID(row["withdrawn_cause"])),
        withdrawn_reason=row["withdrawn_reason"],
        reason=row["reason"],
        at=req_ts(row.get("at"), "at"),
    )


def verdict_row_to_row(verdict_row: VerdictRow) -> dict[str, Any]:
    """Convert a ``VerdictRow`` to a flat LanceDB row dict."""
    return {
        "id": verdict_row.id,
        "created_at": to_utc(verdict_row.created_at),
        "caller": verdict_row.caller,
        "method": verdict_row.method,
        "bounce_id": verdict_row.bounce_id,
        "candidate_id": verdict_row.candidate_id,
        "probe_rank": verdict_row.probe_rank,
        "similarity": verdict_row.similarity,
        "pick": verdict_row.pick,
        "odds_json": verdict_row.odds_json,
        "confidence": verdict_row.confidence,
        "reason_class": verdict_row.reason_class,
        "unavailable_reason": verdict_row.unavailable_reason,
        "unavailable_detail": verdict_row.unavailable_detail,
        "model": verdict_row.model,
        "wording_hash": verdict_row.wording_hash,
        "latency_ms": verdict_row.latency_ms,
        "input_tokens": verdict_row.input_tokens,
        "output_tokens": verdict_row.output_tokens,
        "request_id": verdict_row.request_id,
        "status": verdict_row.status,
        "action": verdict_row.action,
        "actor": verdict_row.actor,
        "matched": verdict_row.matched,
        "effect_ref": verdict_row.effect_ref,
        "new_record_id": verdict_row.new_record_id,
        "incoming_text": verdict_row.incoming_text,
        "override": verdict_row.override,
        "override_at": to_utc(verdict_row.override_at) if verdict_row.override_at else None,
        "stamped_at": to_utc(verdict_row.stamped_at) if verdict_row.stamped_at else None,
    }


def row_to_verdict_row(row: dict[str, Any]) -> VerdictRow:
    """Reconstruct a ``VerdictRow`` from a flat LanceDB row dict."""
    return VerdictRow(
        id=row["id"],
        created_at=req_ts(row.get("created_at"), "created_at"),
        caller=row["caller"],
        method=row["method"],
        bounce_id=row.get("bounce_id"),
        candidate_id=row.get("candidate_id"),
        probe_rank=row.get("probe_rank"),
        similarity=row.get("similarity"),
        pick=row.get("pick"),
        odds_json=row.get("odds_json"),
        confidence=row.get("confidence"),
        reason_class=row.get("reason_class"),
        unavailable_reason=row.get("unavailable_reason"),
        unavailable_detail=row.get("unavailable_detail"),
        model=row.get("model"),
        wording_hash=row.get("wording_hash"),
        latency_ms=row.get("latency_ms"),
        input_tokens=row.get("input_tokens"),
        output_tokens=row.get("output_tokens"),
        request_id=row.get("request_id"),
        status=row["status"],
        action=row.get("action"),
        actor=row.get("actor"),
        matched=row.get("matched"),
        effect_ref=row.get("effect_ref"),
        new_record_id=row.get("new_record_id"),
        incoming_text=row.get("incoming_text"),
        override=row.get("override"),
        override_at=from_ts(row.get("override_at")),
        stamped_at=from_ts(row.get("stamped_at")),
    )


# ---------------------------------------------------------------------------
# records: column readers, row metadata and search hits
# ---------------------------------------------------------------------------


def row_id(row: dict[str, Any]) -> HypomnemaId:
    """The ``id`` column of a ``records`` row."""
    return HypomnemaId(uuid.UUID(row["id"]))


def row_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    """The raw ``id`` column strings of ``rows``, in order, for a merge source table."""
    return [row["id"] for row in rows]


def row_vector(row: dict[str, Any]) -> list[float] | None:
    """The stored vector of a ``records`` row, or ``None`` when the row has none."""
    raw = row.get("vector")
    return [float(x) for x in raw] if raw is not None else None


def row_embedding_model(row: dict[str, Any]) -> str | None:
    """The ``embedding_model`` stamp of a ``records`` row, or ``None``."""
    return row.get("embedding_model")


def row_created_at(row: dict[str, Any]) -> datetime:
    """The non-null ``created_at`` of a ``records`` row."""
    return req_ts(row.get("created_at"), "created_at")


def content_key_columns(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Merge-source columns (``id``, ``content_key``, ``content_hash``) derived from ``rows``.

    ``rows`` carry ``id`` and ``content`` only, as the startup content-key pass selects them.
    """
    return {
        "id": [row["id"] for row in rows],
        "content_key": [content_key(row["content"]) for row in rows],
        "content_hash": [content_hash(row["content"]) for row in rows],
    }


def row_to_row_meta(row: dict[str, Any]) -> RowMeta:
    """Reconstruct a ``RowMeta`` from a flat ``records`` row dict."""
    return RowMeta(
        id=HypomnemaId(uuid.UUID(row["id"])),
        type=row["type"],
        scope=Scope(row["scope"]),
        lifecycle=row["lifecycle"],
        validity=row["validity"],
        recall_count=int(row["recall_count"]),
        last_referenced_at=from_ts(row.get("last_referenced_at")),
        feedback_score=float(row.get("feedback_score") or 0.0),
        decay_due_at=from_ts(row.get("decay_due_at")),
        embedding_model=row.get("embedding_model"),
    )


@dataclass(frozen=True, slots=True)
class HitRow:
    """One search-result row: the decoded record and the leg's raw score.

    ``raw`` is the cosine distance for a vector-leg row and the BM25 score for a
    text-leg row.
    """

    record: Fact | Decision
    raw: float


def row_to_vector_hit_row(row: dict[str, Any]) -> HitRow:
    """Decode a vector-search row; ``raw`` is ``_distance`` (0.0 when absent)."""
    return HitRow(record=row_to_record(row), raw=float(row.get("_distance", 0.0)))


def row_to_text_hit_row(row: dict[str, Any]) -> HitRow:
    """Decode a text-search row; ``raw`` is ``_score`` (0.0 when absent)."""
    return HitRow(record=row_to_record(row), raw=float(row.get("_score", 0.0)))


# ---------------------------------------------------------------------------
# Tombstone <-> row conversion
# ---------------------------------------------------------------------------


def tombstone_to_row(tombstone: Tombstone) -> dict[str, Any]:
    """Convert a ``Tombstone`` to a flat LanceDB row dict."""
    return {
        "tombstone_id": str(tombstone.id),
        "buried_id": str(tombstone.buried_id),
        "type": tombstone.type,
        "reason": tombstone.reason,
        "timestamp": to_utc(tombstone.timestamp),
        "model_generated": tombstone.model_generated,
        "restores": str(tombstone.restores) if tombstone.restores is not None else None,
        "restores_content_hash": tombstone.restores_content_hash,
    }


def row_to_tombstone(row: dict[str, Any]) -> Tombstone:
    """Reconstruct a ``Tombstone`` from a flat LanceDB row dict."""
    restores_raw = row.get("restores")
    return Tombstone(
        id=uuid.UUID(row["tombstone_id"]),
        buried_id=HypomnemaId(uuid.UUID(row["buried_id"])),
        type=row["type"],
        reason=row["reason"],
        timestamp=req_ts(row.get("timestamp"), "timestamp"),
        model_generated=bool(row["model_generated"]),
        restores=HypomnemaId(uuid.UUID(restores_raw)) if restores_raw else None,
        restores_content_hash=row.get("restores_content_hash"),
    )


# ---------------------------------------------------------------------------
# ReinforcementEvent and ReadEvent <-> row conversion
# ---------------------------------------------------------------------------


def event_to_row(event: ReinforcementEvent) -> dict[str, Any]:
    """Convert a ``ReinforcementEvent`` to a flat LanceDB row dict."""
    return {
        "target_id": str(event.target_id),
        "at": to_utc(event.at),
        "source": event.source,
        "matched_from_id": str(event.matched_from_id) if event.matched_from_id else None,
        "value": float(event.value) if event.value is not None else None,
        "reason": event.reason,
        "similarity": float(event.similarity) if event.similarity is not None else None,
    }


def row_to_event(row: dict[str, Any]) -> ReinforcementEvent:
    """Reconstruct a ``ReinforcementEvent`` from a flat LanceDB row dict."""
    return ReinforcementEvent(
        target_id=HypomnemaId(uuid.UUID(row["target_id"])),
        at=req_ts(row.get("at"), "at"),
        source=row["source"],
        matched_from_id=(
            HypomnemaId(uuid.UUID(row["matched_from_id"])) if row.get("matched_from_id") else None
        ),
        value=float(row["value"]) if row.get("value") is not None else None,
        reason=row.get("reason"),
        similarity=float(row["similarity"]) if row.get("similarity") is not None else None,
    )


def read_event_to_row(event: ReadEvent) -> dict[str, Any]:
    """Convert a ``ReadEvent`` to a flat LanceDB row dict."""
    return {
        "target_id": str(event.target_id),
        "at": to_utc(event.at),
        "source": event.source,
    }


# ---------------------------------------------------------------------------
# DistinctPair <-> row conversion
# ---------------------------------------------------------------------------


def distinct_pair_to_row(pair: DistinctPair) -> dict[str, Any]:
    """Convert a ``DistinctPair`` to a flat LanceDB row dict."""
    return {
        "lower_id": str(pair.lower_id),
        "higher_id": str(pair.higher_id),
        "source": pair.source,
        "judged_at": to_utc(pair.judged_at),
        "similarity": float(pair.similarity) if pair.similarity is not None else None,
    }


def row_to_distinct_pair(row: dict[str, Any]) -> DistinctPair:
    """Reconstruct a ``DistinctPair`` from a flat LanceDB row dict."""
    return DistinctPair(
        lower_id=HypomnemaId(uuid.UUID(row["lower_id"])),
        higher_id=HypomnemaId(uuid.UUID(row["higher_id"])),
        source=row["source"],
        judged_at=req_ts(row.get("judged_at"), "judged_at"),
        similarity=float(row["similarity"]) if row.get("similarity") is not None else None,
    )


# ---------------------------------------------------------------------------
# Registries (scopes, facets, roles) and sessions
# ---------------------------------------------------------------------------


def registry_name_to_row(name: str) -> dict[str, Any]:
    """Convert a registry entry to a flat LanceDB row dict."""
    return {"name": name}


def row_to_registry_name(row: dict[str, Any]) -> str:
    """Read the name from a registry row."""
    return str(row["name"])


def row_to_ended_at(row: dict[str, Any]) -> datetime:
    """Read the non-null ``ended_at`` from a ``sessions`` row."""
    return req_ts(row.get("ended_at"), "ended_at")


# ---------------------------------------------------------------------------
# BackfillTarget
# ---------------------------------------------------------------------------


def row_to_backfill_target(row: dict[str, Any]) -> BackfillTarget:
    """Reconstruct a ``BackfillTarget`` from a row that selects only its three columns."""
    return BackfillTarget(
        id=HypomnemaId(uuid.UUID(row["id"])),
        content=row["content"],
        content_hash=row["content_hash"],
    )
