"""Audit and log rows: events, pass history, reviews, judgments and verdicts."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from tabularium.domain import (
    HypomnemaId,
    InvalidationReason,
    Kind,
    Scope,
    SupersessionReason,
)

# ---------------------------------------------------------------------------
# ReinforcementEvent
# ---------------------------------------------------------------------------


class ReinforcementEvent(BaseModel):
    """Append-only reinforcement event.

    The event log is the source of truth. ``recall_count``, ``last_referenced_at``,
    ``feedback_score`` and ``decay_due_at`` are projections the engine recomputes
    and persists through ``update_projections`` after each append.

    - ``value``: meaningful only for ``source="feedback"``; ``None`` otherwise.
    - ``matched_from_id``: the existing record that triggered reinforcement (the
      dedup probe's winner), for lineage tracing and the offline reconciliation
      judge.
    - ``similarity``: the raw cosine the gate's probe (``select_best_candidate`` in
      ``engine/gate.py``) computed for the targeted candidate, threaded through from
      the probe rather than recomputed. Set only for ``source="dedup"``; ``None`` for
      other sources (a ``"revival"`` candidate has no bounce to carry a score) and
      for events written before the field existed.
    - ``reason``: the caller-supplied audit string of ``feedback(id, value,
      reason=None)``, kept as given; ``None``, never an empty string, when omitted.
      ``source="shelved"`` (``set_shelved``, both directions) also populates it,
      because ``set_validity``'s in-memory audit map does not survive a restart.
    - ``source="edit"``: written by a content-edit method that no longer exists;
      nothing appends it now, and the value stays in the ``Literal`` so old rows
      decode. ``reason`` is ``None``.
    - ``source="unsupersede"`` is retired (undo's trail is the ``unsupersedes``
      table); the value stays in the ``Literal`` so old rows decode.
    - ``"shelve_reverted"``: written by the Cleaner when it undoes its own failed
      trim save. Revisit's protection check counts only ``"shelved"`` events, so a
      self-revert never reads as an owner round trip.

    ``shelved``, ``shelve_reverted``, ``unsupersede`` and ``edit`` are excluded from
    every projection formula (``engine/projections.py``): they are demotions or
    curator mutations, not references or feedback.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    at: datetime
    source: Literal[
        "dedup",
        "feedback",
        "revival",
        "compact",
        "shelved",
        "shelve_reverted",
        "unsupersede",
        "edit",
    ]
    matched_from_id: HypomnemaId | None = None
    value: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None
    similarity: float | None = None


# ---------------------------------------------------------------------------
# ReadEvent: a separate, append-only log from ReinforcementEvent

ReadSource = Literal["grounding", "recall", "console", "decision-view"]


class ReadEvent(BaseModel):
    """Append-only read event, in its own table separate from ``ReinforcementEvent``.

    Same shape (id, timestamp, source; no content), but reads must not move
    ``recall_count``, ``last_referenced_at``, ``decay_due_at`` or any other
    importance-decay field. ``update_projections`` never reads this log and nothing
    computed over it is persisted onto a record.

    One row is appended per record actually delivered, never for a candidate
    scanned but not returned. Live sources: ``"recall"`` (a ``recall()`` hit) and
    ``"decision-view"`` (the ``decision()`` full view). ``"console"`` (the removed
    web console) and ``"grounding"`` (once logged for ``context()`` payloads) are no
    longer written; they stay in ``ReadSource`` so old rows decode, and the
    conformance suite constructs ``source="grounding"`` directly.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    at: datetime
    source: ReadSource


# ---------------------------------------------------------------------------
# CleanerState
# ---------------------------------------------------------------------------


class CleanerState(BaseModel):
    """Per-scope Cleaner pass state (per scope, not per record).

    ``watermark_at``: the ``created_at`` cutoff of the last completed and verified
    pass on this scope; every record with ``created_at <= watermark_at`` has been
    judged and nothing after it has. ``get_cleaner_state`` returns ``None`` (never
    a default or epoch-backed ``CleanerState``) for a scope never judged; the
    caller decides what that means (initialize the watermark to now and judge
    nothing). This type and its readers and writers never manufacture that
    default.

    ``last_pass_at``: the wall-clock instant the most recent completed pass
    finished, distinct from the ``created_at`` cutoff ``watermark_at``.

    ``report``: nullable JSON payload.

    NOTE: ``last_pass_at`` and ``report`` are written on every pass but have no
    production reader; the newest ``CleanerPassRecord`` (``list_cleaner_passes``)
    answers "when" and "what".
    """

    model_config = {"frozen": True}

    watermark_at: datetime
    # Written on every pass; no production reader (see ``CleanerPassRecord.pass_at``).
    last_pass_at: datetime
    # Written on every completed or blocked pass; no production reader (see
    # ``CleanerPassRecord.payload``).
    report: str | None = None


# ---------------------------------------------------------------------------
# CleanerPassRecord
# ---------------------------------------------------------------------------


class CleanerPassRecord(BaseModel):
    """One row of the append-only pass-history table, beside ``cleaner_state``.

    ``cleaner_state`` is upserted (one row per scope) because the watermark is a
    current value. This table gives every pass its own row, including a second pass
    for one scope on the same day: the watermark says what has been judged, this
    table says what happened each time. Append-only with no pruning, cap, rollup or
    update/delete leg.

    ``outcome`` mirrors ``CleanSummary.outcome`` (``engine/clean.py``) plus
    ``"errored"``: a crash caught above ``clean()`` has no ``CleanSummary`` to carry
    an outcome.

    ``payload`` is opaque JSON text at the storage seam, like
    ``cleaner_state.report``, so what a row records can grow without a schema
    migration.
    """

    model_config = {"frozen": True}

    scope: Scope
    pass_at: datetime
    outcome: Literal["initialized", "empty_pool", "completed", "blocked", "errored"]
    dry_run: bool
    payload: str


# RecordReview: one row of the revisit axis's append-only review table


ReviewOutcome = Literal[
    "judged", "skipped_no_evidence", "skipped_too_large", "protected", "assessor_resolved"
]


class RecordReview(BaseModel):
    """One row of the revisit axis's append-only ``record_reviews`` table.

    One row per record per pass it was handled on (judged, skipped or protected). A
    skipped record still gets a row, or it would sit at the front of the rotation
    and block everything behind it. The table has no ``id`` column; a row is
    identified positionally by ``target_id`` and ``reviewed_at``, and
    ``RecordJudgment`` carries the judge detail instead.

    ``next_review_due_at`` is computed by the engine at append time
    (``engine.review_schedule.next_review_due_at``), never by the adapter, so
    ``list_due_for_review`` stays a plain ``<=`` comparison.

    ``outcome`` says why the row exists:

    - ``"judged"``: the panel graded it; ``band`` and ``grades`` are populated.
    - ``"skipped_no_evidence"``: no later session returned to the subject and no
      lifecycle event since birth (a deterministic rule, not a judge call).
    - ``"skipped_too_large"``: the record's own content and history exceed the
      prompt ceiling even alone, so no judge ran; ``band`` is ``None``.
    - ``"protected"``: an un-shelved record the panel may never re-shelve.
    - ``"assessor_resolved"``: a relevance or falsity split the Assessor decided
      outright (``split_keep`` / ``split_coexist``), recorded with ``band="keep"``
      and no panel grades; the Assessor's verdict row lives in
      ``assessor_verdicts``.

    ``band`` and ``grades`` are ``None`` and ``()`` on ``"skipped_no_evidence"`` and
    ``"protected"``; ``grades`` is also ``()`` on ``"assessor_resolved"``.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    scope: Scope
    reviewed_at: datetime
    next_review_due_at: datetime
    outcome: ReviewOutcome
    band: Literal["shelve", "keep", "contested", "unclear"] | None = None
    grades: tuple[str, ...] = ()
    rubric_version: str


# ---------------------------------------------------------------------------
# RecordJudgment: Cleaner-owned judge-panel verdicts, separate from RecordReview
# ---------------------------------------------------------------------------


class RecordJudgment(BaseModel):
    """The current judge-panel verdict behind a record's contested status: who judged it, and why.

    Its own table, not columns on ``RecordReview``. Append-only and keyed on the
    record's own id, ``target_id``, which is not unique: a record contested twice has
    two rows, so the earlier panel's ``why`` survives.

    It is not keyed to join a specific ``record_reviews`` row, because
    ``(target_id, reviewed_at)`` is not collision-free:

    1. Within one falsity pass, two newcomers can pick the same older record as
       their best candidate (``engine.falsity._generate_pairs`` has no uniqueness
       constraint). If both pairs land ``"contested"``, that record gets two review
       rows at the same ``reviewed_at``.
    2. Across axes in one Cleaner pass, relevance, falsity and revisit each sample
       their own ``clock.now()``. Under a fixed test clock those are identical, so a
       record contested by two axes can collide.

    Written only by a judge panel's contested branch (``run_revisit_axis``,
    ``run_relevance_axis``, ``run_falsity_axis``); a human adjudication never
    writes here. Human verdicts yield only ``"keep"`` and ``"shelve"`` on
    ``record_reviews``, so the ``"contested"`` band is judge-only.

    ``paired_with`` and ``similarity`` are set only for a falsity-axis contest,
    which disputes two records: both get their own row naming the other via
    ``paired_with`` (the same-pass collision above applies). They are ``None`` for
    relevance and revisit contests.

    Every judge field is required: this table has one kind of writer.
    """

    model_config = {"frozen": True}

    target_id: HypomnemaId
    scope: Scope
    judged_at: datetime
    rubric_version: str
    judge_a_identity: str
    judge_a_grade: str
    judge_a_why: str
    judge_b_identity: str
    judge_b_grade: str
    judge_b_why: str
    paired_with: HypomnemaId | None = None
    similarity: float | None = None


# ---------------------------------------------------------------------------
# UnsupersedeEvent: the undo audit trail, separate from ``events``
# ---------------------------------------------------------------------------


class UnsupersedeEvent(BaseModel):
    """One row of the append-only ``unsupersedes`` table: the content-free undo audit trail.

    Keyed by ``edge_id``, the supersession edge's own id, not by (predecessor,
    successor, reason): a successor can be reused after an undo, and the triple
    could not tell generations apart. ``append_unsupersede`` is idempotent on
    ``edge_id``, so a retry after a crash between the event write and the records
    transaction leaves one row.

    Carries ids, the withdrawn supersession class and note, and the caller's
    retraction ``reason``, never record content. ``withdrawn_reason`` and
    ``withdrawn_note`` are the values undo clears from the restored row.
    ``withdrawn_reason`` is nullable: a row superseded before the reason column
    existed has none even after the startup edge-id pass gives it an edge id, so
    the class is honestly unknown, never fabricated.

    ``refused_at`` is ``None`` until ``mark_unsupersede_refused`` stamps it, the
    first of the two refusal writes (event, then ``withdraw_tombstone``). A crash
    between them leaves a durable signal that a later repair pass can finish or
    leave alone, instead of re-deriving the decision from a tombstone that may be
    gone. It is cleared only by ``append_unsupersede`` reopening a refused edge on
    a fresh live ``unsupersede``.
    """

    model_config = {"frozen": True}

    id: uuid.UUID
    edge_id: str
    predecessor_id: HypomnemaId
    successor_id: HypomnemaId
    arm: Kind
    withdrawn_reason: SupersessionReason | None
    withdrawn_note: str | None
    reason: str
    at: datetime
    refused_at: datetime | None = None


class RevalidationEvent(BaseModel):
    """One row of the append-only ``revalidations`` table: the audit trail for revalidation.

    ``StorageAdapter.revalidate`` writes it before the row flip that clears
    ``invalidated_by`` and ``invalidation_reason``, so a crash between them leaves
    a durable trail of what was about to be revalidated. Re-running
    ``set_validity(valid=True)`` retries the flip and writes a second event: this
    is an audit log, not an idempotency guard like ``unsupersedes``. Content-free.

    ``withdrawn_cause``: the Fact id that invalidated ``decision_id`` (its
    ``invalidated_by``, read fresh). ``withdrawn_reason``: the invalidation's reason
    class, ``"correction"`` or ``"changed"``, never ``None``, since an event is
    written only when an invalidation exists. ``reason``: the caller's
    justification for revalidating, distinct from the reason class.
    """

    model_config = {"frozen": True}

    id: uuid.UUID
    decision_id: HypomnemaId
    withdrawn_cause: HypomnemaId
    withdrawn_reason: InvalidationReason
    reason: str
    at: datetime


# ---------------------------------------------------------------------------
# VerdictRow: the ``assessor_verdicts`` table, one row per Assessor verdict per candidate
# ---------------------------------------------------------------------------


VerdictStatus = Literal["unanswered", "pending", "acted", "failed", "sibling_acted"]


class VerdictRow(BaseModel):
    """One row of the ``assessor_verdicts`` table: append-only in spirit, updated in place.

    Written ``status="unanswered"`` for each probe survivor's verdict on a bounce
    returned to a caller, linked by ``bounce_id`` (``None`` for a row an automatic
    exit dispatched with no bounce shown). A caller replaying the bounce's ``how``
    (naming the same ``bounce_id``), or an automatic exit taking the top verdict,
    moves the row through ``pending`` to ``acted`` or ``failed``; siblings on a
    multi-candidate bounce land ``sibling_acted``.

    ``method``, ``pick``, ``odds_json``, ``confidence``, ``reason_class`` and the
    trace columns (``model``, ``wording_hash``, ``latency_ms``, ``input_tokens``,
    ``output_tokens``, ``request_id``) are ``None`` on an ``unavailable_reason`` row,
    since no verdict was produced. ``odds`` is JSON text because the label set
    differs by method.

    Every column named above except ``method`` and ``latency_ms`` defaults to ``None``, as do
    ``unavailable_reason``, ``unavailable_detail`` and every outcome column below except
    ``new_record_id``, so a writer names only the half of the row that applies.

    ``action``, ``actor``, ``matched``, ``effect_ref`` and ``new_record_id`` are the
    outcome. ``action`` is the exit taken: the bounce-exit vocabulary (``reinforce``,
    ``confirm_distinct``, ``supersede``, ``promoted_from``) or the engine's
    automatic-exit vocabulary (``reinforce``, ``write_new``, ``supersede``,
    ``invalidate``). ``effect_ref`` lets reconcile and the reversal tools find the
    effect again: a supersession edge id, an invalidated Decision id, or a reinforce
    event key (``"{target_id}@{at_iso}"``). ``new_record_id`` is the record the exit
    wrote, if any.

    ``incoming_text`` carries text a discarding exit (reinforce) would otherwise
    lose. It is scrubbed to ``None`` on every row naming ``candidate_id`` when that
    record is buried, and no other field is scrubbed.

    ``override`` and ``override_at`` are stamped by the four reversal tools
    (``unsupersede``, ``revoke``, ``set_validity``, ``set_shelved``) at their write
    point, or by a caller resolve action that does not match the row's own pick
    (``override="caller_against_pick"``, stamped at resolve time). The first
    override wins and is never overwritten.
    """

    model_config = {"frozen": True}

    id: str
    created_at: datetime
    caller: Literal["save", "cleaner_prepass", "cleaner_split", "checkpoint"]
    method: str
    bounce_id: str | None
    candidate_id: str | None
    probe_rank: int | None
    similarity: float | None
    pick: str | None = None
    odds_json: str | None = None
    confidence: float | None = None
    reason_class: str | None = None
    unavailable_reason: str | None = None
    unavailable_detail: str | None = None
    model: str | None = None
    wording_hash: str | None = None
    latency_ms: int | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    status: VerdictStatus
    action: str | None = None
    actor: Literal["engine", "caller", "panel", "owner"] | None = None
    matched: bool | None = None
    effect_ref: str | None = None
    new_record_id: str | None
    incoming_text: str | None = None
    override: str | None = None
    override_at: datetime | None = None
    stamped_at: datetime | None = None
