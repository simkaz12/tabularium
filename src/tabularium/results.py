"""Result models returned by the engine and projected by the MCP tools.

The engine returns these models directly and ``server.py`` hands them to FastMCP, so their
field names, ``Field`` descriptions and class docstrings reach the tool output schemas. The
class docstrings that appear in the tools golden are frozen with it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from tabularium.domain import (
    AdmissionAction,
    DecisionLifecycle,
    Fact,
    HypomnemaId,
    InvalidationReason,
    Kind,
    Scope,
    SupersessionReason,
    UnsupersedeShape,
    Validity,
)
from tabularium.storage.types import TombstoneId

# ---------------------------------------------------------------------------
# HypomnemaRef
# ---------------------------------------------------------------------------


class HypomnemaRef(BaseModel):
    """Minimal reference to a saved record — id + type.

    Carried on ``SaveResult`` so the caller can reference the record without
    the engine returning the full domain object.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    kind: Kind


# ---------------------------------------------------------------------------
# JudgedVerdict / JudgedUnavailable / JudgedBlock
# ---------------------------------------------------------------------------


class JudgedVerdict(BaseModel):
    """One method's verdict on the bounce, ``trace`` stripped — the wire never carries what
    only the verdict log (A2) reads."""

    model_config = {"frozen": True}

    status: Literal["verdict"] = "verdict"
    pick: str
    odds: dict[str, float]
    confidence: float
    reason_class: SupersessionReason | None = None


class JudgedUnavailable(BaseModel):
    """One method's typed no-verdict on the bounce (D5) — never a bare ``None``."""

    model_config = {"frozen": True}

    status: Literal["unavailable"] = "unavailable"
    reason: Literal["absent", "timeout", "error"]
    detail: str | None = None


JudgedBlock = dict[str, Annotated[JudgedVerdict | JudgedUnavailable, Field(discriminator="status")]]


# ---------------------------------------------------------------------------
# BounceNeighbor
# ---------------------------------------------------------------------------


class BounceNeighbor(BaseModel):
    """A same-probe candidate above its arm's gray floor, other than the primary.

    Ranked below ``_BounceBase.candidate_id`` by similarity. Content is
    intentionally omitted — the primary candidate's content is all
    adjudication needs.

    ``similarity`` (P7 admission-order piece): ``None`` exactly when this
    neighbour was surfaced by the embedder-down BM25-only probe, so there is no cosine to report
    under an outage.
    ``bm25_rank`` is the mirror: set only on that same outage path (1-indexed
    BM25 rank), ``None`` on the ordinary live-vector probe.

    ``judged`` (A4 bounce-judged piece, decision D12): this neighbour's own ``relation``
    odds, same ``JudgedBlock`` shape as ``_BounceBase.judged`` — ``None`` exactly when this
    neighbour never got an Assessor call (no Assessor configured for the call site, or the
    neighbour sat below the population this dispatch actually judged). Populated by
    ``engine.facts``/``engine.decisions`` once the survivors' verdicts come back, never by
    ``_bounce_neighbors`` itself, which builds ``other_candidates`` before any Assessor call
    has run.
    """

    model_config = {"frozen": True}

    candidate_id: HypomnemaId
    similarity: float | None
    guard_flag: str | None = None
    bm25_rank: int | None = None
    judged: JudgedBlock | None = None


WithheldReason = Literal[
    "switch_off",
    "competing_candidate",
    "cross_scope_replacement",
    "elaborated_against_decision",
    "decision_against_fact",
    "cross_kind_restatement",
    "candidate_retired",
    "candidate_changed",
    "no_reason_class",
]
"""Fixed tokens naming why a bounce carrying a judged verdict took no automatic exit
(P8 assessor-automatic-exits piece, decision D11; ``competing_candidate`` added,
``upward_restatement`` removed, A4 bounce-judged piece decision D14) — tests pin these
tokens, never prose.

``switch_off`` — the judged relation's own automatic-exit switch is off.
``competing_candidate`` (A4, D14) — the top survivor's own verdict was switched on and
legal, but another survivor outside the upward-promotion ``moved`` set carries its own
``restatement`` or ``replacement`` verdict — the picture is not clean,
so nothing dispatches automatically even though the top candidate alone would have. On
the Decision arm this token also covers the arm's own upward-promotion cap (owner ruling
2026-09-24): two or more same-kind Decision survivors judged ``restatement`` upward never
auto-promote, even with the switch on, since the arm's single-id ``supersedes`` could
never later undo a successor shared by several predecessors.
``cross_scope_replacement`` — a replacement verdict against a candidate outside the
successor's own scope.
``elaborated_against_decision`` — a Fact replacement verdict of ``elaborated`` against a
Decision candidate; only ``correction``/``changed`` can invalidate a Decision.
``decision_against_fact`` — a Decision save judged against a Fact candidate, any relation.
``cross_kind_restatement`` — a restatement verdict between a Fact and a Decision.
``candidate_retired`` — the judged candidate is superseded, revoked, invalid, or otherwise
outside its kind's admission population.
``candidate_changed`` — the candidate's content or population state moved between
judgement (outside the lock) and dispatch (back inside it).
``no_reason_class`` — a replacement verdict carried no reason class, or carried ``moved``,
neither of which an automatic replacement may act on.
"""


# ---------------------------------------------------------------------------
# _BounceBase / BounceExit / FactBounce / DecisionBounce
# ---------------------------------------------------------------------------


class BounceExit(BaseModel):
    """One named exit from a bounce — the engine names the exits and ranks none of
    them; which one fits is a judgement about content that similarity cannot
    make. Declining every exit is legitimate only as a verdict — you read the
    candidate and concluded your atom was not worth writing. It is not a way to
    skip the judgement: a bounce left unresolved drops the atom silently.

    ``how``:  the exact call the agent would make to take this exit.
    ``when``: the content judgement that selects it.
    """

    model_config = {"frozen": True}

    action: AdmissionAction
    how: str
    when: str


class _BounceBase(BaseModel):
    """Structured bounce returned to the calling agent in place of a write.

    The bounce itself writes no Fact or Decision; the agent acts through one of the subclass's
    ``exits``, and an ignored bounce leaves the atom unsaved.

    ``candidate_id``, ``candidate_content`` and ``candidate_scope`` name the primary match.
    ``other_candidates`` holds the other records of the same probe that are above their own
    arm's gray floor, in descending similarity and without the primary; it is empty when there
    are none. A ``resolve`` targets one ``candidate_id``.

    ``verdict`` is ``near_dup``, ``decision_adjacent``, ``guard_trip`` or ``exact_key``.
    ``guard_flag`` names the discriminative guard that fired (``numeral``, ``version``,
    ``entity`` or ``negation``) and is ``None`` when none did; a guard only annotates the
    bounce. ``exact_key`` is the upward exact-key path: the incoming content's canonical key
    matched a live row in a named scope while the save targets ``global``. It has
    ``similarity=1.0``, no ``other_candidates`` and exactly one exit, a ``moved`` supersede.

    ``similarity`` is ``None`` when the embedder was down and the bounce came from the
    BM25-only probe.

    ``judged`` carries the Assessor ``relation`` verdict on the top candidate as
    ``judged["relation"]``, or ``JudgedUnavailable`` when the call failed, timed out or no
    Assessor was configured. On the live-vector path with no Assessor configured it is
    ``None``. On the embedder-outage path (``similarity=None``) the verdict is always
    requested, so ``judged`` is set, and it is evidence only: no automatic exit runs and
    ``withheld`` stays ``None``. Elsewhere ``withheld`` names why a judged verdict took no
    automatic exit (see ``WithheldReason``); it is ``None`` when no verdict applied, and can
    be set while this bounce's ``judged`` is ``None``, for ``candidate_changed``.

    ``bounce_id`` is set on every bounce and carried by every exit's ``how``. A follow-up
    ``save`` that omits it leaves the bounce's verdict rows ``unanswered`` instead of failing.
    """

    model_config = {"frozen": True}

    candidate_id: HypomnemaId
    candidate_content: str
    candidate_scope: str
    similarity: float | None
    guard_flag: str | None = None
    verdict: Literal["near_dup", "decision_adjacent", "guard_trip", "exact_key"]
    other_candidates: list[BounceNeighbor] = Field(default_factory=list)
    judged: JudgedBlock | None = None
    withheld: WithheldReason | None = None
    bounce_id: str


BounceExits = (
    tuple[BounceExit]
    | tuple[BounceExit, BounceExit]
    | tuple[BounceExit, BounceExit, BounceExit]
)


class FactBounce(_BounceBase):
    """A bounce for a Fact save — discriminated on the kind being
    SAVED, never on the candidate's kind. Two exits — ``reinforce`` and
    ``confirm_distinct`` — plus a third, ``supersede`` (2026-08-10
    fact-supersession brief,
    NOT YET owner-ratified), when the candidate is itself a Fact AND
    ``engine.exits.is_fact_supersedable(candidate)`` (currently: the
    candidate is valid — an already-superseded Fact is always invalid too).
    A Fact cannot supersede a Decision, so a Decision candidate never gets
    the third exit — same suppress-rather-than-refuse discipline
    ``DecisionBounce``'s ``promoted_from`` exit already follows (ADR 0022).

    ``exits`` also carries a ONE-exit variant (A4 bounce-judged piece, decision
    D3): the byte-identical upward exact-key path (``engine.exits.
    build_upward_exits(..., exact_key=True)``), whose ``moved`` supersede is the
    only exit offered — never ``confirm_distinct``, since a matching canonical
    key already rules out "these are genuinely distinct" (unchanged from the
    pre-A4 ``exact_key`` rationale, just narrower now that ``reinforce`` is
    gone from this ONE case too).
    """

    bounce_kind: Literal["fact"] = "fact"
    exits: BounceExits


class DecisionBounce(_BounceBase):
    """A bounce for a Decision save — discriminated on the kind
    being SAVED, never on the candidate's kind.  Three exits — ``reinforce``,
    ``confirm_distinct``, plus a third set by the candidate: ``supersede``
    when the candidate is a Decision (always legal — only in-force Decisions
    ever reach this bounce), ``promoted_from`` when it is a promotable Fact
    (``lifecycle`` in ``{active, hot, ember}``) — **except** two exits when
    the candidate is a Fact whose ``lifecycle`` is ``archived`` or
    ``shelved``: promotion is not legal against either, so the offer site
    (``engine.exits.build_exits``) suppresses the third exit rather than
    naming one the engine would then refuse (2026-08-07 fix, ADR 0022
    amendment).

    ``exits`` also carries the same ONE-exit upward variant ``FactBounce``
    documents (A4, D3) — the byte-identical upward exact-key path, ``moved``
    the only exit.
    """

    bounce_kind: Literal["decision"] = "decision"
    exits: BounceExits


# ---------------------------------------------------------------------------
# EscalationSuggestion
# ---------------------------------------------------------------------------


class EscalationExit(BaseModel):
    """One of the two rung-2 exits (SPEC §8.4). Always surfaced as a pair —
    the engine names both and ranks neither; which one fits is an owner
    judgement about content (SPEC §1.4's road-not-taken test), not something
    a recurrence count can answer.

    ``how``:  the exact call the owner would make to take this exit.
    ``when``: the content judgement that selects it.
    """

    model_config = {"frozen": True}

    action: Literal["codify", "foreground"]
    how: str
    when: str


class EscalationSuggestion(BaseModel):
    """Rung-2 escalation suggestion (SPEC §8.4), carried on the SAME
    ``SaveResult`` envelope a reinforced ember already returns — never a
    second mechanism.

    Fires on the **crossing**: an owner-asserted recurrence event that takes
    ``owner_recurrence_count`` from below ``threshold`` to at-or-above it —
    never on the standing value, so a Fact recurring past the threshold many
    times over only ever carries this once. ``set_hot(id, hot=False)``
    resets the ember's ``escalation_baseline`` on demotion so the ladder can
    re-arm.

    Suggested, never automatic (SPEC §8.4): the engine does not choose
    between ``exits`` — the owner takes one, or neither.
    """

    model_config = {"frozen": True}

    ember_id: HypomnemaId
    owner_recurrence_count: int
    threshold: int
    exits: tuple[EscalationExit, EscalationExit]  # always both, never one


# ---------------------------------------------------------------------------
# BounceResolution
# ---------------------------------------------------------------------------


class BounceResolution(BaseModel):
    """Agent adjudication of a bounce (SPEC §4.4 / §4.5 [A6]).

    Passed as ``save(resolve=BounceResolution(...))`` to re-run the gate with
    an explicit adjudication action:

    - ``confirm_distinct``: the agent judges the new content and the candidate
      are genuinely distinct — write the new record and register the pair in
      the distinct-pair registry (idempotent; first-judgment-wins).
    - ``reinforce``: the agent confirms the new content is redundant with the
      candidate — reinforce the named candidate directly without writing new.

    If the candidate is no longer the best match (buried, superseded, or
    overtaken), the engine returns a **fresh bounce** rather than blindly
    applying a stale adjudication [A6].

    ``bounce_id`` (A2 verdict-log piece, decision D14): optional, additive — copied
    from the bounce's own ``bounce_id`` field. Present, it stamps the verdict row this
    resolution names; absent (an old, copy-pasted ``how``), the resolve still runs
    exactly as before and the row stays ``unanswered`` (D15).
    """

    model_config = {"frozen": True}

    candidate_id: HypomnemaId
    action: Literal["confirm_distinct", "reinforce"]
    bounce_id: str | None = None


# ---------------------------------------------------------------------------
# SaveResult
# ---------------------------------------------------------------------------


class SaveResult(BaseModel):
    """Result of a ``save`` / ``ember`` call (SPEC §9.1).

    Outcomes:
      ``saved``        — novel content written as a new record.
      ``reinforced``   — near-duplicate matched an active/hot Fact; event
                         appended, projections updated; no new record.
      ``revived``      — re-assertion of an archived Fact; in-place
                         ``archived→active`` transition + reinforcement.
      ``reconciled``   — new content written past an invalid match; the new
                         record carries ``reconciles=<invalid_id>`` [A6].
      ``bounced``      — gate returned a structured bounce; nothing written.
      ``already_saved``— admission step 1 (P7 admission-order piece): a
                         raw byte-identical live row already exists in the
                         same scope, either arm. Idempotent ack; never
                         writes a NEW record, never appends a reinforcement
                         event — a replay is not new evidence. Fact-arm
                         exception (SPEC §4.5 [A6] amendment): if the
                         matching prior record is currently ``archived``,
                         this outcome additionally revives it in place
                         (``archived→active``) before returning — still not
                         a new record, but not a pure no-op read either.

    A save/ember never fails because the embedder is down (P6 vector-less-
    rows piece, D1 — the old typed-failure outcome is gone): under an
    outage, ``outcome`` lands as ``saved``/``reconciled`` with
    ``embedded=False`` — the row exists with ``vector=None``, readable
    through the text leg and grounding, invisible to the vector probe until
    backfilled — UNLESS the caller supplied an explicit ``resolve`` naming
    a candidate: the similarity probe needs a vector, but a
    named candidate does not, so ``reinforce``/``confirm_distinct`` are
    still honored under an outage and can still land ``outcome="reinforced"``
    or ``"revived"`` (no fresh row, ``embedded`` stays its default ``True``)
    alongside the vector-less ``"saved"``/``"reconciled"`` case.

    ``hypomnema``: set when ``outcome`` is one of ``saved | reconciled | already_saved``
      (``saved``/``reconciled`` — a new record was just written;
      ``already_saved`` — the id of the record a PRIOR call already wrote).
      ``None`` for reinforced/revived/bounced
      (reinforced/revived carry the candidate id in the ``reinforced_id`` field).
    ``bounce``: set when ``outcome="bounced"``; absent otherwise.
    ``reason``: human-readable explanation for a non-``saved`` outcome.
    ``reinforced_id``: the id of the record that was reinforced or revived.
      ``None`` for saved/reconciled/bounced/already_saved.
    ``embedded``: ``False`` exactly when this call's own write landed
      without a vector (embedder unavailable at write time); ``True``
      otherwise, including on every outcome that never produced a fresh
      write of its own (``bounced``/``already_saved`` only ever run
      against a live vector — the outage branch bypasses the similarity
      probe entirely; ``reinforced``/``revived`` normally do too, but can
      also land under an outage via a caller-named ``resolve`` (see
      ``engine.facts``/``engine.decisions``).
    ``escalation_suggestion`` (SPEC §8.4 rung 2, ADR 0020 amendment): set
      when a ``reinforced`` outcome reinforces an ember AND the reinforcing
      event was owner-asserted (``ember(asserted_by="owner")``) AND it
      crossed ``escalation_threshold``. ``None`` on every other outcome, and
      ``None`` on an agent-asserted or below-threshold ember reinforcement.
    """

    model_config = {"frozen": True}

    outcome: Literal[
        "saved", "reinforced", "revived", "reconciled", "bounced", "already_saved"
    ]
    hypomnema: HypomnemaRef | None = None
    bounce: Annotated[FactBounce | DecisionBounce, Field(discriminator="bounce_kind")] | None = (
        None
    )
    reason: str | None = None
    reinforced_id: HypomnemaId | None = None
    embedded: bool = True
    escalation_suggestion: EscalationSuggestion | None = None


# ---------------------------------------------------------------------------
# RecalledHit
# ---------------------------------------------------------------------------


class RecalledHit(BaseModel):
    """A single hit from ``recall`` (SPEC §9.1).

    ``scope`` carries the origin label per SPEC §2 / §0015 — a hit from
    ``global`` in a scoped recall is labeled ``"global"``.

    ``purpose`` is present but may be ``None`` when the record was saved
    without a purpose value.

    ``superseded_by`` (2026-08-10 fact-supersession brief): the forward
    pointer, when the hit is a superseded Fact.
    Non-``None`` only ever on an ``invalid`` record (``domain.Fact``'s own
    consistency validator), so it is visible only under
    ``include_invalid=True`` — not by any special-casing here, but as a
    direct consequence of the default recall filter already excluding
    invalid records. A caller who explicitly asks for invalid records is
    exactly the caller who needs to know where the replacement is.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    kind: Kind
    content: str
    lifecycle: str
    validity: Validity
    superseded_by: HypomnemaId | None = None
    score: float = Field(ge=0.0, description="Post-boost composite score")
    scope: str  # origin scope label
    purpose: str | None = None


# ---------------------------------------------------------------------------
# RecallResult
# ---------------------------------------------------------------------------


class RecallResult(BaseModel):
    """Result of a ``recall`` call (SPEC §9.1).

    ``degraded``: True when the embedder was unavailable and only the BM25 leg
    ran.  The caller should treat results as lower-confidence.

    ``reason``: non-None when ``degraded=True``; describes the outage.

    ``legs_used``: which search legs contributed.  Normal = ``["vector", "text"]``;
    degraded = ``["text"]``.  Empty when the store has no matches.
    """

    model_config = {"frozen": True}

    hits: list[RecalledHit]
    degraded: bool = False
    reason: str | None = None
    legs_used: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DecisionLink
# ---------------------------------------------------------------------------


class DecisionLink(BaseModel):
    """Lean triage link returned by ``decisions_query`` (SPEC §9.1 / [A11]).

    Intentionally excludes live stats (recall_count, last_referenced_at, tip)
    and the render — those live in ``DecisionView`` only.  ``uri`` follows the
    ``decision://{id}`` scheme that the resource will serve.

    ``validity`` mirrors the row and ``DecisionView.validity`` (P9 invalid-
    Decision read contract, decision D5): ``"valid"`` unless the caller widened
    with ``decisions_query(include_invalid=True)``.
    """

    model_config = {"frozen": True}

    uri: str                                      # "decision://<id>"
    id: HypomnemaId
    title: str                                    # derived first-sentence title
    lifecycle: DecisionLifecycle
    validity: Validity
    superseded_by: HypomnemaId | None = None
    scope: str                                    # origin scope label (SPEC §2 / §0015)


# ---------------------------------------------------------------------------
# RevalidationRecord
# ---------------------------------------------------------------------------


class RevalidationRecord(BaseModel):
    """One past revalidation of a Decision (P9 invalid-Decision read contract,
    decision D8) — a results-layer model built from the storage
    ``RevalidationEvent`` (``storage.types.RevalidationEvent``), never the
    storage type exposed on the API.

    ``withdrawn_cause``: the Fact id whose invalidation this revalidation
    withdrew. ``withdrawn_reason``: that invalidation's own reason class.
    ``reason``: the caller's own audit-trail justification for revalidating,
    distinct from ``withdrawn_reason``.
    """

    model_config = {"frozen": True}

    withdrawn_cause: HypomnemaId
    withdrawn_reason: InvalidationReason
    reason: str
    at: datetime


# ---------------------------------------------------------------------------
# DecisionView
# ---------------------------------------------------------------------------


class DecisionView(BaseModel):
    """Full view returned by the ``decision`` tool (SPEC §9.1).

    Includes live stats (recall_count, last_referenced_at) and tip (derived
    live end of the chain).  ``rendered`` is byte-identical to what the
    ``decision://{id}`` resource will serve — one render function, two exposures.

    ``supersedes``: the id of the immediate predecessor (chain[-1].id), or
    ``None`` if this decision has no backward chain.

    ``tip``: the live end of the chain, derived by following ``superseded_by``
    pointers.  ``None`` when this decision IS the live end (i.e. self is the
    tip); the id of the live end otherwise.

    ``promoted_from``: list of ember ids that were promoted into this decision,
    or ``None`` if empty.

    ``invalidated_by``/``invalidation_reason``: the Fact id and reason class
    of the current invalidation (P8), or both ``None`` when valid or never
    invalidated. ``revalidations``: every past revalidation, oldest first (P9
    invalid-Decision read contract, decision D8) — empty when none happened.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    content: str
    lifecycle: DecisionLifecycle
    validity: Validity
    supersedes: HypomnemaId | None              # chain[-1].id
    superseded_by: HypomnemaId | None
    tip: HypomnemaId | None                     # None = self is the live end
    promoted_from: list[HypomnemaId] | None
    invalidated_by: str | None
    invalidation_reason: InvalidationReason | None
    revalidations: list[RevalidationRecord]
    recall_count: int
    last_referenced_at: datetime | None
    purpose: str | None = None
    created_at: datetime
    rendered: str                               # byte-identical to the resource


# ---------------------------------------------------------------------------
# ReviseResult
# ---------------------------------------------------------------------------


class ReviseResult(BaseModel):
    """Result of a ``set_validity`` or ``revoke`` call (SPEC §9.1).

    Shared by both tools: each flips one lifecycle/validity bit and reports
    the after-state.

    ``outcome``:
      ``"unchanged"`` — idempotent ack (SPEC §9.2 monotonic re-op): a
        same-value ``set_validity`` call, or a re-``revoke`` on an
        already-``revoked`` Decision. No adapter write happened.
      ``"revised"``   — state actually moved: ``set_validity`` flipped the
        bit, or ``revoke`` transitioned ``in_force → revoked``.

    ``lifecycle``: the record's lifecycle after the call (unchanged by
      ``set_validity``; ``"revoked"`` after a successful ``revoke``).
    ``validity``: the record's validity after the call (flipped by
      ``set_validity``; unchanged by ``revoke`` — validity and lifecycle are
      orthogonal axes, SPEC §1.2).
    ``reason``: echoes the caller-supplied audit-trail reason (SPEC §9.1:
      "reason required on all three" disposition tools).

    ``successor_validity`` (2026-08-11 fact-supersession-undo):
    RETIRED, no live writer as of the P4 two-arm unsupersede
    piece — the Fact-only undo verb this field was built for now returns
    ``UnsupersedeResult`` instead of ``ReviseResult``,
    and its arm-general successor never populates this field. Kept on the
    model (always ``None``) rather than removed — deleting a field is a
    wider wire-shape change than this piece's scope, and every current
    ``ReviseResult`` producer (``set_validity``/``revoke``/``set_shelved``)
    already has no successor concept to report here.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    outcome: Literal["revised", "unchanged"]
    lifecycle: str
    validity: Validity
    reason: str
    successor_validity: Validity | None = None


# ---------------------------------------------------------------------------
# SupersedeResult
# ---------------------------------------------------------------------------


class SupersedeResult(BaseModel):
    """Result of ``supersede`` — draws edges from one or more existing
    predecessors to an existing successor, all in one records transaction.

    ``edges``: one ``(predecessor_id, edge_id)`` pair per predecessor, in
    the caller's own order — the edge id the engine minted before the
    write, the same id P4's undo keys on.
    """

    model_config = {"frozen": True}

    successor_id: HypomnemaId
    edges: tuple[tuple[HypomnemaId, str], ...]


# ---------------------------------------------------------------------------
# UnsupersedeResult
# ---------------------------------------------------------------------------


class UnsupersedeResult(BaseModel):
    """Result of ``unsupersede(id, reason)`` — undoes a supersession on
    either arm and names which shape ran, replacing the
    ``ReviseResult`` shape its Fact-only predecessor
    used to return.

    ``shape``: ``"fact_both_live"`` — the pointer clears and validity
    restores on the predecessor; the successor is untouched, both rows stay
    live. ``"decision_restored"`` — the predecessor returns to
    ``in_force``; the successor is deleted, its tombstone carrying
    ``restores=predecessor_id``.

    ``withdrawn_reason``/``withdrawn_note`` echo the supersession class/note
    being undone — the SAME values the undo event
    persists, even though they no longer live on the restored row itself.
    ``withdrawn_reason`` is ``None`` for an edge drawn before the reason
    class column existed (pre-P3) — the withdrawn class is honestly
    unknown, never fabricated.
    """

    model_config = {"frozen": True}

    predecessor_id: HypomnemaId
    successor_id: HypomnemaId
    edge_id: str
    arm: Kind
    shape: UnsupersedeShape
    withdrawn_reason: SupersessionReason | None
    withdrawn_note: str | None
    at: datetime


# ---------------------------------------------------------------------------
# BuryResult
# ---------------------------------------------------------------------------


class BuryResult(BaseModel):
    """Result of a ``bury`` call (SPEC §9.1 / §9.2).

    ``outcome``:
      ``"buried"``         — the record was live and is now gone; a fresh
        tombstone was written.
      ``"already_buried"`` — idempotent re-op (SPEC §9.2 monotonic re-op):
        the record was already buried; ``tombstone_id`` is the id of the
        ORIGINAL tombstone, not a new one — ``bury`` writes exactly one
        tombstone per id, ever.

    ``buried_id``: the id that was (or already was) buried.
    ``tombstone_id``: the surviving trace of the buried record (SPEC §1.3).
    ``reason``: echoes the caller-supplied audit-trail reason.
    """

    model_config = {"frozen": True}

    buried_id: HypomnemaId
    tombstone_id: TombstoneId
    outcome: Literal["buried", "already_buried"]
    reason: str


# ---------------------------------------------------------------------------
# FeedbackResult
# ---------------------------------------------------------------------------


class FeedbackResult(BaseModel):
    """Result of a ``feedback`` call (SPEC §9.1 / §8.1 [A8] / §8.2).

    ``feedback_score``: the Fact's ``EWMA(alpha=0.3, init 0.0)`` feedback
    projection AFTER this event is applied (SPEC §8.2) — the same value
    persisted via ``update_projections``, echoed back so the caller does not
    need a second read to see the effect.

    ``importance``: the Fact's full §8.1 composite, recomputed at the same
    instant this event was applied — computed-lazy, never stored (SPEC
    §8.1), so this is a snapshot, not a live-tracking value.

    ``reason``: echoes the caller-supplied audit-trail reason, or ``None``
    when omitted — same convention as ``ReviseResult``/``BuryResult``.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    feedback_score: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0)
    reason: str | None = None


# ---------------------------------------------------------------------------
# PackedTier / ContextPayload
# ---------------------------------------------------------------------------


class PackedTier(BaseModel):
    """One tier after packing (SPEC §7 mechanics: greedy whole-entry fill,
    waterfall spill, never truncate — see ``engine.context.pack`` for the
    full spill mechanics).

    ``entries``: the whole, unmodified records admitted into this tier's
    budget. ``budget_used``: total counter cost of the admitted entries.
    ``skipped``: count of entries that did not fit and were omitted whole
    (never partially included).
    """

    model_config = {"frozen": True}

    name: str
    entries: list[Fact] = Field(default_factory=list)
    budget_used: int = 0
    skipped: int = 0


class CleanerContextSummary(BaseModel):
    """A short, BOUNDED rendering of the last Cleaner pass for one scope
    (2026-08-10 cleaner plan) — carried on ``ContextPayload.cleaner``
    so a session-start caller sees "what happened last time" without a
    second round trip.

    Deliberately small: this competes with grounding for the session-start
    budget, and the tiers above are what the payload is actually for. The
    full report (every judge pair, every id, every band count) stays in the
    ``report`` column (``engine.report.CleanerReport``) for anyone who asks
    — ``recall``/``decisions_query`` are the on-demand path, this is not it.

    ``owner_attention`` is capped at ``rubric.parameters.report_render_cap``
    entries (``engine.report.render_context_summary``) — a pass with
    hundreds of unclear/contradicted/failed items never turns into hundreds
    of lines here; an overflow marker is appended instead of the remainder.

    ``outcome="errored"`` (2026-08-10 cleaner plan follow-up) is a
    THIRD value: the pass crashed before ``engine.clean.clean`` ever reached
    its own ``"completed"``/``"blocked"`` write — distinct from both of
    those (an axis genuinely ran) and from ``ContextPayload.cleaner`` being
    ``None`` entirely (the scope has never been cleaned at all — see
    ``engine.report.load_context_summary``).

    ``outcome`` widened to the full five-value set (cleaner-telemetry change):
    ``"initialized"``/``"empty_pool"`` join ``"completed"``/``"blocked"``/``"errored"`` because
    this now renders the newest row of the append-only pass history whatever outcome it carries,
    rather than the (pre-existing) ``CleanerState.report``, which the `initialized`/
    `empty_pool` outcomes never wrote to at all — a pass that judged nothing used to render as
    ``None`` here; now it renders as itself, distinguishably from every other outcome, so a run of
    consecutive empty passes (the trigger silently stopped firing) is visible at session start.

    ``blocked_streak`` counts the consecutive newest non-dry-run pass rows whose outcome is
    ``"blocked"`` or ``"errored"``; ``blocked_since`` is the ``pass_at`` of the oldest row in that
    streak. ``0`` and ``None`` when the newest row is neither.
    """

    model_config = {"frozen": True}

    scope: Scope
    pass_at: datetime
    outcome: Literal["initialized", "empty_pool", "completed", "blocked", "errored"]
    summary_line: str
    owner_attention: tuple[str, ...] = ()
    blocked_streak: int = 0
    blocked_since: datetime | None = None


class ContextPayload(BaseModel):
    """Result of ``engine.context.pack`` (SPEC §7).

    ``budget_used`` and ``skipped`` are the totals across all tiers (each
    also observable per-tier on ``PackedTier``) — the packer's headline
    observability contract (the acceptance criteria).

    ``cleaner``: the requested scope's last Cleaner pass, rendered small
    (2026-08-10 cleaner plan) — ``None`` when the scope has never
    had a completed or blocked pass (never judged, or only ever
    initialized/empty). Additive: a scope with no Cleaner state gets
    ``cleaner=None`` and every other field is exactly what ``pack`` already
    produced — selection, ranking and the token-budget accounting in
    ``tiers``/``budget_used``/``skipped`` are entirely unaffected by this
    field's presence.
    """

    model_config = {"frozen": True}

    tiers: list[PackedTier] = Field(default_factory=list)
    budget_used: int = 0
    skipped: int = 0
    cleaner: CleanerContextSummary | None = None


# ---------------------------------------------------------------------------
# ProjectionDrift / ProjectionRebuildReport
# ---------------------------------------------------------------------------


class ProjectionDrift(BaseModel):
    """One record whose stored projections disagree with the event-log recomputation.

    Each populated field is a ``(stored, recomputed)`` pair; a field that matches is ``None``.
    ``feedback_score`` and ``decay_due_at`` are always ``None`` for a Decision.
    """

    model_config = {"frozen": True}

    id: HypomnemaId
    kind: Kind
    recall_count: tuple[int, int] | None = None
    last_referenced_at: tuple[datetime | None, datetime | None] | None = None
    feedback_score: tuple[float, float] | None = None
    decay_due_at: tuple[datetime | None, datetime | None] | None = None


class ProjectionRebuildReport(BaseModel):
    """Result of ``rebuild_projections``.

    ``mode="report"`` writes nothing and ``repaired`` is ``0``. ``mode="repair"`` also writes
    each drifted record's recomputed values through ``update_projections``; ``repaired`` counts
    the writes that landed. A record buried between the scan and its repair write stays in
    ``drifted`` and is not counted in ``repaired``.
    """

    model_config = {"frozen": True}

    mode: Literal["report", "repair"]
    scanned: int
    drifted: list[ProjectionDrift] = Field(default_factory=list)
    repaired: int = 0


# ---------------------------------------------------------------------------
# Doctor report
# ---------------------------------------------------------------------------


class FacetCoverageReport(BaseModel):
    """``tabularium doctor`` facet coverage: records carrying a ``purpose`` and an
    ``audience_hint``, and the ``purpose`` distribution.

    ``scanned`` is the denominator for every ratio: the live Facts and Decisions from
    ``scan_records(lifecycle_in=None)``, which excludes buried and includes archived.
    ``purpose_counts`` starts from every registered facet at zero.
    """

    model_config = {"frozen": True}

    scanned: int
    fact_count: int
    decision_count: int
    with_purpose: int
    purpose_counts: dict[str, int] = Field(default_factory=dict)
    with_audience_hint: int


class DeadVocabularyReport(BaseModel):
    """``tabularium doctor`` dead vocabulary: registered facet values.

    ``dead`` lists the registered values no dispatch path reads; the doctor currently reports
    every registered value as dead. ``unused_in_data`` lists the registered values with no
    live record in the same scan as ``FacetCoverageReport``. The two are independent.
    """

    model_config = {"frozen": True}

    registered: list[str] = Field(default_factory=list)
    dead: list[str] = Field(default_factory=list)
    unused_in_data: list[str] = Field(default_factory=list)


class AwaitingEmbeddingReport(BaseModel):
    """``tabularium doctor`` rows awaiting embedding.

    ``count`` is ``len(adapter.backfill_targets())``, the list the backfill pass consumes: rows
    with no vector, or stamped with a stale ``embedding_model``, buried rows excluded.
    """

    model_config = {"frozen": True}

    count: int = Field(ge=0)


class DoctorReport(BaseModel):
    """Full ``tabularium doctor`` report. Read-only output; nothing here feeds a rejection or a
    rewrite.
    """

    model_config = {"frozen": True}

    facet_coverage: FacetCoverageReport
    dead_vocabulary: DeadVocabularyReport
    awaiting_embedding: AwaitingEmbeddingReport
    projection_drift: ProjectionRebuildReport
