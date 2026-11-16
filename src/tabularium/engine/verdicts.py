"""The verdict log: row building, stamping and the startup reconcile for Assessor verdicts.

``engine/facts.py`` and ``engine/decisions.py`` call into this leaf, and ``core.py`` delegates to
it. An Assessor verdict for a probe survivor, or a ``timeout``/``error`` answer, becomes one
``VerdictRow``; nothing here deletes rows. Two disjoint write paths land a row on the save side:

- Bounce path: every survivor's verdict on a bounce that returns to a caller is written
  ``status="unanswered"`` under the bounce's own ``bounce_id``. A caller replaying the bounce's own
  ``how`` moves the named candidate's row ``unanswered -> pending -> acted``/``failed`` and stamps
  the other rows ``sibling_acted``.
- Automatic-exit path: the top survivor's verdict, when a switch is on and the pairing is legal, is
  written ``pending`` (no ``bounce_id``, since no bounce is shown) inside the lock hold the apply
  function already takes, mutated, then stamped ``acted``/``failed``.

Crash order is pending, then mutate, then stamp. A ``pending`` row left by a crash is resolved by
``reconcile_pending_verdicts`` at startup.

``matched``/``override``: a row's ``pick`` is what the Assessor recommended, and ``legal_exit``
called with every switch forced on gives the exit that pick would dispatch to. A caller or automatic
exit that took a different exit is ``matched=False``. A caller mismatch also stamps
``override="caller_against_pick"`` when the row moves to ``pending``.

``VerdictRow`` keeps ``method``, ``pick``, ``reason_class`` and ``action`` as free strings, so
decoding a stored row never gets stricter. The ``_parse_*`` functions narrow them to a ``Literal``
at the point of use.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Literal, TypeGuard, assert_never, get_args
from uuid import uuid4

from tabularium.config import EngineConfig
from tabularium.domain import (
    AdmissionAction,
    Decision,
    Fact,
    HypomnemaId,
    Kind,
    SupersessionReason,
    VerdictActor,
)
from tabularium.engine.admission import (
    AutoExitSwitches,
    Direction,
    Invalidate,
    Promote,
    Reinforce,
    Supersede,
    Withheld,
    WriteNew,
    legal_exit,
)
from tabularium.engine.assessor import (
    ContradictionVerdict,
    Relation,
    RelationVerdict,
    Trace,
    Unavailable,
    Verdict,
)
from tabularium.storage.audit import VerdictStatus
from tabularium.storage.types import StorageAdapter, VerdictRow

_log = logging.getLogger("tabularium")

CallerName = Literal["save", "cleaner_prepass", "cleaner_split"]

#: The Cleaner's action vocabulary. Free strings at the storage seam (``VerdictRow.action``); this
#: Literal types the caller side only.
CleanerAction = Literal[
    "panel_keep",
    "panel_shelve",
    "panel_contested",
    "panel_unclear",
    "skipped_keep",
    "split_keep",
    "split_annotated",
    "split_shelve",
    "split_supersede",
    "split_coexist",
    "reason_by_assessor",
    "contested",
    "global_reported",
]

#: The ``VerdictRow.action`` values the startup reconcile has a read path for. Free strings at the
#: storage seam; ``_parse_pending_action`` narrows them.
PendingAction = Literal[
    "reinforce",
    "invalidate",
    "supersede",
    "split_supersede",
    "reason_by_assessor",
    "write_new",
    "confirm_distinct",
    "promoted_from",
    "split_shelve",
    "panel_shelve",
    "split_keep",
    "split_coexist",
]

#: The subset of ``CleanerAction`` whose row follows the pending -> write -> stamp crash order.
#: Every other action is appended already resolved.
_CLEANER_WRITE_ACTIONS: frozenset[str] = frozenset(
    {"split_keep", "split_shelve", "split_coexist", "split_supersede", "reason_by_assessor"}
)

_ALL_SWITCHES_ON = AutoExitSwitches(
    restatement=True, replacement=True, follow_up=True, distinct=True
)

_IDEAL_ACTION_BY_EXIT: dict[type, str] = {
    Reinforce: "reinforce",
    WriteNew: "confirm_distinct",
    Supersede: "supersede",
    Promote: "supersede",
}

WritePending = Callable[[Sequence[VerdictRow]], Awaitable[None]]

# ---------------------------------------------------------------------------
# Parsing the free strings a stored row carries
# ---------------------------------------------------------------------------


def _is_pending_action(value: str | None) -> TypeGuard[PendingAction]:
    return value in get_args(PendingAction)


def _parse_pending_action(row: VerdictRow) -> PendingAction:
    """The row's ``action`` as a ``PendingAction``. Raises ``RuntimeError`` for any other value,
    ``None`` included, because the reconcile has no read path for it.
    """
    action = row.action
    if _is_pending_action(action):
        return action
    raise RuntimeError(
        f"reconcile_pending_verdicts: no read path for pending row action={row.action!r} "
        f"id={row.id!r} — stop and report rather than defaulting"
    )


def _is_relation(value: str) -> TypeGuard[Relation]:
    return value in get_args(Relation)


def _parse_relation(pick: str) -> Relation:
    """``pick`` as a ``Relation``. ``legal_exit`` has no branch for a label outside the four and
    answers it as ``distinct``, so such a pick parses to ``"distinct"`` and never raises.
    """
    if _is_relation(pick):
        return pick
    return "distinct"


def _is_supersession_reason(value: str) -> TypeGuard[SupersessionReason]:
    return value in get_args(SupersessionReason)


def _parse_supersession_reason(reason_class: str | None) -> SupersessionReason | None:
    """``reason_class`` as a ``SupersessionReason``. ``legal_exit`` tells ``None``, ``"moved"``,
    ``"elaborated"`` and ``"correction"`` apart and answers every other label as it answers
    ``"changed"``, so such a label parses to ``"changed"`` and never raises.
    """
    if reason_class is None:
        return None
    if _is_supersession_reason(reason_class):
        return reason_class
    return "changed"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RowFrame:
    """The columns of a new row that do not come from the verdict: identity, status and outcome."""

    row_id: str
    created_at: datetime
    caller: CallerName
    candidate_id: str
    status: VerdictStatus
    bounce_id: str | None = None
    probe_rank: int | None = None
    similarity: float | None = None
    new_record_id: str | None = None
    action: str | None = None
    actor: VerdictActor | None = None
    matched: bool | None = None
    effect_ref: str | None = None
    incoming_text: str | None = None
    stamped_at: datetime | None = None


def _answered_row(
    verdict: Verdict | ContradictionVerdict, *, reason_class: str | None, frame: _RowFrame
) -> VerdictRow:
    return VerdictRow(
        id=frame.row_id,
        created_at=frame.created_at,
        caller=frame.caller,
        method=verdict.method,
        bounce_id=frame.bounce_id,
        candidate_id=frame.candidate_id,
        probe_rank=frame.probe_rank,
        similarity=frame.similarity,
        pick=verdict.pick,
        odds_json=json.dumps(verdict.odds),
        confidence=verdict.confidence,
        reason_class=reason_class,
        model=verdict.trace.model,
        wording_hash=verdict.trace.wording_hash,
        latency_ms=verdict.trace.latency_ms,
        input_tokens=verdict.trace.input_tokens,
        output_tokens=verdict.trace.output_tokens,
        request_id=verdict.trace.request_id,
        status=frame.status,
        action=frame.action,
        actor=frame.actor,
        matched=frame.matched,
        effect_ref=frame.effect_ref,
        new_record_id=frame.new_record_id,
        incoming_text=frame.incoming_text,
        stamped_at=frame.stamped_at,
    )


def _unavailable_row(verdict: Unavailable, frame: _RowFrame) -> VerdictRow:
    return VerdictRow(
        id=frame.row_id,
        created_at=frame.created_at,
        caller=frame.caller,
        method=verdict.method,
        bounce_id=frame.bounce_id,
        candidate_id=frame.candidate_id,
        probe_rank=frame.probe_rank,
        similarity=frame.similarity,
        unavailable_reason=verdict.reason,
        unavailable_detail=verdict.detail,
        latency_ms=verdict.latency_ms,
        status=frame.status,
        action=frame.action,
        actor=frame.actor,
        matched=frame.matched,
        effect_ref=frame.effect_ref,
        new_record_id=frame.new_record_id,
        incoming_text=frame.incoming_text,
        stamped_at=frame.stamped_at,
    )


def _relation_row(verdict: RelationVerdict | Unavailable, frame: _RowFrame) -> VerdictRow:
    if isinstance(verdict, Unavailable):
        return _unavailable_row(verdict, frame)
    return _answered_row(verdict, reason_class=verdict.reason_class, frame=frame)


def build_unanswered_rows(
    *,
    caller: CallerName = "save",
    bounce_id: str | None,
    candidate_ids: Sequence[HypomnemaId],
    similarities: Sequence[float | None],
    verdicts: Sequence[RelationVerdict | Unavailable],
    now: datetime,
    start_rank: int = 0,
) -> list[VerdictRow]:
    """One row per probe survivor, ``status="unanswered"``.

    ``bounce_id`` is the bounce this batch returns under, or ``None`` for the secondary survivors of
    an automatic exit's own verdict batch, which are never shown to a caller. ``start_rank`` is the
    original probe position of ``candidate_ids[0]``; a tail slice such as ``candidate_ids[1:]`` must
    pass ``start_rank=1``, or its rows share ``probe_rank=0`` with the candidate the exit acted on.
    """
    return [
        _relation_row(
            verdict,
            _RowFrame(
                row_id=str(uuid4()),
                created_at=now,
                caller=caller,
                candidate_id=str(candidate_id),
                status="unanswered",
                bounce_id=bounce_id,
                probe_rank=probe_rank,
                similarity=similarity,
            ),
        )
        for probe_rank, (candidate_id, similarity, verdict) in enumerate(
            zip(candidate_ids, similarities, verdicts, strict=True), start=start_rank
        )
    ]


def build_pending_row(
    *,
    caller: CallerName = "save",
    candidate_id: HypomnemaId,
    similarity: float | None,
    verdict: RelationVerdict,
    action: Literal["reinforce", "write_new", "supersede", "invalidate"],
    effect_ref: str | None,
    new_record_id: str | None,
    now: datetime,
    incoming_text: str | None = None,
) -> VerdictRow:
    """The top survivor's row for an automatic exit: ``status="pending"`` and no ``bounce_id``.

    ``action``, ``actor`` and ``matched`` are already set (``matched`` is always true, since the
    engine acts on the verdict's own pick), so a crash between this write and the stamp still tells
    the reconcile which effect to look for. ``incoming_text`` is the discarded content and is set
    only on the automatic ``reinforce`` exit; every other automatic exit writes a new record,
    carried in ``new_record_id``.
    """
    return _relation_row(
        verdict,
        _RowFrame(
            row_id=str(uuid4()),
            created_at=now,
            caller=caller,
            candidate_id=str(candidate_id),
            status="pending",
            probe_rank=0,
            similarity=similarity,
            new_record_id=new_record_id,
            action=action,
            actor="engine",
            matched=True,
            effect_ref=effect_ref,
            incoming_text=incoming_text,
        ),
    )


# ---------------------------------------------------------------------------
# Cleaner row builders: rows for the pre-pass and split paths. They carry no bounce, probe rank or
# similarity, because the Cleaner asks the Assessor about one record or one pair.
# ---------------------------------------------------------------------------


def _cleaner_row(
    verdict: Verdict | ContradictionVerdict | Unavailable, frame: _RowFrame
) -> VerdictRow:
    if isinstance(verdict, Unavailable):
        return _unavailable_row(verdict, frame)
    reason_class = verdict.reason_class if isinstance(verdict, ContradictionVerdict) else None
    return _answered_row(verdict, reason_class=reason_class, frame=frame)


def build_cleaner_resolved_row(
    *,
    caller: CallerName,
    candidate_id: HypomnemaId,
    new_record_id: HypomnemaId | None = None,
    verdict: Verdict | ContradictionVerdict | Unavailable,
    action: CleanerAction | None,
    actor: VerdictActor | None,
    matched: bool | None,
    now: datetime,
) -> VerdictRow:
    """A row for a Cleaner outcome the Assessor's vote did not cause (``panel_*``, ``skipped_keep``,
    ``contested``, ``global_reported``).

    The row is built ``acted`` when a real verdict landed and ``unanswered`` when the Assessor was
    ``Unavailable``. The builder does not produce a ``pending`` row; ``resolve_panel_shelve_write``
    sets that status itself for ``panel_shelve``.
    """
    status: Literal["unanswered", "acted"] = (
        "unanswered" if isinstance(verdict, Unavailable) else "acted"
    )
    return _cleaner_row(
        verdict,
        _RowFrame(
            row_id=str(uuid4()),
            created_at=now,
            caller=caller,
            candidate_id=str(candidate_id),
            status=status,
            new_record_id=str(new_record_id) if new_record_id is not None else None,
            action=action,
            actor=actor,
            matched=matched,
            stamped_at=now if status == "acted" else None,
        ),
    )


def build_cleaner_pending_row(
    *,
    caller: CallerName,
    candidate_id: HypomnemaId,
    new_record_id: HypomnemaId | None,
    verdict: Verdict | ContradictionVerdict,
    action: CleanerAction,
    effect_ref: str | None,
    now: datetime,
) -> VerdictRow:
    """The ``pending`` row for a write the Assessor's vote caused (``split_keep``, ``split_shelve``,
    ``split_coexist``, ``split_supersede``, ``reason_by_assessor``).

    The intended outcome is set on the row at this write (``action``, ``actor="engine"``,
    ``matched=True``, ``effect_ref``), so a crash before the final stamp leaves the reconcile a
    complete row to flip. ``action`` must be in ``_CLEANER_WRITE_ACTIONS``.
    """
    assert action in _CLEANER_WRITE_ACTIONS, (
        f"build_cleaner_pending_row: action={action!r} is not an Assessor-caused write"
    )
    return _cleaner_row(
        verdict,
        _RowFrame(
            row_id=str(uuid4()),
            created_at=now,
            caller=caller,
            candidate_id=str(candidate_id),
            status="pending",
            new_record_id=str(new_record_id) if new_record_id is not None else None,
            action=action,
            actor="engine",
            matched=True,
            effect_ref=effect_ref,
        ),
    )


def newest_pick_bearing_row(
    rows: Sequence[VerdictRow], *, caller: CallerName
) -> VerdictRow | None:
    """The newest row for ``caller`` that carries a pick. ``rows`` must be newest first, as
    ``StorageAdapter.verdicts_for_record`` returns them.
    """
    for row in rows:
        if row.caller == caller and row.pick is not None:
            return row
    return None


def verdict_from_cleaner_row(row: VerdictRow) -> Verdict | Unavailable:
    """Rebuild the ``Verdict``, or the ``Unavailable`` when the row has no pick, that a Cleaner row
    was built from. This inverts ``_cleaner_row``.

    The revisit axis reuses an existing ``cleaner_prepass`` row's original pick this way instead of
    calling the Assessor again. ``method`` is taken from the row. ``reason_class`` is not restored,
    so a row built from a ``ContradictionVerdict`` comes back as a plain ``Verdict``. A row with no
    pick must carry ``unavailable_reason``.

    Raises pydantic's ``ValidationError`` when the row's ``method`` is outside ``Method``, or, for
    a row with no pick, when ``unavailable_reason`` is outside ``UnavailableReason``.
    """
    if row.pick is None:
        assert row.unavailable_reason is not None, (
            "verdict_from_cleaner_row: a row with no pick must carry unavailable_reason"
        )
        return Unavailable.model_validate(
            {
                "method": row.method,
                "reason": row.unavailable_reason,
                "detail": row.unavailable_detail,
                "latency_ms": row.latency_ms,
            }
        )
    assert row.odds_json is not None and row.model is not None and row.wording_hash is not None
    return Verdict.model_validate(
        {
            "method": row.method,
            "pick": row.pick,
            "odds": json.loads(row.odds_json),
            "confidence": row.confidence if row.confidence is not None else 0.0,
            "trace": Trace(
                model=row.model,
                wording_hash=row.wording_hash,
                latency_ms=row.latency_ms if row.latency_ms is not None else 0,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                request_id=row.request_id,
            ),
        }
    )


# ---------------------------------------------------------------------------
# Transitions: every change to an existing row goes through one of these functions
# ---------------------------------------------------------------------------


def settle_row(
    row: VerdictRow,
    *,
    status: VerdictStatus,
    stamped_at: datetime | None,
    new_record_id: str | None = None,
) -> VerdictRow:
    """``row`` with ``status`` and ``stamped_at`` replaced, and ``new_record_id`` too when given."""
    update: dict[str, object] = {"status": status, "stamped_at": stamped_at}
    if new_record_id is not None:
        update["new_record_id"] = new_record_id
    return row.model_copy(update=update)  # model_copy: frozen row, typed args


def _with_exit_outcome(
    row: VerdictRow,
    *,
    action: AdmissionAction,
    matched: bool,
    effect_ref: str | None,
    new_record_id: str | None,
    incoming_text: str | None,
) -> VerdictRow:
    """``row`` moved to ``pending`` with the outcome a caller's exit intends."""
    return row.model_copy(  # model_copy: frozen row, typed args
        update={
            "status": "pending",
            "action": action,
            "actor": "caller",
            "matched": matched,
            "effect_ref": effect_ref,
            "new_record_id": new_record_id,
            "incoming_text": incoming_text,
        }
    )


def _overridden(row: VerdictRow, *, tool: str, at: datetime) -> VerdictRow:
    """``row`` with ``override`` and ``override_at`` set."""
    update = {"override": tool, "override_at": at}
    return row.model_copy(update=update)  # model_copy: frozen row, typed args


def restamp_cleaner_row(
    row: VerdictRow,
    *,
    action: CleanerAction,
    actor: VerdictActor,
    matched: bool | None,
    now: datetime,
) -> VerdictRow:
    """Move an existing ``acted`` Cleaner row back to ``pending`` under a new action, for the
    revisit axis's reuse of a pre-pass verdict.

    ``created_at`` and the verdict fields (``pick``, ``odds_json``, trace fields) are kept; only the
    outcome fields change. ``effect_ref`` becomes ``now.isoformat()``. ``created_at`` is the
    original pre-pass time, so ``_effect_landed`` needs this attempt's own start to avoid accepting
    an older resolving review. For ``split_shelve``, ``new_record_id`` is set to
    ``row.candidate_id`` so a crash after the shelve still leaves a linked row; for other actions it
    is unchanged.
    """
    return row.model_copy(  # model_copy: frozen row, typed args
        update={
            "status": "pending",
            "action": action,
            "actor": actor,
            "matched": matched,
            "effect_ref": now.isoformat(),
            "new_record_id": row.candidate_id if action == "split_shelve" else row.new_record_id,
            "stamped_at": None,
        }
    )


def _sibling_acted(rows: Sequence[VerdictRow], *, now: datetime) -> list[VerdictRow]:
    return [settle_row(row, status="sibling_acted", stamped_at=now) for row in rows]


# ---------------------------------------------------------------------------
# Split-band resolution, shared by the relevance and revisit axes.
# ---------------------------------------------------------------------------

SplitAction = Literal["split_keep", "split_shelve", "split_annotated", "panel_contested", "none"]


def decide_split_action(
    prepass_verdict: Verdict | Unavailable | None, *, cfg: EngineConfig
) -> SplitAction:
    """Classify a relevance or revisit split from the record's pre-pass (or fresh split-call)
    verdict.

    ``None`` (no Assessor configured) gives ``"none"``. ``Unavailable`` gives ``"panel_contested"``.
    A ``keep-*`` pick gives ``"split_keep"``. Any other pick gives ``"split_shelve"`` when
    ``cfg.cleaner_split_shelve`` is on, else ``"split_annotated"``.
    """
    if prepass_verdict is None:
        return "none"
    if isinstance(prepass_verdict, Unavailable):
        return "panel_contested"
    if prepass_verdict.pick.startswith("keep"):
        return "split_keep"
    if cfg.cleaner_split_shelve:
        return "split_shelve"
    return "split_annotated"


# ---------------------------------------------------------------------------
# Stamping (crash order: pending -> mutate -> stamp)
# ---------------------------------------------------------------------------


_SHELVE_ACTIONS = frozenset({"panel_shelve", "split_shelve"})


async def stamp_failed_for_reverted_shelve(
    adapter: StorageAdapter, *, record_id: HypomnemaId, now: datetime
) -> None:
    """Stamp ``failed`` the newest ``acted`` shelve row (``panel_shelve``/``split_shelve``) of
    ``record_id`` that carries no override, because the Cleaner has undone that shelve itself.
    """
    rows = await adapter.verdicts_by_effect(str(record_id))
    shelves = [
        row
        for row in rows
        if row.action in _SHELVE_ACTIONS and row.status == "acted" and row.override is None
    ]
    if not shelves:
        return
    newest = max(shelves, key=lambda row: row.stamped_at or row.created_at)
    await stamp_failed_many(adapter, [newest], now=now)


async def stamp_failed_for_reverted_edge(
    adapter: StorageAdapter, *, edge_id: str, now: datetime
) -> None:
    """Stamp ``failed`` every ``acted`` row of the supersession edge ``edge_id`` that carries no
    override. No ``override`` is stamped: the Cleaner undid its own edge, which is not owner
    disagreement.
    """
    rows = await adapter.verdicts_by_effect(edge_id)
    targets = [row for row in rows if row.status == "acted" and row.override is None]
    await stamp_failed_many(adapter, targets, now=now)


# ---------------------------------------------------------------------------
# Caller-exit stamping (bounce resolve path)
# ---------------------------------------------------------------------------


async def find_bounce_row(
    adapter: StorageAdapter, *, bounce_id: str | None, candidate_id: HypomnemaId
) -> VerdictRow | None:
    """The one ``unanswered`` row a caller's resolve names, or ``None``.

    A missing ``bounce_id``, an unknown one, an unmatched candidate or an already-resolved row
    returns ``None`` without raising. All but the missing ``bounce_id`` log one DEBUG line.
    """
    if bounce_id is None:
        return None
    rows = await adapter.verdicts_by_bounce(bounce_id)
    target = next(
        (
            row
            for row in rows
            if row.candidate_id == str(candidate_id) and row.status == "unanswered"
        ),
        None,
    )
    if target is None:
        _log.debug(
            "verdict_bounce_unresolved bounce_id=%s candidate_id=%s", bounce_id, candidate_id
        )
        return None
    return target


def caller_exit_matches_pick(
    row: VerdictRow,
    *,
    action: AdmissionAction,
    incoming_kind: Kind,
    candidate: Fact | Decision,
    direction: Direction,
) -> bool:
    """Whether ``action`` equals the exit ``legal_exit`` dispatches for this row's ``pick`` and
    ``reason_class`` with every switch forced on.

    ``promoted_from`` never matches, a row with no pick never matches, and neither does a
    ``Withheld`` or ``Invalidate`` exit. ``Promote`` maps to ``"supersede"``. A ``pick`` or
    ``reason_class`` outside the Assessor's sets is read the way ``_parse_relation`` and
    ``_parse_supersession_reason`` say.
    """
    if row.pick is None or action == "promoted_from":
        return False
    ideal = legal_exit(
        relation=_parse_relation(row.pick),
        reason_class=_parse_supersession_reason(row.reason_class),
        incoming_kind=incoming_kind,
        candidate=candidate,
        direction=direction,
        switches=_ALL_SWITCHES_ON,
    )
    if isinstance(ideal, Withheld) or isinstance(ideal, Invalidate):
        return False
    return _IDEAL_ACTION_BY_EXIT.get(type(ideal)) == action


async def begin_caller_exit(
    adapter: StorageAdapter,
    row: VerdictRow,
    *,
    action: AdmissionAction,
    matched: bool,
    effect_ref: str | None,
    new_record_id: str | None,
    incoming_text: str | None,
    now: datetime,
) -> VerdictRow:
    """Move the named row ``unanswered -> pending`` before the mutation runs. Call it in the same
    lock hold as the mutation.

    The full intended outcome is set on the row at this write: ``action``, ``actor="caller"``,
    ``matched``, ``effect_ref``, ``new_record_id``, ``incoming_text``, and ``override`` plus
    ``override_at`` when ``matched`` is false. A crash before the final stamp therefore leaves the
    reconcile a complete row to flip.
    """
    pending = _with_exit_outcome(
        row,
        action=action,
        matched=matched,
        effect_ref=effect_ref,
        new_record_id=new_record_id,
        incoming_text=incoming_text,
    )
    if not matched:
        pending = _overridden(pending, tool="caller_against_pick", at=now)
    await adapter.update_verdicts([pending])
    return pending


async def stamp_siblings(
    adapter: StorageAdapter, *, bounce_id: str, exclude_candidate_id: HypomnemaId, now: datetime
) -> None:
    """Move every other ``unanswered`` row on ``bounce_id`` to ``sibling_acted`` once the caller has
    acted on one candidate.
    """
    rows = await adapter.verdicts_by_bounce(bounce_id)
    siblings = [
        row
        for row in rows
        if row.candidate_id != str(exclude_candidate_id) and row.status == "unanswered"
    ]
    if not siblings:
        return
    await adapter.update_verdicts(_sibling_acted(siblings, now=now))


async def stamp_acted_many(
    adapter: StorageAdapter,
    rows: Sequence[VerdictRow],
    *,
    now: datetime,
    new_record_id: str | None = None,
) -> None:
    """Stamp ``pending`` rows ``acted`` after the mutation that landed them, in one
    ``update_verdicts`` call. A composed ``moved`` set writes several per-predecessor rows that
    share one successor. No-op on empty ``rows``.
    """
    if not rows:
        return
    await adapter.update_verdicts(
        [
            settle_row(row, status="acted", stamped_at=now, new_record_id=new_record_id)
            for row in rows
        ]
    )


async def stamp_failed_many(
    adapter: StorageAdapter, rows: Sequence[VerdictRow], *, now: datetime
) -> None:
    """Stamp rows ``failed`` in one ``update_verdicts`` call: their mutation raised, or the
    reconcile found ``pending`` rows whose effect never landed. No-op on empty ``rows``.
    """
    if not rows:
        return
    await adapter.update_verdicts(
        [settle_row(row, status="failed", stamped_at=now) for row in rows]
    )


class StampFailedOnError:
    """Async context manager for the block between a pending write and its ``acted`` stamp.
    When the block raises an ``Exception``, each row group is stamped ``failed`` through
    ``stamp_failed_many``, one call per group in the order given, and the same exception
    propagates unchanged. A ``BaseException`` that is not an ``Exception`` (cancellation) is
    never stamped; ``reconcile_pending_verdicts`` settles those rows later. ``at`` is called
    once per group, only on failure, at the moment that group is stamped. A ``None`` group is
    skipped and does not call ``at``; an empty group calls ``at`` and stamps nothing. The
    ``acted`` stamp stays with the caller, after the block.
    """

    def __init__(
        self,
        adapter: StorageAdapter,
        at: Callable[[], datetime],
        *row_groups: Sequence[VerdictRow] | None,
    ) -> None:
        self._adapter = adapter
        self._at = at
        self._row_groups = row_groups

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None or not isinstance(exc, Exception):
            return
        for rows in self._row_groups:
            if rows is None:
                continue
            await stamp_failed_many(self._adapter, rows, now=self._at())


async def stamp_acted_many_with_siblings(
    adapter: StorageAdapter,
    rows: Sequence[VerdictRow],
    *,
    bounce_id: str,
    exclude_candidate_ids: AbstractSet[HypomnemaId],
    now: datetime,
) -> None:
    """Stamp caller-named rows ``acted`` under one prior bounce, and every other ``unanswered`` row
    on ``bounce_id`` whose candidate is not in ``exclude_candidate_ids`` ``sibling_acted``, all in
    one ``update_verdicts`` call. No-op on empty ``rows``.

    NOTE: one call matters. Once a primary is ``acted`` it leaves ``pending_verdicts()``, so the
    reconcile never revisits it, and two separate calls with a crash between them would leave the
    siblings ``unanswered`` forever. ``reconcile_pending_verdicts`` stamps siblings while the
    primary is still ``pending`` and flips the primary in a later batched update, so that loop can
    be retried after a crash.
    """
    if not rows:
        return
    stamped_primaries = [settle_row(row, status="acted", stamped_at=now) for row in rows]
    bounce_rows = await adapter.verdicts_by_bounce(bounce_id)
    exclude_strs = {str(cid) for cid in exclude_candidate_ids}
    siblings = [
        r for r in bounce_rows if r.candidate_id not in exclude_strs and r.status == "unanswered"
    ]
    await adapter.update_verdicts([*stamped_primaries, *_sibling_acted(siblings, now=now)])


# ---------------------------------------------------------------------------
# Override stamping
# ---------------------------------------------------------------------------


async def stamp_override(
    adapter: StorageAdapter, *, ref: str, tool: str, now: datetime
) -> None:
    """Stamp ``override=tool`` and ``override_at`` on every row whose ``effect_ref`` or
    ``new_record_id`` equals ``ref`` and has no override yet; the first override wins.

    The reversal paths in ``engine/dispositions.py`` call it inside their own lock hold.
    """
    rows = await adapter.verdicts_by_effect(ref)
    targets = [row for row in rows if row.override is None]
    if not targets:
        return
    await adapter.update_verdicts([_overridden(row, tool=tool, at=now) for row in targets])


# ---------------------------------------------------------------------------
# Startup reconcile
# ---------------------------------------------------------------------------


async def _record_or_none(adapter: StorageAdapter, record_id: str) -> Fact | Decision | None:
    try:
        return await adapter.get_by_id(HypomnemaId(uuid.UUID(record_id)))
    except KeyError:
        return None


async def _reinforce_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when the target has a reinforcement event at the instant ``effect_ref`` names
    (``"{target_id}@{at_iso}"``). An unparseable ``effect_ref`` is not landed.
    """
    if row.effect_ref is None:
        return False
    target_id_str, sep, at_str = row.effect_ref.rpartition("@")
    if not sep:
        return False
    try:
        at = datetime.fromisoformat(at_str)
        target_id = HypomnemaId(uuid.UUID(target_id_str))
    except ValueError:
        return False
    events = await adapter.query_events(target_id=target_id)
    return any(event.at == at for event in events)


async def _invalidate_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when the candidate reads back ``invalid``."""
    if row.candidate_id is None:
        return False
    record = await _record_or_none(adapter, row.candidate_id)
    return record is not None and record.validity == "invalid"


async def _supersession_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when the candidate carries the row's ``effect_ref`` as its supersession edge id."""
    if row.candidate_id is None or row.effect_ref is None:
        return False
    record = await _record_or_none(adapter, row.candidate_id)
    return record is not None and getattr(record, "supersession_edge_id", None) == row.effect_ref


async def _successor_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when the record the exit wrote (``new_record_id``) exists."""
    if row.new_record_id is None:
        return False
    return await _record_or_none(adapter, row.new_record_id) is not None


async def _shelved_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when a fresh read shows the candidate shelved. ``panel_shelve`` shares this read for
    the crash window its pending row opens.
    """
    if row.candidate_id is None:
        return False
    record = await _record_or_none(adapter, row.candidate_id)
    return record is not None and getattr(record, "lifecycle", None) == "shelved"


async def _keep_review_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    """Landed when a ``band="keep"`` review exists at or after this attempt's start for the older
    participant (``candidate_id``): the review that resolves the split (relevance) or the pair
    (falsity).

    The start is ``effect_ref`` when it parses as an ISO time (a restamp sets it to now), else
    ``created_at``, since a restamped row's ``created_at`` is the original pre-pass time and could
    let an older keep review pass. A ``split_coexist`` row with a ``new_record_id`` also needs the
    newcomer's own keep review, since it resolves a pair with one review per participant. The
    comparison is ``>=`` because the pending row and its resolving review can come from the same
    clock sample.
    """
    if row.candidate_id is None:
        return False
    since = row.created_at
    if row.effect_ref is not None:
        try:
            since = datetime.fromisoformat(row.effect_ref)
        except ValueError:
            since = row.created_at
    target_ids = [HypomnemaId(uuid.UUID(row.candidate_id))]
    if row.action == "split_coexist" and row.new_record_id is not None:
        target_ids.append(HypomnemaId(uuid.UUID(row.new_record_id)))

    for target_id in target_ids:
        reviews = await adapter.list_reviews_for(target_id=target_id)
        landed = any(review.band == "keep" and review.reviewed_at >= since for review in reviews)
        if not landed:
            return False
    return True


async def _effect_landed(adapter: StorageAdapter, row: VerdictRow) -> bool:
    action = _parse_pending_action(row)
    match action:
        case "reinforce":
            return await _reinforce_landed(adapter, row)
        case "invalidate":
            return await _invalidate_landed(adapter, row)
        case "supersede" | "split_supersede" | "reason_by_assessor":
            return await _supersession_landed(adapter, row)
        case "write_new" | "confirm_distinct" | "promoted_from":
            return await _successor_landed(adapter, row)
        case "split_shelve" | "panel_shelve":
            return await _shelved_landed(adapter, row)
        case "split_keep" | "split_coexist":
            return await _keep_review_landed(adapter, row)
        case _:
            assert_never(action)


async def reconcile_pending_verdicts(adapter: StorageAdapter, *, now: datetime) -> tuple[int, int]:
    """Resolve every ``pending`` row left by a crash: ``acted`` when its effect is present in the
    store, else ``failed``. Returns ``(acted_count, failed_count)``.

    ``run_startup_passes`` calls it inside its lock hold. For a landed row with a ``bounce_id`` it
    stamps the siblings while the primary is still ``pending``; all primary stamps are written in
    one ``update_verdicts`` call at the end.

    A landed ``panel_shelve`` row that carries ``unavailable_reason`` is stamped ``unanswered`` with
    no ``stamped_at``, matching ``resolve_panel_shelve_write``'s normal completion, because an
    ``Unavailable`` pre-pass does not become a real verdict when the panel shelves the record. It
    still counts in ``acted_count``, since the write landed.

    Raises ``RuntimeError`` for a ``pending`` row whose ``action`` is not a ``PendingAction``.
    """
    pending = await adapter.pending_verdicts()
    stamped: list[VerdictRow] = []
    acted = 0
    failed = 0
    for row in pending:
        landed = await _effect_landed(adapter, row)
        if landed:
            acted += 1
            if row.action == "panel_shelve" and row.unavailable_reason is not None:
                stamped.append(settle_row(row, status="unanswered", stamped_at=None))
            else:
                stamped.append(settle_row(row, status="acted", stamped_at=now))
            if row.bounce_id is not None and row.candidate_id is not None:
                await stamp_siblings(
                    adapter, bounce_id=row.bounce_id,
                    exclude_candidate_id=HypomnemaId(uuid.UUID(row.candidate_id)), now=now,
                )
        else:
            failed += 1
            stamped.append(settle_row(row, status="failed", stamped_at=now))
    if stamped:
        await adapter.update_verdicts(stamped)
    return acted, failed
