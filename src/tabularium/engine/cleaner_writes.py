"""Cleaner-side text saves: trim on a shelve, merge on a falsity replace.

Both go through ``facts.save_internal``, the path the MCP ``save`` takes, so the dedup gate decides
whether the text is already covered. The helper passes ``assessor=None``, because the Assessor's
automatic exits could otherwise supersede or reinforce an unrelated record from inside a Cleaner
pass.

``save_cleaner_fact`` takes ``lock`` the way ``save_internal`` does, so a caller must not hold it
when calling any function here (``asyncio.Lock`` is not reentrant).

Trim: the record is already shelved when ``trim_after_shelve`` runs, and the ``keep`` text is saved
as a new Fact. A save that raises or bounces, or a cancellation during the save, attempts to
un-shelve the record, so the text is not lost along with the shelve. A failure of that un-shelve
propagates. Merge: the falsity edge is drawn first, then the newer record plus ``carry`` is saved. A
bounce, a raise or a cancellation attempts to undo the edge, so the older record keeps the part the
newer one lacks. If the undo fails with a ``ToolError`` and the edge still stands, the result is
``revert_failed``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from tabularium.clock import Clock
from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope
from tabularium.embedding import Embedder
from tabularium.engine._errors import ToolError
from tabularium.engine._types import RequireScope
from tabularium.engine.assessor import Unavailable, Verdict
from tabularium.engine.backfill import NoBackfill
from tabularium.engine.dispositions import set_shelved, unsupersede
from tabularium.engine.facts import save_internal
from tabularium.engine.verdicts import (
    SplitAction,
    WritePending,
    build_cleaner_resolved_row,
    settle_row,
    stamp_acted_many,
    stamp_failed_for_reverted_edge,
    stamp_failed_for_reverted_shelve,
    stamp_failed_many,
)
from tabularium.results import SaveResult
from tabularium.storage.types import RecordReview, StorageAdapter, VerdictRow


async def _accept_scope(scope: Scope) -> None:
    return None


async def _accept_purpose(purpose: str | None) -> None:
    return None


async def _accept_roles(roles: list[str] | None) -> None:
    return None


class TrimmedPair(BaseModel):
    """A shelved record and the Fact saved from its ``keep`` text; ``new_id`` is ``None`` in a dry
    run, where nothing is saved."""

    model_config = {"frozen": True}

    old_id: HypomnemaId
    new_id: HypomnemaId | None = None


class MergedTriple(BaseModel):
    """A falsity edge (older superseded by newer) and the record that carries the older record's
    surviving text and supersedes the newer one; ``merged_id`` is ``None`` in a dry run."""

    model_config = {"frozen": True}

    older_id: HypomnemaId
    newer_id: HypomnemaId
    merged_id: HypomnemaId | None = None


class MergeRevertedEntry(BaseModel):
    """A falsity edge that was drawn and then undone because the merged save bounced or raised."""

    model_config = {"frozen": True}

    older_id: HypomnemaId
    newer_id: HypomnemaId
    error: str


class ShelveVoter(Protocol):
    keep: str | None

    @property
    def votes_shelve(self) -> bool: ...


def winning_keep(verdicts: Sequence[ShelveVoter]) -> str | None:
    """The ``keep`` text of the first seat (A, then B) that voted shelve and supplied one."""
    return next(
        (verdict.keep for verdict in verdicts if verdict.votes_shelve and verdict.keep is not None),
        None,
    )


def pick_carry(carries: Sequence[str | None]) -> str | None:
    """The first ``carry`` text supplied, in the order given (seat A before seat B)."""
    return next((text for text in carries if text is not None), None)


@dataclass(frozen=True)
class TrimResult:
    status: Literal["trimmed", "covered", "reverted"]
    new_id: HypomnemaId | None = None
    error: str | None = None


@dataclass(frozen=True)
class MergeResult:
    status: Literal["merged", "covered", "reverted", "revert_failed"]
    new_id: HypomnemaId | None = None
    error: str | None = None


@dataclass(frozen=True)
class SplitResolution:
    """What a split resolver decided and whether its write landed.

    ``action`` is the resolved split action. ``write`` is ``"none"`` when this resolver attempted
    no action write: any action other than ``split_keep`` and ``split_shelve``, which includes
    ``split_annotated`` and ``panel_contested`` (the resolver may still record the Cleaner's own
    row, but that is not the action's write) and ``none`` (no Assessor). For ``split_keep`` it is
    ``"landed"`` when the resolving keep review landed and ``"failed"`` when it did not (the
    attempt is already stamped ``failed``). For ``split_shelve`` it is ``"landed"`` unless
    ``set_shelved`` refused, then ``"failed"``. Under ``dry_run`` the two write actions report
    ``"landed"`` and nothing is written.
    """

    action: SplitAction
    write: Literal["none", "landed", "failed"]

    @property
    def landed(self) -> bool:
        return self.write == "landed"


@dataclass(frozen=True)
class EdgeRevert:
    """What ``_revert_edge`` did. ``error`` is the text to record for the failed merge: the
    original error, with a suffix when the edge was already changed or the undo failed.
    ``edge_stands`` is ``True`` only when the undo failed in storage and the edge is still the one
    this merge drew; in every other branch (undone, or already changed by another writer) it is
    ``False``.
    """

    error: str
    edge_stands: bool


@dataclass(frozen=True)
class TrimApplied:
    """What ``apply_trim`` did, for the caller to record. ``pair`` is set when a trim is to be
    counted: the saved pair after a live ``trimmed`` save, or ``old_id`` alone in a dry run; it is
    ``None`` when there was no ``keep`` text, no embedder, the text was already covered, or the
    save failed. ``refusal`` is set only when the save failed and the shelve was reverted; the
    caller records the record as refused. The two are never both set.
    """

    pair: TrimmedPair | None = None
    refusal: str | None = None


@dataclass(frozen=True)
class MergeApplied:
    """What ``apply_merge`` did, for the caller to record. At most one field is set.

    ``merged`` is set when the merge is to be counted: the saved triple after a live ``merged``
    save, or the triple without ``merged_id`` in a dry run. ``reverted`` is set when the merged save
    failed and the edge was undone (or was already changed by another writer). ``revert_failed`` is
    set when the undo itself failed and the edge still stands. All three are ``None`` when the
    merged text was already covered.
    """

    merged: MergedTriple | None = None
    reverted: MergeRevertedEntry | None = None
    revert_failed: MergeRevertedEntry | None = None

    @property
    def edge_stands(self) -> bool:
        """``False`` only when the edge was undone, so the caller drops it from its own report."""
        return self.reverted is None


async def save_cleaner_fact(
    *,
    content: str,
    scope: Scope,
    purpose: str | None,
    source: str,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_scope: RequireScope | None = None,
    supersedes: list[HypomnemaId] | None = None,
    reason: Literal["elaborated"] | None = None,
    note: str | None = None,
) -> SaveResult:
    return await save_internal(
        content=content,
        scope=str(scope),
        source=source,
        purpose=purpose,
        audience_hint=None,
        lifecycle="active",
        valence=None,
        resolve=None,
        adapter=adapter,
        embedder=embedder,
        clock=clock,
        cfg=cfg,
        lock=lock,
        require_scope=require_scope if require_scope is not None else _accept_scope,
        require_purpose=_accept_purpose,
        require_roles=_accept_roles,
        backfill=NoBackfill(),
        supersedes=supersedes,
        reason=reason,
        note=note,
        assessor=None,
    )


def _bounce_text(result: SaveResult, subject: str = "merged text") -> str:
    if result.bounce is None:
        return result.reason or f"{subject} bounced"
    return f"{subject} bounced ({result.bounce.verdict}) against {result.bounce.candidate_id}"


async def _revert_shelve(
    *,
    record: Fact,
    error: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> None:
    restore: Literal["active", "archived"] = (
        "archived" if record.lifecycle == "archived" else "active"
    )
    await set_shelved(
        id=record.id,
        shelved=False,
        reason=f"trim save failed: {error}; shelve reverted",
        adapter=adapter,
        clock=clock,
        lock=lock,
        skip_override_stamp=True,
        cleaner_revert=True,
        restore_lifecycle=restore,
    )
    async with lock:
        await stamp_failed_for_reverted_shelve(adapter, record_id=record.id, now=clock.now())


async def _current_edge_id(*, older: Fact, newer: Fact, adapter: StorageAdapter) -> str | None:
    try:
        current = await adapter.get_by_id(older.id)
    except KeyError:
        return None
    if current.superseded_by != newer.id:
        return None
    return current.supersession_edge_id


async def _revert_edge(
    *,
    older: Fact,
    newer: Fact,
    edge_id: str | None,
    error: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> EdgeRevert:
    """Undo the ``older`` -> ``newer`` edge this merge started from, and only that edge: if the
    edge was already undone or replaced while the save ran, there is nothing of ours to roll back
    and the other writer's state is left alone. ``edge_id`` is ``None`` only when reading it before
    the save failed; the edge is then identified by ``older`` -> ``newer`` alone. Returns the text
    to record as the error and whether the edge still stands: ``True`` only when the undo itself
    failed in storage and the edge is still the one this merge drew.
    """
    current_edge_id = await _current_edge_id(older=older, newer=newer, adapter=adapter)
    if current_edge_id is None or (edge_id is not None and current_edge_id != edge_id):
        return EdgeRevert(
            error=f"{error}; the edge was already changed, nothing to undo", edge_stands=False
        )

    try:
        await unsupersede(
            id=older.id,
            reason=f"merge save failed: {error}; edge reverted",
            adapter=adapter,
            clock=clock,
            lock=lock,
            skip_override_stamp=True,
        )
    except ToolError as exc:
        still_standing = await _current_edge_id(older=older, newer=newer, adapter=adapter)
        if still_standing is not None and still_standing == current_edge_id:
            return EdgeRevert(error=f"{error}; revert failed: {exc}", edge_stands=True)
        return EdgeRevert(
            error=f"{error}; the edge was already changed, nothing to undo", edge_stands=False
        )
    async with lock:
        await stamp_failed_for_reverted_edge(adapter, edge_id=current_edge_id, now=clock.now())
    return EdgeRevert(error=error, edge_stands=False)


async def trim_after_shelve(
    *,
    record: Fact,
    keep: str,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_scope: RequireScope | None = None,
) -> TrimResult:
    """Save ``keep`` as a new Fact after ``record`` has been shelved.

    ``saved``/``reconciled`` is ``trimmed``; ``already_saved``/``reinforced``/``revived`` means the
    text already exists and the shelve stands (``covered``). A bounce only means the text is near
    an existing record, so it un-shelves ``record`` like a raise does and reports ``reverted`` with
    the bounce text. A save that raises un-shelves ``record`` and reports ``reverted`` with the
    error. A cancellation during the save un-shelves ``record`` before the cancellation
    propagates.
    """
    try:
        result = await save_cleaner_fact(
            content=keep,
            scope=record.scope,
            purpose=record.purpose,
            source=f"cleaner:trim:{record.id}",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=cfg,
            lock=lock,
            require_scope=require_scope,
        )
    except asyncio.CancelledError:
        await asyncio.shield(
            _revert_shelve(
                record=record,
                error="cancelled",
                adapter=adapter,
                clock=clock,
                lock=lock,
            )
        )
        raise
    except Exception as exc:  # deliberately broad -- any failed save must revert the shelve
        error = str(exc) or type(exc).__name__
        await _revert_shelve(record=record, error=error, adapter=adapter, clock=clock, lock=lock)
        return TrimResult(status="reverted", error=error)

    if result.outcome in ("saved", "reconciled") and result.hypomnema is not None:
        return TrimResult(status="trimmed", new_id=result.hypomnema.id)
    if result.outcome == "bounced":
        error = _bounce_text(result, "trim text")
        await _revert_shelve(record=record, error=error, adapter=adapter, clock=clock, lock=lock)
        return TrimResult(status="reverted", error=error)
    return TrimResult(status="covered")


async def merge_carry(
    *,
    newer: Fact,
    older: Fact,
    carry: str,
    adapter: StorageAdapter,
    embedder: Embedder,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    require_scope: RequireScope | None = None,
) -> MergeResult:
    """Save ``newer.content + "\\n\\n" + carry`` as a record that supersedes ``newer``, after the
    ``older`` -> ``newer`` edge was drawn.

    ``saved``/``reconciled`` is ``merged``; ``already_saved``/``reinforced``/``revived`` means the
    merged text already exists and the edge stands. A bounce or a raise undoes the
    ``older`` -> ``newer`` edge (both records stay live) and reports ``reverted`` with the cause;
    if that undo itself fails in storage and the edge still stands, it reports ``revert_failed``.
    A cancellation during the save undoes the edge before the cancellation propagates.
    """
    error: str | None = None
    edge_id: str | None = None
    try:
        edge_id = await _current_edge_id(older=older, newer=newer, adapter=adapter)
        result = await save_cleaner_fact(
            content=newer.content + "\n\n" + carry,
            scope=newer.scope,
            purpose=newer.purpose,
            source=f"cleaner:merge:{older.id}",
            adapter=adapter,
            embedder=embedder,
            clock=clock,
            cfg=cfg,
            lock=lock,
            require_scope=require_scope,
            supersedes=[newer.id],
            reason="elaborated",
            note=f"merged: carries what {older.id} still held",
        )
    except asyncio.CancelledError:
        await asyncio.shield(
            _revert_edge(
                older=older,
                newer=newer,
                edge_id=edge_id,
                error="cancelled",
                adapter=adapter,
                clock=clock,
                lock=lock,
            )
        )
        raise
    except Exception as exc:  # deliberately broad -- any failed save must undo the edge
        error = str(exc) or type(exc).__name__
    else:
        if result.outcome in ("saved", "reconciled") and result.hypomnema is not None:
            return MergeResult(status="merged", new_id=result.hypomnema.id)
        if result.outcome != "bounced":
            return MergeResult(status="covered")
        error = _bounce_text(result)

    revert = await _revert_edge(
        older=older,
        newer=newer,
        edge_id=edge_id,
        error=error,
        adapter=adapter,
        clock=clock,
        lock=lock,
    )
    return MergeResult(
        status="revert_failed" if revert.edge_stands else "reverted", error=revert.error
    )


async def apply_trim(
    *,
    record: Fact,
    verdicts: Sequence[ShelveVoter],
    embedder: Embedder | None,
    require_scope: RequireScope | None,
    adapter: StorageAdapter,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    dry_run: bool,
) -> TrimApplied:
    """Trim an axis's just-shelved ``record``; the caller appends ``pair`` to its own ``trimmed``
    list and records ``refusal`` (the save failed and the shelve was reverted) as a refused record.

    The text is ``winning_keep(verdicts)``; no seat supplying one means no trim. A dry run returns
    the pair and writes nothing. Without an ``embedder`` the save cannot run and the shelve stands
    untouched.
    """
    keep = winning_keep(verdicts)
    if keep is None:
        return TrimApplied()
    if dry_run:
        return TrimApplied(pair=TrimmedPair(old_id=record.id))
    if embedder is None:
        return TrimApplied()

    result = await trim_after_shelve(
        record=record,
        keep=keep,
        adapter=adapter,
        embedder=embedder,
        clock=clock,
        cfg=cfg,
        lock=lock,
        require_scope=require_scope,
    )
    if result.status == "trimmed":
        return TrimApplied(pair=TrimmedPair(old_id=record.id, new_id=result.new_id))
    if result.status == "reverted":
        return TrimApplied(refusal=f"trim save failed: {result.error}; shelve reverted")
    return TrimApplied()


async def apply_merge(
    *,
    newer: Fact,
    older: Fact,
    carry: str,
    embedder: Embedder,
    require_scope: RequireScope | None,
    adapter: StorageAdapter,
    clock: Clock,
    cfg: EngineConfig,
    lock: asyncio.Lock,
    dry_run: bool,
) -> MergeApplied:
    """Merge an axis's just-drawn ``older`` -> ``newer`` edge with ``carry``; the caller appends
    the returned entry to its own ``merged``, ``merge_reverted`` or ``merge_revert_failed`` list
    and drops the edge from its own report when ``edge_stands`` is ``False``.

    A dry run returns the merged triple and writes nothing.
    """
    if dry_run:
        return MergeApplied(merged=MergedTriple(older_id=older.id, newer_id=newer.id))

    result = await merge_carry(
        newer=newer,
        older=older,
        carry=carry,
        adapter=adapter,
        embedder=embedder,
        clock=clock,
        cfg=cfg,
        lock=lock,
        require_scope=require_scope,
    )
    if result.status == "merged":
        return MergeApplied(
            merged=MergedTriple(older_id=older.id, newer_id=newer.id, merged_id=result.new_id)
        )
    if result.status in ("reverted", "revert_failed"):
        entry = MergeRevertedEntry(
            older_id=older.id, newer_id=newer.id, error=result.error or "merge save failed"
        )
        if result.status == "revert_failed":
            return MergeApplied(revert_failed=entry)
        return MergeApplied(reverted=entry)
    return MergeApplied()


async def resolve_split_keep_write(
    *,
    pending_row: VerdictRow,
    write_pending: WritePending,
    record: Fact,
    rubric_version: str,
    next_review_due_at: datetime,
    now: datetime,
    adapter: StorageAdapter,
    lock: asyncio.Lock,
    on_review_append_failure: Callable[[Exception], Awaitable[None]] | None = None,
) -> bool:
    """Resolve a ``split_keep`` by appending a ``band="keep"`` ``RecordReview``. Returns whether the
    review landed.

    ``write_pending`` is the caller's write of ``pending_row``: ``adapter.append_verdicts`` for a
    fresh row (the relevance axis), ``adapter.update_verdicts`` for an existing ``cleaner_prepass``
    row being restamped (the revisit axis). Under one lock hold the row lands ``pending``, the
    review is appended, and the row is stamped ``acted``.

    The review is what resolves the split, since the record's latest review decides whether storage
    still lists it as contested. When the append raises, the row is stamped ``failed``,
    ``on_review_append_failure`` is called if given, and the function returns ``False`` so the
    caller falls back to its contested write.
    """
    review = RecordReview(
        target_id=record.id,
        scope=record.scope,
        reviewed_at=now,
        next_review_due_at=next_review_due_at,
        outcome="assessor_resolved",
        band="keep",
        grades=(),
        rubric_version=rubric_version,
    )
    async with lock:
        await write_pending([pending_row])
        try:
            await adapter.append_review(review)
        except Exception as exc:  # deliberately broad -- mirrors _append_review_safely
            if on_review_append_failure is not None:
                await on_review_append_failure(exc)
            await stamp_failed_many(adapter, [pending_row], now=now)
            return False
        await stamp_acted_many(adapter, [pending_row], now=now)
    return True


async def resolve_split_shelve_write(
    *,
    pending_row: VerdictRow,
    write_pending: WritePending,
    record: Fact,
    reason: str,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
    require_valid_and_unsuperseded: bool = False,
) -> bool:
    """Resolve a ``split_shelve`` by calling ``set_shelved``. Returns whether the shelve landed.

    ``write_pending`` is the caller's write of ``pending_row``, as for ``resolve_split_keep_write``.
    The row lands ``pending`` in its own lock hold, ``set_shelved`` runs outside any lock hold of
    this function, and the row is then stamped ``acted`` in a second hold. A ``ToolError`` from
    ``set_shelved`` stamps the row ``failed`` and returns ``False``, and the caller reports the
    refusal.

    ``pending_row`` already carries ``new_record_id`` = the record's own id, set by
    ``build_cleaner_pending_row`` or ``restamp_cleaner_row``, so a crash after ``set_shelved`` or a
    racing owner reversal finds a linked row. ``skip_override_stamp=True`` keeps ``set_shelved``
    from taking this write for an owner's reversal of an earlier one. The second hold re-reads the
    row before stamping, because an owner override may have landed on it during ``set_shelved`` and
    stamping the caller's stale copy would erase it.

    ``require_valid_and_unsuperseded`` is passed to ``set_shelved`` unchanged. The revisit axis
    passes ``True``, which re-checks validity and supersession in the same lock hold as the shelve;
    the relevance axis uses the default ``False``.
    """
    async with lock:
        await write_pending([pending_row])
    try:
        await set_shelved(
            id=record.id, shelved=True, reason=reason, adapter=adapter, clock=clock, lock=lock,
            require_valid_and_unsuperseded=require_valid_and_unsuperseded,
            skip_override_stamp=True,
        )
    except ToolError:
        async with lock:
            await stamp_failed_many(adapter, [pending_row], now=clock.now())
        return False
    async with lock:
        fresh_rows = await adapter.verdicts_for_record(record.id)
        fresh_row = next((r for r in fresh_rows if r.id == pending_row.id), pending_row)
        await stamp_acted_many(adapter, [fresh_row], now=clock.now(), new_record_id=str(record.id))
    return True


async def resolve_panel_shelve_write(
    *,
    candidate_id: HypomnemaId,
    verdict: Verdict | Unavailable,
    matched: bool | None,
    reason: str,
    now: datetime,
    adapter: StorageAdapter,
    clock: Clock,
    lock: asyncio.Lock,
) -> None:
    """Shelve a record on a unanimous panel verdict and log its ``panel_shelve`` row, in the same
    order as ``resolve_split_shelve_write``.

    The row lands ``pending`` first with ``new_record_id=candidate_id``, the key ``stamp_override``
    looks rows up by, so an owner reversal that races the shelve finds a row. ``set_shelved`` runs
    with ``skip_override_stamp=True``. The row is then re-read under the lock and stamped, so a
    concurrent override survives. The final status is ``unanswered`` (no ``stamped_at``) when
    ``verdict`` is ``Unavailable``, else ``acted``.

    Raises the ``ToolError`` from ``set_shelved`` after stamping the row ``failed``;
    ``run_relevance_axis`` catches it per record.

    NOTE: ``panel_shelve`` is not an Assessor-caused write. The ``pending`` row exists so the
    override lookup works; ``_effect_landed`` also reads ``panel_shelve`` so a crash before the
    stamp is resolved at the next reconcile.
    """
    terminal_status: Literal["unanswered", "acted"] = (
        "unanswered" if isinstance(verdict, Unavailable) else "acted"
    )
    pending_row = settle_row(
        build_cleaner_resolved_row(
            caller="cleaner_prepass",
            candidate_id=candidate_id,
            new_record_id=candidate_id,
            verdict=verdict,
            action="panel_shelve",
            actor="panel",
            matched=matched,
            now=now,
        ),
        status="pending",
        stamped_at=None,
    )
    async with lock:
        await adapter.append_verdicts([pending_row])
    try:
        await set_shelved(
            id=candidate_id, shelved=True, reason=reason, adapter=adapter, clock=clock,
            lock=lock, skip_override_stamp=True,
        )
    except ToolError:
        async with lock:
            await stamp_failed_many(adapter, [pending_row], now=clock.now())
        raise
    async with lock:
        fresh_rows = await adapter.verdicts_for_record(candidate_id)
        fresh_row = next((r for r in fresh_rows if r.id == pending_row.id), pending_row)
        await adapter.update_verdicts(
            [
                settle_row(
                    fresh_row,
                    status=terminal_status,
                    stamped_at=clock.now() if terminal_status == "acted" else None,
                )
            ]
        )
