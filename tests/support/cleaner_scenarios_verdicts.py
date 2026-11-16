"""Startup-reconcile scenarios for the cleaner golden. Importing the module registers them.

A pass that crashes between a pending verdict row and its stamp leaves the row for
``reconcile_pending_verdicts`` to settle at the next start. These scenarios leave pending rows for
``panel_shelve`` and for each Assessor-caused write action, landed and not landed, and settle them.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from tabularium.engine.assessor import Unavailable
from tabularium.engine.dispositions import set_shelved, supersede
from tabularium.engine.verdicts import (
    build_cleaner_pending_row,
    build_cleaner_resolved_row,
    reconcile_pending_verdicts,
)
from tabularium.storage.types import RecordReview, VerdictRow
from tests.support.cleaner_golden import contradiction_pick, relevance_pick
from tests.support.cleaner_world import (
    AGED,
    BORN,
    Json,
    World,
    dump_store,
    expect_raise,
    make_world,
    put_fact,
    rid,
    scenario,
)

_ATTEMPT = AGED + timedelta(days=40)


async def _pending(
    world: World,
    number: int,
    action: Literal[
        "split_keep", "split_shelve", "split_coexist", "split_supersede", "reason_by_assessor"
    ],
    *,
    effect_ref: str | None = None,
    new_record_id: int | None = None,
) -> VerdictRow:
    verdict = (
        contradiction_pick("changed", 0.9)
        if action in ("split_supersede", "reason_by_assessor")
        else relevance_pick("keep-lean", 0.7)
    )
    row = build_cleaner_pending_row(
        caller="cleaner_split"
        if action in ("split_coexist", "split_supersede")
        else "cleaner_prepass",
        candidate_id=rid(number),
        new_record_id=rid(new_record_id) if new_record_id is not None else None,
        verdict=verdict,
        action=action,
        effect_ref=effect_ref,
        now=_ATTEMPT,
    )
    await world.adapter.append_verdicts([row])
    return row


async def _pending_panel_shelve(world: World, number: int, *, unavailable: bool) -> None:
    verdict = (
        Unavailable(method="relevance", reason="timeout", detail=None, latency_ms=0)
        if unavailable
        else relevance_pick("shelve-clear", 0.9)
    )
    row = build_cleaner_resolved_row(
        caller="cleaner_prepass",
        candidate_id=rid(number),
        new_record_id=rid(number),
        verdict=verdict,
        action="panel_shelve",
        actor="panel",
        matched=None if unavailable else True,
        now=_ATTEMPT,
    ).model_copy(update={"status": "pending", "stamped_at": None})
    await world.adapter.append_verdicts([row])


async def _keep_review(world: World, number: int, at_offset_seconds: int) -> None:
    await world.adapter.append_review(
        RecordReview(
            target_id=rid(number),
            scope=world.scope,
            reviewed_at=_ATTEMPT + timedelta(seconds=at_offset_seconds),
            next_review_due_at=_ATTEMPT + timedelta(days=90),
            outcome="assessor_resolved",
            band="keep",
            grades=(),
            rubric_version=world.rubric.version,
        )
    )


async def _world_with_records(count: int) -> World:
    world = await make_world()
    for number in range(1, count + 1):
        await put_fact(
            world, number, f"Record {number} left behind by a crashed pass", created=BORN
        )
    return world


@scenario
async def verdicts_reconcile_settles_a_pending_row_of_every_cleaner_action() -> Json:
    world = await _world_with_records(20)
    adapter = world.adapter
    shelve_reason = "scripted shelve"
    for number in (1, 3, 4):
        await set_shelved(
            id=rid(number),
            shelved=True,
            reason=shelve_reason,
            adapter=adapter,
            clock=world.clock,
            lock=world.lock,
        )
    await supersede(
        successor_id=rid(10),
        predecessor_ids=[rid(9)],
        reason="changed",
        note=None,
        adapter=adapter,
        clock=world.clock,
        lock=world.lock,
        edge_id="edge-that-landed",
    )
    await supersede(
        successor_id=rid(18),
        predecessor_ids=[rid(17)],
        reason="correction",
        note=None,
        adapter=adapter,
        clock=world.clock,
        lock=world.lock,
        edge_id="edge-reason-landed",
    )
    await _pending(
        world, 17, "reason_by_assessor", effect_ref="edge-reason-landed", new_record_id=18
    )
    await _pending(world, 19, "reason_by_assessor", effect_ref="edge-reason-lost", new_record_id=20)
    await _pending(world, 1, "split_shelve", new_record_id=1)
    await _pending(world, 2, "split_shelve", new_record_id=2)
    await _pending_panel_shelve(world, 3, unavailable=False)
    await _pending_panel_shelve(world, 4, unavailable=True)
    await _pending_panel_shelve(world, 5, unavailable=False)
    await _pending(world, 9, "split_supersede", effect_ref="edge-that-landed", new_record_id=10)
    await _pending(
        world, 11, "split_supersede", effect_ref="edge-that-never-landed", new_record_id=12
    )
    await _pending(world, 6, "split_keep", effect_ref=_ATTEMPT.isoformat())
    await _keep_review(world, 6, 1)
    await _pending(world, 7, "split_keep", effect_ref="not-a-time")
    await _keep_review(world, 7, -3600)
    await _pending(world, 8, "split_keep")
    await _pending(world, 13, "split_coexist", new_record_id=14)
    await _keep_review(world, 13, 0)
    await _keep_review(world, 14, 0)
    await _pending(world, 15, "split_coexist", new_record_id=16)
    await _keep_review(world, 15, 0)
    await _pending(world, 99, "split_shelve")
    await world.freeze_seed()
    acted, failed = await reconcile_pending_verdicts(adapter, now=world.clock.now())
    dump = await dump_store(world)
    dump["counts"] = {"acted": acted, "failed": failed}
    return dump


@scenario
async def verdicts_reconcile_refuses_a_pending_row_with_no_read_path() -> Json:
    world = await _world_with_records(1)
    row = build_cleaner_resolved_row(
        caller="cleaner_prepass",
        candidate_id=rid(1),
        verdict=relevance_pick("keep-lean", 0.7),
        action="panel_keep",
        actor="panel",
        matched=True,
        now=_ATTEMPT,
    ).model_copy(update={"status": "pending", "stamped_at": None})
    await world.adapter.append_verdicts([row])
    await world.freeze_seed()
    raised = await expect_raise(reconcile_pending_verdicts(world.adapter, now=world.clock.now()))
    dump = await dump_store(world)
    dump["raised"] = {"type": type(raised).__name__, "message": str(raised)}
    return dump
