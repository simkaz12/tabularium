"""Falsity-axis scenarios for the cleaner golden. Importing the module registers them."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest import mock

from tabularium.domain import Fact, FactLifecycle
from tabularium.engine import cleaner_writes
from tabularium.engine import judge as judge_mod
from tabularium.engine._errors import ToolError
from tabularium.engine.dispositions import unsupersede
from tabularium.engine.falsity import FalsityVerdict
from tabularium.engine.judge import render_batch_prompt
from tests.support.cleaner_golden import (
    ScriptedAssessor,
    breadth_pick,
    coexist,
    contradiction_pick,
    golden_rubric,
    replace,
)
from tests.support.cleaner_scenario_kit import Panel, flaky_appends, panel
from tests.support.cleaner_world import (
    BORN,
    OLDER,
    Json,
    World,
    expect_raise,
    make_world,
    put_decision,
    put_fact,
    rid,
    run_falsity,
    scenario,
    snapshot,
)

CARRY = "Metrics are exposed on port 9100 <c9>"


async def put_pair(
    world: World,
    i: int,
    *,
    older_scope: str | None = None,
    older_lifecycle: FactLifecycle = "active",
    older_created: datetime = OLDER,
    newer_created: datetime = BORN,
    newer_padding: str = "",
) -> tuple[Fact, Fact]:
    """Seed an older record ``[oI]`` and a newer one ``[nI]`` in cluster ``i`` (cosine 0.96)."""
    older = await put_fact(
        world,
        10 * i + 1,
        f"Pi serves from the old address {i} [o{i}] <c{i}.0>",
        created=older_created,
        scope=older_scope,
        lifecycle=older_lifecycle,
    )
    newer = await put_fact(
        world,
        10 * i + 2,
        f"Pi serves from the new address {i}{newer_padding} [n{i}] <c{i}.1>",
        created=newer_created,
    )
    return older, newer


def both(item: dict[str, object], *pairs: int) -> dict[str, dict[str, object]]:
    return {f"[n{i}]": item for i in pairs}


@scenario
async def falsity_bands_without_assessor() -> Json:
    world = await make_world()
    for i in range(1, 6):
        await put_pair(world, i)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("changed", "A: moved"),
            "[n2]": coexist("A: both stand"),
            "[n3]": replace("correction", "A: it was wrong"),
            "[n4]": replace("changed", "A: moved on"),
            "[n5]": coexist("A: both stand again"),
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": coexist("B: both stand"),
            "[n3]": replace("changed", "B: it moved"),
            "[n4]": coexist("B: still true"),
            "[n5]": replace("changed", "B: moved on"),
        },
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_bands_without_assessor_dry_run() -> Json:
    world = await make_world()
    for i in range(1, 6):
        await put_pair(world, i)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("changed", "A: moved", carry=CARRY),
            "[n2]": coexist("A: both stand"),
            "[n3]": replace("correction", "A: it was wrong", carry="Wrong but partly true <c8>"),
            "[n4]": replace("changed", "A: moved on"),
            "[n5]": coexist("A: both stand again"),
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": coexist("B: both stand"),
            "[n3]": replace("changed", "B: it moved"),
            "[n4]": coexist("B: still true"),
            "[n5]": replace("changed", "B: moved on"),
        },
    )
    summary = await run_falsity(world, judge_a, judge_b, dry_run=True)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary, dry_run=True)


@scenario
async def falsity_empty_pool() -> Json:
    world = await make_world()
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_candidates_are_classified_and_newcomers_excluded() -> Json:
    world = await make_world()
    await put_pair(world, 1)
    await put_decision(world, 21, "Pi serves from a fixed address [d2] <c2.0>")
    await put_fact(world, 22, "Pi serves from the address of record [n2] <c2.1>")
    await put_fact(
        world, 31, "Pi serves from a global address [g3] <c3.0>", scope="global", created=OLDER
    )
    await put_fact(world, 32, "Pi serves from the project address [n3] <c3.1>")
    await put_fact(
        world, 41, "Pi serves from a pinned address [h4] <c4.0>", lifecycle="hot", created=OLDER
    )
    await put_fact(world, 42, "Pi serves from the address under review [n4] <c4.1>")
    await put_fact(
        world, 51, "Pi serves from an ember address [e5] <c5.0>", lifecycle="ember", created=OLDER
    )
    await put_fact(world, 52, "Pi serves from the address in memory [n5] <c5.1>")
    await put_fact(world, 61, "Pi serves from the same-age address [s6] <c6.0>", created=BORN)
    await put_fact(world, 62, "Pi serves from the equal-age address [n6] <c6.1>", created=BORN)
    await put_fact(
        world,
        71,
        "Pi serves from a later hot address [l7] <c7.0>",
        lifecycle="hot",
        created=BORN + timedelta(hours=1),
    )
    await put_fact(world, 72, "Pi serves from the earlier address [n7] <c7.1>")
    await put_fact(world, 82, "Pi serves from an address nobody else shares [n8] <c8.1>")
    await put_decision(world, 91, "Decision born in the window [x9] <c9>", created=BORN)
    await put_fact(world, 92, "Hot record born in the window [x10] <c10>", lifecycle="hot")
    await put_fact(world, 93, "Shelved record born in the window [x11] <c11>", lifecycle="shelved")
    await put_fact(world, 94, "Record the caller barred from the pass [x12] <c12>")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[n1]": replace("changed", "A: moved")})
    summary = await run_falsity(world, judge_a, judge_b, excluded_ids=frozenset({rid(94)}))
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_failed_batches_whatever_the_status() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=1))
    for i in range(1, 5):
        await put_pair(world, i)
    await world.freeze_seed()
    items = {f"[n{i}]": replace("changed", f"{i}: moved") for i in range(1, 5)}
    judge_a, judge_b = panel(
        items,
        a_faults=("unavailable", None, "unexpected", None),
        b_faults=(None, "invalid", "invalid", None),
    )
    summary = await run_falsity(world, judge_a, judge_b)
    statuses = {call.status for call in summary.calls}
    assert statuses == {"judge_unavailable", "invalid_output", "unexpected", "success"}
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_oversize_pair_prompt_fails_its_batch_and_the_next_is_judged() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=1))
    await put_pair(world, 1, newer_padding=" padding" * 120)
    await put_pair(world, 2)
    await world.freeze_seed()
    base = len(
        render_batch_prompt(
            rubric_prose=world.rubric.falsity.prose, contents=[], item_model=FalsityVerdict
        )
    )
    judge_a, judge_b = panel({f"[n{i}]": replace("changed", f"{i}: moved") for i in (1, 2)})
    with mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 400):
        summary = await run_falsity(world, judge_a, judge_b)
    statuses = [call.status for call in summary.calls]
    assert statuses == ["prompt_too_large", "prompt_too_large", "success", "success"]
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_second_newcomer_for_one_candidate_is_refused_by_the_supersede() -> Json:
    world = await make_world()
    await put_fact(world, 11, "Pi serves from the old address 1 [o1] <c1.0>", created=OLDER)
    await put_fact(world, 12, "Pi serves from the new address 1 [n1] <c1.1>")
    await put_fact(
        world,
        13,
        "Pi serves from the newer address 1 [n2] <c1.2>",
        created=BORN + timedelta(minutes=1),
    )
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[n1]": replace("changed", "first"), "[n2]": replace("changed", "second")}
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_write_failure_inside_the_supersede_aborts_the_pass() -> Json:
    world = await make_world()
    await put_pair(world, 1)
    await world.freeze_seed()
    judge_a, judge_b = panel({"[n1]": replace("changed", "moved")})

    async def dead_store(*args: object, **kwargs: object) -> object:
        raise RuntimeError("store unavailable")

    with mock.patch.object(world.adapter, "replace", dead_store):
        raised = await expect_raise(run_falsity(world, judge_a, judge_b))
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, raised=raised)


async def _assessor_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    for i in range(1, 10):
        await put_pair(world, i)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("correction", "A: wrong"),
            "[n2]": replace("correction", "A: wrong"),
            "[n3]": replace("correction", "A: wrong"),
            "[n4]": replace("changed", "A: moved"),
            "[n5]": replace("changed", "A: moved"),
            "[n6]": replace("changed", "A: moved"),
            "[n7]": replace("correction", "A: wrong"),
            "[n8]": replace("changed", "A: moved"),
            "[n9]": coexist("A: both stand"),
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": replace("changed", "B: moved"),
            "[n3]": replace("changed", "B: moved"),
            "[n4]": coexist("B: still true"),
            "[n5]": coexist("B: still true"),
            "[n6]": coexist("B: still true"),
            "[n7]": replace("changed", "B: moved"),
            "[n8]": replace("changed", "B: moved"),
            "[n9]": coexist("B: both stand"),
        },
    )
    assessor = ScriptedAssessor(
        contradiction={
            "[o1]": contradiction_pick("changed", 0.9),
            "[o2]": contradiction_pick("coexist", 0.9),
            "[o3]": TimeoutError("assessor timed out"),
            "[o4]": contradiction_pick("correction", 0.8),
            "[o5]": contradiction_pick("coexist", 0.8),
            "[o6]": TimeoutError("assessor timed out"),
            "[o7]": RuntimeError("assessor exploded"),
        }
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def falsity_assessor_arbitrates_every_split_kind() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_world()
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_assessor_arbitrates_every_split_kind_dry_run() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_world()
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot(
        world,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        falsity=summary,
        dry_run=True,
    )


@scenario
async def falsity_split_coexist_review_append_failures_fall_back_to_contested() -> Json:
    world = await make_world()
    for i in range(1, 5):
        await put_pair(world, i)
    await world.freeze_seed()
    split_a = {f"[n{i}]": replace("changed", "A: moved") for i in range(1, 5)}
    split_b = {f"[n{i}]": coexist("B: still true") for i in range(1, 5)}
    judge_a, judge_b = panel(split_a, split_b)
    assessor = ScriptedAssessor(
        contradiction={f"[o{i}]": contradiction_pick("coexist", 0.8) for i in range(1, 5)}
    )
    failing = {rid(12), rid(21), rid(22), rid(31)}
    with flaky_appends(
        world.adapter,
        review_when=lambda review: (
            review.outcome == "assessor_resolved" and review.target_id in failing
        ),
    ):
        summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_contested_review_and_judgment_appends_fail_independently() -> Json:
    world = await make_world()
    for i in range(1, 4):
        await put_pair(world, i)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {f"[n{i}]": replace("changed", "A: moved") for i in range(1, 4)},
        {f"[n{i}]": coexist("B: still true") for i in range(1, 4)},
    )
    with flaky_appends(
        world.adapter,
        review_when=lambda review: review.target_id in {rid(12), rid(21)},
        judgment_when=lambda judgment: judgment.target_id in {rid(11), rid(32)},
    ):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_assessor_second_newcomer_for_one_candidate_is_refused() -> Json:
    world = await make_world()
    await put_fact(world, 11, "Pi serves from the old address 1 [o1] <c1.0>", created=OLDER)
    await put_fact(world, 12, "Pi serves from the new address 1 [n1] <c1.1>")
    await put_fact(
        world,
        13,
        "Pi serves from the newer address 1 [n2] <c1.2>",
        created=BORN + timedelta(minutes=1),
    )
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[n1]": replace("changed", "A: first"), "[n2]": replace("changed", "A: second")},
        {"[n1]": coexist("B: first"), "[n2]": coexist("B: second")},
    )
    assessor = ScriptedAssessor(contradiction={"[o1]": contradiction_pick("changed", 0.9)})
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_assessor_write_failure_inside_the_supersede_aborts_the_pass() -> Json:
    world = await make_world()
    await put_pair(world, 1)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[n1]": replace("changed", "A: moved")}, {"[n1]": coexist("B: still true")}
    )
    assessor = ScriptedAssessor(contradiction={"[o1]": contradiction_pick("changed", 0.9)})

    async def dead_store(*args: object, **kwargs: object) -> object:
        raise RuntimeError("store unavailable")

    with mock.patch.object(world.adapter, "replace", dead_store):
        raised = await expect_raise(run_falsity(world, judge_a, judge_b, assessor=assessor))
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, raised=raised)


async def _breadth_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    for i in range(1, 8):
        await put_pair(world, i, older_scope="global")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("changed", "A: moved"),
            "[n2]": replace("changed", "A: moved"),
            "[n3]": replace("changed", "A: moved"),
            "[n4]": coexist("A: both stand"),
            "[n5]": replace("changed", "A: moved"),
            "[n6]": replace("changed", "A: moved"),
            "[n7]": replace("correction", "A: wrong"),
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": replace("changed", "B: moved"),
            "[n3]": replace("changed", "B: moved"),
            "[n4]": coexist("B: both stand"),
            "[n5]": coexist("B: still true"),
            "[n6]": replace("changed", "B: moved"),
            "[n7]": replace("changed", "B: moved"),
        },
    )
    assessor = ScriptedAssessor(
        breadth={
            "[n1]": breadth_pick("owner_wide"),
            "[n2]": breadth_pick("project_specific"),
            "[n3]": TimeoutError("assessor timed out"),
            "[n5]": breadth_pick("project_specific"),
            "[n6]": breadth_pick("project_specific"),
            "[n7]": breadth_pick("owner_wide"),
        },
        contradiction={
            "[o2]": contradiction_pick("changed", 0.9),
            "[o5]": contradiction_pick("coexist", 0.9),
            "[o6]": TimeoutError("assessor timed out"),
            "[o7]": contradiction_pick("correction", 0.9),
        },
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def falsity_breadth_gate_allows_gates_and_fails_closed() -> Json:
    world, (judge_a, judge_b), assessor = await _breadth_world()
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_breadth_gate_dry_run() -> Json:
    world, (judge_a, judge_b), assessor = await _breadth_world()
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot(
        world,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        falsity=summary,
        dry_run=True,
    )


@scenario
async def falsity_global_pass_treats_global_pairs_as_ordinary() -> Json:
    world = await make_world(scope="global")
    await put_fact(world, 11, "Pi serves from the old address 1 [o1] <c1.0>", created=OLDER)
    await put_fact(world, 12, "Pi serves from the new address 1 [n1] <c1.1>")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[n1]": replace("changed", "moved")})
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


async def _merge_world(
    *, third: bool = False, existing_merged: bool = False
) -> tuple[World, Fact, Fact]:
    world = await make_world()
    older, newer = await put_pair(world, 1)
    if third:
        await put_fact(world, 13, "Pi also answers on a lab address [x1] <c9.2>", created=OLDER)
    if existing_merged:
        await put_fact(world, 14, newer.content + "\n\n" + CARRY, created=OLDER)
    await world.freeze_seed()
    return world, older, newer


def _carry_panel(*, reason_a: str = "changed", reason_b: str = "changed") -> Panel:
    return panel(
        {"[n1]": replace(reason_a, "A: moved", carry=CARRY)},
        {"[n1]": replace(reason_b, "B: moved")},
    )


@scenario
async def falsity_merge_carried_from_either_seat_saves_a_superseding_record() -> Json:
    world = await make_world()
    await put_pair(world, 1)
    await put_pair(world, 2)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("changed", "A: moved", carry=CARRY),
            "[n2]": replace("changed", "A: moved"),
        },
        {
            "[n1]": replace("changed", "B: moved", carry="Seat B carry that loses <c8>"),
            "[n2]": replace("changed", "B: moved", carry="Seat B carry that wins <c8>"),
        },
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_without_a_carry_or_on_a_coexist_vote_saves_nothing() -> Json:
    world = await make_world()
    await put_pair(world, 1)
    await put_pair(world, 2)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[n1]": replace("changed", "A: moved"), "[n2]": coexist("A: both stand")},
        {"[n1]": replace("changed", "B: moved"), "[n2]": coexist("B: both stand")},
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_reverted_when_the_merged_text_bounces() -> Json:
    world, _, _ = await _merge_world(third=True)
    judge_a, judge_b = panel(
        {"[n1]": replace("changed", "A: moved", carry="Metrics are exposed on port 9100 <c9.2>")},
        {"[n1]": replace("changed", "B: moved")},
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_reverted_when_the_save_raises() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    with mock.patch.object(cleaner_writes, "save_internal", boom):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_revert_failure_keeps_the_standing_edge() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    async def failing_restore(*args: object, **kwargs: object) -> None:
        raise RuntimeError("restore failed")

    with (
        mock.patch.object(cleaner_writes, "save_internal", boom),
        mock.patch.object(world.adapter, "restore", failing_restore),
    ):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_already_present_leaves_the_edge_standing() -> Json:
    world, _, _ = await _merge_world(existing_merged=True)
    judge_a, judge_b = _carry_panel()
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_edge_changed_by_someone_else_is_not_reverted_twice() -> Json:
    world, older, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def owner_undoes_then_fails(**kwargs: object) -> object:
        await unsupersede(
            id=older.id,
            reason="owner undid the edge",
            adapter=world.adapter,
            clock=world.clock,
            lock=world.lock,
            skip_override_stamp=True,
        )
        raise RuntimeError("save failed after the edge changed")

    with mock.patch.object(cleaner_writes, "save_internal", owner_undoes_then_fails):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_cancelled_mid_save_reverts_the_edge() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def cancelled(**kwargs: object) -> object:
        raise asyncio.CancelledError

    raised: BaseException | None = None
    with mock.patch.object(cleaner_writes, "save_internal", cancelled):
        try:
            await run_falsity(world, judge_a, judge_b)
        except asyncio.CancelledError as exc:
            raised = exc
    assert raised is not None
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, raised=raised)


@scenario
async def falsity_merge_dry_run_counts_the_merge_and_saves_nothing() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()
    summary = await run_falsity(world, judge_a, judge_b, dry_run=True)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary, dry_run=True)


async def _assessor_merge_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    await put_pair(world, 1)
    await put_pair(world, 2)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("correction", "A: wrong", carry="Reason disagreement carry <c9>"),
            "[n2]": replace("changed", "A: moved", carry="Action split carry <c8>"),
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": coexist("B: still true"),
        },
    )
    assessor = ScriptedAssessor(
        contradiction={
            "[o1]": contradiction_pick("changed", 0.9),
            "[o2]": contradiction_pick("correction", 0.9),
        }
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def falsity_assessor_decided_merges_are_saved() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_merge_world()
    summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_assessor_decided_merge_reverts_take_back_their_counts() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_merge_world()

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    with mock.patch.object(cleaner_writes, "save_internal", boom):
        summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_assessor_decided_merge_revert_failure_keeps_the_counts() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_merge_world()

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    async def failing_restore(*args: object, **kwargs: object) -> None:
        raise RuntimeError("restore failed")

    with (
        mock.patch.object(cleaner_writes, "save_internal", boom),
        mock.patch.object(world.adapter, "restore", failing_restore),
    ):
        summary = await run_falsity(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, falsity=summary
    )


@scenario
async def falsity_merge_after_a_reason_disagreement_without_assessor_is_reverted() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel(reason_a="correction", reason_b="changed")

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    with mock.patch.object(cleaner_writes, "save_internal", boom):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_older_record_buried_before_the_revert() -> Json:
    world, older, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def bury_then_fail(**kwargs: object) -> object:
        await world.adapter.bury(older.id, "scripted burial")
        raise RuntimeError("save failed after the older record vanished")

    with mock.patch.object(cleaner_writes, "save_internal", bury_then_fail):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_merge_revert_error_after_the_edge_already_went_is_not_a_standing_edge() -> Json:
    world, _, _ = await _merge_world()
    judge_a, judge_b = _carry_panel()

    async def boom(**kwargs: object) -> object:
        raise RuntimeError("no disk")

    async def undo_then_fail(**kwargs: Any) -> object:  # forwards kwargs unchanged
        await unsupersede(**kwargs)
        raise ToolError("late failure after the undo landed")

    with (
        mock.patch.object(cleaner_writes, "save_internal", boom),
        mock.patch.object(cleaner_writes, "unsupersede", undo_then_fail),
    ):
        summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_schema_violations_fail_the_batch_with_the_verdict_class_in_the_error() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=1))
    for i in range(1, 5):
        await put_pair(world, i)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": {"verdict": "replace", "why": "replace without a reason"},
            "[n2]": replace("changed", "A: moved"),
            "[n3]": replace("changed", "A: moved", carry="   "),
            "[n4]": {**coexist("A: both stand"), "carry": "carry on a coexist vote is dropped"},
        },
        {
            "[n1]": replace("changed", "B: moved"),
            "[n2]": {"verdict": "coexist", "reason": "changed", "why": "coexist with a reason"},
            "[n3]": replace("changed", "B: moved"),
            "[n4]": coexist("B: both stand"),
        },
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)


@scenario
async def falsity_reason_disagreement_second_newcomer_is_refused_without_an_assessor() -> Json:
    world = await make_world()
    await put_fact(world, 11, "Pi serves from the old address 1 [o1] <c1.0>", created=OLDER)
    await put_fact(world, 12, "Pi serves from the new address 1 [n1] <c1.1>")
    await put_fact(
        world,
        13,
        "Pi serves from the newer address 1 [n2] <c1.2>",
        created=BORN + timedelta(minutes=1),
    )
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[n1]": replace("correction", "A: first, it was wrong"),
            "[n2]": replace("correction", "A: second, it was wrong"),
        },
        {
            "[n1]": replace("changed", "B: first, it moved"),
            "[n2]": replace("changed", "B: second, it moved"),
        },
    )
    summary = await run_falsity(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, falsity=summary)
