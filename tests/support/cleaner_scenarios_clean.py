"""``clean()`` scenarios for the cleaner golden. Importing the module registers them."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest import mock

from tabularium.config import EngineConfig
from tabularium.engine import clean as clean_module
from tabularium.engine import falsity, relevance, revisit
from tabularium.engine._errors import ToolError
from tabularium.engine.dispositions import set_shelved, supersede
from tabularium.storage.types import CleanerState
from tests.support.cleaner_golden import (
    ByAxis,
    ItemSpec,
    ScriptedAssessor,
    coexist,
    contradiction_pick,
    golden_rubric,
    relevance_pick,
    replace,
    shelf,
)
from tests.support.cleaner_scenario_kit import Panel, flaky_appends, invalidate_on_call, panel
from tests.support.cleaner_scenarios_falsity import put_pair
from tests.support.cleaner_scenarios_revisit import aged
from tests.support.cleaner_world import (
    BORN,
    SINCE,
    Json,
    World,
    expect_raise,
    make_world,
    put_fact,
    rid,
    run_clean,
    scenario,
    snapshot_clean,
)


async def _start_from_watermark(world: World) -> None:
    await world.adapter.set_cleaner_state(
        scope=world.scope,
        state=CleanerState(watermark_at=SINCE, last_pass_at=SINCE, report=None),
    )


PLAIN_A: dict[str, ItemSpec] = {
    "[n1]": ByAxis(relevance=shelf("shelve-clear", "A: stale")),
    "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed", "A: moved")),
    "[n3]": ByAxis(relevance=shelf("shelve-clear", "A: obsolete")),
    "[n4]": ByAxis(relevance=shelf("keep-lean", unclear=True)),
    "[a1]": ByAxis(revisit=shelf("shelve-clear", "A: obsolete")),
}
PLAIN_B: dict[str, ItemSpec] = {
    "[n1]": ByAxis(relevance=shelf("shelve-lean", "B: stale")),
    "[n2]": ByAxis(relevance=shelf("keep-lean"), falsity=replace("changed", "B: moved")),
    "[n3]": ByAxis(relevance=shelf("keep-clear", "B: still true")),
    "[n4]": ByAxis(relevance=shelf("keep-clear")),
    "[a1]": ByAxis(revisit=shelf("shelve-lean", "B: stale")),
}


async def _plain_world(*, with_state: bool = True) -> tuple[World, Panel]:
    world = await make_world()
    await put_pair(world, 2)
    await put_fact(world, 101, "Stale ticket note that both seats shelve [n1]")
    await put_fact(world, 103, "Note the seats disagree about [n3]")
    await put_fact(world, 104, "Note the seat cannot read [n4]")
    await aged(world, 201, "Old note the revisit seats shelve [a1]")
    await aged(world, 202, "Old note nothing happened to [a2]", evidence=False)
    if with_state:
        await _start_from_watermark(world)
    await world.freeze_seed()
    judges = panel(PLAIN_A, PLAIN_B)
    return world, judges


@scenario
async def clean_first_pass_initializes_the_watermark_and_still_revisits() -> Json:
    world, (judge_a, judge_b) = await _plain_world(with_state=False)
    summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_first_pass_dry_run_writes_only_the_history_row() -> Json:
    world, (judge_a, judge_b) = await _plain_world(with_state=False)
    summary = await run_clean(world, judge_a, judge_b, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_empty_pool_keeps_the_watermark_and_still_revisits() -> Json:
    world = await make_world()
    await aged(world, 201, "Old note the revisit seats shelve [a1]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judge_a, judge_b = panel({"[a1]": ByAxis(revisit=shelf("shelve-clear"))})
    summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_empty_pool_dry_run_writes_only_the_history_row() -> Json:
    world = await make_world()
    await aged(world, 201, "Old note the revisit seats shelve [a1]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judge_a, judge_b = panel({"[a1]": ByAxis(revisit=shelf("shelve-clear"))})
    summary = await run_clean(world, judge_a, judge_b, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_completed_pass_persists_the_report_with_the_state() -> Json:
    world, (judge_a, judge_b) = await _plain_world()
    summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_completed_pass_without_a_report_callback_uses_the_fallback_payload() -> Json:
    world, (judge_a, judge_b) = await _plain_world()
    summary = await run_clean(world, judge_a, judge_b, with_report=False)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_completed_dry_run_writes_only_the_history_row() -> Json:
    world, (judge_a, judge_b) = await _plain_world()
    summary = await run_clean(world, judge_a, judge_b, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_pool_limit_bounds_the_snapshot_the_watermark_comes_from() -> Json:
    world = await make_world()
    for number in range(1, 5):
        await put_fact(
            world,
            number,
            f"Born in order record [n{number}]",
            created=BORN + timedelta(hours=number),
        )
    await _start_from_watermark(world)
    await world.freeze_seed()
    keep_all = {f"[n{n}]": ByAxis(relevance=shelf("keep-clear")) for n in range(1, 5)}
    judge_a, judge_b = panel(keep_all)
    summary = await run_clean(world, judge_a, judge_b, pool_limit=2)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_blocked_by_failed_batches_stores_the_report_and_holds_the_watermark() -> Json:
    world, _ = await _plain_world()
    judge_a, judge_b = panel(
        {
            "[n1]": ByAxis(relevance=shelf("shelve-clear")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed")),
            "[n3]": ByAxis(relevance=shelf("keep-clear")),
            "[n4]": ByAxis(relevance=shelf("keep-clear")),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
        },
        a_faults=("unavailable",),
        b_faults=(None, "invalid"),
    )
    summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_blocked_without_a_report_callback_writes_no_state() -> Json:
    world, _ = await _plain_world()
    judge_a, judge_b = panel(
        {
            "[n1]": ByAxis(relevance=shelf("keep-clear")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed")),
            "[n3]": ByAxis(relevance=shelf("keep-clear")),
            "[n4]": ByAxis(relevance=shelf("keep-clear")),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
        },
        a_faults=("unavailable",),
    )
    summary = await run_clean(world, judge_a, judge_b, with_report=False)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_blocked_dry_run_writes_only_the_history_row() -> Json:
    world, _ = await _plain_world()
    judge_a, judge_b = panel(
        {
            "[n1]": ByAxis(relevance=shelf("keep-clear")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed")),
            "[n3]": ByAxis(relevance=shelf("keep-clear")),
            "[n4]": ByAxis(relevance=shelf("keep-clear")),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
        },
        b_faults=("unexpected",),
    )
    summary = await run_clean(world, judge_a, judge_b, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_blocked_pass_then_completed_pass_moves_the_watermark_once() -> Json:
    world, (_, _) = await _plain_world()
    first_a, first_b = panel(
        {
            "[n1]": ByAxis(relevance=shelf("keep-clear")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=coexist()),
            "[n3]": ByAxis(relevance=shelf("keep-clear")),
            "[n4]": ByAxis(relevance=shelf("keep-clear")),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
        },
        a_faults=("unavailable",),
    )
    first = await run_clean(world, first_a, first_b)
    second_a, second_b = panel(
        {
            "[n1]": ByAxis(relevance=shelf("shelve-clear", "A: stale")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=coexist("A: both stand")),
            "[n3]": ByAxis(relevance=shelf("keep-clear")),
            "[n4]": ByAxis(relevance=shelf("keep-lean")),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
        },
        {
            "[n1]": ByAxis(relevance=shelf("shelve-lean", "B: stale")),
            "[n2]": ByAxis(relevance=shelf("keep-lean"), falsity=coexist("B: both stand")),
            "[n3]": ByAxis(relevance=shelf("keep-lean")),
            "[n4]": ByAxis(relevance=shelf("keep-clear")),
            "[a1]": ByAxis(revisit=shelf("keep-lean")),
        },
    )
    second = await run_clean(world, second_a, second_b)
    dump = await snapshot_clean(world, second, judge_a=second_a, judge_b=second_b)
    dump["first_pass_outcome"] = first.outcome
    dump["first_pass_reason"] = first.reason
    return dump


async def _verification_world() -> tuple[World, Panel]:
    world = await make_world()
    await put_pair(world, 2)
    await put_pair(world, 6)
    await put_fact(world, 101, "Shelve that silently does nothing [n1]")
    await put_fact(world, 105, "Shelve that lands and then the record vanishes [n5]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judges = panel(
        {
            "[n1]": ByAxis(relevance=shelf("shelve-clear")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed")),
            "[n5]": ByAxis(relevance=shelf("shelve-clear")),
            "[n6]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed")),
        }
    )
    return world, judges


@scenario
async def clean_blocked_by_writes_that_did_not_verify() -> Json:
    world, (judge_a, judge_b) = await _verification_world()

    async def flaky_shelve(**kwargs: Any) -> object:  # forwards kwargs unchanged
        target = kwargs["id"]
        if target == rid(101):
            return None
        result = await set_shelved(**kwargs)
        if target == rid(105):
            await world.adapter.bury(rid(105), "scripted burial")
        return result

    async def flaky_supersede(**kwargs: Any) -> object:  # forwards kwargs unchanged
        successor = kwargs["successor_id"]
        if successor == rid(22):
            return None
        result = await supersede(**kwargs)
        if successor == rid(62):
            await world.adapter.bury(rid(61), "scripted burial")
        return result

    with (
        mock.patch.object(relevance, "set_shelved", flaky_shelve),
        mock.patch.object(falsity, "supersede", flaky_supersede),
    ):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


async def _revisit_failure_world() -> tuple[World, Panel]:
    world = await make_world()
    await put_fact(world, 101, "Newcomer both seats keep [n1]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judges = panel({"[n1]": ByAxis(relevance=shelf("keep-clear"))})
    return world, judges


async def _explode(**kwargs: object) -> object:
    raise RuntimeError("revisit exploded")


@scenario
async def clean_revisit_failure_is_isolated_on_the_initialized_branch() -> Json:
    world = await make_world()
    await put_fact(world, 101, "Newcomer both seats keep [n1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    with mock.patch.object(clean_module, "run_revisit_axis", _explode):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_revisit_failure_is_isolated_on_the_empty_pool_branch() -> Json:
    world = await make_world()
    await _start_from_watermark(world)
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    with mock.patch.object(clean_module, "run_revisit_axis", _explode):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_revisit_failure_is_isolated_on_the_completed_branch() -> Json:
    world, (judge_a, judge_b) = await _revisit_failure_world()
    with mock.patch.object(clean_module, "run_revisit_axis", _explode):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_revisit_failure_is_isolated_on_the_blocked_branch() -> Json:
    world, _ = await _revisit_failure_world()
    judge_a, judge_b = panel(
        {"[n1]": ByAxis(relevance=shelf("keep-clear"))}, a_faults=("unavailable",)
    )
    with mock.patch.object(clean_module, "run_revisit_axis", _explode):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


async def _assessor_clean_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    await put_pair(world, 2)
    await put_pair(world, 3)
    await put_pair(world, 6)
    await put_pair(world, 7)
    await put_fact(world, 101, "Stale ticket note that both seats shelve [n1]")
    await put_fact(world, 104, "Note the seats disagree about, annotated [n4]")
    await put_fact(world, 105, "Note the seat cannot read [n5]")
    await aged(world, 301, "Old note the revisit seats keep [a1]")
    await aged(world, 302, "Old note the revisit seats shelve [a2]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judges = panel(
        {
            "[n1]": ByAxis(relevance=shelf("shelve-clear", "A: stale")),
            "[n2]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed", "A: moved")),
            "[n3]": ByAxis(
                relevance=shelf("shelve-lean", "A: lean shelve"), falsity=coexist("A: both stand")
            ),
            "[n4]": ByAxis(relevance=shelf("shelve-clear", "A: obsolete")),
            "[n5]": ByAxis(relevance=shelf("keep-lean", unclear=True)),
            "[n6]": ByAxis(relevance=shelf("keep-clear"), falsity=replace("changed", "A: moved")),
            "[n7]": ByAxis(
                relevance=shelf("keep-clear"), falsity=replace("correction", "A: wrong")
            ),
            "[a1]": ByAxis(revisit=shelf("keep-clear")),
            "[a2]": ByAxis(revisit=shelf("shelve-clear", "A: obsolete")),
        },
        {
            "[n1]": ByAxis(relevance=shelf("shelve-lean", "B: stale")),
            "[n2]": ByAxis(relevance=shelf("keep-lean"), falsity=replace("changed", "B: moved")),
            "[n3]": ByAxis(relevance=shelf("keep-clear"), falsity=coexist("B: both stand")),
            "[n4]": ByAxis(relevance=shelf("keep-lean", "B: still true")),
            "[n5]": ByAxis(relevance=shelf("keep-clear")),
            "[n6]": ByAxis(relevance=shelf("keep-lean"), falsity=coexist("B: still true")),
            "[n7]": ByAxis(relevance=shelf("keep-lean"), falsity=replace("changed", "B: moved")),
            "[a1]": ByAxis(revisit=shelf("keep-lean")),
            "[a2]": ByAxis(revisit=shelf("shelve-lean", "B: stale")),
        },
    )
    assessor = ScriptedAssessor(
        relevance={
            "[n1]": relevance_pick("shelve-clear", 0.97),
            "[n2]": relevance_pick("keep-clear", 0.97),
            "[n3]": relevance_pick("keep-lean", 0.8),
            "[n4]": relevance_pick("shelve-lean", 0.8),
            "[n5]": relevance_pick("keep-lean", 0.8),
            "[n6]": relevance_pick("keep-clear", 0.97),
            "[n7]": relevance_pick("keep-clear", 0.97),
        },
        contradiction={
            "[o6]": contradiction_pick("correction", 0.9),
            "[o7]": contradiction_pick("coexist", 0.9),
        },
    )
    return world, judges, assessor


@scenario
async def clean_completed_pass_with_an_assessor_on_every_axis() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_clean_world()
    summary = await run_clean(world, judge_a, judge_b, assessor=assessor)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b, assessor=assessor)


@scenario
async def clean_completed_dry_run_with_an_assessor_on_every_axis() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_clean_world()
    summary = await run_clean(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b, assessor=assessor)


@scenario
async def clean_resolves_the_default_rubric_and_judge_seats() -> Json:
    world = await make_world()
    await _start_from_watermark(world)
    await world.freeze_seed()
    with mock.patch.object(clean_module, "load_rubric", lambda: golden_rubric()):
        summary = await run_clean(world, None, None, rubric=None)
    return await snapshot_clean(world, summary)


@scenario
async def clean_refuses_a_scope_that_is_not_registered() -> Json:
    world = await make_world()
    await world.freeze_seed()

    async def unregistered(scope: object) -> None:
        raise ToolError(f"Unregistered scope {scope!r}")

    raised = await expect_raise(
        clean_module.clean(
            scope="proj-a",
            adapter=world.adapter,
            embedder=world.embedder,
            clock=world.clock,
            lock=world.lock,
            cfg=EngineConfig(),
            require_scope=unregistered,
            rubric=world.rubric,
            judge_a=panel({})[0],
            judge_b=panel({})[1],
        )
    )
    return await snapshot_clean(world, None, raised=raised)


@scenario
async def clean_refuses_a_malformed_scope() -> Json:
    world = await make_world()
    await world.freeze_seed()
    raised = await expect_raise(
        clean_module.clean(
            scope="Not A Scope",
            adapter=world.adapter,
            embedder=world.embedder,
            clock=world.clock,
            lock=world.lock,
            cfg=EngineConfig(),
            require_scope=lambda scope: _accept(),
            rubric=world.rubric,
            judge_a=panel({})[0],
            judge_b=panel({})[1],
        )
    )
    return await snapshot_clean(world, None, raised=raised)


async def _accept() -> None:
    return None


@scenario
async def clean_pass_history_append_failure_is_logged_and_the_pass_still_returns() -> Json:
    world, (judge_a, judge_b) = await _plain_world()

    async def dead_history(*, record: object) -> None:
        raise RuntimeError("pass history store down")

    with mock.patch.object(world.adapter, "append_cleaner_pass", dead_history):
        summary = await run_clean(world, judge_a, judge_b)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_fallback_payload_spends_one_error_budget_across_the_three_axes() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=1, revisit_batch_cap=1))
    for number in range(1, 11):
        await put_fact(world, number, f"Failing record [n{number}]")
    await aged(world, 301, "Old record whose batch fails [a1]")
    await aged(world, 302, "Old record whose batch fails too [a2]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    items: dict[str, ByAxis] = {
        f"[n{number}]": ByAxis(relevance=shelf("keep-clear")) for number in range(1, 11)
    }
    items["[a1]"] = ByAxis(revisit=shelf("keep-clear"))
    items["[a2]"] = ByAxis(revisit=shelf("keep-clear"))
    judge_a, judge_b = panel(
        items,
        a_faults=("unavailable",) * 12,
        fault_padding=" " + "long diagnostic " * 40 + "THE-LIVE-FAILURE-LINE",
    )
    summary = await run_clean(world, judge_a, judge_b, dry_run=True)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)


@scenario
async def clean_fallback_payload_renders_revisit_failures_unverified_writes_and_refusals() -> Json:
    world = await make_world(rubric=golden_rubric(revisit_batch_cap=2))
    for number in range(1, 6):
        await aged(world, number, f"Revisit record [x{number}]")
    await _start_from_watermark(world)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[x1]": ByAxis(revisit=shelf("shelve-clear")),
            "[x2]": ByAxis(revisit=shelf("shelve-clear")),
            "[x3]": ByAxis(revisit=shelf("shelve-clear")),
            "[x4]": ByAxis(revisit=shelf("shelve-clear")),
            "[x5]": ByAxis(revisit=shelf("keep-clear")),
        },
        {
            "[x1]": ByAxis(revisit=shelf("shelve-lean")),
            "[x2]": ByAxis(revisit=shelf("shelve-lean")),
            "[x3]": ByAxis(revisit=shelf("keep-clear")),
            "[x4]": ByAxis(revisit=shelf("keep-clear")),
            "[x5]": ByAxis(revisit=shelf("keep-clear")),
        },
        b_faults=(None, None, "invalid"),
        a_on_call=invalidate_on_call(world, 0, rid(2)),
    )

    async def flaky_shelve(**kwargs: Any) -> object:  # forwards kwargs unchanged
        if kwargs["id"] == rid(1):
            return None
        return await set_shelved(**kwargs)

    with (
        mock.patch.object(revisit, "set_shelved", flaky_shelve),
        flaky_appends(
            world.adapter,
            review_when=lambda review: review.outcome == "judged" and review.target_id == rid(3),
        ),
    ):
        summary = await run_clean(world, judge_a, judge_b, with_report=False)
    return await snapshot_clean(world, summary, judge_a=judge_a, judge_b=judge_b)
