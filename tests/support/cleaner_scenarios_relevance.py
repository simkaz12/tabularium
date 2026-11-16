"""Relevance-axis scenarios for the cleaner golden. Importing the module registers them."""

from __future__ import annotations

import asyncio
from unittest import mock

from tabularium.config import EngineConfig
from tabularium.engine import cleaner_writes, relevance
from tabularium.engine import judge as judge_mod
from tabularium.engine.judge import render_batch_prompt
from tabularium.engine.relevance import RelevanceVerdict
from tests.support.cleaner_golden import (
    ScriptedAssessor,
    golden_rubric,
    relevance_pick,
    shelf,
)
from tests.support.cleaner_scenario_kit import (
    Panel,
    failing_saves,
    flaky_appends,
    hot_on_call,
    panel,
)
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
    run_relevance,
    scenario,
    snapshot,
)


async def _bands_world(*, trim: bool) -> tuple[World, Panel]:
    world = await make_world()
    keep_text = "Disk alerts fire at eighty percent" if trim else None
    await put_fact(world, 1, "Disk usage alert threshold is 80 percent [r1]")
    await put_fact(world, 2, "Backups run nightly at two [r2]")
    await put_fact(world, 3, "Staging box is called pinky [r3]")
    await put_fact(world, 4, "The wiki lives on the old host [r4]")
    await put_fact(world, 5, "Pager rotation changes on monday [r5]")
    await put_fact(world, 6, "Release freeze starts friday [r6]")
    await put_decision(world, 7, "Use postgres for the ledger [r7]", created=BORN)
    await put_fact(world, 8, "Pinned deploy checklist [r8]", lifecycle="hot")
    await put_fact(world, 9, "Already shelved note [r9]", lifecycle="shelved")
    await put_fact(world, 10, "Ember complaint about linting [r10]", lifecycle="ember")
    await put_fact(world, 11, "Shelved with a keep that is only whitespace [r11]")
    await world.freeze_seed()
    judges = panel(
        {
            "[r1]": shelf("shelve-clear", "A: ticket noise", keep=keep_text),
            "[r2]": shelf("keep-clear", keep="A keep text on a keep grade is dropped"),
            "[r3]": shelf("keep-lean", unclear=True),
            "[r4]": shelf("shelve-clear", "A: obsolete host"),
            "[r5]": shelf("keep-lean"),
            "[r6]": shelf("keep-clear"),
            "[r11]": shelf("shelve-clear", "A: stale", keep="   "),
        },
        {
            "[r1]": shelf("shelve-lean", "B: stale"),
            "[r2]": shelf("keep-lean"),
            "[r3]": shelf("keep-clear"),
            "[r4]": shelf("keep-clear", "B: still true"),
            "[r5]": shelf("shelve-clear", "B: out of date"),
            "[r6]": shelf("shelve-lean", unclear=True),
            "[r11]": shelf("shelve-lean", "B: stale"),
        },
    )
    return world, judges


@scenario
async def relevance_bands_without_assessor() -> Json:
    world, (judge_a, judge_b) = await _bands_world(trim=True)
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_bands_without_assessor_dry_run() -> Json:
    world, (judge_a, judge_b) = await _bands_world(trim=True)
    summary = await run_relevance(world, judge_a, judge_b, dry_run=True)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary, dry_run=True)


@scenario
async def relevance_empty_pool() -> Json:
    world = await make_world()
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_global_pass_includes_global_facts() -> Json:
    world = await make_world(scope="global")
    await put_fact(world, 1, "Owner prefers short commit subjects [r1]")
    await put_fact(world, 2, "Owner reads release notes on friday [r2]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[r1]": shelf("shelve-clear"), "[r2]": shelf("keep-clear")},
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_explicit_pool_dry_run_judges_any_lifecycle() -> Json:
    world = await make_world()
    hot = await put_fact(world, 1, "Hot record handed in directly [r1]", lifecycle="hot")
    shelved = await put_fact(
        world, 2, "Shelved record handed in directly [r2]", lifecycle="shelved"
    )
    active = await put_fact(world, 3, "Active record handed in directly [r3]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[r1]": shelf("shelve-clear", keep="Hot part that survives"),
            "[r2]": shelf("keep-clear"),
            "[r3]": shelf("shelve-lean"),
        },
        {
            "[r1]": shelf("shelve-lean"),
            "[r2]": shelf("shelve-clear"),
            "[r3]": shelf("shelve-clear"),
        },
    )
    summary = await run_relevance(
        world, judge_a, judge_b, dry_run=True, explicit_pool=[hot, shelved, active]
    )
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary, dry_run=True)


@scenario
async def relevance_explicit_pool_refuses_a_live_run() -> Json:
    world = await make_world()
    fact = await put_fact(world, 1, "Record handed in directly [r1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[r1]": shelf("shelve-clear")})
    raised = await expect_raise(run_relevance(world, judge_a, judge_b, explicit_pool=[fact]))
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, raised=raised)


async def _assessor_panel_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    await put_fact(world, 1, "Assessed keep, panel keeps [a1]")
    await put_fact(world, 2, "Assessed shelve, panel unclear [a2]")
    await put_fact(world, 3, "Assessed shelve, panel shelves [a3]")
    await put_fact(world, 4, "Assessor times out, panel shelves [a4]")
    await put_fact(world, 5, "Assessed keep, panel splits [a5]")
    await put_fact(world, 6, "Assessed shelve, panel splits [a6]")
    await put_fact(world, 7, "Assessor times out, panel splits [a7]")
    await put_fact(world, 8, "Assessed keep, panel shelves [a8]")
    await put_fact(world, 9, "Assessor errors, panel keeps [a9]")
    await world.freeze_seed()
    judges = panel(
        {
            "[a1]": shelf("keep-clear"),
            "[a2]": shelf("shelve-lean", unclear=True),
            "[a3]": shelf("shelve-clear", "A: gone"),
            "[a4]": shelf("shelve-clear", "A: gone too"),
            "[a5]": shelf("shelve-clear"),
            "[a6]": shelf("shelve-lean"),
            "[a7]": shelf("shelve-clear"),
            "[a8]": shelf("shelve-lean", "A: lean shelve"),
            "[a9]": shelf("keep-lean"),
        },
        {
            "[a1]": shelf("keep-lean"),
            "[a2]": shelf("keep-clear"),
            "[a3]": shelf("shelve-lean"),
            "[a4]": shelf("shelve-clear"),
            "[a5]": shelf("keep-clear"),
            "[a6]": shelf("keep-lean"),
            "[a7]": shelf("keep-clear"),
            "[a8]": shelf("shelve-clear"),
            "[a9]": shelf("keep-clear"),
        },
    )
    assessor = ScriptedAssessor(
        relevance={
            "[a1]": relevance_pick("keep-lean", 0.9),
            "[a2]": relevance_pick("shelve-lean", 0.8),
            "[a3]": relevance_pick("shelve-clear", 0.97),
            "[a4]": TimeoutError("assessor timed out"),
            "[a5]": relevance_pick("keep-lean", 0.7),
            "[a6]": relevance_pick("shelve-lean", 0.7),
            "[a7]": TimeoutError("assessor timed out"),
            "[a8]": relevance_pick("keep-clear", 0.5),
            "[a9]": RuntimeError("assessor exploded"),
        }
    )
    return world, judges, assessor


@scenario
async def relevance_assessor_rows_for_every_panel_band() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_panel_world()
    summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_assessor_rows_for_every_panel_band_dry_run() -> Json:
    world, (judge_a, judge_b), assessor = await _assessor_panel_world()
    summary = await run_relevance(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot(
        world,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        relevance=summary,
        dry_run=True,
    )


async def _skip_and_split_world(*, race: bool) -> tuple[World, Panel, ScriptedAssessor]:
    cfg = EngineConfig(
        cleaner_prepass_skip_keep=True,
        cleaner_prepass_skip_cutoff=0.9,
        cleaner_split_shelve=True,
    )
    world = await make_world(cfg=cfg, rubric=golden_rubric(batch_cap=3))
    await put_fact(world, 1, "Skipped keep one [s1]")
    await put_fact(world, 2, "Skipped keep two [s2]")
    await put_fact(world, 3, "Skipped keep three [s3]")
    await put_fact(world, 4, "Keep below the cutoff goes to the panel [t1]")
    await put_fact(world, 5, "Shelve-clear always goes to the panel [t2]")
    await put_fact(world, 6, "Split shelved with a surviving part [t3]")
    t4 = await put_fact(world, 7, "Split shelve refused by a racing hot promotion [t4]")
    await put_fact(world, 8, "Split shelve whose trim save fails [t5 boom]")
    await put_fact(world, 9, "Split with an unavailable assessor [t6]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[t1]": shelf("keep-clear"),
            "[t2]": shelf("shelve-clear", "A: gone"),
            "[t3]": shelf("shelve-clear", "A: partly gone", keep="The surviving part of t3"),
            "[t4]": shelf("shelve-clear"),
            "[t5": shelf("shelve-clear", keep="The surviving part of t5 boom"),
            "[t6]": shelf("shelve-clear"),
        },
        {
            "[t1]": shelf("keep-lean"),
            "[t2]": shelf("shelve-lean"),
            "[t3]": shelf("keep-clear"),
            "[t4]": shelf("keep-lean"),
            "[t5": shelf("keep-clear"),
            "[t6]": shelf("keep-lean"),
        },
        a_on_call=hot_on_call(world, 1, t4.id) if race else None,
    )
    assessor = ScriptedAssessor(
        relevance={
            "[s1]": relevance_pick("keep-clear", 0.95),
            "[s2]": relevance_pick("keep-clear", 0.99),
            "[s3]": relevance_pick("keep-clear", 0.93),
            "[t1]": relevance_pick("keep-clear", 0.5),
            "[t2]": relevance_pick("shelve-clear", 0.99),
            "[t3]": relevance_pick("shelve-lean", 0.8),
            "[t4]": relevance_pick("shelve-lean", 0.8),
            "[t5": relevance_pick("shelve-lean", 0.8),
            "[t6]": TimeoutError("assessor timed out"),
        }
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def relevance_assessor_skip_switch_and_split_shelve() -> Json:
    world, (judge_a, judge_b), assessor = await _skip_and_split_world(race=True)
    with failing_saves("boom", "raise"):
        summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_assessor_skip_switch_and_split_shelve_dry_run() -> Json:
    world, (judge_a, judge_b), assessor = await _skip_and_split_world(race=False)
    summary = await run_relevance(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot(
        world,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        relevance=summary,
        dry_run=True,
    )


@scenario
async def relevance_retry_recovers_after_invalid_batch_answer() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Retry recovers shelve [r1]")
    await put_fact(world, 2, "Retry recovers keep [r2]")
    await put_fact(world, 3, "Retry recovers split [r3]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[r1]": shelf("shelve-clear"),
            "[r2]": shelf("keep-clear"),
            "[r3]": shelf("shelve-lean"),
        },
        {
            "[r1]": shelf("shelve-lean"),
            "[r2]": shelf("keep-lean"),
            "[r3]": shelf("keep-clear"),
        },
        a_faults=("invalid",),
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_invalid_answers_drop_one_record_per_seat() -> Json:
    world = await make_world()
    for number in range(1, 5):
        await put_fact(world, number, f"Dropped positions record [r{number}]")
    await world.freeze_seed()
    keep_all = {f"[r{number}]": shelf("keep-clear") for number in range(1, 5)}
    judge_a, judge_b = panel(
        {**keep_all, "[r2]": "invalid"},
        {**keep_all, "[r4]": "invalid"},
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_invalid_answer_drops_a_record_for_seat_a_only() -> Json:
    world = await make_world()
    for number in range(1, 4):
        await put_fact(world, number, f"Dropped by seat A record [r{number}]")
    await world.freeze_seed()
    keep_all = {f"[r{number}]": shelf("shelve-clear") for number in range(1, 4)}
    judge_a, judge_b = panel({**keep_all, "[r3]": "invalid"}, keep_all)
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_unavailable_seat_a_fails_its_batch_and_the_next_is_judged() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=2))
    for number in range(1, 5):
        await put_fact(world, number, f"Batch failure record [r{number}]")
    await world.freeze_seed()
    keep_all = {f"[r{number}]": shelf("keep-clear") for number in range(1, 5)}
    judge_a, judge_b = panel(keep_all, a_faults=("unavailable",))
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_invalid_seat_a_and_unexpected_seat_b_fail_the_batch() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Mixed failure record one [r1]")
    await put_fact(world, 2, "Mixed failure record two [r2]")
    await world.freeze_seed()
    keep_all = {"[r1]": shelf("keep-clear"), "[r2]": shelf("keep-clear")}
    judge_a, judge_b = panel(keep_all, a_faults=("invalid",), b_faults=("unexpected",))
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_retry_hard_failure_of_seat_a_fails_the_batch() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Retry hard failure one [r1]")
    await put_fact(world, 2, "Retry hard failure two [r2]")
    await world.freeze_seed()
    keep_all = {"[r1]": shelf("keep-clear"), "[r2]": shelf("keep-clear")}
    judge_a, judge_b = panel(keep_all, a_faults=("invalid", "unavailable"))
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_fallback_hard_failure_of_seat_a_fails_the_batch() -> Json:
    world = await make_world()
    for number in range(1, 4):
        await put_fact(world, number, f"Fallback hard failure record [r{number}]")
    await world.freeze_seed()
    keep_all = {f"[r{number}]": shelf("keep-clear") for number in range(1, 4)}
    judge_a, judge_b = panel(keep_all, a_faults=("invalid", "invalid", None, "unavailable"))
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_hard_failure_of_seat_b_after_seat_a_dropped_a_record() -> Json:
    world = await make_world()
    for number in range(1, 4):
        await put_fact(world, number, f"Seat B hard failure record [r{number}]")
    await world.freeze_seed()
    keep_all = {f"[r{number}]": shelf("keep-clear") for number in range(1, 4)}
    judge_a, judge_b = panel(
        {**keep_all, "[r2]": "invalid"},
        keep_all,
        b_faults=("invalid", "unavailable"),
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_hard_failure_of_seat_b_after_a_clean_resolution() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Seat B retry crash one [r1]")
    await put_fact(world, 2, "Seat B retry crash two [r2]")
    await world.freeze_seed()
    keep_all = {"[r1]": shelf("keep-clear"), "[r2]": shelf("keep-clear")}
    judge_a, judge_b = panel(keep_all, b_faults=("invalid", "unexpected"))
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_oversize_prompt_fails_its_batch_and_the_next_is_judged() -> Json:
    world = await make_world(rubric=golden_rubric(batch_cap=1))
    await put_fact(world, 1, "Oversize record " + "padding " * 100 + "[r1]")
    await put_fact(world, 2, "Small record that fits [r2]")
    await world.freeze_seed()
    base = len(
        render_batch_prompt(
            rubric_prose=world.rubric.relevance.prose, contents=[], item_model=RelevanceVerdict
        )
    )
    keep_all = {"[r1]": shelf("keep-clear"), "[r2]": shelf("keep-clear")}
    judge_a, judge_b = panel(keep_all)
    with mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 300):
        summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_trim_saved_covered_and_taken_from_either_seat() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Stale chatter plus a rule [t1]")
    await put_fact(world, 2, "More chatter plus another rule [t2]")
    await put_fact(world, 3, "Even more chatter plus a third rule [t3]")
    await put_fact(world, 4, "Chatter that repeats a known rule [t4]")
    await put_fact(world, 5, "Plain stale record with no surviving part [t5]")
    await put_fact(world, 6, "Established guidance on pager hygiene", created=OLDER)
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[t1]": shelf("shelve-clear", keep="Rotate credentials quarterly"),
            "[t2]": shelf("shelve-clear"),
            "[t3]": shelf("shelve-lean", keep="Seat A text for the third rule"),
            "[t4]": shelf("shelve-clear", keep="Established guidance on pager hygiene"),
            "[t5]": shelf("shelve-clear"),
        },
        {
            "[t1]": shelf("shelve-lean"),
            "[t2]": shelf("shelve-lean", keep="Seat B text for the second rule"),
            "[t3]": shelf("shelve-clear", keep="Seat B text for the third rule"),
            "[t4]": shelf("shelve-lean"),
            "[t5]": shelf("shelve-lean"),
        },
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


async def _trim_failure_world() -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world()
    await put_fact(world, 1, "Active record with a surviving part [t1]")
    await put_fact(world, 2, "Archived record with a surviving part [t2]", lifecycle="archived")
    await put_fact(world, 3, "Record whose trim save is fine [t3]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[t1]": shelf("shelve-clear", keep="Survivor of t1 boom"),
            "[t2]": shelf("shelve-clear", keep="Survivor of t2 boom"),
            "[t3]": shelf("shelve-clear", keep="Survivor of t3 fine"),
        }
    )
    assessor = ScriptedAssessor(
        relevance={
            "[t1]": relevance_pick("shelve-clear", 0.9),
            "[t2]": relevance_pick("shelve-clear", 0.9),
            "[t3]": relevance_pick("shelve-clear", 0.9),
        }
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def relevance_trim_save_bounced_reverts_the_shelve() -> Json:
    world, (judge_a, judge_b), assessor = await _trim_failure_world()
    with failing_saves("boom", "bounce"):
        summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_trim_save_raising_reverts_the_shelve() -> Json:
    world, (judge_a, judge_b), assessor = await _trim_failure_world()
    with failing_saves("boom", "raise"):
        summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_without_an_embedder_the_shelve_stands_untrimmed() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Chatter with a surviving part but no embedder [r1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[r1]": shelf("shelve-clear", keep="Part that cannot be saved")})
    summary = await run_relevance(world, judge_a, judge_b, with_embedder=False)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_shelve_refused_when_the_record_went_hot_mid_pass() -> Json:
    world = await make_world()
    racer = await put_fact(world, 1, "Promoted to hot while the judges ran [r1]")
    await put_fact(world, 2, "Ordinary record that still shelves [r2]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[r1]": shelf("shelve-clear"), "[r2]": shelf("shelve-clear")},
        a_on_call=hot_on_call(world, 0, racer.id),
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_panel_shelve_refused_when_the_record_went_hot_mid_pass() -> Json:
    world = await make_world()
    racer = await put_fact(world, 1, "Promoted to hot while the judges ran [r1]")
    await put_fact(world, 2, "Ordinary record that still shelves [r2]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[r1]": shelf("shelve-clear"), "[r2]": shelf("shelve-clear")},
        a_on_call=hot_on_call(world, 0, racer.id),
    )
    assessor = ScriptedAssessor(
        relevance={
            "[r1]": relevance_pick("shelve-clear", 0.9),
            "[r2]": relevance_pick("shelve-clear", 0.9),
        }
    )
    summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_storage_failure_during_the_shelve_aborts_the_pass() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Record whose shelve hits a dead store [r1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[r1]": shelf("shelve-clear")})

    async def dead_store(**kwargs: object) -> object:
        raise RuntimeError("store unavailable")

    with mock.patch.object(relevance, "set_shelved", dead_store):
        raised = await expect_raise(run_relevance(world, judge_a, judge_b))
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, raised=raised)


@scenario
async def relevance_contested_review_and_judgment_appends_fail_independently() -> Json:
    world = await make_world()
    for number in range(1, 5):
        await put_fact(world, number, f"Contested record [c{number}]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {f"[c{number}]": shelf("shelve-clear", f"A on c{number}") for number in range(1, 5)},
        {f"[c{number}]": shelf("keep-clear", f"B on c{number}") for number in range(1, 5)},
    )
    with flaky_appends(
        world.adapter,
        review_when=lambda review: review.target_id in {rid(1), rid(2)},
        judgment_when=lambda judgment: judgment.target_id in {rid(1), rid(3)},
    ):
        summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_split_keep_review_append_failure_falls_back_to_contested() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Split keep whose resolving review fails [k1]")
    await put_fact(world, 2, "Split keep that resolves cleanly [k2]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {"[k1]": shelf("shelve-clear"), "[k2]": shelf("shelve-clear")},
        {"[k1]": shelf("keep-clear"), "[k2]": shelf("keep-clear")},
    )
    assessor = ScriptedAssessor(
        relevance={
            "[k1]": relevance_pick("keep-lean", 0.7),
            "[k2]": relevance_pick("keep-lean", 0.7),
        }
    )
    with flaky_appends(
        world.adapter,
        review_when=lambda review: (
            review.outcome == "assessor_resolved" and review.target_id == rid(1)
        ),
    ):
        summary = await run_relevance(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, relevance=summary
    )


@scenario
async def relevance_trim_cancelled_mid_save_reverts_the_shelve() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Chatter with a surviving part, cancelled mid-save [r1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[r1]": shelf("shelve-clear", keep="Part that never lands")})

    async def cancelled(**kwargs: object) -> object:
        raise asyncio.CancelledError

    raised: BaseException | None = None
    with mock.patch.object(cleaner_writes, "save_internal", cancelled):
        try:
            await run_relevance(world, judge_a, judge_b)
        except asyncio.CancelledError as exc:
            raised = exc
    assert raised is not None
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, raised=raised)


@scenario
async def relevance_schema_violations_name_the_verdict_class_in_the_call_error() -> Json:
    world = await make_world()
    for number in range(1, 4):
        await put_fact(world, number, f"Schema violation record [r{number}]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[r1]": shelf("keep-clear"),
            "[r2]": {"grade": "bogus-grade", "why": "not on the scale", "unclear": False},
            "[r3]": {
                "grade": "keep-clear",
                "why": "unclear must be a real boolean",
                "unclear": "no",
            },
        },
        {f"[r{number}]": shelf("keep-clear") for number in range(1, 4)},
    )
    summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)


@scenario
async def relevance_refusals_of_different_kinds_keep_their_collection_order() -> Json:
    world = await make_world()
    await put_fact(world, 1, "Shelve whose trim save fails first in the pool [r1 boom]")
    racer = await put_fact(world, 2, "Promoted to hot while the judges ran [r2]")
    await put_fact(world, 3, "Ordinary record that still shelves [r3]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[r1": shelf("shelve-clear", keep="Survivor of r1 boom"),
            "[r2]": shelf("shelve-clear"),
            "[r3]": shelf("shelve-clear"),
        },
        a_on_call=hot_on_call(world, 0, racer.id),
    )
    with failing_saves("boom", "raise"):
        summary = await run_relevance(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, relevance=summary)
