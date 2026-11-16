"""Revisit-axis scenarios for the cleaner golden. Importing the module registers them."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest import mock

from tabularium.config import EngineConfig
from tabularium.domain import Fact, FactLifecycle
from tabularium.engine import judge as judge_mod
from tabularium.engine import revisit
from tabularium.engine.assessor import Unavailable
from tabularium.engine.dispositions import set_shelved
from tabularium.engine.judge import render_batch_prompt
from tabularium.engine.revisit import RevisitVerdict
from tabularium.engine.verdicts import build_cleaner_resolved_row
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
    invalidate_on_call,
    panel,
)
from tests.support.cleaner_world import (
    AGED,
    BORN,
    Json,
    World,
    make_world,
    put_event,
    put_fact,
    put_review,
    put_session,
    rid,
    run_revisit,
    scenario,
    snapshot,
)

EVIDENCE_AT = AGED + timedelta(days=10)


async def aged(
    world: World,
    number: int,
    content: str,
    *,
    evidence: bool = True,
    lifecycle: FactLifecycle = "active",
    created: datetime = AGED,
) -> Fact:
    """A record old enough for the revisit pool, with one later event as its evidence."""
    fact = await put_fact(world, number, content, created=created, lifecycle=lifecycle)
    if evidence:
        await put_event(world, fact.id, created + timedelta(days=10))
    return fact


async def _bands_world() -> tuple[World, Panel]:
    world = await make_world()
    await aged(world, 1, "Shelved by both seats [v1]")
    await aged(world, 2, "Kept by both seats [v2]")
    await aged(world, 3, "Unclear to seat B [v3]")
    await aged(world, 4, "Contested between the seats [v4]")
    protected = await aged(world, 5, "Shelved and restored twice by the owner [v5]")
    await put_event(world, protected.id, EVIDENCE_AT + timedelta(days=1), "shelved")
    await put_event(world, protected.id, EVIDENCE_AT + timedelta(days=2), "shelved")
    await aged(world, 6, "Nothing happened to this one [v6]", evidence=False)
    streak = await aged(world, 7, "Kept twice before and kept again [v7]")
    for reviewed, due in (
        (datetime(2025, 2, 1, tzinfo=AGED.tzinfo), datetime(2025, 3, 1, tzinfo=AGED.tzinfo)),
        (datetime(2025, 3, 2, tzinfo=AGED.tzinfo), datetime(2025, 4, 1, tzinfo=AGED.tzinfo)),
    ):
        await put_review(world, streak.id, reviewed_at=reviewed, band="keep", due=due)
    contested_before = await aged(world, 8, "Contested before and kept now [v8]")
    await put_review(
        world,
        contested_before.id,
        reviewed_at=datetime(2025, 3, 1, tzinfo=AGED.tzinfo),
        band="contested",
        due=datetime(2025, 5, 1, tzinfo=AGED.tzinfo),
    )
    await aged(world, 10, "Archived record that is kept [v10]", lifecycle="archived")
    await aged(world, 11, "ok fine yes [v11]")
    await put_session(
        world,
        1,
        "Planning session. The seats met about lunch.",
        started=AGED + timedelta(days=3),
        ended=AGED + timedelta(days=3, hours=1),
    )
    not_due = await aged(world, 9, "Reviewed last week, not due yet [v9]")
    await put_review(
        world,
        not_due.id,
        reviewed_at=datetime(2025, 5, 25, tzinfo=AGED.tzinfo),
        band="keep",
        due=datetime(2025, 9, 1, tzinfo=AGED.tzinfo),
    )
    await world.freeze_seed()
    judges = panel(
        {
            "[v1]": shelf("shelve-clear", "A: obsolete"),
            "[v2]": shelf("keep-clear", keep="A keep text on a keep grade is dropped"),
            "[v3]": shelf("keep-lean"),
            "[v4]": shelf("shelve-clear", "A: obsolete"),
            "[v7]": shelf("keep-clear"),
            "[v8]": shelf("keep-lean"),
            "[v10]": shelf("keep-clear"),
            "[v11]": shelf("keep-lean"),
        },
        {
            "[v1]": shelf("shelve-lean", "B: stale", keep="  "),
            "[v2]": shelf("keep-lean"),
            "[v3]": shelf("shelve-lean", unclear=True),
            "[v4]": shelf("keep-clear", "B: still true"),
            "[v7]": shelf("keep-clear"),
            "[v8]": shelf("keep-clear"),
            "[v10]": shelf("keep-lean"),
            "[v11]": shelf("keep-clear"),
        },
    )
    return world, judges


@scenario
async def revisit_bands_without_assessor() -> Json:
    world, (judge_a, judge_b) = await _bands_world()
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_bands_without_assessor_dry_run() -> Json:
    world, (judge_a, judge_b) = await _bands_world()
    summary = await run_revisit(world, judge_a, judge_b, dry_run=True)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary, dry_run=True)


@scenario
async def revisit_empty_pool() -> Json:
    world = await make_world()
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


async def _split_world(*, shelve: bool, race: bool = True) -> tuple[World, Panel, ScriptedAssessor]:
    world = await make_world(cfg=EngineConfig(cleaner_split_shelve=shelve))
    reused = {
        1: ("keep-lean", 0.7),
        3: ("shelve-lean", 0.7),
        6: ("shelve-lean", 0.7),
        8: ("shelve-lean", 0.7),
        10: ("keep-lean", 0.7),
    }
    contents = {
        1: "Split with a reused keep row [p1]",
        2: "Split with a fresh keep call [p2]",
        3: "Split with a reused shelve row [p3]",
        4: "Split with a fresh shelve call [p4]",
        5: "Split with an unavailable assessor [p5]",
        6: "Split shelved with a surviving part [p6]",
        7: "Split shelve whose trim save fails [p7 boom]",
        8: "Split shelve invalidated mid-pass [p8]",
        9: "Split keep whose resolving review fails [p9]",
        10: "Split with a reused row hidden behind a newer unavailable row [p10]",
    }
    for number, content in contents.items():
        await aged(world, number, content)
    for number, (pick, confidence) in reused.items():
        await world.adapter.append_verdicts(
            [
                build_cleaner_resolved_row(
                    caller="cleaner_prepass",
                    candidate_id=rid(number),
                    verdict=relevance_pick(pick, confidence),
                    action="panel_keep" if pick.startswith("keep") else "panel_shelve",
                    actor="panel",
                    matched=True,
                    now=AGED + timedelta(days=5),
                )
            ]
        )
    await world.adapter.append_verdicts(
        [
            build_cleaner_resolved_row(
                caller="cleaner_prepass",
                candidate_id=rid(10),
                verdict=Unavailable(
                    method="relevance", reason="timeout", detail=None, latency_ms=0
                ),
                action="panel_contested",
                actor="panel",
                matched=None,
                now=AGED + timedelta(days=6),
            ),
            build_cleaner_resolved_row(
                caller="cleaner_split",
                candidate_id=rid(10),
                verdict=relevance_pick("shelve-clear", 0.9),
                action="split_annotated",
                actor="engine",
                matched=False,
                now=AGED + timedelta(days=7),
            ),
        ]
    )
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            f"[p{n}": shelf(
                "shelve-clear",
                keep=f"Surviving part of p{n}" + (" boom" if n == 7 else "")
                if n in (6, 7)
                else None,
            )
            for n in contents
        },
        {f"[p{n}": shelf("keep-clear") for n in contents},
        a_on_call=invalidate_on_call(world, 0, rid(8)) if race else None,
    )
    assessor = ScriptedAssessor(
        relevance={
            "[p2]": relevance_pick("keep-lean", 0.7),
            "[p4]": relevance_pick("shelve-lean", 0.7),
            "[p5]": TimeoutError("assessor timed out"),
            "[p7": relevance_pick("shelve-lean", 0.7),
            "[p9]": relevance_pick("keep-lean", 0.7),
        }
    )
    return world, (judge_a, judge_b), assessor


@scenario
async def revisit_assessor_splits_with_split_shelve_off() -> Json:
    world, (judge_a, judge_b), assessor = await _split_world(shelve=False)
    with flaky_appends(
        world.adapter,
        review_when=lambda review: (
            review.outcome == "assessor_resolved" and review.target_id == rid(9)
        ),
    ):
        summary = await run_revisit(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, revisit=summary
    )


@scenario
async def revisit_assessor_splits_with_split_shelve_on() -> Json:
    world, (judge_a, judge_b), assessor = await _split_world(shelve=True)
    with (
        failing_saves("boom", "raise"),
        flaky_appends(
            world.adapter,
            review_when=lambda review: (
                review.outcome == "assessor_resolved" and review.target_id == rid(9)
            ),
        ),
    ):
        summary = await run_revisit(world, judge_a, judge_b, assessor=assessor)
    return await snapshot(
        world, judge_a=judge_a, judge_b=judge_b, assessor=assessor, revisit=summary
    )


@scenario
async def revisit_assessor_splits_with_split_shelve_on_dry_run() -> Json:
    world, (judge_a, judge_b), assessor = await _split_world(shelve=True, race=False)
    summary = await run_revisit(world, judge_a, judge_b, assessor=assessor, dry_run=True)
    return await snapshot(
        world,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        revisit=summary,
        dry_run=True,
    )


async def _shelve_write_world() -> tuple[World, Panel]:
    world = await make_world()
    await aged(world, 1, "Shelved with a surviving part [w1]")
    await aged(world, 2, "Shelved with a part that is already covered [w2]")
    await aged(world, 3, "Shelved with a part whose save fails [w3 boom]")
    await aged(world, 4, "Shelved but invalidated mid-pass [w4]")
    await aged(world, 5, "Shelved with a review that fails to append [w5]")
    await aged(world, 6, "Contested with a judgment that fails to append [w6]")
    await aged(world, 7, "Contested with a review that fails to append [w7]")
    await put_fact(world, 8, "Established guidance on pager hygiene", created=BORN)
    await world.freeze_seed()
    shelved_a = {
        "[w1]": shelf("shelve-clear", keep="Rotate credentials quarterly"),
        "[w2]": shelf("shelve-clear", keep="Established guidance on pager hygiene"),
        "[w3": shelf("shelve-clear", keep="Surviving part of w3 boom"),
        "[w4]": shelf("shelve-clear"),
        "[w5]": shelf("shelve-clear"),
        "[w6]": shelf("shelve-clear"),
        "[w7]": shelf("shelve-clear"),
    }
    judges = panel(
        shelved_a,
        {
            "[w1]": shelf("shelve-lean"),
            "[w2]": shelf("shelve-lean"),
            "[w3": shelf("shelve-lean"),
            "[w4]": shelf("shelve-lean"),
            "[w5]": shelf("shelve-lean"),
            "[w6]": shelf("keep-clear"),
            "[w7]": shelf("keep-clear"),
        },
        a_on_call=invalidate_on_call(world, 0, rid(4)),
    )
    return world, judges


@scenario
async def revisit_unanimous_shelve_writes_and_their_failures() -> Json:
    world, (judge_a, judge_b) = await _shelve_write_world()
    with (
        failing_saves("boom", "raise"),
        flaky_appends(
            world.adapter,
            review_when=lambda review: (
                review.outcome == "judged" and review.target_id in {rid(5), rid(7)}
            ),
            judgment_when=lambda judgment: judgment.target_id == rid(6),
        ),
    ):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_without_an_embedder_the_shelve_stands_untrimmed() -> Json:
    world = await make_world()
    await aged(world, 1, "Shelved with a part that cannot be saved [w1]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[w1]": shelf("shelve-clear", keep="Part that cannot be saved")})
    summary = await run_revisit(world, judge_a, judge_b, with_embedder=False)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


async def _oversize_world(*, batch_cap: int) -> tuple[World, int]:
    world = await make_world(rubric=golden_rubric(revisit_batch_cap=batch_cap))
    await aged(world, 1, "Oversize record " + "padding " * 100 + "[o1]")
    await aged(world, 2, "Small record that fits [o2]")
    await aged(world, 3, "Another small record that fits [o3]")
    await world.freeze_seed()
    base = len(
        render_batch_prompt(
            rubric_prose=world.rubric.relevance.prose + "\n\n" + world.rubric.revisit.prose,
            contents=[],
            item_model=RevisitVerdict,
        )
    )
    return world, base


@scenario
async def revisit_lone_oversize_record_is_skipped_with_a_review_row() -> Json:
    world, base = await _oversize_world(batch_cap=1)
    judge_a, judge_b = panel({"[o2]": shelf("keep-clear"), "[o3]": shelf("keep-clear")})
    with mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 500):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_lone_oversize_record_dry_run_writes_no_review_row() -> Json:
    world, base = await _oversize_world(batch_cap=1)
    judge_a, judge_b = panel({"[o2]": shelf("keep-clear"), "[o3]": shelf("keep-clear")})
    with mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 500):
        summary = await run_revisit(world, judge_a, judge_b, dry_run=True)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary, dry_run=True)


@scenario
async def revisit_lone_oversize_record_review_append_failure_is_recorded() -> Json:
    world, base = await _oversize_world(batch_cap=1)
    judge_a, judge_b = panel({"[o2]": shelf("keep-clear"), "[o3]": shelf("keep-clear")})
    with (
        mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 500),
        flaky_appends(
            world.adapter,
            review_when=lambda review: review.outcome == "skipped_too_large",
        ),
    ):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_oversize_two_record_batch_fails_as_a_whole() -> Json:
    world, base = await _oversize_world(batch_cap=2)
    judge_a, judge_b = panel(
        {"[o1]": shelf("keep-clear"), "[o2]": shelf("keep-clear"), "[o3]": shelf("keep-clear")}
    )
    with mock.patch.object(judge_mod, "PROMPT_CHAR_CEILING", base + 700):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_failed_batches_whatever_the_status() -> Json:
    world = await make_world(rubric=golden_rubric(revisit_batch_cap=1))
    for number in range(1, 6):
        await aged(world, number, f"Failed batch record [f{number}]")
    await world.freeze_seed()
    keep_all = {f"[f{number}]": shelf("keep-clear") for number in range(1, 6)}
    judge_a, judge_b = panel(
        keep_all,
        a_faults=("unavailable", None, "unexpected", None, None),
        b_faults=(None, "invalid", "invalid", None, None),
    )
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_protected_and_unevidenced_review_appends_fail() -> Json:
    world = await make_world()
    for number in (1, 2):
        protected = await aged(world, number, f"Protected record [q{number}]")
        await put_event(world, protected.id, EVIDENCE_AT + timedelta(days=1), "shelved")
        await put_event(world, protected.id, EVIDENCE_AT + timedelta(days=2), "shelved")
    await aged(world, 3, "Unevidenced record one [q3]", evidence=False)
    await aged(world, 4, "Unevidenced record two [q4]", evidence=False)
    await world.freeze_seed()
    judge_a, judge_b = panel({})
    with flaky_appends(
        world.adapter,
        review_when=lambda review: review.target_id in {rid(2), rid(4)},
    ):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_history_block_carries_sessions() -> Json:
    world = await make_world()
    subject = "Deployment pipeline uses blue green rollouts [h1]"
    await put_fact(world, 1, subject, created=AGED)
    await put_fact(
        world,
        2,
        "Deployment pipeline owner is the platform team [h2]",
        created=AGED - timedelta(days=60),
    )
    await put_session(
        world,
        1,
        "Birth digest of the working session. " + "x" * 9000,
        started=AGED - timedelta(hours=1),
        ended=AGED + timedelta(hours=2),
    )
    for number in range(2, 8):
        await put_session(
            world,
            number,
            f"Later session {number} revisits the deployment pipeline. " + "y" * 6900,
            started=AGED + timedelta(days=number),
            ended=AGED + timedelta(days=number, hours=1),
        )
    await put_session(
        world,
        8,
        "Unrelated session about lunch plans and nothing else.",
        started=AGED + timedelta(days=9),
        ended=AGED + timedelta(days=9, hours=1),
    )
    await world.freeze_seed()
    judge_a, judge_b = panel({"[h1]": shelf("keep-clear"), "[h2]": shelf("keep-lean")})
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_shelve_that_did_not_land_fails_verification() -> Json:
    world = await make_world()
    await aged(world, 1, "Shelve silently does nothing [x1]")
    await aged(world, 2, "Shelve lands and the record then vanishes [x2]")
    await aged(world, 3, "Shelve lands and stays [x3]")
    await world.freeze_seed()
    judge_a, judge_b = panel({f"[x{n}]": shelf("shelve-clear") for n in range(1, 4)})

    async def flaky_shelve(**kwargs: Any) -> object:  # forwards kwargs unchanged
        target = kwargs["id"]
        if target == rid(1):
            return None
        result = await set_shelved(**kwargs)
        if target == rid(2):
            await world.adapter.bury(rid(2), "scripted burial")
        return result

    with mock.patch.object(revisit, "set_shelved", flaky_shelve):
        summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


_ = (BORN,)


@scenario
async def revisit_schema_violation_fails_the_batch_with_the_verdict_class_in_the_error() -> Json:
    world = await make_world()
    await aged(world, 1, "Schema violation record one [s1]")
    await aged(world, 2, "Schema violation record two [s2]")
    await world.freeze_seed()
    judge_a, judge_b = panel(
        {
            "[s1]": shelf("keep-clear"),
            "[s2]": {"grade": "bogus-grade", "why": "not on the scale", "unclear": False},
        },
        {"[s1]": shelf("keep-clear"), "[s2]": shelf("keep-clear")},
    )
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)


@scenario
async def revisit_global_pass_revisits_global_records() -> Json:
    world = await make_world(scope="global")
    await aged(world, 1, "Owner rule judged in a global pass [g1]")
    await aged(world, 2, "Owner rule kept in a global pass [g2]")
    await world.freeze_seed()
    judge_a, judge_b = panel({"[g1]": shelf("shelve-clear"), "[g2]": shelf("keep-clear")})
    summary = await run_revisit(world, judge_a, judge_b)
    return await snapshot(world, judge_a=judge_a, judge_b=judge_b, revisit=summary)
