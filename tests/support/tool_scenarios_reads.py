"""Tool-golden scenarios for the read tools: ``recall``, ``decision``, ``decisions_query`` and
``context``.

``recall`` runs against records with different feedback so the importance boost applies, under a
healthy embedder and a down one, with archived records that revive or vanish mid-call. ``context``
covers each role, both push tiers, the budget packer and its refusals. The scenarios that leave an
archived record archived after a revival attempt, a ``decision`` view that follows a chain to its
tip and the lazily built token counter of ``context`` are here too. The tools that write take no
part: a scenario seeds its store with ``ToolWorld.write``, which calls the engine method.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from unittest import mock

from tabularium.domain import HypomnemaId
from tabularium.engine import core
from tests.support.doubles import FakeTokenCounter
from tests.support.tool_golden_assessor import ToolAssessor
from tests.support.tool_golden_world import (
    Json,
    ToolWorld,
    cluster,
    decision,
    ember,
    expect,
    fact,
    golden_config,
    inject,
    make_world,
    saved_id,
    scenario,
)

_GONE = str(uuid.UUID(int=0x404))


def _rid(value: str) -> HypomnemaId:
    return HypomnemaId(uuid.UUID(value))


async def _seed_deployment_corpus(w: ToolWorld) -> dict[str, str]:
    ids = {
        "pipeline": await fact(w, "the deployment pipeline runs on every merge to main"),
        "secrets": await fact(w, "deployment pipeline secrets live in the vault"),
        "rollback": await fact(w, "rollback a deployment by pushing the previous tag"),
        "global": await fact(w, "deployment windows are closed on fridays", scope="global"),
        "other": await fact(w, "deployment pipeline of the other project", scope="q"),
        "decision": await decision(w, "decision: the deployment pipeline uses blue green"),
    }
    await w.write("feedback", {"id": ids["secrets"], "value": 1.0})
    await w.write("feedback", {"id": ids["secrets"], "value": 1.0})
    await w.write("feedback", {"id": ids["rollback"], "value": 0.0})
    return ids


@scenario
async def recall__ranking_limit_boost_and_read_events() -> Json:
    w = await make_world()
    ids = await _seed_deployment_corpus(w)

    await w.call("recall", {"query": "deployment pipeline", "scope": "p"})
    await w.call("recall", {"query": "deployment pipeline", "scope": "p", "limit": 2})
    await w.call("recall", {"query": "deployment pipeline", "scope": "global"})
    await w.call("recall", {"query": "deployment pipeline", "scope": "q", "limit": 1})
    await w.call("recall", {"query": "nothing matches this at all zebra", "scope": "p"})
    await w.call("decision", {"id": ids["decision"]})
    return await w.finish()


@scenario
async def recall__invalid_superseded_and_include_invalid() -> Json:
    w = await make_world()
    old = await fact(w, "the build runs on node 18")
    new = await fact(w, "the build runs on node 22")
    wrong = await fact(w, "the build runs on node 14")
    await w.write("supersede", {"successor_id": new, "predecessor_ids": [old], "reason": "changed"})
    await w.write("set_validity", {"id": wrong, "valid": False, "reason": "never true"})

    await w.call("recall", {"query": "the build runs on node", "scope": "p"})
    await w.call(
        "recall", {"query": "the build runs on node", "scope": "p", "include_invalid": True}
    )
    return await w.finish()


@scenario
async def recall__refusals() -> Json:
    w = await make_world()
    await fact(w, "a fact so the store is not empty")

    await w.call("recall", {"query": "", "scope": "p"})
    await w.call("recall", {"query": "   ", "scope": "p"})
    await w.call("recall", {"query": "anything", "scope": "nowhere"})
    await w.call("recall", {"query": "anything", "scope": "Not A Scope"})
    await w.call("recall", {"query": "anything", "scope": "p", "limit": "many"})
    return await w.finish()


@scenario
async def recall__embedder_down_degrades_to_text() -> Json:
    w = await make_world()
    await _seed_deployment_corpus(w)

    w.embedder.arm()
    await w.call("recall", {"query": "deployment pipeline", "scope": "p"})
    await w.call("recall", {"query": "zebra", "scope": "p"})
    w.embedder.lift()
    await w.call("recall", {"query": "deployment pipeline", "scope": "p", "limit": 3})
    return await w.finish()


async def _archive(w: ToolWorld, record_id: str) -> None:
    await w.adapter.transition(_rid(record_id), "archived", at=w.clock.peek())


@scenario
async def recall__archived_fallback_revives_by_importance() -> Json:
    w = await make_world()
    loved = await fact(w, "the staging cluster restarts nightly at two")
    ignored = await fact(w, "the staging cluster has two spare nodes")
    await fact(w, "an unrelated active fact about parrots")
    for _ in range(3):
        await w.write("feedback", {"id": loved, "value": 1.0})
    await _archive(w, loved)
    await _archive(w, ignored)

    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    w.note("lifecycles after the first recall", await _lifecycles(w, [loved, ignored]))
    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    return await w.finish()


async def _lifecycles(w: ToolWorld, ids: list[str]) -> dict[str, str]:
    return {record_id: (await w.adapter.get_by_id(_rid(record_id))).lifecycle for record_id in ids}


@scenario
async def recall__include_archived_never_revives() -> Json:
    w = await make_world()
    archived = await fact(w, "the staging cluster restarts nightly at two")
    await w.write("feedback", {"id": archived, "value": 1.0})
    await _archive(w, archived)

    await w.call("recall", {"query": "staging cluster", "scope": "p", "include_archived": True})
    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    return await w.finish()


@scenario
async def recall__archived_invalid_is_returned_read_only() -> Json:
    w = await make_world()
    archived = await fact(w, "the staging cluster restarts nightly at two")
    await w.write("feedback", {"id": archived, "value": 1.0})
    await w.write("set_validity", {"id": archived, "valid": False, "reason": "retired"})
    await _archive(w, archived)

    await w.call("recall", {"query": "staging cluster", "scope": "p", "include_invalid": True})
    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    return await w.finish()


@scenario
async def recall__revival_races() -> Json:
    w = await make_world()
    buried = await fact(w, "the staging cluster restarts nightly at two")
    moved = await fact(w, "the staging cluster has two spare nodes")
    gone_after_reinforce = await fact(w, "the staging cluster lives in frankfurt")
    gone_after_transition = await fact(w, "the staging cluster uses spot instances")
    for record_id in (buried, moved, gone_after_reinforce, gone_after_transition):
        await w.write("feedback", {"id": record_id, "value": 1.0})
        await _archive(w, record_id)

    inject(w, "get_by_id", KeyError("buried"), call=1)
    await w.call("recall", {"query": "staging cluster", "scope": "p", "limit": 1})
    w.note("lifecycles", await _lifecycles(w, [buried, moved]))

    original = w.adapter.get_by_id
    revived_elsewhere = {"done": False}

    async def get_by_id_after_a_rival_revival(record_id: HypomnemaId) -> object:
        if not revived_elsewhere["done"]:
            revived_elsewhere["done"] = True
            await w.adapter.transition(record_id, "active", at=w.clock.peek())
        return await original(record_id)

    with mock.patch.object(w.adapter, "get_by_id", get_by_id_after_a_rival_revival):
        await w.call("recall", {"query": "staging cluster", "scope": "p", "limit": 1})

    inject(w, "get_by_id", KeyError("buried"), call=2)
    await w.call("recall", {"query": "staging cluster", "scope": "p", "limit": 3})
    inject(w, "get_by_id", KeyError("buried"), call=3)
    await w.call("recall", {"query": "staging cluster", "scope": "p", "limit": 4})
    return await w.finish()


@scenario
async def decision__view_chain_and_refusals() -> Json:
    w = await make_world()
    old = await decision(w, cluster("decision: use postgres for the ledger", 3, 0))
    new = saved_id(
        await w.write(
            "save",
            {
                "content": cluster("decision: use postgres 16 for the ledger", 3, 1),
                "scope": "p",
                "kind": "decision",
                "supersedes": old,
                "reason": "elaborated",
                "note": "pinned a version",
            },
        )
    )
    fact_id = await fact(w, "a fact is no decision")

    await w.call("decision", {"id": old})
    await w.call("decision", {"id": new})
    await w.call("decision", {"id": fact_id})
    await w.call("decision", {"id": _GONE})
    await w.call("decision", {"id": "not-a-uuid"})
    return await w.finish()


@scenario
async def decision__view_of_promoted_decision_and_revalidations() -> Json:
    w = await make_world(
        assessor=ToolAssessor("correction"),
        config=golden_config(auto_exit_replacement=True),
    )
    promoted_from = saved_id(
        await ember(w, "always pin the lockfile", valence="guideline", audience_hint=["builder"])
    )
    promoted = saved_id(
        await w.write(
            "save",
            {
                "content": "decision: pin the lockfile in every repository",
                "scope": "p",
                "kind": "decision",
                "promoted_from": [promoted_from],
            },
        )
    )
    invalidated = await decision(w, cluster("decision: ship weekly releases", 2, 0))
    expect(
        await w.write(
            "save", {"content": cluster("fact: releases are now fortnightly", 2, 1), "scope": "p"}
        ),
        "saved",
    )
    await w.write(
        "set_validity", {"id": invalidated, "valid": True, "reason": "the fact was wrong"}
    )

    await w.call("decision", {"id": promoted})
    await w.call("decision", {"id": invalidated})
    return await w.finish()


@scenario
async def decisions_query__links_filters_and_refusals() -> Json:
    w = await make_world()
    await decision(w, cluster("decision: use postgres for the ledger", 3, 0))
    revoked = await decision(w, "decision: cache sessions in redis")
    invalid = await decision(w, "decision: use graphql for the public api")
    await decision(w, "decision: the other project uses sqlite", scope="q")
    superseded = await decision(w, cluster("decision: ledger backups run hourly", 4, 0))
    saved_id(
        await w.write(
            "save",
            {
                "content": cluster("decision: ledger backups run every ten minutes", 4, 1),
                "scope": "p",
                "kind": "decision",
                "supersedes": superseded,
                "reason": "changed",
            },
        )
    )
    await w.write("revoke", {"id": revoked, "reason": "dropped"})
    await w.write("set_validity", {"id": invalid, "valid": False, "reason": "rejected"})
    await fact(w, "a fact about the ledger that decisions_query must not return")

    await w.call("decisions_query", {"query": "ledger postgres backups", "scope": "p"})
    await w.call(
        "decisions_query",
        {
            "query": "ledger postgres backups redis graphql",
            "scope": "p",
            "include_superseded": True,
        },
    )
    await w.call(
        "decisions_query",
        {"query": "ledger postgres backups redis graphql", "scope": "p", "include_invalid": True},
    )
    await w.call(
        "decisions_query",
        {
            "query": "ledger postgres backups redis graphql",
            "scope": "p",
            "include_superseded": True,
            "include_invalid": True,
            "limit": 3,
        },
    )
    await w.call("decisions_query", {"query": "sqlite", "scope": "q"})
    await w.call("decisions_query", {"query": "", "scope": "p"})
    await w.call("decisions_query", {"query": "ledger", "scope": "nowhere"})
    return await w.finish()


@scenario
async def decisions_query__embedder_down() -> Json:
    w = await make_world()
    await decision(w, "decision: use postgres for the ledger")
    w.embedder.arm()
    await w.call("decisions_query", {"query": "postgres ledger", "scope": "p"})
    return await w.finish()


async def _seed_push_tiers(w: ToolWorld) -> dict[str, str]:
    ids = {
        "hot_builder": await fact(w, "always run the linter before pushing", scope="p"),
        "hot_both": await fact(w, "never commit secrets to the repository", scope="p"),
        "hot_global": await fact(w, "write commit messages in the imperative", scope="global"),
        "hot_invalid": await fact(w, "a hot fact that later turns invalid", scope="p"),
        "hot_other_scope": await fact(w, "a hot fact of the other project", scope="q"),
        "hot_no_role": await fact(w, "a hot fact that reaches no role", scope="p"),
        "plain": await fact(w, "an ordinary fact that never reaches the payload", scope="p"),
    }
    await w.write("set_hot", {"id": ids["hot_builder"], "hot": True, "roles": ["builder"]})
    await w.write("set_hot", {"id": ids["hot_both"], "hot": True, "roles": ["builder", "reviewer"]})
    await w.write(
        "set_hot", {"id": ids["hot_global"], "hot": True, "roles": ["builder", "decider"]}
    )
    await w.write("set_hot", {"id": ids["hot_invalid"], "hot": True, "roles": ["builder"]})
    await w.write("set_hot", {"id": ids["hot_other_scope"], "hot": True, "roles": ["builder"]})
    await w.write("set_hot", {"id": ids["hot_no_role"], "hot": True, "roles": []})
    await w.write("set_validity", {"id": ids["hot_invalid"], "valid": False, "reason": "wrong"})
    await w.write("feedback", {"id": ids["hot_both"], "value": 1.0})
    await w.write("feedback", {"id": ids["hot_global"], "value": 0.2})
    ids["ember_fresh"] = saved_id(
        await ember(w, "never force push to a shared branch", audience_hint=["builder", "reviewer"])
    )
    ids["ember_agent"] = saved_id(await ember(w, "prefer small pull requests", valence="guideline"))
    ids["ember_other"] = saved_id(
        await ember(w, "ember of the other project", scope="q", audience_hint=["builder"])
    )
    return ids


@scenario
async def context__roles_tiers_and_packing() -> Json:
    w = await make_world()
    await _seed_push_tiers(w)

    for role in ("builder", "reviewer", "decider", "navigator"):
        await w.call("context", {"scope": "p", "budget": 400, "role": role})
    await w.call("context", {"scope": "global", "budget": 400, "role": "builder"})
    await w.call("context", {"scope": "q", "budget": 400, "role": "builder"})
    await w.call("context", {"scope": "p", "budget": 12, "role": "builder"})
    await w.call("context", {"scope": "p", "budget": 0, "role": "builder"})
    await w.call("context", {"scope": "p", "budget": 25, "role": "reviewer"})
    return await w.finish()


@scenario
async def context__ember_fades_below_the_floor() -> Json:
    w = await make_world()
    await ember(w, "never force push to a shared branch", audience_hint=["builder"])
    await w.call("context", {"scope": "p", "budget": 200, "role": "builder"})
    w.clock.advance(timedelta(days=400))
    await w.call("context", {"scope": "p", "budget": 200, "role": "builder"})
    return await w.finish()


@scenario
async def context__refusals() -> Json:
    w = await make_world()
    await fact(w, "a fact so the store is not empty")

    await w.call("context", {"scope": "p", "budget": 100})
    await w.call("context", {"scope": "p", "budget": 100, "role": "nobody"})
    await w.call("context", {"scope": "nowhere", "budget": 100, "role": "builder"})
    await w.call("context", {"scope": "Not A Scope", "budget": 100, "role": "builder"})
    await w.call("context", {"scope": "p", "budget": -1, "role": "builder"})
    await w.call("context", {"scope": "p", "budget": 1.5, "role": "builder"})
    return await w.finish()


@scenario
async def decision__view_follows_a_chain_to_its_tip_and_stops_at_a_gap() -> Json:
    w = await make_world()
    first = await decision(w, cluster("decision: version one of the ledger store", 2, 0))
    second = saved_id(
        await w.write(
            "save",
            {
                "content": cluster("decision: version two of the ledger store", 2, 1),
                "scope": "p",
                "kind": "decision",
                "supersedes": first,
                "reason": "changed",
            },
        )
    )
    third = saved_id(
        await w.write(
            "save",
            {
                "content": cluster("decision: version three of the ledger store", 2, 0) + " x",
                "scope": "p",
                "kind": "decision",
                "supersedes": second,
                "reason": "changed",
            },
        )
    )
    await w.call("decision", {"id": first})
    await w.write("bury", {"id": third, "reason": "the tip is gone"})
    await w.call("decision", {"id": first})
    return await w.finish()


@scenario
async def recall__revival_reinforces_but_stays_archived_below_the_threshold() -> Json:
    w = await make_world(config=golden_config(w_recall=0.0, w_recency=0.0))
    quiet = await fact(w, "the staging cluster restarts nightly at two")
    await w.adapter.transition(_rid(quiet), "archived", at=w.clock.peek())
    w.clock.advance(timedelta(days=400))
    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    await w.call("recall", {"query": "staging cluster", "scope": "p"})
    return await w.finish()


@scenario
async def core__context_builds_its_token_counter_on_first_use() -> Json:
    with mock.patch.object(core, "LitellmTokenCounter", FakeTokenCounter):
        w = await make_world(inject_token_counter=False)
        await fact(w, "alpha fact so the store is not empty")
        await w.call("context", {"scope": "p", "budget": 50, "role": "builder"})
        await w.call("context", {"scope": "p", "budget": 50, "role": "builder"})
        return await w.finish()
