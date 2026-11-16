"""The world a cleaner-golden scenario runs in: store, clock, seeds, runners and the dump.

A scenario builds a ``World``, seeds records, runs one axis or ``clean()``, and returns
``await snapshot(...)``. Nothing in the dump is normalized here: the clock ticks deterministically,
``uuid4`` and the wall-clock timers are pinned by ``pinned_environment``, and every list keeps the
order the store emitted it in.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import sys
import uuid
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal
from unittest import mock

from tabularium import cli
from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, FactLifecycle, HypomnemaId, Scope, Session
from tabularium.engine.assessor import Assessor
from tabularium.engine.axis_common import BandCounts
from tabularium.engine.clean import CleanSummary, clean, fallback_payload
from tabularium.engine.falsity import (
    FalsityAxisSummary,
    FalsityBandCounts,
    run_falsity_axis,
)
from tabularium.engine.judge import JudgeAdapter
from tabularium.engine.relevance import RelevanceAxisSummary, run_relevance_axis
from tabularium.engine.report import (
    build_report,
    render_pass_record,
    run_clean_pass,
)
from tabularium.engine.revisit import (
    RevisitAxisSummary,
    run_revisit_axis,
)
from tabularium.engine.rubric import Rubric
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import (
    CleanerPassRecord,
    RecordReview,
    ReinforcementEvent,
    VerdictRow,
)
from tests.support.cleaner_golden import (
    ClusterEmbedder,
    ScriptedAssessor,
    ScriptedJudge,
    TickingClock,
    golden_rubric,
)
from tests.support.doubles import FakeTokenCounter, TestClock

Json = dict[str, Any]  # a JSON document being built for the golden

CLOCK_START = datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
PASS_AT = datetime(2025, 6, 1, 11, 0, 0, tzinfo=UTC)
SINCE = datetime(2025, 5, 1, tzinfo=UTC)
BORN = datetime(2025, 5, 10, 8, 0, 0, tzinfo=UTC)
OLDER = datetime(2025, 4, 1, 8, 0, 0, tzinfo=UTC)
AGED = datetime(2025, 1, 15, 8, 0, 0, tzinfo=UTC)

_PINNED_UUID_BASE = 0xABC000000
_ORIGINAL_UUID4 = uuid.uuid4


def rid(number: int) -> HypomnemaId:
    return HypomnemaId(uuid.UUID(int=number))


class UuidPin:
    """A ``uuid4`` stand-in that counts from a fixed base and can be reset per scenario."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)

    def reset(self) -> None:
        self._counter = itertools.count(1)

    def __call__(self) -> uuid.UUID:
        return uuid.UUID(int=_PINNED_UUID_BASE + next(self._counter))


@contextmanager
def pinned_environment(pin: UuidPin) -> Iterator[None]:
    """Pin the sources of nondeterminism the scenarios reach.

    - ``uuid.uuid4`` and every ``tabularium`` module that imported the function by name;
    - the ``time`` module as seen by ``engine.judge`` and ``engine.admission`` (durations and
      latencies), replaced by a stub whose ``monotonic`` never advances;
    - ``LitellmTokenCounter`` in the three axis modules, replaced by the ``len // 4`` counter.
    """
    frozen_time = SimpleNamespace(monotonic=lambda: 0.0)
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(uuid, "uuid4", pin))
        for name, module in list(sys.modules.items()):
            if not name.startswith("tabularium"):
                continue
            if getattr(module, "uuid4", None) is _ORIGINAL_UUID4:
                stack.enter_context(mock.patch.object(module, "uuid4", pin))
        for name in ("tabularium.engine.judge", "tabularium.engine.admission"):
            stack.enter_context(mock.patch.object(sys.modules[name], "time", frozen_time))
        for name in (
            "tabularium.engine.relevance",
            "tabularium.engine.falsity",
            "tabularium.engine.revisit",
        ):
            stack.enter_context(
                mock.patch.object(sys.modules[name], "LitellmTokenCounter", FakeTokenCounter)
            )
        yield


@dataclass
class World:
    adapter: MemoryAdapter
    clock: TickingClock
    lock: asyncio.Lock
    cfg: EngineConfig
    rubric: Rubric
    embedder: ClusterEmbedder
    scope: Scope
    seeded: dict[HypomnemaId, Json] = field(default_factory=dict)
    seeded_counts: dict[str, int] = field(default_factory=dict)

    async def freeze_seed(self) -> None:
        self.seeded = {record.id: dump_record(record) for record in await all_records(self.adapter)}
        self.seeded_counts = {
            "reviews": len(self.adapter._record_reviews),
            "judgments": len(self.adapter._record_judgments),
            "events": len(self.adapter._events),
        }


async def make_world(
    *,
    cfg: EngineConfig | None = None,
    rubric: Rubric | None = None,
    scope: str = "proj-a",
) -> World:
    embedder = ClusterEmbedder()
    adapter = MemoryAdapter(
        embedding_model=embedder.model,
        dim=embedder.dim,
        clock=TestClock(CLOCK_START),
    )
    for name in ("proj-a", "global"):
        await adapter.create_scope(Scope(name))
    return World(
        adapter=adapter,
        clock=TickingClock(CLOCK_START),
        lock=asyncio.Lock(),
        cfg=cfg if cfg is not None else EngineConfig(),
        rubric=rubric if rubric is not None else golden_rubric(),
        embedder=embedder,
        scope=Scope(scope),
    )


async def put_fact(
    world: World,
    number: int,
    content: str,
    *,
    created: datetime = BORN,
    scope: str | None = None,
    lifecycle: FactLifecycle = "active",
    superseded_by: HypomnemaId | None = None,
    purpose: str | None = None,
) -> Fact:
    fact = Fact(
        id=rid(number),
        type="fact",
        content=content,
        scope=Scope(scope if scope is not None else str(world.scope)),
        lifecycle=lifecycle,
        validity="invalid" if superseded_by is not None else "valid",
        superseded_by=superseded_by,
        purpose=purpose,
        created_at=created,
        updated_at=created,
    )
    await world.adapter.upsert(fact, world.embedder.vector(content))
    return fact


async def put_decision(
    world: World,
    number: int,
    content: str,
    *,
    created: datetime = OLDER,
    scope: str | None = None,
) -> Decision:
    decision = Decision(
        id=rid(number),
        type="decision",
        content=content,
        scope=Scope(scope if scope is not None else str(world.scope)),
        lifecycle="in_force",
        validity="valid",
        created_at=created,
        updated_at=created,
    )
    await world.adapter.upsert(decision, world.embedder.vector(content))
    return decision


async def put_session(
    world: World,
    number: int,
    content: str,
    *,
    started: datetime,
    ended: datetime,
) -> None:
    await world.adapter.append_session(
        Session(
            id=uuid.UUID(int=0x5000 + number),
            scope=str(world.scope),
            started_at=started,
            ended_at=ended,
            content=content,
        )
    )


async def put_event(
    world: World,
    target: HypomnemaId,
    at: datetime,
    source: Literal["shelved", "shelve_reverted", "dedup"] = "dedup",
) -> None:
    await world.adapter.append_event(ReinforcementEvent(target_id=target, at=at, source=source))


async def put_review(
    world: World,
    target: HypomnemaId,
    *,
    reviewed_at: datetime,
    band: Literal["keep", "contested", "shelve", "unclear"] | None,
    due: datetime,
    scope: str | None = None,
) -> None:
    await world.adapter.append_review(
        RecordReview(
            target_id=target,
            scope=Scope(scope if scope is not None else str(world.scope)),
            reviewed_at=reviewed_at,
            next_review_due_at=due,
            outcome="judged",
            band=band,
            grades=(),
            rubric_version=world.rubric.version,
        )
    )


def dump_record(record: Fact | Decision) -> Json:
    """Every field of the stored record."""
    dumped: Json = record.model_dump(mode="json")
    return dumped


async def all_records(adapter: MemoryAdapter) -> list[Fact | Decision]:
    return [record async for record in await adapter.scan_records()]


def _verdict_row(row: VerdictRow) -> Json:
    dumped: Json = row.model_dump(mode="json")
    return dumped


def _dump_model_list(models: Sequence[Any]) -> list[Json]:  # any pydantic model list
    dumped: list[Json] = [model.model_dump(mode="json") for model in models]
    return dumped


async def dump_store(world: World) -> Json:
    adapter = world.adapter
    records = await all_records(adapter)
    touched = [
        dumped
        for record in records
        if world.seeded.get(record.id) != (dumped := dump_record(record))
    ]
    reviews = adapter._record_reviews[world.seeded_counts.get("reviews", 0) :]
    judgments = adapter._record_judgments[world.seeded_counts.get("judgments", 0) :]
    events = adapter._events[world.seeded_counts.get("events", 0) :]
    return {
        "touched_records": touched,
        "verdict_rows": [_verdict_row(row) for row in adapter._verdicts.values()],
        "record_reviews": _dump_model_list(reviews),
        "record_judgments": _dump_model_list(judgments),
        "reinforcement_events": _dump_model_list(events),
        "unsupersedes": _dump_model_list(adapter._unsupersedes),
        "cleaner_state": dump_cleaner_state(adapter),
        "pass_history": _dump_model_list(adapter._cleaner_pass_history),
    }


def dump_cleaner_state(adapter: MemoryAdapter) -> Json:
    return {
        str(scope): state.model_dump(mode="json") for scope, state in adapter._cleaner_state.items()
    }


def dump_judge(judge: ScriptedJudge) -> list[Json]:
    return [
        {
            "call": number,
            "fault": judge.call_faults[number],
            "chars": len(prompt),
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "items": prompt.count("\n## "),
        }
        for number, prompt in enumerate(judge.prompts)
    ]


def idle_relevance(world: World) -> RelevanceAxisSummary:
    return RelevanceAxisSummary(
        scope=world.scope,
        rubric_version=world.rubric.version,
        judge_pair=("idle-a", "idle-b"),
        pool_size=0,
        excluded_decisions=(),
        excluded_global=(),
        excluded_already_shelved=(),
        counts=BandCounts(),
        shelved_ids=(),
        unclear_ids=(),
        failed_batches=(),
        calls=(),
    )


def idle_falsity(world: World) -> FalsityAxisSummary:
    return FalsityAxisSummary(
        scope=world.scope,
        rubric_version=world.rubric.version,
        judge_pair=("idle-a", "idle-b"),
        newcomer_pool_size=0,
        excluded_decisions=(),
        excluded_global=(),
        excluded_shelved_this_pass=(),
        counts=FalsityBandCounts(),
        edges=(),
        decision_candidates=(),
        global_candidates=(),
        hot_or_ember_candidates=(),
        failed_batches=(),
        calls=(),
    )


def compose_summary(
    world: World,
    *,
    relevance: RelevanceAxisSummary | None = None,
    falsity: FalsityAxisSummary | None = None,
    revisit: RevisitAxisSummary | None = None,
    dry_run: bool = False,
) -> CleanSummary:
    """Wrap the axis summaries a scenario produced in a ``completed`` pass so the report and the
    CLI text render them; an axis the scenario did not run is an idle one.
    """
    return CleanSummary(
        scope=world.scope,
        outcome="completed",
        watermark_before=SINCE,
        watermark_after=PASS_AT,
        pool_size=relevance.pool_size if relevance is not None else 0,
        relevance=relevance if relevance is not None else idle_relevance(world),
        falsity=falsity if falsity is not None else idle_falsity(world),
        revisit=revisit,
        reason="composed for the golden",
        dry_run=dry_run,
    )


def render_all(world: World, summary: CleanSummary) -> Json:
    """The renderings of one ``CleanSummary``: report JSON, fallback payload, CLI text and the
    context summary of the pass-history row a live pass would have written.
    """
    fallback = fallback_payload(summary)
    rendered: Json = {
        "cli_text": cli._render_clean_summary(summary).split("\n"),
        "fallback_payload_json": fallback,
    }
    report = build_report(summary, pass_at=PASS_AT)
    if report is None:
        rendered["cleaner_report_json"] = None
        return rendered

    report_json = report.model_dump_json()
    rendered["cleaner_report_json"] = report_json
    payload = fallback if summary.dry_run else report_json
    row = CleanerPassRecord(
        scope=summary.scope,
        pass_at=PASS_AT,
        outcome=summary.outcome,
        dry_run=summary.dry_run,
        payload=payload,
    )
    rendered["context_summary"] = render_pass_record(
        row, cap=world.rubric.parameters.report_render_cap
    ).model_dump(mode="json")
    return rendered


async def snapshot(
    world: World,
    *,
    judge_a: ScriptedJudge | None = None,
    judge_b: ScriptedJudge | None = None,
    assessor: ScriptedAssessor | None = None,
    relevance: RelevanceAxisSummary | None = None,
    falsity: FalsityAxisSummary | None = None,
    revisit: RevisitAxisSummary | None = None,
    dry_run: bool = False,
    raised: BaseException | None = None,
) -> Json:
    """Dump one axis scenario: the summaries, their renderings, the store and the call logs."""
    dump: Json = {}
    if raised is not None:
        dump["raised"] = {"type": type(raised).__name__, "message": str(raised)}
    if relevance is not None:
        dump["relevance_summary"] = relevance.model_dump(mode="json")
    if falsity is not None:
        dump["falsity_summary"] = falsity.model_dump(mode="json")
    if revisit is not None:
        dump["revisit_summary"] = revisit.model_dump(mode="json")
    if relevance is not None or falsity is not None or revisit is not None:
        dump["rendered"] = render_all(
            world,
            compose_summary(
                world, relevance=relevance, falsity=falsity, revisit=revisit, dry_run=dry_run
            ),
        )
    dump["store"] = await dump_store(world)
    dump["judge_prompts"] = {
        "a": dump_judge(judge_a) if judge_a is not None else [],
        "b": dump_judge(judge_b) if judge_b is not None else [],
    }
    if assessor is not None:
        dump["assessor_calls"] = [dict(entry) for entry in assessor.log]
    return dump


async def expect_raise(awaitable: Awaitable[object]) -> Exception:
    """Await ``awaitable`` and return the ``Exception`` it must raise."""
    try:
        await awaitable
    except Exception as exc:
        return exc
    raise AssertionError("the scenario expected the call to raise")


Scenario = Callable[[], Awaitable[Json]]
SCENARIOS: dict[str, Scenario] = {}


def scenario(function: Scenario) -> Scenario:
    """Register ``function`` under its own name; the name is the golden's scenario key."""
    SCENARIOS[function.__name__] = function
    return function


async def run_relevance(
    world: World,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    *,
    assessor: Assessor | None = None,
    dry_run: bool = False,
    with_embedder: bool = True,
    explicit_pool: Sequence[Fact] | None = None,
    pool_limit: int | None = None,
) -> RelevanceAxisSummary:
    return await run_relevance_axis(
        scope=world.scope,
        since=SINCE,
        adapter=world.adapter,
        clock=world.clock,
        lock=world.lock,
        cfg=world.cfg,
        rubric=world.rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        pool_limit=pool_limit,
        dry_run=dry_run,
        explicit_pool=explicit_pool,
        embedder=world.embedder if with_embedder else None,
    )


async def run_falsity(
    world: World,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    *,
    assessor: Assessor | None = None,
    excluded_ids: frozenset[HypomnemaId] = frozenset(),
    dry_run: bool = False,
) -> FalsityAxisSummary:
    return await run_falsity_axis(
        scope=world.scope,
        since=SINCE,
        adapter=world.adapter,
        clock=world.clock,
        lock=world.lock,
        rubric=world.rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        embedder=world.embedder,
        cfg=world.cfg,
        assessor=assessor,
        excluded_ids=excluded_ids,
        dry_run=dry_run,
    )


async def run_revisit(
    world: World,
    judge_a: JudgeAdapter,
    judge_b: JudgeAdapter,
    *,
    assessor: Assessor | None = None,
    dry_run: bool = False,
    with_embedder: bool = True,
) -> RevisitAxisSummary:
    return await run_revisit_axis(
        scope=world.scope,
        adapter=world.adapter,
        clock=world.clock,
        lock=world.lock,
        cfg=world.cfg,
        rubric=world.rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        dry_run=dry_run,
        embedder=world.embedder if with_embedder else None,
    )


async def require_registered(scope: Scope) -> None:
    return None


async def run_clean(
    world: World,
    judge_a: JudgeAdapter | None,
    judge_b: JudgeAdapter | None,
    *,
    assessor: Assessor | None = None,
    dry_run: bool = False,
    with_report: bool = True,
    pool_limit: int | None = None,
    rubric: Rubric | None = None,
) -> CleanSummary:
    """``run_clean_pass`` when the report is wanted (the production composition), else ``clean``
    with no ``build_report`` callback.
    """
    if with_report:
        return await run_clean_pass(
            scope=str(world.scope),
            adapter=world.adapter,
            embedder=world.embedder,
            clock=world.clock,
            lock=world.lock,
            cfg=world.cfg,
            require_scope=require_registered,
            rubric=rubric if rubric is not None else world.rubric,
            judge_a=judge_a,
            judge_b=judge_b,
            assessor=assessor,
            pool_limit=pool_limit,
            dry_run=dry_run,
        )
    return await clean(
        scope=str(world.scope),
        adapter=world.adapter,
        embedder=world.embedder,
        clock=world.clock,
        lock=world.lock,
        cfg=world.cfg,
        require_scope=require_registered,
        rubric=rubric if rubric is not None else world.rubric,
        judge_a=judge_a,
        judge_b=judge_b,
        assessor=assessor,
        pool_limit=pool_limit,
        dry_run=dry_run,
    )


async def snapshot_clean(
    world: World,
    summary: CleanSummary | None,
    *,
    judge_a: ScriptedJudge | None = None,
    judge_b: ScriptedJudge | None = None,
    assessor: ScriptedAssessor | None = None,
    raised: BaseException | None = None,
) -> Json:
    """Dump one ``clean()`` scenario: the summary, its renderings and everything the store holds."""
    dump: Json = {}
    if raised is not None:
        dump["raised"] = {"type": type(raised).__name__, "message": str(raised)}
    if summary is not None:
        dump["clean_summary"] = summary.model_dump(mode="json")
        dump["cli_text"] = cli._render_clean_summary(summary).split("\n")
        dump["build_report_returns_none"] = build_report(summary, pass_at=PASS_AT) is None
    dump["store"] = await dump_store(world)
    rows = await world.adapter.list_cleaner_passes(scope=world.scope, limit=10)
    dump["context_summary"] = [
        render_pass_record(row, cap=world.rubric.parameters.report_render_cap).model_dump(
            mode="json"
        )
        for row in rows
    ]
    dump["judge_prompts"] = {
        "a": dump_judge(judge_a) if judge_a is not None else [],
        "b": dump_judge(judge_b) if judge_b is not None else [],
    }
    if assessor is not None:
        dump["assessor_calls"] = [dict(entry) for entry in assessor.log]
    return dump


def advance_days(world: World, days: int) -> None:
    world.clock.advance(timedelta(days=days))


__all__ = [
    "AGED",
    "BORN",
    "CLOCK_START",
    "OLDER",
    "PASS_AT",
    "SCENARIOS",
    "SINCE",
    "Scenario",
    "World",
    "advance_days",
    "all_records",
    "compose_summary",
    "dump_cleaner_state",
    "dump_store",
    "expect_raise",
    "idle_falsity",
    "idle_relevance",
    "make_world",
    "pinned_environment",
    "put_decision",
    "put_event",
    "put_fact",
    "put_review",
    "put_session",
    "render_all",
    "rid",
    "run_clean",
    "run_falsity",
    "run_relevance",
    "run_revisit",
    "scenario",
    "snapshot",
    "snapshot_clean",
]
