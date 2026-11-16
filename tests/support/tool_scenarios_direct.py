"""Tool-golden scenarios for engine behaviour no tool reaches.

Each calls an engine function or method directly and logs it as a ``direct`` entry: the registry
seeding of ``Tabularium.initialize``, ``Tabularium.save`` with a kind the wire schema rejects or
with empty content, ``Tabularium.context`` and ``pack`` with a negative budget,
``Tabularium.clean``, and the nightly scheduler's loop, tick and failed-pass recording. The nightly
scenarios pin the process time zone to UTC for their duration, because the scheduler works in
local time.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest import mock

from tabularium.engine import core
from tabularium.engine.context import SelectedTier, pack
from tests.support.doubles import FakeTokenCounter
from tests.support.tool_golden_assessor import ToolAssessor
from tests.support.tool_golden_world import (
    Json,
    ToolWorld,
    golden_config,
    make_world,
    scenario,
)


@scenario
async def core__initialize_seeds_registries_and_retires_the_default_role() -> Json:
    w = await make_world()
    await w.adapter.create_role("default")
    await w.adapter.create_role("operator-added")
    w.note("roles before", await w.adapter.list_roles())
    await w.direct("initialize again", w.engine.initialize())
    w.note("roles after", await w.adapter.list_roles())
    w.note("facets after", await w.adapter.list_facets())
    w.note("scopes after", await w.adapter.list_scopes())
    return await w.finish()


@scenario
async def core__save_with_a_kind_the_wire_schema_rejects() -> Json:
    w = await make_world(assessor=ToolAssessor())
    invented_kind: Any = "invented"
    await w.direct("save kind=invented", w.engine.save("alpha rule", kind=invented_kind, scope="p"))
    await w.direct("save empty content", w.engine.save("", scope="p"))
    await w.direct(
        "context without a role",
        w.engine.context("p", 100),
    )
    return await w.finish()


@scenario
async def save__refusals_for_empty_decision_content() -> Json:
    w = await make_world()
    await w.direct("save empty decision", w.engine.save("", kind="decision", scope="p"))
    await w.direct(
        "save whitespace-only decision", w.engine.save("   ", kind="decision", scope="p")
    )
    await w.call("decisions_query", {"query": "nothing matches this zebra quartz", "scope": "p"})
    return await w.finish()


@contextmanager
def _utc_process() -> Iterator[None]:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


@scenario
async def core__nightly_schedule_tick_and_failed_pass_recording() -> Json:
    with _utc_process():
        w = await make_world(config=golden_config(cleaner_nightly_at="03:00"))
        engine = w.engine

        w.note(
            "next after noon", engine._next_nightly_after(datetime(2025, 6, 1, 12, 0, tzinfo=UTC))
        )
        w.note(
            "next after 02:59", engine._next_nightly_after(datetime(2025, 6, 1, 2, 59, tzinfo=UTC))
        )
        w.note(
            "next at 03:00 exactly",
            engine._next_nightly_after(datetime(2025, 6, 1, 3, 0, tzinfo=UTC)),
        )

        await w.direct("tick before the scheduler has a time", engine._nightly_tick())
        engine._next_nightly = w.clock.peek() + timedelta(hours=5)
        await w.direct("tick before the scheduled instant", engine._nightly_tick())

        passes: list[str] = []

        async def failing_clean(scope: str) -> object:
            passes.append(scope)
            raise RuntimeError(f"the pass for {scope} crashed")

        engine._next_nightly = w.clock.peek() - timedelta(seconds=1)
        with mock.patch.object(engine, "clean", failing_clean):
            await w.direct("tick at the scheduled instant", engine._nightly_tick())
        w.note("scopes passed over", passes)
        w.note("next scheduled", engine._next_nightly)

        await w.direct("single flight for a scope with a pass in flight", _in_flight(w))
        w.note("tasks left", sorted(engine._cleaner_tasks))
        return await w.finish()


async def _in_flight(w: ToolWorld) -> None:
    engine = w.engine
    started = asyncio.Event()
    release = asyncio.Event()
    runs: list[str] = []

    async def slow_clean(scope: str) -> object:
        runs.append(scope)
        started.set()
        await release.wait()
        return None

    with mock.patch.object(engine, "clean", slow_clean):
        first = asyncio.create_task(engine._run_single_flight_pass("p"))
        await started.wait()
        await engine._run_single_flight_pass("p")
        release.set()
        await first
    w.note("clean runs", runs)


@scenario
async def core__context_direct_negative_budget_and_empty_tiers() -> Json:
    w = await make_world()
    await w.direct("context with a negative budget", w.engine.context("p", -1, "builder"))

    async def packed() -> object:
        return pack([], 10, FakeTokenCounter())

    await w.direct("pack with no tiers", packed())

    async def packed_negative() -> object:
        return pack([SelectedTier(name="hot", entries=[])], -1, FakeTokenCounter())

    await w.direct("pack with a negative budget", packed_negative())
    return await w.finish()


@scenario
async def core__clean_forwards_its_arguments_to_the_pass() -> Json:
    w = await make_world(assessor=ToolAssessor())
    calls: list[list[str]] = []

    async def stub(**kwargs: object) -> str:
        calls.append(sorted(kwargs))
        return "stub summary"

    with mock.patch.object(core, "_run_clean_pass", stub):
        await w.direct("clean p", w.engine.clean("p"))
    w.note("keyword arguments the pass received", calls)
    return await w.finish()


@scenario
async def core__nightly_loop_runs_a_tick_and_survives_a_failing_one() -> Json:
    original_sleep = asyncio.sleep
    delays: list[float] = []
    holder: dict[str, ToolWorld] = {}

    async def fake_sleep(delay: float, result: object = None) -> object:
        if delay == 0:
            return await original_sleep(0, result)
        delays.append(delay)
        world = holder["w"]
        if len(delays) == 1:
            due = world.engine._next_nightly
            assert due is not None
            world.clock.set(due + timedelta(seconds=1))
        if len(delays) >= 3:
            raise asyncio.CancelledError
        return None

    ticks: list[int] = []

    async def failing_clean(scope: str) -> object:
        raise RuntimeError(f"the pass for {scope} crashed")

    async def failing_record(**kwargs: object) -> None:
        raise RuntimeError("the failure report could not be stored")

    with _utc_process(), mock.patch.object(asyncio, "sleep", fake_sleep):
        w = await make_world(config=golden_config(enable_digest_cleaner=True))
        holder["w"] = w
        engine = w.engine
        real_tick = engine._nightly_tick

        async def tick_that_fails_once() -> bool:
            ticks.append(len(ticks))
            if len(ticks) == 2:
                raise RuntimeError("tick failed")
            return await real_tick()

        with (
            mock.patch.object(engine, "clean", failing_clean),
            mock.patch.object(engine, "_nightly_tick", tick_that_fails_once),
            mock.patch.object(core, "record_pass_failure", failing_record),
        ):
            task = engine._nightly_task
            assert task is not None
            try:
                await task
            except asyncio.CancelledError:
                w.note("the loop ended cancelled", True)
    w.note("sleep delays in seconds", delays)
    w.note("ticks", ticks)
    w.note("next scheduled", engine._next_nightly)
    return await w.finish()


@scenario
async def core__initialize_twice_keeps_one_nightly_task() -> Json:
    original_sleep = asyncio.sleep

    async def blocking_sleep(delay: float, result: object = None) -> object:
        if delay == 0:
            return await original_sleep(0, result)
        await asyncio.Event().wait()
        return None

    with _utc_process(), mock.patch.object(asyncio, "sleep", blocking_sleep):
        w = await make_world(config=golden_config(enable_digest_cleaner=True))
        first = w.engine._nightly_task
        assert first is not None
        await w.direct("initialize again", w.engine.initialize())
        w.note("the same task keeps running", w.engine._nightly_task is first and not first.done())
        w.note("next scheduled", w.engine._next_nightly)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    return await w.finish()
