"""The nightly Cleaner scheduler (``Tabularium._nightly_tick`` / ``_run_single_flight_pass`` /
``_run_cleaner_pass``) and the retired per-call trigger.

Single-adapter (``MemoryAdapter``): the scheduler never branches on which ``StorageAdapter`` it is
handed. Determinism: every test drives ``_nightly_tick`` directly against a ``TestClock`` and uses
``asyncio.Event`` gates, never a sleep guess at scheduling order.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Scope
from tabularium.engine import Tabularium
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, TestClock

LOCAL_TWO_FIFTY_NINE = datetime(2025, 1, 1, 2, 59).astimezone().astimezone(UTC)


async def _make_engine(
    *, enabled: bool = True, scopes: tuple[str, ...] = ("proj-a", "proj-b")
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    clock = TestClock(LOCAL_TWO_FIFTY_NINE)
    adapter = MemoryAdapter(embedding_model="test-model", dim=8)
    cfg = EngineConfig(enable_digest_cleaner=enabled)
    engine = Tabularium(adapter, FakeEmbedder(dim=8), clock, cfg)
    await engine.initialize()
    for scope in scopes:
        await adapter.create_scope(Scope(scope))
    engine._next_nightly = engine._next_nightly_after(clock.now())
    return engine, adapter, clock


async def _stop_scheduler(engine: Tabularium) -> None:
    task = engine._nightly_task
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class TestConfig:
    def test_nightly_default_is_three_am(self) -> None:
        assert EngineConfig().cleaner_nightly_at == "03:00"

    @pytest.mark.parametrize("value", ["3:00", "24:00", "03:60", "0300", "", "noon"])
    def test_malformed_time_is_refused(self, value: str) -> None:
        with pytest.raises(ValueError):
            EngineConfig(cleaner_nightly_at=value)


class TestSchedulerStartup:
    async def test_initialize_starts_the_loop_only_when_enabled(self) -> None:
        enabled, _a, _c = await _make_engine(enabled=True)
        disabled, _a2, _c2 = await _make_engine(enabled=False)
        try:
            assert enabled._nightly_task is not None
            assert disabled._nightly_task is None
        finally:
            await _stop_scheduler(enabled)

    async def test_next_run_is_the_first_local_three_am_after_now(self) -> None:
        engine, _adapter, clock = await _make_engine(enabled=False)

        nxt = engine._next_nightly_after(clock.now())
        local = nxt.astimezone()

        assert (local.hour, local.minute) == (3, 0)
        assert nxt - clock.now() == timedelta(minutes=1)

        clock.advance(timedelta(minutes=1))
        later = engine._next_nightly_after(clock.now())
        assert later - clock.now() == timedelta(days=1)


class TestNightlyTick:
    async def test_before_three_nothing_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _adapter, _clock = await _make_engine()
        calls: list[str] = []

        async def _fake_clean(scope: str) -> None:
            calls.append(scope)

        monkeypatch.setattr(engine, "clean", _fake_clean)
        try:
            assert await engine._nightly_tick() is False
            assert calls == []
        finally:
            await _stop_scheduler(engine)

    async def test_at_three_one_pass_per_scope_in_sequence_and_once_per_night(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, adapter, clock = await _make_engine()
        order: list[str] = []
        running = 0
        peak = 0

        async def _fake_clean(scope: str) -> None:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            order.append(scope)
            await asyncio.sleep(0)
            running -= 1

        monkeypatch.setattr(engine, "clean", _fake_clean)
        try:
            clock.advance(timedelta(minutes=1))

            assert await engine._nightly_tick() is True
            assert sorted(order) == ["global", "proj-a", "proj-b"]
            assert order == [str(scope) for scope in await adapter.list_scopes()]
            assert peak == 1

            assert await engine._nightly_tick() is False
            clock.advance(timedelta(hours=2))
            assert await engine._nightly_tick() is False
            assert len(order) == 3

            clock.advance(timedelta(days=1))
            assert await engine._nightly_tick() is True
            assert len(order) == 6
        finally:
            await _stop_scheduler(engine)

    async def test_global_pass_is_a_real_pass_not_a_refusal(self) -> None:
        engine, adapter, clock = await _make_engine(scopes=())
        try:
            clock.advance(timedelta(minutes=1))

            await engine._nightly_tick()

            summary = await adapter.get_cleaner_state(scope=Scope("global"))
            assert summary is not None
            history = await adapter.list_cleaner_passes(scope=Scope("global"), limit=5)
            assert [row.outcome for row in history] == ["initialized"]
        finally:
            await _stop_scheduler(engine)


class TestNoTriggerMethod:
    def test_no_trigger_method_remains(self) -> None:
        assert not hasattr(Tabularium, "_trigger_cleaner_pass")


class TestSingleFlight:
    async def test_a_scope_with_a_pass_in_flight_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, _adapter, _clock = await _make_engine()
        calls: list[str] = []
        release = asyncio.Event()
        started = asyncio.Event()

        async def _blocking_clean(scope: str) -> None:
            calls.append(scope)
            started.set()
            await release.wait()

        monkeypatch.setattr(engine, "clean", _blocking_clean)
        try:
            first = asyncio.create_task(engine._run_single_flight_pass("proj-a"))
            await asyncio.wait_for(started.wait(), timeout=2.0)

            await engine._run_single_flight_pass("proj-a")
            assert calls == ["proj-a"]

            release.set()
            await first
            await asyncio.sleep(0)
            assert engine._cleaner_tasks == {}

            await engine._run_single_flight_pass("proj-a")
            assert calls == ["proj-a", "proj-a"]
        finally:
            await _stop_scheduler(engine)


class TestFailuresAreRecordedNotSwallowed:
    async def test_exception_from_clean_is_logged_and_the_tick_continues(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, clock = await _make_engine()
        calls: list[str] = []

        async def _flaky_clean(scope: str) -> None:
            calls.append(scope)
            if scope == "proj-a":
                raise RuntimeError("synthetic failure for proj-a")

        monkeypatch.setattr(engine, "clean", _flaky_clean)
        try:
            clock.advance(timedelta(minutes=1))
            with caplog.at_level(logging.ERROR, logger="tabularium"):
                assert await engine._nightly_tick() is True

            assert "proj-b" in calls
            assert any(
                "synthetic failure for proj-a" in r.getMessage() and r.exc_info is not None
                for r in caplog.records
            )
        finally:
            await _stop_scheduler(engine)

    async def test_crash_after_a_real_first_pass_preserves_the_watermark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, adapter, clock = await _make_engine(scopes=("proj-a",))
        try:
            await engine._run_single_flight_pass("proj-a")
            pre = await adapter.get_cleaner_state(scope=Scope("proj-a"))
            assert pre is not None

            clock.advance(timedelta(hours=1))

            async def _exploding_clean(scope: str) -> None:
                raise RuntimeError("boom")

            monkeypatch.setattr(engine, "clean", _exploding_clean)
            await engine._run_single_flight_pass("proj-a")

            post = await adapter.get_cleaner_state(scope=Scope("proj-a"))
            assert post is not None
            assert post.watermark_at == pre.watermark_at
            payload = await engine.context("proj-a", 2000, role="builder")
            assert payload.cleaner is not None
            assert payload.cleaner.outcome == "errored"
            assert "boom" in payload.cleaner.summary_line
        finally:
            await _stop_scheduler(engine)

    async def test_persisting_the_failure_itself_failing_does_not_propagate(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, adapter, _clock = await _make_engine(scopes=("proj-a",))
        try:
            await engine._run_single_flight_pass("proj-a")

            async def _exploding_clean(scope: str) -> None:
                raise RuntimeError("boom2")

            async def _exploding_set_cleaner_state(*, scope: object, state: object) -> None:
                raise ConnectionError("adapter is down")

            monkeypatch.setattr(engine, "clean", _exploding_clean)
            monkeypatch.setattr(adapter, "set_cleaner_state", _exploding_set_cleaner_state)

            with caplog.at_level(logging.ERROR, logger="tabularium"):
                await engine._run_single_flight_pass("proj-a")

            messages = [r.getMessage() for r in caplog.records]
            assert any("boom2" in m for m in messages)
            assert any("adapter is down" in m or "record_pass_failure" in m for m in messages)
            history = await adapter.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
            assert [r.outcome for r in history] == ["errored", "initialized"]
        finally:
            await _stop_scheduler(engine)
