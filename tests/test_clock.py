"""Tests for clock.py — SystemClock and TestClock behaviour (SPEC §8.7)."""

from __future__ import annotations

import datetime

import pytest

from tabularium.clock import Clock, SystemClock
from tests.support.doubles import TestClock

UTC = datetime.UTC


class TestSystemClock:
    def test_returns_aware_datetime(self) -> None:
        clk = SystemClock()
        t = clk.now()
        assert t.tzinfo is not None

    def test_returns_utc(self) -> None:
        clk = SystemClock()
        t = clk.now()
        # UTC offset must be zero
        assert t.utcoffset() == datetime.timedelta(0)

    def test_satisfies_clock_protocol(self) -> None:
        clk: Clock = SystemClock()
        t = clk.now()
        assert isinstance(t, datetime.datetime)

    def test_monotonically_nondecreasing(self) -> None:
        clk = SystemClock()
        a = clk.now()
        b = clk.now()
        assert b >= a


class TestTestClock:
    def test_initial_time_preserved(self) -> None:
        start = datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        clk = TestClock(start)
        assert clk.now() == start

    def test_advance_moves_clock(self) -> None:
        start = datetime.datetime(2025, 1, 1, tzinfo=UTC)
        clk = TestClock(start)
        clk.advance(datetime.timedelta(days=1))
        assert clk.now() == datetime.datetime(2025, 1, 2, tzinfo=UTC)

    def test_advance_is_additive(self) -> None:
        start = datetime.datetime(2025, 1, 1, tzinfo=UTC)
        clk = TestClock(start)
        clk.advance(datetime.timedelta(hours=1))
        clk.advance(datetime.timedelta(hours=1))
        assert clk.now() == datetime.datetime(2025, 1, 1, 2, 0, 0, tzinfo=UTC)

    def test_set_jumps_to_absolute_time(self) -> None:
        start = datetime.datetime(2025, 1, 1, tzinfo=UTC)
        clk = TestClock(start)
        target = datetime.datetime(2030, 12, 31, tzinfo=UTC)
        clk.set(target)
        assert clk.now() == target

    def test_naive_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            TestClock(datetime.datetime(2025, 1, 1))

    def test_naive_set_rejected(self) -> None:
        clk = TestClock(datetime.datetime(2025, 1, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="tz-aware"):
            clk.set(datetime.datetime(2025, 6, 1))

    def test_satisfies_clock_protocol(self) -> None:
        clk: Clock = TestClock(datetime.datetime(2025, 1, 1, tzinfo=UTC))
        assert isinstance(clk.now(), datetime.datetime)

    def test_returns_utc_aware(self) -> None:
        clk = TestClock(datetime.datetime(2025, 3, 15, tzinfo=UTC))
        t = clk.now()
        assert t.tzinfo is not None
        assert t.utcoffset() == datetime.timedelta(0)

    def test_advance_negative_delta(self) -> None:
        """Negative advance (time travel backward) must also work — evals need it."""
        start = datetime.datetime(2025, 6, 1, tzinfo=UTC)
        clk = TestClock(start)
        clk.advance(datetime.timedelta(days=-30))
        assert clk.now() == datetime.datetime(2025, 5, 2, tzinfo=UTC)
