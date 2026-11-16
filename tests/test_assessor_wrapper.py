"""``engine.admission.ask``/``relation_verdicts`` — the engine-owned timeout wrapper and the
per-survivor fan-out (Assessor map ticket 02, decisions D3/D4/D5).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from tabularium.engine.admission import ask, relation_verdicts
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    PanelDisagreement,
    Relation,
    RelationVerdict,
    Trace,
    Unavailable,
)

_NOW = datetime(2025, 6, 1, tzinfo=UTC)


def _candidate(cid: str) -> AssessorCandidate:
    return AssessorCandidate(
        id=cid, content=f"content {cid}", kind="fact", scope="global", created_at=_NOW
    )


def _trace() -> Trace:
    return Trace(
        model="test", wording_hash="test", latency_ms=0,
        input_tokens=None, output_tokens=None, request_id=None,
    )


def _verdict(pick: Relation = "distinct") -> RelationVerdict:
    return RelationVerdict(
        method="relation", pick=pick, odds={pick: 1.0}, confidence=1.0, trace=_trace(),
    )


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


async def test_ask_absent_when_factory_is_none() -> None:
    result = await ask(None, method="relation", timeout_s=1.0)
    assert isinstance(result, Unavailable)
    assert result.reason == "absent"
    assert result.latency_ms is None


async def test_ask_returns_the_factory_result_on_success() -> None:
    verdict = _verdict()

    async def _factory() -> RelationVerdict:
        return verdict

    result = await ask(_factory, method="relation", timeout_s=1.0)
    assert result is verdict


async def test_ask_times_out() -> None:
    async def _factory() -> RelationVerdict:
        await asyncio.sleep(10)
        return _verdict()

    result = await ask(_factory, method="relation", timeout_s=0.05)
    assert isinstance(result, Unavailable)
    assert result.reason == "timeout"
    assert result.latency_ms is not None


async def test_ask_never_hangs_past_timeout_plus_slack() -> None:
    async def _factory() -> RelationVerdict:
        await asyncio.sleep(10)
        return _verdict()

    start = time.monotonic()
    await ask(_factory, method="relation", timeout_s=0.05)
    assert time.monotonic() - start < 1.0


async def test_ask_maps_a_raise_to_error_naming_the_exception_class() -> None:
    async def _factory() -> RelationVerdict:
        raise ValueError("boom")

    result = await ask(_factory, method="relation", timeout_s=1.0)
    assert isinstance(result, Unavailable)
    assert result.reason == "error"
    assert result.detail == "ValueError"


async def test_ask_reraises_not_implemented_error() -> None:
    async def _factory() -> RelationVerdict:
        raise NotImplementedError("A3 owns this method")

    with pytest.raises(NotImplementedError):
        await ask(_factory, method="relation", timeout_s=1.0)


async def test_ask_logs_one_debug_line_per_call(caplog: pytest.LogCaptureFixture) -> None:
    async def _factory() -> RelationVerdict:
        return _verdict()

    with caplog.at_level(logging.DEBUG, logger="tabularium"):
        await ask(_factory, method="relation", timeout_s=1.0)

    lines = [r for r in caplog.records if r.getMessage().startswith("assessor_call")]
    assert len(lines) == 1
    assert "method=relation" in lines[0].getMessage()
    assert "outcome=verdict" in lines[0].getMessage()


# ---------------------------------------------------------------------------
# relation_verdicts
# ---------------------------------------------------------------------------


class _SleepyAssessor:
    """Records call order and sleeps ``delay_s`` per ``relation`` call — proves the fan-out
    runs concurrently, not sequentially."""

    def __init__(self, delay_s: float) -> None:
        self._delay_s = delay_s
        self.calls: list[str] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.calls.append(candidate.id)
        await asyncio.sleep(self._delay_s)
        return _verdict("restatement" if candidate.id == "c0" else "distinct")

    async def kind(self, text: str, declared_kind: str) -> RelationVerdict:
        raise NotImplementedError

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> RelationVerdict:
        raise NotImplementedError

    async def perishable(self, text: str) -> RelationVerdict:
        raise NotImplementedError

    async def breadth(self, text: str, source_scope: str) -> RelationVerdict:
        raise NotImplementedError

    async def scope(self, text: str, labels: object) -> RelationVerdict:
        raise NotImplementedError

    async def relevance(self, text: str) -> RelationVerdict:
        raise NotImplementedError

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        raise NotImplementedError


async def test_three_survivors_issue_three_concurrent_calls_in_probe_order() -> None:
    assessor = _SleepyAssessor(delay_s=0.3)
    survivors = [(_candidate("c0"), 0.99), (_candidate("c1"), 0.9), (_candidate("c2"), 0.85)]

    start = time.monotonic()
    verdicts = await relation_verdicts(assessor, "incoming", survivors, timeout_s=2.0)
    elapsed = time.monotonic() - start

    assert elapsed < 0.6
    assert assessor.calls == ["c0", "c1", "c2"]
    assert len(verdicts) == 3
    picks = []
    for v in verdicts:
        assert isinstance(v, RelationVerdict)
        picks.append(v.pick)
    assert picks == ["restatement", "distinct", "distinct"]


async def test_relation_verdicts_absent_when_no_assessor() -> None:
    survivors = [(_candidate("c0"), 0.99)]
    verdicts = await relation_verdicts(None, "incoming", survivors, timeout_s=1.0)
    assert len(verdicts) == 1
    assert isinstance(verdicts[0], Unavailable)
    assert verdicts[0].reason == "absent"
