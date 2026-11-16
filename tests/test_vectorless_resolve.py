"""Explicit ``resolve`` under an embedder outage (P6 vector-less-rows
piece).

Before this fix, ``save_internal``'s (facts.py) and ``save_decision``'s
(decisions.py) ``vector is None`` early return wrote a fresh vector-less
row unconditionally, ignoring any caller-supplied ``resolve`` entirely —
``reinforce`` minted a new row instead of reinforcing the named candidate
(no event landed on it), and ``confirm_distinct`` wrote the new row without
registering the distinct pair, so a byte-identical replay after recovery
would not be recognized as already-adjudicated. The similarity probe needs
a query vector; a named candidate does not, so both actions are honored
here the same as the live-vector path (``_handle_resolve`` /
``_handle_resolve_decision``), minus the best-hit/floor comparison there is
no probe to run.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import new_id
from tabularium.engine import Tabularium
from tabularium.engine._errors import ToolError
from tabularium.results import BounceResolution
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import _make_memory_adapter
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.faults import FaultInjectingEmbedder, as_embedder

EPOCH = datetime(2025, 1, 1, tzinfo=UTC)
_DIM = 16

_EngineCtx = tuple[Tabularium, MemoryAdapter, FaultInjectingEmbedder]


def _cfg() -> EngineConfig:
    return EngineConfig(gray_floor=0.80, candidate_k=1000)


@pytest.fixture
async def engine_ctx(tmp_path: pathlib.Path) -> _EngineCtx:
    fake = FakeEmbedder(dim=_DIM)
    fault = FaultInjectingEmbedder(fake)
    clock = TestClock(EPOCH)
    adapter = _make_memory_adapter(embedding_model=fake.model, dim=_DIM, clock=clock)
    engine = Tabularium(adapter, as_embedder(fault), clock, _cfg())
    await engine.initialize()
    return engine, adapter, fault


# ---------------------------------------------------------------------------
# Fact arm
# ---------------------------------------------------------------------------


async def test_fact_reinforce_during_outage_reinforces_candidate_and_writes_no_row(
    engine_ctx: _EngineCtx,
) -> None:
    engine, adapter, fault = engine_ctx
    original = await engine.save("the candidate content", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    result = await engine.save(
        "the candidate content",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "reinforced"
    assert result.reinforced_id == o_id
    assert result.hypomnema is None

    events_after = await adapter.query_events(target_id=o_id, source="dedup")
    assert len(events_after) == 1

    row = await adapter.get_by_id(o_id)
    assert row.recall_count == 1


async def test_fact_confirm_distinct_during_outage_writes_vectorless_row_and_registers_pair(
    engine_ctx: _EngineCtx,
) -> None:
    engine, adapter, fault = engine_ctx
    original = await engine.save("candidate content two", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    result = await engine.save(
        "genuinely distinct new content",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert result.outcome == "saved"
    assert result.embedded is False
    assert result.hypomnema is not None

    landed = await adapter.get_by_id(result.hypomnema.id)
    assert landed.embedding_model is None

    assert await adapter.is_distinct(o_id, result.hypomnema.id) is True


async def test_fact_confirm_distinct_replay_during_outage_is_idempotent(
    engine_ctx: _EngineCtx,
) -> None:
    """A verbatim retry
    of the same vectorless confirm_distinct resolve — e.g. a caller's own
    outage-aware retry loop — must not mint a second vector-less row. The
    similarity probe cannot run without a vector, but replay detection
    does not need one (a vector-independent exact-content lookup)."""
    engine, adapter, fault = engine_ctx
    original = await engine.save("candidate content three", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    first = await engine.save(
        "genuinely distinct replayed content",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert first.outcome == "saved"
    assert first.hypomnema is not None
    n_id = first.hypomnema.id

    replay = await engine.save(
        "genuinely distinct replayed content",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert replay.outcome == "already_saved"
    assert replay.hypomnema is not None
    assert replay.hypomnema.id == n_id

    matches = [
        meta async for meta in await adapter.scan()
        if (await adapter.get_by_id(meta.id)).content == "genuinely distinct replayed content"
    ]
    assert len(matches) == 1


async def test_fact_confirm_distinct_replay_survives_a_bounded_candidate_k(
    tmp_path: pathlib.Path,
) -> None:
    """Replay detection must not depend
    on BM25 top-k retrieval. The tokenizer strips punctuation (letters and
    digits only), so ``"Use WAL."`` and ``"Use WAL!"`` tokenize IDENTICALLY
    and score identically for the same query — a tie the storage layer
    breaks by insertion order, always favoring the OLDER row. With
    ``candidate_k=1`` a ranked lookup for the newer row (the actual
    duplicate N) returns only the older candidate O and never sees N at
    all — a live counterexample confirmed against ``MemoryAdapter``."""
    fake = FakeEmbedder(dim=_DIM)
    fault = FaultInjectingEmbedder(fake)
    clock = TestClock(EPOCH)
    adapter = _make_memory_adapter(embedding_model=fake.model, dim=_DIM, clock=clock)
    cfg = EngineConfig(gray_floor=0.80, candidate_k=1)
    engine = Tabularium(adapter, as_embedder(fault), clock, cfg)
    await engine.initialize()

    original = await engine.save("Use WAL.", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    first = await engine.save(
        "Use WAL!",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert first.outcome == "saved"
    assert first.hypomnema is not None
    n_id = first.hypomnema.id

    replay = await engine.save(
        "Use WAL!",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert replay.outcome == "already_saved"
    assert replay.hypomnema is not None
    assert replay.hypomnema.id == n_id

    matches = [
        meta async for meta in await adapter.scan()
        if (await adapter.get_by_id(meta.id)).content == "Use WAL!"
    ]
    assert len(matches) == 1


async def test_fact_confirm_distinct_replay_excludes_a_shelved_match(
    engine_ctx: _EngineCtx,
) -> None:
    """Shelving a Fact withdraws the
    store's dedup defence of its content (ADR 0021) — the live-vector
    replay guard's own probe never sees a shelved candidate, and the
    vectorless replay scan must not silently reintroduce it as an eligible
    match. Shelve the row the first confirm_distinct call wrote, retry the
    identical resolve during the same outage, and expect a NEW active row
    rather than an idempotent ack of the now-shelved one."""
    engine, adapter, fault = engine_ctx
    original = await engine.save("candidate content four", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    first = await engine.save(
        "content that will be shelved then replayed",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert first.outcome == "saved"
    assert first.hypomnema is not None
    n_id = first.hypomnema.id

    fault.lift()
    await engine.set_shelved(n_id, True, "not worth carrying")

    fault.arm()
    replay = await engine.save(
        "content that will be shelved then replayed",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert replay.outcome == "saved"
    assert replay.hypomnema is not None
    assert replay.hypomnema.id != n_id

    matching_ids = [
        meta.id async for meta in await adapter.scan()
        if (await adapter.get_by_id(meta.id)).content
        == "content that will be shelved then replayed"
    ]
    assert len(matching_ids) == 2
    assert n_id in matching_ids
    assert replay.hypomnema.id in matching_ids


async def test_fact_reinforce_unknown_candidate_during_outage_raises(
    engine_ctx: _EngineCtx,
) -> None:
    engine, _adapter, fault = engine_ctx
    unknown_id = new_id()

    fault.arm()
    with pytest.raises(ToolError, match="unknown/buried"):
        await engine.save(
            "content naming a bogus candidate",
            scope="global",
            resolve=BounceResolution(candidate_id=unknown_id, action="reinforce"),
        )


async def test_fact_reinforce_archived_candidate_during_outage_revives_it(
    engine_ctx: _EngineCtx,
) -> None:
    engine, adapter, fault = engine_ctx
    original = await engine.save("archived candidate content", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id
    await adapter.transition(o_id, "archived", at=EPOCH)

    fault.arm()
    result = await engine.save(
        "archived candidate content",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "revived"
    assert result.reinforced_id == o_id
    row = await adapter.get_by_id(o_id)
    assert row.lifecycle == "active"


async def test_fact_reinforce_shelved_candidate_during_outage_writes_fresh_row(
    engine_ctx: _EngineCtx,
) -> None:
    """ADR 0021 withdraws the store's
    dedup defence of shelved content. The live-vector path never reinforces
    a shelved candidate — ``probe_filters_for`` excludes it from ``best_hit``,
    so a stale resolve against a since-shelved id lands as a fresh save.
    The vectorless outage branch must not silently reinstate it as
    reinforceable just because it skipped the probe."""
    engine, adapter, fault = engine_ctx
    original = await engine.save("content that will be shelved", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id
    await engine.set_shelved(o_id, True, "not worth carrying")

    fault.arm()
    result = await engine.save(
        "content that will be shelved",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "saved"
    assert result.embedded is False
    assert result.hypomnema is not None
    assert result.hypomnema.id != o_id

    events = await adapter.query_events(target_id=o_id, source="dedup")
    assert events == []
    shelved_row = await adapter.get_by_id(o_id)
    assert shelved_row.lifecycle == "shelved"


# ---------------------------------------------------------------------------
# Decision arm
# ---------------------------------------------------------------------------


async def test_decision_reinforce_during_outage_reinforces_candidate_and_writes_no_row(
    engine_ctx: _EngineCtx,
) -> None:
    engine, adapter, fault = engine_ctx
    original = await engine.save("decision candidate content", kind="decision", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    result = await engine.save(
        "decision candidate content",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "reinforced"
    assert result.reinforced_id == o_id
    assert result.hypomnema is None

    events_after = await adapter.query_events(target_id=o_id, source="dedup")
    assert len(events_after) == 1


async def test_decision_confirm_distinct_during_outage_writes_vectorless_row_and_registers_pair(
    engine_ctx: _EngineCtx,
) -> None:
    engine, adapter, fault = engine_ctx
    original = await engine.save("decision candidate content two", kind="decision", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    result = await engine.save(
        "genuinely distinct new decision content",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert result.outcome == "saved"
    assert result.embedded is False
    assert result.hypomnema is not None

    landed = await adapter.get_by_id(result.hypomnema.id)
    assert landed.embedding_model is None
    assert await adapter.is_distinct(o_id, result.hypomnema.id) is True


async def test_decision_confirm_distinct_replay_during_outage_is_idempotent(
    engine_ctx: _EngineCtx,
) -> None:
    """Decision-arm mirror of the Fact-arm replay test above."""
    engine, adapter, fault = engine_ctx
    original = await engine.save(
        "decision candidate content three", kind="decision", scope="global"
    )
    assert original.hypomnema is not None
    o_id = original.hypomnema.id

    fault.arm()
    first = await engine.save(
        "genuinely distinct replayed decision content",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert first.outcome == "saved"
    assert first.hypomnema is not None
    n_id = first.hypomnema.id

    replay = await engine.save(
        "genuinely distinct replayed decision content",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="confirm_distinct"),
    )
    assert replay.outcome == "already_saved"
    assert replay.hypomnema is not None
    assert replay.hypomnema.id == n_id

    replayed_content = "genuinely distinct replayed decision content"
    matches = [
        meta async for meta in await adapter.scan()
        if (await adapter.get_by_id(meta.id)).content == replayed_content
    ]
    assert len(matches) == 1


async def test_decision_reinforce_unknown_candidate_during_outage_raises(
    engine_ctx: _EngineCtx,
) -> None:
    engine, _adapter, fault = engine_ctx
    unknown_id = new_id()

    fault.arm()
    with pytest.raises(ToolError, match="unknown/buried"):
        await engine.save(
            "decision content naming a bogus candidate",
            kind="decision",
            scope="global",
            resolve=BounceResolution(candidate_id=unknown_id, action="reinforce"),
        )


async def test_decision_reinforce_superseded_candidate_during_outage_writes_fresh_row(
    engine_ctx: _EngineCtx,
) -> None:
    """A
    Decision candidate no longer ``in_force`` (superseded here) must not
    be reinforced — the live-vector path's own probe excludes retired
    Decisions from candidacy, and a stale resolve against one falls
    through to a fresh save there."""
    engine, adapter, fault = engine_ctx
    original = await engine.save(
        "decision that will be superseded", kind="decision", scope="global"
    )
    assert original.hypomnema is not None
    o_id = original.hypomnema.id
    successor = await engine.save(
        "a real successor decision",
        kind="decision",
        scope="global",
        supersedes=o_id,
        reason="changed",
    )
    assert successor.outcome == "saved"

    fault.arm()
    result = await engine.save(
        "decision that will be superseded",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "saved"
    assert result.embedded is False
    assert result.hypomnema is not None
    assert result.hypomnema.id != o_id

    events = await adapter.query_events(target_id=o_id, source="dedup")
    assert events == []
    superseded_row = await adapter.get_by_id(o_id)
    assert superseded_row.lifecycle == "superseded"


async def test_decision_reinforce_shelved_fact_candidate_during_outage_writes_fresh_row(
    engine_ctx: _EngineCtx,
) -> None:
    """Decision-arm reinforcing a Fact
    candidate: a shelved Fact must not be reinforced through the Decision
    arm any more than through the Fact arm."""
    engine, adapter, fault = engine_ctx
    original = await engine.save("fact that will be shelved via decision arm", scope="global")
    assert original.hypomnema is not None
    o_id = original.hypomnema.id
    await engine.set_shelved(o_id, True, "not worth carrying")

    fault.arm()
    result = await engine.save(
        "fact that will be shelved via decision arm",
        kind="decision",
        scope="global",
        resolve=BounceResolution(candidate_id=o_id, action="reinforce"),
    )

    assert result.outcome == "saved"
    assert result.embedded is False
    assert result.hypomnema is not None
    assert result.hypomnema.id != o_id

    events = await adapter.query_events(target_id=o_id, source="dedup")
    assert events == []
