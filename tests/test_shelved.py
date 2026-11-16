"""``set_shelved`` — demotion the dedup probe does not see (SPEC §1.2 / §4.1
amendment / §5 amendment / §9.1, ADR 0021).

The one novel behaviour under test: ``shelved`` is the ONLY Fact lifecycle
value the dedup gate's candidate probe (``engine.filters.probe_filters_for``)
never returns, while staying fully visible to ordinary ``recall`` (SPEC §5)
and never reviving on a recall hit (the deliberate difference from
``archived``). Everything else this file tests exists to keep that one
property honest — see ADR 0021.

Dual-parameterized over both storage adapters (``adapter_id`` fixture +
``_build_engine`` helper), same construction pattern as ``tests/
test_set_audience.py`` / ``tests/test_hot.py`` — ``set_shelved`` persists
through the existing ``transition()``/``append_event()`` primitives (no new
adapter method), but the dual-parameterized run still proves the LanceDB
mirror agrees with the in-memory reference adapter. The raw storage-seam
round-trip (the new ``lifecycle="shelved"``/``source="shelved"`` DATA
VALUES flowing through ``transition``/``append_event``/``query_events``,
including survival across a reopen) is covered directly against the
``adapter`` fixture in ``tests/conformance/test_conformance.py``'s
``TestShelvedLifecycle`` — this file is the engine-level behaviour suite.

Falsifiability (recorded in the task verdict, not duplicated here):
  - Putting ``shelved`` back into ``probe_filters_for``'s ``lifecycle_in`` set
    makes ``TestProbeExclusion::
    test_near_duplicate_of_shelved_record_is_not_bounced`` go red on both
    ``[memory]`` and ``[lancedb]`` — the near-duplicate save bounces instead
    of proceeding novel. Confirmed; reverted.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

AnyShelvedAdapter = MemoryAdapter | LanceDBAdapter

# Large dim, same rationale as the sibling per-role-grounding test files:
# keeps FakeEmbedder's SHA-256 slot-collision probability negligible across
# this file's many near-identical fixture strings.
_DIM = 4096

_ALL_FOUR = ("decider", "builder", "reviewer", "navigator")


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


async def _build_engine(
    adapter_id: str,
    tmp_path: pathlib.Path,
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
) -> tuple[Tabularium, AnyShelvedAdapter, FakeEmbedder, TestClock]:
    """A fresh, initialized engine + adapter + embedder + clock.

    Defaults to a BARE ``EngineConfig()`` (``gray_floor=0.80`` included) —
    the real production default, matching ``tests/test_set_audience.py``'s
    own rationale: the probe-exclusion property must hold under the config
    a real deployment actually runs, not a relaxed test override.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else _make_clock()
    adapter: AnyShelvedAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    engine = Tabularium(
        adapter,
        embedder,
        clk,
        cfg if cfg is not None else EngineConfig(),
        token_counter=FakeTokenCounter(),
    )
    await engine.initialize()
    return engine, adapter, embedder, clk


@pytest.fixture(params=["memory", "lancedb"])
def adapter_id(request: pytest.FixtureRequest) -> str:
    param: str = request.param
    return param


def _direct_fact(
    *,
    content: str,
    lifecycle: str,
    audience_hint: tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> Fact:
    """Build a Fact for direct ``adapter.upsert`` — bypasses the engine for
    lifecycle states ``set_shelved``'s own legal-edge guard cannot reach
    directly (e.g. constructing a ``shelved`` fixture to test something
    OTHER than the transition itself, such as context exclusion)."""
    t = now or datetime(2025, 1, 1, tzinfo=UTC)
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope("global"),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        audience_hint=audience_hint,
        created_at=t,
        updated_at=t,
    )


# ---------------------------------------------------------------------------
# The marquee property: the dedup probe never sees a shelved record
# ---------------------------------------------------------------------------


class TestProbeExclusion:
    async def test_near_duplicate_of_shelved_record_is_not_bounced(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The whole point of ADR 0021: shelving withdraws the dedup gate's
        DEFENCE of content, so a later save of the identical atom succeeds
        as a fresh, novel record rather than bouncing against its shelved
        predecessor.

        Falsifiability: see this module's docstring.
        """
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "the quokka is the happiest animal alive, curated away"

        r1 = await engine.save(content, scope="global")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        fid = r1.hypomnema.id

        await engine.set_shelved(fid, True, reason="curator judged this not worth carrying")

        r2 = await engine.save(content, scope="global")
        assert r2.outcome == "saved", (
            f"a near-duplicate of shelved content must not bounce -- got "
            f"outcome={r2.outcome!r}"
        )
        assert r2.hypomnema is not None
        assert r2.hypomnema.id != fid, "the duplicate must be a genuinely NEW record"

    async def test_mirror_active_duplicate_is_bounced(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The mirror: identical content against an ACTIVE (never shelved)
        record still resolves to the SAME pre-existing record (P7
        admission-order piece, step 1 — ``already_saved``, not a new row)
        under the same default config — proves the first test's "not
        bounced, genuinely new record" result comes from shelving, not from
        an unrelated gate misconfiguration."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "the same content, kept active this whole time"

        r1 = await engine.save(content, scope="global")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None

        r2 = await engine.save(content, scope="global")
        assert r2.outcome == "already_saved"
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == r1.hypomnema.id


# ---------------------------------------------------------------------------
# No revival on a recall hit
# ---------------------------------------------------------------------------


class TestNoRevivalOnRecallHit:
    async def test_shelved_record_returned_by_recall_lifecycle_unchanged(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A shelved record is returned by an ordinary ``recall`` (SPEC §5)
        and its lifecycle is STILL ``shelved`` afterwards — asserted after
        the hit, not before (the no-revival property)."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "a fact that will be shelved, then recalled"

        r1 = await engine.save(content, scope="global")
        assert r1.hypomnema is not None
        fid = r1.hypomnema.id
        await engine.set_shelved(fid, True, reason="not worth carrying")

        result = await engine.recall(content, scope="global")
        hit = next((h for h in result.hits if h.id == fid), None)
        assert hit is not None, "an ordinary recall must still return a shelved record"
        assert hit.lifecycle == "shelved"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "shelved", (
            "a recall hit on a shelved record must never revive it -- "
            f"got lifecycle={rec.lifecycle!r} after the hit"
        )

    async def test_shelved_record_returned_under_include_archived_too_without_revival(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``include_archived=True``'s lifecycle FILTER is a superset of the
        default ``recall`` call's (SPEC §5) — ``shelved`` must not be
        STRUCTURALLY excluded from it. The fallback filter used for an explicit
        ``include_archived=True`` history query initially stayed ``{active,
        hot, ember, archived}`` — one value shy of the primary filter's new
        ``shelved`` inclusion — so a shelved record was unconditionally,
        100% reproducibly dropped from every history query regardless of
        competing candidates. Fixed by adding ``shelved`` to
        ``fallback_recall_filters``.

        This test proves that specific structural-omission bug stays fixed
        — it does NOT claim that a shelved hit
        survives ``include_archived=True`` under adversarial candidate
        volume: like every retrieval call in this system, results are
        bounded by ``candidate_k`` per leg (no unbounded mode — ADR 0018),
        so enough higher-scoring ``archived`` competitors could in
        principle push ANY lifecycle value (shelved included) out of a
        single call's bounded top-k — a pre-existing, general property of
        bounded top-k retrieval over a widened filter that predates this
        task and applies identically to ``active``/``hot``/``ember``
        already sharing that filter, not something specific to or
        introduced by ``shelved``. Redesigning ``engine.recall``'s
        retrieval shape to guarantee survival under arbitrary competitor
        volume is out of this task's scope. This fixture has exactly one
        competing record (none), so it is deliberately not exercising that
        separate, pre-existing property."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "a fact shelved, then recalled under both recall modes"

        r1 = await engine.save(content, scope="global")
        assert r1.hypomnema is not None
        fid = r1.hypomnema.id
        await engine.set_shelved(fid, True, reason="not worth carrying")

        default_result = await engine.recall(content, scope="global")
        default_hit = next((h for h in default_result.hits if h.id == fid), None)
        assert default_hit is not None
        assert default_hit.lifecycle == "shelved"

        history_result = await engine.recall(content, scope="global", include_archived=True)
        history_hit = next((h for h in history_result.hits if h.id == fid), None)
        assert history_hit is not None, (
            "include_archived=True's FILTER must not structurally exclude shelved "
            "-- it must not silently drop a shelved record the default call "
            "returned when nothing else is competing for the bounded top-k"
        )
        assert history_hit.lifecycle == "shelved"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "shelved", (
            "neither recall mode may revive a shelved record -- "
            f"got lifecycle={rec.lifecycle!r} after both hits"
        )


# ---------------------------------------------------------------------------
# Never reaches context()
# ---------------------------------------------------------------------------


class TestContextExclusion:
    async def test_shelved_record_never_in_context_for_any_role(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A shelved record never appears in a ``context`` payload for any
        of the four roles — automatic (the packer only ever reads
        ``hot``/``ember``, ADR 0021), proven here even for a shelved record
        hinted at every role."""
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)

        content = "a shelved fact hinted at every registered role"
        fact = _direct_fact(content=content, lifecycle="shelved", audience_hint=_ALL_FOUR)
        await adapter.upsert(fact, embedder.peek(content))

        for role in _ALL_FOUR:
            payload = await engine.context("global", 10_000, role=role)
            seen_ids = {
                entry.id for tier in payload.tiers for entry in tier.entries
            }
            assert fact.id not in seen_ids, (
                f"a shelved record must never reach context() -- found it "
                f"in role={role!r}'s payload"
            )


# ---------------------------------------------------------------------------
# Legal edges
# ---------------------------------------------------------------------------


class TestLegalEdges:
    async def test_active_to_shelved(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("an active fact headed for shelving", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        result = await engine.set_shelved(fid, True, reason="curated away")
        assert result.outcome == "revised"
        assert result.lifecycle == "shelved"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "shelved"

    async def test_archived_to_shelved(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "an archived fact headed for shelving"
        fact = _direct_fact(content=content, lifecycle="archived")
        await adapter.upsert(fact, embedder.peek(content))

        result = await engine.set_shelved(fact.id, True, reason="curated away, was cold")
        assert result.outcome == "revised"
        assert result.lifecycle == "shelved"

    async def test_shelved_to_active(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "a shelved fact headed back to active"
        fact = _direct_fact(content=content, lifecycle="shelved")
        await adapter.upsert(fact, embedder.peek(content))

        result = await engine.set_shelved(fact.id, False, reason="curator reversed the call")
        assert result.outcome == "revised"
        assert result.lifecycle == "active"

    async def test_shelved_to_hot_via_set_hot_raises_naming_unshelve(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``shelved -> hot`` directly via ``set_hot`` is illegal — un-shelve
        first, mirroring the existing "archived Facts are not promotable
        via set_hot" rule (``engine/dispositions.py``)."""
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "a shelved fact someone tries to promote directly"
        fact = _direct_fact(content=content, lifecycle="shelved")
        await adapter.upsert(fact, embedder.peek(content))

        with pytest.raises(ToolError, match="set_shelved"):
            await engine.set_hot(fact.id, hot=True, roles=["builder"])

        rec = await adapter.get_by_id(fact.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "shelved", "the rejected call must not write anything"

    async def test_hot_cannot_be_shelved_directly(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact promoted hot, then shelving attempted", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["builder"])

        with pytest.raises(ToolError, match="set_hot"):
            await engine.set_shelved(fid, True, reason="try to shelve a hot fact")

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "hot", "the rejected call must not write anything"

    async def test_ember_cannot_be_shelved_directly(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.ember(
            "an ember, then shelving attempted",
            scope="global",
            valence="guideline",
            asserted_by="owner",
            audience_hint=["builder"],
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        with pytest.raises(ToolError, match="ADR 0021"):
            await engine.set_shelved(eid, True, reason="try to shelve an ember")

        rec = await adapter.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "ember", "the rejected call must not write anything"


# ---------------------------------------------------------------------------
# Idempotency (SPEC §9.2 monotonic re-op)
# ---------------------------------------------------------------------------


class TestIdempotency:
    async def test_reshelve_already_shelved_is_noop(
        self, adapter_id: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "already shelved, re-shelved with the same verb"
        fact = _direct_fact(content=content, lifecycle="shelved")
        await adapter.upsert(fact, embedder.peek(content))

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("adapter.transition must not be called on a true no-op")

        monkeypatch.setattr(adapter, "transition", _boom)

        result = await engine.set_shelved(fact.id, True, reason="re-shelve, no-op")
        assert result.outcome == "unchanged"
        assert result.lifecycle == "shelved"

    async def test_reunshelve_already_active_is_noop(
        self, adapter_id: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("already active, un-shelved with the same verb", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("adapter.transition must not be called on a true no-op")

        monkeypatch.setattr(_adapter, "transition", _boom)

        result = await engine.set_shelved(fid, False, reason="un-shelve, no-op")
        assert result.outcome == "unchanged"
        assert result.lifecycle == "active"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestSetShelvedErrors:
    async def test_unknown_id_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        with pytest.raises(ToolError, match="Unknown id"):
            await engine.set_shelved(new_id(), True, reason="unknown id")

    async def test_decision_id_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "a decision for the set_shelved Facts-only guard",
            scope="global",
            kind="decision",
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id

        with pytest.raises(ToolError, match="Decision"):
            await engine.set_shelved(did, True, reason="try to shelve a decision")

    async def test_empty_reason_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact for the empty-reason guard", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="reason"):
            await engine.set_shelved(fid, True, reason="")

    async def test_unshelve_on_a_never_shelved_archived_fact_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``shelved=False`` on an archived (never-shelved) Fact does not
        silently apply -- "un-shelve" has no meaning for a Fact this tool
        never shelved."""
        engine, adapter, embedder, _clock = await _build_engine(adapter_id, tmp_path)
        content = "an archived fact, never shelved, un-shelve attempted"
        fact = _direct_fact(content=content, lifecycle="archived")
        await adapter.upsert(fact, embedder.peek(content))

        with pytest.raises(ToolError, match="shelved"):
            await engine.set_shelved(fact.id, False, reason="nonsensical un-shelve")

        rec = await adapter.get_by_id(fact.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "archived", "the rejected call must not write anything"
