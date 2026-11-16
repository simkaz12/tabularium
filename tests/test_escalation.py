"""Ember recurrence ladder — owner/agent split + rung-2 crossing (M2, SPEC
§8.4, ADR 0020's Foreground-exit amendment).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``adapter_id`` fixture + ``_build_engine`` helper, mirroring ``tests/
test_hot.py``'s pattern (task precedent) rather than a third construction
pattern — this feature touches the LanceDB schema/codec directly (three new
columns), so the LanceDB leg is load-bearing here, not incidental.

Falsifiability (per the M2 grounding-slice-C brief): every test below has
been proven sensitive to a real defect by deliberately mutating
``src/tabularium/engine/embers.py`` / ``src/tabularium/engine/dispositions.py``,
watching the named test go red for the stated reason, then restoring. The
mutation-per-test map is recorded in the dispatching agent's verdict, not
duplicated here — each test's docstring names the mutation that would break
it, for a future reader.

Re-arm reachability note (test_rearm_after_baseline_reset_allows_a_second_
crossing): a real end-to-end round trip through ``set_hot(hot=True)`` then
``set_hot(hot=False)`` on the SAME ember cannot, by construction, exercise a
SECOND crossing — promotion moves the record's lifecycle to "hot" (SPEC
§1.2: ember is write-time-seeded, never re-entered), so no further
``engine.ember()`` call can ever route back through the ember-recurrence
dispatch branch for that id. The re-arm mechanic is therefore tested in two
parts that together prove the whole property: (1) a real ``set_hot(hot=
False)`` call, on a Fact whose recurrence counters were advanced first,
correctly resets ``escalation_baseline`` (proving the disposition-side
write), and (2) a direct adapter-level baseline reset (standing in for what
(1) already proved happens for real) shows a fresh crossing DOES fire again
from the new baseline (proving the arithmetic). No single test claims to
exercise both halves in one continuous ``ember()`` → ``set_hot()`` →
``ember()`` chain, because that chain is not reachable given ember's
one-way lifecycle edge.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact
from tabularium.engine import Tabularium, ToolError
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock

AnyEscalationAdapter = MemoryAdapter | LanceDBAdapter

# Large dim to keep FakeEmbedder's SHA-256 slot-collision probability
# negligible across this file's many near-identical fixture strings — same
# rationale/measurement as tests/test_hot.py.
_DIM = 4096


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {
        "candidate_k": 1000,
        "gray_floor": 0.80,
        "escalation_threshold": 3,
    }
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _build_engine(
    adapter_id: str,
    tmp_path: pathlib.Path,
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
) -> tuple[Tabularium, AnyEscalationAdapter, FakeEmbedder, TestClock]:
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else _make_clock()
    adapter: AnyEscalationAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    engine = Tabularium(adapter, embedder, clk, cfg if cfg is not None else _make_config())
    await engine.initialize()
    return engine, adapter, embedder, clk


@pytest.fixture(params=["memory", "lancedb"])
def adapter_id(request: pytest.FixtureRequest) -> str:
    param: str = request.param
    return param


# ---------------------------------------------------------------------------
# Owner vs agent split
# ---------------------------------------------------------------------------


class TestOwnerAgentSplit:
    async def test_agent_recurrence_never_advances_owner_count_or_fires_rung2(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change ``record_ember_recurrence``'s ``else`` branch
        to also increment ``owner_count`` (agent bumping owner too) — this
        test goes red because ``owner_recurrence_count`` would climb past 0
        and, past ``escalation_threshold`` reinforcements, a suggestion would
        fire even though every reinforcement below is agent-sourced (the
        engine.ember default — no ``asserted_by`` passed).
        """
        cfg = _make_config(escalation_threshold=3)
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "agent-only recurring ember, never owner-asserted"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        # Reinforce well past escalation_threshold, purely agent-sourced
        # (no asserted_by passed — the honest "we cannot verify" default).
        for _ in range(6):
            rep = await engine.ember(content, scope="global", valence="guideline")
            assert rep.outcome == "reinforced"
            assert rep.escalation_suggestion is None

        rec = await adapter.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.owner_recurrence_count == 0
        assert rec.agent_recurrence_count == 6

    async def test_owner_recurrence_advances_owner_count(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change the ``asserted_by == "owner"`` branch to
        increment ``agent_count`` instead of ``owner_count`` — this test
        goes red because ``owner_recurrence_count`` would stay 0.
        """
        cfg = _make_config(escalation_threshold=10)  # high enough not to fire here
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "owner-asserted recurring ember below threshold"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        for _ in range(3):
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
            assert rep.outcome == "reinforced"

        rec = await adapter.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.owner_recurrence_count == 3
        assert rec.agent_recurrence_count == 0


# ---------------------------------------------------------------------------
# Rung-2 crossing, not standing value
# ---------------------------------------------------------------------------


class TestRung2Crossing:
    async def test_fires_exactly_on_the_crossing_event(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """escalation_threshold=3: the 1st and 2nd owner-asserted
        reinforcements must NOT carry a suggestion; the 3rd (the crossing)
        must."""
        cfg = _make_config(escalation_threshold=3)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "standing rule that keeps being violated"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None

        results = []
        for _ in range(3):
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
            results.append(rep)

        assert results[0].escalation_suggestion is None
        assert results[1].escalation_suggestion is None
        assert results[2].escalation_suggestion is not None
        assert results[2].escalation_suggestion.owner_recurrence_count == 3
        assert results[2].escalation_suggestion.threshold == 3

    async def test_does_not_refire_on_repeated_events_past_threshold(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change the crossing test from
        ``prev < target <= owner_count`` to ``owner_count >= target`` (drop
        the ``prev`` half) — this test goes red because every subsequent
        owner-asserted reinforcement past the threshold would ALSO carry a
        suggestion (a nag), not just the one that first crossed it.
        """
        cfg = _make_config(escalation_threshold=3)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "standing rule recurring far past the threshold"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None

        suggestions = []
        for _ in range(8):  # crosses at #3, then 5 more owner-asserted events
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
            suggestions.append(rep.escalation_suggestion)

        fired = [i for i, s in enumerate(suggestions) if s is not None]
        assert fired == [2], f"expected exactly one crossing at index 2, got {fired}"

    async def test_suggestion_names_both_exits(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change ``_make_escalation_exits`` to return the
        ``codify`` exit twice — this test goes red because the action set
        collapses to a single value instead of ``{"codify", "foreground"}``.
        """
        cfg = _make_config(escalation_threshold=2)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "ember whose suggestion payload we inspect"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        ember_id = r.hypomnema.id

        rep = None
        for _ in range(2):
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
        assert rep is not None
        suggestion = rep.escalation_suggestion
        assert suggestion is not None
        assert suggestion.ember_id == ember_id
        actions = {e.action for e in suggestion.exits}
        assert actions == {"codify", "foreground"}
        for exit_ in suggestion.exits:
            assert exit_.how  # non-empty
            assert exit_.when  # non-empty
            # the exact call, not a placeholder — the real id is interpolated
            assert str(ember_id) in exit_.how


def test_escalation_exit_how_strings_quote_the_uuid() -> None:
    """Both ``EscalationExit.how`` templates must quote the interpolated
    ``ember_id`` — a bare UUID is a syntax error in both Python and JSON
    argument text, uncopyable by an agent.

    Falsifiability: revert either ``f'...{ember_id!s}...'`` interpolation in
    ``_make_escalation_exits`` back to an unquoted form and this test goes
    red — the quoted-id substring disappears from that exit's ``how``.
    """
    from tabularium.domain import new_id
    from tabularium.engine.ember_recurrence import _make_escalation_exits

    ember_id = new_id()
    codify, foreground = _make_escalation_exits(ember_id)
    assert f'"{ember_id!s}"' in codify.how
    assert f'"{ember_id!s}"' in foreground.how


# ---------------------------------------------------------------------------
# set_hot on an ember (Foreground exit) — succeeds; Decision still ToolError
# ---------------------------------------------------------------------------


class TestForegroundExit:
    async def test_set_hot_promotes_an_ember_to_hot(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: revert the ``set_hot`` promote-check from
        ``record.lifecycle not in ("active", "ember")`` back to ``!=
        "active"`` — this test goes red with a ``ToolError`` where success
        was expected (this is the exact regression test_hot.py's removed
        ``test_ember_cannot_be_promoted`` used to pin the OLD behaviour).
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.ember(
            "standing rule to foreground via set_hot", scope="global", valence="guideline"
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        result = await engine.set_hot(eid, hot=True, roles=["decider"])
        assert isinstance(result, Fact)
        assert result.lifecycle == "hot"

        rec = await adapter.get_by_id(eid)
        assert rec.lifecycle == "hot"

    async def test_set_hot_on_decision_still_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Unchanged behaviour — Decisions were never eligible and stay
        that way; guards against a promote-check relaxation that
        accidentally widens past Facts. ``roles`` is supplied so the
        raise is provably about the Decision/Fact guard, not the
        (curation block) required-roles guard."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "a decision for the foreground-exit exclusion check",
            scope="global",
            kind="decision",
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.set_hot(did, hot=True, roles=["decider"])


# ---------------------------------------------------------------------------
# Demotion — lands on active (never ember), preserves counters, re-arms
# ---------------------------------------------------------------------------


class TestDemotionReArm:
    async def test_demote_lands_on_active_and_preserves_counters(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Promote an ember (with nonzero recurrence already on it) to hot,
        then demote: lifecycle must read "active" — never "ember" (SPEC
        §8.4/§8.5: ember is a write-time-seeded provisional state; a record
        that has been through the ladder is not) — and the recurrence
        counters must survive both transitions unchanged in value.
        """
        cfg = _make_config(escalation_threshold=10)  # avoid firing mid-setup
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "ember with recurrence history, later foregrounded then demoted"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        for _ in range(2):
            await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
        await engine.ember(content, scope="global", valence="guideline")  # one agent event too

        pre = await adapter.get_by_id(eid)
        assert isinstance(pre, Fact)
        assert pre.owner_recurrence_count == 2
        assert pre.agent_recurrence_count == 1

        await engine.set_hot(eid, hot=True, roles=["decider"])
        result = await engine.set_hot(eid, hot=False)
        assert result.lifecycle == "active"

        rec = await adapter.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "active"
        assert rec.owner_recurrence_count == 2
        assert rec.agent_recurrence_count == 1

    async def test_set_hot_demotion_resets_escalation_baseline(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: delete the ``if not hot and updated.
        escalation_baseline != updated.owner_recurrence_count:`` re-arm
        block in ``set_hot`` — this test goes red because
        ``escalation_baseline`` would stay 0 after demotion instead of
        matching the Fact's ``owner_recurrence_count``.

        Uses a plain (non-ember-origin) hot Fact whose recurrence counters
        are advanced directly via ``adapter.update_recurrence`` — standing
        in for what a real ember's history would look like, since a real
        Foreground promotion already moves lifecycle off "ember" (see the
        module docstring's reachability note).
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact promoted straight to hot", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        # Simulate prior ember-recurrence history on this (now-hot) Fact.
        await adapter.update_recurrence(
            fid, owner_recurrence_count=5, agent_recurrence_count=2, escalation_baseline=0
        )

        await engine.set_hot(fid, hot=False)

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "active"
        assert rec.owner_recurrence_count == 5  # untouched
        assert rec.escalation_baseline == 5  # re-armed to the current count

    async def test_rearm_after_baseline_reset_allows_a_second_crossing(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: hardcode ``record_ember_recurrence`` to always
        compare against ``cfg.escalation_threshold`` alone (ignoring
        ``record.escalation_baseline``) — this test goes red because the
        second batch of owner-asserted events (which land above the
        threshold in absolute terms) would fail to produce a NEW crossing,
        since the crossing already happened once and a baseline-blind check
        cannot distinguish "past threshold again" from "still past
        threshold from before".

        See the module docstring for why this uses a direct
        ``adapter.update_recurrence`` baseline reset rather than a real
        ``set_hot`` demotion — the latter is proven separately in
        ``test_set_hot_demotion_resets_escalation_baseline`` above.
        """
        cfg = _make_config(escalation_threshold=3)
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        content = "ember that gets re-armed after a simulated demotion"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        first_crossing_at: int | None = None
        for i in range(3):
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
            if rep.escalation_suggestion is not None:
                first_crossing_at = i
        assert first_crossing_at == 2

        # Re-arm: reset the baseline to the current standing count (what a
        # real set_hot(hot=False) demotion performs — proven separately).
        rec = await adapter.get_by_id(eid)
        assert isinstance(rec, Fact)
        await adapter.update_recurrence(
            eid,
            owner_recurrence_count=rec.owner_recurrence_count,
            agent_recurrence_count=rec.agent_recurrence_count,
            escalation_baseline=rec.owner_recurrence_count,
        )

        second_crossing_at: int | None = None
        for i in range(3):
            rep = await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=["decider"],
            )
            if rep.escalation_suggestion is not None:
                second_crossing_at = i
        assert second_crossing_at == 2, (
            "a fresh crossing from the new baseline must fire again — the "
            "ladder must be able to re-arm, not fire once ever"
        )


# ---------------------------------------------------------------------------
# Counters are not §8.2 projections — survive an unrelated projection update
# ---------------------------------------------------------------------------


async def test_recurrence_counters_are_not_touched_by_ordinary_reinforce(
    tmp_path: pathlib.Path,
) -> None:
    """An active/hot (non-ember) Fact reinforcement must never touch the
    recurrence-ladder counters — they are ember-dispatch-only (SPEC §8.4).
    Uses the memory adapter only; this is a dispatch-routing property, not
    an adapter/codec conformance concern (already covered dual-adapter
    above)."""
    cfg = _make_config(escalation_threshold=3)
    engine, adapter, _embedder, clock = await _build_engine("memory", tmp_path, cfg=cfg)
    r = await engine.save("a plain active fact, never an ember", scope="global")
    assert r.hypomnema is not None
    fid = r.hypomnema.id

    for _ in range(4):
        clock.advance(timedelta(hours=1))
        # A whitespace variant (P7 admission-order piece, step 3 exact-key
        # match) reinforces via the exact-key path — a byte-identical
        # resubmission acks already_saved instead (step 1), which never
        # reinforces.
        rep = await engine.save("a plain active fact, never an ember ", scope="global")
        assert rep.outcome == "reinforced"

    rec = await adapter.get_by_id(fid)
    assert isinstance(rec, Fact)
    assert rec.owner_recurrence_count == 0
    assert rec.agent_recurrence_count == 0
    assert rec.escalation_baseline == 0


# ---------------------------------------------------------------------------
# LanceDB open-time schema migration — regression for a real gap caught by
# review: a store created BEFORE this diff has no
# owner_recurrence_count / agent_recurrence_count / escalation_baseline
# columns, and every write/update path now builds Arrow rows against the
# NEW schema.
# ---------------------------------------------------------------------------


async def test_lancedb_reopen_migrates_missing_recurrence_columns(
    tmp_path: pathlib.Path,
) -> None:
    """Falsifiability: comment out the records-table ``_migrate_table_schema``
    call in ``LanceDBAdapter.open`` (or empty ``_SCALAR_DEFAULTS``) — this test goes
    red because the reopened adapter's ``update_recurrence`` /
    ``transition`` / ``upsert`` calls against the pre-existing row raise
    (Arrow schema mismatch: the row dict ``record_to_row`` builds carries
    the three new keys, the physical table does not) instead of succeeding
    with the columns transparently backfilled to 0.
    """
    import lancedb
    import pyarrow as pa

    from tabularium.domain import Scope, new_id
    from tabularium.storage.lancedb_codec import make_records_schema, record_to_row

    store_path = str(tmp_path / "old_schema_store")
    dim = 8
    embedding_model = "test-model"
    now = datetime(2025, 1, 1, tzinfo=UTC)

    # Build the OLD schema: identical to the current one minus the three
    # recurrence columns this diff adds.
    new_cols = {"owner_recurrence_count", "agent_recurrence_count", "escalation_baseline"}
    current_schema = make_records_schema(dim)
    old_fields = [f for f in current_schema if f.name not in new_cols]
    old_schema = pa.schema(old_fields)

    # Write one row directly against the OLD schema — bypassing this
    # adapter entirely, simulating data written by the pre-diff codebase.
    old_fact_id = new_id()
    old_fact = Fact(
        id=old_fact_id,
        type="fact",
        content="a fact saved before the recurrence-ladder columns existed",
        scope=Scope("global"),
        lifecycle="active",
        created_at=now,
        updated_at=now,
    )
    vector = [0.1] * dim
    full_row = record_to_row(old_fact, vector, embedding_model)
    old_field_names = {f.name for f in old_fields}
    old_row = {k: v for k, v in full_row.items() if k in old_field_names}

    db = await lancedb.connect_async(store_path)
    await db.create_table("records", schema=old_schema)
    old_table = await db.open_table("records")
    await old_table.add([old_row])

    # Reopen with the CURRENT adapter — must migrate the table in place.
    adapter = await LanceDBAdapter.open(
        store_path, embedding_model=embedding_model, dim=dim
    )

    migrated = await adapter.get_by_id(old_fact_id)
    assert isinstance(migrated, Fact)
    assert migrated.owner_recurrence_count == 0
    assert migrated.agent_recurrence_count == 0
    assert migrated.escalation_baseline == 0

    # update_recurrence
    await adapter.update_recurrence(
        old_fact_id, owner_recurrence_count=2, agent_recurrence_count=1, escalation_baseline=0
    )
    rec = await adapter.get_by_id(old_fact_id)
    assert isinstance(rec, Fact)
    assert rec.owner_recurrence_count == 2
    assert rec.agent_recurrence_count == 1

    # transition — counters must survive a lifecycle change untouched
    await adapter.transition(old_fact_id, "hot", at=now)
    rec2 = await adapter.get_by_id(old_fact_id)
    assert isinstance(rec2, Fact)
    assert rec2.lifecycle == "hot"
    assert rec2.owner_recurrence_count == 2

    # upsert — re-saving the current (migrated) record must round-trip clean
    await adapter.upsert(rec2, vector)
    rec3 = await adapter.get_by_id(old_fact_id)
    assert isinstance(rec3, Fact)
    assert rec3.owner_recurrence_count == 2
    assert rec3.escalation_baseline == 0

    # A second reopen (migration already applied) must be a no-op, not an error.
    adapter2 = await LanceDBAdapter.open(
        store_path, embedding_model=embedding_model, dim=dim
    )
    rec4 = await adapter2.get_by_id(old_fact_id)
    assert isinstance(rec4, Fact)
    assert rec4.owner_recurrence_count == 2


# ---------------------------------------------------------------------------
# Per-role grounding — ember(roles=...) retags an already-ember record (this
# brief, curation block, mirroring set_hot's roles param). Overwrite,
# not union; roles=None leaves the existing hint untouched; an unregistered
# role is a ToolError, matching save/set_hot. The write path lives in
# _dispatch_match's ember-recurrence branch (engine/facts.py) — one
# ember(content, roles=[...]) call, on content that dedup-matches an
# existing ember, both reinforces it (the "reignite" — see the ``/ember``
# command) AND persists the new role tag: one call, one write, not two.
#
# "builder"/"reviewer" need no explicit adapter.create_role call — engine.
# initialize() (inside _build_engine) already seeds all five roles.
# ---------------------------------------------------------------------------


class TestEmberRoleRetagging:
    async def test_reignite_with_roles_overwrites_capture_time_hint(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Overwrite, not union: an ember captured with
        ``audience_hint=["builder"]`` then reignited via identical content
        plus ``roles=["reviewer"]`` (the incoming content shares the ember's
        canonical key, so the ember-recurrence branch reinforces — P7
        admission-order piece, decision D14 — rather than a fresh write)
        routes to reviewer and not to builder.

        Falsifiability (step 6): stop forwarding ``roles`` to ``transition``
        in the ember-recurrence branch (drop the persist call
        unconditionally) — the hint stays ``("builder",)`` and this test
        goes red.
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "atom captured with a builder hint, later curated to reviewer"
        r = await engine.ember(
            content, scope="global", valence="guideline", audience_hint=["builder"]
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        rep = await engine.ember(
            content, scope="global", valence="guideline", roles=["reviewer"]
        )
        assert rep.outcome == "reinforced"
        assert rep.reinforced_id == eid

        updated = await adapter.get_by_id(eid)
        assert isinstance(updated, Fact)
        assert updated.audience_hint == ("reviewer",)
        assert "builder" not in (updated.audience_hint or ())

    async def test_roles_none_leaves_existing_hint_untouched(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles=None`` (the default) passes ``None`` through to
        ``transition`` — untouched, so a plain reignite (no ``roles`` arg)
        is exactly what it is today; existing callers unaffected."""
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "atom captured with a builder hint, reignited with no roles arg"
        r = await engine.ember(
            content, scope="global", valence="guideline", audience_hint=["builder"]
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        rep = await engine.ember(content, scope="global", valence="guideline")
        assert rep.outcome == "reinforced"

        updated = await adapter.get_by_id(eid)
        assert isinstance(updated, Fact)
        assert updated.audience_hint == ("builder",)

    async def test_reignite_and_retag_is_one_call(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A single ``ember(content, roles=[...])`` call, on content matching
        an existing ember, both reinforces it (recurrence — the "reignite")
        AND persists the new role tag: one call does both halves, not a
        reignite followed by a separate retag call.
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "atom reignited and retagged in the same call"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        before = await adapter.get_by_id(eid)
        assert isinstance(before, Fact)
        assert before.agent_recurrence_count == 0

        rep = await engine.ember(
            content, scope="global", valence="guideline", roles=["reviewer"]
        )
        assert rep.outcome == "reinforced"

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.agent_recurrence_count == 1, "the reignite half must still fire"
        assert after.audience_hint == ("reviewer",), (
            "the retag half must land in the same call"
        )

    async def test_retag_already_recurring_ember_persists_new_hint(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The named trap: a role-only retag on an ember's SECOND reignite
        (already past its first recurrence, not a brand-new write) must
        still persist. An unconditional early return anywhere ahead of the
        roles-transition write in the ember-recurrence branch — the exact
        shape ``set_hot``'s own roles rollout hit — would silently no-op
        this, which is precisely the live-store curation path the feature
        exists for (most embers worth retagging have already recurred).

        Falsifiability: restore an unconditional early return before the
        roles-transition write in the ember-recurrence branch of
        ``_dispatch_match`` — the retag on this second reignite silently
        no-ops and this test goes red.
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "already-recurring ember retagged on its second reignite"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        # First reignite, unhinted — the ember is already past its first
        # recurrence before the retag under test, not a fresh write.
        rep1 = await engine.ember(content, scope="global", valence="guideline")
        assert rep1.outcome == "reinforced"

        rep2 = await engine.ember(
            content, scope="global", valence="guideline", roles=["reviewer"]
        )
        assert rep2.outcome == "reinforced"

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "ember", "retag must not disturb the lifecycle"
        assert after.audience_hint == ("reviewer",)

    async def test_empty_roles_clears_the_hint(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles=[]`` is the third tri-state leg — an explicit un-hint,
        clearing a previously-set audience so the ember goes back to
        universal reach."""
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "atom hinted then cleared back to universal reach"
        r = await engine.ember(
            content, scope="global", valence="guideline", audience_hint=["builder"]
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        rep = await engine.ember(content, scope="global", valence="guideline", roles=[])
        assert rep.outcome == "reinforced"

        updated = await adapter.get_by_id(eid)
        assert isinstance(updated, Fact)
        assert not updated.audience_hint


class TestEmberRoleRetaggingErrors:
    async def test_unregistered_role_rejected(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Matches ``save``/``set_hot``'s gate — an unregistered role in
        ``roles`` is a ``ToolError``, and nothing is written or reinforced
        (validated before the embed/dedup gate even runs).

        Falsifiability (step 6): remove the ``await require_roles(roles)``
        call in ``save_internal`` — this test goes red (no exception
        raised).
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "ember for the unregistered-role rejection test"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        with pytest.raises(ToolError, match="Unregistered role"):
            await engine.ember(
                content,
                scope="global",
                valence="guideline",
                roles=["never-registered"],
            )

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "ember", "rejected role must not write anything"
        assert after.agent_recurrence_count == 0, "rejected role must not reinforce either"
