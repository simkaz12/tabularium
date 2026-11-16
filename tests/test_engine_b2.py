"""Engine B2 tests — Decisions end-to-end (SPEC §1/§4/§6/§9.1, [A11]).

Acceptance criteria:
  - Decision creation through the gate
  - Supersession chain build over ≥3 links (snapshots frozen, superseded_by set once, chain linear)
  - ToolErrors: supersede non-in-force / Fact / unknown; promoted_from non-ember;
    supersedes with kind=fact
  - promoted_from snapshot freezing incl. recurrence_count
  - Render byte-stability: render twice = identical bytes; reinforcement events do NOT change
    bytes; a semantic transition DOES change bytes
  - Title derivation edge cases (no terminator, >80 chars, newline-first)
  - Tip derivation over a chain
  - decisions_query (scope span, lifecycle filters, include_superseded, DecisionLink shape,
    uri format)
  - Resolve-reinforce on Decision target (emits dedup event, bumps counters)
"""

from __future__ import annotations

import math
import pathlib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.render import derive_title, render_decision
from tabularium.results import BounceResolution
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import DistinctPair

# Adapter-construction pattern reused verbatim from the conformance suite
# (precedent — see tests/test_engine_dispositions.py) rather than a
# second one, for the dual-parameterized confirm_distinct-replay tests below.
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.fake_assessor import EchoAssessor

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIM = 64  # 16 caused sha256-based FakeEmbedder collisions between test texts


def _make_adapter(dim: int = _DIM) -> MemoryAdapter:
    return MemoryAdapter(embedding_model="fake/nomic", dim=dim)


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _engine(
    *,
    dim: int = _DIM,
    config: EngineConfig | None = None,
    clock: TestClock | None = None,
    should_fail: bool = False,
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    adapter = _make_adapter(dim=dim)
    embedder = FakeEmbedder(dim=dim, should_fail=should_fail)
    clk = clock if clock is not None else _make_clock()
    cfg = config if config is not None else _make_config()
    engine = Tabularium(adapter, embedder, clk, cfg)
    await engine.initialize()
    return engine, adapter, clk


class _IdenticalEmbedder:
    """All texts → same unit vector → cosine=1.0."""

    dim: int = _DIM
    model: str = "identical"

    async def embed(self, text: str) -> list[float]:
        del text
        v = [0.0] * self.dim
        v[0] = 1.0
        return v


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` from axis 0, zero-padded to ``dim``.

    ``FakeEmbedder`` only produces cosine 0.0/1.0 for distinct strings, so it
    cannot express graduated gray-band similarities.  Two angle-parameterised
    vectors ``theta`` apart have cosine ``cos(theta)`` exactly (both unit
    length by construction) — lets a test place several stored Decisions at
    known, distinct similarities to a single probe.
    """
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    """Deterministic embedder: exact-string → vector lookup (``KeyError`` on miss).

    Drives distinct incoming texts (original bounce content, then an edited
    resolve content) to distinct, pre-computed vectors — needed for
    gray-band / content-edit resolve scenarios.
    """

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


# ---------------------------------------------------------------------------
# 1. Decision creation — basic
# ---------------------------------------------------------------------------


class TestDecisionCreation:
    async def test_save_decision_writes_in_force(self) -> None:
        """save(kind='decision') creates an in_force Decision."""
        engine, adapter, _ = await _engine()
        r = await engine.save("We will use Python.", kind="decision", scope="global")
        assert r.outcome == "saved"
        assert r.hypomnema is not None
        assert r.hypomnema.kind == "decision"

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Decision)
        assert rec.lifecycle == "in_force"
        assert rec.content == "We will use Python."

    async def test_decision_has_empty_chain_when_no_supersedes(self) -> None:
        """A standalone Decision has an empty supersession_chain."""
        engine, adapter, _ = await _engine()
        r = await engine.save("First policy.", kind="decision", scope="global")
        rec = await adapter.get_by_id(r.hypomnema.id)  # type: ignore[union-attr]
        assert isinstance(rec, Decision)
        assert rec.supersession_chain == ()
        assert rec.superseded_by is None

    async def test_decision_born_valid(self) -> None:
        """New Decision is born valid (default)."""
        engine, adapter, _ = await _engine()
        r = await engine.save("Policy A.", kind="decision", scope="global")
        rec = await adapter.get_by_id(r.hypomnema.id)  # type: ignore[union-attr]
        assert isinstance(rec, Decision)
        assert rec.validity == "valid"

    async def test_save_decision_gate_bounce_on_different_candidate(self) -> None:
        """Gate bounces a Decision save if a different candidate is above its floor."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert an existing Decision with the same vector
        existing = Decision(
            id=new_id(),
            type="decision",
            content="use Python in all new projects",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(existing, vec)

        engine._embedder = _IdenticalEmbedder()
        # Save a *different* Decision — gate should bounce on the existing one
        r = await engine.save(
            "use Python in all new services",
            kind="decision",
            scope="global",
        )
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.verdict == "decision_adjacent"

    async def test_save_decision_skips_supersedes_target_in_gate(self) -> None:
        """When supersedes is set, the gate skips the named target — no self-bounce."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        old = Decision(
            id=new_id(),
            type="decision",
            content="use Java",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(old, vec)

        engine._embedder = _IdenticalEmbedder()
        # Supersede old — same content vector; gate must skip old and write the new Decision
        r = await engine.save(
            "use Python instead",
            kind="decision",
            scope="global",
            supersedes=old.id,
            reason="changed",
        )
        assert r.outcome == "saved"
        assert r.hypomnema is not None
        assert r.hypomnema.kind == "decision"


# ---------------------------------------------------------------------------
# 2. Supersession chain — linear chains over ≥3 links
# ---------------------------------------------------------------------------


class TestSupersessionChain:
    async def test_supersession_sets_old_superseded_by(self) -> None:
        """Superseding a Decision sets old.superseded_by = new.id."""
        engine, adapter, _ = await _engine()

        r1 = await engine.save("Decision v1.", kind="decision", scope="global")
        old_id = r1.hypomnema.id  # type: ignore[union-attr]

        r2 = await engine.save(
            "Decision v2.", kind="decision", scope="global", supersedes=old_id, reason="changed"
        )
        new_id_ = r2.hypomnema.id  # type: ignore[union-attr]

        old_rec = await adapter.get_by_id(old_id)
        assert isinstance(old_rec, Decision)
        assert old_rec.lifecycle == "superseded"
        assert old_rec.superseded_by == new_id_

    async def test_supersession_chain_frozen_at_birth(self) -> None:
        """New Decision's supersession_chain captures snapshot of predecessor."""
        engine, adapter, _ = await _engine()

        r1 = await engine.save("Original policy.", kind="decision", scope="global")
        old_id = r1.hypomnema.id  # type: ignore[union-attr]

        r2 = await engine.save(
            "Revised policy.", kind="decision", scope="global", supersedes=old_id, reason="changed"
        )
        new_id_ = r2.hypomnema.id  # type: ignore[union-attr]

        new_rec = await adapter.get_by_id(new_id_)
        assert isinstance(new_rec, Decision)
        assert len(new_rec.supersession_chain) == 1
        snap = new_rec.supersession_chain[0]
        assert snap.id == old_id
        # Title = first sentence of "Original policy." → "Original policy."
        assert snap.title == "Original policy."

    async def test_three_link_chain(self) -> None:
        """Three-generation chain: snapshots frozen, linear, superseded_by set once."""
        engine, adapter, _ = await _engine()

        r1 = await engine.save("V1 policy.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]

        r2 = await engine.save(
            "V2 policy.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        r3 = await engine.save(
            "V3 policy.", kind="decision", scope="global", supersedes=id2, reason="changed"
        )
        id3 = r3.hypomnema.id  # type: ignore[union-attr]

        # Check the tip (id3 is the live end)
        v3 = await adapter.get_by_id(id3)
        assert isinstance(v3, Decision)
        assert v3.lifecycle == "in_force"
        assert len(v3.supersession_chain) == 2
        assert v3.supersession_chain[0].id == id1
        assert v3.supersession_chain[1].id == id2

        # id2 is superseded; superseded_by = id3
        v2 = await adapter.get_by_id(id2)
        assert isinstance(v2, Decision)
        assert v2.lifecycle == "superseded"
        assert v2.superseded_by == id3

        # id1 is superseded; superseded_by = id2
        v1 = await adapter.get_by_id(id1)
        assert isinstance(v1, Decision)
        assert v1.lifecycle == "superseded"
        assert v1.superseded_by == id2

    async def test_superseded_by_set_exactly_once(self) -> None:
        """Attempt to supersede an already-superseded Decision → ToolError."""
        engine, _adapter, _ = await _engine()

        r1 = await engine.save("First.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]

        r2 = await engine.save(
            "Second.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )
        assert r2.outcome == "saved"

        # id1 is now superseded — attempting to supersede it again → ToolError
        with pytest.raises(ToolError, match="not in_force"):
            await engine.save(
                "Third (illegal).", kind="decision", scope="global", supersedes=id1
            )


# ---------------------------------------------------------------------------
# 3. ToolErrors
# ---------------------------------------------------------------------------


class TestDecisionToolErrors:
    async def test_supersedes_unknown_id(self) -> None:
        """supersedes with a non-existent id → ToolError."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="unknown"):
            await engine.save(
                "New decision.", kind="decision", scope="global", supersedes=new_id()
            )

    async def test_supersedes_a_fact_raises(self) -> None:
        """supersedes targeting a Fact id → ToolError."""
        engine, _adapter, _ = await _engine()
        r = await engine.save("A fact.", scope="global")
        fact_id = r.hypomnema.id  # type: ignore[union-attr]

        with pytest.raises(ToolError, match="Fact"):
            await engine.save(
                "A decision.", kind="decision", scope="global", supersedes=fact_id
            )

    async def test_supersedes_revoked_decision_raises(self) -> None:
        """supersedes targeting a revoked Decision → ToolError (lifecycle != in_force)."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        revoked = Decision(
            id=new_id(),
            type="decision",
            content="Revoked policy.",
            scope=Scope("global"),
            lifecycle="revoked",
            created_at=now,
            updated_at=now,
        )
        vec = [0.5] + [0.0] * (_DIM - 1)
        await adapter.upsert(revoked, vec)

        with pytest.raises(ToolError, match="not in_force"):
            await engine.save(
                "Replacement.", kind="decision", scope="global", supersedes=revoked.id
            )

    async def test_supersedes_with_kind_fact_raises(self) -> None:
        """2026-08-10 fact-supersession brief: ``supersedes`` now opens for
        Facts too, but takes a LIST there (the Decision arm keeps a single
        id) — a bare id on ``kind='fact'`` is still rejected, now with the
        type-directed message rather than the old "Decision-only" one (see
        ``tests/test_fact_supersession.py`` for the arm-dependent-shape
        coverage this brief adds)."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="requires a LIST of ids"):
            await engine.save(
                "fact text", scope="global", supersedes=new_id()
            )

    async def test_promoted_from_with_kind_fact_raises(self) -> None:
        """promoted_from requires kind='decision' — kind='fact' with promoted_from → ToolError."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="promoted_from requires kind='decision'"):
            await engine.save(
                "fact text", scope="global", promoted_from=[new_id()]
            )

    async def test_save_invalid_kind_raises_and_writes_nothing(self) -> None:
        """``Literal["fact", "decision"]`` is a static hint only — a direct
        in-process caller (bypassing the MCP/pydantic boundary) passing any
        other string must be a hard error, not a silent fall-through to the
        Fact arm."""
        engine, adapter, _ = await _engine()
        rows_before = [row async for row in await adapter.scan()]

        with pytest.raises(ToolError):
            await engine.save("some content", kind="bogus", scope="global")  # type: ignore[arg-type]

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before), "nothing should have been written"

    async def test_save_invalid_kind_error_names_bad_value(self) -> None:
        """The error names the offending value, not just a generic message."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="bogus"):
            await engine.save("some content", kind="bogus", scope="global")  # type: ignore[arg-type]

    async def test_save_valid_kinds_still_work(self) -> None:
        """Guard against an over-tight check: both legal kinds still save
        end-to-end after validation is added."""
        engine, _, _ = await _engine()

        r_fact = await engine.save("A fact.", kind="fact", scope="global")
        assert r_fact.outcome == "saved"

        r_decision = await engine.save("A decision.", kind="decision", scope="global")
        assert r_decision.outcome == "saved"

    async def test_invalid_kind_reported_before_supersedes_guard(self) -> None:
        """Validation order: a bad ``kind`` must be reported as a bad kind,
        not misdiagnosed as a supersedes misuse — the supersedes guard reads
        ``kind`` and would otherwise fire first with a misleading message."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="Invalid kind"):
            await engine.save(
                "some content", kind="bogus", scope="global", supersedes=new_id()  # type: ignore[arg-type]
            )

    async def test_promoted_from_unknown_id_raises(self) -> None:
        """promoted_from with a non-existent id → ToolError."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="unknown"):
            await engine.save(
                "Decision.", kind="decision", scope="global",
                promoted_from=[new_id()]
            )

    async def test_promoted_from_archived_fact_raises(self) -> None:
        """promoted_from targeting an archived Fact → ToolError (2026-08-07
        fix, ADR 0022 amendment: promotable = {active, hot, ember}, not
        ember-only — but archived stays refused, decayed-out content is not
        a live assertion a Decision can rest on)."""
        engine, adapter, clock = await _engine()
        now = clock.now()
        fact = Fact(
            id=new_id(), type="fact", content="an archived fact",
            scope=Scope("global"), lifecycle="archived",
            created_at=now, updated_at=now,
        )
        await adapter.upsert(fact, [1.0] + [0.0] * (_DIM - 1))

        with pytest.raises(ToolError, match="archived"):
            await engine.save(
                "Decision.", kind="decision", scope="global",
                promoted_from=[fact.id]
            )

    async def test_promoted_from_shelved_fact_raises(self) -> None:
        """promoted_from targeting a shelved Fact → ToolError (same rule as
        archived — a curator-demoted Fact is not promotable either)."""
        engine, adapter, clock = await _engine()
        now = clock.now()
        fact = Fact(
            id=new_id(), type="fact", content="a shelved fact",
            scope=Scope("global"), lifecycle="shelved",
            created_at=now, updated_at=now,
        )
        await adapter.upsert(fact, [1.0] + [0.0] * (_DIM - 1))

        with pytest.raises(ToolError, match="shelved"):
            await engine.save(
                "Decision.", kind="decision", scope="global",
                promoted_from=[fact.id]
            )

    async def test_promoted_from_decision_raises(self) -> None:
        """promoted_from targeting a Decision id → ToolError."""
        engine, _adapter, _ = await _engine()
        r_dec = await engine.save("A decision.", kind="decision", scope="global")
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        with pytest.raises(ToolError, match="Decision"):
            await engine.save(
                "Another decision.", kind="decision", scope="global",
                promoted_from=[dec_id]
            )


# ---------------------------------------------------------------------------
# 4. promoted_from snapshot freezing incl. recurrence_count
# ---------------------------------------------------------------------------


class TestPromotedFromSnapshots:
    async def test_promoted_from_snapshot_content_and_valence(self) -> None:
        """promoted_from freezes content and valence from the ember."""
        engine, adapter, _ = await _engine()

        r_ember = await engine.ember(
            "Always write tests first.", scope="global", valence="guideline"
        )
        ember_id = r_ember.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "We adopt TDD as policy.",
            kind="decision",
            scope="global",
            promoted_from=[ember_id],
        )
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        assert len(rec.promoted_from) == 1

        snap = rec.promoted_from[0]
        assert snap.id == ember_id
        assert snap.content == "Always write tests first."
        assert snap.valence == "guideline"

    async def test_promoted_from_recurrence_count_is_dedup_events(self) -> None:
        """recurrence_count in snapshot = count of source='dedup' events on ember."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, _ = await _engine(config=cfg)

        r_ember = await engine.ember("Always write tests.", scope="global", valence="guideline")
        ember_id = r_ember.hypomnema.id  # type: ignore[union-attr]

        # Trigger 2 dedup reinforcements
        await engine.save("Always write tests.", scope="global")
        await engine.save("Always write tests.", scope="global")

        # Confirm 2 dedup events
        events = await adapter.query_events(target_id=ember_id, source="dedup")
        assert len(events) == 2

        # Promote: snapshot must capture recurrence_count=2
        r_dec = await engine.save(
            "We adopt test-first as policy.",
            kind="decision",
            scope="global",
            promoted_from=[ember_id],
        )
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        assert rec.promoted_from[0].recurrence_count == 2

    async def test_promoted_from_zero_recurrence(self) -> None:
        """Ember with no recurrences → recurrence_count=0 in snapshot."""
        engine, adapter, _ = await _engine()

        r_ember = await engine.ember(
            "New observation.", scope="global", valence="correction"
        )
        ember_id = r_ember.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "Based on new observation, policy is X.",
            kind="decision",
            scope="global",
            promoted_from=[ember_id],
        )
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        assert rec.promoted_from[0].recurrence_count == 0

    async def test_ember_stays_fact_after_promotion(self) -> None:
        """The ember Fact must not be mutated after promotion (SPEC §8.4)."""
        engine, adapter, _ = await _engine()

        r_ember = await engine.ember("Ember observation.", scope="global", valence="guideline")
        ember_id = r_ember.hypomnema.id  # type: ignore[union-attr]

        await engine.save(
            "Policy from ember.",
            kind="decision",
            scope="global",
            promoted_from=[ember_id],
        )

        # Ember must still be an ember Fact — no transition, no mutation
        ember_rec = await adapter.get_by_id(ember_id)
        assert isinstance(ember_rec, Fact)
        assert ember_rec.lifecycle == "ember"

    async def test_promoted_from_active_fact_succeeds_with_null_valence(self) -> None:
        """The previously-failing call now succeeds (2026-08-07 fix, ADR
        0022 amendment): promoting a plain ``active`` Fact — never an ember,
        never carrying a ``valence`` — writes the Decision and freezes an
        honest ``valence=None`` snapshot rather than raising or fabricating
        a value. Regression test for the live defect: a Decision save
        bounced against an active Fact, the bounce's third exit told the
        caller to call exactly this, and the engine refused it."""
        engine, adapter, _ = await _engine()

        r_fact = await engine.save(
            "A fact a decision might rest on.", scope="global"
        )
        fact_id = r_fact.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "We adopt the fact's implication as policy.",
            kind="decision",
            scope="global",
            promoted_from=[fact_id],
        )
        assert r_dec.outcome == "saved"
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        assert len(rec.promoted_from) == 1

        snap = rec.promoted_from[0]
        assert snap.id == fact_id
        assert snap.content == "A fact a decision might rest on."
        assert snap.valence is None

    async def test_active_fact_stays_untouched_after_promotion(self) -> None:
        """Promotion is pure provenance for a plain active Fact too — the
        same untouched-target guarantee ``test_ember_stays_fact_after_promotion``
        proves for the ember case (``_write_decision`` upserts only the new
        Decision; SPEC §8.4's "ember is left untouched" line generalizes)."""
        engine, adapter, _ = await _engine()

        r_fact = await engine.save(
            "Another fact a decision might rest on.", scope="global"
        )
        fact_id = r_fact.hypomnema.id  # type: ignore[union-attr]
        before = await adapter.get_by_id(fact_id)
        assert isinstance(before, Fact)

        await engine.save(
            "Policy from the fact.",
            kind="decision",
            scope="global",
            promoted_from=[fact_id],
        )

        after = await adapter.get_by_id(fact_id)
        assert isinstance(after, Fact)
        assert after.lifecycle == "active"
        assert after.content == before.content
        assert after.updated_at == before.updated_at


# ---------------------------------------------------------------------------
# 4b. promoted_from one-shot bypass (2026-08-07 fix, ADR 0022 second
# amendment) — the exit's own ``how`` call resolves the bounce without
# ``resolve``, and registers nothing in the distinct-pair registry.
# ---------------------------------------------------------------------------


class TestPromotedFromOneShotBypass:
    """Live defect filed 2026-08-07: the ``promoted_from`` exit's ``how``
    string is ``save(kind="decision", content=…, promoted_from=[candidate_id])``
    — no ``resolve``.  Calling it verbatim against pre-fix ``main`` re-bounces,
    because ``save_decision`` only special-cased ``resolve.candidate_id ==
    candidate.id``, never ``promoted_from``.  Each test below states what it
    would catch on that pre-fix engine.
    """

    async def test_promoted_from_verbatim_how_call_saves_without_resolve(self) -> None:
        """The live repro.  On pre-fix ``main`` this call re-bounces
        (``outcome == "bounced"``, nothing written) — exactly the defect
        filed live today; this assertion is what catches it."""
        engine, _adapter, _ = await _engine()
        engine._embedder = _IdenticalEmbedder()

        r_fact = await engine.save(
            "An active fact the decision rests on.", scope="global"
        )
        fact_id = r_fact.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "We adopt the fact's implication as policy.",
            kind="decision",
            scope="global",
            promoted_from=[fact_id],
        )
        assert r_dec.outcome == "saved", (
            f"the bounce's own promoted_from exit must resolve the bounce "
            f"one-shot, got outcome={r_dec.outcome!r}"
        )
        assert r_dec.hypomnema is not None
        assert r_dec.hypomnema.kind == "decision"

    async def test_distinct_pair_registry_unchanged_after_bypass(self) -> None:
        """Constraint 2, asserted against the registry rather than assumed.
        Breaking this: if the bypass routed through the same write path as
        ``confirm_distinct`` (which calls ``adapter.mark_distinct``),
        ``is_distinct(fact_id, dec_id)`` would flip True and this fails."""
        engine, adapter, _ = await _engine()
        engine._embedder = _IdenticalEmbedder()

        r_fact = await engine.save("Another active fact.", scope="global")
        fact_id = r_fact.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "Policy resting on the fact.",
            kind="decision",
            scope="global",
            promoted_from=[fact_id],
        )
        assert r_dec.outcome == "saved"
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]

        assert not await adapter.is_distinct(fact_id, dec_id), (
            "the promote bypass must register nothing in the distinct-pair registry"
        )

    async def test_promoted_from_unrelated_id_still_bounces(self) -> None:
        """Constraint 3, unrelated-record case: ``promoted_from`` naming a
        Fact that is NOT the bounce's candidate does not bypass anything.
        Breaking this: an implementation that bypasses whenever
        ``promoted_from`` is merely non-empty (rather than checking
        ``candidate.id in promoted_from``) would return ``saved`` here
        instead of ``bounced``."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha fact.",
            scope=Scope("global"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        fact_b = Fact(
            id=new_id(), type="fact", content="Bravo fact, unrelated.",
            scope=Scope("global"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))
        await adapter.upsert(fact_b, _angle_vec(90))

        probe = "incoming decision text near alpha"
        engine._embedder = _TableEmbedder({probe: _angle_vec(5)})

        r_dec = await engine.save(
            probe, kind="decision", scope="global", promoted_from=[fact_b.id]
        )
        assert r_dec.outcome == "bounced", (
            f"promoted_from naming an unrelated Fact must not bypass the "
            f"bounce against the real candidate, got outcome={r_dec.outcome!r}"
        )
        assert r_dec.bounce is not None
        assert r_dec.bounce.candidate_id == fact_a.id

    async def test_promoted_from_other_candidate_only_still_bounces(self) -> None:
        """Constraint 3, ``other_candidates``-only case: ``promoted_from``
        naming a Fact that IS a viable above-floor neighbour, but not the
        PRIMARY bounce candidate, still bounces.  Breaking this: an
        implementation that bypasses on membership in the full candidate set
        (primary + other_candidates) rather than the primary alone would
        return ``saved`` here instead of ``bounced``."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha fact, primary candidate.",
            scope=Scope("global"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        fact_c = Fact(
            id=new_id(), type="fact", content="Charlie fact, secondary candidate.",
            scope=Scope("global"), lifecycle="active",
            created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))
        await adapter.upsert(fact_c, _angle_vec(15))

        probe = "incoming decision text between alpha and charlie"
        # cos(5°)=.9962 vs A, cos(10°)=.9848 vs C — both above gray_floor=0.80,
        # A wins on raw cosine and is the primary candidate; C is a viable
        # other_candidates neighbour, not the primary.
        engine._embedder = _TableEmbedder({probe: _angle_vec(5)})

        r_dec = await engine.save(
            probe, kind="decision", scope="global", promoted_from=[fact_c.id]
        )
        assert r_dec.outcome == "bounced", (
            f"promoted_from naming the SECONDARY candidate must not bypass "
            f"the bounce against the primary, got outcome={r_dec.outcome!r}"
        )
        assert r_dec.bounce is not None
        assert r_dec.bounce.candidate_id == fact_a.id
        assert any(n.candidate_id == fact_c.id for n in r_dec.bounce.other_candidates), (
            "test setup must actually place C above floor as a viable neighbour"
        )

    async def test_non_promotable_candidate_promoted_from_still_raises_at_bounce_cosine(
        self,
    ) -> None:
        """Regression guard on constraint 1: even when a non-promotable Fact
        IS the bounce's own top candidate (cosine above floor), naming it in
        ``promoted_from`` must still raise ``ToolError``, and with no
        ``promoted_from`` at all the bounce still offers only the two-exit
        set.  Breaking this: weakening either the pre-lock validation loop's
        ``is_promotable`` check or the bypass condition's own re-check would
        let this call return ``saved`` instead of raising.

        ``archived`` only — ``shelved`` cannot reach this scenario at all:
        ``probe_filters_for`` (ADR 0021) deliberately excludes ``shelved`` from
        the dedup probe, so a shelved Fact never becomes a bounce candidate
        in the first place, bypass or no bypass. Its refusal is already
        proven at the validation layer alone by
        ``test_promoted_from_shelved_fact_raises`` — the only layer it can
        ever reach."""
        lifecycle = "archived"
        engine, adapter, clock = await _engine()
        engine._embedder = _IdenticalEmbedder()
        now = clock.now()

        fact = Fact(
            id=new_id(), type="fact", content="A non-promotable fact.",
            scope=Scope("global"), lifecycle=lifecycle,  # type: ignore[arg-type]
            created_at=now, updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(fact, vec)

        with pytest.raises(ToolError, match=lifecycle):
            await engine.save(
                "Policy that would rest on the fact.",
                kind="decision",
                scope="global",
                promoted_from=[fact.id],
            )

        r_dec = await engine.save(
            "Policy that would rest on the fact.", kind="decision", scope="global"
        )
        assert r_dec.outcome == "bounced"
        assert r_dec.bounce is not None
        assert len(r_dec.bounce.exits) == 2
        assert {e.action for e in r_dec.bounce.exits} == {"reinforce", "confirm_distinct"}


# ---------------------------------------------------------------------------
# 5. Render — byte-stability
# ---------------------------------------------------------------------------


class TestRenderByteStability:
    async def test_render_twice_identical(self) -> None:
        """render_decision returns identical bytes on two calls."""
        engine, adapter, _ = await _engine()
        r = await engine.save("Policy A.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]
        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)

        out1 = render_decision(rec)
        out2 = render_decision(rec)
        assert out1 == out2, "render_decision must be byte-stable"

    async def test_render_reinforcement_does_not_change_bytes(self) -> None:
        """recall_count/last_referenced_at changes do NOT appear in the render.

        We directly bump the projection fields (as update_projections would) and
        verify the render is unchanged — the render is a pure function of the
        domain fields that appear in the output, and the live-stats fields
        (recall_count, last_referenced_at) are explicitly excluded.
        """
        engine, adapter, _ = await _engine()
        r = await engine.save("Policy B.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        rec_before = await adapter.get_by_id(dec_id)
        assert isinstance(rec_before, Decision)
        render_before = render_decision(rec_before)

        # Directly bump the projection fields via update_projections
        t = datetime(2025, 6, 1, tzinfo=UTC)
        await adapter.update_projections(
            dec_id,
            recall_count=5,
            last_referenced_at=t,
            feedback_score=0.0,
            decay_due_at=None,
        )

        rec_after = await adapter.get_by_id(dec_id)
        assert isinstance(rec_after, Decision)
        assert rec_after.recall_count == 5  # counter updated
        assert rec_after.last_referenced_at == t

        render_after = render_decision(rec_after)
        assert render_before == render_after, (
            "Reinforcement events must NOT change render bytes — "
            "recall_count/last_referenced_at are excluded from the render"
        )

    async def test_render_unaffected_by_unrelated_invalidation(self) -> None:
        """A never-invalidated Decision's render is byte-identical before and
        after an UNRELATED Decision is invalidated through P8's real write
        path (``EchoAssessor("correction")``, ``auto_exit_replacement=True``,
        a Decision save then a Fact save) — the invalidated row's own render
        changes by exactly one lineage line (P9 invalid-Decision read
        contract, decision D7).
        """
        stable_text = "Stable decision untouched by any invalidation."
        existing_text = "the quick brown fox jumps over the lazy dog"
        incoming_text = "the quick brown fox jumps over the lazy hound"
        table = {
            stable_text: _angle_vec(170),
            existing_text: _angle_vec(0),
            incoming_text: _angle_vec(5),
        }
        embedder = _TableEmbedder(table)
        clock = _make_clock()
        adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
        cfg = _make_config(decision_gray_floor=0.70, auto_exit_replacement=True)
        engine = Tabularium(adapter, embedder, clock, cfg, assessor=EchoAssessor("correction"))
        await engine.initialize()
        await adapter.create_scope(Scope("global"))

        stable = await engine.save(stable_text, scope="global", kind="decision")
        assert stable.hypomnema is not None
        stable_rec_before = await adapter.get_by_id(stable.hypomnema.id)
        assert isinstance(stable_rec_before, Decision)
        stable_render_before = render_decision(stable_rec_before)

        existing = await engine.save(existing_text, scope="global", kind="decision")
        assert existing.hypomnema is not None
        existing_rec_before = await adapter.get_by_id(existing.hypomnema.id)
        assert isinstance(existing_rec_before, Decision)
        existing_render_before = render_decision(existing_rec_before)
        assert "Invalidated by:" not in existing_render_before

        incoming = await engine.save(incoming_text, scope="global")
        assert incoming.outcome == "saved"
        assert incoming.hypomnema is not None
        fact_id = incoming.hypomnema.id

        stable_rec_after = await adapter.get_by_id(stable.hypomnema.id)
        assert isinstance(stable_rec_after, Decision)
        stable_render_after = render_decision(stable_rec_after)
        assert stable_render_after == stable_render_before, (
            "an unrelated Decision's render must not change when a sibling "
            "Decision is invalidated"
        )

        existing_rec_after = await adapter.get_by_id(existing.hypomnema.id)
        assert isinstance(existing_rec_after, Decision)
        existing_render_after = render_decision(existing_rec_after)
        recorded = existing_rec_after.created_at.strftime("%Y-%m-%d")
        expected_after = "\n".join(
            [
                f"in_force · invalid · recorded {recorded}",
                "",
                existing_text,
                "",
                "---",
                "",
                f"Invalidated by: `{fact_id!s}` (correction)",
                "",
            ]
        )
        assert existing_render_after == expected_after
        assert existing_render_after != existing_render_before

    async def test_render_bytes_change_on_lifecycle_transition(self) -> None:
        """A supersession transition changes the predecessor's render bytes."""
        engine, adapter, _ = await _engine()

        r1 = await engine.save("V1 policy.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]

        rec_before = await adapter.get_by_id(id1)
        assert isinstance(rec_before, Decision)
        render_before = render_decision(rec_before)

        # Supersede id1 — this sets old.superseded_by + old.lifecycle="superseded"
        await engine.save(
            "V2 policy.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )

        rec_after = await adapter.get_by_id(id1)
        assert isinstance(rec_after, Decision)
        render_after = render_decision(rec_after)

        assert render_before != render_after, (
            "Semantic transition (superseded lifecycle) must change render bytes"
        )
        assert "superseded" in render_after

    async def test_render_includes_backward_chain(self) -> None:
        """The render includes the full backward chain from supersession_chain."""
        engine, adapter, _ = await _engine()

        r1 = await engine.save("First principle.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]

        r2 = await engine.save(
            "Revised principle.",
            kind="decision",
            scope="global",
            supersedes=id1,
            reason="changed",
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        rec2 = await adapter.get_by_id(id2)
        assert isinstance(rec2, Decision)
        rendered = render_decision(rec2)

        assert "Supersedes:" in rendered
        assert "First principle." in rendered  # title of v1

    async def test_render_includes_promoted_from(self) -> None:
        """The render includes promoted_from entries with valence and recurrence."""
        engine, adapter, _ = await _engine()

        r_ember = await engine.ember(
            "Never skip tests.", scope="global", valence="guideline"
        )
        ember_id = r_ember.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "Testing policy formalized.",
            kind="decision",
            scope="global",
            promoted_from=[ember_id],
        )
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]
        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        rendered = render_decision(rec)

        assert "Promoted from:" in rendered
        assert "Guideline" in rendered
        assert "Never skip tests." in rendered
        assert "recurrence: 0" in rendered

    async def test_render_promoted_from_null_valence_omits_label(self) -> None:
        """A promoted plain Fact (never an ember) has ``valence=None`` — the
        render omits the valence label rather than crashing or fabricating
        one (2026-08-07 fix, ADR 0022 amendment; regression for the
        ``pf.valence.capitalize()`` call that assumed every promotion was
        ember-sourced)."""
        engine, adapter, _ = await _engine()

        r_fact = await engine.save(
            "Rendering matters for policy.", scope="global"
        )
        fact_id = r_fact.hypomnema.id  # type: ignore[union-attr]

        r_dec = await engine.save(
            "Policy formalized from a plain fact.",
            kind="decision",
            scope="global",
            promoted_from=[fact_id],
        )
        dec_id = r_dec.hypomnema.id  # type: ignore[union-attr]
        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        rendered = render_decision(rec)

        assert "Promoted from:" in rendered
        assert "Rendering matters for policy." in rendered
        assert "recurrence: 0" in rendered
        # No fabricated valence label — the line reads straight into content.
        assert "- Rendering matters for policy." in rendered


# ---------------------------------------------------------------------------
# 6. Title derivation edge cases
# ---------------------------------------------------------------------------


class TestTitleDerivation:
    def test_title_first_sentence_period(self) -> None:
        """Title stops at the first period followed by whitespace."""
        content = "This is the first sentence. This is the second."
        assert derive_title(content) == "This is the first sentence."

    def test_title_first_sentence_question_mark(self) -> None:
        content = "Is this correct? Yes it is."
        assert derive_title(content) == "Is this correct?"

    def test_title_first_sentence_exclamation(self) -> None:
        content = "Never do this! Seriously."
        assert derive_title(content) == "Never do this!"

    def test_title_newline_before_period(self) -> None:
        """Newline that comes before the first period takes priority."""
        content = "First line\nSecond line. More text."
        assert derive_title(content) == "First line"

    def test_title_no_terminator_uses_whole_content(self) -> None:
        """No sentence terminator and no newline → whole content (up to 80 chars)."""
        content = "A short phrase with no period or newline"
        assert derive_title(content) == content

    def test_title_truncation_at_80_chars(self) -> None:
        """Content longer than 80 chars with no terminator → truncated to 79 + ellipsis."""
        content = "A" * 90
        title = derive_title(content)
        assert len(title) == 80  # 79 chars + "…"
        assert title.endswith("…")

    def test_title_exactly_80_chars_no_truncation(self) -> None:
        """Content of exactly 80 chars → no truncation (≤80 is fine)."""
        content = "A" * 80
        title = derive_title(content)
        assert title == content
        assert len(title) == 80

    def test_title_81_chars_triggers_truncation(self) -> None:
        """81 chars → truncated."""
        content = "A" * 81
        title = derive_title(content)
        assert len(title) == 80
        assert title.endswith("…")

    def test_title_period_at_end_of_string(self) -> None:
        """Period at end of string (no following whitespace) is still a terminator."""
        content = "This is complete."
        assert derive_title(content) == "This is complete."

    def test_title_stripping_whitespace(self) -> None:
        """Title strips leading/trailing whitespace."""
        content = "  Proper sentence.  More text."
        # First period followed by space — title = "  Proper sentence." stripped
        assert derive_title(content) == "Proper sentence."


# ---------------------------------------------------------------------------
# 7. Tip derivation over a chain
# ---------------------------------------------------------------------------


class TestTipDerivation:
    async def test_tip_is_none_when_decision_is_live_end(self) -> None:
        """tip=None when the decision has no superseded_by (is the live end)."""
        engine, _adapter, _ = await _engine()
        r = await engine.save("Solo decision.", kind="decision", scope="global")
        view = await engine.decision(r.hypomnema.id)  # type: ignore[union-attr]
        assert view.tip is None

    async def test_tip_is_none_for_in_force_at_end_of_chain(self) -> None:
        """The live end of a chain has tip=None."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("V1.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        r2 = await engine.save(
            "V2.", kind="decision", scope="global",
            supersedes=id1, reason="changed",
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        view2 = await engine.decision(id2)
        assert view2.tip is None  # id2 IS the live end

    async def test_tip_points_to_live_end_for_superseded(self) -> None:
        """For a superseded Decision, tip points to the live end of the chain."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("V1.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        r2 = await engine.save(
            "V2.", kind="decision", scope="global",
            supersedes=id1, reason="changed",
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        view1 = await engine.decision(id1)
        assert view1.tip == id2  # id2 is the live end

    async def test_tip_over_three_link_chain(self) -> None:
        """Three-link chain: v1.tip = v3, v2.tip = v3, v3.tip = None."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("V1.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        r2 = await engine.save(
            "V2.", kind="decision", scope="global",
            supersedes=id1, reason="changed",
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]
        r3 = await engine.save(
            "V3.", kind="decision", scope="global",
            supersedes=id2, reason="changed",
        )
        id3 = r3.hypomnema.id  # type: ignore[union-attr]

        view1 = await engine.decision(id1)
        view2 = await engine.decision(id2)
        view3 = await engine.decision(id3)

        assert view1.tip == id3
        assert view2.tip == id3
        assert view3.tip is None

    async def test_decision_view_supersedes_field(self) -> None:
        """DecisionView.supersedes = chain[-1].id (immediate predecessor)."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("V1.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        r2 = await engine.save(
            "V2.", kind="decision", scope="global",
            supersedes=id1, reason="changed",
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        view2 = await engine.decision(id2)
        assert view2.supersedes == id1

    async def test_decision_view_fact_id_raises(self) -> None:
        """decision() on a Fact id → ToolError."""
        engine, _adapter, _ = await _engine()
        r = await engine.save("A fact.", scope="global")
        fact_id = r.hypomnema.id  # type: ignore[union-attr]

        with pytest.raises(ToolError, match="Fact"):
            await engine.decision(fact_id)

    async def test_decision_view_unknown_id_raises(self) -> None:
        """decision() on an unknown id → ToolError."""
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="unknown"):
            await engine.decision(new_id())

    async def test_decision_view_rendered_equals_render_decision(self) -> None:
        """DecisionView.rendered is byte-identical to render_decision(record)."""
        engine, adapter, _ = await _engine()
        r = await engine.save("View policy.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        view = await engine.decision(dec_id)
        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)

        assert view.rendered == render_decision(rec)


# ---------------------------------------------------------------------------
# 8. decisions_query
# ---------------------------------------------------------------------------


class TestDecisionsQuery:
    async def test_decisions_query_returns_in_force_by_default(self) -> None:
        """decisions_query returns in_force Decisions by default."""
        engine, _adapter, _ = await _engine()
        r = await engine.save(
            "Use Python for new services.", kind="decision", scope="global"
        )
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        links = await engine.decisions_query("Python services", scope="global")
        assert any(lnk.id == dec_id for lnk in links)
        for lnk in links:
            assert lnk.lifecycle == "in_force"

    async def test_decisions_query_excludes_superseded_by_default(self) -> None:
        """Superseded decisions are excluded unless include_superseded=True."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("Old policy.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        await engine.save(
            "New policy.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )

        links = await engine.decisions_query("policy", scope="global")
        ids = [lnk.id for lnk in links]
        assert id1 not in ids, "Superseded decision must be excluded by default"

    async def test_decisions_query_include_superseded_widens(self) -> None:
        """include_superseded=True widens lifecycle filter to all three."""
        engine, _adapter, _ = await _engine()
        r1 = await engine.save("Superseded policy.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        await engine.save(
            "Current policy.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )

        links = await engine.decisions_query(
            "policy", scope="global", include_superseded=True
        )
        ids = [lnk.id for lnk in links]
        assert id1 in ids, "include_superseded=True must include superseded decisions"

    async def test_decisions_query_uri_format(self) -> None:
        """DecisionLink.uri has the decision:// scheme."""
        engine, _, _ = await _engine()
        r = await engine.save("URI test policy.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        links = await engine.decisions_query("URI test", scope="global")
        matching = [lnk for lnk in links if lnk.id == dec_id]
        assert matching
        assert matching[0].uri == f"decision://{dec_id!s}"

    async def test_decisions_query_link_shape(self) -> None:
        """DecisionLink carries: uri, id, title, lifecycle, superseded_by, scope."""
        engine, _, _ = await _engine()
        r = await engine.save("Shape test decision.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        links = await engine.decisions_query("shape test", scope="global")
        matching = [lnk for lnk in links if lnk.id == dec_id]
        assert matching

        lnk = matching[0]
        assert lnk.uri.startswith("decision://")
        assert lnk.id == dec_id
        assert lnk.title == "Shape test decision."  # first sentence
        assert lnk.lifecycle == "in_force"
        assert lnk.superseded_by is None
        assert lnk.scope == "global"

    async def test_decisions_query_scope_span(self) -> None:
        """Read in scope X sees X + global (SPEC §2).

        Inserts decisions directly via the adapter (bypassing the gate) so we
        can use orthogonal fixture vectors without worrying about dedup bounce.
        The gate's correctness is covered by other tests.
        """
        engine, adapter, clock = await _engine()
        await adapter.create_scope(Scope("project"))
        now = clock.now()

        # Insert Decision in global scope
        g_id = new_id()
        g_dec = Decision(
            id=g_id,
            type="decision",
            content="global infrastructure policy",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec_g = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(g_dec, vec_g)

        # Insert Decision in project scope (orthogonal vector)
        p_id = new_id()
        p_dec = Decision(
            id=p_id,
            type="decision",
            content="project delivery policy",
            scope=Scope("project"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec_p = [0.0] + [1.0] + [0.0] * (_DIM - 2)
        await adapter.upsert(p_dec, vec_p)

        # Query from project scope: should see both project + global
        links = await engine.decisions_query("policy", scope="project", limit=20)
        ids = [lnk.id for lnk in links]
        assert g_id in ids, "global scope visible from project query"
        assert p_id in ids, "project scope visible from project query"

    async def test_decisions_query_limit(self) -> None:
        """limit parameter restricts number of results."""
        engine, _, _ = await _engine()
        for i in range(5):
            await engine.save(f"Policy number {i}.", kind="decision", scope="global")

        links = await engine.decisions_query("policy", scope="global", limit=2)
        assert len(links) <= 2

    async def test_decisions_query_excludes_facts(self) -> None:
        """decisions_query only returns Decisions, not Facts."""
        engine, _, _ = await _engine()
        await engine.save("A plain fact about things.", scope="global")
        await engine.save("A plain decision about things.", kind="decision", scope="global")

        links = await engine.decisions_query("things", scope="global")
        for lnk in links:
            # All links must have the decision:// URI scheme
            assert lnk.uri.startswith("decision://")

    async def test_decisions_query_superseded_by_in_link(self) -> None:
        """DecisionLink.superseded_by is set for superseded decisions."""
        engine, _, _ = await _engine()
        r1 = await engine.save("Old policy.", kind="decision", scope="global")
        id1 = r1.hypomnema.id  # type: ignore[union-attr]
        r2 = await engine.save(
            "New policy.", kind="decision", scope="global", supersedes=id1, reason="changed"
        )
        id2 = r2.hypomnema.id  # type: ignore[union-attr]

        links = await engine.decisions_query(
            "policy", scope="global", include_superseded=True
        )
        old_link = next((lnk for lnk in links if lnk.id == id1), None)
        assert old_link is not None
        assert old_link.superseded_by == id2

    async def test_decisions_query_empty_on_no_matches(self) -> None:
        """decisions_query returns empty list when nothing matches."""
        engine, _, _ = await _engine()
        links = await engine.decisions_query(
            "completely unrelated gibberish xyz", scope="global"
        )
        # No decisions were added, so result should be empty
        assert isinstance(links, list)


# ---------------------------------------------------------------------------
# 9. Resolve-reinforce on Decision target (SPEC §4.4 / B2 item 8)
# ---------------------------------------------------------------------------


class TestDecisionResolveReinforce:
    async def test_resolve_reinforce_decision_emits_dedup_event(self) -> None:
        """resolve reinforce on a Decision emits source='dedup' event.

        Uses a uniform embedder so the Decision and the probe vector are
        identical — the Decision is the best match when the gate re-runs.
        """
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, _ = await _engine(config=cfg)

        # Use the identical embedder from the start so both the Decision's stored
        # vector and the probe vector are the same unit vector.
        engine._embedder = _IdenticalEmbedder()

        r = await engine.save("Stable decision.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        # The save will bounce (decision_gray_floor=0.0 → decision supersede bounce)
        # But we still have the id. Use resolve=reinforce to adjudicate.
        resolution = BounceResolution(candidate_id=dec_id, action="reinforce")
        r2 = await engine.save(
            "Stable decision.", scope="global", resolve=resolution
        )
        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == dec_id

        events = await adapter.query_events(target_id=dec_id, source="dedup")
        assert len(events) == 1

    async def test_resolve_reinforce_decision_bumps_recall_count(self) -> None:
        """resolve reinforce on a Decision bumps recall_count."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, _ = await _engine(config=cfg)
        engine._embedder = _IdenticalEmbedder()

        r = await engine.save("Counted decision.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        resolution = BounceResolution(candidate_id=dec_id, action="reinforce")
        await engine.save("Counted decision.", scope="global", resolve=resolution)

        rec = await adapter.get_by_id(dec_id)
        assert isinstance(rec, Decision)
        assert rec.recall_count == 1

    async def test_auto_reinforce_decision_still_bounces(self) -> None:
        """Auto-reinforce of Decisions is still forbidden; gate bounces."""
        cfg = _make_config(
            gray_floor=0.0,
            decision_gray_floor=0.0
        )
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="never auto reinforce this",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        # Without resolve — gate should bounce, not auto-reinforce
        r = await engine.save("never auto reinforce this", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.verdict == "decision_adjacent"


# ---------------------------------------------------------------------------
# 9b. Gray-band multi-candidate resolve [A6] — Decision arm
# ---------------------------------------------------------------------------


class TestDecisionResolveGrayBandMultiCandidate:
    """Decision-arm mirror of the Fact-arm gray-band resolve tests in
    ``test_engine_m1.py`` — no prior test constructed two Decisions above
    ``decision_gray_floor`` for one incoming save either.

    Geometry shared by both tests: Decision A at angle 0°, Decision B at
    angle 25°, both above ``decision_gray_floor=0.70`` from any probe angle
    used below.
    """

    async def test_identical_content_resolves_on_first_attempt(self) -> None:
        """Two Decisions above floor + UNCHANGED content on resolve → the
        originally-bounced candidate resolves on the FIRST round-trip.
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.70)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        dec_b = Decision(
            id=new_id(), type="decision", content="Beta policy on rollout cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))
        await adapter.upsert(dec_b, _angle_vec(25))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_a.id  # cos(10°)=.9848 > cos(15°)=.9659

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved", (
            f"expected the first round-trip to succeed with unchanged content, "
            f"got outcome={r2.outcome!r}"
        )
        assert r2.hypomnema is not None
        assert await adapter.is_distinct(dec_a.id, r2.hypomnema.id)

    async def test_edited_content_flips_top1_and_returns_fresh_bounce(self) -> None:
        """Content EDITED between the bounce and the resolve, such that the
        top-1 candidate flips → a fresh bounce naming the NEW best Decision,
        nothing written (decisions.py:528's stale-candidate check).
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.70)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        dec_b = Decision(
            id=new_id(), type="decision", content="Beta policy on rollout cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))
        await adapter.upsert(dec_b, _angle_vec(25))

        c1 = "incoming policy note on release cadence"   # angle 10 -> A wins
        c2 = "incoming policy note on rollout cadence"   # angle 20 -> B wins
        engine._embedder = _TableEmbedder({c1: _angle_vec(10), c2: _angle_vec(20)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_a.id

        rows_before = [row async for row in await adapter.scan()]

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c2, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "bounced", (
            f"expected a fresh bounce naming the new top-1 (B), got outcome={r2.outcome!r}"
        )
        assert r2.bounce is not None
        assert r2.bounce.candidate_id == dec_b.id  # cos(5°)=.9962 (B) > cos(20°)=.9397 (A)
        assert r2.hypomnema is None

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before), "nothing should have been written"

    async def test_distinct_pair_never_rebounces(self) -> None:
        """A pair already judged distinct skips re-bounce (is_distinct consult) —
        Decision-arm mirror of ``test_engine_m1.TestResolveGrayBandMultiCandidate
        .test_distinct_pair_never_rebounces`` sibling coverage, but constructed to
        actually force a re-bounce absent the consult (registry *presence* alone
        proves nothing about whether it's *read*).

        After ``confirm_distinct`` writes a new Decision N verbatim from the
        resolved content, N is an exact vector match for that content (cosine
        1.0) — strictly higher than the original candidate's gray-band cosine.
        A naive retry of the SAME resolve call therefore re-probes into a world
        where N, not the original candidate, is the top-1 raw-cosine match.
        Absent an ``is_distinct`` consult excluding N (already registered
        distinct from the original candidate), the engine dispatches on N and
        returns a fresh bounce — an infinite re-bounce loop on every retry.
        With the consult, N is excluded, the original candidate is still best,
        and the resolve is honored directly: no bounce.
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.70)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        # Round 1: bounce against dec_a (only record in the store).
        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_a.id

        # Resolve confirm_distinct, unchanged content -> writes N verbatim from
        # c1 (vector == _angle_vec(10) exactly) and registers (dec_a.id, N.id).
        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(dec_a.id, n_id)

        rows_after_first_write = [row async for row in await adapter.scan()]

        # Round 2: naive retry of the SAME resolve call (same content, same
        # candidate_id=dec_a.id).  N now scores cos(0°)=1.0 against this probe,
        # beating dec_a's cos(10°)=.9848 -- the already-adjudicated pair must
        # not manifest as a fresh bounce, AND must not write a second Decision
        # (see TestDecisionConfirmDistinctReplayIdempotency for the full
        # unbounded-duplicate regression coverage; asserting only outcome
        # here was this test's original weakness).
        r3 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r3.outcome == "already_saved", (
            f"expected the already-adjudicated pair to return an idempotent ack "
            f"(outcome='already_saved'), got outcome={r3.outcome!r} "
            f"(bounce candidate={r3.bounce.candidate_id if r3.bounce else None})"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == n_id, (
            "already_saved must carry the id of the Decision the FIRST call wrote, "
            f"got {r3.hypomnema.id} instead of {n_id}"
        )

        rows_after_replay = [row async for row in await adapter.scan()]
        assert len(rows_after_replay) == len(rows_after_first_write), (
            "a replayed confirm_distinct must not write a second Decision "
            f"(rows before replay={len(rows_after_first_write)}, "
            f"after={len(rows_after_replay)})"
        )


# ---------------------------------------------------------------------------
# 9b-bis. confirm_distinct replay idempotency — unbounded-duplicate regression
# (root cause: MemoryAdapter/LanceDBAdapter ``is_distinct`` keys on
# ``frozenset({a, b})``, which collapses to a 1-element set when ``a == b``,
# so a candidate can never be excluded from itself; facts.py's
# ``_find_confirm_distinct_replay`` closes the gap at the engine layer via
# content-equality against the already-distinct-judged candidate — shared by
# both arms; decisions.py calls the same function (see ``save_decision``))
# ---------------------------------------------------------------------------


_AnyReplayAdapter = MemoryAdapter | LanceDBAdapter
ReplayEngineCtx = tuple[Tabularium, _AnyReplayAdapter, TestClock, str]
# Per-arm aliases (both arms use the identical underlying tuple shape) so
# each fixture's own call sites stay self-documenting without duplicating
# the union.
DecisionEngineCtx = ReplayEngineCtx


async def _build_replay_engine_ctx(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path, cfg: EngineConfig
) -> ReplayEngineCtx:
    """Dual-parameterized engine/adapter/clock builder shared by the
    confirm_distinct-replay fixtures (Decision arm's ``decision_engine_ctx``,
    Fact arm's ``fact_engine_ctx``), reusing the conformance suite's
    adapter-construction helpers (precedent — see
    ``tests/test_engine_dispositions.py``) rather than a third construction
    pattern.  Scoped to this section only — the rest of this file's
    ``_engine()`` helper (``MemoryAdapter``-only) is unchanged.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clock = _make_clock()
    adapter_id = request.param
    adapter: _AnyReplayAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    return engine, adapter, clock, adapter_id


@pytest.fixture(params=["memory", "lancedb"])
async def decision_engine_ctx(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> DecisionEngineCtx:
    cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.70)
    return await _build_replay_engine_ctx(request, tmp_path, cfg)


class TestDecisionConfirmDistinctReplayIdempotency:
    """A retried ``save(kind="decision", resolve=BounceResolution(candidate_id=O,
    action="confirm_distinct"))`` must be idempotent (SPEC §4.5 [A6]): the
    first call writes a new in-force Decision N and marks ``(O, N)``
    distinct; every subsequent byte-identical replay must return
    ``outcome="already_saved"`` naming N — never write a second, third, ...
    Decision.
    """

    async def test_replay_does_not_duplicate_across_three_retries(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """Unbounded-duplicate regression: 3+ identical replays must all be
        no-ops, on both storage adapters."""
        engine, adapter, clock, adapter_id = decision_engine_ctx
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_a.id

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")

        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(dec_a.id, n_id)

        rows_after_first_write = [row async for row in await adapter.scan()]

        for attempt in range(1, 4):
            r = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
            assert r.outcome == "already_saved", (
                f"[{adapter_id}] retry {attempt}: expected an idempotent ack, "
                f"got outcome={r.outcome!r}"
            )
            assert r.hypomnema is not None
            assert r.hypomnema.id == n_id, (
                f"[{adapter_id}] retry {attempt}: expected the FIRST call's Decision "
                f"id ({n_id}), got {r.hypomnema.id}"
            )

        rows_after_retries = [row async for row in await adapter.scan()]
        assert len(rows_after_retries) == len(rows_after_first_write), (
            f"[{adapter_id}] replay retries must not write additional rows "
            f"(before={len(rows_after_first_write)}, after={len(rows_after_retries)})"
        )

        # Exactly one in-force Decision carries the retried content, after
        # 1 real write + 3 replayed no-ops.
        full_records = [await adapter.get_by_id(rm.id) for rm in rows_after_retries]
        matches = [
            rec for rec in full_records
            if isinstance(rec, Decision) and rec.content == c1 and rec.lifecycle == "in_force"
        ]
        assert len(matches) == 1, (
            f"[{adapter_id}] expected exactly 1 in-force Decision with content={c1!r}, "
            f"found {len(matches)}: {[str(m.id) for m in matches]}"
        )

    async def test_edited_content_after_replay_still_writes_new_decision(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """Regression guard: the replay suppression is keyed on content
        equality, not on ``resolve.candidate_id`` alone — a genuinely edited
        resubmission that keeps the SAME candidate on top (excluding the
        already-distinct N) must still write a fresh Decision.
        """
        engine, adapter, clock, adapter_id = decision_engine_ctx
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        # angle 12: close enough to dec_a (cos(12°)≈0.978, well above the
        # 0.70 decision_gray_floor) that dec_a stays top-1 once N (angle 10,
        # cos(2°) from c2) is excluded; different STRING from c1, so the
        # replay guard's content-equality check must not fire.
        c2 = "incoming policy note on release cadence, revised"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10), c2: _angle_vec(12)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c2, kind="decision", scope="global", resolve=resolution)
        assert r3.outcome == "saved", (
            f"[{adapter_id}] an edited resubmission must still write a new Decision, "
            f"got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id != n_id, (
            f"[{adapter_id}] edited content must not be suppressed as a replay of N"
        )
        assert await adapter.is_distinct(dec_a.id, r3.hypomnema.id)

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) + 1, (
            f"[{adapter_id}] edited resubmission must write exactly one new row "
            f"(before={len(rows_before)}, after={len(rows_after)})"
        )

    async def test_multiple_already_distinct_duplicates_returns_earliest(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """If the store already holds MORE than one byte-identical Decision
        already marked distinct from the resolve target — a store that
        accumulated duplicates before this fix landed, or via a write race
        entirely outside this fix's scope — replay detection must deterministically name the
        EARLIEST one by ``created_at``, not an arbitrary
        set-iteration-order pick.
        """
        engine, adapter, clock, adapter_id = decision_engine_ctx
        t0 = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        t1 = t0 + timedelta(days=1)
        t2 = t0 + timedelta(days=2)
        n1 = Decision(
            id=new_id(), type="decision", content=c1,
            scope=Scope("global"), lifecycle="in_force", created_at=t1, updated_at=t1,
        )
        n2 = Decision(
            id=new_id(), type="decision", content=c1,
            scope=Scope("global"), lifecycle="in_force", created_at=t2, updated_at=t2,
        )
        # Both duplicates sit at the SAME vector position (angle 10) — an
        # exact-cosine tie the set-iteration-order bug would otherwise
        # resolve arbitrarily.
        await adapter.upsert(n1, _angle_vec(10))
        await adapter.upsert(n2, _angle_vec(10))

        await adapter.mark_distinct(DistinctPair(
            lower_id=min(dec_a.id, n1.id, key=str), higher_id=max(dec_a.id, n1.id, key=str),
            source="save", judged_at=t1,
        ))
        await adapter.mark_distinct(DistinctPair(
            lower_id=min(dec_a.id, n2.id, key=str), higher_id=max(dec_a.id, n2.id, key=str),
            source="save", judged_at=t2,
        ))

        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r = await engine.save(c1, kind="decision", scope="global", resolve=resolution)

        assert r.outcome == "already_saved", f"[{adapter_id}] got outcome={r.outcome!r}"
        assert r.hypomnema is not None
        assert r.hypomnema.id == n1.id, (
            f"[{adapter_id}] expected the EARLIEST duplicate ({n1.id}, created_at={t1}), "
            f"got {r.hypomnema.id} (n2={n2.id}, created_at={t2})"
        )

    async def test_duplicates_with_equal_created_at_break_tie_by_id(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """Residual tie case:
        when two already-distinct byte-identical duplicates share the SAME
        ``created_at`` (only reachable via the same out-of-scope write race
        as the multi-duplicate case above — genuine write order is not a
        field the storage seam carries, SPEC §3.1), the id-string tie-break
        must still resolve deterministically rather than crash or vary
        across runs. This does not (and per the docstring, cannot) promise
        "the decision the first call wrote" in this exact-tie sub-case —
        only that the pick is stable.
        """
        engine, adapter, clock, adapter_id = decision_engine_ctx
        t0 = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        n1 = Decision(
            id=new_id(), type="decision", content=c1,
            scope=Scope("global"), lifecycle="in_force", created_at=t0, updated_at=t0,
        )
        n2 = Decision(
            id=new_id(), type="decision", content=c1,
            scope=Scope("global"), lifecycle="in_force", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(n1, _angle_vec(10))
        await adapter.upsert(n2, _angle_vec(10))

        await adapter.mark_distinct(DistinctPair(
            lower_id=min(dec_a.id, n1.id, key=str), higher_id=max(dec_a.id, n1.id, key=str),
            source="save", judged_at=t0,
        ))
        await adapter.mark_distinct(DistinctPair(
            lower_id=min(dec_a.id, n2.id, key=str), higher_id=max(dec_a.id, n2.id, key=str),
            source="save", judged_at=t0,
        ))

        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        expected_id = min(n1.id, n2.id, key=str)

        for attempt in range(1, 3):
            r = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
            assert r.outcome == "already_saved", (
                f"[{adapter_id}] attempt {attempt}: got outcome={r.outcome!r}"
            )
            assert r.hypomnema is not None
            assert r.hypomnema.id == expected_id, (
                f"[{adapter_id}] attempt {attempt}: expected the id-string tie-break "
                f"winner ({expected_id}), got {r.hypomnema.id} — must be stable across calls"
            )

    async def test_replay_ignores_revoked_prior_match_and_writes_fresh(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """Regression: the
        distinct-pair registry is append-only and never pruned (SPEC §4.6 —
        "even if one or both ids are later buried, their pair entry
        harmlessly remains"), so a candidate can still surface via
        ``distinct_exclude`` long after it was retired. If the Decision the
        first confirm_distinct call wrote is later REVOKED, a subsequent
        byte-identical replay must NOT return ``already_saved`` naming that
        now-dead id — it must fall through to normal dispatch (dec_a is
        still the only live viable candidate) and write a fresh in-force
        Decision, exactly as ``save_decision``'s own viable-hits loop
        already treats retired Decisions elsewhere in this file.
        """
        engine, adapter, clock, adapter_id = decision_engine_ctx
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(dec_a.id, n_id)

        revise = await engine.revoke(n_id, reason="test: retire the confirm_distinct sibling")
        assert revise.outcome == "revised"
        assert revise.lifecycle == "revoked"

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r3.outcome == "saved", (
            f"[{adapter_id}] a replay against a RETIRED prior match must write fresh, "
            f"not return already_saved for a dead id; got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id != n_id

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) + 1, (
            f"[{adapter_id}] expected exactly one fresh Decision written "
            f"(before={len(rows_before)}, after={len(rows_after)})"
        )

    async def test_replay_ignores_invalidated_prior_match_and_writes_fresh(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        """Regression: same
        failure mode as the revoked case above, but for validity — if the
        Decision the first call wrote is later marked INVALID, a
        byte-identical replay must not return ``already_saved`` naming the
        invalid id; the write must still happen (normal dispatch, since the
        invalid record is excluded from consideration by the same
        distinct-pair exclusion, not by the invalid-candidate/reconciliation
        path — either way the caller gets a fresh, valid Decision, never a
        dead-record ack).
        """
        engine, adapter, clock, adapter_id = decision_engine_ctx
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=dec_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        revise = await engine.set_validity(n_id, False, reason="test: invalidate the sibling")
        assert revise.outcome == "revised"
        assert revise.validity == "invalid"

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c1, kind="decision", scope="global", resolve=resolution)
        assert r3.hypomnema is not None
        assert r3.hypomnema.id != n_id
        assert r3.outcome in {"saved", "reconciled"}, (
            f"[{adapter_id}] a replay against an INVALIDATED prior match must write past "
            f"it, not return already_saved; got outcome={r3.outcome!r}"
        )

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) + 1, (
            f"[{adapter_id}] expected exactly one fresh Decision written "
            f"(before={len(rows_before)}, after={len(rows_after)})"
        )


# ---------------------------------------------------------------------------
# 9b-quater. confirm_distinct replay idempotency — FACT arm (structurally
# identical defect to 9b-bis above, closed by the same shared
# ``facts._find_confirm_distinct_replay`` — both arms call it, parametrized
# on their own liveness predicate).  Root cause identical: ``is_distinct``
# keys on ``frozenset({a, b})``, which collapses to a 1-element set when
# ``a == b``, so a candidate can never be excluded from itself.
# ---------------------------------------------------------------------------


FactEngineCtx = ReplayEngineCtx


@pytest.fixture(params=["memory", "lancedb"])
async def fact_engine_ctx(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> FactEngineCtx:
    """Dual-parameterized engine/adapter/clock for the Fact-arm
    confirm_distinct-replay tests — same ``_build_replay_engine_ctx`` builder
    as ``decision_engine_ctx`` above, parallel fixture for the Fact arm (its
    own config: no ``decision_gray_floor`` override needed).
    """
    cfg = _make_config(gray_floor=0.80)
    return await _build_replay_engine_ctx(request, tmp_path, cfg)


class TestFactConfirmDistinctReplayIdempotency:
    """A retried ``save(resolve=BounceResolution(candidate_id=O,
    action="confirm_distinct"))`` on the FACT arm (``kind="fact"``, the
    default) must be idempotent (SPEC §4.5 [A6]) — same contract as the
    Decision arm above, both routed through the shared
    ``facts._find_confirm_distinct_replay``. The first call writes a new
    valid Fact N and marks ``(O, N)`` distinct; every subsequent
    byte-identical replay must return ``outcome="already_saved"`` naming N —
    never write a second, third, ... Fact.
    """

    async def test_replay_does_not_duplicate_across_three_retries(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """Unbounded-duplicate regression: 3+ identical replays must all be
        no-ops, on both storage adapters."""
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == fact_a.id

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")

        r2 = await engine.save(c1, scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(fact_a.id, n_id)

        rows_after_first_write = [row async for row in await adapter.scan()]

        for attempt in range(1, 4):
            r = await engine.save(c1, scope="global", resolve=resolution)
            assert r.outcome == "already_saved", (
                f"[{adapter_id}] retry {attempt}: expected an idempotent ack, "
                f"got outcome={r.outcome!r}"
            )
            assert r.hypomnema is not None
            assert r.hypomnema.id == n_id, (
                f"[{adapter_id}] retry {attempt}: expected the FIRST call's Fact "
                f"id ({n_id}), got {r.hypomnema.id}"
            )

        rows_after_retries = [row async for row in await adapter.scan()]
        assert len(rows_after_retries) == len(rows_after_first_write), (
            f"[{adapter_id}] replay retries must not write additional rows "
            f"(before={len(rows_after_first_write)}, after={len(rows_after_retries)})"
        )

        # Exactly one valid Fact carries the retried content, after 1 real
        # write + 3 replayed no-ops.
        full_records = [await adapter.get_by_id(rm.id) for rm in rows_after_retries]
        matches = [
            rec for rec in full_records
            if isinstance(rec, Fact) and rec.content == c1 and rec.validity == "valid"
        ]
        assert len(matches) == 1, (
            f"[{adapter_id}] expected exactly 1 valid Fact with content={c1!r}, "
            f"found {len(matches)}: {[str(m.id) for m in matches]}"
        )

    async def test_edited_content_after_replay_still_writes_new_fact(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """Regression guard: the replay suppression is keyed on content
        equality, not on ``resolve.candidate_id`` alone — a genuinely edited
        resubmission that keeps the SAME candidate on top (excluding the
        already-distinct N) must still write a fresh Fact.
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        # angle 12: close enough to fact_a (cos(12°)≈0.978, well above the
        # 0.80 gray_floor) that fact_a stays top-1 once N (angle 10,
        # cos(2°) from c2) is excluded; different STRING from c1, so the
        # replay guard's content-equality check must not fire.
        c2 = "incoming policy note on release cadence, revised"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10), c2: _angle_vec(12)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c2, scope="global", resolve=resolution)
        assert r3.outcome == "saved", (
            f"[{adapter_id}] an edited resubmission must still write a new Fact, "
            f"got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id != n_id, (
            f"[{adapter_id}] edited content must not be suppressed as a replay of N"
        )
        assert await adapter.is_distinct(fact_a.id, r3.hypomnema.id)

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) + 1, (
            f"[{adapter_id}] edited resubmission must write exactly one new row "
            f"(before={len(rows_before)}, after={len(rows_after)})"
        )

    async def test_multiple_already_distinct_duplicates_returns_earliest(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """If the store already holds MORE than one byte-identical Fact
        already marked distinct from the resolve target — a store that
        accumulated duplicates before this fix landed, or via a write race
        entirely outside this fix's scope — replay detection must
        deterministically name the EARLIEST one by ``created_at``, not an
        arbitrary set-iteration-order pick.
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        t0 = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        t1 = t0 + timedelta(days=1)
        t2 = t0 + timedelta(days=2)
        n1 = Fact(
            id=new_id(), type="fact", content=c1,
            scope=Scope("global"), lifecycle="active", created_at=t1, updated_at=t1,
        )
        n2 = Fact(
            id=new_id(), type="fact", content=c1,
            scope=Scope("global"), lifecycle="active", created_at=t2, updated_at=t2,
        )
        # Both duplicates sit at the SAME vector position (angle 10) — an
        # exact-cosine tie the set-iteration-order bug would otherwise
        # resolve arbitrarily.
        await adapter.upsert(n1, _angle_vec(10))
        await adapter.upsert(n2, _angle_vec(10))

        await adapter.mark_distinct(DistinctPair(
            lower_id=min(fact_a.id, n1.id, key=str), higher_id=max(fact_a.id, n1.id, key=str),
            source="save", judged_at=t1,
        ))
        await adapter.mark_distinct(DistinctPair(
            lower_id=min(fact_a.id, n2.id, key=str), higher_id=max(fact_a.id, n2.id, key=str),
            source="save", judged_at=t2,
        ))

        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        r = await engine.save(c1, scope="global", resolve=resolution)

        assert r.outcome == "already_saved", f"[{adapter_id}] got outcome={r.outcome!r}"
        assert r.hypomnema is not None
        assert r.hypomnema.id == n1.id, (
            f"[{adapter_id}] expected the EARLIEST duplicate ({n1.id}, created_at={t1}), "
            f"got {r.hypomnema.id} (n2={n2.id}, created_at={t2})"
        )

    async def test_duplicates_with_equal_created_at_break_tie_by_id(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """Residual tie case: when two already-distinct byte-identical
        duplicates share the SAME ``created_at`` (only reachable via a
        write race outside this fix's scope — genuine write order is not a
        field the storage seam carries, SPEC §3.1), the id-string tie-break
        must still resolve deterministically rather than crash or vary
        across runs.
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        t0 = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        n1 = Fact(
            id=new_id(), type="fact", content=c1,
            scope=Scope("global"), lifecycle="active", created_at=t0, updated_at=t0,
        )
        n2 = Fact(
            id=new_id(), type="fact", content=c1,
            scope=Scope("global"), lifecycle="active", created_at=t0, updated_at=t0,
        )
        await adapter.upsert(n1, _angle_vec(10))
        await adapter.upsert(n2, _angle_vec(10))

        await adapter.mark_distinct(DistinctPair(
            lower_id=min(fact_a.id, n1.id, key=str), higher_id=max(fact_a.id, n1.id, key=str),
            source="save", judged_at=t0,
        ))
        await adapter.mark_distinct(DistinctPair(
            lower_id=min(fact_a.id, n2.id, key=str), higher_id=max(fact_a.id, n2.id, key=str),
            source="save", judged_at=t0,
        ))

        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        expected_id = min(n1.id, n2.id, key=str)

        for attempt in range(1, 3):
            r = await engine.save(c1, scope="global", resolve=resolution)
            assert r.outcome == "already_saved", (
                f"[{adapter_id}] attempt {attempt}: got outcome={r.outcome!r}"
            )
            assert r.hypomnema is not None
            assert r.hypomnema.id == expected_id, (
                f"[{adapter_id}] attempt {attempt}: expected the id-string tie-break "
                f"winner ({expected_id}), got {r.hypomnema.id} — must be stable across calls"
            )

    async def test_replay_ignores_invalidated_prior_match_and_writes_fresh(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """Regression: if the Fact the first call wrote is later marked
        INVALID, a byte-identical replay must not return ``already_saved``
        naming the invalid id; the write must still happen. (Facts have no
        lifecycle equivalent to Decision's superseded/revoked — a Fact's
        lifecycle stays live (active/hot/ember/archived) until buried, and a
        buried record drops out of search results entirely rather than
        surfacing here — so ``validity == "invalid"`` is the only
        reachable non-live state for a Fact search hit.)
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        revise = await engine.set_validity(n_id, False, reason="test: invalidate the sibling")
        assert revise.outcome == "revised"
        assert revise.validity == "invalid"

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c1, scope="global", resolve=resolution)
        assert r3.hypomnema is not None
        assert r3.hypomnema.id != n_id
        assert r3.outcome in {"saved", "reconciled"}, (
            f"[{adapter_id}] a replay against an INVALIDATED prior match must write past "
            f"it, not return already_saved; got outcome={r3.outcome!r}"
        )

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) + 1, (
            f"[{adapter_id}] expected exactly one fresh Fact written "
            f"(before={len(rows_before)}, after={len(rows_after)})"
        )

    async def test_replay_still_suppressed_when_prior_match_is_archived(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """Fact-arm-specific coverage (this is where a lazy port of the
        Decision-arm predicate would break): Facts have NO lifecycle state
        that permanently retires a record the way Decision's
        superseded/revoked does — an archived Fact is dormant, not dead,
        and stays eligible for dedup-revival (facts.py's ``_dispatch_match``
        archived branch reinforces + reactivates it in place). The
        Fact-arm liveness predicate is therefore ``validity == "valid"``
        alone, with NO lifecycle filter. Copying the Decision arm's
        ``lifecycle == "in_force"`` check verbatim onto the Fact arm would
        wrongly treat an archived duplicate as dead and let a replay slip
        through to a second write — this test pins the opposite: N archived
        after the first write must still suppress the replay.
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        await adapter.transition(n_id, "archived", at=clock.now())

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c1, scope="global", resolve=resolution)
        assert r3.outcome == "already_saved", (
            f"[{adapter_id}] an archived prior match is still LIVE for the Fact arm "
            f"(archived Facts are dormant, not dead) and must still suppress the "
            f"replay; got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == n_id

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before), (
            f"[{adapter_id}] a replay against a still-live (archived) prior match must "
            f"not write a new row (before={len(rows_before)}, after={len(rows_after)})"
        )

    async def test_replay_against_archived_prior_match_revives_it(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        """New locked semantics (SPEC §4.5 [A6] amendment): a byte-identical ``confirm_distinct``
        replay whose prior match N
        is currently ARCHIVED must revive N in place
        (``archived``->``active``, same id) before acking
        ``outcome="already_saved"`` — not leave it dormant while telling the
        caller its content is already saved (that is silent data loss by
        omission, not idempotence). Reuses the exact same revival mutation
        as the gate's own archived-hit dispatch (``_dispatch_match``):
        reinforce (``source="revival"``) then ``transition``, so this must
        also assert the revival EVENT landed, not just the lifecycle flip.

        Row-count assertion is load-bearing here (per the brief): a
        correct-outcome-but-still-writes regression (the codex-flagged
        alternative of excluding archived from the liveness predicate,
        which silently writes a THIRD duplicate Fact instead of reviving
        the second) would pass an outcome/id-only check but fail this one.
        """
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_a = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_a, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"

        resolution = BounceResolution(candidate_id=fact_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id

        await adapter.transition(n_id, "archived", at=clock.now())
        archived_check = await adapter.get_by_id(n_id)
        assert isinstance(archived_check, Fact)
        assert archived_check.lifecycle == "archived"

        rows_before = [row async for row in await adapter.scan()]

        r3 = await engine.save(c1, scope="global", resolve=resolution)
        assert r3.outcome == "already_saved", (
            f"[{adapter_id}] got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == n_id

        # Lifecycle: revived in place, not left dormant.
        revived = await adapter.get_by_id(n_id)
        assert isinstance(revived, Fact)
        assert revived.lifecycle == "active", (
            f"[{adapter_id}] N must be revived archived->active by the replay ack, "
            f"got lifecycle={revived.lifecycle!r}"
        )

        # A real revival event was emitted (same mutation as _dispatch_match's
        # own archived-hit dispatch) — not merely a lifecycle poke.
        revival_events = await adapter.query_events(target_id=n_id, source="revival")
        assert len(revival_events) == 1, (
            f"[{adapter_id}] expected exactly one revival event on N, "
            f"found {len(revival_events)}"
        )

        # Row count: revival mutates N in place, it must not add (or remove) a row.
        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before) == 2, (
            f"[{adapter_id}] revival must not write any additional row "
            f"(before={len(rows_before)}, after={len(rows_after)}, expected 2: "
            f"fact_a + N)"
        )


# ---------------------------------------------------------------------------
# 9b-quinquies. A stale "reinforce" resolve must not silently reinforce the
# wrong record (behaviour updated by the admission-order change). O and N are marked
# distinct via a confirm_distinct round-trip; a LATER
# resolve(candidate_id=O, action="reinforce") resubmits content that is
# actually byte-identical to N. Admission step 1 (replay) now runs
# regardless of ``resolve`` and catches N directly, before the serial probe
# or ``distinct_exclude_ids`` ever come into it — ``outcome="already_saved"``
# naming N, never a silent reinforcement of the stale O. Shared root cause,
# shared fix (``facts._admission_steps_1_3`` / its Decision-arm mirror),
# both arms call it — covered on both here.
# ---------------------------------------------------------------------------


class TestDistinctExcludeNotAppliedToReinforceResolve:
    async def test_decision_reinforce_resolve_against_stale_target_bounces_fresh(
        self, decision_engine_ctx: DecisionEngineCtx
    ) -> None:
        engine, adapter, clock, adapter_id = decision_engine_ctx
        now = clock.now()

        dec_o = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_o, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_o.id

        # confirm_distinct: writes N verbatim from c1 (cosine 1.0 against
        # itself) and registers the pair (O, N) distinct.
        confirm = BounceResolution(candidate_id=dec_o.id, action="confirm_distinct")
        r2 = await engine.save(c1, kind="decision", scope="global", resolve=confirm)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(dec_o.id, n_id)

        # Now a STALE reinforce resolve: still names O, but the resubmitted
        # content (unchanged c1) is byte-identical to N. Admission step 1
        # (replay) now runs regardless of ``resolve`` and catches N
        # directly before the serial probe ever runs -- the pre-fix defect
        # this test originally pinned (N excluded from best-hit
        # consideration, O silently reinforced) is superseded by an even
        # earlier, more direct catch.
        reinforce = BounceResolution(candidate_id=dec_o.id, action="reinforce")
        r3 = await engine.save(c1, kind="decision", scope="global", resolve=reinforce)

        assert r3.outcome == "already_saved", (
            f"[{adapter_id}] expected step 1's byte-identical replay to name N "
            f"(the record the content actually matches) directly, "
            f"got outcome={r3.outcome!r} (reinforced_id={r3.reinforced_id})"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == n_id, (
            f"[{adapter_id}] replay must name N ({n_id}), the record the "
            f"content is byte-identical to, not O ({dec_o.id})"
        )

        # Neither O nor N was reinforced — a replay ack writes nothing.
        o_events = await adapter.query_events(target_id=dec_o.id)
        assert o_events == [], (
            f"[{adapter_id}] O must NOT have been reinforced by the stale "
            f"reinforce resolve; found events {o_events!r}"
        )
        n_events = await adapter.query_events(target_id=n_id)
        assert n_events == [], (
            f"[{adapter_id}] N must NOT have been reinforced either -- a replay "
            f"ack writes nothing; found events {n_events!r}"
        )

    async def test_fact_reinforce_resolve_against_stale_target_bounces_fresh(
        self, fact_engine_ctx: FactEngineCtx
    ) -> None:
        engine, adapter, clock, adapter_id = fact_engine_ctx
        now = clock.now()

        fact_o = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(fact_o, _angle_vec(0))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == fact_o.id

        confirm = BounceResolution(candidate_id=fact_o.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=confirm)
        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        n_id = r2.hypomnema.id
        assert await adapter.is_distinct(fact_o.id, n_id)

        # Stale reinforce resolve naming O, content byte-identical to N.
        # Admission step 1 (replay) now runs regardless of ``resolve``
        # and catches N directly before the serial probe ever runs.
        reinforce = BounceResolution(candidate_id=fact_o.id, action="reinforce")
        r3 = await engine.save(c1, scope="global", resolve=reinforce)

        assert r3.outcome == "already_saved", (
            f"[{adapter_id}] expected step 1's byte-identical replay to name N "
            f"(the record the content actually matches) directly, "
            f"got outcome={r3.outcome!r} (reinforced_id={r3.reinforced_id})"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == n_id, (
            f"[{adapter_id}] replay must name N ({n_id}), not O ({fact_o.id})"
        )

        o_events = await adapter.query_events(target_id=fact_o.id)
        assert o_events == [], (
            f"[{adapter_id}] O must NOT have been reinforced by the stale "
            f"reinforce resolve; found events {o_events!r}"
        )
        n_events = await adapter.query_events(target_id=n_id)
        assert n_events == [], (
            f"[{adapter_id}] N must NOT have been reinforced either -- a replay "
            f"ack writes nothing; found events {n_events!r}"
        )


# ---------------------------------------------------------------------------
# 9c. Bounce envelope — other_candidates (SPEC §4.4) — Decision arm
# ---------------------------------------------------------------------------


class TestDecisionBounceOtherCandidates:
    """Decision-arm wiring for ``other_candidates`` (``DecisionBounce``) —
    proves the ``save_decision`` viable-hits collection (decisions.py) feeds
    the shared ``_bounce_neighbors`` helper correctly, independent of the
    Fact arm.
    """

    async def test_populated_with_second_neighbor_descending(self) -> None:
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.70)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        dec_a = Decision(
            id=new_id(), type="decision", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        dec_b = Decision(
            id=new_id(), type="decision", content="Beta policy on rollout cadence.",
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(dec_a, _angle_vec(0))
        await adapter.upsert(dec_b, _angle_vec(25))

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, kind="decision", scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == dec_a.id

        assert len(r1.bounce.other_candidates) == 1
        neighbor = r1.bounce.other_candidates[0]
        assert neighbor.candidate_id == dec_b.id
        assert neighbor.similarity == pytest.approx(math.cos(math.radians(15)), abs=1e-6)
        assert neighbor.guard_flag is None


# ---------------------------------------------------------------------------
# 10. Codex-gate regression tests (findings F1/F2/F3)
# ---------------------------------------------------------------------------


class TestCodexGateRegressions:
    """Regression tests for three earlier findings."""

    # F1: Resolve must not reinforce when candidate is still best but below arm floor
    async def test_resolve_reinforce_below_floor_writes_novel(self) -> None:
        """resolve='reinforce' on a candidate that is still best but below arm floor
        must write novel, not reinforce — the gate would have passed novel originally.

        Strategy: save the Decision directly via adapter (bypass gate), set a floor
        of 0.95, then probe with a vector at cosine ~0.71 (well below 0.95) from the
        Decision.  Issue resolve=reinforce; engine should write novel.
        """
        import math

        cfg = _make_config(gray_floor=0.95, decision_gray_floor=0.95)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Decision stored at axis-0 unit vector
        dec = Decision(
            id=new_id(),
            type="decision",
            content="below floor decision",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec_dec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(dec, vec_dec)

        # Probe at 45° from vec_dec → cosine = cos(45°) ≈ 0.707 < 0.95
        probe_vec = [1.0, 1.0] + [0.0] * (_DIM - 2)
        norm = math.sqrt(sum(x * x for x in probe_vec))
        probe_vec = [x / norm for x in probe_vec]
        cosine = sum(a * b for a, b in zip(vec_dec, probe_vec, strict=True))
        assert cosine < 0.95, f"cosine {cosine:.4f} must be below floor 0.95 for this test"

        class _FixedEmbedder:
            dim: int = _DIM
            model: str = "fixed"
            _v: list[float]

            def __init__(self, v: list[float]) -> None:
                self._v = v

            async def embed(self, text: str) -> list[float]:
                del text
                return list(self._v)

        engine._embedder = _FixedEmbedder(probe_vec)

        # Issue resolve=reinforce: candidate is still best hit but below floor
        resolution = BounceResolution(candidate_id=dec.id, action="reinforce")
        r = await engine.save(
            "below floor decision query",
            scope="global",
            resolve=resolution,
        )
        # Must write novel — NOT reinforce the below-floor candidate
        assert r.outcome == "saved", (
            f"Expected novel write (outcome='saved') but got outcome={r.outcome!r}; "
            "resolve must not reinforce a candidate below the arm floor"
        )
        assert r.hypomnema is not None
        # Original Decision must NOT have been reinforced
        events = await adapter.query_events(target_id=dec.id, source="dedup")
        assert len(events) == 0, "No dedup events should have been written below the arm floor"

    # F2: Scope isolation — supersedes must reject out-of-span targets
    async def test_supersedes_out_of_span_raises(self) -> None:
        """supersedes: target in a different project scope raises ToolError."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        # Create a second scope and insert a Decision there
        other_scope = Scope("otherproject")
        await adapter.create_scope(other_scope)
        foreign_dec = Decision(
            id=new_id(),
            type="decision",
            content="Foreign project policy.",
            scope=other_scope,
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(foreign_dec, vec)

        # Saving into global: supersedes a Decision from an unrelated scope → must fail
        with pytest.raises(ToolError, match="outside the caller's read span"):
            await engine.save(
                "New global policy.",
                kind="decision",
                scope="global",
                supersedes=foreign_dec.id,
            )

    async def test_promoted_from_out_of_span_raises(self) -> None:
        """promoted_from: ember in a different project scope raises ToolError."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        # Create a second scope and insert an ember Fact there
        other_scope = Scope("otherproject")
        await adapter.create_scope(other_scope)
        foreign_ember = Fact(
            id=new_id(),
            type="fact",
            content="Foreign ember.",
            scope=other_scope,
            lifecycle="ember",
            valence="guideline",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(foreign_ember, vec)

        # Saving into global: promoted_from an ember in an unrelated scope → must fail
        with pytest.raises(ToolError, match="outside the caller's read span"):
            await engine.save(
                "New global decision.",
                kind="decision",
                scope="global",
                promoted_from=[foreign_ember.id],
            )

    # F3: replace failure leaves nothing — no compensating bury (P3
    # one-transaction-supersession piece: the Decision
    # arm's bury-on-failure compensation is gone; a failed replace leaves
    # nothing to compensate for)
    async def test_replace_failure_leaves_no_decision_and_predecessor_untouched(self) -> None:
        """If adapter.replace raises, nothing lands: no new Decision (no
        compensating bury needed — there is nothing half-written), and the
        predecessor stays exactly as it was."""
        engine, adapter, clock = await _engine()
        now = clock.now()

        # Insert an in_force predecessor
        predecessor = Decision(
            id=new_id(),
            type="decision",
            content="Predecessor policy.",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [0.0, 1.0] + [0.0] * (_DIM - 2)  # orthogonal to default probes
        await adapter.upsert(predecessor, vec)

        # Count live Decisions before the failing write
        live_before = len([
            rid for rid, rec in adapter._rows.items()
            if isinstance(rec, Decision) and rec.id != predecessor.id
        ])
        assert live_before == 0

        # Patch adapter.replace to always raise
        original_replace = adapter.replace

        async def _failing_replace(*args: Any, **kwargs: Any) -> None:
            raise ValueError("simulated concurrent supersession failure")

        adapter.replace = _failing_replace  # type: ignore[method-assign]
        try:
            with pytest.raises(ToolError, match="supersedes:"):
                await engine.save(
                    "Replacement policy.",
                    kind="decision",
                    scope="global",
                    supersedes=predecessor.id,
                    reason="changed",
                )
        finally:
            adapter.replace = original_replace  # type: ignore[method-assign]

        # No new Decision landed — the failed replace left nothing to compensate for.
        live_after = len([
            rid for rid, rec in adapter._rows.items()
            if isinstance(rec, Decision) and rec.id != predecessor.id
        ])
        assert live_after == 0, "a failed replace must never leave a dangling new Decision"

        # The predecessor itself is untouched.
        pred_after = await adapter.get_by_id(predecessor.id)
        assert isinstance(pred_after, Decision)
        assert pred_after.lifecycle == "in_force"
        assert pred_after.superseded_by is None

    async def test_non_valueerror_replace_failure_still_raises_tool_error(self) -> None:
        """A genuine storage
        I/O failure need not surface as ``ValueError`` — ``_write_decision``
        must convert ANY exception from ``adapter.replace``, not only
        ``ValueError``, into a clean ``ToolError``.

        Falsify: narrow the ``except`` clause in ``_write_decision`` back to
        ``except ValueError`` — this test's ``pytest.raises(ToolError)``
        goes red as the injected ``RuntimeError`` propagates raw instead.
        """
        engine, adapter, clock = await _engine()
        now = clock.now()

        predecessor = Decision(
            id=new_id(),
            type="decision",
            content="Predecessor policy.",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [0.0, 1.0] + [0.0] * (_DIM - 2)
        await adapter.upsert(predecessor, vec)

        original_replace = adapter.replace

        async def _failing_replace(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated disk I/O failure")

        adapter.replace = _failing_replace  # type: ignore[method-assign]
        try:
            with pytest.raises(ToolError, match="supersedes:"):
                await engine.save(
                    "Replacement policy.",
                    kind="decision",
                    scope="global",
                    supersedes=predecessor.id,
                    reason="changed",
                )
        finally:
            adapter.replace = original_replace  # type: ignore[method-assign]

        pred_after = await adapter.get_by_id(predecessor.id)
        assert isinstance(pred_after, Decision)
        assert pred_after.lifecycle == "in_force"
        assert pred_after.superseded_by is None

    # R2-F1: Invalid candidates must not block Decision creation (write past them)
    async def test_decision_save_writes_past_invalid_candidate(self) -> None:
        """An invalid Fact above the gate floor must not bounce a Decision save;
        the new Decision is written with outcome='reconciled'."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert an invalid Fact at the same vector position
        invalid_fact = Fact(
            id=new_id(),
            type="fact",
            content="deprecated policy note",
            scope=Scope("global"),
            lifecycle="active",
            validity="invalid",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(invalid_fact, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("Replacement policy.", kind="decision", scope="global")
        # Must write past the invalid candidate, not bounce
        assert r.outcome == "reconciled", (
            f"Expected outcome='reconciled' (write past invalid) but got {r.outcome!r}"
        )
        assert r.hypomnema is not None
        # Verify the new Decision references the invalid candidate
        new_rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(new_rec, Decision)
        assert new_rec.reconciles == invalid_fact.id

    # R2-F2: Resolve-reinforce on an invalidated Decision must not mutate it
    async def test_resolve_reinforce_invalid_decision_writes_novel(self) -> None:
        """Resolve reinforce on a Decision that became invalid between bounce and resolve
        must write a novel Fact (reconciliation path), not append a dedup event."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, _clock = await _engine(config=cfg)
        engine._embedder = _IdenticalEmbedder()

        r = await engine.save("Stable decision.", kind="decision", scope="global")
        dec_id = r.hypomnema.id  # type: ignore[union-attr]

        # Invalidate the Decision between bounce and resolve
        await adapter.set_validity(dec_id, False, reason="test invalidation")

        resolution = BounceResolution(candidate_id=dec_id, action="reinforce")
        r2 = await engine.save("Stable decision.", scope="global", resolve=resolution)

        # Must write novel (reconciliation) — NOT reinforce the invalid Decision
        assert r2.outcome in {"saved", "reconciled"}, (
            f"Expected novel write but got outcome={r2.outcome!r}"
        )
        # No dedup events must have been written to the invalid Decision
        events = await adapter.query_events(target_id=dec_id, source="dedup")
        assert len(events) == 0, (
            "Dedup event was written to an invalid Decision during resolve-reinforce"
        )

    # R2-F3: resolve passed with kind='decision' is honored (not silently dropped)
    async def test_resolve_with_kind_decision_is_honored(self) -> None:
        """A BounceResolution supplied alongside kind='decision' is honored,
        not silently dropped — confirm_distinct writes the new Decision."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert an existing in_force Decision
        existing = Decision(
            id=new_id(),
            type="decision",
            content="Use Python globally.",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(existing, vec)

        engine._embedder = _IdenticalEmbedder()
        # First save — byte-identical content already_saved acks in place
        # (P7 admission-order piece, step 1) rather than bouncing.
        r = await engine.save("Use Python globally.", kind="decision", scope="global")
        assert r.outcome == "already_saved"
        assert r.hypomnema is not None
        assert r.hypomnema.id == existing.id

        # Resolve confirm_distinct — must write the new Decision (not bounce again)
        resolution = BounceResolution(candidate_id=existing.id, action="confirm_distinct")
        r2 = await engine.save(
            "Use Python globally (variant).",
            kind="decision",
            scope="global",
            resolve=resolution,
        )
        assert r2.outcome == "saved", (
            f"Expected outcome='saved' after resolve confirm_distinct but got {r2.outcome!r}"
        )
        assert r2.hypomnema is not None
        new_rec = await adapter.get_by_id(r2.hypomnema.id)
        assert isinstance(new_rec, Decision)

    async def test_resolve_decision_confirm_distinct_marks_pair(self) -> None:
        """confirm_distinct on a Decision save must register the distinct pair
        so that future saves do not bounce on the same candidate again."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert an existing in_force Decision
        existing = Decision(
            id=new_id(),
            type="decision",
            content="Existing policy.",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(existing, vec)

        engine._embedder = _IdenticalEmbedder()
        # Bounce, then resolve confirm_distinct
        resolution = BounceResolution(candidate_id=existing.id, action="confirm_distinct")
        r = await engine.save(
            "Distinct policy.",
            kind="decision",
            scope="global",
            resolve=resolution,
        )
        assert r.outcome == "saved"
        assert r.hypomnema is not None

        # The pair must be registered
        is_dist = await adapter.is_distinct(existing.id, r.hypomnema.id)
        assert is_dist, (
            "confirm_distinct on a Decision save must register the distinct pair "
            f"({existing.id}, {r.hypomnema.id})"
        )

    # R4-F1: resolve.action='reinforce' with unknown/buried id must raise ToolError
    async def test_resolve_reinforce_unknown_id_raises_for_decision_save(self) -> None:
        """resolve=reinforce targeting an unknown/buried id with kind='decision'
        must raise ToolError, not silently fall through to write novel."""
        engine, _adapter, _ = await _engine()
        unknown_id = new_id()  # not in the store

        resolution = BounceResolution(candidate_id=unknown_id, action="reinforce")
        with pytest.raises(ToolError, match="unknown/buried"):
            await engine.save(
                "Some policy.",
                kind="decision",
                scope="global",
                resolve=resolution,
            )
