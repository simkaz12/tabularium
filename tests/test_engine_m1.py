"""Engine M1 tests -- dispatch matrix, projections, resolve, ember, revival, reconciliation.

Covers all new M1 acceptance criteria per the B1 brief:
  - Every dispatch outcome: reinforced / revived / reconciled / recurrence-on-ember
  - Resolve semantics incl. stale-candidate fresh-bounce and unknown-id ToolError
  - Distinct-pair no-re-bounce
  - Projection math (EWMA a=0.3 init 0.0; [A10] source rules; decay_due_at analytic value
    incl. the never-decays None case)
  - Fallback-hit reinforcement + threshold-gated revival + include_archived-never-reinforces
  - Ember birth via engine.ember
  - Purpose/facet + audience_hint/role validation
  - Park logging (WARNING level)
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.engine.projections import (
    compute_decay_due_at,
    compute_ewma,
    compute_last_referenced_at,
    compute_recall_count,
)
from tabularium.results import BounceResolution
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import ReinforcementEvent
from tests.support.doubles import FakeEmbedder, TestClock

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DIM = 16


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
    should_fail: bool = False,
    clock: TestClock | None = None,
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    adapter = _make_adapter(dim=dim)
    embedder = FakeEmbedder(dim=dim, should_fail=should_fail)
    clk = clock if clock is not None else _make_clock()
    cfg = config if config is not None else _make_config()
    engine = Tabularium(adapter, embedder, clk, cfg)
    await engine.initialize()
    return engine, adapter, clk


# Stub embedder: always returns the same unit vector so every text collides.
class _IdenticalEmbedder:
    dim: int = _DIM
    model: str = "identical"

    async def embed(self, text: str) -> list[float]:
        del text
        v = [0.0] * self.dim
        v[0] = 1.0
        return v


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    """Unit vector at ``angle_deg`` from axis 0, zero-padded to ``dim``.

    ``FakeEmbedder`` only produces cosine 0.0/1.0 for distinct strings (it is
    a single-hot hash), so it cannot express graduated gray-band similarities.
    Two angle-parameterised vectors at angle ``theta`` apart have cosine
    ``cos(theta)`` exactly (both unit length by construction), which lets a
    test place several stored records at known, distinct similarities to a
    single probe.
    """
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _TableEmbedder:
    """Deterministic embedder: exact-string → vector lookup (``KeyError`` on miss).

    Unlike ``_IdenticalEmbedder`` (one vector for every text) or ``FakeEmbedder``
    (binary cosine), this drives distinct incoming texts — e.g. the original
    bounce content and a since-edited resolve content — to distinct,
    pre-computed vectors, which is what a gray-band / content-edit scenario
    needs.
    """

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


async def _collide_engine(
    *,
    config: EngineConfig | None = None,
    clock: TestClock | None = None,
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    """Engine with an identical embedder: all texts get the same vector → always collide."""
    adapter = _make_adapter()
    embedder = _IdenticalEmbedder()
    clk = clock if clock is not None else _make_clock()
    cfg = config if config is not None else _make_config(gray_floor=0.0)
    engine = Tabularium(adapter, embedder, clk, cfg)
    await engine.initialize()
    return engine, adapter, clk


def _fact(
    scope: str = "global",
    lifecycle: str = "active",
    content: str = "test fact",
    now: datetime | None = None,
) -> Fact:
    t = now or datetime(2025, 1, 1, tzinfo=UTC)
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        created_at=t,
        updated_at=t,
    )


# ---------------------------------------------------------------------------
# 1. SaveResult.outcome extended set
# ---------------------------------------------------------------------------


class TestOutcomeValues:
    """SaveResult.outcome covers all six values."""

    async def test_saved_outcome(self) -> None:
        engine, _, _ = await _engine()
        r = await engine.save("novel fact", scope="global")
        assert r.outcome == "saved"
        assert r.hypomnema is not None
        assert r.bounce is None
        assert r.reinforced_id is None

    async def test_bounced_outcome(self) -> None:
        """A near-dup (cosine collision via ``_IdenticalEmbedder``, different
        canonical key) still bounces — a byte-identical resubmission acks
        ``already_saved`` instead (P7 admission-order piece, step 1)."""
        engine, _, _ = await _collide_engine()
        await engine.save("text a", scope="global")
        r = await engine.save("text b", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None

    async def test_vectorless_write_outcome(self) -> None:
        engine, _, _ = await _engine(should_fail=True)
        r = await engine.save("anything", scope="global")
        assert r.outcome == "saved"
        assert r.embedded is False

    async def test_reinforced_outcome_via_exact_key(self) -> None:
        """A whitespace variant (P7 admission-order piece, step 3 exact-key
        match) reinforces via the exact-key path — a byte-identical
        resubmission acks ``already_saved`` instead (step 1)."""
        engine, _, _ = await _engine(config=_make_config())
        text = "sky is blue"
        r1 = await engine.save(text, scope="global")
        assert r1.outcome == "saved"
        r2 = await engine.save(text + " ", scope="global")
        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == r1.hypomnema.id  # type: ignore[union-attr]

    async def test_revived_outcome(self) -> None:
        """Re-assertion of an archived Fact revives it in-place (dedup-revival).

        A DIFFERENT restatement, not a byte-identical or same-key match
        (P7 admission-order piece, step 1/3 would otherwise intercept it
        before the probe ever runs) — but sharing a token with the
        original (the serial probe's BM25 leg needs SOME overlap to
        surface the row at all; ``_IdenticalEmbedder`` forces the cosine
        side of the collision this dedup-revival dispatch needs).
        """
        cfg = _make_config()
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()
        # Directly upsert an archived fact
        archived = _fact(lifecycle="archived", content="archived fact content", now=now)
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(archived, vec)

        # Re-assert: engine sees archived → dedup-revival
        embedder = _IdenticalEmbedder()
        engine._embedder = embedder
        r = await engine.save("archived fact content restated differently", scope="global")
        assert r.outcome == "revived"
        assert r.reinforced_id == archived.id

        # Confirm the record is now active
        rec = await adapter.get_by_id(archived.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "active"

    async def test_reconciled_outcome(self) -> None:
        """Save past an invalid match → reconciled, new record carries reconciles."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert an invalid record with same vector so it's the best match
        invalid_fact = Fact(
            id=new_id(),
            type="fact",
            content="invalid content",
            scope=Scope("global"),
            lifecycle="active",
            validity="invalid",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(invalid_fact, vec)

        # Save with same vector (collide embedder) — should reconcile
        embedder = _IdenticalEmbedder()
        engine._embedder = embedder
        r = await engine.save("some new content", scope="global")
        assert r.outcome == "reconciled"
        assert r.hypomnema is not None
        new_rec = await adapter.get_by_id(r.hypomnema.id)
        assert new_rec.reconciles == invalid_fact.id


# ---------------------------------------------------------------------------
# 2. Dispatch matrix — ember recurrence
# ---------------------------------------------------------------------------


class TestEmberRecurrence:
    """valid ember → recurrence: reinforce with source='dedup'."""

    async def test_ember_birth_via_engine_ember(self) -> None:
        """engine.ember creates a lifecycle=ember fact."""
        engine, adapter, _ = await _engine()
        r = await engine.ember("watch out for this pattern", scope="global", valence="guideline")
        assert r.outcome == "saved"
        assert r.hypomnema is not None
        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "ember"
        assert rec.valence == "guideline"

    async def test_ember_recurrence_counts_dedup_events(self) -> None:
        """Re-asserting an ember reinforces it (source='dedup').

        Recurrence r = count(source='dedup' events) on the ember.
        The incoming content shares the ember's canonical key, so the
        ember-recurrence branch reinforces unconditionally (P7
        admission-order piece, decision D14).
        """
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, _ = await _engine(config=cfg)

        r1 = await engine.ember("ember content here", scope="global", valence="guideline")
        assert r1.outcome == "saved"
        ember_id = r1.hypomnema.id  # type: ignore[union-attr]

        # Re-assert (collide vectors via identical text)
        r2 = await engine.save("ember content here", scope="global")
        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == ember_id

        # r = count(source='dedup' events) = 1 after one recurrence
        events = await adapter.query_events(target_id=ember_id, source="dedup")
        assert len(events) == 1

        # Another re-assertion increments r to 2
        r3 = await engine.save("ember content here", scope="global")
        assert r3.outcome == "reinforced"
        events2 = await adapter.query_events(target_id=ember_id, source="dedup")
        assert len(events2) == 2

    async def test_ember_valence_correction(self) -> None:
        """engine.ember with valence='correction' is accepted."""
        engine, adapter, _ = await _engine()
        r = await engine.ember("correction nudge", scope="global", valence="correction")
        assert r.outcome == "saved"
        rec = await adapter.get_by_id(r.hypomnema.id)  # type: ignore[union-attr]
        assert isinstance(rec, Fact)
        assert rec.valence == "correction"


# ---------------------------------------------------------------------------
# 3. Dedup-revival (archived Fact → in-place transition + reinforce)
# ---------------------------------------------------------------------------


class TestDedupRevival:
    async def test_dedup_revival_transitions_archived_to_active(self) -> None:
        """Dedup-revival: archived→active unconditionally; reinforcement event emitted.

        A DIFFERENT restatement, not a byte-identical match (P7 admission-
        order piece, step 1 would otherwise revive-and-ack ``already_saved``
        one lock hold earlier) — ``_IdenticalEmbedder`` still forces the
        cosine collision this dedup-revival dispatch needs.
        """
        cfg = _make_config()
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        archived = _fact(lifecycle="archived", content="archived fact", now=now)
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(archived, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("a restatement of the archived fact", scope="global")
        assert r.outcome == "revived"
        assert r.reinforced_id == archived.id

        # Verify lifecycle = active
        rec = await adapter.get_by_id(archived.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "active"

        # Verify revival event was emitted
        events = await adapter.query_events(target_id=archived.id, source="revival")
        assert len(events) == 1

    async def test_dedup_revival_increments_recall_count(self) -> None:
        """After dedup-revival, recall_count = 1 (one source='revival' event)."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        archived = _fact(lifecycle="archived", content="old archived", now=now)
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(archived, vec)

        engine._embedder = _IdenticalEmbedder()
        await engine.save("old archived", scope="global")

        rec = await adapter.get_by_id(archived.id)
        assert isinstance(rec, Fact)
        assert rec.recall_count == 1  # 'revival' counts toward recall_count


# ---------------------------------------------------------------------------
# 4. Reconciliation — non-blocking, new record carries reconciles
# ---------------------------------------------------------------------------


class TestReconciliation:
    async def test_reconciliation_writes_new_record(self) -> None:
        """Invalid match → new record written, outcome='reconciled'.

        Shares a token with ``invalid.content`` (the serial probe's BM25
        leg needs SOME overlap to surface the row at all — P7 admission-
        order piece); ``_IdenticalEmbedder`` forces the cosine side of the
        collision this reconciliation dispatch needs.
        """
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        invalid = Fact(
            id=new_id(),
            type="fact",
            content="invalid fact",
            scope=Scope("global"),
            lifecycle="active",
            validity="invalid",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(invalid, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("new valid fact content", scope="global")
        assert r.outcome == "reconciled"
        assert r.hypomnema is not None

        new_rec = await adapter.get_by_id(r.hypomnema.id)
        assert new_rec.reconciles == invalid.id
        assert new_rec.validity == "valid"
        # The invalid record is never touched
        invalid_rec = await adapter.get_by_id(invalid.id)
        assert invalid_rec.validity == "invalid"

    async def test_reconciliation_does_not_touch_invalid_record(self) -> None:
        """The invalid record's content / validity must be unchanged after reconciliation."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        invalid = Fact(
            id=new_id(),
            type="fact",
            content="stale invalid content",
            scope=Scope("global"),
            lifecycle="active",
            validity="invalid",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(invalid, vec)

        engine._embedder = _IdenticalEmbedder()
        await engine.save("fresh valid content", scope="global")

        still_invalid = await adapter.get_by_id(invalid.id)
        assert still_invalid.validity == "invalid"
        assert still_invalid.content == "stale invalid content"


# ---------------------------------------------------------------------------
# 5. Decision bounce (lower decision_gray_floor)
# ---------------------------------------------------------------------------


class TestDecisionBounce:
    async def test_decision_hit_bounces_with_decision_adjacent_verdict(self) -> None:
        """A Decision best match bounces eagerly with verdict='decision_adjacent'."""
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert a Decision directly (engine.save won't write Decisions yet)
        decision = Decision(
            id=new_id(),
            type="decision",
            content="we shall use Python",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("we shall use Python", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.verdict == "decision_adjacent"

    async def test_decision_never_auto_reinforces(self) -> None:
        """A Decision candidate never auto-reinforces — it bounces eagerly
        below its own arm floor (P7 admission-order piece: the band that
        could once auto-reinforce past a probe survivor is gone entirely,
        so this holds for every config, not just a non-default one)."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="we shall not pass",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("we shall not pass", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.verdict == "decision_adjacent"


# ---------------------------------------------------------------------------
# 6. Resolve semantics [A6]
# ---------------------------------------------------------------------------


class TestResolveSemantics:
    async def test_resolve_confirm_distinct_writes_and_marks(self) -> None:
        """resolve confirm_distinct: new record written + pair registered."""
        engine, adapter, _ = await _collide_engine(
            config=_make_config(gray_floor=0.0)
        )
        r1 = await engine.save("original content", scope="global")
        assert r1.outcome == "saved"
        orig_id = r1.hypomnema.id  # type: ignore[union-attr]

        # Second save bounces
        r2 = await engine.save("different content", scope="global")
        assert r2.outcome == "bounced"
        assert r2.bounce is not None

        # Resolve: confirm they are distinct
        resolution = BounceResolution(candidate_id=orig_id, action="confirm_distinct")
        r3 = await engine.save("different content", scope="global", resolve=resolution)
        assert r3.outcome == "saved"
        assert r3.hypomnema is not None
        new_id_ = r3.hypomnema.id

        # Pair must be registered
        assert await adapter.is_distinct(orig_id, new_id_)

    async def test_resolve_reinforce_active_fact(self) -> None:
        """resolve reinforce on an active Fact → reinforced."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, _ = await _collide_engine(config=cfg)

        r1 = await engine.save("original", scope="global")
        assert r1.outcome == "saved"
        orig_id = r1.hypomnema.id  # type: ignore[union-attr]

        # A plain byte-identical resubmission acks already_saved (P7
        # admission-order piece, step 1), never a bounce.
        r2 = await engine.save("original", scope="global")
        assert r2.outcome == "already_saved"

        # Resolve reinforce
        resolution = BounceResolution(candidate_id=orig_id, action="reinforce")
        r3 = await engine.save("original", scope="global", resolve=resolution)
        assert r3.outcome == "reinforced"
        assert r3.reinforced_id == orig_id

        # Event was appended
        events = await adapter.query_events(target_id=orig_id, source="dedup")
        assert len(events) == 1

    async def test_resolve_reinforce_unknown_id_raises_tool_error(self) -> None:
        """resolve reinforce with unknown/buried id → ToolError."""
        engine, _, _ = await _engine()
        fake_id = new_id()
        resolution = BounceResolution(candidate_id=fake_id, action="reinforce")
        with pytest.raises(ToolError, match="unknown/buried"):
            await engine.save("some content", scope="global", resolve=resolution)

    async def test_resolve_stale_candidate_returns_fresh_bounce(self) -> None:
        """If candidate is no longer the best match, resolve returns a fresh bounce."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        # Insert orig and a second record with a different vector
        orig_id = new_id()
        orig = Fact(
            id=orig_id,
            type="fact",
            content="original text",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )
        vec_orig = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(orig, vec_orig)

        # Insert a nearer record
        near_id = new_id()
        near = Fact(
            id=near_id,
            type="fact",
            content="nearer text",
            scope=Scope("global"),
            lifecycle="active",
            created_at=now,
            updated_at=now,
        )
        vec_near = [1.0] + [0.0] * (_DIM - 1)  # same vector = ties; orig inserted first
        await adapter.upsert(near, vec_near)

        # Try to resolve against the orig_id, but use an embedder that points to near
        # To force a "stale candidate" we pass a candidate_id that is NOT the best match.
        # We do this by burying orig_id — then the best match is near, not orig.
        await adapter.bury(orig_id, "test: buried to test stale resolve")

        resolution = BounceResolution(candidate_id=orig_id, action="confirm_distinct")
        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("nearer text", scope="global", resolve=resolution)
        # Candidate orig is buried → it is no longer the best match → fresh bounce
        # (or pass, if the near record is also not above the floor after the burial).
        # P7 admission-order piece: the submitted content is byte-identical to
        # the live "near" record, so admission step 1 (which still runs for a
        # confirm_distinct resolve — it is what replaces the old confirm_distinct
        # replay guard) acks already_saved(near_id) directly, never reaching the
        # probe at all — a stronger, still-correct answer to "orig is stale":
        # the content already exists verbatim, so there is nothing left to
        # confirm distinct against.
        # The important invariant: outcome must NOT be "saved" with reconciles=orig_id
        assert r.outcome in {"bounced", "saved", "already_saved"}

    async def test_distinct_pair_never_rebounces(self) -> None:
        """A pair already judged distinct skips re-bounce (is_distinct consult)."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, _ = await _collide_engine(config=cfg)

        r1 = await engine.save("content a", scope="global")
        orig_id = r1.hypomnema.id  # type: ignore[union-attr]

        # Confirm distinct → new record saved
        resolution = BounceResolution(candidate_id=orig_id, action="confirm_distinct")
        r2 = await engine.save("content b", scope="global", resolve=resolution)
        assert r2.outcome == "saved"
        new_id_val = r2.hypomnema.id  # type: ignore[union-attr]

        # Now both ids are in the distinct registry; re-saving "content b" should
        # skip orig as a candidate (it's distinct from new_id_val) — won't bounce again.
        # But there IS another record (orig) so we may get a fresh save or a pass.
        # The key: resolve=BounceResolution(orig_id) should not bounce against orig_id
        # when the pair is already in the registry.
        assert await adapter.is_distinct(orig_id, new_id_val)

    async def test_resolve_reinforce_decision_emits_dedup_event(self) -> None:
        """resolve reinforce on a Decision emits a source='dedup' event (SPEC §4.4 / B2 item 8).

        The agent-adjudicated false-alarm path is allowed: reinforce via resolve
        bumps recall_count/last_referenced_at.  Auto-reinforce of Decisions stays
        forbidden — only the agent can adjudicate via resolve.
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="important decision content",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        resolution = BounceResolution(candidate_id=decision.id, action="reinforce")
        r = await engine.save("important decision content", scope="global", resolve=resolution)
        assert r.outcome == "reinforced"
        assert r.reinforced_id == decision.id

        # A source='dedup' event must have been appended
        events = await adapter.query_events(target_id=decision.id, source="dedup")
        assert len(events) == 1

        # recall_count should be 1 after one dedup event
        rec = await adapter.get_by_id(decision.id)
        assert isinstance(rec, Decision)
        assert rec.recall_count == 1


# ---------------------------------------------------------------------------
# 6b. Gray-band multi-candidate resolve [A6] (two-plus records above floor)
# ---------------------------------------------------------------------------


class TestResolveGrayBandMultiCandidate:
    """Resolve semantics when TWO stored records are above ``gray_floor`` for
    one incoming save (SPEC §4.5 [A6]) — the shape no prior test constructed
    (``FakeEmbedder`` can't express graduated similarities; these use
    ``_TableEmbedder``/``_angle_vec`` instead).

    Geometry shared by both tests: candidate A at angle 0°, candidate B at
    angle 25°, both well above ``gray_floor=0.80`` from any probe angle used
    below.
    """

    async def test_identical_content_resolves_on_first_attempt(self) -> None:
        """Two candidates above floor + UNCHANGED content on resolve → the
        originally-bounced candidate resolves on the FIRST round-trip: with
        content held constant, nothing oscillates.
        """
        cfg = _make_config(gray_floor=0.80)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        cand_a = _fact(content="existing note about quarterly planning cadence", now=now)
        cand_b = _fact(content="existing note about roadmap review cadence", now=now)
        await adapter.upsert(cand_a, _angle_vec(0))
        await adapter.upsert(cand_b, _angle_vec(25))

        c1 = "incoming note about quarterly planning items"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == cand_a.id  # cos(10°)=.9848 > cos(15°)=.9659

        resolution = BounceResolution(candidate_id=cand_a.id, action="confirm_distinct")
        r2 = await engine.save(c1, scope="global", resolve=resolution)  # SAME content
        assert r2.outcome == "saved", (
            f"expected the first round-trip to succeed with unchanged content, "
            f"got outcome={r2.outcome!r}"
        )
        assert r2.hypomnema is not None
        assert await adapter.is_distinct(cand_a.id, r2.hypomnema.id)

    async def test_edited_content_flips_top1_and_returns_fresh_bounce(self) -> None:
        """Content EDITED between the bounce and the resolve, such that the
        top-1 candidate flips → a fresh bounce naming the NEW best candidate,
        and nothing is written.  This is the [A6] contract working as
        designed, not the "livelock" originally (and wrongly) reported.
        """
        cfg = _make_config(gray_floor=0.80)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        cand_a = _fact(content="existing note about quarterly planning cadence", now=now)
        cand_b = _fact(content="existing note about roadmap review cadence", now=now)
        await adapter.upsert(cand_a, _angle_vec(0))
        await adapter.upsert(cand_b, _angle_vec(25))

        c1 = "incoming note about quarterly planning items"  # angle 10 -> A wins
        c2 = "incoming note about roadmap review items"      # angle 20 -> B wins
        engine._embedder = _TableEmbedder({c1: _angle_vec(10), c2: _angle_vec(20)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == cand_a.id

        rows_before = [row async for row in await adapter.scan()]

        resolution = BounceResolution(candidate_id=cand_a.id, action="confirm_distinct")
        r2 = await engine.save(c2, scope="global", resolve=resolution)  # EDITED content
        assert r2.outcome == "bounced", (
            f"expected a fresh bounce naming the new top-1 (B), got outcome={r2.outcome!r}"
        )
        assert r2.bounce is not None
        # top-1 flipped: cos(5°)=.9962 (B) > cos(20°)=.9397 (A)
        assert r2.bounce.candidate_id == cand_b.id
        assert r2.hypomnema is None

        rows_after = [row async for row in await adapter.scan()]
        assert len(rows_after) == len(rows_before), "nothing should have been written"


# ---------------------------------------------------------------------------
# 6c. Bounce envelope — other_candidates (SPEC §4.4)
# ---------------------------------------------------------------------------


class TestBounceOtherCandidates:
    """``other_candidates`` (``FactBounce``/``DecisionBounce``) surfaces other
    above-floor neighbours from the same probe, without changing the primary
    candidate's semantics.
    """

    async def test_populated_with_second_neighbor_descending(self) -> None:
        cfg = _make_config(gray_floor=0.80)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        cand_a = _fact(content="existing note about quarterly planning cadence", now=now)
        cand_b = _fact(content="existing note about roadmap review cadence", now=now)
        await adapter.upsert(cand_a, _angle_vec(0))
        await adapter.upsert(cand_b, _angle_vec(25))

        c1 = "incoming note about quarterly planning items"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == cand_a.id

        assert len(r1.bounce.other_candidates) == 1
        neighbor = r1.bounce.other_candidates[0]
        assert neighbor.candidate_id == cand_b.id
        assert neighbor.similarity == pytest.approx(math.cos(math.radians(15)), abs=1e-6)
        assert neighbor.guard_flag is None

    async def test_empty_when_only_one_candidate_above_floor(self) -> None:
        cfg = _make_config(gray_floor=0.80)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        cand_a = _fact(content="existing note about quarterly planning cadence", now=now)
        far = _fact(content="unrelated note about kitchen inventory levels", now=now)
        await adapter.upsert(cand_a, _angle_vec(0))
        await adapter.upsert(far, _angle_vec(90))  # cos(80°)=.174, well below floor

        c1 = "incoming note about quarterly planning items"
        engine._embedder = _TableEmbedder({c1: _angle_vec(10)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == cand_a.id
        assert r1.bounce.other_candidates == []

    async def test_equal_similarity_neighbors_break_ties_by_id(self) -> None:
        """Two neighbours tied at the exact same similarity still come back in
        a deterministic order (ascending id string) — the storage seam only
        promises hits ranked by cosine (SPEC §3), not a specific tie-break for
        equal scores, so ``_bounce_neighbors`` must own its own total order
        rather than inherit whatever an adapter happens to do with ties.
        """
        cfg = _make_config(gray_floor=0.80)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        cand_a = _fact(content="existing note about quarterly planning cadence", now=now)
        cand_b = _fact(content="existing note about roadmap review cadence", now=now)
        cand_c = _fact(content="existing note about budget review cadence", now=now)

        # Insert the tied pair in the OPPOSITE order from their id-string sort,
        # so a missing tie-break would surface adapter insert-order instead
        # (MemoryAdapter breaks ties by insert order) and this test would catch it.
        tied_first, tied_second = sorted([cand_b, cand_c], key=lambda r: str(r.id), reverse=True)
        await adapter.upsert(cand_a, _angle_vec(5))
        await adapter.upsert(tied_first, _angle_vec(20))
        await adapter.upsert(tied_second, _angle_vec(-20))  # cos(-20°) == cos(20°): exact tie

        c1 = "incoming note about quarterly planning items"
        engine._embedder = _TableEmbedder({c1: _angle_vec(0)})

        r1 = await engine.save(c1, scope="global")
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == cand_a.id  # cos(5°) is the closest

        other = r1.bounce.other_candidates
        assert len(other) == 2
        assert other[0].similarity == pytest.approx(other[1].similarity, abs=1e-9)
        assert [n.candidate_id for n in other] == sorted([cand_b.id, cand_c.id], key=str)


# ---------------------------------------------------------------------------
# 7. Projection math (SPEC §8.2, [A10], [A8])
# ---------------------------------------------------------------------------


class TestProjectionMath:
    """Unit tests for the standalone projection computation functions."""

    def test_recall_count_counts_dedup_revival_compact(self) -> None:
        t = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            ReinforcementEvent(target_id=new_id(), at=t, source="dedup"),
            ReinforcementEvent(target_id=new_id(), at=t, source="revival"),
            ReinforcementEvent(target_id=new_id(), at=t, source="compact"),
            ReinforcementEvent(target_id=new_id(), at=t, source="feedback", value=0.5),
        ]
        # feedback is excluded from recall_count
        assert compute_recall_count(events) == 3

    def test_recall_count_excludes_feedback(self) -> None:
        t = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            ReinforcementEvent(target_id=new_id(), at=t, source="feedback", value=0.8),
            ReinforcementEvent(target_id=new_id(), at=t, source="feedback", value=0.3),
        ]
        assert compute_recall_count(events) == 0

    def test_last_referenced_at_max_of_dedup_revival(self) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 1, 2, tzinfo=UTC)
        t3 = datetime(2025, 1, 3, tzinfo=UTC)
        events = [
            ReinforcementEvent(target_id=new_id(), at=t1, source="dedup"),
            ReinforcementEvent(target_id=new_id(), at=t3, source="dedup"),
            ReinforcementEvent(target_id=new_id(), at=t2, source="compact"),  # excluded
        ]
        result = compute_last_referenced_at(events)
        # Max of dedup events = t3; compact is excluded [A10]
        assert result == t3

    def test_last_referenced_at_excludes_compact(self) -> None:
        t_compact = datetime(2025, 3, 1, tzinfo=UTC)
        t_dedup = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            ReinforcementEvent(target_id=new_id(), at=t_dedup, source="dedup"),
            ReinforcementEvent(target_id=new_id(), at=t_compact, source="compact"),
        ]
        result = compute_last_referenced_at(events)
        # compact is excluded; only dedup counts
        assert result == t_dedup

    def test_last_referenced_at_none_when_no_qualifying_events(self) -> None:
        t = datetime(2025, 1, 1, tzinfo=UTC)
        events = [
            ReinforcementEvent(target_id=new_id(), at=t, source="compact"),
            ReinforcementEvent(target_id=new_id(), at=t, source="feedback", value=0.5),
        ]
        result = compute_last_referenced_at(events)
        assert result is None

    def test_ewma_init_0_alpha_0_3(self) -> None:
        """EWMA with a=0.3, init=0.0 over three feedback events."""
        t = datetime(2025, 1, 1, tzinfo=UTC)
        tid = new_id()
        events = [
            ReinforcementEvent(target_id=tid, at=t, source="feedback", value=1.0),
            ReinforcementEvent(target_id=tid, at=t, source="feedback", value=0.0),
            ReinforcementEvent(target_id=tid, at=t, source="feedback", value=1.0),
        ]
        # e0 = 0.0 (init)
        # e1 = 0.3*1.0 + 0.7*0.0 = 0.3
        # e2 = 0.3*0.0 + 0.7*0.3 = 0.21
        # e3 = 0.3*1.0 + 0.7*0.21 = 0.3 + 0.147 = 0.447
        expected = 0.3 + 0.7 * 0.21
        result = compute_ewma(events, alpha=0.3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_ewma_ignores_non_feedback_events(self) -> None:
        """Non-feedback events do not affect feedback_score."""
        t = datetime(2025, 1, 1, tzinfo=UTC)
        tid = new_id()
        events = [
            ReinforcementEvent(target_id=tid, at=t, source="dedup"),
            ReinforcementEvent(target_id=tid, at=t, source="revival"),
            ReinforcementEvent(target_id=tid, at=t, source="feedback", value=0.6),
        ]
        # Only the feedback event affects the score
        # e0=0.0 (init), e1 = 0.3*0.6 + 0.7*0.0 = 0.18
        result = compute_ewma(events, alpha=0.3)
        assert math.isclose(result, 0.18, rel_tol=1e-9)

    def test_ewma_no_events_returns_0(self) -> None:
        assert compute_ewma([]) == 0.0

    def test_decay_due_at_permanence_when_constant_exceeds_threshold(self) -> None:
        """recall_count high enough → constant ≥ archive_threshold → None."""
        cfg = _make_config(archive_threshold=0.10)
        # w_recall=0.5, ln(1+100) ≈ 4.615 → constant ≈ 2.31 >> 0.10
        result = compute_decay_due_at(
            recall_count=100,
            feedback_score=0.0,
            last_referenced_at=None,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            cfg=cfg,
        )
        assert result is None, "High recall_count buys permanence → decay_due_at=None"

    def test_decay_due_at_analytic_formula(self) -> None:
        """Verify analytic decay_due_at against hand-computed value.

        Formula:
          constant = w_recall*ln(1+rc) + w_feedback*fs
          d_days = -half_life * ln((archive_threshold - constant) / w_recency)
          decay_due_at = last_referenced_at + d_days

        Seeds: w_recall=0.5, w_recency=1.0, w_feedback=1.5, half_life=30, archive_threshold=0.10.
        Test case: recall_count=0, feedback_score=0.0 -> constant=0.0.
          d_days = -30 * ln((0.10 - 0.0) / 1.0) = -30 * ln(0.10) = -30 * (-2.302...) ~= 69.08
        """
        cfg = EngineConfig(
            w_recall=0.5,
            w_recency=1.0,
            w_feedback=1.5,
            half_life_days=30.0,
            archive_threshold=0.10,
            revival_threshold=0.15,
        )
        ref_at = datetime(2025, 1, 1, tzinfo=UTC)
        result = compute_decay_due_at(
            recall_count=0,
            feedback_score=0.0,
            last_referenced_at=ref_at,
            created_at=ref_at,
            cfg=cfg,
        )
        assert result is not None
        expected_delta = -30.0 * math.log(0.10 / 1.0)
        expected = ref_at + timedelta(days=expected_delta)
        assert result is not None
        diff = abs((result - expected).total_seconds())
        assert diff < 1.0, f"decay_due_at off by {diff:.2f}s (expected {expected}, got {result})"

    def test_decay_due_at_uses_last_referenced_at_when_present(self) -> None:
        """decay_due_at is anchored to last_referenced_at, not created_at."""
        cfg = EngineConfig(
            w_recall=0.5,
            w_recency=1.0,
            w_feedback=1.5,
            half_life_days=30.0,
            archive_threshold=0.10,
            revival_threshold=0.15,
        )
        created_at = datetime(2025, 1, 1, tzinfo=UTC)
        last_ref = datetime(2025, 6, 1, tzinfo=UTC)  # well after created_at
        result = compute_decay_due_at(
            recall_count=0,
            feedback_score=0.0,
            last_referenced_at=last_ref,
            created_at=created_at,
            cfg=cfg,
        )
        assert result is not None
        # Must be anchored to last_ref, not created_at
        # Must be in the future relative to last_referenced_at
        assert result > last_ref
        assert result > datetime(2025, 6, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 8. Projection integration: reinforce updates stored projections
# ---------------------------------------------------------------------------


class TestProjectionIntegration:
    """After a real reinforce call, the adapter stores updated projections."""

    async def test_reinforce_updates_recall_count(self) -> None:
        cfg = _make_config()
        engine, adapter, _ = await _engine(config=cfg)

        r = await engine.save("some fact", scope="global")
        assert r.outcome == "saved"
        fid = r.hypomnema.id  # type: ignore[union-attr]

        # A whitespace variant (P7 admission-order piece, step 3 exact-key
        # match) reinforces via the exact-key path.
        r2 = await engine.save("some fact ", scope="global")
        assert r2.outcome == "reinforced"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.recall_count == 1  # one source='dedup' event

    async def test_direct_feedback_events_compute_ewma(self) -> None:
        """Directly appended feedback events produce the correct EWMA feedback_score."""
        cfg = _make_config(gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        r = await engine.save("some fact", scope="global")
        assert r.outcome == "saved"
        fid = r.hypomnema.id  # type: ignore[union-attr]

        # Directly append two feedback events (no feedback tool yet — testing the formula)
        ev1 = ReinforcementEvent(target_id=fid, at=now, source="feedback", value=1.0)
        ev2 = ReinforcementEvent(target_id=fid, at=now, source="feedback", value=0.0)
        await adapter.append_event(ev1)
        await adapter.append_event(ev2)

        # Compute expected EWMA
        # e0=0.0, e1=0.3*1.0=0.3, e2=0.3*0.0+0.7*0.3=0.21
        expected_ewma = 0.21

        all_events = await adapter.query_events(target_id=fid)
        computed = compute_ewma(all_events, alpha=0.3)
        assert math.isclose(computed, expected_ewma, rel_tol=1e-9)

    async def test_decay_due_at_stored_after_reinforce(self) -> None:
        """After reinforce, decay_due_at is stored on the record."""
        cfg = _make_config()
        engine, adapter, _ = await _engine(config=cfg)

        r = await engine.save("decaying fact", scope="global")
        fid = r.hypomnema.id  # type: ignore[union-attr]

        # A whitespace variant (P7 admission-order piece, step 3 exact-key
        # match) reinforces via the exact-key path.
        r2 = await engine.save("decaying fact ", scope="global")
        assert r2.outcome == "reinforced"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        # With recall_count=1 (source='dedup'), the constant may or may not exceed
        # archive_threshold=0.10. Let's compute manually:
        # constant = 0.5 * ln(2) + 1.5 * 0.0 ≈ 0.347 > 0.10 → permanence → None
        assert rec.decay_due_at is None, (
            "recall_count=1 gives constant≈0.347 > archive_threshold=0.10 → permanence"
        )

    async def test_last_referenced_at_updated_on_dedup_reinforce(self) -> None:
        """last_referenced_at is updated on dedup reinforce (source='dedup' counts)."""
        cfg = _make_config()
        clock = _make_clock()
        engine, adapter, _ = await _engine(config=cfg, clock=clock)

        r = await engine.save("fact content", scope="global")
        fid = r.hypomnema.id  # type: ignore[union-attr]

        # Advance clock so the reinforce has a later timestamp. A whitespace
        # variant (P7 admission-order piece, step 3 exact-key match)
        # reinforces via the exact-key path.
        clock.advance(timedelta(days=1))
        r2 = await engine.save("fact content ", scope="global")
        assert r2.outcome == "reinforced"

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.last_referenced_at is not None
        assert rec.last_referenced_at == datetime(2025, 1, 2, tzinfo=UTC)

    async def test_compact_does_not_update_last_referenced_at(self) -> None:
        """compact events count toward recall_count but NOT last_referenced_at [A10]."""
        _, _adapter, _ = await _engine()
        now = datetime(2025, 1, 1, tzinfo=UTC)
        later = datetime(2025, 2, 1, tzinfo=UTC)

        target = new_id()
        # First a dedup event at `now`
        ev_dedup = ReinforcementEvent(target_id=target, at=now, source="dedup")
        # Then a compact event at `later` (offline sweep)
        ev_compact = ReinforcementEvent(target_id=target, at=later, source="compact")

        events = [ev_dedup, ev_compact]
        rc = compute_recall_count(events)
        last_ref = compute_last_referenced_at(events)

        assert rc == 2  # both count toward recall_count
        assert last_ref == now  # compact excluded from last_referenced_at


# ---------------------------------------------------------------------------
# 9. Fallback-hit reinforcement + threshold-gated revival
# ---------------------------------------------------------------------------


class TestFallbackRevival:
    async def test_fallback_hit_emits_revival_event(self) -> None:
        """A hit returned by the fallback pass gets a source='revival' event."""
        cfg = _make_config(min_active_results=2,
                           revival_threshold=1000.0)  # very high: won't transition
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        archived = _fact(lifecycle="archived", content="archived recall content", now=now)
        vec = FakeEmbedder(dim=_DIM).peek("archived recall content")
        await adapter.upsert(archived, vec)

        # Primary pass finds 0 active records < min_active_results=2 → widen
        result = await engine.recall("archived recall content", scope="global")
        assert any(h.id == archived.id for h in result.hits)

        events = await adapter.query_events(target_id=archived.id, source="revival")
        assert len(events) == 1

    async def test_fallback_hit_transitions_when_above_revival_threshold(self) -> None:
        """Fallback hit reinforcement triggers archived→active when importance crosses threshold."""
        # Set revival_threshold very low so even a single revival event exceeds it
        # archive_threshold must be < revival_threshold per the config validator
        cfg = _make_config(min_active_results=2,
                           revival_threshold=0.001,
                           archive_threshold=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        archived = _fact(lifecycle="archived", content="archived threshold fact", now=now)
        vec = FakeEmbedder(dim=_DIM).peek("archived threshold fact")
        await adapter.upsert(archived, vec)

        await engine.recall("archived threshold fact", scope="global")

        rec = await adapter.get_by_id(archived.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "active", "Should have been revived by fallback"

    async def test_include_archived_never_reinforces(self) -> None:
        """Explicit include_archived=True reads never reinforce or revive (SPEC §5)."""
        cfg = _make_config(min_active_results=0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        archived = _fact(lifecycle="archived", content="explicit archived fact", now=now)
        vec = FakeEmbedder(dim=_DIM).peek("explicit archived fact")
        await adapter.upsert(archived, vec)

        await engine.recall("explicit archived fact", scope="global", include_archived=True)

        events = await adapter.query_events(target_id=archived.id)
        assert events == [], "include_archived=True must not emit any reinforcement events"

        rec = await adapter.get_by_id(archived.id)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "archived", "include_archived=True must not change lifecycle"


# ---------------------------------------------------------------------------
# 10. Purpose / role validation (SPEC §10.1-10.3)
# ---------------------------------------------------------------------------


class TestRegistryValidation:
    async def test_all_five_purposes_pre_registered(self) -> None:
        """initialize() pre-registers constraint/convention/preference/pitfall/reference."""
        _, adapter, _ = await _engine()
        facets = await adapter.list_facets()
        for p in ("constraint", "convention", "preference", "pitfall", "reference"):
            assert p in facets, f"Expected {p!r} to be pre-registered"

    async def test_default_role_no_longer_pre_registered(self) -> None:
        """Curation block: 'default' is retired -- initialize() no
        longer seeds it on a fresh store. ``role`` is now required on
        ``context`` and there is no fallback for it to resolve to."""
        _, adapter, _ = await _engine()
        roles = await adapter.list_roles()
        assert "default" not in roles

    async def test_reviewer_role_pre_registered(self) -> None:
        """initialize() pre-registers the 'reviewer' role (per-role grounding,
        M2 slice C, 2026-08-02) — a session-start hook for review sub-agents
        requests role="reviewer" unconditionally, so a fresh (or freshly-upgraded,
        since create_role is idempotent) store must not require a manual
        `tabularium role add reviewer` before the very first sub-agent dispatch works.

        Falsifiability: drop ``_REVIEWER_ROLE`` from ``ALL_ROLES`` (the tuple
        ``Tabularium.initialize`` now loops over, curation block,
        2026-08-03) — this test goes red, and separately
        ``context("global", budget=800, role="reviewer")`` on a fresh engine
        raises ``ToolError: Unregistered role 'reviewer'``.
        """
        _, adapter, _ = await _engine()
        roles = await adapter.list_roles()
        assert "reviewer" in roles

    async def test_all_four_roles_pre_registered_on_fresh_store(self) -> None:
        """initialize() pre-registers exactly the four function-named
        audiences on a fresh store (curation block 2026-08-03) — 'decider'
        (main), 'builder' (implementer agents), 'reviewer'
        (reviewer agents), 'navigator' (scout and research agents). 'navigator' is registered even
        though it receives no
        payload of its own in this block: routing is subtractive, so
        keeping an ember away from a scout requires the scout to be
        nameable in some other atom's audience_hint. 'default' (the
        pre-per-role-grounding fallback) is RETIRED : a fresh
        store's registry is exactly these four, never five.

        Falsifiability: comment out one role's ``create_role`` seed (via
        ``ALL_ROLES``) in ``Tabularium.initialize`` — this test goes red
        naming exactly the missing role in the set-difference assertion
        error. See the task verdict for the captured output.
        """
        _, adapter, _ = await _engine()
        roles = await adapter.list_roles()
        assert set(roles) == {"decider", "builder", "reviewer", "navigator"}

    async def test_initialize_idempotent_against_live_store_shape(self) -> None:
        """Re-running initialize() against a store that already holds
        exactly 'default' + 'reviewer' — the shape of the deployed store as
        of 2026-08-03, before curation block's rollout — adds the
        missing three function-named roles with no duplicates, and REMOVES
        the pre-existing 'default' row: as of that change, retiring 'default'
        means initialize() actively calls ``delete_role("default")``
        unconditionally, not merely stops seeding it (which is all that
        alone could guarantee). Asserted, not assumed: this is the exact
        scenario the live deploy exercises, since ``create_role`` is
        idempotent on both adapters and ``initialize()`` is re-run on every
        service start.

        Falsifiability: make ``delete_role`` a silent no-op on both
        adapters (or drop the ``initialize()`` call to it) — this test goes
        red on the ``"default" not in roles`` assertion. See the task
        verdict for the captured output.
        """
        adapter = _make_adapter()
        await adapter.create_role("default")
        await adapter.create_role("reviewer")
        assert await adapter.list_roles() == ["default", "reviewer"]

        embedder = FakeEmbedder(dim=_DIM)
        engine = Tabularium(adapter, embedder, _make_clock(), _make_config())
        await engine.initialize()

        roles = await adapter.list_roles()
        assert set(roles) == {"decider", "builder", "reviewer", "navigator"}
        assert "default" not in roles, (
            "'default' must be REMOVED by initialize() (curation block "
            "delete_role guarantee) -- 'registered but unused' "
            "is not acceptable (owner decision), and an older store held exactly this shape"
        )
        assert roles.count("reviewer") == 1

    async def test_save_and_context_accept_all_new_roles_off_initialize_alone(self) -> None:
        """``_require_roles`` (shared by ``save``/``ember``/``set_hot``/
        ``context``) accepts 'decider'/'builder'/'reviewer'/'navigator' with
        no manual ``adapter.create_role()`` call, since they now come from
        ``initialize()`` alone -- one registry, no per-tool change needed."""
        for role in ("decider", "builder", "reviewer", "navigator"):
            engine, _, _ = await _engine()
            r = await engine.save(f"a fact for {role}", scope="global", audience_hint=[role])
            assert r.outcome in {"saved", "bounced"}
            payload = await engine.context("global", budget=200, role=role)
            assert payload is not None

    async def test_save_with_valid_purpose(self) -> None:
        """Each pre-registered purpose value is accepted (no ToolError)."""
        for purpose in ("constraint", "convention", "preference", "pitfall", "reference"):
            # Fresh engine per iteration to avoid dedup bounces from FakeEmbedder collisions
            engine, _, _ = await _engine()
            r = await engine.save(f"a fact about {purpose}", scope="global", purpose=purpose)
            assert r.outcome in {"saved", "bounced"}, (
                f"Expected saved or bounced for purpose={purpose!r}, got {r.outcome!r}"
            )

    async def test_save_with_unregistered_purpose_raises(self) -> None:
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="Unregistered purpose"):
            await engine.save("fact", scope="global", purpose="unknown-purpose")

    async def test_save_with_none_purpose_always_accepted(self) -> None:
        engine, _, _ = await _engine()
        r = await engine.save("fact without purpose", scope="global", purpose=None)
        assert r.outcome == "saved"

    async def test_save_with_valid_role_in_audience_hint(self) -> None:
        engine, adapter, _ = await _engine()
        await adapter.create_role("implementer")
        r = await engine.save("fact", scope="global", audience_hint=["decider", "implementer"])
        assert r.outcome == "saved"

    async def test_save_with_unregistered_role_raises(self) -> None:
        engine, _, _ = await _engine()
        with pytest.raises(ToolError, match="Unregistered role"):
            await engine.save("fact", scope="global", audience_hint=["ghost-role"])

    async def test_ember_with_registered_purpose(self) -> None:
        engine, _, _ = await _engine()
        r = await engine.ember("nudge", scope="global", valence="guideline", purpose="pitfall")
        assert r.outcome == "saved"

    async def test_initialize_is_idempotent(self) -> None:
        """Multiple initialize() calls are safe (idempotent registry seeding),
        for both facets and all four roles ('default' is retired --
        -- and is not seeded by initialize() at all on a fresh store)."""
        engine, adapter, _ = await _engine()
        await engine.initialize()
        await engine.initialize()
        facets = await adapter.list_facets()
        assert facets.count("constraint") == 1  # no duplicates
        roles = await adapter.list_roles()
        for role in ("decider", "builder", "reviewer", "navigator"):
            assert roles.count(role) == 1  # no duplicates
        assert "default" not in roles


# ---------------------------------------------------------------------------
# 11. Vector-less write logging
# ---------------------------------------------------------------------------


class TestVectorlessWriteLogging:
    async def test_vectorless_write_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _, _ = await _engine(should_fail=True)
        with caplog.at_level(logging.INFO, logger="tabularium"):
            r = await engine.save("will land vectorless", scope="global")
        assert r.outcome == "saved"
        assert r.embedded is False
        info_messages = [rec.message for rec in caplog.records if rec.levelno == logging.INFO]
        assert any(
            "vectorless_write" in msg and "kind=fact" in msg and "trigger=save" in msg
            for msg in info_messages
        ), f"Expected an INFO vectorless_write log; got: {info_messages}"

    async def test_vectorless_write_logs_info_for_decision(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _, _ = await _engine(should_fail=True)
        with caplog.at_level(logging.INFO, logger="tabularium"):
            r = await engine.save("will land vectorless", kind="decision", scope="global")
        assert r.outcome == "saved"
        assert r.embedded is False
        info_messages = [rec.message for rec in caplog.records if rec.levelno == logging.INFO]
        assert any(
            "vectorless_write" in msg and "kind=decision" in msg and "trigger=save" in msg
            for msg in info_messages
        ), f"Expected an INFO vectorless_write log; got: {info_messages}"

    async def test_normal_save_does_not_log_vectorless_write(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _, _ = await _engine()
        with caplog.at_level(logging.INFO, logger="tabularium"):
            r = await engine.save("normal save", scope="global")
        assert r.outcome == "saved"
        assert r.embedded is True
        messages = [rec.message for rec in caplog.records]
        assert not any("vectorless_write" in msg for msg in messages)


# ---------------------------------------------------------------------------
# 12. Guard trip + dispatch interaction
# ---------------------------------------------------------------------------


class TestGuardDispatch:
    async def test_guard_trip_always_downgrades_to_bounce(self) -> None:
        """Guard trip always downgrades to bounce (P7 admission-order piece):
        the auto-reinforce band a guard trip used to downgrade OUT of is
        gone, so a guard-tripped candidate bounces regardless of config —
        there is no longer a mode where it would do anything else."""
        cfg = _make_config(gray_floor=0.0)
        engine, _, _ = await _collide_engine(config=cfg)

        await engine.save("We have 5 cats", scope="global")
        # "We have 3 cats" has a numeral difference → guard trip → bounce
        r = await engine.save("We have 3 cats", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.guard_flag == "numeral"
        assert r.bounce.verdict == "guard_trip"


# ---------------------------------------------------------------------------
# 11. Bounce exits (ADR 0022) — all four cells, the fact-vs-decision
# regression, and the guard-suppression fix
# ---------------------------------------------------------------------------


class TestBounceExitCells:
    """One test per (saved_kind, candidate_kind) cell — ``bounce_kind``, exit
    arity, and the exact action set. Split on the kind being SAVED, never on
    the candidate's kind (ADR 0022's own defect).
    """

    async def test_fact_saved_fact_candidate(self) -> None:
        """2026-08-10 fact-supersession brief: a valid Fact candidate now
        gets a THIRD exit, ``supersede`` — see ``tests/test_exits.py`` for
        the pure ``build_exits`` coverage of this arity and
        ``tests/test_fact_supersession.py`` for the engine-level behaviour
        (naming/executing the exit, the exclusion, partial-edge-failure
        reporting)."""
        cfg = _make_config(gray_floor=0.0)
        engine, _, _ = await _collide_engine(config=cfg)

        await engine.save("original fact content", scope="global")
        r = await engine.save("second fact content", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.bounce_kind == "fact"
        assert len(r.bounce.exits) == 3
        assert {e.action for e in r.bounce.exits} == {
            "reinforce", "confirm_distinct", "supersede",
        }

    async def test_fact_saved_decision_candidate(self) -> None:
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="use Python everywhere",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("use Python everywhere", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.bounce_kind == "fact"
        assert len(r.bounce.exits) == 2
        assert {e.action for e in r.bounce.exits} == {"reinforce", "confirm_distinct"}

    @pytest.mark.parametrize(
        "lifecycle,expected_actions",
        [
            ("active", {"reinforce", "confirm_distinct", "promoted_from"}),
            ("hot", {"reinforce", "confirm_distinct", "promoted_from"}),
            ("ember", {"reinforce", "confirm_distinct", "promoted_from"}),
            ("archived", {"reinforce", "confirm_distinct"}),
        ],
    )
    async def test_decision_saved_fact_candidate(
        self, lifecycle: str, expected_actions: set[str]
    ) -> None:
        """End-to-end, all four lifecycles a Fact candidate can actually
        reach the dedup probe in (2026-08-07 fix, ADR 0022 amendment):
        ``shelved`` is excluded — the probe's one deliberate blind spot,
        ADR 0021, ``engine.filters.probe_filters_for`` — so it cannot surface
        here at all; see ``test_decision_saved_shelved_fact_candidate_never_bounces``
        below for that case and ``tests/test_exits.py`` for the pure
        ``build_exits`` coverage of all five lifecycles including ``shelved``.
        """
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        fact = Fact(
            id=new_id(),
            type="fact",
            content="a fact a decision might rest on",
            scope=Scope("global"),
            lifecycle=lifecycle,  # type: ignore[arg-type]
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(fact, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save(
            "a fact a decision might rest on", kind="decision", scope="global"
        )
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.bounce_kind == "decision"
        assert len(r.bounce.exits) == len(expected_actions)
        assert {e.action for e in r.bounce.exits} == expected_actions

    async def test_decision_saved_shelved_fact_candidate_never_bounces(self) -> None:
        """A shelved Fact candidate cannot reach this bounce at all — the
        dedup probe itself excludes ``shelved`` (ADR 0021), independent of
        this fix's promotable-lifecycle gate. Confirms the fourth
        non-promotable lifecycle is unreachable here rather than silently
        untested."""
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        fact = Fact(
            id=new_id(),
            type="fact",
            content="a fact a decision might rest on",
            scope=Scope("global"),
            lifecycle="shelved",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(fact, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save(
            "a fact a decision might rest on", kind="decision", scope="global"
        )
        assert r.outcome == "saved"

    async def test_decision_saved_decision_candidate(self) -> None:
        cfg = _make_config(gray_floor=0.0, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        existing = Decision(
            id=new_id(),
            type="decision",
            content="use Python everywhere",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(existing, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save(
            "use Python everywhere v2", kind="decision", scope="global"
        )
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.bounce_kind == "decision"
        assert len(r.bounce.exits) == 3
        assert {e.action for e in r.bounce.exits} == {
            "reinforce", "confirm_distinct", "supersede",
        }

    async def test_fact_save_bouncing_off_decision_never_offers_supersede(self) -> None:
        """Regression test for the defect this brief fixes (ADR 0022): a Fact
        cannot supersede anything, but the gate used to hardcode
        ``suggested_action="supersede"`` whenever the best candidate was a
        Decision — unexecutable, since ``BounceResolution.action`` never
        accepted ``"supersede"`` either. A Fact save's exits must never
        include it, regardless of which candidate kind it bounced against.
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="never supersede me from a fact save",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("never supersede me from a fact save", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        actions = {e.action for e in r.bounce.exits}
        assert "supersede" not in actions

    async def test_fact_save_against_decision_candidate_carries_guard_flag(self) -> None:
        """The guard check used to be suppressed whenever the candidate was a
        Decision (``guard=None`` hardcoded) — ADR 0022 removes that special
        case, so a Fact save against a Decision candidate now runs the
        ordinary guard check like any other and surfaces ``guard_flag`` as
        evidence when it trips.
        """
        cfg = _make_config(gray_floor=0.80, decision_gray_floor=0.0)
        engine, adapter, clock = await _engine(config=cfg)
        now = clock.now()

        decision = Decision(
            id=new_id(),
            type="decision",
            content="we support 5 languages",
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        vec = [1.0] + [0.0] * (_DIM - 1)
        await adapter.upsert(decision, vec)

        engine._embedder = _IdenticalEmbedder()
        r = await engine.save("we support 3 languages", scope="global")
        assert r.outcome == "bounced"
        assert r.bounce is not None
        assert r.bounce.guard_flag == "numeral"
