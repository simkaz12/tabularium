"""Engine concurrency-control tests (storage-stabilization pass).

Covers the write-time concurrency defect: the dedup gate's check-then-write
window had no lock anywhere in the save/disposition/recall path, so two
cooperatively-interleaved callers on the single asyncio event loop could
both win a dedup probe and both write — proven unrepairable for the
same-predecessor Decision supersession case (an orphaned successor with a
birth-frozen lineage claim, but no predecessor pointer to it, and no data
model signal distinguishing it from the real tip).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``engine_ctx`` fixture, following the established pattern (see
``tests/test_engine_dispositions.py`` / ``tests/test_engine_b2.py``'s
``decision_engine_ctx``).

Falsifiability (non-negotiable per the concurrency-control brief): every
test below was proven sensitive to the defect by temporarily removing the
engine's write lock (a throwaway no-op lock swapped into
``Tabularium.__init__``), running this file, watching each test go red for
its stated reason, then restoring. Test 4 was additionally proven to catch
a *widened* lock (one that wraps the embed call) by temporarily moving the
``async with lock:`` in ``facts.save_internal`` to enclose the embed
step.

Determinism: every test drives its race via ``asyncio.gather``/
``asyncio.create_task`` plus explicit ``asyncio.Event``/deterministic
polling loops (bounded ``await asyncio.sleep(0)`` spins, or a genuine
``asyncio.wait_for`` timeout only where real backend I/O — LanceDB — is in
flight) rather than relying on incidental scheduling order. No test depends
on which of two concurrent coroutines happens to run first; where order
matters, it is forced by construction (an ``asyncio.Event`` gate) rather
than hoped for.
"""

from __future__ import annotations

import asyncio
import hashlib
import pathlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.engine.assessor import (
    AssessorCandidate,
    ContradictionVerdict,
    PanelDisagreement,
    RelationVerdict,
    Verdict,
)
from tabularium.results import BuryResult, SaveResult
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter

# Adapter-construction pattern reused verbatim from the conformance suite
# (precedent — see tests/test_engine_dispositions.py) rather than a
# second one.
from tests.conformance.test_conformance import (
    _flip,
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.cases import MaterialLabel
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import FaultInjectingEmbedder, as_embedder

AnyConcurrencyAdapter = MemoryAdapter | LanceDBAdapter
EngineCtx = tuple[Tabularium, AnyConcurrencyAdapter, FakeEmbedder, TestClock, str]

_DIM = 64  # matches test_engine_b2.py's choice: avoids sha256 slot collisions


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _build_engine_ctx(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path, cfg: EngineConfig
) -> EngineCtx:
    embedder = FakeEmbedder(dim=_DIM)
    clock = _make_clock()
    adapter_id = request.param
    adapter: AnyConcurrencyAdapter
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
    return engine, adapter, embedder, clock, adapter_id


@pytest.fixture(params=["memory", "lancedb"])
async def engine_ctx(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> EngineCtx:
    """Default gate posture:  (auto-reinforce unreachable,
    matches ``EngineConfig``'s own seed default) — near-duplicates bounce,
    they don't silently self-reinforce. Used by the tests that don't need a
    specific reinforce/auto-ceiling posture.
    """
    cfg = _make_config(decision_gray_floor=0.70)
    return await _build_engine_ctx(request, tmp_path, cfg)


@pytest.fixture(params=["memory", "lancedb"])
async def reinforcing_engine_ctx(
    request: pytest.FixtureRequest, tmp_path: pathlib.Path
) -> EngineCtx:
    """A low gray_floor — needed by the bury-vs-reinforce race test, which
    exercises the reinforce mutation path (``reinforce`` ->
    ``update_projections``) via the exact content-key match (P7
    admission-order piece, decision D13/D2; mirrors test_engine_m1.py's
    ``test_reinforced_outcome_via_exact_key`` pattern) rather than the
    bounce path. gray_floor is irrelevant to that step — it runs before
    the probe ever does — kept low only so the fixture stays reusable if
    a future case needs the probe to also see a wide-open floor.
    """
    cfg = _make_config(gray_floor=0.0)
    return await _build_engine_ctx(request, tmp_path, cfg)


def _slot(text: str, dim: int = _DIM) -> int:
    """Same hashing scheme as ``FakeEmbedder`` — lets fixture content be
    picked so distinct strings provably don't collide at ``dim=64``."""
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "big") % dim


class _GatedEmbedder:
    """Wraps a ``FakeEmbedder`` so ``embed()`` blocks on one designated
    "slow" text until released, and returns immediately for everything
    else. Lets a test observe "this coroutine is inside embed()" as a real,
    awaited state (``started``) instead of inferring it from timing, and
    release it at a moment the test controls.
    """

    def __init__(self, *, dim: int, slow_text: str) -> None:
        self._inner = FakeEmbedder(dim=dim)
        self._slow_text = slow_text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model(self) -> str:
        return "gated/" + self._inner.model

    async def embed(self, text: str) -> list[float]:
        if text == self._slow_text:
            self.started.set()
            await self.release.wait()
        return self._inner.peek(text)


# ---------------------------------------------------------------------------
# 1. Two concurrent saves of identical novel content -> exactly one record
# ---------------------------------------------------------------------------


class TestConcurrentIdenticalSaves:
    async def test_identical_content_writes_exactly_one_record(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 1 (brief).

        Two concurrent ``save()`` calls with byte-identical novel content:
        the correct outcome set is exactly one ``saved`` plus one LOSING
        outcome, never two ``saved`` (that would be the proven duplicate-
        write defect) and never two losing outcomes (that would mean
        neither call actually wrote).

        The loser's specific outcome depends on WHICH lock hold its
        interleaving lands in (P7 admission-order piece: the lock is taken
        twice per save — once for admission steps 1-3, released on a miss,
        then again for the probe and the terminal write):
        - if the loser's OWN admission check runs strictly after the
          winner's write has already landed, step 1 catches the byte-
          identical row directly — ``already_saved``.
        - if both calls' admission checks race the same empty-store
          snapshot (both miss) and only the second lock hold's probe
          catches the duplicate — ``bounced``, same as before this piece.
        Both are legitimate, non-duplicating outcomes; which one occurs is
        an adapter-scheduling detail (LanceDB's real I/O creates a wider
        race window than MemoryAdapter's yield-free path), not a property
        this test pins.

        Falsify: remove the engine's write lock -- both calls' dedup gates
        can race the same empty-store snapshot and both write, landing on
        outcomes={"saved"} (a set of size 1, two writes) with 2 rows in the
        store instead of 1.
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        content = "identical content for concurrent save test one"

        r1, r2 = await asyncio.gather(
            engine.save(content, scope="global"),
            engine.save(content, scope="global"),
        )

        outcomes = {r1.outcome, r2.outcome}
        assert outcomes in ({"saved", "bounced"}, {"saved", "already_saved"}), (
            f"[{adapter_id}] expected exactly one saved + one losing outcome, "
            f"got r1={r1.outcome!r} r2={r2.outcome!r}"
        )

        rows = [row async for row in await adapter.scan()]
        assert len(rows) == 1, (
            f"[{adapter_id}] expected exactly one record written, found {len(rows)}"
        )


class _BothGatedEmbedder:
    """Like ``_GatedEmbedder``, but blocks EVERY caller of the designated
    ``slow_text`` until ``release`` fires, and exposes ``both_started`` --
    an ``asyncio.Event`` set only once TWO callers are blocked inside
    ``embed()`` together. Lets a test force two concurrent saves to race
    the SECOND lock hold together, both having already crossed the FIRST
    hold as a miss."""

    def __init__(self, *, dim: int, slow_text: str) -> None:
        self._inner = FakeEmbedder(dim=dim)
        self._slow_text = slow_text
        self._entered = 0
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model(self) -> str:
        return "both-gated/" + self._inner.model

    async def embed(self, text: str) -> list[float]:
        if text == self._slow_text:
            self._entered += 1
            if self._entered >= 2:
                self.both_started.set()
            await self.release.wait()
        return self._inner.peek(text)


class TestConcurrentTokenlessSaves:
    """Regression: BM25's
    tokenizer (``MemoryAdapter._tokenize``, ``re.findall(r"[a-z0-9]+",
    ...)``) yields no tokens for punctuation-only content, so the serial
    probe's BM25 leg never returns a concurrent twin as a survivor -- the
    vector leg alone cannot save it (decision ``d15``: BM25 narrows the
    vector leg, it does not replace it). Steps 1/3 ran only inside the
    FIRST lock hold; two saves that both cross that hold as a miss (empty
    store) then both reach the serial probe with an empty survivor set and
    BOTH write -- two byte-identical rows, both ``outcome="saved"``.

    Fixed by re-running steps 1/3 inside the SECOND lock hold too, right
    before ``serial_probe`` (``facts._admission_steps_1_3`` /
    ``decisions._admission_steps_1_3_decision``).

    Falsify: comment out the second-hold re-check call in
    ``facts.save_internal`` and ``decisions.save_decision`` -- both tests
    below go red with ``outcomes == {"saved"}`` and 2 rows in the store;
    restore to go green again.
    """

    async def test_tokenless_content_dedups_on_second_hold(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        content = "!!! !!! !!!"

        gate = _BothGatedEmbedder(dim=_DIM, slow_text=content)
        engine._embedder = gate

        task1 = asyncio.create_task(engine.save(content, scope="global"))
        task2 = asyncio.create_task(engine.save(content, scope="global"))
        await asyncio.wait_for(gate.both_started.wait(), timeout=5.0)
        gate.release.set()

        r1, r2 = await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        outcomes = {r1.outcome, r2.outcome}
        assert outcomes == {"saved", "already_saved"}, (
            f"[{adapter_id}] expected exactly one saved + one already_saved "
            f"for a raced tokenless restatement, got r1={r1.outcome!r} "
            f"r2={r2.outcome!r}"
        )

        rows = [row async for row in await adapter.scan()]
        assert len(rows) == 1, (
            f"[{adapter_id}] expected exactly one record written, found {len(rows)}"
        )

    async def test_tokenizable_control_still_dedups(self, engine_ctx: EngineCtx) -> None:
        """Control: the same forced race, but content BM25 CAN tokenize --
        proves the defect above is specific to tokenless content, not a
        general hole in the race. With the second-hold re-check in place,
        this converges on the same shape as the tokenless case (the
        re-check's byte-identical replay step catches it before the
        serial probe ever runs), rather than the pre-fix path of getting
        caught later as a ``bounced`` survivor.
        """
        engine, adapter, _embedder, _clock, adapter_id = engine_ctx
        content = "identical tokenizable content for the concurrent control"

        gate = _BothGatedEmbedder(dim=_DIM, slow_text=content)
        engine._embedder = gate

        task1 = asyncio.create_task(engine.save(content, scope="global"))
        task2 = asyncio.create_task(engine.save(content, scope="global"))
        await asyncio.wait_for(gate.both_started.wait(), timeout=5.0)
        gate.release.set()

        r1, r2 = await asyncio.wait_for(asyncio.gather(task1, task2), timeout=5.0)

        outcomes = {r1.outcome, r2.outcome}
        assert outcomes == {"saved", "already_saved"}, (
            f"[{adapter_id}] expected exactly one saved + one already_saved "
            f"for a raced tokenizable restatement, got r1={r1.outcome!r} "
            f"r2={r2.outcome!r}"
        )

        rows = [row async for row in await adapter.scan()]
        assert len(rows) == 1, (
            f"[{adapter_id}] expected exactly one record written, found {len(rows)}"
        )


# ---------------------------------------------------------------------------
# 2. Two concurrent supersessions of the same predecessor -> no orphan
#    (the headline regression -- the probe that proved the defect)
# ---------------------------------------------------------------------------


class TestConcurrentSupersession:
    async def test_no_orphan_on_concurrent_supersede(self, engine_ctx: EngineCtx) -> None:
        """Case 2 (brief) -- THE headline regression.

        Two concurrent ``save(kind="decision", supersedes=X)`` calls
        targeting the SAME predecessor X, with orthogonal content (so
        neither successor can be mistaken for a near-duplicate of the
        other or of X): X cannot point ``superseded_by`` at both. Exactly
        one caller's successor must win (outcome="saved", X.superseded_by
        == winner.id); the other must fail cleanly (a ToolError, per
        ``_write_decision``'s documented contract) and its just-written
        Decision must be compensated (buried) -- never left live, in_force,
        and claiming lineage from X with no predecessor pointer to it.

        Falsify (empirically verified, not just reasoned about): defense in depth means EITHER
        Part 1 (this engine lock) OR Part 3 (``transition``'s adapter-level
        CAS) independently closes this specific race by itself -- removing
        only the lock (CAS intact) leaves this test green, and reverting
        only CAS (lock intact) also leaves it green. Reverting BOTH at once
        -- the lock (a no-op swapped into ``Tabularium.__init__``) AND CAS
        (``LanceDBAdapter.transition``'s ``where`` clause back to
        unconditional; ``MemoryAdapter.transition``'s post-yield re-check
        removed) -- reproduces the original probed defect exactly: both
        calls read X as not-yet-superseded before either writes, both
        successfully transition() X, and both return outcome="saved" (2
        live in_force Decisions, X pointing at only one). This is the
        headline regression's *true* minimal falsifying mutation; a
        lock-only removal does not reach it, which is the intended shape of
        defense in depth, not a gap in this test.
        """
        engine, adapter, embedder, clock, adapter_id = engine_ctx
        now = clock.now()

        predecessor_content = "Predecessor decision: use Python for the backend."
        succ_a_content = (
            "Successor A: use Python 3.12 with strict typing for the backend."
        )
        succ_b_content = "Successor B: switch the backend to Rust for performance."
        # Confirm the three fixture strings don't collide at this dim (would
        # make the test accidentally depend on hash luck rather than logic).
        slots = {_slot(t) for t in (predecessor_content, succ_a_content, succ_b_content)}
        assert len(slots) == 3, "fixture content must not share a FakeEmbedder hash slot"

        predecessor = Decision(
            id=new_id(),
            type="decision",
            content=predecessor_content,
            scope=Scope("global"),
            lifecycle="in_force",
            created_at=now,
            updated_at=now,
        )
        await adapter.upsert(predecessor, embedder.peek(predecessor_content))

        results = await asyncio.gather(
            engine.save(
                succ_a_content, kind="decision", scope="global", supersedes=predecessor.id,
                reason="correction",
            ),
            engine.save(
                succ_b_content, kind="decision", scope="global", supersedes=predecessor.id,
                reason="correction",
            ),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, SaveResult)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, (
            f"[{adapter_id}] expected exactly one successful supersession, "
            f"got {len(successes)} (results={results!r})"
        )
        assert len(failures) == 1, (
            f"[{adapter_id}] expected exactly one failed supersession, "
            f"got {len(failures)} (results={results!r})"
        )
        assert isinstance(failures[0], ToolError), (
            f"[{adapter_id}] the loser must fail cleanly with the documented "
            f"ToolError, got {failures[0]!r}"
        )

        winner = successes[0]
        assert winner.outcome == "saved"
        assert winner.hypomnema is not None

        predecessor_after = await adapter.get_by_id(predecessor.id)
        assert isinstance(predecessor_after, Decision)
        assert predecessor_after.lifecycle == "superseded"
        assert predecessor_after.superseded_by == winner.hypomnema.id

        # No orphan: exactly one live in_force Decision in the whole store
        # (the winner) -- the loser's just-written Decision was compensated
        # (buried), not left live and dangling.
        rows = [row async for row in await adapter.scan()]
        live_in_force = [
            r for r in rows if r.type == "decision" and r.lifecycle == "in_force"
        ]
        assert len(live_in_force) == 1, (
            f"[{adapter_id}] expected exactly one live in_force Decision, "
            f"found {len(live_in_force)}: {[str(r.id) for r in live_in_force]}"
        )
        assert live_in_force[0].id == winner.hypomnema.id
        # predecessor (now superseded) + winner (in_force) = 2 live rows;
        # the loser's compensated write must not appear at all.
        assert len(rows) == 2, (
            f"[{adapter_id}] expected exactly 2 live rows (predecessor + "
            f"winner), found {len(rows)}"
        )


# ---------------------------------------------------------------------------
# 3. Concurrent bury + save touching the same record -> no torn state
# ---------------------------------------------------------------------------


class TestBuryVersusSave:
    async def test_no_torn_state_on_concurrent_bury_and_reinforce(
        self, reinforcing_engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case 3 (brief).

        A second save of a WHITESPACE VARIANT of the first's content shares
        its canonical content key, so the exact-key match (P7
        admission-order piece, step 3) reinforces the first (``reinforce``
        -> ``adapter.update_projections``) rather than bouncing -- the
        mutating path, which is the one a concurrent bury can actually tear.
        A byte-identical resubmission would instead ack ``already_saved``
        from step 1, one lock hold earlier, never reaching this path.

        Determinism: ``adapter.update_projections`` -- the terminal write of
        ``reinforce``, called from inside the save's locked critical
        section -- is gated on an event. The test lets the save's reinforce
        reach exactly that point, THEN starts the concurrent bury, confirms
        bury makes NO progress while gated (proving the lock actually
        serializes bury against save), THEN releases. This forces one
        specific, checkable interleaving on both adapters rather than
        hoping natural scheduling order lands there.

        Falsify: remove the engine's write lock -- bury no longer blocks
        behind the in-flight reinforce; it runs to completion (deletes the
        row, writes the tombstone) *while update_projections is still
        gated*. On release, the real ``update_projections`` -- which does
        its own fresh existence read before writing, on both adapters --
        finds the row gone and raises a bare, undocumented ``KeyError`` out
        of ``engine.save()`` instead of the clean, typed outcome any caller
        is contracted to see.
        """
        engine, adapter, _embedder, _clock, adapter_id = reinforcing_engine_ctx

        content = "fact for concurrent bury-vs-reinforce race"
        r0 = await engine.save(content, scope="global")
        assert r0.outcome == "saved"
        assert r0.hypomnema is not None
        fid = r0.hypomnema.id

        real_update_projections = adapter.update_projections
        gate_started = asyncio.Event()
        gate_release = asyncio.Event()

        async def _gated_update_projections(id: Any, **kw: Any) -> None:
            gate_started.set()
            await gate_release.wait()
            await real_update_projections(id, **kw)

        monkeypatch.setattr(adapter, "update_projections", _gated_update_projections)

        save_task = asyncio.create_task(engine.save(content + " ", scope="global"))
        await asyncio.wait_for(gate_started.wait(), timeout=5.0)

        bury_task = asyncio.create_task(engine.bury(fid, reason="cleanup"))
        # Give bury every chance to run concurrently while the reinforce is
        # gated mid critical-section -- under the fix it must be unable to
        # make any progress at all (blocked on the shared lock). A real
        # (short) wall-clock wait, not a bare sleep(0) spin: LanceDB's bury
        # does genuine disk I/O, which needs real time to complete, not just
        # callback-queue turns -- a sleep(0) loop would pass here for the
        # wrong reason (not enough real time elapsed) rather than the right
        # one (genuinely blocked on the lock). Under the fix this wait
        # always exhausts its budget, since bury is blocked indefinitely
        # (until the gate below releases it), not merely slow.
        try:
            await asyncio.wait_for(asyncio.shield(bury_task), timeout=0.5)
        except TimeoutError:
            pass  # expected: bury made no progress at all
        else:
            pytest.fail(
                f"[{adapter_id}] bury completed while a concurrent save's "
                "reinforce was still inside its locked critical section -- "
                "the shared write lock is not serializing bury against save"
            )

        gate_release.set()
        save_result, bury_result = await asyncio.gather(
            save_task, bury_task, return_exceptions=True
        )

        for label, result in (("save", save_result), ("bury", bury_result)):
            assert not isinstance(result, BaseException), (
                f"[{adapter_id}] {label} raised an undocumented exception "
                f"under the race: {result!r}"
            )

        assert isinstance(save_result, SaveResult)
        assert isinstance(bury_result, BuryResult)
        # Forced order (bury cannot start until save's locked section
        # exits): the reinforce always completes cleanly first.
        assert save_result.outcome == "reinforced"
        assert save_result.reinforced_id == fid
        assert bury_result.outcome == "buried"

        # No torn state: the record is cleanly, fully buried afterward --
        # not half-updated, not still discoverable.
        with pytest.raises(KeyError):
            await adapter.get_by_id(fid)
        rows = [row async for row in await adapter.scan()]
        assert fid not in {r.id for r in rows}


# ---------------------------------------------------------------------------
# 4. The lock must NOT span the embed call
# ---------------------------------------------------------------------------


class TestLockExcludesEmbed:
    async def test_second_save_proceeds_while_first_is_still_embedding(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 4 (brief).

        A stub embedder blocks ``embed()`` for one designated ("slow")
        save's content until the test releases it; a second ("fast") save
        with different content must be able to reach and clear the lock,
        and complete, WHILE the first is still stuck inside its embed call
        -- proving the lock is acquired strictly after embed, never around
        it.

        Determinism: driven entirely by ``asyncio.Event`` (the gate's
        ``started``) plus a genuine ``asyncio.wait_for`` timeout for the
        fast save's completion -- no fixed sleep is used to "probably" let
        the fast save finish. The timeout (5s) is generous enough that any
        real local dedup-gate work (including LanceDB disk I/O) completes
        well inside it under the correct implementation; under the
        mutation described below it never completes at all, so the test
        fails deterministically rather than flaking.

        Falsify: widen the lock to also cover the embed call (acquire it
        BEFORE ``await embedder.embed(content)`` instead of after) -- the
        slow save now holds the lock while stuck inside its gated embed;
        the fast save's own (instant) embed finishes but it then blocks
        trying to acquire the SAME lock, so it cannot complete within the
        timeout and this test times out / fails.
        """
        engine, _adapter, _embedder, _clock, adapter_id = engine_ctx
        slow_text = "slow save content for lock-does-not-span-embed test"
        fast_text = "fast save content for lock-does-not-span-embed test"
        assert _slot(slow_text) != _slot(fast_text), (
            "fixture content must not share a FakeEmbedder hash slot"
        )

        gate = _GatedEmbedder(dim=_DIM, slow_text=slow_text)
        engine._embedder = gate

        slow_task = asyncio.create_task(engine.save(slow_text, scope="global"))
        try:
            await asyncio.wait_for(gate.started.wait(), timeout=5.0)

            fast_task = asyncio.create_task(engine.save(fast_text, scope="global"))
            try:
                await asyncio.wait_for(fast_task, timeout=5.0)
            except TimeoutError:
                fast_task.cancel()
                pytest.fail(
                    f"[{adapter_id}] the fast save did not complete within 5s "
                    "while the slow save was still inside embed() -- the "
                    "lock must be spanning the embed call (see this test's "
                    "docstring)."
                )

            fast_result = fast_task.result()
            assert fast_result.outcome in {
                "saved", "bounced", "reinforced", "revived", "reconciled",
            }
        finally:
            gate.release.set()
            slow_result = await slow_task
            assert slow_result.outcome in {
                "saved", "bounced", "reinforced", "revived", "reconciled",
            }


# ---------------------------------------------------------------------------
# 5. Additional coverage (beyond the four required cases): the adapter-level
#    CAS on transition()'s supersede edge (Part 3, defense in depth).
# ---------------------------------------------------------------------------


class TestAdapterLevelSupersessionCAS:
    """Calls ``adapter.replace()`` DIRECTLY and concurrently -- bypassing
    the engine's write lock entirely (going through ``engine.save()`` would
    fully serialize the two calls, per the lock closing the in-process race
    at the engine layer, so this CAS branch would never actually be
    exercised through the public API). This is what proves ``replace``'s
    fresh-read-under-lock discipline on its own terms: a cross-process (or
    any other out-of-lock) writer racing ``replace`` on the supersede edge
    still can't produce two winners, on either adapter.

    Falsify: build the predecessor's write in ``replace`` from a snapshot
    taken before ``self._supersede_lock``/the yield point rather than a
    fresh in-lock re-read -- both writers "win" and ``get_by_id`` returns
    whichever committed last, silently overwriting the other's
    ``superseded_by`` claim with no error raised at all.
    """

    async def test_concurrent_direct_replace_calls_one_wins(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, adapter, embedder, clock, adapter_id = engine_ctx
        del engine  # unused: this test bypasses the engine entirely by design
        now = clock.now()

        predecessor_content = "CAS predecessor: adopt the seam interface."
        succ_a_content = "CAS successor A: adopt the seam interface, revision one."
        succ_b_content = "CAS successor B: replace the seam interface entirely."
        slots = {_slot(t) for t in (predecessor_content, succ_a_content, succ_b_content)}
        assert len(slots) == 3, "fixture content must not share a FakeEmbedder hash slot"

        predecessor = Decision(
            id=new_id(), type="decision", content=predecessor_content,
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        succ_a = Decision(
            id=new_id(), type="decision", content=succ_a_content,
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        succ_b = Decision(
            id=new_id(), type="decision", content=succ_b_content,
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(predecessor, embedder.peek(predecessor_content))
        await adapter.upsert(succ_a, embedder.peek(succ_a_content))
        await adapter.upsert(succ_b, embedder.peek(succ_b_content))

        results = await asyncio.gather(
            adapter.replace(succ_a, None, [_flip(predecessor.id)], at=now),
            adapter.replace(succ_b, None, [_flip(predecessor.id)], at=now),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, BaseException)]
        failures = [r for r in results if isinstance(r, BaseException)]
        assert len(successes) == 1, (
            f"[{adapter_id}] expected exactly one winning transition, "
            f"got {len(successes)} (results={results!r})"
        )
        assert len(failures) == 1, (
            f"[{adapter_id}] expected exactly one losing transition, "
            f"got {len(failures)} (results={results!r})"
        )
        assert isinstance(failures[0], ValueError), (
            f"[{adapter_id}] the loser must raise ValueError (the documented "
            f"set-exactly-once contract), got {failures[0]!r}"
        )

        updated = await adapter.get_by_id(predecessor.id)
        assert isinstance(updated, Decision)
        assert updated.superseded_by in {succ_a.id, succ_b.id}


# ---------------------------------------------------------------------------
# 6. Additional coverage (found while implementing Part 3, same scope): the
#    CAS predicate on transition()'s supersede edge must assert BOTH halves
#    of the precondition the app-level check assumed -- superseded_by IS
#    NULL is not sufficient on its own.
# ---------------------------------------------------------------------------


class TestConcurrentRevokeVersusSupersede:
    async def test_no_torn_state_on_concurrent_revoke_and_supersede(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Found while implementing Part 3: a concurrent ``revoke(X)`` and
        ``save(supersedes=X)`` race ``transition()``'s precondition too, not
        only the same-predecessor double-supersede case (test 2 above). A
        CAS predicate checking ONLY ``superseded_by IS NULL`` lets a revoked
        Decision be silently flipped to "superseded" -- ``revoke`` moves
        ``lifecycle`` to ``"revoked"`` without touching ``superseded_by``,
        so that half of the precondition alone is blind to it. Both Decision
        edges must originate at ``in_force`` (SPEC S1.2). Fixed by also
        asserting ``lifecycle == "in_force"`` in the CAS predicate (both
        adapters) and its pre-yield/pre-write mirror check.

        Exactly one of {revoke, save} must win; the other must fail cleanly
        (``ToolError``) -- and the predecessor's final state (revoked XOR
        superseded, never a superseded Decision with no revoke record, nor
        a revoked Decision with ``superseded_by`` set) must match whichever
        one won.

        Falsify: revert the ``lifecycle`` half of the CAS predicate on both
        adapters (keep only ``superseded_by IS NULL``) -- when save's
        locked ``transition()`` call runs strictly after a revoke that
        already completed, it no longer detects the conflict: the
        predecessor ends up ``lifecycle="superseded"`` with
        ``superseded_by`` set despite having already been revoked, an
        illegal, torn state no data-model invariant should allow.
        """
        engine, adapter, embedder, clock, adapter_id = engine_ctx
        now = clock.now()

        predecessor_content = "Predecessor for revoke-vs-supersede race."
        succ_content = "Successor for revoke-vs-supersede race."
        assert _slot(predecessor_content) != _slot(succ_content), (
            "fixture content must not share a FakeEmbedder hash slot"
        )

        predecessor = Decision(
            id=new_id(), type="decision", content=predecessor_content,
            scope=Scope("global"), lifecycle="in_force", created_at=now, updated_at=now,
        )
        await adapter.upsert(predecessor, embedder.peek(predecessor_content))

        revoke_result, save_result = await asyncio.gather(
            engine.revoke(predecessor.id, reason="race test"),
            engine.save(
                succ_content, kind="decision", scope="global", supersedes=predecessor.id,
                reason="correction",
            ),
            return_exceptions=True,
        )

        final = await adapter.get_by_id(predecessor.id)
        assert isinstance(final, Decision)

        revoke_ok = not isinstance(revoke_result, BaseException)
        save_ok = not isinstance(save_result, BaseException)
        assert revoke_ok != save_ok, (
            f"[{adapter_id}] exactly one of revoke/save must win, got "
            f"revoke_ok={revoke_ok} save_ok={save_ok} "
            f"(revoke={revoke_result!r}, save={save_result!r})"
        )

        if revoke_ok:
            assert final.lifecycle == "revoked"
            assert final.superseded_by is None
            assert isinstance(save_result, ToolError), (
                f"[{adapter_id}] the losing save must fail with a ToolError, "
                f"got {save_result!r}"
            )
        else:
            assert final.lifecycle == "superseded"
            assert isinstance(save_result, SaveResult)
            assert save_result.outcome == "saved"
            assert save_result.hypomnema is not None
            assert final.superseded_by == save_result.hypomnema.id
            assert isinstance(revoke_result, ToolError), (
                f"[{adapter_id}] the losing revoke must fail with a "
                f"ToolError, got {revoke_result!r}"
            )


# ---------------------------------------------------------------------------
# 7. Recall's fallback-hit revival versus a concurrent bury on the same id
#    ``recall``'s revival loop captures its candidate
#    ``rec`` during the UNLOCKED search phase above (multiple real await
#    points: the primary search, then the archived-fallback search that
#    surfaces this candidate in the first place), then only takes the
#    engine's write lock for the mutation. A concurrent ``bury`` landing in
#    that window (which shares the same lock, but runs and releases it
#    BEFORE recall ever reaches it) used to reach ``reinforce`` ->
#    ``adapter.update_projections``, which raises a bare ``KeyError`` that
#    nothing in the revival loop caught -- escaping all the way out of
#    ``engine.recall()`` (``server.py``'s recall handler only catches
#    ``(EngineToolError, ValueError)``, so this would reach the MCP client
#    as an unhandled exception).
# ---------------------------------------------------------------------------


class TestRecallRevivalVersusConcurrentBury:
    async def test_no_uncaught_keyerror_when_bury_races_fallback_revival(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single archived Fact is the only record in the store, so
        recall's primary (active/hot/ember) pass returns 0 results --- below
        ``min_active_results`` (default 3) --- forcing the fallback pass
        that discovers it. ``adapter.text_search`` is wrapped so that,
        specifically for the fallback pass's archived-only filter (never the
        primary pass, which passes straight through), it pauses AFTER
        computing its real result but BEFORE returning -- forcing the
        recall task to sit exactly between "unlocked search complete" and
        "about to enter the locked revival section" while a concurrent
        ``bury`` of that same record runs to completion (bury does not call
        ``text_search``/``vector_search`` at all, so it is never itself
        gated by this wrapper).

        Determinism: driven entirely by ``asyncio.Event`` (the gate's
        ``fallback_search_done``) plus a genuine ``asyncio.wait_for``
        timeout -- no fixed sleep. The bury is fully awaited (not just
        started) before the gate is released, so by the time recall resumes
        and takes the lock, the row is provably already gone -- not a race
        on whether bury *happens* to finish first.

        Design decision (per the brief): a concurrently-buried hit is
        DROPPED from recall's returned results, not returned with its
        stale pre-bury content -- the row is destroyed by the time the
        locked revival section observes it, so returning content that no
        longer exists anywhere in the store would be misleading (a
        follow-up ``get_by_id``/``recall`` on it would 404).

        Falsify: revert the ``get_by_id`` re-validation this fix added to
        ``recall.py``'s revival loop (restore the direct
        ``await reinforce(..., record=rec, ...)`` call using the stale
        pre-lock ``rec``) -- this test goes red with an uncaught
        ``KeyError`` propagating out of ``engine.recall()``, for exactly the
        reason described above, not for an unrelated reason (import error,
        missing attribute, etc).
        """
        engine, adapter, embedder, clock, adapter_id = engine_ctx

        content = "archived fact for concurrent bury during recall revival test"
        now = clock.now()
        archived = Fact(
            id=new_id(), type="fact", content=content,
            scope=Scope("global"), lifecycle="archived", created_at=now, updated_at=now,
        )
        await adapter.upsert(archived, embedder.peek(content))

        real_text_search = adapter.text_search
        fallback_search_done = asyncio.Event()
        may_proceed = asyncio.Event()

        async def _gated_text_search(query: str, *, filters: Any, k: int) -> Any:
            result = await real_text_search(query, filters=filters, k=k)
            if filters.lifecycle_in == frozenset({"archived"}):
                fallback_search_done.set()
                await may_proceed.wait()
            return result

        monkeypatch.setattr(adapter, "text_search", _gated_text_search)

        recall_task = asyncio.create_task(engine.recall(content, scope="global"))
        await asyncio.wait_for(fallback_search_done.wait(), timeout=5.0)

        bury_result = await engine.bury(
            archived.id, reason="concurrent bury during recall revival test"
        )
        assert bury_result.outcome == "buried"

        may_proceed.set()
        recall_result = await asyncio.wait_for(recall_task, timeout=5.0)

        # No uncaught KeyError -- recall completed cleanly and returned a
        # normal RecallResult (asyncio.wait_for would re-raise otherwise).
        assert recall_result.degraded is False

        # The buried record must not appear in the returned hits -- its
        # content no longer exists anywhere in the store by the time recall
        # observes the bury.
        assert archived.id not in {h.id for h in recall_result.hits}, (
            f"[{adapter_id}] a concurrently-buried hit must be dropped from "
            f"recall's results, not returned with stale content"
        )

        # No torn state: the record stays buried, and gained no revival
        # event after the fact (the revival loop must skip it entirely, not
        # partially apply the mutation).
        with pytest.raises(KeyError):
            await adapter.get_by_id(archived.id)
        revival_events = await adapter.query_events(target_id=archived.id, source="revival")
        assert revival_events == [], (
            f"[{adapter_id}] a buried record must not gain a revival event "
            f"after the fact; found {revival_events!r}"
        )

    async def test_no_uncaught_keyerror_when_bury_races_invalid_archived_hit(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-lock short-circuit
        that skipped invalid hits ran BEFORE the fresh under-lock
        ``get_by_id`` re-read, so it never fired for an INVALID archived hit
        (only reachable via ``include_invalid=True``) -- a concurrently
        buried invalid hit would bypass the fresh read entirely and still be
        returned with its now-destroyed content, and an invalid->valid
        repair racing the same window would be returned as stale-invalid.
        Fixed by making the fresh ``get_by_id`` re-read unconditional for
        every candidate id, before any validity/lifecycle decision.

        Same gating shape as ``test_no_uncaught_keyerror_when_bury_races_
        fallback_revival`` above, but the record is ``validity="invalid"``
        and the call passes ``include_invalid=True`` -- otherwise this
        record would never surface at all and the invalid-specific code
        path this test targets would never run.

        Falsify: move the ``if not isinstance(rec, Fact): continue`` /
        ``if rec.validity != "valid": continue`` pre-lock checks back in
        front of the ``get_by_id`` re-read (as they were before this fix) --
        this test goes red because the buried record is still present in
        ``recall_result.hits`` with its pre-bury content, not for an
        unrelated reason.
        """
        engine, adapter, embedder, clock, adapter_id = engine_ctx

        content = "invalid archived fact for concurrent bury during recall"
        now = clock.now()
        archived_invalid = Fact(
            id=new_id(), type="fact", content=content, validity="invalid",
            scope=Scope("global"), lifecycle="archived", created_at=now, updated_at=now,
        )
        await adapter.upsert(archived_invalid, embedder.peek(content))

        real_text_search = adapter.text_search
        fallback_search_done = asyncio.Event()
        may_proceed = asyncio.Event()

        async def _gated_text_search(query: str, *, filters: Any, k: int) -> Any:
            result = await real_text_search(query, filters=filters, k=k)
            if filters.lifecycle_in == frozenset({"archived"}):
                fallback_search_done.set()
                await may_proceed.wait()
            return result

        monkeypatch.setattr(adapter, "text_search", _gated_text_search)

        recall_task = asyncio.create_task(
            engine.recall(content, scope="global", include_invalid=True)
        )
        await asyncio.wait_for(fallback_search_done.wait(), timeout=5.0)

        bury_result = await engine.bury(
            archived_invalid.id, reason="concurrent bury during recall (invalid hit) test"
        )
        assert bury_result.outcome == "buried"

        may_proceed.set()
        recall_result = await asyncio.wait_for(recall_task, timeout=5.0)
        assert recall_result.degraded is False

        assert archived_invalid.id not in {h.id for h in recall_result.hits}, (
            f"[{adapter_id}] a concurrently-buried invalid hit must be dropped "
            f"from recall's results, not returned with stale/destroyed content"
        )
        with pytest.raises(KeyError):
            await adapter.get_by_id(archived_invalid.id)
        revival_events = await adapter.query_events(
            target_id=archived_invalid.id, source="revival"
        )
        assert revival_events == [], (
            f"[{adapter_id}] an invalid, buried record must never gain a "
            f"revival event; found {revival_events!r}"
        )


# ---------------------------------------------------------------------------
# 8. Recall's fallback-hit revival versus a concurrent revival of the SAME
#    archived record by a different caller (re-validating existence alone is not enough -- a hit
# whose
#    lifecycle already moved off "archived" between the unlocked search and
#    the lock must not be reinforced/transitioned a second time, and the
#    stale pre-lock snapshot must not be what recall returns).
# ---------------------------------------------------------------------------


class TestRecallRevivalVersusConcurrentRevival:
    async def test_no_double_revival_when_save_races_fallback_revival(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same gating shape as the bury-race test above, but the concurrent
        operation is a plain ``save()`` of matching content, which reaches
        ``_dispatch_match``'s own archived-revival branch (cosine 1.0, well
        above ``gray_floor``) and revives the record (``archived``->
        ``active``, one ``source="revival"`` event) WHILE recall is paused
        between its unlocked fallback search and the locked revival section.

        Once recall resumes and re-reads the record fresh under the lock, it
        must recognise the record is no longer archived and skip its own
        revival mutation entirely -- not emit a second, redundant
        ``source="revival"`` event, and not return the hit with the STALE
        pre-lock ``lifecycle="archived"`` when the store's real, current
        state is already ``active``.

        Falsify: revert the ``live_rec.lifecycle != "archived"`` re-check
        this fix added (fall straight through to ``reinforce`` /
        conditional ``transition`` using only the existence + validity + Fact
        re-checks) -- this test goes red on the revival-event-count
        assertion (a bogus second event) and/or the returned-hit-lifecycle
        assertion (stale "archived"), not for an unrelated reason.
        """
        engine, adapter, embedder, clock, adapter_id = engine_ctx

        content = "archived fact revived by a racing save during recall"
        now = clock.now()
        archived = Fact(
            id=new_id(), type="fact", content=content,
            scope=Scope("global"), lifecycle="archived", created_at=now, updated_at=now,
        )
        await adapter.upsert(archived, embedder.peek(content))

        real_text_search = adapter.text_search
        fallback_search_done = asyncio.Event()
        may_proceed = asyncio.Event()

        async def _gated_text_search(query: str, *, filters: Any, k: int) -> Any:
            result = await real_text_search(query, filters=filters, k=k)
            if filters.lifecycle_in == frozenset({"archived"}):
                fallback_search_done.set()
                await may_proceed.wait()
            return result

        monkeypatch.setattr(adapter, "text_search", _gated_text_search)

        recall_task = asyncio.create_task(engine.recall(content, scope="global"))
        await asyncio.wait_for(fallback_search_done.wait(), timeout=5.0)

        # Concurrent save of byte-identical content (P7 admission-order
        # piece): admission step 1 catches the byte-identical replay
        # directly and revives the archived match in place before acking
        # ``already_saved`` -- same revival mutation ``_dispatch_match``'s
        # archived branch used to reach via the probe, one lock hold
        # earlier now.
        save_result = await engine.save(content, scope="global")
        assert save_result.outcome == "already_saved"
        assert save_result.hypomnema is not None
        assert save_result.hypomnema.id == archived.id

        revived_mid_race = await adapter.get_by_id(archived.id)
        assert isinstance(revived_mid_race, Fact)
        assert revived_mid_race.lifecycle == "active", (
            "test setup: the racing save must have already revived the "
            "record before recall resumes"
        )

        may_proceed.set()
        recall_result = await asyncio.wait_for(recall_task, timeout=5.0)
        assert recall_result.degraded is False

        # Exactly ONE revival event -- from the racing save's own dedup
        # revival, never a second one from recall's own (stale) discovery.
        revival_events = await adapter.query_events(target_id=archived.id, source="revival")
        assert len(revival_events) == 1, (
            f"[{adapter_id}] expected exactly one revival event (from the "
            f"racing save), found {len(revival_events)} -- recall must not "
            f"reinforce a record it no longer finds archived"
        )

        # The store's real current lifecycle.
        final_rec = await adapter.get_by_id(archived.id)
        assert isinstance(final_rec, Fact)
        assert final_rec.lifecycle == "active"

        # recall's returned hit (if present) must reflect that current
        # reality, never the stale pre-lock "archived" snapshot.
        hit = next((h for h in recall_result.hits if h.id == archived.id), None)
        if hit is not None:
            assert hit.lifecycle == "active", (
                f"[{adapter_id}] recall returned a stale lifecycle for a "
                f"record concurrently revived by another caller: "
                f"got {hit.lifecycle!r}, store has 'active'"
            )


# ---------------------------------------------------------------------------
# 6. Outage-helper lock handling: the Assessor call on the outage bounce path must run
#    strictly OUTSIDE any write-lock hold, and that must survive real task
#    cancellation under stress, not merely the absence of a bug in the
#    common case.
# ---------------------------------------------------------------------------


class _OwnershipLock(asyncio.Lock):
    """An ``asyncio.Lock`` that tracks WHICH task holds it, so a task that
    releases a lock it does not currently hold (the exact defect:
    a cancelled reacquire leaves the ``async with lock:`` context manager's
    own cleanup releasing whatever OTHER task grabbed the lock in the
    meantime) is recorded as a violation instead of silently corrupting the
    next holder's critical section.

    ``max_concurrent`` derives from the SAME acquire/release bookkeeping:
    under the round-2 defect, the interloper's wrongful early release can
    wake a THIRD task while the true holder is still running, which shows
    up here as a count above 1 -- exactly the "max concurrent lock holders"
    property P7's brief names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.holder: asyncio.Task[Any] | None = None
        self._active = 0
        self.max_concurrent = 0
        self.violations: list[str] = []

    async def acquire(self) -> Literal[True]:
        await super().acquire()
        current = asyncio.current_task()
        if self.holder is not None:
            self.violations.append(
                f"acquire() succeeded while holder={self.holder!r} was still "
                f"set (current={current!r}) -- two tasks both think they hold "
                "the lock"
            )
        self.holder = current
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        return True

    def release(self) -> None:
        current = asyncio.current_task()
        if self.holder is not current:
            self.violations.append(
                f"release() by non-holder: holder={self.holder!r}, "
                f"releaser={current!r}"
            )
        else:
            self.holder = None
        self._active -= 1
        super().release()


class _LockCheckingAssessor:
    """Wraps ``EchoAssessor``, recording whether the engine's write lock was
    held at the moment ``relation()`` was called. D12 requires the Assessor
    to run entirely outside any lock hold; an earlier shape
    called it from INSIDE one, or manually released and reacquired the SAME
    lock around the call."""

    def __init__(self, label: MaterialLabel, lock: asyncio.Lock) -> None:
        self._inner = EchoAssessor(label)
        self._lock = lock
        self.lock_states: list[bool] = []

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.lock_states.append(self._lock.locked())
        await asyncio.sleep(0)
        return await self._inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        return await self._inner.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        return await self._inner.worth(text)

    async def perishable(self, text: str) -> Verdict:
        return await self._inner.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        return await self._inner.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        return await self._inner.scope(text, labels)

    async def relevance(self, text: str) -> Verdict:
        return await self._inner.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._inner.contradiction(existing, incoming, panel=panel)


class TestOutageAssessorLockSafety:
    async def test_assessor_never_called_while_lock_held(self, engine_ctx: EngineCtx) -> None:
        """Deterministic: a single outage bounce, no cancellation needed --
        the Assessor call in the shipped code always sits strictly after
        the second ``async with lock:`` block exits, so ``lock.locked()``
        is always ``False`` at call time on every run.

        Falsify: move the Assessor-call block (``facts.py``, "Assessor
        call [D12]: outside the lock") back inside the second ``async with
        lock:`` hold -- this test's ``all(state is False ...)`` assertion
        goes red (records ``True``) for the same reason holding the global write lock around the
        outage assessment would.
        """
        engine, _adapter, _embedder, _clock, adapter_id = engine_ctx
        seed_text = "alpha bravo charlie delta echo golf hotel"
        r0 = await engine.save(seed_text, scope="global")
        assert r0.outcome == "saved"

        wrapper = FaultInjectingEmbedder(engine._embedder)
        engine._embedder = as_embedder(wrapper)
        checking_assessor = _LockCheckingAssessor("repetition", engine._write_lock)
        engine._assessor = checking_assessor
        wrapper.arm()

        r1 = await engine.save(seed_text + " india", scope="global")

        assert r1.outcome == "bounced", f"[{adapter_id}] expected an outage bounce, got {r1}"
        assert checking_assessor.lock_states, f"[{adapter_id}] assessor was never called"
        assert all(state is False for state in checking_assessor.lock_states), (
            f"[{adapter_id}] Assessor was called while the write lock was held: "
            f"{checking_assessor.lock_states}"
        )


class _GatedAssessor:
    """Blocks ``relation()`` on an event until released — lets a test park a
    save exactly at its Assessor call and control resumption timing (same
    ``started``/``release`` shape as this file's own ``_GatedEmbedder``)."""

    def __init__(self, label: MaterialLabel) -> None:
        self._inner = EchoAssessor(label)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def relation(
        self,
        incoming: str,
        candidate: AssessorCandidate,
        *,
        similarity: float | None,
        neighbours: Sequence[AssessorCandidate] | None = None,
    ) -> RelationVerdict:
        self.started.set()
        await self.release.wait()
        return await self._inner.relation(incoming, candidate, similarity=similarity)

    async def kind(self, text: str, declared_kind: str) -> Verdict:
        return await self._inner.kind(text, declared_kind)

    async def worth(
        self, text: str, *, neighbours: Sequence[AssessorCandidate] | None = None
    ) -> Verdict:
        return await self._inner.worth(text)

    async def perishable(self, text: str) -> Verdict:
        return await self._inner.perishable(text)

    async def breadth(self, text: str, source_scope: str) -> Verdict:
        return await self._inner.breadth(text, source_scope)

    async def scope(self, text: str, labels: Sequence[tuple[str, str | None]]) -> Verdict:
        return await self._inner.scope(text, labels)

    async def relevance(self, text: str) -> Verdict:
        return await self._inner.relevance(text)

    async def contradiction(
        self, existing: str, incoming: str, *, panel: PanelDisagreement | None = None
    ) -> ContradictionVerdict:
        return await self._inner.contradiction(existing, incoming, panel=panel)


class TestOutageCancellationLockOwnershipStress:
    async def test_cancellation_during_reacquisition_cannot_release_another_holders_lock(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Deterministic construction of the exact race, not blind
        stress: the original shape at the outage Assessor call site did
        ``lock.release()`` before the outage Assessor call and ``await
        lock.acquire()`` in a ``finally`` before the caller's own ``async
        with lock:`` exited. A save cancelled while blocked on that
        reacquire, with a DIFFERENT holder currently on the lock, made the
        caller's own ``async with lock:`` cleanup call ``release()`` on a
        lock it did not hold.

        A blind 30-task stress with scattered cancellation cannot be
        trusted to land in this exact window on every run — verified
        empirically: with the original shape reintroduced by hand, six
        consecutive runs of such a stress test (30 tasks, cancellation
        staggered over 0-4 scheduling yields) stayed green, because
        ``MemoryAdapter``'s operations rarely suspend control back to the
        loop outside the deliberately-placed gates. This test forces the
        window by construction instead: the SECOND lock holder is the
        test's own coroutine, acquired directly on the shared
        ``_OwnershipLock``, not a second ``engine.save()`` call racing this
        one's exact timing.

        Falsify: reintroduce the original
        release/reacquire shape around the outage Assessor call in
        ``facts.py`` (nesting the whole "Assessor call [D12]" block back
        inside the second ``async with lock:``, then wrapping the
        ``outage_candidates`` branch's relation-verdicts call in
        ``lock.release(); try: ... finally: await lock.acquire()``) —
        ``ownership_lock.violations`` goes from ``[]`` to a "release() by
        non-holder" entry. The shipped code has no manual acquire/release
        left to reintroduce that shape into without editing production
        code, which is exactly the property this test pins: nothing here
        EVER touches the lock outside a plain ``async with lock:``.
        """
        engine, _adapter, embedder, _clock, adapter_id = engine_ctx
        ownership_lock = _OwnershipLock()
        engine._write_lock = ownership_lock

        seed_text = "alpha bravo charlie delta echo foxtrot golf hotel"
        r0 = await engine.save(seed_text, scope="global")
        assert r0.outcome == "saved"

        wrapper = FaultInjectingEmbedder(embedder)
        engine._embedder = as_embedder(wrapper)
        gated_assessor = _GatedAssessor("repetition")
        engine._assessor = gated_assessor
        wrapper.arm()

        task = asyncio.create_task(engine.save(seed_text + " india juliet", scope="global"))
        await asyncio.wait_for(gated_assessor.started.wait(), timeout=5.0)
        assert ownership_lock.holder is None, (
            f"[{adapter_id}] the save must have released the write lock before "
            "reaching the Assessor call (D12) — test setup is wrong if this fails"
        )

        # Stand in for a second, concurrent lock holder -- acquired directly
        # on the SAME shared lock, so the outcome does not depend on a
        # second save() call racing this one's exact timing. Under the
        # shipped (fixed) code the save has nothing left to do with the
        # lock after the Assessor returns, so it is typically already done
        # by the time ``task.cancel()`` runs below -- a harmless no-op.
        # Under the reintroduced original shape it is still blocked waiting
        # to reacquire, exactly where the cancellation needs to land.
        await ownership_lock.acquire()
        try:
            gated_assessor.release.set()
            for _ in range(50):
                await asyncio.sleep(0)

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            if ownership_lock.locked() and ownership_lock.holder is asyncio.current_task():
                ownership_lock.release()

        assert ownership_lock.violations == [], (
            f"[{adapter_id}] lock ownership corrupted under cancellation: "
            f"{ownership_lock.violations}"
        )

    async def test_cancellation_stress_no_crash_no_duplicate_outcomes(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Complementary breadth check (not a substitute for the
        deterministic test above): 30 concurrent outage saves (mixed
        Fact/Decision), a third of them cancelled at staggered scheduling
        points, driven through the SAME outage-bounce-plus-Assessor path
        the fix touches on both arms
        (``facts._resolve_vectorless_save``,
        ``decisions._resolve_vectorless_save_decision``). Proves no crash,
        no lost/duplicated writes, and every non-cancelled task lands a
        legitimate outcome under heavy concurrent cancellation load.
        """
        engine, _adapter, embedder, _clock, adapter_id = engine_ctx
        ownership_lock = _OwnershipLock()
        engine._write_lock = ownership_lock

        seed_text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"
        r0 = await engine.save(seed_text, scope="global")
        assert r0.outcome == "saved"

        wrapper = FaultInjectingEmbedder(embedder)
        engine._embedder = as_embedder(wrapper)
        checking_assessor = _LockCheckingAssessor("repetition", ownership_lock)
        engine._assessor = checking_assessor
        wrapper.arm()

        async def _one(i: int) -> SaveResult:
            kind: Literal["fact", "decision"] = "fact" if i % 2 == 0 else "decision"
            return await engine.save(f"{seed_text} variant-{i}", scope="global", kind=kind)

        tasks = [asyncio.create_task(_one(i)) for i in range(30)]
        for i, task in enumerate(tasks):
            if i % 3 == 0:
                for _ in range(i % 5):
                    await asyncio.sleep(0)
                task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert ownership_lock.violations == [], (
            f"[{adapter_id}] lock ownership corrupted under cancellation: "
            f"{ownership_lock.violations}"
        )
        assert ownership_lock.max_concurrent <= 1, (
            f"[{adapter_id}] more than one task held the write lock at once: "
            f"max_concurrent={ownership_lock.max_concurrent}"
        )
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                assert isinstance(result, asyncio.CancelledError), (
                    f"[{adapter_id}] task {i} raised {result!r} instead of "
                    "completing or being cleanly cancelled"
                )
            else:
                assert isinstance(result, SaveResult)
                assert result.outcome in {"saved", "bounced"}, (
                    f"[{adapter_id}] task {i} produced an unexpected outcome: {result.outcome!r}"
                )
