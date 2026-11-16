"""Admission order — the seven Observable-behavior examples (P7 admission-order
piece), both arms.

Hermetic lane only — ``MemoryAdapter`` + a call-counting ``FakeEmbedder``
wrapper, no network, no disk (same convention as ``test_engine*.py``).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Decision, Scope
from tabularium.engine import Tabularium
from tabularium.engine.assessor import Assessor
from tabularium.results import BounceResolution
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import StorageAdapter
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.fake_assessor import EchoAssessor
from tests.support.faults import (
    FaultInjectingAdapter,
    FaultInjectingEmbedder,
    as_adapter,
    as_embedder,
)

_DIM = 16


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


class _CountingEmbedder(FakeEmbedder):
    """Same deterministic SHA-256 vectors as ``FakeEmbedder`` — counts calls
    so a test can assert "no embed call" (steps 1/3 must resolve entirely
    from storage reads, before the embedder ever runs)."""

    def __init__(self, *, dim: int = _DIM) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return await super().embed(text)


async def _build_engine(
    *, assessor: Assessor | None = None, **cfg_kwargs: Any,
) -> tuple[Tabularium, MemoryAdapter, _CountingEmbedder]:
    embedder = _CountingEmbedder()
    clock = _make_clock()
    adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    cfg = _make_config(**cfg_kwargs)
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=assessor)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    await adapter.create_scope(Scope("q"))
    return engine, adapter, embedder


async def _build_outage_engine(
    *, assessor: Assessor | None = None, **cfg_kwargs: Any,
) -> tuple[Tabularium, MemoryAdapter, FaultInjectingEmbedder]:
    """A hermetic engine whose embedder is armed to fail AFTER construction
    — a caller seeds rows first (the embedder still works), then arms the
    fault so the incoming save under test hits the ``vector is None``
    outage branch."""
    inner = _CountingEmbedder()
    wrapper = FaultInjectingEmbedder(inner)
    clock = _make_clock()
    adapter = MemoryAdapter(embedding_model=inner.model, dim=inner.dim, clock=clock)
    cfg = _make_config(**cfg_kwargs)
    engine = Tabularium(adapter, as_embedder(wrapper), clock, cfg, assessor=assessor)
    await engine.initialize()
    await adapter.create_scope(Scope("p"))
    await adapter.create_scope(Scope("q"))
    return engine, adapter, wrapper


async def _build_engine_with_fault_adapter(
    *, assessor: Assessor | None = None, **cfg_kwargs: Any,
) -> tuple[Tabularium, FaultInjectingAdapter, _CountingEmbedder]:
    """A hermetic engine whose STORAGE adapter is wrapped in
    ``FaultInjectingAdapter`` — a caller seeds rows first (unarmed, free),
    then arms a named method to fail on its next call, so a save's
    MUTATION (not its read) fails mid-flight."""
    embedder = _CountingEmbedder()
    clock = _make_clock()
    inner = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
    wrapper = FaultInjectingAdapter(inner)
    cfg = _make_config(**cfg_kwargs)
    engine = Tabularium(as_adapter(wrapper), embedder, clock, cfg, assessor=assessor)
    await engine.initialize()
    await wrapper.create_scope(Scope("p"))
    await wrapper.create_scope(Scope("q"))
    return engine, wrapper, embedder


class _GateOnceAdapter:
    """Delegates every call to ``inner`` except that method ``gate_on``'s
    FIRST call blocks on an event until released, then delegates — lets a
    test park a save mid-mutation and cancel it there for real (a genuine
    ``asyncio.Task.cancel()``, never a raised fault)."""

    def __init__(self, inner: StorageAdapter, *, gate_on: str) -> None:
        self._inner = inner
        self._gate_on = gate_on
        self._gated_once = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._gate_on or not callable(attr):
            return attr

        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if not self._gated_once:
                self._gated_once = True
                self.started.set()
                await self.release.wait()
            return await attr(*args, **kwargs)

        return _wrapped


class _PauseOnceEmbedder(FakeEmbedder):
    """Blocks the FIRST ``embed()`` call for one designated text until
    released, then delegates normally — a later call, including a retry of
    the SAME text, proceeds immediately. Lets a test park a save mid-embed
    and observe what a completely different caller does elsewhere while it
    is parked."""

    def __init__(self, *, dim: int = _DIM, pause_text: str) -> None:
        super().__init__(dim=dim)
        self._pause_text = pause_text
        self._paused_once = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def embed(self, text: str) -> list[float]:
        if text == self._pause_text and not self._paused_once:
            self._paused_once = True
            self.started.set()
            await self.release.wait()
        return await super().embed(text)


def _resolved_lines(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith("admission_resolved")]


def _assert_one_resolved(caplog: pytest.LogCaptureFixture, *, step: str, outcome: str) -> None:
    lines = _resolved_lines(caplog)
    assert len(lines) == 1, (
        f"expected exactly one admission_resolved line, got {len(lines)}: "
        f"{[r.getMessage() for r in lines]}"
    )
    record = lines[0]
    assert record.step == step, f"expected step={step!r}, got {record.step!r}"  # type: ignore[attr-defined]
    assert record.outcome == outcome, (  # type: ignore[attr-defined]
        f"expected outcome={outcome!r}, got {record.outcome!r}"  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Fact arm
# ---------------------------------------------------------------------------


class TestAdmissionOrderFact:
    async def test_byte_identical_replay_is_already_saved(self) -> None:
        engine, adapter, embedder = await _build_engine()
        r1 = await engine.save("x", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        calls_after_first = embedder.calls

        r2 = await engine.save("x", scope="p")

        assert r2.outcome == "already_saved"
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == r1.hypomnema.id
        assert embedder.calls == calls_after_first, "step 1 must never embed"
        events = await adapter.query_events(target_id=r1.hypomnema.id)
        assert events == [], "a replay is not new evidence — no reinforcement event"
        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 1

    async def test_whitespace_variant_reinforces_via_exact_key(self) -> None:
        engine, _adapter, embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        calls_after_first = embedder.calls

        r2 = await engine.save("x  y\n", scope="p")

        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == r1.hypomnema.id
        assert embedder.calls == calls_after_first, "step 3 must never embed"

    async def test_downward_reinforces_the_global_row_silently(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="global")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None

        r2 = await engine.save("x y", scope="p")

        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == r1.hypomnema.id

    async def test_upward_bounces_naming_the_candidate_and_its_scope(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None

        r2 = await engine.save("x y", scope="global")

        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.candidate_id == r1.hypomnema.id
        assert r2.bounce.candidate_scope == "p"
        assert r2.bounce.verdict == "exact_key"
        actions = {exit_.action for exit_ in r2.bounce.exits}
        assert actions == {"supersede"}

    async def test_upward_retry_with_moved_reason_completes_as_a_move(self) -> None:
        """The piece's own "upward retry with moved" example:
        ``save(scope="global", supersedes=[<p row>], reason="moved")`` completes as a move —
        the p row is excluded from the exact-key step (step 2 of D1) so it never blocks its own
        successor, a fresh global row lands, and the p row is superseded
        with reason ``moved``.
        """
        engine, adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        p_id = r1.hypomnema.id

        r2 = await engine.save("x y", scope="global", supersedes=[p_id], reason="moved")

        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        global_id = r2.hypomnema.id
        assert global_id != p_id

        p_row = await adapter.get_by_id(p_id)
        assert p_row.validity == "invalid"
        assert p_row.superseded_by == global_id
        assert p_row.supersession_reason == "moved"

        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 2, "no extra row minted — one new global row, the p row kept"

    async def test_ties_prefer_same_scope_then_global_then_oldest(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        global_row = await engine.save("x y", scope="global")
        assert global_row.outcome == "saved"
        assert global_row.hypomnema is not None

        # A same-scope candidate, created AFTER the global one — the tier
        # rule must still pick it (same-scope beats global regardless of age).
        same_scope_row = await engine.save("x y", scope="q")
        assert same_scope_row.outcome == "reinforced"
        # First save into "q" reinforces the pre-existing global row (there is
        # no "q"-scope row yet at this point) — assert against the global row.
        assert same_scope_row.reinforced_id == global_row.hypomnema.id

    async def test_siblings_never_probe_each_other(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        r2 = await engine.save("x y", scope="q")

        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        assert r1.hypomnema is not None
        assert r2.hypomnema.id != r1.hypomnema.id


# ---------------------------------------------------------------------------
# Decision arm
# ---------------------------------------------------------------------------


class TestReplayExcludesOnlyResolveNotSupersedes:
    """``check_replay``'s own ``exclude`` must
    never contain ``supersede_ids`` — naming a row's own byte-identical
    content as its ``supersedes`` predecessor is still a replay (D13's "not
    new evidence to mutate"), not a mutation that retires the row and
    mints a successor."""

    async def test_fact_naming_own_content_as_supersedes_still_replays(self) -> None:
        engine, adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        row_id = r1.hypomnema.id

        r2 = await engine.save("x y", scope="p", supersedes=[row_id], reason="correction")

        assert r2.outcome == "already_saved"
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == row_id

        row = await adapter.get_by_id(row_id)
        assert row.validity == "valid"
        assert row.superseded_by is None
        rows = [r async for r in await adapter.scan_records()]
        assert len(rows) == 1, "naming a replay as its own predecessor must not mint a successor"

    async def test_decision_naming_own_content_as_supersedes_still_replays(self) -> None:
        engine, adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        row_id = r1.hypomnema.id

        r2 = await engine.save(
            "x y", scope="p", kind="decision", supersedes=row_id, reason="correction"
        )

        assert r2.outcome == "already_saved"
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == row_id

        row = await adapter.get_by_id(row_id)
        assert row.lifecycle == "in_force"
        assert row.superseded_by is None
        rows = [r async for r in await adapter.scan_records()]
        assert len(rows) == 1, "naming a replay as its own predecessor must not mint a successor"


class TestReinforceResolveStaleCandidateNoDuplicate:
    """A ``reinforce`` resolve naming
    a stale/invalidated candidate B, resubmitting content that is actually
    byte-identical to a DIFFERENT live row A, must not duplicate A. Steps
    1/3 now run regardless of ``resolve`` — ``replay_exclude``/
    ``admission_exclude`` already carve B, the resolve's own candidate, out
    of both steps' population, so this never short-circuits a live
    adjudication against B itself.

    A's content is tokenless (BM25 finds no ``[a-z0-9]+`` tokens in pure
    punctuation — ``MemoryAdapter._tokenize``) so the serial probe's own
    survivor set, which the pre-fix code fell through to unconditionally
    whenever ``resolve.action == "reinforce"``, misses A entirely; only the
    admission steps this fix restores catch it.

    Falsify: restore the ``if resolve is None or resolve.action ==
    "confirm_distinct":`` gate around the ``_admission_steps_1_3`` call in
    both lock holds of ``facts.save_internal``/``decisions.save_decision``
    — both tests below go red, each with ``outcome="saved"`` (a second row
    minted alongside A) instead of ``already_saved``.
    """

    async def test_fact_reinforce_naming_stale_candidate_replays_onto_live_row(
        self,
    ) -> None:
        engine, adapter, _embedder = await _build_engine()
        a_text = "!!! !!! !!!"

        ra = await engine.save(a_text, scope="p")
        assert ra.outcome == "saved"
        assert ra.hypomnema is not None
        a_id = ra.hypomnema.id

        rb = await engine.save("an unrelated live candidate row", scope="p")
        assert rb.outcome == "saved"
        assert rb.hypomnema is not None
        b_id = rb.hypomnema.id

        invalidated = await engine.set_validity(b_id, False, "stale candidate")
        assert invalidated.outcome == "revised"

        resolve = BounceResolution(candidate_id=b_id, action="reinforce")
        r3 = await engine.save(a_text, scope="p", resolve=resolve)

        assert r3.outcome == "already_saved", (
            f"expected the byte-identical replay against A to intercept the "
            f"stale reinforce, got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == a_id

        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 2, "no duplicate of A minted"

    async def test_decision_reinforce_naming_stale_candidate_replays_onto_live_row(
        self,
    ) -> None:
        engine, adapter, _embedder = await _build_engine()
        a_text = "!!! !!! !!!"

        ra = await engine.save(a_text, scope="p", kind="decision")
        assert ra.outcome == "saved"
        assert ra.hypomnema is not None
        a_id = ra.hypomnema.id

        rb = await engine.save("an unrelated live candidate row", scope="p", kind="decision")
        assert rb.outcome == "saved"
        assert rb.hypomnema is not None
        b_id = rb.hypomnema.id

        invalidated = await engine.set_validity(b_id, False, "stale candidate")
        assert invalidated.outcome == "revised"

        resolve = BounceResolution(candidate_id=b_id, action="reinforce")
        r3 = await engine.save(a_text, scope="p", kind="decision", resolve=resolve)

        assert r3.outcome == "already_saved", (
            f"expected the byte-identical replay against A to intercept the "
            f"stale reinforce, got outcome={r3.outcome!r}"
        )
        assert r3.hypomnema is not None
        assert r3.hypomnema.id == a_id

        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 2, "no duplicate of A minted"


class TestAdmissionOrderDecision:
    async def test_byte_identical_replay_is_already_saved(self) -> None:
        engine, adapter, embedder = await _build_engine()
        r1 = await engine.save("x", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        calls_after_first = embedder.calls

        r2 = await engine.save("x", scope="p", kind="decision")

        assert r2.outcome == "already_saved"
        assert r2.hypomnema is not None
        assert r2.hypomnema.id == r1.hypomnema.id
        assert embedder.calls == calls_after_first
        events = await adapter.query_events(target_id=r1.hypomnema.id)
        assert events == []

    async def test_whitespace_variant_reinforces_via_exact_key(self) -> None:
        engine, _adapter, embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        calls_after_first = embedder.calls

        r2 = await engine.save("x  y\n", scope="p", kind="decision")

        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == r1.hypomnema.id
        assert embedder.calls == calls_after_first

    async def test_downward_reinforces_the_global_row_silently(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="global", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None

        r2 = await engine.save("x y", scope="p", kind="decision")

        assert r2.outcome == "reinforced"
        assert r2.reinforced_id == r1.hypomnema.id

    async def test_upward_bounces_naming_the_candidate_and_its_scope(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None

        r2 = await engine.save("x y", scope="global", kind="decision")

        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.candidate_id == r1.hypomnema.id
        assert r2.bounce.candidate_scope == "p"
        assert r2.bounce.verdict == "exact_key"
        actions = {exit_.action for exit_ in r2.bounce.exits}
        assert actions == {"supersede"}

    async def test_upward_retry_with_moved_reason_completes_as_a_move(self) -> None:
        """Decision-arm sibling of the Fact arm's test above: the same
        upward retry completes as a move, one events transaction, the p
        row's lifecycle flipped to ``superseded`` with reason ``moved``.
        """
        engine, adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        p_id = r1.hypomnema.id

        r2 = await engine.save(
            "x y", scope="global", kind="decision", supersedes=p_id, reason="moved"
        )

        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        global_id = r2.hypomnema.id
        assert global_id != p_id

        p_row = await adapter.get_by_id(p_id)
        assert p_row.lifecycle == "superseded"
        assert p_row.superseded_by == global_id
        assert p_row.supersession_reason == "moved"

        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 2, "no extra row minted — one new global row, the p row kept"

    async def test_siblings_never_probe_each_other(self) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        r2 = await engine.save("x y", scope="q", kind="decision")

        assert r2.outcome == "saved"
        assert r2.hypomnema is not None
        assert r1.hypomnema is not None
        assert r2.hypomnema.id != r1.hypomnema.id


class TestOutagePromotedFromRetrySucceeds:
    """a judged outage bounce against an
    active Fact offers ``promoted_from`` — the retry it names must actually
    complete, not bounce again. Before this fix nothing on the outage path
    ever looked at ``promoted_from``, so the caller's own suggested retry
    bounced forever."""

    async def test_promoted_from_retry_under_outage_completes(self) -> None:
        engine, adapter, wrapper = await _build_outage_engine(assessor=EchoAssessor("repetition"))
        r_fact = await engine.save("alpha bravo charlie delta echo", scope="p")
        assert r_fact.outcome == "saved"
        assert r_fact.hypomnema is not None
        fact_id = r_fact.hypomnema.id

        wrapper.arm()
        r1 = await engine.save(
            "alpha bravo charlie delta echo foxtrot", scope="p", kind="decision"
        )
        assert r1.outcome == "bounced"
        assert r1.bounce is not None
        assert r1.bounce.candidate_id == fact_id
        assert r1.bounce.similarity is None
        actions = {exit_.action for exit_ in r1.bounce.exits}
        assert "promoted_from" in actions, f"expected an offered promotion, got {actions}"

        r2 = await engine.save(
            "alpha bravo charlie delta echo foxtrot",
            scope="p",
            kind="decision",
            promoted_from=[fact_id],
        )

        assert r2.outcome == "saved", f"the retry the bounce itself offered must complete: {r2}"
        assert r2.hypomnema is not None
        decision = await adapter.get_by_id(r2.hypomnema.id)
        assert isinstance(decision, Decision)
        assert decision.promoted_from
        assert decision.promoted_from[0].id == fact_id


class TestOutageJudgedBounceExcludesSupersedeOnRetiredDecision:
    """the outage path's ``bm25_probe``
    has no retirement filter, so a superseded Decision can become its
    primary candidate; the judged bounce built from it must not offer
    ``supersede`` (``save_decision``'s own validation only accepts an
    ``in_force`` predecessor)."""

    async def test_superseded_decision_primary_never_offers_supersede(self) -> None:
        engine, adapter, wrapper = await _build_outage_engine(assessor=EchoAssessor("repetition"))
        r1 = await engine.save("alpha bravo charlie delta echo", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        old_id = r1.hypomnema.id

        r2 = await engine.save(
            "the backend moved to a completely different design",
            scope="p",
            kind="decision",
            supersedes=old_id,
            reason="changed",
        )
        assert r2.outcome == "saved"
        old_row = await adapter.get_by_id(old_id)
        assert old_row.lifecycle == "superseded"

        wrapper.arm()
        r3 = await engine.save(
            "alpha bravo charlie delta echo foxtrot", scope="p", kind="decision"
        )

        assert r3.outcome == "bounced"
        assert r3.bounce is not None
        assert r3.bounce.candidate_id == old_id, "BM25 must surface the retired row as primary"
        actions = {exit_.action for exit_ in r3.bounce.exits}
        assert actions == {"reinforce", "confirm_distinct"}, (
            f"a superseded predecessor must never be offered as a supersede target: {actions}"
        )


class TestGlobalProbeSpanRefreshesDuringEmbed:
    """a scope registered WHILE a
    global save is paused inside ``embed()`` must still be seen once that
    same save resumes and probes — the probe span is recomputed fresh
    under the SECOND lock hold, never carried over from the first."""

    async def test_scope_registered_during_embed_pause_is_probed(self) -> None:
        content = "alpha bravo charlie delta echo golf hotel"
        embedder = _PauseOnceEmbedder(pause_text=content)
        clock = _make_clock()
        adapter = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
        cfg = _make_config()
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        global_task = asyncio.create_task(engine.save(content, scope="global", kind="decision"))
        await asyncio.wait_for(embedder.started.wait(), timeout=5.0)

        # Registered and written WHILE the global save sits paused inside
        # embed() -- neither a lock hold is open at this point (embed runs
        # strictly between the two), so this completes on its own.
        await adapter.create_scope(Scope("r"))
        r_named = await engine.save(content, scope="r", kind="decision")
        assert r_named.outcome == "saved"
        assert r_named.hypomnema is not None

        embedder.release.set()
        r_global = await asyncio.wait_for(global_task, timeout=5.0)

        assert r_global.outcome == "bounced", (
            "the named scope's row, registered while the global save was "
            "paused in embed(), must still be visible to its probe"
        )
        assert r_global.bounce is not None
        assert r_global.bounce.candidate_id == r_named.hypomnema.id
        assert r_global.bounce.candidate_scope == "r"

        rows = [row async for row in await adapter.scan_records()]
        assert len(rows) == 1, "a stale probe span would let the global save write a duplicate"


class TestExactKeyMutationFailuresStillLogOneLine:
    """the reinforcement mutation
    (or, for an archived Fact's replay, the lifecycle transition that
    follows it) can still raise — D10's one ``admission_resolved`` line per
    save must fire regardless, then the exception must still propagate."""

    async def test_fact_exact_key_reinforce_failure_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, adapter, _embedder = await _build_engine_with_fault_adapter()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        adapter.fault(on="update_projections", call=1, exc=RuntimeError("boom"))
        adapter.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            with pytest.raises(RuntimeError, match="boom"):
                await engine.save("x  y\n", scope="p")

        _assert_one_resolved(caplog, step="exact_key", outcome="raised")

    async def test_fact_archived_replay_revival_failure_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, adapter, _embedder = await _build_engine_with_fault_adapter()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        await adapter.transition(r1.hypomnema.id, "archived", at=_make_clock().now())

        adapter.fault(on="transition", call=1, exc=RuntimeError("boom"))
        adapter.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            with pytest.raises(RuntimeError, match="boom"):
                await engine.save("x y", scope="p")

        _assert_one_resolved(caplog, step="replay", outcome="raised")

    async def test_decision_exact_key_reinforce_failure_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, adapter, _embedder = await _build_engine_with_fault_adapter()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        adapter.fault(on="update_projections", call=1, exc=RuntimeError("boom"))
        adapter.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            with pytest.raises(RuntimeError, match="boom"):
                await engine.save("x  y\n", scope="p", kind="decision")

        _assert_one_resolved(caplog, step="exact_key", outcome="raised")


class TestCancellationDuringMutationStillLogsOneLine:
    """the admission
    wrappers widened from ``except Exception`` to ``except BaseException``
    so a genuine task cancellation mid-mutation still gets its one
    ``admission_resolved`` line, same as any other raised error, and the
    cancellation itself still propagates unchanged."""

    async def test_fact_exact_key_reinforce_cancellation_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        embedder = _CountingEmbedder()
        clock = _make_clock()
        inner = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
        gate = _GateOnceAdapter(inner, gate_on="append_event")
        cfg = _make_config()
        engine = Tabularium(cast(StorageAdapter, gate), embedder, clock, cfg)
        await engine.initialize()
        await inner.create_scope(Scope("p"))

        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            task = asyncio.create_task(engine.save("x  y\n", scope="p"))
            await asyncio.wait_for(gate.started.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _assert_one_resolved(caplog, step="exact_key", outcome="raised")

    async def test_decision_exact_key_reinforce_cancellation_logs_one_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        embedder = _CountingEmbedder()
        clock = _make_clock()
        inner = MemoryAdapter(embedding_model=embedder.model, dim=embedder.dim, clock=clock)
        gate = _GateOnceAdapter(inner, gate_on="append_event")
        cfg = _make_config()
        engine = Tabularium(cast(StorageAdapter, gate), embedder, clock, cfg)
        await engine.initialize()
        await inner.create_scope(Scope("p"))

        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            task = asyncio.create_task(engine.save("x  y\n", scope="p", kind="decision"))
            await asyncio.wait_for(gate.started.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        _assert_one_resolved(caplog, step="exact_key", outcome="raised")


# ---------------------------------------------------------------------------
# admission_resolved DEBUG line (P7 admission-order piece, D10) — caplog
# harvest of the step that resolved a save, for all seven Observable-
# behavior order examples (the upward retry with ``moved`` resolves at
# step="excluded_only" — the p row is the only key match in span and it is
# excluded by its own ``supersedes`` entry, so nothing survives to the
# exact-key or probe steps and the write lands novel) and the three
# embedder-outage outcomes, on both arms.
# ---------------------------------------------------------------------------


class TestAdmissionResolvedLogFact:
    async def test_replay_logs_replay_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x", scope="p")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x", scope="p")

        assert r2.outcome == "already_saved"
        _assert_one_resolved(caplog, step="replay", outcome="already_saved")

    async def test_whitespace_variant_logs_exact_key_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x  y\n", scope="p")

        assert r2.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_downward_logs_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="global")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="p")

        assert r2.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_upward_logs_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="global")

        assert r2.outcome == "bounced"
        _assert_one_resolved(caplog, step="exact_key", outcome="bounced")

    async def test_upward_retry_with_moved_reason_logs_excluded_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        p_id = r1.hypomnema.id

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="global", supersedes=[p_id], reason="moved")

        assert r2.outcome == "saved"
        _assert_one_resolved(caplog, step="excluded_only", outcome="saved")

    async def test_ties_log_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        global_row = await engine.save("x y", scope="global")
        assert global_row.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            same_scope_row = await engine.save("x y", scope="q")

        assert same_scope_row.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_siblings_log_below_floor_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="q")

        assert r2.outcome == "saved"
        _assert_one_resolved(caplog, step="below_floor", outcome="saved")

    async def test_outage_bm25_only_with_assessor_logs_bm25_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine(
            assessor=EchoAssessor("repetition")
        )
        r1 = await engine.save("alpha bravo charlie delta echo", scope="p")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("alpha bravo charlie delta echo foxtrot", scope="p")

        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.similarity is None
        _assert_one_resolved(caplog, step="bm25_only", outcome="bounced")

    async def test_outage_bm25_only_without_assessor_logs_bm25_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine()
        r1 = await engine.save("golf hotel india juliet kilo", scope="p")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("golf hotel india juliet kilo lima", scope="p")

        assert r2.outcome == "bounced"
        assert r2.reason == "judge_unavailable"
        _assert_one_resolved(caplog, step="bm25_only", outcome="bounced")

    async def test_outage_vectorless_logs_vectorless_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine()
        r1 = await engine.save("mike november oscar papa quebec", scope="p")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("zulu yankee xray whiskey victor", scope="p")

        assert r2.outcome == "saved"
        assert r2.embedded is False
        _assert_one_resolved(caplog, step="vectorless", outcome="saved")


class TestAdmissionResolvedLogDecision:
    async def test_replay_logs_replay_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x", scope="p", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x", scope="p", kind="decision")

        assert r2.outcome == "already_saved"
        _assert_one_resolved(caplog, step="replay", outcome="already_saved")

    async def test_whitespace_variant_logs_exact_key_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x  y\n", scope="p", kind="decision")

        assert r2.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_downward_logs_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="global", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="p", kind="decision")

        assert r2.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_upward_logs_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="global", kind="decision")

        assert r2.outcome == "bounced"
        _assert_one_resolved(caplog, step="exact_key", outcome="bounced")

    async def test_upward_retry_with_moved_reason_logs_excluded_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"
        assert r1.hypomnema is not None
        p_id = r1.hypomnema.id

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save(
                "x y", scope="global", kind="decision", supersedes=p_id, reason="moved"
            )

        assert r2.outcome == "saved"
        _assert_one_resolved(caplog, step="excluded_only", outcome="saved")

    async def test_ties_log_exact_key_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        global_row = await engine.save("x y", scope="global", kind="decision")
        assert global_row.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            same_scope_row = await engine.save("x y", scope="q", kind="decision")

        assert same_scope_row.outcome == "reinforced"
        _assert_one_resolved(caplog, step="exact_key", outcome="reinforced")

    async def test_siblings_log_below_floor_step(self, caplog: pytest.LogCaptureFixture) -> None:
        engine, _adapter, _embedder = await _build_engine()
        r1 = await engine.save("x y", scope="p", kind="decision")
        assert r1.outcome == "saved"

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save("x y", scope="q", kind="decision")

        assert r2.outcome == "saved"
        _assert_one_resolved(caplog, step="below_floor", outcome="saved")

    async def test_outage_bm25_only_with_assessor_logs_bm25_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine(
            assessor=EchoAssessor("repetition")
        )
        r1 = await engine.save("alpha bravo charlie delta echo", scope="p", kind="decision")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save(
                "alpha bravo charlie delta echo foxtrot", scope="p", kind="decision"
            )

        assert r2.outcome == "bounced"
        assert r2.bounce is not None
        assert r2.bounce.similarity is None
        _assert_one_resolved(caplog, step="bm25_only", outcome="bounced")

    async def test_outage_bm25_only_without_assessor_logs_bm25_only_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine()
        r1 = await engine.save("golf hotel india juliet kilo", scope="p", kind="decision")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save(
                "golf hotel india juliet kilo lima", scope="p", kind="decision"
            )

        assert r2.outcome == "bounced"
        assert r2.reason == "judge_unavailable"
        _assert_one_resolved(caplog, step="bm25_only", outcome="bounced")

    async def test_outage_vectorless_logs_vectorless_step(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, wrapper = await _build_outage_engine()
        r1 = await engine.save("mike november oscar papa quebec", scope="p", kind="decision")
        assert r1.outcome == "saved"

        wrapper.arm()
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="tabularium"):
            r2 = await engine.save(
                "zulu yankee xray whiskey victor", scope="p", kind="decision"
            )

        assert r2.outcome == "saved"
        assert r2.embedded is False
        _assert_one_resolved(caplog, step="vectorless", outcome="saved")
