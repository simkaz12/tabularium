"""Backfill pass tests (vector-less-rows piece).

Dual-parameterized over both storage adapters via the ``engine_ctx`` fixture
for the pass's general shape (fills vector-less rows, skips buried ones,
byte-identical columns); concurrency-timing tests (guard, rerun, mid-pass
outage) run against ``MemoryAdapter`` only — deterministic single-process
interleaving is what those tests are actually about, not adapter mechanics,
the same precedent the codebase's other concurrency-control tests already
follow.
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig, StoreConfig, config_from_env
from tabularium.domain import Fact, Scope, new_id
from tabularium.embedding import Embedder
from tabularium.engine import Tabularium
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
    _raw_vector,
    _reopen_at_model,
)
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.edits import edit_content
from tests.support.faults import FaultInjectingEmbedder, as_embedder
from tests.support.lance_raw import raw_table

AnyBackfillAdapter = MemoryAdapter | LanceDBAdapter

_DIM = 16
EPOCH = datetime(2025, 1, 1, tzinfo=UTC)


def _clock(at: datetime = EPOCH) -> TestClock:
    return TestClock(at)


def _cfg(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"gray_floor": 0.80, "candidate_k": 1000}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


def _fact(*, content: str, scope: str = "global") -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content=content,
        scope=Scope(scope),
        lifecycle="active",
        created_at=EPOCH,
        updated_at=EPOCH,
    )


class _BlockingEmbedder:
    """Wraps a real ``Embedder``; every ``embed`` call blocks on ``gate``
    before delegating — lets a test pin a backfill pass mid-embed."""

    def __init__(self, inner: Embedder, *, gate: asyncio.Event) -> None:
        self._inner = inner
        self._gate = gate

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model(self) -> str:
        return self._inner.model

    async def embed(self, text: str) -> list[float]:
        await self._gate.wait()
        return await self._inner.embed(text)


@dataclass
class _Ctx:
    engine: Tabularium
    adapter: AnyBackfillAdapter
    fault: FaultInjectingEmbedder
    fake: FakeEmbedder
    clock: TestClock


@pytest.fixture(params=["memory", "lancedb"])
async def engine_ctx(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> _Ctx:
    """A fresh engine wrapping a ``FaultInjectingEmbedder`` around a real
    ``FakeEmbedder`` — ``arm()``/``lift()``/``fault()`` simulate the outage
    and mid-pass-failure scenarios without a second embedder type."""
    fake = FakeEmbedder(dim=_DIM)
    fault = FaultInjectingEmbedder(fake)
    clock = _clock()
    adapter: AnyBackfillAdapter
    if request.param == "memory":
        adapter = _make_memory_adapter(embedding_model=fake.model, dim=_DIM, clock=clock)
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=fake.model, dim=_DIM, clock=clock
        )
    engine = Tabularium(adapter, as_embedder(fault), clock, _cfg())
    await engine.initialize()
    return _Ctx(engine=engine, adapter=adapter, fault=fault, fake=fake, clock=clock)


# ---------------------------------------------------------------------------
# Main shape: fills vector-less rows, skips buried, byte-identical columns
# ---------------------------------------------------------------------------


class TestBackfillFillsVectorlessRows:
    async def test_fills_two_live_rows_skips_the_buried_one(self, engine_ctx: _Ctx) -> None:
        engine, adapter, fault = engine_ctx.engine, engine_ctx.adapter, engine_ctx.fault

        fault.arm()
        r1 = await engine.save("aardvark migration patterns", scope="global")
        r2 = await engine.save("bumblebee pollination routes", scope="global")
        r3 = await engine.save("cheetah sprint mechanics", kind="decision", scope="global")
        assert r1.embedded is False
        assert r2.embedded is False
        assert r3.embedded is False
        assert r1.hypomnema is not None
        assert r2.hypomnema is not None
        assert r3.hypomnema is not None

        await adapter.bury(r2.hypomnema.id, "test burial")

        before_r1 = await adapter.get_by_id(r1.hypomnema.id)
        before_r3 = await adapter.get_by_id(r3.hypomnema.id)

        fault.lift()
        r4 = await engine.save("dolphin echolocation frequency", scope="global")
        assert r4.embedded is True

        await engine.pending_backfill

        after_r1 = await adapter.get_by_id(r1.hypomnema.id)
        after_r3 = await adapter.get_by_id(r3.hypomnema.id)
        assert after_r1.embedding_model == engine_ctx.fake.model
        assert after_r3.embedding_model == engine_ctx.fake.model

        # Every OTHER column is byte-identical to before the pass touched it.
        assert before_r1.model_dump(exclude={"embedding_model"}) == after_r1.model_dump(
            exclude={"embedding_model"}
        )
        assert before_r3.model_dump(exclude={"embedding_model"}) == after_r3.model_dump(
            exclude={"embedding_model"}
        )

        # The buried row was never a target — excluded, not merely skipped.
        remaining_targets = {t.id for t in await adapter.backfill_targets()}
        assert r2.hypomnema.id not in remaining_targets
        assert r1.hypomnema.id not in remaining_targets
        assert r3.hypomnema.id not in remaining_targets

        # T1 (P6 vector-less-rows piece, backfill test-gap fix): the buried
        # id's own refusal, not merely its absence from backfill_targets().
        # bury() physically deletes the row on both adapters, so a
        # DIRECT backfill() call against it is refused by the row's own
        # absence regardless of the id's own `_buried_ids` guard — that
        # guard is isolated below by adding a still-LIVE row's id to
        # `_buried_ids` without burying it (simulating the guard's own
        # refusal independent of physical deletion), then confirming its
        # raw vector never materializes.
        from tabularium.storage.content_key import content_hash

        live_but_marked = await engine.save("live row wrongly marked buried", scope="global")
        assert live_but_marked.hypomnema is not None
        assert live_but_marked.embedded is True
        adapter._buried_ids.add(live_but_marked.hypomnema.id)
        try:
            guard_write = await adapter.backfill(
                live_but_marked.hypomnema.id,
                engine_ctx.fake.peek("live row wrongly marked buried"),
                embedding_model=engine_ctx.fake.model,
                expected_hash=content_hash("live row wrongly marked buried"),
            )
            assert guard_write is False
        finally:
            adapter._buried_ids.discard(live_but_marked.hypomnema.id)

        assert await _raw_vector(adapter, r2.hypomnema.id) is None
        with pytest.raises(KeyError):
            await adapter.get_by_id(r2.hypomnema.id)

        # The filled rows are now real vector-search matches on their own content.
        v1 = engine_ctx.fake.peek("aardvark migration patterns")
        v3 = engine_ctx.fake.peek("cheetah sprint mechanics")
        from tabularium.storage.types import Filters

        filters = Filters(
            scopes=[Scope("global")],
            validity=frozenset({"valid"}),
            lifecycle_in=frozenset({"active", "in_force"}),
        )
        hits1 = await adapter.vector_search(v1, filters=filters, k=5)
        hits3 = await adapter.vector_search(v3, filters=filters, k=5)
        assert any(h.record.id == r1.hypomnema.id for h in hits1)
        assert any(h.record.id == r3.hypomnema.id for h in hits3)


# ---------------------------------------------------------------------------
# Stale-model: a row stamped under an old model gets re-embedded
# ---------------------------------------------------------------------------


class _SeededFakeEmbedder:
    """Wraps a real ``FakeEmbedder``; every call hashes ``f"{seed}:{text}"``
    instead of ``text`` alone. ``FakeEmbedder``'s own vector depends ONLY on
    the text's SHA-256 digest, never on ``model`` (see its own docstring) --
    two ``FakeEmbedder`` instances with different ``model`` values still
    produce IDENTICAL vectors for the same text, so a stale-model test built
    on plain ``FakeEmbedder(model=...)`` alone cannot distinguish a genuine
    re-embed from a stamp-only write that left the old vector in place. A
    seeded hash input is deterministic but genuinely different, closing
    that gap (T2, P6 vector-less-rows piece, backfill test-gap fix)."""

    def __init__(self, inner: FakeEmbedder, *, seed: str) -> None:
        self._inner = inner
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model(self) -> str:
        return self._inner.model

    async def embed(self, text: str) -> list[float]:
        return await self._inner.embed(f"{self._seed}:{text}")

    def peek(self, text: str) -> list[float]:
        return self._inner.peek(f"{self._seed}:{text}")


class TestBackfillReembedsStaleModel:
    async def test_stale_model_row_reembedded_on_restart(
        self, engine_ctx: _Ctx, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, clock = engine_ctx.engine, engine_ctx.adapter, engine_ctx.clock
        saved = await engine.save("stale model fact", scope="global")
        assert saved.embedded is True
        assert saved.hypomnema is not None

        vector_before = await _raw_vector(adapter, saved.hypomnema.id)
        assert vector_before is not None

        adapter_v2 = await _reopen_at_model(
            adapter, embedding_model="nomic-v2", tmp_path=tmp_path, clock=clock, dim=_DIM
        )
        # A genuine model change redeploys BOTH the adapter's configured
        # stamp AND the embedder actually in use — a fresh, seeded embedder
        # reporting the NEW model, never the old ``engine_ctx.fake`` and
        # never producing the SAME vectors it would (see
        # ``_SeededFakeEmbedder``'s own docstring for why that distinction
        # matters here).
        fake_v2 = _SeededFakeEmbedder(FakeEmbedder(dim=_DIM, model="nomic-v2"), seed="v2")
        fault_v2 = FaultInjectingEmbedder(fake_v2)
        engine_v2 = Tabularium(adapter_v2, as_embedder(fault_v2), clock, _cfg())
        await engine_v2.initialize()
        await engine_v2.pending_backfill

        after = await adapter_v2.get_by_id(saved.hypomnema.id)
        assert after.embedding_model == "nomic-v2"
        assert after.content == "stale model fact"

        vector_after = await _raw_vector(adapter_v2, saved.hypomnema.id)
        assert vector_after is not None
        assert vector_after != vector_before, (
            "the row must carry a genuinely RE-EMBEDDED vector, not the old "
            "one under a relabeled stamp"
        )
        assert vector_after == fake_v2.peek("stale model fact")


# ---------------------------------------------------------------------------
# Concurrency-timing: guard, rerun, mid-pass outage — MemoryAdapter only
# ---------------------------------------------------------------------------


class TestBackfillGuard:
    async def test_row_edited_while_pass_waits_for_the_lock_is_skipped(self) -> None:
        """T3 (P6 vector-less-rows piece, backfill test-gap fix): narrower
        than the embed-duration window the test below covers — an edit
        landing in the exact gap between the pass finishing its embed
        (having already captured the target's STALE ``content_hash`` at
        scan time) and actually acquiring the lock to write. The pass's
        own hash check re-reads the row FRESH under that same lock,
        immediately before the write (never a pre-lock check against a
        possibly-stale read — see ``reproB_218_hash_race.py``'s own
        write-up of the design this guards against), so even this
        narrowest slice of the window must still be caught."""
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        engine = Tabularium(adapter, fake, clock, _cfg())
        await engine.initialize()

        target = _fact(content="will be edited while the pass waits for the lock")
        await adapter.upsert(target, None)

        from tabularium.engine.backfill import run_pass

        async with engine._write_lock:
            pass_task = asyncio.ensure_future(
                run_pass(
                    adapter, fake, engine._write_lock,
                    embedding_model=fake.model, trigger="embed",
                )
            )
            await asyncio.sleep(0)
            assert not pass_task.done(), (
                "the pass must be blocked waiting for the lock this test "
                "already holds, not finished or stuck earlier"
            )

            await edit_content(
                adapter, target.id, "edited while the pass waits for the lock", None
            )

        report = await pass_task

        assert report.skipped_hash == 1
        assert report.filled == 0
        after = await adapter.get_by_id(target.id)
        assert after.content == "edited while the pass waits for the lock"
        assert after.embedding_model is None

    async def test_row_edited_mid_pass_is_skipped_then_filled_next_pass(self) -> None:
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        gate = asyncio.Event()
        blocking = _BlockingEmbedder(fake, gate=gate)
        engine = Tabularium(adapter, blocking, clock, _cfg())
        await engine.initialize()

        target = _fact(content="will be edited mid pass")
        await adapter.upsert(target, None)

        from tabularium.engine.backfill import run_pass

        pass_task = asyncio.ensure_future(
            run_pass(
                adapter, blocking, engine._write_lock,
                embedding_model=fake.model, trigger="embed",
            )
        )
        # Give the pass a chance to start embedding (it will block on `gate`).
        await asyncio.sleep(0)

        async with engine._write_lock:
            await edit_content(adapter, target.id, "edited while the pass was mid-embed", None)

        gate.set()
        report = await pass_task

        assert report.skipped_hash == 1
        after_first_pass = await adapter.get_by_id(target.id)
        assert after_first_pass.content == "edited while the pass was mid-embed"
        assert after_first_pass.embedding_model is None

        gate2 = asyncio.Event()
        gate2.set()
        blocking2 = _BlockingEmbedder(fake, gate=gate2)
        second_report = await run_pass(
            adapter, blocking2, engine._write_lock, embedding_model=fake.model, trigger="embed"
        )
        assert second_report.filled == 1
        after_second_pass = await adapter.get_by_id(target.id)
        assert after_second_pass.embedding_model == fake.model


class TestBackfillRerun:
    async def test_trigger_twice_while_blocked_runs_exactly_two_passes(self) -> None:
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        gate = asyncio.Event()
        blocking = _BlockingEmbedder(fake, gate=gate)
        engine = Tabularium(adapter, blocking, clock, _cfg())
        await engine.initialize()

        await adapter.upsert(_fact(content="rerun target one"), None)

        pass_count = 0

        async def _counted_pass() -> Any:
            nonlocal pass_count
            pass_count += 1
            from tabularium.engine.backfill import run_pass

            return await run_pass(
                adapter, blocking, engine._write_lock,
                embedding_model=fake.model, trigger="embed",
            )

        state = engine._backfiller
        state.dirty = True
        state.start(_counted_pass)
        await asyncio.sleep(0)
        assert state.task is not None
        assert not state.task.done()

        # A second trigger while the first pass is still blocked on the gate.
        state.dirty = True
        state.start(_counted_pass)
        assert state.rerun_pending is True

        pending = asyncio.ensure_future(engine.pending_backfill)
        await asyncio.sleep(0)
        assert not pending.done(), "pending_backfill must not resolve mid-rerun"

        gate.set()
        await pending

        assert pass_count == 2
        assert state.rerun_pending is False
        assert state.task is None


class TestBackfillMidPassOutage:
    async def test_embedder_outage_on_second_row_counts_the_remainder_as_failed(self) -> None:
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        fault = FaultInjectingEmbedder(fake)
        engine = Tabularium(adapter, as_embedder(fault), clock, _cfg())
        await engine.initialize()

        f1 = _fact(content="mid-outage target one")
        f2 = _fact(content="mid-outage target two")
        f3 = _fact(content="mid-outage target three")
        for f in (f1, f2, f3):
            await adapter.upsert(f, None)

        from tabularium.embedding import EmbedderUnavailable
        from tabularium.engine.backfill import run_pass

        fault.fault(on="embed", call=2, exc=EmbedderUnavailable("mid-pass outage"))
        fault.arm()

        report = await run_pass(
            adapter, as_embedder(fault), engine._write_lock,
            embedding_model=fake.model, trigger="embed",
        )
        assert report.targets == 3
        assert report.filled == 1
        assert report.failed == 2

        fault.lift()
        second_report = await run_pass(
            adapter, as_embedder(fault), engine._write_lock,
            embedding_model=fake.model, trigger="embed",
        )
        assert second_report.filled == 2
        assert second_report.failed == 0


# ---------------------------------------------------------------------------
# CLI: tabularium backfill
# ---------------------------------------------------------------------------


class TestBackfillCli:
    async def test_cli_backfill_prints_filled_skipped_failed(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tabularium.cli import _cmd_backfill
        from tabularium.storage import open_adapter

        store_dir = str(tmp_path / "store")
        fake = FakeEmbedder(dim=_DIM)
        adapter = await open_adapter(
            "lancedb", embedding_model=fake.model, dim=_DIM, store_path=store_dir
        )
        await adapter.create_scope(Scope("global"))
        await adapter.upsert(_fact(content="cli backfill target"), None)

        cfg = StoreConfig(
            adapter="lancedb",
            path=store_dir,
            model=fake.model,
            dim=_DIM,
            api_base="http://localhost:11434",
        )
        rc = await _cmd_backfill(cfg, embedder=fake)

        out = capsys.readouterr().out
        assert rc == 0
        assert "filled=1 skipped=0 failed=0" in out

    async def test_cli_backfill_wires_api_base_env_into_embedder(
        self,
        tmp_path: pathlib.Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prior version built ``LitellmEmbedder(model=..., dim=...)`` with no ``api_base`` at
        all inside ``_cmd_backfill``, so ``TABULARIUM_API_BASE`` was silently ignored and the
        command always talked to litellm's default host (found by verification, P6). Exercises
        the real chain — ``config_from_env`` reading the env var into the command's
        ``StoreConfig``, then ``_cmd_backfill`` building the real ``LitellmEmbedder`` from it
        (``embedder=None``, the production path) — rather than asserting on either half in
        isolation.
        """
        import tabularium.cli as cli_module
        from tabularium.embedding import LitellmEmbedder
        from tabularium.storage import open_adapter

        sentinel = "http://sentinel-host:9999"
        monkeypatch.setenv("TABULARIUM_API_BASE", sentinel)
        monkeypatch.setenv("TABULARIUM_MODEL", "nomic-v2")
        monkeypatch.setenv("TABULARIUM_DIM", str(_DIM))

        captured: dict[str, object] = {}
        real_init = LitellmEmbedder.__init__

        def _capturing_init(self: LitellmEmbedder, **kwargs: Any) -> None:
            captured.update(kwargs)
            real_init(self, **kwargs)

        monkeypatch.setattr(LitellmEmbedder, "__init__", _capturing_init)

        store_dir = str(tmp_path / "store")
        await open_adapter("lancedb", embedding_model="nomic-v2", dim=_DIM, store_path=store_dir)

        cfg = config_from_env(path_override=store_dir, adapter_override="lancedb")
        rc = await cli_module._cmd_backfill(cfg)

        capsys.readouterr()
        assert rc == 0
        assert captured["api_base"] == sentinel

    async def test_cli_backfill_populates_legacy_content_hash_first(
        self, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A store predating the P1 schema piece has no ``content_hash``
        column until ``LanceDBAdapter.open()`` adds it as null; only
        ``Tabularium.initialize()``'s startup pass ever fills it in.
        ``tabularium backfill`` opens the adapter directly and never runs
        that pass —
        a stale-model row with a still-null ``content_hash`` must not crash
        the command before a single row gets embedded.
        """
        import shutil

        from tabularium.cli import _cmd_backfill

        fixture = pathlib.Path(__file__).parent / "fixtures" / "legacy-store-294fa14"
        dest = tmp_path / "legacy-store"
        shutil.copytree(fixture, dest)

        cfg = StoreConfig(
            adapter="lancedb",
            path=str(dest),
            model="nomic-v2",
            dim=8,
            api_base="http://localhost:11434",
        )
        fake = FakeEmbedder(dim=8, model="nomic-v2")
        rc = await _cmd_backfill(cfg, embedder=fake)

        out = capsys.readouterr().out
        assert rc == 0
        assert "filled=12 skipped=0 failed=0" in out


# ---------------------------------------------------------------------------
# Cross-connection safety: `tabularium backfill` opens its own LanceDBAdapter
# against a store a running server may hold open concurrently (P6 vector-
# less-rows piece). `backfill()`'s atomic conditional
# `table.update(where="id = ... AND content_hash = ...")` must make this
# safe: the predicate is evaluated against the true COMMITTED row at commit
# time, never a stale prior `query()` on this connection, and there is no
# whole-row fallback to accidentally clobber a concurrent edit.
# ---------------------------------------------------------------------------


class TestBackfillCrossConnectionSafety:
    async def test_atomic_update_noops_against_a_row_a_second_connection_already_edited(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The CLI's own guard read (a stale ``query()`` on its own
        connection) would spuriously see the pre-edit content and hash. The atomic conditional
        write must still evaluate against the SERVER's already-committed
        edit, not this connection's stale view, and no-op."""
        from tabularium.storage.content_key import content_hash

        clock = _clock()
        cli_adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model="test-model", dim=_DIM, clock=clock
        )
        fact = _fact(content="original content, landed vector-less")
        await cli_adapter.upsert(fact, None)
        stale_hash = content_hash(fact.content)

        server_adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model="test-model", dim=_DIM, clock=clock
        )
        await edit_content(server_adapter, fact.id, "server's edit, lands before backfill", None)

        ok = await cli_adapter.backfill(
            fact.id, [0.1] * _DIM, embedding_model="test-model", expected_hash=stale_hash
        )
        assert ok is False, (
            "the atomic conditional update must evaluate against the TRUE "
            "current row (the server's edit, a different hash), not this "
            "connection's stale pre-edit read — a spurious pass here would "
            "mean the staleness gap survived the fix"
        )

        after = await server_adapter.get_by_id(fact.id)
        assert after.content == "server's edit, lands before backfill"
        assert after.embedding_model is None

    async def test_write_conflict_during_update_propagates_rather_than_falling_back(
        self, tmp_path: pathlib.Path
    ) -> None:
        """No whole-row fallback:
        if ``table.update()`` itself raises, ``backfill()`` must let that
        exception propagate rather than rewriting a captured, possibly
        stale, whole row."""
        from tabularium.storage.content_key import content_hash

        clock = _clock()
        cli_adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model="test-model", dim=_DIM, clock=clock
        )
        fact = _fact(content="content whose update will be forced to raise")
        await cli_adapter.upsert(fact, None)
        expected_hash = content_hash(fact.content)

        real_update = raw_table(cli_adapter._records).update

        async def _raising_update(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("simulated LanceDB concurrent-write conflict")

        raw_table(cli_adapter._records).update = _raising_update
        try:
            with pytest.raises(RuntimeError, match="simulated LanceDB concurrent-write conflict"):
                await cli_adapter.backfill(
                    fact.id, [0.1] * _DIM, embedding_model="test-model",
                    expected_hash=expected_hash,
                )
        finally:
            raw_table(cli_adapter._records).update = real_update

        after = await cli_adapter.get_by_id(fact.id)
        assert after.content == "content whose update will be forced to raise"
        assert after.embedding_model is None


# ---------------------------------------------------------------------------
# Dirty-flag race: a pass that scans before a concurrent outage write
# commits must not erase that write's own dirty mark
# ---------------------------------------------------------------------------


class _DelayedReturnAdapter:
    """Delegates everything to ``inner`` except ``backfill_targets``, which
    runs the real scan immediately, then blocks on ``gate`` before handing
    the (already-computed) result back — manufactures a real interleaving
    window right after a scan, independent of whether it found anything."""

    def __init__(self, inner: MemoryAdapter, *, gate: asyncio.Event) -> None:
        self._inner = inner
        self._gate = gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def backfill_targets(self) -> Any:
        result = await self._inner.backfill_targets()
        await self._gate.wait()
        return result


class TestBackfillDirtyRace:
    async def test_pass_scanning_before_a_concurrent_write_commits_does_not_erase_its_mark(
        self,
    ) -> None:
        """The scan (``backfill_targets()``) is a snapshot taken with no
        lock held (D5/D6). A row that commits AFTER that snapshot was
        already taken, but before the pass finishes and does its
        end-of-run bookkeeping, must not have its own ``dirty`` mark
        erased by that bookkeeping — regression test. ``_DelayedReturnAdapter`` holds the
        ALREADY-COMPUTED (empty) scan result before handing it back, so
        the racing write commits strictly between the real scan and the
        pass receiving that result — the exact window the bug lived in."""
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        return_gate = asyncio.Event()
        return_gate.set()  # open during setup — only the actual test scan blocks
        delayed = _DelayedReturnAdapter(adapter, gate=return_gate)

        engine = Tabularium(delayed, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill
        return_gate.clear()
        assert engine._backfiller.dirty is False

        engine._backfiller.dirty = True
        engine._backfiller.schedule(trigger="embed")
        await asyncio.sleep(0)  # the pass's scan has run (found nothing) and is now blocked

        # The racing write: commits, then marks dirty itself — exactly the
        # order facts.py/decisions.py/edit.py use (write first, mark
        # after, no `await` in between) — while the pass above is still
        # holding its now-stale empty scan result.
        racer = _fact(content="racing write commits mid pass")
        await adapter.upsert(racer, None)
        engine._backfiller.dirty = True

        return_gate.set()
        await engine.pending_backfill

        assert engine._backfiller.dirty is True, (
            "the racing write's own post-commit mark must survive this "
            "pass's end-of-run bookkeeping, even though the pass's scan "
            "predates the write"
        )

        # A real trigger now finds and fills the still-vector-less racer.
        return_gate.set()
        engine._backfiller.schedule(trigger="embed")
        await engine.pending_backfill
        after = await adapter.get_by_id(racer.id)
        assert after.embedding_model == fake.model


# ---------------------------------------------------------------------------
# Existing-store migration: embedding_model must become nullable on a
# pre-P6 store, not only on a freshly created one
# ---------------------------------------------------------------------------


class TestExistingStoreMigratesEmbeddingModelNullable:
    async def test_legacy_store_accepts_a_vectorless_write_after_open(
        self, tmp_path: pathlib.Path
    ) -> None:
        import shutil

        import lancedb

        fixture = pathlib.Path(__file__).parent / "fixtures" / "legacy-store-294fa14"
        dest = tmp_path / "legacy-store"
        shutil.copytree(fixture, dest)

        pre_db = await lancedb.connect_async(str(dest))
        pre_schema = await (await pre_db.open_table("records")).schema()
        assert pre_schema.field("embedding_model").nullable is False, (
            "fixture drifted: this test needs a genuinely non-nullable "
            "embedding_model column to be a real regression check"
        )

        adapter = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=8)

        physical_schema = await adapter._records.schema()
        assert physical_schema.field("embedding_model").nullable is True

        # The actual regression: before the fix this raised a Lance
        # RuntimeError ("contained null values even though the field is
        # marked non-null") on any outage write against a pre-existing
        # store.
        landed = _fact(content="vectorless write against a migrated legacy store", scope="proj-a")
        await adapter.upsert(landed, None)
        after = await adapter.get_by_id(landed.id)
        assert after.embedding_model is None

        # Pre-existing rows are untouched.
        old_rows = await raw_table(adapter._records).query().to_list()
        stamped = [r for r in old_rows if r["id"] != str(landed.id)]
        assert stamped
        for row in stamped:
            assert row["embedding_model"] is not None

        # A second open (idempotent) does not fail or re-alter redundantly.
        adapter2 = await LanceDBAdapter.open(str(dest), embedding_model="test-model", dim=8)
        physical_schema2 = await adapter2._records.schema()
        assert physical_schema2.field("embedding_model").nullable is True


# ---------------------------------------------------------------------------
# Pass exception safety, real-trigger rerun, and edit's own dirty-mark
# timing
# ---------------------------------------------------------------------------


class _FlakyScanAdapter:
    """Delegates everything to ``inner``; ``backfill_targets`` raises once
    when ``should_fail`` is set, then clears the flag and delegates
    normally on every later call."""

    def __init__(self, inner: MemoryAdapter) -> None:
        self._inner = inner
        self.should_fail = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def backfill_targets(self) -> Any:
        if self.should_fail:
            self.should_fail = False
            raise RuntimeError("transient storage read failure")
        return await self._inner.backfill_targets()


class TestBackfillPassExceptionSafety:
    async def test_a_failed_scan_restores_dirty_for_the_next_trigger(self) -> None:
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        real_adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        flaky = _FlakyScanAdapter(real_adapter)
        engine = Tabularium(flaky, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        target = _fact(content="found after a transient scan failure")
        await real_adapter.upsert(target, None)
        engine._backfiller.dirty = True

        flaky.should_fail = True
        engine._backfiller.schedule(trigger="embed")
        with pytest.raises(RuntimeError, match="transient storage read failure"):
            await engine.pending_backfill

        assert engine._backfiller.dirty is True, (
            "a failed scan must restore dirty, not strand it False forever"
        )

        engine._backfiller.schedule(trigger="embed")
        await engine.pending_backfill
        after = await real_adapter.get_by_id(target.id)
        assert after.embedding_model == fake.model

    async def test_a_failed_pass_resets_task_so_a_later_pending_backfill_resolves_cleanly(
        self,
    ) -> None:
        """P6 vector-less-rows piece, state-hygiene half:
        before the fix, a failed pass left ``state.task`` pointing at the
        done-with-exception task forever, so a caller with no part in the
        failure who awaited ``pending_backfill`` afterward re-raised the
        SAME stale error indefinitely. The original awaiter must still see
        the failure; a later one must not."""
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        real_adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        flaky = _FlakyScanAdapter(real_adapter)
        engine = Tabularium(flaky, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        engine._backfiller.dirty = True
        flaky.should_fail = True
        engine._backfiller.schedule(trigger="embed")

        with pytest.raises(RuntimeError, match="transient storage read failure"):
            await engine.pending_backfill

        assert engine._backfiller.task is None, (
            "a failed pass must reset state.task, never strand a "
            "done-with-exception task in place"
        )

        await engine.pending_backfill  # a later caller must not re-raise the stale error


class _ScanBlocksThenFailsOnNthCall:
    """Delegates everything to ``inner``; the ``fail_on_call``-th call to
    ``backfill_targets`` blocks on ``gate`` before raising — every other
    call delegates normally. Manufactures the exact window the original report names: a rerun
    requested WHILE a pass's own scan is in
    flight, immediately before that same scan fails."""

    def __init__(self, inner: MemoryAdapter, *, fail_on_call: int, gate: asyncio.Event) -> None:
        self._inner = inner
        self._fail_on_call = fail_on_call
        self._gate = gate
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def backfill_targets(self) -> Any:
        self._calls += 1
        if self._calls == self._fail_on_call:
            await self._gate.wait()
            raise RuntimeError("simulated scan failure while a rerun is already pending")
        return await self._inner.backfill_targets()


class TestBackfillRerunPendingAtFailureIsRescheduled:
    async def test_rerun_queued_during_a_failing_scan_still_fills_the_row_unattended(
        self,
    ) -> None:
        """T4 (P6 vector-less-rows piece): a rerun requested
        while the CURRENT pass's own scan is still in flight, then lost
        when that same scan fails, must not be abandoned — it is
        rescheduled from ``_run_and_absorb_reruns``'s own ``finally``, so
        the row it exists to catch is filled without any further external
        trigger. ``_ScanBlocksThenFailsOnNthCall``'s call 1 is the startup
        sweep (empty store, delegates normally); call 2 is the pass under
        test, whose scan blocks on ``gate`` then fails once released."""
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        real_adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        gate = asyncio.Event()
        flaky = _ScanBlocksThenFailsOnNthCall(real_adapter, fail_on_call=2, gate=gate)
        engine = Tabularium(flaky, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        r1 = _fact(content="row present when the failing scan started")
        await real_adapter.upsert(r1, None)
        engine._backfiller.dirty = True
        engine._backfiller.schedule(trigger="embed")
        await asyncio.sleep(0)

        assert engine._backfiller.task is not None
        assert not engine._backfiller.task.done()

        r2 = _fact(content="row that lands after the scan started, wants the rerun")
        await real_adapter.upsert(r2, None)
        engine._backfiller.dirty = True
        engine._backfiller.schedule(trigger="embed")
        assert engine._backfiller.rerun_pending is True

        gate.set()

        with pytest.raises(RuntimeError, match="simulated scan failure"):
            await engine.pending_backfill

        await engine.pending_backfill

        after_r1 = await real_adapter.get_by_id(r1.id)
        after_r2 = await real_adapter.get_by_id(r2.id)
        assert after_r1.embedding_model == fake.model, (
            "call 2's own scan raised before ever returning a target list, "
            "so r1 was never processed by it either — the rescheduled "
            "pass's fresh scan (call 3) must re-discover and fill it too"
        )
        assert after_r2.embedding_model == fake.model, (
            "the rescheduled pass's own scan (call 3) must find and fill r2 "
            "with no further trigger from the test"
        )
        assert engine._backfiller.rerun_pending is False
        assert engine._backfiller.task is None


class TestBackfillRealTriggerDuringActivePass:
    async def test_a_real_save_while_a_pass_is_active_sets_rerun_not_dropped(self) -> None:
        fake = FakeEmbedder(dim=_DIM)
        clock = _clock()
        adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        engine = Tabularium(adapter, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        decoy = _fact(content="decoy embedded slowly")
        await adapter.upsert(decoy, None)

        pass_gate = asyncio.Event()
        blocking = _BlockingEmbedder(fake, gate=pass_gate)
        engine._backfiller.dirty = True
        engine._embedder = blocking
        engine._backfiller._embedder = blocking
        engine._backfiller.schedule(trigger="embed")
        await asyncio.sleep(0)  # the pass clears dirty and blocks embedding the decoy

        assert engine._backfiller.dirty is False
        assert engine._backfiller.task is not None
        assert not engine._backfiller.task.done()

        # Restore the real embedder for the SAVE below — the already-running
        # pass captured `blocking` when it called `run_pass`, so this does
        # not affect it.
        engine._embedder = fake
        engine._backfiller._embedder = fake
        save_task = asyncio.ensure_future(
            engine.save("real trigger while a pass is active", scope="global")
        )
        saved = await save_task
        assert saved.embedded is True

        assert engine._backfiller.rerun_pending is True, (
            "a real successful-embed trigger arriving while a pass is "
            "active must set rerun_pending, never be silently dropped "
            "because `dirty` currently reads False"
        )

        pass_gate.set()
        await engine.pending_backfill
        assert engine._backfiller.rerun_pending is False

        after = await adapter.get_by_id(decoy.id)
        assert after.embedding_model == fake.model


# ---------------------------------------------------------------------------
# Save's dirty mark must survive a cancellation during post-commit
# maintenance, and the CLI backfill command must populate a legacy store's
# content_hash before scanning it
# ---------------------------------------------------------------------------


class _BlockingMaintenanceAdapter:
    """Delegates everything to ``inner``; ``maintenance_debt`` signals
    ``entered`` then blocks on ``gate`` — lets a test cancel the caller
    while it is parked inside ``maybe_upkeep``, after the row's own write
    has already committed."""

    def __init__(
        self, inner: MemoryAdapter, *, gate: asyncio.Event, entered: asyncio.Event
    ) -> None:
        self._inner = inner
        self._gate = gate
        self._entered = entered

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def maintenance_debt(self) -> Any:
        self._entered.set()
        await self._gate.wait()
        return await self._inner.maintenance_debt()


class TestSaveMarksDirtyBeforeMaintenanceCanCancel:
    async def test_save_marks_dirty_even_when_maintenance_is_cancelled(self) -> None:
        fake = FakeEmbedder(dim=_DIM, should_fail=True)
        clock = _clock()
        real_adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        gate = asyncio.Event()
        entered = asyncio.Event()
        blocking = _BlockingMaintenanceAdapter(real_adapter, gate=gate, entered=entered)
        engine = Tabularium(blocking, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        save_task = asyncio.ensure_future(
            engine.save("fact under outage, cancelled mid-maintenance", scope="global")
        )
        await entered.wait()
        save_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await save_task

        assert engine._backfiller.dirty is True, (
            "the vector-less write's own commit must mark dirty even though "
            "a later step (maintenance) never got to finish"
        )

    async def test_save_decision_marks_dirty_even_when_maintenance_is_cancelled(self) -> None:
        fake = FakeEmbedder(dim=_DIM, should_fail=True)
        clock = _clock()
        real_adapter = MemoryAdapter(embedding_model=fake.model, dim=_DIM, clock=clock)
        gate = asyncio.Event()
        entered = asyncio.Event()
        blocking = _BlockingMaintenanceAdapter(real_adapter, gate=gate, entered=entered)
        engine = Tabularium(blocking, fake, clock, _cfg())
        await engine.initialize()
        await engine.pending_backfill

        save_task = asyncio.ensure_future(
            engine.save(
                "decision under outage, cancelled mid-maintenance",
                kind="decision",
                scope="global",
            )
        )
        await entered.wait()
        save_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await save_task

        assert engine._backfiller.dirty is True, (
            "the vector-less decision write's own commit must mark dirty "
            "even though a later step (maintenance) never got to finish"
        )
