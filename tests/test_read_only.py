"""Maintenance-write freeze (``TABULARIUM_READ_ONLY`` / ``EngineConfig.read_only``).

Three surfaces: the engine's ``initialize()`` (no scheduler, no startup writes), ``recall`` (no
revival of an archived hit), and the ``tabularium clean`` refusal. Default mode is asserted
unchanged beside each. The server lists the same four read tools in either mode; that is pinned in
``tests/test_server.py``.
"""

from __future__ import annotations

import asyncio
import pathlib
from datetime import UTC, datetime

import pytest

from tabularium.cli import CliError, _cmd_clean
from tabularium.config import EngineConfig, StoreConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium
from tabularium.server import engine_config_from_env
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

_DIM = 16


def _engine(
    *, read_only: bool = False, enable_digest_cleaner: bool = False
) -> tuple[Tabularium, MemoryAdapter, FakeEmbedder]:
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    config = EngineConfig(
        candidate_k=1000,
        min_active_results=2,
        read_only=read_only,
        enable_digest_cleaner=enable_digest_cleaner,
    )
    engine = Tabularium(adapter, embedder, clock, config, token_counter=FakeTokenCounter())
    return engine, adapter, embedder


class TestEnv:
    def test_unset_means_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TABULARIUM_READ_ONLY", raising=False)

        assert engine_config_from_env().read_only is False

    @pytest.mark.parametrize("spelling", ["1", "true", "yes"])
    def test_truthy_spellings_turn_it_on(
        self, monkeypatch: pytest.MonkeyPatch, spelling: str
    ) -> None:
        monkeypatch.setenv("TABULARIUM_READ_ONLY", spelling)

        assert engine_config_from_env().read_only is True

    def test_zero_keeps_it_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TABULARIUM_READ_ONLY", "0")

        assert engine_config_from_env().read_only is False


class TestScheduler:
    async def test_read_only_never_starts_the_nightly_loop(self) -> None:
        engine, _adapter, _embedder = _engine(read_only=True, enable_digest_cleaner=True)

        await engine.initialize()

        assert engine._nightly_task is None

    async def test_cleaner_flag_alone_still_starts_the_loop(self) -> None:
        engine, _adapter, _embedder = _engine(enable_digest_cleaner=True)

        await engine.initialize()
        task = engine._nightly_task

        assert task is not None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_read_only_initialize_writes_no_registry_rows(self) -> None:
        engine, adapter, _embedder = _engine(read_only=True)

        await engine.initialize()

        assert await adapter.list_scopes() == []
        assert await adapter.list_roles() == []


class TestRecallRevival:
    async def _archived_store(
        self, *, read_only: bool = False
    ) -> tuple[Tabularium, MemoryAdapter, Fact]:
        engine, adapter, embedder = _engine(read_only=read_only)
        await adapter.create_scope(Scope("global"))
        now = datetime(2025, 1, 1, tzinfo=UTC)
        fact = Fact(
            id=new_id(),
            type="fact",
            content="archived useful knowledge",
            scope=Scope("global"),
            lifecycle="archived",
            created_at=now,
            updated_at=now,
        )
        await adapter.upsert(fact, embedder.peek(fact.content))
        return engine, adapter, fact

    async def test_default_recall_revives_an_archived_hit(self) -> None:
        engine, adapter, fact = await self._archived_store()

        await engine.recall("archived useful knowledge", scope="global")

        assert await adapter.query_events(target_id=fact.id) != []

    async def test_read_only_recall_returns_the_hit_and_writes_no_event(self) -> None:
        engine, adapter, fact = await self._archived_store(read_only=True)

        result = await engine.recall("archived useful knowledge", scope="global")

        assert [h.id for h in result.hits] == [fact.id]
        assert await adapter.query_events(target_id=fact.id) == []
        assert (await adapter.get_by_id(fact.id)).lifecycle == "archived"


def _store(path: pathlib.Path) -> StoreConfig:
    return StoreConfig(
        adapter="lancedb",
        path=str(path),
        model="fake/nomic",
        dim=_DIM,
        api_base="http://localhost:11434",
    )


class TestCleanCli:
    @pytest.mark.parametrize("dry_run", [True, False])
    async def test_clean_refuses_and_opens_no_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, dry_run: bool
    ) -> None:
        monkeypatch.setenv("TABULARIUM_READ_ONLY", "1")
        store_dir = tmp_path / "store"

        with pytest.raises(CliError, match="TABULARIUM_READ_ONLY"):
            await _cmd_clean("proj-a", _store(store_dir), dry_run=dry_run)

        assert not store_dir.exists()

    async def test_clean_history_still_reads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setenv("TABULARIUM_READ_ONLY", "1")

        rc = await _cmd_clean("proj-a", _store(tmp_path / "store"), dry_run=False, history=True)

        assert rc == 0

    async def test_clean_is_not_refused_when_the_flag_is_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setenv("TABULARIUM_READ_ONLY", "0")

        with pytest.raises(CliError, match="Unregistered scope") as excinfo:
            await _cmd_clean("proj-a", _store(tmp_path / "store"), dry_run=True)

        assert "TABULARIUM_READ_ONLY" not in str(excinfo.value)
