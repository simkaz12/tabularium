"""CLI tests — ``tabularium sync`` + registry verbs (B4, SPEC §13).

Uses a tmp LanceDB store (via open_adapter) to exercise the real adapter
path.  All async helpers run under pytest-asyncio.

Acceptance coverage
-------------------
  sync: writes one file per decision
  sync: second run writes nothing (byte-idempotent)
  sync: a live supersession changes exactly the two expected files
  sync: buried decision's file is removed
  registry: scope add registers (visible via list) and is idempotent
  registry: facet add registers and is idempotent
  registry: role add registers and is idempotent
"""

from __future__ import annotations

import json
import math
import pathlib
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.cli import CliError
from tabularium.config import EngineConfig, StoreConfig
from tabularium.domain import HypomnemaId, Scope
from tabularium.engine import Tabularium
from tabularium.storage import open_adapter
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.types import CleanerState
from tests.support.doubles import FakeEmbedder, FakeJudgeAdapter, TestClock
from tests.support.fake_assessor import EchoAssessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 16


async def _make_lancedb_engine(
    store_path: str,
) -> tuple[Tabularium, LanceDBAdapter]:
    """Open a real LanceDB adapter + engine at ``store_path``."""
    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_path,
    )
    assert isinstance(adapter, LanceDBAdapter)
    embedder = FakeEmbedder(dim=_DIM)
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(candidate_k=1000, gray_floor=0.80)
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    return engine, adapter


def _cfg(store_path: str) -> StoreConfig:
    return StoreConfig(
        adapter="lancedb",
        path=store_path,
        model="fake/nomic",
        dim=_DIM,
        api_base="http://localhost:11434",
    )


# ---------------------------------------------------------------------------
# sync: one file per decision
# ---------------------------------------------------------------------------


async def test_sync_writes_one_file_per_decision(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    out_dir = tmp_path / "decisions"
    engine, _ = await _make_lancedb_engine(store_dir)

    await engine.save(
        "Use pydantic v2 for validation.", kind="decision", scope="global"
    )
    await engine.save(
        "Use LanceDB as the vector store.", kind="decision", scope="global"
    )

    from tabularium.cli import _cmd_sync

    rc = await _cmd_sync(out_dir, _cfg(store_dir))
    assert rc == 0

    md_files = list(out_dir.glob("*.md"))
    assert len(md_files) == 2, f"Expected 2 .md files, got {len(md_files)}"
    for f in md_files:
        assert f.read_text(encoding="utf-8").strip() != ""


# ---------------------------------------------------------------------------
# sync: second run writes nothing (byte-idempotent)
# ---------------------------------------------------------------------------


async def test_sync_byte_idempotent(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    out_dir = tmp_path / "decisions"
    engine, _ = await _make_lancedb_engine(store_dir)

    await engine.save("Prefer stdlib over third-party.", kind="decision", scope="global")

    from tabularium.cli import _cmd_sync

    await _cmd_sync(out_dir, _cfg(store_dir))

    # Record mtimes
    mtimes_before = {f: f.stat().st_mtime_ns for f in out_dir.glob("*.md")}

    # Second run
    await _cmd_sync(out_dir, _cfg(store_dir))

    mtimes_after = {f: f.stat().st_mtime_ns for f in out_dir.glob("*.md")}

    assert mtimes_before == mtimes_after, (
        "Second sync must not touch files whose bytes have not changed"
    )


# ---------------------------------------------------------------------------
# sync: supersession changes exactly the two expected files
# ---------------------------------------------------------------------------


async def test_sync_supersession_changes_two_files(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    out_dir = tmp_path / "decisions"
    engine, _ = await _make_lancedb_engine(store_dir)

    # Save original decision
    r1 = await engine.save(
        "Use MemoryAdapter for tests.", kind="decision", scope="global"
    )
    assert r1.hypomnema is not None
    orig_id = str(r1.hypomnema.id)

    from tabularium.cli import _cmd_sync

    # First sync — establish baseline
    await _cmd_sync(out_dir, _cfg(store_dir))
    orig_bytes_before = (out_dir / f"{orig_id}.md").read_bytes()

    # Supersede the original decision
    r2 = await engine.save(
        "Use MemoryAdapter for tests; LanceDB in production.",
        kind="decision",
        scope="global",
        supersedes=HypomnemaId(uuid.UUID(orig_id)),
        reason="changed",
    )
    assert r2.hypomnema is not None
    new_id = str(r2.hypomnema.id)

    # Second sync
    await _cmd_sync(out_dir, _cfg(store_dir))

    # Original file should be updated (now shows lifecycle=superseded)
    orig_bytes_after = (out_dir / f"{orig_id}.md").read_bytes()
    assert orig_bytes_before != orig_bytes_after, (
        "Original decision file must change after supersession"
    )

    # New decision file must exist
    new_file = out_dir / f"{new_id}.md"
    assert new_file.exists(), "Superseding decision file must be created"


# ---------------------------------------------------------------------------
# sync: buried decision's file is removed
# ---------------------------------------------------------------------------


async def test_sync_removes_buried_file(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    out_dir = tmp_path / "decisions"
    engine, adapter = await _make_lancedb_engine(store_dir)

    r = await engine.save(
        "Temporary decision to be buried.", kind="decision", scope="global"
    )
    assert r.hypomnema is not None
    buried_id = str(r.hypomnema.id)

    from tabularium.cli import _cmd_sync

    # First sync — file exists
    await _cmd_sync(out_dir, _cfg(store_dir))
    assert (out_dir / f"{buried_id}.md").exists()

    # Bury the decision
    await adapter.bury(HypomnemaId(uuid.UUID(buried_id)), reason="test cleanup")

    # Second sync — file must be removed
    await _cmd_sync(out_dir, _cfg(store_dir))
    assert not (out_dir / f"{buried_id}.md").exists(), (
        "Buried decision file must be removed by sync"
    )


# ---------------------------------------------------------------------------
# sync: exported render carries a revalidation line (P9 invalid-Decision
# read contract) — must match engine.decision(id).rendered byte-for-byte
# ---------------------------------------------------------------------------


def _angle_vec(angle_deg: float, dim: int = _DIM) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (dim - 2)


class _AngleEmbedder:
    dim = _DIM
    model = "fake/angle"

    def __init__(self, table: dict[str, float]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return _angle_vec(self._table[text])


async def test_sync_exports_revalidation_line(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    out_dir = tmp_path / "decisions"

    decision_text = "the quick brown fox jumps over the lazy dog"
    fact_text = "the quick brown fox jumps over the lazy hound"

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/angle",
        dim=_DIM,
        store_path=store_dir,
    )
    embedder = _AngleEmbedder({decision_text: 0.0, fact_text: 5.0})
    clock = TestClock(datetime(2025, 1, 1, tzinfo=UTC))
    cfg = EngineConfig(
        candidate_k=1000,
        gray_floor=0.80,
        decision_gray_floor=0.70,
        auto_exit_replacement=True,
    )
    engine = Tabularium(adapter, embedder, clock, cfg, assessor=EchoAssessor("correction"))
    await engine.initialize()

    d = await engine.save(decision_text, kind="decision", scope="global")
    assert d.hypomnema is not None
    d_id = d.hypomnema.id

    fact = await engine.save(fact_text, scope="global")
    assert fact.hypomnema is not None
    assert fact.outcome == "saved"

    revised = await engine.set_validity(d_id, valid=True, reason="reviewed")
    assert revised.outcome == "revised"

    from tabularium.cli import _cmd_sync

    rc = await _cmd_sync(out_dir, _cfg(store_dir))
    assert rc == 0

    exported = (out_dir / f"{d_id}.md").read_text(encoding="utf-8")
    view = await engine.decision(d_id)
    assert exported == view.rendered
    assert "Revalidated" in exported


# ---------------------------------------------------------------------------
# registry: scope add — registers and is idempotent
# ---------------------------------------------------------------------------


async def test_scope_add_registers(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    rc = await _cmd_registry_add("scope", "myproject", cfg)
    assert rc == 0

    # Verify via adapter
    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    scopes = [str(s) for s in await adapter.list_scopes()]
    assert "myproject" in scopes


async def test_scope_add_idempotent(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    rc1 = await _cmd_registry_add("scope", "myproject", cfg)
    rc2 = await _cmd_registry_add("scope", "myproject", cfg)
    assert rc1 == 0
    assert rc2 == 0

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    scopes = [str(s) for s in await adapter.list_scopes()]
    # Must appear only once
    assert scopes.count("myproject") == 1


async def test_scope_add_invalid_name_returns_error(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    with pytest.raises(CliError, match="invalid scope name 'BadScope!'"):
        await _cmd_registry_add("scope", "BadScope!", cfg)


# ---------------------------------------------------------------------------
# registry: facet add
# ---------------------------------------------------------------------------


async def test_facet_add_registers(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    rc = await _cmd_registry_add("facet", "finding", cfg)
    assert rc == 0

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    facets = await adapter.list_facets()
    assert "finding" in facets


async def test_facet_add_idempotent(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    await _cmd_registry_add("facet", "finding", cfg)
    await _cmd_registry_add("facet", "finding", cfg)

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    facets = await adapter.list_facets()
    assert facets.count("finding") == 1


# ---------------------------------------------------------------------------
# registry: role add
# ---------------------------------------------------------------------------


async def test_role_add_registers(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    rc = await _cmd_registry_add("role", "implementer", cfg)
    assert rc == 0

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    roles = await adapter.list_roles()
    assert "implementer" in roles


async def test_role_add_idempotent(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_registry_add

    await _cmd_registry_add("role", "implementer", cfg)
    await _cmd_registry_add("role", "implementer", cfg)

    adapter = await open_adapter(
        "lancedb",
        embedding_model="fake/nomic",
        dim=_DIM,
        store_path=store_dir,
    )
    roles = await adapter.list_roles()
    assert roles.count("implementer") == 1


# ---------------------------------------------------------------------------
# reclaim: irreversible, named door — confirmation gate (B2, SPEC §13)
# ---------------------------------------------------------------------------


async def test_reclaim_confirmed(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """reclaim() is called on the adapter when the interactive prompt is confirmed."""
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_reclaim, _cmd_registry_add

    # Ensure the store exists before reclaim opens it.
    await _cmd_registry_add("scope", "myproject", cfg)

    calls = {"n": 0}
    original_reclaim = LanceDBAdapter.reclaim

    async def _spy_reclaim(self: LanceDBAdapter) -> None:
        calls["n"] += 1
        return await original_reclaim(self)

    monkeypatch.setattr(LanceDBAdapter, "reclaim", _spy_reclaim)
    monkeypatch.setattr("builtins.input", lambda *_a: "y")

    rc = await _cmd_reclaim(cfg, yes=False)
    assert rc == 0
    assert calls["n"] == 1


async def test_reclaim_declined(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """reclaim() is NOT called when the interactive prompt is declined."""
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_reclaim, _cmd_registry_add

    await _cmd_registry_add("scope", "myproject", cfg)

    calls = {"n": 0}
    original_reclaim = LanceDBAdapter.reclaim

    async def _spy_reclaim(self: LanceDBAdapter) -> None:
        calls["n"] += 1
        return await original_reclaim(self)

    monkeypatch.setattr(LanceDBAdapter, "reclaim", _spy_reclaim)
    monkeypatch.setattr("builtins.input", lambda *_a: "n")

    rc = await _cmd_reclaim(cfg, yes=False)
    assert rc != 0
    assert calls["n"] == 0


async def test_reclaim_yes_flag(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """--yes skips the interactive prompt entirely and still calls reclaim()."""
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_reclaim, _cmd_registry_add

    await _cmd_registry_add("scope", "myproject", cfg)

    calls = {"n": 0}
    original_reclaim = LanceDBAdapter.reclaim

    async def _spy_reclaim(self: LanceDBAdapter) -> None:
        calls["n"] += 1
        return await original_reclaim(self)

    monkeypatch.setattr(LanceDBAdapter, "reclaim", _spy_reclaim)

    def _fail_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() must not be called when --yes is passed")

    monkeypatch.setattr("builtins.input", _fail_input)

    rc = await _cmd_reclaim(cfg, yes=True)
    assert rc == 0
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# doctor: runs, exits 0, deterministic output (SPEC §13, M2 slice B)
# ---------------------------------------------------------------------------


async def test_doctor_runs_and_exits_zero_on_empty_store(
    tmp_path: pathlib.Path,
) -> None:
    """``doctor`` observes; it never polices — always exit 0, even on an
    empty, never-initialized store.
    """
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_doctor

    rc = await _cmd_doctor(cfg)
    assert rc == 0


async def test_doctor_output_is_non_vacuous_and_names_all_four_sections(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """Every section prints something, never a bare/empty heading — the
    acceptance criterion that a section with nothing to report SAYS so.
    """
    store_dir = str(tmp_path / "store")
    engine, _ = await _make_lancedb_engine(store_dir)
    await engine.save("some content", scope="global", purpose="constraint")

    from tabularium.cli import _cmd_doctor

    cfg = _cfg(store_dir)
    rc = await _cmd_doctor(cfg)
    assert rc == 0

    out = capsys.readouterr().out
    assert "1. Facet coverage" in out
    assert "2. Dead-vocabulary flags" in out
    assert "3. Rows awaiting embedding" in out
    assert "4. Projection drift" in out
    # Non-vacuous: real numbers appear, not an empty heading.
    assert "scanned (denominator): 1 live records" in out


async def test_doctor_output_is_deterministic_across_runs(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """Two runs against an unchanged store produce byte-identical output —
    no timestamp/ordering churn (the acceptance criterion).
    """
    store_dir = str(tmp_path / "store")
    engine, _ = await _make_lancedb_engine(store_dir)
    await engine.save("some content", scope="global", purpose="constraint")
    await engine.save("more content", scope="global")

    from tabularium.cli import _cmd_doctor

    cfg = _cfg(store_dir)
    await _cmd_doctor(cfg)
    first = capsys.readouterr().out
    await _cmd_doctor(cfg)
    second = capsys.readouterr().out

    assert first == second
    assert first != ""


# ---------------------------------------------------------------------------
# supersede
# ---------------------------------------------------------------------------


async def test_supersede_draws_the_edge(tmp_path: pathlib.Path, capsys: Any) -> None:
    """Drawing an edge by hand: the predecessor ends up invalid and pointed
    at the successor, exit code 0, a confirming line printed.

    Verifies via a FRESHLY opened adapter connection, never the one
    ``_make_lancedb_engine`` already holds open — ``_cmd_supersede`` opens
    its own second connection against the same store path, and a second
    LanceDB handle does not see a first handle's writes for free
    (``LanceDBAdapter``'s own documented cross-connection staleness note);
    the CLI's OWN process always opens exactly one connection in practice,
    this dual-connection shape is a test-only artifact of driving two CLI
    commands against one store in a single process.
    """
    store_dir = str(tmp_path / "store")
    engine, _first_adapter = await _make_lancedb_engine(store_dir)
    pred = await engine.save("predecessor content", scope="global")
    succ = await engine.save("successor content", scope="global")
    assert pred.hypomnema is not None
    assert succ.hypomnema is not None

    from tabularium.cli import _cmd_supersede

    cfg = _cfg(store_dir)
    rc = await _cmd_supersede(
        str(pred.hypomnema.id), str(succ.hypomnema.id), "correction", None, cfg
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "supersede:" in out

    fresh_adapter = await open_adapter(
        "lancedb", embedding_model="fake/nomic", dim=_DIM, store_path=store_dir,
    )
    updated = await fresh_adapter.get_by_id(pred.hypomnema.id)
    assert updated.validity == "invalid"
    assert updated.superseded_by == succ.hypomnema.id


async def test_supersede_invalid_uuid_returns_error(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    cfg = _cfg(store_dir)

    from tabularium.cli import _cmd_supersede

    with pytest.raises(CliError, match="predecessor 'not-a-uuid' is not a valid UUID"):
        await _cmd_supersede("not-a-uuid", str(uuid.uuid4()), "correction", None, cfg)


async def test_supersede_unknown_id_returns_error(tmp_path: pathlib.Path) -> None:
    store_dir = str(tmp_path / "store")
    engine, _ = await _make_lancedb_engine(store_dir)
    succ = await engine.save("successor content", scope="global")
    assert succ.hypomnema is not None

    from tabularium.cli import _cmd_supersede

    cfg = _cfg(store_dir)
    with pytest.raises(CliError, match="supersedes: unknown id"):
        await _cmd_supersede(
            str(uuid.uuid4()), str(succ.hypomnema.id), "correction", None, cfg
        )


# ---------------------------------------------------------------------------
# clean (2026-08-10 cleaner plan)
# ---------------------------------------------------------------------------


def _dir_snapshot(path: pathlib.Path, *, exclude_prefix: str | None = None) -> dict[str, bytes]:
    """Every file under ``path``, by relative path -> bytes — used to assert a LanceDB store is
    byte-for-byte unchanged (acceptance criterion: "--dry-run ... Assert the store is
    byte-unchanged"). Comparing the two dicts catches an added file, a removed file, or a
    modified file equally, unlike comparing only a file count or a total size.

    ``exclude_prefix`` (cleaner-telemetry change) drops any relative path
    starting with it — used to carve the ``pass_history`` table itself out of the "byte-unchanged"
    claim: the earlier decision list is explicit that "a dry run writes exactly one thing: its
    history row", so the ORIGINAL pre-existing "zero bytes touched, anywhere" claim this helper
    enforced is no longer true of the WHOLE store — only of everything except that one table now.
    """
    return {
        str(f.relative_to(path)): f.read_bytes()
        for f in sorted(path.rglob("*"))
        if f.is_file()
        and (exclude_prefix is None or not str(f.relative_to(path)).startswith(exclude_prefix))
    }


def _shelve_verdict_payload() -> str:
    return json.dumps(
        {"verdicts": [{"index": 1, "grade": "shelve-clear", "unclear": False, "why": "stale"}]}
    )


def _shelve_judges() -> tuple[FakeJudgeAdapter, FakeJudgeAdapter]:
    """A relevance panel that unanimously shelves whatever single-record batch it is shown — used
    so a dry run has something real to report ("would shelve") and a live run has something real
    to write, rather than exercising a no-op pass either way."""
    payload = _shelve_verdict_payload()
    return (
        FakeJudgeAdapter(identity="judge-a", response=payload),
        FakeJudgeAdapter(identity="judge-b", response=payload),
    )


_CLEAN_T0 = datetime(2024, 12, 31, tzinfo=UTC)  # strictly before the engine's own TestClock start


async def _seed_clean_scope(store_dir: str) -> tuple[Tabularium, HypomnemaId]:
    """Register scope 'proj-a', seed a Cleaner watermark at ``_CLEAN_T0``, and save one Fact born
    after it (the engine's own ``TestClock`` starts 2025-01-01, strictly after ``_CLEAN_T0``) —
    the shared setup every clean CLI test below builds on."""
    engine, adapter = await _make_lancedb_engine(store_dir)
    await adapter.create_scope(Scope("proj-a"))
    await adapter.set_cleaner_state(
        scope=Scope("proj-a"),
        state=CleanerState(watermark_at=_CLEAN_T0, last_pass_at=_CLEAN_T0, report=None),
    )
    r = await engine.save("a fact the panel will vote to shelve", scope="proj-a")
    assert r.hypomnema is not None
    return engine, r.hypomnema.id


async def test_clean_dry_run_writes_nothing_and_reports_what_it_would_do(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """The real falsifiability target (Verify step 1's break, 2026-08-10 cleaner plan): a
    dry run against a real judge panel and a real pool must leave the on-disk store byte-for-byte
    identical OUTSIDE the pass-history table, and must never move the watermark — while still
    reporting the real verdict the panel reached.

    The "byte-for-byte identical" claim is now scoped to exclude ``pass_history`` (2026-08-11
    cleaner-telemetry change): that task's own decision list is explicit that "a dry run
    writes exactly one thing: its history row" — verified NOT by assumption but by first running
    this test with the ORIGINAL unscoped ``before == after`` assertion and observing it fail on
    exactly (and only) ``pass_history.lance/*``, with all 93 other on-disk files identical; see
    the task verdict for the captured diff.
    """
    store_dir = str(tmp_path / "store")
    _engine, fact_id = await _seed_clean_scope(store_dir)

    from tabularium.cli import _cmd_clean

    store_path = pathlib.Path(store_dir)
    before = _dir_snapshot(store_path, exclude_prefix="pass_history.lance")

    judge_a, judge_b = _shelve_judges()
    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=True,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=judge_a,
        judge_b=judge_b,
    )
    assert rc == 0

    after = _dir_snapshot(store_path, exclude_prefix="pass_history.lance")
    assert before == after, (
        "dry-run must not write a single byte to the store outside pass_history"
    )

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "1 would shelve" in out
    assert _CLEAN_T0.isoformat() in out  # unmoved watermark, printed verbatim
    # review follow-up (2026-08-11): outcome:/watermark: must not read identically to a live
    # pass's — a human skimming past the mode: banner must still see the qualifier.
    assert "watermark (would move):" in out
    assert "outcome: completed (dry run — UNVERIFIED" in out
    # 2026-08-12 rubric-gate-retune fix (verifiability): a dry run must report WHICH record
    # would flip and WHY, not just a count — `_shelve_verdict_payload`'s judge `why` is "stale".
    assert "would shelve:" in out
    assert str(fact_id) in out
    assert "stale" in out

    fresh = await open_adapter(
        "lancedb", embedding_model="fake/nomic", dim=_DIM, store_path=store_dir
    )
    state = await fresh.get_cleaner_state(scope=Scope("proj-a"))
    assert state is not None
    assert state.watermark_at == _CLEAN_T0  # unmoved
    assert state.report is None  # never written

    stored = await fresh.get_by_id(fact_id)
    assert stored.lifecycle == "active"  # never shelved

    # The ONE thing a dry run does write (cleaner-telemetry change acceptance
    # criterion): its own pass-history row, flagged.
    history = await fresh.list_cleaner_passes(scope=Scope("proj-a"), limit=10)
    assert len(history) == 1
    assert history[0].dry_run is True
    assert history[0].outcome == "completed"


async def test_clean_honours_engine_config_env_vars_not_just_defaults(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    """A hand-run ``tabularium clean``
    must build the SAME ``EngineConfig`` ``server.main()``'s own digest-triggered pass does
    (``engine_config_from_env()``) -- a bare ``EngineConfig()`` here would silently ignore every
    Cleaner env var, so `TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S`/`TABULARIUM_CLEANER_SPLIT_SHELVE`
    would configure the digest-triggered pass but never a hand-run one."""
    store_dir = str(tmp_path / "store")
    await _seed_clean_scope(store_dir)

    monkeypatch.setenv("TABULARIUM_ASSESSOR_CLEANER_TIMEOUT_S", "42")
    monkeypatch.setenv("TABULARIUM_CLEANER_SPLIT_SHELVE", "true")

    from tabularium.cli import _cmd_clean
    from tabularium.engine import report as report_mod

    real_run_clean_pass = report_mod.run_clean_pass
    captured: dict[str, Any] = {}

    async def _spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return await real_run_clean_pass(**kwargs)

    monkeypatch.setattr(report_mod, "run_clean_pass", _spy)

    judge_a, judge_b = _shelve_judges()
    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=True,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=judge_a,
        judge_b=judge_b,
    )

    assert rc == 0
    assert captured["cfg"].assessor_cleaner_timeout_s == 42.0
    assert captured["cfg"].cleaner_split_shelve is True


async def test_clean_live_run_writes_and_prints_warning(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """Without --dry-run: the concurrent-write warning prints first, the verdict actually lands,
    and the watermark actually advances — the real end-to-end path (acceptance criteria)."""
    store_dir = str(tmp_path / "store")
    _engine, fact_id = await _seed_clean_scope(store_dir)

    from tabularium.cli import _cmd_clean

    judge_a, judge_b = _shelve_judges()
    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=False,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=judge_a,
        judge_b=judge_b,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "second OS process" in out  # the concurrent-write warning
    assert "LIVE" in out
    assert "1 shelved" in out
    # 2026-08-12 rubric-gate-retune fix: the same detail rendering also appears on a live run
    # (unconditional, not dry-run-gated) -- "shelved:" (no "would"), the id, and the judge `why`.
    assert "shelved:" in out
    assert str(fact_id) in out
    assert "stale" in out

    fresh = await open_adapter(
        "lancedb", embedding_model="fake/nomic", dim=_DIM, store_path=store_dir
    )
    state = await fresh.get_cleaner_state(scope=Scope("proj-a"))
    assert state is not None
    assert state.watermark_at > _CLEAN_T0  # actually advanced
    assert state.report is not None  # report actually persisted

    stored = await fresh.get_by_id(fact_id)
    assert stored.lifecycle == "shelved"  # actually written


def _clean_summary_for_one_shelved_why(why: str) -> Any:
    """Build a minimal ``CleanSummary`` carrying a single shelved record whose deciding-judge
    ``why`` is exactly ``why`` — the shared fixture for the ``_render_clean_summary`` sanitization
    tests below (2026-08-12). No store, no judge panel: the
    property under test is rendering, not the pass itself."""
    from tabularium.domain import new_id
    from tabularium.engine.axis_common import BandCounts, RecordVerdict
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.relevance import RelevanceAxisSummary

    record_id = new_id()
    return record_id, CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_CLEAN_T0,
        watermark_after=datetime(2025, 1, 10, tzinfo=UTC),
        pool_size=1,
        relevance=RelevanceAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_already_shelved=(),
            counts=BandCounts(shelve=1, keep=0, unclear=0),
            shelved_ids=(record_id,),
            unclear_ids=(),
            failed_batches=(),
            calls=(),
            verdicts=(RecordVerdict(id=record_id, band="shelve", why=why),),
        ),
        reason="test",
        dry_run=True,
    )


def test_render_clean_summary_collapses_multiline_judge_why() -> None:
    """``RelevanceVerdict.why`` is validated only
    ``min_length=1`` — nothing on the judge seam stops a judge's raw text from containing a
    newline or a run of control/whitespace characters, and ``_render_clean_summary``'s shelve
    detail block is fixed-order, one-record-per-line text. A `why` with an embedded newline must
    not fork the record's line in two or make the id-to-reason block unparseable.
    """
    from tabularium.cli import _render_clean_summary

    dirty_why = "line one\nline two\t\twith a tab\r\nand a third line"
    record_id, summary = _clean_summary_for_one_shelved_why(dirty_why)

    rendered = _render_clean_summary(summary)

    lines = rendered.splitlines()
    matching = [line for line in lines if str(record_id) in line]
    assert len(matching) == 1, "the shelved record must render on exactly one line"
    assert "line one line two" in matching[0]  # newline/tab/CR collapsed to single spaces
    assert "\n" not in matching[0] and "\t" not in matching[0] and "\r" not in matching[0]


def test_render_clean_summary_strips_ansi_control_bytes_in_judge_why() -> None:
    """Whitespace collapse alone still let a non-whitespace control byte — e.g. an ANSI
    cursor/erase escape sequence — reach
    the terminal verbatim through a judge's ``why``. Every C0/C1 control byte other than
    tab/LF/CR (already handled by whitespace collapse) must be dropped outright, not merely left
    embedded in an otherwise-collapsed line.
    """
    from tabularium.cli import _render_clean_summary

    ansi_why = "before\x1b[2Kafter\x07bell\x9cst"  # cursor erase, BEL, and a C1 control byte
    record_id, summary = _clean_summary_for_one_shelved_why(ansi_why)

    rendered = _render_clean_summary(summary)

    lines = rendered.splitlines()
    matching = [line for line in lines if str(record_id) in line]
    assert len(matching) == 1, "the shelved record must render on exactly one line"
    # Only the control BYTES are dropped -- the escape sequence's ordinary printable characters
    # ("[2K") are left as harmless literal text once the ESC byte that would have made a terminal
    # interpret them as a command is gone.
    assert "before[2Kafterbellst" in matching[0]
    for control_byte in ("\x1b", "\x07", "\x9c"):
        assert control_byte not in matching[0]


def test_render_clean_summary_prints_relevance_batch_failure_errors() -> None:
    """Defect-1 fix (2026-08-14): before this fix, a failed relevance batch rendered ONLY
    ``"N relevance batch(es) failed"`` -- the count, never the reason. Falsifiability: revert the
    per-batch ``for failure in relevance.failed_batches`` loop in ``_render_clean_summary`` back to
    the count-only line and re-run -- red, because ``"code -32602"`` (the error text) no longer
    appears anywhere in the rendered output even though the count line still does."""
    from tabularium.cli import _render_clean_summary
    from tabularium.domain import new_id
    from tabularium.engine.axis_common import BandCounts, BatchFailure
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.relevance import RelevanceAxisSummary

    record_id = new_id()
    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="blocked",
        watermark_before=_CLEAN_T0,
        watermark_after=_CLEAN_T0,
        pool_size=1,
        relevance=RelevanceAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_already_shelved=(),
            counts=BandCounts(),
            shelved_ids=(),
            unclear_ids=(),
            failed_batches=(
                BatchFailure(
                    batch_index=0,
                    record_ids=(record_id,),
                    judge_a_outcome="unavailable",
                    judge_b_outcome="ok",
                    error="codex CLI exited 1: Error: turn/start failed (code -32602)",
                ),
            ),
            calls=(),
            verdicts=(),
        ),
        reason="test",
        dry_run=False,
    )

    rendered = _render_clean_summary(summary)

    assert "1 relevance batch(es) failed" in rendered
    assert "batch 0" in rendered
    assert "code -32602" in rendered
    assert "judge_a=unavailable" in rendered


def test_render_clean_summary_prints_falsity_batch_failure_errors() -> None:
    """Same fix, falsity axis (``cli.py``'s "fix all three, this is systemic" instruction)."""
    from tabularium.cli import _render_clean_summary
    from tabularium.domain import new_id
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.falsity import (
        FalsityAxisSummary,
        FalsityBandCounts,
        FalsityBatchFailure,
    )

    newcomer_id, candidate_id = new_id(), new_id()
    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="blocked",
        watermark_before=_CLEAN_T0,
        watermark_after=_CLEAN_T0,
        pool_size=1,
        falsity=FalsityAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            newcomer_pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_shelved_this_pass=(),
            counts=FalsityBandCounts(),
            edges=(),
            decision_candidates=(),
            global_candidates=(),
            hot_or_ember_candidates=(),
            failed_batches=(
                FalsityBatchFailure(
                    batch_index=2,
                    pairs=((newcomer_id, candidate_id),),
                    judge_a_outcome="ok",
                    judge_b_outcome="invalid_output",
                    error="response is not valid JSON: Expecting value: line 1 column 1 (char 0)",
                ),
            ),
            calls=(),
        ),
        reason="test",
        dry_run=False,
    )

    rendered = _render_clean_summary(summary)

    assert "1 falsity batch(es) failed" in rendered
    assert "batch 2" in rendered
    assert "Expecting value" in rendered
    assert "judge_b=invalid_output" in rendered


def test_render_clean_summary_prints_revisit_batch_failure_errors() -> None:
    """Same fix, revisit axis -- the exact incident that motivated the fix ("a real pass had all
    four revisit batches fail, and the only artifact anywhere was the integer 4")."""
    from tabularium.cli import _render_clean_summary
    from tabularium.domain import new_id
    from tabularium.engine.axis_common import BandCounts, BatchFailure
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.revisit import RevisitAxisSummary

    record_id = new_id()
    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_CLEAN_T0,
        watermark_after=datetime(2025, 1, 10, tzinfo=UTC),
        pool_size=1,
        revisit=RevisitAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=1,
            excluded_global=(),
            excluded_non_fact=(),
            protected_ids=(),
            skipped_no_evidence_ids=(),
            counts=BandCounts(),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(),
            failed_batches=(
                BatchFailure(
                    batch_index=3,
                    record_ids=(record_id,),
                    judge_a_outcome="unavailable",
                    judge_b_outcome="unavailable",
                    error=(
                        "claude CLI exited 1: some diagnostic; "
                        "codex CLI exited 1: Error: turn/start failed (code -32602)"
                    ),
                ),
            ),
            calls=(),
            verdicts=(),
            contested_verdicts=(),
            verification_failures=(),
        ),
        reason="test",
        dry_run=False,
    )

    rendered = _render_clean_summary(summary)

    assert "1 revisit batch(es) failed" in rendered
    assert "batch 3" in rendered
    assert "code -32602" in rendered
    assert "judge_a=unavailable" in rendered and "judge_b=unavailable" in rendered


def test_render_batch_error_truncates_a_long_error_to_exactly_the_cap() -> None:
    """The three tests above only
    ever exercise ``_render_batch_error`` through short fixtures ("code -32602",
    "Expecting value: line 1 column 1 (char 0)") and only assert a substring appears -- none of
    them prove the cap arithmetic itself, which is exactly where ``_truncate_why``'s own docstring
    warns "a naive ``why[:cap] + '...'`` would silently render up to ``cap + 3`` characters,
    defeating the cap it claims to enforce". ``_render_batch_error`` uses the mirror-image shape
    (a LEADING ellipsis plus a TAIL slice, not a trailing one), so the off-by-one has to be proven
    against THIS arithmetic, not borrowed from ``_truncate_why``'s own test coverage.

    Falsifiability: change ``_render_batch_error``'s truncation branch from
    ``"..." + single[-(cap - 3):]`` to the naive ``"..." + single[-cap:]`` (ellipsis ADDED on top
    of a full ``cap``-length slice, rather than carved out of it) and re-run -- ``len(rendered) ==
    300`` fails with 303.
    """
    from tabularium.cli import _BATCH_ERROR_RENDER_CAP, _render_batch_error

    assert _BATCH_ERROR_RENDER_CAP == 300
    long_error = ("A" * 500) + "TAIL-MARKER-END"

    rendered = _render_batch_error(long_error)

    assert len(rendered) == _BATCH_ERROR_RENDER_CAP
    assert rendered.startswith("...")
    # The retained content is the TAIL of the original, not the head -- the marker planted at the
    # very end of the 516-character input survives into the capped result.
    assert rendered.endswith("TAIL-MARKER-END")
    assert rendered == "..." + long_error[-(_BATCH_ERROR_RENDER_CAP - 3) :]


def test_render_batch_error_leaves_a_short_error_unchanged() -> None:
    """The complementary case: at or under the cap, no ellipsis is added and no content is cut --
    the same "nothing to cut, return verbatim" shape ``_truncate_why`` establishes."""
    from tabularium.cli import _render_batch_error

    short_error = "codex CLI exited 1: Error: turn/start failed (code -32602)"
    assert _render_batch_error(short_error) == short_error
    assert "..." not in _render_batch_error(short_error)


def test_render_clean_summary_prints_the_revisit_section() -> None:
    """2026-08-12 cleaner-revisit-pass plan, scope 5: ``_render_clean_summary`` renders the
    revisit axis's own shelve/contested detail alongside the existing relevance/falsity blocks,
    isolated in MEANING (the top-level ``outcome``/``watermark`` lines still describe the birth
    axes alone) but not in VISIBILITY."""
    from tabularium.cli import _render_clean_summary
    from tabularium.domain import new_id
    from tabularium.engine.axis_common import (
        BandCounts,
        ContestedVerdict,
        RecordVerdict,
    )
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.revisit import RevisitAxisSummary

    shelved_id = new_id()
    contested_id = new_id()
    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_CLEAN_T0,
        watermark_after=datetime(2025, 1, 10, tzinfo=UTC),
        pool_size=1,
        revisit=RevisitAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=2,
            excluded_global=(),
            excluded_non_fact=(),
            protected_ids=(new_id(),),
            skipped_no_evidence_ids=(new_id(), new_id()),
            counts=BandCounts(shelve=1, keep=0, unclear=0, contested=1),
            shelved_ids=(shelved_id,),
            unclear_ids=(),
            contested_ids=(contested_id,),
            failed_batches=(),
            calls=(),
            verdicts=(RecordVerdict(id=shelved_id, band="shelve", why="stale now"),),
            contested_verdicts=(
                ContestedVerdict(
                    id=contested_id,
                    judge_a_identity="judge-a",
                    judge_a_grade="shelve-lean",
                    judge_a_why="perishable",
                    judge_b_identity="judge-b",
                    judge_b_grade="keep-clear",
                    judge_b_why="durable",
                ),
            ),
            verification_failures=(),
        ),
        reason="test",
        dry_run=False,
    )

    rendered = _render_clean_summary(summary)

    assert "revisit:" in rendered
    assert "1 shelved" in rendered
    assert str(shelved_id) in rendered
    assert "stale now" in rendered
    assert str(contested_id) in rendered
    assert "perishable" in rendered and "durable" in rendered
    assert "protected 1" in rendered
    assert "skipped-no-evidence 2" in rendered


def test_render_clean_summary_prints_revisit_error_when_the_axis_crashed() -> None:
    from tabularium.cli import _render_clean_summary
    from tabularium.engine.clean import CleanSummary

    summary = CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_CLEAN_T0,
        watermark_after=datetime(2025, 1, 10, tzinfo=UTC),
        pool_size=1,
        revisit=None,
        revisit_error="kaboom -- revisit axis exploded",
        reason="test",
        dry_run=False,
    )

    rendered = _render_clean_summary(summary)

    assert "revisit" in rendered
    assert "kaboom" in rendered
    # ISOLATED: the top-level outcome is untouched by the revisit crash.
    assert "outcome: completed" in rendered


async def test_clean_dry_run_prints_no_concurrent_write_warning(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """--dry-run never writes, so it has nothing to warn about (acceptance criterion: "Without
    --dry-run, the command prints the concurrent-write warning ...")."""
    store_dir = str(tmp_path / "store")
    _engine, _fact_id = await _seed_clean_scope(store_dir)

    from tabularium.cli import _cmd_clean

    judge_a, judge_b = _shelve_judges()
    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=True,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=judge_a,
        judge_b=judge_b,
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "second OS process" not in out


async def test_clean_global_scope_runs(tmp_path: pathlib.Path) -> None:
    """``--scope global`` runs like any scope, matching the engine (``clean()`` accepts it)."""
    store_dir = str(tmp_path / "store")
    engine, _adapter = await _make_lancedb_engine(store_dir)
    del engine

    from tabularium.cli import _cmd_clean

    rc = await _cmd_clean("global", _cfg(store_dir), dry_run=True)
    assert rc == 0


async def test_clean_unregistered_scope_is_a_clean_error(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """An unregistered scope is a clean error (caught ``ToolError``, printed to stderr), never a
    traceback."""
    store_dir = str(tmp_path / "store")
    engine, _adapter = await _make_lancedb_engine(store_dir)
    del engine

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="Unregistered scope"):
        await _cmd_clean("never-registered", _cfg(store_dir), dry_run=True)


async def test_clean_memory_adapter_refused(tmp_path: pathlib.Path, capsys: Any) -> None:
    """The memory adapter is refused with a clear message, as ``sync`` already does — an
    ephemeral store is not something to clean."""
    cfg = StoreConfig(
        adapter="memory",
        path=None,
        model="fake/nomic",
        dim=_DIM,
        api_base="http://localhost:11434",
    )

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="memory adapter never persists data"):
        await _cmd_clean("proj-a", cfg, dry_run=True)


async def test_clean_blocked_outcome_returns_nonzero_exit(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """Exit code is non-zero when the pass failed (an unreachable judge), so a wrapper can tell."""
    store_dir = str(tmp_path / "store")
    _engine, _fact_id = await _seed_clean_scope(store_dir)

    from tabularium.cli import _cmd_clean

    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=False,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=FakeJudgeAdapter(identity="judge-a", should_fail=True),
        judge_b=FakeJudgeAdapter(identity="judge-b", response=_shelve_verdict_payload()),
    )
    assert rc == 1

    out = capsys.readouterr().out
    assert "outcome: blocked" in out


# ---------------------------------------------------------------------------
# clean --rubric (rubric-calibration dry-run seam)
# ---------------------------------------------------------------------------


async def test_clean_rubric_without_dry_run_exits_before_opening_the_store(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """--rubric is dry-run only (locked decision): passed without --dry-run, this exits 1 BEFORE
    the store is ever opened -- proven by pointing --rubric at a file that does not exist. If the
    gate fired after an attempted load, the error would instead be a "file not found" message; if
    it opened the store first, the store directory would exist afterward."""
    store_dir = tmp_path / "store"

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="dry-run only") as excinfo:
        await _cmd_clean(
            "proj-a",
            _cfg(str(store_dir)),
            dry_run=False,
            rubric_path=tmp_path / "never-created.toml",
        )
    assert "not found" not in str(excinfo.value)
    assert not store_dir.exists()  # never opened


async def test_clean_rubric_missing_file_reports_a_named_error(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    store_dir = str(tmp_path / "store")
    _engine, _fact_id = await _seed_clean_scope(store_dir)
    missing = tmp_path / "does-not-exist.toml"

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="not found") as excinfo:
        await _cmd_clean(
            "proj-a",
            _cfg(store_dir),
            dry_run=True,
            rubric_path=missing,
        )
    assert str(missing) in str(excinfo.value)


async def test_clean_rubric_malformed_toml_reports_a_named_error(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    store_dir = str(tmp_path / "store")
    _engine, _fact_id = await _seed_clean_scope(store_dir)
    bad_path = tmp_path / "custom.toml"
    bad_path.write_text("this is [ not valid toml =", encoding="utf-8")

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="TOML") as excinfo:
        await _cmd_clean(
            "proj-a",
            _cfg(store_dir),
            dry_run=True,
            rubric_path=bad_path,
        )
    assert str(bad_path) in str(excinfo.value)


async def test_clean_dry_run_with_custom_rubric_names_it_in_the_summary(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """End-to-end (Verify step): --rubric against a temp TOML whose only delta from the packaged
    artifact is `revisit_min_age_days`, run with --dry-run against a fixture store -- never the
    live store. The summary names the custom rubric's own path and its own (differing) version.
    """
    from tabularium.engine.rubric import load_rubric, load_rubric_from_path
    from tests.test_rubric import packaged_toml_with_revisit_min_age_days

    store_dir = str(tmp_path / "store")
    _engine, fact_id = await _seed_clean_scope(store_dir)

    custom_path = tmp_path / "custom.toml"
    custom_path.write_text(packaged_toml_with_revisit_min_age_days(45), encoding="utf-8")
    expected_custom_rubric = load_rubric_from_path(custom_path)
    packaged_version = load_rubric().version
    assert expected_custom_rubric.version != packaged_version  # sanity: genuinely differs

    from tabularium.cli import _cmd_clean

    judge_a, judge_b = _shelve_judges()
    rc = await _cmd_clean(
        "proj-a",
        _cfg(store_dir),
        dry_run=True,
        rubric_path=custom_path,
        embedder=FakeEmbedder(dim=_DIM),
        clock=TestClock(datetime(2025, 1, 10, tzinfo=UTC)),
        judge_a=judge_a,
        judge_b=judge_b,
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert f"rubric: {custom_path} (version {expected_custom_rubric.version})" in out
    assert packaged_version not in out
    assert "1 would shelve" in out
    assert str(fact_id) in out

    # Never touched the store the way the packaged-rubric dry-run test already proves byte-for-
    # byte -- here it is enough to confirm the watermark truly never moved (fixture store only).
    fresh = await open_adapter(
        "lancedb", embedding_model="fake/nomic", dim=_DIM, store_path=store_dir
    )
    state = await fresh.get_cleaner_state(scope=Scope("proj-a"))
    assert state is not None
    assert state.watermark_at == _CLEAN_T0


async def test_clean_history_and_rubric_are_mutually_exclusive(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    """--history runs no pass and never reads
    a rubric of any kind, so combining it with --rubric used to either silently ignore --rubric
    (a live-defaulting --history call) or silently skip validating a bad --rubric path (a dry-run
    --history call) -- both surprising. Refused outright instead, before either branch runs."""
    store_dir = str(tmp_path / "store")
    _engine, _fact_id = await _seed_clean_scope(store_dir)

    from tabularium.cli import _cmd_clean

    with pytest.raises(CliError, match="--history") as excinfo:
        await _cmd_clean(
            "proj-a",
            _cfg(store_dir),
            dry_run=True,
            history=True,
            rubric_path=tmp_path / "never-read.toml",
        )
    assert "--rubric" in str(excinfo.value)
    assert capsys.readouterr().out == ""  # _cmd_clean_history never ran


def test_clean_rubric_empty_string_argument_is_not_treated_as_absent(
    capsys: Any, monkeypatch: Any
) -> None:
    """`main()` converted `--rubric PATH` to
    `_cmd_clean`'s `rubric_path` with a truthiness check (`if args.rubric else None`), so
    `--rubric ''` (e.g. an empty shell variable) silently resolved to `rubric_path=None` --
    bypassing the dry-run-only gate entirely and falling through toward an ordinary LIVE pass.
    Driven through the real argument parser and `main()`, not `_cmd_clean` directly, since the
    bug lived in that conversion. `--adapter memory` keeps this hermetic: the rubric gate fires
    before the memory-adapter refusal, so no store is ever opened and no network call is made
    either way -- if it fires."""
    from tabularium.cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["tabularium", "--adapter", "memory", "clean", "--scope", "proj-a", "--rubric", ""],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    err = capsys.readouterr().err
    assert "dry-run only" in err


def _summary_with_trim_and_merge(
    *,
    trimmed: int,
    merged: int,
    merge_reverted: int,
    merge_revert_failed: int = 0,
    dry_run: bool = False,
) -> Any:
    from tabularium.domain import new_id
    from tabularium.engine.axis_common import BandCounts
    from tabularium.engine.clean import CleanSummary
    from tabularium.engine.cleaner_writes import MergedTriple, MergeRevertedEntry, TrimmedPair
    from tabularium.engine.falsity import FalsityAxisSummary, FalsityBandCounts
    from tabularium.engine.relevance import RelevanceAxisSummary
    from tabularium.engine.revisit import RevisitAxisSummary

    def _trims() -> tuple[TrimmedPair, ...]:
        return tuple(TrimmedPair(old_id=new_id(), new_id=new_id()) for _ in range(trimmed))

    return CleanSummary(
        scope=Scope("proj-a"),
        outcome="completed",
        watermark_before=_CLEAN_T0,
        watermark_after=datetime(2025, 1, 10, tzinfo=UTC),
        pool_size=1,
        relevance=RelevanceAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_already_shelved=(),
            counts=BandCounts(shelve=trimmed),
            shelved_ids=(),
            unclear_ids=(),
            failed_batches=(),
            calls=(),
            trimmed=_trims(),
        ),
        falsity=FalsityAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            newcomer_pool_size=1,
            excluded_decisions=(),
            excluded_global=(),
            excluded_shelved_this_pass=(),
            counts=FalsityBandCounts(replace=merged + merge_reverted),
            edges=(),
            decision_candidates=(),
            global_candidates=(),
            hot_or_ember_candidates=(),
            failed_batches=(),
            calls=(),
            merged=tuple(
                MergedTriple(older_id=new_id(), newer_id=new_id(), merged_id=new_id())
                for _ in range(merged)
            ),
            merge_reverted=tuple(
                MergeRevertedEntry(older_id=new_id(), newer_id=new_id(), error="bounced")
                for _ in range(merge_reverted)
            ),
            merge_revert_failed=tuple(
                MergeRevertedEntry(older_id=new_id(), newer_id=new_id(), error="restore failed")
                for _ in range(merge_revert_failed)
            ),
        ),
        revisit=RevisitAxisSummary(
            scope=Scope("proj-a"),
            rubric_version="v1",
            judge_pair=("judge-a", "judge-b"),
            pool_size=1,
            excluded_global=(),
            excluded_non_fact=(),
            protected_ids=(),
            skipped_no_evidence_ids=(),
            counts=BandCounts(shelve=trimmed),
            shelved_ids=(),
            unclear_ids=(),
            contested_ids=(),
            failed_batches=(),
            calls=(),
            verdicts=(),
            contested_verdicts=(),
            verification_failures=(),
            trimmed=_trims(),
        ),
        reason="test",
        dry_run=dry_run,
    )


def test_render_clean_summary_shows_trimmed_and_merged_counts_on_each_axis_line() -> None:
    from tabularium.cli import _render_clean_summary

    rendered = _render_clean_summary(
        _summary_with_trim_and_merge(trimmed=2, merged=3, merge_reverted=1)
    )

    lines = {line.split(":")[0]: line for line in rendered.splitlines() if ": " in line}
    assert "0 contested, 2 trimmed (judges" in lines["relevance"]
    assert "0 contested, 3 merged, 1 merge-reverted (judges" in lines["falsity"]
    assert "0 contested, 2 trimmed (judges" in lines["revisit"]


def test_render_clean_summary_labels_failed_undos_apart_from_reverted_merges() -> None:
    from tabularium.cli import _render_clean_summary

    rendered = _render_clean_summary(
        _summary_with_trim_and_merge(
            trimmed=0, merged=0, merge_reverted=1, merge_revert_failed=2
        )
    )

    falsity_line = next(line for line in rendered.splitlines() if line.startswith("falsity:"))
    assert "1 merge-reverted, 2 merge-revert-failed" in falsity_line


def test_render_clean_summary_omits_trim_and_merge_counts_when_zero() -> None:
    from tabularium.cli import _render_clean_summary

    rendered = _render_clean_summary(
        _summary_with_trim_and_merge(trimmed=0, merged=0, merge_reverted=0)
    )

    assert "trimmed" not in rendered
    assert "merged" not in rendered
    assert "merge-reverted" not in rendered


# ---------------------------------------------------------------------------
# dispatch, doctor config, supersede reason, concurrent-write warning
# ---------------------------------------------------------------------------


def test_doctor_malformed_engine_env_prints_error_and_exits_one(
    tmp_path: pathlib.Path, capsys: Any, monkeypatch: Any
) -> None:
    from tabularium.cli import main

    monkeypatch.setenv("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", "x")
    monkeypatch.setattr(
        "sys.argv",
        ["tabularium", "--adapter", "lancedb", "--store", str(tmp_path / "store"), "doctor"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1

    captured = capsys.readouterr()
    assert captured.err == (
        "error: Invalid TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S='x'; must be a float\n"
    )
    assert not (tmp_path / "store").exists()


async def test_doctor_reads_the_engine_env_config(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    captured: dict[str, Any] = {}

    async def fake_report(adapter: Any, cfg: Any) -> Any:
        captured["cfg"] = cfg
        raise RuntimeError("stop after the config reaches the report")

    monkeypatch.setattr("tabularium.engine.doctor.build_doctor_report", fake_report)
    monkeypatch.setenv("TABULARIUM_ASSESSOR_SAVE_TIMEOUT_S", "9")

    from tabularium.cli import _cmd_doctor

    with pytest.raises(RuntimeError, match="stop after"):
        await _cmd_doctor(_cfg(str(tmp_path / "store")))
    assert captured["cfg"].assessor_save_timeout_s == 9.0


def test_supersede_reason_is_parsed_at_the_argparse_edge() -> None:
    from tabularium.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["supersede", "a", "b", "--reason", "elaborated"])
    assert args.reason == "elaborated"

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["supersede", "a", "b", "--reason", "bogus"])
    assert excinfo.value.code == 2


def test_supersede_reason_error_names_the_choices(capsys: Any) -> None:
    from tabularium.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["supersede", "a", "b", "--reason", "bogus"])

    assert (
        "argument --reason: invalid choice: 'bogus' "
        "(choose from correction, changed, elaborated, moved)"
    ) in capsys.readouterr().err


def test_supersede_reason_metavar_lists_every_supersession_reason() -> None:
    from typing import get_args

    from tabularium.cli import _REASON_METAVAR
    from tabularium.domain import SupersessionReason

    assert _REASON_METAVAR == "{" + ",".join(get_args(SupersessionReason)) + "}"


def test_bare_command_group_prints_its_help_and_exits_zero(
    tmp_path: pathlib.Path, capsys: Any, monkeypatch: Any
) -> None:
    from tabularium.cli import main

    monkeypatch.setattr(
        "sys.argv",
        ["tabularium", "--adapter", "memory", "scope"],
    )

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("usage: tabularium scope")


def test_concurrent_write_warning_names_the_command_and_the_per_process_locks() -> None:
    from tabularium.cli import _concurrent_write_warning

    clean = _concurrent_write_warning("clean")
    backfill = _concurrent_write_warning("backfill")

    assert clean.startswith("clean: this command writes from a second OS process.")
    assert backfill.startswith("backfill: this command writes from a second OS process.")
    assert clean.replace("clean:", "backfill:", 1) == backfill
    assert "per-process" in clean
    assert "compare-and-swap" not in clean


async def test_backfill_prints_its_own_name_in_the_warning(
    tmp_path: pathlib.Path, capsys: Any
) -> None:
    from tabularium.cli import _cmd_backfill

    rc = await _cmd_backfill(_cfg(str(tmp_path / "store")), embedder=FakeEmbedder(dim=_DIM))
    assert rc == 0

    out = capsys.readouterr().out
    assert "backfill: this command writes from a second OS process." in out
    assert "clean:" not in out
