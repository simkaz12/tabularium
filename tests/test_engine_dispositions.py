"""Engine disposition tests — ``set_validity``/``bury``/``revoke`` (M2A, SPEC §1.3/§9.1/§9.2).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``engine_ctx`` fixture, which reuses the conformance suite's adapter
construction helpers (precedent — see ``tests/test_eval_b5.py``)
rather than a third construction pattern.

Falsifiability (non-negotiable per the M2A brief): every test here has been
proven sensitive to a real defect by deliberately mutating
``src/tabularium/engine/dispositions.py`` (or, for one case,
``render.py``), watching the named test go red for the stated reason, then
restoring. Each test's docstring names the mutation that
would break it, for a future reader.
"""

from __future__ import annotations

import logging
import pathlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pydantic
import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.render import derive_title
from tabularium.server import make_app
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import Filters, UnsupersedeEvent

# Adapter-construction pattern reused verbatim from the conformance suite
# (see tests/test_eval_b5.py) rather than a second one.
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock
from tests.support.edits import edit_content
from tests.support.faults import FaultInjectingAdapter, as_adapter
from tests.support.lance_raw import raw_table

AnyDispositionAdapter = MemoryAdapter | LanceDBAdapter
EngineCtx = tuple[Tabularium, AnyDispositionAdapter, FakeEmbedder, TestClock, str]

_DIM = 64


async def _tombstone_count(adapter: AnyDispositionAdapter) -> int:
    if isinstance(adapter, MemoryAdapter):
        return len(adapter._tombstones)

    return int(await raw_table(adapter._tombstones_tbl).count_rows())


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


@pytest.fixture(params=["memory", "lancedb"])
async def engine_ctx(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> EngineCtx:
    """A fresh, initialized engine + adapter + embedder + clock.

    Parameterized over ``request.param`` in ``{"memory", "lancedb"}`` — every
    test taking this fixture runs once per adapter.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clock = _make_clock()
    adapter_id = request.param
    adapter: AnyDispositionAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clock
        )
    cfg = _make_config()
    engine = Tabularium(adapter, embedder, clock, cfg)
    await engine.initialize()
    return engine, adapter, embedder, clock, adapter_id


# ---------------------------------------------------------------------------
# set_validity
# ---------------------------------------------------------------------------


class TestSetValidity:
    async def test_flips_validity_and_preserves_record(self, engine_ctx: EngineCtx) -> None:
        """Case 1 (brief).

        Falsify: comment out the ``await adapter.set_validity(...)`` call in
        ``set_validity`` — validity never actually flips.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for validity flip test one", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        before = await adapter.get_by_id(fid)

        result = await engine.set_validity(fid, valid=False, reason="found to be false")
        assert result.outcome == "revised"
        assert result.validity == "invalid"

        after = await adapter.get_by_id(fid)
        assert after.validity == "invalid"
        assert after.content == before.content
        assert after.lifecycle == before.lifecycle
        assert after.created_at == before.created_at

    async def test_round_trip_restores_valid(self, engine_ctx: EngineCtx) -> None:
        """Case 2 (brief): False -> True restores validity="valid"."""
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for validity round trip test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        await engine.set_validity(fid, valid=False, reason="temporarily false")
        result = await engine.set_validity(fid, valid=True, reason="restored true")
        assert result.outcome == "revised"
        assert result.validity == "valid"

        rec = await adapter.get_by_id(fid)
        assert rec.validity == "valid"

    async def test_same_value_is_noop_ack(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Case 3 (brief): same-value ``set_validity`` returns "unchanged" and
        performs NO write — proven here by monkeypatching ``adapter.set_validity``
        to raise if called at all.

        Falsify: remove the "already equals -> unchanged, no write" early
        return in ``set_validity`` so it always calls the adapter.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for same value validity test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "adapter.set_validity must not be called on a same-value re-call"
            )

        monkeypatch.setattr(adapter, "set_validity", _boom)

        result = await engine.set_validity(fid, valid=True, reason="already valid, no-op")
        assert result.outcome == "unchanged"
        assert result.validity == "valid"

    async def test_leaves_updated_at_unmoved(self, engine_ctx: EngineCtx) -> None:
        """Case 4 (brief): the exact ``updated_at`` datetime is unchanged.

        Falsify: make ``set_validity`` call ``adapter.transition`` instead of
        ``adapter.set_validity``.
        """
        engine, adapter, _embedder, clock, _id = engine_ctx
        r = await engine.save("fact for updated_at stability test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        before = await adapter.get_by_id(fid)

        clock.advance(timedelta(days=3))
        await engine.set_validity(fid, valid=False, reason="testing updated_at stability")

        after = await adapter.get_by_id(fid)
        assert after.updated_at == before.updated_at

    async def test_render_date_tokens_stable_across_invalidate(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 5 (brief) — corrected.

        The brief literally asks for full byte-identity of the render across
        an invalidate. That is unreachable by design: ``render_decision``'s
        status line embeds ``decision.validity`` directly, and both
        ``render.py``'s own module docstring and SPEC §6 (line 314) name
        "validity flip" as one of the THREE semantic transitions whose bytes
        ARE required to change. What case 4 (``updated_at`` unmoved) actually
        protects at the render level is narrower and still real: the render's
        DATE tokens (derived from ``created_at``/``updated_at``) must not
        shift when only validity changes. This test asserts that — using a
        SUPERSEDED Decision, whose render carries a ``superseded {date}``
        token derived from ``updated_at`` (an ``in_force`` Decision's render
        never shows ``updated_at`` at all, so it could not exercise this).

        Falsify: same mutation as case 4 (call ``adapter.transition`` instead
        of ``adapter.set_validity``) — the superseded-date token shifts (or,
        for a Decision, the mutated ``transition`` call raises outright since
        ``superseded_by`` is already set — either way, red).
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        ra = await engine.save(
            "Decision A for render stability test", scope="global", kind="decision"
        )
        assert ra.hypomnema is not None
        a_id = ra.hypomnema.id
        rb = await engine.save(
            "Decision B superseding A for render stability test",
            scope="global",
            kind="decision",
            supersedes=a_id,
            reason="changed",
        )
        assert rb.outcome == "saved"

        before_view = await engine.decision(a_id)
        rendered_before = before_view.rendered
        assert "superseded " in rendered_before

        result = await engine.set_validity(a_id, valid=False, reason="render stability check")
        assert result.outcome == "revised"

        after_view = await engine.decision(a_id)
        rendered_after = after_view.rendered

        # The validity token DOES change (a real semantic transition per
        # render.py's own contract) -- bytes are NOT identical overall.
        assert rendered_before != rendered_after

        def _superseded_date_token(rendered: str) -> str:
            first_line = rendered.splitlines()[0]
            tokens = first_line.split(" · ")
            date_tokens = [t for t in tokens if t.startswith("superseded ")]
            assert len(date_tokens) == 1, (
                f"expected exactly one 'superseded <date>' token, got {tokens!r}"
            )
            return date_tokens[0]

        assert _superseded_date_token(rendered_before) == _superseded_date_token(
            rendered_after
        ), "superseded date must not shift across an invalidate (updated_at unmoved)"

    async def test_render_omits_invalidated_line_for_manual_validity_flip(
        self, engine_ctx: EngineCtx
    ) -> None:
        """A plain ``set_validity(valid=False)`` call, with no Fact cause behind
        it, flips only the status-line validity token — it never writes
        ``invalidated_by``/``invalidation_reason`` (P8 D2), so the render's
        lineage section gains no ``Invalidated by:`` line (P9 invalid-Decision
        read contract, decision D7: that line is gated on ``invalidated_by``
        being set, not on the ``validity`` token alone).

        Falsify: change the render's gate from ``invalidated_by is not None``
        to ``decision.validity == "invalid"`` alone — this test goes red.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save(
            "Decision manually marked invalid, no Fact cause", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        dec_id = r.hypomnema.id

        result = await engine.set_validity(dec_id, valid=False, reason="manual flip, no cause")
        assert result.outcome == "revised"

        view = await engine.decision(dec_id)
        assert view.validity == "invalid"
        assert view.invalidated_by is None
        assert "invalid" in view.rendered.splitlines()[0]
        assert "Invalidated by:" not in view.rendered

    async def test_unknown_id_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        with pytest.raises(ToolError):
            await engine.set_validity(new_id(), valid=False, reason="does not exist")

    async def test_empty_reason_raises(self, engine_ctx: EngineCtx) -> None:
        """SPEC §9.1 requires ``reason`` on all three
        disposition tools, not just ``bury``.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for set_validity empty reason test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.set_validity(fid, valid=False, reason="")

    async def test_race_bury_between_check_and_write_raises_tool_error(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concurrent burial between the initial
        existence check and the write must surface SPEC §9.2's ToolError
        contract, never a raw adapter ``KeyError``.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for set_validity race test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        async def _raise_key_error(*_args: object, **_kwargs: object) -> None:
            raise KeyError(f"Unknown id: {fid!s}")

        monkeypatch.setattr(adapter, "set_validity", _raise_key_error)

        with pytest.raises(ToolError):
            await engine.set_validity(fid, valid=False, reason="racing a concurrent bury")


# ---------------------------------------------------------------------------
# bury
# ---------------------------------------------------------------------------


class TestBury:
    async def test_returns_tombstone_and_removes_record(self, engine_ctx: EngineCtx) -> None:
        """Case 6 (brief).

        Falsify: make ``bury`` return a fabricated ``BuryResult`` without
        calling ``adapter.bury`` at all.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact to be buried for tombstone test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        result = await engine.bury(fid, reason="no longer true")
        assert result.outcome == "buried"
        assert result.buried_id == fid

        tombstone = await adapter.get_tombstone(fid)
        assert tombstone is not None
        assert tombstone.id == result.tombstone_id
        assert tombstone.buried_id == fid
        assert tombstone.reason == "no longer true"
        assert tombstone.type == "fact"

        with pytest.raises(KeyError):
            await adapter.get_by_id(fid)

    async def test_completeness_absent_from_search_and_recall(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 7 (brief): burial is complete across vector, text, and recall.

        Falsify: same mutation as case 6 — a fabricated success that never
        calls ``adapter.bury`` leaves the record fully searchable.
        """
        engine, adapter, embedder, _clock, _id = engine_ctx
        content = "unique content for burial completeness search test"
        r = await engine.save(content, scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        recall_before = await engine.recall(content, scope="global")
        assert any(h.id == fid for h in recall_before.hits)

        await engine.bury(fid, reason="burial completeness check")

        recall_after = await engine.recall(content, scope="global")
        assert all(h.id != fid for h in recall_after.hits)

        vec = await embedder.embed(content)
        filters = Filters(
            scopes=[Scope("global")],
            validity=frozenset({"valid", "invalid"}),
            lifecycle_in=frozenset({"active", "hot", "ember", "archived"}),
        )
        v_hits = await adapter.vector_search(vec, filters=filters, k=50)
        assert all(h.record.id != fid for h in v_hits)

        t_hits = await adapter.text_search(content, filters=filters, k=50)
        assert all(h.record.id != fid for h in t_hits)

    async def test_rebury_is_idempotent_same_tombstone_no_second_write(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 8 (brief).

        Falsify: remove the ``except KeyError`` disambiguation branch in
        ``bury`` (or let it always raise ``ToolError`` regardless of
        ``get_tombstone``) — the second ``bury`` call then either raises or
        returns a mismatched/new tombstone id.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("fact for re-bury idempotence test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        first = await engine.bury(fid, reason="first burial")
        assert first.outcome == "buried"
        count_after_first = await _tombstone_count(adapter)

        second = await engine.bury(fid, reason="second burial attempt")
        assert second.outcome == "already_buried"
        assert second.tombstone_id == first.tombstone_id
        assert second.buried_id == fid

        count_after_second = await _tombstone_count(adapter)
        assert count_after_second == count_after_first, "re-bury must not write a second tombstone"

    async def test_unknown_id_and_empty_reason_raise(self, engine_ctx: EngineCtx) -> None:
        """Case 9 (brief).

        Falsify (empty reason): comment out the ``if not reason or not
        reason.strip(): raise ToolError(...)`` guard in ``bury``.
        Falsify (unknown id): change the ``if tombstone is None: raise
        ToolError`` branch to instead fabricate a ``BuryResult`` for the
        unknown id.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        with pytest.raises(ToolError):
            await engine.bury(new_id(), reason="does not exist")

        r = await engine.save("fact for empty reason bury test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.bury(fid, reason="")


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------


class TestRevoke:
    async def test_moves_in_force_to_revoked_and_updates_timestamp(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 10 (brief).

        Falsify: skip the ``await adapter.transition(id, "revoked", at=now)``
        call in ``revoke`` (return the unchanged record instead).
        """
        engine, adapter, _embedder, clock, _id = engine_ctx
        r = await engine.save("Decision to be revoked", scope="global", kind="decision")
        assert r.hypomnema is not None
        did = r.hypomnema.id
        before = await adapter.get_by_id(did)

        clock.advance(timedelta(days=1))
        result = await engine.revoke(did, reason="no longer applies")
        assert result.outcome == "revised"
        assert result.lifecycle == "revoked"

        after = await adapter.get_by_id(did)
        assert after.lifecycle == "revoked"
        assert after.updated_at != before.updated_at
        assert after.updated_at == clock.now()

    async def test_revoked_render_shows_revoked_date_from_updated_at(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 11 (brief).

        Falsify: same mutation as case 10 — with no transition, the render
        never carries a "revoked" status token at all.
        """
        engine, _adapter, _embedder, clock, _id = engine_ctx
        r = await engine.save(
            "Decision for revoked render test", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id

        clock.advance(timedelta(days=2))
        await engine.revoke(did, reason="render date check")

        view = await engine.decision(did)
        expected_date = clock.now().strftime("%Y-%m-%d")
        assert f"revoked {expected_date}" in view.rendered

    async def test_re_revoke_is_idempotent_and_does_not_move_updated_at(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 12 (brief).

        Falsify: remove the ``lifecycle == "revoked" -> unchanged`` early
        return in ``revoke`` so it always calls ``transition`` again.
        """
        engine, adapter, _embedder, clock, _id = engine_ctx
        r = await engine.save("Decision for re-revoke test", scope="global", kind="decision")
        assert r.hypomnema is not None
        did = r.hypomnema.id

        await engine.revoke(did, reason="first revoke")
        after_first = await adapter.get_by_id(did)

        clock.advance(timedelta(days=5))
        result = await engine.revoke(did, reason="second revoke attempt")
        assert result.outcome == "unchanged"

        after_second = await adapter.get_by_id(did)
        assert after_second.updated_at == after_first.updated_at

    async def test_on_fact_and_on_superseded_decision_raise(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 13 (brief).

        Falsify (Fact case): remove the ``isinstance(record, Decision)``
        guard in ``revoke``.
        Falsify (superseded case): remove the ``lifecycle == "superseded" ->
        raise ToolError`` guard in ``revoke``.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        rf = await engine.save("a plain fact, not a decision", scope="global")
        assert rf.hypomnema is not None
        fid = rf.hypomnema.id
        with pytest.raises(ToolError):
            await engine.revoke(fid, reason="facts cannot be revoked")

        ra = await engine.save(
            "Decision A to be superseded", scope="global", kind="decision"
        )
        assert ra.hypomnema is not None
        a_id = ra.hypomnema.id
        rb = await engine.save(
            "Decision B superseding A",
            scope="global",
            kind="decision",
            supersedes=a_id,
            reason="changed",
        )
        assert rb.outcome == "saved"

        with pytest.raises(ToolError):
            await engine.revoke(a_id, reason="cannot revoke a superseded decision")

    async def test_empty_reason_raises(self, engine_ctx: EngineCtx) -> None:
        """SPEC §9.1 requires ``reason`` on all three
        disposition tools, not just ``bury``.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save(
            "Decision for revoke empty reason test", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.revoke(did, reason="")

    async def test_race_bury_between_check_and_write_raises_tool_error(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concurrent burial between the initial
        existence check and the write must surface SPEC §9.2's ToolError
        contract, never a raw adapter ``KeyError``.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save(
            "Decision for revoke race test", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id

        async def _raise_key_error(*_args: object, **_kwargs: object) -> None:
            raise KeyError(f"Unknown id: {did!s}")

        monkeypatch.setattr(adapter, "transition", _raise_key_error)

        with pytest.raises(ToolError):
            await engine.revoke(did, reason="racing a concurrent bury")


# ---------------------------------------------------------------------------
# Lineage exception + buried-decision view guard
# ---------------------------------------------------------------------------


class TestLineageAndBuriedView:
    async def test_lineage_exception_survives_ancestor_burial(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 14 (brief): burying an ancestor does NOT scrub its content
        from a descendant's birth-frozen ``supersession_chain`` snapshot
        (SPEC §10.4.3).

        Falsify: this asserts EXISTING, already-shipped behaviour (bury never
        touches a descendant's frozen snapshot; render_decision never reads
        the adapter). Falsify by removing the "Backward chain" rendering
        block in ``render.py``'s ``render_decision`` — the surviving snapshot
        title then disappears from the render even though the underlying
        data is intact, which is exactly the "a future reader mistakes this
        for an incomplete-burial bug" scenario this test guards against.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        ancestor_content = "Ancestor decision content for lineage test"
        ra = await engine.save(ancestor_content, scope="global", kind="decision")
        assert ra.hypomnema is not None
        a_id = ra.hypomnema.id
        rb = await engine.save(
            "Descendant decision content for lineage test",
            scope="global",
            kind="decision",
            supersedes=a_id,
            reason="changed",
        )
        assert rb.hypomnema is not None
        b_id = rb.hypomnema.id

        await engine.bury(a_id, reason="ancestor burial for lineage exception test")

        view = await engine.decision(b_id)
        expected_title = derive_title(ancestor_content)
        assert expected_title in view.rendered

        # decision(B) still succeeds after A is buried.
        again = await engine.decision(b_id)
        assert again.id == b_id

    async def test_buried_decision_raises_from_tool_and_resource(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 15 (brief): both the ``decision`` engine method AND the
        ``decision://{id}`` MCP resource raise for a buried Decision.

        Falsify: change the ``except KeyError: raise ToolError(...)`` branch
        in ``decision_view`` (decisions.py) to swallow the error and return
        a stub view instead.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save(
            "Decision to be buried for view test", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id
        await engine.bury(did, reason="buried decision view test")

        with pytest.raises(ToolError):
            await engine.decision(did)

        app = make_app(engine)
        # Scope 3 step 5 (SEP-2164, SDK v2 migration 2026-08-12): a buried
        # Decision is a resource-not-found, propagating untouched through
        # v2's resource-template wrapper with the original message intact.
        from mcp.server.mcpserver.exceptions import ResourceNotFoundError

        with pytest.raises(ResourceNotFoundError, match="unknown or buried"):
            await app.read_resource(f"decision://{did!s}")


# ---------------------------------------------------------------------------
# include_invalid recall filter after a validity flip
# ---------------------------------------------------------------------------


class TestIncludeInvalidAfterFlip:
    async def test_include_invalid_filter_after_validity_flip(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Case 16 (brief): guards against a validity flip writing the wrong
        column — ``include_invalid=False`` (default) still excludes the
        invalidated record; ``include_invalid=True`` returns it.

        Falsify: swap the boolean in ``set_validity``'s call to
        ``adapter.set_validity(id, valid, reason)`` to
        ``adapter.set_validity(id, not valid, reason)``.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        content = "unique content for include_invalid filter test"
        r = await engine.save(content, scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        result = await engine.set_validity(fid, valid=False, reason="mark invalid for filter test")
        assert result.validity == "invalid"

        default_recall = await engine.recall(content, scope="global")
        assert all(h.id != fid for h in default_recall.hits)

        included_recall = await engine.recall(content, scope="global", include_invalid=True)
        assert any(h.id == fid for h in included_recall.hits)


# ---------------------------------------------------------------------------
# supersede (P3 one-transaction-supersession piece, grown
# from the pre-P3 CLI-only supersede_fact)
# ---------------------------------------------------------------------------


class TestSupersedeFact:
    async def test_supersedes_and_flips_validity_atomically(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Predecessor ends up invalid AND pointed at the successor, both
        fields observable in ONE read — no intermediate state.

        Falsify: split ``adapter.replace``'s single ``merge_insert`` into two
        separate writes — a read taken between the two would see validity
        flipped but ``superseded_by`` still None (or vice versa).
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred_r = await engine.save("predecessor fact: use v1 of the API", scope="global")
        succ_r = await engine.save("successor fact: use v2 of the API", scope="global")
        assert pred_r.hypomnema is not None
        assert succ_r.hypomnema is not None
        pred_id = pred_r.hypomnema.id
        succ_id = succ_r.hypomnema.id

        result = await engine.supersede(succ_id, [pred_id], "changed")
        assert result.successor_id == succ_id
        assert len(result.edges) == 1
        assert result.edges[0][0] == pred_id

        reread = await adapter.get_by_id(pred_id)
        assert reread.validity == "invalid"
        assert reread.superseded_by == succ_id

    async def test_self_edge_raises(self, engine_ctx: EngineCtx) -> None:
        """A Fact cannot supersede itself — the distinct-pair registry's
        ``frozenset({a, b})`` bug (2026-07-31) is the standing reminder that
        same-id pairs collapse into nonsense.

        Falsify: remove the ``successor_id in predecessor_ids`` guard.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("a fact that cannot supersede itself", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.supersede(fid, [fid], "correction")

    async def test_unknown_predecessor_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("a real successor fact", scope="global")
        assert r.hypomnema is not None
        with pytest.raises(ToolError):
            await engine.supersede(r.hypomnema.id, [new_id()], "changed")

    async def test_unknown_successor_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("a real predecessor fact", scope="global")
        assert r.hypomnema is not None
        with pytest.raises(ToolError):
            await engine.supersede(new_id(), [r.hypomnema.id], "changed")

    async def test_decision_predecessor_raises(self, engine_ctx: EngineCtx) -> None:
        """Fact-arm supersede is Facts-only — a Decision on either side raises.

        Falsify: drop the ``isinstance(predecessor, Fact)`` guard.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        d = await engine.save("a decision, not a fact", scope="global", kind="decision")
        f = await engine.save("a fact successor", scope="global")
        assert d.hypomnema is not None
        assert f.hypomnema is not None
        with pytest.raises(ToolError):
            await engine.supersede(f.hypomnema.id, [d.hypomnema.id], "changed")

    async def test_decision_successor_raises(self, engine_ctx: EngineCtx) -> None:
        """Falsify: drop the ``isinstance(successor, Fact)`` guard."""
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        f = await engine.save("a fact predecessor", scope="global")
        d = await engine.save("a decision, not a fact successor", scope="global", kind="decision")
        assert f.hypomnema is not None
        assert d.hypomnema is not None
        with pytest.raises(ToolError):
            await engine.supersede(d.hypomnema.id, [f.hypomnema.id], "changed")

    async def test_already_invalid_predecessor_raises(self, engine_ctx: EngineCtx) -> None:
        """A predecessor that is already invalid (for an unrelated reason —
        plain ``set_validity(valid=False)``, never named a successor) cannot
        be superseded.

        Falsify: drop the ``predecessor.validity != "valid"`` guard.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("a fact invalidated for an unrelated reason", scope="global")
        succ = await engine.save("an unrelated successor candidate", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        await engine.set_validity(pred.hypomnema.id, valid=False, reason="unrelated invalidation")
        with pytest.raises(ToolError):
            await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")

    async def test_pre_minted_edge_id_names_the_written_edge(self, engine_ctx: EngineCtx) -> None:
        """A3 cleaner-integration piece, D14: a caller that must name the edge in a `pending`
        verdict row BEFORE the write can pre-mint `edge_id` -- the write uses it verbatim."""
        import uuid as _uuid

        from tabularium.engine.dispositions import supersede

        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for a pre-minted edge id", scope="global")
        succ = await engine.save("successor for a pre-minted edge id", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        minted = str(_uuid.uuid4())

        result = await supersede(
            successor_id=succ.hypomnema.id,
            predecessor_ids=[pred.hypomnema.id],
            reason="changed",
            note=None,
            adapter=adapter,
            clock=engine._clock,
            lock=engine._write_lock,
            edge_id=minted,
        )

        assert result.edges == ((pred.hypomnema.id, minted),)
        reread = await adapter.get_by_id(pred.hypomnema.id)
        assert getattr(reread, "supersession_edge_id", None) == minted

    async def test_edge_id_with_two_predecessors_raises(self, engine_ctx: EngineCtx) -> None:
        from tabularium.engine.dispositions import supersede

        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred_a = await engine.save("first predecessor, edge_id misuse", scope="global")
        pred_b = await engine.save("second predecessor, edge_id misuse", scope="global")
        succ = await engine.save("successor, edge_id misuse", scope="global")
        assert pred_a.hypomnema is not None
        assert pred_b.hypomnema is not None
        assert succ.hypomnema is not None

        with pytest.raises(ToolError, match="edge_id"):
            await supersede(
                successor_id=succ.hypomnema.id,
                predecessor_ids=[pred_a.hypomnema.id, pred_b.hypomnema.id],
                reason="changed",
                note=None,
                adapter=adapter,
                clock=engine._clock,
                lock=engine._write_lock,
                edge_id="some-edge-id",
            )

    async def test_already_superseded_predecessor_raises(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Second edge on the same predecessor raises — first edge wins.

        Falsify: drop the ``predecessor.superseded_by is not None`` guard.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor targeted twice", scope="global")
        succ_a = await engine.save("successor A wins the race", scope="global")
        succ_b = await engine.save("successor B loses the race", scope="global")
        assert pred.hypomnema is not None
        assert succ_a.hypomnema is not None
        assert succ_b.hypomnema is not None

        await engine.supersede(succ_a.hypomnema.id, [pred.hypomnema.id], "changed")
        with pytest.raises(ToolError):
            await engine.supersede(succ_b.hypomnema.id, [pred.hypomnema.id], "changed")

    async def test_already_superseded_successor_raises(self, engine_ctx: EngineCtx) -> None:
        """Naming an already-superseded record as a replacement is refused,
        not silently accepted. ``retired_b`` is first made superseded (by
        an unrelated record), then a THIRD fact tries to name it as ITS
        successor.

        Falsify: drop the ``successor.superseded_by is not None`` guard.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        retired_b = await engine.save("fact B, will itself be retired", scope="global")
        b_successor = await engine.save("fact D, retires B", scope="global")
        c = await engine.save("fact C, tries to name the retired B", scope="global")
        assert retired_b.hypomnema is not None
        assert b_successor.hypomnema is not None
        assert c.hypomnema is not None

        await engine.supersede(b_successor.hypomnema.id, [retired_b.hypomnema.id], "changed")
        with pytest.raises(ToolError):
            await engine.supersede(retired_b.hypomnema.id, [c.hypomnema.id], "changed")

    async def test_lost_race_value_error_reports_as_a_race(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented CAS-race ``ValueError`` (``adapter.replace``'s
        fresh-under-lock recheck) still converts to a ``ToolError`` worded
        as a lost race — the fact-supersession-fixes review, defect 3
        synthesis keeps this conversion, it does not narrow it. Same
        monkeypatch-injection idiom as ``TestSetValidity``/``TestBury``'s
        own race tests in this file (``adapter.replace`` swapped for a
        stand-in that raises exactly what the storage layer raises on a
        lost race).

        Falsify: make ``supersede`` treat EVERY ``ValueError`` as
        "unexpected" (drop the ``pydantic.ValidationError`` discrimination
        entirely, in the other direction from defect 3's own falsifiability
        test below) — this test's ``match`` would then fail because the
        message reads "unexpected" instead of "lost a concurrent... race".
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for the lost-race wording test", scope="global")
        succ = await engine.save("successor for the lost-race wording test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id

        async def _raise_cas_race(*_args: object, **_kwargs: object) -> None:
            raise ValueError(
                f"superseded_by is already set on Fact {pred_id!s} — cannot "
                "supersede twice (lost a concurrent supersession race)"
            )

        monkeypatch.setattr(adapter, "replace", _raise_cas_race)

        with pytest.raises(ToolError, match="lost a concurrent supersession race"):
            await engine.supersede(succ.hypomnema.id, [pred_id], "changed")

    async def test_unexpected_validation_error_does_not_read_as_a_race(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fact-supersession-fixes review, defect 3: ``pydantic.
        ValidationError`` subclasses ``ValueError``, so a genuine
        data-integrity failure escaping from deep inside the storage
        layer's ``Fact.model_validate`` call must NOT be folded into the
        same "lost a concurrent race" wording as an ordinary lost race
        (the two ``ValueError``s the sibling test above exercises) — the
        conversion to ``ToolError`` stays either way (SPEC: report, never crash after a landed
        write), but a
        ``ValidationError`` is marked as an UNEXPECTED failure.

        The injected exception is a REAL ``pydantic.ValidationError``
        (triggered via the same invariant ``Fact._check_supersession_
        consistency`` enforces — ``superseded_by`` set with
        ``validity='valid'``), not a hand-built stand-in, so this proves
        the discrimination is on TYPE, not on message text (the module
        docstring's own "message text is not a contract" principle).

        Falsify: remove the ``isinstance(exc, pydantic.ValidationError)``
        branch in ``supersede`` (fold everything back into the
        single generic wording) — this test's ``match`` would then fail
        because the message reads "lost a concurrent race" instead of
        "unexpected"/"NOT a lost race".
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save(
            "predecessor for the validation-error wording test", scope="global"
        )
        succ = await engine.save(
            "successor for the validation-error wording test", scope="global"
        )
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        stored = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(stored, Fact)
        with pytest.raises(pydantic.ValidationError) as exc_info:
            Fact.model_validate(
                {
                    **stored.model_dump(mode="python"),
                    "superseded_by": new_id(),
                    "validity": "valid",
                }
            )
        real_validation_error = exc_info.value

        async def _raise_validation_error(*_args: object, **_kwargs: object) -> None:
            raise real_validation_error

        monkeypatch.setattr(adapter, "replace", _raise_validation_error)

        with pytest.raises(ToolError, match="unexpected data-integrity failure") as tool_exc_info:
            await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")
        assert "NOT a lost race" in str(tool_exc_info.value)

    async def test_non_valueerror_replace_failure_still_reports_cleanly(
        self, engine_ctx: EngineCtx, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine storage
        I/O failure need not surface as ``ValueError`` — ``supersede`` must
        convert ANY exception from ``adapter.replace``, not only
        ``ValueError``, into a clean ``ToolError``.

        Falsify: narrow the ``except`` clause in ``supersede`` back to
        ``except ValueError`` — this test's ``pytest.raises(ToolError)``
        goes red as the injected ``RuntimeError`` propagates raw instead.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save(
            "predecessor for the non-ValueError supersede fault test", scope="global"
        )
        succ = await engine.save(
            "successor for the non-ValueError supersede fault test", scope="global"
        )
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        async def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated disk I/O failure")

        monkeypatch.setattr(adapter, "replace", _raise_runtime_error)

        with pytest.raises(ToolError, match="supersede:"):
            await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")

    async def test_set_validity_true_on_superseded_fact_is_reachable_and_undefined(
        self, engine_ctx: EngineCtx
    ) -> None:
        """TRIPWIRE (brief Scope 3 step 4, Out-of-scope "Un-supersession"):
        ``set_validity(valid=True)`` on a superseded Fact IS reachable — the
        OPERATION remains permanently blocked VIA THIS TOOL — but the ERROR
        SHAPE is no longer undefined (fact-supersession-fixes review, defect
        4): ``set_validity`` now checks for this state proactively, before
        ever calling the adapter, and raises a clean ``ToolError`` naming
        it. Before this fix, the same call surfaced a raw
        ``pydantic.ValidationError`` escaping the engine boundary
        (``Fact``'s own consistency validator rejecting the out-of-sync
        reconstruction the adapter attempted) — this test used to document
        that reachability with a bare ``pytest.raises(Exception)``; it is
        tightened here to the specific, now-defined exception type, keeping
        the same documenting intent (the STATE stays undefined and blocked
        via ``set_validity`` specifically; only the exception shape is no
        longer an accident of pydantic's own validator surfacing raw). This
        assertion is UNCHANGED by the 2026-08-11 fact-supersession-undo
        brief: ``set_validity`` still refuses this
        exact call, worded exactly this way (``match="superseded"`` still
        matches, the message now also names ``unsupersede_fact`` as the
        correct tool — see ``TestUnsupersedeFact`` below for the actual
        repair path, which does NOT go through ``set_validity`` at all).

        Falsifiability: remove the proactive guard in ``set_validity`` and
        this test goes red with a raw ``pydantic.ValidationError`` instead
        of the expected ``ToolError``.
        """
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for the set_validity tripwire", scope="global")
        succ = await engine.save("successor for the set_validity tripwire", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")

        with pytest.raises(ToolError, match="superseded"):
            await engine.set_validity(
                pred.hypomnema.id, valid=True, reason="attempting to un-supersede"
            )


# ---------------------------------------------------------------------------
# unsupersede
# — the undo path on both arms, replacing the Fact-only unsupersede_fact
#.
# ---------------------------------------------------------------------------


class TestUnsupersede:
    """Fact-arm happy path + refusals."""

    async def test_unknown_id_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        with pytest.raises(ToolError):
            await engine.unsupersede(new_id(), reason="unknown id")

    async def test_buried_id_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("a fact that will be buried", scope="global")
        assert r.hypomnema is not None
        await engine.bury(r.hypomnema.id, reason="buried before unsupersede attempt")
        with pytest.raises(ToolError):
            await engine.unsupersede(r.hypomnema.id, reason="buried id")

    async def test_empty_reason_raises(self, engine_ctx: EngineCtx) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for the empty-reason test", scope="global")
        succ = await engine.save("successor for the empty-reason test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")
        with pytest.raises(ToolError, match="reason"):
            await engine.unsupersede(pred.hypomnema.id, reason="")

    async def test_never_superseded_fact_refuses(self, engine_ctx: EngineCtx) -> None:
        """Design call: unlike the
        Fact-only predecessor this replaces, calling ``unsupersede`` on a
        row that is not currently superseded REFUSES — it is not a
        monotonic re-op.

        Falsify: remove the ``record.superseded_by is None`` guard in
        ``unsupersede`` — this call would then proceed toward
        ``adapter.restore``, which raises a differently-worded
        ``ValueError`` instead of this specific ``ToolError``.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        r = await engine.save("a fact never superseded", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        before = await adapter.get_by_id(fid)

        with pytest.raises(ToolError, match="not currently superseded"):
            await engine.unsupersede(fid, reason="nothing to undo")

        after = await adapter.get_by_id(fid)
        assert isinstance(after, Fact)
        assert after.updated_at == before.updated_at

    async def test_decision_id_missing_successor_state_named(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        d = await engine.save("a decision, not a fact", scope="global", kind="decision")
        assert d.hypomnema is not None
        with pytest.raises(ToolError, match="not currently superseded"):
            await engine.unsupersede(d.hypomnema.id, reason="wrong arm")

    async def test_successful_clear_restores_validity_and_clears_pointer(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Happy path: the predecessor's pointer, reason class, note, and
        edge id all clear, and validity restores — one write, observed in
        ONE read.

        Falsify: split the storage-layer write into two calls — a read
        taken between them would see one field restored and not the other.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred_r = await engine.save(
            "wrongly superseded predecessor", scope="global"
        )
        succ_r = await engine.save("the successor that superseded it", scope="global")
        assert pred_r.hypomnema is not None
        assert succ_r.hypomnema is not None
        pred_id = pred_r.hypomnema.id
        succ_id = succ_r.hypomnema.id

        await engine.supersede(succ_id, [pred_id], "changed", note="a free-text note")
        superseded = await adapter.get_by_id(pred_id)
        assert superseded.validity == "invalid"
        assert superseded.superseded_by == succ_id

        result = await engine.unsupersede(pred_id, reason="the edge was wrong")
        assert result.shape == "fact_both_live"
        assert result.arm == "fact"
        assert result.predecessor_id == pred_id
        assert result.successor_id == succ_id
        assert result.withdrawn_reason == "changed"
        assert result.withdrawn_note == "a free-text note"

        reread = await adapter.get_by_id(pred_id)
        assert isinstance(reread, Fact)
        assert reread.validity == "valid"
        assert reread.superseded_by is None
        assert reread.supersession_reason is None
        assert reread.supersession_note is None
        assert reread.supersession_edge_id is None

        succ_after = await adapter.get_by_id(succ_id)
        assert isinstance(succ_after, Fact)

    async def test_called_twice_refuses_on_second_call(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Idempotent RETRY (edge-id keyed event) is not the same as an
        idempotent CALL: the second call refuses (a prior decision's own
        worked example)."""
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor, unsuperseded once, then again", scope="global")
        succ = await engine.save("successor for the double-unsupersede test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id
        await engine.supersede(succ.hypomnema.id, [pred_id], "changed")
        await engine.unsupersede(pred_id, reason="first undo")
        with pytest.raises(ToolError, match="not currently superseded"):
            await engine.unsupersede(pred_id, reason="second undo")

    async def test_event_persisted_in_unsupersedes_table(
        self, engine_ctx: EngineCtx
    ) -> None:
        """The undo is recorded in the ``unsupersedes`` table — content-free,
        keyed by edge id — never as a
        ``ReinforcementEvent`` (D10: that write path is removed)."""
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for the event-persistence test", scope="global")
        succ = await engine.save("successor for the event-persistence test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id

        await engine.supersede(succ.hypomnema.id, [pred_id], "changed", note="a note")
        edge_id_before = (await adapter.get_by_id(pred_id)).supersession_edge_id
        assert edge_id_before is not None

        await engine.unsupersede(
            pred_id, reason="the Cleaner drew this edge on a true Fact — retracting it"
        )

        events = await adapter.list_unsupersedes()
        assert len(events) == 1
        event = events[0]
        assert event.edge_id == edge_id_before
        assert event.predecessor_id == pred_id
        assert event.successor_id == succ.hypomnema.id
        assert event.arm == "fact"
        assert event.withdrawn_reason == "changed"
        assert event.withdrawn_note == "a note"
        assert event.reason == (
            "the Cleaner drew this edge on a true Fact — retracting it"
        )

        reinforcement_events = await adapter.query_events(
            target_id=pred_id, source="unsupersede"
        )
        assert len(reinforcement_events) == 0

    async def test_second_supersede_after_unsupersede_succeeds_with_fresh_edge_id(
        self, engine_ctx: EngineCtx
    ) -> None:
        """A second ``supersede`` on the now-cleared predecessor succeeds
        with a FRESH edge id — proving the guard
        RE-OPENS rather than was weakened, and that the new generation is
        distinguishable from the undone one.

        Falsify: change ``unsupersede``/the storage resolver to leave some
        other marker of "was once superseded" that the forward edge's guard
        also checks, or have the forward edge REUSE the cleared edge id —
        this test would then fail either on the ``supersede`` call itself
        or on the ``first_edge_id != second_edge_id`` assertion.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save(
            "predecessor, superseded twice with an undo between", scope="global"
        )
        wrong_succ = await engine.save("the wrong successor", scope="global")
        right_succ = await engine.save("the correct successor", scope="global")
        assert pred.hypomnema is not None
        assert wrong_succ.hypomnema is not None
        assert right_succ.hypomnema is not None
        pred_id = pred.hypomnema.id

        await engine.supersede(wrong_succ.hypomnema.id, [pred_id], "changed")
        first_edge_id = (await adapter.get_by_id(pred_id)).supersession_edge_id
        assert first_edge_id is not None
        await engine.unsupersede(pred_id, reason="wrong successor named")

        await engine.supersede(right_succ.hypomnema.id, [pred_id], "changed")
        updated = await adapter.get_by_id(pred_id)
        assert updated.validity == "invalid"
        assert updated.superseded_by == right_succ.hypomnema.id
        assert updated.supersession_edge_id is not None
        assert updated.supersession_edge_id != first_edge_id

    async def test_log_unsupersede_completed_emitted(
        self, engine_ctx: EngineCtx, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("predecessor for the completed-log test", scope="global")
        succ = await engine.save("successor for the completed-log test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")

        with caplog.at_level(logging.INFO, logger="tabularium"):
            await engine.unsupersede(pred.hypomnema.id, reason="log check")

        matching = [
            r for r in caplog.records if r.getMessage().startswith("unsupersede_completed")
        ]
        assert len(matching) == 1, caplog.records
        record = matching[0]
        assert record.arm == "fact"  # type: ignore[attr-defined]
        assert record.shape == "fact_both_live"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# unsupersede — Decision arm
# ---------------------------------------------------------------------------


class TestUnsupersedeDecisionArm:
    async def test_happy_path_restores_and_deletes_successor(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("Decision A, wrongly superseded", scope="global", kind="decision")
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ = await engine.save(
            "Decision B superseding A",
            scope="global",
            kind="decision",
            supersedes=pred_id,
            reason="changed",
        )
        assert succ.hypomnema is not None
        succ_id = succ.hypomnema.id

        result = await engine.unsupersede(pred_id, reason="wrong edge, an automated judge")
        assert result.shape == "decision_restored"
        assert result.arm == "decision"
        assert result.successor_id == succ_id

        restored = await adapter.get_by_id(pred_id)
        assert restored.lifecycle == "in_force"
        assert restored.superseded_by is None
        with pytest.raises(KeyError):
            await adapter.get_by_id(succ_id)
        tombstone = await adapter.get_tombstone(succ_id)
        assert tombstone is not None
        assert tombstone.restores == pred_id

    async def test_validity_preserved_when_it_was_invalid_before(
        self, engine_ctx: EngineCtx
    ) -> None:
        """D8: the restore never forces validity back to valid on the
        Decision arm — it rides through exactly as it was."""
        engine, adapter, _embedder, _clock, _id = engine_ctx
        pred = await engine.save("Decision A, superseded then separately invalidated",
                                  scope="global", kind="decision")
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ = await engine.save(
            "Decision B", scope="global", kind="decision", supersedes=pred_id, reason="changed",
        )
        assert succ.hypomnema is not None
        await engine.set_validity(pred_id, valid=False, reason="separately invalid")

        await engine.unsupersede(pred_id, reason="undo anyway")
        restored = await adapter.get_by_id(pred_id)
        assert restored.lifecycle == "in_force"
        assert restored.validity == "invalid"

    async def test_successor_superseded_refused_naming_state(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a.hypomnema.id, reason="changed",
        )
        assert b.hypomnema is not None
        c = await engine.save(
            "Decision C", scope="global", kind="decision",
            supersedes=b.hypomnema.id, reason="changed",
        )
        assert c.hypomnema is not None

        with pytest.raises(ToolError, match="not in_force"):
            await engine.unsupersede(a.hypomnema.id, reason="B has its own history now")

    async def test_successor_revoked_refused_naming_state(
        self, engine_ctx: EngineCtx
    ) -> None:
        """A revoked Decision has its own history the undo's delete would
        erase — refused, naming that state (D9)."""
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a.hypomnema.id, reason="changed",
        )
        assert b.hypomnema is not None
        await engine.revoke(b.hypomnema.id, reason="B itself turned out wrong")

        with pytest.raises(ToolError, match="not in_force"):
            await engine.unsupersede(a.hypomnema.id, reason="B is revoked now")

    async def test_successor_buried_refused_naming_state(
        self, engine_ctx: EngineCtx
    ) -> None:
        engine, _adapter, _embedder, _clock, _id = engine_ctx
        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a.hypomnema.id, reason="changed",
        )
        assert b.hypomnema is not None
        await engine.bury(b.hypomnema.id, reason="unrelated cleanup")

        with pytest.raises(ToolError, match="unknown or buried"):
            await engine.unsupersede(a.hypomnema.id, reason="B is gone now")

    async def test_conflicting_tombstone_refused_before_the_destructive_transaction(
        self, engine_ctx: EngineCtx
    ) -> None:
        """The live undo
        path used to check only ``existing_tombstone is None`` before
        skipping the plant step and going straight to ``restore`` — never
        the tombstone's own ``restores`` field. A directly-planted
        tombstone naming an unrelated predecessor must refuse, not
        silently authorise the delete.
        """
        engine, adapter, _embedder, _clock, _id = engine_ctx
        a = await engine.save("Decision A", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B", scope="global", kind="decision",
            supersedes=a_id, reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id
        x = await engine.save("Decision X, unrelated", scope="global", kind="decision")
        assert x.hypomnema is not None
        x_id = x.hypomnema.id

        await adapter.plant_tombstone(b_id, "unrelated crafted tombstone", restores=x_id)

        with pytest.raises(ToolError, match="does not authorise"):
            await engine.unsupersede(a_id, reason="wrong successor")

        unchanged_a = await adapter.get_by_id(a_id)
        assert unchanged_a.lifecycle == "superseded"
        assert unchanged_a.superseded_by == b_id
        still_present_b = await adapter.get_by_id(b_id)
        assert still_present_b is not None

    async def test_stale_pending_tombstone_never_refuses_the_live_call(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Edit-during-pending-undo fix, point 4: a
        pending tombstone left by an earlier attempt can go stale (the
        successor was edited since it was planted) without ever being
        noticed until the NEXT live call. Unlike startup repair, the live
        path must NEVER refuse for staleness — the caller is asking right
        now and sees the successor as it stands right now. It withdraws
        the stale tombstone and replants fresh with the current hash
        before completing the same call.
        """
        engine, adapter, embedder, _clock, _id = engine_ctx
        a = await engine.save("Decision A, wrongly superseded", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B superseding A",
            scope="global",
            kind="decision",
            supersedes=a_id,
            reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id
        edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert edge_id is not None

        await adapter.plant_tombstone(b_id, "unsupersede: earlier attempt", restores=a_id)

        edited_content = "Decision B, edited after the plant"
        await edit_content(adapter, b_id, edited_content, await embedder.embed(edited_content))

        result = await engine.unsupersede(a_id, reason="wrong successor, retried live")
        assert result.shape == "decision_restored"

        restored_a = await adapter.get_by_id(a_id)
        assert restored_a.lifecycle == "in_force"
        with pytest.raises(KeyError):
            await adapter.get_by_id(b_id)

        events = [e for e in await adapter.list_unsupersedes() if e.edge_id == edge_id]
        assert len(events) == 1

    async def test_live_retry_reopens_a_refused_edge_and_completes(
        self, engine_ctx: EngineCtx
    ) -> None:
        """Durable refusal follow-up, point 3: a live ``unsupersede`` call
        against an edge whose event is ALREADY refused (``refused_at`` set,
        its tombstone already withdrawn — a fully-resolved refusal from an
        earlier repair pass) is a fresh request, not blocked by the
        durable marker. ``append_unsupersede`` clears ``refused_at`` as
        part of reopening the edge; the call then plants and restores as
        an ordinary undo, leaving exactly one event row behind with
        ``refused_at`` NULL again.
        """
        engine, adapter, _embedder, clock, _id = engine_ctx
        a = await engine.save("Decision A, wrongly superseded", scope="global", kind="decision")
        assert a.hypomnema is not None
        a_id = a.hypomnema.id
        b = await engine.save(
            "Decision B superseding A",
            scope="global",
            kind="decision",
            supersedes=a_id,
            reason="changed",
        )
        assert b.hypomnema is not None
        b_id = b.hypomnema.id
        edge_id = (await adapter.get_by_id(a_id)).supersession_edge_id
        assert edge_id is not None
        now = clock.now()

        event = UnsupersedeEvent(
            id=uuid.uuid4(), edge_id=edge_id, predecessor_id=a_id, successor_id=b_id,
            arm="decision", withdrawn_reason="changed", withdrawn_note=None,
            reason="an earlier undo attempt, since refused", at=now,
        )
        await adapter.append_unsupersede(event)
        await adapter.mark_unsupersede_refused(edge_id, at=now)

        result = await engine.unsupersede(a_id, reason="retry, deliberate, right now")
        assert result.shape == "decision_restored"

        restored_a = await adapter.get_by_id(a_id)
        assert restored_a.lifecycle == "in_force"
        with pytest.raises(KeyError):
            await adapter.get_by_id(b_id)

        events = [e for e in await adapter.list_unsupersedes() if e.edge_id == edge_id]
        assert len(events) == 1
        assert events[0].refused_at is None


# ---------------------------------------------------------------------------
# unsupersede — storage-failure honesty, both arms (fault-injected via
# FaultInjectingAdapter, the same tool test_fact_supersession.py uses)
# ---------------------------------------------------------------------------


class TestUnsupersedeFaultInjection:
    async def _engine(self) -> tuple[Tabularium, MemoryAdapter, FaultInjectingAdapter]:
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        raw_adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
        wrapper = FaultInjectingAdapter(raw_adapter)
        cfg = _make_config()
        engine = Tabularium(as_adapter(wrapper), embedder, clock, cfg)
        await engine.initialize()
        return engine, raw_adapter, wrapper

    async def test_fact_restore_failure_leaves_event_and_is_retryable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine, raw_adapter, wrapper = await self._engine()
        pred = await engine.save("predecessor for the restore-fault test", scope="global")
        succ = await engine.save("successor for the restore-fault test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None
        pred_id = pred.hypomnema.id
        await engine.supersede(succ.hypomnema.id, [pred_id], "changed")

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated restore failure"))
        wrapper.arm()

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError):
                await engine.unsupersede(pred_id, reason="the edge was wrong")

        matching = [
            r for r in caplog.records if r.getMessage().startswith("unsupersede_failed")
        ]
        assert len(matching) == 1, caplog.records

        still_superseded = await raw_adapter.get_by_id(pred_id)
        assert still_superseded.superseded_by == succ.hypomnema.id
        events = await raw_adapter.list_unsupersedes()
        assert len(events) == 1

        result = await engine.unsupersede(pred_id, reason="retry after the fault")
        assert result.shape == "fact_both_live"
        events_after_retry = await raw_adapter.list_unsupersedes()
        assert len(events_after_retry) == 1, "the retry must not append a second undo row"

    async def test_decision_restore_failure_leaves_tombstone_and_is_retryable(
        self,
    ) -> None:
        engine, raw_adapter, wrapper = await self._engine()
        pred = await engine.save("Decision A for the restore-fault test",
                                  scope="global", kind="decision")
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ = await engine.save(
            "Decision B", scope="global", kind="decision", supersedes=pred_id, reason="changed",
        )
        assert succ.hypomnema is not None
        succ_id = succ.hypomnema.id

        wrapper.fault(on="restore", call=1, exc=RuntimeError("simulated restore failure"))
        wrapper.arm()

        with pytest.raises(ToolError):
            await engine.unsupersede(pred_id, reason="the edge was wrong")

        still_present = await raw_adapter.get_by_id(succ_id)
        assert still_present is not None
        tombstone = await raw_adapter.get_tombstone(succ_id)
        assert tombstone is not None
        assert tombstone.restores == pred_id

        result = await engine.unsupersede(pred_id, reason="retry after the fault")
        assert result.shape == "decision_restored"
        with pytest.raises(KeyError):
            await raw_adapter.get_by_id(succ_id)

    async def test_decision_plant_tombstone_failure_leaves_only_the_event(
        self,
    ) -> None:
        engine, raw_adapter, wrapper = await self._engine()
        pred = await engine.save("Decision A for the plant-tombstone-fault test",
                                  scope="global", kind="decision")
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id
        succ = await engine.save(
            "Decision B", scope="global", kind="decision", supersedes=pred_id, reason="changed",
        )
        assert succ.hypomnema is not None
        succ_id = succ.hypomnema.id

        wrapper.fault(on="plant_tombstone", call=1, exc=RuntimeError("simulated fault"))
        wrapper.arm()

        with pytest.raises(ToolError):
            await engine.unsupersede(pred_id, reason="the edge was wrong")

        events = await raw_adapter.list_unsupersedes()
        assert len(events) == 1
        tombstone = await raw_adapter.get_tombstone(succ_id)
        assert tombstone is None
        still_superseded = await raw_adapter.get_by_id(pred_id)
        assert still_superseded.superseded_by == succ_id

        result = await engine.unsupersede(pred_id, reason="retry after the fault")
        assert result.shape == "decision_restored"
        events_after_retry = await raw_adapter.list_unsupersedes()
        assert len(events_after_retry) == 1, "the retry must not append a second undo row"
