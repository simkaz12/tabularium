"""``set_hot`` — owner-curated hot membership (M2 slice C, SPEC §1.2 /
§8.5 / §9.1 / §9.2, ADR 0020).

Dual-parameterized over both storage adapters (memory / lancedb) via the
``adapter_id`` fixture + ``_build_engine`` helper, mirroring the
``engine_ctx`` pattern in ``tests/test_engine_dispositions.py`` (precedent) but allowing a per-test
``cfg`` override — needed here because
several tests below tune ``hot_cap`` away from its default.

Falsifiability (step 6 of the task): the cap-guardrail test and one
housekeeping-demotion test (invalidate) were each proven sensitive to a real
defect by deliberately breaking the corresponding guard in
``src/tabularium/engine/dispositions.py`` / ``src/tabularium/engine/context.py``,
watching the named test go red for the stated reason, then restoring. The
captured failures are reported in the task verdict, not duplicated here —
each affected test's docstring names the mutation that would break it.
"""

from __future__ import annotations

import asyncio
import pathlib
import typing
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.engine.backfill import NoBackfill
from tabularium.engine.context import select_default
from tabularium.engine.engine_context import EngineContext
from tabularium.engine.scoring import compute_importance
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter

# Adapter-construction pattern reused verbatim from the conformance suite
# (see tests/test_eval_b5.py / tests/test_engine_dispositions.py)
# rather than a third construction pattern.
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, TestClock

AnyHotAdapter = MemoryAdapter | LanceDBAdapter
# A larger dim than the B2 suite's usual 64 — several tests here save a
# double-digit number of near-identical-looking (but content-distinct)
# fixture strings, and ``FakeEmbedder``'s SHA-256 slot collision probability
# is ~1/dim per pair; 64 was measured to actually collide for a couple of
# this file's fixture strings, which is a deterministic false near-dup
# bounce, not flakiness (SHA-256 has no PYTHONHASHSEED dependence). 4096
# makes that collision probability negligible at this file's fixture count.
_DIM = 4096


def _select_ctx(adapter: AnyHotAdapter, cfg: EngineConfig, clock: TestClock) -> EngineContext:
    return EngineContext(
        adapter=adapter,
        embedder=FakeEmbedder(),
        clock=clock,
        cfg=cfg,
        write_lock=asyncio.Lock(),
        assessor=None,
        token_counter=None,
        backfill=NoBackfill(),
    )


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _build_engine(
    adapter_id: str,
    tmp_path: pathlib.Path,
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
    dim: int = _DIM,
) -> tuple[Tabularium, AnyHotAdapter, FakeEmbedder, TestClock]:
    """A fresh, initialized engine + adapter + embedder + clock.

    Unlike ``test_engine_dispositions.py``'s ``engine_ctx`` fixture, ``cfg``
    is caller-supplied (not hard-coded inside the fixture) — several tests
    here need a non-default ``hot_cap``.
    """
    embedder = FakeEmbedder(dim=dim)
    clk = clock if clock is not None else _make_clock()
    adapter: AnyHotAdapter
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
# THE marquee property: nothing anywhere auto-promotes to hot
# ---------------------------------------------------------------------------


class TestNoAutoPromotion:
    async def test_high_importance_never_auto_promotes_to_hot(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The property this whole task exists to guarantee: no engine write
        path (save's auto-reinforce, ember's recurrence-reinforce, recall's
        fallback-tier revival) ever moves a Fact's lifecycle to "hot" — only
        ``set_hot`` does. Every call below is a REAL engine call (``engine.
        save`` / ``engine.ember`` / ``engine.recall``), not a mock — this
        drives the actual write paths, not an assertion on a stub.

        The active-Fact loop below reinforces via a whitespace variant of
        the same content (P7 admission-order piece, step 3 exact-key match)
        rather than a byte-identical resubmission, since the latter now
        acks ``already_saved`` (step 1) and never reinforces at all — which
        would make the "high importance" setup vacuous.
        """
        cfg = _make_config()
        clock = _make_clock()
        engine, adapter, _embedder, clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg, clock=clock
        )

        # --- drive `save` + its auto-reinforce path, on several Facts ---
        active_ids: list[HypomnemaId] = []
        for i in range(4):
            content = f"hostile hot-promotion candidate active fact number {i}"
            r = await engine.save(content, scope="global")
            assert r.hypomnema is not None
            fid = r.hypomnema.id
            active_ids.append(fid)
            for _ in range(12):
                clock.advance(timedelta(hours=1))
                # A whitespace variant (P7 admission-order piece, step 3
                # exact-key match) reinforces via the exact-key path — a
                # byte-identical resubmission acks already_saved instead
                # (step 1), which never reinforces.
                rep = await engine.save(content + " ", scope="global")
                assert rep.outcome == "reinforced"

        # --- drive `ember` + its own recurrence-reinforce path ---
        for i in range(3):
            content = f"hostile hot-promotion candidate ember fact number {i}"
            r = await engine.ember(content, scope="global", valence="guideline")
            assert r.hypomnema is not None
            for _ in range(12):
                clock.advance(timedelta(hours=1))
                rep = await engine.ember(content, scope="global", valence="guideline")
                assert rep.outcome == "reinforced"

        # --- drive `recall`'s fallback-revival write path on an archived Fact ---
        archived_query = "hostile hot-promotion candidate archived fact for revival"
        archived_r = await engine.save(archived_query, scope="global")
        assert archived_r.hypomnema is not None
        archived_id = archived_r.hypomnema.id
        await adapter.transition(archived_id, "archived", at=clock.now())
        for _ in range(6):
            clock.advance(timedelta(hours=1))
            await engine.recall(archived_query, scope="global")

        # Sanity: importance really did get high for the most-reinforced
        # active Fact — otherwise "nothing promotes" would hold for the
        # boring reason that nothing ever qualified for promotion.
        rec = await adapter.get_by_id(active_ids[0])
        assert isinstance(rec, Fact)
        imp = compute_importance(rec, now=clock.now(), cfg=cfg)
        assert imp > 0.60, (
            f"test setup failed to reach high importance ({imp}); the "
            "'nothing auto-promotes' assertion below would be vacuous"
        )

        # THE PROPERTY: no Fact anywhere carries lifecycle="hot" — the owner
        # never called set_hot, so the hot set must be empty, full stop,
        # regardless of how important anything got via any write path.
        hot_ids: set[HypomnemaId] = set()
        async for meta in await adapter.scan(lifecycle_in=frozenset({"hot"})):
            hot_ids.add(meta.id)
        assert hot_ids == set()


# ---------------------------------------------------------------------------
# Promote / demote / idempotency
# ---------------------------------------------------------------------------


class TestSetHotPromoteDemote:
    async def test_promotes_active_to_hot(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("active fact to be promoted to hot", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        before = await adapter.get_by_id(fid)

        result = await engine.set_hot(fid, hot=True, roles=["decider"])
        assert isinstance(result, Fact)
        assert result.lifecycle == "hot"

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "hot"
        assert after.content == before.content

    async def test_demotes_hot_to_active(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("hot fact to be demoted to active", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        result = await engine.set_hot(fid, hot=False)
        assert result.lifecycle == "active"

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "active"

    async def test_promote_idempotent_on_already_hot_with_same_roles_no_write(
        self, adapter_id: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falsify: remove the ``tuple(roles) == (record.audience_hint or
        ()) -> return record`` early return in ``set_hot`` so it always
        calls ``adapter.transition``.

        Curation block REVERSES the original shape of this test:
        ``roles`` is now required whenever ``hot=True`` (``roles=None``
        raises unconditionally — see ``TestSetHotRequiresRoles`` below), so
        the idempotent no-write path on the promote side is reachable only
        by supplying the EXACT SAME ``roles`` on both calls, never by
        omitting them.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("already-hot fact for idempotency test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "adapter.transition must not be called on an already-hot "
                "re-promote with the SAME roles"
            )

        monkeypatch.setattr(adapter, "transition", _boom)

        result = await engine.set_hot(fid, hot=True, roles=["decider"])
        assert result.lifecycle == "hot"
        assert result.audience_hint == ("decider",)

    async def test_demote_idempotent_on_already_active_no_write(
        self, adapter_id: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falsify: remove the ``lifecycle == "active" -> return record``
        early return in ``set_hot`` so it always calls ``adapter.transition``.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("already-active fact for idempotency test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "adapter.transition must not be called on an already-active re-demote"
            )

        monkeypatch.setattr(adapter, "transition", _boom)

        result = await engine.set_hot(fid, hot=False)
        assert result.lifecycle == "active"


# ---------------------------------------------------------------------------
# roles required on hot=True (curation block) — "no caller has no
# role", reversing the original "existing callers unaffected" clause.
# ---------------------------------------------------------------------------


class TestSetHotRequiresRoles:
    async def test_promote_without_roles_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``set_hot(id, hot=True)`` with no ``roles`` argument raises --
        unconditionally, before the record is even fetched (an unknown id
        would otherwise raise, too, but for the wrong reason).

        Falsifiability (step 6): delete the ``if hot and roles is None:
        raise ToolError(...)`` guard in ``set_hot`` -- this test's
        ``pytest.raises`` block goes red (no exception). See the task
        verdict for the captured failure output.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact for the roles-required test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="requires roles"):
            await engine.set_hot(fid, hot=True)

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "active", "rejected promotion must not write anything"

    async def test_promote_without_roles_error_names_registered_roles(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("another fact for the roles-required test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError) as exc_info:
            await engine.set_hot(fid, hot=True)
        msg = str(exc_info.value)
        for role in ("decider", "builder", "reviewer", "navigator"):
            assert role in msg

    async def test_promote_without_roles_raises_even_with_existing_capture_time_hint(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """This DELIBERATELY REVERSES the grounding brief's original
        "roles=None leaves any existing hint untouched, every pre-existing
        caller is unaffected" clause: even a Fact that already carries a
        real ``audience_hint`` from write time still requires an EXPLICIT
        ``roles`` at promotion time -- the caller must consider the
        audience at the moment of curation, not rely on a capture-time
        guess it never confirmed."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("builder")
        r = await engine.save(
            "atom captured with a builder hint, promotion attempted with no roles",
            scope="global",
            audience_hint=["builder"],
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="requires roles"):
            await engine.set_hot(fid, hot=True)

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "active", "rejected promotion must not write anything"
        assert after.audience_hint == ("builder",), "the capture-time hint must survive untouched"

    async def test_demote_without_roles_still_works(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The other half of the acceptance criterion: ``hot=False`` needs
        no audience decision to remove something from the tier."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a hot fact to demote with no roles", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        result = await engine.set_hot(fid, hot=False)
        assert result.lifecycle == "active"

    async def test_promote_with_explicit_empty_roles_is_legal(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles=[]`` (explicit empty) is a deliberate "hot, reaching no
        role" choice -- distinct from silently omitting the argument
        (``roles=None``), which raises. Only the latter is a caller bug."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "a fact promoted hot with an explicit empty roles list", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        result = await engine.set_hot(fid, hot=True, roles=[])
        assert result.lifecycle == "hot"
        # A falsy tuple normalizes to None on write (both adapters agree —
        # see StorageAdapter.transition's tri-state docstring).
        assert result.audience_hint is None


# ---------------------------------------------------------------------------
# hot_cap — payload-size guardrail enforced by rejection, never eviction
# ---------------------------------------------------------------------------


class TestSetHotCap:
    async def test_promoting_past_cap_raises_and_store_unchanged(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability (step 6, break #1): delete the
        ``if len(occupants) >= cap: raise ToolError(...)`` guard in
        ``set_hot`` and watch this test fail — the third ``set_hot`` call
        would silently succeed instead of raising, and ``ids[2]`` would end
        up ``hot`` instead of staying ``active``. See the task verdict for
        the captured failure output.
        """
        cfg = _make_config(hot_cap_global=2)
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )

        ids: list[HypomnemaId] = []
        for i in range(3):
            r = await engine.save(f"hot cap candidate fact number {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        hot1 = await engine.set_hot(ids[0], hot=True, roles=["decider"])
        assert hot1.lifecycle == "hot"
        hot2 = await engine.set_hot(ids[1], hot=True, roles=["decider"])
        assert hot2.lifecycle == "hot"

        with pytest.raises(ToolError) as exc_info:
            await engine.set_hot(ids[2], hot=True, roles=["decider"])
        msg = str(exc_info.value)
        assert str(ids[0]) in msg
        assert str(ids[1]) in msg

        # Store unchanged: the third candidate stayed active.
        rec3 = await adapter.get_by_id(ids[2])
        assert rec3.lifecycle == "active"

        hot_ids: set[HypomnemaId] = set()
        async for meta in await adapter.scan(lifecycle_in=frozenset({"hot"})):
            hot_ids.add(meta.id)
        assert hot_ids == {ids[0], ids[1]}


# ---------------------------------------------------------------------------
# Per-(scope, role) hot_cap (curation block, owner decision
# 2026-08-03: `global` 8 / project scopes 5). Counted via
# `engine.context.routes_to` — the same predicate the push-tier delivery
# filter uses — so cap and payload cannot drift apart.
# ---------------------------------------------------------------------------


class TestSetHotPerRoleCap:
    async def test_cap_is_independent_per_role(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Filling one role's cap must not block a DIFFERENT role in the
        same scope — the pre-existing per-SCOPE cap would have rejected
        this, since it counted every hot Fact in the scope regardless of
        audience.

        Falsifiability: reverting the count to per-scope (the old
        ``meta.scope == record.scope`` scan with no role filter) makes this
        test fail — the third ``set_hot`` call raises instead of
        succeeding, because the scope-wide count is already 2 (from the two
        ``decider`` promotions) at ``hot_cap=2``. See the run verdict for
        the captured failure output.
        """
        cfg = _make_config(hot_cap_global=2)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        ids: list[HypomnemaId] = []
        for i in range(3):
            r = await engine.save(f"per-role independence fact {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        await engine.set_hot(ids[0], hot=True, roles=["decider"])
        await engine.set_hot(ids[1], hot=True, roles=["decider"])  # decider full, 2/2

        # reviewer's own cap is untouched by decider's saturation.
        result = await engine.set_hot(ids[2], hot=True, roles=["reviewer"])
        assert result.lifecycle == "hot"

    async def test_cap_error_names_the_overflowing_role_and_its_occupants(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The `ToolError` must name WHICH role overflowed and list that
        role's current occupants — not scope occupants generically (the
        pre-existing message), which would now be actively misleading with
        four independent per-role caps sharing one scope."""
        cfg = _make_config(hot_cap_global=2)
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        ids: list[HypomnemaId] = []
        for i in range(3):
            r = await engine.save(f"cap error message fact {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        await engine.set_hot(ids[0], hot=True, roles=["decider"])
        await engine.set_hot(ids[1], hot=True, roles=["decider"])

        with pytest.raises(ToolError) as exc_info:
            await engine.set_hot(ids[2], hot=True, roles=["decider"])
        msg = str(exc_info.value)
        assert "decider" in msg
        assert str(ids[0]) in msg
        assert str(ids[1]) in msg

        rec = await adapter.get_by_id(ids[2])
        assert rec.lifecycle == "active"

    async def test_multi_role_promotion_rejected_wholly_on_one_overflowing_role(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """No partial write: a promotion requesting BOTH a saturated role
        (`decider`) and a role with room (`reviewer`) is rejected in full —
        the Fact must not end up hot for `reviewer` either.

        Falsifiability (step 6, break #2): disable the whole-reject guard
        (``if len(occupants) >= cap:`` becomes unreachable) and this test
        fails with "DID NOT RAISE ToolError" — ``ids[2]`` would end up hot
        and tagged both ``decider`` AND ``reviewer`` despite the ``decider``
        overflow. See the run verdict for the captured failure output.
        """
        cfg = _make_config(hot_cap_global=2)
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        ids: list[HypomnemaId] = []
        for i in range(3):
            r = await engine.save(f"whole-reject fact {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        await engine.set_hot(ids[0], hot=True, roles=["decider"])
        await engine.set_hot(ids[1], hot=True, roles=["decider"])  # decider full

        with pytest.raises(ToolError, match="decider"):
            await engine.set_hot(ids[2], hot=True, roles=["decider", "reviewer"])

        rec = await adapter.get_by_id(ids[2])
        assert rec.lifecycle == "active", "the whole promotion must be rejected, not partial"
        assert rec.audience_hint is None, "no role may be tagged when the promotion is rejected"

    async def test_retag_leaving_a_saturated_role_frees_a_slot(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        cfg = _make_config(hot_cap_global=2)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        ids: list[HypomnemaId] = []
        for i in range(3):
            r = await engine.save(f"retag-frees-slot fact {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        await engine.set_hot(ids[0], hot=True, roles=["decider"])
        await engine.set_hot(ids[1], hot=True, roles=["decider"])  # decider full, 2/2

        with pytest.raises(ToolError):
            await engine.set_hot(ids[2], hot=True, roles=["decider"])

        # Retag ids[0] OUT of decider entirely -- frees a slot.
        retagged = await engine.set_hot(ids[0], hot=True, roles=["reviewer"])
        assert retagged.audience_hint == ("reviewer",)

        result = await engine.set_hot(ids[2], hot=True, roles=["decider"])
        assert result.lifecycle == "hot"

    async def test_retag_keeping_an_already_held_role_does_not_trip_its_own_cap(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The idempotent-adjacent case this task's acceptance criteria
        name explicitly: a Fact occupying one of a FULL role's slots is
        retagged with a DIFFERENT roles list that still names that role
        (so this is a genuine retag -- the tuple-equality early return does
        not fire, unlike the exact-same-roles idempotent case already
        covered by ``TestSetHotPromoteDemote``). Excluding the Fact from
        its own occupant count is what lets this succeed even though every
        OTHER occupant already saturates the cap.

        Falsifiability (step 6, break #3): stop excluding ``id`` from the
        per-role occupant scan (drop the ``candidate.id != id`` guard) and
        this test fails — the retag raises, since the Fact's own existing
        occupancy of `decider` would count against its own cap. See the
        run verdict for the captured failure output.
        """
        cfg = _make_config(hot_cap_global=2)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        ids: list[HypomnemaId] = []
        for i in range(2):
            r = await engine.save(f"self-exclusion fact {i}", scope="global")
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)

        await engine.set_hot(ids[0], hot=True, roles=["decider"])
        await engine.set_hot(ids[1], hot=True, roles=["decider"])  # decider full, 2/2

        # Genuine retag: KEEPS decider, ADDS reviewer.
        result = await engine.set_hot(ids[0], hot=True, roles=["decider", "reviewer"])
        assert result.lifecycle == "hot"
        assert set(result.audience_hint or ()) == {"decider", "reviewer"}

    async def test_unhotting_frees_slots_in_every_role_it_was_tagged_for(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        cfg = _make_config(hot_cap_global=1)
        engine, _adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        r1 = await engine.save("universal fact later un-hotted", scope="global")
        assert r1.hypomnema is not None
        id1 = r1.hypomnema.id
        r2 = await engine.save("second decider-only fact", scope="global")
        assert r2.hypomnema is not None
        id2 = r2.hypomnema.id
        r3 = await engine.save("second reviewer-only fact", scope="global")
        assert r3.hypomnema is not None
        id3 = r3.hypomnema.id

        await engine.set_hot(id1, hot=True, roles=["decider", "reviewer"])

        # Both caps (1 each) are now full via id1 alone.
        with pytest.raises(ToolError):
            await engine.set_hot(id2, hot=True, roles=["decider"])
        with pytest.raises(ToolError):
            await engine.set_hot(id3, hot=True, roles=["reviewer"])

        await engine.set_hot(id1, hot=False)  # un-hot -- frees BOTH roles at once

        result2 = await engine.set_hot(id2, hot=True, roles=["decider"])
        assert result2.lifecycle == "hot"
        result3 = await engine.set_hot(id3, hot=True, roles=["reviewer"])
        assert result3.lifecycle == "hot"

    async def test_global_and_project_scope_caps_are_the_seeded_8_and_5(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Exercises the DEFAULT config (no override) to pin the actual
        seeded numbers (owner decision 2026-08-03: `global` 8, project
        scopes 5) rather than only ever testing with an artificially small
        override cap."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_scope(Scope("proj-a"))

        for i in range(5):
            r = await engine.save(f"project cap seed fact {i}", scope="proj-a")
            assert r.hypomnema is not None
            await engine.set_hot(r.hypomnema.id, hot=True, roles=["decider"])

        extra = await engine.save("sixth project fact over cap", scope="proj-a")
        assert extra.hypomnema is not None
        with pytest.raises(ToolError, match="hot_cap=5"):
            await engine.set_hot(extra.hypomnema.id, hot=True, roles=["decider"])

        # global's larger cap (8) is untouched by the project scope's saturation.
        global_extra = await engine.save(
            "global fact unaffected by project saturation", scope="global"
        )
        assert global_extra.hypomnema is not None
        result = await engine.set_hot(global_extra.hypomnema.id, hot=True, roles=["decider"])
        assert result.lifecycle == "hot"


# ---------------------------------------------------------------------------
# Exclusions — Decisions and archived Facts cannot be promoted (embers CAN,
# as of the §8.4 rung-2 Foreground-exit amendment — see tests/test_escalation.py)
# ---------------------------------------------------------------------------


class TestSetHotExclusions:
    async def test_decision_cannot_be_promoted(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles`` is supplied so the raise is provably about the
        Decision/Fact guard, not the (curation block)
        required-roles guard."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "a decision for set_hot exclusion test", scope="global", kind="decision"
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.set_hot(did, hot=True, roles=["decider"])

    async def test_archived_cannot_be_promoted(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles`` is supplied so the raise is provably about the
        archived-exclusion guard, not the (curation block)
        required-roles guard."""
        engine, adapter, _embedder, clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "an active fact to archive for set_hot exclusion test", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await adapter.transition(fid, "archived", at=clock.now())

        with pytest.raises(ToolError):
            await engine.set_hot(fid, hot=True, roles=["decider"])

    async def test_demote_on_ember_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A Fact that is neither active nor hot cannot be touched by
        ``set_hot`` in either direction."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.ember(
            "an ember fact for set_hot demote-illegal-state test",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        with pytest.raises(ToolError):
            await engine.set_hot(eid, hot=False)


# ---------------------------------------------------------------------------
# Housekeeping demotion — automatic, but consistency, never curation
# ---------------------------------------------------------------------------


class TestSetHotHousekeepingDemotion:
    async def test_bury_removes_hot_fact_from_tier(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Real disposition path: ``engine.bury``."""
        cfg = _make_config()
        engine, adapter, _embedder, clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        r = await engine.save(
            "hot fact to be buried for housekeeping demotion test", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])
        assert (await adapter.get_by_id(fid)).lifecycle == "hot"

        await engine.bury(fid, reason="housekeeping demotion via bury")

        with pytest.raises(KeyError):
            await adapter.get_by_id(fid)

        tiers = await select_default(_select_ctx(adapter, cfg, clock), "global", "decider")
        hot_tier = next(t for t in tiers if t.name == "hot")
        assert all(e.id != fid for e in hot_tier.entries)

    async def test_invalidate_removes_hot_fact_from_tier(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Real disposition path: ``engine.set_validity(valid=False)``.

        This is the case that proves "consistency, not curation": the
        record's ``lifecycle`` field stays exactly "hot" (``set_validity``
        never touches lifecycle) — it leaves the *presented* tier purely
        because ``select_default``'s existing valid-only filter excludes it,
        never because any policy re-judged the owner's pin.

        Falsifiability (step 6, break #2): remove the
        ``meta.validity != _VALID: continue`` guard in ``_select_hot``
        (``src/tabularium/engine/context.py``) and watch this test fail —
        see the task verdict for the captured failure output.
        """
        cfg = _make_config()
        engine, adapter, _embedder, clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        r = await engine.save(
            "hot fact to be invalidated for housekeeping demotion test", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        await engine.set_validity(fid, valid=False, reason="found false while hot")

        rec = await adapter.get_by_id(fid)
        assert rec.lifecycle == "hot"  # untouched -- proves this is consistency
        assert rec.validity == "invalid"

        tiers = await select_default(_select_ctx(adapter, cfg, clock), "global", "decider")
        hot_tier = next(t for t in tiers if t.name == "hot")
        assert all(e.id != fid for e in hot_tier.entries)

    async def test_archive_removes_hot_fact_from_tier(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Decay (``{active,hot}->archived``) is not yet built (a later milestone).
        This drives the same ``adapter.transition`` seam decay will
        eventually call — the sanctioned mutation path per ``domain.py``'s
        design notes, not "writing the lifecycle field directly" (which
        would mean bypassing the adapter via ``model_copy`` or similar) — to
        prove the tier is structurally consistent ahead of that feature
        landing: ``Fact.lifecycle`` is a single field, so once a record
        reads "archived" it cannot simultaneously read "hot".
        """
        cfg = _make_config()
        engine, adapter, _embedder, clock = await _build_engine(
            adapter_id, tmp_path, cfg=cfg
        )
        r = await engine.save(
            "hot fact to be archived for housekeeping demotion test", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        await adapter.transition(fid, "archived", at=clock.now())

        rec = await adapter.get_by_id(fid)
        assert rec.lifecycle == "archived"

        tiers = await select_default(_select_ctx(adapter, cfg, clock), "global", "decider")
        hot_tier = next(t for t in tiers if t.name == "hot")
        assert all(e.id != fid for e in hot_tier.entries)


def test_fact_lifecycle_structurally_excludes_superseded() -> None:
    """Facts cannot be "superseded" — supersession is a Decision-only edge
    (SPEC §1.2); ``Fact.lifecycle``'s ``Literal`` is exactly
    ``{active, hot, ember, archived, shelved}`` (ADR 0021 added ``shelved``
    as the fifth value — updated here, not deleted, per that ADR: this
    assertion's whole point is a closed enumeration, so it must track the
    real one exactly, never merely "still excludes superseded"). Since
    ``set_hot`` is Facts-only, a hot Fact can therefore never reach
    "superseded" in the first place — this closes the housekeeping-demotion
    enumeration's fourth case (bury/archive/supersede/invalidate) as
    vacuously true rather than silently unconsidered. Pure domain-level
    check, no adapter involved.
    """
    args = typing.get_args(Fact.model_fields["lifecycle"].annotation)
    assert "superseded" not in args
    assert set(args) == {"active", "hot", "ember", "archived", "shelved"}


# ---------------------------------------------------------------------------
# Per-role grounding — set_hot(roles=...) persists audience_hint (this
# brief). Overwrite, not union (owner-ratified); roles=None leaves the
# existing hint untouched; an unregistered role is a ToolError, matching
# save/ember.
# ---------------------------------------------------------------------------


class TestSetHotRoles:
    async def test_promote_with_roles_overwrites_capture_time_hint(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Overwrite, not union: an atom saved with
        ``audience_hint=["builder"]`` then pinned
        ``set_hot(id, True, roles=["reviewer"])`` routes to reviewer and not
        to builder.

        Falsifiability (step 6): stop ``set_hot`` from forwarding ``roles``
        to ``transition`` (pass ``audience_hint=None`` unconditionally,
        reproducing the exact silent-discard bug the storage-seam
        prerequisite brief fixed) — the persisted hint stays ``("builder",)``
        and this test goes red. See the run verdict for the captured
        failure output.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("builder")
        await adapter.create_role("reviewer")
        r = await engine.save(
            "atom captured with a builder hint, later curated to reviewer",
            scope="global",
            audience_hint=["builder"],
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        await engine.set_hot(fid, hot=True, roles=["reviewer"])

        updated = await adapter.get_by_id(fid)
        assert updated.audience_hint == ("reviewer",)
        assert "builder" not in (updated.audience_hint or ())

    async def test_demote_roles_none_leaves_existing_hint_untouched(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The half of the original "roles=None leaves any existing hint
        untouched" clause that SURVIVES curation block:
        ``hot=False`` (demotion) still leaves ``roles`` optional, since
        removing something from the tier needs no audience decision. (The
        ``hot=True`` half of this clause is REVERSED — see
        ``TestSetHotRequiresRoles.test_promote_without_roles_raises_even_with_existing_capture_time_hint``.)
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("builder")
        r = await engine.save(
            "atom captured with a builder hint, promoted then demoted with no roles arg",
            scope="global",
            audience_hint=["builder"],
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["builder"])

        await engine.set_hot(fid, hot=False)

        updated = await adapter.get_by_id(fid)
        assert updated.lifecycle == "active"
        assert updated.audience_hint == ("builder",)

    async def test_retag_already_hot_fact_persists_new_roles(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The live-store curation path: an atom already hot from BEFORE
        this task (unhinted — curation block makes the engine's
        OWN ``set_hot`` refuse to create such a state going forward, so
        this simulates a legacy record the same way ``test_context.py``
        simulates a legacy ``archived`` Fact — via a direct
        ``adapter.transition`` call, bypassing ``set_hot``) must still be
        retaggable with ``set_hot(id, True, roles=[...])`` — the idempotent
        "same audience, no write" early return must not swallow a
        role-only retag. The atoms
        worth `reviewer`-tagging are typically already hot, so a curation
        pass that can only tag at promotion time is broken for its primary
        real use.

        Falsifiability: restore the old unconditional early return (``if
        record.lifecycle == "hot": return record``, dropping the
        roles-vs-current-hint equality check) — the retag silently no-ops
        and this test goes red.
        """
        engine, adapter, _embedder, clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("reviewer")
        r = await engine.save(
            "atom already hot before the reviewer curation pass runs",
            scope="global",
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        # Simulate a pre-existing legacy hot-but-unhinted Fact directly via
        # the adapter -- set_hot itself can no longer create this state.
        await adapter.transition(fid, "hot", at=clock.now())

        before = await adapter.get_by_id(fid)
        assert before.lifecycle == "hot"
        assert before.audience_hint is None

        await engine.set_hot(fid, hot=True, roles=["reviewer"])  # retag while already hot

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "hot", "retag must not disturb the lifecycle"
        assert after.audience_hint == ("reviewer",)

    async def test_retag_already_active_fact_does_not_rearm_baseline(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The mirror case on the demote side: ``set_hot(id, False,
        roles=[...])`` on an already-``active`` Fact must persist the hint
        without falsely re-arming the rung-2 escalation baseline (that
        re-arm is reserved for a genuine ``hot``->``active`` edge)."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("reviewer")
        r = await engine.save(
            "already-active atom retagged without ever being hot", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        before = await adapter.get_by_id(fid)
        assert isinstance(before, Fact)
        assert before.lifecycle == "active"
        baseline_before = before.escalation_baseline
        recurrence_before = before.owner_recurrence_count

        await engine.set_hot(fid, hot=False, roles=["reviewer"])

        after = await adapter.get_by_id(fid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "active"
        assert after.audience_hint == ("reviewer",)
        assert after.escalation_baseline == baseline_before
        assert after.owner_recurrence_count == recurrence_before


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestSetHotErrors:
    async def test_unknown_id_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles`` is supplied so the raise is provably about the
        unknown-id guard, not the (curation block) required-roles
        guard."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        with pytest.raises(ToolError):
            await engine.set_hot(new_id(), hot=True, roles=["decider"])

    async def test_unregistered_role_rejected(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Matches ``save``/``ember``'s gate — an unregistered role in
        ``roles`` is a ``ToolError``, and nothing is written.

        Falsifiability (step 6): remove the ``await require_roles(roles)``
        call in ``set_hot`` — this test goes red (no exception raised). See
        the run verdict for the captured failure output.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "atom for the unregistered-role rejection test", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="Unregistered role"):
            await engine.set_hot(fid, hot=True, roles=["never-registered"])

        after = await adapter.get_by_id(fid)
        assert after.lifecycle == "active", "rejected role must not write anything"
