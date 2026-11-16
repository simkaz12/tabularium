"""``set_audience`` — rule on a Fact's audience without foregrounding it into
the hot tier (SPEC §1.2 / §8.4 rung 0 / §9.1, curation block).

The gap this closes: the default gives an agent-authored ember all four roles
and ``provisional=True`` — an owner verdict on who it actually reaches is
outstanding. SPEC §8.4 already names two clearing paths (the ember-reignite
``roles`` retag, and ``set_hot(hot=True, roles=[...])`` promotion), but the
reignite retag only fires when a reignite's content shares the ember's
canonical key (P7 admission-order piece, decision D14) — a paraphrase that
keeps the idea but changes the wording bounces instead, and ``ember`` has no
``resolve`` argument to adjudicate that bounce — leaving ``set_hot``
promotion as the only reachable path when the wording differs, which ALSO
promotes the record into the always-on hot tier. The ordinary case has no
path short of promotion: "yes, this is for builders and reviewers; no, it
does not belong in every session's payload yet." ``set_audience`` is that
path.

Every test that exercises the marquee property below constructs the engine
with a bare ``EngineConfig()`` — the real default — since ``set_audience``
never touches the dedup gate at all and so is unaffected by any gate
tuning.

Dual-parameterized over both storage adapters (``adapter_id`` fixture +
``_build_engine`` helper), same construction pattern as ``tests/
test_ember_creation_audience.py`` / ``tests/test_hot.py`` — this feature
persists through the existing ``transition()`` tri-state fields (no new
adapter method), but the dual-parameterized run still proves the LanceDB
mirror agrees with the in-memory reference adapter.

Falsifiability (recorded in the task verdict, not duplicated here):
  - Making ``set_audience`` skip its own write (returning the record
    immediately after the idempotency check, never reaching
    ``adapter.transition``) makes
    ``TestRulesWithoutForegrounding::test_rules_provisional_ember_audience_without_promoting``
    go red on both ``[memory]`` and ``[lancedb]`` — ``provisional`` stays
    ``True`` instead of clearing. Confirmed; reverted.
  - Making ``set_audience`` promote to hot as a side effect (passing
    ``"hot"`` instead of ``record.lifecycle`` to ``adapter.transition``)
    makes 12 of THIS FILE's own tests go red across
    ``TestRulesWithoutForegrounding``, ``TestSetAudienceNeverPromotes``, and
    ``TestSetAudienceIdempotency`` (both adapters). It does **NOT** make
    ``tests/test_hot.py::TestNoAutoPromotion`` go red, and it cannot:
    that test drives only ``save``/``ember``/``recall`` (see
    ``tests/test_hot.py:110-182``) and never calls ``set_audience``, so it
    is structurally blind to a regression in this function regardless of
    how it is broken. ``TestSetAudienceNeverPromotes`` in THIS file is the
    actual regression guard for this verb; the exhaustive, future-proof
    guarantee is the static invariant that ``set_hot`` is the SOLE writer
    of the literal ``lifecycle="hot"`` transition target anywhere in
    ``src/`` (grep-checkable — see ``engine/dispositions.py``'s module
    docstring). Confirmed; reverted.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId
from tabularium.engine import Tabularium, ToolError
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
)
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

AnyAudienceAdapter = MemoryAdapter | LanceDBAdapter

# Large dim, same rationale as the sibling per-role-grounding test files:
# keeps FakeEmbedder's SHA-256 slot-collision probability negligible across
# this file's many near-identical fixture strings.
_DIM = 4096

_ALL_FOUR = ["decider", "builder", "reviewer", "navigator"]


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


async def _build_engine(
    adapter_id: str,
    tmp_path: pathlib.Path,
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
) -> tuple[Tabularium, AnyAudienceAdapter, FakeEmbedder, TestClock]:
    """A fresh, initialized engine + adapter + embedder + clock.

    Defaults to a BARE ``EngineConfig()`` — the real default. ``set_audience``
    never touches the dedup gate, so unlike several sibling test files whose
    ``_make_config`` helper tunes the gate for their own reinforce-path
    tests, this file has no need to override anything here.
    """
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else _make_clock()
    adapter: AnyAudienceAdapter
    if adapter_id == "memory":
        adapter = _make_memory_adapter(
            embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    else:
        adapter = await _make_lancedb_adapter(
            tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clk
        )
    engine = Tabularium(
        adapter,
        embedder,
        clk,
        cfg if cfg is not None else EngineConfig(),
        token_counter=FakeTokenCounter(),
    )
    await engine.initialize()
    return engine, adapter, embedder, clk


@pytest.fixture(params=["memory", "lancedb"])
def adapter_id(request: pytest.FixtureRequest) -> str:
    param: str = request.param
    return param


# ---------------------------------------------------------------------------
# The marquee property: ruling on a provisional ember's audience never
# promotes it, under the REAL default config.
# ---------------------------------------------------------------------------


class TestRulesWithoutForegrounding:
    async def test_rules_provisional_ember_audience_without_promoting(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The exact scenario this exists for: an agent-authored ember
        (all four roles, ``provisional=True``) gets an owner ruling — "yes,
        for builders and reviewers" — WITHOUT becoming hot.

        Falsifiability: see this module's docstring.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "agent-surfaced correction awaiting an owner ruling on audience",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        before = await adapter.get_by_id(eid)
        assert isinstance(before, Fact)
        assert before.lifecycle == "ember"
        assert before.provisional is True
        assert set(before.audience_hint or ()) == set(_ALL_FOUR)

        await engine.set_audience(eid, ["builder", "reviewer"])

        # Assert against the STORE, not a return value, and assert the
        # lifecycle EXPLICITLY rather than merely "not hot" (task brief).
        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "ember", (
            f"set_audience must never change lifecycle; got {after.lifecycle!r}"
        )
        assert after.provisional is False
        assert set(after.audience_hint or ()) == {"builder", "reviewer"}

    async def test_ruling_reaching_no_role_is_legal_and_still_clears_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles=[]`` is a deliberate "this reaches no role" ruling — still
        a ruling, so it still clears ``provisional``."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "agent-surfaced correction the owner rules should reach no role",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        await engine.set_audience(eid, [])

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "ember"
        assert after.provisional is False
        assert after.audience_hint is None

    async def test_works_on_a_plain_active_fact_too(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Not ember-specific: an ordinary ``active`` Fact can also have its
        audience curated without ever touching lifecycle."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        await adapter.create_role("builder")
        r = await engine.save(
            "an ordinary active fact whose audience gets curated", scope="global"
        )
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        await engine.set_audience(fid, ["builder"])

        after = await adapter.get_by_id(fid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "active"
        assert after.audience_hint == ("builder",)
        assert after.provisional is False


# ---------------------------------------------------------------------------
# No-auto-promotion — the same property test_hot.py::TestNoAutoPromotion
# guards, scoped to this new verb specifically.
# ---------------------------------------------------------------------------


class TestSetAudienceNeverPromotes:
    async def test_repeated_calls_across_many_embers_never_produce_a_hot_fact(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Drives the REAL ``engine.set_audience`` repeatedly, under the
        real default config, and checks the STORE for any Fact that ended
        up ``lifecycle="hot"`` — there must be none. ``set_hot`` remains
        the sole writer of ``lifecycle="hot"`` anywhere in the engine.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        ids: list[HypomnemaId] = []
        for i in range(5):
            r = await engine.ember(
                f"hostile audience-ruling candidate ember number {i}",
                scope="global",
                valence="guideline",
            )
            assert r.hypomnema is not None
            ids.append(r.hypomnema.id)
            await engine.set_audience(r.hypomnema.id, ["builder", "reviewer"])
            # Rule again, different roles -- still must never touch lifecycle.
            await engine.set_audience(r.hypomnema.id, ["navigator"])

        hot_ids: set[HypomnemaId] = set()
        async for meta in await adapter.scan(lifecycle_in=frozenset({"hot"})):
            hot_ids.add(meta.id)
        assert hot_ids == set()

        for eid in ids:
            rec = await adapter.get_by_id(eid)
            assert isinstance(rec, Fact)
            assert rec.lifecycle == "ember"
            assert rec.provisional is False


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestSetAudienceIdempotency:
    async def test_repeat_call_same_roles_once_cleared_is_a_noop(
        self, adapter_id: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falsifiability: remove the ``(record.audience_hint or ()) ==
        target_hint and record.provisional is False -> return record`` early
        return in ``set_audience`` -- ``adapter.transition`` would then be
        called a second time and this test's monkeypatched ``_boom`` fires.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.ember(
            "ember ruled on once, then queried again with the same roles",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        await engine.set_audience(eid, ["reviewer"])

        async def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError(
                "adapter.transition must not be called on a true idempotent re-op"
            )

        monkeypatch.setattr(adapter, "transition", _boom)

        result = await engine.set_audience(eid, ["reviewer"])
        assert result.lifecycle == "ember"
        assert result.audience_hint == ("reviewer",)
        assert result.provisional is False

    async def test_same_roles_as_current_hint_but_still_provisional_still_writes(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The agent-authored default hint is exactly the four canonical
        roles. If the owner rules with THOSE SAME four roles, the call must
        still write (to clear ``provisional``) -- a tuple-equality check on
        ``roles`` alone would wrongly treat this as a no-op forever."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.ember(
            "agent-surfaced correction the owner rules IS for everyone",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        before = await adapter.get_by_id(eid)
        assert isinstance(before, Fact)
        assert set(before.audience_hint or ()) == set(_ALL_FOUR)
        assert before.provisional is True

        await engine.set_audience(eid, list(_ALL_FOUR))

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert set(after.audience_hint or ()) == set(_ALL_FOUR)
        assert after.provisional is False, (
            "ruling with the SAME roles the record already carried must still "
            "clear provisional -- it is the first genuine ruling"
        )

    async def test_cleared_ruling_survives_reopen(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Asserted against a REOPENED adapter handle, not a return value."""
        clk = _make_clock()
        engine, adapter, embedder, clk = await _build_engine(
            adapter_id, tmp_path, clock=clk
        )
        r = await engine.ember(
            "ember ruled on then the store reopened",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        await engine.set_audience(eid, ["builder"])

        from tests.conformance.test_conformance import _reopen_at_model

        reopened = await _reopen_at_model(
            adapter,
            embedding_model=embedder.model,
            tmp_path=tmp_path,
            clock=clk,
            dim=embedder.dim,
        )
        rec = await reopened.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "ember"
        assert rec.provisional is False
        assert rec.audience_hint == ("builder",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestSetAudienceErrors:
    async def test_unknown_id_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        from tabularium.domain import new_id

        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        with pytest.raises(ToolError, match="Unknown id"):
            await engine.set_audience(new_id(), ["decider"])

    async def test_roles_none_raises_toolerror_not_a_bare_typeerror(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``roles`` has no default (a type-checked caller cannot omit it),
        but nothing at runtime stops an untyped direct caller from passing
        ``None`` explicitly. Without an explicit guard, ``None`` slips past ``require_roles``
        (which treats falsy input as "nothing to validate") and hits a bare
        ``TypeError`` at ``tuple(roles)`` instead of this family's
        ``ToolError`` contract -- checked BEFORE the record is fetched, so a
        hot Fact's own guard cannot mask the missing parameter either.

        Falsifiability: remove the ``if roles is None: raise ToolError(...)``
        guard in ``set_audience`` -- this test goes red with a raw
        ``TypeError`` instead of the expected ``ToolError``.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact for the roles=None guard test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="requires roles"):
            await engine.set_audience(fid, None)  # type: ignore[arg-type]

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.audience_hint is None, "rejected call must not write anything"

    async def test_roles_none_raises_even_on_a_hot_fact(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The ``roles is None`` guard runs before the record is even
        fetched, so it must fire with its OWN message even on a hot Fact --
        proving the missing-parameter case is never masked by the
        hot-Fact/``set_hot`` guard."""
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact promoted hot for the roles=None guard test", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        with pytest.raises(ToolError, match="requires roles"):
            await engine.set_audience(fid, None)  # type: ignore[arg-type]

    async def test_decision_id_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save(
            "a decision for the set_audience Facts-only guard",
            scope="global",
            kind="decision",
        )
        assert r.hypomnema is not None
        did = r.hypomnema.id

        with pytest.raises(ToolError, match="Decision"):
            await engine.set_audience(did, ["decider"])

    async def test_hot_fact_raises_naming_set_hot(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A hot Fact's audience must be retagged via ``set_hot`` (which
        re-checks ``hot_cap``) -- ``set_audience`` refuses instead of
        opening a second, uncapped route to the same field."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("an active fact to be promoted then re-audienced", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id
        await engine.set_hot(fid, hot=True, roles=["decider"])

        with pytest.raises(ToolError, match="set_hot"):
            await engine.set_audience(fid, ["reviewer"])

        # Nothing changed -- rejected call must not write anything.
        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.lifecycle == "hot"
        assert rec.audience_hint == ("decider",)

    async def test_unregistered_role_rejected(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        r = await engine.save("a fact for the unregistered-role guard", scope="global")
        assert r.hypomnema is not None
        fid = r.hypomnema.id

        with pytest.raises(ToolError, match="Unregistered role"):
            await engine.set_audience(fid, ["never-registered"])

        rec = await adapter.get_by_id(fid)
        assert isinstance(rec, Fact)
        assert rec.audience_hint is None, "rejected call must not write anything"
