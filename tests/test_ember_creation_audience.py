"""Ember creation-time audience rule (curation block, SPEC §8.4
amendment): a brand-new ember cannot be born reaching nobody, by either
authorship path.

A FALSY ``audience_hint`` (omitted, OR an explicit ``[]``) is treated
identically by both authorship paths — the headline acceptance property is
unqualified ("no ember can be born reaching nobody"), so there is no
"deliberately reach nobody" option at creation time, unlike the SEPARATE
``roles`` retag parameter:

- ``asserted_by="owner"`` with a falsy ``audience_hint`` — RAISES, naming
  the registered roles. An ember is authored, not inferred; the author is
  right there and can say who it is for.
- ``asserted_by="agent"`` (the default) with a falsy ``audience_hint`` —
  written with the FOUR CANONICAL roles (``engine._types.ALL_ROLES``, not
  a live registry read) and marked ``provisional`` (``Fact.provisional``):
  an owner verdict on who this actually reaches is outstanding.
- A NON-EMPTY explicit ``audience_hint`` on either path is honored verbatim
  and never marked provisional.
- The provisional mark is a persisted field, not an inference from the
  hint's shape (an all-four hint alone cannot distinguish "the owner said
  everyone" from "an agent guessed and nobody has ruled").
- The defined path by which the owner rules and the mark clears: an
  owner-asserted ``roles=[...]`` retag on the SAME ember's reignite.
  Gated on ``asserted_by`` — an agent-asserted retag narrows the hint but
  does not clear the mark ("an agent surfaces a correction, it does not
  get to decide who it is for").

Dual-parameterized over both storage adapters (``adapter_id`` fixture +
``_build_engine`` helper), same construction pattern as ``tests/
test_escalation.py`` — this feature touches the LanceDB schema/codec
directly (one new column), so the LanceDB leg is load-bearing, not
incidental.

Falsifiability: every test below has been proven sensitive to a real
defect by deliberately reverting the corresponding branch in
``src/tabularium/engine/embers.py`` / ``src/tabularium/engine/facts.py``,
watching the named test go red for the stated reason, then restoring. The
mutation-per-test map is recorded in the dispatching agent's verdict.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, HypomnemaId
from tabularium.engine import Tabularium, ToolError
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.memory import MemoryAdapter
from tests.conformance.test_conformance import (
    _make_lancedb_adapter,
    _make_memory_adapter,
    _reopen_at_model,
)
from tests.support.doubles import FakeEmbedder, FakeTokenCounter, TestClock

AnyEmberAdapter = MemoryAdapter | LanceDBAdapter

# Large dim, same rationale as test_escalation.py / test_hot.py: keeps
# FakeEmbedder's SHA-256 slot-collision probability negligible across this
# file's many near-identical fixture strings.
_DIM = 4096

_ALL_FOUR = ["decider", "builder", "reviewer", "navigator"]


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {
        "candidate_k": 1000,
        "gray_floor": 0.80,
    }
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _build_engine(
    adapter_id: str,
    tmp_path: pathlib.Path,
    *,
    cfg: EngineConfig | None = None,
    clock: TestClock | None = None,
) -> tuple[Tabularium, AnyEmberAdapter, FakeEmbedder, TestClock]:
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else _make_clock()
    adapter: AnyEmberAdapter
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
        cfg if cfg is not None else _make_config(),
        token_counter=FakeTokenCounter(),
    )
    await engine.initialize()
    return engine, adapter, embedder, clk


@pytest.fixture(params=["memory", "lancedb"])
def adapter_id(request: pytest.FixtureRequest) -> str:
    param: str = request.param
    return param


# ---------------------------------------------------------------------------
# Owner path — required, raises when absent
# ---------------------------------------------------------------------------


class TestOwnerCreationRequiresAudience:
    async def test_owner_without_audience_hint_raises_naming_registered_roles(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: remove the ``asserted_by == "owner"`` raise
        branch in ``save_ember`` — this test goes red (no exception; the
        ember would silently be written unhinted, reaching no role)."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        with pytest.raises(ToolError, match="Registered roles"):
            await engine.ember(
                "an owner correction with no stated audience",
                scope="global",
                valence="guideline",
                asserted_by="owner",
            )

        # The raise must fire before any write (mirrors set_hot's
        # before-the-record-is-fetched precedent).
        count = 0
        async for _row in await adapter.scan():
            count += 1
        assert count == 0, "nothing may be written when the raise fires"

    async def test_owner_with_explicit_audience_hint_succeeds_and_is_not_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "an owner correction naming its own audience",
            scope="global",
            valence="guideline",
            asserted_by="owner",
            audience_hint=["builder"],
        )
        assert r.outcome == "saved"
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert rec.audience_hint == ("builder",)
        assert rec.provisional is False

    async def test_owner_with_explicit_empty_audience_hint_also_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change the guard from ``if not audience_hint:``
        back to ``if audience_hint is None:`` — this test goes red (no
        exception raised for an explicit ``[]``).

        Unlike ``set_hot(roles=[])`` or the ember RETAG path's
        ``roles=[]`` (both legitimate "reach nobody" choices on an
        EXISTING record), there is no such escape hatch at ember CREATION
        time: the headline
        acceptance property is unqualified — "no ember can be born
        reaching nobody, by either authorship path" — so an explicit
        ``audience_hint=[]`` is treated exactly like an omitted one.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        with pytest.raises(ToolError, match="Registered roles"):
            await engine.ember(
                "an owner correction with an explicitly empty audience",
                scope="global",
                valence="guideline",
                asserted_by="owner",
                audience_hint=[],
            )

        count = 0
        async for _row in await adapter.scan():
            count += 1
        assert count == 0, "nothing may be written when the raise fires"


# ---------------------------------------------------------------------------
# Agent path — defaults to every role, marks provisional
# ---------------------------------------------------------------------------


class TestAgentCreationDefaultsAndMarksProvisional:
    async def test_agent_without_audience_hint_defaults_to_every_role_and_is_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: remove the ``audience_hint = await adapter.
        list_roles()`` / ``provisional = True`` branch in ``save_ember`` —
        this test goes red because the Fact would be written with
        ``audience_hint=None`` (reaching no role) instead of all four, and
        ``provisional`` would stay ``False``.

        Neither ``asserted_by`` nor ``audience_hint`` is passed here —
        the honest "we cannot verify who is calling" default.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "an agent-surfaced correction with no stated audience",
            scope="global",
            valence="guideline",
        )
        assert r.outcome == "saved"
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert set(rec.audience_hint or ()) == set(_ALL_FOUR)
        assert rec.provisional is True

    async def test_agent_default_excludes_a_stale_retired_default_role(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: revert ``save_ember``'s agent-default branch to
        read ``await adapter.list_roles()`` instead of the fixed
        ``engine._types.ALL_ROLES`` tuple — this test goes red because the
        stored hint would include ``"default"`` too.

        An upgraded store can still carry a ``"default"`` role row from
        before curation block retired it from ``initialize()``'s
        seed list (ADR 0016's own honesty finding: retirement means "no
        longer seeded", never "guaranteed absent"). A brand-new
        agent-authored provisional ember must be born reaching exactly the
        four canonical roles regardless — never a live registry snapshot
        carrying that historical cruft.
        """
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)
        # Simulate the pre-existing registry state this store would have
        # carried forward: initialize() already ran (in _build_engine); a
        # "default" role additionally lingers from before it was retired.
        await adapter.create_role("default")

        r = await engine.ember(
            "an agent-surfaced correction on a store carrying a stale default role",
            scope="global",
            valence="guideline",
        )
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert set(rec.audience_hint or ()) == set(_ALL_FOUR), (
            "must be exactly the four canonical roles, never the live "
            "registry (which can carry stale entries like 'default')"
        )

    async def test_agent_with_explicit_audience_hint_is_honored_and_not_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "an agent correction with a stated audience",
            scope="global",
            valence="guideline",
            audience_hint=["reviewer"],
        )
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert rec.audience_hint == ("reviewer",)
        assert rec.provisional is False

    async def test_agent_with_explicit_empty_audience_hint_also_defaults_and_is_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """An explicit ``[]`` on the agent path is treated exactly like an
        omitted ``audience_hint`` — there is no "reach nobody" option at
        ember creation, on either authorship path."""
        engine, adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        r = await engine.ember(
            "an agent correction with an explicitly empty audience",
            scope="global",
            valence="guideline",
            audience_hint=[],
        )
        assert r.hypomnema is not None

        rec = await adapter.get_by_id(r.hypomnema.id)
        assert isinstance(rec, Fact)
        assert set(rec.audience_hint or ()) == set(_ALL_FOUR)
        assert rec.provisional is True


# ---------------------------------------------------------------------------
# Provisional clearing — the defined path by which the owner rules
# ---------------------------------------------------------------------------


class TestProvisionalClearing:
    async def test_owner_supplied_roles_on_reignite_clears_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """Falsifiability: change the ember-recurrence retag branch's
        ``provisional=False if asserted_by == "owner" else None`` to always
        pass ``provisional=None`` — this test goes red because the mark
        stays ``True`` forever, with no path to clear it.
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "agent-surfaced correction later ruled on by the owner"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        before = await adapter.get_by_id(eid)
        assert isinstance(before, Fact)
        assert before.provisional is True

        # Owner reignites the SAME content and narrows the audience —
        # audience_hint is required now (owner path) but irrelevant to the
        # reignite outcome itself; roles is what actually retags.
        rep = await engine.ember(
            content,
            scope="global",
            valence="guideline",
            asserted_by="owner",
            audience_hint=["builder"],
            roles=["reviewer"],
        )
        assert rep.outcome == "reinforced"

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.audience_hint == ("reviewer",)
        assert after.provisional is False

    async def test_owner_retag_without_audience_hint_still_raises(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The full call shape that clears provisional is ``ember(
        asserted_by="owner", audience_hint=[...], roles=[...])`` — BOTH
        arguments together, never ``roles`` alone.
        ``audience_hint`` is required unconditionally on every
        ``asserted_by="owner"`` call, checked before the dedup gate even
        determines this is a reignite — so a caller cannot skip it just
        because they intend to retag rather than create.
        """
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "agent-surfaced correction an owner tries to retag without audience_hint"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        with pytest.raises(ToolError, match="Registered roles"):
            await engine.ember(
                content,
                scope="global",
                valence="guideline",
                asserted_by="owner",
                roles=["reviewer"],
            )

        # Nothing about the pre-existing provisional ember changed.
        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.provisional is True
        assert set(after.audience_hint or ()) == set(_ALL_FOUR)

    async def test_agent_supplied_roles_on_reignite_does_not_clear_provisional(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """An agent narrowing the hint does not get to rule — only an
        owner-asserted retag clears the mark (SPEC §8.4 amendment: "an
        agent surfaces a correction, it does not get to decide who it is
        for")."""
        engine, adapter, _embedder, _clock = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config()
        )
        content = "agent-surfaced correction retagged by another agent call"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        rep = await engine.ember(
            content, scope="global", valence="guideline", roles=["reviewer"]
        )
        assert rep.outcome == "reinforced"

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.audience_hint == ("reviewer",)
        assert after.provisional is True, (
            "an agent-asserted retag must not clear the provisional mark "
            "— only the owner rules"
        )

    async def test_cleared_provisional_survives_reopen(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """A cleared record stays cleared across a reopen — asserted
        against the store via a real reopened adapter handle, not a
        return value."""
        clk = _make_clock()
        engine, adapter, embedder, clk = await _build_engine(
            adapter_id, tmp_path, cfg=_make_config(), clock=clk
        )
        content = "agent-surfaced correction, owner-ruled, then reopened"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id

        await engine.ember(
            content,
            scope="global",
            valence="guideline",
            asserted_by="owner",
            audience_hint=["builder"],
            roles=["reviewer"],
        )

        reopened = await _reopen_at_model(
            adapter,
            embedding_model=embedder.model,
            tmp_path=tmp_path,
            clock=clk,
            dim=embedder.dim,
        )
        rec = await reopened.get_by_id(eid)
        assert isinstance(rec, Fact)
        assert rec.provisional is False
        assert rec.audience_hint == ("reviewer",)

    async def test_set_hot_promotion_clears_provisional_under_the_default_config(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """``set_hot(hot=True, roles=[...])`` is a SECOND, definitely-
        reachable path by which the owner rules and ``provisional``
        clears: the
        ember-reignite ``roles`` retag path only fires when a reignite's
        content shares the ember's canonical key (P7 admission-order
        piece, decision D14) — a paraphrase that keeps the idea but
        changes the wording never reaches it, and ``ember`` has no
        ``resolve`` argument to adjudicate the bounce it gets instead.
        ``set_hot`` never touches the dedup gate at all, so it clears
        the mark regardless. This test uses the TRUE default config
        (not this file's ``_make_config()``, tuned for the reignite-path
        tests above) to prove the point.

        Falsifiability: drop the ``provisional=provisional_update`` kwarg
        from ``set_hot``'s ``adapter.transition(...)`` call — this test
        goes red because the promoted Fact stays ``provisional=True``.
        """
        embedder = FakeEmbedder(dim=_DIM)
        clk = _make_clock()
        adapter: AnyEmberAdapter
        if adapter_id == "memory":
            adapter = _make_memory_adapter(
                embedding_model=embedder.model, dim=embedder.dim, clock=clk
            )
        else:
            adapter = await _make_lancedb_adapter(
                tmp_path, embedding_model=embedder.model, dim=embedder.dim, clock=clk
            )
        engine = Tabularium(
            adapter, embedder, clk, EngineConfig(), token_counter=FakeTokenCounter()
        )
        await engine.initialize()

        content = "agent-surfaced correction later foregrounded by the owner"
        r = await engine.ember(content, scope="global", valence="guideline")
        assert r.hypomnema is not None
        eid = r.hypomnema.id
        before = await adapter.get_by_id(eid)
        assert isinstance(before, Fact)
        assert before.provisional is True

        await engine.set_hot(eid, hot=True, roles=["reviewer"])

        after = await adapter.get_by_id(eid)
        assert isinstance(after, Fact)
        assert after.lifecycle == "hot"
        assert after.audience_hint == ("reviewer",)
        assert after.provisional is False


# ---------------------------------------------------------------------------
# End-to-end proof — the unit test is not the proof (the earlier change's own words):
# write an ember by each path, then call context() for all four roles and
# confirm each appears in at least one payload.
# ---------------------------------------------------------------------------


class TestContextEndToEndAcrossAllRoles:
    async def test_owner_and_agent_authored_embers_each_reach_at_least_one_role_payload(
        self, adapter_id: str, tmp_path: pathlib.Path
    ) -> None:
        """The exact defect class this task closes: a unit-level green
        (``audience_hint``/``provisional`` set correctly on the stored
        Fact) coexisting with an empty push payload for every role. Drives
        the real ``engine.context`` call for each of the four registered
        roles — not a mock, not a direct ``routes_to`` call.
        """
        engine, _adapter, _embedder, _clock = await _build_engine(adapter_id, tmp_path)

        owner_content = "owner-authored correction naming builder only, for context proof"
        owner_r = await engine.ember(
            owner_content,
            scope="global",
            valence="guideline",
            asserted_by="owner",
            audience_hint=["builder"],
        )
        assert owner_r.hypomnema is not None
        owner_id = owner_r.hypomnema.id

        agent_content = "agent-surfaced correction with no stated audience, for context proof"
        agent_r = await engine.ember(agent_content, scope="global", valence="guideline")
        assert agent_r.hypomnema is not None
        agent_id = agent_r.hypomnema.id

        owner_reaches: set[str] = set()
        agent_reaches: set[str] = set()
        for role in _ALL_FOUR:
            payload = await engine.context(scope="global", budget=5000, role=role)
            ember_tier = next(t for t in payload.tiers if t.name == "ember")
            ids: set[HypomnemaId] = {e.id for e in ember_tier.entries}
            if owner_id in ids:
                owner_reaches.add(role)
            if agent_id in ids:
                agent_reaches.add(role)

        assert owner_reaches, "the owner-authored ember must appear in at least one payload"
        assert owner_reaches == {"builder"}, (
            f"the owner-authored ember must reach exactly its named role, got {owner_reaches}"
        )
        assert agent_reaches, (
            "the agent-authored (provisional) ember must appear in at least one payload — "
            "this is the exact silent-empty-payload defect class this task closes"
        )
        assert agent_reaches == set(_ALL_FOUR), (
            f"the agent-authored (provisional) ember must reach every role until the "
            f"owner rules, got {agent_reaches}"
        )
