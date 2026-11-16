"""Fact-arm supersession — ``save(kind="fact", supersedes=[...])`` and the
third Fact bounce exit (fact-supersession brief; one-transaction-supersession piece).

Scope 1 (the field) and Scope 2 (the storage edge under a lock) are covered
directly against the storage seam in ``tests/conformance/test_conformance.py``
(``TestFactSupersededByRoundTrip`` / ``TestFactSupersedeConcurrency`` /
``TestReplaceAtomicity``). Scope 3 (the explicit engine call, ``supersede``)
is covered in ``tests/test_engine_dispositions.py`` (``TestSupersedeFact``).
This file is the engine-level behaviour suite for Scope 4
(``save(supersedes=[...])``) and Scope 5 (the third Fact bounce exit) — the
two axes that most need adversarial coverage — plus Scope 6's read side
(``RecalledHit.superseded_by``
and the ``find_superseded_by`` backward query; the CLI subcommand is
covered in ``tests/test_cli.py``).

P3 replaced the two-write shape (successor ``upsert`` under the lock,
predecessor edges drawn afterward, each its own commit) with one
``adapter.replace`` call landing the successor and every predecessor's flip
in a single transaction. ``save(supersedes=[...])`` now requires a `reason``
and is all-or-nothing (D2): one stale predecessor refuses the WHOLE call,
not just its own edge — the old partial-failure shape (write lands, one
edge fails, the failure named in ``SaveResult.reason``) no longer exists.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime
from typing import Any

import pytest

from tabularium.config import EngineConfig
from tabularium.domain import Fact, Scope, new_id
from tabularium.engine import Tabularium, ToolError
from tabularium.results import BounceResolution, SaveResult
from tabularium.storage.memory import MemoryAdapter
from tests.support.doubles import FakeEmbedder, TestClock, find_superseded_by
from tests.support.faults import FaultInjectingAdapter, as_adapter

_DIM = 16


def _make_clock(year: int = 2025) -> TestClock:
    return TestClock(datetime(year, 1, 1, tzinfo=UTC))


def _make_config(**kwargs: Any) -> EngineConfig:
    defaults: dict[str, Any] = {"candidate_k": 1000, "gray_floor": 0.80}
    defaults.update(kwargs)
    return EngineConfig(**defaults)


async def _engine(
    *, config: EngineConfig | None = None, clock: TestClock | None = None
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    """Plain engine — FakeEmbedder's binary-cosine hash, distinct content
    never collides."""
    adapter = MemoryAdapter(embedding_model="fake/nomic", dim=_DIM)
    embedder = FakeEmbedder(dim=_DIM)
    clk = clock if clock is not None else _make_clock()
    cfg = config if config is not None else _make_config()
    engine = Tabularium(adapter, embedder, clk, cfg)
    await engine.initialize()
    return engine, adapter, clk


class _IdenticalEmbedder:
    """Every text embeds to the SAME unit vector — every save collides with
    every existing record (cosine 1.0). Same device ``test_engine_m1.py``
    uses for the identical reason."""

    dim: int = _DIM
    model: str = "identical"

    async def embed(self, text: str) -> list[float]:
        del text
        v = [0.0] * self.dim
        v[0] = 1.0
        return v


class _TableEmbedder:
    """Deterministic exact-string → vector lookup (``KeyError`` on miss) —
    same device ``test_engine_m1.py`` uses to place records at known,
    distinct cosine similarities."""

    dim: int = _DIM
    model: str = "test/table"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    async def embed(self, text: str) -> list[float]:
        return list(self._table[text])


def _angle_vec(angle_deg: float) -> list[float]:
    a = math.radians(angle_deg)
    return [math.cos(a), math.sin(a)] + [0.0] * (_DIM - 2)


async def _collide_engine(
    *, config: EngineConfig | None = None, clock: TestClock | None = None
) -> tuple[Tabularium, MemoryAdapter, TestClock]:
    adapter = MemoryAdapter(embedding_model="identical", dim=_DIM)
    embedder = _IdenticalEmbedder()
    clk = clock if clock is not None else _make_clock()
    cfg = config if config is not None else _make_config(gray_floor=0.0)
    engine = Tabularium(adapter, embedder, clk, cfg)
    await engine.initialize()
    return engine, adapter, clk


def _no_supersession_failed_logged(caplog: pytest.LogCaptureFixture) -> bool:
    return not any(
        r.getMessage().startswith("supersession_failed") for r in caplog.records
    )


# ---------------------------------------------------------------------------
# Scope 4 — save(kind="fact", supersedes=[...])
# ---------------------------------------------------------------------------


class TestSaveSupersedesTypeDirected:
    async def test_fact_arm_rejects_bare_id(self) -> None:
        """A bare id on the Fact arm is the Decision arm's shape — rejected.

        Falsify: drop the ``not isinstance(supersedes, list)`` guard in
        ``core.py``'s ``save``.
        """
        engine, _adapter, _clock = await _engine()
        with pytest.raises(ToolError):
            await engine.save(
                "a fact", scope="global", kind="fact", supersedes=new_id()
            )

    async def test_decision_arm_rejects_list(self) -> None:
        """A list on the Decision arm is the Fact arm's shape — rejected.

        Falsify: drop the ``isinstance(supersedes, list)`` guard in
        ``core.py``'s ``save``.
        """
        engine, _adapter, _clock = await _engine()
        d = await engine.save("a decision to supersede", scope="global", kind="decision")
        assert d.hypomnema is not None
        with pytest.raises(ToolError):
            await engine.save(
                "a decision", scope="global", kind="decision",
                supersedes=[d.hypomnema.id],
            )


class TestSaveSupersedesEdgeIdAfterUndo:
    async def test_post_undo_resave_supersedes_mints_a_fresh_edge_id(self) -> None:
        """P4 two-arm unsupersede piece: A superseded by
        B through ``save(supersedes=[...])``, undone via ``engine.
        unsupersede``, then A superseded by B AGAIN through
        ``save(supersedes=[...])`` — the second edge gets a FRESH id, not a
        reused one, so a later undo can never confuse the two generations.

        Falsify: have ``resolve_replace_predecessor`` copy the caller's own
        ``flip.edge_id`` from a stale cached value instead of the one
        ``save``/``supersede`` mints fresh per call — this test's
        ``first_edge_id != second_edge_id`` assertion goes red.
        """
        engine, adapter, _clock = await _engine()
        pred = await engine.save("predecessor, superseded then undone then re-superseded",
                                  scope="global")
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id

        first = await engine.save(
            "first successor", scope="global", supersedes=[pred_id], reason="changed",
        )
        assert first.hypomnema is not None
        after_first = await adapter.get_by_id(pred_id)
        assert isinstance(after_first, Fact)
        first_edge_id = after_first.supersession_edge_id
        assert first_edge_id is not None

        await engine.unsupersede(pred_id, reason="wrong successor named")

        second = await engine.save(
            "second successor", scope="global", supersedes=[pred_id], reason="changed",
        )
        assert second.hypomnema is not None
        after_second = await adapter.get_by_id(pred_id)
        assert isinstance(after_second, Fact)
        second_edge_id = after_second.supersession_edge_id
        assert second_edge_id is not None
        assert second_edge_id != first_edge_id
        assert after_second.superseded_by == second.hypomnema.id


class TestSaveSupersedesExclusion:
    async def test_naming_predecessor_does_not_self_bounce(self) -> None:
        """A save naming its predecessor does not bounce against it, even
        when every other candidate collides at cosine 1.0.

        Falsify: remove the ``candidate_exclude`` widening (fold only
        ``distinct_exclude`` into candidate selection) — this test goes red
        with ``outcome == "bounced"``.
        """
        engine, _adapter, _clock = await _collide_engine()
        pred = await engine.save("predecessor content", scope="global")
        assert pred.hypomnema is not None

        result = await engine.save(
            "successor content", scope="global",
            supersedes=[pred.hypomnema.id], reason="changed",
        )
        assert result.outcome in ("saved", "reconciled"), result

    async def test_naming_unrelated_record_still_bounces_against_y(self) -> None:
        """The exclusion is narrow, not a gate bypass: naming predecessor P
        does not shield an UNNAMED, genuinely-similar candidate Y from the
        gate. Distinct angled vectors (never colliding with each other) so
        ``pred`` and ``y`` can both be written cleanly, then the probe's
        cosine ranking is controlled precisely: ``y`` is placed CLOSER to
        the new content than ``pred`` is, so it is the natural best match
        with or without the exclusion — the exclusion must not also hide it.

        Falsify: fold ``candidate_exclude`` into ``distinct_exclude`` itself
        (rather than a separate combined set) in a way that widens beyond
        the named ids — this test would then also swallow Y.
        """
        pred_text = "predecessor content, orthogonal to both Y and the new content"
        y_text = "Y content, 10 degrees off the new content — genuinely close"
        new_text = "new content, replaces the predecessor"
        table = {
            # pred sits 90 degrees from y (cosine 0, below floor — pred's own
            # save never bounces against y) and ~100 degrees from new
            # (irrelevant either way, since pred is excluded by name).
            pred_text: _angle_vec(100.0),
            y_text: _angle_vec(10.0),
            new_text: _angle_vec(0.0),
        }
        adapter = MemoryAdapter(embedding_model="test/table", dim=_DIM)
        embedder = _TableEmbedder(table)
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.10)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()

        pred = await engine.save(pred_text, scope="global")
        y = await engine.save(y_text, scope="global")
        assert pred.hypomnema is not None
        assert y.hypomnema is not None

        result = await engine.save(
            new_text, scope="global",
            supersedes=[pred.hypomnema.id], reason="changed",
        )
        assert result.outcome == "bounced", result
        assert result.bounce is not None
        assert result.bounce.candidate_id == y.hypomnema.id

    async def test_three_predecessors_all_point_at_the_same_successor(self) -> None:
        """A save naming three predecessors leaves all three pointing at the
        SAME successor, all three flips landed in the SAME transaction as
        the successor row.

        Falsify: only draw an edge for the FIRST id in ``supersedes``
        (an off-by-loop bug) — two of the three would stay ``valid`` with no
        pointer.
        """
        engine, adapter, _clock = await _engine()
        p1 = await engine.save("predecessor one: policy A v1", scope="global")
        p2 = await engine.save("predecessor two: policy B v1", scope="global")
        p3 = await engine.save("predecessor three: policy C v1", scope="global")
        assert p1.hypomnema is not None
        assert p2.hypomnema is not None
        assert p3.hypomnema is not None

        result = await engine.save(
            "successor: unified policy v2, replaces A, B, and C",
            scope="global",
            supersedes=[p1.hypomnema.id, p2.hypomnema.id, p3.hypomnema.id],
            reason="changed",
        )
        assert result.outcome == "saved", result
        assert result.hypomnema is not None
        successor_id = result.hypomnema.id

        for pred in (p1, p2, p3):
            assert pred.hypomnema is not None
            rec = await adapter.get_by_id(pred.hypomnema.id)
            assert isinstance(rec, Fact)
            assert rec.validity == "invalid"
            assert rec.superseded_by == successor_id


class TestFactBounceSupersedeExitExecutes:
    """Scope 5's own Verify: a test that EXECUTES the exit the Fact bounce
    names and asserts the engine accepts it — the ADR 0022 rule as a test,
    not a comment. The exit-set/engine-legality coupling is one of the two
    axes that most need adversarial coverage.
    """

    async def test_taking_the_named_supersede_exit_succeeds(self) -> None:
        """Executes the ACTUAL exit's ``how`` shape against the ACTUAL
        bounce the gate produced — not a hand-rolled call.

        Falsifiability: make ``is_fact_supersedable`` return ``True``
        unconditionally (offering the exit for an invalid candidate) —
        ``tests/test_exits.py``'s
        ``test_fact_save_fact_candidate_offered_split_pinned_by_validity``
        goes red immediately; independently, ``engine.dispositions.
        supersede``'s OWN legality check (not merely the offer site) still
        refuses an edge against an invalid predecessor even if one were
        ever offered — see
        ``tests/test_engine_dispositions.py::TestSupersedeFact::
        test_already_invalid_predecessor_raises``.
        """
        engine, adapter, _clock = await _collide_engine()
        pred = await engine.save("predecessor content", scope="global")
        assert pred.hypomnema is not None

        bounced = await engine.save("distinct successor content", scope="global")
        assert bounced.outcome == "bounced", bounced
        assert bounced.bounce is not None
        exits_by_action = {e.action: e for e in bounced.bounce.exits}
        assert "supersede" in exits_by_action, (
            "the gate must name the third exit for a valid Fact candidate"
        )
        supersede_exit = exits_by_action["supersede"]
        candidate_id = bounced.bounce.candidate_id
        bounce_id = bounced.bounce.bounce_id
        assert supersede_exit.how == (
            f'save(kind="fact", content=…, supersedes=["{candidate_id!s}"], '
            f'reason="changed", bounce_id="{bounce_id}")'
        )

        # Take the exit exactly as named — the how template's reason class
        # included verbatim, no ad-hoc addition.
        result = await engine.save(
            "distinct successor content", scope="global",
            supersedes=[candidate_id], reason="changed", bounce_id=bounce_id,
        )
        assert result.outcome in ("saved", "reconciled"), result

        predecessor_after = await adapter.get_by_id(candidate_id)
        assert isinstance(predecessor_after, Fact)
        assert predecessor_after.validity == "invalid"
        assert result.hypomnema is not None
        assert predecessor_after.superseded_by == result.hypomnema.id

    async def test_cross_scope_candidate_exit_names_moved_reason_and_succeeds(
        self,
    ) -> None:
        """A
        bounce candidate living in ``global`` (reachable from any scope's
        own read span) gets a ``supersede`` exit whose ``how`` literal
        names ``reason="moved"`` — the only class D10 accepts for a
        cross-scope predecessor — not the same-scope ``"changed"`` the
        template used to hardcode regardless of the candidate's own scope.

        Falsify: hardcode ``reason="changed"`` in ``build_exits`` again —
        this test's own literal execution goes red with "reason='moved' is
        refused" instead of landing.
        """
        engine, adapter, _clock = await _collide_engine()
        await engine._adapter.create_scope(Scope("beta"))
        pred = await engine.save("predecessor content in global scope", scope="global")
        assert pred.hypomnema is not None

        bounced = await engine.save(
            "distinct successor content in beta scope", scope="beta"
        )
        assert bounced.outcome == "bounced", bounced
        assert bounced.bounce is not None
        exits_by_action = {e.action: e for e in bounced.bounce.exits}
        supersede_exit = exits_by_action["supersede"]
        candidate_id = bounced.bounce.candidate_id
        bounce_id = bounced.bounce.bounce_id
        assert supersede_exit.how == (
            f'save(kind="fact", content=…, supersedes=["{candidate_id!s}"], '
            f'reason="moved", bounce_id="{bounce_id}")'
        )

        result = await engine.save(
            "distinct successor content in beta scope", scope="beta",
            supersedes=[candidate_id], reason="moved", bounce_id=bounce_id,
        )
        assert result.outcome in ("saved", "reconciled"), result

        predecessor_after = await adapter.get_by_id(candidate_id)
        assert isinstance(predecessor_after, Fact)
        assert predecessor_after.validity == "invalid"


class TestSaveSupersedesAllOrNothing:
    async def test_one_dead_predecessor_refuses_the_whole_save(self) -> None:
        """D2 (P3 one-transaction-supersession piece): multi-id ``supersedes`` is all-or-nothing.
        One named
        predecessor already superseded by someone else BEFORE this save
        runs refuses the ENTIRE call — the successor is never written, and
        the OTHER, healthy predecessor is left completely untouched, not
        superseded on its own with the dead edge merely reported.

        Falsify: catch ``adapter.replace``'s ``ValueError`` inside
        ``_write_fact`` and still return ``outcome="saved"`` with the
        failure named in ``SaveResult.reason`` (the pre-P3 partial-failure
        shape) — this test goes red because the successor Fact WOULD exist
        and the healthy predecessor WOULD be flipped.
        """
        engine, adapter, _clock = await _engine()
        already_superseded = await engine.save(
            "predecessor already retired by someone else", scope="global"
        )
        its_successor = await engine.save("the record that retired it", scope="global")
        healthy_pred = await engine.save("a healthy predecessor", scope="global")
        assert already_superseded.hypomnema is not None
        assert its_successor.hypomnema is not None
        assert healthy_pred.hypomnema is not None

        await engine.supersede(
            its_successor.hypomnema.id, [already_superseded.hypomnema.id], "changed"
        )

        with pytest.raises(ToolError, match=str(already_superseded.hypomnema.id)):
            await engine.save(
                "new successor naming one dead edge and one healthy one",
                scope="global",
                supersedes=[already_superseded.hypomnema.id, healthy_pred.hypomnema.id],
                reason="changed",
            )

        healthy_after = await adapter.get_by_id(healthy_pred.hypomnema.id)
        assert isinstance(healthy_after, Fact)
        assert healthy_after.validity == "valid"
        assert healthy_after.superseded_by is None

        found = await find_superseded_by(healthy_pred.hypomnema.id, adapter=adapter)
        assert found == []


class TestSaveSupersedesReasonRequired:
    async def test_supersedes_without_reason_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D1-D3: ``reason`` is required whenever ``supersedes`` is
        non-empty, checked BEFORE the embedder runs — a pre-write refusal,
        never a ``replace`` failure, so no ``supersession_failed`` line.
        """
        engine, adapter, _clock = await _engine()
        pred = await engine.save(
            "predecessor for the reason-required test", scope="global"
        )
        assert pred.hypomnema is not None

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="reason"):
                await engine.save(
                    "successor for the reason-required test", scope="global",
                    supersedes=[pred.hypomnema.id],
                )
        assert _no_supersession_failed_logged(caplog)

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_reason_without_supersedes_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The inverse refusal: ``reason`` set with no ``supersedes`` at
        all — an ordinary save carrying a stray reason class is refused,
        not silently ignored.
        """
        engine, _adapter, _clock = await _engine()
        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="reason"):
                await engine.save(
                    "an ordinary fact with a stray reason", scope="global",
                    reason="changed",
                )
        assert _no_supersession_failed_logged(caplog)


class TestSupersedeCrossScopeRejected:
    """D3/D10 (P3 one-transaction-supersession piece, decisions
    earlier decisions): the caller's own ``supersedes`` list must be
    all same-scope or all cross-scope relative to the successor, and
    ``reason="moved"`` is legal ONLY against a cross-scope predecessor —
    every other reason class is same-scope only.

    ``save``'s Fact arm ADDITIONALLY keeps the pre-existing read-span guard
    (Scope 2 step 1: "the existing span check stays") — a predecessor must
    be a member of ``read_span(successor_scope)`` (the successor's own
    scope, or ``global``), so the only reachable "cross-scope" pair via
    ``save`` is (non-global scope, ``global``); an unrelated sibling scope
    (e.g. ``alpha`` naming a predecessor while saving in ``beta``) is
    refused regardless of ``reason``. The explicit ``supersede(...)`` op is the deliberately
    UNRESTRICTED counterpart — the
    same administrative category as ``tabularium reclaim`` — and reaches
    arbitrary sibling-scope pairs the read span itself never would.
    """

    async def test_out_of_span_predecessor_refused_regardless_of_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A save in one scope naming a predecessor living in an UNRELATED
        sibling scope (neither the successor's own scope nor ``global``) is
        refused before any write, even with ``reason="moved"`` — mirrors
        the cross-scope guard the Decision arm's own
        ``supersedes``/``promoted_from`` already carry.

        Falsify: drop the ``span`` check in ``check_supersedes_reason`` —
        this test goes red with ``outcome == "saved"`` and the alpha-scope
        predecessor flipped to invalid.
        """
        engine, adapter, _clock = await _engine()
        await engine._adapter.create_scope(Scope("alpha"))
        await engine._adapter.create_scope(Scope("beta"))

        pred = await engine.save("predecessor fact in alpha scope", scope="alpha")
        assert pred.hypomnema is not None

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="read span"):
                await engine.save(
                    "successor fact written in beta scope", scope="beta",
                    supersedes=[pred.hypomnema.id], reason="moved",
                )
        assert _no_supersession_failed_logged(caplog)

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_supersede_direct_call_reaches_an_out_of_span_pair(self) -> None:
        """The explicit ``supersede(...)`` op passes no span — an
        administrative action, unlike ``save``, is not scope-restricted:
        it reaches the exact sibling-scope pair the test above refuses via
        ``save``.
        """
        engine, adapter, _clock = await _engine()
        await engine._adapter.create_scope(Scope("alpha"))
        await engine._adapter.create_scope(Scope("beta"))

        pred = await engine.save("predecessor fact in alpha scope", scope="alpha")
        succ = await engine.save("successor fact in beta scope", scope="beta")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        result = await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "moved")
        assert result.successor_id == succ.hypomnema.id

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "invalid"
        assert pred_after.superseded_by == succ.hypomnema.id

    async def test_cross_scope_global_predecessor_with_wrong_reason_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The one cross-scope pair a Fact save CAN reach (a non-global
        successor naming a ``global``-scope predecessor — the only other
        member of ``read_span``) still requires ``reason="moved"``; a
        same-scope-only class is refused.

        Falsify: drop the ``cross_scope and reason != "moved"`` check in
        ``check_supersedes_reason``.
        """
        engine, adapter, _clock = await _engine()
        await engine._adapter.create_scope(Scope("beta"))

        pred = await engine.save("predecessor fact in global scope", scope="global")
        assert pred.hypomnema is not None

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="moved"):
                await engine.save(
                    "successor fact written in beta scope", scope="beta",
                    supersedes=[pred.hypomnema.id], reason="correction",
                )
        assert _no_supersession_failed_logged(caplog)

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_cross_scope_global_predecessor_with_moved_reason_succeeds(
        self,
    ) -> None:
        """The legitimate cross-scope path: a non-global successor naming a
        ``global``-scope predecessor with ``reason="moved"`` succeeds, one
        commit, exactly like a same-scope supersession.
        """
        engine, adapter, _clock = await _engine()
        await engine._adapter.create_scope(Scope("beta"))

        pred = await engine.save("predecessor fact in global scope", scope="global")
        assert pred.hypomnema is not None

        result = await engine.save(
            "successor fact written in beta scope", scope="beta",
            supersedes=[pred.hypomnema.id], reason="moved",
        )
        assert result.outcome == "saved", result
        assert result.hypomnema is not None

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "invalid"
        assert pred_after.superseded_by == result.hypomnema.id
        assert pred_after.supersession_reason == "moved"

    async def test_same_scope_predecessor_with_moved_reason_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``reason="moved"`` against a same-scope predecessor is refused —
        "moved" is for cross-scope supersession only (D10).
        """
        engine, adapter, _clock = await _engine()
        pred = await engine.save(
            "same-scope predecessor for the moved-reason test", scope="global"
        )
        assert pred.hypomnema is not None

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="moved"):
                await engine.save(
                    "successor for the moved-reason test", scope="global",
                    supersedes=[pred.hypomnema.id], reason="moved",
                )
        assert _no_supersession_failed_logged(caplog)

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_mixed_scope_predecessors_refused_before_write(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D3: a caller's own list must be all same-scope or all
        cross-scope relative to the successor — one same-scope and one
        cross-scope predecessor in the SAME call is refused, naming both
        ids and saying to split by scope, before any write. Both
        predecessors are within the successor's own read span (``beta``
        and ``global``) so this exercises the mixed-scope check itself,
        not the separate out-of-span refusal above.
        """
        engine, adapter, _clock = await _engine()
        await engine._adapter.create_scope(Scope("beta"))

        same_scope_pred = await engine.save(
            "same-scope predecessor for the mixed-scope test", scope="beta"
        )
        cross_scope_pred = await engine.save(
            "cross-scope predecessor for the mixed-scope test", scope="global"
        )
        assert same_scope_pred.hypomnema is not None
        assert cross_scope_pred.hypomnema is not None

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError, match="split into two calls"):
                await engine.save(
                    "successor naming a mixed-scope list", scope="beta",
                    supersedes=[
                        same_scope_pred.hypomnema.id, cross_scope_pred.hypomnema.id,
                    ],
                    reason="changed",
                )
        assert _no_supersession_failed_logged(caplog)

        for pred in (same_scope_pred, cross_scope_pred):
            assert pred.hypomnema is not None
            rec = await adapter.get_by_id(pred.hypomnema.id)
            assert isinstance(rec, Fact)
            assert rec.validity == "valid"
            assert rec.superseded_by is None



class TestSupersessionFailedLogLine:
    async def test_replace_failure_emits_one_supersession_failed_line(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D8 (P3 one-transaction-supersession piece): a failed ``replace`` emits exactly ONE
        WARNING on
        the ``tabularium`` logger, message ``supersession_failed``, with
        all eight fields — never an events row.

        Fault injected through ``FaultInjectingAdapter(on="replace")``
        (a prior decision: the call is named ``replace`` so fault
        tooling can target it directly).
        """
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        raw_adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
        wrapper = FaultInjectingAdapter(raw_adapter)
        cfg = _make_config()
        engine = Tabularium(as_adapter(wrapper), embedder, clock, cfg)
        await engine.initialize()

        pred = await engine.save(
            "predecessor for the fault-injection log-line test", scope="global"
        )
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id

        wrapper.fault(on="replace", call=1, exc=ValueError("simulated replace failure"))
        wrapper.arm()

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError):
                await engine.save(
                    "successor for the fault-injection log-line test",
                    scope="global", supersedes=[pred_id], reason="changed",
                )

        matching = [
            r for r in caplog.records if r.getMessage().startswith("supersession_failed")
        ]
        assert len(matching) == 1, caplog.records
        record = matching[0]
        assert record.op == "save"  # type: ignore[attr-defined]
        assert record.arm == "fact"  # type: ignore[attr-defined]
        assert record.successor_id  # type: ignore[attr-defined]
        assert record.predecessor_ids == [str(pred_id)]  # type: ignore[attr-defined]
        assert record.reason == "changed"  # type: ignore[attr-defined]
        assert record.scope == "global"  # type: ignore[attr-defined]
        assert record.error == "ValueError"  # type: ignore[attr-defined]
        assert "simulated replace failure" in record.detail  # type: ignore[attr-defined]

        raw_pred = await raw_adapter.get_by_id(pred_id)
        assert isinstance(raw_pred, Fact)
        assert raw_pred.validity == "valid"
        assert raw_pred.superseded_by is None

    async def test_non_valueerror_replace_failure_still_logs_and_raises_toolerror(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A genuine
        storage I/O failure (disk fault, permission error) does not
        necessarily surface as ``ValueError`` — ``replace``'s own
        legality-refusal contract is ``ValueError``-only, but the adapter
        itself can raise anything (``OSError``, a raw Lance panic). Every
        ``replace`` caller must convert ANY exception, not only
        ``ValueError``, into the log line + ``ToolError`` — never let a
        non-``ValueError`` failure escape as a raw internal error.

        Falsify: narrow the ``except`` clause in ``_write_fact`` back to
        ``except ValueError`` — this test's ``pytest.raises(ToolError)``
        goes red as the injected ``RuntimeError`` propagates raw instead.
        """
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        raw_adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
        wrapper = FaultInjectingAdapter(raw_adapter)
        cfg = _make_config()
        engine = Tabularium(as_adapter(wrapper), embedder, clock, cfg)
        await engine.initialize()

        pred = await engine.save(
            "predecessor for the non-ValueError fault-injection test", scope="global"
        )
        assert pred.hypomnema is not None
        pred_id = pred.hypomnema.id

        wrapper.fault(on="replace", call=1, exc=RuntimeError("simulated disk I/O failure"))
        wrapper.arm()

        with caplog.at_level(logging.WARNING, logger="tabularium"):
            with pytest.raises(ToolError):
                await engine.save(
                    "successor for the non-ValueError fault-injection test",
                    scope="global", supersedes=[pred_id], reason="changed",
                )

        matching = [
            r for r in caplog.records if r.getMessage().startswith("supersession_failed")
        ]
        assert len(matching) == 1, caplog.records
        assert matching[0].error == "RuntimeError"  # type: ignore[attr-defined]

        raw_pred = await raw_adapter.get_by_id(pred_id)
        assert isinstance(raw_pred, Fact)
        assert raw_pred.validity == "valid"
        assert raw_pred.superseded_by is None


# ---------------------------------------------------------------------------
# Scope 6 — the read side
# ---------------------------------------------------------------------------


class TestRecallSurfacesSupersededBy:
    async def test_include_invalid_returns_the_pointer(self) -> None:
        """A recall with ``include_invalid=True`` returns the pointer; the
        same recall without the flag returns neither the record nor the
        field (a superseded Fact is always invalid, so the default
        valid-only filter already excludes it — no special-casing needed).
        """
        engine, _adapter, _clock = await _engine()
        pred_content = "predecessor content for the recall pointer test"
        pred = await engine.save(pred_content, scope="global")
        succ = await engine.save("successor content for the recall pointer test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        await engine.supersede(succ.hypomnema.id, [pred.hypomnema.id], "changed")

        default_recall = await engine.recall(pred_content, scope="global")
        assert all(h.id != pred.hypomnema.id for h in default_recall.hits)

        included_recall = await engine.recall(pred_content, scope="global", include_invalid=True)
        hit = next(h for h in included_recall.hits if h.id == pred.hypomnema.id)
        assert hit.superseded_by == succ.hypomnema.id

    async def test_ordinary_recall_hit_carries_no_pointer(self) -> None:
        """A never-superseded Fact's hit carries ``superseded_by=None``."""
        engine, _adapter, _clock = await _engine()
        content = "an ordinary fact, never superseded"
        r = await engine.save(content, scope="global")
        assert r.hypomnema is not None

        recalled = await engine.recall(content, scope="global")
        hit = next(h for h in recalled.hits if h.id == r.hypomnema.id)
        assert hit.superseded_by is None


class TestFindSupersededBy:
    async def test_returns_all_three_predecessors_in_one_call(self) -> None:
        """The backward query returns all three predecessors from the
        Scope 4 multi-supersession scenario in one call — the query that
        justifies the forward-only pointer design.

        Falsify: only scan/match the first id (an off-by-loop bug), or drop
        the ``isinstance(rec, Fact)`` guard so a Decision row raises.
        """
        engine, adapter, _clock = await _engine()
        p1 = await engine.save("predecessor one: policy A v1", scope="global")
        p2 = await engine.save("predecessor two: policy B v1", scope="global")
        p3 = await engine.save("predecessor three: policy C v1", scope="global")
        assert p1.hypomnema is not None
        assert p2.hypomnema is not None
        assert p3.hypomnema is not None

        result = await engine.save(
            "successor: unified policy v2, replaces A, B, and C",
            scope="global",
            supersedes=[p1.hypomnema.id, p2.hypomnema.id, p3.hypomnema.id],
            reason="changed",
        )
        assert result.outcome == "saved", result
        assert result.hypomnema is not None

        predecessors = await find_superseded_by(result.hypomnema.id, adapter=adapter)
        assert {p.id for p in predecessors} == {
            p1.hypomnema.id, p2.hypomnema.id, p3.hypomnema.id,
        }
        assert all(p.validity == "invalid" for p in predecessors)

    async def test_returns_empty_list_when_nothing_supersedes_the_id(self) -> None:
        engine, adapter, _clock = await _engine()
        r = await engine.save("a fact nothing supersedes", scope="global")
        assert r.hypomnema is not None

        predecessors = await find_superseded_by(r.hypomnema.id, adapter=adapter)
        assert predecessors == []


# ---------------------------------------------------------------------------
# Two reproduced findings, fixed. Reworked for P3's all-or-nothing,
# no-drawn-edges-after-the-lock design: the old contract had ``reinforced``/
# ``revived``/``already_saved`` draw an edge against whatever existing
# record they named (retry-safety), narrowed by Decision 1 to a LOUD skip
# reported in ``SaveResult.reason``. P3 removes ``SaveResult.reason`` as a
# skip-reporting channel entirely (Scope 2 step 2: "drop the partial-edge
# text from SaveResult.reason") — these three outcomes still draw no edge,
# but the caller is no longer told why via ``reason``; only the storage
# state (predecessor untouched) is asserted below.
# ---------------------------------------------------------------------------


class TestSupersedeSkippedOnGateMatchedSuccessor:
    """2026-08-10 Decision 1: edge-drawing is narrowed to
    ``saved``/``reconciled`` only — ``reinforced``/``revived``/
    ``already_saved`` can name a record the GATE matched, not content the
    caller submitted, so no edge is ever drawn for them. ``reason`` names
    every un-drawn predecessor and why (the loud skip Decision 1 itself
    calls for) — separate from, and NOT reintroduced by, P3's own removal
    of the all-or-nothing partial-EDGE-failure reporting on the
    saved/reconciled write path (D1/D2): this is "the edge was never
    attempted", not "the edge was attempted and one of several failed".

    Restored after
    an unintentional regression during P3 (my own new ``_write_fact``
    docstring read "the two outcomes whose successor is content THIS call
    actually wrote" as license to drop the field for the other three too,
    which the brief never asked for).
    """

    async def test_revived_outcome_draws_no_edge(self) -> None:
        """Reviewer's own reproduction: an unrelated ARCHIVED Fact sits
        above ``gray_floor`` against new content — the save resolves to
        ``revived``, the caller's content is never written at all, and
        without Decision 1 the predecessor would be retired pointing at the
        archived record instead of staying live and un-superseded.

        Controlled-angle embedder against ``MemoryAdapter``, the same
        device this file's other tests use: the archived Fact and the new
        content sit ~1 degree apart (cosine ~0.9998, comfortably above
        ``gray_floor``); the named predecessor sits 90 degrees off both
        (cosine 0) so it is never itself a candidate.

        Falsify: widen the outcome tuple in ``save_internal`` back to
        include ``revived`` — this test goes red with the predecessor
        flipped to ``invalid`` and pointed at the archived record.
        """
        archived_content = "an unrelated archived fact the gate happens to match"
        pred_content = "the actual predecessor the caller means to supersede"
        new_content = "new content that collides with the archived fact by coincidence"
        table = {
            archived_content: _angle_vec(0.0),
            pred_content: _angle_vec(90.0),
            new_content: _angle_vec(1.0),
        }
        adapter = MemoryAdapter(embedding_model="test/table", dim=_DIM)
        embedder = _TableEmbedder(table)
        clock = _make_clock()
        cfg = _make_config(gray_floor=0.80)
        engine = Tabularium(adapter, embedder, clock, cfg)
        await engine.initialize()
        now = clock.now()

        archived = Fact(
            id=new_id(), type="fact", content=archived_content,
            scope=Scope("global"), lifecycle="archived", created_at=now, updated_at=now,
        )
        await adapter.upsert(archived, _angle_vec(0.0))

        pred = await engine.save(pred_content, scope="global")
        assert pred.hypomnema is not None

        result = await engine.save(
            new_content, scope="global",
            supersedes=[pred.hypomnema.id], reason="changed",
        )
        assert result.outcome == "revived", result
        assert result.reinforced_id == archived.id
        assert result.reason is not None
        assert str(pred.hypomnema.id) in result.reason

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

        # The archived record itself was genuinely revived — the outcome is
        # real, only the edge is withheld.
        archived_after = await adapter.get_by_id(archived.id)
        assert isinstance(archived_after, Fact)
        assert archived_after.lifecycle == "active"

    async def test_reinforced_outcome_draws_no_edge(self) -> None:
        """A ``save(content=X, supersedes=[pred])`` that resolves to
        ``reinforced`` (X already exists, auto-reinforce fires) draws NO
        edge — the successor is an existing record the gate matched, not
        content this call submitted (Decision 1). The predecessor stays
        valid.

        Falsify: widen the outcome tuple in ``save_internal`` back to
        include ``reinforced`` — this test goes red with the predecessor
        flipped to ``invalid``.
        """
        engine, adapter, _clock = await _engine(config=_make_config())
        pred = await engine.save("predecessor content for retry test", scope="global")
        succ = await engine.save("successor content for retry test", scope="global")
        assert pred.hypomnema is not None
        assert succ.hypomnema is not None

        # Byte-identical to succ's own content: admission step 1 (P7
        # admission-order piece) acks already_saved(succ) directly, before
        # the probe/dispatch this test originally exercised ever runs —
        # same "no edge for pred" invariant, carried on ``hypomnema``
        # instead of ``reinforced_id`` (already_saved never reinforces).
        retry = await engine.save(
            "successor content for retry test", scope="global",
            supersedes=[pred.hypomnema.id], reason="changed",
        )
        assert retry.outcome == "already_saved", retry
        assert retry.hypomnema is not None
        assert retry.hypomnema.id == succ.hypomnema.id
        assert retry.reason is not None
        assert str(pred.hypomnema.id) in retry.reason

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None

    async def test_already_saved_replay_draws_no_edge(self) -> None:
        """The ``already_saved`` outcome (confirm_distinct replay guard)
        names a record the GATE matched via ``hypomnema.id`` — never the
        caller's own submitted content — so no edge is drawn (Decision 1).

        Constructs a genuine ``already_saved`` replay directly (same
        ``_TableEmbedder`` + ``upsert``-a-candidate device
        ``tests/test_engine_b2.py``'s own replay-guard tests use): an
        existing candidate O, a ``resolve(candidate_id=O,
        action="confirm_distinct")`` call that writes N, then the SAME
        resolve replayed a second time — which the replay guard recognizes
        and acks as ``already_saved`` naming N, never a new write.

        Falsify: widen the outcome tuple in ``save_internal`` back to
        include ``already_saved`` — this test goes red with the
        predecessor flipped to ``invalid``.
        """
        engine, adapter, clock = await _engine()
        now = clock.now()

        candidate = Fact(
            id=new_id(), type="fact", content="Alpha policy on release cadence.",
            scope=Scope("global"), lifecycle="active", created_at=now, updated_at=now,
        )
        await adapter.upsert(candidate, _angle_vec(0.0))

        pred = await engine.save("predecessor for already_saved retry test", scope="global")
        assert pred.hypomnema is not None

        c1 = "incoming policy note on release cadence"
        engine._embedder = _TableEmbedder({
            "predecessor for already_saved retry test": _angle_vec(180.0),
            c1: _angle_vec(10.0),
        })

        bounced = await engine.save(c1, scope="global")
        assert bounced.outcome == "bounced", bounced

        resolution = BounceResolution(candidate_id=candidate.id, action="confirm_distinct")
        written = await engine.save(c1, scope="global", resolve=resolution)
        assert written.outcome == "saved", written
        assert written.hypomnema is not None
        successor_id = written.hypomnema.id

        replay = await engine.save(
            c1, scope="global", resolve=resolution,
            supersedes=[pred.hypomnema.id], reason="changed",
        )
        assert replay.outcome == "already_saved", replay
        assert replay.hypomnema is not None
        assert replay.hypomnema.id == successor_id
        assert replay.reason is not None
        assert str(pred.hypomnema.id) in replay.reason

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "valid"
        assert pred_after.superseded_by is None


class TestSupersedeConcurrentEdgeLoserRaises:
    async def test_concurrent_save_supersedes_loser_gets_tool_error(self) -> None:
        """D1/D2 (P3 one-transaction-supersession piece): two independent
        ``Tabularium`` instances (mirroring two separate MCP server
        processes/requests, not two calls sharing one engine's write lock)
        share ONE ``MemoryAdapter`` and race
        ``save(content=<distinct>, supersedes=[pred], reason=...)`` against
        the SAME predecessor. Exactly one call's write lands (its new Fact
        AND the predecessor flip, one commit); the OTHER call's ENTIRE
        write is refused — including its own new successor content, since
        the successor row and every predecessor flip are now one atomic
        ``replace`` call, never two. The loser gets a ``ToolError``, not a
        silently-landed row.

        This reverses the pre-P3 contract (a lost edge was reported in
        ``SaveResult.reason``, both writes landed): D1 states a failed
        supersession means NOTHING landed, and ``save`` raises ``ToolError``
        — no exception ever escaping ``engine.save`` after a partial write
        is no longer the property being protected, because there is no
        longer a partial write to protect against.

        Falsify: catch ``replace``'s ``ValueError`` inside ``_write_fact``
        and still return ``outcome="saved"`` for the loser (the pre-P3
        partial-failure shape) — this test goes red because BOTH gather
        results would be clean ``SaveResult``s and a second Fact naming the
        loser's own content would exist in the store.
        """
        embedder = FakeEmbedder(dim=_DIM)
        clock = _make_clock()
        adapter = MemoryAdapter(embedding_model=embedder.model, dim=_DIM, clock=clock)
        cfg = _make_config()

        engine_a = Tabularium(adapter, embedder, clock, cfg)
        await engine_a.initialize()
        engine_b = Tabularium(adapter, embedder, clock, cfg)

        pred = await engine_a.save(
            "predecessor content for concurrent-loser test", scope="global"
        )
        assert pred.hypomnema is not None

        results = await asyncio.gather(
            engine_a.save(
                "successor A content, concurrent-loser test",
                scope="global", supersedes=[pred.hypomnema.id], reason="changed",
            ),
            engine_b.save(
                "successor B content, concurrent-loser test",
                scope="global", supersedes=[pred.hypomnema.id], reason="changed",
            ),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, SaveResult)]
        failures = [r for r in results if isinstance(r, ToolError)]
        assert len(successes) == 1, results
        assert len(failures) == 1, results
        assert successes[0].outcome == "saved"
        assert successes[0].hypomnema is not None

        pred_after = await adapter.get_by_id(pred.hypomnema.id)
        assert isinstance(pred_after, Fact)
        assert pred_after.validity == "invalid"
        assert pred_after.superseded_by == successes[0].hypomnema.id
