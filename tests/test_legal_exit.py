"""``legal_exit`` table-driven test (P8 assessor-automatic-exits piece, decisions D2/D10) —
pins the Constraints matrix row for row, on both arms, including the fixed check order and
every ``WithheldReason`` token at least once.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import get_args

import pytest

from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine.admission import (
    AutoExitSwitches,
    Invalidate,
    Moved,
    Promote,
    Reinforce,
    Supersede,
    Withheld,
    WriteNew,
    clean_picture,
    legal_exit,
)
from tabularium.engine.assessor import RelationVerdict, Trace, Unavailable
from tabularium.results import WithheldReason

_NOW = datetime(2025, 1, 1, tzinfo=UTC)

_ALL_ON = AutoExitSwitches(restatement=True, replacement=True, follow_up=True, distinct=True)
_ALL_OFF = AutoExitSwitches(restatement=False, replacement=False, follow_up=False, distinct=False)


def _fact(*, scope: str = "p", lifecycle: str = "active", validity: str = "valid") -> Fact:
    return Fact(
        id=new_id(),
        type="fact",
        content="candidate content",
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=_NOW,
        updated_at=_NOW,
    )


def _decision(
    *, scope: str = "p", lifecycle: str = "in_force", validity: str = "valid"
) -> Decision:
    return Decision(
        id=new_id(),
        type="decision",
        content="candidate content",
        scope=Scope(scope),
        lifecycle=lifecycle,  # type: ignore[arg-type]
        validity=validity,  # type: ignore[arg-type]
        created_at=_NOW,
        updated_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Order of checks: retired candidate, switch, kind pair, direction, reason class.
# ---------------------------------------------------------------------------


def test_retired_candidate_beats_switch_off() -> None:
    """A retired candidate with its relation's switch OFF still reports
    ``candidate_retired``, never ``switch_off`` — retired is checked first."""
    superseded_fact = _fact(validity="invalid")
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=superseded_fact,
        direction="same",
        switches=_ALL_OFF,
    )
    assert result == Withheld(reason="candidate_retired")


def test_retired_decision_candidate_beats_switch_on() -> None:
    revoked_decision = _decision(lifecycle="revoked")
    result = legal_exit(
        relation="replacement",
        reason_class="correction",
        incoming_kind="fact",
        candidate=revoked_decision,
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="candidate_retired")


def test_ember_fact_candidate_is_retired() -> None:
    """A Fact outside the admission population (``ember``) is ``candidate_retired`` —
    the parenthetical in the matrix's own wording ("not in admission population")."""
    ember_fact = _fact(lifecycle="ember")
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=ember_fact,
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="candidate_retired")


def test_decision_against_fact_beats_cross_scope_replacement() -> None:
    """Kind pair is checked ahead of direction: a Decision save judged against a
    Fact candidate reports ``decision_against_fact`` even when the pairing is
    ALSO cross-scope."""
    cross_scope_fact = _fact(scope="q")
    result = legal_exit(
        relation="replacement",
        reason_class="correction",
        incoming_kind="decision",
        candidate=cross_scope_fact,
        direction="upward",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="decision_against_fact")


def test_decision_against_fact_beats_no_reason_class() -> None:
    """Kind pair is checked ahead of reason class: a Decision save judged against a
    Fact candidate reports ``decision_against_fact`` even with no reason class."""
    fact_candidate = _fact()
    result = legal_exit(
        relation="replacement",
        reason_class=None,
        incoming_kind="decision",
        candidate=fact_candidate,
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="decision_against_fact")


def test_cross_scope_replacement_beats_no_reason_class() -> None:
    """Direction is checked ahead of reason class: a cross-scope Fact/Fact pairing
    reports ``cross_scope_replacement`` even with no reason class."""
    cross_scope_fact = _fact(scope="q")
    result = legal_exit(
        relation="replacement",
        reason_class=None,
        incoming_kind="fact",
        candidate=cross_scope_fact,
        direction="upward",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="cross_scope_replacement")


# ---------------------------------------------------------------------------
# restatement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["same", "downward"])
def test_restatement_fact_fact_reinforces(direction: str) -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=_fact(),
        direction=direction,  # type: ignore[arg-type]
        switches=_ALL_ON,
    )
    assert result == Reinforce()


@pytest.mark.parametrize("direction", ["same", "downward"])
def test_restatement_decision_decision_reinforces(direction: str) -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="decision",
        candidate=_decision(),
        direction=direction,  # type: ignore[arg-type]
        switches=_ALL_ON,
    )
    assert result == Reinforce()


def test_restatement_upward_promotes() -> None:
    """A4 bounce-judged piece: a legal same-kind upward restatement is a per-candidate
    promotion target (``Promote``), never dispatched alone — ``clean_picture`` composes
    it (``tests/test_auto_exits.py``). ``upward_restatement`` is gone from
    ``WithheldReason`` entirely (D14)."""
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=_fact(),
        direction="upward",
        switches=_ALL_ON,
    )
    assert result == Promote()


def test_restatement_upward_decision_promotes() -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="decision",
        candidate=_decision(),
        direction="upward",
        switches=_ALL_ON,
    )
    assert result == Promote()


def test_restatement_cross_kind_fact_vs_decision_withheld() -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=_decision(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="cross_kind_restatement")


def test_restatement_cross_kind_decision_vs_fact_withheld() -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="decision",
        candidate=_fact(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="cross_kind_restatement")


def test_restatement_switch_off() -> None:
    result = legal_exit(
        relation="restatement",
        reason_class=None,
        incoming_kind="fact",
        candidate=_fact(),
        direction="same",
        switches=_ALL_OFF,
    )
    assert result == Withheld(reason="switch_off")


# ---------------------------------------------------------------------------
# replacement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["correction", "changed", "elaborated"])
def test_replacement_fact_fact_supersedes(reason: str) -> None:
    result = legal_exit(
        relation="replacement",
        reason_class=reason,  # type: ignore[arg-type]
        incoming_kind="fact",
        candidate=_fact(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Supersede(reason=reason)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["correction", "changed", "elaborated"])
def test_replacement_decision_decision_supersedes(reason: str) -> None:
    result = legal_exit(
        relation="replacement",
        reason_class=reason,  # type: ignore[arg-type]
        incoming_kind="decision",
        candidate=_decision(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Supersede(reason=reason)  # type: ignore[arg-type]


@pytest.mark.parametrize("reason", ["correction", "changed"])
def test_replacement_fact_invalidates_decision(reason: str) -> None:
    result = legal_exit(
        relation="replacement",
        reason_class=reason,  # type: ignore[arg-type]
        incoming_kind="fact",
        candidate=_decision(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Invalidate(reason=reason)  # type: ignore[arg-type]


def test_replacement_fact_elaborates_decision_withheld() -> None:
    result = legal_exit(
        relation="replacement",
        reason_class="elaborated",
        incoming_kind="fact",
        candidate=_decision(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="elaborated_against_decision")


@pytest.mark.parametrize("reason_class", [None, "moved"])
def test_replacement_no_reason_class_withheld(reason_class: str | None) -> None:
    result = legal_exit(
        relation="replacement",
        reason_class=reason_class,  # type: ignore[arg-type]
        incoming_kind="fact",
        candidate=_fact(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="no_reason_class")


def test_replacement_cross_scope_withheld() -> None:
    result = legal_exit(
        relation="replacement",
        reason_class="changed",
        incoming_kind="fact",
        candidate=_fact(scope="q"),
        direction="upward",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="cross_scope_replacement")


def test_replacement_decision_against_fact_withheld() -> None:
    result = legal_exit(
        relation="replacement",
        reason_class="correction",
        incoming_kind="decision",
        candidate=_fact(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="decision_against_fact")


def test_replacement_switch_off() -> None:
    result = legal_exit(
        relation="replacement",
        reason_class="correction",
        incoming_kind="fact",
        candidate=_fact(),
        direction="same",
        switches=_ALL_OFF,
    )
    assert result == Withheld(reason="switch_off")


# ---------------------------------------------------------------------------
# follow_up / distinct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("relation", ["follow_up", "distinct"])
@pytest.mark.parametrize(
    ("incoming_kind", "candidate"),
    [
        ("fact", _fact()),
        ("fact", _decision()),
        ("decision", _decision()),
    ],
)
def test_follow_up_distinct_write_new(
    relation: str, incoming_kind: str, candidate: Fact | Decision
) -> None:
    result = legal_exit(
        relation=relation,  # type: ignore[arg-type]
        reason_class=None,
        incoming_kind=incoming_kind,  # type: ignore[arg-type]
        candidate=candidate,
        direction="same",
        switches=_ALL_ON,
    )
    assert result == WriteNew()


@pytest.mark.parametrize("relation", ["follow_up", "distinct"])
def test_follow_up_distinct_decision_against_fact_withheld(relation: str) -> None:
    result = legal_exit(
        relation=relation,  # type: ignore[arg-type]
        reason_class=None,
        incoming_kind="decision",
        candidate=_fact(),
        direction="same",
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="decision_against_fact")


@pytest.mark.parametrize("relation", ["follow_up", "distinct"])
def test_follow_up_distinct_switch_off(relation: str) -> None:
    switches = AutoExitSwitches(
        restatement=True,
        replacement=True,
        follow_up=relation != "follow_up",
        distinct=relation != "distinct",
    )
    result = legal_exit(
        relation=relation,  # type: ignore[arg-type]
        reason_class=None,
        incoming_kind="fact",
        candidate=_fact(),
        direction="same",
        switches=switches,
    )
    assert result == Withheld(reason="switch_off")


# ---------------------------------------------------------------------------
# Every WithheldReason token fires at least once above.
# ---------------------------------------------------------------------------


def test_every_legal_exit_withheld_reason_token_is_exercised() -> None:
    """Every ``WithheldReason`` token ``legal_exit`` itself can return, exercised above at
    least once. ``candidate_changed`` is the one token this pure function structurally
    cannot produce — it names a candidate that changed BETWEEN the judged snapshot and a
    fresh re-read, a comparison ``legal_exit`` never sees (it takes only the fresh
    ``candidate``, per D15); the calling arm detects and reports it directly before ever
    reaching this function (covered in ``tests/test_auto_exits.py``). ``competing_candidate``
    (A4, D14) is likewise never returned by ``legal_exit`` itself — only ``clean_picture``,
    which composes several candidates' own ``legal_exit`` calls, can detect a competing
    survivor (covered in ``tests/test_legal_exit.py``'s ``clean_picture`` table and
    ``tests/test_auto_exits.py``)."""
    exercised: set[str] = {
        "candidate_retired",
        "switch_off",
        "cross_kind_restatement",
        "decision_against_fact",
        "cross_scope_replacement",
        "no_reason_class",
        "elaborated_against_decision",
    }
    assert exercised == set(get_args(WithheldReason)) - {"candidate_changed", "competing_candidate"}


# ---------------------------------------------------------------------------
# clean_picture (A4 bounce-judged piece, decision D16) — the composed
# multi-candidate dispatch decision.
# ---------------------------------------------------------------------------


def _trace() -> Trace:
    return Trace(
        model="test", wording_hash="test", latency_ms=0,
        input_tokens=None, output_tokens=None, request_id=None,
    )


def _verdict(
    pick: str, *, reason_class: str | None = None
) -> RelationVerdict:
    odds = {"restatement": 0.0, "distinct": 0.0, "follow_up": 0.0, "replacement": 0.0}
    odds[pick] = 1.0
    return RelationVerdict(
        method="relation", pick=pick, odds=odds, confidence=1.0,  # type: ignore[arg-type]
        reason_class=reason_class, trace=_trace(),  # type: ignore[arg-type]
    )


def _unavailable() -> Unavailable:
    return Unavailable(method="relation", reason="timeout", detail=None, latency_ms=50)


def test_clean_picture_restatement_second_replacement_competes() -> None:
    """Observable: two survivors, top restatement switched on, second replacement:
    bounce, competing_candidate."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("restatement"),
        top_candidate=_fact(),
        other_verdicts=[_verdict("replacement", reason_class="correction")],
        other_candidates=[_fact()],
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="competing_candidate")


def test_clean_picture_restatement_second_restatement_competes() -> None:
    """Observable: the same, but the second is restatement: bounce, competing_candidate."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("restatement"),
        top_candidate=_fact(),
        other_verdicts=[_verdict("restatement")],
        other_candidates=[_fact()],
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="competing_candidate")


def test_clean_picture_restatement_second_distinct_reinforces() -> None:
    """Observable: the same, but the second is distinct or follow_up: auto-reinforce
    on the top candidate."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("restatement"),
        top_candidate=_fact(),
        other_verdicts=[_verdict("distinct")],
        other_candidates=[_fact()],
        switches=_ALL_ON,
    )
    assert result == Reinforce()


def test_clean_picture_second_unavailable_does_not_break_clean() -> None:
    """Observable: a second survivor whose verdict is unavailable does not break the
    clean picture."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("restatement"),
        top_candidate=_fact(),
        other_verdicts=[_unavailable()],
        other_candidates=[_fact()],
        switches=_ALL_ON,
    )
    assert result == Reinforce()


def test_clean_picture_replacement_second_distinct_auto_supersedes() -> None:
    """Observable: top replacement/correction, switch on, second distinct:
    auto-supersede of the top, as P8."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("replacement", reason_class="correction"),
        top_candidate=_fact(),
        other_verdicts=[_verdict("distinct")],
        other_candidates=[_fact()],
        switches=_ALL_ON,
    )
    assert result == Supersede(reason="correction")


def test_clean_picture_top_withheld_propagates() -> None:
    """The top's own withheld reason propagates untouched — the primary candidate stays
    the top survivor, no other survivor is even consulted."""
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("p"),
        top_verdict=_verdict("restatement"),
        top_candidate=_fact(),
        other_verdicts=[_verdict("replacement", reason_class="correction")],
        other_candidates=[_fact()],
        switches=_ALL_OFF,
    )
    assert result == Withheld(reason="switch_off")


def test_clean_picture_upward_two_candidates_composes_moved() -> None:
    """Observable: upward, A in p, B in q, both restatement of an incoming global text,
    switch on: one Moved set naming both, survivor rank order."""
    top = _fact(scope="p")
    second = _fact(scope="q")
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("global"),
        top_verdict=_verdict("restatement"),
        top_candidate=top,
        other_verdicts=[_verdict("restatement")],
        other_candidates=[second],
        switches=_ALL_ON,
    )
    assert result == Moved(candidates=(top, second))


def test_clean_picture_upward_plus_same_scope_replacement_competes() -> None:
    """Observable: upward with A (p, restatement) plus C (global, replacement):
    not clean, competing_candidate."""
    top = _fact(scope="p")
    third = _fact(scope="global")
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("global"),
        top_verdict=_verdict("restatement"),
        top_candidate=top,
        other_verdicts=[_verdict("replacement", reason_class="changed")],
        other_candidates=[third],
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="competing_candidate")


def test_clean_picture_upward_cross_kind_restatement_competes() -> None:
    """Observable: a Decision restating an incoming Fact stays cross_kind_restatement and
    never joins the moved set; because it carries restatement, the picture is not clean."""
    top = _fact(scope="p")
    cross_kind = _decision(scope="q")
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("global"),
        top_verdict=_verdict("restatement"),
        top_candidate=top,
        other_verdicts=[_verdict("restatement")],
        other_candidates=[cross_kind],
        switches=_ALL_ON,
    )
    assert result == Withheld(reason="competing_candidate")


def test_clean_picture_upward_switch_off_withholds_on_top_reason() -> None:
    """Observable: the same [upward two-candidate] with the switch off: bounce,
    withheld=switch_off, primary candidate is the top survivor."""
    top = _fact(scope="p")
    second = _fact(scope="q")
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("global"),
        top_verdict=_verdict("restatement"),
        top_candidate=top,
        other_verdicts=[_verdict("restatement")],
        other_candidates=[second],
        switches=_ALL_OFF,
    )
    assert result == Withheld(reason="switch_off")


def test_clean_picture_upward_downward_second_reinforces_clean() -> None:
    """A downward/same-scope second survivor, distinct, does not disturb an upward
    Moved dispatch of the top alone."""
    top = _fact(scope="p")
    result = clean_picture(
        incoming_kind="fact",
        typed_scope=Scope("global"),
        top_verdict=_verdict("restatement"),
        top_candidate=top,
        other_verdicts=[_verdict("distinct")],
        other_candidates=[_fact(scope="global")],
        switches=_ALL_ON,
    )
    assert result == Moved(candidates=(top,))


# ---------------------------------------------------------------------------
# _WHEN_SUPERSEDE_MOVED rendering (D10/D15) and the no-verdict-in-wording check
# ---------------------------------------------------------------------------


def test_when_supersede_moved_single_scope_pinned() -> None:
    from tabularium.engine.exits import _render_when_supersede_moved

    assert _render_when_supersede_moved(["p"]) == (
        "the p row says what yours says, and the claim holds beyond p — "
        "yours becomes the one copy and that row is kept as moved"
    )


def test_when_supersede_moved_two_scopes_joined() -> None:
    from tabularium.engine.exits import _render_when_supersede_moved

    assert _render_when_supersede_moved(["q", "p"]) == (
        "the p and q rows say what yours says, and the claim holds beyond p and q — "
        "yours becomes the one copy and those rows are kept as moved"
    )


def test_when_supersede_moved_three_scopes_oxford_less() -> None:
    from tabularium.engine.exits import _render_when_supersede_moved

    assert _render_when_supersede_moved(["r", "p", "q"]) == (
        "the p, q and r rows say what yours says, and the claim holds beyond p, q and r — "
        "yours becomes the one copy and those rows are kept as moved"
    )


_NO_VERDICT_PATTERN = re.compile(r"\d|odds|confidence|restatement|replacement|distinct|follow_up")


@pytest.mark.parametrize("scopes", [["p"], ["p", "q"], ["r", "p", "q"]])
def test_supersede_moved_when_never_mentions_verdict_or_number(scopes: list[str]) -> None:
    """D2/D9: the new upward ``moved`` ``when`` wording — one scope and several — never
    mentions a number, ``odds``, ``confidence`` or a verdict label (D2's static-wording
    rule, unaffected by this piece's own D15 scope-plural rendering)."""
    from tabularium.engine.exits import _render_when_supersede_moved

    rendered = _render_when_supersede_moved(scopes)
    assert not _NO_VERDICT_PATTERN.search(rendered), rendered
