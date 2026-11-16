"""``build_exits`` (ADR 0022) — the seven ``when`` strings, byte-for-byte, and
the exit-set arity rule.

All seven strings are owner-ratified: the original six
(``2026-08-06-bounce-envelope-exits.md``), plus a seventh, Fact-arm
``supersede`` (fact-supersession brief), ratified the same day with revised
wording that excludes the design-vs-sequencing and authorization-vs-
field-report failure shapes.

Arity: Decision save = 3 against a Decision or a promotable Fact candidate,
2 against a non-promotable (archived/shelved) Fact candidate (2026-08-07
amendment). Fact save = 3 against a Fact candidate that
``is_fact_supersedable`` (currently: every Fact candidate that reaches a
bounce, since an invalid one is reconciled past earlier in the dispatch
matrix — see ``engine/facts.py``'s ``_dispatch_match``), 2 against a
Decision candidate (a Fact cannot supersede a Decision).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tabularium.domain import Decision, Fact, Scope, new_id
from tabularium.engine.exits import build_exits
from tabularium.engine.filters import read_span

_NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _fact(**overrides: object) -> Fact:
    defaults: dict[str, object] = {
        "id": new_id(),
        "type": "fact",
        "content": "a candidate fact",
        "scope": Scope("global"),
        "lifecycle": "active",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return Fact(**defaults)  # type: ignore[arg-type]


def _decision(**overrides: object) -> Decision:
    defaults: dict[str, object] = {
        "id": new_id(),
        "type": "decision",
        "content": "a candidate decision",
        "scope": Scope("global"),
        "lifecycle": "in_force",
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    defaults.update(overrides)
    return Decision(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The six pinned ``when`` strings — byte-for-byte against the brief's table
# ---------------------------------------------------------------------------

_WHEN_REINFORCE_SAME_KIND = (
    "yours restates the candidate and adds nothing a reader would miss — no new "
    "subject, no new qualifier, no changed value"
)
_WHEN_REINFORCE_CROSS_KIND = (
    "the candidate already carries your claim whole; across kinds this is rare — "
    "a Fact recording what a Decision caused is usually distinct, not redundant"
)
_WHEN_CONFIRM_DISTINCT_SAME_KIND = (
    "both stand — they read alike because they share a subject, not because "
    "either repeats the other; the common case when several atoms come out of "
    "one design session"
)
_WHEN_CONFIRM_DISTINCT_CROSS_KIND = (
    "a Fact recording a consequence and the Decision that caused it are "
    "different records; the shared vocabulary is expected and is not redundancy"
)
_WHEN_SUPERSEDE = (
    "same question, different answer — the candidate's answer is the one yours "
    "replaces, not merely one yours sits beside"
)
_WHEN_PROMOTED_FROM = (
    "the candidate is the observation your decision rests on, and yours names "
    "what was chosen against what was rejected (SPEC §1.4's road-not-taken test)"
)
_WHEN_SUPERSEDE_FACT = (
    "the candidate was true and yours is what made it false — same claim at "
    "a later state, not a different aspect of it, and not a report that "
    "disputes it without replacing it"
)


def test_fact_save_fact_candidate_same_kind_when_strings() -> None:
    """A valid Fact candidate now gets THREE exits — see the arity tests
    below for the byte-for-byte string check on the third."""
    candidate = _fact()
    exits = build_exits(
        saved_kind="fact", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].when == _WHEN_REINFORCE_SAME_KIND
    assert by_action["confirm_distinct"].when == _WHEN_CONFIRM_DISTINCT_SAME_KIND
    assert by_action["supersede"].when == _WHEN_SUPERSEDE_FACT


def test_fact_save_decision_candidate_cross_kind_when_strings() -> None:
    candidate = _decision()
    exits = build_exits(
        saved_kind="fact", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 2
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].when == _WHEN_REINFORCE_CROSS_KIND
    assert by_action["confirm_distinct"].when == _WHEN_CONFIRM_DISTINCT_CROSS_KIND


def test_decision_save_decision_candidate_same_kind_when_strings() -> None:
    candidate = _decision()
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].when == _WHEN_REINFORCE_SAME_KIND
    assert by_action["confirm_distinct"].when == _WHEN_CONFIRM_DISTINCT_SAME_KIND
    assert by_action["supersede"].when == _WHEN_SUPERSEDE


def test_decision_save_fact_candidate_cross_kind_when_strings() -> None:
    candidate = _fact()
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].when == _WHEN_REINFORCE_CROSS_KIND
    assert by_action["confirm_distinct"].when == _WHEN_CONFIRM_DISTINCT_CROSS_KIND
    assert by_action["promoted_from"].when == _WHEN_PROMOTED_FROM


@pytest.mark.parametrize("lifecycle", ["active", "hot", "ember"])
def test_decision_save_fact_candidate_when_strings_hold_for_every_promotable_lifecycle(
    lifecycle: str,
) -> None:
    """The reinforce/confirm_distinct ``when`` strings do not vary by the
    Fact candidate's lifecycle — only whether the third exit is offered
    does (see the arity tests below)."""
    candidate = _fact(lifecycle=lifecycle)
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].when == _WHEN_REINFORCE_CROSS_KIND
    assert by_action["confirm_distinct"].when == _WHEN_CONFIRM_DISTINCT_CROSS_KIND
    assert by_action["promoted_from"].when == _WHEN_PROMOTED_FROM


# ---------------------------------------------------------------------------
# Arity — Fact save: 3 against a supersedable Fact candidate, 2 against a
# Decision candidate or a non-supersedable (invalid) Fact candidate.
# DecisionBounce arity is unchanged (2026-08-07 amendment) — see below.
# ---------------------------------------------------------------------------


def test_fact_save_arity_is_three_for_a_valid_fact_candidate() -> None:
    exits = build_exits(
        saved_kind="fact", candidate=_fact(), saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct", "supersede"}


def test_fact_save_arity_is_two_for_a_decision_candidate() -> None:
    """A Fact cannot supersede a Decision — the third exit is never offered
    regardless of the Decision candidate's own state."""
    exits = build_exits(
        saved_kind="fact", candidate=_decision(), saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 2
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct"}


@pytest.mark.parametrize(
    "validity,expected_actions",
    [
        ("valid", {"reinforce", "confirm_distinct", "supersede"}),
        ("invalid", {"reinforce", "confirm_distinct"}),
    ],
)
def test_fact_save_fact_candidate_offered_split_pinned_by_validity(
    validity: str, expected_actions: set[str]
) -> None:
    """Both Fact validities, pinned explicitly: ``is_fact_supersedable``
    depends only on ``validity``, never ``lifecycle`` — the opposite axis
    from the Decision-arm's ``promoted_from`` exit, which is lifecycle-gated
    (see the parametrized lifecycle test below, and
    ``test_decision_save_fact_candidate_offered_split_pinned_by_lifecycle``
    above for that exit's own lifecycle split)."""
    candidate = _fact(validity=validity)
    exits = build_exits(
        saved_kind="fact", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == len(expected_actions)
    assert {e.action for e in exits} == expected_actions


@pytest.mark.parametrize("lifecycle", ["active", "hot", "ember", "archived", "shelved"])
def test_fact_save_arity_holds_across_every_lifecycle_when_valid(lifecycle: str) -> None:
    """Unlike the Decision-arm's ``promoted_from`` exit, the Fact-arm
    ``supersede`` exit does not vary by lifecycle at all — only validity."""
    candidate = _fact(lifecycle=lifecycle, validity="valid")
    exits = build_exits(
        saved_kind="fact", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct", "supersede"}


@pytest.mark.parametrize("candidate_factory", [_fact, _decision])
def test_decision_save_arity_is_three_for_promotable_candidates(
    candidate_factory: object,
) -> None:
    """A Decision candidate is always promotable (supersede — only in-force
    Decisions ever reach ``build_exits``, filtered upstream in
    ``engine/decisions.py``); ``_fact()``'s default ``lifecycle="active"``
    is promotable too. The non-promotable Fact split is its own test below —
    2026-08-07 fix, ADR 0022 amendment."""
    candidate = candidate_factory()  # type: ignore[operator]
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == 3


@pytest.mark.parametrize(
    "lifecycle,expected_actions",
    [
        ("active", {"reinforce", "confirm_distinct", "promoted_from"}),
        ("hot", {"reinforce", "confirm_distinct", "promoted_from"}),
        ("ember", {"reinforce", "confirm_distinct", "promoted_from"}),
        ("archived", {"reinforce", "confirm_distinct"}),
        ("shelved", {"reinforce", "confirm_distinct"}),
    ],
)
def test_decision_save_fact_candidate_offered_split_pinned_by_lifecycle(
    lifecycle: str, expected_actions: set[str]
) -> None:
    """All five Fact lifecycles, pinned explicitly: the offered/suppressed
    split for the ``promoted_from`` exit (2026-08-07 fix, ADR 0022
    amendment). ``archived``/``shelved`` are not promotable — the third
    exit is suppressed rather than offered and then refused by the engine
    guard (the live defect this fixes)."""
    candidate = _fact(lifecycle=lifecycle)
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert len(exits) == len(expected_actions)
    assert {e.action for e in exits} == expected_actions


def test_decision_save_third_exit_is_supersede_for_decision_candidate() -> None:
    exits = build_exits(
        saved_kind="decision", candidate=_decision(), saved_scope=Scope("global"),
        bounce_id="B",
    )
    actions = {e.action for e in exits}
    assert actions == {"reinforce", "confirm_distinct", "supersede"}


def test_decision_save_third_exit_is_promoted_from_for_fact_candidate() -> None:
    exits = build_exits(
        saved_kind="decision", candidate=_fact(), saved_scope=Scope("global"),
        bounce_id="B",
    )
    actions = {e.action for e in exits}
    assert actions == {"reinforce", "confirm_distinct", "promoted_from"}


# ---------------------------------------------------------------------------
# ``how`` templates — quoted ids, exact call shape (brief's pinned table)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ``promoted_from`` span guard — a
# promotable Fact candidate outside ``read_span(saved_scope)`` must not
# get the third exit, even though the WIDER admission probe span (D6) is
# what surfaced it as a candidate in the first place.
# ---------------------------------------------------------------------------


def test_decision_save_promoted_from_suppressed_for_out_of_span_fact_candidate() -> None:
    """A global save's own read span is ``["global"]`` alone (SPEC §2) — a
    promotable Fact from an unrelated named scope can reach ``build_exits``
    as a P7 probe-span candidate (D6 widened the PROBE, not the read span),
    but ``save_decision``'s own ``promoted_from`` validation would refuse
    it; the exit must not be offered."""
    candidate = _fact(scope=Scope("other-project"))
    assert candidate.scope not in read_span(Scope("global"))
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct"}


def test_decision_save_promoted_from_offered_for_in_span_fact_candidate() -> None:
    same_scope = _fact(scope=Scope("p"))
    exits = build_exits(
        saved_kind="decision", candidate=same_scope, saved_scope=Scope("p"),
        bounce_id="B",
    )
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct", "promoted_from"}

    global_candidate = _fact(scope=Scope("global"))
    exits = build_exits(
        saved_kind="decision", candidate=global_candidate, saved_scope=Scope("p"),
        bounce_id="B",
    )
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct", "promoted_from"}


def test_decision_save_promoted_from_suppressed_for_sibling_scope_fact_candidate() -> None:
    """A named-scope save's own read span is ``[scope, global]`` — a
    promotable Fact from a DIFFERENT named scope is out of span too."""
    candidate = _fact(scope=Scope("q"))
    assert candidate.scope not in read_span(Scope("p"))
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("p"),
        bounce_id="B",
    )
    assert {e.action for e in exits} == {"reinforce", "confirm_distinct"}


# ---------------------------------------------------------------------------
# ``supersede`` (Decision candidate) lifecycle guard — the outage path's ``bm25_probe`` has no
# retirement
# filter, so a superseded/revoked Decision can reach ``build_exits`` as a
# candidate; ``save_decision``'s own ``supersedes`` validation only accepts
# an ``in_force`` predecessor.
# ---------------------------------------------------------------------------


def test_decision_save_supersede_offered_only_for_in_force_decision_candidate() -> None:
    in_force = _decision(lifecycle="in_force")
    superseded = _decision(lifecycle="superseded", superseded_by=new_id())
    revoked = _decision(lifecycle="revoked")

    exits_in_force = build_exits(
        saved_kind="decision", candidate=in_force, saved_scope=Scope("global"), bounce_id="B"
    )
    exits_superseded = build_exits(
        saved_kind="decision", candidate=superseded, saved_scope=Scope("global"), bounce_id="B"
    )
    exits_revoked = build_exits(
        saved_kind="decision", candidate=revoked, saved_scope=Scope("global"), bounce_id="B"
    )

    assert {e.action for e in exits_in_force} == {"reinforce", "confirm_distinct", "supersede"}
    assert {e.action for e in exits_superseded} == {"reinforce", "confirm_distinct"}
    assert {e.action for e in exits_revoked} == {"reinforce", "confirm_distinct"}


def test_how_templates_match_pinned_table() -> None:
    candidate = _decision()
    cid = candidate.id
    exits = build_exits(
        saved_kind="decision", candidate=candidate, saved_scope=Scope("global"), bounce_id="B"
    )
    by_action = {e.action: e for e in exits}
    assert by_action["reinforce"].how == (
        f'save(resolve={{"candidate_id": "{cid!s}", "action": "reinforce", '
        f'"bounce_id": "B"}})'
    )
    assert by_action["confirm_distinct"].how == (
        f'save(resolve={{"candidate_id": "{cid!s}", "action": "confirm_distinct", '
        f'"bounce_id": "B"}})'
    )
    assert by_action["supersede"].how == (
        f'save(kind="decision", content=…, supersedes="{cid!s}", reason="changed", '
        f'bounce_id="B")'
    )

    fact_candidate = _fact()
    fcid = fact_candidate.id
    fact_exits = build_exits(
        saved_kind="decision", candidate=fact_candidate, saved_scope=Scope("global"),
        bounce_id="B",
    )
    fact_by_action = {e.action: e for e in fact_exits}
    assert fact_by_action["promoted_from"].how == (
        f'save(kind="decision", content=…, promoted_from=["{fcid!s}"], bounce_id="B")'
    )

    fact_save_exits = build_exits(
        saved_kind="fact", candidate=fact_candidate, saved_scope=Scope("global"), bounce_id="B",
    )
    fact_save_by_action = {e.action: e for e in fact_save_exits}
    assert fact_save_by_action["supersede"].how == (
        f'save(kind="fact", content=…, supersedes=["{fcid!s}"], reason="changed", '
        f'bounce_id="B")'
    )


# ---------------------------------------------------------------------------
# build_upward_exits (A4 bounce-judged piece, decisions D3/D9/D10/D15)
# ---------------------------------------------------------------------------


def test_build_upward_exits_single_candidate_two_exits_supersede_first() -> None:
    from tabularium.engine.exits import build_upward_exits

    candidate = _fact(scope=Scope("p"))
    exits = build_upward_exits(saved_kind="fact", candidates=[candidate], bounce_id="B")
    assert len(exits) == 2
    assert exits[0].action == "supersede"
    assert exits[1].action == "confirm_distinct"
    assert exits[0].how == (
        f'save(kind="fact", content=…, scope="global", supersedes=["{candidate.id!s}"], '
        f'reason="moved", bounce_id="B")'
    )


def test_build_upward_exits_exact_key_single_exit() -> None:
    from tabularium.engine.exits import build_upward_exits

    candidate = _fact(scope=Scope("p"))
    exits = build_upward_exits(
        saved_kind="fact", candidates=[candidate], bounce_id="B", exact_key=True
    )
    assert len(exits) == 1
    assert exits[0].action == "supersede"


def test_build_upward_exits_two_candidates_names_both_in_order() -> None:
    from tabularium.engine.exits import build_upward_exits

    first = _fact(scope=Scope("p"))
    second = _fact(scope=Scope("q"))
    exits = build_upward_exits(saved_kind="fact", candidates=[first, second], bounce_id="B")
    assert exits[0].how == (
        f'save(kind="fact", content=…, scope="global", '
        f'supersedes=["{first.id!s}", "{second.id!s}"], reason="moved", bounce_id="B")'
    )


def test_build_upward_exits_when_reflects_single_and_multi_scope() -> None:
    from tabularium.engine.exits import build_upward_exits

    one = build_upward_exits(
        saved_kind="fact", candidates=[_fact(scope=Scope("p"))], bounce_id="B"
    )
    assert "the p row says" in one[0].when

    two = build_upward_exits(
        saved_kind="fact",
        candidates=[_fact(scope=Scope("p")), _fact(scope=Scope("q"))],
        bounce_id="B",
    )
    assert "the p and q rows say" in two[0].when


def test_build_upward_exits_decision_single_candidate_singular_how() -> None:
    from tabularium.engine.exits import build_upward_exits

    candidate = _decision(scope=Scope("p"))
    exits = build_upward_exits(saved_kind="decision", candidates=[candidate], bounce_id="B")
    assert exits[0].how == (
        f'save(kind="decision", content=…, scope="global", supersedes="{candidate.id!s}", '
        f'reason="moved", bounce_id="B")'
    )
