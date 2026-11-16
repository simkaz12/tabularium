"""Exit builder for bounce envelopes. Pure module, no I/O.

``make_bounce`` call sites in ``facts`` and ``decisions`` build their exits with
``build_exits`` or ``build_upward_exits``; the embedder-down ``judge_unavailable`` bounce
builds its two exits with ``judge_unavailable_exits``. All of them build each ``resolve``
exit's ``how`` with ``how_resolve``. The ``_WHEN_*`` strings are wire text, pinned verbatim.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from tabularium.domain import Decision, Fact, HypomnemaId, Kind, Scope
from tabularium.engine.filters import read_span
from tabularium.results import BounceExit

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

# The multi-scope ``moved`` variant is a separate plural string rather than a shared
# template, so the singular stays pinned and each carries its own verb agreement.
_WHEN_SUPERSEDE_MOVED = (
    "the {scope} row says what yours says, and the claim holds beyond "
    "{scope} — yours becomes the one copy and that row is kept as moved"
)
_WHEN_SUPERSEDE_MOVED_PLURAL = (
    "the {scopes} rows say what yours says, and the claim holds beyond "
    "{scopes} — yours becomes the one copy and those rows are kept as moved"
)


def _join_scopes(scopes: Sequence[str]) -> str:
    """Sort the distinct scopes and join them with commas and a final ``" and "``.

    ``["p"] -> "p"``, ``["p", "q"] -> "p and q"``, ``["q", "p", "r"] -> "p, q and r"``.
    """
    ordered = sorted(set(scopes))
    if len(ordered) == 1:
        return ordered[0]
    return ", ".join(ordered[:-1]) + " and " + ordered[-1]


def _render_when_supersede_moved(scopes: Sequence[str]) -> str:
    """One distinct scope renders the singular string; several render the plural one."""
    joined = _join_scopes(scopes)
    if len(set(scopes)) == 1:
        return _WHEN_SUPERSEDE_MOVED.format(scope=joined)
    return _WHEN_SUPERSEDE_MOVED_PLURAL.format(scopes=joined)


# Offer-time subset of ``dispositions.supersede``'s legality, shared with it so the two cannot
# drift. At offer time only validity varies per candidate, and an already-superseded Fact is
# always invalid (``Fact`` validator), so validity alone covers "not already superseded".
def is_fact_supersedable(fact: Fact) -> bool:
    """Whether ``fact`` may be named as the predecessor of a Fact-arm supersede."""
    return fact.validity == "valid"



# Shared with ``decisions.save_decision``'s ``promoted_from`` check so offer and guard agree.
# Promotion only adds a Decision and never mutates the target Fact, so any live Fact
# qualifies: ``archived`` and ``shelved`` do not, ``active``, ``hot`` and ``ember`` do.
_PROMOTABLE_FACT_LIFECYCLES: frozenset[str] = frozenset({"active", "hot", "ember"})


def is_promotable(fact: Fact) -> bool:
    """Whether ``fact`` may legally be named as a ``promoted_from`` target."""
    return fact.lifecycle in _PROMOTABLE_FACT_LIFECYCLES


def how_resolve(
    action: Literal["reinforce", "confirm_distinct"], cid: HypomnemaId, bounce_id: str
) -> str:
    """The ``save(resolve={...})`` call text that replays ``action`` against candidate ``cid``."""
    return (
        f'save(resolve={{"candidate_id": "{cid!s}", "action": "{action}", '
        f'"bounce_id": "{bounce_id}"}})'
    )


def build_exits(
    *,
    saved_kind: Kind,
    candidate: Fact | Decision,
    saved_scope: Scope,
    bounce_id: str,
) -> tuple[BounceExit, ...]:
    """Build the named exits for a bounce.

    ``saved_kind`` is the kind being saved, not the candidate's. It picks the same-kind or
    cross-kind ``when`` strings. Every ``how`` carries ``bounce_id``, so replaying it stamps
    the verdict row the bounce wrote.

    ``saved_scope`` picks the ``supersede`` reason: ``changed`` when the candidate shares
    that scope, otherwise ``moved``.

    - Fact save: ``(reinforce, confirm_distinct)``, plus ``supersede`` when the candidate is a
      Fact and ``is_fact_supersedable``. A Decision candidate gets no third exit.
    - Decision save, Decision candidate: plus ``supersede``, but only when the candidate is
      ``in_force``. The BM25 outage probe applies no retired-Decision filter, and
      ``save_decision`` refuses to supersede a retired one.
    - Decision save, Fact candidate: plus ``promoted_from``, only when ``is_promotable`` and
      the candidate's scope is in ``read_span(saved_scope)``. A global save probes every
      registered scope, and ``save_decision`` rejects a ``promoted_from`` target outside
      that span.
    """
    cid = candidate.id
    same_kind = saved_kind == candidate.type
    supersede_reason = "changed" if candidate.scope == saved_scope else "moved"

    reinforce = BounceExit(
        action="reinforce",
        how=how_resolve("reinforce", cid, bounce_id),
        when=_WHEN_REINFORCE_SAME_KIND if same_kind else _WHEN_REINFORCE_CROSS_KIND,
    )
    confirm_distinct = BounceExit(
        action="confirm_distinct",
        how=how_resolve("confirm_distinct", cid, bounce_id),
        when=(
            _WHEN_CONFIRM_DISTINCT_SAME_KIND if same_kind else _WHEN_CONFIRM_DISTINCT_CROSS_KIND
        ),
    )

    if saved_kind == "fact":
        if isinstance(candidate, Fact) and is_fact_supersedable(candidate):
            fact_third = BounceExit(
                action="supersede",
                how=(
                    f'save(kind="fact", content=…, supersedes=["{cid!s}"], '
                    f'reason="{supersede_reason}", bounce_id="{bounce_id}")'
                ),
                when=_WHEN_SUPERSEDE_FACT,
            )
            return (reinforce, confirm_distinct, fact_third)
        return (reinforce, confirm_distinct)

    if isinstance(candidate, Decision):
        if candidate.lifecycle != "in_force":
            return (reinforce, confirm_distinct)
        third = BounceExit(
            action="supersede",
            how=(
                f'save(kind="decision", content=…, supersedes="{cid!s}", '
                f'reason="{supersede_reason}", bounce_id="{bounce_id}")'
            ),
            when=_WHEN_SUPERSEDE,
        )
        return (reinforce, confirm_distinct, third)

    if not is_promotable(candidate) or candidate.scope not in read_span(saved_scope):
        return (reinforce, confirm_distinct)

    third = BounceExit(
        action="promoted_from",
        how=(
            f'save(kind="decision", content=…, promoted_from=["{cid!s}"], '
            f'bounce_id="{bounce_id}")'
        ),
        when=_WHEN_PROMOTED_FROM,
    )
    return (reinforce, confirm_distinct, third)



def build_upward_exits(
    *,
    saved_kind: Kind,
    candidates: Sequence[Fact | Decision],
    bounce_id: str,
    exact_key: bool = False,
) -> tuple[BounceExit, ...]:
    """The upward-promotion exit set: one ``supersede`` exit with ``reason="moved"``.

    It names every candidate id in survivor rank order and is always returned first. With
    ``exact_key=True`` (byte-identical upward path, which has exactly one candidate) it is the
    only exit; otherwise ``confirm_distinct`` follows.

    ``when`` is ``_WHEN_SUPERSEDE_MOVED`` rendered over the candidates' scopes. ``how`` names
    ids as a list on the Fact arm. The Decision arm's ``save`` takes one id and
    ``clean_picture`` caps its ``Moved`` set at one candidate, so it always gets the singular
    literal.
    """
    assert candidates, "build_upward_exits: at least one candidate is required"
    assert saved_kind == "fact" or len(candidates) == 1, (
        "build_upward_exits: the Decision arm names exactly one candidate "
        "(clean_picture caps its Moved set at one row there)"
    )
    ids = [c.id for c in candidates]
    scopes = [str(c.scope) for c in candidates]
    when = _render_when_supersede_moved(scopes)
    if saved_kind == "fact" or len(ids) > 1:
        supersedes_literal = "[" + ", ".join(f'"{cid!s}"' for cid in ids) + "]"
    else:
        supersedes_literal = f'"{ids[0]!s}"'
    supersede = BounceExit(
        action="supersede",
        how=(
            f'save(kind="{saved_kind}", content=…, scope="global", '
            f'supersedes={supersedes_literal}, reason="moved", bounce_id="{bounce_id}")'
        ),
        when=when,
    )
    if exact_key:
        return (supersede,)
    confirm_distinct = BounceExit(
        action="confirm_distinct",
        how=how_resolve("confirm_distinct", ids[0], bounce_id),
        when=_WHEN_CONFIRM_DISTINCT_SAME_KIND,
    )
    return (supersede, confirm_distinct)


def judge_unavailable_exits(
    candidate_id: HypomnemaId, bounce_id: str
) -> tuple[BounceExit, BounceExit]:
    """The two exits offered when the embedder and the Assessor are both unavailable.

    Returned in the order ``confirm_distinct``, ``reinforce``. They carry no ``supersede`` or
    ``promoted_from``: with no judge, the engine can say only whether the content restates the
    candidate or is distinct from it.
    """
    return (
        BounceExit(
            action="confirm_distinct",
            how=how_resolve("confirm_distinct", candidate_id, bounce_id),
            when="the Assessor is unavailable; judge for yourself whether this is distinct",
        ),
        BounceExit(
            action="reinforce",
            how=how_resolve("reinforce", candidate_id, bounce_id),
            when="the Assessor is unavailable; judge for yourself whether this restates it",
        ),
    )
