"""Revisit selection and the history block: deterministic rules in code, with no I/O.

Selection (``select_revisit_pool``) is rules in code, not prompt instructions, so the panel sees
only records with mixed evidence. It applies, in order, to the pool ``adapter.list_due_for_review``
returned (Facts only, already filtered by scope, lifecycle and age):

1. ``scope == "global"`` outside a ``global`` pass, or a non-Fact: excluded. These are
   belt-and-braces over the caller's scope choice and ``list_due_for_review``'s own Facts-only
   filter.
2. Protected (``is_protected``): excluded from the panel, never re-shelved by it. The caller
   appends a ``protected`` review row.
3. A birth session is looked up. A record with no covering session is annotated ``provenance
   unavailable (no covering session)``, not excluded.
4. No evidence since birth: no later session that returns to the subject and no event with ``at >
   created_at``. The record is excluded and the caller appends a ``skipped_no_evidence`` review
   row. This keeps the birth digest from being the only signal on every judged record.

Protected: a Fact whose lifecycle is ``active`` with at least two net ``shelved``-source events.
Only an un-shelve lands a Fact back on ``active`` from ``shelved``, and each leg of a completed
shelve-then-unshelve round trip appends one such event. The threshold is two, not one, because
``set_shelved`` appends its audit event before the lifecycle transition, so one failed shelve
attempt leaves a single stray event. A ``shelve_reverted`` event (the Cleaner undoing its own failed
trim save) cancels one ``shelved`` event.

Later-session lookup: the caller prefetches the scope's session rows once per pass
(``adapter.search_sessions`` with no query, so no embedding and no LLM call) and one
``query_events`` result per pool record. ``select_revisit_pool`` then does no I/O.

- The birth session is the row whose ``[started_at, ended_at]`` covers the record's ``created_at``,
  a range-matching link. It is inferred, never stored.
- A later session is any other row starting strictly after the birth session's ``ended_at`` (or
  after ``created_at`` when there is no birth session) that shares at least two of the record's
  distinctive terms, or all of them if it has fewer than two (``session_mentions_subject``). The
  check is plain term overlap, not BM25, embeddings or an LLM, so co-occurrence in the scope alone
  does not count.

History block (``assemble_history_block``): a compact markdown block appended to a candidate's own
content before batching. It is not a field on the judge schema and never contains a record id or a
session id. The birth-session excerpt is labelled as the author's own account, written at the time:
admissible as the baseline of a comparison, not as an argument that the record matters because its
author said so. The ``[revisit].prose`` rubric section states the same rule for the judge. The judge
never sees a predecessor's verdict, only the session evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from tabularium.domain import (
    Decision,
    Fact,
    Hypomnema,
    HypomnemaId,
    Scope,
    Session,
)
from tabularium.engine.filters import GLOBAL
from tabularium.engine.rubric import RubricParameters
from tabularium.storage.types import ReinforcementEvent

# ---------------------------------------------------------------------------
# Selection: deterministic rules in code
# ---------------------------------------------------------------------------


class RevisitCandidate(BaseModel):
    """One record that survived selection, paired with the session evidence
    ``assemble_history_block`` needs. The caller renders it into a content string before it reaches
    a judge.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    record: Fact
    birth_session: Session | None
    later_sessions: tuple[Session, ...]


class RevisitPool(BaseModel):
    model_config = {"frozen": True}

    candidates: tuple[RevisitCandidate, ...]
    # Never re-shelved by the panel; the caller appends a `protected` review row.
    protected: tuple[HypomnemaId, ...]
    # No later session returned to the subject and no event since birth.
    skipped_no_evidence: tuple[HypomnemaId, ...]
    excluded_global: tuple[HypomnemaId, ...]
    excluded_non_fact: tuple[HypomnemaId, ...]


def is_protected(record: Fact, events: Sequence[ReinforcementEvent]) -> bool:
    """Whether ``record`` is protected from re-shelving: lifecycle ``active`` and at least two net
    ``shelved``-source events (``shelved`` minus ``shelve_reverted``).

    Only an un-shelve lands a Fact back on ``active``, and a completed round trip appends one
    ``shelved``-source event per leg, so a count is enough and no event ordering is inspected. The
    threshold is two because ``set_shelved`` appends its audit event before the lifecycle
    transition, so a single failed shelve on an ``active`` Fact leaves one stray event. Each
    ``shelve_reverted`` event cancels one ``shelved`` event, since the Cleaner undid its own shelve
    and that is not an owner action; repeated failed trims therefore never add up to a round trip.
    """
    if record.lifecycle != "active":
        return False
    shelved = sum(1 for event in events if event.source == "shelved")
    reverted = sum(1 for event in events if event.source == "shelve_reverted")
    return shelved - reverted >= 2


def find_birth_session(record: Fact, sessions: Sequence[Session]) -> Session | None:
    """The one session row whose range covers ``record.created_at``, or ``None`` when no row does (a
    record that predates the scope's first Session row). Matched by range, never by a stored link.
    """
    for session in sessions:
        if session.started_at <= record.created_at <= session.ended_at:
            return session
    return None


_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "along",
        "always",
        "and",
        "because",
        "been",
        "before",
        "being",
        "between",
        "could",
        "does",
        "doing",
        "during",
        "each",
        "either",
        "every",
        "from",
        "have",
        "having",
        "into",
        "itself",
        "never",
        "not",
        "only",
        "other",
        "over",
        "should",
        "since",
        "some",
        "still",
        "such",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
    }
)

_TERM_RE = re.compile(r"[a-zA-Z0-9_]+")


def _significant_terms(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens of length 5 or more, minus a small stopword list. Coarser than
    ``storage.memory``'s BM25 tokenizer on purpose, since this is a subject-overlap check and not a
    ranking function.
    """
    return frozenset(
        term
        for term in (m.group(0).lower() for m in _TERM_RE.finditer(text))
        if len(term) >= 5 and term not in _STOPWORDS
    )


def session_mentions_subject(record_content: str, session_content: str) -> bool:
    """Whether a session returns to the record's subject: it shares at least two of the record's
    significant terms, or all of them if the record has fewer than two. Deterministic term overlap,
    with no BM25, embedding or LLM call.
    """
    record_terms = _significant_terms(record_content)
    if not record_terms:
        return False
    shared = record_terms & _significant_terms(session_content)
    return len(shared) >= min(2, len(record_terms))


def find_later_sessions(
    record: Fact, birth_session: Session | None, sessions: Sequence[Session]
) -> tuple[Session, ...]:
    """Sessions starting strictly after the birth session's ``ended_at`` (or after
    ``record.created_at`` when there is none) whose content returns to the record's subject, ordered
    by ``ended_at`` ascending.
    """
    cutoff = birth_session.ended_at if birth_session is not None else record.created_at
    birth_id = birth_session.id if birth_session is not None else None
    later = [
        session
        for session in sessions
        if session.started_at > cutoff
        and session.id != birth_id
        and session_mentions_subject(record.content, session.content)
    ]
    later.sort(key=lambda session: session.ended_at)
    return tuple(later)


def select_revisit_pool(
    records: Sequence[Hypomnema],
    *,
    pass_scope: Scope | None = None,
    events_by_id: Mapping[HypomnemaId, Sequence[ReinforcementEvent]],
    sessions_by_scope: Mapping[Scope, Sequence[Session]],
    params: RubricParameters,
) -> RevisitPool:
    """Filter ``list_due_for_review``'s pool down to this axis's candidates. The module docstring
    has the rule order. Due-ness was decided inside ``list_due_for_review``.
    """
    candidates: list[RevisitCandidate] = []
    protected: list[HypomnemaId] = []
    skipped_no_evidence: list[HypomnemaId] = []
    excluded_global: list[HypomnemaId] = []
    excluded_non_fact: list[HypomnemaId] = []

    for record in records:
        if record.scope == GLOBAL and pass_scope != GLOBAL:
            excluded_global.append(record.id)
            continue
        if isinstance(record, Decision) or not isinstance(record, Fact):
            excluded_non_fact.append(record.id)
            continue

        events = events_by_id.get(record.id, ())

        if is_protected(record, events):
            protected.append(record.id)
            continue

        scope_sessions = sessions_by_scope.get(record.scope, ())
        birth_session = find_birth_session(record, scope_sessions)
        later_sessions = find_later_sessions(record, birth_session, scope_sessions)

        events_since_birth = tuple(event for event in events if event.at > record.created_at)
        if not later_sessions and not events_since_birth:
            skipped_no_evidence.append(record.id)
            continue

        candidates.append(
            RevisitCandidate(
                record=record, birth_session=birth_session, later_sessions=later_sessions
            )
        )

    return RevisitPool(
        candidates=tuple(candidates),
        protected=tuple(protected),
        skipped_no_evidence=tuple(skipped_no_evidence),
        excluded_global=tuple(excluded_global),
        excluded_non_fact=tuple(excluded_non_fact),
    )


# ---------------------------------------------------------------------------
# The history block: evidence in hand, never a record id
# ---------------------------------------------------------------------------

_NO_COVERING_SESSION = "provenance unavailable (no covering session)"


HISTORY_BUDGET_CHARS = 40_000
EXCERPT_CAP_CHARS = 8_000


def _clip_excerpt(content: str) -> str:
    """Keep the head of ``content`` up to ``EXCERPT_CAP_CHARS`` and mark the cut."""
    if len(content) <= EXCERPT_CAP_CHARS:
        return content
    dropped = len(content) - EXCERPT_CAP_CHARS
    return f"{content[:EXCERPT_CAP_CHARS]}\n[... clipped, {dropped} more chars]"


def assemble_history_block(
    *, birth_session: Session | None, later_sessions: Sequence[Session]
) -> str:
    """Render a compact markdown history block for one candidate, appended to the record's own
    content by the caller before batching. It contains no record id or session id.

    Fill order against the shared ``HISTORY_BUDGET_CHARS``: the birth digest clipped to
    ``EXCERPT_CAP_CHARS``; then later sessions, newest first, each clipped the same way, until the
    next would push the excerpt text past the budget; then one line counting the older sessions left
    out. The fixed header, label and omitted-count lines do not count against it.
    """
    lines = [
        "",
        "---",
        "",
        "### History (input to weigh -- not part of the record's own text; do not grade it)",
        "",
    ]
    used = 0
    if birth_session is not None:
        birth_text = _clip_excerpt(birth_session.content)
        used += len(birth_text)
        lines.append(
            "**Birth session** -- the author's own account, written at the time this record "
            "was saved:"
        )
        lines.append(birth_text)
    else:
        lines.append(f"**Birth session**: {_NO_COVERING_SESSION}")

    newest_first = sorted(later_sessions, key=lambda session: session.ended_at, reverse=True)
    included: list[str] = []
    for session in newest_first:
        text = _clip_excerpt(session.content)
        if used + len(text) > HISTORY_BUDGET_CHARS:
            break
        used += len(text)
        included.append(text)

    if included:
        lines.append("")
        lines.append("**Later sessions that returned to this subject:**")
        lines.extend(f"- {text}" for text in included)

    omitted = len(newest_first) - len(included)
    if omitted:
        lines.append("")
        lines.append(f"({omitted} older later session(s) omitted to fit the history budget)")

    return "\n".join(lines)
