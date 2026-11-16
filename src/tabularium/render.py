"""Decision render and title derivation.

``render_decision`` is a pure function of its two arguments: the Decision row and the caller's
revalidation records. It reads nothing else, so the output is byte-stable for equal inputs. The
``decision`` tool's ``DecisionView.rendered``, the ``decision://{id}`` resource and the CLI sync
export all call it.

``derive_title`` takes the first sentence of the content, capped at 80 characters, and
``make_supersedes_snapshot`` uses it for ``SupersedesSnapshot.title``.

The render omits ``recall_count``, ``last_referenced_at``, ``tip`` and ``purpose``; those are on
``DecisionView`` only.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime

from tabularium.domain import Decision, SupersedesSnapshot
from tabularium.results import RevalidationRecord

# ---------------------------------------------------------------------------
# Title derivation
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def derive_title(content: str) -> str:
    """Derive a display title from record content.

    Cut at the first ``.``, ``!`` or ``?`` followed by whitespace or end of string, or at the
    first newline, whichever comes first, and strip the result. Content with neither uses the
    whole string; content that strips to nothing falls back to the stripped content. A title
    longer than 80 characters becomes its first 79 plus ``…``.
    """
    sentence_match = _SENTENCE_END.search(content)
    sentence_end = sentence_match.end() if sentence_match else len(content)

    nl_pos = content.find("\n")
    newline_end = nl_pos if nl_pos >= 0 else len(content)

    cut = min(sentence_end, newline_end)
    title = content[:cut].strip()

    if not title:
        title = content.strip()

    if len(title) > 80:
        title = title[:79] + "…"

    return title


# ---------------------------------------------------------------------------
# Date formatting helpers
# ---------------------------------------------------------------------------


def _fmt_date(dt: datetime) -> str:
    """Format the date part of ``dt`` as YYYY-MM-DD, in ``dt``'s own timezone."""
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# render_decision
# ---------------------------------------------------------------------------


def render_decision(
    decision: Decision,
    *,
    revalidations: Sequence[RevalidationRecord] = (),
) -> str:
    """Render a Decision to a plain-text / Markdown string.

    Pure function of ``decision`` and ``revalidations``; it reads nothing else.

    Layout:
      Status line     - lifecycle, validity, "recorded YYYY-MM-DD", plus "superseded
                        YYYY-MM-DD" (lifecycle superseded with ``superseded_by`` set) or
                        "revoked YYYY-MM-DD" (lifecycle revoked), both dated from ``updated_at``
      Blank line
      Verbatim body   - ``decision.content``
      Blank line
      Lineage section - after a ``---`` rule, only when it has any entry:
                        the supersession chain, in stored order
                        + ``superseded_by`` id
                        + invalidation ``invalidated_by`` and reason, only while validity
                        is invalid and ``invalidated_by`` is set
                        + one "Revalidated" line per revalidation, in the order given
                        + ``promoted_from`` entries (valence, content, recurrence_count)
    """
    parts: list[str] = []

    recorded = _fmt_date(decision.created_at)
    status_tokens: list[str] = [
        decision.lifecycle,
        decision.validity,
        f"recorded {recorded}",
    ]
    if decision.lifecycle == "superseded" and decision.superseded_by is not None:
        superseded_date = _fmt_date(decision.updated_at)
        status_tokens.append(f"superseded {superseded_date}")
    elif decision.lifecycle == "revoked":
        revoked_date = _fmt_date(decision.updated_at)
        status_tokens.append(f"revoked {revoked_date}")

    parts.append(" · ".join(status_tokens))
    parts.append("")

    parts.append(decision.content)
    parts.append("")

    has_invalidation = decision.validity == "invalid" and decision.invalidated_by is not None
    has_lineage = (
        len(decision.supersession_chain) > 0
        or decision.superseded_by is not None
        or len(decision.promoted_from) > 0
        or has_invalidation
        or len(revalidations) > 0
    )

    if has_lineage:
        parts.append("---")
        parts.append("")

    if decision.supersession_chain:
        parts.append("**Supersedes:**")
        for snap in decision.supersession_chain:
            parts.append(f"- {snap.title} (recorded {_fmt_date(snap.created_at)})")
        parts.append("")

    if decision.superseded_by is not None:
        parts.append(f"**Superseded by:** `{decision.superseded_by!s}`")
        parts.append("")

    if has_invalidation:
        parts.append(
            f"Invalidated by: `{decision.invalidated_by}` ({decision.invalidation_reason})"
        )
        parts.append("")

    for rev in revalidations:
        parts.append(
            f"Revalidated {_fmt_date(rev.at)}: withdrew {rev.withdrawn_reason} "
            f"by `{rev.withdrawn_cause!s}` ({rev.reason})"
        )
        parts.append("")

    if decision.promoted_from:
        parts.append("**Promoted from:**")
        for pf in decision.promoted_from:
            # NOTE: valence is None for a promoted Fact that was never an ember; no label.
            prefix = f"{pf.valence.capitalize()} — " if pf.valence is not None else ""
            parts.append(
                f"- {prefix}{pf.content}"
                f" (recurrence: {pf.recurrence_count})"
            )
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# make_supersedes_snapshot
# ---------------------------------------------------------------------------


def make_supersedes_snapshot(predecessor: Decision) -> SupersedesSnapshot:
    """Snapshot the predecessor's id, derived title and ``created_at``."""
    return SupersedesSnapshot(
        id=predecessor.id,
        title=derive_title(predecessor.content),
        created_at=predecessor.created_at,
    )
