"""The upsert gate both adapters call before they write over an existing row."""

from __future__ import annotations

from tabularium.domain import Decision, Fact


def check_upsert_allowed(existing: Fact | Decision, incoming: Fact | Decision) -> None:
    """Raise ``ValueError`` when ``incoming`` changes a field that ``upsert`` may not change.

    ``existing`` is the stored row with the same id. The adapter checks the vector
    dimension and the buried id before it reads ``existing``; this function covers
    the rest. When several fields differ, the first violation in this order is
    reported: Decision content, lifecycle, validity, ``recall_count``,
    ``last_referenced_at``, then the Fact fields (``feedback_score``, ``decay_due_at``,
    the recurrence counters, ``superseded_by``) or the Decision fields
    (``superseded_by``, ``supersession_chain``, ``promoted_from``).
    """
    if isinstance(existing, Decision) and existing.content != incoming.content:
        raise ValueError(
            f"Decision content is immutable (id={incoming.id!s}): "
            "existing content differs from new content"
        )
    if existing.lifecycle != incoming.lifecycle:
        raise ValueError(
            f"Cannot change lifecycle via upsert (id={incoming.id!s}): "
            f"existing={existing.lifecycle!r}, new={incoming.lifecycle!r}; "
            "use transition() — it is the ONLY lifecycle-change path"
        )
    if existing.validity != incoming.validity:
        raise ValueError(
            f"Cannot change validity via upsert (id={incoming.id!s}): "
            f"existing={existing.validity!r}, new={incoming.validity!r}; "
            "use set_validity() so the audit list is populated"
        )
    if existing.recall_count != incoming.recall_count:
        raise ValueError(
            f"Cannot change recall_count via upsert (id={incoming.id!s}); "
            "use update_projections() [A3]"
        )
    if existing.last_referenced_at != incoming.last_referenced_at:
        raise ValueError(
            f"Cannot change last_referenced_at via upsert (id={incoming.id!s}); "
            "use update_projections() [A3]"
        )

    if isinstance(existing, Fact) and isinstance(incoming, Fact):
        _check_fact_fields(existing, incoming)
    if isinstance(existing, Decision) and isinstance(incoming, Decision):
        _check_decision_lineage(existing, incoming)


def _check_fact_fields(existing: Fact, incoming: Fact) -> None:
    if existing.feedback_score != incoming.feedback_score:
        raise ValueError(
            f"Cannot change feedback_score via upsert (id={incoming.id!s}); "
            "use update_projections() [A3]"
        )
    if existing.decay_due_at != incoming.decay_due_at:
        raise ValueError(
            f"Cannot change decay_due_at via upsert (id={incoming.id!s}); "
            "use update_projections() [A3]"
        )
    if existing.owner_recurrence_count != incoming.owner_recurrence_count:
        raise ValueError(
            f"Cannot change owner_recurrence_count via upsert (id={incoming.id!s}); "
            "use update_recurrence() (SPEC §8.4)"
        )
    if existing.agent_recurrence_count != incoming.agent_recurrence_count:
        raise ValueError(
            f"Cannot change agent_recurrence_count via upsert (id={incoming.id!s}); "
            "use update_recurrence() (SPEC §8.4)"
        )
    if existing.escalation_baseline != incoming.escalation_baseline:
        raise ValueError(
            f"Cannot change escalation_baseline via upsert (id={incoming.id!s}); "
            "use update_recurrence() (SPEC §8.4)"
        )
    if existing.superseded_by != incoming.superseded_by:
        raise ValueError(
            f"Fact superseded_by must not change via upsert "
            f"(id={incoming.id!s}); use transition()"
        )


def _check_decision_lineage(existing: Decision, incoming: Decision) -> None:
    if existing.superseded_by != incoming.superseded_by:
        raise ValueError(
            f"Decision superseded_by must not change via upsert "
            f"(id={incoming.id!s}); use transition()"
        )
    if existing.supersession_chain != incoming.supersession_chain:
        raise ValueError(
            f"Decision supersession_chain is birth-frozen and must "
            f"not change via upsert (id={incoming.id!s})"
        )
    if existing.promoted_from != incoming.promoted_from:
        raise ValueError(
            f"Decision promoted_from is birth-frozen and must "
            f"not change via upsert (id={incoming.id!s})"
        )
