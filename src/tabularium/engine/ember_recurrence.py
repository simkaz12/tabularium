"""The owner/agent recurrence ladder for a reinforced ember.

``facts._dispatch_match`` calls ``record_ember_recurrence`` when a save reinforces an ember. The
function lives apart from ``embers`` so that ``embers`` can import ``facts.save_internal`` at module
level while ``facts`` imports this module.
"""

from __future__ import annotations

from tabularium.config import EngineConfig
from tabularium.domain import Asserter, Fact, HypomnemaId
from tabularium.results import EscalationExit, EscalationSuggestion
from tabularium.storage.types import StorageAdapter


def _make_escalation_exits(ember_id: HypomnemaId) -> tuple[EscalationExit, EscalationExit]:
    """The codify and foreground exits, always returned as a pair.

    ``how`` carries the real ``ember_id`` so each call can be copied as written.
    """
    return (
        EscalationExit(
            action="codify",
            how=f'save(kind="decision", promoted_from=["{ember_id!s}"])',
            when=(
                "the ember records a choice: something was chosen and something "
                "else was live and rejected (SPEC §1.4's road-not-taken test)"
            ),
        ),
        EscalationExit(
            action="foreground",
            how=f'set_hot("{ember_id!s}", hot=True)',
            when=(
                "the ember is a standing rule that keeps being violated, rather "
                "than a choice that keeps being re-derived"
            ),
        ),
    )


async def record_ember_recurrence(
    *,
    record: Fact,
    asserted_by: Asserter,
    adapter: StorageAdapter,
    cfg: EngineConfig,
) -> EscalationSuggestion | None:
    """Bump the owner or agent recurrence count on a reinforced ember and detect a rung-2 crossing.

    Call after the generic ``reinforce``, only for ``record.lifecycle == "ember"``. The three
    counters are not projections, so they are persisted with ``adapter.update_recurrence``.

    ``asserted_by="agent"`` bumps ``agent_recurrence_count`` and never triggers rung 2.
    ``"owner"`` bumps ``owner_recurrence_count`` and is checked against the threshold.

    A crossing is the pre-increment count below ``escalation_baseline + escalation_threshold``
    and the post-increment count at or above it, so a suggestion is returned once per
    crossing. ``set_hot(id, hot=False)`` resets ``escalation_baseline`` so the ladder can
    re-arm.

    Returns the ``EscalationSuggestion`` on a crossing, else ``None``.
    """
    owner_count = record.owner_recurrence_count
    agent_count = record.agent_recurrence_count
    suggestion: EscalationSuggestion | None = None

    if asserted_by == "owner":
        prev = owner_count
        owner_count = prev + 1
        target = record.escalation_baseline + cfg.escalation_threshold
        if prev < target <= owner_count:
            suggestion = EscalationSuggestion(
                ember_id=record.id,
                owner_recurrence_count=owner_count,
                threshold=cfg.escalation_threshold,
                exits=_make_escalation_exits(record.id),
            )
    else:
        agent_count += 1

    await adapter.update_recurrence(
        record.id,
        owner_recurrence_count=owner_count,
        agent_recurrence_count=agent_count,
        escalation_baseline=record.escalation_baseline,
    )
    return suggestion
