"""Compounding review intervals: a pure function of one record's own prior ``RecordReview`` history.

Callers fetch the history with ``adapter.list_reviews_for`` immediately before each write.
Consecutive unanimous keeps earn a longer gap and a contested review shortens it, using rubric
parameters, never a hint in a prompt. ``list_reviews_for`` is a single-id lookback. It does not
enumerate a scope or compute due-ness, so it cannot rebuild the rotation pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from tabularium.engine.rubric import RubricParameters
from tabularium.storage.types import RecordReview


def next_review_due_at(
    now: datetime, *, history: Sequence[RecordReview], params: RubricParameters
) -> datetime:
    """A record's next revisit due date from its own prior ``RecordReview`` history, oldest first.

    The base interval is ``revisit_interval_days``. It is multiplied by
    ``revisit_stable_multiplier`` once per consecutive ``keep`` review counted back from the most
    recent. It is multiplied by ``revisit_contested_multiplier`` when the most recent review was
    contested; that cannot combine with a keep streak, since a streak means the latest review was a
    keep. The result is clamped to ``revisit_max_interval_days``. Empty ``history`` gives the base
    interval.
    """
    streak = 0
    for review in reversed(history):
        if review.band != "keep":
            break
        streak += 1

    interval_days = params.revisit_interval_days * (params.revisit_stable_multiplier**streak)
    if history and history[-1].band == "contested":
        interval_days *= params.revisit_contested_multiplier
    interval_days = min(interval_days, params.revisit_max_interval_days)

    return now + timedelta(days=interval_days)
