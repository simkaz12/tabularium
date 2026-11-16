"""Pure rules behind ``list_due_for_review``, shared by both adapters."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import NamedTuple

from tabularium.domain import Fact, HypomnemaId, Scope
from tabularium.storage.types import RecordReview

REVISIT_POOL_LIFECYCLES: tuple[str, ...] = ("active", "archived")


class PoolCandidate[T](NamedTuple):
    """One pool row: id and creation time for the due check and sort, plus the adapter's payload."""

    id: HypomnemaId
    created_at: datetime
    payload: T


def latest_review_per_target(reviews: Iterable[RecordReview]) -> dict[HypomnemaId, RecordReview]:
    """Keep the review with the greatest ``reviewed_at`` per ``target_id``.

    The comparison is strict ``>``, so among equal ``reviewed_at`` values the first
    review in iteration order wins.
    """
    latest: dict[HypomnemaId, RecordReview] = {}
    for review in reviews:
        existing = latest.get(review.target_id)
        if existing is None or review.reviewed_at > existing.reviewed_at:
            latest[review.target_id] = review
    return latest


def in_revisit_pool(fact: Fact, *, scope: Scope, created_before: datetime) -> bool:
    """True for a valid, unsuperseded, live Fact of ``scope`` created by ``created_before``."""
    return (
        fact.scope == scope
        and fact.lifecycle in REVISIT_POOL_LIFECYCLES
        and fact.validity == "valid"
        and fact.superseded_by is None
        and fact.created_at <= created_before
    )


def order_due_candidates[T](
    candidates: Iterable[PoolCandidate[T]],
    latest: dict[HypomnemaId, RecordReview],
    *,
    now: datetime,
) -> list[T]:
    """Keep never-reviewed and due candidates, never-reviewed first, then oldest review first.

    A candidate is due when its latest review's ``next_review_due_at`` is at or
    before ``now``. Ties on the sort key keep the order of ``candidates``.
    """
    picked: list[tuple[datetime | None, PoolCandidate[T]]] = []
    for candidate in candidates:
        review = latest.get(candidate.id)
        if review is None:
            picked.append((None, candidate))
        elif review.next_review_due_at <= now:
            picked.append((review.reviewed_at, candidate))

    picked.sort(
        key=lambda pair: (
            pair[0] is not None,
            pair[0] or datetime.min.replace(tzinfo=UTC),
            pair[1].created_at,
        )
    )
    return [candidate.payload for _, candidate in picked]
