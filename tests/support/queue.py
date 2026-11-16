from __future__ import annotations

from tabularium.domain import HypomnemaId, Scope
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import RecordJudgment, RecordReview


def contested_ids(adapter: MemoryAdapter, scope: Scope) -> set[HypomnemaId]:
    latest: dict[HypomnemaId, RecordReview] = {}
    for review in adapter._record_reviews:
        if review.scope != scope:
            continue

        current = latest.get(review.target_id)
        if current is None or review.reviewed_at > current.reviewed_at:
            latest[review.target_id] = review

    return {
        target_id
        for target_id, review in latest.items()
        if review.band == "contested" and target_id in adapter._rows
    }


def latest_judgment(adapter: MemoryAdapter, target_id: HypomnemaId) -> RecordJudgment | None:
    latest: RecordJudgment | None = None
    for judgment in adapter._record_judgments:
        if judgment.target_id != target_id:
            continue

        if latest is None or judgment.judged_at > latest.judged_at:
            latest = judgment

    return latest
