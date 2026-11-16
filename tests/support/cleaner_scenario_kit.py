"""Helpers shared by the cleaner-golden scenario modules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal
from unittest import mock

from tabularium.domain import HypomnemaId
from tabularium.engine import cleaner_writes
from tabularium.engine.facts import save_internal as real_save_internal
from tabularium.results import SaveResult
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import RecordJudgment, RecordReview
from tests.support.cleaner_golden import Fault, ItemSpec, ScriptedJudge
from tests.support.cleaner_world import World

Panel = tuple[ScriptedJudge, ScriptedJudge]


def panel(
    a: Mapping[str, ItemSpec],
    b: Mapping[str, ItemSpec] | None = None,
    *,
    a_faults: tuple[Fault | None, ...] = (),
    b_faults: tuple[Fault | None, ...] = (),
    a_on_call: Callable[[int], Awaitable[None]] | None = None,
    fault_padding: str = "",
) -> Panel:
    return (
        ScriptedJudge(
            "judge-a", a, faults=a_faults, fault_padding=fault_padding, on_call=a_on_call
        ),
        ScriptedJudge(
            "judge-b", a if b is None else b, faults=b_faults, fault_padding=fault_padding
        ),
    )


@contextmanager
def failing_saves(marker: str, how: Literal["raise", "bounce"]) -> Iterator[None]:
    """Make the Cleaner's save fail for content carrying ``marker``; other saves pass through."""
    real = real_save_internal

    async def flaky(**kwargs: Any) -> SaveResult:  # forwards kwargs unchanged
        if marker in str(kwargs["content"]):
            if how == "raise":
                raise RuntimeError("scripted save failure")
            return SaveResult(outcome="bounced", reason="scripted bounce")
        result: SaveResult = await real(**kwargs)
        return result

    with mock.patch.object(cleaner_writes, "save_internal", flaky):
        yield


def hot_on_call(world: World, number: int, target: HypomnemaId) -> Callable[[int], Awaitable[None]]:
    """A judge hook that promotes ``target`` to hot while the ``number``-th call is answered."""

    async def hook(call: int) -> None:
        if call == number:
            await world.adapter.transition(target, "hot", at=world.clock.peek())

    return hook


@contextmanager
def flaky_appends(
    adapter: MemoryAdapter,
    *,
    review_when: Callable[[RecordReview], bool] = lambda review: False,
    judgment_when: Callable[[RecordJudgment], bool] = lambda judgment: False,
) -> Iterator[None]:
    """Make ``append_review`` and ``append_record_judgment`` raise for the rows picked."""
    real_review = adapter.append_review
    real_judgment = adapter.append_record_judgment

    async def flaky_review(review: RecordReview) -> None:
        if review_when(review):
            raise RuntimeError(f"review store down for {review.target_id} ({review.outcome})")
        await real_review(review)

    async def flaky_judgment(judgment: RecordJudgment) -> None:
        if judgment_when(judgment):
            raise RuntimeError(f"judgment store down for {judgment.target_id}")
        await real_judgment(judgment)

    with (
        mock.patch.object(adapter, "append_review", flaky_review),
        mock.patch.object(adapter, "append_record_judgment", flaky_judgment),
    ):
        yield


def invalidate_on_call(
    world: World, number: int, target: HypomnemaId
) -> Callable[[int], Awaitable[None]]:
    """A judge hook that marks ``target`` invalid while the ``number``-th call is answered."""

    async def hook(call: int) -> None:
        if call == number:
            await world.adapter.set_validity(target, False, "scripted invalidation")

    return hook
