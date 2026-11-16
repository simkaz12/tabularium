"""Named lookups for the dispositions and feedback: one place for the ``Unknown id`` refusal."""

from __future__ import annotations

from tabularium.domain import Decision, Fact, HypomnemaId
from tabularium.engine._errors import ToolError
from tabularium.storage.types import StorageAdapter


def unknown_id(id: HypomnemaId) -> ToolError:
    """The refusal for an id that is unknown or buried, for callers that raise it ``from None``."""
    return ToolError(f"Unknown id: {id!s}")


async def get_record(adapter: StorageAdapter, id: HypomnemaId) -> Fact | Decision:
    """Read ``id`` through ``adapter.get_by_id``; a ``KeyError`` becomes ``unknown_id(id)``.

    Raises:
        ToolError: ``Unknown id: <id>``, with no ``__cause__``.
    """
    try:
        return await adapter.get_by_id(id)
    except KeyError:
        raise unknown_id(id) from None


async def get_fact(adapter: StorageAdapter, id: HypomnemaId) -> Fact:
    """Read ``id`` as ``get_record`` does and narrow it to a Fact.

    Meant for a re-read of an id already known to be a Fact.

    Raises:
        ToolError: ``Unknown id: <id>``, with no ``__cause__``.
        TypeError: the record is a Decision.
    """
    record = await get_record(adapter, id)
    if not isinstance(record, Fact):
        raise TypeError(f"id={id!s} is a Decision, not a Fact")

    return record
