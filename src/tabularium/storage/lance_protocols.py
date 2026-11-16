"""Structural types for the LanceDB handles ``LanceDBAdapter`` holds.

LanceDB ships no ``py.typed`` marker, so mypy sees ``lancedb`` as untyped and cannot
check these Protocols against ``AsyncTable`` / ``AsyncConnection``. They are written
against the signatures in lancedb 0.34 and cover only the calls the adapter makes.
``stats()`` returns a plain ``dict`` at runtime, so it is a ``TypedDict``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable, Sequence
from datetime import timedelta
from typing import Any, Protocol, Self, TypedDict

import pyarrow as pa


class LanceFragmentStats(TypedDict):
    num_small_fragments: int


class LanceTableStats(TypedDict):
    num_rows: int
    fragment_stats: LanceFragmentStats


class LanceIndexConfig(Protocol):
    @property
    def num_unindexed_rows(self) -> int: ...


class LanceUpdateResult(Protocol):
    @property
    def rows_updated(self) -> int: ...


class LanceMergeResult(Protocol):
    @property
    def num_updated_rows(self) -> int: ...


class LanceMergeInsert(Protocol):
    def when_matched_update_all(self) -> Self: ...

    def when_not_matched_insert_all(self) -> Self: ...

    def when_not_matched_by_source_delete(self, condition: str) -> Self: ...

    def execute(self, new_data: pa.Table) -> Awaitable[LanceMergeResult]: ...


class LanceQuery(Protocol):
    """One query-builder surface for plain, vector and full-text queries.

    ``where`` takes a SQL string or a ``lancedb.expr`` expression, which is why its
    parameter is ``object``.
    """

    def where(self, predicate: object) -> Self: ...

    def select(self, columns: list[str]) -> Self: ...

    def limit(self, limit: int) -> Self: ...

    def order_by(self, ordering: list[Any]) -> Self: ...  # lancedb ColumnOrdering is untyped

    def nearest_to(self, query_vector: list[float]) -> Self: ...

    def nearest_to_text(self, query: str) -> Self: ...

    def distance_type(self, distance_type: str) -> Self: ...

    def bypass_vector_index(self) -> Self: ...

    async def to_list(self) -> list[dict[str, Any]]: ...  # rows are decoded by the codec


class LanceTable(Protocol):
    async def schema(self) -> pa.Schema: ...

    def query(self) -> LanceQuery: ...

    async def add(self, data: Sequence[dict[str, Any]]) -> object: ...  # codec-built row dicts

    async def delete(self, where: object) -> object: ...

    def merge_insert(self, on: str) -> LanceMergeInsert: ...

    async def update(
        self,
        updates: dict[str, Any],  # column name to a Python value
        *,
        where: str,
    ) -> LanceUpdateResult: ...

    async def add_columns(self, transforms: dict[str, str]) -> object: ...

    async def alter_columns(self, *alterations: dict[str, Any]) -> object: ...  # lancedb dict spec

    async def create_index(self, column: str, *, config: object, replace: bool) -> None: ...

    async def optimize(
        self,
        *,
        cleanup_older_than: timedelta,
        delete_unverified: bool = False,
    ) -> object: ...

    async def stats(self) -> LanceTableStats: ...

    async def list_indices(self) -> Iterable[LanceIndexConfig]: ...


class LanceTableListing(Protocol):
    @property
    def tables(self) -> list[str]: ...


class LanceConnection(Protocol):
    async def list_tables(self) -> LanceTableListing | Iterable[str]: ...

    async def open_table(self, name: str) -> LanceTable: ...

    async def create_table(self, name: str, *, schema: pa.Schema) -> LanceTable: ...
