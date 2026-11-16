"""LanceDB born-correct indexing: a store opened from source has all three
scalar/text indexes present, even before a single row is written.

FTS on ``content`` was already built at ``open()``; this locks in the two
that were missing — BTree on ``id``, Bitmap on ``scope`` — and proves both
work against an empty table, which is the case that matters: it is what
makes a store built from source *born correct* rather than born pathological
(SPEC §3.3, ADR 0017).

The bottom section (2026-08-02 sessions plan) covers the eighth
table, ``sessions`` — the first non-``records`` table to carry a full-text
index, and the first with no vector column at all.
"""

from __future__ import annotations

import pathlib

from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import SESSIONS_SCHEMA
from tests.support.lance_raw import raw_table


def _index_columns(indices: list[object]) -> set[str]:
    """Flatten ``IndexConfig.columns`` (each a list) into a set of column names."""
    covered: set[str] = set()
    for idx in indices:
        covered.update(idx.columns)  # type: ignore[attr-defined]
    return covered


async def test_fresh_empty_store_has_all_three_indexes(tmp_path: pathlib.Path) -> None:
    """A store opened on an empty table already carries all three indexes."""
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "fresh_store"),
        embedding_model="test-model",
        dim=4,
    )

    # The empty-table case is the one that matters — assert it explicitly so
    # this test cannot start passing only because rows happened to exist.
    assert await raw_table(adapter._records).count_rows() == 0

    indices = await raw_table(adapter._records).list_indices()
    assert len(indices) == 3
    assert _index_columns(indices) == {"content", "id", "scope"}


async def test_reopen_is_idempotent_and_still_reports_three(
    tmp_path: pathlib.Path,
) -> None:
    """Reopening an existing store must not raise and must not duplicate indexes."""
    path = str(tmp_path / "reopen_store")
    await LanceDBAdapter.open(path, embedding_model="test-model", dim=4)

    adapter = await LanceDBAdapter.open(path, embedding_model="test-model", dim=4)

    indices = await raw_table(adapter._records).list_indices()
    assert len(indices) == 3
    assert _index_columns(indices) == {"content", "id", "scope"}


# ---------------------------------------------------------------------------
# Sessions table (2026-08-02 sessions plan) — the eighth table, and
# the first non-``records`` table to carry a full-text index.
# ---------------------------------------------------------------------------


def test_sessions_schema_field_names_prove_no_vector_column() -> None:
    """``SESSIONS_SCHEMA`` carries exactly the six declared fields — no
    ``vector`` column, not even a nullable one (unlike ``records``, this
    table is never embedded)."""
    assert SESSIONS_SCHEMA.names == [
        "id",
        "scope",
        "started_at",
        "ended_at",
        "content",
        "source",
    ]
    assert "vector" not in SESSIONS_SCHEMA.names


async def test_sessions_table_has_fts_and_btree_indexes(tmp_path: pathlib.Path) -> None:
    """A store opened on an empty ``sessions`` table already carries FTS on
    ``content`` and BTree on ``ended_at`` — read from the table's own index
    list, not inferred from a successful query."""
    adapter = await LanceDBAdapter.open(
        str(tmp_path / "sessions_fresh_store"),
        embedding_model="test-model",
        dim=4,
    )

    assert await raw_table(adapter._sessions).count_rows() == 0

    indices = await raw_table(adapter._sessions).list_indices()
    assert len(indices) == 2
    assert _index_columns(indices) == {"content", "ended_at"}

    by_column = {idx.columns[0]: idx for idx in indices}
    assert by_column["content"].index_type == "FTS"
    assert by_column["ended_at"].index_type == "BTree"
