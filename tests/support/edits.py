from __future__ import annotations

import pyarrow as pa

from tabularium.domain import HypomnemaId
from tabularium.storage.lancedb_adapter import LanceDBAdapter
from tabularium.storage.lancedb_codec import record_to_row, row_to_record
from tabularium.storage.memory import MemoryAdapter
from tabularium.storage.types import StorageAdapter


async def edit_content(
    adapter: StorageAdapter,
    id: HypomnemaId,
    content: str,
    vector: list[float] | None,
) -> None:
    if isinstance(adapter, MemoryAdapter):
        _edit_memory(adapter, id, content, vector)
        return

    if isinstance(adapter, LanceDBAdapter):
        await _edit_lancedb(adapter, id, content, vector)
        return

    raise TypeError(f"cannot edit content through {type(adapter).__name__}")


def _edit_memory(
    adapter: MemoryAdapter,
    id: HypomnemaId,
    content: str,
    vector: list[float] | None,
) -> None:
    row = adapter._rows.get(id)
    if row is None:
        raise KeyError(f"Unknown id: {id!s}")

    row.record = type(row.record).model_validate(
        {
            **row.record.model_dump(mode="python"),
            "content": content,
            "previous_content": row.record.content,
        }
    )
    row.vector = list(vector) if vector else None
    row.embedding_model = adapter._embedding_model if vector else None


async def _edit_lancedb(
    adapter: LanceDBAdapter,
    id: HypomnemaId,
    content: str,
    vector: list[float] | None,
) -> None:
    rows = await adapter._records.query().where(f"id = '{id!s}'").to_list()
    if not rows:
        raise KeyError(f"Unknown id: {id!s}")

    existing = row_to_record(rows[0])
    updated = type(existing).model_validate(
        {
            **existing.model_dump(mode="python"),
            "content": content,
            "previous_content": existing.content,
        }
    )
    built_row = record_to_row(updated, vector, adapter._embedding_model)
    arrow_table = pa.table(
        {k: [v] for k, v in built_row.items()},
        schema=adapter._records_schema,
    )
    await adapter._records.merge_insert("id").when_matched_update_all().execute(arrow_table)
