from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from eval_v1v2 import paths
from eval_v1v2.paths import out_path
from eval_v1v2.rows import CorpusRow, write_jsonl

POOL_PATH = paths.DATA_ROOT / "burial" / "pool-full.json"
EXCLUDED_SCOPES = frozenset({"project-a", "project-i"})
LIVE_LIFECYCLES = frozenset({"active", "in_force", "hot"})
CORPUS_NAME = "corpus.jsonl"
CORE_FIELDS = frozenset(
    {"id", "type", "content", "scope", "lifecycle", "vector", "recall_count", "created_at"}
)


def is_live(record: dict[str, Any]) -> bool:
    return record.get("validity") == "valid" and record.get("lifecycle") in LIVE_LIFECYCLES


def to_row(record: dict[str, Any]) -> CorpusRow:
    extras = {key: value for key, value in record.items() if key not in CORE_FIELDS}
    return CorpusRow(
        id=record["id"],
        kind=record["type"],
        content=record["content"],
        scope=record["scope"],
        lifecycle=record["lifecycle"],
        vector=record["vector"],
        recall_count=record.get("recall_count") or 0,
        created_at=record["created_at"],
        **extras,
    )


def select(records: list[dict[str, Any]]) -> Iterator[CorpusRow]:
    for record in records:
        if record.get("scope") in EXCLUDED_SCOPES:
            continue
        if not is_live(record):
            continue
        yield to_row(record)


def build(pool_path: Path = POOL_PATH, out: Path | None = None) -> list[CorpusRow]:
    with pool_path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    rows = list(select(records))
    write_jsonl(out or out_path(CORPUS_NAME), rows)
    return rows


def main() -> None:
    argparse.ArgumentParser(description="Build the search corpus.").parse_args()
    rows = build()
    counts = Counter(row.scope for row in rows)
    print(f"corpus rows: {len(rows)}")
    for scope, count in sorted(counts.items()):
        print(f"  {scope}: {count}")


if __name__ == "__main__":
    main()
