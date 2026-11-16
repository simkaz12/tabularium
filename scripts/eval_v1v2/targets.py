from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval_v1v2 import judges
from eval_v1v2.rows import LabelRow, NoiseRow, PairRow, SaveRow, System, read_jsonl

Action = Literal["merged", "flagged"]

TARGETS_DIR = "targets"
SPLIT = "judges split"
UNRESOLVED = "unresolved"
CATEGORIES: tuple[str, ...] = (
    "restatement",
    "replacement",
    "follow_up",
    "distinct",
    SPLIT,
    UNRESOLVED,
)
MERGE_OUTCOMES: frozenset[str] = frozenset({"reinforced", "revived", "already_saved"})
FLAG_OUTCOMES: frozenset[str] = frozenset({"bounced", "approved"})


class Pointed(BaseModel):
    system: System
    pair_id: str
    action: Action
    target_id: str | None
    category: str = UNRESOLVED


def action_of(raw_outcome: str) -> Action:
    head = raw_outcome.split(":", 1)[0]

    if head in MERGE_OUTCOMES:
        return "merged"

    if head in FLAG_OUTCOMES:
        return "flagged"

    raise ValueError(f"unknown raw_outcome {raw_outcome!r}")


def target_id_of(raw_outcome: str) -> str | None:
    parts = raw_outcome.split(":")

    if parts[0] in MERGE_OUTCOMES and len(parts) == 2:
        return parts[1]

    return parts[-1] if len(parts) >= 3 else None


def pointed_rows(rows: Sequence[SaveRow]) -> list[Pointed]:
    return [
        Pointed(
            system=row.system,
            pair_id=row.pair_id,
            action=action_of(row.raw_outcome),
            target_id=target_id_of(row.raw_outcome),
        )
        for row in rows
        if row.bucket == "wrong_target"
    ]


def item_key(row: Pointed) -> str:
    return f"{row.system}:{row.pair_id}"


def build_items(
    pointed: Sequence[Pointed], texts: Mapping[str, str], incoming: Mapping[str, str]
) -> tuple[list[judges.Item], dict[str, str]]:
    items: list[judges.Item] = []
    canonical: dict[str, str] = {}
    first_seen: dict[tuple[str, str], str] = {}

    for row in pointed:
        if row.target_id is None or row.target_id not in texts or row.pair_id not in incoming:
            continue

        pair = (texts[row.target_id], incoming[row.pair_id])
        item_id = item_key(row)
        canonical[item_id] = first_seen.setdefault(pair, item_id)

        if canonical[item_id] == item_id:
            items.append(
                judges.Item(item_id=item_id, fields={"existing": pair[0], "incoming": pair[1]})
            )

    return items, canonical


def categorise(
    pointed: Sequence[Pointed], canonical: Mapping[str, str], labels: Sequence[LabelRow]
) -> list[Pointed]:
    agreed = {row.item_id: str(row.label) for row in judges.agreed_key(labels)}
    done: list[Pointed] = []

    for row in pointed:
        item_id = canonical.get(item_key(row))
        category = UNRESOLVED if item_id is None else agreed.get(item_id, SPLIT)
        done.append(row.model_copy(update={"category": category}))

    return done


def noise_texts(base: Path) -> dict[str, str]:
    if not (base / "noise.jsonl").exists():
        return {}

    return {row.id: row.content for row in read_jsonl(base / "noise.jsonl", NoiseRow)}


def incoming_texts(base: Path) -> dict[str, str]:
    if not (base / "pairs.jsonl").exists():
        return {}

    return {row.pair_id: row.incoming for row in read_jsonl(base / "pairs.jsonl", PairRow)}


def load_save_rows(base: Path) -> list[SaveRow]:
    rows: list[SaveRow] = []

    for system in ("v1", "v2"):
        path = base / f"save-{system}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path, SaveRow))

    return rows


def items_for(base: Path) -> tuple[list[Pointed], list[judges.Item], dict[str, str]]:
    pointed = pointed_rows(load_save_rows(base))
    items, canonical = build_items(pointed, noise_texts(base), incoming_texts(base))

    return pointed, items, canonical


def judged_pointed(base: Path) -> list[Pointed]:
    pointed, _items, canonical = items_for(base)
    labels_path = base / TARGETS_DIR / "labels-relation.jsonl"
    labels = read_jsonl(labels_path, LabelRow) if labels_path.exists() else []

    return categorise(pointed, canonical, labels)
