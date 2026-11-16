from __future__ import annotations

import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import yaml

from eval_v1v2 import paths
from eval_v1v2.rows import Kind, NoiseRow, PairRow, write_jsonl

SEED = 20261112
BOUNCE_PER_QUARTILE = 30
CUT_LENGTH_SHARE = 0.05
CUT_LENGTH_MIN_TEXTS = 30
CUT_LENGTH_FLOOR = 100
NOISE_COUNT = 1000
EXCLUDED_SCOPES = frozenset({"project-a", "project-i", "project-f", "project-k"})
REPO_SCOPES = {
    "project-a-api": "project-a",
    "project-a-web": "project-a",
    "project-i-api": "project-i",
    "project-i-app": "project-i",
    "project-f-api": "project-f",
    "project-f-web": "project-f",
    "project-k": "project-k",
}
LIVE_LIFECYCLES = frozenset({"active", "in_force", "hot"})
CONTENT_RELATION_CLASSES = frozenset(
    {
        "admission-auto-restatement",
        "admission-auto-replacement",
        "admission-auto-write",
        "admission-invalidation",
    }
)

REPO_ROOT = paths.REPO_ROOT
JEV_PATH = REPO_ROOT / "tests" / "fixtures" / "jev_pairs_sample.json"
BOUNCE_PATH = paths.DATA_ROOT / "bounce-log.json"
LEDGER_PATH = paths.DATA_ROOT / "ledger-export.json"
ADMISSION_DIR = REPO_ROOT / "tests" / "cases" / "admission"
POOL_PATH = paths.DATA_ROOT / "burial" / "pool-full.json"


def _kind(value: object) -> Kind:
    return "decision" if value == "decision" else "fact"


def bounce_scope(new_scope: object) -> str:
    name = str(new_scope).split("/", 1)[-1]

    return REPO_SCOPES.get(name, name)


def jev_pairs(path: Path = JEV_PATH) -> list[PairRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows: list[PairRow] = []

    for item in raw:
        if {item.get("existing_scope"), item.get("incoming_scope")} & EXCLUDED_SCOPES:
            continue

        similarity = item.get("similarity")
        rows.append(
            PairRow(
                pair_id=f"jev-{item['pair_id']}",
                source="jev_pairs",
                kind=_kind(item.get("incoming_type")),
                existing=item["existing_content"],
                incoming=item["incoming_content"],
                similarity=None if similarity is None else float(similarity),
                old_label=item.get("store_label"),
            )
        )

    return rows


def _ledger_scope(atom: dict[str, Any]) -> str:
    return f"{atom['scope_kind']}/{atom['scope_id']}"


def resolve_bounce_row(
    item: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    by_scope: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[str, str, Kind] | None, str | None]:
    incoming = by_id.get(str(item.get("new_pi_id")))

    if incoming is None:
        return None, "missing_new_pi_id"

    incoming_text = str(incoming["content"]).strip()

    if not incoming_text.startswith(str(item["new_content"])):
        return None, "incoming_prefix_mismatch"

    prefix = str(item["candidate_content"])
    scope = str(item.get("new_scope") or _ledger_scope(incoming))
    texts = {
        str(atom["content"]).strip()
        for atom in by_scope.get(scope, [])
        if str(atom["content"]).strip().startswith(prefix)
    }

    if not texts:
        return None, "candidate_not_found"

    if len(texts) > 1:
        return None, "candidate_ambiguous"

    kind: Kind = "decision" if incoming.get("kind") == "decision" else "fact"

    return (texts.pop(), incoming_text, kind), None


def bounce_pairs(
    path: Path = BOUNCE_PATH,
    seed: int = SEED,
    per_quartile: int = BOUNCE_PER_QUARTILE,
    ledger_path: Path = LEDGER_PATH,
    drops: Counter[str] | None = None,
) -> list[PairRow]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    by_id = {str(atom["id"]): atom for atom in ledger}
    by_scope: dict[str, list[dict[str, Any]]] = {}

    for atom in ledger:
        by_scope.setdefault(_ledger_scope(atom), []).append(atom)

    dropped = drops if drops is not None else Counter()
    resolved: list[tuple[int, tuple[str, str, Kind], float]] = []

    for index, item in enumerate(raw):
        if bounce_scope(item.get("new_scope")) in EXCLUDED_SCOPES:
            continue

        texts, reason = resolve_bounce_row(item, by_id, by_scope)

        if texts is None:
            dropped[str(reason)] += 1
            continue

        resolved.append((index, texts, float(item["similarity"])))

    ordered = sorted(resolved, key=lambda entry: (entry[2], entry[0]))
    total = len(ordered)
    rng = random.Random(seed)
    rows: list[PairRow] = []

    for quartile in range(4):
        chunk = ordered[quartile * total // 4 : (quartile + 1) * total // 4]
        picked = rng.sample(chunk, min(per_quartile, len(chunk)))
        picked.sort(key=lambda entry: entry[0])

        for index, (existing, incoming, kind), similarity in picked:
            rows.append(
                PairRow(
                    pair_id=f"bounce-{index:04d}",
                    source="bounce_log",
                    kind=kind,
                    existing=existing,
                    incoming=incoming,
                    similarity=similarity,
                    old_label=None,
                )
            )

    print(f"bounce log rows in scope: {len(resolved) + sum(dropped.values())}")
    print(f"bounce log rows resolved to full text: {len(resolved)}")
    print(f"bounce log rows dropped: {sum(dropped.values())}")

    for reason, count in sorted(dropped.items()):
        print(f"  dropped {reason}: {count}")

    return rows


def golden_pairs() -> list[PairRow]:
    scripts_dir = str(Path(__file__).resolve().parents[1])
    added = scripts_dir not in sys.path

    if added:
        sys.path.insert(0, scripts_dir)

    try:
        from freeze_vectors import GOLDEN_PAIRS
    finally:
        if added:
            sys.path.remove(scripts_dir)

    rows: list[PairRow] = []

    for index, item in enumerate(GOLDEN_PAIRS):
        rows.append(
            PairRow(
                pair_id=f"golden-{index:02d}",
                source="golden",
                kind="fact",
                existing=str(item["text_a"]),
                incoming=str(item["text_b"]),
                similarity=None,
                old_label=str(item["label"]),
            )
        )

    return rows


def admission_pairs(directory: Path = ADMISSION_DIR) -> list[PairRow]:
    rows: list[PairRow] = []

    for path in sorted(directory.glob("*.yaml")):
        case = yaml.safe_load(path.read_text(encoding="utf-8"))
        expect = case.get("expect") or {}
        incoming = case.get("incoming")
        existing = case.get("existing") or []

        if case.get("class") not in CONTENT_RELATION_CLASSES:
            continue

        if len(existing) != 1 or not incoming or not expect.get("material"):
            continue

        rows.append(
            PairRow(
                pair_id=f"adm-{case['id']}",
                source="admission",
                kind=_kind(incoming.get("kind")),
                existing=existing[0]["text"],
                incoming=incoming["content"],
                similarity=None,
                old_label=expect["material"],
            )
        )

    return rows


def dedupe(rows: Iterable[PairRow]) -> list[PairRow]:
    seen: set[tuple[str, str]] = set()
    kept: list[PairRow] = []

    for row in rows:
        key = (row.existing, row.incoming)

        if key in seen:
            continue

        seen.add(key)
        kept.append(row)

    return kept


def build_pairs(
    jev_path: Path = JEV_PATH,
    bounce_path: Path = BOUNCE_PATH,
    admission_dir: Path = ADMISSION_DIR,
    seed: int = SEED,
    ledger_path: Path = LEDGER_PATH,
) -> list[PairRow]:
    rows = dedupe(
        [
            *jev_pairs(jev_path),
            *bounce_pairs(bounce_path, seed, ledger_path=ledger_path),
            *golden_pairs(),
            *admission_pairs(admission_dir),
        ]
    )
    check_no_cut_texts(rows)

    return rows


def check_no_cut_texts(rows: Iterable[PairRow], share: float = CUT_LENGTH_SHARE) -> None:
    texts: dict[str, set[str]] = {}

    for row in rows:
        texts.setdefault(row.source, set()).update((row.existing, row.incoming))

    for source, distinct in sorted(texts.items()):
        if len(distinct) < CUT_LENGTH_MIN_TEXTS:
            continue

        values = [len(text) for text in distinct if len(text) >= CUT_LENGTH_FLOOR]

        if not values:
            continue

        length, count = Counter(values).most_common(1)[0]

        if count > share * len(distinct):
            raise ValueError(
                f"{source}: {count} of {len(distinct)} texts are exactly {length} characters; "
                "the texts look cut"
            )


def pair_texts(pairs: Iterable[PairRow]) -> set[str]:
    texts: set[str] = set()

    for pair in pairs:
        texts.add(pair.existing.strip())
        texts.add(pair.incoming.strip())

    return texts


def build_noise(
    pool: list[dict[str, Any]],
    excluded_texts: set[str],
    seed: int = SEED,
    count: int = NOISE_COUNT,
) -> list[NoiseRow]:
    eligible = [
        record
        for record in pool
        if record.get("validity") == "valid"
        and record.get("lifecycle") in LIVE_LIFECYCLES
        and record.get("scope") not in EXCLUDED_SCOPES
        and str(record["content"]).strip() not in excluded_texts
    ]
    eligible.sort(key=lambda record: str(record["id"]))
    picked = random.Random(seed).sample(eligible, min(count, len(eligible)))

    return [
        NoiseRow(
            id=str(record["id"]),
            kind=_kind(record.get("type")),
            content=record["content"],
            vector=record["vector"],
            recall_count=int(record.get("recall_count") or 0),
            created_at=str(record["created_at"]),
            lifecycle=str(record["lifecycle"]),
        )
        for record in picked
    ]


def main() -> None:
    pairs = build_pairs()
    pool = cast("list[dict[str, Any]]", json.loads(POOL_PATH.read_text(encoding="utf-8")))
    noise = build_noise(pool, pair_texts(pairs))
    del pool

    pair_count = write_jsonl(paths.out_path("pairs.jsonl"), pairs)
    noise_count = write_jsonl(paths.out_path("noise.jsonl"), noise)

    for source, count in sorted(Counter(pair.source for pair in pairs).items()):
        print(f"{source}: {count}")

    print(f"pairs: {pair_count}")
    print(f"noise: {noise_count}")


if __name__ == "__main__":
    main()
