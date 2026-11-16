from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PICKS = {
    "follow_up only": {"follow_up"},
    "follow_up or unclear": {"follow_up", "unclear"},
    "any non-distinct": {"follow_up", "unclear", "restatement", "replacement"},
}
QUESTIONS = ("nothing_lost", "nothing_invented", "newer_wins", "one_record")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def recount(base: Path, writer: str, floor: float, enhances: float | None, picks: str) -> str:
    cdir = base / "experiments" / "covers"
    signal = {r["pair_id"]: r for r in rows(base / "experiments" / "dup-rules" / "signals.jsonl")}
    covers = {r["pair_id"]: r for r in rows(cdir / "signals.jsonl")}
    relation = {r["item_id"]: r["label"] for r in rows(base / "key-relation.jsonl")}
    cover_key = {r["item_id"]: r["label"] == "yes" for r in rows(cdir / "key-covers.jsonl")}
    fold_key = {
        r["item_id"]: r["label"] == "yes" for r in rows(base / "experiments/fold/key-fold.jsonl")
    }
    folded = {
        r["pair_id"] for r in rows(base / "experiments/fold/folds.jsonl") if r["writer"] == writer
    }

    restated = replaced = folds = bad = wrong = lost = 0

    for pid, sig in signal.items():
        label = relation.get(pid)
        neighbour = sig["existing"]
        pick = None if neighbour is None else neighbour["pick"]
        sim = None if neighbour is None else neighbour["similarity"]
        covered = covers[pid]["new_covers_old"]["p_yes"]
        boost = covers[pid]["enhances"]["p_yes"]
        action = "new"

        if pick == "restatement":
            action = "reinforce"
        elif sim is not None and covered is not None and covered >= 0.90:
            action = "supersede"
        elif pick == "replacement":
            action = "flag"
        elif (
            sim is not None
            and sim >= floor
            and pick in PICKS[picks]
            and (enhances is None or (boost is not None and boost >= enhances))
        ):
            action = "fold"

        answers = {q: fold_key.get(f"{pid}~{writer}#{q}") for q in QUESTIONS}

        if action == "fold":
            assert pid in folded
            folds += 1
            bad += any(v is False for v in answers.values())

        if action == "supersede" and cover_key.get(f"{pid}#new_covers_old") is False:
            wrong += 1

        if action == "reinforce" and label in ("follow_up", "distinct", "replacement"):
            lost += 1

        if label == "restatement":
            restated += action in ("reinforce", "supersede") or (
                action == "fold" and all(v is True for v in answers.values())
            )
        elif label == "replacement":
            replaced += action in ("flag", "supersede") or (
                action == "fold"
                and answers["newer_wins"] is True
                and answers["nothing_invented"] is True
            )

    return (
        f"writer={writer} rule=sim>={floor} enhances={enhances} picks={picks}: "
        f"restatements={restated} replacements={replaced} folds={folds} bad={bad} "
        f"wrong_supersedes={wrong} lost={lost}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recount a fold rule from raw rows.")
    parser.add_argument("base", type=Path)
    parser.add_argument("writer")
    parser.add_argument("floor", type=float)
    parser.add_argument("enhances", help="a number or none")
    parser.add_argument("picks", choices=sorted(PICKS))
    args = parser.parse_args()
    enhances = None if args.enhances == "none" else float(args.enhances)
    print(recount(args.base, args.writer, args.floor, enhances, args.picks))


if __name__ == "__main__":
    main()
