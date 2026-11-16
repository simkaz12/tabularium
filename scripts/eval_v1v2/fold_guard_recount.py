from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

QUESTIONS = ("nothing_lost", "nothing_invented", "newer_wins", "one_record")
GUARDS = ("fold_covers_old", "fold_covers_new", "sources_cover_fold", "fold_single")
KINDS = {
    "off": (),
    "coverage": GUARDS[:2],
    "coverage+sources": GUARDS[:3],
    "all": GUARDS,
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def recount(
    base: Path,
    writer: str,
    floor: float,
    enhances: float | None,
    subject: float | None,
    replacement: float | None,
    kind: str,
    threshold: float | None,
) -> str:
    cdir = base / "experiments" / "covers"
    gdir = base / "experiments" / "fold-guard"
    signal = {r["pair_id"]: r for r in rows(base / "experiments" / "dup-rules" / "signals.jsonl")}
    covers = {r["pair_id"]: r for r in rows(cdir / "signals.jsonl")}
    relation = {r["item_id"]: r["label"] for r in rows(base / "key-relation.jsonl")}
    cover_key = {r["item_id"]: r["label"] == "yes" for r in rows(cdir / "key-covers.jsonl")}
    fold_key = {
        r["item_id"]: r["label"] == "yes" for r in rows(base / "experiments/fold/key-fold.jsonl")
    }
    folded = {
        r["pair_id"]
        for r in rows(base / "experiments/fold/folds.jsonl")
        if r["writer"] == writer and r.get("error") is None and r.get("text")
    }
    guard = {r["pair_id"]: r for r in rows(gdir / "guard-signals.jsonl") if r["writer"] == writer}
    band = {r["pair_id"]: r for r in rows(gdir / "band-signals.jsonl")}

    restated = replaced = loose_restated = loose_replaced = 0
    fired = accepted = bad = disputed = good_rejected = wrong = lost = 0

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

        answers = {q: fold_key.get(f"{pid}~{writer}#{q}") for q in QUESTIONS}
        good = all(v is True for v in answers.values())
        has_no = any(v is False for v in answers.values())

        if (
            action == "new"
            and pick == "follow_up"
            and sim is not None
            and sim >= 0.75
            and pid in folded
        ):
            probability = neighbour["probabilities"].get("replacement", 0.0)
            gate = sim >= floor
            gate = gate and (enhances is None or (boost is not None and boost >= enhances))
            subject_p = band[pid]["same_subject"]["p_yes"] if pid in band else None
            gate = gate and (subject is None or (subject_p is not None and subject_p >= subject))

            if gate or (replacement is not None and probability >= replacement):
                fired += 1
                readings = [guard[pid][q]["p_yes"] if pid in guard else None for q in KINDS[kind]]
                passes = threshold is None or all(
                    r is not None and r >= threshold for r in readings
                )

                if passes:
                    action = "fold"
                    accepted += 1
                    bad += has_no
                    disputed += not has_no and not good
                elif good:
                    good_rejected += 1

        if action == "supersede" and cover_key.get(f"{pid}#new_covers_old") is False:
            wrong += 1

        if action == "reinforce" and label in ("follow_up", "distinct", "replacement"):
            lost += 1

        newer_clean = answers["newer_wins"] is True and answers["nothing_invented"] is True
        newer_unfailed = answers["newer_wins"] is not False
        clean_unfailed = answers["nothing_invented"] is not False

        if label == "restatement":
            restated += action in ("reinforce", "supersede") or (action == "fold" and good)
            loose_restated += action in ("reinforce", "supersede") or (
                action == "fold" and not has_no
            )
        elif label == "replacement":
            replaced += action in ("flag", "supersede") or (action == "fold" and newer_clean)
            loose_replaced += action in ("flag", "supersede") or (
                action == "fold" and newer_unfailed and clean_unfailed
            )

    return (
        f"writer={writer} floor={floor} enhances={enhances} same_subject={subject} "
        f"p_replacement={replacement} guards={kind}>={threshold}: restatements={restated} "
        f"replacements={replaced} loose_restatements={loose_restated} "
        f"loose_replacements={loose_replaced} fired={fired} accepted={accepted} bad={bad} "
        f"disputed={disputed} good_rejected={good_rejected} wrong_supersedes={wrong} lost={lost}"
    )


def optional(value: str) -> float | None:
    return None if value == "none" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recount a guarded fold rule from raw rows.")
    parser.add_argument("base", type=Path)
    parser.add_argument("writer")
    parser.add_argument("floor", type=float)
    parser.add_argument("enhances", help="a number or none")
    parser.add_argument("subject", help="a number or none")
    parser.add_argument("replacement", help="a number or none")
    parser.add_argument("kind", choices=sorted(KINDS))
    parser.add_argument("threshold", help="a number or none")
    args = parser.parse_args()
    print(
        recount(
            args.base,
            args.writer,
            args.floor,
            optional(args.enhances),
            optional(args.subject),
            optional(args.replacement),
            args.kind,
            optional(args.threshold),
        )
    )


if __name__ == "__main__":
    main()
