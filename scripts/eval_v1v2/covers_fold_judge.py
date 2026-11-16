from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval_v1v2 import fold, paths
from eval_v1v2.agent_fold_ask import load_asks
from eval_v1v2.judges import Item, Seat, run_task
from eval_v1v2.noise_relation import mentions_client
from eval_v1v2.prod_check import COVERS_ASKS_FILE, COVERS_FOLD_DIR, REPLAY_ARM, left_out_file
from eval_v1v2.rows import PairRow, read_jsonl


def fold_notes(replies: Path, arm: str) -> dict[str, str]:
    return {
        pair_id: row.note
        for (pair_id, name), row in load_asks(replies).items()
        if name == arm and row.folded and row.note is not None
    }


def fold_items(
    notes: Mapping[str, str], pairs: Mapping[str, PairRow], arm: str
) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    left_out: list[str] = []

    for pair_id, note in sorted(notes.items()):
        pair = pairs[pair_id]

        if any(mentions_client(text) for text in (pair.existing, pair.incoming, note)):
            left_out.append(pair_id)
            continue

        items += [
            Item(
                item_id=fold.fold_item_id(pair_id, arm, question),
                fields={
                    "question": question,
                    "existing": pair.existing,
                    "incoming": pair.incoming,
                    "fold": note,
                },
            )
            for question in fold.QUESTIONS
        ]

    return items, left_out


def screen_folds(base: Path, arm: str) -> tuple[list[Item], list[str], Path]:
    replay = base / "experiments" / "prod-replay"
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    notes = fold_notes(replay / COVERS_ASKS_FILE, arm)
    items, left_out = fold_items(notes, pairs, arm)
    out_dir = replay / COVERS_FOLD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / left_out_file(arm)).write_text(
        "".join(f"{pair_id}\n" for pair_id in left_out), "utf-8"
    )

    return items, left_out, out_dir


def judge_command(
    base: Path, arm: str = REPLAY_ARM, seats: Sequence[Seat] | None = None
) -> tuple[int, list[str]]:
    items, left_out, out_dir = screen_folds(base, arm)
    outcome = run_task("fold", items, seats=seats, out_dir=out_dir)

    return len(outcome.key), left_out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge the replayed covers folds.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--arm", default=REPLAY_ARM)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args(argv)

    if args.screen_only:
        _, left_out, _ = screen_folds(args.base, args.arm)
        sys.stdout.write(f"left out {left_out}\n")

        return 0

    keyed, left_out = judge_command(args.base, args.arm)
    sys.stdout.write(f"keyed {keyed} left out {left_out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
