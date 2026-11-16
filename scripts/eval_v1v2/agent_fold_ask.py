from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

from eval_v1v2 import fold, paths
from eval_v1v2.rows import PairRow, Row, read_jsonl
from eval_v1v2.second_judge import CALL_TIMEOUT, CONCURRENCY, MAX_ATTEMPTS
from tabularium.engine.judge import JudgeUnavailable, SubprocessJudgeAdapter

Wording = Literal["choice", "fold_default"]
Framing = Literal["cold", "in_task"]
ReplyAction = Literal["fold", "keep_both", "continue"]
AskAction = Literal["fold", "keep_both", "continue", "invalid"]

MODELS: tuple[str, ...] = ("sonnet", "opus")
SHAPES: tuple[tuple[Wording, Framing], ...] = (
    ("choice", "cold"),
    ("fold_default", "cold"),
    ("fold_default", "in_task"),
)
RAW_LIMIT = 600
EXISTING_TOKEN = "<<EXISTING>>"
INCOMING_TOKEN = "<<INCOMING>>"
_TOKEN = re.compile(r"<<(EXISTING|INCOMING)>>")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

FOLD_RULES = """\
Keep every concrete detail from both notes: names, numbers, paths, config keys, dates, rejected
options and the reasons they were rejected.
Add nothing that is in neither note. No new advice, no inference, no framing of your own.
Where the notes conflict, your note is the newer one and wins. Give the newer value and say that
the older value was replaced.
Write it as one fact or one decision, not as two notes joined together.
The note field holds the note text only: no preamble, no heading, no quotes, no explanation."""

NOTES_BLOCK = """\
EXISTING NOTE (already in the store):
<<EXISTING>>

YOUR NOTE (the one you just sent):
<<INCOMING>>"""

CHOICE_ASK = f"""\
Your note was not saved yet. The store already holds a note that may cover the same ground.

{NOTES_BLOCK}

Choose one reply.

1. Send one note that holds everything from both. It replaces both in the store.
{FOLD_RULES}
Reply: {{"action": "fold", "note": "<the one note>"}}

2. If the two are separate facts, each worth keeping as its own note, answer keep_both. Both stay
in the store as they are.
Reply: {{"action": "keep_both", "reason": "<one short sentence>"}}

Reply with one JSON object and nothing else."""

FOLD_DEFAULT_ASK = f"""\
Your note was not saved yet. The store says it looks like the same thing as a note it already
holds.

{NOTES_BLOCK}

Send back the one merged note that holds everything from both. It replaces both in the store.
{FOLD_RULES}
Reply: {{"action": "fold", "note": "<the one note>"}}

The only other reply is keep_both, and it is allowed only with a one-line reason that names the
fact in the existing note that your note is not about. A reason that does not name such a fact is
not accepted.
Reply: {{"action": "keep_both", "reason": "<one line naming that fact>"}}

Reply with one JSON object and nothing else."""

ASKS: dict[Wording, str] = {"choice": CHOICE_ASK, "fold_default": FOLD_DEFAULT_ASK}

IN_TASK_TASK = """\
You are a coding agent working in a repository for a small internal inventory tool. Your task: add
a `--dry-run` flag to the nightly `export_orders` command, so that it prints the rows it would send
to the reporting database and writes nothing. Done means: the flag shows in `--help`, the existing
tests still pass, and one new test covers the dry run."""

IN_TASK_PROGRESS = """\
Work so far: you read `cli/export_orders.py` and `reporting/writer.py`, found that the writer is
called from one place, and decided the flag check belongs just before that call. You have not edited
any file yet. While reading you noticed something worth keeping for later sessions and called the
memory save tool with a note about it."""

IN_TASK_FRAME = f"""\
{IN_TASK_TASK}

{IN_TASK_PROGRESS}

The tool result for that save call is below.

--- tool result ---
<<ASK>>
--- end of tool result ---

Besides the replies the tool result lists, you may reply {{"action": "continue"}} to carry on with
your task without answering the tool result. Reply with one JSON object and nothing else."""

FRAMES: dict[Framing, str] = {"cold": "<<ASK>>", "in_task": IN_TASK_FRAME}
ASK_TOKEN = "<<ASK>>"


@dataclass(frozen=True)
class Arm:
    model: str
    wording: Wording
    framing: Framing

    @property
    def id(self) -> str:
        return f"{self.model}-{self.wording}-{self.framing}"


def all_arms(models: Sequence[str] = MODELS) -> list[Arm]:
    return [Arm(model, wording, framing) for model in models for wording, framing in SHAPES]


def arm_model(arm_id: str) -> str:
    return arm_id.partition("-")[0]


def prompt_template(wording: Wording, framing: Framing) -> str:
    return FRAMES[framing].replace(ASK_TOKEN, ASKS[wording])


def render_prompt(wording: Wording, framing: Framing, existing: str, incoming: str) -> str:
    texts = {"EXISTING": existing, "INCOMING": incoming}

    return _TOKEN.sub(lambda found: texts[found.group(1)], prompt_template(wording, framing))


class Reply(BaseModel):
    action: ReplyAction
    note: str | None = None
    reason: str | None = None


class UnparseableReply(Exception):
    pass


def parse_reply(text: str, wording: Wording, framing: Framing) -> Reply:
    found = _JSON_OBJECT.search(text)

    if found is None:
        raise UnparseableReply(text[:200])

    try:
        reply = Reply.model_validate(json.loads(found.group(0)))
    except (ValueError, ValidationError) as exc:
        raise UnparseableReply(str(exc)[:200]) from exc

    if reply.action == "fold" and not (reply.note or "").strip():
        raise UnparseableReply("fold without a note")

    if (
        reply.action == "keep_both"
        and wording == "fold_default"
        and not (reply.reason or "").strip()
    ):
        raise UnparseableReply("keep_both without a reason")

    if reply.action == "continue" and framing != "in_task":
        raise UnparseableReply("continue is not offered in the cold framing")

    return reply


class AskRow(Row):
    pair_id: str
    arm: str
    model: str
    wording: Wording
    framing: Framing
    action: AskAction | None
    note: str | None
    reason: str | None
    raw: str | None
    elapsed_ms: float
    attempts: int
    error: str | None

    @property
    def answered(self) -> bool:
        return self.error is None and self.action is not None

    @property
    def folded(self) -> bool:
        return self.answered and self.action == "fold" and bool(self.note)


def load_asks(path: Path) -> dict[tuple[str, str], AskRow]:
    if not path.exists():
        return {}

    return {(row.pair_id, row.arm): row for row in read_jsonl(path, AskRow)}


def pending_pairs(
    band: Sequence[fold.BandRow], asks: Mapping[tuple[str, str], AskRow], arm: Arm
) -> list[fold.BandRow]:
    return [
        row
        for row in band
        if (row.pair_id, arm.id) not in asks or not asks[(row.pair_id, arm.id)].answered
    ]


async def ask_pair(adapter: SubprocessJudgeAdapter, arm: Arm, row: fold.BandRow) -> AskRow:
    prompt = render_prompt(arm.wording, arm.framing, row.existing, row.incoming)
    started = time.perf_counter()
    reply: Reply | None = None
    raw: str | None = None
    error: str | None = None
    attempts = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts = attempt

        try:
            raw = await adapter.call(prompt)
            error = None
            reply = parse_reply(raw, arm.wording, arm.framing)
        except JudgeUnavailable as exc:
            error = f"JudgeUnavailable: {str(exc)[:200]}"
        except UnparseableReply:
            continue

        if reply is not None:
            break

    return _ask_row(arm, row.pair_id, reply, raw, error, started, attempts)


def _ask_row(
    arm: Arm,
    pair_id: str,
    reply: Reply | None,
    raw: str | None,
    error: str | None,
    started: float,
    attempts: int,
) -> AskRow:
    action: AskAction | None = None

    if reply is not None:
        action = reply.action
    elif error is None:
        action = "invalid"

    return AskRow(
        pair_id=pair_id,
        arm=arm.id,
        model=arm.model,
        wording=arm.wording,
        framing=arm.framing,
        action=action,
        note=None if reply is None else reply.note,
        reason=None if reply is None else reply.reason,
        raw=raw[:RAW_LIMIT] if action == "invalid" and raw is not None else None,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        attempts=attempts,
        error=error,
    )


async def ask_all(
    adapter: SubprocessJudgeAdapter,
    arm: Arm,
    rows: Sequence[fold.BandRow],
    sink: Callable[[AskRow], None],
) -> list[AskRow]:
    gate = asyncio.Semaphore(CONCURRENCY)
    failures = 0
    results: list[AskRow] = []

    async def one(row: fold.BandRow) -> None:
        nonlocal failures

        if failures >= fold.STOP_AFTER_FAILURES:
            return

        async with gate:
            if failures >= fold.STOP_AFTER_FAILURES:
                return

            result = await ask_pair(adapter, arm, row)

        failures = 0 if result.answered else failures + 1
        sink(result)
        results.append(result)

    await asyncio.gather(*(one(row) for row in rows))

    return results


def write_prompts(out_dir: Path) -> None:
    target = out_dir / "ask-prompts"
    target.mkdir(parents=True, exist_ok=True)

    for wording in ASKS:
        for framing in FRAMES:
            target.joinpath(f"{wording}-{framing}.txt").write_text(
                prompt_template(wording, framing) + "\n", encoding="utf-8"
            )


def ask_arms(log: Path, rows: Sequence[fold.BandRow], arms: Sequence[Arm]) -> bool:
    binary = shutil.which("claude")

    if binary is None:
        print("claude CLI not found", file=sys.stderr)

        return False

    log = log.resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")

    def sink(row: AskRow) -> None:
        handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()

    os.chdir(tempfile.mkdtemp(prefix="agent-fold-"))
    complete = True

    for arm in arms:
        todo = pending_pairs(rows, load_asks(log), arm)
        adapter = SubprocessJudgeAdapter(
            cli="claude", binary=binary, model=arm.model, timeout=CALL_TIMEOUT
        )
        asyncio.run(ask_all(adapter, arm, todo, sink))
        left = len(pending_pairs(rows, load_asks(log), arm))
        print(f"{arm.id}: {len(todo)} attempted, {left} of {len(rows)} still pending", flush=True)
        complete = complete and left == 0

    handle.close()

    return complete


def run_asks(out_dir: Path, base: Path, models: Sequence[str]) -> int:
    band = read_jsonl(base / "experiments" / "fold" / "band.jsonl", fold.BandRow)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_prompts(out_dir)

    return 0 if ask_arms(out_dir / "asks.jsonl", band, all_arms(models)) else 2


def pair_band_rows(pairs: Sequence[PairRow], only: Collection[str]) -> list[fold.BandRow]:
    return [
        fold.BandRow(
            pair_id=pair.pair_id,
            source=pair.source,
            relation=None,
            similarity=pair.similarity or 0.0,
            pick=None,
            enhances=None,
            existing=pair.existing,
            incoming=pair.incoming,
        )
        for pair in pairs
        if pair.pair_id in only
    ]


def read_pair_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_pair_asks(log: Path, pairs_path: Path, ids_path: Path) -> int:
    rows = pair_band_rows(read_jsonl(pairs_path, PairRow), set(read_pair_ids(ids_path)))
    arm = Arm("opus", "fold_default", "in_task")

    return 0 if ask_arms(log, rows, [arm]) else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent fold: ask every band pair, per arm.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "agent-fold"
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--only", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if args.pairs is not None or args.only is not None or args.log is not None:
        if args.pairs is None or args.only is None or args.log is None:
            parser.error("--pairs, --only and --log go together")

        sys.exit(run_pair_asks(args.log, args.pairs, args.only))

    sys.exit(run_asks(args.out_dir, args.base, args.models))


if __name__ == "__main__":
    main()
