from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from eval_v1v2 import paths
from eval_v1v2.rows import Judge, KeyRow, LabelRow, PairRow, read_jsonl, write_jsonl
from tabularium.engine.judge import (
    IndexedVerdictBase,
    JudgeInvalidOutput,
    JudgeUnavailable,
    SubprocessJudgeAdapter,
    parse_batch_verdicts,
)

BATCH_SIZE = 20
DEFAULT_SEED = 20261112
DEFAULT_MODELS: dict[Judge, str] = {"claude": "opus", "codex": "gpt-5.6-sol"}
RELATION_LABELS = ("restatement", "replacement", "follow_up", "distinct")
RELEVANCE_GRADES = (0, 1, 2)
CALL_TIMEOUT = 600.0

LabelValue = str | int


class LabelledVerdict(IndexedVerdictBase):
    label: LabelValue
    reason: str


class RelationVerdict(LabelledVerdict):
    label: Literal["restatement", "replacement", "follow_up", "distinct"]


class RelevanceVerdict(LabelledVerdict):
    label: int = Field(strict=True, ge=0, le=2)


class CoversVerdict(LabelledVerdict):
    label: Literal["yes", "no"]


class Caller(Protocol):
    def call(self, prompt: str) -> Awaitable[str]: ...


@dataclass(frozen=True)
class Item:
    item_id: str
    fields: Mapping[str, str]


@dataclass(frozen=True)
class Seat:
    judge: Judge
    model: str
    adapter: Caller


@dataclass(frozen=True)
class Task:
    name: str
    required: tuple[str, ...]
    verdict_model: type[LabelledVerdict]
    preamble: str
    render: Callable[[Item], str]
    label_hint: str


@dataclass
class Outcome:
    labels: list[LabelRow]
    key: list[KeyRow]
    agreement: dict[str, object]
    unlabelled: Mapping[str, int] = field(default_factory=dict)


RELATION_PREAMBLE = """\
You compare two short notes from a memory store: EXISTING was saved first, INCOMING arrives later.
Give each pair exactly one label.

- restatement: INCOMING says the same thing as EXISTING in other words; nothing new.
  Example: EXISTING "The deploy script lives in bin/deploy.sh."
  INCOMING "bin/deploy.sh is the deploy script."
- replacement: INCOMING covers the same subject and changes or corrects it, so EXISTING is out
  of date.
  Example: EXISTING "The cache TTL is 60 seconds." INCOMING "The cache TTL is now 300 seconds."
- follow_up: INCOMING is about the same subject and adds new information; EXISTING stays true
  and both are worth keeping.
  Example: EXISTING "We moved the queue to Redis."
  INCOMING "The Redis queue needs maxmemory-policy set to noeviction."
- distinct: the two notes are about different things, even if they share words.
  Example: EXISTING "The cache TTL is 60 seconds." INCOMING "Deploys run on Tuesdays."

Judge from the text alone."""

RELEVANCE_PREAMBLE = """\
You grade how well a stored RECORD answers a QUESTION that someone asked a memory system.

- 2: the record answers the question, or holds the specific fact it asks for. Grade 2 is rare.
- 1: the record is on topic and would help, but does not answer the question by itself.
- 0: the record is off topic or only shares words with the question.

Expected spread over a batch: about one in ten grade 2, about one in four grade 1, the rest grade 0.
Use grade 2 only when you could stop searching after reading the record.
Judge from the text alone."""


COVERS_PREAMBLE = """\
You compare two short notes from a memory store: EXISTING was saved first, INCOMING arrives later.
Each item names one QUESTION about whether one note contains the other. Answer yes or no.

- new_covers_old: does INCOMING state everything EXISTING states? Answer yes only when no fact,
  number, name, reason, decision or caveat in EXISTING is missing from INCOMING. Rewording counts as
  stated, and extra detail in INCOMING does not matter. If INCOMING changes or contradicts something
  EXISTING says, answer no.
  Example: EXISTING "The cache TTL is 60 seconds." INCOMING "The cache TTL is 60 seconds; the
  setting lives in config/cache.php." Answer yes.
  Example: EXISTING "The cache TTL is 60 seconds and keys expire lazily." INCOMING "The cache TTL
  is 60 seconds." Answer no, the lazy expiry is gone.
- old_covers_new: does EXISTING state everything INCOMING states? Answer yes only when no fact,
  number, name, reason, decision or caveat in INCOMING is missing from EXISTING. Rewording counts as
  stated, and extra detail in EXISTING does not matter. If INCOMING changes or contradicts
  something EXISTING says, answer no.
  Example: EXISTING "The cache TTL is 60 seconds; the setting lives in config/cache.php."
  INCOMING "The cache TTL is 60 seconds." Answer yes.

The two questions are independent. Judge each item from the text alone."""


FOLD_PREAMBLE = """\
You check a FOLD: one note a writer made by merging two short notes from a memory store. EXISTING
was saved first. INCOMING is newer and wins where the two disagree. Each item names one QUESTION
about the FOLD. Answer yes or no.

- nothing_lost: does the FOLD contain every claim made in EXISTING and in INCOMING? Answer yes only
  when no fact, number, name, path, config key, date, decision, rejected option or reason from
  either note is missing. Rewording counts as present. A value that INCOMING overrides counts as
  present when the FOLD says it was replaced.
- nothing_invented: is every claim in the FOLD found in EXISTING or INCOMING? Answer yes only when
  the FOLD adds no fact, number, name, reason, advice or inference that neither note states.
  Connecting words and a statement that a value was replaced are not added claims.
- newer_wins: where EXISTING and INCOMING conflict, does the FOLD give INCOMING's version as the
  current one? Answer yes when the two notes do not conflict. Answer no when they conflict and the
  FOLD keeps EXISTING's version, presents both as current, or leaves the conflict unresolved.
- one_record: does the FOLD read as one fact or one decision about one subject, not two separate
  things glued together? Answer no when it reads as two notes joined by a heading, a list of
  unrelated points, or a bridge between different topics.

The four questions are independent. Judge each item from the text alone."""


def _render_fold(item: Item) -> str:
    return (
        f"QUESTION: {item.fields['question']}\n"
        f"EXISTING: {item.fields['existing']}\nINCOMING: {item.fields['incoming']}\n"
        f"FOLD: {item.fields['fold']}"
    )


def _render_cover(item: Item) -> str:
    return (
        f"QUESTION: {item.fields['question']}\n"
        f"EXISTING: {item.fields['existing']}\nINCOMING: {item.fields['incoming']}"
    )


def _render_pair(item: Item) -> str:
    return f"EXISTING: {item.fields['existing']}\nINCOMING: {item.fields['incoming']}"


def _render_question(item: Item) -> str:
    return f"QUESTION: {item.fields['question']}\nRECORD: {item.fields['record']}"


TASKS: dict[str, Task] = {
    "relation": Task(
        name="relation",
        required=("existing", "incoming"),
        verdict_model=RelationVerdict,
        preamble=RELATION_PREAMBLE,
        render=_render_pair,
        label_hint="one of restatement, replacement, follow_up, distinct",
    ),
    "relevance": Task(
        name="relevance",
        required=("question", "record"),
        verdict_model=RelevanceVerdict,
        preamble=RELEVANCE_PREAMBLE,
        render=_render_question,
        label_hint="the integer 0, 1 or 2 (a JSON number, not a string)",
    ),
    "covers": Task(
        name="covers",
        required=("question", "existing", "incoming"),
        verdict_model=CoversVerdict,
        preamble=COVERS_PREAMBLE,
        render=_render_cover,
        label_hint="yes or no",
    ),
    "fold": Task(
        name="fold",
        required=("question", "existing", "incoming", "fold"),
        verdict_model=CoversVerdict,
        preamble=FOLD_PREAMBLE,
        render=_render_fold,
        label_hint="yes or no",
    ),
}


def build_prompt(task: Task, batch: Sequence[Item]) -> str:
    entries = "\n\n".join(
        f"[{position}]\n{task.render(item)}" for position, item in enumerate(batch, start=1)
    )
    return (
        f"{task.preamble}\n\n"
        f"There are {len(batch)} items below, numbered 1 to {len(batch)}.\n\n"
        f"{entries}\n\n"
        "Reply with one JSON object and nothing else: "
        '{"verdicts": [{"index": 1, "label": ..., "reason": "..."}, ...]}. '
        f"The label is {task.label_hint}. The reason is one short sentence. "
        f"Give each index from 1 to {len(batch)} exactly once. "
        "The first character of your reply is `{`. No code fences."
    )


def build_seats(
    models: Mapping[Judge, str] | None = None, timeout: float = CALL_TIMEOUT
) -> list[Seat]:
    chosen = {**DEFAULT_MODELS, **(models or {})}
    seats: list[Seat] = []
    for name in ("claude", "codex"):
        binary = shutil.which(name)
        if binary is None:
            raise JudgeUnavailable(f"{name} binary not found on PATH")
        judge: Judge = "claude" if name == "claude" else "codex"
        adapter = SubprocessJudgeAdapter(
            cli=judge, binary=binary, model=chosen[judge], timeout=timeout
        )
        seats.append(Seat(judge=judge, model=chosen[judge], adapter=adapter))
    return seats


def _validate_items(task: Task, items: Sequence[Item]) -> None:
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"duplicate item_id {item.item_id}")
        seen.add(item.item_id)
        missing = [name for name in task.required if name not in item.fields]
        if missing:
            raise ValueError(f"item {item.item_id} lacks fields {missing} for task {task.name}")


def _batches(items: Sequence[Item], judge: Judge, seed: int, size: int) -> list[list[Item]]:
    ordered = list(items)
    random.Random(f"{seed}:{judge}").shuffle(ordered)
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


async def _label_batch(task: Task, seat: Seat, batch: Sequence[Item]) -> list[LabelRow] | None:
    prompt = build_prompt(task, batch)
    for _attempt in range(2):
        try:
            raw = await seat.adapter.call(prompt)
            verdicts = parse_batch_verdicts(
                raw, item_model=task.verdict_model, batch_size=len(batch)
            )
        except (JudgeUnavailable, JudgeInvalidOutput):
            continue
        return [
            LabelRow.model_validate(
                {
                    "item_id": item.item_id,
                    "judge": seat.judge,
                    "model": seat.model,
                    "label": verdict.label,
                    "reason": verdict.reason,
                }
            )
            for item, verdict in zip(batch, verdicts, strict=True)
        ]
    return None


async def _run_seat(
    task: Task,
    seat: Seat,
    items: Sequence[Item],
    labels: list[LabelRow],
    seed: int,
    size: int,
    persist: Callable[[], None],
) -> int:
    done = {row.item_id for row in labels if row.judge == seat.judge}
    pending = [item for item in items if item.item_id not in done]
    unlabelled = 0
    for batch in _batches(pending, seat.judge, seed, size):
        rows = await _label_batch(task, seat, batch)
        if rows is None:
            unlabelled += len(batch)
            continue
        labels.extend(rows)
        persist()
    return unlabelled


def agreement_stats(labels: Sequence[LabelRow]) -> dict[str, object]:
    by_judge: dict[str, dict[str, LabelValue]] = {"claude": {}, "codex": {}}
    for row in labels:
        by_judge[row.judge][row.item_id] = row.label
    shared = sorted(set(by_judge["claude"]) & set(by_judge["codex"]))
    total = len(shared)
    confusion: dict[str, dict[str, int]] = {}
    agree = 0
    claude_marginal: Counter[str] = Counter()
    codex_marginal: Counter[str] = Counter()
    for item_id in shared:
        first = by_judge["claude"][item_id]
        second = by_judge["codex"][item_id]
        confusion.setdefault(str(first), {}).setdefault(str(second), 0)
        confusion[str(first)][str(second)] += 1
        claude_marginal[str(first)] += 1
        codex_marginal[str(second)] += 1
        agree += first == second
    raw = agree / total if total else 0.0
    chance = (
        sum(claude_marginal[k] * codex_marginal[k] for k in claude_marginal) / (total * total)
        if total
        else 0.0
    )
    if total == 0:
        kappa = 0.0
    elif chance >= 1.0:
        kappa = 1.0 if raw == 1.0 else 0.0
    else:
        kappa = (raw - chance) / (1.0 - chance)
    return {
        "items_labelled_by_both": total,
        "items_agreed": agree,
        "raw_agreement": raw,
        "cohens_kappa": kappa,
        "confusion": confusion,
    }


def agreed_key(labels: Sequence[LabelRow], prior: Sequence[KeyRow] = ()) -> list[KeyRow]:
    by_judge: dict[str, dict[str, LabelValue]] = {"claude": {}, "codex": {}}
    for row in labels:
        by_judge[row.judge][row.item_id] = row.label
    owner = {row.item_id: row for row in prior if row.source == "owner"}
    key: dict[str, KeyRow] = dict(owner)
    for item_id in sorted(set(by_judge["claude"]) & set(by_judge["codex"])):
        if item_id in owner:
            continue
        first = by_judge["claude"][item_id]
        if first == by_judge["codex"][item_id]:
            key[item_id] = KeyRow.model_validate(
                {"item_id": item_id, "label": first, "source": "agreed"}
            )
    return list(key.values())


def _load[RowT: LabelRow | KeyRow](path: Path, model: type[RowT]) -> list[RowT]:
    return read_jsonl(path, model) if path.exists() else []


def run_task(
    task_name: str,
    items: Sequence[Item],
    *,
    seats: Sequence[Seat] | None = None,
    seed: int = DEFAULT_SEED,
    batch_size: int = BATCH_SIZE,
    out_dir: Path | None = None,
) -> Outcome:
    task = TASKS[task_name]
    _validate_items(task, items)
    chosen = list(seats) if seats is not None else build_seats()
    base = out_dir if out_dir is not None else paths.ensure_out_dir()
    labels_path = base / f"labels-{task_name}.jsonl"
    key_path = base / f"key-{task_name}.jsonl"
    agreement_path = base / f"agreement-{task_name}.json"

    known = {item.item_id for item in items}
    labels = [row for row in _load(labels_path, LabelRow) if row.item_id in known]
    prior_key = _load(key_path, KeyRow)

    def persist() -> None:
        write_jsonl(labels_path, labels)

    async def drive() -> list[int]:
        return list(
            await asyncio.gather(
                *(
                    _run_seat(task, seat, items, labels, seed, batch_size, persist)
                    for seat in chosen
                )
            )
        )

    counts = asyncio.run(drive())
    persist()

    key = agreed_key(labels, prior_key)
    stats = agreement_stats(labels)
    write_jsonl(key_path, key)
    agreement_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    unlabelled: dict[str, int] = {
        seat.judge: count for seat, count in zip(chosen, counts, strict=True)
    }
    return Outcome(labels=labels, key=key, agreement=stats, unlabelled=unlabelled)


def items_from_pairs(path: Path, limit: int | None = None) -> list[Item]:
    pairs = read_jsonl(path, PairRow)
    if limit is not None:
        pairs = pairs[:limit]
    return [
        Item(item_id=pair.pair_id, fields={"existing": pair.existing, "incoming": pair.incoming})
        for pair in pairs
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label relation pairs with claude and codex.")
    parser.add_argument("--pairs", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    pairs_path = args.pairs if args.pairs is not None else paths.out_path("pairs.jsonl")
    items = items_from_pairs(pairs_path, args.limit)
    outcome = run_task("relation", items, seed=args.seed, out_dir=args.out_dir)
    print(
        json.dumps(
            {
                "items": len(items),
                "key_rows": len(outcome.key),
                "unlabelled": outcome.unlabelled,
                "raw_agreement": outcome.agreement["raw_agreement"],
                "cohens_kappa": outcome.agreement["cohens_kappa"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
