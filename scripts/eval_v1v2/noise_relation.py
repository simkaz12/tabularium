from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval_v1v2 import agent_fold, fold, paths
from eval_v1v2.agent_fold import AgentData
from eval_v1v2.covers import Case
from eval_v1v2.judges import Item, Seat, run_task
from eval_v1v2.prod_check import (
    COVERS_FOLD_DIR,
    JUDGED_FILE,
    JudgedRow,
    load_all_asks,
    noise_events,
)
from eval_v1v2.rows import PairRow, Row, SaveRow, read_jsonl, write_jsonl

FIRST_RUN = "first run"
JUDGE_DIR = "noise-relation-judge"
COVERS_DIR = "noise-judge"
COVERS_QUESTION = "new_covers_old"
ASK_QUESTION = "one_record"
JUDGED_OUT = "noise-relation-judged.jsonl"
WRONG_REINFORCE_OUT = "wrong-noise-reinforce.md"
SUMMARY_OUT = "summary.md"
REPLAY_ARM = "opus-fold_default-in_task"
RELATION_ACTIONS = ("approved:supersede:", "reinforced:reinforce:")
CLIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"project-a", re.IGNORECASE),
    re.compile(r"\b(?:project-f|project-k|project-i)-(?:api|web|app)\b", re.IGNORECASE),
    re.compile(r"[`'\"](?:project-f|project-k|project-i)[`'\"]", re.IGNORECASE),
)
RIGHT_PAIRS = frozenset({("reinforce", "restatement"), ("supersede", "replacement")})
KINDS: tuple[str, ...] = ("containment", "ask", "relation suggestion")
Verdict = Literal["right", "wrong", "unkeyed", "left_out"]
VERDICTS: tuple[Verdict, ...] = ("right", "wrong", "unkeyed", "left_out")
RELATION_RULE = (
    "A reinforce is right when the judges key the noise row as a restatement of the incoming note. "
    "A supersede suggestion is right when they key it as a replacement. A follow_up or distinct "
    "key makes either action wrong, and so does the crossed case (a reinforce on a replacement, "
    "a supersede suggestion on a restatement). A split between the judges is unkeyed."
)
RESTATEMENT_RULE = (
    "A restatement pair counts as resolved when v2 reinforced the seed, superseded it by "
    "containment (runs before run-9 only; later runs send every covers hit to the ask), or "
    "folded it and the fold is graded good, meaning the judges key yes on all four fold "
    "questions (nothing_lost, nothing_invented, newer_wins, one_record). The grade belongs to "
    "the bench note from the Opus arm that the replay sends, from the agent-fold key for band "
    "pairs and from the covers-fold-judge key for the pairs outside the band. A fold graded "
    "bad or disputed does not count, and neither does a note saved as new beside the seed or a "
    "held note that expired without a fold."
)
READING_RULE = (
    "A seed pair keyed restatement also counts as resolved when v2 reinforced a "
    "noise row the judges key as a restatement of the incoming note, because the incoming note was "
    "not written and no new duplicate went live. A seed pair keyed replacement also counts when "
    "v2 put a supersede suggestion on a noise row the judges key as a replacement of the incoming "
    "note."
)


class RelationEvent(BaseModel):
    pair_id: str
    noise_id: str
    actions: dict[str, str]


class RelationJudgedRow(Row):
    pair_id: str
    noise_id: str
    actions: dict[str, str]
    label: str | None
    labels: dict[str, str]


@dataclass
class Tally:
    counts: Counter[str] = field(default_factory=Counter)
    reinforced_wrong: list[tuple[str, str]] = field(default_factory=list)

    def add(self, verdict: Verdict) -> None:
        self.counts[verdict] += 1

    def total(self) -> int:
        return sum(self.counts.values())


def action_of(raw_outcome: str) -> str:
    return "reinforce" if raw_outcome.startswith("reinforced:") else "supersede"


def relation_events(runs: Mapping[str, Sequence[SaveRow]]) -> list[RelationEvent]:
    merged: dict[tuple[str, str], dict[str, str]] = {}

    for run, rows in runs.items():
        for row in rows:
            if row.bucket != "wrong_target" or not row.raw_outcome.startswith(RELATION_ACTIONS):
                continue

            noise_id = row.raw_outcome.rsplit(":", 1)[-1]
            merged.setdefault((row.pair_id, noise_id), {})[run] = action_of(row.raw_outcome)

    return [
        RelationEvent(pair_id=pair_id, noise_id=noise_id, actions=actions)
        for (pair_id, noise_id), actions in sorted(merged.items())
    ]


def mentions_client(text: str) -> bool:
    return any(pattern.search(text) for pattern in CLIENT_PATTERNS)


def event_item_id(event: RelationEvent) -> str:
    return f"{event.pair_id}|{event.noise_id}"


def merge_earlier_relation(
    events: Sequence[RelationEvent], earlier: Sequence[RelationJudgedRow]
) -> list[RelationEvent]:
    merged = {(event.pair_id, event.noise_id): event for event in events}

    for row in earlier:
        key = (row.pair_id, row.noise_id)
        current = merged.get(key)
        actions = row.actions if current is None else {**row.actions, **current.actions}
        merged[key] = RelationEvent(pair_id=row.pair_id, noise_id=row.noise_id, actions=actions)

    return [merged[key] for key in sorted(merged)]


def split_sendable(
    events: Sequence[RelationEvent],
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
) -> tuple[list[RelationEvent], list[RelationEvent]]:
    sendable: list[RelationEvent] = []
    left_out: list[RelationEvent] = []

    for event in events:
        texts = (noise[event.noise_id], incoming[event.pair_id])

        if any(mentions_client(text) for text in texts):
            left_out.append(event)
        else:
            sendable.append(event)

    return sendable, left_out


def judge_events(
    events: Sequence[RelationEvent],
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
    out_dir: Path,
    seats: Sequence[Seat] | None = None,
) -> list[RelationJudgedRow]:
    items = [
        Item(
            item_id=event_item_id(event),
            fields={"existing": noise[event.noise_id], "incoming": incoming[event.pair_id]},
        )
        for event in events
    ]
    outcome = run_task("relation", items, seats=seats, out_dir=out_dir)
    keyed = {row.item_id: str(row.label) for row in outcome.key}
    labels: dict[str, dict[str, str]] = {}

    for label_row in outcome.labels:
        labels.setdefault(label_row.item_id, {})[label_row.judge] = str(label_row.label)

    return [
        RelationJudgedRow(
            pair_id=event.pair_id,
            noise_id=event.noise_id,
            actions=event.actions,
            label=keyed.get(event_item_id(event)),
            labels=labels.get(event_item_id(event), {}),
        )
        for event in events
    ]


def relation_verdict(action: str, label: str | None) -> Verdict:
    if label is None:
        return "unkeyed"

    return "right" if (action, label) in RIGHT_PAIRS else "wrong"


def yes_no_verdict(answer: str | None) -> Verdict:
    if answer is None:
        return "unkeyed"

    return "right" if answer == "yes" else "wrong"


@dataclass(frozen=True)
class NoiseJudgments:
    earlier: Mapping[tuple[str, str], JudgedRow]
    relation: Mapping[tuple[str, str], RelationJudgedRow]
    asked: Mapping[tuple[str, str], JudgedRow] = field(default_factory=dict)


def event_verdict(
    kind: str, action: str, target: tuple[str, str], judged: NoiseJudgments
) -> Verdict:
    if kind == "containment":
        earlier = judged.earlier.get(target)

        return "left_out" if earlier is None else yes_no_verdict(earlier.new_covers_old)

    if kind == "ask":
        asked = judged.asked.get(target)

        return "left_out" if asked is None else yes_no_verdict(asked.one_record)

    row = judged.relation.get(target)

    return "left_out" if row is None else relation_verdict(action, row.label)


def noise_tallies(save: Sequence[SaveRow], judged: NoiseJudgments) -> dict[str, Tally]:
    tallies = {kind: Tally() for kind in KINDS}
    by_pair = {row.pair_id: row for row in save}

    for pair_id, kind, noise_id in noise_events(save):
        action = action_of(by_pair[pair_id].raw_outcome)
        verdict = event_verdict(kind, action, (pair_id, noise_id), judged)
        tallies[kind].add(verdict)

        if kind == "relation suggestion" and action == "reinforce" and verdict == "wrong":
            tallies[kind].reinforced_wrong.append((pair_id, noise_id))

    return tallies


class ReplayRule:
    name = "replay"

    def __init__(self, save: Mapping[str, SaveRow], folded: frozenset[str]) -> None:
        self.save = save
        self.folded = folded

    def decide(self, case: Case) -> fold.Action:
        bucket = self.save[case.pair_id].bucket

        if bucket == "caught":
            return "reinforce"

        if bucket == "superseded":
            return "supersede"

        if bucket == "flagged":
            return "flag"

        return "fold" if case.pair_id in self.folded else "new"


class Readings(BaseModel):
    restatements: int
    replacements: int
    seed_resolved: int
    seed_caught: int
    extra_resolved: list[str]
    extra_caught: list[str]
    loose_resolved: int
    loose_caught: int

    @property
    def resolved_no_new_duplicate(self) -> int:
        return self.seed_resolved + len(self.extra_resolved)

    @property
    def caught_no_new_duplicate(self) -> int:
        return self.seed_caught + len(self.extra_caught)


def folded_pairs(directory: Path) -> frozenset[str]:
    folded = {row.pair_id for row in load_save(directory / "save-v2.jsonl") if is_folded(row)}
    answers = directory / "answer-v2.jsonl"

    if answers.exists():
        rows = (
            json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines() if line
        )
        folded |= {row["pair_id"] for row in rows if row.get("end_state") == "folded"}

    return frozenset(folded)


def is_folded(row: SaveRow) -> bool:
    return row.answer is not None and row.answer.end_state == "folded"


def noise_extras(
    save: Sequence[SaveRow],
    relation: Mapping[str, str],
    judged: Mapping[tuple[str, str], RelationJudgedRow],
    folded: frozenset[str],
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    caught: list[str] = []

    for row in save:
        if row.bucket != "wrong_target" or row.pair_id in folded:
            continue

        noise_id = row.raw_outcome.rsplit(":", 1)[-1]
        verdict = judged.get((row.pair_id, noise_id))

        if verdict is None or not row.raw_outcome.startswith(RELATION_ACTIONS):
            continue

        action = action_of(row.raw_outcome)
        seed = relation.get(row.pair_id)

        if action == "reinforce" and seed == "restatement" and verdict.label == "restatement":
            resolved.append(row.pair_id)

        if action == "supersede" and seed == "replacement" and verdict.label == "replacement":
            caught.append(row.pair_id)

    return resolved, caught


def score_readings(
    save: Sequence[SaveRow],
    folded: frozenset[str],
    data: agent_fold.AgentData,
    judged: Mapping[tuple[str, str], RelationJudgedRow],
) -> Readings:
    inputs = data.inputs
    grades = fold.writer_grades(data.grades, REPLAY_ARM)
    rule = ReplayRule({row.pair_id: row for row in save}, folded)
    score = fold.score_fold(rule, inputs.cases, inputs.relation, inputs.cover, grades)
    agent_fold.credit_replacements(score, rule, inputs, grades, None)
    resolved, caught = noise_extras(save, inputs.relation, judged, folded)

    return Readings(
        restatements=score.base.restatements,
        replacements=score.base.replacements,
        seed_resolved=agent_fold.resolved(score, "strict"),
        seed_caught=agent_fold.caught(score, "strict"),
        extra_resolved=resolved,
        extra_caught=caught,
        loose_resolved=agent_fold.resolved(score, "loose"),
        loose_caught=agent_fold.caught(score, "loose"),
    )


class Interruptions(BaseModel):
    keyed_new: int
    asked: int
    asked_seed: int
    flagged: int
    folded: int
    covers_asked: int = 0


def interruptions(
    save: Sequence[SaveRow], relation: Mapping[str, str], folded: frozenset[str]
) -> Interruptions:
    new_pairs = {pair for pair, label in relation.items() if label in ("follow_up", "distinct")}
    rows = [row for row in save if row.pair_id in new_pairs]

    return Interruptions(
        keyed_new=len(new_pairs),
        asked=sum(row.raw_outcome.startswith("asked:") for row in rows),
        asked_seed=sum(row.bucket == "asked" for row in rows),
        flagged=sum(row.bucket == "flagged" for row in rows),
        folded=sum(row.pair_id in folded for row in rows),
        covers_asked=sum(row.raw_outcome.startswith("asked:covers:") for row in rows),
    )


def load_save(path: Path) -> list[SaveRow]:
    return read_jsonl(path, SaveRow)


def run_dir(replay: Path, run: str) -> Path:
    return replay if run == FIRST_RUN else replay / run


def run_name(run: str) -> str:
    return run if run == FIRST_RUN else Path(run).name


def discover_runs(replay: Path) -> list[str]:
    found = [path.name for path in replay.glob("run-*") if (path / "save-v2.jsonl").exists()]

    return sorted(found, key=lambda name: (len(name), name))


def resolve_runs(replay: Path, requested: Sequence[str]) -> list[str]:
    runs = list(requested) or discover_runs(replay)

    if not runs:
        raise ValueError(f"no run directories under {replay}")

    missing = [run for run in runs if not (run_dir(replay, run) / "save-v2.jsonl").exists()]

    if missing:
        raise ValueError(f"no save-v2.jsonl in {', '.join(missing)}")

    return runs


def load_noise_texts(base: Path) -> dict[str, str]:
    return {row["id"]: row["content"] for row in map(json.loads, (base / "noise.jsonl").open())}


def covers_item_id(pair_id: str, noise_id: str, recorded: Mapping[tuple[str, str], str]) -> str:
    return recorded.get((pair_id, noise_id)) or f"{pair_id}|{noise_id}#{COVERS_QUESTION}"


def recorded_item_ids(prior: Sequence[JudgedRow]) -> dict[tuple[str, str], str]:
    return {
        (row.pair_id, row.noise_id): row.item_id or f"{row.pair_id}#{COVERS_QUESTION}"
        for row in prior
    }


def containment_pairs(runs: Mapping[str, Sequence[SaveRow]]) -> list[tuple[str, str]]:
    found = {
        (pair_id, noise_id)
        for rows in runs.values()
        for pair_id, kind, noise_id in noise_events(rows)
        if kind == "containment"
    }

    return sorted(found)


def judge_containment(
    events: Sequence[tuple[str, str]],
    earlier: Sequence[JudgedRow],
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
    out_dir: Path,
    seats: Sequence[Seat] | None = None,
) -> list[JudgedRow]:
    prior = [row for row in earlier if row.event == "contained"]
    recorded = recorded_item_ids(prior)
    wanted = sorted(recorded.keys() | set(events))
    sendable = [
        pair
        for pair in wanted
        if pair in recorded
        or not any(mentions_client(text) for text in (noise[pair[1]], incoming[pair[0]]))
    ]
    items = [
        Item(
            item_id=covers_item_id(pair_id, noise_id, recorded),
            fields={
                "question": COVERS_QUESTION,
                "existing": noise[noise_id],
                "incoming": incoming[pair_id],
            },
        )
        for pair_id, noise_id in sendable
    ]
    outcome = run_task("covers", items, seats=seats, out_dir=out_dir)
    keyed = {row.item_id: str(row.label) for row in outcome.key}
    labels: dict[str, dict[str, str]] = {}

    for label_row in outcome.labels:
        labels.setdefault(label_row.item_id, {})[label_row.judge] = str(label_row.label)

    rows: list[JudgedRow] = []

    for pair_id, noise_id in sendable:
        item_id = covers_item_id(pair_id, noise_id, recorded)
        answer = keyed.get(item_id)
        rows.append(
            JudgedRow.model_validate(
                {
                    "pair_id": pair_id,
                    "event": "contained",
                    "noise_id": noise_id,
                    "new_covers_old": answer,
                    "new_covers_old_labels": labels.get(item_id, {}),
                    "item_id": item_id,
                }
            )
        )

    return rows


def reply_notes(base: Path, arm: str = REPLAY_ARM) -> dict[str, str]:
    return {
        pair_id: row.note
        for (pair_id, name), row in load_all_asks(base).items()
        if name == arm and row.folded and row.note is not None
    }


def ask_pairs(runs: Mapping[str, Sequence[SaveRow]]) -> list[tuple[str, str]]:
    found = {
        (pair_id, noise_id)
        for rows in runs.values()
        for pair_id, kind, noise_id in noise_events(rows)
        if kind == "ask"
    }

    return sorted(found)


def recorded_ask_ids(prior: Sequence[JudgedRow]) -> dict[tuple[str, str], str]:
    return {
        (row.pair_id, row.noise_id): row.item_id or f"{row.pair_id}#{ASK_QUESTION}" for row in prior
    }


def ask_item_id(pair_id: str, noise_id: str, recorded: Mapping[tuple[str, str], str]) -> str:
    return recorded.get((pair_id, noise_id)) or f"{pair_id}|{noise_id}#{ASK_QUESTION}"


def judge_asks(
    events: Sequence[tuple[str, str]],
    earlier: Sequence[JudgedRow],
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
    notes: Mapping[str, str],
    out_dir: Path,
    seats: Sequence[Seat] | None = None,
) -> list[JudgedRow]:
    prior = {(row.pair_id, row.noise_id): row for row in earlier if row.event == "asked"}
    recorded = recorded_ask_ids(list(prior.values()))
    wanted = sorted(recorded.keys() | set(events))
    sendable = [
        pair
        for pair in wanted
        if pair[0] in notes
        and (
            pair in recorded
            or not any(
                mentions_client(text)
                for text in (noise[pair[1]], incoming[pair[0]], notes[pair[0]])
            )
        )
    ]
    items = [
        Item(
            item_id=ask_item_id(pair_id, noise_id, recorded),
            fields={
                "question": ASK_QUESTION,
                "existing": noise[noise_id],
                "incoming": incoming[pair_id],
                "fold": notes[pair_id],
            },
        )
        for pair_id, noise_id in sendable
    ]
    outcome = run_task("fold", items, seats=seats, out_dir=out_dir)
    keyed = {row.item_id: str(row.label) for row in outcome.key}
    labels: dict[str, dict[str, str]] = {}

    for label_row in outcome.labels:
        labels.setdefault(label_row.item_id, {})[label_row.judge] = str(label_row.label)

    rows: list[JudgedRow] = []

    for pair_id, noise_id in sendable:
        item_id = ask_item_id(pair_id, noise_id, recorded)
        before = prior.get((pair_id, noise_id))
        rows.append(
            JudgedRow.model_validate(
                {
                    "pair_id": pair_id,
                    "event": "asked",
                    "noise_id": noise_id,
                    "new_covers_old": None if before is None else before.new_covers_old,
                    "new_covers_old_labels": {} if before is None else before.new_covers_old_labels,
                    "one_record": keyed.get(item_id),
                    "one_record_labels": labels.get(item_id, {}),
                    "item_id": item_id,
                }
            )
        )

    dropped = [row for pair, row in prior.items() if pair not in set(sendable)]

    return [*rows, *dropped]


def judge_command(
    base: Path,
    replay: Path,
    requested: Sequence[str] = (),
    seats: Sequence[Seat] | None = None,
) -> int:
    names = resolve_runs(replay, requested)
    runs = {run_name(run): load_save(run_dir(replay, run) / "save-v2.jsonl") for run in names}
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    noise = load_noise_texts(base)
    earlier_rows = read_jsonl(replay / JUDGED_FILE, JudgedRow)
    relation_path = replay / JUDGED_OUT
    earlier_relation = (
        read_jsonl(relation_path, RelationJudgedRow) if relation_path.exists() else []
    )
    events = merge_earlier_relation(relation_events(runs), earlier_relation)
    incoming = {pair_id: pair.incoming for pair_id, pair in pairs.items()}
    sendable, left_out = split_sendable(events, noise, incoming)
    judged = judge_events(sendable, noise, incoming, replay / JUDGE_DIR, seats)
    first = {FIRST_RUN: load_save(replay / "save-v2.jsonl")}
    contained = judge_containment(
        containment_pairs({**first, **runs}),
        earlier_rows,
        noise,
        incoming,
        replay / COVERS_DIR,
        seats,
    )
    asked = judge_asks(
        ask_pairs({**first, **runs}),
        earlier_rows,
        noise,
        incoming,
        reply_notes(base),
        replay / COVERS_DIR,
        seats,
    )

    write_jsonl(replay / JUDGED_OUT, judged)
    write_jsonl(replay / JUDGED_FILE, [*contained, *asked])
    sys.stdout.write(
        f"events {len(events)} sent {len(sendable)} left out {len(left_out)} "
        f"keyed {sum(row.label is not None for row in judged)} "
        f"containment judged {len(contained)} asks judged {len(asked)}\n"
    )
    for event in left_out:
        sys.stdout.write(f"left out {event.pair_id} {event.noise_id}\n")

    return 0


def load_judgments(replay: Path) -> NoiseJudgments:
    earlier = read_jsonl(replay / JUDGED_FILE, JudgedRow)
    relation = read_jsonl(replay / JUDGED_OUT, RelationJudgedRow)

    return NoiseJudgments(
        earlier={(row.pair_id, row.noise_id): row for row in earlier if row.event == "contained"},
        relation={(row.pair_id, row.noise_id): row for row in relation},
        asked={(row.pair_id, row.noise_id): row for row in earlier if row.event == "asked"},
    )


def covers_grades(replay: Path) -> dict[tuple[str, str], fold.FoldGrade]:
    return fold.load_fold_grades(replay / COVERS_FOLD_DIR / "key-fold.jsonl")


def with_covers_grades(data: AgentData, replay: Path) -> AgentData:
    return dataclasses.replace(data, grades={**covers_grades(replay), **data.grades})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge relation suggestions that hit noise rows.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("runs", nargs="*", help="run directory names under --replay")
    args = parser.parse_args(argv)
    replay = args.replay or args.base / "experiments" / "prod-replay"

    return judge_command(args.base, replay, args.runs)


if __name__ == "__main__":
    sys.exit(main())
