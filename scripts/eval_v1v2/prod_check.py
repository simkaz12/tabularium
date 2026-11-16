from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eval_v1v2 import covers, dup_rules, fold, fold_guard, paths
from eval_v1v2.agent_fold_ask import AskRow, load_asks
from eval_v1v2.rows import EndState, NeighbourRow, Replay, Row, SaveRow, read_jsonl

REPLAY_ARM = "opus-fold_default-in_task"
CONTAINMENT_FLOOR = 0.95
FOLD_SIMILARITY_FLOOR = 0.80
FOLD_SUBJECT_FLOOR = 0.70
REINFORCED_EXPECTED = 24
REINFORCED_TOLERANCE = 2
ASKED_TOLERANCE = 2
COVERS_ASK_PREFIX = "asked:covers:"
COVERS_ASKS_FILE = "asks-covers.jsonl"
COVERS_FOLD_DIR = "covers-fold-judge"
LEFT_OUT_PREFIX = "left-out-fold"
NEAR_MARGIN = 0.06
JUDGED_FILE = "noise-judged.jsonl"
NO_REPLY_STATES = ("no_reply", "unreplayed")
EXPECTED_END: dict[str, str] = {
    "fold": "folded",
    "keep_both": "both_live",
    "continue": "both_live",
    "none": "no_reply",
}


class AskRecord(Row):
    existing_id: str
    similarity: float
    same_subject: float
    new_covers_old: float
    is_seed: bool


class AnswerRow(Row):
    pair_id: str
    asked: bool
    replay: Replay
    status: str
    ask_answer: str | None
    end_state: EndState
    existing_live: bool
    existing_superseded_by_fold: bool
    fold_live: bool
    held_live: bool
    fold_covers_old: float | None
    fold_covers_new: float | None
    error: str | None
    neighbours: list[NeighbourRow] | None = None
    ask: AskRecord | None = None


class JudgedRow(Row):
    pair_id: str
    event: Literal["contained", "asked"]
    noise_id: str
    new_covers_old: Literal["yes", "no"] | None
    new_covers_old_labels: dict[str, str] = Field(default_factory=dict)
    one_record: Literal["yes", "no"] | None = None
    one_record_labels: dict[str, str] = Field(default_factory=dict)
    item_id: str | None = None


class SeedProbs(BaseModel):
    new_covers_old: float | None
    same_subject: float | None


class Bound(BaseModel):
    name: str
    expected: int
    tolerance: int
    actual: int
    adjustment: int = 0

    @property
    def judged(self) -> int:
        return self.actual + self.adjustment

    @property
    def ok(self) -> bool:
        return abs(self.judged - self.expected) <= self.tolerance


EventKind = Literal["containment", "ask", "relation suggestion"]


class Unkeyed(BaseModel):
    pair_id: str
    kind: EventKind
    noise_id: str
    verdict: str
    action_verdict: Literal["yes", "no"] | None = None


class Check(BaseModel):
    bounds: list[Bound]
    reinforced: list[str]
    superseded: list[str]
    contained_on_noise: list[str]
    asked: list[str]
    covers_asked: list[str]
    wrong_target: list[str]
    wrong_target_new_paths: list[str]
    asked_outside_bench: list[str]
    asked_missing: list[str]
    asked_on_noise: list[str] = Field(default_factory=list)
    ask_suppressed: list[str] = Field(default_factory=list)
    reinforced_outside_bench: list[str]
    reinforced_missing: list[str]
    near_threshold: list[str]
    near_flips: list[str]
    near_flip_notes: list[str]
    unkeyed: list[Unkeyed]
    end_states: dict[str, int]
    end_state_mismatches: list[str]
    asked_without_answer_row: list[str]
    no_reply: list[str] = Field(default_factory=list)
    fold_grades: dict[str, int] = Field(default_factory=dict)
    bad_folds: list[str] = Field(default_factory=list)
    ungraded_folds: list[str] = Field(default_factory=list)
    screened_folds: list[str] = Field(default_factory=list)
    noise_only_asks: list[str] = Field(default_factory=list)

    @property
    def findings(self) -> list[str]:
        return findings_for(self)

    @property
    def ok(self) -> bool:
        return not self.findings


def decide_all(cases: Mapping[str, covers.Case], floor: float) -> dict[str, str]:
    rule = covers.jev_rule("new_covers_old", False, floor, None)

    return {pair_id: rule.decide(case) for pair_id, case in cases.items()}


def bench_actions(base: Path, floor: float = CONTAINMENT_FLOOR) -> dict[str, str]:
    experiments = base / "experiments"
    signals = dup_rules.load_signals(experiments / "dup-rules" / "signals.jsonl")
    signal = covers.load_covers(experiments / "covers" / "signals.jsonl")

    return decide_all(covers.build_cases(signals, signal, {}), floor)


def bench_probs(base: Path) -> dict[str, SeedProbs]:
    experiments = base / "experiments"
    signal = covers.load_covers(experiments / "covers" / "signals.jsonl")
    subjects = fold_guard.load_band_signals(experiments / "fold-guard" / "band-signals.jsonl")
    probs: dict[str, SeedProbs] = {}

    for pair_id in signal.keys() | subjects.keys():
        cover = signal.get(pair_id)
        subject = subjects.get(pair_id)
        probs[pair_id] = SeedProbs(
            new_covers_old=None if cover is None else cover.new_covers_old.p_yes,
            same_subject=None if subject is None else subject.same_subject.p_yes,
        )

    return probs


def within(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value - threshold) <= NEAR_MARGIN + 1e-9


def near_threshold(probs: Mapping[str, SeedProbs], floor: float) -> set[str]:
    return {
        pair_id
        for pair_id, prob in probs.items()
        if within(prob.new_covers_old, floor) or within(prob.same_subject, FOLD_SUBJECT_FLOOR)
    }


def crosses_opposite_side(value: float | None, threshold: float, observed_above: bool) -> bool:
    if value is None or not within(value, threshold):
        return False

    return value < threshold if observed_above else value >= threshold


def near_flips_of(
    probs: Mapping[str, SeedProbs],
    floor: float,
    bench_covers: set[str],
    bench_ask: set[str],
    asked: set[str],
) -> tuple[list[str], list[str]]:
    cover_flips: list[str] = []
    ask_flips: list[str] = []

    for pair_id in sorted(bench_ask ^ asked):
        prob = probs.get(pair_id)

        if prob is None:
            continue

        observed = pair_id in asked

        if crosses_opposite_side(prob.new_covers_old, floor, observed):
            cover_flips.append(pair_id)
        elif crosses_opposite_side(prob.same_subject, FOLD_SUBJECT_FLOOR, observed):
            ask_flips.append(pair_id)

    return cover_flips[: len(bench_covers) // 2], ask_flips


def trigger_asked(base: Path, actions: Mapping[str, str]) -> set[str]:
    experiments = base / "experiments"
    band = read_jsonl(experiments / "fold" / "band.jsonl", fold.BandRow)
    subjects = fold_guard.load_band_signals(experiments / "fold-guard" / "band-signals.jsonl")
    asked: set[str] = set()

    for row in band:
        subject = subjects.get(row.pair_id)
        p_yes = None if subject is None else subject.same_subject.p_yes

        if (
            row.similarity >= FOLD_SIMILARITY_FLOOR
            and p_yes is not None
            and p_yes >= FOLD_SUBJECT_FLOOR
            and row.pick != "distinct"
            and actions.get(row.pair_id) == "new"
        ):
            asked.add(row.pair_id)

    return asked


def covers_set(actions: Mapping[str, str]) -> set[str]:
    return {pair_id for pair_id, action in actions.items() if action == "supersede"}


def bench_asked(base: Path, actions: Mapping[str, str]) -> set[str]:
    return trigger_asked(base, actions) | covers_set(actions)


def replay_of(ask: AskRow | None) -> Replay:
    if ask is None or not ask.answered:
        return "none"

    if ask.action == "fold":
        return "fold"

    if ask.action == "keep_both":
        return "keep_both"

    return "continue" if ask.action == "continue" else "none"


def pairs_in(rows: Sequence[SaveRow], bucket: str) -> list[str]:
    return sorted(row.pair_id for row in rows if row.bucket == bucket)


def answers_of(save: Sequence[SaveRow], answers: Sequence[AnswerRow]) -> dict[str, AnswerRow]:
    merged = {
        row.pair_id: AnswerRow.model_validate(
            {"pair_id": row.pair_id, "asked": True, **row.answer.model_dump()}
        )
        for row in save
        if row.answer is not None
    }

    return {row.pair_id: row for row in answers} | merged


def end_state_checks(
    asked: Sequence[str],
    answers: Mapping[str, AnswerRow],
    asks: Mapping[tuple[str, str], AskRow],
    arm: str,
) -> tuple[list[str], list[str], list[str]]:
    mismatches: list[str] = []
    missing: list[str] = []
    unreplied: list[str] = []

    for pair_id in asked:
        row = answers.get(pair_id)

        if row is None:
            missing.append(pair_id)
            continue

        replay = replay_of(asks.get((pair_id, arm)))
        expected = EXPECTED_END[replay]
        matches = (
            row.end_state in NO_REPLY_STATES and row.held_live
            if replay == "none"
            else row.end_state == expected
        )

        if not matches or not row.asked:
            unsaved = ", held note not live" if replay == "none" and not row.held_live else ""
            mismatches.append(
                f"{pair_id}: {row.end_state} (asked={row.asked}{unsaved}), expected {expected}"
            )
        elif replay == "none":
            unreplied.append(pair_id)

    return mismatches, missing, unreplied


def bounds_for(
    reinforced: int,
    superseded: int,
    asked: int,
    expected_asked: int,
    asked_adjustment: int = 0,
) -> list[Bound]:
    return [
        Bound(
            name="reinforced",
            expected=REINFORCED_EXPECTED,
            tolerance=REINFORCED_TOLERANCE,
            actual=reinforced,
        ),
        Bound(name="superseded", expected=0, tolerance=0, actual=superseded),
        Bound(
            name="asked",
            expected=expected_asked,
            tolerance=ASKED_TOLERANCE,
            actual=asked,
            adjustment=asked_adjustment,
        ),
    ]


def adjustment_for(bench: set[str], actual: set[str], explained: set[str]) -> int:
    return len((bench - actual) & explained) - len((actual - bench) & explained)


def suppressed_asks_of(
    save: Sequence[SaveRow], bench_ask: set[str], seed_asked: set[str]
) -> set[str]:
    replaced = {
        row.pair_id
        for row in save
        if row.bucket == "wrong_target" and row.raw_outcome.startswith("approved:supersede:")
    }

    return (bench_ask & replaced) - seed_asked


def noise_events(save: Sequence[SaveRow]) -> list[tuple[str, EventKind, str]]:
    events: list[tuple[str, EventKind, str]] = []

    for row in save:
        if row.bucket != "wrong_target":
            continue

        outcome = row.raw_outcome
        target = outcome.rsplit(":", 1)[-1]

        if ":contained:" in outcome:
            events.append((row.pair_id, "containment", target))
        elif outcome.startswith("asked:"):
            events.append((row.pair_id, "ask", target))
        elif outcome.startswith(("approved:supersede:", "reinforced:reinforce:")):
            events.append((row.pair_id, "relation suggestion", target))

    return sorted(events)


JUDGED_EVENT: dict[str, Literal["contained", "asked"]] = {
    "containment": "contained",
    "ask": "asked",
}


def action_verdict(kind: EventKind, judged: JudgedRow | None) -> Literal["yes", "no"] | None:
    if judged is None:
        return None

    return judged.new_covers_old if kind == "containment" else judged.one_record


def verdict_text(kind: EventKind, judged: JudgedRow | None) -> str:
    if kind == "relation suggestion" or judged is None:
        return "not judged"

    covers_key = judged.new_covers_old or "no agreement"

    if kind == "containment":
        return f"new_covers_old key {covers_key}"

    record_key = f"one_record key {judged.one_record or 'no agreement'}"

    if judged.new_covers_old is None:
        return record_key

    return f"new_covers_old key {covers_key}, {record_key}"


def unkeyed_events(save: Sequence[SaveRow], judged: Sequence[JudgedRow]) -> list[Unkeyed]:
    by_event = {(row.pair_id, row.noise_id, row.event): row for row in judged}
    events: list[Unkeyed] = []

    for pair_id, kind, noise_id in noise_events(save):
        event = JUDGED_EVENT.get(kind)
        row = None if event is None else by_event.get((pair_id, noise_id, event))
        events.append(
            Unkeyed(
                pair_id=pair_id,
                kind=kind,
                noise_id=noise_id,
                verdict=verdict_text(kind, row),
                action_verdict=action_verdict(kind, row),
            )
        )

    return events


def live_seed(row: SaveRow | None) -> NeighbourRow | None:
    if row is None or row.neighbours is None:
        return None

    return next((neighbour for neighbour in row.neighbours if neighbour.is_seed), None)


def flip_note(
    pair_id: str, probs: Mapping[str, SeedProbs], row: SaveRow | None, bucket: str
) -> str:
    prob = probs.get(pair_id)
    seed = live_seed(row)
    bench_cover = None if prob is None else prob.new_covers_old
    bench_subject = None if prob is None else prob.same_subject
    text = (
        f"{pair_id} ({bucket}): bench new_covers_old {_p(bench_cover)}, "
        f"same_subject {_p(bench_subject)}"
    )

    if seed is not None:
        text += (
            f"; live new_covers_old {_p(seed.new_covers_old)}, same_subject {_p(seed.same_subject)}"
        )

    return text


def _p(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def findings_for(check: Check) -> list[str]:
    found = [
        f"{bound.name} is {bound.actual}"
        + (
            f", {bound.judged} after the {bound.adjustment:+d} explained-pair adjustment"
            if bound.adjustment
            else ""
        )
        + f", bench {bound.expected} +-{bound.tolerance}"
        for bound in check.bounds
        if not bound.ok
    ]

    if check.superseded:
        found.append(f"superseded without a fold answer: {check.superseded}")

    if check.contained_on_noise:
        found.append(f"containment superseded a noise row: {check.contained_on_noise}")

    if check.end_state_mismatches:
        found.append(f"answered pairs with the wrong end state: {check.end_state_mismatches}")

    if check.asked_without_answer_row:
        found.append(f"asked pairs with no answer row: {check.asked_without_answer_row}")

    if check.ungraded_folds:
        found.append(f"folded pairs with no fold grade: {check.ungraded_folds}")

    return found


def fold_grade_counts(
    answered: Mapping[str, AnswerRow],
    grades: Mapping[str, fold.FoldGrade],
    screened: Collection[str] = (),
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    counts: Counter[str] = Counter()
    bad: list[str] = []
    ungraded: list[str] = []
    left_out: list[str] = []

    for pair_id, row in sorted(answered.items()):
        if row.end_state != "folded":
            continue

        grade = grades.get(pair_id)

        if grade is None:
            (left_out if pair_id in screened else ungraded).append(pair_id)
            continue

        counts[grade.status] += 1

        if grade.status == "bad":
            bad.append(pair_id)

    return dict(counts), bad, ungraded, left_out


def noise_only_asks_of(
    unkeyed: Sequence[Unkeyed], bench_ask: set[str], seed_asked: set[str], suppressed: set[str]
) -> list[Unkeyed]:
    return [
        event
        for event in unkeyed
        if event.kind == "ask"
        and event.pair_id in bench_ask
        and event.pair_id not in seed_asked
        and event.pair_id not in suppressed
    ]


def run_check(
    save: Sequence[SaveRow],
    answers: Sequence[AnswerRow],
    actions: Mapping[str, str],
    bench_ask: set[str],
    asks: Mapping[tuple[str, str], AskRow],
    arm: str = REPLAY_ARM,
    probs: Mapping[str, SeedProbs] | None = None,
    judged: Sequence[JudgedRow] = (),
    floor: float = CONTAINMENT_FLOOR,
    grades: Mapping[str, fold.FoldGrade] | None = None,
    screened: Collection[str] = (),
) -> Check:
    seed_probs = probs or {}
    rows = {row.pair_id: row for row in save}
    reinforced = pairs_in(save, "caught")
    superseded = pairs_in(save, "superseded")
    asked = pairs_in(save, "asked")
    bench_reinforce = {pair for pair, action in actions.items() if action == "reinforce"}
    seed_asked = set(asked)
    suppressed = suppressed_asks_of(save, bench_ask, seed_asked)
    unkeyed = unkeyed_events(save, judged)
    noise_only = noise_only_asks_of(unkeyed, bench_ask, seed_asked, suppressed)
    noise_credited = {event.pair_id for event in noise_only if event.action_verdict == "yes"}
    credited = suppressed | noise_credited
    open_ask = bench_ask - credited
    rejected = {event.pair_id for event in noise_only} - credited
    near = near_threshold(seed_probs, floor)
    cover_flips, ask_flips = near_flips_of(
        seed_probs, floor, covers_set(actions), open_ask - rejected, seed_asked
    )
    noisy = {
        event.pair_id
        for event in unkeyed
        if event.kind == "containment" and event.action_verdict == "yes"
    }
    explained = (set(cover_flips) | set(ask_flips) | noisy) - credited - rejected
    near_flips = sorted(set(cover_flips) | set(ask_flips))
    answered = answers_of(save, answers)
    mismatches, missing_rows, unreplied = end_state_checks(asked, answered, asks, arm)
    counts, bad_folds, ungraded, screened_folds = (
        ({}, [], [], []) if grades is None else fold_grade_counts(answered, grades, screened)
    )

    return Check(
        bounds=bounds_for(
            len(reinforced),
            len(superseded),
            len(seed_asked),
            len(open_ask),
            asked_adjustment=adjustment_for(open_ask, seed_asked, explained),
        ),
        reinforced=reinforced,
        superseded=superseded,
        contained_on_noise=[
            f"{event.pair_id} on {event.noise_id}"
            for event in unkeyed
            if event.kind == "containment"
        ],
        asked=asked,
        covers_asked=sorted(
            row.pair_id
            for row in save
            if row.bucket == "asked" and row.raw_outcome.startswith(COVERS_ASK_PREFIX)
        ),
        wrong_target=pairs_in(save, "wrong_target"),
        wrong_target_new_paths=sorted(
            {event.pair_id for event in unkeyed if event.kind != "relation suggestion"}
        ),
        asked_outside_bench=sorted(set(asked) - bench_ask),
        asked_missing=sorted(open_ask - seed_asked),
        asked_on_noise=sorted(noise_credited),
        ask_suppressed=sorted(suppressed),
        reinforced_outside_bench=sorted(set(reinforced) - bench_reinforce),
        reinforced_missing=sorted(bench_reinforce - set(reinforced)),
        near_threshold=sorted(near),
        near_flips=near_flips,
        near_flip_notes=[
            flip_note(
                pair, seed_probs, rows.get(pair), "asked" if pair in seed_asked else "not asked"
            )
            for pair in near_flips
        ],
        unkeyed=unkeyed,
        end_states=dict(Counter(row.end_state for row in answered.values())),
        end_state_mismatches=mismatches,
        asked_without_answer_row=missing_rows,
        no_reply=unreplied,
        fold_grades=counts,
        bad_folds=bad_folds,
        ungraded_folds=ungraded,
        screened_folds=screened_folds,
        noise_only_asks=[f"{event.pair_id}: {event.verdict}" for event in noise_only],
    )


def listed(title: str, pairs: Sequence[str]) -> list[str]:
    return [f"- {title}: {len(pairs)}" + (f" ({', '.join(pairs)})" if pairs else "")]


def render(check: Check, floor: float = CONTAINMENT_FLOOR) -> str:
    lines = ["# Production replay against the bench", "", f"Containment floor {floor:.2f}.", ""]
    lines += [
        "| bucket | actual | bench | tolerance | explained adjustment | "
        "judged (actual + adjustment) | within |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {bound.name} | {bound.actual} | {bound.expected} | {bound.tolerance} | "
        f"{bound.adjustment:+d} | {bound.judged} | {'yes' if bound.ok else 'NO'} |"
        for bound in check.bounds
    ]
    lines.append("")
    lines += listed("superseded without a fold answer", check.superseded)
    lines += listed("containment superseded a noise row", check.contained_on_noise)
    lines += listed(
        "asked outside the bench ask set (trigger and covers)", check.asked_outside_bench
    )
    lines += listed("asked in the bench ask set but not asked", check.asked_missing)
    lines += listed(
        "bench asks seen only on a noise row, not counted as asked", check.noise_only_asks
    )
    lines += listed(
        "of those, keyed one_record yes and subtracted from the asked bench", check.asked_on_noise
    )
    lines += listed(
        "bench asks suppressed by a replacement suggestion on a noise row, "
        "subtracted from the asked bench",
        check.ask_suppressed,
    )
    lines += listed("asked through covers", check.covers_asked)
    lines += listed("reinforced outside the bench set", check.reinforced_outside_bench)
    lines += listed("reinforced in the bench set but not reinforced", check.reinforced_missing)
    lines += listed("wrong_target (any path, informational)", check.wrong_target)
    lines += listed("asked pairs with no answer row", check.asked_without_answer_row)
    lines.append(f"- answer end states: {check.end_states}")
    lines.append(f"- fold grades on folded pairs: {check.fold_grades}")
    lines += listed("folds graded bad", check.bad_folds)
    lines += listed("folded pairs with no grade", check.ungraded_folds)
    lines += listed("not gradable (client-name screen)", check.screened_folds)
    lines += listed("end states that differ from the recorded reply", check.end_state_mismatches)
    lines.append("")
    lines.append(
        f"## Asked with no recorded reply: {len(check.no_reply)}, expired and saved as new, "
        "not findings"
    )
    lines += [f"- {pair_id}" for pair_id in check.no_reply]
    lines.append("")
    lines.append(
        f"## Excused near flips (bench probability within {NEAR_MARGIN:.2f} of a threshold, "
        f"on the side opposite to the outcome): {len(check.near_flips)} of "
        f"{len(check.near_threshold)} near pairs"
    )
    lines += [f"- {note}" for note in check.near_flip_notes]
    lines.append("")
    lines.append(
        f"## Events on noise rows: {len(check.unkeyed)}, a keyed yes excuses a bound shortfall"
    )
    lines += [
        f"- {event.pair_id}: {event.kind} on {event.noise_id[:8]}; {event.verdict}"
        for event in check.unkeyed
    ]
    lines.append("")
    lines.append("Findings: " + ("none" if check.ok else ""))
    lines += [f"- {finding}" for finding in check.findings]

    return "\n".join(lines) + "\n"


def load_save(path: Path) -> list[SaveRow]:
    return [row for row in read_jsonl(path, SaveRow) if row.system == "v2"]


def load_all_asks(base: Path) -> dict[tuple[str, str], AskRow]:
    experiments = base / "experiments"
    extra = load_asks(experiments / "prod-replay" / COVERS_ASKS_FILE)

    return extra | load_asks(experiments / "agent-fold" / "asks.jsonl")


def load_grades(base: Path, arm: str) -> dict[str, fold.FoldGrade]:
    experiments = base / "experiments"
    keys = (
        experiments / "prod-replay" / COVERS_FOLD_DIR / "key-fold.jsonl",
        experiments / "agent-fold" / "key-fold.jsonl",
    )
    grades: dict[str, fold.FoldGrade] = {}

    for key in keys:
        grades |= fold.writer_grades(fold.load_fold_grades(key), arm)

    return grades


def left_out_file(arm: str) -> str:
    return f"{LEFT_OUT_PREFIX}-{arm}.txt"


def load_screened(base: Path, arm: str) -> set[str]:
    path = base / "experiments" / "prod-replay" / COVERS_FOLD_DIR / left_out_file(arm)

    if not path.exists():
        return set()

    return {line for line in path.read_text(encoding="utf-8").splitlines() if line}


def superseded_in_runs(replay: Path) -> set[str]:
    files = [replay / "save-v2.jsonl", *sorted(replay.glob("run-*/save-v2.jsonl"))]

    return {
        row.pair_id
        for file in files
        if file.exists()
        for row in load_save(file)
        if row.bucket == "superseded"
    }


def pairs_needing_replies(
    base: Path, replay: Path, floor: float = CONTAINMENT_FLOOR, arm: str = REPLAY_ARM
) -> list[str]:
    wanted = covers_set(bench_actions(base, floor)) | superseded_in_runs(replay)
    recorded = load_asks(base / "experiments" / "agent-fold" / "asks.jsonl")

    return sorted(
        pair_id
        for pair_id in wanted
        if (pair_id, arm) not in recorded or not recorded[(pair_id, arm)].answered
    )


def run(
    base: Path,
    replay: Path,
    arm: str,
    floor: float = CONTAINMENT_FLOOR,
    judged_path: Path | None = None,
) -> Check:
    actions = bench_actions(base, floor)
    answers_path = replay / "answer-v2.jsonl"
    answers = read_jsonl(answers_path, AnswerRow) if answers_path.exists() else []
    judged_file = judged_path or base / "experiments" / "prod-replay" / JUDGED_FILE
    judged = read_jsonl(judged_file, JudgedRow) if judged_file.exists() else []

    return run_check(
        save=load_save(replay / "save-v2.jsonl"),
        answers=answers,
        actions=actions,
        bench_ask=bench_asked(base, actions),
        asks=load_all_asks(base),
        arm=arm,
        probs=bench_probs(base),
        judged=judged,
        floor=floor,
        grades=load_grades(base, arm),
        screened=load_screened(base, arm),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the production replay with the bench.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("--arm", default=REPLAY_ARM)
    parser.add_argument("--containment-floor", type=float, default=CONTAINMENT_FLOOR)
    parser.add_argument("--judged", type=Path, default=None)
    parser.add_argument("--needed-replies", type=Path, default=None)
    args = parser.parse_args()
    replay = args.replay or args.base / "experiments" / "prod-replay"

    if args.needed_replies is not None:
        needed = pairs_needing_replies(args.base, replay, args.containment_floor, args.arm)
        args.needed_replies.write_text(
            "".join(f"{pair_id}\n" for pair_id in needed), encoding="utf-8"
        )
        sys.stdout.write(f"{len(needed)} pairs need a reply\n")
        sys.exit(0)

    check = run(args.base, replay, args.arm, args.containment_floor, args.judged)
    text = render(check, args.containment_floor)

    (replay / "check.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    sys.exit(0 if check.ok else 1)


if __name__ == "__main__":
    main()
