from __future__ import annotations

import argparse
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval_v1v2 import fold, paths
from eval_v1v2.covers import (
    BASELINE,
    REPLACEMENT_BAR,
    RESTATEMENT_BAR,
    Case,
    Confusion,
    best_threshold,
    confusion_at,
    v1_counts,
)
from eval_v1v2.dup_rules import load_signals
from eval_v1v2.rows import Row, read_jsonl
from eval_v1v2.score_save import median

GUARD_QUESTIONS: tuple[str, ...] = (
    "fold_covers_old",
    "fold_covers_new",
    "sources_cover_fold",
    "fold_single",
)
GUARD_KEYS: dict[str, str] = {
    "fold_covers_old": "nothing_lost",
    "fold_covers_new": "nothing_lost",
    "sources_cover_fold": "nothing_invented",
    "fold_single": "one_record",
}
GUARD_KINDS: dict[str, tuple[str, ...]] = {
    "off": (),
    "coverage": ("fold_covers_old", "fold_covers_new"),
    "coverage+sources": ("fold_covers_old", "fold_covers_new", "sources_cover_fold"),
    "all": GUARD_QUESTIONS,
}
FLOORS: tuple[float, ...] = (0.75, 0.80, 0.85)
GATES: tuple[tuple[float | None, float | None], ...] = (
    (None, None),
    (0.70, None),
    (0.90, None),
    (None, 0.50),
    (None, 0.70),
    (None, 0.90),
)
REPLACEMENT_TRIGGERS: tuple[float | None, ...] = (None, 0.25, 0.35)
GUARD_THRESHOLDS: tuple[float, ...] = (0.70, 0.80, 0.90)
REPEAT_CROSSING = 0.70
Reading = Literal["strict", "loose"]
READINGS: tuple[Reading, ...] = ("strict", "loose")


class TimedAnswer(Row):
    answer: Literal["yes", "no"] | None
    p_yes: float | None
    probabilities: dict[str, float] | None
    error: str | None
    elapsed_ms: float


class GuardSignal(Row):
    pair_id: str
    writer: str
    run: int
    fold_covers_old: TimedAnswer
    fold_covers_new: TimedAnswer
    sources_cover_fold: TimedAnswer
    fold_single: TimedAnswer
    error: str | None

    def reading(self, name: str) -> TimedAnswer:
        answer: TimedAnswer = getattr(self, name)

        return answer

    @property
    def sequential_ms(self) -> float:
        return sum(self.reading(name).elapsed_ms for name in GUARD_QUESTIONS)

    @property
    def parallel_ms(self) -> float:
        return max(self.reading(name).elapsed_ms for name in GUARD_QUESTIONS)


class BandSignal(Row):
    pair_id: str
    run: int
    same_subject: TimedAnswer
    error: str | None


def load_guard_signals(path: Path) -> dict[tuple[str, str], GuardSignal]:
    if not path.exists():
        return {}

    return {(row.pair_id, row.writer): row for row in read_jsonl(path, GuardSignal)}


def load_band_signals(path: Path) -> dict[str, BandSignal]:
    if not path.exists():
        return {}

    return {row.pair_id: row for row in read_jsonl(path, BandSignal)}


@dataclass(frozen=True)
class GuardData:
    inputs: fold.Inputs
    guards: Mapping[tuple[str, str], GuardSignal]
    subjects: Mapping[str, BandSignal]
    p_replacement: Mapping[str, float]
    out_dir: Path


def load_data(base: Path, out_dir: Path) -> GuardData:
    inputs = fold.load_inputs(base, base / "experiments" / "fold")
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    replacement: dict[str, float] = {}

    for pair_id, signal in signals.items():
        neighbour = signal.existing

        if neighbour is not None and neighbour.probabilities is not None:
            replacement[pair_id] = neighbour.probabilities.get("replacement", 0.0)

    return GuardData(
        inputs=inputs,
        guards=load_guard_signals(out_dir / "guard-signals.jsonl"),
        subjects=load_band_signals(out_dir / "band-signals.jsonl"),
        p_replacement=replacement,
        out_dir=out_dir,
    )


@dataclass(frozen=True)
class BoundRule:
    rule: GuardRule
    writer: str
    data: GuardData

    @property
    def name(self) -> str:
        return self.rule.name

    def decide(self, case: Case) -> fold.Action:
        base = fold.CONTAINMENT.decide(case)

        if base != "new":
            return base

        return "fold" if self.fires(case) and self.accepts(case) else "new"

    def fires(self, case: Case) -> bool:
        if fold.CONTAINMENT.decide(case) != "new" or case.pick != "follow_up":
            return False

        similarity = case.similarity
        folded = self.data.inputs.folds.get((case.pair_id, self.writer))

        if similarity is None or similarity < fold.BAND_FLOOR or folded is None:
            return False

        if not folded.answered:
            return False

        return self.by_gate(case, similarity) or self.by_replacement(case)

    def by_gate(self, case: Case, similarity: float) -> bool:
        rule = self.rule

        if similarity < rule.floor:
            return False

        if rule.enhances is not None:
            reading = case.p_yes("enhances")

            return reading is not None and reading >= rule.enhances

        if rule.subject is not None:
            signal = self.data.subjects.get(case.pair_id)
            answer = None if signal is None else signal.same_subject.p_yes

            return answer is not None and answer >= rule.subject

        return True

    def by_replacement(self, case: Case) -> bool:
        threshold = self.rule.replacement

        return threshold is not None and self.data.p_replacement.get(case.pair_id, 0.0) >= threshold

    def accepts(self, case: Case) -> bool:
        rule = self.rule
        names = GUARD_KINDS[rule.kind]

        if not names or rule.threshold is None:
            return True

        signal = self.data.guards.get((case.pair_id, self.writer))

        if signal is None:
            return False

        for name in names:
            reading = signal.reading(name).p_yes

            if reading is None or reading < rule.threshold:
                return False

        return True


@dataclass(frozen=True)
class GuardRule:
    floor: float
    enhances: float | None
    subject: float | None
    replacement: float | None
    kind: str
    threshold: float | None

    @property
    def name(self) -> str:
        gate = (
            f" enhances>={self.enhances:.2f}"
            if self.enhances is not None
            else f" same_subject>={self.subject:.2f}"
            if self.subject is not None
            else ""
        )
        extra = (
            ""
            if self.replacement is None
            else f" OR p_replacement>={self.replacement:.2f} (band floor {fold.BAND_FLOOR:.2f})"
        )
        guard = "" if self.threshold is None else f" guards={self.kind}>={self.threshold:.2f}"

        return f"sim>={self.floor:.2f}{gate}{extra}{guard}"

    @property
    def knobs(self) -> int:
        return 1 + sum(
            value is not None
            for value in (self.enhances, self.subject, self.replacement, self.threshold)
        )

    def bind(self, writer: str, data: GuardData) -> BoundRule:
        return BoundRule(rule=self, writer=writer, data=data)


def guard_rule(
    floor: float,
    gate: tuple[float | None, float | None],
    replacement: float | None,
    kind: str,
    threshold: float | None,
) -> GuardRule:
    return GuardRule(
        floor=floor,
        enhances=gate[0],
        subject=gate[1],
        replacement=replacement,
        kind=kind,
        threshold=threshold,
    )


def all_guard_rules() -> list[GuardRule]:
    guards: list[tuple[str, float | None]] = [("off", None)] + [
        (kind, threshold) for kind in GUARD_KINDS if kind != "off" for threshold in GUARD_THRESHOLDS
    ]

    return [
        guard_rule(floor, gate, replacement, kind, threshold)
        for floor in FLOORS
        for gate in GATES
        for replacement in REPLACEMENT_TRIGGERS
        for kind, threshold in guards
    ]


class GuardScore(BaseModel):
    result: fold.FoldScore
    fired: list[str] = []
    rejected_good: list[str] = []
    rejected_bad: list[str] = []
    rejected_disputed: list[str] = []

    @property
    def accepted(self) -> int:
        return len(self.result.triggered)

    def resolved(self, reading: Reading) -> int:
        return (
            self.result.resolved_restatement
            if reading == "strict"
            else self.result.loose_restatement
        )

    def caught(self, reading: Reading) -> int:
        return (
            self.result.caught_replacement if reading == "strict" else self.result.loose_replacement
        )

    def errors(self, reading: Reading) -> int:
        return self.result.strict_errors if reading == "strict" else self.result.errors

    def meets_bars(self, reading: Reading) -> bool:
        return (
            self.resolved(reading) >= RESTATEMENT_BAR
            and self.caught(reading) >= REPLACEMENT_BAR
            and self.errors(reading) == 0
        )


def score_guard(
    rule: GuardRule, data: GuardData, writer: str, only: frozenset[str] | None = None
) -> GuardScore:
    inputs = data.inputs
    bound = rule.bind(writer, data)
    grades = fold.writer_grades(inputs.grades, writer)
    score = GuardScore(
        result=fold.score_fold(bound, inputs.cases, inputs.relation, inputs.cover, grades, only)
    )
    accepted = set(score.result.triggered)

    for pair_id, case in inputs.cases.items():
        if (only is not None and pair_id not in only) or not bound.fires(case):
            continue

        score.fired.append(pair_id)

        if pair_id in accepted:
            continue

        status = grades.get(pair_id, fold.FoldGrade()).status
        {
            "good": score.rejected_good,
            "bad": score.rejected_bad,
            "disputed": score.rejected_disputed,
        }[status].append(pair_id)

    return score


def shortfall(resolved: int, caught: int) -> int:
    return max(0, RESTATEMENT_BAR - resolved) + max(0, REPLACEMENT_BAR - caught)


def rank_reading(score: GuardScore, reading: Reading) -> tuple[int, int, int, int]:
    resolved, caught = score.resolved(reading), score.caught(reading)

    return (
        shortfall(resolved, caught),
        score.errors(reading),
        -(resolved + caught),
        score.accepted,
    )


def best_guard_rule(
    rules: Sequence[GuardRule],
    data: GuardData,
    writer: str,
    reading: Reading,
    only: frozenset[str] | None = None,
) -> GuardRule:
    return min(rules, key=lambda rule: rank_reading(score_guard(rule, data, writer, only), reading))


def best_clean_rule(
    scored: Sequence[tuple[GuardRule, GuardScore]], reading: Reading
) -> tuple[GuardRule, GuardScore] | None:
    clean = [pair for pair in scored if pair[1].errors(reading) == 0]

    if not clean:
        return None

    return min(
        clean,
        key=lambda pair: (-pair[1].resolved(reading), -pair[1].caught(reading), pair[1].accepted),
    )


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def fixed(value: float | None, places: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def split_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none"


def sweep_row(rule: GuardRule, writer: str, score: GuardScore, total: int) -> str:
    base = score.result.base

    return (
        f"| {rule.name} | {writer} | {rule.knobs} "
        f"| {_ratio(score.result.resolved_restatement, base.restatements)} "
        f"({base.reinforced_restatement}+{base.superseded_restatement}"
        f"+{score.result.folded_good_restatement}) "
        f"| {_ratio(score.result.caught_replacement, base.replacements)} "
        f"({base.flagged_replacement}+{base.superseded_replacement}"
        f"+{score.result.folded_good_replacement}) "
        f"| {_ratio(score.result.loose_restatement, base.restatements)} "
        f"| {_ratio(score.result.loose_replacement, base.replacements)} "
        f"| {len(score.fired)} ({len(score.fired) / total:.1%}) "
        f"| {score.accepted} ({score.accepted / total:.1%}) "
        f"| {len(score.result.bad)} | {split_counts(score.result.bad_by_answer)} "
        f"| {split_counts(score.result.bad_by_label)} | {len(score.result.disputed)} "
        f"| {len(score.rejected_good)}/{len(score.rejected_bad)}/{len(score.rejected_disputed)} "
        f"| {score.result.follow_up_folds} ({score.result.follow_up_one_record_yes}/"
        f"{score.result.follow_up_one_record_no}) "
        f"| {len(score.result.distinct_folds)} ({len(score.result.distinct_not_one_record)}) "
        f"| {base.wrong} | {base.lost} | {base.false_flag} "
        f"| {'YES' if score.meets_bars('strict') else ''} "
        f"| {'YES' if score.meets_bars('loose') else ''} |"
    )


SWEEP_HEAD = (
    "| rule | writer | knobs on | restatements resolved (reinforced+superseded+folded good) "
    "| replacements caught (flagged+superseded+folded) | loose restatements "
    "| loose replacements | folds fired (share of 242) | folds accepted (share of 242) "
    "| accepted bad | failing answer | relation label | accepted disputed "
    "| rejected good/bad/disputed | accepted follow_up folds (one_record yes/no) "
    "| accepted distinct folds (not keyed one_record yes) | wrong supersedes | lost "
    "| false flags | meets bars strict | meets bars loose |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def scored_rules(
    rules: Sequence[GuardRule], data: GuardData, writer: str
) -> list[tuple[GuardRule, GuardScore]]:
    return [(rule, score_guard(rule, data, writer)) for rule in rules]


def sweep_text(
    data: GuardData, scored: Mapping[str, Sequence[tuple[GuardRule, GuardScore]]]
) -> str:
    total = len(data.inputs.cases)
    lines = [
        "# Sweep",
        "",
        "Rules are ranked by the strict reading: shortfall against the bars, then errors "
        "(accepted bad and disputed folds, wrong supersedes, lost records), then total resolved. "
        "A fired fold whose guards fail is saved as new, so it counts as neither resolved nor bad.",
        "",
        SWEEP_HEAD,
    ]

    for writer, rows in scored.items():
        ordered = sorted(rows, key=lambda pair: rank_reading(pair[1], "strict"))
        lines += [sweep_row(rule, writer, score, total) for rule, score in ordered]

    return "\n".join(lines)


def accepted_bad_text(
    data: GuardData, picks: Sequence[tuple[str, Reading, GuardRule, GuardScore]]
) -> str:
    inputs = data.inputs
    lines = [
        "# Accepted bad and disputed folds",
        "",
        "Holds record text; stays under dev-corpus. For the best rule of each writer and reading, "
        "every fold the rule accepts whose judge key has a no (bad) or a missing answer "
        "(disputed). Each entry lists the failing answer, the relation label, the two sources and "
        "the fold.",
        "",
    ]
    seen: set[tuple[str, str]] = set()

    for writer, reading, rule, score in picks:
        grades = fold.writer_grades(inputs.grades, writer)
        lines += [f"## {writer}, best rule under the {reading} reading: `{rule.name}`", ""]
        entries = 0

        for pair_id in score.result.triggered:
            grade = grades.get(pair_id, fold.FoldGrade())

            if grade.status == "good":
                continue

            entries += 1
            row = next(item for item in inputs.band if item.pair_id == pair_id)
            folded = inputs.folds[(pair_id, writer)]
            duplicate = (writer, pair_id) in seen
            seen.add((writer, pair_id))
            label = inputs.relation.get(pair_id, "disputed")
            lines += [
                f"### {pair_id} ({writer}, relation key: {label}, {grade.status}"
                f"{', also listed above' if duplicate else ''})",
                "",
                f"Failing: {', '.join(grade.failing) or 'none'}. "
                f"Answers: {fold.answers_text(grade)}.",
                "",
                f"- existing: {' '.join(row.existing.split())}",
                f"- incoming: {' '.join(row.incoming.split())}",
                f"- fold: {' '.join((folded.text or '').split())}",
                "",
            ]

        if entries == 0:
            lines += ["No accepted bad or disputed folds.", ""]

    return "\n".join(lines)


def guard_confusions(data: GuardData, name: str, writer: str | None) -> list[tuple[bool, float]]:
    found: list[tuple[bool, float]] = []

    for (pair_id, who), signal in data.guards.items():
        if writer is not None and who != writer:
            continue

        truth = data.inputs.grades.get((pair_id, who), fold.FoldGrade()).answers.get(
            GUARD_KEYS[name]
        )
        reading = signal.reading(name).p_yes

        if truth is not None and reading is not None:
            found.append((truth, reading))

    return found


def _cells(table: Confusion) -> str:
    return (
        f"{table.true_yes_pred_yes} | {table.true_yes_pred_no} "
        f"| {table.true_no_pred_yes} | {table.true_no_pred_no}"
    )


def guard_accuracy_table(data: GuardData) -> str:
    lines = [
        "| guard | judge answer it is scored against | writer | folds keyed | share keyed yes "
        "| accuracy at 0.50 | best threshold | accuracy there | key yes, guard yes "
        "| key yes, guard no | key no, guard yes | key no, guard no |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for name in GUARD_QUESTIONS:
        for writer in (None, *fold.WRITERS):
            pairs = guard_confusions(data, name, writer)

            if not pairs:
                continue

            threshold, table = best_threshold(pairs)
            at_half = confusion_at(pairs, 0.5)
            yes = sum(1 for truth, _ in pairs if truth)
            lines.append(
                f"| {name} | {GUARD_KEYS[name]} | {writer or 'both'} | {len(pairs)} "
                f"| {yes / len(pairs):.0%} | {at_half.accuracy:.1%} | {threshold:.2f} "
                f"| {table.accuracy:.1%} | {_cells(table)} |"
            )

    return "\n".join(lines)


def same_subject_table(data: GuardData) -> str:
    lines = [
        "| relation key | band pairs | mean p(yes) | p>=0.50 | p>=0.70 | p>=0.90 |",
        "|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[float]] = {}

    for row in data.inputs.band:
        signal = data.subjects.get(row.pair_id)

        if signal is not None and signal.same_subject.p_yes is not None:
            groups.setdefault(row.relation or "disputed", []).append(signal.same_subject.p_yes)

    for label, values in sorted(groups.items()):
        lines.append(
            f"| {label} | {len(values)} | {statistics.fmean(values):.2f} "
            f"| {sum(v >= 0.5 for v in values)} | {sum(v >= 0.7 for v in values)} "
            f"| {sum(v >= 0.9 for v in values)} |"
        )

    return "\n".join(lines)


def repeat_rows(
    first: Mapping[str, TimedAnswer], second: Mapping[str, TimedAnswer]
) -> tuple[int, int, float, float, int]:
    gaps: list[float] = []
    flips = crossings = 0

    for key in sorted(set(first) & set(second)):
        one, two = first[key], second[key]

        if one.p_yes is None or two.p_yes is None:
            continue

        gaps.append(abs(one.p_yes - two.p_yes))
        flips += one.answer != two.answer
        crossings += (one.p_yes >= REPEAT_CROSSING) != (two.p_yes >= REPEAT_CROSSING)

    mean = sum(gaps) / len(gaps) if gaps else 0.0

    return len(gaps), flips, mean, max(gaps, default=0.0), crossings


def repeatability_table(data: GuardData) -> str:
    repeat_guards = load_guard_signals(data.out_dir / "guard-signals-repeat.jsonl")
    repeat_band = load_band_signals(data.out_dir / "band-signals-repeat.jsonl")
    lines = [
        "| question | items run twice | answers that flipped | mean absolute change in p(yes) "
        "| largest change | items crossing 0.70 |",
        "|---|---|---|---|---|---|",
    ]

    for name in GUARD_QUESTIONS:
        first = {f"{pid}~{who}": signal.reading(name) for (pid, who), signal in data.guards.items()}
        second = {
            f"{pid}~{who}": signal.reading(name) for (pid, who), signal in repeat_guards.items()
        }
        count, flips, mean, largest, crossings = repeat_rows(first, second)
        lines.append(f"| {name} | {count} | {flips} | {mean:.3f} | {largest:.3f} | {crossings} |")

    count, flips, mean, largest, crossings = repeat_rows(
        {pid: signal.same_subject for pid, signal in data.subjects.items()},
        {pid: signal.same_subject for pid, signal in repeat_band.items()},
    )
    lines.append(f"| same_subject | {count} | {flips} | {mean:.3f} | {largest:.3f} | {crossings} |")

    return "\n".join(lines)


def latency_text(data: GuardData, accepted: Mapping[str, Sequence[str]]) -> str:
    lines = [
        "| scope | folds | same_subject median ms | four guards, one after another, median ms "
        "| four guards in parallel, median ms | median added per fold, sequential "
        "| median added per fold, guards in parallel |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(label: str, keys: Sequence[tuple[str, str]]) -> str:
        signals = [data.guards[key] for key in keys if key in data.guards]
        subject = [
            data.subjects[key[0]].same_subject.elapsed_ms for key in keys if key[0] in data.subjects
        ]
        sequential = [
            data.subjects[s.pair_id].same_subject.elapsed_ms + s.sequential_ms
            for s in signals
            if s.pair_id in data.subjects
        ]
        parallel = [
            data.subjects[s.pair_id].same_subject.elapsed_ms + s.parallel_ms
            for s in signals
            if s.pair_id in data.subjects
        ]

        return (
            f"| {label} | {len(signals)} | {fixed(median(subject))} "
            f"| {fixed(median([s.sequential_ms for s in signals]))} "
            f"| {fixed(median([s.parallel_ms for s in signals]))} "
            f"| {fixed(median(sequential))} | {fixed(median(parallel))} |"
        )

    lines.append(row("all folds", sorted(data.guards)))

    for writer, pair_ids in accepted.items():
        lines.append(
            row(f"accepted folds of the best {writer} rule", [(p, writer) for p in pair_ids])
        )

    return "\n".join(lines)


def headline_line(name: str, score: GuardScore, reading: Reading) -> str:
    base = score.result.base

    return (
        f"| {name} | {_ratio(score.resolved(reading), base.restatements)} "
        f"| {_ratio(score.caught(reading), base.replacements)} "
        f"| {len(score.result.bad)} | {len(score.result.disputed)} | {base.wrong} "
        f"| {base.lost_record}/{base.unrelated} | {base.lost_update}/{base.replacements} "
        f"| {score.accepted} |"
    )


HEADLINE_HEAD = (
    "| system | restatements resolved | replacements caught | accepted bad folds "
    "| accepted disputed folds | wrong supersedes | lost records | lost updates "
    "| folds accepted |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


EXPERIMENT_4_LOOSE = GuardRule(0.80, 0.70, None, None, "off", None)


def headline_table(
    data: GuardData,
    best: Mapping[tuple[str, Reading], GuardRule],
    scores: Mapping[tuple[str, Reading], GuardScore],
    guarded: Mapping[tuple[str, Reading], tuple[GuardRule, GuardScore]],
) -> str:
    inputs = data.inputs
    empty: fold.Grades = {}
    today = fold.score_fold(BASELINE, inputs.cases, inputs.relation, inputs.cover, empty)
    contained = fold.score_fold(
        fold.CONTAINMENT, inputs.cases, inputs.relation, inputs.cover, empty
    )
    v1 = v1_counts(inputs.published_v1, inputs.relation)
    lines = [
        HEADLINE_HEAD,
        headline_line("v2 today (Jev pick)", GuardScore(result=today), "strict"),
        headline_line("v2 plus containment rule", GuardScore(result=contained), "strict"),
        "| v1 as published "
        f"| {_ratio(v1.resolved_restatement, v1.restatements)} "
        f"| {_ratio(v1.flagged_replacement, v1.replacements)} | n/a | n/a | n/a "
        f"| {v1.lost_record}/{v1.unrelated} | {v1.lost_update}/{v1.replacements} | n/a |",
    ]

    for writer in fold.WRITERS:
        loose = score_guard(EXPERIMENT_4_LOOSE, data, writer)
        lines.append(
            headline_line(
                f"experiment 4 loose rule, {writer}, strict: {EXPERIMENT_4_LOOSE.name}",
                loose,
                "strict",
            )
        )

    for writer in fold.WRITERS:
        for reading in READINGS:
            rule = best[(writer, reading)]
            lines.append(
                headline_line(
                    f"best rule in the sweep, {writer}, {reading}: {rule.name}",
                    scores[(writer, reading)],
                    reading,
                )
            )

    for writer in fold.WRITERS:
        for reading in READINGS:
            rule, score = guarded[(writer, reading)]
            lines.append(
                headline_line(
                    f"best rule with a guard on, {writer}, {reading}: {rule.name}", score, reading
                )
            )

    return "\n".join(lines)


def loso_table(
    rules: Sequence[GuardRule], data: GuardData, writer: str, reading: Reading
) -> tuple[str, dict[str, int]]:
    inputs = data.inputs
    sources = {pid: inputs.pairs[pid].source for pid in inputs.cases}
    everyone = frozenset(sources)
    lines = [
        "| held-out source | rule chosen on the other sources | restatements resolved "
        "| replacements caught | accepted bad | accepted disputed | wrong supersedes | lost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    pooled = {"restatements": 0, "replacements": 0, "bad": 0, "disputed": 0, "wrong": 0, "lost": 0}
    totals = {"restatements": 0, "replacements": 0}

    for held in sorted(set(sources.values())):
        test = frozenset(pid for pid, source in sources.items() if source == held)
        chosen = best_guard_rule(rules, data, writer, reading, everyone - test)
        score = score_guard(chosen, data, writer, test)
        base = score.result.base
        pooled["restatements"] += score.resolved(reading)
        pooled["replacements"] += score.caught(reading)
        pooled["bad"] += len(score.result.bad)
        pooled["disputed"] += len(score.result.disputed)
        pooled["wrong"] += base.wrong
        pooled["lost"] += base.lost
        totals["restatements"] += base.restatements
        totals["replacements"] += base.replacements
        lines.append(
            f"| {held} | {chosen.name} | {_ratio(score.resolved(reading), base.restatements)} "
            f"| {_ratio(score.caught(reading), base.replacements)} | {len(score.result.bad)} "
            f"| {len(score.result.disputed)} | {base.wrong} | {base.lost} |"
        )

    lines.append(
        f"| pooled held-out | | {_ratio(pooled['restatements'], totals['restatements'])} "
        f"| {_ratio(pooled['replacements'], totals['replacements'])} | {pooled['bad']} "
        f"| {pooled['disputed']} | {pooled['wrong']} | {pooled['lost']} |"
    )

    return "\n".join(lines), pooled


def miss_text(score: GuardScore, reading: Reading) -> str:
    misses: list[str] = []
    gap_rs = RESTATEMENT_BAR - score.resolved(reading)
    gap_rp = REPLACEMENT_BAR - score.caught(reading)

    if gap_rs > 0:
        misses.append(f"restatements by {gap_rs}")

    if gap_rp > 0:
        misses.append(f"replacements by {gap_rp}")

    if score.result.bad:
        misses.append(f"{len(score.result.bad)} accepted bad fold(s)")

    if reading == "strict" and score.result.disputed:
        misses.append(f"{len(score.result.disputed)} accepted disputed fold(s)")

    if score.result.base.wrong:
        misses.append(f"{score.result.base.wrong} wrong supersede(s)")

    if score.result.base.lost:
        misses.append(f"{score.result.base.lost} lost record(s) or update(s)")

    return ", ".join(misses) if misses else "nothing"


def verdict_text(
    best: Mapping[tuple[str, Reading], GuardRule],
    scores: Mapping[tuple[str, Reading], GuardScore],
    winners: Mapping[tuple[str, Reading], int],
) -> str:
    sentences: list[str] = []

    for writer in fold.WRITERS:
        for reading in READINGS:
            key = (writer, reading)
            sentences.append(
                f"{writer}, {reading}: {winners[key]} rule(s) reach {RESTATEMENT_BAR}+ "
                f"restatements and {REPLACEMENT_BAR} replacements with zero accepted bad folds"
                f"{' and zero accepted disputed folds' if reading == 'strict' else ''}, zero "
                f"wrong supersedes and zero lost records. Closest rule `{best[key].name}` "
                f"resolves {scores[key].resolved(reading)} restatements and catches "
                f"{scores[key].caught(reading)} replacements; it misses on "
                f"{miss_text(scores[key], reading)}."
            )

    return " ".join(sentences)


EFFECT_TRIGGERS: tuple[GuardRule, ...] = (
    EXPERIMENT_4_LOOSE,
    GuardRule(0.80, None, 0.70, None, "off", None),
    GuardRule(0.80, None, 0.70, 0.25, "off", None),
)


def guard_effect_table(data: GuardData) -> str:
    lines = [
        "| writer | trigger | guards | threshold | folds fired | accepted | accepted bad "
        "| accepted disputed | good folds rejected | bad folds rejected "
        "| restatements resolved (strict) | replacements caught (strict) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for writer in fold.WRITERS:
        for trigger in EFFECT_TRIGGERS:
            variants = [(trigger.kind, None)] + [
                (kind, threshold)
                for kind in GUARD_KINDS
                if kind != "off"
                for threshold in GUARD_THRESHOLDS
            ]

            for kind, threshold in variants:
                rule = GuardRule(
                    trigger.floor, trigger.enhances, trigger.subject, trigger.replacement,
                    kind, threshold,
                )  # fmt: skip
                score = score_guard(rule, data, writer)
                lines.append(
                    f"| {writer} | {rule.name.split(' guards=')[0]} | {kind} "
                    f"| {'' if threshold is None else f'{threshold:.2f}'} | {len(score.fired)} "
                    f"| {score.accepted} | {len(score.result.bad)} | {len(score.result.disputed)} "
                    f"| {len(score.rejected_good)} | {len(score.rejected_bad)} "
                    f"| {score.resolved('strict')}/{score.result.base.restatements} "
                    f"| {score.caught('strict')}/{score.result.base.replacements} |"
                )

    return "\n".join(lines)


def clean_lines(
    scored: Mapping[str, Sequence[tuple[GuardRule, GuardScore]]],
) -> list[str]:
    lines = [
        "| writer | reading | rule with zero errors that resolves the most restatements "
        "| restatements | replacements | folds accepted | knobs on |",
        "|---|---|---|---|---|---|---|",
    ]

    for writer, rows in scored.items():
        for reading in READINGS:
            pick = best_clean_rule(rows, reading)

            if pick is None:
                lines.append(f"| {writer} | {reading} | none | | | | |")
                continue

            rule, score = pick
            lines.append(
                f"| {writer} | {reading} | {rule.name} "
                f"| {_ratio(score.resolved(reading), score.result.base.restatements)} "
                f"| {_ratio(score.caught(reading), score.result.base.replacements)} "
                f"| {score.accepted} | {rule.knobs} |"
            )

    return lines


def best_detail(
    data: GuardData, writer: str, reading: Reading, rule: GuardRule, score: GuardScore
) -> list[str]:
    total = len(data.inputs.cases)
    fold_score = score.result

    return [
        f"Folds that fire: {len(score.fired)} of {total} saves "
        f"({len(score.fired) / total:.1%}); accepted after guards: {score.accepted} "
        f"({score.accepted / total:.1%}). Good {len(fold_score.good)}, bad "
        f"{len(fold_score.bad)}, disputed {len(fold_score.disputed)}.",
        "",
        f"Accepted bad folds by failing answer: {split_counts(fold_score.bad_by_answer)}. "
        f"By relation key: {split_counts(fold_score.bad_by_label)}.",
        "",
        f"Good folds the guards rejected: {len(score.rejected_good)}; bad folds rejected: "
        f"{len(score.rejected_bad)}; disputed folds rejected: {len(score.rejected_disputed)}.",
        "",
        f"Accepted folds on true follow_up pairs: {fold_score.follow_up_folds}; one_record keyed "
        f"yes {fold_score.follow_up_one_record_yes}, keyed no "
        f"{fold_score.follow_up_one_record_no}.",
        "",
        f"Accepted folds on true distinct pairs: {len(fold_score.distinct_folds)}; not keyed "
        f"one_record yes: {len(fold_score.distinct_not_one_record)}.",
        "",
        f"Knobs on: {rule.knobs} (floor{'' if rule.enhances is None else ', enhances'}"
        f"{'' if rule.subject is None else ', same_subject'}"
        f"{'' if rule.replacement is None else ', p_replacement'}"
        f"{'' if rule.threshold is None else ', guard threshold'}) of the "
        f"{len(all_guard_rules())} rules in the sweep.",
    ]


def summary_text(
    data: GuardData,
    rules: Sequence[GuardRule],
    scored: Mapping[str, Sequence[tuple[GuardRule, GuardScore]]],
) -> str:
    best: dict[tuple[str, Reading], GuardRule] = {}
    scores: dict[tuple[str, Reading], GuardScore] = {}
    winners: dict[tuple[str, Reading], int] = {}
    guarded: dict[tuple[str, Reading], tuple[GuardRule, GuardScore]] = {}

    for writer, rows in scored.items():
        for reading in READINGS:
            guarded[(writer, reading)] = min(
                (pair for pair in rows if pair[0].kind != "off"),
                key=lambda pair: rank_reading(pair[1], reading),
            )
            rule, score = min(rows, key=lambda pair: rank_reading(pair[1], reading))
            best[(writer, reading)] = rule
            scores[(writer, reading)] = score
            winners[(writer, reading)] = sum(1 for _, s in rows if s.meets_bars(reading))

    accepted = {writer: scores[(writer, "strict")].result.triggered for writer in fold.WRITERS}
    lines = [
        "# Experiment 5: fold guards",
        "",
        verdict_text(best, scores, winners),
        "",
        f"Pairs: {len(data.inputs.cases)}; relation-keyed: {len(data.inputs.relation)}; folds "
        f"with guard signals: {len(data.guards)} of {len(data.inputs.folds)}; band pairs with "
        f"same_subject: {len(data.subjects)} of {len(data.inputs.band)}. Rules per writer: "
        f"{len(rules)}.",
        "",
        "Strict: an accepted disputed fold is not resolved and counts as an error, as does an "
        "accepted bad one. Loose: an accepted fold counts as resolved unless a judge pair keyed a "
        "needed answer no, and only accepted bad folds count against. Rules rank by shortfall "
        f"against the bars ({RESTATEMENT_BAR} restatements, {REPLACEMENT_BAR} replacements), then "
        "by errors under the reading, then by total resolved.",
        "",
        "A p_replacement term is an OR trigger. A save in the band also fires when its "
        f"p(replacement) reaches the threshold, and that path uses the {fold.BAND_FLOOR} band "
        "floor, not the rule's own similarity floor. In `sim>=0.80 same_subject>=0.70 OR "
        "p_replacement>=0.25 (band floor 0.75)` the 0.80 floor applies only to the "
        "same_subject path.",
        "",
        "## Headline",
        "",
        headline_table(data, best, scores, guarded),
        "",
        "## What each guard does to a fixed trigger",
        "",
        guard_effect_table(data),
        "",
        "## Zero-error rule that resolves the most restatements",
        "",
        *clean_lines(scored),
        "",
        "The zero-error Haiku rule accepts 2 folds and resolves 28 restatements, one more than "
        "the containment rule's 27.",
        "",
        "## Guard accuracy against the matching judge answer",
        "",
        guard_accuracy_table(data),
        "",
        "fold_covers_old and fold_covers_new are both scored against nothing_lost, which asks "
        "about both sources at once, so each guard sees half of what the key asks.",
        "",
        "## same_subject by relation key (no judge answer to score it against)",
        "",
        same_subject_table(data),
        "",
        "## Repeatability on a 40-item sample (two runs)",
        "",
        repeatability_table(data),
        "",
        "## Added Jev latency per fold",
        "",
        latency_text(data, accepted),
        "",
        "The writer's own latency is in experiment 4 and is not repeated here.",
        "",
    ]

    for writer in fold.WRITERS:
        for reading in READINGS:
            key = (writer, reading)
            lines += [
                f"## Best rule, {writer}, {reading}: `{best[key].name}`",
                "",
                *best_detail(data, writer, reading, best[key], scores[key]),
                "",
            ]
            table, pooled = loso_table(rules, data, writer, reading)
            lines += [
                f"### Leave one source out, {writer}, {reading}",
                "",
                table,
                "",
                f"Pooled held-out: {pooled['restatements']} restatements, "
                f"{pooled['replacements']} replacements, {pooled['bad']} accepted bad folds, "
                f"{pooled['disputed']} accepted disputed.",
                "",
            ]

    hand_check = data.out_dir / "hand-check.md"

    if hand_check.exists():
        lines += ["## Hand check", "", hand_check.read_text(encoding="utf-8").strip(), ""]

    lines += [
        "## Files",
        "",
        "- guard-signals.jsonl, band-signals.jsonl, guard-signals-repeat.jsonl, "
        "band-signals-repeat.jsonl, sweep.md, accepted-bad.md, summary.md, hand-check.md, "
        "run-signals.sh, signals-run.log",
        "",
    ]

    return "\n".join(lines)


def pick_rows(
    scored: Mapping[str, Sequence[tuple[GuardRule, GuardScore]]],
) -> list[tuple[str, Reading, GuardRule, GuardScore]]:
    picks: list[tuple[str, Reading, GuardRule, GuardScore]] = []

    for writer, rows in scored.items():
        for reading in READINGS:
            rule, score = min(rows, key=lambda pair: rank_reading(pair[1], reading))
            picks.append((writer, reading, rule, score))

    return picks


def unfinished(data: GuardData) -> tuple[int, int]:
    folds_missing = sum(
        1
        for key, row in data.inputs.folds.items()
        if row.answered and (key not in data.guards or data.guards[key].error is not None)
    )
    pairs_missing = sum(
        1
        for row in data.inputs.band
        if row.pair_id not in data.subjects or data.subjects[row.pair_id].error is not None
    )

    return folds_missing, pairs_missing


def run_score(base: Path, out_dir: Path) -> int:
    data = load_data(base, out_dir)
    folds_missing, pairs_missing = unfinished(data)

    if folds_missing or pairs_missing:
        print(
            f"signals incomplete: {folds_missing} folds without guards, {pairs_missing} band "
            "pairs without same_subject; run run-signals.sh again",
            flush=True,
        )

        return 2

    rules = all_guard_rules()
    scored = {writer: scored_rules(rules, data, writer) for writer in fold.WRITERS}
    out_dir.joinpath("sweep.md").write_text(sweep_text(data, scored) + "\n", encoding="utf-8")
    out_dir.joinpath("accepted-bad.md").write_text(
        accepted_bad_text(data, pick_rows(scored)) + "\n", encoding="utf-8"
    )
    out_dir.joinpath("summary.md").write_text(
        summary_text(data, rules, scored) + "\n", encoding="utf-8"
    )
    print("scored; wrote sweep.md, accepted-bad.md, summary.md", flush=True)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold guard experiment: scorer.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "fold-guard"
    )
    args = parser.parse_args()
    code = run_score(args.base, args.out_dir)

    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
