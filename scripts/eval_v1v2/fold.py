from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from eval_v1v2 import paths
from eval_v1v2.covers import (
    BASELINE,
    REPLACEMENT_BAR,
    RESTATEMENT_BAR,
    Case,
    Rule,
    Score,
    build_cases,
    jev_rule,
    load_cover_key,
    load_covers,
    score_rule,
    v1_counts,
)
from eval_v1v2.covers_calls import load_sonnet_calls
from eval_v1v2.dup_rules import load_signals
from eval_v1v2.judges import Item, agreement_stats, run_task
from eval_v1v2.rows import KeyRow, LabelRow, PairRow, Row, SaveRow, read_jsonl, write_jsonl
from eval_v1v2.score_save import load_key, median, percentile
from eval_v1v2.second_judge import CALL_TIMEOUT, CONCURRENCY, MAX_ATTEMPTS
from tabularium.engine.judge import JudgeUnavailable, SubprocessJudgeAdapter

Action = Literal["reinforce", "supersede", "flag", "new", "fold"]
Grade = Literal["good", "bad", "disputed"]

WRITERS: tuple[str, ...] = ("haiku", "sonnet")
QUESTIONS: tuple[str, ...] = ("nothing_lost", "nothing_invented", "newer_wins", "one_record")
BAND_FLOOR = 0.75
SIMILARITY_FLOORS: tuple[float, ...] = (0.75, 0.80, 0.85)
ENHANCES_FLOORS: tuple[float | None, ...] = (None, 0.70, 0.80, 0.90)
NON_DISTINCT = frozenset({"restatement", "replacement", "follow_up", "unclear"})
PICK_SETS: dict[str, frozenset[str]] = {
    "follow_up only": frozenset({"follow_up"}),
    "follow_up or unclear": frozenset({"follow_up", "unclear"}),
    "any non-distinct": NON_DISTINCT,
}
CONTAINMENT_THRESHOLD = 0.90
STUCK = (
    "bounce-0796",
    "bounce-0872",
    "bounce-0990",
    "bounce-1010",
    "bounce-1100",
    "bounce-1140",
    "bounce-1242",
    "bounce-1260",
    "bounce-1370",
    "jev-c112",
    "jev-c113",
)
STOP_AFTER_FAILURES = 6
OLDER_TOKEN = "<<OLDER>>"
NEWER_TOKEN = "<<NEWER>>"

WRITER_PROMPT = """\
You merge two short notes from a memory store into one note. OLDER was saved first. NEWER arrives
now and is current.

Write one note that holds everything from both.

- Keep every concrete detail from both notes: names, numbers, paths, config keys, dates, rejected
  options and the reasons they were rejected.
- Add nothing that is in neither note. No new advice, no inference, no framing of your own.
- Where the notes conflict, NEWER wins. Give the newer value and say that the older value was
  replaced.
- Write it as one fact or one decision, not as two notes joined together.
- Reply with the merged note text only. No preamble, no heading, no quotes, no explanation.

OLDER:
<<OLDER>>

NEWER:
<<NEWER>>
"""


class BandRow(Row):
    pair_id: str
    source: str
    relation: str | None
    similarity: float
    pick: str | None
    enhances: float | None
    existing: str
    incoming: str


class FoldRow(Row):
    pair_id: str
    writer: str
    text: str | None
    elapsed_ms: float
    attempts: int
    error: str | None

    @property
    def answered(self) -> bool:
        return self.error is None and bool(self.text)


class FoldGrade(BaseModel):
    answers: dict[str, bool] = {}

    @property
    def failing(self) -> list[str]:
        return [name for name in QUESTIONS if self.answers.get(name) is False]

    @property
    def status(self) -> Grade:
        if self.failing:
            return "bad"

        return "good" if all(self.answers.get(name) is True for name in QUESTIONS) else "disputed"


Grades = Mapping[str, FoldGrade]


class Decider(Protocol):
    @property
    def name(self) -> str: ...

    def decide(self, case: Case) -> Action: ...


@dataclass(frozen=True)
class FoldRule:
    name: str
    floor: float
    enhances: float | None
    picks_name: str
    picks: frozenset[str]

    def decide(self, case: Case) -> Action:
        base = CONTAINMENT.decide(case)

        if base != "new":
            return base

        return "fold" if self.triggers(case) else "new"

    def triggers(self, case: Case) -> bool:
        if case.similarity is None or case.similarity < self.floor or case.pick not in self.picks:
            return False

        if self.enhances is None:
            return True

        reading = case.p_yes("enhances")

        return reading is not None and reading >= self.enhances


CONTAINMENT = jev_rule("new_covers_old", False, CONTAINMENT_THRESHOLD, None)


def fold_rule(floor: float, enhances: float | None, picks_name: str) -> FoldRule:
    gate = "" if enhances is None else f" enhances>={enhances:.2f}"

    return FoldRule(
        name=f"sim>={floor:.2f}{gate} pick={picks_name}",
        floor=floor,
        enhances=enhances,
        picks_name=picks_name,
        picks=PICK_SETS[picks_name],
    )


def all_fold_rules() -> list[FoldRule]:
    return [
        fold_rule(floor, enhances, picks_name)
        for floor in SIMILARITY_FLOORS
        for enhances in ENHANCES_FLOORS
        for picks_name in PICK_SETS
    ]


LOOSEST = fold_rule(BAND_FLOOR, None, "any non-distinct")


def build_band(
    cases: Mapping[str, Case], relation: Mapping[str, str], pairs: Mapping[str, PairRow]
) -> list[BandRow]:
    band: list[BandRow] = []

    for pair_id, case in cases.items():
        similarity = case.similarity

        if similarity is None or similarity < BAND_FLOOR or CONTAINMENT.decide(case) != "new":
            continue

        pair = pairs[pair_id]
        band.append(
            BandRow(
                pair_id=pair_id,
                source=pair.source,
                relation=relation.get(pair_id),
                similarity=similarity,
                pick=case.pick,
                enhances=case.p_yes("enhances"),
                existing=pair.existing,
                incoming=pair.incoming,
            )
        )

    return band


def render_writer_prompt(older: str, newer: str) -> str:
    return WRITER_PROMPT.replace(OLDER_TOKEN, older).replace(NEWER_TOKEN, newer)


async def write_once(
    adapter: SubprocessJudgeAdapter, writer: str, row: BandRow, attempts: int
) -> FoldRow:
    started = time.perf_counter()
    text: str | None = None
    error: str | None = None

    try:
        text = (await adapter.call(render_writer_prompt(row.existing, row.incoming))).strip()

        if not text:
            error = "EmptyFold"
            text = None
    except JudgeUnavailable as exc:
        error = f"JudgeUnavailable: {str(exc)[:200]}"

    return FoldRow(
        pair_id=row.pair_id,
        writer=writer,
        text=text,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        attempts=attempts,
        error=error,
    )


async def write_fold(adapter: SubprocessJudgeAdapter, writer: str, row: BandRow) -> FoldRow:
    result = await write_once(adapter, writer, row, 1)

    for attempt in range(2, MAX_ATTEMPTS + 1):
        if result.answered:
            break

        result = await write_once(adapter, writer, row, attempt)

    return result


async def write_all(
    adapter: SubprocessJudgeAdapter,
    writer: str,
    rows: Sequence[BandRow],
    sink: Callable[[FoldRow], None],
) -> list[FoldRow]:
    gate = asyncio.Semaphore(CONCURRENCY)
    failures = 0
    results: list[FoldRow] = []

    async def one(row: BandRow) -> None:
        nonlocal failures

        if failures >= STOP_AFTER_FAILURES:
            return

        async with gate:
            if failures >= STOP_AFTER_FAILURES:
                return

            result = await write_fold(adapter, writer, row)

        failures = 0 if result.answered else failures + 1
        sink(result)
        results.append(result)

    await asyncio.gather(*(one(row) for row in rows))

    return results


def load_folds(path: Path) -> dict[tuple[str, str], FoldRow]:
    if not path.exists():
        return {}

    latest: dict[tuple[str, str], FoldRow] = {}

    for row in read_jsonl(path, FoldRow):
        latest[(row.pair_id, row.writer)] = row

    return latest


def pending_band(
    band: Sequence[BandRow], folds: Mapping[tuple[str, str], FoldRow], writer: str
) -> list[BandRow]:
    return [
        row
        for row in band
        if (row.pair_id, writer) not in folds or not folds[(row.pair_id, writer)].answered
    ]


def fold_item_id(pair_id: str, writer: str, question: str) -> str:
    return f"{pair_id}~{writer}#{question}"


def split_fold_item(value: str) -> tuple[str, str, str]:
    head, _, question = value.partition("#")
    pair_id, _, writer = head.rpartition("~")

    return pair_id, writer, question


def fold_items(band: Sequence[BandRow], folds: Mapping[tuple[str, str], FoldRow]) -> list[Item]:
    items: list[Item] = []

    for row in band:
        for writer in WRITERS:
            fold = folds.get((row.pair_id, writer))

            if fold is None or not fold.answered or fold.text is None:
                continue

            for question in QUESTIONS:
                items.append(
                    Item(
                        item_id=fold_item_id(row.pair_id, writer, question),
                        fields={
                            "question": question,
                            "existing": row.existing,
                            "incoming": row.incoming,
                            "fold": fold.text,
                        },
                    )
                )

    return items


def load_fold_grades(path: Path) -> dict[tuple[str, str], FoldGrade]:
    grades: dict[tuple[str, str], FoldGrade] = {}

    if not path.exists():
        return grades

    for row in read_jsonl(path, KeyRow):
        pair_id, writer, question = split_fold_item(row.item_id)
        grades.setdefault((pair_id, writer), FoldGrade()).answers[question] = row.label == "yes"

    return grades


def writer_grades(grades: Mapping[tuple[str, str], FoldGrade], writer: str) -> dict[str, FoldGrade]:
    return {pair_id: grade for (pair_id, name), grade in grades.items() if name == writer}


class FoldScore(BaseModel):
    base: Score
    triggered: list[str] = []
    good: list[str] = []
    bad: list[str] = []
    disputed: list[str] = []
    bad_by_answer: dict[str, int] = {}
    bad_by_label: dict[str, int] = {}
    folded_good_restatement: int = 0
    folded_good_replacement: int = 0
    folded_unbad_restatement: int = 0
    folded_unbad_replacement: int = 0
    follow_up_folds: int = 0
    follow_up_one_record_yes: int = 0
    follow_up_one_record_no: int = 0
    distinct_folds: list[str] = []
    distinct_not_one_record: list[str] = []

    @property
    def resolved_restatement(self) -> int:
        return self.base.resolved_restatement + self.folded_good_restatement

    @property
    def caught_replacement(self) -> int:
        return self.base.handled_replacement + self.folded_good_replacement

    @property
    def errors(self) -> int:
        return len(self.bad) + self.base.wrong + self.base.lost

    @property
    def strict_errors(self) -> int:
        return self.errors + len(self.disputed)

    @property
    def loose_restatement(self) -> int:
        return self.base.resolved_restatement + self.folded_unbad_restatement

    @property
    def loose_replacement(self) -> int:
        return self.base.handled_replacement + self.folded_unbad_replacement

    @property
    def meets_bars(self) -> bool:
        return (
            self.resolved_restatement >= RESTATEMENT_BAR
            and self.caught_replacement >= REPLACEMENT_BAR
            and self.errors == 0
        )


CoverAction = Literal["reinforce", "supersede", "flag", "new"]


def _as_containment(decide: Callable[[Case], Action]) -> Callable[[Case], CoverAction]:
    def inner(case: Case) -> CoverAction:
        action = decide(case)

        return "new" if action == "fold" else action

    return inner


def score_fold(
    rule: Decider | Rule,
    cases: Mapping[str, Case],
    relation: Mapping[str, str],
    cover: Mapping[str, Mapping[str, bool]],
    grades: Grades,
    only: frozenset[str] | None = None,
) -> FoldScore:
    decide: Callable[[Case], Action] = rule.decide
    flat = Rule(name=rule.name, family="fold", decide=_as_containment(decide))
    score = FoldScore(base=score_rule(flat, cases, relation, cover, only))

    for pair_id, case in cases.items():
        if (only is not None and pair_id not in only) or decide(case) != "fold":
            continue

        grade = grades.get(pair_id, FoldGrade())
        _tally_fold(score, pair_id, relation.get(pair_id, "disputed"), grade)

    return score


def _tally_fold(score: FoldScore, pair_id: str, label: str, grade: FoldGrade) -> None:
    score.triggered.append(pair_id)
    status = grade.status

    if status == "good":
        score.good.append(pair_id)
    elif status == "bad":
        score.bad.append(pair_id)
        score.bad_by_label[label] = score.bad_by_label.get(label, 0) + 1

        for name in grade.failing:
            score.bad_by_answer[name] = score.bad_by_answer.get(name, 0) + 1
    else:
        score.disputed.append(pair_id)

    if label == "restatement" and status == "good":
        score.folded_good_restatement += 1

    if label == "restatement" and status != "bad":
        score.folded_unbad_restatement += 1

    answers = grade.answers
    newer_and_clean = answers.get("newer_wins") is True and answers.get("nothing_invented") is True
    newer_unfailed = answers.get("newer_wins") is not False
    clean_unfailed = answers.get("nothing_invented") is not False

    if label == "replacement" and newer_and_clean:
        score.folded_good_replacement += 1

    if label == "replacement" and newer_unfailed and clean_unfailed:
        score.folded_unbad_replacement += 1

    if label == "follow_up":
        score.follow_up_folds += 1
        score.follow_up_one_record_yes += answers.get("one_record") is True
        score.follow_up_one_record_no += answers.get("one_record") is False

    if label == "distinct":
        score.distinct_folds.append(pair_id)

        if answers.get("one_record") is not True:
            score.distinct_not_one_record.append(pair_id)


def rank(score: FoldScore) -> tuple[int, int, int, int]:
    return (
        score.errors,
        -(score.resolved_restatement + score.caught_replacement),
        score.base.false_flag,
        len(score.triggered),
    )


def rank_strict(score: FoldScore) -> tuple[int, int, int, int]:
    return (
        score.strict_errors,
        -(score.resolved_restatement + score.caught_replacement),
        score.base.false_flag,
        len(score.triggered),
    )


def rank_loose(score: FoldScore) -> tuple[int, int, int, int]:
    return (
        score.errors,
        -(score.loose_restatement + score.loose_replacement),
        score.base.false_flag,
        len(score.triggered),
    )


def best_fold_rule(
    rules: Sequence[FoldRule],
    cases: Mapping[str, Case],
    relation: Mapping[str, str],
    cover: Mapping[str, Mapping[str, bool]],
    grades: Grades,
    only: frozenset[str] | None = None,
    ranking: Callable[[FoldScore], tuple[int, int, int, int]] = rank,
) -> FoldRule:
    return min(
        rules, key=lambda rule: ranking(score_fold(rule, cases, relation, cover, grades, only))
    )


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def _fixed(value: float | None, places: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _split(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none"


def sweep_row(name: str, writer: str, score: FoldScore, total: int) -> str:
    base = score.base

    return (
        f"| {name} | {writer} "
        f"| {_ratio(score.resolved_restatement, base.restatements)} "
        f"({base.reinforced_restatement}+{base.superseded_restatement}"
        f"+{score.folded_good_restatement}) "
        f"| {_ratio(score.caught_replacement, base.replacements)} "
        f"({base.flagged_replacement}+{base.superseded_replacement}"
        f"+{score.folded_good_replacement}) "
        f"| {len(score.triggered)} ({len(score.triggered) / total:.1%}) "
        f"| {len(score.good)}/{len(score.bad)}/{len(score.disputed)} "
        f"| {score.follow_up_folds} ({score.follow_up_one_record_yes}/"
        f"{score.follow_up_one_record_no}) "
        f"| {len(score.distinct_folds)} ({len(score.distinct_not_one_record)}) "
        f"| {base.wrong} | {base.lost_record} | {base.lost_update} | {base.false_flag} "
        f"| {'YES' if score.meets_bars else ''} |"
    )


SWEEP_HEAD = (
    "| rule | writer | restatements resolved (reinforced+superseded+folded good) "
    "| replacements caught (flagged+superseded+folded) | folds (share of all saves) "
    "| good/bad/disputed | follow_up folds (one_record yes/no) "
    "| distinct folds (not keyed one_record yes) | wrong supersedes | lost records "
    "| lost updates | false flags | meets bars |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


@dataclass(frozen=True)
class Inputs:
    pairs: Mapping[str, PairRow]
    cases: Mapping[str, Case]
    relation: Mapping[str, str]
    cover: Mapping[str, Mapping[str, bool]]
    band: Sequence[BandRow]
    folds: Mapping[tuple[str, str], FoldRow]
    grades: Mapping[tuple[str, str], FoldGrade]
    labels: Sequence[LabelRow]
    published_v1: Sequence[SaveRow]
    out_dir: Path


def load_inputs(base: Path, out_dir: Path) -> Inputs:
    covers_dir = base / "experiments" / "covers"
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    relation = {
        pid: label for pid, label in load_key(base / "key-relation.jsonl").items() if pid in signals
    }
    labels_path = out_dir / "labels-fold.jsonl"

    return Inputs(
        pairs=pairs,
        cases=build_cases(
            signals,
            load_covers(covers_dir / "signals.jsonl"),
            load_sonnet_calls(covers_dir / "sonnet-calls.jsonl"),
        ),
        relation=relation,
        cover=load_cover_key(covers_dir / "key-covers.jsonl"),
        band=read_jsonl(out_dir / "band.jsonl", BandRow),
        folds=load_folds(out_dir / "folds.jsonl"),
        grades=load_fold_grades(out_dir / "key-fold.jsonl"),
        labels=read_jsonl(labels_path, LabelRow) if labels_path.exists() else [],
        published_v1=[
            row for row in read_jsonl(base / "save-v1.jsonl", SaveRow) if row.system == "v1"
        ],
        out_dir=out_dir,
    )


def agreement_rows(
    labels: Sequence[LabelRow],
) -> list[tuple[str, int, int, float, float, int, int]]:
    rows: list[tuple[str, int, int, float, float, int, int]] = []

    for name in QUESTIONS:
        subset = [row for row in labels if split_fold_item(row.item_id)[2] == name]
        stats = agreement_stats(subset)
        confusion = stats["confusion"]
        yes = confusion.get("yes", {}).get("yes", 0) if isinstance(confusion, dict) else 0
        no = confusion.get("no", {}).get("no", 0) if isinstance(confusion, dict) else 0
        rows.append(
            (
                name,
                int(str(stats["items_labelled_by_both"])),
                int(str(stats["items_agreed"])),
                float(str(stats["raw_agreement"])),
                float(str(stats["cohens_kappa"])),
                yes,
                no,
            )
        )

    return rows


def agreement_table(labels: Sequence[LabelRow]) -> str:
    lines = [
        "| answer | folds both judges labelled | agreed | raw agreement | Cohen's kappa "
        "| keyed yes | keyed no |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, both, agreed, raw, kappa, yes, no in agreement_rows(labels):
        lines.append(f"| {name} | {both} | {agreed} | {raw:.1%} | {kappa:.3f} | {yes} | {no} |")

    return "\n".join(lines)


def fold_quality_table(inputs: Inputs) -> str:
    lines = [
        "| writer | folds | good | bad | disputed | nothing_lost no | nothing_invented no "
        "| newer_wins no | one_record no | median latency ms | p90 latency ms "
        "| median length ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    band = {row.pair_id: row for row in inputs.band}

    for writer in WRITERS:
        rows = [inputs.folds[(pid, writer)] for pid in band if (pid, writer) in inputs.folds]
        answered = [row for row in rows if row.answered]
        grades = [inputs.grades.get((row.pair_id, writer), FoldGrade()) for row in answered]
        status = Counter(grade.status for grade in grades)
        times = sorted(row.elapsed_ms for row in answered)
        ratios = [length_ratio(row, band[row.pair_id]) for row in answered]
        failing = Counter(name for grade in grades for name in grade.failing)
        lines.append(
            f"| {writer} | {len(answered)}/{len(band)} | {status['good']} | {status['bad']} "
            f"| {status['disputed']} | {failing['nothing_lost']} | {failing['nothing_invented']} "
            f"| {failing['newer_wins']} | {failing['one_record']} | {_fixed(median(times))} "
            f"| {_fixed(percentile(times, 0.9))} | {_fixed(median(ratios), 2)} |"
        )

    return "\n".join(lines)


def length_ratio(fold: FoldRow, row: BandRow) -> float:
    return len(fold.text or "") / max(len(row.existing), len(row.incoming))


def band_split(band: Sequence[BandRow]) -> str:
    return _split(Counter(row.relation or "disputed" for row in band))


def bad_folds_text(inputs: Inputs, best: Mapping[str, FoldRule]) -> str:
    lines = [
        "# Bad folds",
        "",
        "Holds record text; stays under dev-corpus. A fold is bad when either judge pair keyed one "
        "of the four answers no. Every bad fold over the whole candidate band is listed, whether "
        "or not a rule would trigger it.",
        "",
    ]
    count = 0

    for writer in WRITERS:
        for row in inputs.band:
            grade = inputs.grades.get((row.pair_id, writer))
            fold = inputs.folds.get((row.pair_id, writer))

            if grade is None or fold is None or grade.status != "bad":
                continue

            count += 1
            fires = best[writer].decide(inputs.cases[row.pair_id]) == "fold"
            lines += [
                f"### {row.pair_id} ({writer}, relation key: {row.relation or 'disputed'})",
                "",
                f"Failing: {', '.join(grade.failing)}. Fires under the best {writer} rule "
                f"`{best[writer].name}`: {'yes' if fires else 'no'}. "
                f"Answers: {answers_text(grade)}.",
                "",
                f"- existing: {' '.join(row.existing.split())}",
                f"- incoming: {' '.join(row.incoming.split())}",
                f"- fold: {' '.join((fold.text or '').split())}",
                "",
            ]

    lines.insert(2, f"{count} bad folds.")
    lines.insert(3, "")

    return "\n".join(lines)


def answers_text(grade: FoldGrade) -> str:
    marks = {True: "yes", False: "no"}

    return ", ".join(
        f"{name}={marks[grade.answers[name]] if name in grade.answers else 'unkeyed'}"
        for name in QUESTIONS
    )


SHORT_NAMES = {"nothing_lost": "l", "nothing_invented": "n", "newer_wins": "w", "one_record": "o"}


def short_answers(grade: FoldGrade | None) -> str:
    if grade is None:
        return "not graded"

    marks = {True: "y", False: "n"}

    return " ".join(
        f"{SHORT_NAMES[name]}={marks[grade.answers[name]] if name in grade.answers else '-'}"
        for name in QUESTIONS
    )


def stuck_text(inputs: Inputs, best: Mapping[str, FoldRule]) -> str:
    lines = [
        "# Restatements the containment key does not cover",
        "",
        "The list is the 11 the covers experiment left stuck. Twelve restatements stay "
        "unresolved under the containment rule; golden-17 (flagged) is the one not listed.",
        "",
        "Answers read l=nothing_lost, n=nothing_invented, w=newer_wins, o=one_record; y keyed "
        "yes, n keyed no, - not keyed (the judges split or a label is missing). Outcome is under "
        "the best rule for that writer.",
        "",
        "| pair_id | relation key | containment rule | similarity | Jev pick | enhances "
        "| loosest rule fires | "
        + " | ".join(
            f"{w}: best rule fires | {w}: answers | {w}: grade | {w}: outcome" for w in WRITERS
        )
        + " |",
        "|---|---|---|---|---|---|---|" + "---|---|---|---|" * len(WRITERS),
    ]

    for pair_id in STUCK:
        case = inputs.cases[pair_id]
        cells: list[str] = []

        for writer in WRITERS:
            action = best[writer].decide(case)
            grade = inputs.grades.get((pair_id, writer))
            fired = action == "fold"
            outcome = (
                f"fold {grade.status if grade else 'ungraded'}"
                + (" -> resolved" if grade is not None and grade.status == "good" else "")
                if fired
                else action
            )
            cells += [
                "yes" if fired else "no",
                short_answers(grade) if fired or grade is not None else "no fold",
                grade.status if grade is not None else "n/a",
                outcome,
            ]

        similarity = "none" if case.similarity is None else f"{case.similarity:.3f}"
        lines.append(
            f"| {pair_id} | {inputs.relation.get(pair_id, 'disputed')} "
            f"| {CONTAINMENT.decide(case)} | {similarity} | {case.pick} "
            f"| {_fixed(case.p_yes('enhances'), 2)} "
            f"| {'yes' if LOOSEST.decide(case) == 'fold' else 'no'} | " + " | ".join(cells) + " |"
        )

    return "\n".join(lines)


def leave_one_source_out(
    rules: Sequence[FoldRule], inputs: Inputs, writer: str
) -> tuple[str, dict[str, int]]:
    grades = writer_grades(inputs.grades, writer)
    sources = {pid: inputs.pairs[pid].source for pid in inputs.cases}
    everyone = frozenset(sources)
    lines = [
        "| held-out source | rule chosen on the other sources | restatements resolved "
        "| replacements caught | bad folds | wrong supersedes | lost |",
        "|---|---|---|---|---|---|---|",
    ]
    pooled = {"restatements": 0, "replacements": 0, "bad": 0, "wrong": 0, "lost": 0}
    totals = {"restatements": 0, "replacements": 0}

    for held in sorted(set(sources.values())):
        test = frozenset(pid for pid, source in sources.items() if source == held)
        chosen = best_fold_rule(
            rules, inputs.cases, inputs.relation, inputs.cover, grades, everyone - test
        )
        score = score_fold(chosen, inputs.cases, inputs.relation, inputs.cover, grades, test)
        pooled["restatements"] += score.resolved_restatement
        pooled["replacements"] += score.caught_replacement
        pooled["bad"] += len(score.bad)
        pooled["wrong"] += score.base.wrong
        pooled["lost"] += score.base.lost
        totals["restatements"] += score.base.restatements
        totals["replacements"] += score.base.replacements
        lines.append(
            f"| {held} | {chosen.name} "
            f"| {_ratio(score.resolved_restatement, score.base.restatements)} "
            f"| {_ratio(score.caught_replacement, score.base.replacements)} "
            f"| {len(score.bad)} | {score.base.wrong} | {score.base.lost} |"
        )

    lines.append(
        f"| pooled held-out | | {_ratio(pooled['restatements'], totals['restatements'])} "
        f"| {_ratio(pooled['replacements'], totals['replacements'])} | {pooled['bad']} "
        f"| {pooled['wrong']} | {pooled['lost']} |"
    )

    return "\n".join(lines), pooled


def headline_rows(
    inputs: Inputs, best: Mapping[str, FoldRule], scores: Mapping[str, FoldScore]
) -> str:
    relation = inputs.relation
    empty: Grades = {}
    baseline = score_fold(BASELINE, inputs.cases, relation, inputs.cover, empty)
    containment = score_fold(CONTAINMENT, inputs.cases, relation, inputs.cover, empty)
    v1 = v1_counts(inputs.published_v1, relation)
    lines = [
        "| system | restatements resolved | replacements caught | bad folds | wrong supersedes "
        "| lost records | lost updates |",
        "|---|---|---|---|---|---|---|",
        _headline_line("v2 today (Jev pick)", baseline),
        _headline_line("v2 plus containment rule", containment),
        "| v1 as published "
        f"| {_ratio(v1.resolved_restatement, v1.restatements)} "
        f"| {_ratio(v1.flagged_replacement, v1.replacements)} | n/a | n/a "
        f"| {v1.lost_record}/{v1.unrelated} | {v1.lost_update}/{v1.replacements} |",
    ]

    for writer in WRITERS:
        lines.append(
            _headline_line(f"best fold rule, {writer}: {best[writer].name}", scores[writer])
        )

    return "\n".join(lines)


def _headline_line(name: str, score: FoldScore) -> str:
    base = score.base

    return (
        f"| {name} | {_ratio(score.resolved_restatement, base.restatements)} "
        f"| {_ratio(score.caught_replacement, base.replacements)} | {len(score.bad)} "
        f"| {base.wrong} | {base.lost_record}/{base.unrelated} "
        f"| {base.lost_update}/{base.replacements} |"
    )


def verdict_text(
    best: Mapping[str, FoldRule], scores: Mapping[str, FoldScore], winners: Mapping[str, int]
) -> str:
    sentences: list[str] = []

    for writer in WRITERS:
        score = scores[writer]
        gap_rs = RESTATEMENT_BAR - score.resolved_restatement
        gap_rp = REPLACEMENT_BAR - score.caught_replacement
        misses: list[str] = []

        if gap_rs > 0:
            misses.append(f"restatements by {gap_rs}")

        if gap_rp > 0:
            misses.append(f"replacements by {gap_rp}")

        if score.bad:
            misses.append(f"{len(score.bad)} bad fold(s)")

        if score.base.wrong:
            misses.append(f"{score.base.wrong} wrong supersede(s)")

        if score.base.lost:
            misses.append(f"{score.base.lost} lost record(s) or update(s)")

        sentences.append(
            f"{writer}: {winners[writer]} rule(s) reach {RESTATEMENT_BAR}+ restatements and "
            f"{REPLACEMENT_BAR} replacements with zero bad folds, wrong supersedes and lost "
            f"records. Best rule `{best[writer].name}` resolves {score.resolved_restatement} "
            f"restatements and catches {score.caught_replacement} replacements; it misses on "
            + (", ".join(misses) if misses else "nothing")
            + "."
        )

    return " ".join(sentences)


def readings_text(inputs: Inputs) -> str:
    rules = all_fold_rules()
    lines = [
        "| reading | writer | best rule | restatements resolved | replacements caught "
        "| errors counted against | bad | disputed | wrong supersedes | lost | meets bars |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for writer in WRITERS:
        grades = writer_grades(inputs.grades, writer)

        for reading, ranking in (("strict", rank_strict), ("loose", rank_loose)):
            chosen = best_fold_rule(
                rules, inputs.cases, inputs.relation, inputs.cover, grades, ranking=ranking
            )
            score = score_fold(chosen, inputs.cases, inputs.relation, inputs.cover, grades)
            strict = reading == "strict"
            resolved = score.resolved_restatement if strict else score.loose_restatement
            caught = score.caught_replacement if strict else score.loose_replacement
            errors = score.strict_errors if strict else score.errors
            meets = resolved >= RESTATEMENT_BAR and caught >= REPLACEMENT_BAR and errors == 0
            lines.append(
                f"| {reading} | {writer} | {chosen.name} "
                f"| {_ratio(resolved, score.base.restatements)} "
                f"| {_ratio(caught, score.base.replacements)} | {errors} | {len(score.bad)} "
                f"| {len(score.disputed)} | {score.base.wrong} | {score.base.lost} "
                f"| {'YES' if meets else ''} |"
            )

    return "\n".join(lines)


def pair_table(pair_ids: Iterable[str], inputs: Inputs, writer: str) -> str:
    lines = ["| pair_id | similarity | Jev pick | grade | one_record |", "|---|---|---|---|---|"]

    for pair_id in pair_ids:
        case = inputs.cases[pair_id]
        grade = inputs.grades.get((pair_id, writer), FoldGrade())
        record = grade.answers.get("one_record")
        lines.append(
            f"| {pair_id} | {_fixed(case.similarity, 3)} | {case.pick} | {grade.status} "
            f"| {'unkeyed' if record is None else 'yes' if record else 'no'} |"
        )

    return "\n".join(lines)


def unjudged_count(inputs: Inputs) -> int:
    seen: dict[str, set[str]] = {}

    for row in inputs.labels:
        seen.setdefault(row.item_id, set()).add(row.judge)

    wanted = {
        fold_item_id(pid, writer, question)
        for (pid, writer), fold in inputs.folds.items()
        if fold.answered
        for question in QUESTIONS
    }

    return sum(1 for item in wanted if len(seen.get(item, set())) < 2)


def summary_text(inputs: Inputs) -> str:
    rules = all_fold_rules()
    grades = {writer: writer_grades(inputs.grades, writer) for writer in WRITERS}
    best = {
        writer: best_fold_rule(rules, inputs.cases, inputs.relation, inputs.cover, grades[writer])
        for writer in WRITERS
    }
    scores = {
        writer: score_fold(
            best[writer], inputs.cases, inputs.relation, inputs.cover, grades[writer]
        )
        for writer in WRITERS
    }
    sweep = {
        (writer, rule.name): score_fold(
            rule, inputs.cases, inputs.relation, inputs.cover, grades[writer]
        )
        for writer in WRITERS
        for rule in rules
    }
    winners = {
        writer: sum(1 for (w, _), score in sweep.items() if w == writer and score.meets_bars)
        for writer in WRITERS
    }
    unjudged = unjudged_count(inputs)
    lines = ["# Experiment 4: fold", ""]

    if unjudged:
        lines += [f"WARNING: {unjudged} fold answers lack a label from one or both judges.", ""]

    lines += [
        verdict_text(best, scores, winners),
        "",
        f"Candidate band: {len(inputs.band)} pairs (containment rule outputs save as new, "
        f"similarity >= {BAND_FLOOR}). Split by relation key: {band_split(inputs.band)}. "
        f"Pairs: {len(inputs.cases)}; relation-keyed: {len(inputs.relation)}.",
        "",
        "Inside the band every Jev pick is follow_up or distinct, because a restatement or "
        "replacement pick is already reinforced or flagged. The three pick sets in the rule "
        "family therefore select the same pairs.",
        "",
        "## Headline",
        "",
        headline_rows(inputs, best, scores),
        "",
        "## Strict and loose readings",
        "",
        "Strict: a disputed fold is not resolved and counts as an error, as does a keyed-bad one. "
        "Loose: a fold counts as resolved unless a judge pair keyed a needed answer no, and only "
        "keyed-bad folds count against. The best rule is the one with the fewest errors, then the "
        "most resolved, under that reading.",
        "",
        readings_text(inputs),
        "",
        "## Fold judges: agreement per answer",
        "",
        agreement_table(inputs.labels),
        "",
        "## Folds by writer (whole band)",
        "",
        fold_quality_table(inputs),
        "",
        "Length ratio is fold length over the longer source, in characters. Writers: the v2 "
        "cleaner has no LLM merge prompt, so no third arm was run.",
        "",
    ]

    for writer in WRITERS:
        score = scores[writer]
        lines += [
            f"## Best rule, {writer}: `{best[writer].name}`",
            "",
            f"Folds fired: {len(score.triggered)} of {len(inputs.cases)} saves "
            f"({len(score.triggered) / len(inputs.cases):.1%}). Good {len(score.good)}, "
            f"bad {len(score.bad)}, disputed {len(score.disputed)}.",
            "",
            f"Bad folds by failing answer: {_split(score.bad_by_answer)}. "
            f"By relation key: {_split(score.bad_by_label)}.",
            "",
            f"Folds on true follow_up pairs: {score.follow_up_folds}; one_record keyed yes "
            f"{score.follow_up_one_record_yes}, keyed no {score.follow_up_one_record_no}.",
            "",
            f"Folds on true distinct pairs: {len(score.distinct_folds)}; not keyed one_record yes: "
            f"{len(score.distinct_not_one_record)}.",
            "",
            pair_table(score.distinct_folds, inputs, writer) if score.distinct_folds else "",
            "",
            f"Lost records {score.base.lost_record}/{score.base.unrelated}, lost updates "
            f"{score.base.lost_update}/{score.base.replacements}, wrong supersedes "
            f"{score.base.wrong}.",
            "",
        ]
        table, _ = leave_one_source_out(rules, inputs, writer)
        lines += [
            f"### Leave one source out, {writer}",
            "",
            table,
            "",
        ]

    hand_check = inputs.out_dir / "hand-check.md"

    if hand_check.exists():
        lines += ["## Hand check", "", hand_check.read_text(encoding="utf-8").strip(), ""]

    lines += [
        "## Files",
        "",
        "- band.jsonl, folds.jsonl, labels-fold.jsonl, key-fold.jsonl, agreement-fold.json, "
        "sweep.md, bad-folds.md, stuck.md",
        "",
    ]

    return "\n".join(lines)


def sweep_text(inputs: Inputs) -> str:
    rules = all_fold_rules()
    total = len(inputs.cases)
    lines = [SWEEP_HEAD]
    empty: Grades = {}
    containment = score_fold(CONTAINMENT, inputs.cases, inputs.relation, inputs.cover, empty)
    lines.append(sweep_row("containment only (no fold)", "none", containment, total))

    for writer in WRITERS:
        grades = writer_grades(inputs.grades, writer)
        scored = [
            (rule, score_fold(rule, inputs.cases, inputs.relation, inputs.cover, grades))
            for rule in rules
        ]

        for rule, score in sorted(scored, key=lambda pair: rank(pair[1])):
            lines.append(sweep_row(rule.name, writer, score, total))

    return "\n".join(lines)


def run_band(out_dir: Path, base: Path) -> int:
    covers_dir = base / "experiments" / "covers"
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    relation = {
        pid: label for pid, label in load_key(base / "key-relation.jsonl").items() if pid in signals
    }
    cases = build_cases(
        signals,
        load_covers(covers_dir / "signals.jsonl"),
        load_sonnet_calls(covers_dir / "sonnet-calls.jsonl"),
    )
    band = build_band(cases, relation, pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "band.jsonl", band)
    print(f"band: {len(band)} pairs; {band_split(band)}", flush=True)

    return 0


def run_writers(out_dir: Path, writers: Sequence[str]) -> int:
    band = read_jsonl(out_dir / "band.jsonl", BandRow)
    binary = shutil.which("claude")

    if binary is None:
        print("claude CLI not found", file=sys.stderr)

        return 1

    out_dir.joinpath("writer-prompt.txt").write_text(WRITER_PROMPT, encoding="utf-8")
    log = out_dir / "folds.jsonl"
    handle = log.open("a", encoding="utf-8")

    def sink(row: FoldRow) -> None:
        handle.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()

    os.chdir(tempfile.mkdtemp(prefix="fold-writers-"))
    complete = True

    for writer in writers:
        todo = pending_band(band, load_folds(log), writer)
        adapter = SubprocessJudgeAdapter(
            cli="claude", binary=binary, model=writer, timeout=CALL_TIMEOUT
        )
        asyncio.run(write_all(adapter, writer, todo, sink))
        left = len(pending_band(band, load_folds(log), writer))
        print(f"{writer}: {len(todo)} attempted, {left} of {len(band)} still pending", flush=True)
        complete = complete and left == 0

    handle.close()

    return 0 if complete else 2


def run_judges(out_dir: Path) -> int:
    band = read_jsonl(out_dir / "band.jsonl", BandRow)
    items = fold_items(band, load_folds(out_dir / "folds.jsonl"))
    outcome = run_task("fold", items, out_dir=out_dir)
    keyed = {row.item_id for row in outcome.key}
    seen: dict[str, set[str]] = {}

    for row in outcome.labels:
        seen.setdefault(row.item_id, set()).add(row.judge)

    missing = sum(1 for item in items if len(seen.get(item.item_id, set())) < 2)
    print(
        json.dumps(
            {
                "items": len(items),
                "keyed": len(keyed),
                "missing_a_judge": missing,
                "unlabelled_this_run": outcome.unlabelled,
            }
        ),
        flush=True,
    )

    return 0 if missing == 0 else 2


def run_score(out_dir: Path, base: Path) -> int:
    inputs = load_inputs(base, out_dir)
    rules = all_fold_rules()
    grades = {writer: writer_grades(inputs.grades, writer) for writer in WRITERS}
    best = {
        writer: best_fold_rule(rules, inputs.cases, inputs.relation, inputs.cover, grades[writer])
        for writer in WRITERS
    }
    out_dir.joinpath("sweep.md").write_text(sweep_text(inputs) + "\n", encoding="utf-8")
    out_dir.joinpath("bad-folds.md").write_text(
        bad_folds_text(inputs, best) + "\n", encoding="utf-8"
    )
    out_dir.joinpath("stuck.md").write_text(stuck_text(inputs, best) + "\n", encoding="utf-8")
    out_dir.joinpath("summary.md").write_text(summary_text(inputs) + "\n", encoding="utf-8")
    print("scored; wrote sweep.md, bad-folds.md, stuck.md, summary.md", flush=True)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Fold experiment: band, writers, judges, scorer.")
    parser.add_argument("phase", choices=("band", "write", "judge", "score", "all"))
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "fold")
    parser.add_argument("--writers", nargs="+", default=list(WRITERS))
    args = parser.parse_args()
    phases = ("band", "write", "judge", "score") if args.phase == "all" else (args.phase,)

    for phase in phases:
        if phase == "band":
            code = run_band(args.out_dir, args.base)
        elif phase == "write":
            code = run_writers(args.out_dir, args.writers)
        elif phase == "judge":
            code = run_judges(args.out_dir)
        else:
            code = run_score(args.out_dir, args.base)

        if code != 0:
            sys.exit(code)


if __name__ == "__main__":
    main()
