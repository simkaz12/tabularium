from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval_v1v2 import paths
from eval_v1v2.covers_calls import QUESTIONS, SonnetCall, load_sonnet_calls, split_item_id
from eval_v1v2.dup_rules import SignalRow, load_signals
from eval_v1v2.judges import agreement_stats
from eval_v1v2.rows import LabelRow, PairRow, Row, SaveRow, read_jsonl
from eval_v1v2.score_save import load_key, median, percentile
from eval_v1v2.second_judge import band_pair_ids, v1_bars

Action = Literal["reinforce", "supersede", "flag", "new"]
CoverSignalName = Literal["new_covers_old", "enhances"]
UNRELATED = frozenset({"follow_up", "distinct"})
THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(9))
SIMILARITY_FLOORS: tuple[float | None, ...] = (None, 0.80, 0.85)
RESTATEMENT_BAR = 35
REPLACEMENT_BAR = 42
V1_NOTICED_BAR = 34
V1_FLAGGED_BAR = 41
REPEAT_SAMPLE = 40
LISTED_PAIR = re.compile(r"^\| ([A-Za-z_]+-[A-Za-z]?\d+) \|", re.MULTILINE)


class Answer(Row):
    answer: Literal["yes", "no"] | None
    p_yes: float | None
    probabilities: dict[str, float] | None
    error: str | None


class CoversSignal(Row):
    pair_id: str
    run: int
    new_covers_old: Answer
    old_covers_new: Answer
    enhances: Answer
    elapsed_ms: float
    error: str | None


@dataclass(frozen=True)
class Case:
    pair_id: str
    pick: str | None
    similarity: float | None
    in_band: bool
    covers: CoversSignal | None
    sonnet: SonnetCall | None

    def p_yes(self, name: str) -> float | None:
        if self.covers is None:
            return None

        answer: Answer = getattr(self.covers, name)

        return answer.p_yes


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    decide: Callable[[Case], Action]


class Score(BaseModel):
    restatements: int = 0
    reinforced_restatement: int = 0
    superseded_restatement: int = 0
    replacements: int = 0
    superseded_replacement: int = 0
    flagged_replacement: int = 0
    unrelated: int = 0
    lost_record: int = 0
    lost_update: int = 0
    false_flag: int = 0
    superseded: list[str] = []
    wrong_supersedes: list[str] = []
    covered_unrelated: list[str] = []
    unverified_supersedes: list[str] = []
    reinforce_drops: list[str] = []

    @property
    def resolved_restatement(self) -> int:
        return self.reinforced_restatement + self.superseded_restatement

    @property
    def handled_replacement(self) -> int:
        return self.superseded_replacement + self.flagged_replacement

    @property
    def wrong(self) -> int:
        return len(self.wrong_supersedes)

    @property
    def lost(self) -> int:
        return self.lost_record + self.lost_update

    @property
    def beats_bars(self) -> bool:
        return (
            self.resolved_restatement >= RESTATEMENT_BAR
            and self.handled_replacement >= REPLACEMENT_BAR
            and self.wrong == 0
            and self.lost == 0
        )


CoverKey = Mapping[str, Mapping[str, bool]]


class _KeyRow(Row):
    item_id: str
    label: Literal["yes", "no"]
    source: str


def load_cover_key(path: Path) -> dict[str, dict[str, bool]]:
    key: dict[str, dict[str, bool]] = {}

    for row in read_jsonl(path, _KeyRow):
        pair_id, question = split_item_id(row.item_id)
        key.setdefault(pair_id, {})[question] = row.label == "yes"

    return key


def load_cover_labels(path: Path) -> list[LabelRow]:
    return read_jsonl(path, LabelRow)


def load_covers(path: Path) -> dict[str, CoversSignal]:
    return {row.pair_id: row for row in read_jsonl(path, CoversSignal)}


def build_cases(
    signals: Mapping[str, SignalRow],
    covers: Mapping[str, CoversSignal],
    sonnet: Mapping[str, SonnetCall],
) -> dict[str, Case]:
    band = set(band_pair_ids(signals, None))
    cases: dict[str, Case] = {}

    for pair_id, signal in signals.items():
        existing = signal.existing
        call = sonnet.get(pair_id)
        cases[pair_id] = Case(
            pair_id=pair_id,
            pick=None if existing is None else existing.pick,
            similarity=None if existing is None else existing.similarity,
            in_band=pair_id in band,
            covers=covers.get(pair_id),
            sonnet=call if call is not None and call.answered else None,
        )

    return cases


def clears(case: Case, floor: float | None) -> bool:
    if case.similarity is None:
        return False

    return floor is None or case.similarity >= floor


def baseline_action(case: Case) -> Action:
    if case.pick == "restatement":
        return "reinforce"

    return "flag" if case.pick == "replacement" else "new"


def _floor_text(floor: float | None) -> str:
    return "" if floor is None else f" sim>={floor:.2f}"


def jev_rule(signal: CoverSignalName, subset: bool, threshold: float, floor: float | None) -> Rule:
    family = signal + ("+subset" if subset else "")

    def decide(case: Case) -> Action:
        if case.pick == "restatement":
            return "reinforce"

        covered = case.p_yes(signal)

        if clears(case, floor) and covered is not None and covered >= threshold:
            return "supersede"

        old_covers = case.p_yes("old_covers_new")
        new_covers = case.p_yes("new_covers_old")
        inside = (
            subset
            and old_covers is not None
            and new_covers is not None
            and old_covers >= threshold
            and new_covers < threshold
        )

        if inside and clears(case, floor):
            return "reinforce"

        return baseline_action(case)

    return Rule(
        name=f"{family} t>={threshold:.2f}{_floor_text(floor)}", family=family, decide=decide
    )


def sonnet_rule(subset: bool, floor: float | None) -> Rule:
    family = "sonnet+subset" if subset else "sonnet"

    def decide(case: Case) -> Action:
        call = case.sonnet

        if case.pick == "restatement":
            return "reinforce"

        if case.in_band and call is not None and clears(case, floor):
            if call.new_covers_old == "yes":
                return "supersede"

            if subset and call.old_covers_new == "yes":
                return "reinforce"

        return baseline_action(case)

    return Rule(name=f"{family}{_floor_text(floor)}", family=family, decide=decide)


BASELINE = Rule(name="baseline", family="baseline", decide=baseline_action)


def jev_rules() -> list[Rule]:
    signals: tuple[CoverSignalName, ...] = ("new_covers_old", "enhances")

    return [
        jev_rule(signal, subset, threshold, floor)
        for signal in signals
        for subset in (False, True)
        for floor in SIMILARITY_FLOORS
        for threshold in THRESHOLDS
    ]


def sonnet_rules() -> list[Rule]:
    return [sonnet_rule(subset, floor) for subset in (False, True) for floor in SIMILARITY_FLOORS]


def score_rule(
    rule: Rule,
    cases: Mapping[str, Case],
    relation: Mapping[str, str],
    cover: CoverKey,
    only: frozenset[str] | None = None,
) -> Score:
    score = Score()

    for pair_id, case in cases.items():
        if only is not None and pair_id not in only:
            continue

        action = rule.decide(case)
        label = relation.get(pair_id)
        _tally_relation(score, label, action)
        _tally_cover(score, pair_id, label, action, cover.get(pair_id, {}))

    return score


def _tally_relation(score: Score, label: str | None, action: Action) -> None:
    if label == "restatement":
        score.restatements += 1
        score.reinforced_restatement += action == "reinforce"
        score.superseded_restatement += action == "supersede"
    elif label == "replacement":
        score.replacements += 1
        score.lost_update += action == "reinforce"
        score.superseded_replacement += action == "supersede"
        score.flagged_replacement += action == "flag"
    elif label in UNRELATED:
        score.unrelated += 1
        score.lost_record += action == "reinforce"
        score.false_flag += action == "flag"


def _tally_cover(
    score: Score, pair_id: str, label: str | None, action: Action, answers: Mapping[str, bool]
) -> None:
    if action == "reinforce" and answers.get("old_covers_new") is False:
        score.reinforce_drops.append(pair_id)

    if action != "supersede":
        return

    score.superseded.append(pair_id)
    covers = answers.get("new_covers_old")

    if covers is None:
        score.unverified_supersedes.append(pair_id)
    elif covers is False:
        score.wrong_supersedes.append(pair_id)
    elif label in UNRELATED:
        score.covered_unrelated.append(pair_id)


def rank_key(score: Score) -> tuple[int, int, int, int, int]:
    return (
        score.wrong,
        score.lost,
        -(score.resolved_restatement + score.handled_replacement),
        score.false_flag,
        len(score.unverified_supersedes),
    )


def best_of(
    rules: Sequence[Rule],
    cases: Mapping[str, Case],
    relation: Mapping[str, str],
    cover: CoverKey,
    only: frozenset[str] | None = None,
) -> Rule:
    return min(rules, key=lambda rule: rank_key(score_rule(rule, cases, relation, cover, only)))


def pool(parts: Iterable[Score]) -> Score:
    total = Score()

    for part in parts:
        for name in (
            "restatements",
            "reinforced_restatement",
            "superseded_restatement",
            "replacements",
            "superseded_replacement",
            "flagged_replacement",
            "unrelated",
            "lost_record",
            "lost_update",
            "false_flag",
        ):
            setattr(total, name, getattr(total, name) + getattr(part, name))

        total.superseded = [*total.superseded, *part.superseded]
        total.wrong_supersedes = [*total.wrong_supersedes, *part.wrong_supersedes]
        total.covered_unrelated = [*total.covered_unrelated, *part.covered_unrelated]
        total.unverified_supersedes = [*total.unverified_supersedes, *part.unverified_supersedes]
        total.reinforce_drops = [*total.reinforce_drops, *part.reinforce_drops]

    return total


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def rule_row(rule: Rule, score: Score) -> str:
    return (
        f"| {rule.name} "
        f"| {_ratio(score.resolved_restatement, score.restatements)} "
        f"({score.reinforced_restatement}+{score.superseded_restatement}) "
        f"| {_ratio(score.handled_replacement, score.replacements)} "
        f"({score.superseded_replacement}+{score.flagged_replacement}) "
        f"| {len(score.superseded)} | {score.wrong} | {len(score.covered_unrelated)} "
        f"| {len(score.unverified_supersedes)} "
        f"| {score.lost_record} | {score.lost_update} | {score.false_flag} "
        f"| {len(score.reinforce_drops)} | {'YES' if score.beats_bars else ''} |"
    )


RULE_HEAD = (
    "| rule | restatements resolved (reinforced+superseded) "
    "| replacements handled (superseded+flagged) | supersedes | wrong supersedes "
    "| covered follow_up or distinct supersedes | supersedes with no key "
    "| lost records | lost updates | false flags | reinforces that drop incoming detail "
    "| meets both bars |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def rule_table(rules: Sequence[Rule], scores: Mapping[str, Score]) -> str:
    return "\n".join([RULE_HEAD, *(rule_row(rule, scores[rule.name]) for rule in rules)])


def ranked(rules: Sequence[Rule], scores: Mapping[str, Score]) -> list[Rule]:
    return sorted(rules, key=lambda rule: rank_key(scores[rule.name]))


def frontier_table(rules: Sequence[Rule], scores: Mapping[str, Score]) -> str:
    best: dict[int, Rule] = {}

    for rule in ranked(rules, scores):
        score = scores[rule.name]

        if score.lost == 0 and score.wrong not in best:
            best[score.wrong] = rule

    lines = [
        "| wrong supersedes | most handled with no lost records | rule |",
        "|---|---|---|",
    ]

    for wrong in sorted(best)[:6]:
        score = scores[best[wrong].name]
        lines.append(
            f"| {wrong} | {score.resolved_restatement}/{score.restatements} restatements, "
            f"{score.handled_replacement}/{score.replacements} replacements | {best[wrong].name} |"
        )

    return "\n".join(lines)


def leave_one_source_out(
    rules: Sequence[Rule],
    cases: Mapping[str, Case],
    relation: Mapping[str, str],
    cover: CoverKey,
    sources: Mapping[str, str],
) -> str:
    names = sorted(set(sources.values()))
    everyone = frozenset(sources)
    parts: list[Score] = []
    lines = [
        "| held-out source | rule chosen on the other sources | restatements resolved "
        "| replacements handled | wrong supersedes | lost records | lost updates |",
        "|---|---|---|---|---|---|---|",
    ]

    for held in names:
        test = frozenset(pid for pid, source in sources.items() if source == held)
        chosen = best_of(rules, cases, relation, cover, everyone - test)
        score = score_rule(chosen, cases, relation, cover, test)
        parts.append(score)
        lines.append(_loso_row(held, chosen.name, score))

    lines.append(_loso_row("pooled held-out", "", pool(parts)))

    return "\n".join(lines)


def _loso_row(held: str, chosen: str, score: Score) -> str:
    return (
        f"| {held} | {chosen} "
        f"| {_ratio(score.resolved_restatement, score.restatements)} "
        f"| {_ratio(score.handled_replacement, score.replacements)} "
        f"| {score.wrong} | {score.lost_record} | {score.lost_update} |"
    )


class Confusion(BaseModel):
    true_yes_pred_yes: int = 0
    true_yes_pred_no: int = 0
    true_no_pred_yes: int = 0
    true_no_pred_no: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_yes_pred_yes
            + self.true_yes_pred_no
            + self.true_no_pred_yes
            + self.true_no_pred_no
        )

    @property
    def accuracy(self) -> float:
        right = self.true_yes_pred_yes + self.true_no_pred_no

        return right / self.total if self.total else 0.0

    def add(self, truth: bool, predicted: bool) -> None:
        if truth:
            self.true_yes_pred_yes += predicted
            self.true_yes_pred_no += not predicted
        else:
            self.true_no_pred_yes += predicted
            self.true_no_pred_no += not predicted


def confusion_at(pairs: Iterable[tuple[bool, float]], threshold: float) -> Confusion:
    table = Confusion()

    for truth, probability in pairs:
        table.add(truth, probability >= threshold)

    return table


def derived_enhances_key(cover: CoverKey) -> dict[str, bool]:
    return {
        pid: answers["new_covers_old"] and not answers["old_covers_new"]
        for pid, answers in cover.items()
        if "new_covers_old" in answers and "old_covers_new" in answers
    }


def signal_pairs(
    name: str,
    covers: Mapping[str, CoversSignal],
    cover: CoverKey,
    only: frozenset[str] | None = None,
) -> list[tuple[bool, float]]:
    truth_by_pair = (
        derived_enhances_key(cover)
        if name == "enhances"
        else {pid: answers[name] for pid, answers in cover.items() if name in answers}
    )
    found: list[tuple[bool, float]] = []

    for pair_id, truth in truth_by_pair.items():
        signal = covers.get(pair_id)
        probability = None if signal is None else getattr(signal, name).p_yes

        if probability is not None and (only is None or pair_id in only):
            found.append((truth, probability))

    return found


def best_threshold(pairs: Sequence[tuple[bool, float]]) -> tuple[float, Confusion]:
    options = [(threshold, confusion_at(pairs, threshold)) for threshold in THRESHOLDS]

    return max(options, key=lambda option: (option[1].accuracy, -option[0]))


def _confusion_cells(table: Confusion) -> str:
    return (
        f"{table.true_yes_pred_yes} | {table.true_yes_pred_no} "
        f"| {table.true_no_pred_yes} | {table.true_no_pred_no}"
    )


def jev_signal_table(
    covers: Mapping[str, CoversSignal], cover: CoverKey, only: frozenset[str] | None = None
) -> str:
    lines = [
        "| signal | pairs | share keyed yes | accuracy at 0.50 | best threshold | accuracy there "
        "| key yes, signal yes | key yes, signal no | key no, signal yes | key no, signal no |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for name in (*QUESTIONS, "enhances"):
        pairs = signal_pairs(name, covers, cover, only)

        if not pairs:
            continue

        threshold, table = best_threshold(pairs)
        yes_share = sum(1 for truth, _ in pairs if truth) / len(pairs)
        lines.append(
            f"| {name} | {len(pairs)} | {yes_share:.0%} | {confusion_at(pairs, 0.5).accuracy:.1%} "
            f"| {threshold:.2f} | {table.accuracy:.1%} | {_confusion_cells(table)} |"
        )

    return "\n".join(lines)


def sonnet_confusions(calls: Mapping[str, SonnetCall], cover: CoverKey) -> dict[str, Confusion]:
    tables = {name: Confusion() for name in QUESTIONS}

    for pair_id, call in calls.items():
        if not call.answered:
            continue

        for name in QUESTIONS:
            truth = cover.get(pair_id, {}).get(name)
            answer = getattr(call, name)

            if truth is not None and answer is not None:
                tables[name].add(truth, answer == "yes")

    return tables


def sonnet_signal_table(calls: Mapping[str, SonnetCall], cover: CoverKey) -> str:
    lines = [
        "| question | pairs | accuracy | key yes, sonnet yes | key yes, sonnet no "
        "| key no, sonnet yes | key no, sonnet no |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, table in sonnet_confusions(calls, cover).items():
        lines.append(
            f"| {name} | {table.total} | {table.accuracy:.1%} | {_confusion_cells(table)} |"
        )

    return "\n".join(lines)


def agreement_table(labels: Sequence[LabelRow]) -> str:
    lines = [
        "| question | items both judges labelled | agreed | raw agreement | Cohen's kappa "
        "| keyed yes | keyed no |",
        "|---|---|---|---|---|---|---|",
    ]

    for name in QUESTIONS:
        subset = [row for row in labels if split_item_id(row.item_id)[1] == name]
        stats = agreement_stats(subset)
        confusion = stats["confusion"]
        yes_yes = confusion.get("yes", {}).get("yes", 0) if isinstance(confusion, dict) else 0
        no_no = confusion.get("no", {}).get("no", 0) if isinstance(confusion, dict) else 0
        lines.append(
            f"| {name} | {stats['items_labelled_by_both']} | {stats['items_agreed']} "
            f"| {float(str(stats['raw_agreement'])):.1%} | {float(str(stats['cohens_kappa'])):.3f} "
            f"| {yes_yes} | {no_no} |"
        )

    return "\n".join(lines)


def repeatability_table(
    first: Mapping[str, CoversSignal], second: Mapping[str, CoversSignal]
) -> str:
    shared = sorted(set(first) & set(second))
    lines = [
        "| question | pairs run twice | answers that flipped | mean absolute change in p(yes) "
        "| largest change | pairs crossing 0.70 |",
        "|---|---|---|---|---|---|",
    ]

    for name in (*QUESTIONS, "enhances"):
        gaps: list[float] = []
        flips = 0
        crossings = 0

        for pair_id in shared:
            one, two = getattr(first[pair_id], name), getattr(second[pair_id], name)

            if one.p_yes is None or two.p_yes is None:
                continue

            gaps.append(abs(one.p_yes - two.p_yes))
            flips += one.answer != two.answer
            crossings += (one.p_yes >= 0.7) != (two.p_yes >= 0.7)

        mean = sum(gaps) / len(gaps) if gaps else 0.0
        lines.append(
            f"| {name} | {len(gaps)} | {flips} | {mean:.3f} | {max(gaps, default=0.0):.3f} "
            f"| {crossings} |"
        )

    return "\n".join(lines)


def v1_counts(published: Sequence[SaveRow], relation: Mapping[str, str]) -> Score:
    score = Score()

    for row in published:
        label = relation.get(row.pair_id)
        noticed = row.bucket in ("caught", "flagged")

        if label == "restatement":
            score.restatements += 1
            score.reinforced_restatement += noticed
        elif label == "replacement":
            score.replacements += 1
            score.flagged_replacement += row.bucket == "flagged"
            score.lost_update += row.bucket == "caught"
        elif label in UNRELATED:
            score.unrelated += 1
            score.lost_record += row.bucket == "caught"
            score.false_flag += row.bucket == "flagged"

    return score


def headline_table(rows: Sequence[tuple[str, Score, bool]]) -> str:
    lines = [
        "| system | restatements resolved | replacements handled | wrong supersedes "
        "| lost records | lost updates | false flags |",
        "|---|---|---|---|---|---|---|",
    ]

    for name, score, automatic in rows:
        wrong = str(score.wrong) if automatic else "n/a"
        lines.append(
            f"| {name} | {_ratio(score.resolved_restatement, score.restatements)} "
            f"| {_ratio(score.handled_replacement, score.replacements)} | {wrong} "
            f"| {score.lost_record}/{score.unrelated} | {score.lost_update}/{score.replacements} "
            f"| {score.false_flag} |"
        )

    return "\n".join(lines)


@dataclass(frozen=True)
class Inputs:
    pairs: Mapping[str, PairRow]
    cases: Mapping[str, Case]
    relation: Mapping[str, str]
    cover: CoverKey
    labels: Sequence[LabelRow]
    covers: Mapping[str, CoversSignal]
    repeat: Mapping[str, CoversSignal]
    sonnet: Mapping[str, SonnetCall]
    published_v1: Sequence[SaveRow]
    published_v2: Sequence[SaveRow]
    signals: Mapping[str, SignalRow]
    listed: Sequence[str]


def pair_block(pair: PairRow, inputs: Inputs, score_label: str, extra: str) -> list[str]:
    case = inputs.cases[pair.pair_id]
    answers = inputs.cover.get(pair.pair_id, {})
    keyed = ", ".join(f"{name}={'yes' if value else 'no'}" for name, value in answers.items())

    return [
        f"### {pair.pair_id} ({pair.source}, relation key: {score_label})",
        "",
        f"Jev relation pick: {case.pick}; similarity "
        f"{'none' if case.similarity is None else f'{case.similarity:.3f}'}; "
        f"containment key: {keyed or 'none'}. {extra}",
        "",
        f"- existing: {' '.join(pair.existing.split())}",
        f"- incoming: {' '.join(pair.incoming.split())}",
        "",
    ]


def wrong_supersedes_text(entries: Sequence[tuple[str, Rule, Score]], inputs: Inputs) -> str:
    lines = [
        "# Wrong supersedes",
        "",
        "Holds record text; stays under dev-corpus. A wrong supersede is a pair the rule retires "
        "the existing record for, where the containment key says the incoming note does not "
        "state everything the existing note states.",
        "",
    ]

    for title, rule, score in entries:
        split = Counter(inputs.relation.get(pid, "disputed") for pid in score.wrong_supersedes)
        lines += [
            f"## {title}: `{rule.name}` ({score.wrong})",
            "",
            "By relation key: "
            + (", ".join(f"{name} {count}" for name, count in sorted(split.items())) or "none")
            + ".",
            "",
        ]

        for pair_id in score.wrong_supersedes:
            case = inputs.cases[pair_id]
            probability = case.p_yes("new_covers_old")
            note = (
                f"new_covers_old p(yes)={probability:.2f}."
                if probability is not None
                else "no Jev containment signal."
            )
            lines += pair_block(
                inputs.pairs[pair_id], inputs, inputs.relation.get(pair_id, "disputed"), note
            )

    return "\n".join(lines)


def _probabilities(case: Case) -> str:
    if case.covers is None:
        return "none"

    parts = []

    for name in (*QUESTIONS, "enhances"):
        probability = getattr(case.covers, name).p_yes
        parts.append("n/a" if probability is None else f"{probability:.2f}")

    return " / ".join(parts)


def _judge_cell(labels: Sequence[LabelRow], pair_id: str, question: str) -> str:
    said = {row.judge: str(row.label) for row in labels if row.item_id == f"{pair_id}#{question}"}

    return "/".join(said.get(judge, "-") for judge in ("claude", "codex"))


def hard_cases_text(inputs: Inputs, rules: Mapping[str, Rule]) -> str:
    silent = [
        pid
        for pid, label in inputs.relation.items()
        if label == "restatement" and inputs.cases[pid].pick == "follow_up"
    ]
    unmerged = [
        pid
        for pid, label in inputs.relation.items()
        if label == "restatement" and BASELINE.decide(inputs.cases[pid]) != "reinforce"
    ]
    extra = [pid for pid in inputs.listed if pid in inputs.cases and pid not in unmerged]
    ordered = [*silent, *[pid for pid in unmerged if pid not in silent], *extra]
    head = (
        "| pair_id | relation key | Jev pick | similarity "
        "| p(yes): new_covers_old / old_covers_new / enhances "
        "| judges new_covers_old (claude/codex) | judges old_covers_new (claude/codex) "
        "| sonnet new_covers_old / old_covers_new | "
        + " | ".join(f"outcome: {title}" for title in rules)
        + " |\n|---|---|---|---|---|---|---|---|"
        + "---|" * len(rules)
    )
    lines = [
        "# Hard cases",
        "",
        "Stays under dev-corpus. The first "
        f"{len(silent)} rows are the restatements v2 saved silently because Jev picked follow_up; "
        f"then {len(unmerged) - len(silent)} other restatements the baseline does not reinforce; "
        "then pairs named in second-judge/lost.md or dup-rules/hard-cases.md "
        f"({len(extra)} more).",
        "",
        head,
    ]

    for pair_id in ordered:
        case = inputs.cases[pair_id]
        call = case.sonnet
        sonnet = "not asked" if call is None else f"{call.new_covers_old} / {call.old_covers_new}"
        outcomes = " | ".join(rule.decide(case) for rule in rules.values())
        lines.append(
            f"| {pair_id} | {inputs.relation.get(pair_id, 'disputed')} | {case.pick} "
            f"| {'none' if case.similarity is None else f'{case.similarity:.3f}'} "
            f"| {_probabilities(case)} "
            f"| {_judge_cell(inputs.labels, pair_id, 'new_covers_old')} "
            f"| {_judge_cell(inputs.labels, pair_id, 'old_covers_new')} | {sonnet} | {outcomes} |"
        )

    return "\n".join(lines)


def latency_text(covers: Mapping[str, CoversSignal], sonnet: Mapping[str, SonnetCall]) -> str:
    jev = sorted(row.elapsed_ms for row in covers.values() if row.error is None)
    llm = sorted(call.elapsed_ms for call in sonnet.values() if call.answered)

    return "\n".join(
        [
            "| signal | calls | median ms | p95 ms |",
            "|---|---|---|---|",
            f"| Jev, three questions in one call | {len(jev)} | {median(jev)} "
            f"| {percentile(jev, 0.95)} |",
            f"| Sonnet, both questions in one call | {len(llm)} | {median(llm)} "
            f"| {percentile(llm, 0.95)} |",
        ]
    )


def verdict_text(
    jev_best: tuple[Rule, Score], sonnet_best: tuple[Rule, Score], winners: int
) -> str:
    sentences = []

    if winners:
        sentences.append(
            f"{winners} rule(s) reach {RESTATEMENT_BAR}+ restatements and {REPLACEMENT_BAR} "
            "replacements with zero wrong supersedes and zero lost records."
        )
    else:
        sentences.append(
            f"No rule reaches {RESTATEMENT_BAR}+ restatements and {REPLACEMENT_BAR} "
            "replacements with zero wrong supersedes and zero lost records."
        )

    for title, (rule, score) in (("Jev", jev_best), ("Sonnet", sonnet_best)):
        gap_rs = RESTATEMENT_BAR - score.resolved_restatement
        gap_rp = REPLACEMENT_BAR - score.handled_replacement
        misses = []

        if gap_rs > 0:
            misses.append(f"restatements by {gap_rs}")

        if gap_rp > 0:
            misses.append(f"replacements by {gap_rp}")

        if score.wrong:
            misses.append(f"{score.wrong} wrong supersede(s)")

        if score.lost:
            misses.append(f"{score.lost} lost record(s) or update(s)")

        sentences.append(
            f"Best {title} rule `{rule.name}` resolves {score.resolved_restatement} restatements "
            f"and handles {score.handled_replacement} replacements; it misses the bars on "
            + (", ".join(misses) if misses else "nothing")
            + "."
        )

    return " ".join(sentences)


def summary_text(inputs: Inputs, sweep: dict[str, Score], jev: list[Rule], son: list[Rule]) -> str:
    relation, cover, cases = inputs.relation, inputs.cover, inputs.cases
    sources = {pid: inputs.pairs[pid].source for pid in cases}
    jev_best = best_of(jev, cases, relation, cover)
    son_best = best_of(son, cases, relation, cover)
    winners = [rule for rule in [*jev, *son] if sweep[rule.name].beats_bars]
    only_keyed = {pid: label for pid, label in relation.items() if pid in cases}
    v1 = v1_counts(inputs.published_v1, only_keyed)
    v2_published = v1_counts(inputs.published_v2, only_keyed)
    noticed, flagged = v1_bars(inputs.published_v1, only_keyed)
    band = band_pair_ids(inputs.signals, None)
    answered = sum(1 for pid in band if cases[pid].sonnet is not None)
    band_only = frozenset(band)
    v2_merged = sum(
        1
        for row in inputs.published_v2
        if only_keyed.get(row.pair_id) == "restatement" and row.bucket == "caught"
    )

    return "\n".join(
        [
            "# Containment experiment",
            "",
            verdict_text(
                (jev_best, sweep[jev_best.name]), (son_best, sweep[son_best.name]), len(winners)
            ),
            "",
            f"Pairs: {len(cases)}; relation-keyed: {len(only_keyed)}; containment-keyed "
            f"new_covers_old: {sum(1 for a in cover.values() if 'new_covers_old' in a)}, "
            f"old_covers_new: {sum(1 for a in cover.values() if 'old_covers_new' in a)}. "
            f"Sonnet arm: {answered} of {len(band)} follow_up-band pairs answered.",
            "",
            "## Headline",
            "",
            headline_table(
                [
                    ("v2 baseline (Jev pick, signal replay)", sweep["baseline"], True),
                    ("v2 as published (save-v2.jsonl)", v2_published, False),
                    ("v1 as published (save-v1.jsonl)", v1, False),
                    (f"best Jev rule: {jev_best.name}", sweep[jev_best.name], True),
                    (f"best Sonnet rule: {son_best.name}", sweep[son_best.name], True),
                ]
            ),
            "",
            f"v1 notices {noticed} restatements and flags {flagged} replacements on this key; "
            f"v2 as published reinforces {v2_merged} and notices "
            f"{v2_published.resolved_restatement}. Resolved means reinforced or superseded for the "
            "rules, caught or flagged for the published runs. Handled means superseded or flagged. "
            "Wrong supersedes are counted against the containment key, "
            "so a pair the judges disagreed on cannot be wrong.",
            "",
            "## Containment key: judge agreement",
            "",
            agreement_table(inputs.labels),
            "",
            "## Added latency per save",
            "",
            latency_text(inputs.covers, inputs.sonnet),
            "",
            "## Repeatability on a 40-pair sample (two runs)",
            "",
            repeatability_table(inputs.covers, inputs.repeat),
            "",
            "## How well each Jev signal matches the containment key",
            "",
            "The enhances signal is compared with the derived key new_covers_old yes and "
            "old_covers_new no, because no judge was asked that question directly.",
            "",
            jev_signal_table(inputs.covers, cover),
            "",
            "## Sonnet against the containment key (follow_up band only)",
            "",
            sonnet_signal_table(inputs.sonnet, cover),
            "",
            "Jev on the same band:",
            "",
            jev_signal_table(inputs.covers, cover, band_only),
            "",
            "## Best Jev rule: wrong supersedes by relation key",
            "",
            _split_text(sweep[jev_best.name], relation),
            "",
            "## Best Sonnet rule: wrong supersedes by relation key",
            "",
            _split_text(sweep[son_best.name], relation),
            "",
            "## Frontier, Jev rules",
            "",
            frontier_table(jev, sweep),
            "",
            "## Frontier, Sonnet rules",
            "",
            frontier_table(son, sweep),
            "",
            "## Leave one source out, Jev rules (rule chosen on three sources)",
            "",
            leave_one_source_out(jev, cases, relation, cover, {p: sources[p] for p in relation}),
            "",
            "## Leave one source out, Sonnet rules",
            "",
            leave_one_source_out(son, cases, relation, cover, {p: sources[p] for p in relation}),
            "",
            "## Files",
            "",
            "- sweep.md, wrong-supersedes.md, hard-cases.md, signals.jsonl, signals-repeat.jsonl, "
            "labels-covers.jsonl, key-covers.jsonl, sonnet-calls.jsonl",
            "",
        ]
    )


def _split_text(score: Score, relation: Mapping[str, str]) -> str:
    wrong = Counter(relation.get(pid, "disputed") for pid in score.wrong_supersedes)
    covered = Counter(relation.get(pid, "disputed") for pid in score.covered_unrelated)

    return (
        f"Wrong supersedes: {score.wrong} ("
        + (", ".join(f"{name} {count}" for name, count in sorted(wrong.items())) or "none")
        + f"). Supersedes the containment key backs on pairs the relation key calls follow_up or "
        f"distinct: {len(score.covered_unrelated)} ("
        + (", ".join(f"{name} {count}" for name, count in sorted(covered.items())) or "none")
        + f"). Supersedes with no containment key: {len(score.unverified_supersedes)}."
    )


def load_inputs(base: Path, out_dir: Path) -> Inputs:
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    covers = load_covers(out_dir / "signals.jsonl")
    repeat = load_covers(out_dir / "signals-repeat.jsonl")
    sonnet = load_sonnet_calls(out_dir / "sonnet-calls.jsonl")
    relation = {
        pid: label for pid, label in load_key(base / "key-relation.jsonl").items() if pid in signals
    }
    listed = _listed_pairs(
        [
            base / "experiments" / "second-judge" / "lost.md",
            base / "experiments" / "dup-rules" / "hard-cases.md",
        ]
    )

    return Inputs(
        pairs=pairs,
        cases=build_cases(signals, covers, sonnet),
        relation=relation,
        cover=load_cover_key(out_dir / "key-covers.jsonl"),
        labels=load_cover_labels(out_dir / "labels-covers.jsonl"),
        covers=covers,
        repeat=repeat,
        sonnet=sonnet,
        published_v1=[
            row for row in read_jsonl(base / "save-v1.jsonl", SaveRow) if row.system == "v1"
        ],
        published_v2=[
            row for row in read_jsonl(base / "save-v2.jsonl", SaveRow) if row.system == "v2"
        ],
        signals=signals,
        listed=listed,
    )


def _listed_pairs(files: Sequence[Path]) -> list[str]:
    found: list[str] = []

    for path in files:
        for pair_id in LISTED_PAIR.findall(path.read_text(encoding="utf-8")):
            if pair_id not in found:
                found.append(pair_id)

    return found


def run(base: Path, out_dir: Path) -> dict[str, Score]:
    inputs = load_inputs(base, out_dir)
    jev, son = jev_rules(), sonnet_rules()
    every = [BASELINE, *jev, *son]
    sweep = {
        rule.name: score_rule(rule, inputs.cases, inputs.relation, inputs.cover) for rule in every
    }
    jev_best = best_of(jev, inputs.cases, inputs.relation, inputs.cover)
    son_best = best_of(son, inputs.cases, inputs.relation, inputs.cover)

    out_dir.joinpath("sweep.md").write_text(
        rule_table(ranked(every, sweep), sweep) + "\n", encoding="utf-8"
    )
    out_dir.joinpath("wrong-supersedes.md").write_text(
        wrong_supersedes_text(
            [
                ("Best Jev rule", jev_best, sweep[jev_best.name]),
                ("Best Sonnet rule", son_best, sweep[son_best.name]),
            ],
            inputs,
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir.joinpath("hard-cases.md").write_text(
        hard_cases_text(
            inputs, {"baseline": BASELINE, "best Jev": jev_best, "best Sonnet": son_best}
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir.joinpath("summary.md").write_text(
        summary_text(inputs, sweep, jev, son) + "\n", encoding="utf-8"
    )

    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the containment save rule.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "covers")
    args = parser.parse_args()
    sweep = run(args.base, args.out_dir)

    meeting = sum(1 for score in sweep.values() if score.beats_bars)

    print(f"{len(sweep)} rules scored; {meeting} meet both bars")


if __name__ == "__main__":
    main()
