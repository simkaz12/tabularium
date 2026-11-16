from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval_v1v2 import agent_fold_ask as ask
from eval_v1v2 import agent_fold_recount as recount
from eval_v1v2 import fold, fold_guard, paths
from eval_v1v2.covers import BASELINE, REPLACEMENT_BAR, RESTATEMENT_BAR, Case, v1_counts
from eval_v1v2.judges import Item, run_task
from eval_v1v2.rows import LabelRow, read_jsonl
from eval_v1v2.score_save import median, percentile

Reading = fold_guard.Reading
READINGS = fold_guard.READINGS
GUARDS: tuple[tuple[str, float | None], ...] = (
    ("off", None),
    ("coverage", 0.70),
    ("coverage", 0.90),
)
TRIGGERS: tuple[fold_guard.GuardRule, ...] = (
    fold_guard.GuardRule(0.75, None, None, None, "off", None),
    fold_guard.GuardRule(0.80, None, None, None, "off", None),
    fold_guard.GuardRule(0.80, None, 0.70, None, "off", None),
    fold_guard.GuardRule(0.80, 0.70, None, None, "off", None),
    fold_guard.GuardRule(0.85, 0.90, None, None, "off", None),
)
WIDE_TRIGGER = TRIGGERS[2]
LABELS = ("restatement", "replacement", "follow_up", "distinct", "disputed")
LAZY_THRESHOLDS = (0.50, 0.70, 0.90)
NEAR_COPY = 0.95
SHAPE_ORDER = {f"{wording}-{framing}": index for index, (wording, framing) in enumerate(ask.SHAPES)}

CRITERION = (
    "Best arm per model is chosen on the choice replies alone, before any fold is judged: the "
    "lowest easy-exit rate (share of true restatements in the band where the agent did not fold), "
    "ties broken by the lowest wrong-fold rate (share of true distinct pairs folded), then by arm "
    "order choice-cold, fold_default-cold, fold_default-in_task. Rates use answered pairs only. "
    "The fold_default / in_task arm of each model is judged as well when it is a different arm."
)

UPPER_BOUND = (
    "A cold model, or a model handed a scripted in-task paragraph, is not an agent under real "
    "task pressure, so every agent figure here is an upper bound on agent behaviour."
)

CREDIT_RULE = (
    "Strict counts a disputed fold as an error and not resolved. A fold earns replacement "
    "credit under strict only when the judges grade it good, all four answers keyed yes. Under "
    "loose it earns credit when it is not graded bad and newer_wins and nothing_invented are "
    "both keyed yes. Experiments 4 and 5 used the narrower two-answer credit under both "
    "readings (newer_wins and nothing_invented keyed yes, whatever the other two answers), so "
    "a disputed fold could count as a strict catch there; the server-side row below is that "
    "experiment 5 figure and keeps it."
)

JUDGE_SEAT = (
    "The claude judge seat is Opus, so Opus-written folds are judged partly by the same model "
    "family. The last column is the accepted bad count when Codex alone grades the same "
    "rule; the accepted bad column before it is the two-judge count."
)

LIMITS = (
    "A cold model, or a model handed a scripted in-task paragraph, is not an agent under real "
    "task pressure: it has no tool loop, no half-finished edit to get back to and no history of "
    "earlier saves, and the in-task text is the same invented paragraph for every pair, unrelated "
    "to the notes. These numbers are an upper bound on agent behaviour. The band is 132 pairs "
    "with 10 true restatements and 4 true replacements, so a single pair moves a restatement "
    "rate by 10 points. The judges are two models, not people, and the claude seat is Opus. "
    "Some band pairs may carry project names in the note text. Scope exclusion is applied by "
    "record scope, not by note text. The lazy-resend table is a computed proxy that assumes a "
    "resent "
    "note's fold_covers_old equals the pair's new_covers_old; no real near-copy of the agent's "
    "own note was observed in any arm."
)


@dataclass(frozen=True)
class AgentData:
    guard: fold_guard.GuardData
    asks: Mapping[tuple[str, str], ask.AskRow]
    grades: Mapping[tuple[str, str], fold.FoldGrade]
    signals: Mapping[tuple[str, str], fold_guard.GuardSignal]
    labels: Sequence[LabelRow]
    out_dir: Path

    @property
    def inputs(self) -> fold.Inputs:
        return self.guard.inputs

    @property
    def band(self) -> Sequence[fold.BandRow]:
        return self.guard.inputs.band


def load_data(base: Path, out_dir: Path) -> AgentData:
    labels_path = out_dir / "labels-fold.jsonl"
    key_path = out_dir / "key-fold.jsonl"

    return AgentData(
        guard=fold_guard.load_data(base, base / "experiments" / "fold-guard"),
        asks=ask.load_asks(out_dir / "asks.jsonl"),
        grades=fold.load_fold_grades(key_path) if key_path.exists() else {},
        signals=fold_guard.load_guard_signals(out_dir / "guard-signals.jsonl"),
        labels=read_jsonl(labels_path, LabelRow) if labels_path.exists() else [],
        out_dir=out_dir,
    )


def label_of(row: fold.BandRow) -> str:
    return row.relation or "disputed"


def arm_ids(data: AgentData) -> list[str]:
    return sorted({arm for _, arm in data.asks}, key=arm_order)


def arm_order(arm_id: str) -> tuple[int, int]:
    model = ask.arm_model(arm_id)
    shape = arm_id.partition("-")[2]

    return (ask.MODELS.index(model) if model in ask.MODELS else len(ask.MODELS), SHAPE_ORDER[shape])


def answered_rows(data: AgentData, arm_id: str) -> list[tuple[fold.BandRow, ask.AskRow]]:
    pairs: list[tuple[fold.BandRow, ask.AskRow]] = []

    for row in data.band:
        reply = data.asks.get((row.pair_id, arm_id))

        if reply is not None and reply.answered:
            pairs.append((row, reply))

    return pairs


def rate(count: int, total: int) -> float | None:
    return count / total if total else None


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def ratio(count: int, total: int) -> str:
    return f"{count}/{total}"


def ms(values: Sequence[float], tail: bool = False) -> str:
    ordered = sorted(values)
    value = percentile(ordered, 0.9) if tail else median(ordered)

    return "n/a" if value is None else f"{value:.0f}"


def easy_exit_rate(data: AgentData, arm_id: str) -> float | None:
    rows = [reply for row, reply in answered_rows(data, arm_id) if label_of(row) == "restatement"]

    return rate(sum(1 for reply in rows if not reply.folded), len(rows))


def wrong_fold_rate(data: AgentData, arm_id: str) -> float | None:
    rows = [reply for row, reply in answered_rows(data, arm_id) if label_of(row) == "distinct"]

    return rate(sum(1 for reply in rows if reply.folded), len(rows))


def selection_key(data: AgentData, arm_id: str) -> tuple[float, float, int]:
    easy = easy_exit_rate(data, arm_id)
    wrong = wrong_fold_rate(data, arm_id)

    return (1.0 if easy is None else easy, 1.0 if wrong is None else wrong, arm_order(arm_id)[1])


def select_arms(data: AgentData) -> dict[str, list[str]]:
    chosen: dict[str, list[str]] = {}
    everything = arm_ids(data)

    for model in ask.MODELS:
        own = [
            arm for arm in everything if ask.arm_model(arm) == model and answered_rows(data, arm)
        ]

        if not own:
            continue

        best = min(own, key=lambda arm: selection_key(data, arm))
        picks = [best]
        in_task = f"{model}-fold_default-in_task"

        if in_task in own and in_task != best:
            picks.append(in_task)

        chosen[model] = picks

    return chosen


def labelled_arms(labels: Sequence[LabelRow]) -> set[str]:
    return {fold.split_fold_item(row.item_id)[1] for row in labels}


def judged_arms(data: AgentData) -> list[str]:
    picked = {arm for arms in select_arms(data).values() for arm in arms}

    return sorted(picked | labelled_arms(data.labels), key=arm_order)


def fold_items(data: AgentData, arms: Sequence[str]) -> list[Item]:
    items: list[Item] = []
    band = {row.pair_id: row for row in data.band}

    for arm in arms:
        for row in data.band:
            reply = data.asks.get((row.pair_id, arm))

            if reply is None or not reply.folded or reply.note is None:
                continue

            source = band[row.pair_id]
            items += [
                Item(
                    item_id=fold.fold_item_id(row.pair_id, arm, question),
                    fields={
                        "question": question,
                        "existing": source.existing,
                        "incoming": source.incoming,
                        "fold": reply.note,
                    },
                )
                for question in fold.QUESTIONS
            ]

    return items


def guard_target_rows(data: AgentData, arms: Sequence[str]) -> list[dict[str, object]]:
    return [
        {"pair_id": row.pair_id, "writer": arm, "text": reply.note, "error": None}
        for arm in arms
        for row in data.band
        if (reply := data.asks.get((row.pair_id, arm))) is not None and reply.folded
    ]


@dataclass(frozen=True)
class AgentRule:
    arm: str
    trigger: fold_guard.GuardRule
    kind: str
    threshold: float | None
    data: AgentData

    @property
    def name(self) -> str:
        guard = "" if self.threshold is None else f" guard={self.kind}>={self.threshold:.2f}"

        return f"{self.arm}, {self.trigger.name}{guard}"

    @property
    def model(self) -> str:
        return ask.arm_model(self.arm)

    def asked(self, case: Case) -> bool:
        similarity = case.similarity

        if fold.CONTAINMENT.decide(case) != "new" or case.pick != "follow_up":
            return False

        if similarity is None or similarity < fold.BAND_FLOOR:
            return False

        bound = fold_guard.BoundRule(rule=self.trigger, writer="", data=self.data.guard)

        return bound.by_gate(case, similarity)

    def reply(self, case: Case) -> ask.AskRow | None:
        return self.data.asks.get((case.pair_id, self.arm))

    def passes_guard(self, case: Case) -> bool:
        names = fold_guard.GUARD_KINDS[self.kind]

        if not names or self.threshold is None:
            return True

        signal = self.data.signals.get((case.pair_id, self.arm))

        if signal is None:
            return False

        readings = [signal.reading(name).p_yes for name in names]

        return all(value is not None and value >= self.threshold for value in readings)

    def accepts(self, case: Case) -> bool:
        reply = self.reply(case)

        return self.asked(case) and reply is not None and reply.folded and self.passes_guard(case)

    def decide(self, case: Case) -> fold.Action:
        base = fold.CONTAINMENT.decide(case)

        if base != "new":
            return base

        return "fold" if self.accepts(case) else "new"


def rules_for(data: AgentData, arms: Sequence[str]) -> list[AgentRule]:
    return [
        AgentRule(arm, trigger, kind, threshold, data)
        for arm in arms
        for trigger in TRIGGERS
        for kind, threshold in GUARDS
    ]


def score_agent(
    rule: AgentRule,
    only: frozenset[str] | None = None,
    grades: Mapping[str, fold.FoldGrade] | None = None,
) -> fold.FoldScore:
    inputs = rule.data.inputs
    chosen = grades if grades is not None else fold.writer_grades(rule.data.grades, rule.arm)

    score = fold.score_fold(rule, inputs.cases, inputs.relation, inputs.cover, chosen, only)
    credit_replacements(score, rule, inputs, chosen, only)

    return score


def credit_replacements(
    score: fold.FoldScore,
    rule: fold.Decider,
    inputs: fold.Inputs,
    grades: Mapping[str, fold.FoldGrade],
    only: frozenset[str] | None,
) -> None:
    score.folded_good_replacement = 0
    score.folded_unbad_replacement = 0

    for pair_id, case in inputs.cases.items():
        if (only is not None and pair_id not in only) or rule.decide(case) != "fold":
            continue

        if inputs.relation.get(pair_id) != "replacement":
            continue

        grade = grades.get(pair_id, fold.FoldGrade())
        keyed_clean = (
            grade.answers.get("newer_wins") is True
            and grade.answers.get("nothing_invented") is True
        )
        score.folded_good_replacement += grade.status == "good"
        score.folded_unbad_replacement += grade.status != "bad" and keyed_clean


@dataclass
class AskStats:
    asked: int = 0
    replied_fold: int = 0
    accepted: int = 0
    keep_both: int = 0
    continued: int = 0
    invalid: int = 0
    unanswered: int = 0
    rejected_good: int = 0
    rejected_bad: int = 0
    rejected_disputed: int = 0


def ask_stats(rule: AgentRule) -> tuple[AskStats, list[float]]:
    stats = AskStats()
    latencies: list[float] = []
    grades = fold.writer_grades(rule.data.grades, rule.arm)

    for case in rule.data.inputs.cases.values():
        if not rule.asked(case):
            continue

        stats.asked += 1
        reply = rule.reply(case)

        if reply is None or not reply.answered:
            stats.unanswered += 1

            continue

        latencies.append(reply.elapsed_ms)

        if reply.action == "keep_both":
            stats.keep_both += 1
        elif reply.action == "continue":
            stats.continued += 1
        elif reply.action == "invalid":
            stats.invalid += 1

        if not reply.folded:
            continue

        stats.replied_fold += 1

        if rule.passes_guard(case):
            stats.accepted += 1

            continue

        status = grades.get(case.pair_id, fold.FoldGrade()).status
        stats.rejected_good += status == "good"
        stats.rejected_bad += status == "bad"
        stats.rejected_disputed += status == "disputed"

    return stats, latencies


def guard_score(score: fold.FoldScore) -> fold_guard.GuardScore:
    return fold_guard.GuardScore(result=score)


def rank(score: fold.FoldScore, reading: Reading) -> tuple[int, int, int, int]:
    return fold_guard.rank_reading(guard_score(score), reading)


def best_rule(
    rules: Sequence[AgentRule], reading: Reading, only: frozenset[str] | None = None
) -> AgentRule:
    return min(rules, key=lambda rule: rank(score_agent(rule, only), reading))


def resolved(score: fold.FoldScore, reading: Reading) -> int:
    return guard_score(score).resolved(reading)


def caught(score: fold.FoldScore, reading: Reading) -> int:
    return guard_score(score).caught(reading)


def meets(score: fold.FoldScore, reading: Reading) -> bool:
    return guard_score(score).meets_bars(reading)


def split_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "none"


def tally_by_label(data: AgentData, arm_id: str) -> dict[str, Counter[str]]:
    table: dict[str, Counter[str]] = {label: Counter() for label in LABELS}

    for row in data.band:
        reply = data.asks.get((row.pair_id, arm_id))
        outcome = "pending" if reply is None or not reply.answered else str(reply.action)
        table[label_of(row)][outcome] += 1

    return table


def choice_overview(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| arm | answered | fold | keep_both | continue | invalid | pending | fold rate "
        "| keep_both rate | continue rate | easy-exit rate (restatements) "
        "| wrong-fold rate (distinct) | median ms | p90 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for arm in arms:
        rows = answered_rows(data, arm)
        counts = Counter(str(reply.action) for _, reply in rows)
        total = len(rows)
        restated = sum(1 for row, _ in rows if label_of(row) == "restatement")
        distinct = sum(1 for row, _ in rows if label_of(row) == "distinct")
        times = [reply.elapsed_ms for _, reply in rows]
        easy = easy_exit_rate(data, arm)
        wrong = wrong_fold_rate(data, arm)
        lines.append(
            f"| {arm} | {total} | {counts['fold']} | {counts['keep_both']} | {counts['continue']} "
            f"| {counts['invalid']} | {len(data.band) - total} "
            f"| {pct(rate(counts['fold'], total))} | {pct(rate(counts['keep_both'], total))} "
            f"| {pct(rate(counts['continue'], total))} "
            f"| {pct(easy)} ({restated} restatements) | {pct(wrong)} ({distinct} distinct) "
            f"| {ms(times)} | {ms(times, True)} |"
        )

    return "\n".join(lines)


def choice_by_label(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| arm | relation key | pairs | fold | keep_both | continue | invalid | pending |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for arm in arms:
        for label, counts in tally_by_label(data, arm).items():
            lines.append(
                f"| {arm} | {label} | {sum(counts.values())} | {counts['fold']} "
                f"| {counts['keep_both']} | {counts['continue']} | {counts['invalid']} "
                f"| {counts['pending']} |"
            )

    return "\n".join(lines)


def proxy_answer(data: AgentData, pair_id: str) -> str:
    grade = data.inputs.grades.get((pair_id, "sonnet"))
    answer = None if grade is None else grade.answers.get("one_record")

    return "unkeyed" if answer is None else "yes" if answer else "no"


def follow_up_proxy(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| arm | follow_up pairs | folded, proxy one_record yes | folded, proxy no "
        "| folded, proxy unkeyed | not folded, proxy yes | not folded, proxy no "
        "| not folded, proxy unkeyed | choice agrees with proxy (of keyed) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for arm in arms:
        cells: Counter[tuple[bool, str]] = Counter()

        for row, reply in answered_rows(data, arm):
            if label_of(row) == "follow_up":
                cells[(reply.folded, proxy_answer(data, row.pair_id))] += 1

        agree = cells[(True, "yes")] + cells[(False, "no")]
        keyed = agree + cells[(True, "no")] + cells[(False, "yes")]
        lines.append(
            f"| {arm} | {sum(cells.values())} | {cells[(True, 'yes')]} | {cells[(True, 'no')]} "
            f"| {cells[(True, 'unkeyed')]} | {cells[(False, 'yes')]} | {cells[(False, 'no')]} "
            f"| {cells[(False, 'unkeyed')]} | {ratio(agree, keyed)} |"
        )

    return "\n".join(lines)


def shift_table(data: AgentData) -> str:
    lines = [
        "| model | comparison | pairs answered in both | fold rate before | fold rate after "
        "| pairs whose choice changed | folded before, not after | not folded before, folded "
        "after | continue rate after |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    pairs = (
        ("fold_default-cold", "fold_default-in_task", "cold -> in_task, fold_default"),
        ("choice-cold", "fold_default-cold", "choice -> fold_default, cold"),
    )

    for model in ask.MODELS:
        for before, after, title in pairs:
            first = {r.pair_id: reply for r, reply in answered_rows(data, f"{model}-{before}")}
            second = {r.pair_id: reply for r, reply in answered_rows(data, f"{model}-{after}")}
            shared = sorted(set(first) & set(second))

            if not shared:
                continue

            lost = sum(1 for p in shared if first[p].folded and not second[p].folded)
            gained = sum(1 for p in shared if not first[p].folded and second[p].folded)
            continued = sum(1 for p in shared if second[p].action == "continue")
            lines.append(
                f"| {model} | {title} | {len(shared)} "
                f"| {pct(rate(sum(1 for p in shared if first[p].folded), len(shared)))} "
                f"| {pct(rate(sum(1 for p in shared if second[p].folded), len(shared)))} "
                f"| {lost + gained} | {lost} | {gained} | {pct(rate(continued, len(shared)))} |"
            )

    return "\n".join(lines)


def choice_text(data: AgentData) -> str:
    arms = arm_ids(data)
    picks = select_arms(data)
    lines = [
        "# Choice scoring",
        "",
        "No judge calls here. Every figure counts the agent's reply (fold, keep_both, continue) "
        "on the 132 band pairs, split by the two-judge relation key.",
        "",
        "## Selection criterion (fixed before any fold was judged)",
        "",
        CRITERION,
        "",
        "Selected: " + "; ".join(f"{model}: {', '.join(arms)}" for model, arms in picks.items()),
        "",
        "## Rates per arm",
        "",
        choice_overview(data, arms),
        "",
        "## Replies by relation key",
        "",
        choice_by_label(data, arms),
        "",
        "## True follow_up pairs against the experiment 4 Sonnet fold (a proxy)",
        "",
        "The proxy is the one_record answer the two judges keyed on the Sonnet server-side fold "
        "of the same pair in experiment 4. It says whether the two notes could be written as one "
        "record, not whether this agent's own fold would be one record, so it is a proxy and "
        "not a key for the agent's choice.",
        "",
        follow_up_proxy(data, arms),
        "",
        "## Shift between arms of one model",
        "",
        shift_table(data),
        "",
    ]

    return "\n".join(lines)


def trigger_label(trigger: fold_guard.GuardRule) -> str:
    return "follow_up band " + trigger.name if trigger is TRIGGERS[0] else trigger.name


def sweep_row(rule: AgentRule, total: int) -> str:
    score = score_agent(rule)
    stats, latencies = ask_stats(rule)
    base = score.base

    return (
        f"| {rule.arm} | {trigger_label(rule.trigger)} "
        f"| {'off' if rule.threshold is None else f'{rule.kind}>={rule.threshold:.2f}'} "
        f"| {stats.asked} ({stats.asked / total:.1%}) | {stats.replied_fold} | {stats.accepted} "
        f"| {ratio(resolved(score, 'strict'), base.restatements)} "
        f"({base.reinforced_restatement}+{base.superseded_restatement}"
        f"+{score.folded_good_restatement}) "
        f"| {ratio(caught(score, 'strict'), base.replacements)} "
        f"({base.flagged_replacement}+{base.superseded_replacement}"
        f"+{score.folded_good_replacement}) "
        f"| {ratio(resolved(score, 'loose'), base.restatements)} "
        f"| {ratio(caught(score, 'loose'), base.replacements)} "
        f"| {len(score.bad)} | {split_counts(score.bad_by_answer)} "
        f"| {split_counts(score.bad_by_label)} | {len(score.disputed)} "
        f"| {stats.rejected_good}/{stats.rejected_bad}/{stats.rejected_disputed} "
        f"| {score.follow_up_folds} ({score.follow_up_one_record_yes}/"
        f"{score.follow_up_one_record_no}) "
        f"| {len(score.distinct_folds)} | {base.wrong} | {base.lost} "
        f"| {ms(latencies)} / {ms(latencies, True)} "
        f"| {'YES' if meets(score, 'strict') else ''} | {'YES' if meets(score, 'loose') else ''} |"
    )


SWEEP_HEAD = (
    "| arm | trigger | guard | asked (share of 242) | replied fold | accepted "
    "| restatements strict (reinforced+superseded+folded good) "
    "| replacements strict (flagged+superseded+folded) | restatements loose | replacements loose "
    "| accepted bad | failing answer | relation key | accepted disputed "
    "| guard-rejected folds good/bad/disputed | accepted follow_up folds (one_record yes/no) "
    "| accepted distinct folds | wrong supersedes | lost | ask latency median / p90 ms "
    "| meets bars strict | meets bars loose |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def sweep_text(data: AgentData, arms: Sequence[str]) -> str:
    total = len(data.inputs.cases)
    lines = [
        "# Sweep",
        "",
        "Each row is one judged arm, one trigger and one guard setting, scored over all 242 "
        "pairs: containment rule first, then the trigger decides which saves get an ask, then "
        "the agent's reply, then the optional Jev coverage guard on a fold "
        "(fold_covers_old and fold_covers_new both at or above the threshold). A fold that fails "
        "the guard is saved as new. Strict counts a disputed fold as an error and not resolved; "
        "loose counts it as resolved unless a judge pair keyed a needed answer no. Only judged "
        "arms appear: an arm whose folds were not sent to the judges cannot be scored.",
        "",
        SWEEP_HEAD,
    ]

    for rule in rules_for(data, arms):
        lines.append(sweep_row(rule, total))

    return "\n".join(lines)


def wide_band_note(data: AgentData) -> str:
    rule = AgentRule("", TRIGGERS[0], "off", None, data)
    asked = sum(rule.asked(data.inputs.cases[row.pair_id]) for row in data.band)
    never = len(data.band) - asked

    return (
        f"The follow_up band trigger asks {asked} of the {len(data.band)} band pairs: the "
        f"{never} band pairs Jev picks as distinct are never asked."
    )


def interruption_table(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| arm | trigger | asks | share of 242 saves | fold | keep_both | continue | invalid "
        "| unanswered | share of asks that end in a fold |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    total = len(data.inputs.cases)

    for arm in arms:
        for trigger in TRIGGERS:
            stats, _ = ask_stats(AgentRule(arm, trigger, "off", None, data))
            lines.append(
                f"| {arm} | {trigger_label(trigger)} | {stats.asked} | {stats.asked / total:.1%} "
                f"| {stats.replied_fold} | {stats.keep_both} | {stats.continued} | {stats.invalid} "
                f"| {stats.unanswered} | {pct(rate(stats.replied_fold, stats.asked))} |"
            )

    return "\n".join(lines)


def server_score(
    data: AgentData, trigger: fold_guard.GuardRule, kind: str, threshold: float | None
) -> fold_guard.GuardScore:
    rule = fold_guard.GuardRule(
        trigger.floor, trigger.enhances, trigger.subject, None, kind, threshold
    )

    return fold_guard.score_guard(rule, data.guard, "sonnet")


def side_by_side(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| trigger | system | asks or server folds | accepted folds | restatements strict "
        "| replacements strict | restatements loose | replacements loose | accepted bad "
        "| accepted disputed | follow_up folds (one_record yes/no) | distinct folds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for trigger in TRIGGERS:
        for kind, threshold in GUARDS[:2]:
            theirs = server_score(data, trigger, kind, threshold)
            result = theirs.result
            label = "off" if threshold is None else f"{kind}>={threshold:.2f}"
            lines.append(
                f"| {trigger_label(trigger)} | server-side Sonnet fold, guard {label} "
                f"| {len(theirs.fired)} | {theirs.accepted} "
                f"| {ratio(theirs.resolved('strict'), result.base.restatements)} "
                f"| {ratio(theirs.caught('strict'), result.base.replacements)} "
                f"| {ratio(theirs.resolved('loose'), result.base.restatements)} "
                f"| {ratio(theirs.caught('loose'), result.base.replacements)} "
                f"| {len(result.bad)} | {len(result.disputed)} "
                f"| {result.follow_up_folds} ({result.follow_up_one_record_yes}/"
                f"{result.follow_up_one_record_no}) | {len(result.distinct_folds)} |"
            )

            for arm in arms:
                rule = AgentRule(arm, trigger, kind, threshold, data)
                score = score_agent(rule)
                stats, _ = ask_stats(rule)
                lines.append(
                    f"| {trigger_label(trigger)} | agent {arm}, guard {label} "
                    f"| {stats.asked} | {stats.accepted} "
                    f"| {ratio(resolved(score, 'strict'), score.base.restatements)} "
                    f"| {ratio(caught(score, 'strict'), score.base.replacements)} "
                    f"| {ratio(resolved(score, 'loose'), score.base.restatements)} "
                    f"| {ratio(caught(score, 'loose'), score.base.replacements)} "
                    f"| {len(score.bad)} | {len(score.disputed)} "
                    f"| {score.follow_up_folds} ({score.follow_up_one_record_yes}/"
                    f"{score.follow_up_one_record_no}) | {len(score.distinct_folds)} |"
                )

    return "\n".join(lines)


def server_bad_pairs(data: AgentData) -> tuple[list[str], list[str]]:
    result = server_score(data, WIDE_TRIGGER, "off", None).result

    return list(result.bad), list(result.disputed)


def outcome_cell(data: AgentData, pair_id: str, arm: str) -> str:
    reply = data.asks.get((pair_id, arm))

    if reply is None or not reply.answered:
        return "not asked"

    if not reply.folded:
        return str(reply.action)

    grade = data.grades.get((pair_id, arm))

    return "fold, not judged" if grade is None else f"fold, {grade.status}"


def bad_fold_actions(data: AgentData, arms: Sequence[str], pair_ids: Sequence[str]) -> str:
    server = data.inputs.grades
    lines = [
        "| pair | relation key | source | server-side Sonnet fold | " + " | ".join(arms) + " |",
        "|---|---|---|---|" + "---|" * len(arms),
    ]

    for pair_id in pair_ids:
        grade = server.get((pair_id, "sonnet"), fold.FoldGrade())
        lines.append(
            f"| {pair_id} | {data.inputs.relation.get(pair_id, 'disputed')} "
            f"| {data.inputs.pairs[pair_id].source} "
            f"| {grade.status}: {', '.join(grade.failing) or 'unkeyed answer'} | "
            + " | ".join(outcome_cell(data, pair_id, arm) for arm in arms)
            + " |"
        )

    return "\n".join(lines)


def glued_summary(data: AgentData, arms: Sequence[str], pair_ids: Sequence[str]) -> str:
    lines = [
        "| arm | of the server-side bad pairs: not folded | folded and bad | folded and good "
        "| folded and disputed | folded, not judged |",
        "|---|---|---|---|---|---|",
    ]

    for arm in arms:
        cells = Counter(
            "not folded"
            if (reply := data.asks.get((pid, arm))) is None or not reply.folded
            else "not judged"
            if (grade := data.grades.get((pid, arm))) is None
            else grade.status
            for pid in pair_ids
        )
        lines.append(
            f"| {arm} | {cells['not folded']} | {cells['bad']} | {cells['good']} "
            f"| {cells['disputed']} | {cells['not judged']} |"
        )

    return "\n".join(lines)


def lazy_resend_table(data: AgentData) -> str:
    lines = [
        "| guard threshold | band pairs | lazy resend rejected | restatement | replacement "
        "| follow_up | distinct | disputed |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for threshold in LAZY_THRESHOLDS:
        rejected: Counter[str] = Counter()

        for row in data.band:
            reading = data.inputs.cases[row.pair_id].p_yes("new_covers_old")

            if reading is None or reading < threshold:
                rejected[label_of(row)] += 1

        totals = Counter(label_of(row) for row in data.band)
        lines.append(
            f"| {threshold:.2f} | {len(data.band)} | {sum(rejected.values())} "
            + "".join(f"| {ratio(rejected[label], totals[label])} " for label in LABELS[:5])
            + "|"
        )

    return "\n".join(lines)


def min_coverage(data: AgentData, pair_id: str, arm: str) -> float | None:
    signal = data.signals.get((pair_id, arm))

    if signal is None:
        return None

    readings = [signal.reading(name).p_yes for name in fold_guard.GUARD_KINDS["coverage"]]
    present = [value for value in readings if value is not None]

    return None if len(present) < len(readings) else min(present)


def near_copy(note: str, incoming: str) -> bool:
    return difflib.SequenceMatcher(None, note, incoming).ratio() >= NEAR_COPY


def agent_guard_table(data: AgentData, arms: Sequence[str]) -> str:
    lines = [
        "| arm | threshold | folds with signals | good folds | good rejected | bad folds "
        "| bad rejected | disputed folds | disputed rejected | near-copies of the agent's note "
        "| near-copies rejected |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    band = {row.pair_id: row for row in data.band}

    for arm in arms:
        folds = [
            (row, reply)
            for row, reply in answered_rows(data, arm)
            if reply.folded and (row.pair_id, arm) in data.signals
        ]

        for threshold in LAZY_THRESHOLDS:
            cells: Counter[str] = Counter()

            for row, reply in folds:
                status = data.grades.get((row.pair_id, arm), fold.FoldGrade()).status
                reading = min_coverage(data, row.pair_id, arm)
                rejected = reading is None or reading < threshold
                cells[status] += 1
                cells[f"{status} rejected"] += rejected

                if near_copy(reply.note or "", band[row.pair_id].incoming):
                    cells["copy"] += 1
                    cells["copy rejected"] += rejected

            lines.append(
                f"| {arm} | {threshold:.2f} | {len(folds)} | {cells['good']} "
                f"| {cells['good rejected']} | {cells['bad']} | {cells['bad rejected']} "
                f"| {cells['disputed']} | {cells['disputed rejected']} | {cells['copy']} "
                f"| {cells['copy rejected']} |"
            )

    return "\n".join(lines)


def codex_grades(labels: Sequence[LabelRow]) -> dict[tuple[str, str], fold.FoldGrade]:
    grades: dict[tuple[str, str], fold.FoldGrade] = {}

    for row in labels:
        if row.judge != "codex":
            continue

        pair_id, arm, question = fold.split_fold_item(row.item_id)
        grades.setdefault((pair_id, arm), fold.FoldGrade()).answers[question] = row.label == "yes"

    return grades


def grade_table(data: AgentData, arms: Sequence[str]) -> str:
    codex = codex_grades(data.labels)
    lines = [
        "| arm | folds | two-judge good | two-judge bad | two-judge disputed "
        "| Codex-alone good | Codex-alone bad | Codex-alone incomplete |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for arm in arms:
        folded = [
            pid for row, reply in answered_rows(data, arm) if reply.folded for pid in [row.pair_id]
        ]
        two = Counter(data.grades.get((pid, arm), fold.FoldGrade()).status for pid in folded)
        one = Counter(codex.get((pid, arm), fold.FoldGrade()).status for pid in folded)
        lines.append(
            f"| {arm} | {len(folded)} | {two['good']} | {two['bad']} | {two['disputed']} "
            f"| {one['good']} | {one['bad']} | {one['disputed']} |"
        )

    return "\n".join(lines)


def agreement_blocks(data: AgentData) -> str:
    blocks: list[str] = []

    for title, model in (("all arms", None), *((f"{m} arms", m) for m in ask.MODELS)):
        subset = [
            row
            for row in data.labels
            if model is None or ask.arm_model(fold.split_fold_item(row.item_id)[1]) == model
        ]

        if subset:
            blocks += [f"### Judge agreement, {title}", "", fold.agreement_table(subset), ""]

    return "\n".join(blocks)


def headline_table(data: AgentData, arms: Sequence[str]) -> str:
    inputs = data.inputs
    empty: fold.Grades = {}
    today = fold.score_fold(BASELINE, inputs.cases, inputs.relation, inputs.cover, empty)
    contained = fold.score_fold(
        fold.CONTAINMENT, inputs.cases, inputs.relation, inputs.cover, empty
    )
    v1 = v1_counts(inputs.published_v1, inputs.relation)
    server = server_score(data, WIDE_TRIGGER, "off", None)
    lines = [
        "| system | restatements strict | replacements strict | restatements loose "
        "| replacements loose | accepted bad | accepted disputed | share of 242 saves asked "
        "| accepted folds | accepted bad, Codex alone |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| v2 today (Jev pick) | {ratio(today.resolved_restatement, today.base.restatements)} "
        f"| {ratio(today.caught_replacement, today.base.replacements)} "
        f"| {ratio(today.loose_restatement, today.base.restatements)} "
        f"| {ratio(today.loose_replacement, today.base.replacements)} | 0 | 0 | 0% | 0 | n/a |",
        f"| v2 plus containment rule "
        f"| {ratio(contained.resolved_restatement, contained.base.restatements)} "
        f"| {ratio(contained.caught_replacement, contained.base.replacements)} "
        f"| {ratio(contained.loose_restatement, contained.base.restatements)} "
        f"| {ratio(contained.loose_replacement, contained.base.replacements)} "
        f"| 0 | 0 | 0% | 0 | n/a |",
        f"| v1 as published (stops and asks the agent) "
        f"| {ratio(v1.resolved_restatement, v1.restatements)} "
        f"| {ratio(v1.flagged_replacement, v1.replacements)} "
        f"| {ratio(v1.resolved_restatement, v1.restatements)} "
        f"| {ratio(v1.flagged_replacement, v1.replacements)} | n/a | n/a | n/a | n/a | n/a |",
        f"| server-side Sonnet fold, {WIDE_TRIGGER.name} "
        f"| {ratio(server.resolved('strict'), server.result.base.restatements)} "
        f"| {ratio(server.caught('strict'), server.result.base.replacements)} "
        f"| {ratio(server.resolved('loose'), server.result.base.restatements)} "
        f"| {ratio(server.caught('loose'), server.result.base.replacements)} "
        f"| {len(server.result.bad)} | {len(server.result.disputed)} "
        f"| {server.accepted / len(inputs.cases):.1%} | {server.accepted} | n/a |",
    ]
    codex = codex_grades(data.labels)

    for model in ask.MODELS:
        own = [arm for arm in arms if ask.arm_model(arm) == model]

        if not own:
            continue

        rules = rules_for(data, own)

        for reading in READINGS:
            chosen = best_rule(rules, reading)
            score = score_agent(chosen)
            stats, _ = ask_stats(chosen)
            alone = score_agent(
                chosen,
                grades={pid: grade for (pid, arm), grade in codex.items() if arm == chosen.arm},
            )
            lines.append(
                f"| best agent rule, {model}, {reading} ranking: {chosen.name} "
                f"| {ratio(resolved(score, 'strict'), score.base.restatements)} "
                f"| {ratio(caught(score, 'strict'), score.base.replacements)} "
                f"| {ratio(resolved(score, 'loose'), score.base.restatements)} "
                f"| {ratio(caught(score, 'loose'), score.base.replacements)} "
                f"| {len(score.bad)} | {len(score.disputed)} "
                f"| {stats.asked / len(inputs.cases):.1%} | {stats.accepted} | {len(alone.bad)} |"
            )

    return "\n".join(lines)


def verdict_text(data: AgentData, arms: Sequence[str]) -> str:
    sentences: list[str] = []

    for model in ask.MODELS:
        own = [arm for arm in arms if ask.arm_model(arm) == model]

        if not own:
            sentences.append(f"{model}: no judged arm.")

            continue

        rules = rules_for(data, own)

        for reading in READINGS:
            winners = [rule.name for rule in rules if meets(score_agent(rule), reading)]
            chosen = best_rule(rules, reading)
            score = score_agent(chosen)
            misses = miss_list(score, reading)
            sentences.append(
                f"{model}, {reading}: {len(winners)} arm/trigger/guard combination(s) reach "
                f"{RESTATEMENT_BAR}+ restatements and {REPLACEMENT_BAR} replacements with zero "
                f"accepted bad folds{disputed_clause(reading)}"
                f", zero wrong supersedes and zero lost records. Closest `{chosen.name}` resolves "
                f"{resolved(score, reading)} restatements and catches {caught(score, reading)} "
                f"replacements; it misses on {misses}."
            )

    return " ".join(sentences)


def disputed_clause(reading: Reading) -> str:
    return " and zero accepted disputed folds" if reading == "strict" else ""


def miss_list(score: fold.FoldScore, reading: Reading) -> str:
    misses: list[str] = []
    gap_rs = RESTATEMENT_BAR - resolved(score, reading)
    gap_rp = REPLACEMENT_BAR - caught(score, reading)

    if gap_rs > 0:
        misses.append(f"restatements by {gap_rs}")

    if gap_rp > 0:
        misses.append(f"replacements by {gap_rp}")

    if score.bad:
        misses.append(f"{len(score.bad)} accepted bad fold(s)")

    if reading == "strict" and score.disputed:
        misses.append(f"{len(score.disputed)} accepted disputed fold(s)")

    if score.base.wrong:
        misses.append(f"{score.base.wrong} wrong supersede(s)")

    if score.base.lost:
        misses.append(f"{score.base.lost} lost record(s) or update(s)")

    return ", ".join(misses) if misses else "nothing"


def tie_row(held: str, rule: AgentRule, test: frozenset[str], reading: Reading) -> str:
    score = score_agent(rule, test)

    return (
        f"| {held} (tied on the other sources, not pooled) | {rule.name} "
        f"| {ratio(resolved(score, reading), score.base.restatements)} "
        f"| {ratio(caught(score, reading), score.base.replacements)} | {len(score.bad)} "
        f"| {len(score.disputed)} | {score.base.wrong} | {score.base.lost} |"
    )


def loso_table(
    rules: Sequence[AgentRule], data: AgentData, reading: Reading
) -> tuple[str, dict[str, int]]:
    inputs = data.inputs
    sources = {pid: inputs.pairs[pid].source for pid in inputs.cases}
    everyone = frozenset(sources)
    lines = [
        "| held-out source | rule chosen on the other sources | restatements | replacements "
        "| accepted bad | accepted disputed | wrong supersedes | lost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    pooled = {"restatements": 0, "replacements": 0, "bad": 0, "disputed": 0, "wrong": 0, "lost": 0}
    totals = {"restatements": 0, "replacements": 0}

    for held in sorted(set(sources.values())):
        test = frozenset(pid for pid, source in sources.items() if source == held)
        train = everyone - test
        chosen = best_rule(rules, reading, train)
        score = score_agent(chosen, test)
        best_rank = rank(score_agent(chosen, train), reading)
        ties = [
            rule
            for rule in rules
            if rule is not chosen and rank(score_agent(rule, train), reading) == best_rank
        ]
        pooled["restatements"] += resolved(score, reading)
        pooled["replacements"] += caught(score, reading)
        pooled["bad"] += len(score.bad)
        pooled["disputed"] += len(score.disputed)
        pooled["wrong"] += score.base.wrong
        pooled["lost"] += score.base.lost
        totals["restatements"] += score.base.restatements
        totals["replacements"] += score.base.replacements
        lines.append(
            f"| {held} | {chosen.name} "
            f"| {ratio(resolved(score, reading), score.base.restatements)} "
            f"| {ratio(caught(score, reading), score.base.replacements)} | {len(score.bad)} "
            f"| {len(score.disputed)} | {score.base.wrong} | {score.base.lost} |"
        )
        lines += [tie_row(held, rule, test, reading) for rule in ties]

    lines.append(
        f"| pooled held-out | | {ratio(pooled['restatements'], totals['restatements'])} "
        f"| {ratio(pooled['replacements'], totals['replacements'])} | {pooled['bad']} "
        f"| {pooled['disputed']} | {pooled['wrong']} | {pooled['lost']} |"
    )

    return "\n".join(lines), pooled


def codex_rescore(data: AgentData, arms: Sequence[str]) -> str:
    codex = codex_grades(data.labels)
    lines = [
        "| model | reading | rule (chosen on the two-judge key) | two-judge restatements "
        "| two-judge replacements | two-judge accepted bad | Codex-alone restatements "
        "| Codex-alone replacements | Codex-alone accepted bad |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for model in ask.MODELS:
        own = [arm for arm in arms if ask.arm_model(arm) == model]

        if not own:
            continue

        for reading in READINGS:
            chosen = best_rule(rules_for(data, own), reading)
            two = score_agent(chosen)
            alone = score_agent(
                chosen,
                grades={pid: grade for (pid, arm), grade in codex.items() if arm == chosen.arm},
            )
            lines.append(
                f"| {model} | {reading} | {chosen.name} "
                f"| {ratio(resolved(two, reading), two.base.restatements)} "
                f"| {ratio(caught(two, reading), two.base.replacements)} | {len(two.bad)} "
                f"| {ratio(resolved(alone, reading), alone.base.restatements)} "
                f"| {ratio(caught(alone, reading), alone.base.replacements)} | {len(alone.bad)} |"
            )

    return "\n".join(lines)


def figures(rule: AgentRule) -> str:
    score = score_agent(rule)
    stats, _ = ask_stats(rule)
    trigger = rule.trigger

    return (
        f"arm={rule.arm} floor={trigger.floor} enhances={trigger.enhances} "
        f"same_subject={trigger.subject} guard={rule.kind}>={rule.threshold}: "
        f"restatements={score.resolved_restatement} replacements={score.caught_replacement} "
        f"loose_restatements={score.loose_restatement} "
        f"loose_replacements={score.loose_replacement} asked={stats.asked} "
        f"replied_fold={stats.replied_fold} accepted={stats.accepted} bad={len(score.bad)} "
        f"disputed={len(score.disputed)} wrong_supersedes={score.base.wrong} lost={score.base.lost}"
    )


def top_rules(data: AgentData, arms: Sequence[str]) -> list[AgentRule]:
    found: dict[str, AgentRule] = {}

    for model in ask.MODELS:
        own = [arm for arm in arms if ask.arm_model(arm) == model]

        if own:
            for reading in READINGS:
                chosen = best_rule(rules_for(data, own), reading)
                found[chosen.name] = chosen

    return list(found.values())


def hand_check_text(data: AgentData, base: Path, arms: Sequence[str]) -> str:
    extra = [AgentRule(arm, TRIGGERS[0], "off", None, data) for arm in arms[:1]] + [
        AgentRule(arm, WIDE_TRIGGER, "coverage", 0.70, data) for arm in arms[-1:]
    ]
    lines = [
        "Recount from raw rows with scripts/eval_v1v2/agent_fold_recount.py (reads signals, "
        "covers, key-relation, key-covers, the fold-guard same_subject rows, asks.jsonl, "
        "key-fold.jsonl and guard-signals.jsonl directly; imports nothing from the scorer). Each "
        "pair of lines is the recount, then the scorer's figures for the same rule. A line "
        "starting MATCH means every figure agrees.",
        "",
        "```",
    ]

    for rule in [*top_rules(data, arms), *extra]:
        trigger = rule.trigger
        counted = recount.recount(
            base,
            data.out_dir,
            rule.arm,
            trigger.floor,
            trigger.enhances,
            trigger.subject,
            rule.kind,
            rule.threshold,
        )
        scored = figures(rule)
        mark = "MATCH" if counted == scored else "DIFFERENT"
        lines += [f"{mark} {counted}", f"       scorer: {scored.split(': ', 1)[1]}"]

    lines.append("```")

    return "\n".join(lines)


def bad_folds_text(data: AgentData, arms: Sequence[str]) -> str:
    inputs = data.inputs
    lines = [
        "# Bad folds",
        "",
        "Holds record text; stays under dev-corpus. A fold is bad when either judge pair keyed "
        "one of the four answers no. Every bad fold in every judged arm is listed over the whole "
        "candidate band, whether or not a rule would accept it. 'Accepted by' names the best "
        "agent rule of that model (strict ranking, then loose ranking) that accepts it.",
        "",
    ]
    best = {
        (model, reading): best_rule(
            rules_for(data, [a for a in arms if ask.arm_model(a) == model]), reading
        )
        for model in ask.MODELS
        if any(ask.arm_model(a) == model for a in arms)
        for reading in READINGS
    }
    entries: list[str] = []

    for arm in arms:
        for row in data.band:
            grade = data.grades.get((row.pair_id, arm))
            reply = data.asks.get((row.pair_id, arm))

            if grade is None or reply is None or grade.status != "bad":
                continue

            accepted = [
                f"{reading} rule `{rule.name}`"
                for (model, reading), rule in best.items()
                if model == ask.arm_model(arm)
                and rule.arm == arm
                and rule.accepts(inputs.cases[row.pair_id])
            ]
            entries += [
                f"### {row.pair_id} ({arm}, relation key: {label_of(row)}, "
                f"source: {inputs.pairs[row.pair_id].source})",
                "",
                f"Failing: {', '.join(grade.failing)}. Accepted by: "
                f"{'; '.join(accepted) or 'no best rule'}. Answers: {fold.answers_text(grade)}.",
                "",
                f"- existing: {' '.join(row.existing.split())}",
                f"- incoming: {' '.join(row.incoming.split())}",
                f"- fold: {' '.join((reply.note or '').split())}",
                "",
            ]

    lines += [f"{sum(1 for e in entries if e.startswith('### '))} bad folds.", "", *entries]

    return "\n".join(lines)


def call_counts(data: AgentData, arms: Sequence[str]) -> str:
    lines = ["| arm | calls landed | pending | invalid replies |", "|---|---|---|---|"]

    for arm in arms:
        rows = answered_rows(data, arm)
        invalid = sum(1 for _, reply in rows if reply.action == "invalid")
        lines.append(f"| {arm} | {len(rows)} | {len(data.band) - len(rows)} | {invalid} |")

    return "\n".join(lines)


def unjudged_count(data: AgentData, arms: Sequence[str]) -> int:
    seen: dict[str, set[str]] = {}

    for row in data.labels:
        seen.setdefault(row.item_id, set()).add(row.judge)

    return sum(1 for item in fold_items(data, arms) if len(seen.get(item.item_id, set())) < 2)


def summary_text(data: AgentData) -> str:
    asked_arms = arm_ids(data)
    arms = judged_arms(data)
    picks = select_arms(data)
    best = {
        (model, reading): best_rule(
            rules_for(data, [a for a in arms if ask.arm_model(a) == model]), reading
        )
        for model in ask.MODELS
        if any(ask.arm_model(a) == model for a in arms)
        for reading in READINGS
    }
    bad_pairs, disputed_pairs = server_bad_pairs(data)
    unjudged = unjudged_count(data, arms)
    lines = ["# Experiment 6: agent fold", ""]

    if unjudged:
        lines += [f"WARNING: {unjudged} fold answers lack a label from one or both judges.", ""]

    lines += [
        verdict_text(data, arms),
        "",
        "## Arms and calls",
        "",
        CRITERION,
        "",
        "Judged arms: "
        + (", ".join(arms) or "none yet")
        + ". Selected on the criterion: "
        + "; ".join(f"{m}: {', '.join(a)}" for m, a in picks.items())
        + ".",
        "",
        call_counts(data, asked_arms),
        "",
        "The prompts are in ask-prompts/. The choice-in_task combination was not run.",
        "",
        "## Headline",
        "",
        "Server-side Sonnet fold is the experiment 5 trigger `sim>=0.80 same_subject>=0.70` with "
        "no guard. v1 only stops and asks; its figures are copied from the published run.",
        "",
        UPPER_BOUND,
        "",
        CREDIT_RULE,
        "",
        headline_table(data, arms),
        "",
        JUDGE_SEAT,
        "",
        "## Choice rates and easy exit",
        "",
        choice_overview(data, asked_arms),
        "",
        "Full split by relation key, the follow_up proxy and the cold against in_task shift are "
        "in choice.md.",
        "",
        "## Interruption cost per arm and trigger",
        "",
        "An ask is one interruption of the saving agent. The share is over all 242 saves. "
        + wide_band_note(data),
        "",
        interruption_table(data, asked_arms),
        "",
        "## Against the server-side Sonnet fold on the same trigger",
        "",
        side_by_side(data, arms),
        "",
        f"### The {len(bad_pairs)} server-side bad folds on `{WIDE_TRIGGER.name}`",
        "",
        "The wide trigger here is the experiment 5 best Sonnet rule. These are the folds it "
        "accepted that the judges keyed bad, then the pairs it accepted that stayed disputed, "
        "and what each agent arm did with them.",
        "",
        bad_fold_actions(data, asked_arms, bad_pairs),
        "",
        glued_summary(data, asked_arms, bad_pairs),
        "",
        f"Disputed server-side folds on the same trigger ({len(disputed_pairs)}):",
        "",
        bad_fold_actions(data, asked_arms, disputed_pairs),
        "",
        "## Lazy-return guard",
        "",
        "A lazy agent resends its own note as the fold. The first table is a computed proxy: it "
        "assumes the resent note's fold_covers_old equals the pair's new_covers_old signal and "
        "that its fold_covers_new, the note against itself, passes. It counts the band pairs "
        "that resend would fail at each threshold. The second table is what the same guard "
        "(both coverage readings at or above the threshold) does to the real folds of the "
        "judged arms, and how many of those folds are near-copies of the agent's own note "
        "(character match of 0.95 or more). No real near-copy of the agent's own note was "
        "observed in any arm.",
        "",
        lazy_resend_table(data),
        "",
        agent_guard_table(data, arms),
        "",
        "## Judges",
        "",
        "The claude judge seat is Opus, so Opus-written folds are judged partly by the same model "
        "family. The table gives the two-judge grade next to a Codex-alone grade.",
        "",
        agreement_blocks(data),
        grade_table(data, arms),
        "",
        "Best rule per model under each ranking, rescored with Codex-alone grades:",
        "",
        codex_rescore(data, arms),
        "",
        "## Leave one source out",
        "",
    ]

    for (model, reading), rule in best.items():
        own = [a for a in arms if ask.arm_model(a) == model]
        table, _ = loso_table(rules_for(data, own), data, reading)
        lines += [
            f"### {model}, {reading} ranking (full-data best rule: `{rule.name}`)",
            "",
            table,
            "",
        ]

    hand_check = data.out_dir / "hand-check.md"

    if hand_check.exists():
        lines += ["## Hand check", "", hand_check.read_text(encoding="utf-8").strip(), ""]

    lines += [
        "## Limits of the simulation",
        "",
        LIMITS,
        "",
        "## Files",
        "",
        "- ask-prompts/, asks.jsonl, labels-fold.jsonl, key-fold.jsonl, guard-signals.jsonl, "
        "choice.md, sweep.md, bad-folds.md, summary.md, hand-check.md",
        "",
    ]

    return "\n".join(lines)


def run_judge(out_dir: Path, base: Path) -> int:
    data = load_data(base, out_dir)
    arms = judged_arms(data)
    items = fold_items(data, arms)
    outcome = run_task("fold", items, out_dir=out_dir)
    seen: dict[str, set[str]] = {}

    for row in outcome.labels:
        seen.setdefault(row.item_id, set()).add(row.judge)

    missing = sum(1 for item in items if len(seen.get(item.item_id, set())) < 2)
    print(
        json.dumps(
            {
                "arms": arms,
                "items": len(items),
                "keyed": len(outcome.key),
                "missing_a_judge": missing,
                "unlabelled_this_run": outcome.unlabelled,
            }
        ),
        flush=True,
    )

    return 0 if missing == 0 else 2


def run_guard_rows(out_dir: Path, base: Path) -> int:
    data = load_data(base, out_dir)
    rows = guard_target_rows(data, judged_arms(data))
    path = out_dir / "judged-folds.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} fold rows to {path.name}", flush=True)

    return 0


def guard_missing(data: AgentData, arms: Sequence[str]) -> int:
    return sum(
        1
        for arm in arms
        for row, reply in answered_rows(data, arm)
        if reply.folded
        and ((row.pair_id, arm) not in data.signals or data.signals[(row.pair_id, arm)].error)
    )


def run_score(out_dir: Path, base: Path) -> int:
    data = load_data(base, out_dir)
    arms = judged_arms(data)
    missing = guard_missing(data, arms)

    if missing:
        print(f"{missing} judged folds lack guard signals; run run-guard-signals.sh", flush=True)

        return 2

    out_dir.joinpath("choice.md").write_text(choice_text(data) + "\n", encoding="utf-8")
    out_dir.joinpath("sweep.md").write_text(sweep_text(data, arms) + "\n", encoding="utf-8")
    out_dir.joinpath("bad-folds.md").write_text(bad_folds_text(data, arms) + "\n", encoding="utf-8")
    out_dir.joinpath("hand-check.md").write_text(
        hand_check_text(data, base, arms) + "\n", encoding="utf-8"
    )
    out_dir.joinpath("summary.md").write_text(summary_text(data) + "\n", encoding="utf-8")
    print("scored; wrote choice.md, sweep.md, bad-folds.md, hand-check.md, summary.md", flush=True)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent fold experiment: judge, guard rows, score.")
    parser.add_argument("phase", choices=("judge", "guard-rows", "score"))
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "agent-fold"
    )
    args = parser.parse_args()
    actions: dict[str, Callable[[Path, Path], int]] = {
        "judge": run_judge,
        "guard-rows": run_guard_rows,
        "score": run_score,
    }
    code = actions[args.phase](args.out_dir, args.base)

    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
