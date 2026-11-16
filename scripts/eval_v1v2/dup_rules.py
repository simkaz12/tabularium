from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from eval_v1v2 import paths
from eval_v1v2.rows import Bucket, PairRow, Row, SaveRow, read_jsonl
from eval_v1v2.score_save import load_key, median, percentile

Decision = Literal["merge", "flag_replace", "new"]
MergeKind = Literal["p_restatement", "no_new_info", "both", "either"]

MERGE_KINDS: tuple[MergeKind, ...] = ("p_restatement", "no_new_info", "both", "either")
THRESHOLDS: tuple[float, ...] = (0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15)
NO_NEW_INFO_FLOORS: tuple[float | None, ...] = (None, 0.9, 0.8, 0.7, 0.6)
SIMILARITY_FLOORS: tuple[float | None, ...] = (0.90, 0.85, 0.80, None)
V1_MERGE_BAR = 34
V1_FLAG_BAR = 41
TOP = 12
FRONTIER_ROWS = 6
DECISIONS: tuple[Decision, ...] = ("merge", "flag_replace", "new")
UNRELATED = frozenset({"follow_up", "distinct"})


class NeighbourSignal(Row):
    id: str
    pick: str | None
    confidence: float | None
    probabilities: dict[str, float] | None
    reason_class: str | None
    similarity: float


class AddsInformation(Row):
    answer: Literal["yes", "no"] | None
    probability: float | None
    probabilities: dict[str, float] | None
    elapsed_ms: float
    error: str | None


class SignalRow(Row):
    pair_id: str
    bucket: Bucket
    raw_outcome: str
    found_existing: bool
    relation_pick: str | None
    degraded: str | None
    existing: NeighbourSignal | None
    top_other: NeighbourSignal | None
    adds_information: AddsInformation
    propose_ms: float


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    decide: Callable[[SignalRow], Decision]


class Score(BaseModel):
    restatements: int = 0
    replacements: int = 0
    unrelated: int = 0
    merged_restatement: int = 0
    lost_record: int = 0
    lost_update: int = 0
    flagged_replacement: int = 0
    false_flag: int = 0
    disputed: dict[str, int] = {}

    @property
    def lost(self) -> int:
        return self.lost_record + self.lost_update

    @property
    def beats_v1(self) -> bool:
        return (
            self.merged_restatement > V1_MERGE_BAR
            and self.flagged_replacement >= V1_FLAG_BAR
            and self.lost_record == 0
            and self.lost_update == 0
        )


def baseline_decision(signal: SignalRow) -> Decision:
    existing = signal.existing

    if existing is None:
        return "new"

    if existing.pick == "restatement":
        return "merge"

    if existing.pick == "replacement":
        return "flag_replace"

    return "new"


def probability_of(signal: SignalRow, relation: str) -> float:
    existing = signal.existing

    if existing is None or existing.probabilities is None:
        return 0.0

    return existing.probabilities.get(relation, 0.0)


def adds_nothing(signal: SignalRow, floor: float | None) -> bool:
    existing = signal.existing
    answer = signal.adds_information

    if existing is None or existing.pick != "follow_up" or answer.answer != "no":
        return False

    return answer.probability is not None and (floor is None or answer.probability >= floor)


def clears_similarity(signal: SignalRow, floor: float | None) -> bool:
    existing = signal.existing

    return existing is not None and (floor is None or existing.similarity >= floor)


def merge_condition(
    kind: MergeKind, signal: SignalRow, threshold: float | None, no_new_info_floor: float | None
) -> bool:
    by_probability = threshold is not None and probability_of(signal, "restatement") >= threshold
    by_question = adds_nothing(signal, no_new_info_floor)

    if kind == "p_restatement":
        return by_probability

    if kind == "no_new_info":
        return by_question

    if kind == "both":
        return by_probability and by_question

    return by_probability or by_question


def _suffix(similarity_floor: float | None) -> str:
    return "" if similarity_floor is None else f" @sim>={similarity_floor:.2f}"


def merge_rule(
    kind: MergeKind,
    threshold: float | None,
    no_new_info_floor: float | None,
    similarity_floor: float | None,
) -> Rule:
    parts = []

    if threshold is not None:
        parts.append(f"p_restatement>={threshold}")

    if kind != "p_restatement":
        floor = "" if no_new_info_floor is None else f"(p>={no_new_info_floor})"
        parts.append(f"no_new_info{floor}")

    joiner = {"p_restatement": "", "no_new_info": "", "both": " AND ", "either": " OR "}[kind]
    name = f"{kind}: {joiner.join(parts)}{_suffix(similarity_floor)}"

    def decide(signal: SignalRow) -> Decision:
        if clears_similarity(signal, similarity_floor) and merge_condition(
            kind, signal, threshold, no_new_info_floor
        ):
            return "merge"

        return baseline_decision(signal)

    return Rule(name=name, family=kind, decide=decide)


def replacement_rule(threshold: float, similarity_floor: float | None) -> Rule:
    def decide(signal: SignalRow) -> Decision:
        existing = signal.existing
        flag = (
            existing is not None
            and existing.pick == "follow_up"
            and clears_similarity(signal, similarity_floor)
            and probability_of(signal, "replacement") >= threshold
        )

        return "flag_replace" if flag else baseline_decision(signal)

    return Rule(
        name=f"p_replacement>={threshold}{_suffix(similarity_floor)}",
        family="p_replacement",
        decide=decide,
    )


def compose(merge: Rule, flag: Rule) -> Rule:
    def decide(signal: SignalRow) -> Decision:
        merged = merge.decide(signal)

        return flag.decide(signal) if merged == baseline_decision(signal) else merged

    return Rule(name=f"{merge.name} + {flag.name}", family=merge.family, decide=decide)


BASELINE = Rule(name="baseline", family="baseline", decide=baseline_decision)


def merge_rules_at(similarity_floor: float | None) -> Iterator[Rule]:
    for kind in MERGE_KINDS:
        floors = (None,) if kind == "p_restatement" else NO_NEW_INFO_FLOORS
        thresholds: tuple[float | None, ...] = (None,) if kind == "no_new_info" else THRESHOLDS

        for threshold in thresholds:
            for floor in floors:
                yield merge_rule(kind, threshold, floor, similarity_floor)


def merge_rules() -> Iterator[Rule]:
    for similarity_floor in SIMILARITY_FLOORS:
        yield from merge_rules_at(similarity_floor)


def replacement_rules() -> Iterator[Rule]:
    for similarity_floor in SIMILARITY_FLOORS:
        for threshold in THRESHOLDS:
            yield replacement_rule(threshold, similarity_floor)


def composite_rules() -> Iterator[Rule]:
    for similarity_floor in SIMILARITY_FLOORS:
        for merge in merge_rules_at(similarity_floor):
            for threshold in THRESHOLDS:
                yield compose(merge, replacement_rule(threshold, similarity_floor))


def score_rule(
    rule: Rule,
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    only: frozenset[str] | None = None,
) -> Score:
    score = Score()
    disputed: Counter[str] = Counter()

    for pair_id, signal in signals.items():
        if only is not None and pair_id not in only:
            continue

        decision = rule.decide(signal)
        label = key.get(pair_id)

        if label is None:
            disputed[decision] += 1
            continue

        _tally(score, label, decision)

    score.disputed = {name: disputed.get(name, 0) for name in DECISIONS}

    return score


def _tally(score: Score, label: str, decision: Decision) -> None:
    if label == "restatement":
        score.restatements += 1
        score.merged_restatement += decision == "merge"
    elif label == "replacement":
        score.replacements += 1
        score.lost_update += decision == "merge"
        score.flagged_replacement += decision == "flag_replace"
    elif label in UNRELATED:
        score.unrelated += 1
        score.lost_record += decision == "merge"
        score.false_flag += decision == "flag_replace"


def rank_key(score: Score) -> tuple[int, int, int]:
    return (score.lost, -(score.merged_restatement + score.flagged_replacement), score.false_flag)


def ranked(rules: Sequence[Rule], scores: Mapping[str, Score]) -> list[Rule]:
    return sorted(rules, key=lambda rule: rank_key(scores[rule.name]))


def select_for(
    rules: Sequence[Rule],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    train: frozenset[str],
) -> Rule:
    return min(rules, key=lambda rule: rank_key(score_rule(rule, signals, key, train)))


def stability(
    signals: Sequence[SignalRow], published: Sequence[SaveRow]
) -> tuple[int, Counter[tuple[str, str]]]:
    before = {row.pair_id: row.relation_pick or "none" for row in published}
    moves: Counter[tuple[str, str]] = Counter()

    for signal in signals:
        old = before.get(signal.pair_id, "missing")
        new = signal.relation_pick or "none"

        if old != new:
            moves[(old, new)] += 1

    return sum(moves.values()), moves


def _cell(count: int, total: int) -> str:
    return f"{count}/{total}"


def table_row(rule: Rule, score: Score) -> str:
    beats = "YES" if score.beats_v1 else ""
    disputed = score.disputed

    return (
        f"| {rule.name} | {_cell(score.merged_restatement, score.restatements)} "
        f"| {_cell(score.lost_record, score.unrelated)} "
        f"| {_cell(score.lost_update, score.replacements)} "
        f"| {_cell(score.flagged_replacement, score.replacements)} | {score.false_flag} "
        f"| {disputed['merge']}/{disputed['flag_replace']}/{disputed['new']} | {beats} |"
    )


TABLE_HEAD = (
    "| rule | restatements merged | lost records | lost updates | replacements flagged "
    "| false flags "
    "| disputed merge/flag/new | beats v1 |\n|---|---|---|---|---|---|---|---|"
)


def render_table(
    rules: Sequence[Rule], scores: Mapping[str, Score], twins: Mapping[str, int] | None = None
) -> str:
    if twins is None:
        return "\n".join([TABLE_HEAD, *(table_row(rule, scores[rule.name]) for rule in rules)])

    head = TABLE_HEAD.replace("| beats v1 |", "| beats v1 | rules with this behaviour |").replace(
        "|---|---|---|---|---|---|---|---|", "|---|---|---|---|---|---|---|---|---|"
    )

    return "\n".join(
        [head, *(f"{table_row(rule, scores[rule.name])} {twins[rule.name]} |" for rule in rules)]
    )


def source_table(
    rules: Sequence[Rule],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    sources: Mapping[str, str],
) -> str:
    names = sorted(set(sources.values()))
    lines = [
        "| rule | source | restatements merged | baseline merged | lost records "
        "| lost updates | replacements flagged |",
        "|---|---|---|---|---|---|---|",
    ]

    for rule in rules:
        gaining = []

        for name in names:
            only = frozenset(pid for pid, source in sources.items() if source == name)
            score = score_rule(rule, signals, key, only)
            base = score_rule(BASELINE, signals, key, only)

            if score.merged_restatement > base.merged_restatement:
                gaining.append(name)

            lines.append(
                f"| {rule.name} | {name} | {_cell(score.merged_restatement, score.restatements)} "
                f"| {base.merged_restatement} | {score.lost_record} | {score.lost_update} "
                f"| {_cell(score.flagged_replacement, score.replacements)} |"
            )

        verdict = "gains on " + ", ".join(gaining) if gaining else "gains on no source"
        scope = "one source only" if len(gaining) == 1 else f"{len(gaining)} sources"
        lines.append(f"| {rule.name} | note | {verdict} ({scope}) | | | | |")

    return "\n".join(lines)


def leave_one_source_out(
    candidates: Mapping[str, Sequence[Rule]],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    sources: Mapping[str, str],
) -> str:
    names = sorted(set(sources.values()))
    lines = [
        "| family | held-out source | rule chosen on the other three "
        "| restatements merged | lost records "
        "| lost updates | replacements flagged |",
        "|---|---|---|---|---|---|---|",
    ]

    for family, rules in candidates.items():
        pooled = Score()

        for held in names:
            test = frozenset(pid for pid, source in sources.items() if source == held)
            train = frozenset(sources) - test
            chosen = select_for(rules, signals, key, train)
            score = score_rule(chosen, signals, key, test)

            _pool(pooled, score)
            lines.append(
                f"| {family} | {held} | {chosen.name} "
                f"| {_cell(score.merged_restatement, score.restatements)} "
                f"| {score.lost_record} | {score.lost_update} "
                f"| {_cell(score.flagged_replacement, score.replacements)} |"
            )

        lines.append(
            f"| {family} | pooled held-out | "
            f"| {_cell(pooled.merged_restatement, pooled.restatements)} "
            f"| {pooled.lost_record} | {pooled.lost_update} "
            f"| {_cell(pooled.flagged_replacement, pooled.replacements)} |"
        )

    return "\n".join(lines)


def _pool(total: Score, part: Score) -> None:
    total.restatements += part.restatements
    total.replacements += part.replacements
    total.unrelated += part.unrelated
    total.merged_restatement += part.merged_restatement
    total.lost_record += part.lost_record
    total.lost_update += part.lost_update
    total.flagged_replacement += part.flagged_replacement
    total.false_flag += part.false_flag


def _flat(text: str, limit: int = 200) -> str:
    return " ".join(text.split())[:limit].replace("|", "/")


def _probabilities(signal: SignalRow) -> str:
    existing = signal.existing

    if existing is None or existing.probabilities is None:
        return "none"

    short = {"restatement": "rs", "replacement": "rp", "follow_up": "fu", "distinct": "di"}

    return " ".join(f"{short[name]}={existing.probabilities.get(name, 0.0):.2f}" for name in short)


def hard_case_row(pair: PairRow, signal: SignalRow, label: str) -> str:
    existing = signal.existing
    answer = signal.adds_information
    adds = (
        "none"
        if answer.answer is None or answer.probability is None
        else f"{answer.answer} {answer.probability:.2f}"
    )

    return (
        f"| {pair.pair_id} | {pair.source} | {label} | "
        f"{'none' if existing is None else f'{existing.similarity:.3f}'} "
        f"| {'none' if existing is None else existing.pick} | {_probabilities(signal)} | {adds} "
        f"| {_flat(pair.existing)} | {_flat(pair.incoming)} |"
    )


HARD_HEAD = (
    "| pair_id | source | key | similarity | pick | probabilities | adds information "
    "| existing | incoming |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def hard_cases(
    pairs: Mapping[str, PairRow],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    every_rule: Sequence[Rule],
    best: Sequence[Rule],
) -> str:
    unmerged = [
        pid
        for pid, label in key.items()
        if label == "restatement"
        and not any(rule.decide(signals[pid]) == "merge" for rule in every_rule)
    ]
    lines = [
        "# Hard cases (holds record text; stays under dev-corpus)",
        "",
        f"## True restatements that no rule in the sweep merges ({len(unmerged)})",
        "",
        HARD_HEAD,
        *(hard_case_row(pairs[pid], signals[pid], key[pid]) for pid in unmerged),
        "",
    ]

    for rule in best:
        lost = [
            pid
            for pid, label in key.items()
            if label in UNRELATED and rule.decide(signals[pid]) == "merge"
        ]
        lines += [
            f"## True follow-ups or distinct pairs merged by `{rule.name}` ({len(lost)})",
            "",
            HARD_HEAD,
            *(hard_case_row(pairs[pid], signals[pid], key[pid]) for pid in lost),
            "",
        ]

    return "\n".join(lines)


@dataclass(frozen=True)
class Sweep:
    merges: list[Rule]
    flags: list[Rule]
    composites: list[Rule]
    scores: dict[str, Score]
    twins: dict[str, int]
    shapes: dict[str, tuple[Decision, ...]]

    def distinct(self, rules: Sequence[Rule]) -> list[Rule]:
        seen: set[tuple[Decision, ...]] = set()
        kept = []

        for rule in rules:
            if self.shapes[rule.name] not in seen:
                seen.add(self.shapes[rule.name])
                kept.append(rule)

        return kept

    @property
    def everything(self) -> list[Rule]:
        return [BASELINE, *self.merges, *self.flags, *self.composites]

    def winners(self) -> list[Rule]:
        return [rule for rule in self.everything if self.scores[rule.name].beats_v1]


def run_sweep(signals: Mapping[str, SignalRow], key: Mapping[str, str]) -> Sweep:
    merges = list(merge_rules())
    flags = list(replacement_rules())
    composites = list(composite_rules())
    every = [BASELINE, *merges, *flags, *composites]
    scores = {rule.name: score_rule(rule, signals, key) for rule in every}
    shapes = {rule.name: tuple(rule.decide(row) for row in signals.values()) for rule in every}
    sizes = Counter(shapes.values())
    twins = {name: sizes[shape] for name, shape in shapes.items()}

    return Sweep(
        merges=merges,
        flags=flags,
        composites=composites,
        scores=scores,
        twins=twins,
        shapes=shapes,
    )


def frontier_table(sweep: Sweep) -> str:
    best: dict[int, Rule] = {}

    for rule in ranked(sweep.merges, sweep.scores):
        score = sweep.scores[rule.name]

        if (
            score.lost not in best
            or score.merged_restatement > sweep.scores[best[score.lost].name].merged_restatement
        ):
            best[score.lost] = rule

    lines = [
        "| lost records plus lost updates | most restatements merged | rule |",
        "|---|---|---|",
    ]

    for lost in sorted(best)[:FRONTIER_ROWS]:
        score = sweep.scores[best[lost].name]
        lines.append(
            f"| {lost} | {_cell(score.merged_restatement, score.restatements)} "
            f"| {best[lost].name} |"
        )

    return "\n".join(lines)


def summary_text(
    sweep: Sweep,
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    sources: Mapping[str, str],
    published: Sequence[SaveRow],
    files: Mapping[str, Path],
) -> str:
    scores = sweep.scores
    changed, moves = stability(list(signals.values()), published)
    latency = sorted(row.adds_information.elapsed_ms for row in signals.values())
    top_composite = sweep.distinct(ranked(sweep.composites, scores))[:TOP]
    top_merge = sweep.distinct(ranked(sweep.merges, scores))[:TOP]
    twins = sweep.twins
    winners = sweep.winners()
    by_family: dict[str, list[Rule]] = {
        family: [rule for rule in sweep.composites if rule.family == family]
        for family in MERGE_KINDS
    }
    flag_plain = [rule for rule in sweep.flags if "@sim" not in rule.name]

    return "\n".join(
        [
            "# Duplicate-resolution rule sweep",
            "",
            f"Pairs with signals: {len(signals)}; keyed: {len(key)}; "
            f"disputed: {len(signals) - len(key)}.",
            "",
            "## Stability against the published save-v2 run",
            "",
            f"Pick changed on {changed} of {len(signals)} pairs.",
            *(f"- {old} -> {new}: {count}" for (old, new), count in moves.most_common()),
            "",
            "## Latency of the extra question (alone, sequential)",
            "",
            f"median {median(latency)} ms, p95 {percentile(latency, 0.95)} ms, n={len(latency)}",
            "",
            "## Baseline",
            "",
            render_table([BASELINE], scores),
            "",
            f"## Top {TOP} merge rules (merging only)",
            "",
            render_table(top_merge, scores, twins),
            "",
            "## Replacement rules alone (no similarity floor)",
            "",
            render_table(flag_plain, scores),
            "",
            f"## Top {TOP} composite rules (merge rule + replacement rule)",
            "",
            render_table(top_composite, scores, twins),
            "",
            "## Best restatements merged for each number of lost records (merge rules)",
            "",
            frontier_table(sweep),
            "",
            f"## Rules that beat v1 (merged > {V1_MERGE_BAR}, flagged >= {V1_FLAG_BAR}, "
            "0 lost records, 0 lost updates)",
            "",
            f"{len(winners)} rule(s)." if winners else "None.",
            *(f"- {rule.name}" for rule in winners[:TOP]),
            "",
            "## Per source, best three composite rules",
            "",
            source_table(top_composite[:3], signals, key, sources),
            "",
            "## Leave one source out (rule chosen on three sources, scored on the fourth)",
            "",
            leave_one_source_out(by_family, signals, key, sources),
            "",
            "## Files",
            "",
            *(f"- {name}: {path}" for name, path in files.items()),
            "",
        ]
    )


def load_signals(path: Path) -> dict[str, SignalRow]:
    return {row.pair_id: row for row in read_jsonl(path, SignalRow)}


def run(out_dir: Path, base: Path) -> Sweep:
    signals = load_signals(out_dir / "signals.jsonl")
    key = {
        pid: label for pid, label in load_key(base / "key-relation.jsonl").items() if pid in signals
    }
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    sources = {pid: pairs[pid].source for pid in key}
    published = read_jsonl(base / "save-v2.jsonl", SaveRow)
    sweep = run_sweep(signals, key)
    scores = sweep.scores
    files = {
        "full merge and replacement sweep": out_dir / "sweep.md",
        "full composite sweep": out_dir / "sweep-composite.md",
        "hard cases": out_dir / "hard-cases.md",
    }
    plain = [*sweep.merges, *sweep.flags]
    ranked_composites = ranked(sweep.composites, scores)
    loosest = max(
        sweep.composites,
        key=lambda rule: (scores[rule.name].merged_restatement, -scores[rule.name].lost),
    )
    safest = sweep.distinct(ranked_composites)[0]
    hard_rules = [safest, *([loosest] if loosest.name != safest.name else [])]

    out_dir.joinpath("summary.md").write_text(
        summary_text(sweep, signals, key, sources, published, files), encoding="utf-8"
    )
    files["full merge and replacement sweep"].write_text(
        render_table(ranked([BASELINE, *plain], scores), scores, sweep.twins) + "\n",
        encoding="utf-8",
    )
    files["full composite sweep"].write_text(
        render_table(ranked_composites, scores, sweep.twins) + "\n", encoding="utf-8"
    )
    files["hard cases"].write_text(
        hard_cases(pairs, signals, key, [*plain, *sweep.composites], hard_rules) + "\n",
        encoding="utf-8",
    )

    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep duplicate-resolution rules over captured signals."
    )
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "dup-rules")
    args = parser.parse_args()
    sweep = run(args.out_dir, args.base)

    print(f"{len(sweep.everything)} rules scored; {len(sweep.winners())} beat v1")


if __name__ == "__main__":
    main()
