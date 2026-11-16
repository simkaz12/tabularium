from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from eval_v1v2 import agent_fold, paths
from eval_v1v2.noise_relation import (
    FIRST_RUN,
    JUDGE_DIR,
    KINDS,
    READING_RULE,
    RELATION_RULE,
    RESTATEMENT_RULE,
    SUMMARY_OUT,
    VERDICTS,
    WRONG_REINFORCE_OUT,
    Interruptions,
    NoiseJudgments,
    Readings,
    Tally,
    Verdict,
    folded_pairs,
    interruptions,
    load_judgments,
    load_noise_texts,
    load_save,
    noise_tallies,
    relation_events,
    resolve_runs,
    run_dir,
    run_name,
    score_readings,
    split_sendable,
    with_covers_grades,
)
from eval_v1v2.prod_check import noise_events
from eval_v1v2.rows import PairRow, SaveRow, read_jsonl

CELLS = r"\| (\d+) of (\d+).*?\| (\d+) of (\d+)"
ROW_PATTERNS: Mapping[str, str] = {
    "noticed": r"\| Restatement noticed \(caught or flagged\) " + CELLS,
    "flagged": r"\| Replacement flagged " + CELLS,
    "alarm": r"\| Follow-up or distinct flagged \(false alarm\) " + CELLS,
    "caught": r"\| Restatement caught " + CELLS,
}
BENCH_AGENT_FOLD = {"restatement": "36 of 39", "replacement": "41 of 42", "asks": "38 of 242"}


@dataclass(frozen=True)
class RunResult:
    name: str
    readings: Readings
    interrupted: Interruptions
    tallies: dict[str, Tally] | None


def published_rows(report: Path) -> dict[str, tuple[str, str]]:
    text = report.read_text(encoding="utf-8")
    rows: dict[str, tuple[str, str]] = {}

    for name, pattern in ROW_PATTERNS.items():
        match = re.search(pattern, text)

        if match is None:
            raise ValueError(f"row {name} not found in {report}")

        first, second, third, fourth = match.groups()
        rows[name] = (f"{first} of {second}", f"{third} of {fourth}")

    return rows


def gather(
    replay: Path,
    judged: NoiseJudgments,
    data: agent_fold.AgentData,
    runs: Sequence[str],
) -> list[RunResult]:
    results: list[RunResult] = []

    for run in (FIRST_RUN, *runs):
        directory = run_dir(replay, run)
        save = load_save(directory / "save-v2.jsonl")
        folded = folded_pairs(directory)
        relation = judged.relation if run != FIRST_RUN else {}
        results.append(
            RunResult(
                name=run_name(run),
                readings=score_readings(save, folded, data, relation),
                interrupted=interruptions(save, data.inputs.relation, folded),
                tallies=noise_tallies(save, judged) if run != FIRST_RUN else None,
            )
        )

    return results


def tally_table(results: Sequence[RunResult]) -> list[str]:
    lines = [
        "| Run | Event kind | Right | Wrong | Unkeyed | Left out | Total |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        if result.tallies is None:
            continue

        overall = Tally()

        for kind in KINDS:
            tally = result.tallies[kind]
            overall.counts.update(tally.counts)
            cells = " | ".join(str(tally.counts[verdict]) for verdict in VERDICTS)
            lines.append(f"| {result.name} | {kind} | {cells} | {tally.total()} |")

        cells = " | ".join(f"**{overall.counts[verdict]}**" for verdict in VERDICTS)
        lines.append(f"| {result.name} | all noise events | {cells} | **{overall.total()}** |")

    return lines


def readings_table(results: Sequence[RunResult]) -> list[str]:
    lines = [
        "| Run | Reading | Restatements resolved | Replacements caught |",
        "| --- | --- | --- | --- |",
    ]

    for result in results:
        r = result.readings
        lines.append(
            f"| {result.name} | seed only | {r.seed_resolved} of {r.restatements} | "
            f"{r.seed_caught} of {r.replacements} |"
        )

        if result.tallies is not None:
            lines.append(
                f"| {result.name} | no new duplicate | "
                f"{r.resolved_no_new_duplicate} of {r.restatements} | "
                f"{r.caught_no_new_duplicate} of {r.replacements} |"
            )

    return lines


def extras_lines(results: Sequence[RunResult]) -> list[str]:
    lines: list[str] = []

    for result in results:
        if result.tallies is None:
            continue

        restated = ", ".join(result.readings.extra_resolved) or "none"
        replaced = ", ".join(result.readings.extra_caught) or "none"
        lines.append(
            f"- {result.name}: restatement pairs added {restated}; "
            f"replacement pairs added {replaced}."
        )

    return lines


def comparison_table(
    results: Sequence[RunResult], published: Mapping[str, tuple[str, str]]
) -> list[str]:
    v1, v2 = 0, 1
    lines = [
        "| Row | Restatements | Replacements | Interruptions on follow_up or distinct pairs |",
        "| --- | --- | --- | --- |",
        f"| v1 published | {published['noticed'][v1]} noticed | "
        f"{published['flagged'][v1]} flagged | "
        f"{published['alarm'][v1]} (bounces, nothing written) |",
        f"| v2 published (report.md, before containment and folds) | "
        f"{published['noticed'][v2]} noticed, {published['caught'][v2]} caught | "
        f"{published['flagged'][v2]} flagged | {published['alarm'][v2]} flagged |",
        f"| Bench Opus agent-fold | {BENCH_AGENT_FOLD['restatement']} | "
        f"{BENCH_AGENT_FOLD['replacement']} | asks on {BENCH_AGENT_FOLD['asks']} saves, "
        "0 truly distinct pairs folded |",
    ]

    for result in results:
        r = result.readings
        i = result.interrupted
        floor = "0.90" if result.name == FIRST_RUN else "0.95"
        through = f" ({i.covers_asked} through covers)" if i.covers_asked else ""
        tail = f"{i.asked} asks{through}, {i.flagged} flags, {i.folded} folded of {i.keyed_new}"
        lines.append(
            f"| replay {result.name} (floor {floor}), seed only | "
            f"{r.seed_resolved} of {r.restatements} resolved | "
            f"{r.seed_caught} of {r.replacements} caught | {tail} |"
        )

        if result.tallies is not None:
            lines.append(
                f"| replay {result.name}, no new duplicate | "
                f"{r.resolved_no_new_duplicate} of {r.restatements} resolved | "
                f"{r.caught_no_new_duplicate} of {r.replacements} caught | same |"
            )

    return lines


def wrong_reinforce_text(
    results: Sequence[RunResult],
    judged: NoiseJudgments,
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
    left_out: Mapping[tuple[str, str], Mapping[str, str]],
) -> str:
    wrong: dict[tuple[str, str], list[str]] = {}

    for result in results:
        if result.tallies is None:
            continue

        for key in result.tallies["relation suggestion"].reinforced_wrong:
            wrong.setdefault(key, []).append(result.name)

    lines = [
        "# Reinforces on noise rows the judges do not key as a restatement",
        "",
        f"Wrong reinforces: {len(wrong)} pairs. Per run: "
        + ", ".join(
            f"{result.name} {len(result.tallies['relation suggestion'].reinforced_wrong)}"
            for result in results
            if result.tallies is not None
        )
        + ".",
        "",
    ]

    for (pair_id, noise_id), runs in sorted(wrong.items()):
        row = judged.relation[(pair_id, noise_id)]
        lines += [
            f"## {pair_id}",
            "",
            f"- Noise id: {noise_id}",
            f"- Runs: {', '.join(runs)}",
            f"- Key: {row.label}, judges {row.labels}",
            f"- Existing (noise note): {noise[noise_id]}",
            f"- Incoming (not saved): {incoming[pair_id]}",
            "",
        ]

    unkeyed = [
        row
        for row in judged.relation.values()
        if row.label is None and "reinforce" in row.actions.values()
    ]
    lines += ["## Reinforces with no key", ""]
    lines.append(
        "These are not counted as wrong. The judges split, so the key gives no answer. "
        "The incoming note was not saved in either case."
    )
    lines.append("")

    for row in sorted(unkeyed, key=lambda item: item.pair_id):
        lines += [
            f"### {row.pair_id}",
            "",
            f"- Noise id: {row.noise_id}",
            f"- Judges: {row.labels}",
            f"- Existing (noise note): {noise[row.noise_id]}",
            f"- Incoming (not saved): {incoming[row.pair_id]}",
            "",
        ]

    held = sorted(
        f"{pair} ({sorted(set(actions.values()))[0]})" for (pair, _), actions in left_out.items()
    )
    lines += [
        "## Left out",
        "",
        f"Not sent to the judges because a text names a client: {', '.join(held) or 'none'}. "
        "A reinforce among them dropped the incoming note and is unverified.",
        "",
    ]

    return "\n".join(lines)


def noise_probability(row: SaveRow, noise_id: str) -> float | None:
    neighbours = row.neighbours or []

    return next((n.new_covers_old for n in neighbours if n.id == noise_id), None)


def wrong_containments(
    runs: Mapping[str, Sequence[SaveRow]], judged: NoiseJudgments
) -> dict[tuple[str, str], dict[str, float | None]]:
    found: dict[tuple[str, str], dict[str, float | None]] = {}

    for name, save in runs.items():
        by_pair = {row.pair_id: row for row in save}

        for pair_id, kind, noise_id in noise_events(save):
            verdict = judged.earlier.get((pair_id, noise_id))

            if kind == "containment" and verdict is not None and verdict.new_covers_old == "no":
                probability = noise_probability(by_pair[pair_id], noise_id)
                found.setdefault((pair_id, noise_id), {})[name] = probability

    return found


def probability_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def wrong_containment_lines(
    runs: Mapping[str, Sequence[SaveRow]],
    judged: NoiseJudgments,
    noise: Mapping[str, str],
    incoming: Mapping[str, str],
) -> list[str]:
    found = wrong_containments(runs, judged)
    hit_runs = sorted({name for by_run in found.values() for name in by_run}, key=len_then_name)
    lines = [
        "## Containments on noise rows the judges key as not covered",
        "",
        f"{len(hit_runs)} of {len(runs)} runs contain a wrong noise containment "
        f"({', '.join(hit_runs) or 'none'}). The probability is the noise row's "
        "`new_covers_old` in that run's propose call.",
        "",
    ]

    for (pair_id, noise_id), by_run in sorted(found.items()):
        per_run = ", ".join(
            f"{name} {probability_text(by_run[name])}" for name in sorted(by_run, key=len_then_name)
        )
        lines += [
            f"- {pair_id} on noise row {noise_id}: {per_run}.",
            "",
            f"  Incoming note ({pair_id}), in full:",
            "",
            *quoted(incoming[pair_id]),
            "",
            f"  Noise row {noise_id}, in full:",
            "",
            *quoted(noise[noise_id]),
            "",
        ]

    return lines


def quoted(text: str) -> list[str]:
    return [f"  > {line}".rstrip() for line in text.splitlines()]


def len_then_name(name: str) -> tuple[int, str]:
    return len(name), name


def count(result: RunResult, kind: str, verdict: Verdict) -> int:
    assert result.tallies is not None

    return result.tallies[kind].counts[verdict]


def total(result: RunResult, kind: str) -> int:
    assert result.tallies is not None

    return result.tallies[kind].total()


def per_run(results: Sequence[RunResult], render: Callable[[RunResult], str]) -> str:
    return "; ".join(f"{result.name} {render(result)}" for result in results if result.tallies)


def right_wrong(kind: str) -> Callable[[RunResult], str]:
    return lambda result: (
        f"{count(result, kind, 'right')} right, {count(result, kind, 'wrong')} wrong, "
        f"{count(result, kind, 'unkeyed')} unkeyed, {count(result, kind, 'left_out')} left out"
    )


def reading_lines(
    results: Sequence[RunResult],
    kappa: float,
    agreed: int,
    judged_total: int,
    left_out_total: int,
    wrong_total: int,
) -> list[str]:
    scored = [result for result in results if result.tallies is not None]

    def added(result: RunResult) -> str:
        r = result.readings

        return f"{len(r.extra_resolved)} restatements and {len(r.extra_caught)} replacements"

    return [
        f"- Relation events: {judged_total + left_out_total} distinct (pair, noise row) events "
        f"across {len(scored)} runs. {judged_total} were sent to the judges and they agreed on "
        f"{agreed} (kappa {kappa:.2f}). "
        f"Per run: {per_run(scored, right_wrong('relation suggestion'))}.",
        f"- Containment: {per_run(scored, right_wrong('containment'))}.",
        f"- Asks: {per_run(scored, right_wrong('ask'))}. An ask on a noise row is graded only "
        "when a merged note for it was judged; the others are left out.",
        f"- Wrong reinforces: {wrong_total} pairs keyed wrong. Reinforces with no key or left out "
        f"are listed in `{WRONG_REINFORCE_OUT}`; the incoming note was dropped on each of them "
        "and I cannot say whether that was right.",
        f"- Re-scoring: counting noise actions adds {per_run(scored, added)}.",
    ]


def support_lines(
    results: Sequence[RunResult],
    published: Mapping[str, tuple[str, str]],
    left_out: Mapping[tuple[str, str], Mapping[str, str]],
    wrong_total: int,
) -> list[str]:
    scored = [result for result in results if result.tallies is not None]
    published_resolved = int(published["noticed"][1].split(" of ")[0])
    seed = [result.readings.seed_resolved for result in results]
    unkeyed = max(count(result, "relation suggestion", "unkeyed") for result in scored)
    held = sorted({pair for pair, _ in left_out})
    wrong_line = (
        "no keyed reinforce dropped a note that should have been kept."
        if wrong_total == 0
        else f"{wrong_total} pairs had a keyed reinforce on a note that is not a restatement."
    )

    return [
        "Supported:",
        "",
        f"- Where the judges agree, v2's action on a noise row was right or wrong as counted "
        f"per run in the verdict table, and {wrong_line}",
        f"- Seed-only restatements resolved run from {min(seed)} to {max(seed)} across the runs "
        f"in the table, against {published_resolved} noticed in the published v2 row, which "
        "predates containment and folds.",
        "",
        "Not supported:",
        "",
        "- That the replay beats v1. The v1 figure counts a bounce as noticed and writes nothing. "
        "The replay figure counts a write that left no duplicate, and its folds replay recorded "
        "Opus answers, not live agents under task pressure.",
        f"- That the unkeyed relation events are fine. The judges split on up to {unkeyed} of "
        "them in one run and nobody else has looked.",
        f"- Anything about the left-out pairs ({', '.join(held) or 'none'}).",
        "- That ask-on-noise is harmless in production. It cost an interruption here only "
        "because the replayed agent always answered or the ask expired.",
        "- That the follow_up and distinct column matches the v1 column. v1 stops on any "
        "similar record; the replay column counts asks and flags on a keyed pair, and its folds "
        "on follow_up pairs are not graded as errors.",
    ]


def summary_text(
    results: Sequence[RunResult],
    published: Mapping[str, tuple[str, str]],
    left_out: Mapping[tuple[str, str], Mapping[str, str]],
    wrong_total: int,
    agreement: Mapping[str, object],
    judged_total: int,
    containment_lines: Sequence[str] = (),
) -> str:
    held = ", ".join(f"{pair} (noise {noise[:8]})" for pair, noise in sorted(left_out))
    restatement = f"- Restatement rule: {RESTATEMENT_RULE}"
    reading = f"- Reading `no new duplicate`: {READING_RULE}"
    seed = (
        "- Reading `seed only` is the current scoring: a seed-labelled pair counts only when "
        "the action hit the seed."
    )

    return "\n".join(
        [
            "# Production replay: what v2 did to noise rows",
            "",
            "## Question",
            "",
            "In each replay v2 sometimes acted on a noise row instead of the seed. "
            "This file asks whether that action was right for that row, then re-scores the "
            "answer key under a reading that counts those actions.",
            "",
            "## Rules",
            "",
            f"- Relation events: {RELATION_RULE}",
            "- Containment events (runs before run-9 only; later runs send every covers hit to "
            "the ask): right when the judges key `new_covers_old` yes on (noise note, incoming).",
            "- Ask events, including a covers ask on a noise row: right when the judges key "
            "`one_record` yes on the merged note.",
            restatement,
            reading,
            seed,
            "",
            "## Noise-event verdicts",
            "",
            *tally_table(results),
            "",
            f"Left out of the relation judging because a text names a client: {held}.",
            "",
            *containment_lines,
            "## Re-scored under both readings",
            "",
            *readings_table(results),
            "",
            "Pairs added by `no new duplicate`:",
            "",
            *extras_lines(results),
            "",
            f"Wrong reinforces: {wrong_total} pairs, listed in `{WRONG_REINFORCE_OUT}`.",
            "",
            "## Reading the verdicts",
            "",
            *reading_lines(
                results,
                float(str(agreement["cohens_kappa"])),
                int(str(agreement["items_agreed"])),
                judged_total,
                len(left_out),
                wrong_total,
            ),
            "",
            "## Comparison rows",
            "",
            *comparison_table(results, published),
            "",
            "What the rows support and what they do not.",
            "",
            *support_lines(results, published, left_out, wrong_total),
            "",
        ]
    )


def report_command(base: Path, replay: Path, requested: Sequence[str] = ()) -> int:
    names = resolve_runs(replay, requested)
    judged = load_judgments(replay)
    data = with_covers_grades(
        agent_fold.load_data(base, base / "experiments" / "agent-fold"), replay
    )
    results = gather(replay, judged, data, names)
    noise = load_noise_texts(base)
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    incoming = {pair_id: pair.incoming for pair_id, pair in pairs.items()}
    runs = {run_name(run): load_save(run_dir(replay, run) / "save-v2.jsonl") for run in names}
    _, left = split_sendable(relation_events(runs), noise, incoming)
    held = {(event.pair_id, event.noise_id): event.actions for event in left}
    wrong = {
        key
        for result in results
        if result.tallies is not None
        for key in result.tallies["relation suggestion"].reinforced_wrong
    }
    published = published_rows(base / "report.md")

    (replay / WRONG_REINFORCE_OUT).write_text(
        wrong_reinforce_text(results, judged, noise, incoming, held), encoding="utf-8"
    )
    agreement = json.loads((replay / JUDGE_DIR / "agreement-relation.json").read_text())
    text = summary_text(
        results,
        published,
        held,
        len(wrong),
        agreement,
        len(judged.relation),
        wrong_containment_lines(runs, judged, noise, incoming),
    )
    (replay / SUMMARY_OUT).write_text(text, encoding="utf-8")
    sys.stdout.write(text)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score and write the noise-event summary.")
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument("--replay", type=Path, default=None)
    parser.add_argument("runs", nargs="*", help="run directory names under --replay")
    args = parser.parse_args(argv)
    replay = args.replay or args.base / "experiments" / "prod-replay"

    return report_command(args.base, replay, args.runs)


if __name__ == "__main__":
    sys.exit(main())
