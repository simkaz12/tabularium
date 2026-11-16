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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from eval_v1v2 import paths
from eval_v1v2.dup_rules import (
    BASELINE,
    Decision,
    Rule,
    Score,
    SignalRow,
    baseline_decision,
    load_signals,
    score_rule,
    source_table,
)
from eval_v1v2.rows import LabelRow, PairRow, Row, SaveRow, read_jsonl
from eval_v1v2.score_save import load_key, median, percentile
from tabularium.engine.judge import JudgeUnavailable, SubprocessJudgeAdapter

MODELS: tuple[str, ...] = ("haiku", "sonnet")
LABELS = ("restatement", "replacement", "follow_up", "distinct")
NONE = "none"
CONCURRENCY = 4
CALL_TIMEOUT = 300.0
MAX_ATTEMPTS = 2
NONE_STOP_SHARE = 0.05
SIMILARITY_FLOORS: tuple[float | None, ...] = (None, 0.80, 0.85)
CONFIDENCE_FLOORS: tuple[float | None, ...] = (None, 0.7, 0.8, 0.9)
HARD_PAIRS = ("bounce-0796", "bounce-0872", "bounce-0990", "bounce-1010", "bounce-1370")
UNRELATED = frozenset({"follow_up", "distinct"})
LABEL_DECISION: dict[str, Decision] = {
    "restatement": "merge",
    "replacement": "flag_replace",
    "follow_up": "new",
    "distinct": "new",
}
EXISTING_TOKEN = "<<EXISTING>>"
INCOMING_TOKEN = "<<INCOMING>>"

PROMPT = """\
You guard a memory store against duplicates. EXISTING is a note already saved. INCOMING is a note
about to be saved. Decide how INCOMING relates to EXISTING.

Answer with exactly one label:
- restatement: INCOMING says nothing EXISTING does not already say, even if it is worded
  differently, is longer, or is told as a narrative of what happened.
- replacement: INCOMING contradicts or updates EXISTING, so EXISTING is now wrong or stale.
- follow_up: same topic, and INCOMING adds a fact, decision or consequence EXISTING lacks.
- distinct: different subjects, even if some words overlap.

The two mistakes do not cost the same. Calling a follow_up a restatement throws INCOMING away and
the new information is lost for good. Calling a restatement a follow_up only leaves a duplicate,
which is cheap to clean up later. When you are not sure, answer follow_up. When you answer
follow_up, name the new information in one short phrase.

Reply with one JSON object and nothing else:
{"label": "<label>", "confidence": <number from 0 to 1>,
"new_information": "<what INCOMING adds, or empty>"}

EXISTING:
<<EXISTING>>

INCOMING:
<<INCOMING>>
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class SecondCall(Row):
    pair_id: str
    model: str
    label: str
    confidence: float | None
    new_information: str | None
    elapsed_ms: float
    attempts: int
    error: str | None


class Answer(Row):
    label: Literal["restatement", "replacement", "follow_up", "distinct"]
    confidence: float
    new_information: str = ""


class UnparseableAnswer(Exception):
    pass


Calls = Mapping[str, Mapping[str, SecondCall]]


def render_prompt(existing: str, incoming: str) -> str:
    return PROMPT.replace(EXISTING_TOKEN, existing).replace(INCOMING_TOKEN, incoming)


def parse_answer(text: str) -> Answer:
    found = _JSON_OBJECT.search(text)

    if found is None:
        raise UnparseableAnswer(text[:200])

    try:
        answer = Answer.model_validate(json.loads(found.group(0)))
    except (ValueError, ValidationError) as exc:
        raise UnparseableAnswer(str(exc)[:200]) from exc

    if not 0.0 <= answer.confidence <= 1.0:
        raise UnparseableAnswer(f"confidence {answer.confidence} outside 0 to 1")

    return answer


def in_band(signal: SignalRow, similarity_floor: float | None) -> bool:
    existing = signal.existing

    if not signal.found_existing or existing is None or existing.pick != "follow_up":
        return False

    return similarity_floor is None or existing.similarity >= similarity_floor


def band_pair_ids(signals: Mapping[str, SignalRow], similarity_floor: float | None) -> list[str]:
    return [pid for pid, signal in signals.items() if in_band(signal, similarity_floor)]


async def ask_once(
    adapter: SubprocessJudgeAdapter, model: str, pair: PairRow, attempts: int
) -> SecondCall:
    started = time.perf_counter()
    error: str | None = None
    answer: Answer | None = None

    try:
        answer = parse_answer(await adapter.call(render_prompt(pair.existing, pair.incoming)))
    except (JudgeUnavailable, UnparseableAnswer) as exc:
        error = f"{type(exc).__name__}: {str(exc)[:200]}"

    elapsed = round((time.perf_counter() - started) * 1000, 2)

    return SecondCall(
        pair_id=pair.pair_id,
        model=model,
        label=NONE if answer is None else answer.label,
        confidence=None if answer is None else answer.confidence,
        new_information=None if answer is None else answer.new_information,
        elapsed_ms=elapsed,
        attempts=attempts,
        error=error,
    )


async def ask(adapter: SubprocessJudgeAdapter, model: str, pair: PairRow) -> SecondCall:
    result = await ask_once(adapter, model, pair, 1)

    for attempt in range(2, MAX_ATTEMPTS + 1):
        if result.label != NONE:
            break

        result = await ask_once(adapter, model, pair, attempt)

    return result


async def ask_all(
    adapter: SubprocessJudgeAdapter,
    model: str,
    pairs: Sequence[PairRow],
    sink: Callable[[SecondCall], None],
) -> list[SecondCall]:
    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(pair: PairRow) -> SecondCall:
        async with gate:
            result = await ask(adapter, model, pair)

        sink(result)

        return result

    return list(await asyncio.gather(*(one(pair) for pair in pairs)))


def none_share(calls: Sequence[SecondCall]) -> float:
    return sum(1 for call in calls if call.label == NONE) / len(calls) if calls else 0.0


def usable(call: SecondCall | None, confidence_floor: float | None) -> bool:
    if call is None or call.label == NONE or call.confidence is None:
        return False

    return confidence_floor is None or call.confidence >= confidence_floor


def _floor_text(prefix: str, value: float | None) -> str:
    return "" if value is None else f" {prefix}>={value:.2f}"


def second_rule(
    models: tuple[str, ...],
    calls: Calls,
    similarity_floor: float | None,
    confidence_floor: float | None,
) -> Rule:
    label = "both agree" if len(models) > 1 else models[0]
    name = f"{label}{_floor_text('sim', similarity_floor)}{_floor_text('conf', confidence_floor)}"

    def decide(signal: SignalRow) -> Decision:
        if not in_band(signal, similarity_floor):
            return baseline_decision(signal)

        answers = [calls.get(model, {}).get(signal.pair_id) for model in models]

        if not all(usable(answer, confidence_floor) for answer in answers):
            return baseline_decision(signal)

        labels = {answer.label for answer in answers if answer is not None}

        return LABEL_DECISION[labels.pop()] if len(labels) == 1 else baseline_decision(signal)

    return Rule(name=name, family=label, decide=decide)


def all_second_rules(calls: Calls) -> list[Rule]:
    groups: tuple[tuple[str, ...], ...] = (("haiku",), ("sonnet",), ("haiku", "sonnet"))

    return [
        second_rule(models, calls, sim, conf)
        for models in groups
        for sim in SIMILARITY_FLOORS
        for conf in CONFIDENCE_FLOORS
    ]


def calls_needed(rule_name: str, signals: Mapping[str, SignalRow]) -> int:
    match = re.search(r"sim>=(\d\.\d+)", rule_name)

    return len(band_pair_ids(signals, float(match.group(1)) if match else None))


def load_codex_key(path: Path) -> dict[str, str]:
    return {
        row.item_id: str(row.label) for row in read_jsonl(path, LabelRow) if row.judge == "codex"
    }


def v1_bars(published: Sequence[SaveRow], key: Mapping[str, str]) -> tuple[int, int]:
    noticed = sum(
        1
        for row in published
        if key.get(row.pair_id) == "restatement" and row.bucket in ("caught", "flagged")
    )
    flagged = sum(
        1 for row in published if key.get(row.pair_id) == "replacement" and row.bucket == "flagged"
    )

    return noticed, flagged


def beats(score: Score, bars: tuple[int, int]) -> bool:
    return (
        score.merged_restatement > bars[0]
        and score.flagged_replacement >= bars[1]
        and score.lost_record == 0
        and score.lost_update == 0
    )


def rule_table(
    rules: Sequence[Rule],
    scores: Mapping[str, Score],
    signals: Mapping[str, SignalRow],
    bars: tuple[int, int],
) -> str:
    lines = [
        "| rule | restatements merged | lost records | lost updates | replacements flagged "
        "| false flags | disputed merge/flag/new | second-judge calls | beats v1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for rule in rules:
        score = scores[rule.name]
        disputed = score.disputed
        calls = 0 if rule.name == BASELINE.name else calls_needed(rule.name, signals)
        lines.append(
            f"| {rule.name} | {score.merged_restatement}/{score.restatements} "
            f"| {score.lost_record}/{score.unrelated} "
            f"| {score.lost_update}/{score.replacements} "
            f"| {score.flagged_replacement}/{score.replacements} | {score.false_flag} "
            f"| {disputed['merge']}/{disputed['flag_replace']}/{disputed['new']} "
            f"| {calls} | {'YES' if beats(score, bars) else ''} |"
        )

    return "\n".join(lines)


def cell(call: SecondCall | None) -> str:
    if call is None:
        return "not asked"

    if call.label == NONE:
        return "none"

    return f"{call.label} {call.confidence:.2f}"


def hard_pairs_table(
    ids: Sequence[str],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    calls: Calls,
) -> str:
    lines = [
        "| pair_id | key | Jev pick | similarity | haiku | sonnet | haiku new information "
        "| sonnet new information |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for pid in ids:
        existing = signals[pid].existing
        haiku, sonnet = calls["haiku"].get(pid), calls["sonnet"].get(pid)
        lines.append(
            f"| {pid} | {key.get(pid, 'disputed')} "
            f"| {'none' if existing is None else existing.pick} "
            f"| {'none' if existing is None else f'{existing.similarity:.3f}'} "
            f"| {cell(haiku)} | {cell(sonnet)} "
            f"| {_flat(None if haiku is None else haiku.new_information, 120)} "
            f"| {_flat(None if sonnet is None else sonnet.new_information, 120)} |"
        )

    return "\n".join(lines)


def _flat(text: str | None, limit: int = 200) -> str:
    return " ".join((text or "").split())[:limit].replace("|", "/")


def timing_table(calls: Calls) -> str:
    lines = [
        "| model | calls | median ms | p95 ms | retried | ended as none |",
        "|---|---|---|---|---|---|",
    ]

    for model in MODELS:
        rows = list(calls[model].values())
        times = sorted(call.elapsed_ms for call in rows)
        lines.append(
            f"| {model} | {len(rows)} | {median(times)} | {percentile(times, 0.95)} "
            f"| {sum(1 for call in rows if call.attempts > 1)} "
            f"| {sum(1 for call in rows if call.label == NONE)} |"
        )

    return "\n".join(lines)


def share_table(signals: Mapping[str, SignalRow]) -> str:
    lines = [
        "| similarity floor | pairs needing a second call | share of all pairs |",
        "|---|---|---|",
    ]

    for floor in SIMILARITY_FLOORS:
        count = len(band_pair_ids(signals, floor))
        lines.append(
            f"| {'none' if floor is None else f'{floor:.2f}'} | {count} "
            f"| {count / len(signals):.1%} |"
        )

    return "\n".join(lines)


def lost_text(
    pairs: Mapping[str, PairRow],
    signals: Mapping[str, SignalRow],
    key: Mapping[str, str],
    codex: Mapping[str, str],
    calls: Calls,
) -> str:
    lines = [
        "# Records a second judge would merge although the key calls them follow-up or distinct",
        "",
        "Holds record text; stays under dev-corpus. A row appears when either model answered "
        "restatement inside the band and the two-judge key or the Codex label is "
        "follow_up or distinct.",
        "",
        "| pair_id | source | similarity | two-judge key | codex | haiku | sonnet | existing "
        "| incoming |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for pid in band_pair_ids(signals, None):
        said = {m: calls[m].get(pid) for m in MODELS}
        merging = any(call is not None and call.label == "restatement" for call in said.values())
        unrelated = key.get(pid) in UNRELATED or codex.get(pid) in UNRELATED

        if not (merging and unrelated):
            continue

        existing = signals[pid].existing
        similarity = 0.0 if existing is None else existing.similarity
        lines.append(
            f"| {pid} | {pairs[pid].source} | {similarity:.3f} | {key.get(pid, 'disputed')} "
            f"| {codex.get(pid, 'none')} | {cell(said['haiku'])} | {cell(said['sonnet'])} "
            f"| {_flat(pairs[pid].existing)} | {_flat(pairs[pid].incoming)} |"
        )

    return "\n".join(lines)


def best_rule(
    rules: Sequence[Rule], scores: Mapping[str, Score], signals: Mapping[str, SignalRow]
) -> Rule:
    def order(rule: Rule) -> tuple[int, int, int, int]:
        score = scores[rule.name]

        return (
            score.lost,
            -(score.merged_restatement + score.flagged_replacement),
            score.false_flag,
            calls_needed(rule.name, signals),
        )

    return min(rules, key=order)


def summary_text(
    signals: Mapping[str, SignalRow],
    pairs: Mapping[str, PairRow],
    key: Mapping[str, str],
    codex: Mapping[str, str],
    calls: Calls,
    published: Sequence[SaveRow],
) -> str:
    rules = all_second_rules(calls)
    sections = ["# Second judge on the follow_up band", ""]
    sections += [
        f"Band size: {len(band_pair_ids(signals, None))} pairs. "
        f"Two-judge key: {len(key)} pairs; Codex-alone key: {len(codex)} pairs.",
        "",
        "## Call times and failures",
        "",
        timing_table(calls),
        "",
        "## Share of pairs that need a second call",
        "",
        share_table(signals),
        "",
        "This pair set is full of near-duplicates; the share in real traffic is unknown and lower.",
        "",
    ]
    winners: dict[str, list[str]] = {}

    for title, labels in (("two-judge key", key), ("Codex-alone key", codex)):
        bars = v1_bars(published, labels)
        scores = {rule.name: score_rule(rule, signals, labels) for rule in [BASELINE, *rules]}
        best = best_rule(rules, scores, signals)
        winners[title] = [rule.name for rule in rules if beats(scores[rule.name], bars)]
        only = frozenset(labels)
        sources = {pid: pairs[pid].source for pid in only}
        sections += [
            f"## Rules against the {title}",
            "",
            f"v1 notices {bars[0]} restatements and flags {bars[1]} replacements on this key; "
            "a rule beats v1 when it merges more than the first, flags at least the second, "
            "and loses nothing.",
            "",
            rule_table([BASELINE, *rules], scores, signals, bars),
            "",
            f"Best rule by fewest losses, then most right, then fewest false flags: `{best.name}`.",
            "",
            source_table([BASELINE, best], signals, labels, sources),
            "",
            f"Rules that beat v1 on this key: "
            f"{', '.join(winners[title]) if winners[title] else 'none'}.",
            "",
        ]

    sections += [
        "## The five restatements no rule merged before",
        "",
        hard_pairs_table(HARD_PAIRS, signals, key, calls),
        "",
    ]

    return "\n".join(sections)


def load_calls(path: Path) -> dict[str, dict[str, SecondCall]]:
    loaded: dict[str, dict[str, SecondCall]] = {model: {} for model in MODELS}

    if path.exists():
        for call in read_jsonl(path, SecondCall):
            loaded.setdefault(call.model, {})[call.pair_id] = call

    return loaded


def run_calls(out_dir: Path, base: Path, models: Sequence[str]) -> int:
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    band = [pairs[pid] for pid in band_pair_ids(signals, None)]
    binary = shutil.which("claude")

    if binary is None:
        print("claude CLI not found", file=sys.stderr)

        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("prompt.txt").write_text(PROMPT, encoding="utf-8")
    log = out_dir / "calls.jsonl"
    handle = log.open("a", encoding="utf-8")

    def sink(call: SecondCall) -> None:
        handle.write(json.dumps(call.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()

    os.chdir(tempfile.mkdtemp(prefix="second-judge-"))

    for model in models:
        adapter = SubprocessJudgeAdapter(
            cli="claude", binary=binary, model=model, timeout=CALL_TIMEOUT
        )
        results = asyncio.run(ask_all(adapter, model, band, sink))
        share = none_share(results)
        print(f"{model}: {len(results)} calls, none share {share:.1%}", flush=True)

        if share > NONE_STOP_SHARE:
            print(f"{model} exceeded {NONE_STOP_SHARE:.0%} none; stopping", file=sys.stderr)

            return 2

    handle.close()

    return 0


def run_score(out_dir: Path, base: Path) -> None:
    signals = load_signals(base / "experiments" / "dup-rules" / "signals.jsonl")
    pairs = {pair.pair_id: pair for pair in read_jsonl(base / "pairs.jsonl", PairRow)}
    full_key = load_key(base / "key-relation.jsonl")
    key = {pid: label for pid, label in full_key.items() if pid in signals}
    codex = load_codex_key(base / "labels-relation.jsonl")
    calls = load_calls(out_dir / "calls.jsonl")
    published = [row for row in read_jsonl(base / "save-v1.jsonl", SaveRow) if row.system == "v1"]
    out_dir.joinpath("summary.md").write_text(
        summary_text(signals, pairs, key, codex, calls, published) + "\n", encoding="utf-8"
    )
    out_dir.joinpath("lost.md").write_text(
        lost_text(pairs, signals, key, codex, calls) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Second judge on the follow_up band.")
    parser.add_argument("phase", choices=("call", "score"))
    parser.add_argument("--base", type=Path, default=paths.OUT_DIR)
    parser.add_argument(
        "--out-dir", type=Path, default=paths.OUT_DIR / "experiments" / "second-judge"
    )
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    if args.phase == "call":
        sys.exit(run_calls(args.out_dir, args.base, args.models))

    run_score(args.out_dir, args.base)


if __name__ == "__main__":
    main()
