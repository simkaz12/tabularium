from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from eval_v1v2 import paths
from eval_v1v2.dispute_page import find_disputes
from eval_v1v2.rows import KeyRow, LabelRow, PairRow, QuestionRow, SaveRow, read_jsonl
from eval_v1v2.score_save import RELATIONS, Rate, SaveScore, SystemScore, median
from eval_v1v2.score_search import ARMS, ArmScore, SearchScore, VariantScore
from eval_v1v2.targets import CATEGORIES

REPORT_MD = "report.md"
REPORT_HTML = "report.html"

LIMITS: tuple[str, ...] = (
    "The replay sees only each system's first verdict. v1 bounces are not answered and v2 "
    "supersede suggestions are not applied.",
    "Times are this machine, in-process, not the deployed server. They exclude process start "
    "and the "
    "first call of each search arm.",
    "v2 legacy rows use the fallback shape, which splits content without a model, not the "
    "migration's rewritten headlines.",
    "The answer key is two-judge agreement until the owner settles the disputes. Pairs the "
    "judges split on are outside the save scores until then.",
    "v1 ran with code defaults: no auto-exit flag on, the Jev Assessor on for saves.",
    "40 questions is a small set. Read the interval, not the point value.",
    "The search corpus is a snapshot from the pool export, not the store at cutover.",
    "Each search driver runs its self-check and warm-up recalls before the timed questions, and "
    "both engines log reads, so about 20 records carry a few extra read events when the timed "
    "questions run.",
    "The v1 store is folded with upkeep() after the bulk load, as production does on its save "
    "path.",
)

SELECTION_LIMIT = (
    "{bounce} of the {pairs} pairs come from the bounce log: pairs v1's own gate bounced when the "
    "ledger was first loaded. That v1 flags them again is partly built into how they were chosen. "
    "It raises both v1's noticed rate on true duplicates and its false-alarm rate on that "
    "source. v2 faces no such selection. The per-source table above shows the split."
)

SAVE_TIME_LIMIT = (
    "v1's save median mixes cheap saves where the gate raised no candidate with saves that "
    "waited on the Jev Assessor: {plain} saves without an Assessor call have a median of "
    "{plain_ms} ms, {assessed} saves with one have a median of {assessed_ms} ms. The save-time "
    "gap between v1 and v2 reflects how many pairs reached the Assessor and is not an "
    "engine-speed comparison."
)

AGREEMENT_LIMIT = (
    "Relevance agreement is low{figures}. The key takes the lower of the two grades, which "
    "pushes absolute nDCG down, and the per-judge tables show the spread. Compare arms within "
    "one table, not across tables."
)

COARSE_LIMIT = (
    "Only questions with at least one graded record above 0 count toward nDCG: {counts}. "
    "Grade-2 hits are rare, so hit@1, hit@5 and MRR are too coarse to rank the arms."
)

BOUNCE_NOTE = (
    "v1 stopped and asked the agent what to do; v2 decides by itself. A v1 bounce writes "
    "nothing, so it is counted as flagged, never caught. v1 reaches caught only when it "
    "reinforces, revives or reports already saved on the existing record. On a true "
    "duplicate, flagged for v1 and caught for v2 are both the system noticing, so the "
    "tables show the strict buckets and a noticed rate (caught or flagged on the right "
    "record)."
)


class AgreementSummary(BaseModel):
    task: str
    both: int
    agreed: int
    raw: float
    kappa: float


class ReportData(BaseModel):
    warnings: list[str] = []
    limit: int | None = None
    pairs: int
    key_size: int
    key_agreed: int
    key_owner: int
    key_by_label: dict[str, int]
    disputed: int
    waiting_on_owner: int
    unlabelled_by_judge: dict[str, int]
    v1_assessor_calls: int = 0
    v1_assessor_median_ms: float | None = None
    v1_plain_calls: int = 0
    v1_plain_median_ms: float | None = None
    corpus_rows: int
    mined: int
    generated: int
    agreements: list[AgreementSummary]
    bounce_pairs: int = 0
    save: SaveScore
    search: SearchScore


@dataclass(frozen=True)
class Heading:
    level: int
    text: str


@dataclass(frozen=True)
class Para:
    text: str


@dataclass(frozen=True)
class Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Bars:
    title: str
    items: tuple[tuple[str, float, str], ...]


@dataclass(frozen=True)
class Bullets:
    items: tuple[str, ...]


Block = Heading | Para | Table | Bars | Bullets


def fmt_rate(rate: Rate, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    if rate.total == 0:
        return f"0 of 0{suffix}, n/a"

    return f"{rate.count} of {rate.total}{suffix}, {round(100 * rate.count / rate.total)}%"


def fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def fmt_num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _share(rate: Rate) -> float:
    return rate.count / rate.total if rate.total else 0.0


def _system_table(systems: Sequence[SystemScore]) -> Table:
    rows: list[tuple[str, Callable[[SystemScore], str]]] = [
        ("Pairs scored", lambda s: f"{s.scored} of {s.rows} ({s.unkeyed} not in key)"),
        ("Restatement caught", lambda s: fmt_rate(s.duplicate_caught)),
        ("Restatement noticed (caught or flagged)", lambda s: fmt_rate(s.duplicate_noticed)),
        ("Replacement flagged", lambda s: fmt_rate(s.replacement_flagged)),
        ("Replacement caught (lost update)", lambda s: fmt_rate(s.lost_update)),
        ("Follow-up or distinct caught (lost record)", lambda s: fmt_rate(s.lost_record)),
        ("Follow-up or distinct flagged (false alarm)", lambda s: fmt_rate(s.false_alarm)),
        ("Lost records, any target", lambda s: fmt_rate(s.lost_record_any, "scored pairs")),
        ("False alarms, any target", lambda s: fmt_rate(s.false_alarm_any, "scored pairs")),
        ("Existing record raised as a candidate", lambda s: fmt_rate(s.found_existing)),
        ("Relation pick right, given found", lambda s: fmt_rate(s.relation_pick)),
        ("Degraded saves", lambda s: fmt_rate(s.degraded)),
        ("Save time median, ms", lambda s: f"{fmt_ms(s.median_ms)} over {s.timed_calls} calls"),
        ("Save time p95, ms", lambda s: f"{fmt_ms(s.p95_ms)} over {s.timed_calls} calls"),
    ]

    return Table(
        headers=("Measure", *(s.system for s in systems)),
        rows=tuple((label, *(cell(s) for s in systems)) for label, cell in rows),
    )


CANDIDATE_NOTE = (
    "For v1 a record is raised only when its similarity clears the gate floor (0.80 for facts, "
    "0.70 for decisions), while v2 always passes its 3 nearest records to the judge, so the two "
    "rates are not the same measure."
)

ANY_TARGET_NOTE = (
    "The any-target rows add the wrong-target saves among scored pairs to the rows above them: "
    "a merge onto a record both judges call follow-up or distinct is a lost record, and a flag "
    "on one is a false alarm. Their denominator is all scored pairs."
)


def _source_table(systems: Sequence[SystemScore]) -> Table:
    rows = tuple(
        (
            system.system,
            row.source,
            str(row.scored),
            fmt_rate(row.restatement_noticed),
            fmt_rate(row.replacement_flagged),
            fmt_rate(row.false_alarm),
        )
        for system in systems
        for row in system.by_source
    )

    return Table(
        headers=(
            "System",
            "Source",
            "Pairs scored",
            "Restatement noticed",
            "Replacement flagged",
            "Follow-up or distinct flagged (false alarm)",
        ),
        rows=rows,
    )


def _ndcg_counts(data: ReportData) -> str:
    questions = data.mined + data.generated
    tables = (("as called", data.search.as_called), ("facts only", data.search.facts_only))
    parts: list[str] = []

    for title, variants in tables:
        scored = next((v for v in variants if v.variant == "min"), None)
        if scored is None or not scored.arms:
            continue

        counts = sorted({arm.ndcg_questions for arm in scored.arms})
        figure = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
        parts.append(f"{figure} of {questions} in the {title} table")

    return ", ".join(parts)


def limits(data: ReportData) -> tuple[str, ...]:
    relevance = next((a for a in data.agreements if a.task == "relevance"), None)
    figures = (
        f": raw agreement {fmt_num(relevance.raw)}, Cohen's kappa {fmt_num(relevance.kappa)}"
        if relevance
        else ""
    )

    return (
        *LIMITS,
        SAVE_TIME_LIMIT.format(
            plain=data.v1_plain_calls,
            plain_ms=fmt_ms(data.v1_plain_median_ms),
            assessed=data.v1_assessor_calls,
            assessed_ms=fmt_ms(data.v1_assessor_median_ms),
        ),
        SELECTION_LIMIT.format(bounce=data.bounce_pairs, pairs=data.pairs),
        AGREEMENT_LIMIT.format(figures=figures),
        COARSE_LIMIT.format(counts=_ndcg_counts(data)),
    )


def _pointed_table(system: SystemScore) -> Table:
    cells = system.pointed.cells

    return Table(
        headers=("Action", *CATEGORIES, "total"),
        rows=tuple(
            (
                action,
                *(str(cells.get(action, {}).get(category, 0)) for category in CATEGORIES),
                str(sum(cells.get(action, {}).values())),
            )
            for action in ("merged", "flagged")
        ),
    )


def _pointed_sentences(system: SystemScore) -> Bullets:
    p = system.pointed

    return Bullets(
        (
            f"{p.merged_right} merges onto a record both judges call a restatement: "
            "the system was right.",
            f"{p.merged_lost_record} merges onto a record both judges call follow-up or "
            "distinct: a lost record.",
            f"{p.cells.get('merged', {}).get('replacement', 0)} merges onto a replacement: "
            "a lost update.",
            f"{p.flagged_right} flags on a true restatement or replacement: right.",
            f"{p.flagged_false_alarm} flags on follow-up or distinct: false alarm.",
            f"{p.unresolved} wrong-target saves could not be resolved to a record text.",
        )
    )


def _relation_table(system: SystemScore) -> Table:
    buckets = ("caught", "flagged", "new", "wrong_target")
    ordered = {row.relation: row for row in system.by_relation}
    rows = tuple(
        (
            relation,
            str(ordered[relation].total),
            *(str(ordered[relation].buckets.get(bucket, 0)) for bucket in buckets),
            str(ordered[relation].noticed),
        )
        for relation in RELATIONS
        if relation in ordered
    )

    return Table(headers=("Key label", "Pairs", *buckets, "noticed"), rows=rows)


def _arm_table(variant: VariantScore) -> Table:
    rows = tuple(
        (
            arm.arm,
            f"{fmt_num(arm.ndcg)} over {arm.ndcg_questions} questions",
            fmt_rate(arm.hit1),
            fmt_rate(arm.hit5),
            fmt_rate(arm.precision5, "slots"),
            f"{fmt_num(arm.mrr)} over {arm.mrr_questions} questions",
        )
        for arm in variant.arms
    )

    return Table(
        headers=(
            "Arm",
            "nDCG@10",
            "hit@1 (grade 2)",
            "hit@5 (grade 2)",
            "precision@5 (grade 1+)",
            "MRR",
        ),
        rows=rows,
    )


def _versus_table(variant: VariantScore) -> Table:
    rows = tuple(
        (
            arm.arm,
            f"{arm.wins} / {arm.ties} / {arm.losses} of "
            f"{(arm.wins or 0) + (arm.ties or 0) + (arm.losses or 0)}",
            fmt_signed(arm.diff_mean),
            f"[{fmt_signed(arm.diff_low)}, {fmt_signed(arm.diff_high)}]",
        )
        for arm in variant.arms
        if arm.wins is not None
    )

    return Table(
        headers=("Arm", "Wins / ties / losses vs v1", "nDCG difference", "95% bootstrap interval"),
        rows=rows,
    )


AS_CALLED_NOTE = (
    "v1 recall returns facts only by design, so this table is what an agent got from one recall "
    "call in each system."
)

FACTS_ONLY_NOTE = (
    "This is the like-for-like retrieval comparison: the v2 arms filter to kind fact, and the "
    "judged pool and the ideal ranking hold fact records only."
)

VARIANT_TITLES = {
    "min": "Key grade is the lower of the two judges",
    "claude": "Claude's grades alone",
    "codex": "Codex's grades alone",
}


def _time_table(search: SearchScore) -> Table:
    mixes = {mix.arm: mix for mix in search.kind_mix}
    timed: dict[str, ArmScore] = {}

    for variants in (search.as_called, search.facts_only):
        for variant in variants:
            if variant.variant == "min":
                timed.update({arm.arm: arm for arm in variant.arms if arm.arm not in timed})

    rows = tuple(
        (
            arm.arm,
            fmt_ms(arm.median_ms),
            fmt_ms(arm.p95_ms),
            str(arm.timed_calls),
            str(mixes[arm.arm].facts) if arm.arm in mixes else "n/a",
            str(mixes[arm.arm].decisions) if arm.arm in mixes else "n/a",
        )
        for arm in sorted(timed.values(), key=lambda scored: ARMS.index(scored.arm))
    )

    return Table(
        headers=(
            "Arm",
            "Median ms",
            "p95 ms",
            "Calls",
            "Facts in top 10",
            "Decisions in top 10",
        ),
        rows=rows,
    )


def _variant_blocks(variants: Sequence[VariantScore]) -> list[Block]:
    blocks: list[Block] = []

    for variant in variants:
        blocks.append(Heading(4, VARIANT_TITLES.get(variant.variant, variant.variant)))
        blocks.append(
            Para(
                f"The judged pool is the union of all five arms' top 10s, {variant.graded} "
                "records. The ideal ranking in this table uses the whole pool (facts only in the "
                f"facts-only table); {variant.leftover} pooled records sit outside this table's "
                f"own arms. {variant.unjudged} returned results carry no grade and count as 0."
            )
        )
        blocks.append(_arm_table(variant))
        silent = [
            f"{arm.arm} {arm.questions - arm.answered}"
            for arm in variant.arms
            if arm.answered < arm.questions
        ]

        if silent:
            blocks.append(Para("Questions with no results, scored 0: " + ", ".join(silent) + "."))

        blocks.append(_versus_table(variant))

        if variant.variant == "min":
            blocks.append(
                Bars(
                    "nDCG@10 by arm",
                    tuple(
                        (
                            arm.arm,
                            arm.ndcg or 0.0,
                            f"{fmt_num(arm.ndcg)} over {arm.ndcg_questions} questions",
                        )
                        for arm in variant.arms
                    ),
                )
            )

    return blocks


def _search_blocks(search: SearchScore) -> list[Block]:
    return [
        Heading(3, "As called"),
        Para(AS_CALLED_NOTE),
        *_variant_blocks(search.as_called),
        Heading(3, "Facts only"),
        Para(FACTS_ONLY_NOTE),
        *_variant_blocks(search.facts_only),
        Heading(3, "Time per recall call"),
        _time_table(search),
    ]


def build_blocks(data: ReportData) -> list[Block]:
    blocks: list[Block] = [Heading(1, "v1 vs v2 eval")]

    if data.warnings:
        blocks.append(Heading(2, "Degraded run"))
        blocks.append(
            Para("This report was written with --allow-degraded. Treat the numbers as unreliable.")
        )
        blocks.append(Bullets(tuple(data.warnings)))

    if data.limit is not None:
        blocks.append(Para(f"Smoke run: limited to {data.limit} pairs and questions."))

    key_labels = ", ".join(f"{label} {count}" for label, count in sorted(data.key_by_label.items()))
    blocks.append(Heading(2, "Answer key"))
    blocks.append(
        Bullets(
            (
                f"Relation key: {data.key_size} of {data.pairs} pairs "
                f"({data.key_agreed} judges agreed, {data.key_owner} owner). {key_labels}.",
                f"Disputed pairs: {data.disputed}, "
                f"of which {data.waiting_on_owner} wait on the owner.",
                "Pairs a judge left unlabelled: "
                + ", ".join(f"{judge} {count}" for judge, count in data.unlabelled_by_judge.items())
                + ".",
                f"Questions: {data.mined + data.generated}, "
                f"{data.mined} mined from transcripts and {data.generated} generated.",
                f"Search corpus: {data.corpus_rows} records.",
            )
        )
    )

    if data.agreements:
        blocks.append(
            Table(
                headers=("Task", "Labelled by both", "Agreed", "Raw agreement", "Cohen's kappa"),
                rows=tuple(
                    (
                        a.task,
                        str(a.both),
                        fmt_rate(Rate(count=a.agreed, total=a.both)),
                        fmt_num(a.raw),
                        fmt_num(a.kappa),
                    )
                    for a in data.agreements
                ),
            )
        )

    blocks.append(Heading(2, "Save side"))
    blocks.append(Para(BOUNCE_NOTE))
    blocks.append(_system_table(data.save.systems))
    blocks.append(Heading(3, "Save results by pair source"))
    blocks.append(_source_table(data.save.systems))
    blocks.append(Para(CANDIDATE_NOTE))
    blocks.append(Para(ANY_TARGET_NOTE))
    blocks.append(
        Bars(
            "Restatements noticed (caught or flagged)",
            tuple(
                (s.system, _share(s.duplicate_noticed), fmt_rate(s.duplicate_noticed))
                for s in data.save.systems
            ),
        )
    )
    blocks.append(
        Bars(
            "Restatements strictly caught",
            tuple(
                (s.system, _share(s.duplicate_caught), fmt_rate(s.duplicate_caught))
                for s in data.save.systems
            ),
        )
    )

    for system in data.save.systems:
        blocks.append(Heading(3, f"{system.system} buckets by key label"))
        blocks.append(_relation_table(system))

    blocks.append(Heading(2, "Pointed at another record"))
    blocks.append(
        Para(
            "Saves bucketed wrong_target, over all pairs and not only keyed ones. The target is "
            "the record the system pointed at; two judges labelled its relation to the incoming "
            "text, and a cell reads judges split when they disagreed. These labels live apart "
            "from the answer key."
        )
    )

    for system in data.save.systems:
        blocks.append(Heading(3, f"{system.system} pointed at another record"))
        blocks.append(_pointed_table(system))
        blocks.append(_pointed_sentences(system))

    blocks.append(Heading(2, "Search side"))
    blocks.append(
        Para(
            "Grades are 0, 1 and 2, given blind to which arm returned the record. "
            "nDCG uses the grade as the gain and the ideal list from the judged pool. "
            "Questions with no graded record above 0 are left out of nDCG. "
            "Generated questions are listed apart from mined ones above."
        )
    )
    blocks.extend(_search_blocks(data.search))

    blocks.append(Heading(2, "Limits"))
    blocks.append(Bullets(limits(data)))

    return blocks


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(data: ReportData) -> str:
    lines: list[str] = []

    for block in build_blocks(data):
        if isinstance(block, Heading):
            lines.extend(["", f"{'#' * block.level} {block.text}", ""])
        elif isinstance(block, Para):
            lines.extend([block.text, ""])
        elif isinstance(block, Bullets):
            lines.extend([f"- {item}" for item in block.items])
            lines.append("")
        elif isinstance(block, Table):
            lines.append("| " + " | ".join(_md_cell(h) for h in block.headers) + " |")
            lines.append("|" + "|".join(" --- " for _ in block.headers) + "|")
            lines.extend("| " + " | ".join(_md_cell(c) for c in row) + " |" for row in block.rows)
            lines.append("")
        else:
            lines.extend([f"**{block.title}**", ""])
            lines.extend(f"- {label}: {text}" for label, _, text in block.items)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


STYLE = """
:root { color-scheme: light dark; --bg: #ffffff; --fg: #1b1f24; --muted: #57606a;
  --line: #d0d7de; --bar: #2f6fdb; --track: #e6ebf1; --warn: #9a3412; --warn-bg: #fff1e6; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171a; --fg: #e6e9ec; --muted: #9aa4af; --line: #38404a;
    --bar: #6aa0ff; --track: #262c33; --warn: #ffb27a; --warn-bg: #2a1d12; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif;
  max-width: 62rem; margin: 2rem auto; padding: 0 1rem; }
table { border-collapse: collapse; margin: 1rem 0; width: 100%; }
th, td { border: 1px solid var(--line); padding: 0.35rem 0.6rem; text-align: left;
  vertical-align: top; }
th { color: var(--muted); font-weight: 600; }
.bars { margin: 1rem 0; }
.bar-row { display: grid; grid-template-columns: 7rem 1fr 16rem; gap: 0.6rem;
  align-items: center; margin: 0.25rem 0; }
.track { background: var(--track); height: 0.9rem; border-radius: 3px; }
.fill { background: var(--bar); height: 100%; border-radius: 3px; }
.warn { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn);
  padding: 0.5rem 0.8rem; border-radius: 4px; }
"""


def _percent(value: float) -> float:
    return max(0.0, min(value, 1.0)) * 100


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _html_block(block: Block, warn: bool) -> str:
    if isinstance(block, Heading):
        return f"<h{block.level}>{_esc(block.text)}</h{block.level}>"
    if isinstance(block, Para):
        return f"<p>{_esc(block.text)}</p>"
    if isinstance(block, Bullets):
        items = "".join(f"<li>{_esc(item)}</li>" for item in block.items)
        return f'<ul class="warn">{items}</ul>' if warn else f"<ul>{items}</ul>"
    if isinstance(block, Table):
        head = "".join(f"<th>{_esc(h)}</th>" for h in block.headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in block.rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    rows = "".join(
        f'<div class="bar-row"><span>{_esc(label)}</span>'
        f'<div class="track"><div class="fill" style="width:{_percent(value):.1f}%">'
        f"</div></div><span>{_esc(text)}</span></div>"
        for label, value, text in block.items
    )

    return f'<div class="bars"><strong>{_esc(block.title)}</strong>{rows}</div>'


def render_html(data: ReportData) -> str:
    parts: list[str] = []
    in_degraded = False

    for block in build_blocks(data):
        if isinstance(block, Heading):
            in_degraded = block.text == "Degraded run"
        parts.append(_html_block(block, in_degraded and isinstance(block, Bullets)))

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>v1 vs v2 eval</title><style>{STYLE}</style></head><body>"
        + "\n".join(parts)
        + "</body></html>\n"
    )


def _read_json(path: Path) -> dict[str, object]:
    payload: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))

    return payload


def _agreement(base: Path, task: str) -> AgreementSummary | None:
    path = base / f"agreement-{task}.json"
    if not path.exists():
        return None

    raw = _read_json(path)

    return AgreementSummary(
        task=task,
        both=int(str(raw["items_labelled_by_both"])),
        agreed=int(str(raw["items_agreed"])),
        raw=float(str(raw["raw_agreement"])),
        kappa=float(str(raw["cohens_kappa"])),
    )


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


JUDGES = ("claude", "codex")


def _assessor_call(row: SaveRow) -> bool:
    return row.relation_pick is not None or row.raw_outcome.startswith("bounced")


def _v1_time_split(base: Path) -> tuple[list[float], list[float]]:
    path = base / "save-v1.jsonl"
    rows = read_jsonl(path, SaveRow)[1:] if path.exists() else []
    assessed = [row.elapsed_ms for row in rows if _assessor_call(row)]
    plain = [row.elapsed_ms for row in rows if not _assessor_call(row)]

    return assessed, plain


def collect(base: Path, warnings: Sequence[str] = (), limit: int | None = None) -> ReportData:
    pairs = read_jsonl(base / "pairs.jsonl", PairRow)
    pairs = pairs[:limit]
    ids = {pair.pair_id for pair in pairs}
    labels = read_jsonl(base / "labels-relation.jsonl", LabelRow)
    key_path = base / "key-relation.jsonl"
    key = (
        [row for row in read_jsonl(key_path, KeyRow) if row.item_id in ids]
        if key_path.exists()
        else []
    )
    disputed = {item for item in find_disputes(labels) if item in ids}
    settled = {row.item_id for row in key if row.source == "owner"}
    unlabelled = {
        judge: len(ids - {row.item_id for row in labels if row.judge == judge}) for judge in JUDGES
    }
    questions = read_jsonl(base / "questions.jsonl", QuestionRow)[:limit]
    origins = Counter(question.origin for question in questions)
    agreements = [a for a in (_agreement(base, t) for t in ("relation", "relevance")) if a]

    assessed, plain = _v1_time_split(base)

    return ReportData(
        warnings=list(warnings),
        limit=limit,
        pairs=len(pairs),
        bounce_pairs=sum(1 for pair in pairs if pair.source == "bounce_log"),
        key_size=len(key),
        key_agreed=sum(1 for row in key if row.source == "agreed"),
        key_owner=sum(1 for row in key if row.source == "owner"),
        key_by_label=dict(Counter(str(row.label) for row in key)),
        disputed=len(disputed),
        waiting_on_owner=len(disputed - settled),
        unlabelled_by_judge=unlabelled,
        v1_assessor_calls=len(assessed),
        v1_assessor_median_ms=median(assessed),
        v1_plain_calls=len(plain),
        v1_plain_median_ms=median(plain),
        corpus_rows=_line_count(base / "corpus.jsonl"),
        mined=origins.get("mined", 0),
        generated=origins.get("generated", 0),
        agreements=agreements,
        save=SaveScore.model_validate(_read_json(base / "score-save.json")),
        search=SearchScore.model_validate(_read_json(base / "score-search.json")),
    )


def write_report(base: Path, warnings: Sequence[str] = (), limit: int | None = None) -> None:
    data = collect(base, warnings, limit)
    (base / REPORT_MD).write_text(render_markdown(data), encoding="utf-8")
    (base / REPORT_HTML).write_text(render_html(data), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write report.md and report.html.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    write_report(paths.ensure_out_dir(), limit=args.limit)


if __name__ == "__main__":
    main()
