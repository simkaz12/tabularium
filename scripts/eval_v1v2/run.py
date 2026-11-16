from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from eval_v1v2 import (
    corpus,
    dispute_page,
    fingerprint,
    judges,
    paths,
    report,
    score_save,
    score_search,
    targets,
)
from eval_v1v2.rows import (
    CorpusRow,
    LabelRow,
    PairRow,
    QuestionRow,
    SaveRow,
    SearchRow,
    read_jsonl,
)

REPO_ROOT = paths.REPO_ROOT
SCRIPTS = REPO_ROOT / "scripts"
V2_DIR = REPO_ROOT / "v2"
DEGRADED_LIMIT = 0.05
UNLABELLED_LIMIT = 0.05
TOP = 10
V2_ARMS = ("v2_rerank", "v2_plain", "v2_rerank_facts", "v2_plain_facts")

STAGES: tuple[str, ...] = (
    "pairs",
    "judge-relation",
    "disputes",
    "save-v1",
    "save-v2",
    "judge-targets",
    "score-save",
    "corpus",
    "questions",
    "search-v1",
    "search-v2",
    "judge-relevance",
    "score-search",
    "report",
)

Runner = Callable[[Sequence[str], Path, dict[str, str]], int]


def run_process(command: Sequence[str], cwd: Path, env: dict[str, str]) -> int:
    return subprocess.run(list(command), cwd=cwd, env=env, check=False).returncode


@dataclass
class Context:
    base: Path
    limit: int | None = None
    allow_degraded: bool = False
    force: bool = False
    runner: Runner = run_process
    log: Callable[[str], None] = field(default=lambda text: print(text, file=sys.stderr))

    def path(self, name: str) -> Path:
        return self.base / name

    def python(self, module: str, *args: str) -> list[str]:
        return [sys.executable, "-m", f"eval_v1v2.{module}", *args]

    def limit_args(self) -> list[str]:
        return [] if self.limit is None else ["--limit", str(self.limit)]

    def exec(self, command: Sequence[str], cwd: Path) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SCRIPTS), env.get("PYTHONPATH", "")]))
        code = self.runner(command, cwd, env)

        if code != 0:
            raise StageFailed(f"{' '.join(command[:3])} exited {code}")


class StageFailed(RuntimeError):
    pass


def _load[RowT: BaseModel](path: Path, model: type[RowT]) -> list[RowT]:
    return read_jsonl(path, model) if path.exists() and path.stat().st_size else []


def _pairs(ctx: Context) -> list[PairRow]:
    return _load(ctx.path("pairs.jsonl"), PairRow)[: ctx.limit]


def _questions(ctx: Context) -> list[QuestionRow]:
    return _load(ctx.path("questions.jsonl"), QuestionRow)[: ctx.limit]


def _fresh(output: Path, inputs: Sequence[Path]) -> bool:
    if not output.exists():
        return False
    stamp = output.stat().st_mtime

    return all(path.exists() and path.stat().st_mtime <= stamp for path in inputs)


def relevance_items(ctx: Context) -> list[judges.Item]:
    texts = {row.id: row.content for row in _load(ctx.path("corpus.jsonl"), CorpusRow)}
    asked = {question.question_id: question.query for question in _questions(ctx)}
    seen: dict[str, judges.Item] = {}

    for arm in ("v1", *V2_ARMS):
        for row in _load(ctx.path(f"search-{arm}.jsonl"), SearchRow):
            item_id = f"{row.question_id}:{row.record_id}"
            if row.rank > TOP or item_id in seen or row.question_id not in asked:
                continue
            if row.record_id not in texts:
                continue
            seen[item_id] = judges.Item(
                item_id=item_id,
                fields={"question": asked[row.question_id], "record": texts[row.record_id]},
            )

    return list(seen.values())


def _labelled(path: Path, item_ids: Sequence[str]) -> dict[str, set[str]]:
    wanted = set(item_ids)
    by_judge: dict[str, set[str]] = {"claude": set(), "codex": set()}

    for row in _load(path, LabelRow):
        if row.item_id in wanted:
            by_judge[row.judge].add(row.item_id)

    return by_judge


def _judged_complete(path: Path, item_ids: Sequence[str]) -> bool:
    if not item_ids:
        return False

    return all(len(done) == len(set(item_ids)) for done in _labelled(path, item_ids).values())


def _save_complete(ctx: Context, system: str) -> bool:
    wanted = {pair.pair_id for pair in _pairs(ctx)}
    done = {row.pair_id for row in _load(ctx.path(f"save-{system}.jsonl"), SaveRow)}

    return bool(wanted) and wanted <= done


def _search_complete(ctx: Context, arms: Sequence[str]) -> bool:
    wanted = {question.question_id for question in _questions(ctx)}

    if not wanted:
        return False

    for arm in arms:
        done = {row.question_id for row in _load(ctx.path(f"search-{arm}.jsonl"), SearchRow)}
        if not wanted <= done:
            return False

    return True


def is_complete(ctx: Context, stage: str) -> bool:
    inputs = {
        "disputes": ("labels-relation.jsonl",),
        "score-save": (
            "key-relation.jsonl",
            "save-v1.jsonl",
            "save-v2.jsonl",
            f"{targets.TARGETS_DIR}/labels-relation.jsonl",
        ),
        "score-search": (
            "labels-relevance.jsonl",
            "search-v1.jsonl",
            *(f"search-{arm}.jsonl" for arm in V2_ARMS),
        ),
        "report": ("score-save.json", "score-search.json", "key-relation.jsonl"),
    }
    outputs = {
        "disputes": "disputes-relation.html",
        "score-save": "score-save.json",
        "score-search": "score-search.json",
        "report": "report.md",
    }

    if stage in outputs:
        return _fresh(ctx.path(outputs[stage]), [ctx.path(name) for name in inputs[stage]])

    checks: dict[str, Callable[[], bool]] = {
        "pairs": lambda: all(
            ctx.path(name).exists() and ctx.path(name).stat().st_size > 0
            for name in ("pairs.jsonl", "noise.jsonl")
        ),
        "judge-targets": lambda: _judged_complete(
            ctx.path(targets.TARGETS_DIR) / "labels-relation.jsonl",
            [item.item_id for item in targets.items_for(ctx.base)[1]],
        ),
        "judge-relation": lambda: _judged_complete(
            ctx.path("labels-relation.jsonl"), [pair.pair_id for pair in _pairs(ctx)]
        ),
        "save-v1": lambda: _save_complete(ctx, "v1"),
        "save-v2": lambda: _save_complete(ctx, "v2"),
        "corpus": lambda: bool(_load(ctx.path("corpus.jsonl"), CorpusRow)),
        "questions": lambda: bool(_questions(ctx)),
        "search-v1": lambda: _search_complete(ctx, ("v1",)),
        "search-v2": lambda: _search_complete(ctx, V2_ARMS),
        "judge-relevance": lambda: _judged_complete(
            ctx.path("labels-relevance.jsonl"), [item.item_id for item in relevance_items(ctx)]
        ),
    }

    return checks[stage]()


def check_quality(ctx: Context) -> list[str]:
    breaches: list[str] = []

    for system in ("v1", "v2"):
        rows = _load(ctx.path(f"save-{system}.jsonl"), SaveRow)
        degraded = sum(1 for row in rows if row.degraded is not None)
        if rows and degraded / len(rows) > DEGRADED_LIMIT:
            breaches.append(
                f"{system}: {degraded} of {len(rows)} save rows degraded, "
                f"{round(100 * degraded / len(rows))}%, "
                f"over the {round(DEGRADED_LIMIT * 100)}% limit"
            )

    populations = {
        "relation": [pair.pair_id for pair in _pairs(ctx)],
        "relevance": [item.item_id for item in relevance_items(ctx)],
    }

    for task, item_ids in populations.items():
        if not item_ids:
            continue
        for judge, done in _labelled(ctx.path(f"labels-{task}.jsonl"), item_ids).items():
            missing = len(set(item_ids)) - len(done)
            if missing / len(set(item_ids)) > UNLABELLED_LIMIT:
                breaches.append(
                    f"{judge} left {missing} of {len(set(item_ids))} {task} items unlabelled, "
                    f"{round(100 * missing / len(set(item_ids)))}%, over the "
                    f"{round(UNLABELLED_LIMIT * 100)}% limit"
                )

    return breaches


def run_stage(ctx: Context, stage: str) -> None:
    pairs, noise = ctx.path("pairs.jsonl"), ctx.path("noise.jsonl")
    corpus_path, questions_path = ctx.path("corpus.jsonl"), ctx.path("questions.jsonl")

    if stage == "pairs":
        ctx.exec(ctx.python("pairs"), REPO_ROOT)
    elif stage == "judge-relation":
        items = judges.items_from_pairs(pairs, ctx.limit)
        judges.run_task("relation", items, out_dir=ctx.base)
    elif stage == "disputes":
        dispute_page.build_page(ctx.base)
    elif stage == "save-v1":
        out = ctx.path("save-v1.jsonl")
        args = ["--pairs", str(pairs), "--noise", str(noise), "--out", str(out)]
        ctx.exec(ctx.python("v1_save", *args, *ctx.limit_args()), REPO_ROOT)
    elif stage == "save-v2":
        out = ctx.path("save-v2.jsonl")
        args = ["--pairs", str(pairs), "--noise", str(noise), "--out", str(out)]
        ctx.exec(["node", "eval/save.ts", *args, *ctx.limit_args()], V2_DIR)
    elif stage == "judge-targets":
        items = targets.items_for(ctx.base)[1][: ctx.limit]
        judges.run_task("relation", items, out_dir=ctx.path(targets.TARGETS_DIR))
    elif stage == "score-save":
        score_save.score_files(ctx.base)
    elif stage == "corpus":
        corpus.build(out=corpus_path)
    elif stage == "questions":
        ctx.exec(ctx.python("questions", "--top-up"), REPO_ROOT)
    elif stage == "search-v1":
        out = ctx.path("search-v1.jsonl")
        args = ["--corpus", str(corpus_path), "--questions", str(questions_path), "--out", str(out)]
        ctx.exec(ctx.python("v1_search", *args, *ctx.limit_args()), REPO_ROOT)
    elif stage == "search-v2":
        args = ["--corpus", str(corpus_path), "--questions", str(questions_path)]
        ctx.exec(
            ["node", "eval/search.ts", *args, "--out-dir", str(ctx.base), *ctx.limit_args()], V2_DIR
        )
    elif stage == "judge-relevance":
        judges.run_task("relevance", relevance_items(ctx), out_dir=ctx.base)
    elif stage == "score-search":
        score_search.score_files(ctx.base)
    elif stage == "report":
        write_report(ctx)
    else:
        raise StageFailed(f"unknown stage {stage}")


def write_report(ctx: Context) -> None:
    breaches = check_quality(ctx)

    if breaches and not ctx.allow_degraded:
        raise StageFailed(
            "refusing to write report.md, the run is degraded:\n  "
            + "\n  ".join(breaches)
            + "\npass --allow-degraded to write it with a warning at the top"
        )

    report.write_report(ctx.base, breaches, ctx.limit)


def refuse_stale(ctx: Context, stage: str) -> None:
    touched = (*fingerprint.STAGE_OUTPUTS.get(stage, ()), *fingerprint.STAGE_READS.get(stage, ()))
    changed = fingerprint.stale(ctx.base, touched)

    if changed:
        raise StageFailed(
            f"{stage}: inputs changed since these outputs were produced: {', '.join(changed)}. "
            f"Delete them from {ctx.base} and rerun; resuming would keep labels and rows for "
            "text that no longer matches."
        )


def warn_on_resume(ctx: Context, stage: str) -> None:
    out = ctx.path(f"{stage}.jsonl")

    if stage in {"save-v1", "save-v2"} and out.exists() and out.stat().st_size > 0:
        ctx.log(
            f"{stage}: resuming a partial {out.name}; the new process pays its own cold call, "
            "so timings from this file include cold calls beyond the first row"
        )


def run_pipeline(ctx: Context, only: str | None = None) -> list[str]:
    ran: list[str] = []

    fingerprint.adopt(ctx.base)

    for stage in [only] if only is not None else STAGES:
        refuse_stale(ctx, stage)

        if not ctx.force and is_complete(ctx, stage):
            ctx.log(f"skip {stage}: output complete")
            continue

        ctx.log(f"run {stage}")
        warn_on_resume(ctx, stage)
        run_stage(ctx, stage)
        fingerprint.record(ctx.base, fingerprint.STAGE_OUTPUTS.get(stage, ()))
        ran.append(stage)

    return ran


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the v1 vs v2 eval stage by stage.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stage", choices=STAGES, default=None)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    ctx = Context(
        base=paths.ensure_out_dir(),
        limit=args.limit,
        allow_degraded=args.allow_degraded,
        force=args.force,
    )

    try:
        run_pipeline(ctx, args.stage)
    except StageFailed as error:
        print(str(error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
