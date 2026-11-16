from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import fingerprint, report, run, score_save, score_search, targets  # noqa: E402
from eval_v1v2.rows import (  # noqa: E402
    CorpusRow,
    LabelRow,
    NoiseRow,
    PairRow,
    QuestionRow,
    SaveRow,
    SearchRow,
    write_jsonl,
)

sys.path.pop(0)

VECTOR = [0.1] * 768


def save_row(
    pair_id: str,
    system: str,
    bucket: str,
    *,
    found: bool,
    pick: str | None,
    ms: float,
    degraded: str | None = None,
) -> SaveRow:
    return SaveRow.model_validate(
        {
            "pair_id": pair_id,
            "system": system,
            "bucket": bucket,
            "found_existing": found,
            "relation_pick": pick,
            "raw_outcome": "x",
            "elapsed_ms": ms,
            "degraded": degraded,
        }
    )


KEY = {
    "p1": "restatement",
    "p2": "restatement",
    "p3": "restatement",
    "p4": "restatement",
    "p5": "replacement",
    "p6": "replacement",
    "p7": "distinct",
    "p8": "distinct",
    "p9": "follow_up",
}

V1_ROWS = [
    save_row("p1", "v1", "flagged", found=True, pick="restatement", ms=10),
    save_row("p2", "v1", "caught", found=True, pick="replacement", ms=20),
    save_row("p3", "v1", "new", found=False, pick=None, ms=30),
    save_row("p4", "v1", "wrong_target", found=False, pick=None, ms=40),
    save_row("p5", "v1", "flagged", found=True, pick="replacement", ms=50),
    save_row("p6", "v1", "caught", found=True, pick="restatement", ms=60),
    save_row("p7", "v1", "new", found=False, pick=None, ms=70),
    save_row("p8", "v1", "flagged", found=True, pick="follow_up", ms=80),
    save_row("p9", "v1", "caught", found=True, pick=None, ms=90, degraded="assessor_withheld"),
    save_row("p10", "v1", "new", found=False, pick=None, ms=100),
]


def test_save_rates_have_known_counts() -> None:
    score = score_save.score_save(V1_ROWS, KEY, ("v1",))
    v1 = score.systems[0]

    assert (v1.rows, v1.scored, v1.unkeyed) == (10, 9, 1)
    assert (v1.duplicate_caught.count, v1.duplicate_caught.total) == (1, 4)
    assert (v1.duplicate_noticed.count, v1.duplicate_noticed.total) == (2, 4)
    assert (v1.replacement_flagged.count, v1.replacement_flagged.total) == (1, 2)
    assert (v1.lost_update.count, v1.lost_update.total) == (1, 2)
    assert (v1.lost_record.count, v1.lost_record.total) == (1, 3)
    assert (v1.false_alarm.count, v1.false_alarm.total) == (1, 3)
    assert (v1.found_existing.count, v1.found_existing.total) == (6, 9)
    assert (v1.relation_pick.count, v1.relation_pick.total) == (2, 5)
    assert (v1.degraded.count, v1.degraded.total) == (1, 10)
    assert v1.median_ms == 60
    assert v1.p95_ms == 100
    assert v1.timed_calls == 9


def test_save_relation_table_counts_buckets() -> None:
    v1 = score_save.score_save(V1_ROWS, KEY, ("v1",)).systems[0]
    table = {row.relation: row for row in v1.by_relation}

    assert table["restatement"].buckets == {
        "caught": 1,
        "flagged": 1,
        "new": 1,
        "wrong_target": 1,
    }
    assert table["restatement"].noticed == 2
    assert table["distinct"].buckets["flagged"] == 1


def test_save_systems_stay_apart() -> None:
    other = save_row("p1", "v2", "caught", found=True, pick="restatement", ms=5)
    score = score_save.score_save([*V1_ROWS, other], KEY)
    v2 = score.systems[1]

    assert v2.scored == 1
    assert v2.duplicate_caught.count == 1
    assert score.systems[0].duplicate_caught.count == 1


def test_percentile_is_nearest_rank() -> None:
    assert score_save.percentile([1, 2, 3, 4], 0.5) == 2
    assert score_save.percentile([1, 2, 3, 4], 0.95) == 4
    assert score_save.percentile([], 0.5) is None


def label(item_id: str, judge: str, grade: int) -> LabelRow:
    return LabelRow.model_validate(
        {"item_id": item_id, "judge": judge, "model": "m", "label": grade, "reason": "r"}
    )


def hit(question: str, arm: str, rank: int, record: str, ms: float) -> SearchRow:
    return SearchRow.model_validate(
        {
            "question_id": question,
            "arm": arm,
            "rank": rank,
            "record_id": record,
            "score": 1.0,
            "elapsed_ms": ms,
        }
    )


GRADES = {
    "q1:a": (2, 2),
    "q1:b": (1, 2),
    "q1:c": (0, 0),
    "q1:d": (2, 1),
    "q2:e": (2, 2),
    "q2:f": (0, 0),
}
LABELS = [
    label(item, judge, grade)
    for item, (claude, codex) in GRADES.items()
    for judge, grade in (("claude", claude), ("codex", codex))
]
KINDS = dict.fromkeys("abcdef", "fact")
HITS = [
    *(hit("q1", "v1", r, rec, 10) for r, rec in enumerate("cba", start=1)),
    *(hit("q2", "v1", r, rec, 30) for r, rec in enumerate("fe", start=1)),
    *(hit("q1", "v2_plain", r, rec, 20) for r, rec in enumerate("abc", start=1)),
    *(hit("q2", "v2_plain", r, rec, 40) for r, rec in enumerate("ef", start=1)),
]


def test_grade_is_lower_of_two_judges() -> None:
    low = score_search.grades_for(LABELS, "min")

    assert low[("q1", "b")] == 1
    assert low[("q1", "d")] == 1
    assert score_search.grades_for(LABELS, "claude")[("q1", "d")] == 2
    assert score_search.grades_for(LABELS, "codex")[("q1", "b")] == 2


def test_grade_needs_both_judges_for_min() -> None:
    grades = score_search.grades_for([label("q1:a", "claude", 2)], "min")

    assert grades == {}


def test_ndcg_and_rank_metrics_have_known_values() -> None:
    score = score_search.score_search(HITS, ["q1", "q2"], LABELS, KINDS, ("min",))
    arms = {arm.arm: arm for arm in score.as_called[0].arms}
    v1, v2 = arms["v1"], arms["v2_plain"]
    ideal_q1 = 2 + 1 / math.log2(3) + 1 / 2

    assert v1.ndcg == pytest.approx(((1 / math.log2(3) + 1) / ideal_q1 + 1 / math.log2(3)) / 2)
    assert v2.ndcg == pytest.approx(((2 + 1 / math.log2(3)) / ideal_q1 + 1) / 2)
    assert (v1.hit1.count, v1.hit1.total) == (0, 2)
    assert (v2.hit1.count, v2.hit1.total) == (2, 2)
    assert (v1.hit5.count, v1.hit5.total) == (2, 2)
    assert (v1.precision5.count, v1.precision5.total) == (3, 10)
    assert v1.mrr == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert v2.mrr == pytest.approx(1.0)


def test_versus_baseline_counts_and_interval() -> None:
    score = score_search.score_search(HITS, ["q1", "q2"], LABELS, KINDS, ("min",))
    arms = {arm.arm: arm for arm in score.as_called[0].arms}
    v2 = arms["v2_plain"]

    assert arms["v1"].wins is None
    assert (v2.wins, v2.ties, v2.losses) == (2, 0, 0)
    assert v2.diff_mean == pytest.approx(0.344237, abs=1e-5)
    assert v2.diff_low is not None
    assert v2.diff_high is not None
    assert 0.319 <= v2.diff_low <= v2.diff_high <= 0.370


def test_timing_counts_one_call_per_question() -> None:
    score = score_search.score_search(HITS, ["q1", "q2"], LABELS, KINDS, ("min",))
    arms = {arm.arm: arm for arm in score.as_called[0].arms}

    assert arms["v1"].timed_calls == 2
    assert arms["v1"].median_ms == 20
    assert arms["v2_plain"].p95_ms == 40


def test_unjudged_results_count_as_zero() -> None:
    extra = [*HITS, hit("q1", "v1", 4, "zzz", 10)]
    variant = score_search.score_search(extra, ["q1", "q2"], LABELS, KINDS, ("min",)).as_called[0]

    assert variant.unjudged == 1
    assert variant.graded == 6


DECISION_LABELS = [
    label(item, judge, grade)
    for item, grade in (("q1:x", 2), ("q1:a", 1), ("q1:b", 1))
    for judge in ("claude", "codex")
]
DECISION_KINDS = {"x": "decision", "a": "fact", "b": "fact"}
DECISION_HITS = [
    *(hit("q1", "v1", r, rec, 10) for r, rec in enumerate("xab", start=1)),
    *(hit("q1", "v2_plain", r, rec, 10) for r, rec in enumerate("xa", start=1)),
    *(hit("q1", "v2_plain_facts", r, rec, 10) for r, rec in enumerate("ab", start=1)),
]


def test_facts_only_pool_drops_decisions_from_results_and_ideal() -> None:
    score = score_search.score_search(
        DECISION_HITS, ["q1"], DECISION_LABELS, DECISION_KINDS, ("min",)
    )
    called = {arm.arm: arm for arm in score.as_called[0].arms}
    facts = {arm.arm: arm for arm in score.facts_only[0].arms}

    assert (score.as_called[0].graded, score.facts_only[0].graded) == (3, 2)
    assert (called["v1"].hit1.count, facts["v1"].hit1.count) == (1, 0)
    assert called["v1"].ndcg == pytest.approx(1.0)
    assert facts["v1"].ndcg == pytest.approx(1.0)
    assert facts["v2_plain_facts"].ndcg == pytest.approx(1.0)
    assert called["v2_plain"].ndcg == pytest.approx(
        (2 + 1 / math.log2(3)) / (2 + 1 / math.log2(3) + 1 / 2)
    )
    assert "v2_plain" not in facts
    assert "v2_plain_facts" not in called


def test_decision_at_rank_one_still_counts_as_called() -> None:
    score = score_search.score_search(
        DECISION_HITS, ["q1"], DECISION_LABELS, DECISION_KINDS, ("min",)
    )
    plain = {arm.arm: arm for arm in score.as_called[0].arms}["v2_plain"]

    assert plain.hit1.count == 1
    assert plain.mrr == pytest.approx(1.0)


def test_kind_mix_counts_facts_and_decisions_per_arm() -> None:
    score = score_search.score_search(
        DECISION_HITS, ["q1"], DECISION_LABELS, DECISION_KINDS, ("min",)
    )
    mix = {entry.arm: (entry.facts, entry.decisions) for entry in score.kind_mix}

    assert mix == {"v1": (2, 1), "v2_plain": (1, 1), "v2_plain_facts": (2, 0)}


PINNED_BOOTSTRAP = (0.037233333333333334, 0.28386666666666666)


def test_bootstrap_is_seeded_and_pinned() -> None:
    diffs = [
        0.5173,
        -0.1042,
        0.3391,
        0.0,
        0.2218,
        0.4407,
        -0.2269,
        0.1135,
        0.0766,
        0.2951,
        -0.0518,
        0.3604,
    ]
    first = score_search.bootstrap(diffs)
    again = score_search.bootstrap(diffs)

    assert first == again
    assert score_search.bootstrap(diffs, seed=1) != first
    assert first == pytest.approx(PINNED_BOOTSTRAP)
    assert score_search.bootstrap([1.0] * 5) == (1.0, 1.0)
    assert score_search.bootstrap([]) is None


def fake_data(**overrides: Any) -> report.ReportData:
    v1 = score_save.score_save(V1_ROWS, KEY, ("v1",))
    search = score_search.score_search(HITS, ["q1", "q2"], LABELS, KINDS, ("min",))
    base: dict[str, Any] = {
        "pairs": 10,
        "key_size": 9,
        "key_agreed": 8,
        "key_owner": 1,
        "key_by_label": {"restatement": 4},
        "disputed": 3,
        "waiting_on_owner": 2,
        "unlabelled_by_judge": {"claude": 0, "codex": 2},
        "corpus_rows": 50,
        "mined": 35,
        "generated": 5,
        "agreements": [],
        "save": v1,
        "search": search,
    }
    base.update(overrides)

    return report.ReportData.model_validate(base)


def test_markdown_states_every_rate_with_its_count() -> None:
    text = report.render_markdown(fake_data())

    assert "1 of 4, 25%" in text
    assert "2 of 4, 50%" in text
    assert "35 mined" in text
    assert "5 generated" in text
    assert "2 wait on the owner" in text
    assert "Relation key: 9 of 10 pairs" in text
    assert "stopped and asked the agent" in text
    assert "40 questions is a small set" in text
    assert "snapshot from the pool export" in text
    assert "Degraded run" not in text


def test_markdown_splits_search_into_as_called_and_facts_only() -> None:
    text = report.render_markdown(fake_data())

    assert text.index("### As called") < text.index("### Facts only")
    assert text.index("### Facts only") < text.index("### Time per recall call")
    assert "v1 recall returns facts only by design" in text
    assert "like-for-like retrieval comparison" in text
    assert "The judged pool is the union of all five arms' top 10s, 6 records." in text
    assert "folded with upkeep() after the bulk load" in text


def test_degraded_banner_sits_above_everything() -> None:
    text = report.render_markdown(fake_data(warnings=["v1: 5 of 10 save rows degraded"]))

    assert text.index("Degraded run") < text.index("Answer key")
    assert "v1: 5 of 10 save rows degraded" in text


def test_rate_with_no_rows_reads_na() -> None:
    assert report.fmt_rate(score_save.Rate(count=0, total=0)) == "0 of 0, n/a"
    assert report.fmt_rate(score_save.Rate(count=31, total=48)) == "31 of 48, 65%"


def test_html_is_self_contained_and_escaped() -> None:
    page = report.render_html(fake_data(warnings=["<script>alert(1)</script>"]))

    assert "http://" not in page
    assert "https://" not in page
    assert "<link" not in page
    assert "src=" not in page
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "prefers-color-scheme: dark" in page
    assert 'class="fill"' in page
    assert "<table>" in page


def make_pair(pair_id: str) -> PairRow:
    return PairRow(
        pair_id=pair_id,
        source="golden",
        kind="fact",
        existing="e " + pair_id,
        incoming="i " + pair_id,
        similarity=None,
        old_label=None,
    )


def make_context(tmp_path: pathlib.Path, calls: list[list[str]], **kwargs: Any) -> run.Context:
    def runner(command: Any, cwd: pathlib.Path, env: dict[str, str]) -> int:
        calls.append(list(command))
        return 0

    return run.Context(base=tmp_path, runner=runner, log=lambda text: None, **kwargs)


def test_complete_stage_is_skipped_and_incomplete_runs_with_limit(tmp_path: pathlib.Path) -> None:
    calls: list[list[str]] = []
    ctx = make_context(tmp_path, calls, limit=2)
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a"), make_pair("b"), make_pair("c")])

    ran = run.run_pipeline(ctx, "save-v1")

    assert ran == ["save-v1"]
    assert calls[0][-2:] == ["--limit", "2"]

    write_jsonl(
        tmp_path / "save-v1.jsonl",
        [
            save_row("a", "v1", "new", found=False, pick=None, ms=1),
            save_row("b", "v1", "new", found=False, pick=None, ms=1),
        ],
    )

    assert run.run_pipeline(ctx, "save-v1") == []
    assert len(calls) == 1


def test_force_reruns_a_complete_stage(tmp_path: pathlib.Path) -> None:
    calls: list[list[str]] = []
    ctx = make_context(tmp_path, calls, force=True)
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a")])
    write_jsonl(
        tmp_path / "save-v2.jsonl", [save_row("a", "v2", "new", found=False, pick=None, ms=1)]
    )

    assert run.run_pipeline(ctx, "save-v2") == ["save-v2"]
    assert calls[0][:2] == ["node", "eval/save.ts"]


def corpus_row(record_id: str, content: str) -> CorpusRow:
    return CorpusRow(
        id=record_id,
        kind="fact",
        content=content,
        scope="global",
        lifecycle="live",
        vector=VECTOR,
        recall_count=0,
        created_at="2026-01-01T00:00:00Z",
    )


def test_relevance_items_are_the_deduplicated_union_of_top_ten(tmp_path: pathlib.Path) -> None:
    ctx = make_context(tmp_path, [])
    write_jsonl(tmp_path / "corpus.jsonl", [corpus_row(r, f"text {r}") for r in ("a", "b", "c")])
    write_jsonl(
        tmp_path / "questions.jsonl",
        [QuestionRow(question_id="q1", query="what?", scope="global", origin="mined")],
    )
    write_jsonl(
        tmp_path / "search-v1.jsonl", [hit("q1", "v1", 1, "a", 1), hit("q1", "v1", 2, "b", 1)]
    )
    write_jsonl(
        tmp_path / "search-v2_plain.jsonl",
        [
            hit("q1", "v2_plain", 1, "b", 1),
            hit("q1", "v2_plain", 2, "c", 1),
            hit("q1", "v2_plain", 11, "a", 1),
        ],
    )

    items = run.relevance_items(ctx)

    assert [item.item_id for item in items] == ["q1:a", "q1:b", "q1:c"]
    assert items[0].fields == {"question": "what?", "record": "text a"}


def seed_degraded(tmp_path: pathlib.Path, degraded: int, total: int) -> None:
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair(f"p{i}") for i in range(total)])
    write_jsonl(
        tmp_path / "save-v1.jsonl",
        [
            save_row(
                f"p{i}",
                "v1",
                "new",
                found=False,
                pick=None,
                ms=1,
                degraded="x" if i < degraded else None,
            )
            for i in range(total)
        ],
    )
    labels = [label(f"p{i}", j, 0) for i in range(total) for j in ("claude", "codex")]
    write_jsonl(
        tmp_path / "labels-relation.jsonl",
        [row.model_copy(update={"label": "distinct"}) for row in labels],
    )


def test_degraded_run_refuses_the_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_degraded(tmp_path, degraded=2, total=20)
    written: list[tuple[list[str]]] = []
    monkeypatch.setattr(
        report, "write_report", lambda base, warnings, limit: written.append((list(warnings),))
    )
    ctx = make_context(tmp_path, [])

    with pytest.raises(run.StageFailed, match="2 of 20 save rows degraded"):
        run.write_report(ctx)

    assert written == []


def test_allow_degraded_writes_with_the_warning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_degraded(tmp_path, degraded=2, total=20)
    written: list[list[str]] = []
    monkeypatch.setattr(
        report, "write_report", lambda base, warnings, limit: written.append(list(warnings))
    )
    ctx = make_context(tmp_path, [], allow_degraded=True)

    run.write_report(ctx)

    assert len(written) == 1
    assert "2 of 20 save rows degraded" in written[0][0]


def test_five_percent_degraded_is_accepted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_degraded(tmp_path, degraded=1, total=20)
    written: list[list[str]] = []
    monkeypatch.setattr(
        report, "write_report", lambda base, warnings, limit: written.append(list(warnings))
    )

    run.write_report(make_context(tmp_path, []))

    assert written == [[]]


def test_unlabelled_judge_items_refuse_the_report(tmp_path: pathlib.Path) -> None:
    seed_degraded(tmp_path, degraded=0, total=20)
    kept = [
        row
        for row in run._load(tmp_path / "labels-relation.jsonl", LabelRow)
        if not (row.judge == "codex" and row.item_id in {"p0", "p1"})
    ]
    write_jsonl(tmp_path / "labels-relation.jsonl", kept)

    with pytest.raises(run.StageFailed, match="codex left 2 of 20 relation items unlabelled"):
        run.write_report(make_context(tmp_path, []))


def aimed_row(pair_id: str, system: str, bucket: str, raw: str) -> SaveRow:
    row = save_row(pair_id, system, bucket, found=False, pick=None, ms=1)

    return row.model_copy(update={"raw_outcome": raw})


def label_rows(item_id: str, claude: str, codex: str) -> list[LabelRow]:
    return [
        LabelRow.model_validate(
            {"item_id": item_id, "judge": judge, "model": "m", "label": label, "reason": "r"}
        )
        for judge, label in (("claude", claude), ("codex", codex))
    ]


def test_target_id_parses_old_and_new_merge_forms() -> None:
    assert targets.target_id_of("reinforced:n1") == "n1"
    assert targets.target_id_of("already_saved:n2") == "n2"
    assert targets.target_id_of("reinforced") is None
    assert targets.target_id_of("reinforced:reinforce:n3") == "n3"
    assert targets.target_id_of("bounced:near_dup:n4") == "n4"


def test_pointed_actions_follow_the_raw_outcome() -> None:
    rows = [
        aimed_row("a", "v1", "wrong_target", "bounced:near_dup:n1"),
        aimed_row("b", "v2", "wrong_target", "reinforced:reinforce:n2"),
        aimed_row("c", "v2", "wrong_target", "approved:supersede:n3"),
        aimed_row("d", "v1", "wrong_target", "reinforced"),
        aimed_row("e", "v1", "new", "saved"),
    ]

    pointed = targets.pointed_rows(rows)

    assert [(p.pair_id, p.action, p.target_id) for p in pointed] == [
        ("a", "flagged", "n1"),
        ("b", "merged", "n2"),
        ("c", "flagged", "n3"),
        ("d", "merged", None),
    ]
    with pytest.raises(ValueError):
        targets.action_of("mystery:x:y")


def test_identical_target_and_incoming_text_is_judged_once_and_fans_out() -> None:
    pointed = [
        targets.Pointed(system="v1", pair_id="a", action="flagged", target_id="n1"),
        targets.Pointed(system="v2", pair_id="a", action="merged", target_id="n2"),
        targets.Pointed(system="v2", pair_id="b", action="merged", target_id="n1"),
        targets.Pointed(system="v1", pair_id="c", action="merged", target_id="gone"),
    ]
    texts = {"n1": "same text", "n2": "same text"}
    incoming = {"a": "in a", "b": "in b", "c": "in c"}

    items, canonical = targets.build_items(pointed, texts, incoming)

    assert [item.item_id for item in items] == ["v1:a", "v2:b"]
    assert items[0].fields == {"existing": "same text", "incoming": "in a"}
    assert canonical == {"v1:a": "v1:a", "v2:a": "v1:a", "v2:b": "v2:b"}

    labels = label_rows("v1:a", "restatement", "restatement") + label_rows(
        "v2:b", "follow_up", "distinct"
    )
    done = {
        f"{p.system}:{p.pair_id}": p.category
        for p in targets.categorise(pointed, canonical, labels)
    }

    assert done == {
        "v1:a": "restatement",
        "v2:a": "restatement",
        "v2:b": "judges split",
        "v1:c": "unresolved",
    }


def pointed_fixture() -> tuple[list[SaveRow], list[targets.Pointed]]:
    rows = [
        aimed_row("p1", "v2", "wrong_target", "reinforced:reinforce:t1"),
        aimed_row("p2", "v2", "wrong_target", "reinforced:reinforce:t2"),
        aimed_row("p3", "v2", "wrong_target", "reinforced:reinforce:t3"),
        aimed_row("p4", "v2", "wrong_target", "approved:supersede:t4"),
        aimed_row("p5", "v2", "wrong_target", "approved:supersede:t5"),
        aimed_row("p6", "v2", "wrong_target", "approved:supersede:t6"),
        aimed_row("p11", "v2", "wrong_target", "approved:supersede:t6"),
        aimed_row("p7", "v2", "wrong_target", "reinforced:reinforce:t7"),
        aimed_row("p8", "v2", "wrong_target", "reinforced:reinforce:missing"),
        aimed_row("p9", "v2", "caught", "reinforced:reinforce:own"),
        aimed_row("p10", "v2", "flagged", "approved:supersede:own"),
    ]
    texts = {f"t{n}": f"target {n}" for n in range(1, 8)}
    incoming = {f"p{n}": f"incoming {n}" for n in range(1, 12)}
    pointed = targets.pointed_rows(rows)
    items, canonical = targets.build_items(pointed, texts, incoming)
    relation = {
        "t1": "restatement",
        "t2": "replacement",
        "t3": "follow_up",
        "t4": "restatement",
        "t5": "replacement",
        "t6": "distinct",
    }
    labels: list[LabelRow] = []

    for item in items:
        agreed = relation.get(item.fields["existing"].replace("target ", "t"))
        pair = (agreed, agreed) if agreed is not None else ("distinct", "follow_up")
        labels.extend(label_rows(item.item_id, *pair))

    return rows, targets.categorise(pointed, canonical, labels)


def test_pointed_table_counts_each_action_and_relation_cell() -> None:
    rows, pointed = pointed_fixture()
    key = {
        "p3": "follow_up",
        "p6": "distinct",
        "p1": "restatement",
        "p9": "distinct",
        "p10": "distinct",
        "p11": "distinct",
    }

    v2 = score_save.score_save(rows, key, ("v2",), pointed).systems[0]
    cells = v2.pointed.cells

    assert cells["merged"] == {
        "restatement": 1,
        "replacement": 1,
        "follow_up": 1,
        "distinct": 0,
        "judges split": 1,
        "unresolved": 1,
    }
    assert cells["flagged"] == {
        "restatement": 1,
        "replacement": 1,
        "follow_up": 0,
        "distinct": 2,
        "judges split": 0,
        "unresolved": 0,
    }
    assert v2.pointed.total == 9
    assert v2.pointed.merged_right == 1
    assert v2.pointed.merged_lost_record == 1
    assert v2.pointed.merged_lost_update == 1
    assert v2.pointed.flagged_right == 2
    assert v2.pointed.flagged_false_alarm == 2
    assert v2.pointed.unresolved == 1
    assert (v2.lost_record.count, v2.lost_record_any.count, v2.lost_record_any.total) == (1, 2, 6)
    assert (v2.false_alarm.count, v2.false_alarm_any.count) == (1, 3)


def test_target_labels_stay_out_of_the_answer_key(tmp_path: pathlib.Path) -> None:
    ctx = make_context(tmp_path, [])
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a")])
    write_jsonl(
        tmp_path / "save-v2.jsonl",
        [aimed_row("a", "v2", "wrong_target", "reinforced:reinforce:n1")],
    )
    noise = NoiseRow(
        id="n1",
        kind="fact",
        content="noise text",
        vector=VECTOR,
        recall_count=0,
        created_at="2026-01-01T00:00:00Z",
        lifecycle="live",
    )
    write_jsonl(tmp_path / "noise.jsonl", [noise])

    _pointed, items, _canonical = targets.items_for(tmp_path)

    assert [item.item_id for item in items] == ["v2:a"]
    assert ctx.path(targets.TARGETS_DIR) == tmp_path / "targets"
    assert "judge-targets" in run.STAGES
    assert run.STAGES.index("judge-targets") == run.STAGES.index("save-v2") + 1
    assert not (tmp_path / "labels-relation.jsonl").exists()
    assert not (tmp_path / "key-relation.jsonl").exists()


def test_judge_targets_stage_writes_only_under_targets(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake(name: str, items: Any, **kwargs: Any) -> None:
        seen.update(name=name, ids=[i.item_id for i in items], out=kwargs["out_dir"])

    monkeypatch.setattr("eval_v1v2.judges.run_task", fake)
    ctx = make_context(tmp_path, [])
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a")])
    write_jsonl(
        tmp_path / "save-v1.jsonl", [aimed_row("a", "v1", "wrong_target", "bounced:near_dup:n1")]
    )
    noise = NoiseRow(
        id="n1",
        kind="fact",
        content="noise text",
        vector=VECTOR,
        recall_count=0,
        created_at="2026-01-01T00:00:00Z",
        lifecycle="live",
    )
    write_jsonl(tmp_path / "noise.jsonl", [noise])

    run.run_pipeline(ctx, "judge-targets")

    assert seen == {"name": "relation", "ids": ["v1:a"], "out": tmp_path / "targets"}


def test_markdown_carries_pointed_section_and_new_rows() -> None:
    rows, pointed = pointed_fixture()
    key = {"p3": "follow_up", "p6": "distinct"}
    v2 = score_save.score_save(rows, key, ("v2",), pointed)
    text = report.render_markdown(fake_data(save=v2))

    assert "Pointed at another record" in text
    assert "Lost records, any target" in text
    assert "False alarms, any target" in text
    assert "Existing record raised as a candidate" in text
    assert "Found the existing record" not in text
    assert "0.80 for facts, 0.70 for decisions" in text
    assert "1 merges onto a record both judges call a restatement" in text


SOURCES = {
    "p1": "jev_pairs",
    "p2": "jev_pairs",
    "p3": "bounce_log",
    "p4": "bounce_log",
    "p5": "bounce_log",
    "p6": "golden",
    "p7": "bounce_log",
    "p8": "golden",
    "p9": "admission",
}


def test_by_source_counts_each_source_with_known_answers() -> None:
    score = score_save.score_save(V1_ROWS, KEY, ("v1",), sources=SOURCES)
    rows = {row.source: row for row in score.systems[0].by_source}

    assert [row.source for row in score.systems[0].by_source] == [
        "jev_pairs",
        "bounce_log",
        "golden",
        "admission",
    ]
    assert rows["jev_pairs"].scored == 2
    assert rows["jev_pairs"].restatement_noticed == score_save.Rate(count=2, total=2)
    assert rows["bounce_log"].scored == 4
    assert rows["bounce_log"].restatement_noticed == score_save.Rate(count=0, total=2)
    assert rows["bounce_log"].replacement_flagged == score_save.Rate(count=1, total=1)
    assert rows["bounce_log"].false_alarm == score_save.Rate(count=0, total=1)
    assert rows["golden"].scored == 2
    assert rows["golden"].false_alarm == score_save.Rate(count=1, total=1)
    assert rows["golden"].replacement_flagged == score_save.Rate(count=0, total=1)
    assert rows["admission"].scored == 1


def test_limits_carry_selection_agreement_and_coarseness_numbers() -> None:
    relevance = report.AgreementSummary(
        task="relevance", both=845, agreed=588, raw=0.696, kappa=0.355
    )
    save = score_save.score_save(V1_ROWS, KEY, ("v1",), sources=SOURCES)
    text = report.render_markdown(fake_data(bounce_pairs=4, agreements=[relevance], save=save))
    limits = text.split("## Limits")[1]

    assert "4 of the 10 pairs come from the bounce log" in limits
    assert "raw agreement 0.696, Cohen's kappa 0.355" in limits
    assert "Compare arms within one table" in limits
    assert "of 40 in the as called table" in limits
    assert "too coarse to rank the arms" in limits
    assert "Save results by pair source" in text
    assert "| v1 | bounce_log | 4 | 0 of 2, 0% | 1 of 1, 100% | 0 of 1, 0% |" in text


def test_collect_counts_unlabelled_per_judge_and_splits_v1_time(tmp_path: pathlib.Path) -> None:
    data = fake_data()
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair(f"p{i}") for i in range(4)])
    write_jsonl(
        tmp_path / "labels-relation.jsonl",
        [
            *(label(f"p{i}", "claude", 0) for i in range(4)),
            *(label(f"p{i}", "codex", 0) for i in range(2)),
            label("p0", "codex", 0),
        ],
    )
    cheap = save_row("p0", "v1", "new", found=False, pick=None, ms=1)
    cold = save_row("p1", "v1", "new", found=False, pick=None, ms=500)
    plain = save_row("p2", "v1", "new", found=False, pick=None, ms=80)
    judged = save_row("p3", "v1", "flagged", found=True, pick="restatement", ms=300)
    bounced = save_row("p4", "v1", "flagged", found=True, pick=None, ms=200)
    bounced = bounced.model_copy(update={"raw_outcome": "bounced:near_dup:x"})
    write_jsonl(tmp_path / "save-v1.jsonl", [cold, plain, judged, bounced, cheap])
    write_jsonl(tmp_path / "questions.jsonl", [])
    (tmp_path / "corpus.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "score-save.json").write_text(data.save.model_dump_json(), encoding="utf-8")
    (tmp_path / "score-search.json").write_text(data.search.model_dump_json(), encoding="utf-8")

    collected = report.collect(tmp_path)

    assert collected.unlabelled_by_judge == {"claude": 0, "codex": 2}
    assert (collected.v1_plain_calls, collected.v1_plain_median_ms) == (2, 40.5)
    assert (collected.v1_assessor_calls, collected.v1_assessor_median_ms) == (2, 250.0)
    assert "left unlabelled: claude 0, codex 2." in report.render_markdown(collected)
    assert "2 saves without an Assessor call have a median of 40.5 ms" in report.render_markdown(
        collected
    )


def test_question_with_no_rows_still_counts_and_scores_zero(tmp_path: pathlib.Path) -> None:
    write_jsonl(
        tmp_path / "questions.jsonl",
        [
            QuestionRow(question_id=q, query=q, scope="global", origin="mined")
            for q in ("q1", "q2", "q3")
        ],
    )
    write_jsonl(tmp_path / "corpus.jsonl", [corpus_row(r, r) for r in "abcdef"])
    write_jsonl(tmp_path / "labels-relevance.jsonl", LABELS)
    write_jsonl(tmp_path / "search-v1.jsonl", [row for row in HITS if row.arm == "v1"])
    write_jsonl(tmp_path / "search-v2_plain.jsonl", [row for row in HITS if row.arm == "v2_plain"])

    score = score_search.score_files(tmp_path)
    arms = {arm.arm: arm for arm in score.as_called[0].arms}

    assert arms["v1"].questions == 3
    assert arms["v1"].answered == 2
    assert arms["v1"].hit1.total == 3
    assert arms["v1"].precision5.total == 15
    assert "Questions with no results, scored 0: v1 1, v2_plain 1." in report.render_markdown(
        fake_data(search=score)
    )


def make_noise() -> NoiseRow:
    return NoiseRow(
        id="n9",
        kind="fact",
        content="extra noise",
        vector=VECTOR,
        recall_count=0,
        created_at="2026-01-01T00:00:00Z",
        lifecycle="live",
    )


def test_first_run_adopts_existing_outputs_then_a_changed_pair_text_blocks_the_stage(
    tmp_path: pathlib.Path,
) -> None:
    calls: list[list[str]] = []
    ctx = make_context(tmp_path, calls)
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a")])
    write_jsonl(tmp_path / "noise.jsonl", [])
    write_jsonl(
        tmp_path / "save-v1.jsonl", [save_row("a", "v1", "new", found=False, pick=None, ms=1)]
    )

    assert run.run_pipeline(ctx, "save-v1") == []
    assert (tmp_path / "inputs.json").exists()

    changed = make_pair("a").model_copy(update={"incoming": "different text"})
    write_jsonl(tmp_path / "pairs.jsonl", [changed])

    with pytest.raises(run.StageFailed, match=r"save-v1\.jsonl"):
        run.run_pipeline(ctx, "save-v1")

    assert calls == []

    with pytest.raises(run.StageFailed, match=r"save-v1\.jsonl"):
        run.run_pipeline(ctx, "score-save")


def test_stage_run_records_the_input_hashes_it_used(tmp_path: pathlib.Path) -> None:
    ctx = make_context(tmp_path, [])
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a")])
    write_jsonl(tmp_path / "noise.jsonl", [])

    def runner(command: Any, cwd: pathlib.Path, env: dict[str, str]) -> int:
        write_jsonl(
            tmp_path / "save-v1.jsonl", [save_row("a", "v1", "new", found=False, pick=None, ms=1)]
        )
        return 0

    ctx.runner = runner

    assert run.run_pipeline(ctx, "save-v1") == ["save-v1"]
    assert fingerprint.stale(tmp_path, ["save-v1.jsonl"]) == []

    write_jsonl(tmp_path / "noise.jsonl", [make_noise()])

    assert fingerprint.stale(tmp_path, ["save-v1.jsonl"]) == ["save-v1.jsonl"]


def test_resuming_a_partial_save_file_warns_about_cold_calls(tmp_path: pathlib.Path) -> None:
    logged: list[str] = []
    ctx = make_context(tmp_path, [])
    ctx.log = logged.append
    write_jsonl(tmp_path / "pairs.jsonl", [make_pair("a"), make_pair("b")])
    write_jsonl(
        tmp_path / "save-v2.jsonl", [save_row("a", "v2", "new", found=False, pick=None, ms=1)]
    )

    run.run_pipeline(ctx, "save-v2")

    assert any("include cold calls" in line for line in logged)
