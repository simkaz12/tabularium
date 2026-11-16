"""Tests for scripts/eval_v1v2/pairs.py.

Falsifiability: matching ``startswith`` on the log field instead of the ledger text in
``resolve_bounce_row`` turned ``test_bounce_resolves_full_text_from_ledger`` red; removing the
distinct-text check turned ``test_bounce_drops_ambiguous_candidate_prefix`` red; raising the
``check_no_cut_texts`` share to 1.0 turned ``test_cut_guard_fails_on_repeated_length`` red.
Earlier:  replacing ``random.Random(seed)`` in ``bounce_pairs`` with an unseeded
``random.Random()`` turned ``test_bounce_sample_is_seeded`` red; slicing the quartiles by
``total * quartile // 4`` against a similarity-unsorted list turned ``test_bounce_quartile_counts``
red; dropping the ``excluded_texts`` filter in ``build_noise`` turned
``test_noise_excludes_pair_text`` red; dropping the ``EXCLUDED_SCOPES`` filter turned
``test_noise_filters_scope_validity_lifecycle`` red. All reverted.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter
from typing import Any

import pytest

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import pairs as pairs_mod  # noqa: E402
from eval_v1v2.rows import PairRow  # noqa: E402

sys.path.pop(0)


def _bounce_file(
    tmp_path: pathlib.Path,
    count: int = 80,
    scopes: list[str] | None = None,
) -> pathlib.Path:
    rows = []
    ledger = []

    for index in range(count):
        scope = scopes[index % len(scopes)] if scopes else "project/tabularium"
        kind, scope_id = scope.split("/")
        rows.append(
            {
                "new_content": f"new {index:04d}",
                "new_pi_id": index,
                "new_scope": scope,
                "candidate_content": f"candidate {index:04d}",
                "similarity": str(round(0.5 + (index * 37 % count) / (count * 2), 4)),
            }
        )
        ledger.append(_atom(index, f"new {index:04d} in full", scope_kind=kind, scope_id=scope_id))
        ledger.append(
            _atom(
                10_000 + index,
                f"candidate {index:04d} in full",
                scope_kind=kind,
                scope_id=scope_id,
            )
        )

    path = tmp_path / "bounce.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    return path


def _atom(atom_id: int, content: str, **overrides: Any) -> dict[str, Any]:
    atom: dict[str, Any] = {
        "id": atom_id,
        "content": content,
        "kind": "fact",
        "scope_kind": "project",
        "scope_id": "tabularium",
    }
    atom.update(overrides)

    return atom


def _ledger(path: pathlib.Path) -> pathlib.Path:
    return path.parent / "ledger.json"


def _pool_record(index: int, **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"id-{index:04d}",
        "type": "fact",
        "content": f"record {index}",
        "scope": "tabularium",
        "validity": "valid",
        "lifecycle": "active",
        "recall_count": 2,
        "created_at": "2026-01-01T00:00:00+00:00",
        "vector": [0.0] * 768,
    }
    record.update(overrides)

    return record


def test_bounce_sample_is_seeded(tmp_path: pathlib.Path) -> None:
    path = _bounce_file(tmp_path)

    first = pairs_mod.bounce_pairs(path, seed=7, per_quartile=5, ledger_path=_ledger(path))
    second = pairs_mod.bounce_pairs(path, seed=7, per_quartile=5, ledger_path=_ledger(path))
    other = pairs_mod.bounce_pairs(path, seed=8, per_quartile=5, ledger_path=_ledger(path))

    assert [row.pair_id for row in first] == [row.pair_id for row in second]
    assert [row.pair_id for row in first] != [row.pair_id for row in other]


def test_bounce_quartile_counts(tmp_path: pathlib.Path) -> None:
    path = _bounce_file(tmp_path, count=80)

    rows = pairs_mod.bounce_pairs(path, seed=1, per_quartile=5, ledger_path=_ledger(path))
    similarities = sorted(
        float(json.loads(path.read_text())[int(row.pair_id.split("-")[1])]["similarity"])
        for row in rows
    )
    all_sims = sorted(float(item["similarity"]) for item in json.loads(path.read_text()))
    edges = [all_sims[0], all_sims[20], all_sims[40], all_sims[60], all_sims[-1] + 1]
    per_bucket = [
        sum(1 for value in similarities if edges[bucket] <= value < edges[bucket + 1])
        for bucket in range(4)
    ]

    assert len(rows) == 20
    assert per_bucket == [5, 5, 5, 5]
    assert all(row.source == "bounce_log" and row.kind == "fact" for row in rows)
    assert all(row.old_label is None for row in rows)


def test_bounce_maps_candidate_to_existing(tmp_path: pathlib.Path) -> None:
    path = _bounce_file(tmp_path, count=8)
    rows = pairs_mod.bounce_pairs(path, seed=1, per_quartile=2, ledger_path=_ledger(path))

    assert len(rows) == 8
    assert all(row.existing.startswith("candidate") for row in rows)
    assert all(row.incoming.startswith("new") for row in rows)
    assert all(isinstance(row.similarity, float) for row in rows)


def test_jev_pairs_keep_label_and_kind(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "jev.json"
    path.write_text(
        json.dumps(
            [
                {
                    "pair_id": "a000",
                    "similarity": 0.7,
                    "existing_content": "old",
                    "incoming_content": "new",
                    "incoming_type": "decision",
                    "store_label": "replacement",
                },
                {
                    "pair_id": "a001",
                    "similarity": 0.6,
                    "existing_content": "old2",
                    "incoming_content": "new2",
                    "incoming_type": "fact",
                    "store_label": None,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = pairs_mod.jev_pairs(path)

    assert [row.kind for row in rows] == ["decision", "fact"]
    assert [row.old_label for row in rows] == ["replacement", None]
    assert rows[0].pair_id == "jev-a000"


def _jev_item(pair_id: str, existing_scope: str, incoming_scope: str) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "similarity": 0.7,
        "existing_content": f"old {pair_id}",
        "incoming_content": f"new {pair_id}",
        "incoming_type": "fact",
        "store_label": None,
        "existing_scope": existing_scope,
        "incoming_scope": incoming_scope,
    }


@pytest.mark.parametrize("scope", ["project-a", "project-i", "project-f", "project-k"])
def test_jev_drops_excluded_scope_on_either_side(tmp_path: pathlib.Path, scope: str) -> None:
    path = tmp_path / "jev.json"
    path.write_text(
        json.dumps(
            [
                _jev_item("keep", "tabularium", "global"),
                _jev_item("left", scope, "global"),
                _jev_item("right", "global", scope),
                _jev_item("both", scope, scope),
            ]
        ),
        encoding="utf-8",
    )

    assert [row.pair_id for row in pairs_mod.jev_pairs(path)] == ["jev-keep"]


def test_bounce_drops_client_repo_rows_before_sampling(tmp_path: pathlib.Path) -> None:
    scopes = ["project/project-a-api", "project/project-a-web"]
    scopes += ["project/project-i-app", "project/project-f-web"]
    scopes += ["project/project-k", "project/project-b", "user/1"]
    path = _bounce_file(tmp_path, count=28, scopes=scopes)

    picked = pairs_mod.bounce_pairs(path, seed=1, per_quartile=100, ledger_path=_ledger(path))
    kept = {int(row.pair_id.split("-")[1]) for row in picked}

    assert kept == {index for index in range(28) if index % len(scopes) in (5, 6)}


def _single_row(tmp_path: pathlib.Path, ledger: list[dict[str, Any]], **overrides: Any) -> Any:
    row = {
        "new_content": "the incoming text begins",
        "new_pi_id": 1,
        "new_scope": "project/tabularium",
        "candidate_content": "the existing text begins",
        "similarity": 0.9,
    }
    row.update(overrides)
    path = tmp_path / "bounce.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    drops: Counter[str] = Counter()
    rows = pairs_mod.bounce_pairs(
        path, seed=1, per_quartile=5, ledger_path=ledger_path, drops=drops
    )

    return rows, drops


def _full_ledger() -> list[dict[str, Any]]:
    return [
        _atom(1, "the incoming text begins and goes on well past the log field", kind="decision"),
        _atom(2, "the existing text begins and also goes on past the log field"),
    ]


def test_bounce_resolves_full_text_from_ledger(tmp_path: pathlib.Path) -> None:
    rows, drops = _single_row(tmp_path, _full_ledger())

    assert len(rows) == 1
    assert rows[0].incoming == "the incoming text begins and goes on well past the log field"
    assert rows[0].existing == "the existing text begins and also goes on past the log field"
    assert rows[0].kind == "decision"
    assert not drops


def test_bounce_kind_falls_back_to_fact(tmp_path: pathlib.Path) -> None:
    ledger = _full_ledger()
    ledger[0]["kind"] = "archived"

    rows, _ = _single_row(tmp_path, ledger)

    assert rows[0].kind == "fact"


def test_bounce_drops_ambiguous_candidate_prefix(tmp_path: pathlib.Path) -> None:
    ledger = [*_full_ledger(), _atom(3, "the existing text begins but says something else")]

    rows, drops = _single_row(tmp_path, ledger)

    assert rows == []
    assert drops == {"candidate_ambiguous": 1}


def test_bounce_same_candidate_text_twice_is_not_ambiguous(tmp_path: pathlib.Path) -> None:
    ledger = [
        *_full_ledger(),
        _atom(3, "the existing text begins and also goes on past the log field"),
    ]

    rows, drops = _single_row(tmp_path, ledger)

    assert len(rows) == 1
    assert not drops


def test_bounce_candidate_is_resolved_in_the_same_scope(tmp_path: pathlib.Path) -> None:
    ledger = [*_full_ledger(), _atom(3, "the existing text begins elsewhere", scope_id="other")]

    rows, drops = _single_row(tmp_path, ledger)

    assert len(rows) == 1
    assert not drops


def test_bounce_drops_missing_new_pi_id(tmp_path: pathlib.Path) -> None:
    rows, drops = _single_row(tmp_path, _full_ledger(), new_pi_id=99)

    assert rows == []
    assert drops == {"missing_new_pi_id": 1}


def test_bounce_drops_incoming_prefix_mismatch(tmp_path: pathlib.Path) -> None:
    rows, drops = _single_row(tmp_path, _full_ledger(), new_content="something else entirely")

    assert rows == []
    assert drops == {"incoming_prefix_mismatch": 1}


def test_bounce_drops_unknown_candidate(tmp_path: pathlib.Path) -> None:
    rows, drops = _single_row(tmp_path, _full_ledger(), candidate_content="never saved")

    assert rows == []
    assert drops == {"candidate_not_found": 1}


def test_full_incoming_text_is_excluded_from_noise(tmp_path: pathlib.Path) -> None:
    rows, _ = _single_row(tmp_path, _full_ledger())
    full = rows[0].incoming
    pool = [_pool_record(index) for index in range(5)]
    pool.append(_pool_record(9, content=full))
    pool.append(_pool_record(10, content=full[:20]))

    noise = pairs_mod.build_noise(pool, pairs_mod.pair_texts(rows), seed=1, count=100)

    assert full not in {row.content for row in noise}
    assert full[:20] in {row.content for row in noise}


def _guard_row(index: int, existing: str, incoming: str, source: str = "bounce_log") -> PairRow:
    return PairRow(
        pair_id=f"g-{index}",
        source=source,  # type: ignore[arg-type]
        kind="fact",
        existing=existing,
        incoming=incoming,
        similarity=None,
        old_label=None,
    )


def test_cut_guard_fails_on_repeated_length() -> None:
    rows = [
        _guard_row(index, f"{index:03d}" + "e" * 197, f"{index:03d}" + "i" * 197)
        for index in range(40)
    ]

    with pytest.raises(ValueError, match="bounce_log"):
        pairs_mod.check_no_cut_texts(rows)


def test_cut_guard_passes_on_varied_lengths() -> None:
    rows = [_guard_row(index, "e" * (150 + index), "i" * (250 + index)) for index in range(40)]

    pairs_mod.check_no_cut_texts(rows)


def test_cut_guard_judges_each_source_alone() -> None:
    cut = [
        _guard_row(index, f"{index:03d}" + "e" * 197, "i" * (210 + index)) for index in range(40)
    ]
    varied = [
        _guard_row(100 + index, "e" * (150 + index), "i" * (250 + index), "jev_pairs")
        for index in range(40)
    ]

    with pytest.raises(ValueError, match="bounce_log"):
        pairs_mod.check_no_cut_texts([*cut, *varied])


def test_golden_pairs_are_forty() -> None:
    rows = pairs_mod.golden_pairs()

    assert len(rows) == 40
    assert {row.old_label for row in rows} == {"same", "distinct_near", "unrelated"}
    assert all(row.similarity is None for row in rows)


def _case(directory: pathlib.Path, name: str, text: str) -> None:
    (directory / f"{name}.yaml").write_text(text, encoding="utf-8")


def test_admission_filters_to_single_existing_content_relations(tmp_path: pathlib.Path) -> None:
    keep = """
id: keep
class: admission-auto-restatement
existing:
  - {alias: e1, text: "old text", kind: fact}
incoming: {content: "new text", scope: p, kind: fact}
expect: {material: repetition}
"""
    two_existing = keep.replace("id: keep", "id: two").replace(
        "existing:\n", 'existing:\n  - {alias: e0, text: "x", kind: fact}\n'
    )
    wrong_class = keep.replace("id: keep", "id: span").replace(
        "admission-auto-restatement", "admission-probe-span"
    )
    falsity = """
id: fals
class: admission-falsity
existing:
  - {alias: e1, text: "old text", kind: fact}
incoming: {content: "new text", scope: p, kind: fact}
falsity: {assessor: correction}
"""
    _case(tmp_path, "keep", keep)
    _case(tmp_path, "two", two_existing)
    _case(tmp_path, "span", wrong_class)
    _case(tmp_path, "fals", falsity)

    rows = pairs_mod.admission_pairs(tmp_path)

    assert [row.pair_id for row in rows] == ["adm-keep"]
    assert rows[0].old_label == "repetition"
    assert rows[0].existing == "old text"
    assert rows[0].incoming == "new text"


def test_real_admission_cases_load() -> None:
    rows = pairs_mod.admission_pairs()

    assert rows
    assert all(row.source == "admission" for row in rows)


def test_dedupe_drops_exact_duplicate_pairs() -> None:
    def make(pair_id: str, existing: str, incoming: str) -> PairRow:
        return PairRow(
            pair_id=pair_id,
            source="golden",
            kind="fact",
            existing=existing,
            incoming=incoming,
            similarity=None,
            old_label=None,
        )

    rows = pairs_mod.dedupe(
        [make("a", "x", "y"), make("b", "x", "y"), make("c", "y", "x"), make("d", "x", "z")]
    )

    assert [row.pair_id for row in rows] == ["a", "c", "d"]


def test_noise_is_seeded_and_sized() -> None:
    pool = [_pool_record(index) for index in range(50)]

    first = pairs_mod.build_noise(pool, set(), seed=3, count=10)
    second = pairs_mod.build_noise(pool, set(), seed=3, count=10)
    other = pairs_mod.build_noise(pool, set(), seed=4, count=10)

    assert len(first) == 10
    assert [row.id for row in first] == [row.id for row in second]
    assert [row.id for row in first] != [row.id for row in other]
    assert all(len(row.vector) == 768 for row in first)


def test_noise_excludes_pair_text() -> None:
    pool = [_pool_record(index) for index in range(20)]

    rows = pairs_mod.build_noise(pool, {"record 3", "record 4"}, seed=1, count=100)

    assert len(rows) == 18
    assert all(row.content not in {"record 3", "record 4"} for row in rows)


def test_noise_filters_scope_validity_lifecycle() -> None:
    pool = [
        _pool_record(0),
        _pool_record(1, scope="project-a"),
        _pool_record(2, scope="project-i"),
        _pool_record(6, scope="project-f"),
        _pool_record(7, scope="project-k"),
        _pool_record(3, validity="invalid"),
        _pool_record(4, lifecycle="superseded"),
        _pool_record(5, type="decision", lifecycle="in_force"),
    ]

    rows = pairs_mod.build_noise(pool, set(), seed=1, count=100)

    assert sorted(row.id for row in rows) == ["id-0000", "id-0005"]
    assert {row.kind for row in rows} == {"fact", "decision"}


def test_pair_texts_strip() -> None:
    row = PairRow(
        pair_id="p",
        source="golden",
        kind="fact",
        existing=" a ",
        incoming="b\n",
        similarity=None,
        old_label=None,
    )

    assert pairs_mod.pair_texts([row]) == {"a", "b"}


@pytest.mark.parametrize("count", [3])
def test_build_pairs_end_to_end(tmp_path: pathlib.Path, count: int) -> None:
    jev = tmp_path / "jev.json"
    jev.write_text(
        json.dumps(
            [
                {
                    "pair_id": f"p{index}",
                    "similarity": 0.5,
                    "existing_content": f"e{index}",
                    "incoming_content": f"i{index}",
                    "incoming_type": "fact",
                    "store_label": None,
                }
                for index in range(count)
            ]
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "cases"
    empty.mkdir()

    bounce = _bounce_file(tmp_path, count=8)
    rows = pairs_mod.build_pairs(jev, bounce, empty, seed=1, ledger_path=_ledger(bounce))

    sources = [row.source for row in rows]
    assert sources.count("jev_pairs") == count
    assert sources.count("bounce_log") == 8
    assert sources.count("golden") == 40
    assert len({row.pair_id for row in rows}) == len(rows)
