"""Tests for scripts/eval_v1v2/rows.py and paths.py.

Falsifiability: changing ``Bucket`` in rows.py to drop ``"wrong_target"`` turned
``test_round_trip`` red for the save shape's wrong_target case; widening ``Bucket`` to ``str``
turned ``test_bad_bucket_rejected`` red. Both reverted.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

_SCRIPTS_DIR = pathlib.Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import paths  # noqa: E402
from eval_v1v2.rows import (  # noqa: E402
    CorpusRow,
    KeyRow,
    LabelRow,
    NoiseRow,
    PairRow,
    QuestionRow,
    SaveRow,
    SearchRow,
    read_jsonl,
    write_jsonl,
)

sys.path.pop(0)

_VEC = [0.5] * 768

_SAMPLES: list[BaseModel] = [
    PairRow(
        pair_id="p1",
        source="jev_pairs",
        kind="fact",
        existing="a",
        incoming="b",
        similarity=0.91,
        old_label=None,
    ),
    NoiseRow(
        id="n1",
        kind="decision",
        content="c",
        vector=_VEC,
        recall_count=2,
        created_at="2026-01-01T00:00:00Z",
        lifecycle="live",
    ),
    LabelRow(item_id="p1", judge="claude", model="m", label="follow_up", reason="r"),
    LabelRow(item_id="q1:r1", judge="codex", model="m", label=2, reason="r"),
    KeyRow(item_id="p1", label="distinct", source="owner"),
    SaveRow(
        pair_id="p1",
        system="v2",
        bucket="wrong_target",
        found_existing=False,
        relation_pick=None,
        raw_outcome="approved",
        elapsed_ms=12.5,
        degraded="jev_unavailable",
    ),
    CorpusRow(
        id="r1",
        kind="fact",
        content="c",
        scope="eval",
        lifecycle="live",
        vector=_VEC,
        recall_count=0,
        created_at="2026-01-01T00:00:00Z",
        **{"extra_field": "kept"},
    ),
    QuestionRow(question_id="q1", query="what", scope="tabularium", origin="mined"),
    SearchRow(
        question_id="q1",
        arm="v2_plain",
        rank=1,
        record_id="r1",
        score=0.3,
        elapsed_ms=4.0,
    ),
]


@pytest.mark.parametrize("row", _SAMPLES, ids=lambda r: type(r).__name__)
def test_round_trip(row: BaseModel, tmp_path: pathlib.Path) -> None:
    path = tmp_path / "rows.jsonl"

    written = write_jsonl(path, [row, row])
    back = read_jsonl(path, type(row))

    assert written == 2
    assert back == [row, row]


def _save_payload(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "pair_id": "p",
        "system": "v1",
        "bucket": "caught",
        "found_existing": True,
        "relation_pick": None,
        "raw_outcome": "reinforced",
        "elapsed_ms": 1.0,
        "degraded": None,
    }
    base.update(over)
    return base


def test_bad_bucket_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "save.jsonl"
    path.write_text(SaveRow(**_save_payload()).model_dump_json() + "\n")
    assert len(read_jsonl(path, SaveRow)) == 1

    bad = SaveRow(**_save_payload()).model_dump_json().replace('"caught"', '"merged"')
    path.write_text(bad + "\n")

    with pytest.raises(ValidationError):
        read_jsonl(path, SaveRow)


def test_bad_label_and_arm_rejected() -> None:
    with pytest.raises(ValidationError):
        LabelRow.model_validate(
            {"item_id": "x", "judge": "claude", "model": "m", "label": "maybe", "reason": "r"}
        )
    with pytest.raises(ValidationError):
        LabelRow.model_validate(
            {"item_id": "x", "judge": "claude", "model": "m", "label": 3, "reason": "r"}
        )
    with pytest.raises(ValidationError):
        SearchRow.model_validate(
            {
                "question_id": "q",
                "arm": "v3",
                "rank": 1,
                "record_id": "r",
                "score": 0.1,
                "elapsed_ms": 1.0,
            }
        )


def test_wrong_vector_length_rejected() -> None:
    with pytest.raises(ValidationError):
        NoiseRow.model_validate(
            {
                "id": "n",
                "kind": "fact",
                "content": "c",
                "vector": [0.1] * 767,
                "recall_count": 0,
                "created_at": "t",
                "lifecycle": "live",
            }
        )


def test_default_out_dir_is_dev_corpus() -> None:
    assert paths.OUT_DIR.parts[-3:] == ("dev-corpus", "evals", "v1v2")


def test_out_dir_created_on_demand(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "evals" / "v1v2"
    monkeypatch.setattr(paths, "OUT_DIR", target)
    assert not target.exists()

    result = paths.out_path("report.md")

    assert target.is_dir()
    assert result == target / "report.md"
