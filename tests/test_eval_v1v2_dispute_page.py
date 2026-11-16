from __future__ import annotations

import json
import pathlib
import re
import sys

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_v1v2 import dispute_page  # noqa: E402
from eval_v1v2.rows import KeyRow, LabelRow, PairRow, read_jsonl, write_jsonl  # noqa: E402

sys.path.pop(0)


def _label(item_id: str, judge: str, label: str) -> LabelRow:
    return LabelRow.model_validate(
        {
            "item_id": item_id,
            "judge": judge,
            "model": "m",
            "label": label,
            "reason": f"{judge} says",
        }
    )


def _pair(pair_id: str, existing: str = "old", incoming: str = "new") -> PairRow:
    return PairRow(
        pair_id=pair_id,
        source="golden",
        kind="fact",
        existing=existing,
        incoming=incoming,
        similarity=0.9,
        old_label="SECRET_OLD_LABEL",
    )


def _seed(base: pathlib.Path) -> None:
    write_jsonl(
        base / "labels-relation.jsonl",
        [
            _label("agree", "claude", "distinct"),
            _label("agree", "codex", "distinct"),
            _label("split1", "claude", "restatement"),
            _label("split1", "codex", "replacement"),
            _label("split2", "claude", "follow_up"),
            _label("split2", "codex", "distinct"),
            _label("half", "claude", "distinct"),
        ],
    )
    write_jsonl(
        base / "pairs.jsonl",
        [_pair("agree"), _pair("split1", "a </script> b", "x & y"), _pair("split2"), _pair("half")],
    )
    write_jsonl(
        base / "key-relation.jsonl", [KeyRow(item_id="agree", label="distinct", source="agreed")]
    )


def _signed(base: pathlib.Path, entries: list[dict[str, object]]) -> list[dict[str, object]]:
    current = {
        str(entry["item_id"]): entry["fingerprint"]
        for entry in dispute_page.build_entries(
            read_jsonl(base / "labels-relation.jsonl", LabelRow),
            read_jsonl(base / "pairs.jsonl", PairRow),
        )
    }

    return [
        {**entry, "fingerprint": current[str(entry["item_id"])]}
        if str(entry["item_id"]) in current
        else entry
        for entry in entries
    ]


def _export(path: pathlib.Path, entries: list[dict[str, object]]) -> pathlib.Path:
    entries = _signed(path.parent, entries)
    path.write_text(json.dumps({"export_version": 1, "task": "relation", "entries": entries}))
    return path


class TestPage:
    def test_only_split_pairs_appear_with_both_picks(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)

        page = dispute_page.build_page(tmp_path)

        html = page.read_text()
        data = json.loads(
            re.search(
                r'<script id="data" type="application/json">(.*?)</script>', html, re.S
            ).group(1)  # type: ignore[union-attr]
        )
        ids = [entry["item_id"] for entry in data["entries"]]
        assert ids == ["split1", "split2"]
        assert data["entries"][0]["claude"]["label"] == "restatement"
        assert data["entries"][0]["codex"]["reason"] == "codex says"

    def test_page_is_self_contained_blind_and_script_safe(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)

        html = dispute_page.build_page(tmp_path).read_text()

        assert "SECRET_OLD_LABEL" not in html
        assert html.count("</script>") == 2
        assert "a </script> b" not in html
        for network in ("http://", "https://", "fetch(", "XMLHttpRequest", "src=", "<link"):
            assert network not in html
        for feature in ("localStorage", "keydown", "reviewed", "Blob"):
            assert feature in html

    def test_find_disputes_ignores_single_judge_items(self) -> None:
        rows = [
            _label("a", "claude", "distinct"),
            _label("b", "claude", "distinct"),
            _label("b", "codex", "distinct"),
        ]

        assert dispute_page.find_disputes(rows) == []


class TestApply:
    def test_reviewed_picks_become_owner_rows(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        export = _export(
            tmp_path / "export.json",
            [
                {"item_id": "split1", "label": "follow_up", "reviewed": True},
                {"item_id": "split2", "label": "distinct", "reviewed": True},
            ],
        )

        counts = dispute_page.apply_export(export, tmp_path)

        key = {row.item_id: row for row in read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)}
        assert counts["applied"] == 2
        assert key["split1"].label == "follow_up"
        assert key["split1"].source == "owner"
        assert key["agree"].source == "agreed"

    def test_unreviewed_pairs_stay_out_of_the_key(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        export = _export(
            tmp_path / "export.json",
            [
                {"item_id": "split1", "label": "follow_up", "reviewed": False},
                {"item_id": "split2", "label": None, "reviewed": False},
            ],
        )

        counts = dispute_page.apply_export(export, tmp_path)

        key = read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)
        assert [row.item_id for row in key] == ["agree"]
        assert counts["unreviewed"] == 2

    def test_agreed_row_is_never_overwritten(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        write_jsonl(
            tmp_path / "key-relation.jsonl",
            [KeyRow(item_id="split1", label="restatement", source="agreed")],
        )
        export = _export(
            tmp_path / "export.json",
            [{"item_id": "split1", "label": "distinct", "reviewed": True}],
        )

        counts = dispute_page.apply_export(export, tmp_path)

        key = read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)
        assert [(row.label, row.source) for row in key] == [("restatement", "agreed")]
        assert counts["skipped_agreed"] == 1

    def test_non_disputed_and_invalid_labels_are_refused(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        export = _export(
            tmp_path / "export.json",
            [
                {"item_id": "agree", "label": "replacement", "reviewed": True},
                {"item_id": "ghost", "label": "distinct", "reviewed": True},
                {"item_id": "split1", "label": "1", "reviewed": True},
            ],
        )

        counts = dispute_page.apply_export(export, tmp_path)

        key = read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)
        assert [row.item_id for row in key] == ["agree"]
        assert key[0].label == "distinct"
        assert counts["invalid"] == 1
        assert counts["skipped_unknown"] == 2

    def test_reapply_updates_owner_row(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        first = _export(
            tmp_path / "e1.json", [{"item_id": "split1", "label": "distinct", "reviewed": True}]
        )
        second = _export(
            tmp_path / "e2.json", [{"item_id": "split1", "label": "follow_up", "reviewed": True}]
        )

        dispute_page.apply_export(first, tmp_path)
        dispute_page.apply_export(second, tmp_path)

        key = {row.item_id: row for row in read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)}
        assert key["split1"].label == "follow_up"

    def test_entries_whose_texts_or_picks_changed_are_refused(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        export = _export(
            tmp_path / "export.json",
            [
                {"item_id": "split1", "label": "follow_up", "reviewed": True},
                {"item_id": "split2", "label": "distinct", "reviewed": True},
            ],
        )
        write_jsonl(
            tmp_path / "pairs.jsonl",
            [
                _pair("agree"),
                _pair("split1", "a </script> b", "rewritten"),
                _pair("split2"),
                _pair("half"),
            ],
        )

        counts = dispute_page.apply_export(export, tmp_path)

        key = read_jsonl(tmp_path / "key-relation.jsonl", KeyRow)
        assert [row.item_id for row in key] == ["agree", "split2"]
        assert counts["stale"] == 1
        assert counts["applied"] == 1

    def test_entry_without_a_fingerprint_is_refused(self, tmp_path: pathlib.Path) -> None:
        _seed(tmp_path)
        export = tmp_path / "export.json"
        export.write_text(
            json.dumps({"entries": [{"item_id": "split1", "label": "distinct", "reviewed": True}]})
        )

        counts = dispute_page.apply_export(export, tmp_path)

        assert counts["stale"] == 1
        assert counts["applied"] == 0
