from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from eval_v1v2 import paths
from eval_v1v2.rows import KeyRow, LabelRow, PairRow, read_jsonl, write_jsonl

TASK = "relation"
LABELS = ("restatement", "replacement", "follow_up", "distinct")
PAGE_NAME = "disputes-relation.html"
EXPORT_VERSION = 1


def find_disputes(labels: Sequence[LabelRow]) -> list[str]:
    picks: dict[str, dict[str, str | int]] = {}
    for row in labels:
        picks.setdefault(row.item_id, {})[row.judge] = row.label
    return sorted(
        item_id
        for item_id, by_judge in picks.items()
        if "claude" in by_judge and "codex" in by_judge and by_judge["claude"] != by_judge["codex"]
    )


def entry_fingerprint(
    item_id: str, kind: str, existing: str, incoming: str, claude: object, codex: object
) -> str:
    payload = json.dumps([item_id, kind, existing, incoming, claude, codex], ensure_ascii=False)

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_entries(labels: Sequence[LabelRow], pairs: Sequence[PairRow]) -> list[dict[str, object]]:
    text = {pair.pair_id: pair for pair in pairs}
    judged: dict[str, dict[str, dict[str, str | int]]] = {}
    for row in labels:
        judged.setdefault(row.item_id, {})[row.judge] = {
            "label": row.label,
            "reason": row.reason,
            "model": row.model,
        }
    entries: list[dict[str, object]] = []
    for item_id in find_disputes(labels):
        pair = text.get(item_id)
        if pair is None:
            continue
        entries.append(
            {
                "item_id": item_id,
                "kind": pair.kind,
                "existing": pair.existing,
                "incoming": pair.incoming,
                "claude": judged[item_id]["claude"],
                "codex": judged[item_id]["codex"],
                "fingerprint": entry_fingerprint(
                    item_id,
                    pair.kind,
                    pair.existing,
                    pair.incoming,
                    judged[item_id]["claude"]["label"],
                    judged[item_id]["codex"]["label"],
                ),
            }
        )
    return entries


def _embed(payload: object) -> str:
    return (
        json.dumps(payload, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(chr(0x2028), "\\u2028")
        .replace(chr(0x2029), "\\u2029")
    )


def storage_key(entries: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha1(
        "\n".join(str(entry["item_id"]) for entry in entries).encode()
    ).hexdigest()
    return f"v1v2-disputes-{TASK}-{digest[:12]}"


def render_page(entries: Sequence[dict[str, object]]) -> str:
    data = {
        "task": TASK,
        "labels": list(LABELS),
        "storage_key": storage_key(entries),
        "export_version": EXPORT_VERSION,
        "entries": list(entries),
    }
    return PAGE_TEMPLATE.replace("__DATA__", _embed(data))


def build_page(out_dir: Path | None = None) -> Path:
    base = out_dir if out_dir is not None else paths.ensure_out_dir()
    labels = read_jsonl(base / f"labels-{TASK}.jsonl", LabelRow)
    pairs = read_jsonl(base / "pairs.jsonl", PairRow)
    page = base / PAGE_NAME
    page.write_text(render_page(build_entries(labels, pairs)), encoding="utf-8")
    return page


def apply_export(export_path: Path, out_dir: Path | None = None) -> dict[str, int]:
    base = out_dir if out_dir is not None else paths.ensure_out_dir()
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    key_path = base / f"key-{TASK}.jsonl"
    key = read_jsonl(key_path, KeyRow) if key_path.exists() else []
    labels = read_jsonl(base / f"labels-{TASK}.jsonl", LabelRow)
    pairs = read_jsonl(base / "pairs.jsonl", PairRow)
    current = {
        str(entry["item_id"]): entry["fingerprint"] for entry in build_entries(labels, pairs)
    }
    disputed = set(find_disputes(labels))
    by_id = {row.item_id: row for row in key}

    counts = {
        "applied": 0,
        "unreviewed": 0,
        "skipped_agreed": 0,
        "skipped_unknown": 0,
        "invalid": 0,
        "stale": 0,
    }
    for entry in payload.get("entries", []):
        item_id = entry.get("item_id")
        label = entry.get("label")
        if not entry.get("reviewed"):
            counts["unreviewed"] += 1
            continue
        if label not in LABELS:
            counts["invalid"] += 1
            continue
        if item_id not in disputed:
            counts["skipped_unknown"] += 1
            continue
        if entry.get("fingerprint") != current.get(item_id):
            counts["stale"] += 1
            continue
        existing = by_id.get(item_id)
        if existing is not None and existing.source == "agreed":
            counts["skipped_agreed"] += 1
            continue
        by_id[item_id] = KeyRow.model_validate(
            {"item_id": item_id, "label": label, "source": "owner"}
        )
        counts["applied"] += 1

    write_jsonl(key_path, by_id.values())
    return counts


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relation disputes</title>
<style>
:root{--bg:#12141a;--panel:#191c22;--border:#2a2f38;--text:#e8eaed;--dim:#9aa3ad;--accent:#6cb6ff;--good:#63d296;--warn:#f2954c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:system-ui,sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--border);
  padding:12px 24px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header .grow{flex:1}
button{background:var(--panel);color:var(--text);border:1px solid var(--border);
  border-radius:4px;padding:6px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--accent)}
button.picked{border-color:var(--good);background:#1f3328}
main{max-width:1100px;margin:0 auto;padding:24px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.col{background:var(--panel);border:1px solid var(--border);border-radius:6px;padding:14px}
.col h3{margin:0 0 8px;font-size:13px;color:var(--dim);text-transform:uppercase}
.col p{margin:0;white-space:pre-wrap;line-height:1.5}
.judges{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
.judge{background:var(--panel);border:1px dashed var(--border);border-radius:6px;padding:12px}
.judge .pick{font-weight:600;color:var(--accent)}
.judge .why{color:var(--dim);margin-top:6px}
.choices{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}
.status{color:var(--dim)}
.status.done{color:var(--good)}
.hint{color:var(--dim);font-size:13px;margin-top:20px;line-height:1.7}
kbd{border:1px solid var(--border);border-radius:3px;padding:0 5px;background:var(--panel)}
.empty{padding:40px;text-align:center;color:var(--dim)}
</style>
</head>
<body>
<header>
  <strong>Relation disputes</strong>
  <span id="progress" class="status"></span>
  <span class="grow"></span>
  <button id="next-open">Next unreviewed</button>
  <button id="export">Export JSON</button>
</header>
<main id="main"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  var data = JSON.parse(document.getElementById("data").textContent);
  var entries = data.entries;
  var labels = data.labels;
  var state = {};
  var index = 0;

  function load(){
    try {
      var raw = window.localStorage.getItem(data.storage_key);
      if (raw) { state = JSON.parse(raw); }
    } catch (e) { state = {}; }
  }
  function save(){
    try { window.localStorage.setItem(data.storage_key, JSON.stringify(state)); } catch (e) {}
  }
  function entryState(id){
    if (!state[id]) { state[id] = {label: null, reviewed: false}; }
    return state[id];
  }
  function el(tag, cls, text){
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined) { node.textContent = text; }
    return node;
  }
  function reviewedCount(){
    return entries.filter(function(e){ return entryState(e.item_id).reviewed; }).length;
  }
  function setLabel(label){
    var s = entryState(entries[index].item_id);
    s.label = label;
    s.reviewed = true;
    save();
    render();
  }
  function toggleReviewed(){
    var s = entryState(entries[index].item_id);
    if (!s.reviewed && !s.label) { return; }
    s.reviewed = !s.reviewed;
    save();
    render();
  }
  function clearPick(){
    var s = entryState(entries[index].item_id);
    s.label = null;
    s.reviewed = false;
    save();
    render();
  }
  function move(delta){
    if (!entries.length) { return; }
    index = Math.max(0, Math.min(entries.length - 1, index + delta));
    render();
  }
  function nextOpen(){
    for (var step = 1; step <= entries.length; step++) {
      var at = (index + step) % entries.length;
      if (!entryState(entries[at].item_id).reviewed) { index = at; render(); return; }
    }
  }
  function judgeBox(name, verdict, key){
    var box = el("div", "judge");
    box.appendChild(el("div", "", name + " (" + verdict.model + ") suggests, key " + key));
    box.appendChild(el("div", "pick", String(verdict.label)));
    box.appendChild(el("div", "why", verdict.reason));
    return box;
  }
  function render(){
    var main = document.getElementById("main");
    main.textContent = "";
    var progress = document.getElementById("progress");
    progress.textContent = reviewedCount() + " of " + entries.length + " reviewed";
    if (!entries.length) {
      main.appendChild(el("div", "empty", "No disputed pairs."));
      return;
    }
    var entry = entries[index];
    var s = entryState(entry.item_id);
    var head = el("div", "status" + (s.reviewed ? " done" : ""),
      "Pair " + (index + 1) + " of " + entries.length + " - " + entry.item_id + " - " + entry.kind +
      (s.reviewed ? " - reviewed" : " - open"));
    main.appendChild(head);

    var cols = el("div", "cols");
    var left = el("div", "col");
    left.appendChild(el("h3", "", "Existing"));
    left.appendChild(el("p", "", entry.existing));
    var right = el("div", "col");
    right.appendChild(el("h3", "", "Incoming"));
    right.appendChild(el("p", "", entry.incoming));
    cols.appendChild(left);
    cols.appendChild(right);
    main.appendChild(cols);

    var judges = el("div", "judges");
    judges.appendChild(judgeBox("Claude", entry.claude, "c"));
    judges.appendChild(judgeBox("Codex", entry.codex, "x"));
    main.appendChild(judges);

    var choices = el("div", "choices");
    labels.forEach(function(label, i){
      var button = el("button", s.label === label ? "picked" : "", (i + 1) + "  " + label);
      button.addEventListener("click", function(){ setLabel(label); });
      choices.appendChild(button);
    });
    main.appendChild(choices);

    var hint = el("div", "hint");
    hint.appendChild(document.createTextNode("Keys: "));
    hint.appendChild(el("kbd", "", "1-4"));
    hint.appendChild(document.createTextNode(" pick a label and mark reviewed, "));
    hint.appendChild(el("kbd", "", "c"));
    hint.appendChild(document.createTextNode(" take Claude's pick, "));
    hint.appendChild(el("kbd", "", "x"));
    hint.appendChild(document.createTextNode(" take Codex's pick, "));
    hint.appendChild(el("kbd", "", "r"));
    hint.appendChild(document.createTextNode(" toggle reviewed, "));
    hint.appendChild(el("kbd", "", "u"));
    hint.appendChild(document.createTextNode(" clear, "));
    hint.appendChild(el("kbd", "", "j"));
    hint.appendChild(document.createTextNode(" next, "));
    hint.appendChild(el("kbd", "", "k"));
    hint.appendChild(document.createTextNode(" previous."));
    hint.appendChild(document.createTextNode(" Progress saves in this browser as you go."));
    main.appendChild(hint);
  }
  function exportJson(){
    var out = {
      export_version: data.export_version,
      task: data.task,
      exported_at: new Date().toISOString(),
      entries: entries.map(function(e){
        var s = entryState(e.item_id);
        return {
          item_id: e.item_id,
          label: s.label,
          reviewed: s.reviewed,
          fingerprint: e.fingerprint,
          claude: e.claude.label,
          codex: e.codex.label
        };
      })
    };
    var blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
    var link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "disputes-" + data.task + "-export.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
  document.addEventListener("keydown", function(event){
    if (event.ctrlKey || event.metaKey || event.altKey || !entries.length) { return; }
    var key = event.key;
    if (key >= "1" && key <= "4") { setLabel(labels[Number(key) - 1]); }
    else if (key === "c") { setLabel(entries[index].claude.label); }
    else if (key === "x") { setLabel(entries[index].codex.label); }
    else if (key === "r") { toggleReviewed(); }
    else if (key === "u") { clearPick(); }
    else if (key === "j") { move(1); }
    else if (key === "k") { move(-1); }
    else { return; }
    event.preventDefault();
  });
  document.getElementById("export").addEventListener("click", exportJson);
  document.getElementById("next-open").addEventListener("click", nextOpen);
  load();
  render();
})();
</script>
</body>
</html>
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the relation dispute page or apply an export."
    )
    parser.add_argument("--apply", type=Path, default=None, metavar="EXPORT_JSON")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.apply is not None:
        print(json.dumps(apply_export(args.apply, args.out_dir)))
        return 0

    page = build_page(args.out_dir)
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
