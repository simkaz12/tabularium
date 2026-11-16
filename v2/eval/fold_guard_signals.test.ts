import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { stubJev } from "../src/testing.ts";
import type { PairRow } from "./engine.ts";
import {
  finishedKeys,
  foldKey,
  foldTargets,
  guardsForFold,
  runBand,
  runGuards,
  sameSubjectForPair,
} from "./fold_guard_signals.ts";
import type { FoldRow } from "./fold_guard_signals.ts";

let scratch: string[] = [];

afterEach(() => {
  for (const dir of scratch) {
    rmSync(dir, { recursive: true, force: true });
  }

  scratch = [];
});

function pair(id: string): PairRow {
  return {
    pair_id: id,
    source: "golden",
    kind: "fact",
    existing: "The nightly backup copies the dump to cold storage.",
    incoming: "The backup runs after midnight.",
    similarity: null,
    old_label: null,
  };
}

function yesNo(choice: string, yes: number) {
  return () => ({ type: "choice", choice, confidence: Math.max(yes, 1 - yes), probabilities: { yes, no: 1 - yes } });
}

function stateOf(call: { state?: unknown } | undefined): Record<string, string> {
  return (call?.state ?? {}) as Record<string, string>;
}

describe("fold guard capture", () => {
  it("asks four single-question calls and sends the right text on each side", async () => {
    const stub = stubJev({
      fold_covers_old: yesNo("yes", 0.9),
      fold_covers_new: yesNo("no", 0.2),
      sources_cover_fold: yesNo("yes", 0.8),
      fold_single: yesNo("no", 0.1),
    });
    const row = await guardsForFold(stub, pair("p1"), "sonnet", "The merged note text.", "stub-model", 1);

    expect(stub.calls).toHaveLength(4);
    expect(stub.calls.map((call) => Object.keys(call.questions))).toEqual([
      ["fold_covers_old"],
      ["fold_covers_new"],
      ["sources_cover_fold"],
      ["fold_single"],
    ]);
    expect(stateOf(stub.calls[0]).existing).toContain("cold storage");
    expect(stateOf(stub.calls[0]).incoming).toContain("merged note");
    expect(stateOf(stub.calls[1]).existing).toContain("after midnight");
    expect(stateOf(stub.calls[2]).existing).toContain("cold storage");
    expect(stateOf(stub.calls[2]).existing).toContain("after midnight");
    expect(stateOf(stub.calls[3]).text).toContain("merged note");
    expect(row.fold_covers_old.p_yes).toBe(0.9);
    expect(row.fold_covers_new.answer).toBe("no");
    expect(row.fold_single.p_yes).toBe(0.1);
    expect(row.error).toBeNull();
    expect(row.writer).toBe("sonnet");
  });

  it("records a failed call as unanswered on every guard", async () => {
    const stub = stubJev();

    stub.failWith = new Error("jev down");

    const row = await guardsForFold(stub, pair("p2"), "haiku", "merged", "stub-model", 2);

    expect(row.error).toBe("jev down");
    expect(row.fold_covers_old.p_yes).toBeNull();
    expect(row.fold_single.error).toBe("jev down");
    expect(row.fold_covers_old.elapsed_ms).toBeGreaterThanOrEqual(0);
  });

  it("asks same_subject once on the two source notes", async () => {
    const stub = stubJev({ same_subject: yesNo("yes", 0.65) });
    const row = await sameSubjectForPair(stub, pair("p3"), "stub-model", 1);

    expect(stub.calls).toHaveLength(1);
    expect(Object.keys(stub.calls[0]?.questions ?? {})).toEqual(["same_subject"]);
    expect(stateOf(stub.calls[0]).incoming).toContain("after midnight");
    expect(row.same_subject.p_yes).toBe(0.65);
  });

  it("builds targets from the latest answered fold per pair and writer", () => {
    const folds: FoldRow[] = [
      { pair_id: "a", writer: "haiku", text: null, error: "down" },
      { pair_id: "a", writer: "haiku", text: "second try", error: null },
      { pair_id: "a", writer: "sonnet", text: "other", error: null },
      { pair_id: "missing", writer: "sonnet", text: "orphan", error: null },
    ];
    const targets = foldTargets(folds, new Map([["a", pair("a")]]));

    expect(targets.map((target) => [target.writer, target.fold])).toEqual([
      ["haiku", "second try"],
      ["sonnet", "other"],
    ]);
  });

  it("resumes: skips finished folds and retries failed ones", async () => {
    const dir = mkdtempSync(join(tmpdir(), "fold-guard-test-"));
    const out = join(dir, "guards.jsonl");

    scratch.push(dir);

    const targets = [
      { pair: pair("a"), writer: "haiku", fold: "one" },
      { pair: pair("b"), writer: "haiku", fold: "two" },
    ];
    const done = new Set([foldKey("b", "haiku")]);
    const rows = await runGuards(stubJev(), targets, out, "stub-model", 1, undefined, done);

    expect(rows.map((row) => row.pair_id)).toEqual(["a"]);
    expect(readFileSync(out, "utf8").trim().split("\n")).toHaveLength(1);

    const finished = finishedKeys(
      [
        { key: "x", error: null },
        { key: "y", error: "down" },
      ],
      (row) => row.key,
    );

    expect([...finished]).toEqual(["x"]);
  });

  it("appends one band line per pair and skips finished pairs", async () => {
    const dir = mkdtempSync(join(tmpdir(), "fold-band-test-"));
    const out = join(dir, "band.jsonl");

    scratch.push(dir);

    const rows = await runBand(stubJev(), [pair("a"), pair("b")], out, "stub-model", 1, undefined, new Set(["b"]));

    expect(rows.map((row) => row.pair_id)).toEqual(["a"]);
    expect(readFileSync(out, "utf8").trim().split("\n")).toHaveLength(1);
  });
});
