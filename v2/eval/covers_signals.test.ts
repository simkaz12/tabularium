import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { stubJev } from "../src/testing.ts";
import { coversForPair, evenSample, runCovers } from "./covers_signals.ts";
import type { PairRow } from "./engine.ts";

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
    existing: "The nightly backup job copies the database dump to cold storage.",
    incoming: "The nightly backup job copies the database dump to cold storage after midnight.",
    similarity: null,
    old_label: null,
  };
}

function yesNo(choice: string, yes: number) {
  return () => ({ type: "choice", choice, confidence: Math.max(yes, 1 - yes), probabilities: { yes, no: 1 - yes } });
}

describe("containment signal capture", () => {
  it("asks the three questions in one call and keeps the probability of yes", async () => {
    const stub = stubJev({
      new_covers_old: yesNo("yes", 0.92),
      old_covers_new: yesNo("no", 0.25),
      enhances: yesNo("yes", 0.7),
    });
    const row = await coversForPair(stub, pair("p1"), "stub-model", 1);

    expect(stub.calls).toHaveLength(1);
    expect(Object.keys(stub.calls[0]?.questions ?? {})).toEqual(["new_covers_old", "old_covers_new", "enhances"]);
    expect(row.new_covers_old.p_yes).toBe(0.92);
    expect(row.old_covers_new.answer).toBe("no");
    expect(row.old_covers_new.p_yes).toBe(0.25);
    expect(row.enhances.p_yes).toBe(0.7);
    expect(row.error).toBeNull();
    expect(row.run).toBe(1);
  });

  it("records a failed call as unanswered on every question, not a guess", async () => {
    const stub = stubJev();

    stub.failWith = new Error("jev down");

    const row = await coversForPair(stub, pair("p2"), "stub-model", 2);

    expect(row.error).toBe("jev down");
    expect(row.new_covers_old.answer).toBeNull();
    expect(row.new_covers_old.p_yes).toBeNull();
    expect(row.enhances.error).toBe("jev down");
  });

  it("flags an unexpected choice instead of reading it as no", async () => {
    const stub = stubJev({ enhances: () => ({ type: "choice", choice: "maybe", confidence: 0.5, probabilities: {} }) });
    const row = await coversForPair(stub, pair("p3"), "stub-model", 1);

    expect(row.enhances.answer).toBeNull();
    expect(row.enhances.error).toContain("unexpected choice maybe");
  });

  it("appends one line per pair and skips finished pairs", async () => {
    const dir = mkdtempSync(join(tmpdir(), "covers-signals-test-"));
    const out = join(dir, "signals.jsonl");

    scratch.push(dir);

    const rows = await runCovers(stubJev(), [pair("a"), pair("b")], out, "stub-model", 1, undefined, new Set(["b"]));
    const lines = readFileSync(out, "utf8").trim().split("\n");

    expect(rows.map((row) => row.pair_id)).toEqual(["a"]);
    expect(lines).toHaveLength(1);
  });

  it("takes an even sample without repeats", () => {
    const items = Array.from({ length: 242 }, (_, index) => index);
    const sample = evenSample(items, 40);

    expect(sample).toHaveLength(40);
    expect(new Set(sample).size).toBe(40);
    expect(sample[0]).toBe(0);
    expect(sample.at(-1)).toBeLessThan(242);
  });
});
