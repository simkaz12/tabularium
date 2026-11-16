import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HashEmbedder } from "../src/embed.ts";
import { stubJev } from "../src/testing.ts";
import { runSignals, replayPairSignals } from "./dup_signals.ts";
import { openEval } from "./engine.ts";
import type { EvalHarness, PairRow } from "./engine.ts";
import { countRecords } from "./save.ts";

const EXISTING_TEXT =
  "The nightly backup job copies the database dump to the cold storage bucket after midnight.";
const INCOMING_TEXT = "The nightly backup job copies the database dump to cold storage after midnight.";
const PROBABILITIES = { restatement: 0.31, replacement: 0.04, follow_up: 0.58, distinct: 0.07 };

let harnesses: EvalHarness[] = [];
let scratch: string[] = [];

afterEach(() => {
  for (const harness of harnesses) {
    rmSync(harness.dir, { recursive: true, force: true });
  }

  for (const dir of scratch) {
    rmSync(dir, { recursive: true, force: true });
  }

  harnesses = [];
  scratch = [];
});

function pair(id: string): PairRow {
  return {
    pair_id: id,
    source: "golden",
    kind: "fact",
    existing: EXISTING_TEXT,
    incoming: INCOMING_TEXT,
    similarity: null,
    old_label: null,
  };
}

function relationAnswer(pick: string, confidence: number, probabilities: Record<string, number>) {
  return () => ({ type: "choice", choice: pick, confidence, probabilities });
}

async function harnessWith(overrides: Parameters<typeof stubJev>[0]) {
  const stub = stubJev(overrides);
  const harness = await openEval({ embedder: new HashEmbedder(64), jev: stub });

  harnesses.push(harness);

  return { harness, stub };
}

describe("dup signal capture", () => {
  it("writes the relation probabilities and confidence through unchanged", async () => {
    const { harness } = await harnessWith({
      relation: relationAnswer("follow_up", 0.58, PROBABILITIES),
      adds_information: relationAnswer("no", 0.83, { yes: 0.17, no: 0.83 }),
    });
    const base = await countRecords(harness);
    const row = await replayPairSignals(harness, pair("p1"), base, "stub-model");

    expect(row.found_existing).toBe(true);
    expect(row.relation_pick).toBe("follow_up");
    expect(row.bucket).toBe("new");
    expect(row.existing?.pick).toBe("follow_up");
    expect(row.existing?.confidence).toBe(0.58);
    expect(row.existing?.probabilities).toEqual(PROBABILITIES);
    expect(row.existing?.reason_class).toBeNull();
    expect(row.existing?.similarity).toBeGreaterThan(0);
    expect(row.top_other).toBeNull();
    expect(row.adds_information.answer).toBe("no");
    expect(row.adds_information.probability).toBe(0.83);
    expect(row.adds_information.elapsed_ms).toBeGreaterThanOrEqual(0);
    expect(row.adds_information.error).toBeNull();
    expect(await countRecords(harness)).toBe(base);
  });

  it("asks the extra question after the relation calls, never alongside them", async () => {
    const { harness, stub } = await harnessWith({
      relation: relationAnswer("restatement", 0.9, { restatement: 0.9 }),
    });

    await replayPairSignals(harness, pair("p2"), await countRecords(harness), "stub-model");

    const questionSets = stub.calls.map((call) => Object.keys(call.questions));
    const extra = questionSets.filter((ids) => ids.includes("adds_information"));

    expect(extra).toEqual([["adds_information"]]);
    expect(questionSets.at(-1)).toEqual(["adds_information"]);
    expect(stub.calls.at(-1)?.model).toBe("stub-model");
  });

  it("records an unanswered extra question as null with the error, not a guess", async () => {
    const { harness } = await harnessWith({
      relation: relationAnswer("distinct", 0.9, { distinct: 0.9 }),
      adds_information: relationAnswer("maybe", 0.5, { maybe: 0.5 }),
    });
    const row = await replayPairSignals(harness, pair("p3"), await countRecords(harness), "stub-model");

    expect(row.adds_information.answer).toBeNull();
    expect(row.adds_information.probability).toBeNull();
    expect(row.adds_information.error).toContain("unexpected choice maybe");
  });

  it("buckets a restatement pick as caught and appends one line per pair", async () => {
    const { harness } = await harnessWith({
      relation: relationAnswer("restatement", 0.9, { restatement: 0.9, follow_up: 0.1 }),
    });
    const dir = mkdtempSync(join(tmpdir(), "dup-signals-test-"));
    const out = join(dir, "signals.jsonl");

    scratch.push(dir);

    const rows = await runSignals(harness, [pair("a"), pair("b")], out, "stub-model", undefined, new Set(["b"]));
    const lines = readFileSync(out, "utf8").trim().split("\n");

    expect(rows.map((row) => row.pair_id)).toEqual(["a"]);
    expect(lines).toHaveLength(1);
    expect(JSON.parse(lines[0] as string).bucket).toBe("caught");
  });
});
