import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HashEmbedder } from "../src/embed.ts";
import { stubJev } from "../src/testing.ts";
import { openEval } from "./engine.ts";
import type { CorpusRow, EvalHarness, QuestionRow } from "./engine.ts";
import { loadCorpus, runArm, sampleKnownItems, selfCheck, zeroHitCalls } from "./search.ts";

const DIM = 64;
const SUBJECTS = [
  "tide tables",
  "sourdough starters",
  "bearing grease",
  "tax filings",
  "comet orbits",
  "kiln schedules",
  "ferry timetables",
  "beehive audits",
  "glacier surveys",
  "loom threading",
];

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

function corpusOf(count: number): CorpusRow[] {
  return Array.from({ length: count }, (_, index) => {
    const subject = SUBJECTS[index % SUBJECTS.length] as string;
    const decision = index % 3 === 0;

    return {
      id: `r${index}`,
      kind: decision ? "decision" : "fact",
      content: decision
        ? `Decision: handle ${subject} number ${index} through the weekly review. Why: audits on ${subject} slipped.`
        : `Record ${index} says ${subject} number ${index} are checked every Tuesday morning.`,
      scope: "eval",
      lifecycle: "live",
      vector: [],
      recall_count: 0,
      created_at: "2026-07-31T13:54:10+00:00",
    };
  });
}

async function loaded(rows: CorpusRow[]): Promise<EvalHarness> {
  const harness = await openEval({ embedder: new HashEmbedder(DIM), jev: stubJev() });

  harnesses.push(harness);
  await loadCorpus(harness, rows);

  return harness;
}

describe("facts-only arm", () => {
  const question: QuestionRow = {
    question_id: "q1",
    query: "weekly review of tide tables",
    scope: "eval",
    origin: "mined",
  };

  it("returns no decisions with kind fact while the plain arm does", async () => {
    const harness = await loaded(corpusOf(12));
    const dir = mkdtempSync(join(tmpdir(), "eval-search-facts-"));

    scratch.push(dir);

    const all = await runArm(harness.seed.recall, "v2_plain", [question], join(dir, "all.jsonl"));
    const facts = await runArm(
      harness.seed.recall,
      "v2_plain_facts",
      [question],
      join(dir, "facts.jsonl"),
      "fact",
    );
    const kinds = new Map(corpusOf(12).map((row) => [row.id, row.kind]));

    expect(all.some((row) => kinds.get(row.record_id) === "decision")).toBe(true);
    expect(facts.length).toBeGreaterThan(0);
    expect(facts.every((row) => kinds.get(row.record_id) === "fact")).toBe(true);
    expect(facts.every((row) => row.arm === "v2_plain_facts")).toBe(true);
  });
});

describe("known-item self-check", () => {
  it("passes on a loaded store", async () => {
    const rows = corpusOf(30);
    const harness = await loaded(rows);
    const check = await selfCheck(harness.seed.recall, sampleKnownItems(rows, 20, 7));

    expect(check.sampled).toBe(20);
    expect(check.rank_one).toBeGreaterThanOrEqual(18);
    expect(check.passed).toBe(true);
  });

  it("fails when the store is empty", async () => {
    const rows = corpusOf(30);
    const harness = await loaded([]);
    const check = await selfCheck(harness.seed.recall, sampleKnownItems(rows, 20, 7));

    expect(check.rank_one).toBe(0);
    expect(check.ranks.every((rank) => rank === null)).toBe(true);
    expect(check.passed).toBe(false);
  });

  it("samples the same records for the same seed and skips duplicated content", () => {
    const rows = [...corpusOf(30), { ...(corpusOf(1)[0] as CorpusRow), id: "twin" }];
    const first = sampleKnownItems(rows, 10, 3).map((row) => row.id);
    const second = sampleKnownItems(rows, 10, 3).map((row) => row.id);

    expect(first).toEqual(second);
    expect(first).not.toContain("r0");
    expect(first).not.toContain("twin");
  });
});

describe("zero-hit calls", () => {
  it("counts the questions that wrote no row", () => {
    const question = (id: string) => ({ question_id: id, query: id, scope: "global", origin: "mined" }) as const;
    const row = (id: string) =>
      ({ question_id: id, arm: "v2_plain", rank: 1, record_id: "r", score: 1, elapsed_ms: 1 }) as const;

    expect(zeroHitCalls([question("a"), question("b"), question("c")], [row("a"), row("a"), row("c")])).toBe(1);
  });
});
