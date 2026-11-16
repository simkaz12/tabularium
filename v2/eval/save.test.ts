import { mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { Proposal } from "../src/domain.ts";
import { HashEmbedder } from "../src/embed.ts";
import { choiceAnswer, stubJev } from "../src/testing.ts";
import { assertTempPath, clipHeadline, fallbackShape, openEval } from "./engine.ts";
import type { CorpusRow, EvalHarness, NoiseRow, PairRow, QuestionRow } from "./engine.ts";
import { bucketOf, degradedOf, rawOutcomeOf, replayPair, runSave, seedNoise, withProposalOptions } from "./save.ts";
import { loadReplies, readReplies, replayOf } from "./reply.ts";
import type { AskReply } from "./reply.ts";
import { loadCorpus, runArm, toLegacyLine } from "./search.ts";

const DIM = 64;
const EXISTING_TEXT =
  "The nightly backup job copies the database dump to the cold storage bucket after midnight.";
const TOPICS = ["gardening", "astronomy", "pastry", "sailing", "welding", "poetry", "geology", "chess"];

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

function pairWith(id: string, incoming: string): PairRow {
  return {
    pair_id: id,
    source: "golden",
    kind: "fact",
    existing: EXISTING_TEXT,
    incoming,
    similarity: null,
    old_label: null,
  };
}

async function noiseRows(embedder: HashEmbedder, count: number): Promise<NoiseRow[]> {
  const rows: NoiseRow[] = [];

  for (let index = 0; index < count; index += 1) {
    const topic = TOPICS[index % TOPICS.length] as string;
    const content = `Entry ${index} covers ${topic} notes number ${index * 7} with ${topic} details.`;

    rows.push({
      id: `noise-${index}`,
      kind: "fact",
      content,
      vector: await embedder.embed(content),
      recall_count: 0,
      created_at: "2026-10-01T00:00:00.000Z",
      lifecycle: "live",
    });
  }

  return rows;
}

function relationFor(pick: string) {
  return {
    relation: (_question: unknown, request: { state: unknown }) => {
      const state = request.state as { existing?: string; incoming?: unknown };
      const matches =
        state.existing?.includes("nightly backup job") === true &&
        JSON.stringify(state.incoming).includes("nightly backup job");

      return choiceAnswer(matches ? pick : "distinct", 0.9);
    },
  };
}

async function seededHarness(pick: string, noiseCount = 20) {
  const embedder = new HashEmbedder(DIM);
  const harness = await openEval({ embedder, jev: stubJev(relationFor(pick)) });
  const noise = await noiseRows(embedder, noiseCount);

  harnesses.push(harness);
  await seedNoise(harness, noise);

  return { harness, noise };
}

describe("v2 save replay", () => {
  it("buckets a stubbed restatement as caught", async () => {
    const { harness } = await seededHarness("restatement");
    const base = await harness.store.records.countRows();
    const row = await replayPair(
      harness,
      pairWith("p-restatement", "The nightly backup job copies the database dump to cold storage after midnight."),
      base,
    );

    expect(row.bucket).toBe("caught");
    expect(row.found_existing).toBe(true);
    expect(row.relation_pick).toBe("restatement");
    expect(row.raw_outcome).toMatch(/^reinforced:reinforce:/);
    expect(row.system).toBe("v2");
    expect(row.elapsed_ms).toBeGreaterThanOrEqual(0);
    expect(row.degraded).toBeNull();
    expect(await harness.store.records.countRows()).toBe(base);
  });

  it("copies each neighbour's raw answers onto the row and marks the seed", async () => {
    const { harness } = await seededHarness("restatement");
    const base = await harness.store.records.countRows();
    const row = await replayPair(
      harness,
      pairWith("p-raw", "The nightly backup job copies the database dump to cold storage after midnight."),
      base,
    );
    const seeds = row.neighbours.filter((neighbour) => neighbour.is_seed);

    expect(row.neighbours.length).toBeGreaterThan(1);
    expect(seeds).toHaveLength(1);
    expect(seeds[0]?.relation_pick).toBe("restatement");
    expect(seeds[0]?.relation_confidence).toBeCloseTo(0.9);
    expect(seeds[0]?.source).toBe("v2");
    expect(row.neighbours.filter((neighbour) => !neighbour.is_seed).every((neighbour) => neighbour.id.startsWith("noise-"))).toBe(true);
    expect(typeof seeds[0]?.similarity).toBe("number");
    expect(seeds[0]).toHaveProperty("new_covers_old");
    expect(seeds[0]).toHaveProperty("same_subject");
  });

  it("buckets a stubbed replacement as flagged", async () => {
    const { harness } = await seededHarness("replacement");
    const base = await harness.store.records.countRows();
    const row = await replayPair(
      harness,
      pairWith("p-replacement", "The nightly backup job now copies the database dump to cold storage before midnight."),
      base,
    );

    expect(row.bucket).toBe("flagged");
    expect(row.found_existing).toBe(true);
    expect(row.relation_pick).toBe("replacement");
    expect(row.raw_outcome).toMatch(/^approved:supersede:/);
    expect(await harness.store.records.countRows()).toBe(base);
  });

  it("buckets a stubbed distinct as new", async () => {
    const { harness } = await seededHarness("distinct");
    const base = await harness.store.records.countRows();
    const row = await replayPair(
      harness,
      pairWith("p-distinct", "The nightly backup job emails the on-call engineer when the dump grows past a limit."),
      base,
    );

    expect(row.bucket).toBe("new");
    expect(row.relation_pick).toBe("distinct");
    expect(row.raw_outcome).toBe("approved");
    expect(await harness.store.records.countRows()).toBe(base);
  });

  it("returns every pair to base and writes one row per pair", async () => {
    const { harness } = await seededHarness("restatement");
    const out = join(mkdtempSync(join(tmpdir(), "eval-out-")), "save-v2.jsonl");
    const base = await harness.store.records.countRows();

    scratch.push(join(out, ".."));

    const rows = await runSave(
      harness,
      [
        pairWith("a", "The nightly backup job copies the database dump to cold storage after midnight."),
        pairWith("b", "Tomatoes need staking once the first fruit sets."),
        pairWith("c", "The nightly backup job copies the database dump to the cold storage bucket after midnight."),
      ],
      out,
    );

    expect(rows.map((row) => row.bucket)).toEqual(["caught", "new", "caught"]);
    expect(await harness.store.records.countRows()).toBe(base);
    expect(await harness.store.proposals.countRows()).toBe(0);
    expect(readFileSync(out, "utf8").trim().split("\n")).toHaveLength(3);
  });

  it("leaves a noise row untouched when a restatement is reinforced onto it", async () => {
    const embedder = new HashEmbedder(DIM);
    const harness = await openEval({
      embedder,
      jev: stubJev({
        relation: (_question: unknown, request: { state: unknown }) => {
          const state = request.state as { existing?: string };

          return choiceAnswer(state.existing?.startsWith("Entry 3 covers") === true ? "restatement" : "distinct", 0.9);
        },
      }),
    });
    const noise = await noiseRows(embedder, 12);

    harnesses.push(harness);
    await seedNoise(harness, noise);

    const base = await harness.store.records.countRows();
    const before = await harness.store.records.query().where("id = 'noise-3'").toArray();
    const row = await replayPair(harness, pairWith("p-noise", (noise[3] as NoiseRow).content), base);
    const after = await harness.store.records.query().where("id = 'noise-3'").toArray();

    expect(row.bucket).toBe("wrong_target");
    expect(row.raw_outcome).toContain("reinforced:reinforce:noise-3");
    expect(after[0]?.reinforcedCount).toBe(before[0]?.reinforcedCount);
    expect(after[0]?.lastReinforcedAt).toBe(before[0]?.lastReinforcedAt);
    await expect(replayPair(harness, pairWith("p-after", EXISTING_TEXT), base)).resolves.toBeDefined();
  });

  it("refuses to run when a background row carries a change", async () => {
    const { harness } = await seededHarness("restatement");
    const base = await harness.store.records.countRows();

    await harness.seed.records.bumpReinforcement("noise-1");

    await expect(replayPair(harness, pairWith("dirty", EXISTING_TEXT), base)).rejects.toThrow(/not at base/);
  });

  it("refuses to run when the store is not at base", async () => {
    const { harness } = await seededHarness("restatement");
    const base = await harness.store.records.countRows();

    await expect(replayPair(harness, pairWith("off", EXISTING_TEXT), base + 1)).rejects.toThrow(/not at base/);
  });

  it("marks a pair degraded when Jev is down and still returns the store to base", async () => {
    const embedder = new HashEmbedder(DIM);
    const jev = stubJev(relationFor("restatement"));
    const harness = await openEval({ embedder, jev });

    harnesses.push(harness);
    await seedNoise(harness, await noiseRows(embedder, 10));
    jev.failWith = new Error("rate limited");

    const base = await harness.store.records.countRows();
    const row = await replayPair(harness, pairWith("down", EXISTING_TEXT), base);

    expect(row.degraded).toBe("jev_unavailable");
    expect(row.bucket).toBe("new");
    expect(await harness.store.records.countRows()).toBe(base);
  });
});

function yesNo(probability: number) {
  return () => ({
    type: "choice",
    choice: probability >= 0.5 ? "yes" : "no",
    confidence: Math.max(probability, 1 - probability),
    probabilities: { yes: probability, no: 1 - probability },
  });
}

async function pairedHarness(options: { newCoversOld: number; sameSubject: number; foldAsk: boolean }) {
  const embedder = new HashEmbedder(DIM);
  const jev = stubJev({
    ...relationFor("follow_up"),
    new_covers_old: yesNo(options.newCoversOld),
    same_subject: yesNo(options.sameSubject),
  });
  const opened = await openEval({ embedder, jev });
  const harness = withProposalOptions(opened, { foldAsk: options.foldAsk });

  harnesses.push(opened);
  await seedNoise(harness, await noiseRows(embedder, 12));

  return harness;
}

describe("v2 save replay with covers asks", () => {
  const INCOMING = `${EXISTING_TEXT} It runs at 03:00.`;

  it("buckets a covers hit as asked with the covers marker and supersedes nothing", async () => {
    const harness = await pairedHarness({ newCoversOld: 0.95, sameSubject: 0.1, foldAsk: true });
    const base = await harness.store.records.countRows();
    const row = await replayPair(harness, pairWith("p-covers", INCOMING), base);

    expect(row.bucket).toBe("asked");
    expect(row.raw_outcome).toMatch(/^asked:covers:/);
    expect(row.answer?.end_state).toBe("no_reply");
    expect(await harness.store.records.countRows()).toBe(base);
    expect(await harness.store.proposals.countRows()).toBe(0);
  });

  it("buckets a held note as asked and removes the proposal row", async () => {
    const harness = await pairedHarness({ newCoversOld: 0.2, sameSubject: 0.9, foldAsk: true });
    const base = await harness.store.records.countRows();
    const row = await replayPair(harness, pairWith("p-asked", INCOMING), base);

    expect(row.bucket).toBe("asked");
    expect(row.raw_outcome).toMatch(/^asked:/);
    expect(await harness.store.records.countRows()).toBe(base);
    expect(await harness.store.proposals.countRows()).toBe(0);
  });

  it("saves as new when the ask is off", async () => {
    const harness = await pairedHarness({ newCoversOld: 0.2, sameSubject: 0.9, foldAsk: false });
    const base = await harness.store.records.countRows();
    const row = await replayPair(harness, pairWith("p-off", INCOMING), base);

    expect(row.bucket).toBe("new");
  });

  it("buckets a covers ask on a noise row as wrong_target and leaves the noise row live", async () => {
    const embedder = new HashEmbedder(DIM);
    const jev = stubJev({
      relation: () => choiceAnswer("follow_up", 0.9),
      same_subject: yesNo(0.1),
      new_covers_old: (_question: unknown, request: { state: unknown }) => {
        const state = request.state as { existing?: string };

        return yesNo(state.existing?.startsWith("Entry 3 covers") === true ? 0.95 : 0.1)();
      },
    });
    const opened = await openEval({ embedder, jev });
    const harness = withProposalOptions(opened, { foldAsk: true });
    const noise = await noiseRows(embedder, 12);

    harnesses.push(opened);
    await seedNoise(harness, noise);

    const base = await harness.store.records.countRows();
    const row = await replayPair(harness, pairWith("p-wrong", (noise[3] as NoiseRow).content), base);
    const after = await harness.store.records.query().where("id = 'noise-3'").toArray();

    expect(row.bucket).toBe("wrong_target");
    expect(row.raw_outcome).toBe("asked:covers:noise-3");
    expect(after[0]?.lifecycle).toBe("live");
    expect(after[0]?.supersededBy).toBeNull();
    await expect(replayPair(harness, pairWith("p-after", EXISTING_TEXT), base)).resolves.toBeDefined();
  });

  it("refuses to run when a proposal row remains", async () => {
    const harness = await pairedHarness({ newCoversOld: 0.2, sameSubject: 0.9, foldAsk: true });
    const base = await harness.store.records.countRows();

    await harness.judged.proposals.create({
      kind: "fact",
      headline: "left over",
      fields: { claim: "left over claim", rejected: null },
      scope: "eval",
      origin: "agent",
    });

    await expect(replayPair(harness, pairWith("dirty", EXISTING_TEXT), base)).rejects.toThrow(/not at base/);
  });
});

describe("bucket mapping", () => {
  it("maps asked proposals and marks a covers ask", () => {
    const asked = { status: "asked", flags: [], ask: { existing: { id: "mine" } } } as unknown as Proposal;
    const covers = { status: "asked", flags: ["covers:mine"], ask: { existing: { id: "mine" } } } as unknown as Proposal;

    expect(bucketOf(asked, "mine")).toBe("asked");
    expect(bucketOf(asked, "other")).toBe("wrong_target");
    expect(rawOutcomeOf(asked)).toBe("asked:mine");
    expect(bucketOf(covers, "mine")).toBe("asked");
    expect(rawOutcomeOf(covers)).toBe("asked:covers:mine");
    expect(degradedOf({ flags: ["pair_unavailable:x"] } as unknown as Proposal)).toBe("pair_unavailable");
  });

  function proposal(overrides: Partial<Proposal>): Proposal {
    return {
      status: "approved",
      suggestion: null,
      resolution: { recordId: "fresh" },
      flags: [],
      neighbours: [],
      ...overrides,
    } as unknown as Proposal;
  }

  it("treats a reinforce onto another record as wrong_target", () => {
    expect(bucketOf(proposal({ status: "reinforced", resolution: { recordId: "other" } }), "mine")).toBe(
      "wrong_target",
    );
    expect(bucketOf(proposal({ status: "reinforced", resolution: { recordId: "mine" } }), "mine")).toBe("caught");
  });

  it("treats a supersede suggestion on another record as wrong_target", () => {
    const suggestion = { action: "supersede", recordId: "other", source: "v2", reasonClass: null } as const;

    expect(bucketOf(proposal({ suggestion }), "mine")).toBe("wrong_target");
    expect(bucketOf(proposal({ suggestion: { ...suggestion, recordId: "mine" } }), "mine")).toBe("flagged");
  });

  it("reports relation_unavailable apart from jev_unavailable", () => {
    expect(degradedOf(proposal({ flags: ["relation_unavailable:x"] }))).toBe("relation_unavailable");
    expect(degradedOf(proposal({ flags: ["jev_unavailable"] }))).toBe("jev_unavailable");
    expect(degradedOf(proposal({ flags: [] }))).toBeNull();
  });
});

describe("store path guard", () => {
  it("accepts a path under the temp directory", () => {
    expect(assertTempPath(join(tmpdir(), "tabularium-eval-x"))).toBe(join(tmpdir(), "tabularium-eval-x"));
  });

  it("rejects the live store path and a traversal out of temp", () => {
    expect(() => assertTempPath("/var/lib/apps/tabularium-v2-store")).toThrow(/under/);
    expect(() => assertTempPath(join(tmpdir(), "..", "elsewhere"))).toThrow(/under/);
  });

  it("rejects a path whose symlinked parent inside temp points outside it", () => {
    const dir = mkdtempSync(join(tmpdir(), "tabularium-eval-link-"));

    try {
      symlinkSync("/usr/share", join(dir, "link"));

      expect(() => assertTempPath(join(dir, "link", "store"))).toThrow(/under/);
      expect(() => assertTempPath(join(dir, "link"))).toThrow(/under/);
      expect(assertTempPath(join(dir, "plain", "store"))).toBe(join(dir, "plain", "store"));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("fallback shape", () => {
  it("splits a decision in any label order", () => {
    const shape = fallbackShape("decision", "Decision: Use Lance. Rejected: Chroma. Why: it embeds.");

    expect(shape.claim).toBe("Use Lance.");
    expect(shape.why).toBe("it embeds.");
    expect(shape.rejected).toBe("Chroma.");
    expect(shape.headline).toBe("Use Lance");
  });

  it("keeps a fact whole and clips a long headline on a word boundary", () => {
    const content = `${"word ".repeat(40)}end`;
    const shape = fallbackShape("fact", content);

    expect(shape.claim).toBe(content.trim());
    expect(shape.headline.length).toBeLessThanOrEqual(100);
    expect(clipHeadline(content).endsWith("word")).toBe(true);
  });
});

describe("v2 search replay", () => {
  function corpus(): CorpusRow[] {
    const rows: Array<[string, string, string]> = [
      ["c1", "alpha", "Redis cache eviction uses allkeys-lru in production."],
      ["c2", "alpha", "Thumbnail workers restart after every release."],
      ["c3", "beta", "Redis cache eviction in beta is noeviction."],
      ["c4", "global", "Redis persistence is disabled everywhere."],
    ];

    return rows.map(([id, scope, content]) => ({
      id,
      kind: "fact",
      content,
      scope,
      lifecycle: "live",
      vector: [],
      recall_count: 2,
      created_at: "2026-09-01T00:00:00.000Z",
    }));
  }

  const question: QuestionRow = {
    question_id: "q1",
    query: "redis cache eviction",
    scope: "alpha",
    origin: "mined",
  };

  it("maps corpus rows onto legacy lines with the fallback split", () => {
    const line = toLegacyLine({ ...(corpus()[0] as CorpusRow), kind: "decision", content: "plain text" });

    expect(line).toMatchObject({
      legacyId: "c1",
      kind: "decision",
      lifecycle: "live",
      readCount: 2,
      oldSource: "v1",
      collapsedIds: [],
    });
  });

  it("returns scoped results for both arms and only calls Jev on the rerank arm", async () => {
    const embedder = new HashEmbedder(DIM);
    const harness = await openEval({ embedder, jev: stubJev() });
    const dir = mkdtempSync(join(tmpdir(), "eval-search-"));

    harnesses.push(harness);
    scratch.push(dir);
    await loadCorpus(harness, corpus());

    const plain = await runArm(harness.seed.recall, "v2_plain", [question], join(dir, "plain.jsonl"));
    const callsAfterPlain = harness.jev?.calls ?? 0;
    const rerank = await runArm(harness.judged.recall, "v2_rerank", [question], join(dir, "rerank.jsonl"));

    expect(callsAfterPlain).toBe(0);
    expect(harness.jev?.calls ?? 0).toBeGreaterThan(0);
    expect(plain.map((row) => row.record_id).sort()).toEqual(["c1", "c2", "c4"]);
    expect(rerank.map((row) => row.rank)).toEqual([1, 2, 3]);
    expect(plain.every((row) => row.arm === "v2_plain" && row.question_id === "q1")).toBe(true);
    expect(rerank.every((row) => row.arm === "v2_rerank")).toBe(true);
    expect(readFileSync(join(dir, "plain.jsonl"), "utf8").trim().split("\n")).toHaveLength(3);
  });
});

describe("one-pass replay of recorded replies", () => {
  const INCOMING = `${EXISTING_TEXT} It runs at 03:00 and we rejected hourly runs.`;
  const FOLD_NOTE = `${EXISTING_TEXT} It runs at 03:00 and hourly runs were rejected.`;

  function reply(overrides: Partial<AskReply>): AskReply {
    return {
      pair_id: "p",
      arm: "opus-fold_default-in_task",
      action: "fold",
      note: FOLD_NOTE,
      reason: null,
      error: null,
      ...overrides,
    };
  }

  async function askingHarness(foldAsk = true) {
    const harness = await pairedHarness({ newCoversOld: 0.2, sameSubject: 0.9, foldAsk });

    return { harness, base: await harness.store.records.countRows() };
  }

  async function expectAtBase(harness: EvalHarness, base: number) {
    expect(await harness.store.records.countRows()).toBe(base);
    expect(await harness.store.proposals.countRows()).toBe(0);
  }

  it("answers a fold on the same ask row and ends with the fold live and the seed superseded by it", async () => {
    const { harness, base } = await askingHarness();
    const row = await replayPair(harness, pairWith("p", INCOMING), base, reply({}));

    expect(row.bucket).toBe("asked");
    expect(row.answer?.replay).toBe("fold");
    expect(row.answer?.status).toBe("folded");
    expect(row.answer?.end_state).toBe("folded");
    expect(row.answer?.existing_live).toBe(false);
    expect(row.answer?.existing_superseded_by_fold).toBe(true);
    expect(row.answer?.fold_live).toBe(true);
    expect(row.answer?.ask_answer).toBe("fold");
    await expectAtBase(harness, base);
  });

  it("ends keep_both with both records live", async () => {
    const { harness, base } = await askingHarness();
    const kept = reply({ action: "keep_both", note: null, reason: "the existing note is about the schedule only" });
    const row = await replayPair(harness, pairWith("p", INCOMING), base, kept);

    expect(row.answer?.end_state).toBe("both_live");
    expect(row.answer?.ask_answer).toBe("keep_both");
    expect(row.answer?.existing_live).toBe(true);
    expect(row.answer?.held_live).toBe(true);
    await expectAtBase(harness, base);
  });

  it("replays continue as expiry and saves the held note as unanswered", async () => {
    const { harness, base } = await askingHarness();
    const row = await replayPair(harness, pairWith("p", INCOMING), base, reply({ action: "continue", note: null }));

    expect(row.answer?.replay).toBe("continue");
    expect(row.answer?.end_state).toBe("both_live");
    expect(row.answer?.ask_answer).toBe("unanswered");
    await expectAtBase(harness, base);
  });

  it("ends an asked pair with no recorded reply as no_reply and saves the held note as new", async () => {
    const { harness, base } = await askingHarness();
    const row = await replayPair(harness, pairWith("p", INCOMING), base);

    expect(row.bucket).toBe("asked");
    expect(row.answer?.replay).toBe("none");
    expect(row.answer?.end_state).toBe("no_reply");
    expect(row.answer?.ask_answer).toBe("unanswered");
    expect(row.answer?.held_live).toBe(true);
    expect(row.answer?.existing_live).toBe(true);
    await expectAtBase(harness, base);
  });

  it("keeps failed and records held_live false when the sweep did not save the held note", async () => {
    const { harness, base } = await askingHarness();
    const sweep = vi.spyOn(harness.judged.proposals, "expireAsks").mockResolvedValue([]);
    const row = await replayPair(harness, pairWith("p", INCOMING), base);

    sweep.mockRestore();
    expect(row.bucket).toBe("asked");
    expect(row.answer?.end_state).toBe("failed");
    expect(row.answer?.held_live).toBe(false);
    await expectAtBase(harness, base);
  });

  it("removes the fold record and the ask row after a fold answer that threw", async () => {
    const { harness, base } = await askingHarness();
    const answer = harness.judged.proposals.answer.bind(harness.judged.proposals);
    const failing = vi.spyOn(harness.judged.proposals, "answer").mockImplementation(async (id, raw) => {
      await answer(id, raw);

      throw new Error("connection dropped after the fold was written");
    });
    const row = await replayPair(harness, pairWith("p", INCOMING), base, reply({}));

    failing.mockRestore();
    expect(row.answer?.end_state).toBe("failed");
    await expectAtBase(harness, base);
    await replayPair(harness, pairWith("q", INCOMING), base, reply({ action: "keep_both", note: null, reason: "x" }));
    await expectAtBase(harness, base);
  });

  it("treats a recorded reply that errored as no reply", async () => {
    const { harness, base } = await askingHarness();
    const errored = reply({ action: null, note: null, error: "JudgeUnavailable" });
    const row = await replayPair(harness, pairWith("p", INCOMING), base, errored);

    expect(row.answer?.end_state).toBe("no_reply");
    await expectAtBase(harness, base);
  });

  it("records a rejected answer as failed and still returns to base", async () => {
    const { harness, base } = await askingHarness();
    const blank = reply({ action: "keep_both", note: null, reason: "   " });
    const row = await replayPair(harness, pairWith("p", INCOMING), base, blank);

    expect(row.answer?.end_state).toBe("failed");
    expect(row.answer?.error).toMatch(/^400/);
    await expectAtBase(harness, base);
  });

  it("carries no answer on a pair the server did not ask", async () => {
    const { harness, base } = await askingHarness(false);
    const row = await replayPair(harness, pairWith("p", INCOMING), base, reply({}));

    expect(row.bucket).toBe("new");
    expect(row.answer).toBeNull();
    await expectAtBase(harness, base);
  });

  it("writes one row per pair, answers each pair in the same pass and resumes past finished pairs", async () => {
    const { harness, base } = await askingHarness();
    const dir = mkdtempSync(join(tmpdir(), "eval-onepass-"));
    const out = join(dir, "save-v2.jsonl");
    const replies = loadReplies([reply({ pair_id: "a" }), reply({ pair_id: "c", action: "continue", note: null })]);
    const pairs = [pairWith("a", INCOMING), pairWith("b", INCOMING), pairWith("c", INCOMING)];

    scratch.push(dir);

    const rows = await runSave(harness, pairs, out, new Set(), replies);
    const resumed = await runSave(harness, [pairs[0] as PairRow, pairs[2] as PairRow], out, new Set(["a"]), replies);

    expect(rows.map((row) => row.answer?.end_state)).toEqual(["folded", "no_reply", "both_live"]);
    expect(resumed.map((row) => row.pair_id)).toEqual(["c"]);
    expect(readFileSync(out, "utf8").trim().split("\n")).toHaveLength(4);
    await expectAtBase(harness, base);
  });
});

describe("reply loading", () => {
  it("keeps the last row per pair for the arm and treats errors as no reply", () => {
    const row = (overrides: Partial<AskReply>): AskReply => ({
      pair_id: "p",
      arm: "opus-fold_default-in_task",
      action: "fold",
      note: "n",
      reason: null,
      error: null,
      ...overrides,
    });
    const replies = loadReplies([
      row({ pair_id: "a", action: null, note: null, error: "down" }),
      row({ pair_id: "a", action: "keep_both", note: null, reason: "r" }),
      row({ pair_id: "b", action: "fold" }),
      row({ pair_id: "b", action: null, note: null, error: "down" }),
      row({ pair_id: "c", arm: "sonnet-fold_default-in_task" }),
    ]);

    expect([...replies.keys()].sort()).toEqual(["a", "b"]);
    expect(replayOf(replies.get("a"))).toBe("keep_both");
    expect(replayOf(replies.get("b"))).toBe("none");
    expect(replayOf(undefined)).toBe("none");
  });

  it("reads replies from a second file and lets the primary file win on a shared pair", () => {
    const dir = mkdtempSync(join(tmpdir(), "eval-replies-"));
    const line = (pair_id: string, action: string) =>
      JSON.stringify({ pair_id, arm: "opus-fold_default-in_task", action, note: "n", reason: "r", error: null });

    scratch.push(dir);
    writeFileSync(join(dir, "primary.jsonl"), `${line("a", "keep_both")}\n`);
    writeFileSync(join(dir, "extra.jsonl"), `${line("a", "fold")}\n${line("b", "fold")}\n`);

    const replies = readReplies(join(dir, "primary.jsonl"), join(dir, "extra.jsonl"));

    expect([...replies.keys()].sort()).toEqual(["a", "b"]);
    expect(replayOf(replies.get("a"))).toBe("keep_both");
    expect(replayOf(replies.get("b"))).toBe("fold");
    expect([...readReplies(join(dir, "primary.jsonl"), undefined).keys()]).toEqual(["a"]);
  });
});
