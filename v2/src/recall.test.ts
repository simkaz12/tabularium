import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { selectWhere } from "./store.ts";
import { makeHarness, seedLegacy, seedRecord, stubJev } from "./testing.ts";
import type { Harness } from "./testing.ts";
import type { JevRequest, ReadLogEntry } from "./domain.ts";
import { DEFAULT_LIMIT, TASK_MAX_CHARS } from "./recall.ts";

let harness: Harness;

beforeEach(async () => {
  harness = await makeHarness();
});

afterEach(() => {
  harness.cleanup();
});

const repl = {
  headline: "repl readline crashes",
  claim: "repl aborts with a readline error",
  why: "fixtures need repl",
};

describe("recall", () => {
  it("returns legacy and v2 together with v2 first on a tie", async () => {
    const legacy = await seedLegacy(harness, repl);
    const current = await seedRecord(harness, repl);

    const { results } = await harness.services.recall.recall({ q: "repl readline error", scope: "project-a" });

    expect(results.map((entry) => [entry.id, entry.source])).toEqual([
      [current.id, "v2"],
      [legacy.id, "legacy"],
    ]);
    expect(results[0]?.score).toBe(results[1]?.score);
    expect(results[0]?.headline).toBe(repl.headline);
    expect(results[0]?.fields?.claim).toBe(repl.claim);
  });

  it("writes one read_log row per returned result", async () => {
    await seedLegacy(harness, repl);
    await seedRecord(harness, repl);
    await seedRecord(harness, { headline: "cats", claim: "unrelated cats", why: "none", scope: "project-a" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error", scope: "project-a", limit: 2 });

    const log = await selectWhere<ReadLogEntry>(harness.store.readLog, undefined);
    expect(results.length).toBe(2);
    expect(log.length).toBe(2);
    expect(log.map((entry) => entry.resultId).sort()).toEqual(results.map((entry) => entry.id).sort());
    expect(log.every((entry) => entry.query === "repl readline error")).toBe(true);
    expect(log.map((entry) => entry.source).sort()).toEqual(results.map((entry) => entry.source).sort());
  });

  it("logs nothing when nothing is returned", async () => {
    const { results } = await harness.services.recall.recall({ q: "nothing here" });

    expect(results).toEqual([]);
    expect(await harness.store.readLog.countRows()).toBe(0);
  });

  it("excludes superseded and retired records", async () => {
    await seedRecord(harness, { ...repl, lifecycle: "superseded" });
    await seedRecord(harness, { ...repl, lifecycle: "retired" });
    const live = await seedRecord(harness, repl);

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results.map((entry) => entry.id)).toEqual([live.id]);
  });

  it("excludes retired and superseded legacy rows", async () => {
    await seedLegacy(harness, { ...repl, lifecycle: "retired", supersededBy: "rec-x" });
    await seedLegacy(harness, { ...repl, lifecycle: "superseded" });
    const live = await seedLegacy(harness, repl);

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results.map((entry) => entry.id)).toEqual([live.id]);
  });

  it("filters by scope, keeping global, and by kind", async () => {
    const mine = await seedRecord(harness, { ...repl, scope: "project-a" });
    const shared = await seedRecord(harness, { ...repl, scope: "global" });
    await seedRecord(harness, { ...repl, scope: "project-b" });
    const decision = await seedRecord(harness, { ...repl, scope: "project-a", kind: "decision" });

    const scoped = await harness.services.recall.recall({ q: "repl readline error", scope: "project-a" });
    expect(scoped.results.map((entry) => entry.id).sort()).toEqual([decision.id, mine.id, shared.id].sort());

    const decisions = await harness.services.recall.recall({
      q: "repl readline error",
      scope: "project-a",
      kind: "decision",
    });
    expect(decisions.results.map((entry) => entry.id)).toEqual([decision.id]);
  });

  it("returns builder and unlabelled rows for audience builder and drops other roles", async () => {
    const builder = await seedRecord(harness, { ...repl, audience: "builder" });
    const unlabelled = await seedRecord(harness, { ...repl, audience: null });
    const reviewer = await seedRecord(harness, { ...repl, audience: "reviewer" });
    const legacyBuilder = await seedLegacy(harness, { ...repl, audience: "builder" });
    const legacyNavigator = await seedLegacy(harness, { ...repl, audience: "navigator" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error", audience: "builder" });
    const ids = results.map((entry) => entry.id);

    expect(ids).toEqual(expect.arrayContaining([builder.id, unlabelled.id, legacyBuilder.id]));
    expect(ids).not.toContain(reviewer.id);
    expect(ids).not.toContain(legacyNavigator.id);
    expect(results.filter((entry) => entry.source === "v2").length).toBe(2);

    const everyone = await harness.services.recall.recall({ q: "repl readline error" });
    expect(everyone.results.length).toBe(5);
  });

  it("carries audience, reader and stage and drops why, rejected and empty keys", async () => {
    await seedRecord(harness, { ...repl, why: null, audience: "decider", reader: "ruling", stage: "design" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results[0]).toMatchObject({
      audience: "decider",
      reader: "ruling",
      stage: "design",
      fields: { claim: repl.claim },
    });

    for (const row of results) {
      expect(row.fields).not.toHaveProperty("why");
      expect(row.fields).not.toHaveProperty("rejected");
      expect(Object.values(row).some((value) => value === null || value === "")).toBe(false);
    }

    expect(results[0]).not.toHaveProperty("where");
    expect(results[0]).not.toHaveProperty("defectStatus");
    expect(results[0]).not.toHaveProperty("takenBy");
  });

  it("returns defects with location and takenBy, open taken and closed alike", async () => {
    const base = {
      kind: "defect" as const,
      headline: "export drops rows",
      claim: "export writes only 500 rows",
      where: "app/Jobs/SyncReport.php",
      trigger: "more than 500 rows",
      leftBecause: "outside the ticket",
    };
    const open = await seedRecord(harness, { ...base, defectStatus: "open" });
    const taken = await seedRecord(harness, {
      ...base,
      defectStatus: "taken",
      takenThreadId: "thread-2",
      takenLink: "t3-thread://thread-2",
      takenAt: "2026-11-01T09:00:00.000Z",
    });
    const closed = await seedRecord(harness, { ...base, defectStatus: "closed", closedBy: "p-1" });
    await seedRecord(harness, { ...repl, headline: "export drops rows", claim: "export writes only 500 rows" });

    const { results } = await harness.services.recall.recall({ q: "export rows", kind: "defect" });
    const byId = new Map(results.map((entry) => [entry.id, entry]));

    expect(results.length).toBe(3);
    expect(byId.get(open.id)).toMatchObject({
      where: "app/Jobs/SyncReport.php",
      trigger: "more than 500 rows",
      leftBecause: "outside the ticket",
      defectStatus: "open",
    });
    expect(byId.get(open.id)).not.toHaveProperty("takenBy");
    expect(byId.get(taken.id)).toMatchObject({
      defectStatus: "taken",
      takenBy: { threadId: "thread-2", link: "t3-thread://thread-2", at: "2026-11-01T09:00:00.000Z" },
    });
    expect(byId.get(closed.id)?.defectStatus).toBe("closed");
  });

  it("finds a defect by its location text", async () => {
    const shared = {
      kind: "defect" as const,
      headline: "rows dropped",
      claim: "rows missing",
      trigger: "big report",
      defectStatus: "open" as const,
    };
    await seedRecord(harness, { ...shared, where: "app/Models/Invoice.php" });
    const defect = await seedRecord(harness, { ...shared, where: "app/Jobs/SyncReport.php" });
    await seedRecord(harness, { ...repl });

    const { results } = await harness.services.recall.recall({ q: "SyncReport", kind: "defect" });

    expect(results.length).toBe(2);
    expect(results[0]?.id).toBe(defect.id);
  });

  it("rejects an empty query, a bad kind, a bad audience and a bad limit", async () => {
    await expect(harness.services.recall.recall({ q: "x", audience: "wizard" })).rejects.toMatchObject({
      status: 400,
    });
    await expect(harness.services.recall.recall({ q: "  " })).rejects.toMatchObject({ status: 400 });
    await expect(harness.services.recall.recall({ q: "x", kind: "note" })).rejects.toMatchObject({ status: 400 });
    await expect(harness.services.recall.recall({ q: "x", limit: 0 })).rejects.toMatchObject({ status: 400 });
  });

  it("returns 17 rows by default from 20 candidates", async () => {
    for (let index = 0; index < 20; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(DEFAULT_LIMIT).toBe(17);
    expect(results.length).toBe(17);
  });

  it("finds a task-only record only when a task is given", async () => {
    await seedRecord(harness, { headline: "repl readline crashes", claim: "repl readline error", why: "w" });
    const taskOnly = await seedRecord(harness, {
      headline: "zebra quartz migration",
      claim: "zebra quartz rule",
      why: "w",
    });

    const without = await harness.services.recall.recall({ q: "repl readline error", limit: 1 });
    const withTask = await harness.services.recall.recall({
      q: "repl readline error",
      task: "zebra quartz migration zebra quartz rule",
      limit: 2,
    });

    expect(without.results.map((entry) => entry.id)).not.toContain(taskOnly.id);
    expect(withTask.results.map((entry) => entry.id)).toContain(taskOnly.id);
  });

  it("cuts a long task to 300 characters before embedding", async () => {
    await seedRecord(harness, repl);
    const seen: string[] = [];
    const embed = harness.embedder.embed.bind(harness.embedder);
    harness.embedder.embed = async (text: string) => {
      seen.push(text);

      return embed(text);
    };

    await harness.services.recall.recall({ q: "repl readline error", task: "a".repeat(1000) });

    expect(seen).toContain("a".repeat(TASK_MAX_CHARS));
    expect(seen.every((text) => text.length <= TASK_MAX_CHARS)).toBe(true);
  });

  it("treats a blank task as absent", async () => {
    await seedRecord(harness, repl);
    const seen: string[] = [];
    const embed = harness.embedder.embed.bind(harness.embedder);
    harness.embedder.embed = async (text: string) => {
      seen.push(text);

      return embed(text);
    };

    await harness.services.recall.recall({ q: "repl readline error", task: "   " });

    expect(seen).toEqual(["repl readline error"]);
  });
});

const scoreAnswer = (score: number) => ({ type: "score", score, confidence: 0.9, legend: {}, probabilities: {} });

describe("recall floor, claim slots and log", () => {
  it("drops a jev-scored row under the floor", async () => {
    const stub = stubJev({
      rerank: (_q, req: JevRequest) => scoreAnswer((req.state as any).record.headline.includes("weak") ? 0.5 : 2),
    });
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    await seedRecord(harness, { headline: "repl readline weak", claim: "repl readline error", why: "a" });
    const strong = await seedRecord(harness, { headline: "repl readline strong", claim: "repl readline error", why: "b" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results.map((entry) => entry.id)).toEqual([strong.id]);
  });

  it("gives claims to exactly the first five rows at or above the line", async () => {
    const stub = stubJev({ rerank: () => scoreAnswer(2) });
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });

    for (let index = 0; index < 7; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline", limit: 17 });

    expect(results.length).toBe(7);
    expect(results.map((entry) => entry.fields !== undefined)).toEqual([
      true,
      true,
      true,
      true,
      true,
      false,
      false,
    ]);
  });

  it("returns a row between the floor and the line without a claim", async () => {
    const stub = stubJev({ rerank: () => scoreAnswer(0.8) });
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    await seedRecord(harness, repl);

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results.length).toBe(1);
    expect(results[0]).not.toHaveProperty("fields");
    expect(results[0]?.score).toBe(0.8);
  });

  it("keeps rows unfiltered with claims on the first five when every jev call fails", async () => {
    const stub = stubJev();
    stub.failWith = new Error("down");
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });

    for (let index = 0; index < 7; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(results.length).toBe(7);
    expect(results.filter((entry) => entry.fields !== undefined).length).toBe(5);
    expect(results.slice(0, 5).every((entry) => entry.fields !== undefined)).toBe(true);
  });

  it("keeps rows unfiltered with claims on the first five when jev is off", async () => {
    for (let index = 0; index < 7; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(results.length).toBe(7);
    expect(results.slice(0, 5).every((entry) => entry.fields !== undefined)).toBe(true);
    expect(results.slice(5).some((entry) => entry.fields !== undefined)).toBe(false);
  });

  it("drops candidates beyond the rerank pool when jev ran", async () => {
    const judged = new Set<string>();
    const stub = stubJev({
      rerank: (_q, req: JevRequest) => {
        const headline = (req.state as any).record.headline as string;
        judged.add(headline);

        return scoreAnswer(headline.endsWith(" 0") || headline.endsWith(" 1") ? 2 : 0.5);
      },
    });
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });

    for (let index = 0; index < 30; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline", limit: 25 });

    expect(judged.size).toBe(20);
    expect(results.length).toBeGreaterThan(0);
    expect(results.length).toBeLessThan(20);
    expect(results.every((entry) => judged.has(entry.headline))).toBe(true);
    expect(results.every((entry) => entry.score >= 0.6)).toBe(true);
  });

  it("logs the rows that survive the floor and the limit", async () => {
    const stub = stubJev({
      rerank: (_q, req: JevRequest) => scoreAnswer((req.state as any).record.headline.includes("weak") ? 0.5 : 2),
    });
    harness.cleanup();
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    await seedRecord(harness, { headline: "repl readline weak", claim: "repl readline error", why: "a" });
    await seedRecord(harness, { headline: "repl readline one", claim: "repl readline error", why: "b" });
    await seedRecord(harness, { headline: "repl readline two", claim: "repl readline error", why: "c" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error", limit: 1 });
    const log = await selectWhere<ReadLogEntry>(harness.store.readLog, undefined);

    expect(results.length).toBe(1);
    expect(log.map((entry) => entry.resultId)).toEqual(results.map((entry) => entry.id));
  });
});
