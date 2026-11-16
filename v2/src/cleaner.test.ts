import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Cleaner, nextRunAfter, previousRunAtOrBefore, scheduleAt, scheduleWeekly } from "./cleaner.ts";
import { loadConfig } from "./config.ts";
import { start } from "./main.ts";
import { insert, selectWhere } from "./store.ts";
import { choiceAnswer, makeHarness, proposalBody, seedLegacy, seedRecord, stubJev } from "./testing.ts";
import type { Harness } from "./testing.ts";
import type { ReadLogEntry } from "./domain.ts";

const NOW = new Date("2026-11-01T10:00:00.000Z");
const daysAgo = (days: number) => new Date(NOW.getTime() - days * 86_400_000).toISOString();

let harness: Harness | null = null;

afterEach(() => {
  vi.useRealTimers();
  harness?.cleanup();
  harness = null;
});

async function setup() {
  harness = await makeHarness();
  harness.clock.value = NOW.toISOString();

  return harness;
}

function cleanerFor(h: Harness, jev: ReturnType<typeof stubJev> | null, extra: object = {}) {
  return new Cleaner({
    store: h.store,
    proposals: h.services.proposals,
    jev,
    model: "jev-test",
    now: () => NOW,
    log: () => undefined,
    ...extra,
  });
}

const unit = (angle: number) => {
  const vector = new Array<number>(64).fill(0);
  vector[0] = Math.cos(angle);
  vector[1] = Math.sin(angle);

  return vector;
};

const NEAR_DUPLICATE_ANGLE = Math.acos(0.95);

async function seedPortPair(h: Harness, angle: number) {
  const older = await seedRecord(h, {
    createdAt: daysAgo(90),
    vector: unit(0),
    headline: "old port",
    claim: "port 8787",
    why: "a",
  });
  const newer = await seedRecord(h, {
    createdAt: daysAgo(40),
    vector: unit(angle),
    headline: "new port",
    claim: "port 8789",
    why: "b",
  });

  return { older, newer };
}

const shelve = () => ({
  type: "score",
  score: 0.2,
  confidence: 0.9,
  legend: {},
  probabilities: { "0": 0.9, "1": 0.05, "2": 0.03, "3": 0.02 },
});

const keep = () => ({
  type: "score",
  score: 2.9,
  confidence: 0.9,
  legend: {},
  probabilities: { "0": 0.01, "1": 0.02, "2": 0.07, "3": 0.9 },
});

async function seedStale(h: Harness, count: number, age = 60) {
  const ids: string[] = [];

  for (let index = 0; index < count; index += 1) {
    const record = await seedRecord(h, {
      headline: `stale topic number${index} alpha${index * 7} beta${index * 13}`,
      claim: `unique claim ${index} gamma${index * 17} delta${index * 19}`,
      why: `why ${index}`,
      createdAt: daysAgo(age),
      approvedAt: daysAgo(age),
    });
    ids.push(record.id);
  }

  return ids;
}

describe("cleaner", () => {
  it("caps a run at five proposals", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    await seedStale(h, 8);

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed.length).toBe(5);
    expect(report.proposed.every((entry) => entry.origin === "cleaner")).toBe(true);
    expect(report.proposed.every((entry) => entry.target?.action === "retire")).toBe(true);
    expect((await h.services.proposals.list("pending", null)).length).toBe(5);
  });

  it("never writes to the records table", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    await seedStale(h, 3);
    await seedLegacy(h);
    const spies = ["add", "mergeInsert", "update", "delete"].map((method) =>
      vi.spyOn(h.store.records, method as "add"),
    );

    await cleanerFor(h, stub).run();

    for (const spy of spies) {
      expect(spy).not.toHaveBeenCalled();
    }

    expect(await h.store.records.countRows()).toBe(3);
  });

  it("builds a retire tile for a defect record that keeps its defect fields and no why", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    const defect = await seedRecord(h, {
      kind: "defect",
      headline: "export drops rows",
      claim: "export writes only 500 rows",
      where: "app/Jobs/SyncReport.php",
      trigger: "more than 500 rows",
      leftBecause: "outside the ticket",
      defectStatus: "open",
      createdAt: daysAgo(60),
      approvedAt: daysAgo(60),
    });

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({ recordId: defect.id, action: "retire" });
    expect(report.proposed[0]?.fields).toMatchObject({
      claim: "export writes only 500 rows",
      where: "app/Jobs/SyncReport.php",
      trigger: "more than 500 rows",
      leftBecause: "outside the ticket",
    });
    expect(report.proposed[0]?.fields).not.toHaveProperty("why");
    expect(report.proposed[0]?.provenance.stage).toBe("");
  });

  it("skips records younger than thirty days", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    await seedStale(h, 2, 10);

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed).toEqual([]);
    expect(stub.calls.length).toBe(0);
  });

  it("skips a target that has a pending proposal and one rejected within 90 days", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    const [pendingId, rejectedId, freeId] = await seedStale(h, 3);
    const pending = await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: pendingId as string, source: "v2", action: "retire", reason: "x" },
      }),
    );
    const rejected = await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: rejectedId as string, source: "v2", action: "retire", reason: "x" },
      }),
    );
    await h.services.proposals.reject(rejected.id, { reason: "keep it" });

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed.map((entry) => entry.target?.recordId)).toEqual([freeId]);
    expect(pending.status).toBe("pending");
  });

  it("proposes again once a rejection is older than 90 days", async () => {
    const stub = stubJev({ relevance: shelve });
    const h = await setup();
    const [id] = await seedStale(h, 1);
    const old = await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: id as string, source: "v2", action: "retire", reason: "x" },
      }),
    );
    h.clock.value = daysAgo(120);
    await h.services.proposals.reject(old.id, { reason: "no" });
    h.clock.value = NOW.toISOString();

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed.length).toBe(1);
  });

  it("does not retire records the relevance check keeps", async () => {
    const stub = stubJev({ relevance: keep });
    const h = await setup();
    await seedStale(h, 3);

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed).toEqual([]);
    expect(report.jevCalls).toBe(3);
  });

  it("proposes a promotion for a legacy record returned three times in 30 days", async () => {
    const h = await setup();
    const hot = await seedLegacy(h, { headline: "popular repl note", claim: "repl readline", why: "fixtures" });
    const cold = await seedLegacy(h, { headline: "rarely read", claim: "other", why: "none" });
    const reads = (resultId: string, count: number, age: number) =>
      Array.from({ length: count }, (_, index) => ({
        id: `${resultId}-${age}-${index}`,
        ts: daysAgo(age),
        query: "q",
        scope: "project-a",
        resultId,
        source: "legacy",
        rank: 0,
        score: 1,
      }));
    await insert(h.store.readLog, [...reads(hot.id, 3, 5), ...reads(cold.id, 2, 5), ...reads(cold.id, 5, 45)]);

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]).toMatchObject({
      origin: "promotion",
      headline: "popular repl note",
      target: { recordId: hot.id, source: "legacy", action: "promote", recordHeadline: "popular repl note" },
    });

    const again = await cleanerFor(h, null).run();
    expect(again.proposed).toEqual([]);
  });

  it("counts recall log rows written by real recalls", async () => {
    const h = await setup();
    const hot = await seedLegacy(h, { headline: "repl readline", claim: "repl readline error", why: "fixtures" });

    for (let index = 0; index < 3; index += 1) {
      await h.services.recall.recall({ q: "repl readline error", scope: "project-a" });
    }

    const log = await selectWhere<ReadLogEntry>(h.store.readLog, undefined);
    expect(log.length).toBe(3);

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.map((entry) => entry.target?.recordId)).toEqual([hot.id]);
  });

  it("does not propose promoting a legacy record already in v2", async () => {
    const h = await setup();
    const hot = await seedLegacy(h);
    await seedRecord(h, { legacyId: hot.id });
    await insert(
      h.store.readLog,
      Array.from({ length: 3 }, (_, index) => ({
        id: `r${index}`,
        ts: daysAgo(1),
        query: "q",
        scope: null,
        resultId: hot.id,
        source: "legacy",
        rank: 0,
        score: 1,
      })),
    );

    expect((await cleanerFor(h, null).run()).proposed).toEqual([]);
  });

  it("proposes a merge retiring the older near-duplicate", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    const older = await seedRecord(h, { ...text, createdAt: daysAgo(90) });
    const newer = await seedRecord(h, { ...text, createdAt: daysAgo(40) });

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: older.id,
      action: "merge",
      replacementId: newer.id,
      recordHeadline: "dup headline",
      replacementHeadline: "dup headline",
    });
  });

  it("never pairs a record with a near-duplicate in another scope", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    await seedRecord(h, { ...text, scope: "alpha", createdAt: daysAgo(90) });
    await seedRecord(h, { ...text, scope: "beta", createdAt: daysAgo(40) });

    expect((await cleanerFor(h, null).run()).proposed).toEqual([]);
  });

  it("pairs a scoped record with a newer keeper in global but not the reverse", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    const scoped = await seedRecord(h, { ...text, scope: "alpha", createdAt: daysAgo(90) });
    const keeper = await seedRecord(h, { ...text, scope: "global", createdAt: daysAgo(40) });

    const forward = await cleanerFor(h, null).run();

    expect(forward.proposed.map((entry) => entry.target)).toMatchObject([
      { recordId: scoped.id, action: "merge", replacementId: keeper.id },
    ]);

    harness?.cleanup();
    const reverse = await setup();
    await seedRecord(reverse, { ...text, scope: "global", createdAt: daysAgo(90) });
    await seedRecord(reverse, { ...text, scope: "alpha", createdAt: daysAgo(40) });

    expect((await cleanerFor(reverse, null).run()).proposed).toEqual([]);
  });

  it("proposes a supersede when contradiction says the newer replaces the older", async () => {
    const stub = stubJev({
      contradiction: () => choiceAnswer("changed"),
      relevance: keep,
    });
    const h = await setup();
    const older = await seedRecord(h, {
      createdAt: daysAgo(90),
      vector: unit(0),
      headline: "old port",
      claim: "port 8787",
      why: "a",
    });
    const newer = await seedRecord(h, {
      createdAt: daysAgo(40),
      vector: unit(0.6),
      headline: "new port",
      claim: "port 8789",
      why: "b",
    });

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: older.id,
      action: "supersede",
      replacementId: newer.id,
      reason: "changed",
    });
    expect(report.proposed[0]?.fields.claim).toContain("jev confidence 0.90");
  });

  it("leaves coexisting pairs alone", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("coexist"), relevance: keep });
    const h = await setup();
    await seedRecord(h, { createdAt: daysAgo(90), vector: unit(0) });
    await seedRecord(h, { createdAt: daysAgo(40), vector: unit(0.6) });

    expect((await cleanerFor(h, stub).run()).proposed).toEqual([]);
  });

  it("keeps asking after jev fails and still returns", async () => {
    const stub = stubJev();
    stub.failWith = new Error("down");
    const h = await setup();
    await seedStale(h, 3);

    const report = await cleanerFor(h, stub).run();

    expect(report.proposed).toEqual([]);
    expect(report.jevCalls).toBe(3);
  });

  it("turns a 0.95 changed pair into a supersede above the floor and a merge below it", async () => {
    const confident = await setup();
    const { older, newer } = await seedPortPair(confident, NEAR_DUPLICATE_ANGLE);
    const stub = stubJev({ contradiction: () => choiceAnswer("changed", 0.8), relevance: keep });

    const report = await cleanerFor(confident, stub).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: older.id,
      action: "supersede",
      replacementId: newer.id,
      reason: "changed",
    });
    expect(report.proposed[0]?.fields.claim).toContain("jev confidence 0.80");

    harness?.cleanup();
    const unsure = await setup();
    const pair = await seedPortPair(unsure, NEAR_DUPLICATE_ANGLE);
    const doubtful = stubJev({ contradiction: () => choiceAnswer("changed", 0.5), relevance: keep });

    const merged = await cleanerFor(unsure, doubtful).run();

    expect(merged.proposed.length).toBe(1);
    expect(merged.proposed[0]?.target).toMatchObject({
      recordId: pair.older.id,
      action: "merge",
      replacementId: pair.newer.id,
    });
    expect(merged.proposed[0]?.target?.reason).toContain("0.50");
  });

  it("merges a 0.95 pair when jev is not configured", async () => {
    const h = await setup();
    const { older, newer } = await seedPortPair(h, NEAR_DUPLICATE_ANGLE);

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: older.id,
      action: "merge",
      replacementId: newer.id,
    });
  });

  it("merges a 0.95 pair when the jev budget is spent", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("changed", 0.9), relevance: keep });
    const h = await setup();
    await seedPortPair(h, NEAR_DUPLICATE_ANGLE);

    const report = await cleanerFor(h, stub, { maxJevCalls: 0 }).run();

    expect(report.proposed.map((entry) => entry.target?.action)).toEqual(["merge"]);
    expect(stub.calls.length).toBe(0);
  });

  it("leaves a 0.95 pair out when the contradiction check fails", async () => {
    const stub = stubJev();
    stub.failWith = new Error("down");
    const h = await setup();
    await seedPortPair(h, NEAR_DUPLICATE_ANGLE);

    expect((await cleanerFor(h, stub).run()).proposed).toEqual([]);
  });

  it("applies the stored confidence floor to supersede tiles", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("changed", 0.8), relevance: keep });
    const h = await setup();
    await seedPortPair(h, 0.6);
    await h.services.settings.updateSettings({ cleanerConfidenceFloor: 0.9 });

    expect((await cleanerFor(h, stub).run()).proposed).toEqual([]);

    await h.services.settings.updateSettings({ cleanerConfidenceFloor: 0.7 });

    expect((await cleanerFor(h, stub).run()).proposed.length).toBe(1);
  });

  it("retires a shelve-lean record only at or above the floor", async () => {
    const lean = () => ({
      type: "score",
      score: 0.8,
      confidence: 0.8,
      legend: {},
      probabilities: { "0": 0.1, "1": 0.8, "2": 0.05, "3": 0.05 },
    });
    const h = await setup();
    await seedStale(h, 1);
    await h.services.settings.updateSettings({ cleanerConfidenceFloor: 1 });

    const blocked = await cleanerFor(h, stubJev({ relevance: lean })).run();

    expect(blocked.proposed).toEqual([]);

    await h.services.settings.updateSettings({ cleanerConfidenceFloor: 0 });

    const allowed = await cleanerFor(h, stubJev({ relevance: lean })).run();

    expect(allowed.proposed.length).toBe(1);
    expect(allowed.proposed[0]?.target?.reason).toMatch(/shelve-lean \(confidence \d\.\d\d\)/);
  });

  it("spends no jev calls on a remembered coexist pair and remembered keeps", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("coexist"), relevance: keep });
    const h = await setup();
    await seedPortPair(h, 0.6);

    const first = await cleanerFor(h, stub).run();
    const second = await cleanerFor(h, stub).run();

    expect(first.jevCalls).toBe(3);
    expect(second.jevCalls).toBe(0);
  });

  it("asks again once a remembered record changes", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("coexist"), relevance: keep });
    const h = await setup();
    await seedPortPair(h, 0.6);

    await cleanerFor(h, stub).run();

    const clock = new Date(NOW.getTime() + 100 * 86_400_000);
    const later = await cleanerFor(h, stub, { now: () => clock }).run();

    expect(later.jevCalls).toBe(3);
  });

  it("proposes exactly one merge for three identical records", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    const ids = [
      (await seedRecord(h, { ...text, createdAt: daysAgo(90) })).id,
      (await seedRecord(h, { ...text, createdAt: daysAgo(80) })).id,
      (await seedRecord(h, { ...text, createdAt: daysAgo(40) })).id,
    ];

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);

    const target = report.proposed[0]?.target;

    expect(target?.action).toBe("merge");
    expect(ids).toContain(target?.recordId);
    expect(ids).toContain(target?.replacementId);
  });

  it("keeps producing tiles after one jev call throws", async () => {
    const stub = stubJev({ relevance: shelve });
    const original = stub.systemOne.bind(stub);
    let calls = 0;
    stub.systemOne = async (request, options) => {
      calls += 1;

      if (calls === 2) {
        throw new Error("one bad call");
      }

      return original(request, options);
    };
    const h = await setup();
    await seedStale(h, 3);

    const report = await cleanerFor(h, stub).run();

    expect(report.jevCalls).toBe(3);
    expect(report.proposed.length).toBe(2);
  });

  it("approves a cleaner supersede tile with no reason override", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("changed", 0.8), relevance: keep });
    const h = await setup();
    const { older, newer } = await seedPortPair(h, 0.6);

    const report = await cleanerFor(h, stub).run();
    const tile = report.proposed[0];

    expect(tile?.target?.reason).toBe("changed");

    const approved = await h.services.proposals.approve(tile?.id as string, {});
    const view = await h.services.records.get(older.id);

    expect(approved.id).toBe(newer.id);
    expect(view.lifecycle).toBe("superseded");
    expect(view.lineage.supersededBy?.id).toBe(newer.id);
  });

  it("ends the jev stages after three consecutive failures and still produces promotions", async () => {
    const logs: string[] = [];
    const stub = stubJev();
    stub.failWith = new Error("jev unreachable");
    const h = await setup();
    await seedStale(h, 6);
    const hot = await seedLegacy(h, { headline: "popular repl note", claim: "repl readline", why: "fixtures" });
    await insert(
      h.store.readLog,
      Array.from({ length: 3 }, (_, index) => ({
        id: `r${index}`,
        ts: daysAgo(1),
        query: "q",
        scope: null,
        resultId: hot.id,
        source: "legacy",
        rank: 0,
        score: 1,
      })),
    );

    const report = await cleanerFor(h, stub, { log: (message: string) => logs.push(message) }).run();

    expect(report.jevCalls).toBe(3);
    expect(stub.calls.length).toBe(3);
    expect(report.proposed.map((entry) => entry.target?.action)).toEqual(["promote"]);
    expect(logs.filter((message) => message.includes("ending the jev stages")).length).toBe(1);
  });

  it("blocks the replacement id of a pending proposal from an earlier run", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    const a = await seedRecord(h, { ...text, createdAt: daysAgo(90) });
    const b = await seedRecord(h, { ...text, createdAt: daysAgo(60) });
    await seedRecord(h, { ...text, createdAt: daysAgo(40) });
    await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: a.id, source: "v2", action: "merge", replacementId: b.id, reason: "earlier run" },
      }),
    );

    const report = await cleanerFor(h, null).run();

    expect(report.proposed).toEqual([]);
  });

  it("blocks the replacement id of a recently rejected proposal", async () => {
    const h = await setup();
    const text = { headline: "dup headline", claim: "dup claim words", why: "dup why" };
    const a = await seedRecord(h, { ...text, createdAt: daysAgo(90) });
    const b = await seedRecord(h, { ...text, createdAt: daysAgo(60) });
    await seedRecord(h, { ...text, createdAt: daysAgo(40) });
    const earlier = await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: a.id, source: "v2", action: "merge", replacementId: b.id, reason: "earlier run" },
      }),
    );
    await h.services.proposals.reject(earlier.id, { reason: "no" });

    expect((await cleanerFor(h, null).run()).proposed).toEqual([]);
  });

  it("still merges a remembered coexist pair at 0.9 or higher after the breaker trips", async () => {
    const stub = stubJev();
    stub.failWith = new Error("jev unreachable");
    const h = await setup();
    const failing = [2, 4, 6];

    for (const dim of failing) {
      for (const [age, angle] of [
        [90, 0],
        [40, Math.acos(0.99)],
      ] as const) {
        const vector = new Array<number>(64).fill(0);
        vector[dim] = Math.cos(angle);
        vector[dim + 1] = Math.sin(angle);
        await seedRecord(h, {
          createdAt: daysAgo(age),
          vector,
          headline: `outage ${dim} ${age}`,
          claim: `claim ${dim} ${age}`,
          why: "w",
        });
      }
    }

    const { older, newer } = await seedPortPair(h, NEAR_DUPLICATE_ANGLE);
    await h.services.cleanerState.rememberPair(
      { id: older.id, updatedAt: older.createdAt },
      { id: newer.id, updatedAt: newer.createdAt },
    );

    const report = await cleanerFor(h, stub).run();

    expect(report.jevCalls).toBe(3);
    expect(report.proposed.map((entry) => entry.target)).toMatchObject([
      { recordId: older.id, action: "merge", replacementId: newer.id },
    ]);
  });

  it("does not trip the breaker on three client errors in a row", async () => {
    const stub = stubJev();
    stub.failWith = Object.assign(new Error("bad request"), { status: 400 });
    const h = await setup();
    await seedStale(h, 6);

    const report = await cleanerFor(h, stub).run();

    expect(report.jevCalls).toBe(6);
  });

  it("does not trip the breaker on malformed answers", async () => {
    const stub = stubJev({ relevance: () => ({ type: "score", score: 1, confidence: 0.9, legend: {} }) });
    const h = await setup();
    await seedStale(h, 5);

    const report = await cleanerFor(h, stub).run();

    expect(report.jevCalls).toBe(5);
  });

  it.each([
    ["a timeout", new DOMException("timed out", "TimeoutError")],
    ["a network error", new TypeError("fetch failed")],
    ["a 503", Object.assign(new Error("unavailable"), { status: 503 })],
    ["a 401", Object.assign(new Error("unauthorized"), { status: 401 })],
    ["a 403", Object.assign(new Error("forbidden"), { status: 403 })],
    ["a 429", Object.assign(new Error("slow down"), { status: 429 })],
  ])("trips the breaker after three failures that are %s", async (_name, error) => {
    const stub = stubJev();
    stub.failWith = error as Error;
    const h = await setup();
    await seedStale(h, 6);

    const report = await cleanerFor(h, stub).run();

    expect(report.jevCalls).toBe(3);
  });

  it("resets the failure count after a successful jev call", async () => {
    const stub = stubJev({ relevance: shelve });
    const original = stub.systemOne.bind(stub);
    let calls = 0;
    stub.systemOne = async (request, options) => {
      calls += 1;

      if (calls % 3 !== 0) {
        throw new Error("flaky");
      }

      return original(request, options);
    };
    const h = await setup();
    await seedStale(h, 9);

    const report = await cleanerFor(h, stub, { cap: 9 }).run();

    expect(report.jevCalls).toBe(9);
    expect(report.proposed.length).toBe(3);
  });

  it("does not hold the store lock while waiting on jev", async () => {
    const outcomes: string[] = [];
    const stub = stubJev({ relevance: keep });
    const original = stub.systemOne.bind(stub);
    const h = await setup();
    stub.systemOne = async (request, options) => {
      const probe = await Promise.race([
        h.store.withLock(async () => "free"),
        new Promise<string>((resolve) => setTimeout(() => resolve("held"), 200)),
      ]);

      outcomes.push(probe);

      return original(request, options);
    };
    await seedStale(h, 2);

    await cleanerFor(h, stub).run();

    expect(outcomes.length).toBe(2);
    expect(outcomes.every((outcome) => outcome === "free")).toBe(true);
  });

  it("logs which step failed and keeps going when creating a tile throws", async () => {
    const logs: string[] = [];
    const h = await setup();
    const hot = await seedLegacy(h, { headline: "popular repl note", claim: "repl readline", why: "fixtures" });
    await insert(
      h.store.readLog,
      Array.from({ length: 3 }, (_, index) => ({
        id: `r${index}`,
        ts: daysAgo(1),
        query: "q",
        scope: null,
        resultId: hot.id,
        source: "legacy",
        rank: 0,
        score: 1,
      })),
    );
    vi.spyOn(h.services.proposals, "createInternal").mockRejectedValue(new Error("create exploded"));

    const report = await cleanerFor(h, null, { log: (message: string) => logs.push(message) }).run();

    expect(report.proposed).toEqual([]);
    expect(logs.some((message) => message.includes("promote") && message.includes("create exploded"))).toBe(true);
  });
});

describe("schedule", () => {
  it("finds the next sunday 03:00 strictly after the start", () => {
    const next = nextRunAfter(new Date(2026, 10, 1, 10, 0, 0));

    expect(next.getDay()).toBe(0);
    expect([next.getHours(), next.getMinutes()]).toEqual([3, 0]);
    expect(next.getTime()).toBeGreaterThan(new Date(2026, 10, 1, 10, 0, 0).getTime());
    expect(next.getTime() - new Date(2026, 10, 1, 10, 0, 0).getTime()).toBeLessThanOrEqual(7 * 86_400_000);

    const sundayEarly = nextRunAfter(new Date(2026, 10, 1, 1, 0, 0));
    expect(sundayEarly.getDate()).toBe(1);

    const exactly = nextRunAfter(new Date(2026, 10, 1, 3, 0, 0));
    expect(exactly.getDate()).toBe(8);
  });

  it("runs once at the due time and re-arms for the following week", async () => {
    vi.useFakeTimers({ now: new Date(2026, 9, 31, 12, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);

    const schedule = scheduleWeekly(run);
    await vi.advanceTimersByTimeAsync(24 * 3600 * 1000);
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(3 * 86_400_000);
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(5 * 86_400_000);
    expect(run).toHaveBeenCalledTimes(2);

    schedule.stop();
  });

  it("finds the latest sunday 03:00 at or before a moment", () => {
    const slot = previousRunAtOrBefore(new Date(2026, 10, 1, 3, 0, 30));

    expect([slot.getDate(), slot.getHours(), slot.getMinutes()]).toEqual([1, 3, 0]);

    const earlier = previousRunAtOrBefore(new Date(2026, 10, 1, 2, 59, 0));

    expect([earlier.getMonth(), earlier.getDate()]).toEqual([9, 25]);
  });

  it("catches up a missed run within a minute of startup", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 1, 3, 0, 30) });
    const run = vi.fn().mockResolvedValue(undefined);
    const state = fakeState(new Date(2026, 9, 25, 3, 0, 5).toISOString());

    const schedule = scheduleWeekly(run, { state });
    await vi.advanceTimersByTimeAsync(60_000);

    expect(run).toHaveBeenCalledTimes(1);
    expect(state.setLastRunAt).toHaveBeenCalledTimes(1);

    schedule.stop();
  });

  it("does not catch up when the latest slot already ran", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 1, 10, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);
    const state = fakeState(new Date(2026, 10, 1, 3, 0, 5).toISOString());

    const schedule = scheduleWeekly(run, { state });
    await vi.advanceTimersByTimeAsync(60_000);

    expect(run).not.toHaveBeenCalled();

    schedule.stop();
  });

  it("records now and does not run when no run was ever recorded", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 1, 3, 0, 30) });
    const run = vi.fn().mockResolvedValue(undefined);
    const state = fakeState(null);

    const schedule = scheduleWeekly(run, { state });
    await vi.advanceTimersByTimeAsync(60_000);

    expect(run).not.toHaveBeenCalled();
    expect(state.setLastRunAt).toHaveBeenCalledWith(new Date(2026, 10, 1, 3, 0, 30).toISOString());

    schedule.stop();
  });

  it("finds the next 04:00 strictly after the start when daily", () => {
    const morning = nextRunAfter(new Date(2026, 10, 1, 3, 0, 0), null, 4);
    expect([morning.getDate(), morning.getHours()]).toEqual([1, 4]);

    const exactly = nextRunAfter(new Date(2026, 10, 1, 4, 0, 0), null, 4);
    expect([exactly.getDate(), exactly.getHours()]).toEqual([2, 4]);
  });

  it("runs a daily schedule at 04:00 and re-arms 24 hours later", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 1, 12, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);

    const schedule = scheduleAt(run, { weekday: null, hour: 4 });
    await vi.advanceTimersByTimeAsync(15 * 3600 * 1000);
    expect(run).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(2 * 3600 * 1000);
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(20 * 3600 * 1000);
    expect(run).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(4 * 3600 * 1000);
    expect(run).toHaveBeenCalledTimes(2);

    schedule.stop();
  });

  it("catches up a daily schedule when yesterday's slot was missed", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 2, 10, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);
    const state = fakeState(new Date(2026, 10, 1, 4, 0, 5).toISOString());

    const schedule = scheduleAt(run, { weekday: null, hour: 4, state });
    await vi.advanceTimersByTimeAsync(60_000);

    expect(run).toHaveBeenCalledTimes(1);
    expect(state.setLastRunAt).toHaveBeenCalledTimes(1);

    schedule.stop();
  });

  it("does not catch up a daily schedule when today's slot already ran", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 2, 10, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);
    const state = fakeState(new Date(2026, 10, 2, 4, 0, 5).toISOString());

    const schedule = scheduleAt(run, { weekday: null, hour: 4, state });
    await vi.advanceTimersByTimeAsync(60_000);

    expect(run).not.toHaveBeenCalled();

    schedule.stop();
  });
});

function fakeState(initial: string | null) {
  let last = initial;

  return {
    getLastRunAt: vi.fn(async () => last),
    setLastRunAt: vi.fn(async (at: string) => {
      last = at;
    }),
  };
}

async function setupWithJev(stub: ReturnType<typeof stubJev>) {
  harness = await makeHarness({ proposals: { jev: stub } });
  harness.clock.value = NOW.toISOString();

  return harness;
}

const doubtingReader = () =>
  stubJev({
    reader: (_question, request) =>
      choiceAnswer((request.state as { headline: string }).headline.includes("weak") ? "none" : "lookup"),
  });

const weakBody = (index: number) =>
  proposalBody({
    headline: `weak note ${index} alpha${index * 7} beta${index * 13}`,
    fields: { claim: `unique claim ${index} gamma${index * 17} delta${index * 19}` },
  });

describe("cleaner doubts pass", () => {
  it("queues a retire tile for a record whose proposal carries a demote flag", async () => {
    const h = await setupWithJev(doubtingReader());
    const doubted = await h.services.proposals.create(weakBody(1));
    const fine = await h.services.proposals.create(proposalBody({ headline: "a plain note" }));

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]).toMatchObject({
      origin: "cleaner",
      status: "pending",
      target: {
        recordId: doubted.id,
        source: "v2",
        action: "retire",
        reason: "jev doubted at propose: demote:reader=none",
      },
    });
    expect(report.proposed[0]?.target?.recordId).not.toBe(fine.id);
    expect((await h.services.records.get(doubted.id)).lifecycle).toBe("live");
  });

  it("queues a retire tile for a record flagged shape_low on a shape question", async () => {
    const h = await setupWithJev(stubJev({ headline_faithful: () => ({ type: "noul", noul: 0.1 }) }));
    const doubted = await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: doubted.id,
      action: "retire",
      reason: "jev doubted at propose: shape_low:headline_faithful",
    });
  });

  it("queues a retire tile for a record flagged low_confidence on a gate question", async () => {
    const h = await setupWithJev(stubJev({ worth: () => choiceAnswer("keep", 0.3) }));
    const doubted = await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: doubted.id,
      action: "retire",
      reason: "jev doubted at propose: low_confidence:worth",
    });
  });

  it("queues no retire tile for a record flagged only low_confidence on a relation", async () => {
    const h = await setupWithJev(stubJev({ relation: () => choiceAnswer("distinct", 0.3) }));
    await seedRecord(h, {
      headline: proposalBody().headline,
      claim: proposalBody().fields.claim,
      createdAt: daysAgo(3),
    });
    const saved = await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();

    const flags = (await h.services.proposals.get(saved.id)).flags;

    expect(flags.some((flag) => flag.startsWith("low_confidence:relation:"))).toBe(true);
    expect(report.proposed.filter((entry) => entry.target?.action === "retire")).toEqual([]);
  });

  it("queues one retire tile listing every doubt flag on a mixed record", async () => {
    const h = await setupWithJev(
      stubJev({
        reader: () => choiceAnswer("none", 0.9),
        worth: () => choiceAnswer("keep", 0.3),
        headline_faithful: () => ({ type: "noul", noul: 0.1 }),
      }),
    );
    await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();
    const reason = String(report.proposed[0]?.target?.reason);

    expect(report.proposed.length).toBe(1);
    expect(reason).toContain("demote:reader=none");
    expect(reason).toContain("shape_low:headline_faithful");
    expect(reason).toContain("low_confidence:worth");
  });

  it("queues a supersede tile for a record saved with a supersede suggestion", async () => {
    const h = await setupWithJev(stubJev({ relation: () => choiceAnswer("replacement", 0.9) }));
    const older = await seedRecord(h, {
      headline: proposalBody().headline,
      claim: proposalBody().fields.claim,
      createdAt: daysAgo(3),
    });
    const saved = await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.length).toBe(1);
    expect(report.proposed[0]?.target).toMatchObject({
      recordId: older.id,
      source: "v2",
      action: "supersede",
      replacementId: saved.id,
      reason: "changed",
    });
    expect((await h.services.records.get(older.id)).lifecycle).toBe("live");
  });

  it("gives a record that qualifies for both only the supersede tile", async () => {
    const h = await setupWithJev(
      stubJev({
        relation: () => choiceAnswer("replacement", 0.9),
        reader: () => choiceAnswer("none", 0.9),
      }),
    );
    const older = await seedRecord(h, {
      headline: proposalBody().headline,
      claim: proposalBody().fields.claim,
      createdAt: daysAgo(3),
    });
    await h.services.proposals.create(proposalBody());

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.map((entry) => entry.target?.action)).toEqual(["supersede"]);
    expect(report.proposed[0]?.target?.recordId).toBe(older.id);
  });

  it("skips records already retired, blocked by a pending tile, or approved before the doubts offset", async () => {
    const h = await setupWithJev(doubtingReader());
    const retired = await h.services.proposals.create(weakBody(1));
    const blocked = await h.services.proposals.create(weakBody(2));
    h.clock.value = daysAgo(2);
    await h.services.proposals.create(weakBody(3));
    h.clock.value = NOW.toISOString();
    await h.services.records.markRetired(retired.id, "obsolete");
    await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: blocked.id, source: "v2", action: "retire", reason: "stale" },
      }),
    );
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(1));

    const report = await cleanerFor(h, null).run();

    expect(report.proposed).toEqual([]);
    expect((await h.services.proposals.list("pending", null)).length).toBe(1);
  });

  it("looks back seven days when the cleaner has never run", async () => {
    const h = await setupWithJev(doubtingReader());
    h.clock.value = daysAgo(9);
    await h.services.proposals.create(weakBody(1));
    h.clock.value = daysAgo(6);
    const recent = await h.services.proposals.create(weakBody(2));
    h.clock.value = NOW.toISOString();

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.map((entry) => entry.target?.recordId)).toEqual([recent.id]);
  });

  it("caps the doubts pass at 10 newest, outside the cap of five", async () => {
    const h = await setupWithJev(doubtingReader());
    const ids: string[] = [];

    for (let index = 0; index < 12; index += 1) {
      h.clock.value = new Date(NOW.getTime() - (12 - index) * 60_000).toISOString();
      ids.push((await h.services.proposals.create(weakBody(index))).id);
    }

    h.clock.value = NOW.toISOString();

    const report = await cleanerFor(h, null).run();
    const queued = report.proposed.map((entry) => entry.target?.recordId);

    expect(queued.length).toBe(10);
    expect(new Set(queued)).toEqual(new Set(ids.slice(2)));
    expect(queued[0]).toBe(ids[11]);
  });

  it("advances the doubts offset to the run start when the pass clears its backlog", async () => {
    const h = await setupWithJev(doubtingReader());
    await h.services.proposals.create(weakBody(1));
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(3));

    await cleanerFor(h, null).run();

    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(NOW.toISOString());
    expect(await h.services.cleanerState.getLastRunAt()).toBeNull();
  });

  it("keeps the doubts offset when the cap trims so the next run sees the dropped records", async () => {
    const h = await setupWithJev(doubtingReader());
    const ids: string[] = [];

    for (let index = 0; index < 12; index += 1) {
      h.clock.value = new Date(NOW.getTime() - (12 - index) * 60_000).toISOString();
      ids.push((await h.services.proposals.create(weakBody(index))).id);
    }

    h.clock.value = NOW.toISOString();
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(1));

    const first = await cleanerFor(h, null).run();

    expect(first.proposed.length).toBe(10);
    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(daysAgo(1));

    const second = await cleanerFor(h, null).run();

    expect(new Set(second.proposed.map((entry) => entry.target?.recordId))).toEqual(new Set(ids.slice(0, 2)));
    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(NOW.toISOString());
  });

  it("does not queue a retire tile after the supersede tile was approved between two capped runs", async () => {
    const dual = weakBody(5);
    const h = await setupWithJev(
      stubJev({
        relation: (_question, request) =>
          choiceAnswer(
            (request.state as { incoming: string }).incoming.includes(dual.headline) ? "replacement" : "distinct",
            0.9,
          ),
        reader: (_question, request) =>
          choiceAnswer((request.state as { headline: string }).headline.includes("weak") ? "none" : "lookup"),
      }),
    );
    const older = await seedRecord(h, {
      headline: dual.headline,
      claim: dual.fields.claim,
      createdAt: daysAgo(3),
    });
    const ids: string[] = [];

    for (let index = 0; index < 11; index += 1) {
      h.clock.value = new Date(NOW.getTime() - (11 - index) * 60_000).toISOString();
      ids.push((await h.services.proposals.create(index === 5 ? dual : weakBody(index))).id);
    }

    h.clock.value = NOW.toISOString();
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(1));

    const first = await cleanerFor(h, null).run();
    const supersedeTile = first.proposed.find((entry) => entry.target?.action === "supersede");

    expect(first.proposed.length).toBe(10);
    expect(supersedeTile?.target).toMatchObject({ recordId: older.id, replacementId: ids[5] });
    expect(first.proposed.some((entry) => entry.target?.recordId === ids[5])).toBe(false);
    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(daysAgo(1));

    await h.services.proposals.approve(supersedeTile?.id as string, {});

    expect((await h.services.records.get(older.id)).lifecycle).toBe("superseded");

    const second = await cleanerFor(h, null).run();

    expect(second.proposed.some((entry) => entry.target?.recordId === ids[5])).toBe(false);
    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(NOW.toISOString());
  });

  it("queues the retire tile when the older record was retired by hand and the replacement carries a demote flag", async () => {
    const dual = weakBody(5);
    const h = await setupWithJev(
      stubJev({
        relation: (_question, request) =>
          choiceAnswer(
            (request.state as { incoming: string }).incoming.includes(dual.headline) ? "replacement" : "distinct",
            0.9,
          ),
        reader: (_question, request) =>
          choiceAnswer((request.state as { headline: string }).headline.includes("weak") ? "none" : "lookup"),
      }),
    );
    const older = await seedRecord(h, {
      headline: dual.headline,
      claim: dual.fields.claim,
      createdAt: daysAgo(3),
    });
    const created = await h.services.proposals.create(dual);
    await h.services.records.markRetired(older.id, "obsolete");
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(1));

    const report = await cleanerFor(h, null).run();
    const targets = report.proposed.map((entry) => entry.target);

    expect((await h.services.records.get(older.id)).supersededBy).toBeNull();
    expect(targets.some((target) => target?.action === "supersede")).toBe(false);
    expect(targets).toHaveLength(1);
    expect(targets[0]).toMatchObject({ recordId: created.id, action: "retire" });
    expect(targets[0]?.reason).toMatch(/^jev doubted at propose:/);
  });

  it("holds the doubts offset when the read hits its limit", async () => {
    const h = await setupWithJev(doubtingReader());

    for (let index = 0; index < 4; index += 1) {
      h.clock.value = new Date(NOW.getTime() - (4 - index) * 60_000).toISOString();
      await h.services.proposals.create(weakBody(index));
    }

    h.clock.value = NOW.toISOString();
    await h.services.cleanerState.setLastDoubtsAt(daysAgo(1));

    await cleanerFor(h, null, { doubtReadLimit: 3 }).run();

    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(daysAgo(1));

    await cleanerFor(h, null).run();

    expect(await h.services.cleanerState.getLastDoubtsAt()).toBe(NOW.toISOString());
  });

  it("doubts a record approved during the previous run", async () => {
    const h = await setupWithJev(doubtingReader());
    const runStart = new Date(NOW.getTime() - 24 * 60 * 60 * 1000);
    h.clock.value = new Date(runStart.getTime() - 60_000).toISOString();
    await h.services.proposals.create(weakBody(1));
    h.clock.value = new Date(runStart.getTime() + 5 * 60_000).toISOString();
    const during = await h.services.proposals.create(weakBody(2));
    h.clock.value = NOW.toISOString();
    await h.services.cleanerState.setLastDoubtsAt(runStart.toISOString());
    await h.services.cleanerState.setLastRunAt(new Date(runStart.getTime() + 10 * 60_000).toISOString());

    const report = await cleanerFor(h, null).run();

    expect(report.proposed.map((entry) => entry.target?.recordId)).toEqual([during.id]);
  });
});

describe("main wiring", () => {
  it("records the last run at startup so catch-up is live", async () => {
    const dir = mkdtempSync(join(tmpdir(), "tabularium-main-"));
    const config = { ...loadConfig({}), host: "127.0.0.1", port: 0, storePath: dir, embedDim: 64 };
    const running = await start(config);

    try {
      const last = await vi.waitFor(async () => {
        const value = await running.services.cleanerState.getLastRunAt();

        expect(value).not.toBeNull();

        return value;
      });

      expect(Number.isNaN(Date.parse(last as string))).toBe(false);
    } finally {
      running.schedule.stop();
      running.defectSweepSchedule.stop();
      await new Promise<void>((resolve) => running.server.close(() => resolve()));
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
