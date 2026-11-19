import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { scheduleAt } from "./cleaner.ts";
import { sweepDefects } from "./defectSweep.ts";
import type { DefectEvent, MemoryRecord } from "./domain.ts";
import { keyedRunClock } from "./settings.ts";
import { createApprovedDefect, defectBody, makeHarness, proposalBody } from "./testing.ts";
import type { Harness } from "./testing.ts";
import { findById, selectWhere } from "./store.ts";

const DAY_MS = 86_400_000;
const NOW = new Date("2026-11-03T10:00:00.000Z");
const TAKER = { threadId: "thread-9", link: "t3-thread://thread-9" };

let harness: Harness;

beforeEach(async () => {
  harness = await makeHarness();
});

afterEach(() => {
  harness.cleanup();
});

function at(ageDays: number): string {
  return new Date(NOW.getTime() - ageDays * DAY_MS).toISOString();
}

async function defectAged(ageDays: number, headline = `defect aged ${ageDays}`): Promise<MemoryRecord> {
  harness.clock.value = at(ageDays);

  const record = await createApprovedDefect(harness, { headline });

  harness.clock.value = NOW.toISOString();

  return record;
}

async function takeAged(record: MemoryRecord, ageDays: number): Promise<void> {
  harness.clock.value = at(ageDays);
  await harness.services.records.take(record.id, TAKER);
  harness.clock.value = NOW.toISOString();
}

async function stored(id: string): Promise<MemoryRecord> {
  return (await findById<MemoryRecord>(harness.store.records, id)) as MemoryRecord;
}

function sweep(overrides: Partial<Parameters<typeof sweepDefects>[0]> = {}) {
  return sweepDefects({
    store: harness.store,
    records: harness.services.records,
    proposals: harness.services.proposals,
    now: () => NOW,
    log: () => {},
    ...overrides,
  });
}

async function pending() {
  return harness.services.proposals.list("pending", null);
}

describe("sweepDefects releases stale takes", () => {
  it("reverts a 15-day take to open and clears the taken fields", async () => {
    const defect = await defectAged(20);
    await takeAged(defect, 15);

    const report = await sweep();
    const after = await stored(defect.id);

    expect(report).toEqual({ released: 1, asked: 0 });
    expect(after).toMatchObject({
      defectStatus: "open",
      takenThreadId: null,
      takenLink: null,
      takenAt: at(15),
    });
  });

  it("keeps the last-take stamp so a released defect is not asked about until it passes 90 days", async () => {
    const defect = await defectAged(120);
    await takeAged(defect, 15);

    const releasedOnDayZero = await sweep();
    const dayOne = await sweep({ now: () => new Date(NOW.getTime() + DAY_MS) });
    const pastCutoff = await sweep({ now: () => new Date(NOW.getTime() + 76 * DAY_MS) });

    expect(releasedOnDayZero).toEqual({ released: 1, asked: 0 });
    expect(dayOne).toEqual({ released: 0, asked: 0 });
    expect(pastCutoff).toEqual({ released: 0, asked: 1 });
    expect(await pending()).toHaveLength(1);
  });

  it("keeps a 13-day take", async () => {
    const defect = await defectAged(20);
    await takeAged(defect, 13);

    const report = await sweep();
    const after = await stored(defect.id);

    expect(report).toEqual({ released: 0, asked: 0 });
    expect(after).toMatchObject({ defectStatus: "taken", takenThreadId: TAKER.threadId });
  });

  it("does not ask about a defect it released in the same run", async () => {
    const defect = await defectAged(120);
    await takeAged(defect, 15);

    const report = await sweep();

    expect(report).toEqual({ released: 1, asked: 0 });
    expect(await pending()).toHaveLength(0);
  });

  it("leaves a closed defect alone", async () => {
    const defect = await defectAged(120);
    await harness.services.proposals.createInternal(closeFor(defect));

    const report = await sweep();

    expect(report).toEqual({ released: 0, asked: 0 });
    expect((await stored(defect.id)).defectStatus).toBe("closed");
  });
});

function closeFor(defect: MemoryRecord) {
  return proposalBody({
    kind: "defect",
    headline: "closing note",
    origin: "agent",
    target: { recordId: defect.id, source: "v2", action: "close" },
  });
}

function retireFor(defect: MemoryRecord) {
  return defectBody({
    headline: "retire note",
    origin: "cleaner",
    target: { recordId: defect.id, source: "v2", action: "retire", reason: "obsolete" },
  });
}

describe("sweepDefects asks about stale open defects", () => {
  it("creates one cleaner close tile for a 91-day open defect", async () => {
    const defect = await defectAged(91, "export drops rows");

    const report = await sweep();
    const tiles = await pending();

    expect(report).toEqual({ released: 0, asked: 1 });
    expect(tiles).toHaveLength(1);
    expect(tiles[0]).toMatchObject({
      origin: "cleaner",
      headline: "Still open? export drops rows",
      target: { recordId: defect.id, source: "v2", action: "close", reason: "open 90d" },
      provenance: { threadId: "cleaner" },
    });
    expect(tiles[0]?.fields.claim).toBe(defect.claim);
  });

  it("ignores an 89-day open defect", async () => {
    await defectAged(89);

    const report = await sweep();

    expect(report).toEqual({ released: 0, asked: 0 });
    expect(await pending()).toHaveLength(0);
  });

  it("ignores an old defect that is still taken inside the window", async () => {
    const defect = await defectAged(120);
    await takeAged(defect, 5);

    const report = await sweep();

    expect(report).toEqual({ released: 0, asked: 0 });
  });

  it("does not ask twice while the first tile is pending", async () => {
    await defectAged(100);

    await sweep();
    const second = await sweep();

    expect(second.asked).toBe(0);
    expect(await pending()).toHaveLength(1);
  });

  it("cuts a 95-character headline so the close tile headline fits the 100-character cap", async () => {
    const headline = "x".repeat(95);
    await defectAged(100, headline);

    const report = await sweep();
    const tiles = await pending();

    expect(report).toEqual({ released: 0, asked: 1 });
    expect(tiles).toHaveLength(1);
    expect(tiles[0]!.headline.length).toBeLessThanOrEqual(100);
    expect(tiles[0]!.headline).toBe(`Still open? ${"x".repeat(87)}…`);
  });

  it("keeps a headline of exactly 88 characters whole", async () => {
    const headline = "y".repeat(88);
    await defectAged(100, headline);

    await sweep();
    const tiles = await pending();

    expect(tiles[0]!.headline).toBe(`Still open? ${headline}`);
  });

  it("does not ask while a pending retire tile sits on the defect", async () => {
    const defect = await defectAged(100);
    await harness.services.proposals.createInternal(retireFor(defect));

    const report = await sweep();
    const tiles = await pending();

    expect(report).toEqual({ released: 0, asked: 0 });
    expect(tiles).toHaveLength(1);
    expect(tiles[0]!.target).toMatchObject({ action: "retire" });
  });

  it("does not ask after a retire tile on the defect was rejected inside 90 days", async () => {
    const defect = await defectAged(100);
    const retire = await harness.services.proposals.createInternal(retireFor(defect));

    harness.clock.value = at(30);
    await harness.services.proposals.reject(retire.id, { reason: "still real" });
    harness.clock.value = NOW.toISOString();

    const report = await sweep();

    expect(report).toEqual({ released: 0, asked: 0 });
    expect(await pending()).toHaveLength(0);
  });

  it("does not ask again after a rejection inside 90 days", async () => {
    await defectAged(100);
    await sweep();
    const [tile] = await pending();

    harness.clock.value = at(30);
    await harness.services.proposals.reject(tile!.id, { reason: "still real" });
    harness.clock.value = NOW.toISOString();

    const second = await sweep();

    expect(second.asked).toBe(0);
    expect(await pending()).toHaveLength(0);
  });

  it("asks again once the rejection is older than 90 days", async () => {
    await defectAged(300);
    await sweep({ now: () => new Date(NOW.getTime() - 200 * DAY_MS) });
    const [tile] = await pending();

    harness.clock.value = at(100);
    await harness.services.proposals.reject(tile!.id, { reason: "still real" });
    harness.clock.value = NOW.toISOString();

    const second = await sweep();

    expect(second.asked).toBe(1);
  });

  it("asks about at most five defects per run, oldest first", async () => {
    for (let index = 0; index < 7; index += 1) {
      await defectAged(100 + index, `defect ${index}`);
    }

    const report = await sweep();
    const headlines = (await pending()).map((tile) => tile.headline).sort();

    expect(report.asked).toBe(5);
    expect(headlines).toEqual([
      "Still open? defect 2",
      "Still open? defect 3",
      "Still open? defect 4",
      "Still open? defect 5",
      "Still open? defect 6",
    ]);
  });

  it("logs a failed tile and carries on with the next defect", async () => {
    const first = await defectAged(120, "first");
    await defectAged(110, "second");
    const real = harness.services.proposals;
    const log = vi.fn();
    const failing = {
      createInternal: vi.fn(async (body: unknown) => {
        const target = (body as { target: { recordId: string } }).target;

        if (target.recordId === first.id) {
          throw new Error("boom");
        }

        return real.createInternal(body);
      }),
    };

    const report = await sweep({ proposals: failing, log });

    expect(report.asked).toBe(1);
    expect(log).toHaveBeenCalledWith(expect.stringContaining(first.id));
  });

  it("never touches a fact or a retired defect", async () => {
    harness.clock.value = at(200);
    await harness.services.proposals.create(proposalBody());
    harness.clock.value = NOW.toISOString();
    const retired = await defectAged(200, "retired one");
    await harness.services.records.markRetired(retired.id, "obsolete");

    const report = await sweep();

    expect(report).toEqual({ released: 0, asked: 0 });
  });
});

describe("defect sweep schedule", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  function stubState() {
    const saved = new Map<string, string>();

    return {
      saved,
      state: {
        getLastRunAt: vi.fn(async (key?: string) => saved.get(key ?? "") ?? null),
        setLastRunAt: vi.fn(async (at: string, key?: string) => {
          saved.set(key ?? "", at);
        }),
      } as unknown as Parameters<typeof keyedRunClock>[0],
    };
  }

  it("runs at 05:00, not at 04:00, and records meta:lastDefectSweepAt", async () => {
    vi.useFakeTimers({ now: new Date(2026, 10, 3, 3, 0, 0) });
    const run = vi.fn().mockResolvedValue(undefined);
    const { saved, state } = stubState();

    const schedule = scheduleAt(run, {
      weekday: null,
      hour: 5,
      label: "defect-sweep",
      state: keyedRunClock(state, "meta:lastDefectSweepAt"),
    });
    await vi.advanceTimersByTimeAsync(90 * 60_000);
    expect(run).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(60 * 60_000);
    expect(run).toHaveBeenCalledTimes(1);
    expect(saved.has("meta:lastDefectSweepAt")).toBe(true);

    schedule.stop();
  });
});

describe("sweepDefects logs auto-release events", () => {
  it("writes an auto-release event with a null thread id per released take", async () => {
    const defect = await defectAged(20);
    await takeAged(defect, 15);

    await sweep();

    const events = await selectWhere<DefectEvent>(harness.store.defectEvents, "event = 'auto-release'");
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ recordId: defect.id, threadId: null });
  });

  it("writes no auto-release event for a kept take", async () => {
    const defect = await defectAged(20);
    await takeAged(defect, 13);

    await sweep();

    const events = await selectWhere<DefectEvent>(harness.store.defectEvents, "event = 'auto-release'");
    expect(events).toEqual([]);
  });
});

describe.each(["t3", "standalone"] as const)("defect sweep in %s mode", (mode) => {
  const closes = mode === "t3";

  it(`${closes ? "creates" : "skips"} the close tile for a stale open defect`, async () => {
    await defectAged(91, "export drops rows");

    const report = await sweep({ mode });

    expect(report.asked).toBe(closes ? 1 : 0);
    expect(await pending()).toHaveLength(closes ? 1 : 0);
  });

  it("still releases a stale take", async () => {
    const defect = await defectAged(20);
    await takeAged(defect, 15);

    const report = await sweep({ mode });

    expect(report.released).toBe(1);
    expect((await stored(defect.id)).defectStatus).toBe("open");
  });
});
