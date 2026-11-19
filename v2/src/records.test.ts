import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { DefectEvent } from "./domain.ts";
import { selectWhere } from "./store.ts";
import { createApprovedDefect, failNthWrite, makeHarness, proposalBody, seedLegacy, seedRecord } from "./testing.ts";
import type { Harness } from "./testing.ts";

let harness: Harness;

const repl = {
  scope: "project-a",
  headline: "repl readline crashes",
  claim: "repl aborts with a readline error",
  why: "fixtures need repl",
};

beforeEach(async () => {
  harness = await makeHarness();
});

afterEach(() => {
  harness.cleanup();
});

async function recalledIds(): Promise<string[]> {
  const { results } = await harness.services.recall.recall({ q: "repl readline error", scope: "project-a" });

  return results.map((entry) => entry.id);
}

describe("archive", () => {
  it("drops a legacy row and a v2 record from recall, then restores them", async () => {
    const legacy = await seedLegacy(harness, repl);
    const current = await seedRecord(harness, repl);
    expect((await recalledIds()).sort()).toEqual([current.id, legacy.id].sort());

    await harness.services.records.archive(legacy.id, "legacy history");
    await harness.services.records.archive(current.id, "old design");

    expect(await recalledIds()).toEqual([]);

    await harness.services.records.unarchive(legacy.id);
    await harness.services.records.unarchive(current.id);

    expect((await recalledIds()).sort()).toEqual([current.id, legacy.id].sort());
  });

  it("keeps the archived row readable with its lifecycle and note", async () => {
    const legacy = await seedLegacy(harness, repl);

    const archived = await harness.services.records.archive(legacy.id, "legacy history");
    const read = await harness.services.records.get(legacy.id);

    expect(archived.lifecycle).toBe("archived");
    expect(read.lifecycle).toBe("archived");
    expect(read.lifecycleNote).toBe("legacy history");
    expect(read.source).toBe("legacy");
  });

  it("clears the note on unarchive", async () => {
    const current = await seedRecord(harness, repl);

    await harness.services.records.archive(current.id, "old design");
    const restored = await harness.services.records.unarchive(current.id);

    expect(restored.lifecycle).toBe("live");
    expect(restored.lifecycleNote).toBeNull();
  });

  it("is not a neighbour for a new proposal", async () => {
    const legacy = await seedLegacy(harness, repl);
    const current = await seedRecord(harness, repl);
    await harness.services.records.archive(legacy.id, "legacy history");
    await harness.services.records.archive(current.id, "old design");

    const proposal = await harness.services.proposals.create(proposalBody());

    expect(proposal.neighbours).toEqual([]);
  });

  it("refuses a superseded, retired or already archived row", async () => {
    const superseded = await seedRecord(harness, { ...repl, lifecycle: "superseded" });
    const retired = await seedLegacy(harness, { ...repl, lifecycle: "retired" });
    const archived = await seedRecord(harness, repl);
    await harness.services.records.archive(archived.id, "old design");

    await expect(harness.services.records.archive(superseded.id, "x")).rejects.toMatchObject({ status: 409 });
    await expect(harness.services.records.archive(retired.id, "x")).rejects.toMatchObject({ status: 409 });
    await expect(harness.services.records.archive(archived.id, "x")).rejects.toMatchObject({ status: 409 });
  });

  it("refuses a defect", async () => {
    const defect = await createApprovedDefect(harness);

    await expect(harness.services.records.archive(defect.id, "x")).rejects.toMatchObject({ status: 409 });
    expect((await harness.services.records.get(defect.id)).lifecycle).toBe("live");
  });

  it("answers 404 for a missing id and 400 for a missing reason", async () => {
    const current = await seedRecord(harness, repl);

    await expect(harness.services.records.archive("nope", "x")).rejects.toMatchObject({ status: 404 });
    await expect(harness.services.records.unarchive("nope")).rejects.toMatchObject({ status: 404 });
    await expect(harness.services.records.archive(current.id, "")).rejects.toMatchObject({ status: 400 });
  });

  it("refuses unarchive on a live row", async () => {
    const current = await seedRecord(harness, repl);

    await expect(harness.services.records.unarchive(current.id)).rejects.toMatchObject({ status: 409 });
  });

  it("is not undone by unsupersede", async () => {
    const legacy = await seedLegacy(harness, repl);
    const current = await seedRecord(harness, repl);
    await harness.services.records.archive(legacy.id, "legacy history");
    await harness.services.records.archive(current.id, "old design");

    await expect(harness.services.records.unsupersede(legacy.id, "x")).rejects.toMatchObject({ status: 409 });
    await expect(harness.services.records.unsupersede(current.id, "x")).rejects.toMatchObject({ status: 409 });
    expect((await harness.services.records.get(legacy.id)).lifecycle).toBe("archived");
  });
});

describe("retireLegacy replacement guard", () => {
  it("refuses a replacement that is not live", async () => {
    const legacy = await seedLegacy(harness, repl);
    const replacement = await seedRecord(harness, repl);
    await harness.services.records.archive(replacement.id, "old design");

    await expect(
      harness.services.records.retireLegacy(legacy.id, replacement.id, "promoted"),
    ).rejects.toMatchObject({ status: 409 });
    expect((await harness.services.records.get(legacy.id)).lifecycle).toBe("live");
  });

  it("keeps the legacy row live when a reinforcement retry meets an archived copy", async () => {
    const legacy = await seedLegacy(harness, repl);
    const proposal = await harness.services.proposals.createInternal(proposalBody({ origin: "cleaner" }));
    const retire = vi.spyOn(harness.services.records, "retireLegacy").mockRejectedValueOnce(new Error("write failed"));

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: legacy.id }),
    ).rejects.toThrow("write failed");
    retire.mockRestore();
    await harness.services.records.archive(proposal.id, "mistaken copy");

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: legacy.id }),
    ).rejects.toMatchObject({ status: 409 });
    expect((await harness.services.records.get(legacy.id)).lifecycle).toBe("live");
  });
});

describe("defect events", () => {
  const taker = { threadId: "thread-9", link: "t3-thread://thread-9" };

  async function events(): Promise<DefectEvent[]> {
    return selectWhere<DefectEvent>(harness.store.defectEvents, undefined);
  }

  it("writes a take event with the taker thread id and link", async () => {
    const defect = await createApprovedDefect(harness);

    await harness.services.records.take(defect.id, taker);

    expect(await events()).toEqual([
      expect.objectContaining({
        recordId: defect.id,
        event: "take",
        threadId: "thread-9",
        link: "t3-thread://thread-9",
        proposalId: null,
      }),
    ]);
  });

  it("writes a release event for the releasing thread", async () => {
    const defect = await createApprovedDefect(harness);
    await harness.services.records.take(defect.id, taker);

    await harness.services.records.release(defect.id, { threadId: "thread-5" });

    const released = (await events()).filter((event) => event.event === "release");
    expect(released).toHaveLength(1);
    expect(released[0]).toMatchObject({ recordId: defect.id, threadId: "thread-5" });
  });

  it("writes no event for a no-op release or a refused take", async () => {
    const defect = await createApprovedDefect(harness);

    await harness.services.records.release(defect.id, { threadId: "thread-5" });
    await harness.services.records.take(defect.id, taker);
    await expect(harness.services.records.take(defect.id, taker)).rejects.toMatchObject({ status: 409 });

    expect((await events()).map((event) => event.event)).toEqual(["take"]);
  });

  it("keeps a call succeeding when the event insert fails and logs one stderr line", async () => {
    const defect = await createApprovedDefect(harness);
    const stderr = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const restore = failNthWrite(harness, "defectEvents", 1);

    const taken = await harness.services.records.take(defect.id, taker);
    restore();

    expect(taken.defectStatus).toBe("taken");
    expect(await events()).toEqual([]);
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(JSON.parse(String(stderr.mock.calls[0]?.[0]))).toMatchObject({ event: "defect-event-failed" });
    stderr.mockRestore();
  });

  it("writes one close event with the proposal id and its provenance thread, and none on re-apply", async () => {
    const defect = await createApprovedDefect(harness);
    const tile = await harness.services.proposals.create(
      proposalBody({
        kind: "fact",
        headline: "export now pages with a cursor",
        fields: { claim: "the cursor fix landed with test SyncReportTest" },
        target: { recordId: defect.id, source: "v2", action: "close" },
      }),
    );

    const closed = (await events()).filter((event) => event.event === "close");
    expect(closed).toHaveLength(1);
    expect(closed[0]).toMatchObject({ recordId: defect.id, proposalId: tile.id, threadId: "thread-1" });

    await (harness.services.proposals as any).applyClose(tile, tile.target);

    expect((await events()).filter((event) => event.event === "close")).toHaveLength(1);
  });

  it("writes a get row only through read, not through take or release", async () => {
    const defect = await createApprovedDefect(harness);

    await harness.services.records.take(defect.id, taker);
    await harness.services.records.release(defect.id);

    expect(await harness.store.readLog.countRows()).toBe(0);
  });
});
