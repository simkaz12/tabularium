import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createApprovedDefect, makeHarness, proposalBody, seedLegacy, seedRecord } from "./testing.ts";
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
