import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { findById, proposalToRow, selectWhere, upsert } from "./store.ts";
import { parseProposalBody } from "./proposals.ts";
import {
  choiceAnswer,
  createApprovedDefect,
  defectBody,
  failNthWrite,
  makeHarness,
  proposalBody,
  seedLegacy,
  seedRecord,
  stubJev,
} from "./testing.ts";
import type { Harness } from "./testing.ts";
import type { LegacyRecord, MemoryRecord } from "./domain.ts";

let harness: Harness;

beforeEach(async () => {
  harness = await makeHarness();
});

afterEach(() => {
  harness.cleanup();
});

async function recordCount(): Promise<number> {
  return harness.store.records.countRows();
}

function pendingTile(overrides: Parameters<typeof proposalBody>[0] = {}) {
  return harness.services.proposals.createInternal(proposalBody({ origin: "cleaner", ...overrides }));
}

describe("propose", () => {
  it("writes a live record on propose and answers approved with the record id", async () => {
    const proposal = await harness.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.checks.status).toBe("unavailable");
    expect(proposal.flags).toEqual(["jev_unavailable"]);
    expect(proposal.resolvedAt).not.toBeNull();
    expect(proposal.resolution).toEqual({ recordId: proposal.id, reason: null });
    expect(await recordCount()).toBe(1);
    expect(await harness.services.proposals.list("pending", null)).toEqual([]);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
  });

  it("writes a live record with provenance", async () => {
    const proposal = await harness.services.proposals.create(proposalBody());

    const record = await harness.services.records.get(proposal.resolution?.recordId as string);

    expect(record.lifecycle).toBe("live");
    expect(record.headline).toBe(proposal.headline);
    expect(record.threadId).toBe("thread-1");
    expect(record.link).toBe("t3-thread://thread-1");
    expect(record.excerpt).toBe("user: repl fails");
    expect(record.origin).toBe("agent");
    expect(await recalled("repl readline history file")).toEqual([record.id]);
  });

  it("applies owner edits to headline, fields and scope on a cleaner tile", async () => {
    const proposal = await pendingTile();

    const record = await harness.services.proposals.approve(proposal.id, {
      edits: {
        headline: "Edited headline",
        fields: { rejected: "edited rejected" },
        audience: "reviewer",
        scope: "game",
        topic: "tooling",
      },
    });

    expect(record.headline).toBe("Edited headline");
    expect(record.rejected).toBe("edited rejected");
    expect(record.audience).toBe("reviewer");
    expect(record.claim).toBe(proposal.fields.claim);
    expect(record.scope).toBe("game");
    expect(record.topic).toBe("tooling");
  });

  it("rejects a second decision on the same proposal", async () => {
    const proposal = await pendingTile();
    await harness.services.proposals.approve(proposal.id, {});

    await expect(harness.services.proposals.approve(proposal.id, {})).rejects.toMatchObject({ status: 409 });
    await expect(
      harness.services.proposals.reject(proposal.id, { reason: "late" }),
    ).rejects.toMatchObject({ status: 409 });
  });

  it("rejects a headline over 100 characters and a blank claim", async () => {
    await expect(
      harness.services.proposals.create(proposalBody({ headline: "x".repeat(101) })),
    ).rejects.toMatchObject({ status: 400 });
    await expect(
      harness.services.proposals.create(proposalBody({ fields: { claim: " " } })),
    ).rejects.toMatchObject({ status: 400 });
  });

  it("accepts a body with only a claim and ignores a fields.why key", async () => {
    const plain = await harness.services.proposals.create({ ...proposalBody(), fields: { claim: "only a claim" } });
    const legacyShape = await harness.services.proposals.create({
      ...proposalBody(),
      fields: { claim: "claim with a why", why: "ignored" },
    });

    expect(plain.fields).toEqual({ claim: "only a claim", rejected: null });
    expect(legacyShape.fields).not.toHaveProperty("why");
    expect(await recordCount()).toBe(2);
  });

  it("copies audience, reader and stage onto a record that carries no why", async () => {
    const jevHarness = await makeHarness({
      proposals: {
        jev: stubJev({ reader: () => choiceAnswer("trap"), audience: () => choiceAnswer("navigator") }),
      },
    });

    try {
      const body = proposalBody();
      const proposal = await jevHarness.services.proposals.create({
        ...body,
        provenance: { ...body.provenance, stage: "recon" },
      });
      const record = await jevHarness.services.records.get(proposal.resolution?.recordId as string);

      expect(record).toMatchObject({ why: null, audience: "navigator", reader: "trap", stage: "recon" });
      expect(record.defectStatus).toBeNull();
    } finally {
      jevHarness.cleanup();
    }
  });

  it("lets edits.audience override the audience on a cleaner tile", async () => {
    const proposal = await pendingTile({ audience: "builder" });
    const record = await harness.services.proposals.approve(proposal.id, { edits: { audience: "decider" } });

    expect(proposal.audience).toBe("builder");
    expect(record.audience).toBe("decider");
  });

  it("rejects an unknown audience", async () => {
    await expect(
      harness.services.proposals.create({ ...proposalBody(), audience: "wizard" }),
    ).rejects.toMatchObject({ status: 400 });
  });

  it("lists neighbours with similarity and source", async () => {
    const existing = await seedRecord(harness, {
      headline: "repl readline crashes on the history file",
      claim: "repl aborts with a readline error when the history file is unwritable",
      why: "the test harness depends on repl for fixtures",
    });
    await seedLegacy(harness, {
      headline: "repl readline history",
      claim: "readline history file permissions break repl",
      why: "fixtures",
    });

    const proposal = await harness.services.proposals.create(proposalBody());

    expect(proposal.neighbours.length).toBe(2);
    expect(proposal.neighbours[0]?.id).toBe(existing.id);
    expect(proposal.neighbours[0]?.source).toBe("v2");
    expect(proposal.neighbours[0]?.similarity).toBeGreaterThan(0.99);
    expect(proposal.neighbours.map((entry) => entry.source)).toContain("legacy");
  });
});

describe("reject and reinforce", () => {
  it("requires a reason to reject and leaves the proposal pending without one", async () => {
    const proposal = await pendingTile();

    for (const body of [undefined, {}, { reason: "" }, { reason: "   " }]) {
      await expect(harness.services.proposals.reject(proposal.id, body)).rejects.toMatchObject({
        status: 400,
        message: "reason is required",
      });
    }

    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");
  });

  it("rejects with a reason and writes no record", async () => {
    const proposal = await pendingTile();

    const rejected = await harness.services.proposals.reject(proposal.id, { reason: "not durable" });

    expect(rejected.status).toBe("rejected");
    expect(rejected.resolution?.reason).toBe("not durable");
    expect(await recordCount()).toBe(0);
  });

  it("reinforce records the count on the target and writes no record", async () => {
    const existing = await seedRecord(harness);
    const proposal = await pendingTile();

    const reinforced = await harness.services.proposals.reinforce(proposal.id, { recordId: existing.id });

    expect(reinforced.reinforcedCount).toBe(1);
    expect(reinforced.lastReinforcedAt).toBe(harness.clock.value);
    expect(await recordCount()).toBe(1);

    const closed = await harness.services.proposals.get(proposal.id);
    expect(closed.status).toBe("reinforced");
    expect(closed.resolution?.recordId).toBe(existing.id);
  });

  it("reinforce of a missing record is a 404 and leaves the proposal pending", async () => {
    const proposal = await pendingTile();

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: "nope" }),
    ).rejects.toMatchObject({ status: 404 });

    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");
  });
});

describe("supersede", () => {
  it("marks the old record superseded with link and reason", async () => {
    const old = await seedRecord(harness, { headline: "old claim" });
    const proposal = await pendingTile();

    const created = await harness.services.proposals.approve(proposal.id, {
      supersedes: [old.id],
      reason: "changed",
    });

    const view = await harness.services.records.get(old.id);
    expect(view.lifecycle).toBe("superseded");
    expect(view.supersededBy).toBe(created.id);
    expect(view.supersessionReason).toBe("changed");
    expect(view.lineage.supersededBy?.id).toBe(created.id);

    const fresh = await harness.services.records.get(created.id);
    expect(fresh.lifecycle).toBe("live");
    expect(fresh.lineage.supersedes.map((entry) => entry.id)).toEqual([old.id]);
  });

  it("requires a reason class and writes nothing without one", async () => {
    const old = await seedRecord(harness);
    const proposal = await pendingTile();

    await expect(
      harness.services.proposals.approve(proposal.id, { supersedes: [old.id] }),
    ).rejects.toMatchObject({ status: 400 });
    await expect(
      harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "because" }),
    ).rejects.toMatchObject({ status: 400 });

    expect(await recordCount()).toBe(1);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");
  });

  it("can supersede a legacy record", async () => {
    const legacy = await seedLegacy(harness);
    const proposal = await pendingTile();

    const created = await harness.services.proposals.approve(proposal.id, {
      supersedes: [legacy.id],
      reason: "correction",
    });

    const archived = await findById<{ lifecycle: string; supersededBy: string }>(
      harness.store.legacy,
      legacy.id,
    );
    expect(archived?.lifecycle).toBe("retired");
    expect(archived?.supersededBy).toBe(created.id);
  });

  it("refuses to supersede a record that is not live", async () => {
    const old = await seedRecord(harness, { lifecycle: "retired" });
    const proposal = await pendingTile();

    await expect(
      harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" }),
    ).rejects.toMatchObject({ status: 409 });
  });

  it("treats a repeated supersede id as one and approves", async () => {
    const old = await seedRecord(harness);
    const proposal = await pendingTile();

    const created = await harness.services.proposals.approve(proposal.id, {
      supersedes: [old.id, old.id],
      reason: "changed",
    });

    const view = await harness.services.records.get(old.id);
    expect(view.lifecycle).toBe("superseded");
    expect(view.supersededBy).toBe(created.id);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
    expect(await recordCount()).toBe(2);
  });

  it("unsupersede restores the old record", async () => {
    const old = await seedRecord(harness);
    const proposal = await pendingTile();
    await harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" });

    const restored = await harness.services.records.unsupersede(old.id, "wrong pairing");

    expect(restored.lifecycle).toBe("live");
    expect(restored.supersededBy).toBeNull();
    expect(restored.supersessionReason).toBeNull();
    expect(restored.lifecycleNote).toBe("wrong pairing");
  });

  it("unsupersede of a live record is a 409 and needs a reason", async () => {
    const live = await seedRecord(harness);

    await expect(harness.services.records.unsupersede(live.id, "x")).rejects.toMatchObject({ status: 409 });
    await expect(harness.services.records.unsupersede(live.id, "")).rejects.toMatchObject({ status: 400 });
  });
});

describe("targeted proposals", () => {
  it("retire approval marks the record retired with the reason", async () => {
    const stale = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: stale.id, source: "v2", action: "retire", reason: "relevance: shelve-clear" },
      }),
    );

    const retired = await harness.services.proposals.approve(proposal.id, {});

    expect(retired.lifecycle).toBe("retired");
    expect(retired.retiredReason).toBe("relevance: shelve-clear");
    expect(await recordCount()).toBe(1);
  });

  it("retire without any reason is a 400", async () => {
    const stale = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({ origin: "cleaner", target: { recordId: stale.id, source: "v2", action: "retire" } }),
    );

    await expect(harness.services.proposals.approve(proposal.id, {})).rejects.toMatchObject({ status: 400 });
  });

  it("merge retires the older record and returns the keeper", async () => {
    const older = await seedRecord(harness);
    const newer = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: older.id, source: "v2", action: "merge", replacementId: newer.id },
      }),
    );

    const keeper = await harness.services.proposals.approve(proposal.id, {});

    expect(keeper.id).toBe(newer.id);
    expect(keeper.lifecycle).toBe("live");

    const view = await harness.services.records.get(older.id);
    expect(view.lifecycle).toBe("retired");
    expect(view.retiredReason).toBe(`merged into ${newer.id}`);
  });

  it("carries the headline of the record and of the keeper on a merge", async () => {
    const older = await seedRecord(harness, { headline: "old duplicate" });
    const newer = await seedRecord(harness, { headline: "newer keeper" });

    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: older.id, source: "v2", action: "merge", replacementId: newer.id },
      }),
    );

    expect(proposal.target).toMatchObject({
      recordHeadline: "old duplicate",
      replacementHeadline: "newer keeper",
    });
    expect((await harness.services.proposals.get(proposal.id)).target).toMatchObject({
      recordHeadline: "old duplicate",
      replacementHeadline: "newer keeper",
    });
  });

  it("resolves a legacy target and a legacy keeper from the legacy table", async () => {
    const legacy = await seedLegacy(harness, { headline: "legacy note" });
    const keeper = await seedLegacy(harness, { headline: "legacy keeper" });

    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "promotion",
        target: { recordId: legacy.id, source: "legacy", action: "supersede", replacementId: keeper.id },
      }),
    );

    expect(proposal.target).toMatchObject({
      recordHeadline: "legacy note",
      replacementHeadline: "legacy keeper",
    });
  });

  it("leaves a headline null when the id is unknown or unset and ignores client-sent headlines", async () => {
    const stale = await seedRecord(harness, { headline: "stale one" });

    const retire = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: {
          recordId: stale.id,
          source: "v2",
          action: "retire",
          reason: "x",
          recordHeadline: "forged",
          replacementHeadline: "forged",
        },
      }),
    );
    const missing = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: stale.id, source: "v2", action: "merge", replacementId: "no-such-id" },
      }),
    );

    expect(retire.target).toMatchObject({ recordHeadline: "stale one", replacementHeadline: null });
    expect(missing.target).toMatchObject({ recordHeadline: "stale one", replacementHeadline: null });
  });

  it("reads back a pending proposal stored without the headline fields as null", async () => {
    const proposal = await pendingTile();
    const row = proposalToRow(proposal);

    row.target = JSON.stringify({ recordId: "r1", source: "v2", action: "merge", replacementId: "r2" });
    await upsert(harness.store.proposals, [row]);

    const read = await harness.services.proposals.get(proposal.id);

    expect(read.target).toEqual({
      recordId: "r1",
      source: "v2",
      action: "merge",
      replacementId: "r2",
      recordHeadline: null,
      replacementHeadline: null,
    });
  });

  it("merge of a record into itself is a 400 and leaves it live", async () => {
    const only = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: only.id, source: "v2", action: "merge", replacementId: only.id },
      }),
    );

    await expect(harness.services.proposals.approve(proposal.id, {})).rejects.toMatchObject({ status: 400 });

    expect((await harness.services.records.get(only.id)).lifecycle).toBe("live");
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");
  });

  it("supersede with a replacement links the two existing records", async () => {
    const older = await seedRecord(harness);
    const newer = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: {
          recordId: older.id,
          source: "v2",
          action: "supersede",
          replacementId: newer.id,
          reason: "changed",
        },
      }),
    );

    await harness.services.proposals.approve(proposal.id, {});

    const view = await harness.services.records.get(older.id);
    expect(view.lifecycle).toBe("superseded");
    expect(view.supersededBy).toBe(newer.id);
    expect(await recordCount()).toBe(2);
  });

  it("promote copies the legacy record into v2 with its legacyId", async () => {
    const legacy = await seedLegacy(harness, { purpose: "pitfall", topic: "tooling" });
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "promotion",
        headline: legacy.headline,
        fields: { claim: legacy.claim },
        target: { recordId: legacy.id, source: "legacy", action: "promote" },
      }),
    );

    expect(proposal.purpose).toBe("pitfall");

    const created = await harness.services.proposals.approve(proposal.id, {});

    expect(created.legacyId).toBe(legacy.id);
    expect(created.origin).toBe("promotion");
    expect(created.purpose).toBe("pitfall");
    expect(created.topic).toBe("tooling");

    const rows = await selectWhere<MemoryRecord>(harness.store.records, undefined);
    expect(rows.map((row) => row.id)).toEqual([created.id]);
  });
});

describe("promotion of a legacy record", () => {
  it("writes the legacy why into the promoted record and approves a why-less legacy row", async () => {
    const withWhy = await seedLegacy(harness, { why: "kept from the archive" });
    const withoutWhy = await seedLegacy(harness, { why: "" });
    const promote = (legacy: LegacyRecord) =>
      harness.services.proposals.createInternal(
        proposalBody({
          origin: "promotion",
          headline: legacy.headline,
          fields: { claim: legacy.claim },
          target: { recordId: legacy.id, source: "legacy", action: "promote" },
        }),
      );

    const first = await harness.services.proposals.approve((await promote(withWhy)).id, {});
    const second = await harness.services.proposals.approve((await promote(withoutWhy)).id, {});

    expect(first.why).toBe("kept from the archive");
    expect(second.why).toBe("");
    expect(await recordCount()).toBe(2);
  });
});

describe("remove", () => {
  it("deletes a record and requires a reason", async () => {
    const secret = await seedRecord(harness);

    await expect(harness.services.records.remove(secret.id, undefined)).rejects.toMatchObject({ status: 400 });
    await harness.services.records.remove(secret.id, "contains a secret");

    await expect(harness.services.records.get(secret.id)).rejects.toMatchObject({ status: 404 });
  });
});

async function legacyRow(id: string): Promise<LegacyRecord> {
  return (await findById<LegacyRecord>(harness.store.legacy, id)) as LegacyRecord;
}

async function recalled(q: string): Promise<string[]> {
  const { results } = await harness.services.recall.recall({ q });

  return results.map((entry) => entry.id);
}

describe("approve converges on retry", () => {
  it("retries after a failed resolve into one record, one supersession and an approved proposal", async () => {
    const old = await seedRecord(harness, { headline: "old claim" });
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "proposals", 1);

    await expect(
      harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" }),
    ).rejects.toThrow("injected");
    restore();

    expect(await recordCount()).toBe(2);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");

    const created = await harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" });

    expect(await recordCount()).toBe(2);
    expect((await harness.services.records.get(old.id)).supersededBy).toBe(created.id);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
    expect((await harness.services.proposals.get(proposal.id)).resolution?.recordId).toBe(created.id);
  });

  it("retries after a failed supersession without a duplicate record", async () => {
    const old = await seedRecord(harness);
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "records", 2);

    await expect(
      harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" }),
    ).rejects.toThrow("injected");
    restore();

    const created = await harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" });

    expect(await recordCount()).toBe(2);
    expect((await harness.services.records.get(old.id)).lifecycle).toBe("superseded");
    expect((await harness.services.records.get(old.id)).supersededBy).toBe(created.id);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
  });

  it("retries a plain approve after a failed resolve without a duplicate record", async () => {
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "proposals", 1);

    await expect(harness.services.proposals.approve(proposal.id, {})).rejects.toThrow("injected");
    restore();
    await harness.services.proposals.approve(proposal.id, {});

    expect(await recordCount()).toBe(1);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
  });

  it("retries a retire approval after a failed resolve", async () => {
    const stale = await seedRecord(harness);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: stale.id, source: "v2", action: "retire", reason: "stale" },
      }),
    );
    const restore = failNthWrite(harness, "proposals", 1);

    await expect(harness.services.proposals.approve(proposal.id, {})).rejects.toThrow("injected");
    restore();
    await harness.services.proposals.approve(proposal.id, {});

    expect((await harness.services.proposals.get(proposal.id)).status).toBe("approved");
  });
});

describe("reinforce idempotency and guards", () => {
  it("never bumps twice when resolve fails and the call is retried", async () => {
    const existing = await seedRecord(harness);
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "proposals", 1);

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: existing.id }),
    ).rejects.toThrow("injected");
    restore();
    await harness.services.proposals.reinforce(proposal.id, { recordId: existing.id });

    expect((await harness.services.records.get(existing.id)).reinforcedCount).toBe(1);
  });

  it("never bumps twice when the bump fails and the call is retried", async () => {
    const existing = await seedRecord(harness);
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "records", 1);

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: existing.id }),
    ).rejects.toThrow("injected");
    restore();
    await harness.services.proposals.reinforce(proposal.id, { recordId: existing.id }).catch(() => undefined);

    expect((await harness.services.records.get(existing.id)).reinforcedCount).toBeLessThanOrEqual(1);
  });

  it("reinforce on a retired v2 record is a 409 and leaves the proposal pending", async () => {
    const retired = await seedRecord(harness, { lifecycle: "retired" });
    const proposal = await pendingTile();

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: retired.id }),
    ).rejects.toMatchObject({ status: 409 });

    expect((await harness.services.proposals.get(proposal.id)).status).toBe("pending");
    expect((await harness.services.records.get(retired.id)).reinforcedCount).toBe(0);
  });

  it("bumpReinforcement refuses a record that is not live", async () => {
    const retired = await seedRecord(harness, { lifecycle: "retired" });

    await expect(harness.services.records.bumpReinforcement(retired.id)).rejects.toMatchObject({ status: 409 });
  });
});

describe("legacy retirement", () => {
  const pitfall = { headline: "legacy quartz note", claim: "quartz clocks drift", why: "measured on the bench" };

  it("promote retires the legacy row, links it to the copy and hides it from recall", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "promotion",
        headline: legacy.headline,
        fields: { claim: legacy.claim },
        target: { recordId: legacy.id, source: "legacy", action: "promote" },
      }),
    );

    expect(await recalled("quartz clocks drift")).toEqual([legacy.id]);

    const created = await harness.services.proposals.approve(proposal.id, {});
    const archived = await legacyRow(legacy.id);

    expect(archived.lifecycle).toBe("retired");
    expect(archived.supersededBy).toBe(created.id);
    expect(archived.claim).toBe(legacy.claim);
    expect(await recalled("quartz clocks drift")).toEqual([created.id]);
  });

  it("a supersede naming a legacy id retires it and points it at the new record", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await pendingTile({
      headline: "quartz clocks drift less",
      fields: { claim: "quartz clocks drift less" },
    });

    const created = await harness.services.proposals.approve(proposal.id, {
      supersedes: [legacy.id],
      reason: "correction",
    });
    const archived = await legacyRow(legacy.id);

    expect(archived.lifecycle).toBe("retired");
    expect(archived.supersededBy).toBe(created.id);
    expect(await recalled("quartz clocks drift")).toEqual([created.id]);
  });

  it("a reinforce on a legacy id promotes it with a count of one", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await pendingTile();

    const copy = await harness.services.proposals.reinforce(proposal.id, { recordId: legacy.id });
    const archived = await legacyRow(legacy.id);

    expect(copy.legacyId).toBe(legacy.id);
    expect(copy.reinforcedCount).toBe(1);
    expect(copy.headline).toBe(legacy.headline);
    expect(archived.lifecycle).toBe("retired");
    expect(archived.supersededBy).toBe(copy.id);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("reinforced");
    expect(await recalled("quartz clocks drift")).toEqual([copy.id]);
  });

  it("a legacy reinforce retried after a failed resolve leaves one copy with a count of one", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await pendingTile();
    const restore = failNthWrite(harness, "proposals", 1);

    await expect(
      harness.services.proposals.reinforce(proposal.id, { recordId: legacy.id }),
    ).rejects.toThrow("injected");
    restore();

    const copy = await harness.services.proposals.reinforce(proposal.id, { recordId: legacy.id });

    expect(await recordCount()).toBe(1);
    expect(copy.reinforcedCount).toBe(1);
    expect((await harness.services.proposals.get(proposal.id)).status).toBe("reinforced");
  });

  it("a retire on a legacy id retires it and hides it from recall", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: legacy.id, source: "legacy", action: "retire", reason: "stale" },
      }),
    );

    await harness.services.proposals.approve(proposal.id, {});

    expect((await legacyRow(legacy.id)).lifecycle).toBe("retired");
    expect(await recalled("quartz clocks drift")).toEqual([]);
  });

  it("a second promote of the same legacy row is a 409", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const promote = () =>
      harness.services.proposals.createInternal(
        proposalBody({
          origin: "promotion",
          headline: legacy.headline,
          fields: { claim: legacy.claim },
          target: { recordId: legacy.id, source: "legacy", action: "promote" },
        }),
      );
    const first = await promote();
    const second = await promote();

    await harness.services.proposals.approve(first.id, {});

    await expect(harness.services.proposals.approve(second.id, {})).rejects.toMatchObject({ status: 409 });
    expect(await recordCount()).toBe(1);
  });

  it("unsupersede on a promoted legacy row is a 409", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "promotion",
        headline: legacy.headline,
        fields: { claim: legacy.claim },
        target: { recordId: legacy.id, source: "legacy", action: "promote" },
      }),
    );
    await harness.services.proposals.approve(proposal.id, {});

    await expect(harness.services.records.unsupersede(legacy.id, "oops")).rejects.toMatchObject({ status: 409 });
    expect((await legacyRow(legacy.id)).lifecycle).toBe("retired");
  });

  it("unsupersede restores a retired legacy row that no v2 record took over", async () => {
    const legacy = await seedLegacy(harness, pitfall);
    const proposal = await harness.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: legacy.id, source: "legacy", action: "retire", reason: "stale" },
      }),
    );
    await harness.services.proposals.approve(proposal.id, {});

    const restored = await harness.services.records.unsupersede(legacy.id, "kept after all");

    expect(restored.lifecycle).toBe("live");
    expect(restored.supersededBy).toBeNull();
    expect(await recalled("quartz clocks drift")).toEqual([legacy.id]);
  });
});

describe("remove with dependents", () => {
  it("is a 409 listing the ids while another record's supersededBy points at the target", async () => {
    const old = await seedRecord(harness);
    const proposal = await pendingTile();
    const created = await harness.services.proposals.approve(proposal.id, { supersedes: [old.id], reason: "changed" });

    await expect(harness.services.records.remove(created.id, "secret")).rejects.toMatchObject({
      status: 409,
      message: expect.stringContaining(old.id),
    });
    expect((await harness.services.records.get(created.id)).lifecycle).toBe("live");

    await harness.services.records.remove(old.id, "secret");
    await harness.services.records.remove(created.id, "secret");
  });
});

describe("origin guard", () => {
  it("parseProposalBody rejects cleaner, promotion and harvest origins with 400 and keeps agent", () => {
    expect(() => parseProposalBody(proposalBody({ origin: "cleaner" }))).toThrow(
      expect.objectContaining({ status: 400 }),
    );
    expect(() => parseProposalBody(proposalBody({ origin: "promotion" }))).toThrow(
      expect.objectContaining({ status: 400 }),
    );
    expect(() => parseProposalBody(proposalBody({ origin: "harvest" }))).toThrow(
      expect.objectContaining({ status: 400 }),
    );
    expect(parseProposalBody(proposalBody({ origin: "agent" })).origin).toBe("agent");
    expect(parseProposalBody(proposalBody({ origin: "cleaner" }), { internal: true }).origin).toBe("cleaner");
  });

  it("create refuses a cleaner or harvest origin and createInternal accepts a cleaner one", async () => {
    await expect(
      harness.services.proposals.create(proposalBody({ origin: "cleaner" })),
    ).rejects.toMatchObject({ status: 400 });
    await expect(
      harness.services.proposals.create(proposalBody({ origin: "harvest" })),
    ).rejects.toMatchObject({ status: 400 });

    const proposal = await harness.services.proposals.createInternal(proposalBody({ origin: "cleaner" }));

    expect(proposal.origin).toBe("cleaner");
    expect(proposal.status).toBe("pending");
    expect(await recordCount()).toBe(0);
  });
});

describe("jev suggestions on propose", () => {
  async function jevHarnessWith(relation: string) {
    return makeHarness({
      proposals: { jev: stubJev({ relation: () => choiceAnswer(relation, 0.9) }) },
    });
  }

  it("applies a Jev reinforce suggestion at once and writes no record", async () => {
    const jevHarness = await jevHarnessWith("restatement");

    try {
      const existing = await seedRecord(jevHarness, {
        headline: proposalBody().headline,
        claim: proposalBody().fields.claim,
      });

      const proposal = await jevHarness.services.proposals.create(proposalBody());
      const counted = await jevHarness.services.records.get(existing.id);

      expect(proposal.status).toBe("reinforced");
      expect(proposal.resolution).toEqual({ recordId: existing.id });
      expect(proposal.suggestion).toMatchObject({ action: "reinforce", recordId: existing.id });
      expect(counted.reinforcedCount).toBe(1);
      expect(await jevHarness.store.records.countRows()).toBe(1);
      expect(await jevHarness.services.proposals.list("pending", null)).toEqual([]);
      expect((await jevHarness.services.proposals.get(proposal.id)).status).toBe("reinforced");
    } finally {
      jevHarness.cleanup();
    }
  });

  it("saves a record with a supersede suggestion and leaves the suggestion on the proposal row", async () => {
    const jevHarness = await jevHarnessWith("replacement");

    try {
      const existing = await seedRecord(jevHarness, {
        headline: proposalBody().headline,
        claim: proposalBody().fields.claim,
      });

      const proposal = await jevHarness.services.proposals.create(proposalBody());
      const row = await jevHarness.services.proposals.get(proposal.id);

      expect(proposal.status).toBe("approved");
      expect(row.suggestion).toMatchObject({ action: "supersede", recordId: existing.id });
      expect(row.flags).toContain(`replacement:${existing.id}`);
      expect((await jevHarness.services.records.get(proposal.id)).lifecycle).toBe("live");
      expect((await jevHarness.services.records.get(existing.id)).lifecycle).toBe("live");
      expect(await jevHarness.store.records.countRows()).toBe(2);
    } finally {
      jevHarness.cleanup();
    }
  });

  it("counts a reinforce suggestion on a defect and writes no second defect", async () => {
    const jevHarness = await jevHarnessWith("restatement");

    try {
      const body = defectBody();
      const existing = await seedRecord(jevHarness, {
        kind: "defect",
        headline: body.headline,
        claim: body.fields.claim,
        where: body.fields.where,
        trigger: body.fields.trigger,
        defectStatus: "open",
      });

      const proposal = await jevHarness.services.proposals.create(body);

      expect(proposal.status).toBe("reinforced");
      expect(proposal.resolution).toEqual({ recordId: existing.id });
      expect(await jevHarness.store.records.countRows()).toBe(1);
    } finally {
      jevHarness.cleanup();
    }
  });
});

describe("defect proposals", () => {
  const taker = { threadId: "thread-9", link: "t3-thread://thread-9" };

  async function approvedDefect(): Promise<MemoryRecord> {
    return createApprovedDefect(harness);
  }

  it("approves a defect on creation into an open record with its location", async () => {
    const proposal = await harness.services.proposals.create(
      defectBody({
        fields: { ...defectBody().fields, fixHint: "page with an offset", impact: "silent data loss" },
      }),
    );

    expect(proposal.kind).toBe("defect");
    expect(proposal.status).toBe("approved");
    expect(proposal.resolvedAt).not.toBeNull();
    expect(proposal.resolution).toEqual({ recordId: proposal.id, reason: null });
    expect(proposal.flags).toEqual(["jev_unavailable"]);
    expect(proposal.neighbours).toEqual([]);
    expect(await harness.services.proposals.list("pending", null)).toEqual([]);

    const record = await findById<MemoryRecord>(harness.store.records, proposal.resolution?.recordId as string);

    expect(record).toMatchObject({
      kind: "defect",
      lifecycle: "live",
      defectStatus: "open",
      where: "app/Jobs/SyncReport.php",
      trigger: "export a report with more than 500 rows",
      leftBecause: "the fix needs an offset change outside the ticket",
      fixHint: "page with an offset",
      impact: "silent data loss",
      takenThreadId: null,
      closedBy: null,
    });
    expect(await recalled("export job drops rows")).toContain(proposal.id);
  });

  it("closes the defect on propose with status approved and the defect's recordId", async () => {
    const defect = await approvedDefect();

    const tile = await harness.services.proposals.create({
      ...proposalBody(),
      kind: "defect",
      fields: { claim: "fixed by the offset change" },
      target: { recordId: defect.id, source: "v2", action: "close" },
    });

    expect(tile.status).toBe("approved");
    expect(tile.resolution).toMatchObject({ recordId: defect.id });
    expect(await harness.services.proposals.list("pending", null)).toEqual([]);
    expect(await harness.services.records.get(defect.id)).toMatchObject({
      defectStatus: "closed",
      closedBy: tile.id,
    });
  });

  it("accepts snake_case defect keys and a symptom alias on the wire", async () => {
    const proposal = await harness.services.proposals.create({
      ...defectBody(),
      fields: {
        symptom: "rows vanish",
        where: "app/Sync.php",
        trigger: "more than 500 rows",
        left_because: "outside the ticket",
        fix_hint: "use an offset",
        impact: "data loss",
      },
    });

    expect(proposal.fields).toMatchObject({
      claim: "rows vanish",
      leftBecause: "outside the ticket",
      fixHint: "use an offset",
      impact: "data loss",
    });
  });

  it.each([
    ["where", "where is required for a defect"],
    ["trigger", "trigger is required for a defect"],
    ["leftBecause", "left_because is required for a defect"],
    ["claim", "symptom is required for a defect"],
  ])("returns 400 naming %s when a defect lacks it", async (key, message) => {
    const body = defectBody();
    const fields = { ...body.fields } as Record<string, unknown>;
    delete fields[key];

    await expect(harness.services.proposals.create({ ...body, fields })).rejects.toMatchObject({
      status: 400,
      message,
    });
  });

  it("does not ask a fact for defect fields and drops them from the stored fields", async () => {
    const proposal = await harness.services.proposals.create({
      ...proposalBody(),
      fields: { claim: "a fact", where: "somewhere" },
    });

    expect(proposal.fields).not.toHaveProperty("where");
  });

  it("takes an open defect, refuses a second take and releases it", async () => {
    const defect = await approvedDefect();

    const taken = await harness.services.records.take(defect.id, taker);

    expect(taken).toMatchObject({
      defectStatus: "taken",
      takenThreadId: "thread-9",
      takenLink: "t3-thread://thread-9",
      takenAt: harness.clock.value,
      takenBy: { threadId: "thread-9", link: "t3-thread://thread-9", at: harness.clock.value },
    });
    await expect(harness.services.records.take(defect.id, taker)).rejects.toMatchObject({ status: 409 });

    const released = await harness.services.records.release(defect.id);

    expect(released).toMatchObject({
      defectStatus: "open",
      takenThreadId: null,
      takenLink: null,
      takenAt: taken.takenAt,
    });
    expect(released.takenBy).toBeNull();
    expect((await harness.services.records.take(defect.id, taker)).defectStatus).toBe("taken");
  });

  it("releases an untaken defect as a no-op", async () => {
    const defect = await approvedDefect();

    expect((await harness.services.records.release(defect.id)).defectStatus).toBe("open");
  });

  it("refuses take and release on a non-defect with 400 and a missing record with 404", async () => {
    const fact = await seedRecord(harness);

    await expect(harness.services.records.take(fact.id, taker)).rejects.toMatchObject({ status: 400 });
    await expect(harness.services.records.release(fact.id)).rejects.toMatchObject({ status: 400 });
    await expect(harness.services.records.take("missing", taker)).rejects.toMatchObject({ status: 404 });
  });

  it("requires a thread id and a link to take", async () => {
    const defect = await approvedDefect();

    await expect(harness.services.records.take(defect.id, { threadId: "t" })).rejects.toMatchObject({ status: 400 });
    await expect(harness.services.records.take(defect.id, undefined)).rejects.toMatchObject({ status: 400 });
  });

  it("closes a taken defect on propose and writes no record", async () => {
    const defect = await approvedDefect();
    await harness.services.records.take(defect.id, taker);
    const tile = await harness.services.proposals.create(
      proposalBody({
        kind: "fact",
        headline: "export now pages with an offset",
        fields: { claim: "the offset fix landed with test SyncReportTest" },
        target: { recordId: defect.id, source: "v2", action: "close" },
      }),
    );

    expect(tile.status).toBe("approved");
    expect(tile.checks.status).toBe("skipped");
    expect(tile.target).toMatchObject({ action: "close", recordHeadline: defect.headline });

    const closed = await harness.services.records.get(defect.id);
    expect(closed).toMatchObject({ defectStatus: "closed", closedBy: tile.id, closedAt: harness.clock.value });
    expect(await recordCount()).toBe(1);
    expect((await harness.services.proposals.get(tile.id)).status).toBe("approved");
    await expect(harness.services.records.take(defect.id, taker)).rejects.toMatchObject({ status: 409 });
    expect((await harness.services.records.release(defect.id)).defectStatus).toBe("closed");
  });

  it("accepts a close tile that names a defect kind without defect fields", async () => {
    const defect = await approvedDefect();

    const tile = await harness.services.proposals.create({
      ...defectBody(),
      fields: { claim: "fixed in the offset change" },
      target: { recordId: defect.id, source: "v2", action: "close" },
    });

    expect(tile.fields).toEqual({ claim: "fixed in the offset change", rejected: null });
  });

  it("refuses a close tile for a missing record, a fact or an already closed defect", async () => {
    const defect = await approvedDefect();
    const fact = await seedRecord(harness);
    const close = (recordId: string) =>
      harness.services.proposals.create(
        proposalBody({ target: { recordId, source: "v2", action: "close" } }),
      );
    await close(defect.id);

    await expect(close("missing")).rejects.toMatchObject({ status: 404 });
    await expect(close(fact.id)).rejects.toMatchObject({ status: 409 });
    await expect(close(defect.id)).rejects.toMatchObject({ status: 409 });
  });

  it("returns 409 on the second create when two close proposals race for one defect", async () => {
    const defect = await approvedDefect();
    const make = () =>
      harness.services.proposals.create(
        proposalBody({ target: { recordId: defect.id, source: "v2", action: "close" } }),
      );

    const results = await Promise.allSettled([make(), make()]);
    const rejected = results.filter((result) => result.status === "rejected");
    const fulfilled = results.filter((result) => result.status === "fulfilled");

    expect(fulfilled).toHaveLength(1);
    expect(rejected).toHaveLength(1);
    expect((rejected[0] as PromiseRejectedResult).reason).toMatchObject({ status: 409 });
    expect((await harness.services.records.get(defect.id)).defectStatus).toBe("closed");
    await expect(make()).rejects.toMatchObject({ status: 409 });
  });
});

describe("covers routing on propose", () => {
  function coverage(probability: number) {
    return () => ({
      type: "choice",
      choice: probability >= 0.5 ? "yes" : "no",
      confidence: Math.max(probability, 1 - probability),
      probabilities: { yes: probability, no: 1 - probability },
    });
  }

  async function coversHarness(
    probability: number,
    options: { relation?: string; containment?: boolean; foldAsk?: boolean } = {},
  ) {
    return makeHarness({
      proposals: {
        jev: stubJev({
          new_covers_old: coverage(probability),
          relation: () => choiceAnswer(options.relation ?? "distinct", 0.9),
          reason: () => choiceAnswer("changed", 0.9),
        }),
        containment: options.containment,
        foldAsk: options.foldAsk ?? true,
      },
    });
  }

  const sameText = () => ({ headline: proposalBody().headline, claim: proposalBody().fields.claim });

  function coversFlags(flags: string[]): string[] {
    return flags.filter((flag) => flag.startsWith("covers:"));
  }

  it("asks on a v2 neighbour at 0.95 and supersedes nothing", async () => {
    const h = await coversHarness(0.95);

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("asked");
      expect(proposal.ask?.existing.id).toBe(existing.id);
      expect(proposal.flags).toContain(`covers:${existing.id}`);
      expect((await h.services.proposals.get(proposal.id)).flags).toContain(`covers:${existing.id}`);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect(await h.store.records.countRows()).toBe(1);
    } finally {
      h.cleanup();
    }
  });

  it("asks at 0.95 even when similarity and same_subject sit below the fold trigger", async () => {
    const h = await makeHarness({
      proposals: {
        foldAsk: true,
        jev: stubJev({
          new_covers_old: coverage(0.95),
          same_subject: coverage(0.1),
          relation: () => choiceAnswer("distinct", 0.9),
        }),
      },
    });

    try {
      const existing = await seedRecord(h, {
        headline: "thumbnail worker restarts",
        claim: "the thumbnail worker restarts after every release",
      });

      const proposal = await h.services.proposals.create(proposalBody());
      const neighbour = proposal.neighbours.find((entry) => entry.id === existing.id);

      expect(neighbour?.similarity).toBeLessThan(0.8);
      expect(neighbour?.pair?.sameSubject).toBeLessThan(0.7);
      expect(proposal.status).toBe("asked");
      expect(proposal.ask?.existing.id).toBe(existing.id);
      expect(proposal.flags).toContain(`covers:${existing.id}`);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(existing.id)).supersededBy).toBeNull();
    } finally {
      h.cleanup();
    }
  });

  it("asks on a legacy neighbour at 0.95 and leaves the legacy row live", async () => {
    const h = await coversHarness(0.95);

    try {
      const legacy = await seedLegacy(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("asked");
      expect(proposal.ask?.existing).toMatchObject({ id: legacy.id, source: "legacy" });
      expect(proposal.flags).toContain(`covers:${legacy.id}`);
      expect((await h.services.records.get(legacy.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("saves as new with the flag and supersedes nothing when foldAsk is off", async () => {
    const h = await coversHarness(0.95, { foldAsk: false });

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("approved");
      expect(proposal.ask).toBeNull();
      expect(proposal.flags).toContain(`covers:${existing.id}`);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(proposal.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(existing.id)).supersededBy).toBeNull();
    } finally {
      h.cleanup();
    }
  });

  it("does not route a neighbour at 0.94 through covers", async () => {
    const h = await coversHarness(0.94);

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("approved");
      expect(coversFlags(proposal.flags)).toEqual([]);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(proposal.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("ignores a global neighbour when the proposal is project scoped", async () => {
    const h = await coversHarness(0.95);

    try {
      const existing = await seedRecord(h, { ...sameText(), scope: "global" });

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("approved");
      expect(coversFlags(proposal.flags)).toEqual([]);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(proposal.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("ignores a defect neighbour", async () => {
    const h = await coversHarness(0.95);

    try {
      const existing = await seedRecord(h, {
        ...sameText(),
        kind: "defect",
        where: "app/x.php",
        trigger: "run it",
        leftBecause: "later",
        defectStatus: "open",
      });

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("approved");
      expect(coversFlags(proposal.flags)).toEqual([]);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("does not route a defect proposal", async () => {
    const h = await coversHarness(0.95);

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(defectBody());

      expect(proposal.status).toBe("approved");
      expect(coversFlags(proposal.flags)).toEqual([]);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(proposal.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("reinforces on a restatement even at 0.95 and supersedes nothing", async () => {
    const h = await coversHarness(0.95, { relation: "restatement" });

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());
      const counted = await h.services.records.get(existing.id);

      expect(proposal.status).toBe("reinforced");
      expect(counted.lifecycle).toBe("live");
      expect(counted.reinforcedCount).toBe(1);
      expect(await h.store.records.countRows()).toBe(1);
    } finally {
      h.cleanup();
    }
  });

  it("asks on the covered neighbour before a replacement suggestion is stored", async () => {
    const h = await coversHarness(0.95, { relation: "replacement" });

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("asked");
      expect(proposal.ask?.existing.id).toBe(existing.id);
      expect(proposal.flags).toEqual(
        expect.arrayContaining([`replacement:${existing.id}`, `covers:${existing.id}`]),
      );
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("stores the replacement suggestion and supersedes nothing when foldAsk is off", async () => {
    const h = await coversHarness(0.95, { relation: "replacement", foldAsk: false });

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());
      const row = await h.services.proposals.get(proposal.id);

      expect(proposal.status).toBe("approved");
      expect(row.suggestion).toMatchObject({ action: "supersede", recordId: existing.id });
      expect(row.flags).toEqual(
        expect.arrayContaining([`replacement:${existing.id}`, `covers:${existing.id}`]),
      );
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("asks on the neighbour with the highest probability only", async () => {
    const h = await makeHarness({
      proposals: {
        foldAsk: true,
        jev: stubJev({
          new_covers_old: (_q, req) =>
            coverage((req.state as any).existing.includes("closer") ? 0.99 : 0.96)(),
        }),
      },
    });

    try {
      const weaker = await seedRecord(h, sameText());
      const stronger = await seedRecord(h, { headline: "closer repl note", claim: "closer repl readline" });

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("asked");
      expect(proposal.ask?.existing.id).toBe(stronger.id);
      expect(coversFlags(proposal.flags)).toEqual([`covers:${stronger.id}`]);
      expect((await h.services.records.get(stronger.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(weaker.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });

  it("does nothing when containment is off", async () => {
    const h = await coversHarness(0.95, { containment: false });

    try {
      const existing = await seedRecord(h, sameText());

      const proposal = await h.services.proposals.create(proposalBody());

      expect(proposal.status).toBe("approved");
      expect(coversFlags(proposal.flags)).toEqual([]);
      expect((await h.services.records.get(existing.id)).lifecycle).toBe("live");
      expect((await h.services.records.get(proposal.id)).lifecycle).toBe("live");
    } finally {
      h.cleanup();
    }
  });
});
