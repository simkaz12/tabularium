import { request } from "node:http";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, describe, expect, it, vi } from "vitest";

import { recordText } from "./domain.ts";
import type { MemoryRecord } from "./domain.ts";
import { createHttpServer } from "./http.ts";
import type { ServiceOptions } from "./http.ts";
import {
  choiceAnswer,
  defectBody,
  failNthWrite,
  makeHarness,
  proposalBody,
  seedRecord,
  stubJev,
} from "./testing.ts";
import type { Harness } from "./testing.ts";

const HOST = "allowed.test:8789";

let harness: Harness | null = null;
let server: Server | null = null;

afterEach(async () => {
  if (server !== null) {
    await new Promise<void>((resolve) => server?.close(() => resolve()));
    server = null;
  }

  harness?.cleanup();
  harness = null;
});

function yesNo(probability: number) {
  return () => ({
    type: "choice",
    choice: probability >= 0.5 ? "yes" : "no",
    confidence: Math.max(probability, 1 - probability),
    probabilities: { yes: probability, no: 1 - probability },
  });
}

interface Setup {
  relation?: string;
  sameSubject?: number;
  newCoversOld?: number;
  foldAsk?: boolean;
  timeoutMs?: number;
}

async function setup(options: Setup = {}): Promise<Harness> {
  const proposals: NonNullable<ServiceOptions["proposals"]> = {
    jev: stubJev({
      relation: () => choiceAnswer(options.relation ?? "follow_up", 0.9),
      same_subject: yesNo(options.sameSubject ?? 0.8),
      new_covers_old: yesNo(options.newCoversOld ?? 0.3),
    }),
    foldAsk: options.foldAsk ?? true,
    foldAskTimeoutMs: options.timeoutMs ?? 600_000,
  };

  harness = await makeHarness({ proposals });

  return harness;
}

async function vectorAt(h: Harness, similarity: number): Promise<number[]> {
  const body = proposalBody();
  const base = await h.embedder.embed(recordText({ kind: body.kind, headline: body.headline, ...body.fields }));
  const other = await h.embedder.embed("zebra quartz mango lattice");
  const overlap = base.reduce((sum, value, index) => sum + value * (other[index] as number), 0);
  const raw = other.map((value, index) => value - overlap * (base[index] as number));
  const norm = Math.hypot(...raw);
  const side = Math.sqrt(1 - similarity * similarity);

  return base.map((value, index) => similarity * value + (side * (raw[index] as number)) / norm);
}

async function seedNear(h: Harness, overrides: Partial<MemoryRecord> = {}, similarity = 0.9) {
  return seedRecord(h, {
    headline: "repl readline history note",
    claim: "repl keeps its history in a readline file",
    vector: await vectorAt(h, similarity),
    ...overrides,
  });
}

const foldInput = () => ({
  action: "fold",
  kind: "fact",
  headline: "repl history file notes",
  fields: { claim: "repl keeps its history in a readline file and aborts when it is unwritable" },
});

function live(h: Harness, id: string) {
  return h.services.records.get(id);
}

function call(path: string, method = "GET", body?: unknown) {
  const payload = body === undefined ? undefined : JSON.stringify(body);

  return new Promise<{ status: number; body: any }>((resolve, reject) => {
    const { port } = (server as Server).address() as AddressInfo;
    const req = request(
      `http://127.0.0.1:${port}/v2${path}`,
      {
        method,
        headers: {
          host: HOST,
          "content-type": "application/json",
          "content-length": Buffer.byteLength(payload ?? ""),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];

        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () =>
          resolve({ status: res.statusCode ?? 0, body: JSON.parse(Buffer.concat(chunks).toString("utf8")) }),
        );
      },
    );

    req.on("error", reject);
    req.end(payload);
  });
}

async function serve(h: Harness): Promise<void> {
  server = createHttpServer(h.services, [HOST]);
  await new Promise<void>((resolve) => (server as Server).listen(0, "127.0.0.1", resolve));
}

describe("fold ask trigger", () => {
  it("holds the note at the floors with a follow_up pick and returns the existing record", async () => {
    const h = await setup({ sameSubject: 0.7 });
    const existing = await seedNear(h, {}, 0.801);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("asked");
    expect(proposal.resolvedAt).toBeNull();
    expect(await h.store.records.countRows()).toBe(1);
    expect(proposal.ask?.existing).toEqual({
      id: existing.id,
      source: "v2",
      kind: "fact",
      headline: existing.headline,
      fields: { claim: existing.claim, rejected: null },
    });
    expect(proposal.ask?.answer).toBeNull();
    expect(proposal.ask?.existingIsFold).toBe(false);
    expect(proposal.ask?.expiresAt).toBe("2026-11-01T10:10:00.000Z");
    expect((await h.services.proposals.get(proposal.id)).status).toBe("asked");
    expect(await h.services.proposals.list("pending", null)).toEqual([]);
  });

  it("carries the existing record's why in the held ask", async () => {
    const h = await setup({ sameSubject: 0.7 });
    const existing = await seedNear(h, { why: "Confirm the copy with diff -rq a b." }, 0.801);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("asked");
    expect(proposal.ask?.existing.fields).toEqual({
      claim: existing.claim,
      why: "Confirm the copy with diff -rq a b.",
      rejected: null,
    });
    expect((await h.services.proposals.get(proposal.id)).ask?.existing.fields.why).toBe(
      "Confirm the copy with diff -rq a b.",
    );
  });

  it("marks an existing record that already replaced another as a fold", async () => {
    const h = await setup();
    const older = await seedRecord(h, { headline: "older", claim: "older claim" });
    const existing = await seedNear(h);
    await h.services.records.markSuperseded(older.id, existing.id, "elaborated");

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.ask?.existingIsFold).toBe(true);
  });

  it("saves as new below the similarity floor", async () => {
    const h = await setup();
    await seedNear(h, {}, 0.79);

    expect((await h.services.proposals.create(proposalBody())).status).toBe("approved");
  });

  it("saves as new below the subject floor", async () => {
    const h = await setup({ sameSubject: 0.69 });
    await seedNear(h);

    expect((await h.services.proposals.create(proposalBody())).status).toBe("approved");
  });

  it("saves as new on a distinct pick", async () => {
    const h = await setup({ relation: "distinct" });
    await seedNear(h);

    expect((await h.services.proposals.create(proposalBody())).status).toBe("approved");
  });

  it("saves as new when the neighbour is in another scope", async () => {
    const h = await setup();
    await seedNear(h, { scope: "global" });

    expect((await h.services.proposals.create(proposalBody())).status).toBe("approved");
  });

  it("saves as new when the neighbour is a defect", async () => {
    const h = await setup();
    await seedNear(h, {
      kind: "defect",
      where: "app/x.php",
      trigger: "run it",
      leftBecause: "later",
      defectStatus: "open",
    });

    expect((await h.services.proposals.create(proposalBody())).status).toBe("approved");
  });

  it("saves a defect proposal as new", async () => {
    const h = await setup();
    await seedNear(h);

    expect((await h.services.proposals.create(defectBody())).status).toBe("approved");
  });

  it("saves as new when foldAsk is off", async () => {
    const h = await setup({ foldAsk: false });
    await seedNear(h);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.ask).toBeNull();
  });

  it("asks on a covers hit at 0.95 whatever the similarity and subject", async () => {
    const h = await setup({ newCoversOld: 0.95, sameSubject: 0.1, relation: "distinct" });
    const existing = await seedNear(h, {}, 0.7);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("asked");
    expect(proposal.ask?.existing.id).toBe(existing.id);
    expect(proposal.ask?.similarity).toBeCloseTo(0.7, 2);
    expect(proposal.ask?.sameSubject).toBeCloseTo(0.1);
    expect(proposal.ask?.newCoversOld).toBeCloseTo(0.95);
    expect(proposal.flags).toContain(`covers:${existing.id}`);
    expect((await live(h, existing.id)).lifecycle).toBe("live");
    expect(await h.store.records.countRows()).toBe(1);
  });

  it("does not ask through covers at 0.94 when the trigger does not fire", async () => {
    const h = await setup({ newCoversOld: 0.94, sameSubject: 0.1, relation: "distinct" });
    await seedNear(h, {}, 0.7);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.flags.some((flag) => flag.startsWith("covers:"))).toBe(false);
  });

  it("saves as new with the covers flag when foldAsk is off", async () => {
    const h = await setup({ newCoversOld: 0.95, sameSubject: 0.1, foldAsk: false });
    const existing = await seedNear(h, {}, 0.7);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.ask).toBeNull();
    expect(proposal.flags).toContain(`covers:${existing.id}`);
    expect((await live(h, existing.id)).lifecycle).toBe("live");
    expect((await live(h, existing.id)).supersededBy).toBeNull();
  });

  it("ignores a covers hit in another scope", async () => {
    const h = await setup({ newCoversOld: 0.95, sameSubject: 0.1 });
    await seedNear(h, { scope: "global" }, 0.7);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.flags.some((flag) => flag.startsWith("covers:"))).toBe(false);
  });

  it("ignores a covers hit on a defect", async () => {
    const h = await setup({ newCoversOld: 0.95, sameSubject: 0.1 });
    await seedNear(
      h,
      { kind: "defect", where: "app/x.php", trigger: "run it", leftBecause: "later", defectStatus: "open" },
      0.7,
    );

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.flags.some((flag) => flag.startsWith("covers:"))).toBe(false);
  });

  it("reinforces on a restatement even when the neighbour is covered", async () => {
    const h = await setup({ newCoversOld: 0.95, relation: "restatement" });
    const existing = await seedNear(h);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("reinforced");
    expect((await live(h, existing.id)).lifecycle).toBe("live");
  });

  it("folds on a covers ask and supersedes the existing record only then", async () => {
    const h = await setup({ newCoversOld: 0.95, sameSubject: 0.1, relation: "distinct" });
    const existing = await seedNear(h, {}, 0.7);
    const asked = await h.services.proposals.create(proposalBody());

    expect((await live(h, existing.id)).lifecycle).toBe("live");

    const answered = await h.services.proposals.answer(asked.id, foldInput());

    expect(answered.status).toBe("folded");
    expect((await live(h, existing.id)).supersededBy).toBe(answered.resolution?.recordId);
  });

  it("does not ask when the relation is a replacement", async () => {
    const h = await setup({ relation: "replacement" });
    await seedNear(h);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.suggestion?.action).toBe("supersede");
  });
});

describe("fold answer", () => {
  it("saves the fold, supersedes the existing record and logs coverage", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());

    const answered = await h.services.proposals.answer(asked.id, foldInput());
    const foldId = answered.resolution?.recordId as string;
    const fold = await live(h, foldId);
    const old = await live(h, existing.id);

    expect(answered.status).toBe("folded");
    expect(answered.ask?.answer).toBe("fold");
    expect(answered.ask?.foldRecordId).toBe(foldId);
    expect(answered.ask?.foldCoversOld).toBeCloseTo(0.3);
    expect(answered.ask?.foldCoversNew).toBeCloseTo(0.3);
    expect(fold.lifecycle).toBe("live");
    expect(fold.headline).toBe("repl history file notes");
    expect(fold.threadId).toBe(asked.provenance.threadId);
    expect(fold.link).toBe(asked.provenance.link);
    expect(fold.excerpt).toBe(asked.provenance.excerpt);
    expect(fold.scope).toBe(asked.scope);
    expect(old.lifecycle).toBe("superseded");
    expect(old.supersededBy).toBe(foldId);
    expect(old.supersessionReason).toBe("elaborated");
    expect(await h.store.records.countRows()).toBe(2);

    const restored = await h.services.records.unsupersede(existing.id, "wrong fold");

    expect(restored.lifecycle).toBe("live");
    expect((await h.services.proposals.get(asked.id)).status).toBe("folded");
  });

  it("keep_both without a reason is a 400", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());

    await expect(h.services.proposals.answer(asked.id, { action: "keep_both" })).rejects.toMatchObject({
      status: 400,
    });
    await expect(
      h.services.proposals.answer(asked.id, { action: "keep_both", reason: "  " }),
    ).rejects.toMatchObject({ status: 400 });
    expect((await h.services.proposals.get(asked.id)).status).toBe("asked");
  });

  it("keep_both with a reason saves the held note beside the existing record", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());

    const answered = await h.services.proposals.answer(asked.id, {
      action: "keep_both",
      reason: "the existing note is about the history path, mine is about the abort",
    });

    expect(answered.status).toBe("approved");
    expect(answered.ask?.answer).toBe("keep_both");
    expect(answered.ask?.reason).toContain("history path");
    expect(answered.resolution?.recordId).toBe(asked.id);
    expect((await live(h, asked.id)).lifecycle).toBe("live");
    expect((await live(h, existing.id)).lifecycle).toBe("live");
  });

  it("answers 409 to a second, different answer", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    await h.services.proposals.answer(asked.id, { action: "keep_both", reason: "separate facts" });

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });
    await expect(
      h.services.proposals.answer(asked.id, { action: "keep_both", reason: "another reason" }),
    ).rejects.toMatchObject({ status: 409 });
  });

  it("replays a completed fold answer without writing again", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const first = await h.services.proposals.answer(asked.id, foldInput());
    const rows = await h.store.records.countRows();

    const replay = await h.services.proposals.answer(asked.id, foldInput());

    expect(replay.status).toBe("folded");
    expect(replay.resolution?.recordId).toBe(first.resolution?.recordId);
    expect(await h.store.records.countRows()).toBe(rows);
  });

  it("replays a completed keep_both answer without writing again", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const input = { action: "keep_both", reason: "separate facts" };
    await h.services.proposals.answer(asked.id, input);
    const rows = await h.store.records.countRows();

    const replay = await h.services.proposals.answer(asked.id, input);

    expect(replay.status).toBe("approved");
    expect(await h.store.records.countRows()).toBe(rows);
  });

  it("answers 409 and keeps the row asked when the existing record is no longer live", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const other = await seedRecord(h, { headline: "other", claim: "other claim" });
    await h.services.records.markSuperseded(existing.id, other.id, "changed");

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });

    const row = await h.services.proposals.get(asked.id);

    expect(row.status).toBe("asked");
    expect(row.ask?.answer).toBeNull();
  });

  it("answers 404 for an unknown id and 409 for a proposal that was never asked", async () => {
    const h = await setup({ foldAsk: false });

    await expect(h.services.proposals.answer("nope", foldInput())).rejects.toMatchObject({ status: 404 });

    const plain = await h.services.proposals.create(proposalBody());

    await expect(h.services.proposals.answer(plain.id, foldInput())).rejects.toMatchObject({ status: 409 });
  });

  it("leaves both coverage numbers null when the question fails", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const stub = stubJev({ new_covers_old: () => ({ type: "choice", choice: "maybe", confidence: 0.5 }) });
    (h.services.proposals as any).jev = stub;

    const answered = await h.services.proposals.answer(asked.id, foldInput());

    expect(answered.status).toBe("folded");
    expect(answered.ask?.foldCoversOld).toBeNull();
    expect(answered.ask?.foldCoversNew).toBeNull();
  });
});

describe("fold ask expiry", () => {
  it("saves the held note as new one millisecond past expiresAt", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();

    const expired = await h.services.proposals.expireAsks(past);
    const row = await h.services.proposals.get(asked.id);

    expect(expired).toEqual([asked.id]);
    expect(row.status).toBe("approved");
    expect(row.ask?.answer).toBe("unanswered");
    expect(row.ask?.answeredAt).toBe(past);
    expect((await live(h, asked.id)).lifecycle).toBe("live");
    expect((await live(h, existing.id)).lifecycle).toBe("live");
  });

  it("leaves the note held one millisecond before expiresAt", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const before = new Date(Date.parse(asked.ask?.expiresAt as string) - 1).toISOString();

    expect(await h.services.proposals.expireAsks(before)).toEqual([]);
    expect((await h.services.proposals.get(asked.id)).status).toBe("asked");
    expect(await h.store.records.countRows()).toBe(1);
  });

  it("does not expire an answered ask and rejects a late answer", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();
    await h.services.proposals.expireAsks(past);

    expect(await h.services.proposals.expireAsks(past)).toEqual([]);
    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });
  });
});

describe("fold answer intent", () => {
  const steps = [
    { label: "the fold insert", table: "records", nth: 1 },
    { label: "the supersede write", table: "records", nth: 2 },
    { label: "the folded status write", table: "proposals", nth: 2 },
  ] as const;

  async function expectFinishedFold(h: Harness, askedId: string, existingId: string) {
    const row = await h.services.proposals.get(askedId);

    expect(row.status).toBe("folded");
    expect(row.resolution?.recordId).toBe(row.ask?.foldRecordId);
    await expectFolded(h, row.ask?.foldRecordId as string, existingId);
    expect(await h.store.records.countRows()).toBe(2);
    await expect(live(h, askedId)).rejects.toMatchObject({ status: 404 });
  }

  for (const step of steps) {
    it(`finishes the fold on a later expireAsks after a failure on ${step.label}`, async () => {
      const h = await setup();
      const existing = await seedNear(h);
      const asked = await h.services.proposals.create(proposalBody());
      const restore = failNthWrite(h, step.table, step.nth);

      await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
      restore();

      const interrupted = await h.services.proposals.get(asked.id);

      expect(interrupted.status).toBe("asked");
      expect(interrupted.ask?.answer).toBe("fold");
      expect(interrupted.ask?.foldRecordId).not.toBeNull();

      const finished = await h.services.proposals.expireAsks(
        new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString(),
      );

      expect(finished).toEqual([asked.id]);
      await expectFinishedFold(h, asked.id, existing.id);
      expect(await h.services.proposals.expireAsks()).toEqual([]);
    });

    it(`finishes the fold on a repeated answer after a failure on ${step.label}`, async () => {
      const h = await setup();
      const existing = await seedNear(h);
      const asked = await h.services.proposals.create(proposalBody());
      const restore = failNthWrite(h, step.table, step.nth);

      await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
      restore();

      const replay = await h.services.proposals.answer(asked.id, foldInput());

      expect(replay.status).toBe("folded");
      await expectFinishedFold(h, asked.id, existing.id);
    });
  }

  async function interruptedThenRetired(h: Harness, existingId: string) {
    const asked = await h.services.proposals.create(proposalBody());
    const restore = failNthWrite(h, "records", 1);

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
    restore();
    await h.services.records.markRetired(existingId, "removed by the owner");

    return asked;
  }

  async function expectFoldedOverGoneExisting(h: Harness, askedId: string, existingId: string) {
    const row = await h.services.proposals.get(askedId);
    const old = await live(h, existingId);

    expect(row.status).toBe("folded");
    expect(row.ask?.existingGone).toBe(true);
    expect((await live(h, row.ask?.foldRecordId as string)).lifecycle).toBe("live");
    expect(old.lifecycle).toBe("retired");
    expect(old.supersededBy).toBeNull();
    expect(await h.store.records.countRows()).toBe(2);
  }

  it("finishes the fold on the sweep when the existing record was retired after the intent write", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await interruptedThenRetired(h, existing.id);
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);

    expect(await h.services.proposals.expireAsks(past)).toEqual([asked.id]);
    await expectFoldedOverGoneExisting(h, asked.id, existing.id);
    expect(await h.services.proposals.expireAsks(past)).toEqual([]);
    expect(errors).not.toHaveBeenCalled();
    errors.mockRestore();
  });

  it("finishes the fold on a repeated answer when the existing record was retired after the intent write", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await interruptedThenRetired(h, existing.id);

    const replay = await h.services.proposals.answer(asked.id, foldInput());

    expect(replay.status).toBe("folded");
    expect(replay.ask?.existingGone).toBe(true);
    await expectFoldedOverGoneExisting(h, asked.id, existing.id);
  });

  it("finishes an interrupted fold on the sweep before the ask is due", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const restore = failNthWrite(h, "records", 1);

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
    restore();

    expect(await h.services.proposals.expireAsks(asked.createdAt)).toEqual([asked.id]);
    await expectFinishedFold(h, asked.id, existing.id);
  });

  it("leaves the ask unanswered with no fold id when the intent write itself failed", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const restore = failNthWrite(h, "proposals", 1);

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
    restore();

    const row = await h.services.proposals.get(asked.id);

    expect(row.status).toBe("asked");
    expect(row.ask?.answer).toBeNull();
    expect(row.ask?.foldRecordId).toBeNull();
    expect(await h.store.records.countRows()).toBe(1);
  });

  it("returns the same folded proposal for two identical answers started together", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());

    const [a, b] = await Promise.all([
      h.services.proposals.answer(asked.id, foldInput()),
      h.services.proposals.answer(asked.id, foldInput()),
    ]);

    expect(a.status).toBe("folded");
    expect(b.status).toBe("folded");
    expect(b.resolution?.recordId).toBe(a.resolution?.recordId);
    await expectFinishedFold(h, asked.id, existing.id);
  });

  it("answers 409 to a replay whose kind differs from the stored fold", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const fields = { claim: "repl keeps its history in a readline file", rejected: "a custom path" };
    await h.services.proposals.answer(asked.id, { ...foldInput(), kind: "fact", fields });

    await expect(
      h.services.proposals.answer(asked.id, { ...foldInput(), kind: "decision", fields }),
    ).rejects.toMatchObject({ status: 409 });
    expect((await h.services.proposals.answer(asked.id, { ...foldInput(), kind: "fact", fields })).status).toBe(
      "folded",
    );
  });
});

describe("held note intent", () => {
  const steps = [
    { label: "the held insert", table: "records", nth: 1 },
    { label: "the approved status write", table: "proposals", nth: 2 },
  ] as const;

  const keepBoth = { action: "keep_both", reason: "separate fact" };

  async function interrupt(
    h: Harness,
    step: { table: "records" | "proposals"; nth: number },
    run: (id: string) => Promise<unknown>,
  ) {
    const asked = await h.services.proposals.create(proposalBody());
    const restore = failNthWrite(h, step.table, step.nth);
    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);

    await run(asked.id).catch(() => undefined);
    restore();
    errors.mockRestore();

    return asked;
  }

  async function expectOneHeldNote(h: Harness, asked: { id: string }, existingId: string, answer: string) {
    const row = await h.services.proposals.get(asked.id);

    expect(row.status).toBe("approved");
    expect(row.ask?.answer).toBe(answer);
    expect(row.resolution?.recordId).toBe(asked.id);
    expect((await live(h, asked.id)).lifecycle).toBe("live");
    expect((await live(h, existingId)).lifecycle).toBe("live");
    expect(await h.store.records.countRows()).toBe(2);
  }

  async function expectRecorded(h: Harness, askedId: string, answer: string) {
    const row = await h.services.proposals.get(askedId);

    expect(row.status).toBe("asked");
    expect(row.ask?.answer).toBe(answer);
    expect(row.ask?.heldRecordId).toBe(askedId);
  }

  for (const step of steps) {
    describe(`keep_both interrupted at ${step.label}`, () => {
      async function interrupted() {
        const h = await setup();
        const existing = await seedNear(h);
        const asked = await interrupt(h, step, (id) => h.services.proposals.answer(id, keepBoth));

        await expectRecorded(h, asked.id, "keep_both");

        return { h, existing, asked };
      }

      it("is finished by a later sweep", async () => {
        const { h, existing, asked } = await interrupted();

        expect(await h.services.proposals.expireAsks(asked.createdAt)).toEqual([asked.id]);
        await expectOneHeldNote(h, asked, existing.id, "keep_both");
      });

      it("is finished by the same answer repeated", async () => {
        const { h, existing, asked } = await interrupted();

        expect((await h.services.proposals.answer(asked.id, keepBoth)).status).toBe("approved");
        await expectOneHeldNote(h, asked, existing.id, "keep_both");
      });

      it("rejects a different answer and then still finishes the recorded one", async () => {
        const { h, existing, asked } = await interrupted();

        await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });
        await expect(
          h.services.proposals.answer(asked.id, { action: "keep_both", reason: "another reason" }),
        ).rejects.toMatchObject({ status: 409 });
        expect(await h.store.records.countRows()).toBe(step.table === "records" ? 1 : 2);
        await h.services.proposals.expireAsks(asked.createdAt);
        await expectOneHeldNote(h, asked, existing.id, "keep_both");
      });
    });

    describe(`expiry interrupted at ${step.label}`, () => {
      async function interrupted() {
        const h = await setup();
        const existing = await seedNear(h);
        const asked = await h.services.proposals.create(proposalBody());
        const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();
        const restore = failNthWrite(h, step.table, step.nth);
        const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);

        await h.services.proposals.expireAsks(past);
        restore();
        errors.mockRestore();
        await expectRecorded(h, asked.id, "unanswered");

        return { h, existing, asked, past };
      }

      it("is finished by a later sweep", async () => {
        const { h, existing, asked, past } = await interrupted();

        expect(await h.services.proposals.expireAsks(past)).toEqual([asked.id]);
        await expectOneHeldNote(h, asked, existing.id, "unanswered");
      });

      it("rejects a keep_both and a fold answer, then the sweep still finishes it", async () => {
        const { h, existing, asked, past } = await interrupted();

        await expect(h.services.proposals.answer(asked.id, keepBoth)).rejects.toMatchObject({ status: 409 });
        await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });
        await h.services.proposals.expireAsks(past);
        await expectOneHeldNote(h, asked, existing.id, "unanswered");
        expect(await h.services.proposals.expireAsks(past)).toEqual([]);
      });
    });
  }

  it("leaves the ask open when the keep_both intent write itself failed", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await interrupt(h, { table: "proposals", nth: 1 }, (id) =>
      h.services.proposals.answer(id, keepBoth),
    );
    const row = await h.services.proposals.get(asked.id);

    expect(row.status).toBe("asked");
    expect(row.ask?.answer).toBeNull();
    expect(row.ask?.heldRecordId ?? null).toBeNull();
    expect(await h.store.records.countRows()).toBe(1);
  });
});

describe("fold record retired before recovery", () => {
  it("marks the row folded and flags the fold as gone instead of failing every sweep", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const restore = failNthWrite(h, "records", 2);

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toThrow("injected write failure");
    restore();

    const interrupted = await h.services.proposals.get(asked.id);
    const foldId = interrupted.ask?.foldRecordId as string;

    await h.services.records.markRetired(foldId, "removed by the owner");

    const errors = vi.spyOn(console, "error").mockImplementation(() => undefined);

    expect(await h.services.proposals.expireAsks(asked.createdAt)).toEqual([asked.id]);
    expect(errors).not.toHaveBeenCalled();
    errors.mockRestore();

    const row = await h.services.proposals.get(asked.id);

    expect(row.status).toBe("folded");
    expect(row.ask?.foldGone).toBe(true);
    expect(row.resolution?.recordId).toBe(foldId);
    expect((await live(h, foldId)).lifecycle).toBe("retired");
    expect((await live(h, existing.id)).lifecycle).toBe("live");
    await expect(live(h, asked.id)).rejects.toMatchObject({ status: 404 });
    expect(await h.services.proposals.expireAsks()).toEqual([]);
  });
});

describe("fold answer racing the expiry sweep", () => {
  async function race(order: "sweep-first" | "answer-first") {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();
    const answer = () =>
      h.services.proposals.answer(asked.id, foldInput()).then(
        (proposal) => ({ proposal }),
        (error: unknown) => ({ error }),
      );
    const sweep = () => h.services.proposals.expireAsks(past);

    if (order === "sweep-first") {
      await Promise.all([sweep(), answer()]);
    } else {
      await Promise.all([answer(), sweep()]);
    }

    return { h, existing, asked };
  }

  for (const order of ["sweep-first", "answer-first"] as const) {
    it(`ends in exactly one outcome with no double write when ${order}`, async () => {
      const { h, existing, asked } = await race(order);
      const row = await h.services.proposals.get(asked.id);
      const heldLive = await live(h, asked.id).then(
        () => true,
        () => false,
      );

      expect(await h.store.records.countRows()).toBe(2);

      if (row.status === "folded") {
        expect(heldLive).toBe(false);
        await expectFolded(h, row.ask?.foldRecordId as string, existing.id);
      } else {
        expect(row.status).toBe("approved");
        expect(row.ask?.answer).toBe("unanswered");
        expect(heldLive).toBe(true);
        expect((await live(h, existing.id)).lifecycle).toBe("live");
      }
    });
  }

  it("rejects a fold answer that arrives after the sweep saved the note", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();

    await h.services.proposals.expireAsks(past);

    await expect(h.services.proposals.answer(asked.id, foldInput())).rejects.toMatchObject({ status: 409 });
    expect(await h.store.records.countRows()).toBe(2);
  });

  it("leaves a folded ask untouched when the sweep runs afterwards", async () => {
    const h = await setup();
    const existing = await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    const past = new Date(Date.parse(asked.ask?.expiresAt as string) + 1).toISOString();
    const folded = await h.services.proposals.answer(asked.id, foldInput());

    expect(await h.services.proposals.expireAsks(past)).toEqual([]);
    await expectFolded(h, folded.resolution?.recordId as string, existing.id);
    expect(await h.store.records.countRows()).toBe(2);
  });
});

async function expectFolded(h: Harness, foldId: string, existingId: string) {
  const old = await live(h, existingId);

  expect((await live(h, foldId)).lifecycle).toBe("live");
  expect(old.lifecycle).toBe("superseded");
  expect(old.supersededBy).toBe(foldId);
}

describe("ask routes", () => {
  it("serves the answer route and the ask log with matching counts", async () => {
    const h = await setup();
    await seedNear(h);
    const folded = await h.services.proposals.create(proposalBody());
    await serve(h);

    const answer = await call(`/proposals/${folded.id}/answer`, "POST", foldInput());
    const again = await call(`/proposals/${folded.id}/answer`, "POST", { action: "keep_both", reason: "no" });
    const missing = await call("/proposals/unknown/answer", "POST", foldInput());
    const log = await call("/asks");

    expect(answer.status).toBe(200);
    expect(answer.body.status).toBe("folded");
    expect(again.status).toBe(409);
    expect(missing.status).toBe(404);
    expect(log.status).toBe(200);
    expect(log.body.counts).toEqual({ fold: 1, keep_both: 0, unanswered: 0, open: 0 });
    expect(log.body.asks.map((entry: { id: string }) => entry.id)).toEqual([folded.id]);
  });

  it("counts keep_both and open asks", async () => {
    const h = await setup();
    await seedNear(h);
    const kept = await h.services.proposals.create(proposalBody());
    await h.services.proposals.answer(kept.id, { action: "keep_both", reason: "separate" });
    const open = await h.services.proposals.create(proposalBody());
    await serve(h);

    const log = await call("/asks");

    expect(open.status).toBe("asked");
    expect(log.body.counts).toEqual({ fold: 0, keep_both: 1, unanswered: 0, open: 1 });
    expect(log.body.asks.length).toBe(2);
  });

  it("filters the ask log by since and rejects a bad timestamp", async () => {
    const h = await setup();
    await seedNear(h);
    await h.services.proposals.create(proposalBody());
    await serve(h);

    const all = await call("/asks?since=2026-10-01T00:00:00.000Z");
    const none = await call("/asks?since=2026-12-01T00:00:00.000Z");
    const bad = await call("/asks?since=yesterday");

    expect(all.body.counts).toEqual({ fold: 0, keep_both: 0, unanswered: 0, open: 1 });
    expect(none.body).toEqual({ counts: { fold: 0, keep_both: 0, unanswered: 0, open: 0 }, asks: [] });
    expect(bad.status).toBe(400);
  });

  it("returns a 400 for keep_both without a reason over http", async () => {
    const h = await setup();
    await seedNear(h);
    const asked = await h.services.proposals.create(proposalBody());
    await serve(h);

    const response = await call(`/proposals/${asked.id}/answer`, "POST", { action: "keep_both" });

    expect(response.status).toBe(400);
  });
});
