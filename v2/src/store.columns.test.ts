import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import * as lancedb from "@lancedb/lancedb";
import { Bool, Field, FixedSizeList, Float32, Float64, Int32, Schema, Utf8 } from "apache-arrow";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ensureColumns, findById, insert, openStore, proposalToRow, rowToProposal, selectWhere } from "./store.ts";
import { TEST_DIM, defectBody, proposalBody } from "./testing.ts";
import type { MemoryRecord, Proposal } from "./domain.ts";

const text = (name: string, nullable = true) => new Field(name, new Utf8(), nullable);

const OLD_RECORD_COLUMNS = [
  "id",
  "kind",
  "headline",
  "claim",
  "why",
  "rejected",
  "scope",
  "purpose",
  "topic",
  "lifecycle",
  "supersededBy",
  "supersessionReason",
  "retiredReason",
  "lifecycleNote",
  "threadId",
  "threadTitle",
  "projectId",
  "branch",
  "machine",
  "link",
  "excerpt",
  "origin",
  "legacyId",
  "reinforcedCount",
  "lastReinforcedAt",
  "createdAt",
  "approvedAt",
];

const OLD_PROPOSAL_COLUMNS = [
  "id",
  "status",
  "kind",
  "headline",
  "scope",
  "origin",
  "purpose",
  "topic",
  "section",
  "targetRecordId",
  "targetAction",
  "createdAt",
  "resolvedAt",
  "fields",
  "provenance",
  "target",
  "checks",
  "flags",
  "neighbours",
  "suggestion",
  "resolution",
];

const NEW_RECORD_COLUMNS = [
  "audience",
  "reader",
  "stage",
  "where",
  "trigger",
  "leftBecause",
  "fixHint",
  "impact",
  "defectStatus",
  "takenThreadId",
  "takenLink",
  "takenAt",
  "closedBy",
  "closedAt",
];

const NEW_PROPOSAL_COLUMNS = ["audience", "reader", "ask"];

function oldRecordSchema(): Schema {
  const fields = OLD_RECORD_COLUMNS.map((name) =>
    name === "reinforcedCount" ? new Field(name, new Int32(), true) : text(name, name !== "id"),
  );
  const vector = new Field("vector", new FixedSizeList(TEST_DIM, new Field("item", new Float32(), true)), false);

  return new Schema([...fields, vector]);
}

function oldLegacySchema(): Schema {
  const extra = [
    text("oldScope"),
    text("oldSource"),
    new Field("readCount", new Int32(), true),
    text("originalCreatedAt"),
    new Field("reviewed", new Bool(), true),
    text("collapsedIds"),
  ];

  return new Schema([...oldRecordSchema().fields, ...extra]);
}

function oldLegacyRow(id: string): Record<string, unknown> {
  return {
    ...oldRecordRow(id),
    oldScope: "project-a",
    oldSource: "v1",
    readCount: 0,
    originalCreatedAt: "2026-01-01T00:00:00.000Z",
    reviewed: false,
    collapsedIds: null,
  };
}

function oldRecordRow(id: string): Record<string, unknown> {
  const row: Record<string, unknown> = {};

  for (const name of OLD_RECORD_COLUMNS) {
    row[name] = name === "reinforcedCount" ? 0 : `${name}-value`;
  }

  row.id = id;
  row.supersededBy = null;
  row.vector = new Array(TEST_DIM).fill(0.1);

  return row;
}

function oldProposalSchema(): Schema {
  return new Schema(OLD_PROPOSAL_COLUMNS.map((name) => text(name, name !== "id")));
}

async function columnsOf(dir: string, name: string): Promise<string[]> {
  const db = await lancedb.connect(dir);
  const table = await db.openTable(name);

  return (await table.schema()).fields.map((field) => field.name);
}

let dir: string;
let logs: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "tabularium-columns-"));
  logs = vi.spyOn(console, "log").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  rmSync(dir, { recursive: true, force: true });
});

async function seedOldStore(): Promise<void> {
  const db = await lancedb.connect(dir);
  const records = await db.createEmptyTable("records", oldRecordSchema());
  await records.add([oldRecordRow("old-1"), oldRecordRow("old-2")]);
  const legacy = await db.createEmptyTable("legacy", oldLegacySchema());
  await legacy.add([oldLegacyRow("old-legacy")]);
  const proposals = await db.createEmptyTable("proposals", oldProposalSchema());
  const body = proposalBody();
  const row: Record<string, unknown> = {};

  for (const name of OLD_PROPOSAL_COLUMNS) {
    row[name] = null;
  }

  Object.assign(row, {
    id: "old-proposal",
    status: "pending",
    kind: "fact",
    headline: body.headline,
    scope: body.scope,
    origin: "agent",
    fields: JSON.stringify({ claim: "c", why: "w" }),
    provenance: JSON.stringify({ threadId: "t" }),
  });
  await proposals.add([row]);
}

describe("ensureColumns", () => {
  it("adds the new columns to existing tables and old rows read null", async () => {
    await seedOldStore();
    const store = await openStore(dir, TEST_DIM);

    for (const name of NEW_RECORD_COLUMNS) {
      expect(await columnsOf(dir, "records")).toContain(name);
      expect(await columnsOf(dir, "legacy")).toContain(name);
    }

    for (const name of NEW_PROPOSAL_COLUMNS) {
      expect(await columnsOf(dir, "proposals")).toContain(name);
    }

    const rows = await selectWhere<MemoryRecord>(store.records, undefined);
    expect(rows.map((row) => row.id).sort()).toEqual(["old-1", "old-2"]);

    for (const row of rows) {
      for (const name of NEW_RECORD_COLUMNS) {
        expect(row[name as keyof MemoryRecord]).toBeNull();
      }

      expect(row.why).toBe("why-value");
      expect(row.headline).toBe("headline-value");
    }

    const events = logs.mock.calls
      .map((call) => JSON.parse(String(call[0])) as { event: string; table: string; columns: string[] })
      .filter((entry) => entry.event === "columns-added");
    expect(events.map((entry) => entry.table).sort()).toEqual(["legacy", "proposals", "records"]);
    expect(events.find((entry) => entry.table === "records")?.columns).toEqual(NEW_RECORD_COLUMNS);
    expect(events.find((entry) => entry.table === "proposals")?.columns).toEqual(NEW_PROPOSAL_COLUMNS);
  });

  it("adds threadId, task and via to an old read_log and old rows read null", async () => {
    const db = await lancedb.connect(dir);
    const oldLog = await db.createEmptyTable(
      "read_log",
      new Schema([
        text("id", false),
        text("ts"),
        text("query"),
        text("scope"),
        text("resultId"),
        text("source"),
        new Field("rank", new Int32(), true),
        new Field("score", new Float64(), true),
      ]),
    );
    await oldLog.add([{ id: "r1", ts: "t", query: "q", scope: null, resultId: "x", source: "legacy", rank: 0, score: 1 }]);

    const store = await openStore(dir, TEST_DIM);

    const columns = await columnsOf(dir, "read_log");
    expect(columns).toEqual(expect.arrayContaining(["threadId", "task", "via"]));

    const [row] = await selectWhere<Record<string, unknown>>(store.readLog, undefined);
    expect(row).toMatchObject({ id: "r1", threadId: null, task: null, via: null });
  });

  it("is a no-op on a second open", async () => {
    await seedOldStore();
    await openStore(dir, TEST_DIM);
    const before = await columnsOf(dir, "records");
    logs.mockClear();

    await openStore(dir, TEST_DIM);

    expect(await columnsOf(dir, "records")).toEqual(before);
    expect(logs.mock.calls.filter((call) => String(call[0]).includes("columns-added"))).toEqual([]);
  });

  it("is a no-op on a freshly created store", async () => {
    await openStore(dir, TEST_DIM);
    logs.mockClear();

    await openStore(dir, TEST_DIM);

    expect(logs.mock.calls.filter((call) => String(call[0]).includes("columns-added"))).toEqual([]);
  });

  it("accepts writes carrying the new columns after the add and filters on where", async () => {
    await seedOldStore();
    const store = await openStore(dir, TEST_DIM);
    const [seed] = await selectWhere<MemoryRecord>(store.records, "id = 'old-1'");
    const defect: MemoryRecord = {
      ...(seed as MemoryRecord),
      id: "new-defect",
      kind: "defect",
      why: null,
      audience: "navigator",
      reader: "trap",
      stage: "recon",
      where: "app/Jobs/SyncReport.php",
      trigger: "export more than 500 rows",
      defectStatus: "open",
    };

    await insert(store.records, [defect as unknown as Record<string, unknown>]);

    const stored = await findById<MemoryRecord>(store.records, "new-defect");
    expect(stored?.defectStatus).toBe("open");
    expect(stored?.where).toBe("app/Jobs/SyncReport.php");
    expect(stored?.audience).toBe("navigator");

    const filtered = await selectWhere<MemoryRecord>(store.records, '"where" IS NOT NULL');
    expect(filtered.map((row) => row.id)).toEqual(["new-defect"]);

    const audience = await selectWhere<MemoryRecord>(store.records, "(audience = 'builder' OR audience IS NULL)");
    expect(audience.map((row) => row.id).sort()).toEqual(["old-1", "old-2"]);
  });

  it("reads old proposals with an empty stage", async () => {
    await seedOldStore();
    const store = await openStore(dir, TEST_DIM);
    const [row] = await selectWhere<Record<string, unknown>>(store.proposals, "id = 'old-proposal'");
    const proposal = rowToProposal(row as Record<string, unknown>);

    expect(proposal.audience).toBeNull();
    expect(proposal.reader).toBeNull();
    expect(proposal.provenance.stage).toBe("");
    expect(proposal.fields.claim).toBe("c");
    expect(proposal.ask).toBeNull();
  });

  it("stores the ask as a string column and reads it back", async () => {
    await seedOldStore();
    const store = await openStore(dir, TEST_DIM);
    const body = proposalBody();
    const ask = {
      existing: { id: "r-1", source: "v2" as const, kind: "fact" as const, headline: "h", fields: { claim: "c" } },
      similarity: 0.9,
      sameSubject: 0.8,
      newCoversOld: 0.1,
      existingIsFold: false,
      askedAt: "2026-11-01T10:00:00.000Z",
      expiresAt: "2026-11-01T10:10:00.000Z",
      answer: null,
      answeredAt: null,
      reason: null,
      foldRecordId: null,
      fold: null,
      foldCoversOld: null,
      foldCoversNew: null,
    };
    const proposal: Proposal = {
      ...body,
      id: "asked-1",
      status: "asked",
      audience: null,
      reader: null,
      purpose: null,
      topic: null,
      checks: { status: "skipped" },
      flags: [],
      neighbours: [],
      suggestion: null,
      createdAt: "2026-11-01T10:00:00.000Z",
      resolvedAt: null,
      resolution: null,
      ask,
    };

    await insert(store.proposals, [proposalToRow(proposal)]);

    const [row] = await selectWhere<Record<string, unknown>>(store.proposals, "ask IS NOT NULL");

    expect(typeof row?.ask).toBe("string");
    expect(rowToProposal(row as Record<string, unknown>).ask).toEqual(ask);
  });

  it("round-trips audience and reader through the proposal row", () => {
    const body = defectBody();
    const proposal: Proposal = {
      ...body,
      id: "p-1",
      status: "pending",
      audience: "builder",
      reader: "trap",
      purpose: null,
      topic: null,
      checks: { status: "skipped" },
      flags: [],
      neighbours: [],
      suggestion: null,
      createdAt: "2026-11-01T10:00:00.000Z",
      resolvedAt: null,
      resolution: null,
      ask: null,
    };

    const back = rowToProposal(proposalToRow(proposal));

    expect(back.audience).toBe("builder");
    expect(back.reader).toBe("trap");
    expect(back.kind).toBe("defect");
    expect(back.fields.where).toBe("app/Jobs/SyncReport.php");
  });

  it("refuses to add a missing non-string column", async () => {
    const db = await lancedb.connect(dir);
    const table = await db.createEmptyTable("probe", new Schema([text("id", false)]));

    await expect(ensureColumns(table, [text("id", false), new Field("count", new Int32(), true)])).rejects.toThrow(
      /missing non-string columns: count/,
    );
  });
});
