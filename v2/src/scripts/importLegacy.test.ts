import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HashEmbedder } from "../embed.ts";
import type { Embedder } from "../embed.ts";
import { openStore, selectWhere, upsert } from "../store.ts";
import type { Store } from "../store.ts";
import { TEST_DIM } from "../testing.ts";
import type { LegacyRecord } from "../domain.ts";
import { importLegacy, parseCliArgs, readLegacyFile } from "./importLegacy.ts";

const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function tmp(): string {
  const dir = mkdtempSync(join(tmpdir(), "tabularium-v2-import-"));
  dirs.push(dir);

  return dir;
}

function fixtureLine(index: number, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    legacyId: `legacy-${index}`,
    kind: index % 2 === 0 ? "decision" : "fact",
    headline: `headline ${index}`,
    claim: `claim number ${index}`,
    why: `why number ${index}`,
    rejected: index % 2 === 0 ? `rejected ${index}` : null,
    scope: "project-a",
    oldScope: index === 1 ? "mobile" : "project-a",
    oldSource: "v1",
    purpose: "pitfall",
    topic: index === 2 ? null : "test harness",
    lifecycle: "live",
    readCount: index,
    originalCreatedAt: "2026-03-01T09:00:00+00:00",
    createdAt: "2026-11-01T10:00:00+00:00",
    reviewed: false,
    collapsedIds: index === 3 ? ["old-a", "old-b"] : [],
    ...overrides,
  };
}

function writeFixture(lines: Array<Record<string, unknown> | string>): string {
  const path = join(tmp(), "legacy.jsonl");
  const body = lines.map((line) => (typeof line === "string" ? line : JSON.stringify(line))).join("\n");
  writeFileSync(path, `${body}\n`);

  return path;
}

function countingEmbedder(): Embedder & { calls: number } {
  const inner = new HashEmbedder(TEST_DIM);
  const stub = {
    dim: TEST_DIM,
    calls: 0,
    async embed(text: string) {
      stub.calls += 1;

      return inner.embed(text);
    },
  };

  return stub;
}

async function freshStore(): Promise<Store> {
  return openStore(tmp(), TEST_DIM);
}

const FIVE = [1, 2, 3, 4, 5].map((index) => fixtureLine(index));

describe("importLegacy", () => {
  it("writes 5 rows from a 5-line fixture and none on a second run", async () => {
    const store = await freshStore();
    const embedder = countingEmbedder();
    const lines = readLegacyFile(writeFixture(FIVE));

    const first = await importLegacy(store, embedder, lines);

    expect(first).toEqual({ lines: 5, written: 5, replaced: 0, skipped: 0 });
    expect(await store.legacy.countRows()).toBe(5);
    expect(embedder.calls).toBe(5);

    const second = await importLegacy(store, embedder, lines);

    expect(second).toEqual({ lines: 5, written: 0, replaced: 0, skipped: 5 });
    expect(await store.legacy.countRows()).toBe(5);
    expect(embedder.calls).toBe(5);
  });

  it("maps every export field onto the legacy row with id equal to legacyId", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture(FIVE));

    await importLegacy(store, countingEmbedder(), lines);

    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-3'");

    expect(row).toMatchObject({
      id: "legacy-3",
      legacyId: "legacy-3",
      kind: "fact",
      headline: "headline 3",
      claim: "claim number 3",
      why: "why number 3",
      rejected: null,
      scope: "project-a",
      oldScope: "project-a",
      oldSource: "v1",
      purpose: "pitfall",
      topic: "test harness",
      lifecycle: "live",
      readCount: 3,
      originalCreatedAt: "2026-03-01T09:00:00+00:00",
      createdAt: "2026-11-01T10:00:00+00:00",
      reviewed: false,
    });
    expect(JSON.parse(row?.collapsedIds ?? "null")).toEqual(["old-a", "old-b"]);
    expect(row?.vector).toHaveLength(TEST_DIM);

    const [empty] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-2'");

    expect(empty?.collapsedIds).toBeNull();
    expect(empty?.topic).toBeNull();
    expect(empty?.rejected).toBe("rejected 2");
  });

  it("keeps a fact whose reshape left why empty", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture([fixtureLine(1, { why: "" })]));

    await importLegacy(store, countingEmbedder(), lines);

    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    expect(row?.why).toBe("");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1, { why: null })]))).toThrow("why must be a string");
  });

  it("keeps a fact whose legacy record had no source", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture([fixtureLine(1, { oldSource: "" })]));

    await importLegacy(store, countingEmbedder(), lines);

    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    expect(row?.oldSource).toBe("");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1, { oldSource: null })]))).toThrow(
      "oldSource must be a string",
    );
  });

  it("imports only the missing rows when the store already holds some", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture(FIVE));

    await importLegacy(store, countingEmbedder(), lines.slice(0, 2));
    const report = await importLegacy(store, countingEmbedder(), lines);

    expect(report).toEqual({ lines: 5, written: 3, replaced: 0, skipped: 2 });
    expect(await store.legacy.countRows()).toBe(5);
  });

  it("writes in batches and survives a batch size smaller than the input", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture(FIVE));

    const report = await importLegacy(store, countingEmbedder(), lines, { batchSize: 2 });

    expect(report.written).toBe(5);
    expect(await store.legacy.countRows()).toBe(5);
  });

  it("writes nothing for a batch whose embedding fails", async () => {
    const store = await freshStore();
    const lines = readLegacyFile(writeFixture(FIVE));
    const broken: Embedder = {
      dim: TEST_DIM,
      async embed() {
        throw new Error("embedder down");
      },
    };

    await expect(importLegacy(store, broken, lines)).rejects.toThrow("embedder down");
    expect(await store.legacy.countRows()).toBe(0);
  });

  it("skips a changed headline on re-import without the flag", async () => {
    const store = await freshStore();
    const embedder = countingEmbedder();

    await importLegacy(store, embedder, readLegacyFile(writeFixture(FIVE)));

    const corrected = FIVE.map((line, index) => (index === 0 ? { ...line, headline: "corrected" } : line));
    const report = await importLegacy(store, embedder, readLegacyFile(writeFixture(corrected)));
    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    expect(report).toEqual({ lines: 5, written: 0, replaced: 0, skipped: 5 });
    expect(row?.headline).toBe("headline 1");
  });

  it("replaces a changed headline with the flag and re-embeds only that row", async () => {
    const store = await freshStore();
    const embedder = countingEmbedder();

    await importLegacy(store, embedder, readLegacyFile(writeFixture(FIVE)));
    const [before] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    const corrected = FIVE.map((line, index) => (index === 0 ? { ...line, headline: "corrected" } : line));
    const report = await importLegacy(store, embedder, readLegacyFile(writeFixture(corrected)), {
      replace: true,
    });
    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    expect(report).toEqual({ lines: 5, written: 0, replaced: 1, skipped: 4 });
    expect(embedder.calls).toBe(6);
    expect(row?.headline).toBe("corrected");
    expect(row?.vector).not.toEqual(before?.vector);
    expect(await store.legacy.countRows()).toBe(5);
  });

  it("keeps the lifecycle of a retired row when --replace rewrites its content", async () => {
    const store = await freshStore();
    const embedder = countingEmbedder();

    await importLegacy(store, embedder, readLegacyFile(writeFixture(FIVE)));
    const [live] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");
    await upsert(store.legacy, [
      { ...live, lifecycle: "retired", supersededBy: "rec-9" },
    ] as unknown as Array<Record<string, unknown>>);

    const corrected = FIVE.map((line, index) => (index === 0 ? { ...line, claim: "new claim" } : line));
    await importLegacy(store, embedder, readLegacyFile(writeFixture(corrected)), { replace: true });
    const [row] = await selectWhere<LegacyRecord>(store.legacy, "legacyId = 'legacy-1'");

    expect(row).toMatchObject({ claim: "new claim", lifecycle: "retired", supersededBy: "rec-9" });
  });

  it("parses the --replace flag and the path in either order", () => {
    expect(parseCliArgs(["node", "x", "a.jsonl"])).toEqual({ path: "a.jsonl", replace: false });
    expect(parseCliArgs(["node", "x", "--replace", "a.jsonl"])).toEqual({ path: "a.jsonl", replace: true });
    expect(parseCliArgs(["node", "x", "a.jsonl", "--replace"])).toEqual({ path: "a.jsonl", replace: true });
    expect(() => parseCliArgs(["node", "x"])).toThrow("usage");
    expect(() => parseCliArgs(["node", "x", "a.jsonl", "--force"])).toThrow("usage");
  });

  it("rejects a malformed file before any store is touched", () => {
    const missing = fixtureLine(1, { claim: "" });

    expect(() => readLegacyFile(writeFixture([missing]))).toThrow("line 1: claim must be a non-empty string");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1), "{broken"]))).toThrow("line 2: not valid JSON");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1), fixtureLine(1)]))).toThrow("duplicate legacyId");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1, { kind: "note" })]))).toThrow("kind must be one of");
    expect(() => readLegacyFile(writeFixture([fixtureLine(1, { collapsedIds: "a" })]))).toThrow("collapsedIds");
  });
});
