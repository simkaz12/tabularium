import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import * as lancedb from "@lancedb/lancedb";

import { hybridSearch, insert, openStore, scopeClause, sqlString } from "./store.ts";
import { makeHarness, seedRecord } from "./testing.ts";

const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function tmp(): string {
  const dir = mkdtempSync(join(tmpdir(), "tabularium-v2-store-"));
  dirs.push(dir);

  return dir;
}

describe("store", () => {
  it("creates the four tables with indexes and reopens idempotently", async () => {
    const path = tmp();
    const first = await openStore(path, 8);
    await insert(first.readLog, [
      { id: "r1", ts: "t", query: "q", scope: null, resultId: "x", source: "v2", rank: 0, score: 1 },
    ]);

    const second = await openStore(path, 8);

    expect(await second.readLog.countRows()).toBe(1);
    expect(await second.records.countRows()).toBe(0);
    expect(await second.legacy.countRows()).toBe(0);
    expect(await second.proposals.countRows()).toBe(0);

    const indexed = (await second.records.listIndices()).map((index) => index.columns[0]).sort();
    expect(indexed).toEqual(["claim", "headline", "lifecycle", "scope"]);
  });

  it("sees rows written through another connection", async () => {
    const path = tmp();
    const reader = await openStore(path, 8);
    const writer = await openStore(path, 8);

    await insert(writer.readLog, [
      { id: "r1", ts: "t", query: "q", scope: null, resultId: "x", source: "v2", rank: 0, score: 1 },
    ]);

    expect(await reader.readLog.countRows()).toBe(1);
  });

  it("builds the scope clause around the given shared scope and defaults it to global", () => {
    expect(scopeClause("acme")).toBe("(scope = 'acme' OR scope = 'global')");
    expect(scopeClause("acme", "global-work")).toBe("(scope = 'acme' OR scope = 'global-work')");
    expect(scopeClause("o'k", "global-w'k")).toBe("(scope = 'o''k' OR scope = 'global-w''k')");
    expect(scopeClause(null, "global-work")).toBeNull();
  });

  it("quotes single quotes in sql strings", () => {
    expect(sqlString("it's")).toBe("'it''s'");
  });

  it("opens a directory that holds only the four original tables and adds the new ones", async () => {
    const path = tmp();
    const first = await openStore(path, 8);
    await insert(first.readLog, [
      { id: "r1", ts: "t", query: "q", scope: null, resultId: "x", source: "v2", rank: 0, score: 1 },
    ]);
    const db = await lancedb.connect(path);
    await db.dropTable("settings");
    await db.dropTable("cleaner_state");
    await db.dropTable("defect_events");
    expect((await db.tableNames()).sort()).toEqual(["legacy", "proposals", "read_log", "records"]);

    const reopened = await openStore(path, 8);

    expect((await db.tableNames()).sort()).toEqual([
      "cleaner_state",
      "defect_events",
      "legacy",
      "proposals",
      "read_log",
      "records",
      "settings",
    ]);
    expect(await reopened.settings.countRows()).toBe(0);
    expect(await reopened.cleanerState.countRows()).toBe(0);
    expect(await reopened.readLog.countRows()).toBe(1);
  });

  it("searches hostile text without throwing and still finds a quoted phrase", async () => {
    const harness = await makeHarness();

    try {
      await seedRecord(harness, { id: "hit", headline: "alpha note", claim: "the alpha claim holds" });
      await seedRecord(harness, { id: "other", headline: "beta note", claim: "unrelated beta text" });
      const vector = await harness.embedder.embed("alpha claim");

      for (const query of ['"alpha claim"', '"', "a:b", "(x", "*", "'", "-", "AND", "\\"]) {
        await expect(
          hybridSearch(harness.store.records, { query, vector, where: "lifecycle = 'live'", limit: 5 }),
        ).resolves.toBeDefined();
      }

      const quoted = await hybridSearch<{ id: string }>(harness.store.records, {
        query: '"alpha claim"',
        vector,
        where: "lifecycle = 'live'",
        limit: 5,
      });

      expect(quoted[0]?.row.id).toBe("hit");

      const punctuation = await hybridSearch<{ id: string }>(harness.store.records, {
        query: "*",
        vector,
        where: "lifecycle = 'live'",
        limit: 5,
      });

      expect(punctuation.length).toBe(2);
    } finally {
      harness.cleanup();
    }
  });
});
