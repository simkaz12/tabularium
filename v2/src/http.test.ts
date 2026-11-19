import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Server } from "node:http";
import type { AddressInfo } from "node:net";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { EmbedError } from "./embed.ts";
import type { Embedder } from "./embed.ts";
import { createHttpServer, createScopeMapRef, createServices, readToken } from "./http.ts";
import { ScopeMap } from "./scopeMap.ts";
import { selectWhere } from "./store.ts";
import type { DefectEvent, ReadLogEntry } from "./domain.ts";
import { createApprovedDefect, defectBody, makeHarness, proposalBody, seedLegacy, seedRecord } from "./testing.ts";
import type { Harness } from "./testing.ts";

let harness: Harness;
let server: Server;
let base: string;
let host: string;

beforeEach(async () => {
  harness = await makeHarness();
  server = createHttpServer(harness.services, ["allowed.test:8789"]);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  base = `http://127.0.0.1:${port}/v2`;
  host = "allowed.test:8789";
});

afterEach(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
  harness.cleanup();
});

async function pendingTile(overrides: Parameters<typeof proposalBody>[0] = {}) {
  return harness.services.proposals.createInternal(proposalBody({ origin: "cleaner", ...overrides }));
}

function call(method: string, path: string, body?: unknown, hostHeader = host) {
  const payload = body === undefined ? undefined : JSON.stringify(body);

  return new Promise<{ status: number; body: any }>((resolve, reject) => {
    const req = request(
      `${base}${path}`,
      {
        method,
        headers: {
          host: hostHeader,
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

describe("http", () => {
  it("rejects a host outside the allowlist", async () => {
    const denied = await call("GET", "/health", undefined, "evil.test");

    expect(denied.status).toBe(403);
    expect((await call("GET", "/health")).status).toBe(200);
  });

  it("reports the embedder check on /health without blocking", async () => {
    const health = await call("GET", "/health");

    expect(health.body).toEqual({ ok: true, embedder: { status: "unchecked" } });
  });

  it("runs propose, list and recall end to end with the record live on create", async () => {
    const created = await call("POST", "/proposals", proposalBody());
    expect(created.status).toBe(201);
    expect(created.body).toMatchObject({ status: "approved", resolution: { recordId: created.body.id } });

    const pending = await call("GET", "/proposals?status=pending");
    expect(pending.body).toEqual([]);

    const approved = await call("GET", "/proposals?status=approved");
    expect(approved.body.map((entry: { id: string }) => entry.id)).toEqual([created.body.id]);

    const recalled = await call("GET", "/recall?q=repl%20readline&scope=project-a");
    expect(recalled.body.results[0]).toMatchObject({ id: created.body.id, source: "v2" });

    const fetched = await call("GET", `/records/${created.body.id}`);
    expect(fetched.body.lifecycle).toBe("live");
    expect(fetched.body.vector).toBeUndefined();
    expect(fetched.body.lineage).toEqual({ supersededBy: null, supersedes: [] });
  });

  it("answers 400 to origin harvest", async () => {
    const refused = await call("POST", "/proposals", proposalBody({ origin: "harvest" }));

    expect(refused.status).toBe(400);
    expect(refused.body.error).toContain("origin must be one of agent");
  });

  it("ignores a section query parameter", async () => {
    await call("POST", "/proposals", proposalBody());

    const listed = await call("GET", "/proposals?status=approved&section=odd");

    expect(listed.status).toBe(200);
    expect(listed.body.length).toBe(1);
    expect(listed.body[0].section).toBeUndefined();
  });

  it("filters proposals by since", async () => {
    const created = await call("POST", "/proposals", proposalBody());

    const later = await call("GET", "/proposals?status=approved&since=2026-12-01T00:00:00Z");
    const earlier = await call("GET", "/proposals?status=approved&since=2026-10-01T00:00:00Z");

    expect(later.body).toEqual([]);
    expect(earlier.body.length).toBe(1);
    expect((await call("GET", "/proposals?since=garbage")).status).toBe(400);
    expect(created.status).toBe(201);
  });

  it("serves reject, reinforce, unsupersede and delete", async () => {
    const reinforcedLegacy = await seedLegacy(harness);
    const supersededLegacy = await seedLegacy(harness);
    const first = await pendingTile();
    const second = await pendingTile();

    const rejected = await call("POST", `/proposals/${first.id}/reject`, { reason: "dup" });
    expect(rejected.body.status).toBe("rejected");

    const reinforced = await call("POST", `/proposals/${second.id}/reinforce`, {
      recordId: reinforcedLegacy.id,
    });
    expect(reinforced.body.reinforcedCount).toBe(1);
    expect(reinforced.body.legacyId).toBe(reinforcedLegacy.id);

    const retiredLegacy = await call("GET", `/records/${reinforcedLegacy.id}`);
    expect(retiredLegacy.body.lifecycle).toBe("retired");
    expect(retiredLegacy.body.lineage.supersededBy.id).toBe(reinforced.body.id);

    const refused = await call("POST", `/records/${reinforcedLegacy.id}/unsupersede`, { reason: "mistake" });
    expect(refused.status).toBe(409);

    const third = await pendingTile();
    const approved = await call("POST", `/proposals/${third.id}/approve`, {
      supersedes: [supersededLegacy.id],
      reason: "changed",
    });
    expect(approved.status).toBe(200);

    const restored = await call("POST", `/records/${supersededLegacy.id}/unsupersede`, { reason: "mistake" });
    expect(restored.body.lifecycle).toBe("live");

    const removed = await call("DELETE", `/records/${supersededLegacy.id}`, {
      reason: "leaked key",
      category: "secret",
    });
    expect(removed.body).toEqual({ id: supersededLegacy.id, deleted: true });
    expect((await call("GET", `/records/${supersededLegacy.id}`)).status).toBe(404);
  });

  it("serves archive and unarchive", async () => {
    const legacy = await seedLegacy(harness);
    const record = await seedRecord(harness);
    const defect = await call("POST", "/proposals", defectBody());

    const archived = await call("POST", `/records/${legacy.id}/archive`, { reason: "legacy history" });
    expect(archived.status).toBe(200);
    expect(archived.body.lifecycle).toBe("archived");
    expect((await call("GET", `/records/${legacy.id}`)).body.lifecycle).toBe("archived");

    expect((await call("POST", `/records/${legacy.id}/archive`, { reason: "again" })).status).toBe(409);
    expect((await call("POST", `/records/${record.id}/archive`, {})).status).toBe(400);
    expect((await call("POST", `/records/${defect.body.resolution.recordId}/archive`, { reason: "x" })).status).toBe(409);
    expect((await call("POST", `/records/${record.id}/unarchive`)).status).toBe(409);
    expect((await call("POST", "/records/missing/archive", { reason: "x" })).status).toBe(404);

    const restored = await call("POST", `/records/${legacy.id}/unarchive`);
    expect(restored.status).toBe(200);
    expect(restored.body.lifecycle).toBe("live");
    expect(restored.body.lifecycleNote).toBeNull();
  });

  it("answers errors as json", async () => {
    expect((await call("POST", "/proposals", { kind: "fact" })).status).toBe(400);
    expect((await call("POST", "/proposals/missing/approve", {})).status).toBe(404);
    expect((await call("GET", "/nope")).status).toBe(404);
  });

  it("answers a malformed percent escape with 400", async () => {
    const response = await call("GET", "/records/%E0%A4%A");

    expect(response.status).toBe(400);
  });

  it("answers an embedder failure with 503", async () => {
    const failing: Embedder = {
      dim: harness.embedder.dim,
      embed: async () => {
        throw new EmbedError("down");
      },
    };
    const broken = createHttpServer(createServices(harness.store, failing), ["allowed.test:8789"]);
    await new Promise<void>((resolve) => broken.listen(0, "127.0.0.1", resolve));
    const original = base;
    base = `http://127.0.0.1:${(broken.address() as AddressInfo).port}/v2`;

    try {
      const proposed = await call("POST", "/proposals", proposalBody());
      const recalled = await call("GET", "/recall?q=anything");

      expect(proposed.status).toBe(503);
      expect(recalled.status).toBe(503);
    } finally {
      base = original;
      await new Promise<void>((resolve) => broken.close(() => resolve()));
    }
  });

  it("recalls a fully quoted query without a 500", async () => {
    await call("POST", "/proposals", proposalBody());

    const quoted = await call("GET", `/recall?q=${encodeURIComponent('"repl readline"')}`);

    expect(quoted.status).toBe(200);
    expect(quoted.body.results.length).toBe(1);
  });

  it("allows delete only for secrets with a reason", async () => {
    const legacy = await seedLegacy(harness);

    expect((await call("DELETE", `/records/${legacy.id}`, { reason: "garbage" })).status).toBe(400);
    expect((await call("DELETE", `/records/${legacy.id}`, { reason: "garbage", category: "garbage" })).status).toBe(
      400,
    );
    expect((await call("DELETE", `/records/${legacy.id}`, { reason: "", category: "secret" })).status).toBe(400);
    expect((await call("DELETE", `/records/${legacy.id}`)).status).toBe(400);
    expect((await call("GET", `/records/${legacy.id}`)).status).toBe(200);
  });

  it("serves and updates settings", async () => {
    expect((await call("GET", "/settings")).body).toEqual({ cleanerConfidenceFloor: 0.7 });

    const updated = await call("PUT", "/settings", { cleanerConfidenceFloor: 0.8 });

    expect(updated.status).toBe(200);
    expect(updated.body).toEqual({ cleanerConfidenceFloor: 0.8 });
    expect((await call("GET", "/settings")).body).toEqual({ cleanerConfidenceFloor: 0.8 });
  });

  it("rejects invalid settings", async () => {
    expect((await call("PUT", "/settings", { cleanerConfidenceFloor: 1.5 })).status).toBe(400);
    expect((await call("PUT", "/settings", { cleanerConfidenceFloor: "x" })).status).toBe(400);
    expect((await call("PUT", "/settings", { other: 1 })).status).toBe(400);
    expect((await call("PUT", "/settings")).status).toBe(400);
    expect((await call("GET", "/settings")).body).toEqual({ cleanerConfidenceFloor: 0.7 });
  });

  it("checks the host on settings", async () => {
    expect((await call("GET", "/settings", undefined, "evil.test")).status).toBe(403);
    expect((await call("PUT", "/settings", { cleanerConfidenceFloor: 0.9 }, "evil.test")).status).toBe(403);
  });

  it("requires a reason to reject over http", async () => {
    const created = await pendingTile();

    const refused = await call("POST", `/proposals/${created.id}/reject`, {});
    const refusedBlank = await call("POST", `/proposals/${created.id}/reject`, { reason: " " });
    const accepted = await call("POST", `/proposals/${created.id}/reject`, { reason: "not durable" });

    expect(refused).toMatchObject({ status: 400, body: { error: "reason is required" } });
    expect(refusedBlank.status).toBe(400);
    expect(accepted.body).toMatchObject({ status: "rejected", resolution: { reason: "not durable" } });
  });

  it("names the missing defect field in a 400 and creates a defect with 201", async () => {
    const body = defectBody();
    const missing = await call("POST", "/proposals", { ...body, fields: { claim: "c", where: "w", trigger: "t" } });
    const created = await call("POST", "/proposals", body);

    expect(missing).toMatchObject({ status: 400, body: { error: "left_because is required for a defect" } });
    expect(created.status).toBe(201);
    expect(created.body).toMatchObject({
      kind: "defect",
      status: "approved",
      audience: null,
      reader: null,
      resolution: { recordId: created.body.id },
    });
    expect(created.body.provenance.stage).toBe("");
  });

  it("takes and releases a defect over http with the 400 and 409 cases", async () => {
    const created = await call("POST", "/proposals", defectBody());
    const defect = (await call("GET", `/records/${created.body.resolution.recordId}`)).body;
    const fact = await seedRecord(harness);
    const taker = { threadId: "thread-3", link: "t3-thread://thread-3" };

    const taken = await call("POST", `/records/${defect.id}/take`, taker);
    const again = await call("POST", `/records/${defect.id}/take`, taker);
    const wrong = await call("POST", `/records/${fact.id}/take`, taker);
    const bare = await call("POST", `/records/${defect.id}/take`, {});
    const released = await call("POST", `/records/${defect.id}/release`);
    const releasedAgain = await call("POST", `/records/${defect.id}/release`);

    expect(taken.status).toBe(200);
    expect(taken.body).toMatchObject({
      defectStatus: "taken",
      takenThreadId: "thread-3",
      takenLink: "t3-thread://thread-3",
      takenBy: { threadId: "thread-3", link: "t3-thread://thread-3" },
    });
    expect(taken.body.takenAt).toEqual(taken.body.takenBy.at);
    expect(taken.body.vector).toBeUndefined();
    expect(again.status).toBe(409);
    expect(wrong.status).toBe(400);
    expect(bare.status).toBe(400);
    expect(released).toMatchObject({
      status: 200,
      body: {
        defectStatus: "open",
        takenThreadId: null,
        takenLink: null,
        takenAt: taken.body.takenAt,
        takenBy: null,
      },
    });
    expect(releasedAgain.status).toBe(200);
  });

  it("closes a defect on propose over http and then answers 409 to take", async () => {
    const created = await call("POST", "/proposals", defectBody());
    const defect = (await call("GET", `/records/${created.body.resolution.recordId}`)).body;
    const tile = await call("POST", "/proposals", {
      ...proposalBody(),
      fields: { claim: "fixed by the offset change" },
      target: { recordId: defect.id, source: "v2", action: "close" },
    });

    expect(tile.status).toBe(201);
    expect(tile.body).toMatchObject({ status: "approved", resolution: { recordId: defect.id } });
    expect((await call("GET", `/records/${defect.id}`)).body.defectStatus).toBe("closed");
    expect((await call("GET", "/proposals?status=pending")).body).toEqual([]);
    expect(
      (await call("POST", `/records/${defect.id}/take`, { threadId: "t", link: "l" })).status,
    ).toBe(409);
  });

  it("recalls by audience over http and returns the new fields", async () => {
    await seedRecord(harness, { headline: "repl readline", claim: "repl readline error", audience: "reviewer" });
    const mine = await seedRecord(harness, {
      headline: "repl readline",
      claim: "repl readline error",
      audience: "builder",
    });

    const recalled = await call("GET", "/recall?q=repl%20readline&audience=builder");
    const refused = await call("GET", "/recall?q=repl&audience=wizard");

    expect(recalled.body.results.map((entry: { id: string }) => entry.id)).toEqual([mine.id]);
    expect(recalled.body.results[0]).toMatchObject({ audience: "builder" });
    expect(refused.status).toBe(400);
  });

  it("passes the recall task through to the search", async () => {
    await seedRecord(harness, { headline: "repl readline", claim: "repl readline error" });
    const taskOnly = await seedRecord(harness, { headline: "zebra quartz migration", claim: "zebra quartz rule" });

    const without = await call("GET", "/recall?q=repl%20readline&limit=1");
    const withTask = await call("GET", "/recall?q=repl%20readline&limit=2&task=zebra%20quartz%20migration%20rule");

    expect(without.body.results.map((entry: { id: string }) => entry.id)).not.toContain(taskOnly.id);
    expect(withTask.body.results.map((entry: { id: string }) => entry.id)).toContain(taskOnly.id);
  });

  it("logs the thread id from the recall query", async () => {
    await seedRecord(harness, { headline: "repl readline", claim: "repl readline error" });

    await call("GET", "/recall?q=repl%20readline&threadId=thread-5&task=fixing%20repl");

    const log = await selectWhere<ReadLogEntry>(harness.store.readLog, undefined);
    expect(log.length).toBeGreaterThan(0);
    expect(log.every((entry) => entry.threadId === "thread-5" && entry.task === "fixing repl")).toBe(true);
  });

  it("writes exactly one get row for a found record and none for a 404", async () => {
    const record = await seedRecord(harness, { headline: "repl readline", claim: "repl readline error" });

    const found = await call("GET", `/records/${record.id}?threadId=thread-5`);
    const missing = await call("GET", "/records/missing?threadId=thread-5");

    const log = await selectWhere<ReadLogEntry>(harness.store.readLog, undefined);
    expect(found.status).toBe(200);
    expect(missing.status).toBe(404);
    expect(log).toHaveLength(1);
    expect(log[0]).toMatchObject({
      resultId: record.id,
      source: "v2",
      threadId: "thread-5",
      via: "get",
      query: null,
      task: null,
      rank: null,
      score: null,
    });
  });

  it("logs a get with no thread id as null", async () => {
    const record = await seedRecord(harness);

    await call("GET", `/records/${record.id}`);

    const log = await selectWhere<ReadLogEntry>(harness.store.readLog, undefined);
    expect(log).toHaveLength(1);
    expect(log[0]?.threadId).toBeNull();
  });

  it("writes release events with the body thread id, or null for a missing or non-string one", async () => {
    const defect = await createApprovedDefect(harness);
    const taker = { threadId: "thread-9", link: "t3-thread://thread-9" };

    await call("POST", `/records/${defect.id}/take`, taker);
    await call("POST", `/records/${defect.id}/release`, { threadId: "thread-5" });
    await call("POST", `/records/${defect.id}/take`, taker);
    await call("POST", `/records/${defect.id}/release`, {});
    await call("POST", `/records/${defect.id}/take`, taker);
    await call("POST", `/records/${defect.id}/release`, { threadId: 42 });

    const events = await selectWhere<DefectEvent>(harness.store.defectEvents, "event = 'release'");
    expect(events.map((event) => event.threadId).sort()).toEqual([null, null, "thread-5"]);
  });

  describe("profiles", () => {
    const profileMap = new ScopeMap(["work", "personal"], new Map([["acme", "work"], ["notes", "personal"]]), []);

    async function seedProfiles() {
      harness.services.scopeMap.current = profileMap;

      const rows = {
        acme: await seedRecord(harness, { scope: "acme" }),
        work: await seedRecord(harness, { scope: "global-work" }),
        notes: await seedRecord(harness, { scope: "notes" }),
        global: await seedRecord(harness, { scope: "global" }),
        legacy: await seedLegacy(harness, { scope: "acme" }),
      };

      await harness.services.proposals.createInternal(proposalBody({ origin: "cleaner", scope: "acme" }));
      await harness.services.proposals.createInternal(proposalBody({ origin: "cleaner", scope: "notes" }));

      return rows;
    }

    it("counts the records and proposals across the profile scopes", async () => {
      await seedProfiles();

      const counted = await call("GET", "/profiles/work/count");

      expect(counted.status).toBe(200);
      expect(counted.body).toEqual({
        profile: "work",
        scopes: ["acme", "global-work"],
        records: 3,
        proposals: 1,
      });
    });

    it("deletes only the profile rows and returns the counts the count route reported", async () => {
      const rows = await seedProfiles();
      const counted = await call("GET", "/profiles/work/count");

      const deleted = await call("DELETE", "/profiles/work");

      expect(deleted.status).toBe(200);
      expect(deleted.body).toEqual(counted.body);

      const left = await selectWhere<{ id: string }>(harness.store.records, undefined);
      expect(left.map((row) => row.id).sort()).toEqual([rows.notes.id, rows.global.id].sort());
      expect(await harness.store.legacy.countRows()).toBe(0);

      const remaining = await selectWhere<{ scope: string }>(harness.store.proposals, undefined);
      expect(remaining.map((row) => row.scope)).toEqual(["notes"]);
    });

    it("refuses the default profile and answers 404 to an unknown one", async () => {
      await seedProfiles();

      expect((await call("DELETE", "/profiles/default")).status).toBe(400);
      expect((await call("GET", "/profiles/default/count")).status).toBe(400);
      expect((await call("DELETE", "/profiles/nope")).status).toBe(404);
      expect(await harness.store.records.countRows()).toBe(4);
    });

    it("reloads the scope map from its file and answers 400 to a bad file", async () => {
      const dir = mkdtempSync(join(tmpdir(), "tabularium-scopes-"));
      const file = join(dir, "scopes.json");
      writeFileSync(file, JSON.stringify({ profiles: ["work"], scopes: { acme: "work" } }));
      harness.services.scopeMap = createScopeMapRef(file);

      expect(harness.services.scopeMap.current.sharedScopeOf("acme")).toBe("global-work");

      writeFileSync(file, JSON.stringify({ profiles: ["work"], scopes: { acme: "work", v2: "work" } }));
      const reloaded = await call("POST", "/scope-map/reload");

      expect(reloaded.body).toEqual({ profiles: ["work"] });
      expect(harness.services.scopeMap.current.scopesOf("work")).toEqual(["acme", "v2", "global-work"]);

      writeFileSync(file, "{");
      const bad = await call("POST", "/scope-map/reload");

      expect(bad.status).toBe(400);
      expect(bad.body.error).toContain(file);
      rmSync(dir, { recursive: true, force: true });
    });
  });
});

describe("http auth", () => {
  const TOKEN = "s3cret-token-value";
  let secured: Server;
  let securedPort: number;

  beforeEach(async () => {
    secured = createHttpServer(harness.services, ["allowed.test:8789"], { token: TOKEN });
    await new Promise<void>((resolve) => secured.listen(0, "127.0.0.1", resolve));
    securedPort = (secured.address() as AddressInfo).port;
  });

  afterEach(async () => {
    await new Promise<void>((resolve) => secured.close(() => resolve()));
  });

  function fetchStatus(path: string, headers: Record<string, string> = {}, port = securedPort) {
    return new Promise<number>((resolve, reject) => {
      const req = request(
        `http://127.0.0.1:${port}/v2${path}`,
        { method: "GET", headers: { host: "allowed.test:8789", ...headers } },
        (res) => {
          res.resume();
          res.on("end", () => resolve(res.statusCode ?? 0));
        },
      );

      req.on("error", reject);
      req.end();
    });
  }

  it("answers 401 with no token", async () => {
    expect(await fetchStatus("/health")).toBe(401);
  });

  it("answers 401 to a wrong token, a wrong scheme and a prefix of the token", async () => {
    expect(await fetchStatus("/health", { authorization: "Bearer nope" })).toBe(401);
    expect(await fetchStatus("/health", { authorization: `Basic ${TOKEN}` })).toBe(401);
    expect(await fetchStatus("/health", { authorization: `Bearer ${TOKEN.slice(0, 5)}` })).toBe(401);
  });

  it("answers 403 to the right token when an Origin header is present", async () => {
    const headers = { authorization: `Bearer ${TOKEN}`, origin: "http://evil.example" };

    expect(await fetchStatus("/health", headers)).toBe(403);
  });

  it("answers 403 to an Origin header even with no token", async () => {
    expect(await fetchStatus("/health", { origin: "http://evil.example" })).toBe(403);
  });

  it("answers 200 to the right token with no Origin", async () => {
    expect(await fetchStatus("/health", { authorization: `Bearer ${TOKEN}` })).toBe(200);
  });

  it("keeps the host allowlist in front of the token", async () => {
    expect(await fetchStatus("/health", { authorization: `Bearer ${TOKEN}`, host: "evil.test" })).toBe(403);
  });

  it("answers 200 with no token and an Origin header in t3 mode", async () => {
    const { port } = server.address() as AddressInfo;

    expect(await fetchStatus("/health", { origin: "http://x.example" }, port)).toBe(200);
  });

  it("reads the token file trimmed and refuses a missing, empty or unset file with its path", () => {
    const dir = mkdtempSync(join(tmpdir(), "tabularium-token-"));
    const file = join(dir, "token");
    const empty = join(dir, "empty");
    writeFileSync(file, `${TOKEN}\n`);
    writeFileSync(empty, "\n");

    expect(readToken(file)).toBe(TOKEN);
    expect(() => readToken(join(dir, "missing"))).toThrow(join(dir, "missing"));
    expect(() => readToken(empty)).toThrow(empty);
    expect(() => readToken(null)).toThrow("TABUMCP_TOKEN_FILE");
    rmSync(dir, { recursive: true, force: true });
  });
});

