import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { request } from "node:http";
import type { Server } from "node:http";
import { createServer as createNetServer } from "node:net";
import type { AddressInfo } from "node:net";
import { hostname, tmpdir } from "node:os";
import { join } from "node:path";

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { recordText } from "./domain.ts";
import { createHttpServer, createScopeMapRef } from "./http.ts";
import type { ServiceOptions } from "./http.ts";
import { ScopeMap } from "./scopeMap.ts";
import { selectWhere } from "./store.ts";
import type { MemoryRecord } from "./domain.ts";
import { choiceAnswer, makeHarness, proposalBody, seedRecord, stubJev } from "./testing.ts";
import type { Harness } from "./testing.ts";

const TOKEN = "mcp-test-token";

let harness: Harness;
let server: Server;
let url: URL;
let root: string;
let client: Client;
let HOST: string;
let port: number;

async function freePort(): Promise<number> {
  const probe = createNetServer();

  await new Promise<void>((resolve) => probe.listen(0, "127.0.0.1", resolve));
  const free = (probe.address() as AddressInfo).port;
  await new Promise<void>((resolve) => probe.close(() => resolve()));

  return free;
}

function repo(name: string, branch: string): string {
  const dir = join(root, name);

  mkdirSync(dir, { recursive: true });
  execFileSync("git", ["-C", dir, "init", "-q", "-b", branch]);

  return dir;
}

function parse(result: unknown): any {
  const content = (result as { content: Array<{ text: string }> }).content;

  return JSON.parse(content[0]?.text ?? "null");
}

function failureText(result: unknown): string {
  const call = result as { isError?: boolean; content: Array<{ text: string }> };

  expect(call.isError).toBe(true);

  return call.content[0]?.text ?? "";
}

async function connect(): Promise<Client> {
  const connected = new Client({ name: "test", version: "0" });

  await connected.connect(
    new StreamableHTTPClientTransport(url, {
      requestInit: { headers: { authorization: `Bearer ${TOKEN}` } },
    }),
  );

  return connected;
}

const proposeArgs = (cwd: unknown, overrides: Record<string, unknown> = {}) => ({
  kind: "fact",
  headline: "repl readline crashes on the history file",
  claim: "repl aborts with a readline error when the history file is unwritable",
  cwd,
  ...overrides,
});

async function start(proposals?: ServiceOptions["proposals"]): Promise<void> {
  root = realpathSync(mkdtempSync(join(tmpdir(), "tabularium-mcp-")));
  const map = new ScopeMap(
    ["work", "personal"],
    new Map([
      ["acme", "work"],
      ["notes", "personal"],
      ["wiki", "work"],
    ]),
    [{ pattern: new RegExp(`^${root}/acme-[^/]*$`), scope: "acme" }],
  );

  harness = await makeHarness({ scopeMap: createScopeMapRef(null, map), ...(proposals ? { proposals } : {}) });
  port = await freePort();
  HOST = `127.0.0.1:${port}`;
  server = createHttpServer(harness.services, [HOST], { token: TOKEN, mcp: true });
  await new Promise<void>((resolve) => server.listen(port, "127.0.0.1", resolve));
  url = new URL(`http://${HOST}/mcp`);
  client = await connect();
}

async function stop(): Promise<void> {
  await client.close();
  await new Promise<void>((resolve) => server.close(() => resolve()));
  harness.cleanup();
  rmSync(root, { recursive: true, force: true });
}

beforeEach(() => start());

afterEach(() => stop());

describe("mcp tools", () => {
  it("lists the six tools with cwd required on every one", async () => {
    const { tools } = await client.listTools();
    const required = Object.fromEntries(tools.map((tool) => [tool.name, tool.inputSchema.required ?? []]));

    const recallScope = tools.find((tool) => tool.name === "memory_recall")?.inputSchema.properties?.scope;

    expect((recallScope as { description?: string }).description).toContain("current project's profile");
    expect(tools.map((tool) => tool.name).sort()).toEqual([
      "memory_answer",
      "memory_get",
      "memory_propose",
      "memory_recall",
      "memory_release",
      "memory_take",
    ]);

    for (const tool of tools) {
      expect(required[tool.name]).toContain("cwd");
    }
  });

  it("saves a proposal under the repo scope with its branch and the machine name", async () => {
    const cwd = repo("acme-api", "TASK-1");

    const saved = parse(await client.callTool({ name: "memory_propose", arguments: proposeArgs(cwd) }));

    expect(saved.status).toBe("approved");

    const [record] = await selectWhere<MemoryRecord>(harness.store.records, `id = '${saved.recordId}'`);

    expect(record).toMatchObject({
      scope: "acme",
      projectId: "acme",
      branch: "TASK-1",
      machine: hostname(),
      origin: "agent",
      threadId: "",
    });
  });

  it("fails a missing or nonexistent cwd with the working-folder wording and writes nothing", async () => {
    const missing = await client.callTool({ name: "memory_propose", arguments: proposeArgs(undefined) });
    const nowhere = await client.callTool({
      name: "memory_propose",
      arguments: proposeArgs(join(root, "does-not-exist")),
    });

    expect(failureText(missing)).toContain("cwd is required: pass the working folder of the session");
    expect(failureText(nowhere)).toContain("working folder");
    expect(await harness.store.records.countRows()).toBe(0);
    expect(await harness.store.proposals.countRows()).toBe(0);
  });

  it("rejects a decision without rejected and a defect without its fields", async () => {
    const cwd = repo("acme-api", "main");

    const decision = await client.callTool({
      name: "memory_propose",
      arguments: proposeArgs(cwd, { kind: "decision" }),
    });
    const defect = await client.callTool({
      name: "memory_propose",
      arguments: proposeArgs(cwd, { kind: "defect" }),
    });

    expect(failureText(decision)).toContain("`rejected`");
    expect(failureText(defect)).toContain("`where`, `trigger`, `left_because`");
    expect(await harness.store.records.countRows()).toBe(0);
  });

  it("never returns a work record to a personal repo and shows it to a work repo", async () => {
    const work = repo("acme-api", "main");
    const personal = repo("notes", "main");
    const saved = parse(await client.callTool({ name: "memory_propose", arguments: proposeArgs(work) }));
    await seedRecord(harness, {
      scope: "global-work",
      headline: "repl readline shared",
      claim: "repl readline error shared across work",
      why: "w",
    });
    const query = { query: "repl readline error", task: "tooling: checking repl" };

    const fromPersonal = parse(
      await client.callTool({ name: "memory_recall", arguments: { ...query, cwd: personal } }),
    );
    const fromWork = parse(await client.callTool({ name: "memory_recall", arguments: { ...query, cwd: work } }));

    expect(fromPersonal.results).toEqual([]);
    expect(fromWork.results.map((entry: { id: string }) => entry.id)).toContain(saved.recordId);
    expect(fromWork.results.map((entry: { scope: string }) => entry.scope).sort()).toEqual([
      "acme",
      "global-work",
    ]);
  });

  it("gives the working-folder guidance for an empty or non-string cwd too", async () => {
    for (const cwd of ["", "   ", 42]) {
      const result = await client.callTool({ name: "memory_propose", arguments: proposeArgs(cwd) });

      expect(failureText(result)).toContain("pass the working folder of the session");
    }
  });

  it("lets an explicit scope on recall override the cwd scope but still needs cwd", async () => {
    const work = repo("acme-api", "main");
    const sibling = repo("wiki", "main");
    const saved = parse(await client.callTool({ name: "memory_propose", arguments: proposeArgs(work) }));
    const query = { query: "repl readline error", task: "tooling: checking repl" };

    const overridden = parse(
      await client.callTool({ name: "memory_recall", arguments: { ...query, cwd: sibling, scope: "acme" } }),
    );
    const noCwd = await client.callTool({ name: "memory_recall", arguments: { ...query, scope: "acme" } });

    expect(overridden.results.map((entry: { id: string }) => entry.id)).toEqual([saved.recordId]);
    expect(failureText(noCwd)).toContain("cwd is required: pass the working folder of the session");
  });

  it("refuses an explicit recall scope from another profile and allows the cwd profile's own", async () => {
    const misc = repo("misc", "main");
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const query = { query: "repl readline error", task: "tooling: checking repl" };
    const recallFrom = (cwd: string, scope: string) =>
      client.callTool({ name: "memory_recall", arguments: { ...query, cwd, scope } });

    expect(failureText(await recallFrom(misc, "acme"))).toContain("another profile");
    expect(failureText(await recallFrom(misc, "global-work"))).toContain("another profile");
    expect(failureText(await recallFrom(notes, "acme"))).toContain("another profile");
    expect(failureText(await recallFrom(work, "notes"))).toContain("another profile");
    expect(failureText(await recallFrom(misc, "acme"))).not.toMatch(/work|personal/);

    for (const [cwd, scope] of [
      [misc, "misc2"],
      [misc, "global"],
      [notes, "global-personal"],
      [work, "wiki"],
      [work, "global-work"],
    ] as const) {
      expect(parse(await recallFrom(cwd, scope))).toHaveProperty("results");
    }
  });

  it("refuses take and release on a defect from another profile", async () => {
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const defect = parse(
      await client.callTool({
        name: "memory_propose",
        arguments: proposeArgs(notes, {
          kind: "defect",
          where: "journal export",
          trigger: "export an empty journal",
          left_because: "outside the task",
        }),
      }),
    );

    for (const name of ["memory_take", "memory_release"]) {
      const result = await client.callTool({ name, arguments: { recordId: defect.recordId, cwd: work } });

      expect(failureText(result)).toContain("another profile");
    }

    const stored = parse(await client.callTool({ name: "memory_get", arguments: { id: defect.recordId, cwd: notes } }));

    expect(stored.defectStatus).toBe("open");
  });

  it("takes and releases a defect keyed on project and branch", async () => {
    const cwd = repo("acme-api", "TASK-1");
    const defect = parse(
      await client.callTool({
        name: "memory_propose",
        arguments: proposeArgs(cwd, {
          kind: "defect",
          where: "app/Jobs/SyncReport.php",
          trigger: "export a report with more than 500 rows",
          left_because: "the fix needs a cursor change",
        }),
      }),
    );

    const taken = parse(
      await client.callTool({ name: "memory_take", arguments: { recordId: defect.recordId, cwd } }),
    );

    expect(taken).toMatchObject({
      recordId: defect.recordId,
      defectStatus: "taken",
      takenBy: { threadId: "acme@TASK-1" },
    });

    const released = parse(
      await client.callTool({ name: "memory_release", arguments: { recordId: defect.recordId, cwd } }),
    );

    expect(released).toMatchObject({ defectStatus: "open", takenBy: null });
  });

  it("refuses an answer that gives neither keep_both nor a merged note, or both", async () => {
    const cwd = repo("acme-api", "main");
    const neither = await client.callTool({ name: "memory_answer", arguments: { answers: "p-1", cwd } });
    const both = await client.callTool({
      name: "memory_answer",
      arguments: { answers: "p-1", cwd, keep_both: "different fact", kind: "fact", headline: "h", claim: "c" },
    });

    expect(failureText(neither)).toContain("keep_both");
    expect(failureText(both)).toContain("not both");
  });

  it("reads a record by id and rejects bad input as a tool error", async () => {
    const cwd = repo("acme-api", "main");
    const saved = parse(await client.callTool({ name: "memory_propose", arguments: proposeArgs(cwd) }));

    const got = parse(await client.callTool({ name: "memory_get", arguments: { id: saved.recordId, cwd } }));
    const tooLong = await client.callTool({
      name: "memory_propose",
      arguments: proposeArgs(cwd, { headline: "x".repeat(101) }),
    });

    expect(got.id).toBe(saved.recordId);
    expect(failureText(tooLong)).toContain("headline");
  });
});

describe("mcp profile guard on ids", () => {
  const record = (cwd: string, overrides: Record<string, unknown> = {}) =>
    client.callTool({ name: "memory_propose", arguments: proposeArgs(cwd, overrides) });

  it("refuses closes on a defect from another profile and leaves it open", async () => {
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const defect = parse(
      await record(notes, {
        kind: "defect",
        where: "journal export",
        trigger: "export an empty journal",
        left_because: "outside the task",
      }),
    );

    const closing = await record(work, { headline: "journal export fixed", closes: defect.recordId });
    const stored = parse(
      await client.callTool({ name: "memory_get", arguments: { id: defect.recordId, cwd: notes } }),
    );

    expect(failureText(closing)).toContain("another profile");
    expect(stored.defectStatus).toBe("open");
  });

  it("refuses supersedes on a note from another profile without naming its scope", async () => {
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const foreign = parse(await record(notes));

    const result = await record(work, { headline: "a replacement note", supersedes: [foreign.recordId] });
    const text = failureText(result);

    expect(text).toContain("another profile");
    expect(text).not.toContain("notes");
    expect(text).not.toContain("global-personal");
  });

  it("allows supersedes on a note in the same profile", async () => {
    const work = repo("acme-api", "main");
    const sameScope = repo("acme-web", "main");
    const older = parse(await record(work));

    const replaced = parse(
      await record(sameScope, { headline: "newer unrelated words entirely", supersedes: [older.recordId] }),
    );

    expect(replaced.recordId).not.toBeNull();
  });

  it("refuses memory_get on a record from another profile and keeps the 404 for unknown ids", async () => {
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const sibling = repo("wiki", "main");
    const saved = parse(await record(work));
    const read = (id: string, cwd: string) => client.callTool({ name: "memory_get", arguments: { id, cwd } });

    expect(failureText(await read(saved.recordId, notes))).toContain("another profile");
    expect(parse(await read(saved.recordId, sibling)).id).toBe(saved.recordId);
    expect(failureText(await read("no-such-id", work))).toContain("not found");
  });
});

describe("mcp profile guard on held proposals", () => {
  const yes = () => ({
    type: "choice",
    choice: "yes",
    confidence: 0.7,
    probabilities: { yes: 0.7, no: 0.3 },
  });
  const no = () => ({
    type: "choice",
    choice: "no",
    confidence: 0.7,
    probabilities: { yes: 0.3, no: 0.7 },
  });

  beforeEach(async () => {
    await stop();
    await start({
      jev: stubJev({
        relation: () => choiceAnswer("follow_up", 0.9),
        same_subject: yes,
        new_covers_old: no,
      }),
      foldAsk: true,
      foldAskTimeoutMs: 600_000,
    });
  });

  async function held(cwd: string, scope: string) {
    const body = proposalBody({ scope });
    const base = await harness.embedder.embed(recordText({ kind: body.kind, headline: body.headline, ...body.fields }));
    const other = await harness.embedder.embed("zebra quartz mango lattice");
    const overlap = base.reduce((sum, value, index) => sum + value * (other[index] as number), 0);
    const raw = other.map((value, index) => value - overlap * (base[index] as number));
    const norm = Math.hypot(...raw);
    const side = Math.sqrt(1 - 0.801 * 0.801);
    const vector = base.map((value, index) => 0.801 * value + (side * (raw[index] as number)) / norm);

    await seedRecord(harness, {
      scope,
      headline: "repl readline history note",
      claim: "repl keeps its history in a readline file",
      vector,
    });

    const result = parse(
      await client.callTool({ name: "memory_propose", arguments: proposeArgs(cwd, { headline: body.headline }) }),
    );

    expect(result.status).toBe("asked");

    return result.proposalId as string;
  }

  const merged = {
    kind: "fact",
    headline: "repl history file notes",
    claim: "repl keeps its history in a readline file and aborts when it is unwritable",
  };

  it("refuses a fold and a keep_both on a held proposal from another profile", async () => {
    const notes = repo("notes", "main");
    const work = repo("acme-api", "main");
    const proposalId = await held(notes, "notes");

    const fold = await client.callTool({ name: "memory_answer", arguments: { answers: proposalId, cwd: work, ...merged } });
    const keep = await client.callTool({
      name: "memory_answer",
      arguments: { answers: proposalId, cwd: work, keep_both: "the existing note is about something else" },
    });

    expect(failureText(fold)).toContain("another profile");
    expect(failureText(keep)).toContain("another profile");
    expect((await harness.services.proposals.get(proposalId)).status).toBe("asked");
  });

  it("answers a held proposal from a cwd in the same profile", async () => {
    const work = repo("acme-api", "main");
    const sibling = repo("wiki", "main");
    const proposalId = await held(work, "acme");

    const answered = parse(
      await client.callTool({ name: "memory_answer", arguments: { answers: proposalId, cwd: sibling, ...merged } }),
    );

    expect(answered.status).toBe("folded");
  });

  it("keeps the 404 for an unknown proposal", async () => {
    const work = repo("acme-api", "main");
    const result = await client.callTool({
      name: "memory_answer",
      arguments: { answers: "no-such-proposal", cwd: work, ...merged },
    });

    expect(failureText(result)).toContain("not found");
  });
});

describe("mcp endpoint access", () => {
  function post(headers: Record<string, string>, target = port) {
    return new Promise<number>((resolve, reject) => {
      const req = request(
        `http://127.0.0.1:${target}/mcp`,
        { method: "POST", headers: { host: HOST, "content-type": "application/json", ...headers } },
        (res) => {
          res.resume();
          res.on("end", () => resolve(res.statusCode ?? 0));
        },
      );

      req.on("error", reject);
      req.end("{}");
    });
  }

  it("sits behind the token and the Origin check", async () => {
    expect(await post({})).toBe(401);
    expect(await post({ authorization: "Bearer wrong" })).toBe(401);
    expect(await post({ authorization: `Bearer ${TOKEN}`, origin: "http://evil.example" })).toBe(403);
  });

  it("answers 405 to GET and 404 when the mcp option is off", async () => {
    const getStatus = await new Promise<number>((resolve, reject) => {
      const req = request(
        url,
        { method: "GET", headers: { host: HOST, authorization: `Bearer ${TOKEN}` } },
        (res) => {
          res.resume();
          res.on("end", () => resolve(res.statusCode ?? 0));
        },
      );

      req.on("error", reject);
      req.end();
    });
    const closedPort = await freePort();
    const closed = createHttpServer(harness.services, [HOST]);
    await new Promise<void>((resolve) => closed.listen(closedPort, "127.0.0.1", resolve));

    try {
      expect(getStatus).toBe(405);
      expect(await post({}, closedPort)).toBe(404);
    } finally {
      await new Promise<void>((resolve) => closed.close(() => resolve()));
    }
  });
});
