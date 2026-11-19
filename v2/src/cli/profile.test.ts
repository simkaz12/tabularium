import { readFileSync, writeFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { makeHarness } from "./harness.ts";
import { install } from "./install.ts";
import { runCli } from "./main.ts";
import { resolvePaths } from "./paths.ts";
import { loadScopeMap } from "../scopeMap.ts";

interface Seen {
  method: string;
  path: string;
  auth: string | undefined;
}

async function withService(replies: Record<string, { status: number; body: unknown }> = {}) {
  const seen: Seen[] = [];
  const h = makeHarness();

  expect(await install(h.ctx)).toBe(0);

  const base = h.ctx.fetch;

  h.ctx.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));

    if (!url.pathname.startsWith("/v2/") || url.pathname === "/v2/health") {
      return base(input, init);
    }

    const method = init?.method ?? "GET";

    seen.push({ method, path: url.pathname, auth: (init?.headers as Record<string, string>).Authorization });

    const reply = replies[`${method} ${url.pathname}`] ?? { status: 200, body: { ok: true } };

    return Response.json(reply.body, { status: reply.status });
  }) as typeof fetch;

  return { h, seen, paths: resolvePaths(h.ctx) };
}

function scopes(file: string): unknown {
  return JSON.parse(readFileSync(file, "utf8"));
}

describe("profile", () => {
  it("add writes the profile and reloads once with the bearer token", async () => {
    const { h, seen, paths } = await withService();

    expect(await runCli(["profile", "add", "work"], h.ctx)).toBe(0);

    expect(scopes(paths.scopesFile)).toMatchObject({ profiles: ["work"] });
    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({ method: "POST", path: "/v2/scope-map/reload" });
    expect(seen[0]?.auth).toBe(`Bearer ${readFileSync(paths.tokenFile, "utf8").trim()}`);
  });

  it("add refuses default and bad names", async () => {
    const { h, seen } = await withService();

    expect(await runCli(["profile", "add", "default"], h.ctx)).toBe(2);
    expect(await runCli(["profile", "add", "a b"], h.ctx)).toBe(2);
    expect(await runCli(["profile", "add"], h.ctx)).toBe(2);
    expect(seen).toEqual([]);
  });

  it("delete answering n sends no DELETE", async () => {
    const { h, seen, paths } = await withService({
      "GET /v2/profiles/work/count": { status: 200, body: { profile: "work", scopes: ["acme", "global-work"], records: 7, proposals: 2 } },
    });

    await runCli(["profile", "add", "work"], h.ctx);
    seen.length = 0;
    h.confirmAnswer.value = false;

    expect(await runCli(["profile", "delete", "work"], h.ctx)).toBe(0);
    expect(seen.map((call) => call.method)).toEqual(["GET"]);
    expect(h.out.join("\n")).toContain("7 records and 2 proposals in acme, global-work");
    expect(scopes(paths.scopesFile)).toMatchObject({ profiles: ["work"] });
  });

  it("delete answering y deletes through the service, then drops the profile and its scopes", async () => {
    const { h, seen, paths } = await withService({
      "GET /v2/profiles/work/count": { status: 200, body: { profile: "work", scopes: ["acme", "global-work"], records: 7, proposals: 2 } },
    });

    await runCli(["profile", "add", "work"], h.ctx);
    await runCli(["scope", "add", "acme", "--profile", "work"], h.ctx);
    seen.length = 0;
    h.confirmAnswer.value = true;

    expect(await runCli(["profile", "delete", "work"], h.ctx)).toBe(0);
    expect(seen.map((call) => `${call.method} ${call.path}`)).toEqual([
      "GET /v2/profiles/work/count",
      "DELETE /v2/profiles/work",
      "POST /v2/scope-map/reload",
    ]);
    expect(scopes(paths.scopesFile)).toMatchObject({ profiles: [], scopes: {} });
  });

  it("delete drops padded profile and scope values and leaves a loadable file", async () => {
    const { h, paths } = await withService({
      "GET /v2/profiles/work/count": { status: 200, body: { profile: "work", scopes: ["acme"], records: 1, proposals: 0 } },
    });

    writeFileSync(
      paths.scopesFile,
      JSON.stringify({ profiles: [" work "], scopes: { " acme ": " work " }, paths: [{ match: " ~/o/** ", scope: " acme " }] }),
    );
    h.confirmAnswer.value = true;

    expect(await runCli(["profile", "delete", "work"], h.ctx)).toBe(0);
    expect(scopes(paths.scopesFile)).toMatchObject({ profiles: [], scopes: {}, paths: [] });
    expect(() => loadScopeMap(paths.scopesFile)).not.toThrow();
  });

  it("delete removes the path rules of the deleted profile's scopes and keeps the others", async () => {
    const { h, paths } = await withService({
      "GET /v2/profiles/work/count": { status: 200, body: { profile: "work", scopes: ["acme"], records: 1, proposals: 0 } },
    });

    writeFileSync(
      paths.scopesFile,
      JSON.stringify({
        profiles: ["work", "other"],
        scopes: { acme: "work", notes: "other" },
        paths: [
          { match: "~/Projects/acme-*", scope: "acme" },
          { match: "~/notes/*", scope: "notes" },
        ],
      }),
    );
    h.confirmAnswer.value = true;

    expect(await runCli(["profile", "delete", "work"], h.ctx)).toBe(0);
    expect(scopes(paths.scopesFile)).toEqual({
      profiles: ["other"],
      scopes: { notes: "other" },
      paths: [{ match: "~/notes/*", scope: "notes" }],
    });
    expect(() => loadScopeMap(paths.scopesFile)).not.toThrow();
  });

  it("delete keeps path rules for scopes the service did not report as deleted", async () => {
    const { h, paths } = await withService({
      "GET /v2/profiles/work/count": { status: 200, body: { profile: "work", scopes: ["acme"], records: 1, proposals: 0 } },
    });

    writeFileSync(
      paths.scopesFile,
      JSON.stringify({
        profiles: ["work"],
        scopes: { acme: "work", unreported: "work" },
        paths: [
          { match: "~/Projects/acme-*", scope: "acme" },
          { match: "~/Projects/other-*", scope: "unreported" },
        ],
      }),
    );
    h.confirmAnswer.value = true;

    expect(await runCli(["profile", "delete", "work"], h.ctx)).toBe(0);
    expect(scopes(paths.scopesFile)).toEqual({
      profiles: [],
      scopes: {},
      paths: [{ match: "~/Projects/other-*", scope: "unreported" }],
    });
    expect(() => loadScopeMap(paths.scopesFile)).not.toThrow();
  });

  it("delete exits 1 on an unknown profile and refuses default without calling the service", async () => {
    const { h, seen } = await withService({
      "GET /v2/profiles/ghost/count": { status: 404, body: { error: "profile ghost not found" } },
    });

    expect(await runCli(["profile", "delete", "ghost"], h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("not found");
    seen.length = 0;
    expect(await runCli(["profile", "delete", "default"], h.ctx)).toBe(2);
    expect(seen).toEqual([]);
  });

  it("reports a refused reload as exit 1", async () => {
    const { h } = await withService({
      "POST /v2/scope-map/reload": { status: 400, body: { error: "bad file /x/scopes.json" } },
    });

    expect(await runCli(["profile", "add", "work"], h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("bad file /x/scopes.json");
  });

  it("list prints default and added profiles", async () => {
    const { h } = await withService();

    await runCli(["profile", "add", "work"], h.ctx);

    expect(await runCli(["profile", "list"], h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("work: shared global-work");
    expect(h.out.join("\n")).toContain("default: shared global");
  });
});

describe("scope", () => {
  it("add maps the scope, records the path rule and triggers exactly one reload", async () => {
    const { h, seen, paths } = await withService();

    await runCli(["profile", "add", "work"], h.ctx);
    seen.length = 0;

    expect(await runCli(["scope", "add", "acme", "--profile", "work", "--path", "~/Projects/acme-*"], h.ctx)).toBe(0);
    expect(seen.map((call) => call.path)).toEqual(["/v2/scope-map/reload"]);
    expect(scopes(paths.scopesFile)).toMatchObject({
      scopes: { acme: "work" },
      paths: [{ match: "~/Projects/acme-*", scope: "acme" }],
    });
  });

  it("add with an unknown profile exits 1 and writes nothing", async () => {
    const { h, seen, paths } = await withService();
    const before = readFileSync(paths.scopesFile, "utf8");

    expect(await runCli(["scope", "add", "acme", "--profile", "ghost"], h.ctx)).toBe(1);
    expect(readFileSync(paths.scopesFile, "utf8")).toBe(before);
    expect(seen).toEqual([]);
  });

  it("scope add refuses the reserved shared scope names and writes nothing", async () => {
    const { h, seen, paths } = await withService();

    await runCli(["profile", "add", "work"], h.ctx);
    seen.length = 0;

    const before = readFileSync(paths.scopesFile, "utf8");

    for (const name of ["global", "global-work"]) {
      expect(await runCli(["scope", "add", name, "--profile", "work"], h.ctx)).toBe(1);
    }

    expect(h.err.join("\n")).toContain("reserved");
    expect(readFileSync(paths.scopesFile, "utf8")).toBe(before);
    expect(seen).toEqual([]);
  });

  it("scope add refuses an empty --path and writes nothing", async () => {
    const { h, seen, paths } = await withService();

    await runCli(["profile", "add", "work"], h.ctx);
    seen.length = 0;

    const before = readFileSync(paths.scopesFile, "utf8");

    for (const glob of ["", "  "]) {
      expect(await runCli(["scope", "add", "acme", "--profile", "work", "--path", glob], h.ctx)).toBe(2);
    }

    expect(h.err.join("\n")).toContain("--path");
    expect(readFileSync(paths.scopesFile, "utf8")).toBe(before);
    expect(seen).toEqual([]);
  });

  it("remove drops the scope and its path rules; list shows what is left", async () => {
    const { h, seen, paths } = await withService();

    await runCli(["profile", "add", "work"], h.ctx);
    await runCli(["scope", "add", "acme", "--profile", "work", "--path", "~/p/acme-*"], h.ctx);
    await runCli(["scope", "add", "tabu", "--profile", "default"], h.ctx);
    seen.length = 0;

    expect(await runCli(["scope", "remove", "acme"], h.ctx)).toBe(0);
    expect(seen.map((call) => call.path)).toEqual(["/v2/scope-map/reload"]);
    expect(scopes(paths.scopesFile)).toMatchObject({ scopes: { tabu: "default" }, paths: [] });
    expect(await runCli(["scope", "remove", "acme"], h.ctx)).toBe(1);

    h.out.length = 0;

    expect(await runCli(["scope", "list"], h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("tabu: profile default");
  });

  it("exits 1 when the service is down after saving", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.ctx.fetch = (async () => {
      throw new Error("refused");
    }) as typeof fetch;
    writeFileSync(resolvePaths(h.ctx).scopesFile, JSON.stringify({ profiles: ["work"], scopes: {}, paths: [] }));

    expect(await runCli(["scope", "add", "x", "--profile", "work"], h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("refused");
  });
});
