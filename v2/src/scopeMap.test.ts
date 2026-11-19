import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { isReservedScope, loadScopeMap } from "./scopeMap.ts";

const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function writeMap(content: string): string {
  const dir = mkdtempSync(join(tmpdir(), "scopemap-"));
  dirs.push(dir);
  const file = join(dir, "scopes.json");
  writeFileSync(file, content);

  return file;
}

const SAMPLE = JSON.stringify({
  profiles: ["work", "personal"],
  scopes: { acme: "work", tabularium: "personal" },
  paths: [{ match: "~/Projects/acme-*", scope: "acme" }],
});

describe("loadScopeMap trimming", () => {
  it("rejects a reserved path target that only differs by whitespace", () => {
    const content = JSON.stringify({ profiles: ["work"], paths: [{ match: "/x/*", scope: " global-work " }] });

    expect(() => loadScopeMap(writeMap(content))).toThrow("reserved");
  });

  it("rejects a reserved scope key that only differs by whitespace", () => {
    const content = JSON.stringify({ profiles: ["work"], scopes: { " global ": "work" } });

    expect(() => loadScopeMap(writeMap(content))).toThrow("reserved");
  });

  it("stores a padded path target trimmed", () => {
    const content = JSON.stringify({ paths: [{ match: " /x/* ", scope: " proj " }] });

    expect(loadScopeMap(writeMap(content)).scopeFor("/x/a", "a")).toBe("proj");
  });
});

describe("loadScopeMap", () => {
  it("returns an empty map when the file is missing", () => {
    const map = loadScopeMap(join(tmpdir(), "scopemap-missing", "scopes.json"));

    expect(map.profileOf("anything")).toBe("default");
    expect(map.sharedScopeOf("anything")).toBe("global");
  });

  it("maps an unmapped scope to default sharing global", () => {
    const map = loadScopeMap(writeMap(SAMPLE));

    expect(map.profileOf("other")).toBe("default");
    expect(map.sharedScopeOf("other")).toBe("global");
  });

  it("gives profile work the shared scope global-work", () => {
    const map = loadScopeMap(writeMap(SAMPLE));

    expect(map.profileOf("acme")).toBe("work");
    expect(map.sharedScopeOf("acme")).toBe("global-work");
    expect(map.scopesOf("work")).toEqual(["acme", "global-work"]);
    expect(map.scopesOf("default")).toEqual(["global"]);
  });

  it("resolves a generated shared scope to its owning profile", () => {
    const map = loadScopeMap(writeMap(SAMPLE));

    expect(map.profileOf("global-work")).toBe("work");
    expect(map.sharedScopeOf("global-work")).toBe("global-work");
    expect(map.profileOf("global-personal")).toBe("personal");
    expect(map.sharedScopeOf("global-personal")).toBe("global-personal");
    expect(map.profileOf("global")).toBe("default");
    expect(map.sharedScopeOf("global")).toBe("global");
  });

  it("rejects a file that maps a reserved shared scope name", () => {
    const viaScopes = JSON.stringify({ profiles: ["work"], scopes: { global: "work" } });
    const viaProfileShared = JSON.stringify({ profiles: ["work"], scopes: { "global-work": "default" } });
    const viaPaths = JSON.stringify({ paths: [{ match: "~/x", scope: "global" }] });

    expect(() => loadScopeMap(writeMap(viaScopes))).toThrow(/reserved/);
    expect(() => loadScopeMap(writeMap(viaProfileShared))).toThrow(/reserved/);
    expect(() => loadScopeMap(writeMap(viaPaths))).toThrow(/reserved/);
    expect(isReservedScope("global")).toBe(true);
    expect(isReservedScope("acme")).toBe(false);
  });

  it("lets a paths match beat the folder name", () => {
    const map = loadScopeMap(writeMap(SAMPLE));

    expect(map.scopeFor(`${homedir()}/Projects/acme-api`, "acme-api")).toBe("acme");
    expect(map.scopeFor(`${homedir()}/Projects/acme-api/sub`, "sub")).toBe("sub");
    expect(map.scopeFor(`${homedir()}/Projects/billing-api`, "billing-api")).toBe("billing-api");
  });

  it("throws with the file path on bad JSON", () => {
    const file = writeMap("{nope");

    expect(() => loadScopeMap(file)).toThrow(file);
  });

  it("throws with the file path on an unknown profile", () => {
    const file = writeMap(JSON.stringify({ profiles: ["work"], scopes: { a: "ghost" } }));

    expect(() => loadScopeMap(file)).toThrow(file);
  });
});
