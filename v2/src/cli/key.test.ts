import { readFileSync, statSync, writeFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { makeHarness } from "./harness.ts";
import { install } from "./install.ts";
import { key } from "./key.ts";
import { runCli } from "./main.ts";
import { resolvePaths } from "./paths.ts";

async function installed() {
  const h = makeHarness();

  expect(await install(h.ctx)).toBe(0);

  return { h, paths: resolvePaths(h.ctx) };
}

describe("key", () => {
  it("writes the typed key into the env file, restarts, and never prints it", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    h.secretAnswer.value = "sk-secret-123";

    expect(await key(h.ctx, ["set"])).toBe(0);

    const env = readFileSync(paths.envFile, "utf8");

    expect(env).toContain("TYPESAFE_API_KEY=sk-secret-123\n");
    expect(env).toContain("TABUMCP_MODE=standalone");
    expect(statSync(paths.envFile).mode & 0o777).toBe(0o600);
    expect(h.calls().slice(before)).toContain("systemctl --user restart tabumcp");
    expect([...h.out, ...h.err, ...h.calls()].join("\n")).not.toContain("sk-secret-123");
  });

  it("replaces an earlier key instead of adding a second line", async () => {
    const { h, paths } = await installed();

    h.secretAnswer.value = "one";
    expect(await key(h.ctx, ["set"])).toBe(0);
    h.secretAnswer.value = "two";
    expect(await key(h.ctx, ["set"])).toBe(0);

    const lines = readFileSync(paths.envFile, "utf8").split("\n").filter((line) => line.startsWith("TYPESAFE_API_KEY"));

    expect(lines).toEqual(["TYPESAFE_API_KEY=two"]);
  });

  it("exits 1 and changes nothing when no key is typed", async () => {
    const { h, paths } = await installed();
    const before = readFileSync(paths.envFile, "utf8");
    const calls = h.calls().length;

    h.secretAnswer.value = "";

    expect(await key(h.ctx, ["set"])).toBe(1);
    expect(readFileSync(paths.envFile, "utf8")).toBe(before);
    expect(h.calls().length).toBe(calls);
  });

  it("clear removes the key and restarts", async () => {
    const { h, paths } = await installed();

    h.secretAnswer.value = "abc";
    await key(h.ctx, ["set"]);

    const before = h.calls().length;

    expect(await key(h.ctx, ["clear"])).toBe(0);
    expect(readFileSync(paths.envFile, "utf8")).not.toContain("TYPESAFE_API_KEY");
    expect(h.calls().slice(before)).toContain("systemctl --user restart tabumcp");
  });

  it("fails without an env file and rejects unknown subcommands", async () => {
    const h = makeHarness();

    h.secretAnswer.value = "abc";

    expect(await key(h.ctx, ["set"])).toBe(1);
    expect(await runCli(["key", "show"], h.ctx)).toBe(2);
    writeFileSync(`${h.home}/unused`, "");
  });
});
