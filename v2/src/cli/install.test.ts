import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, readlinkSync, statSync, writeFileSync } from "node:fs";
import { delimiter, join } from "node:path";

import { describe, expect, it } from "vitest";

import { doctor } from "./doctor.ts";
import { makeHarness } from "./harness.ts";
import { install } from "./install.ts";
import { resolvePaths } from "./paths.ts";

function mode(file: string): number {
  return statSync(file).mode & 0o777;
}

describe("install", () => {
  it.each([
    ["darwin", "x64"],
    ["win32", "x64"],
    ["linux", "ia32"],
  ])("refuses %s/%s and names WSL2 without running anything", async (platform, arch) => {
    const h = makeHarness({ platform: platform as NodeJS.Platform, arch });

    const code = await install(h.ctx);

    expect(code).toBe(1);
    expect(h.err.join("\n")).toContain("WSL2");
    expect(h.calls()).toEqual([]);
    expect(existsSync(resolvePaths(h.ctx).configDir)).toBe(false);
  });

  it("refuses Node older than 22", async () => {
    const h = makeHarness({ nodeVersion: "20.18.0" });

    expect(await install(h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("Node 22");
    expect(h.calls()).toEqual([]);
  });

  it("writes the token mode 600 and never rewrites an existing one", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);

    const first = readFileSync(paths.tokenFile, "utf8");

    expect(first.trim().length).toBeGreaterThanOrEqual(32);
    expect(mode(paths.tokenFile)).toBe(0o600);
    expect(mode(paths.envFile)).toBe(0o600);

    expect(await install(h.ctx)).toBe(0);
    expect(readFileSync(paths.tokenFile, "utf8")).toBe(first);
  });

  it("keeps an existing env file, scopes file and token with their edits", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    writeFileSync(paths.envFile, "TYPESAFE_API_KEY=abc\n");
    writeFileSync(paths.scopesFile, "{\"profiles\":[\"work\"]}\n");
    await install(h.ctx);

    expect(readFileSync(paths.envFile, "utf8")).toBe("TYPESAFE_API_KEY=abc\n");
    expect(readFileSync(paths.scopesFile, "utf8")).toBe("{\"profiles\":[\"work\"]}\n");
  });

  it("installs into a versioned folder, points current at it and runs the unit commands", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    expect(await install(h.ctx)).toBe(0);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
    expect(h.calls()).toContain(`npm install --prefix ${join(paths.appDir, "0.1.0")} tabumcp@0.1.0`);
    expect(h.calls()).toContain("systemctl --user daemon-reload");
    expect(h.calls()).toContain("systemctl --user enable --now tabumcp");
    expect(readFileSync(join(paths.skillDir, "SKILL.md"), "utf8")).toBe("skill body\n");
  });

  it("renders the unit with the current-link ExecStart", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);

    const unit = readFileSync(paths.unitFile, "utf8");

    expect(unit).toContain(`ExecStart="${paths.currentLink}/node_modules/.bin/tabumcp" serve`);
    expect(unit).toContain(`Environment="PATH=/opt/node/bin:`);
    expect(unit).not.toContain("{{");
  });

  it("renders the launchd plist and bootstraps it on darwin arm64", async () => {
    const h = makeHarness({ platform: "darwin", arch: "arm64" });
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);

    const plist = readFileSync(paths.plistFile, "utf8");

    expect(plist).toContain(`<string>${paths.currentLink}/node_modules/.bin/tabumcp</string>`);
    expect(plist).not.toContain("{{");
    expect(h.calls()).toContain(`launchctl bootstrap gui/501 ${paths.plistFile}`);
    expect(paths.dataDir).toContain("Library/Application Support/tabumcp");
  });

  it("prints the next step and exits non-zero when doctor fails", async () => {
    const h = makeHarness();

    h.health.status = 503;

    expect(await install(h.ctx)).toBe(1);
    expect(h.out.join("\n")).toContain("ask your agent to run the tabumcp skill");
  });

  it("pulls the embedding model only when ollama lacks it and the user agrees", async () => {
    const present = makeHarness();

    present.confirmAnswer.value = true;
    await install(present.ctx);

    expect(present.calls().some((line) => line.startsWith("ollama pull"))).toBe(false);

    const declined = makeHarness();

    declined.ctx.env.STUB_OLLAMA_LIST = "llama3:latest";
    await install(declined.ctx);

    expect(declined.calls().some((line) => line.startsWith("ollama pull"))).toBe(false);

    const agreed = makeHarness();

    agreed.ctx.env.STUB_OLLAMA_LIST = "llama3:latest";
    agreed.confirmAnswer.value = true;
    await install(agreed.ctx);

    expect(agreed.calls()).toContain("ollama pull nomic-embed-text");
  });

  it("exits zero with a healthy service", async () => {
    const h = makeHarness();

    expect(await install(h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("MCP URL: http://127.0.0.1:8789/mcp");
  });
});

describe("command link", () => {
  it("links ~/.local/bin/tabumcp through current and prints the PATH line when the folder is missing", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    expect(await install(h.ctx)).toBe(0);
    expect(readlinkSync(paths.binLink)).toBe(join(paths.currentLink, "node_modules", ".bin", "tabumcp"));
    expect(h.out.join("\n")).toContain(`${paths.binDir} is not on your PATH. Add this line to your shell profile:\n  export PATH="$HOME/.local/bin:$PATH"`);
    expect(h.out.join("\n")).toContain("command link:");
    expect(h.out.join("\n")).toContain("not on PATH");
  });

  it("stays quiet about PATH when ~/.local/bin is on it", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    h.ctx.env.PATH = `${paths.binDir}${delimiter}${h.ctx.env.PATH}`;

    expect(await install(h.ctx)).toBe(0);
    expect(h.out.join("\n")).not.toContain("is not on your PATH");
    expect(h.out.join("\n")).toMatch(/command link: .*, on PATH/);
  });

  it("repoints its own link on a second install and leaves a foreign file alone", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    expect(await install(h.ctx)).toBe(0);
    expect(lstatSync(paths.binLink).isSymbolicLink()).toBe(true);

    const other = makeHarness();
    const otherPaths = resolvePaths(other.ctx);

    mkdirSync(otherPaths.binDir, { recursive: true });
    writeFileSync(otherPaths.binLink, "mine\n");

    expect(await install(other.ctx)).toBe(0);
    expect(readFileSync(otherPaths.binLink, "utf8")).toBe("mine\n");
    expect(other.err.join("\n")).toContain("not ours");
  });

  it("never deletes an unrelated file named tabumcp.next beside the link", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);
    const bystander = `${paths.binLink}.next`;

    mkdirSync(paths.binDir, { recursive: true });
    writeFileSync(bystander, "not ours\n");

    expect(await install(h.ctx)).toBe(0);
    expect(readFileSync(bystander, "utf8")).toBe("not ours\n");
    expect(readdirSync(paths.binDir).sort()).toEqual(["tabumcp", "tabumcp.next"]);
  });

  it("doctor reports the link as informational even when it is missing", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.out.length = 0;

    expect(await doctor(h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain(`command link: ${resolvePaths(h.ctx).binLink}`);
  });
});
