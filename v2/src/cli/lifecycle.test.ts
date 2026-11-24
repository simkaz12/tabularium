import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, readlinkSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { COMMANDS, USAGE, runCli } from "./main.ts";
import { doctor } from "./doctor.ts";
import { makeHarness } from "./harness.ts";
import { INSTRUCTION_LINES, instructionBlock } from "./instructions.ts";
import { install } from "./install.ts";
import { resolvePaths } from "./paths.ts";
import { parseEnvFile, serve } from "./serve.ts";
import { uninstall } from "./uninstall.ts";
import { update } from "./update.ts";

describe("update", () => {
  it("installs beside the old version, flips current and restarts", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.ctx.version = "0.2.0";

    expect(await update(h.ctx)).toBe(0);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.2.0"));
    expect(h.calls()).toContain(`npm install --prefix ${join(paths.appDir, "0.2.0")} tabumcp@0.2.0`);
    expect(h.calls()).toContain("systemctl --user restart tabumcp");
  });

  it("flips back and restarts again when the health check fails", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.ctx.version = "0.2.0";
    h.health.status = 503;

    expect(await update(h.ctx)).toBe(1);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
    expect(h.calls().filter((line) => line === "systemctl --user restart tabumcp")).toHaveLength(2);
  });

  it("copies the new version's skill to both skill folders", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    writeFileSync(join(h.ctx.packageRoot, "skill", "tabumcp", "SKILL.md"), "newer skill\n");
    h.ctx.version = "0.2.0";

    expect(await update(h.ctx)).toBe(0);
    expect(readFileSync(join(paths.skillDir, "SKILL.md"), "utf8")).toBe("newer skill\n");
    expect(readFileSync(join(paths.codexSkillDir, "SKILL.md"), "utf8")).toBe("newer skill\n");
  });

  it("hands the post-install steps to the new binary, which writes its own block text", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(join(h.ctx.home, ".claude"), { recursive: true });
    writeFileSync(paths.claudeMd, "top\n<!-- tabumcp:begin -->\nold wording\n<!-- tabumcp:end -->\nbottom\n");
    h.registry.version = "0.2.0";
    h.newBinary.run = async () => {
      writeFileSync(paths.claudeMd, "top\n<!-- tabumcp:begin -->\ntext from the new version\n<!-- tabumcp:end -->\nbottom\n");

      return 0;
    };

    expect(await update(h.ctx)).toBe(0);
    expect(h.calls()).toContain(`${join(paths.appDir, "0.2.0", "node_modules", ".bin", "tabumcp")} post-update --previous 0.1.0`);
    expect(readFileSync(paths.claudeMd, "utf8")).toContain("text from the new version");
    expect(readFileSync(paths.claudeMd, "utf8")).not.toContain(INSTRUCTION_LINES[2] as string);
  });

  it("keeps the new version active and points at post-update when the new binary fails", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.registry.version = "0.2.0";
    h.newBinary.fail = true;

    expect(await update(h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("~/.local/bin/tabumcp post-update");
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.2.0"));
  });

  it("post-update copies the skill, writes its own block text and prunes by --previous", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(join(h.ctx.home, ".claude"), { recursive: true });
    writeFileSync(paths.claudeMd, "top\n<!-- tabumcp:begin -->\nold wording\n<!-- tabumcp:end -->\nbottom\n");
    writeFileSync(join(h.ctx.packageRoot, "skill", "tabumcp", "SKILL.md"), "newer skill\n");
    mkdirSync(join(paths.appDir, "0.0.9"));

    expect(await runCli(["post-update", "--previous", "0.1.0"], h.ctx)).toBe(0);
    expect(readFileSync(paths.claudeMd, "utf8")).toBe(`top\n${instructionBlock()}\nbottom\n`);
    expect(readFileSync(join(paths.skillDir, "SKILL.md"), "utf8")).toBe("newer skill\n");
    expect(existsSync(join(paths.appDir, "0.0.9"))).toBe(false);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
  });

  it("rewrites an existing instruction block on update and leaves the rest and absent clients alone", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(join(h.ctx.home, ".claude"), { recursive: true });
    writeFileSync(paths.claudeMd, "top\n<!-- tabumcp:begin -->\nold wording\n<!-- tabumcp:end -->\nbottom\n");
    h.ctx.version = "0.2.0";
    const callsBefore = h.calls().length;

    expect(await update(h.ctx)).toBe(0);

    const md = readFileSync(paths.claudeMd, "utf8");

    expect(md).not.toContain("old wording");
    expect(md).toBe(`top\n${instructionBlock()}\nbottom\n`);
    expect(existsSync(paths.codexAgentsMd)).toBe(false);
    expect(h.calls().slice(callsBefore).filter((line) => line.startsWith("claude"))).toEqual([]);
  });

  it("keeps the previous version and prunes older ones, never the one current points to", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.ctx.version = "0.2.0";
    await update(h.ctx);
    h.ctx.version = "0.3.0";
    await update(h.ctx);

    const folders = () => readdirSync(paths.appDir).sort();

    expect(folders()).toEqual(["0.2.0", "0.3.0", "current"]);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.3.0"));
  });

  it("installs the version the registry reports, not the running package's own", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.registry.version = "0.4.0";

    expect(await update(h.ctx)).toBe(0);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.4.0"));
    expect(h.calls()).toContain(`npm install --prefix ${join(paths.appDir, "0.4.0")} tabumcp@0.4.0`);
  });

  it("says already on the version and touches nothing when the registry matches the install", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(join(paths.appDir, "0.0.9"));
    h.registry.version = "0.1.0";
    const before = h.calls();

    expect(await update(h.ctx)).toBe(0);
    expect(h.out).toContain("already on 0.1.0");
    expect(h.calls()).toEqual(before);
    expect(existsSync(join(paths.appDir, "0.0.9"))).toBe(true);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
  });

  it("pins the target with --version and skips the registry answer", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.registry.version = "0.9.0";

    expect(await update(h.ctx, ["--version", "0.3.0"])).toBe(0);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.3.0"));
    expect(await update(h.ctx, ["--version=0.1.0"])).toBe(0);
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
    expect(await update(h.ctx, ["--version"])).toBe(2);
    expect(await update(h.ctx, ["--version", "latest"])).toBe(2);
  });

  it("fails without touching anything when the registry cannot be read", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    h.registry.code = 1;

    expect(await update(h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("registry");
    expect(readlinkSync(paths.currentLink)).toBe(join(paths.appDir, "0.1.0"));
  });

  it("prunes nothing when the update is rolled back", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(join(paths.appDir, "0.0.9"));
    h.ctx.version = "0.2.0";
    h.health.status = 503;

    expect(await update(h.ctx)).toBe(1);
    expect(existsSync(join(paths.appDir, "0.0.9"))).toBe(true);
  });

  it("refuses when nothing is installed", async () => {
    const h = makeHarness();

    expect(await update(h.ctx)).toBe(1);
    expect(h.err.join("\n")).toContain("not installed");
    expect(h.calls()).toEqual([]);
  });
});

describe("uninstall", () => {
  it("keeps the store and config without --purge", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(paths.storeDir, { recursive: true });
    writeFileSync(join(paths.storeDir, "rows"), "x");

    expect(await uninstall(h.ctx, { purge: false, yes: false })).toBe(0);
    expect(existsSync(join(paths.storeDir, "rows"))).toBe(true);
    expect(existsSync(paths.tokenFile)).toBe(true);
    expect(existsSync(paths.unitFile)).toBe(false);
    expect(existsSync(paths.appDir)).toBe(false);
    expect(existsSync(paths.skillDir)).toBe(false);
    expect(h.calls()).toContain("systemctl --user disable --now tabumcp");
  });

  it("asks before purging and keeps everything on no", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(paths.storeDir, { recursive: true });

    expect(await uninstall(h.ctx, { purge: true, yes: false })).toBe(0);
    expect(existsSync(paths.storeDir)).toBe(true);
    expect(existsSync(paths.configDir)).toBe(true);
  });

  it("purges the store and config on yes", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    mkdirSync(paths.storeDir, { recursive: true });
    h.confirmAnswer.value = true;

    expect(await uninstall(h.ctx, { purge: true, yes: false })).toBe(0);
    expect(existsSync(paths.storeDir)).toBe(false);
    expect(existsSync(paths.configDir)).toBe(false);
  });

  it("removes the launchd agent on darwin", async () => {
    const h = makeHarness({ platform: "darwin", arch: "arm64" });
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    await uninstall(h.ctx, { purge: false, yes: false });

    expect(h.calls()).toContain("launchctl bootout gui/501/io.github.simkaz12.tabumcp");
    expect(existsSync(paths.plistFile)).toBe(false);
  });
});

describe("uninstall command link", () => {
  it("removes the link it made and leaves one that points elsewhere", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    expect(lstatSync(paths.binLink).isSymbolicLink()).toBe(true);
    await uninstall(h.ctx, { purge: false, yes: false });
    expect(() => lstatSync(paths.binLink)).toThrow();

    const other = makeHarness();
    const otherPaths = resolvePaths(other.ctx);

    await install(other.ctx);
    rmSync(otherPaths.binLink);
    symlinkSync("/usr/bin/true", otherPaths.binLink);
    await uninstall(other.ctx, { purge: false, yes: false });
    expect(readlinkSync(otherPaths.binLink)).toBe("/usr/bin/true");
  });
});

describe("doctor", () => {
  it("passes on a healthy install and prints the MCP URL and token path", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.out.length = 0;

    expect(await doctor(h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("MCP URL: http://127.0.0.1:8789/mcp");
    expect(h.out.join("\n")).toContain(`Token file: ${resolvePaths(h.ctx).tokenFile}`);
  });

  it("fails when Ollama is unreachable", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.tagsOk.value = false;
    h.out.length = 0;

    expect(await doctor(h.ctx)).toBe(1);
    expect(h.out.join("\n")).toMatch(/FAIL {2}ollama/);
  });

  it("points an unreachable Ollama at the download page", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.tagsOk.value = false;
    h.out.length = 0;

    await doctor(h.ctx);

    expect(h.out.join("\n")).toContain("ollama.com/download");
  });

  it("prints the embedder chosen at install", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);
    writeFileSync(paths.envFile, "TABULARIUM_EMBED_MODEL=bge-m3\nTABULARIUM_EMBED_DIM=1024\n");
    h.out.length = 0;

    await doctor(h.ctx);

    expect(h.out.join("\n")).toContain("embedder: bge-m3, 1024 dimensions");
  });

  it("names the log command for an inactive service on each platform", async () => {
    const linux = makeHarness();

    writeFileSync(`${linux.logFile}.stopped`, "");
    await doctor(linux.ctx);

    expect(linux.out.join("\n")).toContain("journalctl --user -u tabumcp -n 20");

    const mac = makeHarness({ platform: "darwin", arch: "arm64" });

    writeFileSync(`${mac.logFile}.stopped`, "");
    await doctor(mac.ctx);

    expect(mac.out.join("\n")).toContain(resolvePaths(mac.ctx).logFile);
  });

  it("fails on an embedder build mismatch", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.health.embedder = "mismatch";

    expect(await doctor(h.ctx)).toBe(1);
  });

  it("does not fail on a missing Jev key", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.out.length = 0;

    expect(await doctor(h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("jev key: not set");
  });

  it("labels a set Jev key as not verified and points at the journal", async () => {
    const h = makeHarness();

    await install(h.ctx);
    h.ctx.env.TYPESAFE_API_KEY = "fake-key-0123456789";
    h.out.length = 0;

    expect(await doctor(h.ctx)).toBe(0);
    expect(h.out.join("\n")).toContain("jev key: set (not verified)");
    expect(h.out.join("\n")).toContain("journalctl --user -u tabumcp");
  });

  it("points at the log file on macOS", async () => {
    const h = makeHarness({ platform: "darwin", arch: "arm64" });

    await install(h.ctx);
    h.ctx.env.TYPESAFE_API_KEY = "fake-key-0123456789";
    h.out.length = 0;

    await doctor(h.ctx);

    expect(h.out.join("\n")).toContain(resolvePaths(h.ctx).logFile);
  });
});

describe("serve", () => {
  it("loads the env file without overriding set variables, then starts", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    mkdirSync(paths.configDir, { recursive: true });
    writeFileSync(paths.envFile, "TABULARIUM_PORT=9001\nTABULARIUM_HOST=0.0.0.0\n# note\n\nTYPESAFE_API_KEY=\"k1\"\n");
    h.ctx.env.TABULARIUM_HOST = "127.0.0.1";
    h.ctx.env.TABULARIUM_STORE = join(h.home, "s");
    h.ctx.env.TABUMCP_TOKEN_FILE = join(h.home, "t");

    let seen: { port: number; host: string; typesafeApiKey: string | null } | null = null;

    await serve(h.ctx, async () => async (config) => {
      seen = config as unknown as { port: number; host: string; typesafeApiKey: string | null };
    });

    expect(seen).toMatchObject({ port: 9001, host: "127.0.0.1", typesafeApiKey: "k1" });
  });

  it("parses env lines", () => {
    expect(parseEnvFile("A=1\n#c\nB='two'\nbad\nC=a=b\n")).toEqual({ A: "1", B: "two", C: "a=b" });
  });
});

describe("main", () => {
  it("exits 2 with usage for an unknown command", async () => {
    const h = makeHarness();

    expect(await runCli(["bogus"], h.ctx)).toBe(2);
    expect(h.err.join("\n")).toContain("usage: tabumcp");
  });

  it("prints usage to stdout and exits 0 for help in every spelling", async () => {
    for (const word of ["--help", "-h", "help"]) {
      const h = makeHarness();

      expect(await runCli([word], h.ctx)).toBe(0);
      expect(h.out.join("\n")).toBe(USAGE);
      expect(h.err).toEqual([]);
    }
  });

  it("prints the version and exits 0", async () => {
    for (const word of ["--version", "-v"]) {
      const h = makeHarness();

      expect(await runCli([word], h.ctx)).toBe(0);
      expect(h.out).toEqual([h.ctx.version]);
    }
  });

  it("exits 2 with usage on stderr when given no arguments", async () => {
    const h = makeHarness();

    expect(await runCli([], h.ctx)).toBe(2);
    expect(h.err.join("\n")).toBe(USAGE);
    expect(h.out).toEqual([]);
  });

  it("registers the lifecycle commands", () => {
    expect(Object.keys(COMMANDS).sort()).toEqual([
      "connect",
      "disconnect",
      "doctor",
      "install",
      "key",
      "post-update",
      "profile",
      "scope",
      "serve",
      "uninstall",
      "update",
    ]);
  });

  it("keeps post-update out of the usage text", () => {
    expect(USAGE).not.toContain("post-update");
  });

  it("passes --purge and --yes to uninstall", async () => {
    const h = makeHarness();
    const paths = resolvePaths(h.ctx);

    await install(h.ctx);

    expect(await runCli(["uninstall", "--purge", "--yes"], h.ctx)).toBe(0);
    expect(existsSync(paths.dataDir)).toBe(false);
    expect(readFileSync(h.logFile, "utf8")).toContain("disable --now tabumcp");
  });
});

describe("uninstall when the service will not stop", () => {
  for (const platform of ["linux", "darwin"] as const) {
    it(`removes nothing and exits 1 on ${platform}, even with --purge`, async () => {
      const h = makeHarness({ platform, arch: platform === "darwin" ? "arm64" : "x64" });
      const paths = resolvePaths(h.ctx);

      await install(h.ctx);
      mkdirSync(paths.storeDir, { recursive: true });
      writeFileSync(join(paths.storeDir, "rows"), "x");
      h.ctx.env.STUB_STUCK = "1";

      expect(await uninstall(h.ctx, { purge: true, yes: true })).toBe(1);
      expect(h.err.join("\n")).toContain("Stop it by hand");
      expect(existsSync(join(paths.storeDir, "rows"))).toBe(true);
      expect(existsSync(paths.appDir)).toBe(true);
      expect(existsSync(platform === "darwin" ? paths.plistFile : paths.unitFile)).toBe(true);
      expect(existsSync(paths.tokenFile)).toBe(true);
    });
  }
});
