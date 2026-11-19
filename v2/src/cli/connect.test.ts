import { chmodSync, existsSync, linkSync, lstatSync, mkdirSync, readFileSync, readdirSync, readlinkSync, realpathSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { basename, delimiter, dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

import { connect, disconnect } from "./connect.ts";
import { execRunner } from "./context.ts";
import { makeHarness } from "./harness.ts";
import { install } from "./install.ts";
import { resolvePaths } from "./paths.ts";
import { runCli } from "./main.ts";
import { uninstall } from "./uninstall.ts";

function put(file: string, text: string): void {
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, text);
}

async function installed() {
  const h = makeHarness();

  expect(await install(h.ctx)).toBe(0);

  return { h, paths: resolvePaths(h.ctx) };
}

function count(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

describe("connect claude", () => {
  it("adds the user-scope HTTP entry with the literal token and one instruction block, twice over", async () => {
    const { h, paths } = await installed();
    const token = readFileSync(paths.tokenFile, "utf8").trim();

    expect(await connect(h.ctx, ["claude"])).toBe(0);
    expect(await connect(h.ctx, ["claude"])).toBe(0);

    const adds = h.calls().filter((line) => line.startsWith("claude mcp add"));

    expect(adds).toEqual([
      `claude mcp add --transport http --scope user tabumcp http://127.0.0.1:8789/mcp --header Authorization: Bearer ${token}`,
      `claude mcp add --transport http --scope user tabumcp http://127.0.0.1:8789/mcp --header Authorization: Bearer ${token}`,
    ]);

    const md = readFileSync(paths.claudeMd, "utf8");

    expect(count(md, "<!-- tabumcp:begin -->")).toBe(1);
    expect(count(md, "<!-- tabumcp:end -->")).toBe(1);
    expect(md.split("\n").length).toBeLessThan(15);
    expect(md).toContain("cwd");
    expect(md).toContain("memory_answer");
  });

  it("refuses a dangling ~/.claude symlink before any claude call", async () => {
    const { h, paths } = await installed();
    const dir = dirname(paths.claudeMd);
    const before = h.calls().length;

    rmSync(dir, { recursive: true, force: true });
    symlinkSync(join(dirname(dir), "no-such-claude"), dir);

    expect(await connect(h.ctx, ["claude"])).toBe(1);
    expect(h.err.join("\n")).toContain(dir);
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(lstatSync(dir).isSymbolicLink()).toBe(true);
  });

  it("refuses a directory at CLAUDE.md before any claude call", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    mkdirSync(paths.claudeMd, { recursive: true });

    expect(await connect(h.ctx, ["claude"])).toBe(1);
    expect(h.err.join("\n")).toContain("not a regular file");
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(lstatSync(paths.claudeMd).isDirectory()).toBe(true);
    expect(readdirSync(paths.claudeMd)).toEqual([]);
  });

  it("returns exit 1 with a message when the token file cannot be read", async () => {
    const { h, paths } = await installed();

    rmSync(paths.tokenFile, { force: true });
    mkdirSync(paths.tokenFile);

    await expect(connect(h.ctx, ["claude"])).resolves.toBe(1);
    expect(h.err.join("\n")).toContain("connect claude");
  });

  it("exits 1 without rejecting when the token file is missing", async () => {
    const h = makeHarness();

    await expect(connect(h.ctx, ["codex"])).resolves.toBe(1);
    expect(h.err.join("\n")).toContain("no token");
  });

  it("refuses a directory at CLAUDE.md on disconnect before any claude call", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    mkdirSync(paths.claudeMd, { recursive: true });

    expect(await disconnect(h.ctx, ["claude"])).toBe(1);
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(lstatSync(paths.claudeMd).isDirectory()).toBe(true);
  });

  it("fails with exit 1 when claude mcp add fails", async () => {
    const { h } = await installed();
    const failing = makeHarness({
      run: async (command, args) =>
        command === "claude" && args[1] === "add"
          ? { code: 3, stdout: "", stderr: "boom" }
          : { code: 0, stdout: "", stderr: "" },
    });

    put(resolvePaths(failing.ctx).tokenFile, "tok\n");

    expect(await connect(failing.ctx, ["claude"])).toBe(1);
    expect(failing.err.join("\n")).toContain("boom");
    expect(existsSync(resolvePaths(failing.ctx).claudeMd)).toBe(false);
    expect(h.err).toEqual([]);
  });

  it("never prints the token when claude fails with empty stderr", async () => {
    const h = makeHarness();

    expect(await install(h.ctx)).toBe(0);

    const token = readFileSync(resolvePaths(h.ctx).tokenFile, "utf8").trim();
    const bin = (h.ctx.env.PATH ?? "").split(delimiter)[0] as string;

    writeFileSync(join(bin, "claude"), `#!/bin/sh\nif [ "$2" = "add" ]; then exit 3; fi\nexit 0\n`);
    chmodSync(join(bin, "claude"), 0o755);

    expect(await connect(h.ctx, ["claude"])).toBe(1);

    const printed = [...h.out, ...h.err].join("\n");

    expect(h.err.join("\n")).toContain("claude mcp add failed");
    expect(printed).not.toContain(token);
  });

  it("execRunner keeps command arguments out of its fallback diagnostic", async () => {
    const h = makeHarness();
    const bin = (h.ctx.env.PATH ?? "").split(delimiter)[0] as string;

    writeFileSync(join(bin, "claude"), "#!/bin/sh\nexit 3\n");
    chmodSync(join(bin, "claude"), 0o755);

    const result = await execRunner("claude", ["--header", "Bearer sekret"], h.ctx.env);

    expect(result.code).toBe(3);
    expect(result.stderr).not.toContain("sekret");
  });

  it("redacts a token that the claude CLI echoes on stderr", async () => {
    const { h, paths } = await installed();
    const token = readFileSync(paths.tokenFile, "utf8").trim();
    const echoing = makeHarness({
      run: async (command, args) =>
        command === "claude" && args[1] === "add"
          ? { code: 3, stdout: "", stderr: `bad header Authorization: Bearer ${token}` }
          : { code: 0, stdout: "", stderr: "" },
    });

    put(resolvePaths(echoing.ctx).tokenFile, `${token}\n`);

    expect(await connect(echoing.ctx, ["claude"])).toBe(1);
    expect(echoing.err.join("\n")).not.toContain(token);
    expect(h.err).toEqual([]);
  });

  it("refuses without a token and with an unknown client", async () => {
    const h = makeHarness();

    expect(await connect(h.ctx, ["claude"])).toBe(1);
    expect(await connect(h.ctx, ["vim"])).toBe(2);
    expect(h.calls()).toEqual([]);
  });

  it("disconnect removes the block and leaves the rest of the file byte-identical", async () => {
    const { h, paths } = await installed();

    for (const original of ["", "# mine\n\nkeep this\n", "no trailing newline", "a\n\n\nb\n"]) {
      put(paths.claudeMd, original);

      expect(await connect(h.ctx, ["claude"])).toBe(0);
      expect(readFileSync(paths.claudeMd, "utf8")).not.toBe(original);
      expect(await disconnect(h.ctx, ["claude"])).toBe(0);
      expect(readFileSync(paths.claudeMd, "utf8")).toBe(original);
    }

    expect(h.calls().filter((line) => line.startsWith("claude mcp remove --scope user tabumcp")).length).toBeGreaterThan(0);
  });
});

describe("connect codex", () => {
  it("writes one marked TOML block with url and http_headers and never runs codex", async () => {
    const { h, paths } = await installed();
    const token = readFileSync(paths.tokenFile, "utf8").trim();

    put(paths.codexConfig, "model = \"o3\"\n");

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(await connect(h.ctx, ["codex"])).toBe(0);

    const toml = readFileSync(paths.codexConfig, "utf8");

    expect(count(toml, "# tabumcp:begin")).toBe(1);
    expect(toml).toContain("[mcp_servers.tabumcp]");
    expect(toml).toContain('url = "http://127.0.0.1:8789/mcp"');
    expect(toml).toContain(`http_headers = { Authorization = "Bearer ${token}" }`);
    expect(toml.startsWith('model = "o3"\n')).toBe(true);
    expect(count(readFileSync(paths.codexAgentsMd, "utf8"), "<!-- tabumcp:begin -->")).toBe(1);
    expect(h.calls().filter((line) => line.startsWith("codex"))).toEqual([]);
  });

  it("creates a new config file mode 600", async () => {
    const { h, paths } = await installed();

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(statSync(paths.codexConfig).mode & 0o777).toBe(0o600);
  });

  it.each([
    '[mcp_servers."tabumcp"]',
    '["mcp_servers".tabumcp]',
    '[ mcp_servers . "tabumcp" ]',
    '[ "mcp_servers" . tabumcp ]',
    "[mcp_servers.'tabumcp']",
    '[mcp_servers."tabumcp".env]',
  ])("refuses the quoted table form %s", async (header) => {
    const { h, paths } = await installed();
    const mine = `${header}\nurl = "http://elsewhere"\n`;

    put(paths.codexConfig, mine);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(readFileSync(paths.codexConfig, "utf8")).toBe(mine);
  });

  it("replaces an existing 0644 config without ever writing the token into the old file", async () => {
    const { h, paths } = await installed();
    const token = readFileSync(paths.tokenFile, "utf8").trim();
    const alias = `${paths.codexConfig}.alias`;

    put(paths.codexConfig, "model = \"o3\"\n");
    chmodSync(paths.codexConfig, 0o644);
    linkSync(paths.codexConfig, alias);

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(readFileSync(alias, "utf8")).not.toContain(token);
    expect(statSync(alias).mode & 0o777).toBe(0o644);
    expect(readFileSync(paths.codexConfig, "utf8")).toContain(token);
    expect(statSync(paths.codexConfig).mode & 0o777).toBe(0o600);
    expect(readdirSync(dirname(paths.codexConfig)).filter((name) => name.endsWith(".tmp"))).toEqual([]);
  });

  it("keeps a symlinked config.toml a symlink and writes its real target mode 600", async () => {
    const { h, paths } = await installed();
    const token = readFileSync(paths.tokenFile, "utf8").trim();
    const target = join(dirname(paths.codexConfig), "dotfiles", "codex.toml");

    put(target, "model = \"o3\"\n");
    chmodSync(target, 0o644);
    symlinkSync(target, paths.codexConfig);

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(lstatSync(paths.codexConfig).isSymbolicLink()).toBe(true);
    expect(readFileSync(target, "utf8")).toContain(token);
    expect(statSync(target).mode & 0o777).toBe(0o600);
  });

  it("refuses a dangling config.toml symlink, writes nothing and leaves the link", async () => {
    const { h, paths } = await installed();
    const target = join(dirname(paths.codexConfig), "dotfiles", "missing.toml");

    mkdirSync(dirname(paths.codexConfig), { recursive: true });
    symlinkSync(target, paths.codexConfig);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain("symlink to a missing target");
    expect(lstatSync(paths.codexConfig).isSymbolicLink()).toBe(true);
    expect(readlinkSync(paths.codexConfig)).toBe(target);
    expect(existsSync(target)).toBe(false);
    expect(existsSync(paths.codexAgentsMd)).toBe(false);
  });

  it("refuses a dangling AGENTS.md symlink before touching config.toml", async () => {
    const { h, paths } = await installed();

    mkdirSync(dirname(paths.codexAgentsMd), { recursive: true });
    symlinkSync(join(dirname(paths.codexAgentsMd), "gone.md"), paths.codexAgentsMd);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(existsSync(paths.codexConfig)).toBe(false);
    expect(lstatSync(paths.codexAgentsMd).isSymbolicLink()).toBe(true);
  });

  it("refuses a dangling ~/.codex symlink and writes nothing", async () => {
    const { h, paths } = await installed();
    const dir = dirname(paths.codexConfig);

    rmSync(dir, { recursive: true, force: true });
    symlinkSync(join(dirname(dir), "no-such-codex"), dir);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain(dir);
    expect(lstatSync(dir).isSymbolicLink()).toBe(true);
    expect(existsSync(join(dirname(dir), "no-such-codex"))).toBe(false);
  });

  it("refuses a directory at AGENTS.md and leaves config.toml byte-identical", async () => {
    const { h, paths } = await installed();

    put(paths.codexConfig, "model = \"x\"\n");
    mkdirSync(paths.codexAgentsMd, { recursive: true });

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain("not a regular file");
    expect(readFileSync(paths.codexConfig, "utf8")).toBe("model = \"x\"\n");
    expect(lstatSync(paths.codexAgentsMd).isDirectory()).toBe(true);
  });

  it("refuses a directory at AGENTS.md and creates no config.toml", async () => {
    const { h, paths } = await installed();

    mkdirSync(paths.codexAgentsMd, { recursive: true });

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(existsSync(paths.codexConfig)).toBe(false);
  });

  it("refuses a directory at config.toml and leaves AGENTS.md untouched", async () => {
    const { h, paths } = await installed();

    put(paths.codexAgentsMd, "keep me\n");
    mkdirSync(paths.codexConfig, { recursive: true });

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain("not a regular file");
    expect(readFileSync(paths.codexAgentsMd, "utf8")).toBe("keep me\n");
    expect(lstatSync(paths.codexConfig).isDirectory()).toBe(true);
  });

  it("refuses a directory at config.toml and creates no AGENTS.md", async () => {
    const { h, paths } = await installed();

    mkdirSync(paths.codexConfig, { recursive: true });

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(existsSync(paths.codexAgentsMd)).toBe(false);
  });

  it("refuses a tabumcp table the installer did not write and writes nothing", async () => {
    const { h, paths } = await installed();
    const mine = "[mcp_servers.tabumcp]\nurl = \"http://elsewhere\"\n";

    put(paths.codexConfig, mine);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain("did not write");
    expect(readFileSync(paths.codexConfig, "utf8")).toBe(mine);
    expect(existsSync(paths.codexAgentsMd)).toBe(false);
  });

  it("disconnect restores config.toml and AGENTS.md byte for byte", async () => {
    const { h, paths } = await installed();
    const config = "model = \"o3\"\n\n[mcp_servers.other]\nurl = \"x\"\n";
    const agents = "# rules\nbe kind\n";

    put(paths.codexConfig, config);
    put(paths.codexAgentsMd, agents);

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(await disconnect(h.ctx, ["codex"])).toBe(0);
    expect(readFileSync(paths.codexConfig, "utf8")).toBe(config);
    expect(readFileSync(paths.codexAgentsMd, "utf8")).toBe(agents);
  });
});

describe("uninstall disconnects connected clients", () => {
  it("removes entries, blocks and both skill copies, and leaves unconnected clients alone", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    put(paths.claudeMd, "mine\n");
    expect(await connect(h.ctx, ["claude"])).toBe(0);
    expect(existsSync(`${h.home}/.agents/skills/tabumcp/SKILL.md`)).toBe(true);
    expect(await uninstall(h.ctx, { purge: false, yes: false })).toBe(0);

    expect(readFileSync(paths.claudeMd, "utf8")).toBe("mine\n");
    expect(h.calls().slice(before).some((line) => line.startsWith("claude mcp remove --scope user tabumcp"))).toBe(true);
    expect(existsSync(paths.codexConfig)).toBe(false);
    expect(existsSync(`${h.home}/.agents/skills/tabumcp`)).toBe(false);
    expect(existsSync(paths.skillDir)).toBe(false);
  });
});

describe("uninstall with a broken client file", () => {
  it("warns, skips that client and still removes the service and skill", async () => {
    const { h, paths } = await installed();

    expect(await connect(h.ctx, ["codex"])).toBe(0);

    const before = h.calls().length;

    mkdirSync(paths.claudeMd, { recursive: true });

    expect(await uninstall(h.ctx, { purge: false, yes: false })).toBe(0);
    expect(h.err.join("\n")).toContain(paths.claudeMd);
    expect(h.calls().slice(before).some((line) => line.startsWith("systemctl --user disable --now"))).toBe(true);
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(lstatSync(paths.claudeMd).isDirectory()).toBe(true);
    expect(readdirSync(paths.claudeMd)).toEqual([]);
    expect(existsSync(paths.skillDir)).toBe(false);
    expect(hasMarker(paths.codexConfig)).toBe(false);
  });

  it("stays silent when the client files are simply missing", async () => {
    const { h } = await installed();

    expect(await uninstall(h.ctx, { purge: false, yes: false })).toBe(0);
    expect(h.err).toEqual([]);
  });
});

function hasMarker(file: string): boolean {
  return existsSync(file) && readFileSync(file, "utf8").includes("tabumcp:begin");
}

describe("read-only targets and rollback", () => {
  it("refuses a read-only AGENTS.md and leaves config.toml byte-identical", async () => {
    const { h, paths } = await installed();

    put(paths.codexConfig, "model = \"x\"\n");
    put(paths.codexAgentsMd, "mine\n");
    chmodSync(paths.codexAgentsMd, 0o444);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(h.err.join("\n")).toContain("not writable");
    expect(readFileSync(paths.codexConfig, "utf8")).toBe("model = \"x\"\n");
    expect(readFileSync(paths.codexAgentsMd, "utf8")).toBe("mine\n");
  });

  it("refuses a read-only AGENTS.md and creates no config.toml", async () => {
    const { h, paths } = await installed();

    put(paths.codexAgentsMd, "mine\n");
    chmodSync(paths.codexAgentsMd, 0o444);

    expect(await connect(h.ctx, ["codex"])).toBe(1);
    expect(existsSync(paths.codexConfig)).toBe(false);
  });

  it("refuses a config.toml in a read-only directory", async () => {
    const { h, paths } = await installed();

    put(paths.codexConfig, "model = \"x\"\n");
    put(paths.codexAgentsMd, "mine\n");
    chmodSync(dirname(paths.codexConfig), 0o555);

    try {
      expect(await connect(h.ctx, ["codex"])).toBe(1);
      expect(readFileSync(paths.codexAgentsMd, "utf8")).toBe("mine\n");
    } finally {
      chmodSync(dirname(paths.codexConfig), 0o755);
    }
  });

  it("refuses a read-only CLAUDE.md before any claude call", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    put(paths.claudeMd, "mine\n");
    chmodSync(paths.claudeMd, 0o444);

    expect(await connect(h.ctx, ["claude"])).toBe(1);
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(readFileSync(paths.claudeMd, "utf8")).toBe("mine\n");
  });

  it("refuses a read-only CLAUDE.md on disconnect before any claude call", async () => {
    const { h, paths } = await installed();

    put(paths.claudeMd, "mine\n");
    expect(await connect(h.ctx, ["claude"])).toBe(0);

    const connected = readFileSync(paths.claudeMd, "utf8");
    const before = h.calls().length;

    chmodSync(paths.claudeMd, 0o444);

    expect(await disconnect(h.ctx, ["claude"])).toBe(1);
    expect(h.calls().slice(before).filter((line) => line.startsWith("claude"))).toEqual([]);
    expect(readFileSync(paths.claudeMd, "utf8")).toBe(connected);
  });

  it("refuses a read-only AGENTS.md on codex disconnect and leaves config.toml connected", async () => {
    const { h, paths } = await installed();

    expect(await connect(h.ctx, ["codex"])).toBe(0);

    const config = readFileSync(paths.codexConfig, "utf8");

    chmodSync(paths.codexAgentsMd, 0o444);

    expect(await disconnect(h.ctx, ["codex"])).toBe(1);
    expect(readFileSync(paths.codexConfig, "utf8")).toBe(config);
  });

  it("rolls config.toml back when the AGENTS.md write fails after the plan passed", async () => {
    const { h, paths } = await installed();

    put(paths.codexConfig, "model = \"x\"\n");
    chmodSync(paths.codexConfig, 0o640);

    const failing = (link: string, text: string, mode?: number): void => {
      if (link === paths.codexAgentsMd) {
        throw Object.assign(new Error("boom"), { code: "EIO" });
      }

      writeRealFile(link, text, mode);
    };

    expect(await connect(h.ctx, ["codex"], failing)).toBe(1);
    expect(h.err.join("\n")).toContain(paths.codexAgentsMd);
    expect(readFileSync(paths.codexConfig, "utf8")).toBe("model = \"x\"\n");
    expect(statSync(paths.codexConfig).mode & 0o777).toBe(0o640);
    expect(existsSync(paths.codexAgentsMd)).toBe(false);
  });

  it("removes a newly created config.toml on rollback", async () => {
    const { h, paths } = await installed();

    const failing = (link: string, text: string, mode?: number): void => {
      if (link === paths.codexAgentsMd) {
        throw new Error("boom");
      }

      writeRealFile(link, text, mode);
    };

    expect(await connect(h.ctx, ["codex"], failing)).toBe(1);
    expect(existsSync(paths.codexConfig)).toBe(false);
  });

  it("removes the claude entry again when the CLAUDE.md write fails after claude mcp add", async () => {
    const { h, paths } = await installed();
    const before = h.calls().length;

    put(paths.claudeMd, "mine\n");

    const failing = (): void => {
      throw new Error("boom");
    };

    expect(await connect(h.ctx, ["claude"], failing)).toBe(1);

    const calls = h.calls().slice(before).filter((line) => line.startsWith("claude"));

    expect(calls.map((line) => line.split(" ").slice(0, 3).join(" "))).toEqual([
      "claude mcp remove",
      "claude mcp add",
      "claude mcp remove",
    ]);
    expect(readFileSync(paths.claudeMd, "utf8")).toBe("mine\n");
  });
});

function writeRealFile(link: string, text: string, mode?: number): void {
  const file = realpathSync(dirname(link)) + "/" + basename(link);

  if (mode === undefined) {
    writeFileSync(file, text);

    return;
  }

  writeFileSync(file, text, { mode });
  chmodSync(file, mode);
}

describe("uninstall with nothing connected", () => {
  it("does not call claude at all", async () => {
    const { h } = await installed();

    expect(await uninstall(h.ctx, { purge: false, yes: false })).toBe(0);
    expect(h.calls().filter((line) => line.startsWith("claude") || line.startsWith("codex"))).toEqual([]);
  });
});

describe("cli routing", () => {
  it("routes connect and disconnect and exits 2 on a bad client", async () => {
    const { h } = await installed();

    expect(await runCli(["connect"], h.ctx)).toBe(2);
    expect(await runCli(["connect", "codex"], h.ctx)).toBe(0);
    expect(await runCli(["disconnect", "codex"], h.ctx)).toBe(0);
    expect(await runCli(["disconnect", "x"], h.ctx)).toBe(2);
  });
});

describe("connect keeps the token private and the file exact", () => {
  it("chmods an existing world-readable config.toml to 600", async () => {
    const { h, paths } = await installed();

    put(paths.codexConfig, "model = \"o3\"\n");
    chmodSync(paths.codexConfig, 0o644);

    expect(await connect(h.ctx, ["codex"])).toBe(0);
    expect(statSync(paths.codexConfig).mode & 0o777).toBe(0o600);
  });

  it("restores a file with text after the block byte for byte", async () => {
    const { h, paths } = await installed();
    const config = "model = \"o3\"\n";
    const agents = "# rules\n";

    put(paths.codexConfig, config);
    put(paths.codexAgentsMd, agents);
    expect(await connect(h.ctx, ["codex"])).toBe(0);
    writeFileSync(paths.codexConfig, `${readFileSync(paths.codexConfig, "utf8")}[other]\nx = 1\n`);
    writeFileSync(paths.codexAgentsMd, `${readFileSync(paths.codexAgentsMd, "utf8")}## later\ntext\n`);

    expect(await disconnect(h.ctx, ["codex"])).toBe(0);
    expect(readFileSync(paths.codexConfig, "utf8")).toBe(`${config}[other]\nx = 1\n`);
    expect(readFileSync(paths.codexAgentsMd, "utf8")).toBe(`${agents}## later\ntext\n`);
  });

  it("restores a file with text before and after a block in the middle", async () => {
    const { h, paths } = await installed();
    const head = "# top\n\n";
    const tail = "## after\nmore\n";

    put(paths.claudeMd, head);
    expect(await connect(h.ctx, ["claude"])).toBe(0);

    const withBlock = readFileSync(paths.claudeMd, "utf8");

    writeFileSync(paths.claudeMd, `${withBlock}${tail}`);

    expect(await disconnect(h.ctx, ["claude"])).toBe(0);
    expect(readFileSync(paths.claudeMd, "utf8")).toBe(`${head}${tail}`);
  });
});
