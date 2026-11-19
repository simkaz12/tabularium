import { appendFileSync, chmodSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";

import { execRunner } from "./context.ts";
import type { Ctx } from "./context.ts";
import { packageRootFrom } from "./context.ts";
import { postUpdate } from "./postUpdate.ts";

export interface Harness {
  ctx: Ctx;
  home: string;
  logFile: string;
  calls: () => string[];
  out: string[];
  err: string[];
  health: { status: number; embedder: string };
  tagsOk: { value: boolean };
  confirmAnswer: { value: boolean };
  secretAnswer: { value: string };
  registry: { version: string; code: number };
  newBinary: { fail: boolean; run: ((flags: string[]) => Promise<number>) | null };
}

const STUBS = ["systemctl", "launchctl", "npm", "ollama", "claude", "codex"];

const STUB_BODY = `#!/bin/sh
echo "$(basename "$0") $*" >> "$STUB_LOG"
STOPPED="$STUB_LOG.stopped"
case "$(basename "$0") $*" in
  "systemctl --user disable --now"*|"launchctl bootout"*) [ -z "$STUB_STUCK" ] && touch "$STOPPED" ;;
  "systemctl --user enable"*|"systemctl --user restart"*|"launchctl bootstrap"*|"launchctl kickstart"*) rm -f "$STOPPED" ;;
esac
if [ "$(basename "$0")" = "systemctl" ] && [ "$2" = "is-active" ]; then
  if [ -e "$STOPPED" ]; then echo inactive; exit 3; fi
  echo active
fi
if [ "$(basename "$0")" = "launchctl" ] && [ "$1" = "print" ] && [ -e "$STOPPED" ]; then exit 113; fi
if [ "$(basename "$0")" = "ollama" ] && [ "$1" = "list" ]; then echo "$STUB_OLLAMA_LIST"; fi
exit 0
`;

export function makeHarness(overrides: Partial<Ctx> = {}): Harness {
  const root = mkdtempSync(join(tmpdir(), "tabumcp-cli-"));
  const home = join(root, "home");
  const bin = join(root, "bin");
  const pkg = join(root, "pkg");
  const logFile = join(root, "stub.log");

  mkdirSync(home, { recursive: true });
  mkdirSync(bin, { recursive: true });
  mkdirSync(join(pkg, "skill", "tabumcp"), { recursive: true });
  writeFileSync(logFile, "");
  writeFileSync(join(pkg, "skill", "tabumcp", "SKILL.md"), "skill body\n");
  cpSync(join(packageRootFrom(import.meta.url), "templates"), join(pkg, "templates"), { recursive: true });

  for (const name of STUBS) {
    writeFileSync(join(bin, name), STUB_BODY);
    chmodSync(join(bin, name), 0o755);
  }

  const out: string[] = [];
  const err: string[] = [];
  const health = { status: 200, embedder: "match" };
  const tagsOk = { value: true };
  const confirmAnswer = { value: false };
  const secretAnswer = { value: "" };
  const registry = { version: "", code: 0 };
  const newBinary: Harness["newBinary"] = { fail: false, run: null };

  const fakeFetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);

    if (url.endsWith("/api/tags")) {
      if (!tagsOk.value) {
        throw new Error("connection refused");
      }

      return Response.json({ models: [{ name: "nomic-embed-text:latest" }] });
    }

    const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;

    const tokenFile = join(home, ".config", "tabumcp", "token");
    const token = existsSync(tokenFile) ? readFileSync(tokenFile, "utf8").trim() : "";

    if (auth !== `Bearer ${token}`) {
      return new Response("{}", { status: 401 });
    }

    return Response.json({ ok: true, embedder: { status: health.embedder } }, { status: health.status });
  }) as typeof fetch;

  const ctx: Ctx = {
    platform: "linux",
    arch: "x64",
    home,
    env: { PATH: `${bin}${delimiter}${process.env.PATH ?? ""}`, STUB_LOG: logFile, STUB_OLLAMA_LIST: "nomic-embed-text:latest" },
    nodeVersion: "22.11.0",
    nodePath: "/opt/node/bin/node",
    uid: 501,
    version: "0.1.0",
    packageRoot: pkg,
    run: async (command, args, env) => {
      if (command === "npm" && args[0] === "view") {
        return { code: registry.code, stdout: `${registry.version || ctx.version}\n`, stderr: "" };
      }

      if (command.endsWith("/node_modules/.bin/tabumcp")) {
        appendFileSync(logFile, `${command} ${args.join(" ")}\n`);

        if (newBinary.fail) {
          return { code: 1, stdout: "", stderr: "tabumcp: command not found" };
        }

        const flags = args.slice(1);
        const code = await (newBinary.run ?? ((rest) => postUpdate(ctx, rest)))(flags);

        return { code, stdout: "", stderr: "" };
      }

      return execRunner(command, args, env);
    },
    fetch: fakeFetch,
    out: (line) => out.push(line),
    err: (line) => err.push(line),
    confirm: async () => confirmAnswer.value,
    secret: async () => secretAnswer.value,
    sleep: async () => {},
    healthAttempts: 1,
    ...overrides,
  };

  return {
    ctx,
    home,
    logFile,
    calls: () => readFileSync(logFile, "utf8").split("\n").filter((line) => line !== ""),
    out,
    err,
    health,
    tagsOk,
    confirmAnswer,
    secretAnswer,
    registry,
    newBinary,
  };
}
