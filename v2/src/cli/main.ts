#!/usr/bin/env node
import { readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";

import type { Ctx } from "./context.ts";
import { defaultContext, packageRootFrom } from "./context.ts";
import { connect, disconnect } from "./connect.ts";
import { doctor } from "./doctor.ts";
import { install } from "./install.ts";
import { key } from "./key.ts";
import { postUpdate } from "./postUpdate.ts";
import { profile } from "./profile.ts";
import { scope } from "./scope.ts";
import { serve } from "./serve.ts";
import { uninstall } from "./uninstall.ts";
import { update } from "./update.ts";

export const USAGE = [
  "usage: tabumcp <command>",
  "  install [--embed-model <name>]   set up the user service",
  "  update [--version <x>]   install the latest version (or <x>) and restart the service",
  "  uninstall [--purge] [--yes]",
  "  doctor             check the install",
  "  serve              run the server in the foreground",
  "  profile add|list|delete   keep work and personal memory apart",
  "  scope add|list|remove     map project folders to scopes and profiles",
  "  connect claude|codex      add the MCP entry and instruction block",
  "  disconnect claude|codex   remove them",
  "  key set|clear             set or remove the TypeSafe key",
].join("\n");

type Handler = (ctx: Ctx, flags: string[]) => Promise<number>;

export const COMMANDS: Record<string, Handler> = {
  install: (ctx, flags) => install(ctx, flags),
  update: (ctx, flags) => update(ctx, flags),
  "post-update": (ctx, flags) => postUpdate(ctx, flags),
  uninstall: (ctx, flags) => uninstall(ctx, { purge: flags.includes("--purge"), yes: flags.includes("--yes") }),
  doctor: (ctx) => doctor(ctx),
  serve: (ctx) => serve(ctx),
  profile: (ctx, flags) => profile(ctx, flags),
  scope: (ctx, flags) => scope(ctx, flags),
  connect: (ctx, flags) => connect(ctx, flags),
  disconnect: (ctx, flags) => disconnect(ctx, flags),
  key: (ctx, flags) => key(ctx, flags),
};

export async function runCli(argv: string[], ctx: Ctx): Promise<number> {
  const [command, ...flags] = argv;

  if (command === "--help" || command === "-h" || command === "help") {
    ctx.out(USAGE);

    return 0;
  }

  if (command === "--version" || command === "-v") {
    ctx.out(ctx.version);

    return 0;
  }

  const handler = command === undefined ? undefined : COMMANDS[command];

  if (handler === undefined) {
    ctx.err(USAGE);

    return 2;
  }

  try {
    return await handler(ctx, flags);
  } catch (error) {
    ctx.err(`${command}: ${(error as Error).message}`);

    return 1;
  }
}

export function packageVersion(moduleUrl: string): string {
  const file = `${packageRootFrom(moduleUrl)}/package.json`;

  return (JSON.parse(readFileSync(file, "utf8")) as { version: string }).version;
}

function invokedDirectly(): boolean {
  const entry = process.argv[1];

  if (entry === undefined) {
    return false;
  }

  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (invokedDirectly()) {
  const ctx = defaultContext(import.meta.url, packageVersion(import.meta.url));

  process.exitCode = await runCli(process.argv.slice(2), ctx);
}
