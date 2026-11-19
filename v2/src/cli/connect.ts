import { randomUUID } from "node:crypto";
import { accessSync, chmodSync, constants, existsSync, lstatSync, mkdirSync, readFileSync, readlinkSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import type { Ctx } from "./context.ts";
import {
  MD_BEGIN,
  MD_END,
  TOML_BEGIN,
  TOML_END,
  hasBlock,
  instructionBlock,
  outsideBlock,
  removeBlock,
  upsertBlock,
} from "./instructions.ts";
import { mcpUrl, resolvePaths } from "./paths.ts";
import { readToken } from "./profile.ts";
import { readEnvFile } from "./serve.ts";

export const SERVER_NAME = "tabumcp";
const TOML_TABLE = /^\s*\[\s*["']?mcp_servers["']?\s*\.\s*["']?tabumcp["']?\s*(\.[^\]]*)?\]/m;

function resolveTarget(file: string): string {
  try {
    return realpathSync(file);
  } catch {
    if (lstatSync(file, { throwIfNoEntry: false })?.isSymbolicLink()) {
      throw new Error(`${file} is a symlink to a missing target (${readlinkSync(file)}), refusing to replace it`);
    }

    return file;
  }
}

function prepareTarget(file: string): void {
  try {
    mkdirSync(dirname(file), { recursive: true });
  } catch (error) {
    throw new Error(`cannot create ${dirname(file)} (${(error as NodeJS.ErrnoException).code ?? "error"}), refusing to continue`);
  }

  resolveTarget(file);
}

interface Target {
  link: string;
  original: string | null;
  mode: number | undefined;
  next: string;
  staged: boolean;
}

type Write = (link: string, text: string, mode?: number) => void;

function requireWritable(file: string, resolved: string, staged: boolean): void {
  try {
    accessSync(existsSync(resolved) ? resolved : dirname(resolved), constants.W_OK);

    if (staged) {
      accessSync(dirname(resolved), constants.W_OK);
    }
  } catch {
    throw new Error(`${file} is not writable, refusing to continue`);
  }
}

function readRegular(file: string): string | null {
  const resolved = resolveTarget(file);

  if (!existsSync(resolved)) {
    return null;
  }

  if (!statSync(resolved).isFile()) {
    throw new Error(`${file} is not a regular file, refusing to replace it`);
  }

  return readFileSync(resolved, "utf8");
}

function planTarget(file: string, staged: boolean): { text: string; original: string | null; mode: number | undefined } {
  prepareTarget(file);

  const original = readRegular(file);
  const resolved = resolveTarget(file);

  requireWritable(file, resolved, staged);

  return {
    text: original ?? "",
    original,
    mode: original === null ? undefined : statSync(resolved).mode & 0o7777,
  };
}

function restore(target: Target): void {
  try {
    if (target.original === null) {
      rmSync(resolveTarget(target.link), { force: true });
    } else {
      writeText(target.link, target.original, target.staged ? target.mode : undefined);
    }
  } catch {
    return;
  }
}

function applyTargets(targets: Target[], write: Write): void {
  const touched: Target[] = [];

  for (const target of targets) {
    touched.push(target);

    try {
      write(target.link, target.next, target.staged ? 0o600 : undefined);
    } catch (error) {
      touched.reverse().forEach(restore);

      throw new Error(`cannot write ${target.link} (${(error as NodeJS.ErrnoException).code ?? (error as Error).message}), restored the other files`);
    }
  }
}

function writeText(link: string, text: string, mode?: number): void {
  mkdirSync(dirname(link), { recursive: true });

  const file = resolveTarget(link);

  if (mode === undefined) {
    writeFileSync(file, text);

    return;
  }

  const staging = `${file}.${randomUUID()}.tmp`;

  try {
    writeFileSync(staging, text, { mode });
    chmodSync(staging, mode);
    renameSync(staging, file);
  } catch (error) {
    rmSync(staging, { force: true });

    throw error;
  }
}

function redact(text: string, token: string): string {
  return token === "" ? text : text.split(token).join("[redacted]");
}

function serverUrl(ctx: Ctx): string {
  return mcpUrl({ env: { ...readEnvFile(resolvePaths(ctx).envFile), ...ctx.env } });
}

function tomlBlock(url: string, token: string): string {
  return [
    TOML_BEGIN,
    `[mcp_servers.${SERVER_NAME}]`,
    `url = ${JSON.stringify(url)}`,
    `http_headers = { Authorization = ${JSON.stringify(`Bearer ${token}`)} }`,
    TOML_END,
  ].join("\n");
}

async function connectClaude(ctx: Ctx, url: string, token: string, write: Write): Promise<number> {
  const paths = resolvePaths(ctx);
  const plan = planTarget(paths.claudeMd, false);
  const target: Target = {
    link: paths.claudeMd,
    original: plan.original,
    mode: plan.mode,
    next: upsertBlock(plan.text, MD_BEGIN, MD_END, instructionBlock()),
    staged: false,
  };

  await ctx.run("claude", ["mcp", "remove", "--scope", "user", SERVER_NAME], ctx.env);

  const added = await ctx.run(
    "claude",
    [
      "mcp",
      "add",
      "--transport",
      "http",
      "--scope",
      "user",
      SERVER_NAME,
      url,
      "--header",
      `Authorization: Bearer ${token}`,
    ],
    ctx.env,
  );

  if (added.code !== 0) {
    const detail = redact(added.stderr, token).trim() || `exit ${added.code}`;

    ctx.err(`connect claude: claude mcp add failed: ${detail}`);

    return 1;
  }

  try {
    applyTargets([target], write);
  } catch (error) {
    await ctx.run("claude", ["mcp", "remove", "--scope", "user", SERVER_NAME], ctx.env);

    throw error;
  }

  ctx.out(`claude connected to ${url}, instructions in ${paths.claudeMd}`);

  return 0;
}

function connectCodex(ctx: Ctx, url: string, token: string, write: Write): number {
  const paths = resolvePaths(ctx);
  const config = planTarget(paths.codexConfig, true);
  const agents = planTarget(paths.codexAgentsMd, false);

  if (TOML_TABLE.test(outsideBlock(config.text, TOML_BEGIN, TOML_END))) {
    ctx.err(
      `connect codex: ${paths.codexConfig} already has a [mcp_servers.${SERVER_NAME}] table that tabumcp did not write. ` +
        "Remove it or rename it, then run connect again.",
    );

    return 1;
  }

  applyTargets(
    [
      {
        link: paths.codexConfig,
        original: config.original,
        mode: config.mode,
        next: upsertBlock(config.text, TOML_BEGIN, TOML_END, tomlBlock(url, token)),
        staged: true,
      },
      {
        link: paths.codexAgentsMd,
        original: agents.original,
        mode: agents.mode,
        next: upsertBlock(agents.text, MD_BEGIN, MD_END, instructionBlock()),
        staged: false,
      },
    ],
    write,
  );
  ctx.out(`codex connected to ${url}, instructions in ${paths.codexAgentsMd}`);

  return 0;
}

function planInstructionRefresh(file: string): Target | null {
  if (!hasBlockSafely(file, MD_BEGIN, MD_END, () => undefined)) {
    return null;
  }

  const plan = planTarget(file, false);

  return {
    link: file,
    original: plan.original,
    mode: plan.mode,
    next: upsertBlock(plan.text, MD_BEGIN, MD_END, instructionBlock()),
    staged: false,
  };
}

export function refreshInstructions(ctx: Ctx, write: Write = writeText): void {
  const paths = resolvePaths(ctx);
  const targets = [paths.claudeMd, paths.codexAgentsMd]
    .map(planInstructionRefresh)
    .filter((target): target is Target => target !== null);

  applyTargets(targets, write);
}

export async function connect(ctx: Ctx, flags: string[], write: Write = writeText): Promise<number> {
  const client = flags[0];

  if (client !== "claude" && client !== "codex") {
    ctx.err("usage: tabumcp connect claude|codex");

    return 2;
  }

  try {
    const token = readToken(ctx);

    if (token === null) {
      ctx.err(`connect ${client}: no token at ${resolvePaths(ctx).tokenFile}, run: tabumcp install`);

      return 1;
    }

    const url = serverUrl(ctx);

    return client === "claude" ? await connectClaude(ctx, url, token, write) : connectCodex(ctx, url, token, write);
  } catch (error) {
    ctx.err(`connect ${client}: ${(error as Error).message}`);

    return 1;
  }
}

function planStrip(file: string, begin: string, end: string): Target | null {
  if (!existsSync(file)) {
    return null;
  }

  if (!statSync(file).isFile()) {
    throw new Error(`${file} is not a regular file, refusing to edit it`);
  }

  const text = readFileSync(file, "utf8");

  if (!hasBlock(text, begin, end)) {
    return null;
  }

  requireWritable(file, resolveTarget(file), false);

  return { link: file, original: text, mode: statSync(file).mode & 0o7777, next: removeBlock(text, begin, end), staged: false };
}

function applyStrips(strips: Array<Target | null>): void {
  applyTargets(
    strips.filter((strip): strip is Target => strip !== null),
    writeText,
  );
}

export async function disconnectClient(ctx: Ctx, client: string): Promise<number> {
  const paths = resolvePaths(ctx);

  if (client === "claude") {
    const strips = [planStrip(paths.claudeMd, MD_BEGIN, MD_END)];
    const removed = await ctx.run("claude", ["mcp", "remove", "--scope", "user", SERVER_NAME], ctx.env);

    applyStrips(strips);

    if (removed.code !== 0) {
      ctx.err(`disconnect claude: claude mcp remove failed: ${removed.stderr.trim() || `exit ${removed.code}`}`);

      return 1;
    }

    ctx.out("claude disconnected");

    return 0;
  }

  if (client === "codex") {
    applyStrips([
      planStrip(paths.codexConfig, TOML_BEGIN, TOML_END),
      planStrip(paths.codexAgentsMd, MD_BEGIN, MD_END),
    ]);
    ctx.out("codex disconnected");

    return 0;
  }

  ctx.err("usage: tabumcp disconnect claude|codex");

  return 2;
}

export async function disconnect(ctx: Ctx, flags: string[]): Promise<number> {
  try {
    return await disconnectClient(ctx, flags[0] ?? "");
  } catch (error) {
    ctx.err(`disconnect ${flags[0]}: ${(error as Error).message}`);

    return 1;
  }
}

function hasBlockSafely(file: string, begin: string, end: string, skipped: (file: string) => void): boolean {
  try {
    if (!existsSync(file)) {
      return false;
    }

    if (!statSync(file).isFile()) {
      skipped(file);

      return false;
    }

    return hasBlock(readFileSync(file, "utf8"), begin, end);
  } catch {
    skipped(file);

    return false;
  }
}

export function connectedClients(ctx: Ctx, skipped: (file: string) => void = () => undefined): string[] {
  const paths = resolvePaths(ctx);
  const clients: string[] = [];

  if (hasBlockSafely(paths.claudeMd, MD_BEGIN, MD_END, skipped)) {
    clients.push("claude");
  }

  if (
    hasBlockSafely(paths.codexConfig, TOML_BEGIN, TOML_END, skipped) ||
    hasBlockSafely(paths.codexAgentsMd, MD_BEGIN, MD_END, skipped)
  ) {
    clients.push("codex");
  }

  return clients;
}
