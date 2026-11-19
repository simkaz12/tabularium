import { randomBytes } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";

import { binDirOnPath, createBinLink, pathLine } from "./binLink.ts";
import type { Ctx } from "./context.ts";
import { doctor, waitHealthy } from "./doctor.ts";
import { DEFAULT_PORT, EMBED_MODEL, nodeProblem, platformProblem, resolvePaths } from "./paths.ts";
import { installService } from "./service.ts";

export async function installVersion(ctx: Ctx, version: string): Promise<void> {
  const paths = resolvePaths(ctx);
  const target = join(paths.appDir, version);

  mkdirSync(target, { recursive: true });

  const result = await ctx.run("npm", ["install", "--prefix", target, `tabumcp@${version}`], ctx.env);

  if (result.code !== 0) {
    throw new Error(`npm install failed: ${result.stderr.trim() || `exit ${result.code}`}`);
  }
}

export function pointCurrentAt(ctx: Ctx, version: string): void {
  const paths = resolvePaths(ctx);
  const staging = `${paths.currentLink}.next`;

  rmSync(staging, { force: true });
  symlinkSync(join(paths.appDir, version), staging, "dir");
  renameSync(staging, paths.currentLink);
}

function writeIfAbsent(file: string, content: string, mode: number): boolean {
  if (existsSync(file)) {
    return false;
  }

  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, content, { mode, flag: "wx" });
  chmodSync(file, mode);

  return true;
}

export function writeConfig(ctx: Ctx): void {
  const paths = resolvePaths(ctx);

  mkdirSync(paths.configDir, { recursive: true, mode: 0o700 });
  mkdirSync(paths.dataDir, { recursive: true });

  const envText = [
    "TABUMCP_MODE=standalone",
    "TABULARIUM_HOST=127.0.0.1",
    `TABULARIUM_PORT=${DEFAULT_PORT}`,
    `TABULARIUM_STORE=${paths.storeDir}`,
    `TABUMCP_TOKEN_FILE=${paths.tokenFile}`,
    `TABUMCP_SCOPES_FILE=${paths.scopesFile}`,
    "",
  ].join("\n");

  writeIfAbsent(paths.envFile, envText, 0o600);
  writeIfAbsent(paths.tokenFile, `${randomBytes(32).toString("hex")}\n`, 0o600);
  writeIfAbsent(paths.scopesFile, `${JSON.stringify({ profiles: [], scopes: {}, paths: [] }, null, 2)}\n`, 0o644);
}

export function copySkill(ctx: Ctx, packageRoot: string = ctx.packageRoot): boolean {
  const source = join(packageRoot, "skill", "tabumcp", "SKILL.md");

  if (!existsSync(source)) {
    return false;
  }

  const paths = resolvePaths(ctx);

  for (const dir of [paths.skillDir, paths.codexSkillDir]) {
    mkdirSync(dir, { recursive: true });
    copyFileSync(source, join(dir, "SKILL.md"));
  }

  return true;
}

async function offerModel(ctx: Ctx): Promise<void> {
  const which = await ctx.run("ollama", ["--version"], ctx.env);

  if (which.code !== 0) {
    return;
  }

  const listed = await ctx.run("ollama", ["list"], ctx.env);

  if (listed.code === 0 && listed.stdout.includes(EMBED_MODEL)) {
    return;
  }

  if (await ctx.confirm(`Pull the embedding model ${EMBED_MODEL} with ollama now?`)) {
    const pulled = await ctx.run("ollama", ["pull", EMBED_MODEL], ctx.env);

    if (pulled.code !== 0) {
      ctx.err(`ollama pull ${EMBED_MODEL} failed: ${pulled.stderr.trim()}`);
    }
  }
}

function linkCommand(ctx: Ctx): void {
  try {
    if (createBinLink(ctx) === "foreign") {
      ctx.err(`install: ${resolvePaths(ctx).binLink} exists and is not ours, left alone`);
    }
  } catch (error) {
    ctx.err(`install: could not link the tabumcp command: ${(error as Error).message}`);
  }
}

export async function install(ctx: Ctx): Promise<number> {
  const problem = platformProblem(ctx) ?? nodeProblem(ctx);

  if (problem !== null) {
    ctx.err(`install: ${problem}`);

    return 1;
  }

  try {
    await installVersion(ctx, ctx.version);
    pointCurrentAt(ctx, ctx.version);
    writeConfig(ctx);
    await offerModel(ctx);
    await installService(ctx);
  } catch (error) {
    ctx.err(`install: ${(error as Error).message}`);

    return 1;
  }

  if (!copySkill(ctx)) {
    ctx.err("install: setup skill not found in the package, skipped");
  }

  linkCommand(ctx);

  await waitHealthy(ctx);

  const code = await doctor(ctx);

  if (!binDirOnPath(ctx)) {
    ctx.out(`${resolvePaths(ctx).binDir} is not on your PATH. Add this line to your shell profile:`);
    ctx.out(`  ${pathLine()}`);
  }

  ctx.out("Next: ask your agent to run the tabumcp skill.");

  return code;
}
