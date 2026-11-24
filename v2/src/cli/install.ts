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
import { DEFAULT_EMBED_MODEL, EMBEDDERS, knownDim } from "./embedders.ts";
import { DEFAULT_PORT, nodeProblem, platformProblem, resolvePaths } from "./paths.ts";
import { readEnvFile } from "./serve.ts";
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

export interface EmbedSetting {
  model: string;
  dim: number;
}

export function writeConfig(ctx: Ctx, embed: EmbedSetting | null = null): void {
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
    ...(embed === null ? [] : [`TABULARIUM_EMBED_MODEL=${embed.model}`, `TABULARIUM_EMBED_DIM=${embed.dim}`]),
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

function embedModelFlag(flags: string[]): string | undefined {
  const inline = flags.find((flag) => flag.startsWith("--embed-model="));

  if (inline !== undefined) {
    return inline.slice("--embed-model=".length) || undefined;
  }

  const at = flags.indexOf("--embed-model");

  if (at === -1) {
    return undefined;
  }

  const value = flags[at + 1];

  if (value === undefined || value.startsWith("--")) {
    throw new Error("--embed-model needs a model name");
  }

  return value;
}

async function askModel(ctx: Ctx): Promise<string> {
  ctx.out("Embedding model (turns memories into vectors for search):");

  EMBEDDERS.forEach((choice, index) => {
    ctx.out(`  ${index + 1}) ${choice.name.padEnd(18)}${choice.note}`);
  });
  ctx.out(`  ${EMBEDDERS.length + 1}) another Ollama embedding model (type its name)`);

  const answer = (await ctx.ask("Choose [1]: ")).trim();

  if (answer === "") {
    return DEFAULT_EMBED_MODEL;
  }

  if (!/^\d+$/.test(answer)) {
    return answer;
  }

  const index = Number(answer) - 1;

  if (index >= 0 && index < EMBEDDERS.length) {
    return EMBEDDERS[index]!.name;
  }

  if (index !== EMBEDDERS.length) {
    throw new Error(`${answer} is not one of the choices`);
  }

  const typed = (await ctx.ask("Model name: ")).trim();

  if (typed === "") {
    throw new Error("no model name given");
  }

  return typed;
}

async function chooseModel(ctx: Ctx, requested: string | undefined): Promise<string> {
  if (requested !== undefined) {
    return requested;
  }

  if (!ctx.interactive) {
    ctx.out(`Using embedding model ${DEFAULT_EMBED_MODEL} (no terminal to ask; pass --embed-model to choose).`);

    return DEFAULT_EMBED_MODEL;
  }

  return askModel(ctx);
}

async function modelPulled(ctx: Ctx, model: string): Promise<boolean> {
  const listed = await ctx.run("ollama", ["list"], ctx.env);

  if (listed.code !== 0) {
    return false;
  }

  return listed.stdout.split("\n").some((line) => {
    const name = line.trim().split(/\s+/)[0];

    return name === model || name === `${model}:latest`;
  });
}

async function ensureModel(ctx: Ctx, model: string): Promise<void> {
  const which = await ctx.run("ollama", ["--version"], ctx.env);

  if (which.code !== 0) {
    ctx.out(
      `Ollama is not installed. TabuMCP needs it to turn memories into vectors. Install it from https://ollama.com/download, then run: ollama pull ${model}`,
    );

    return;
  }

  if (await modelPulled(ctx, model)) {
    return;
  }

  if (!ctx.interactive) {
    ctx.out(`Skipped pulling ${model}: no terminal to confirm. Run: ollama pull ${model}`);

    return;
  }

  if (!(await ctx.confirm(`Pull the embedding model ${model} with ollama now?`))) {
    return;
  }

  const pulled = await ctx.run("ollama", ["pull", model], ctx.env);

  if (pulled.code !== 0) {
    ctx.err(`ollama pull ${model} failed: ${pulled.stderr.trim()}`);
  }
}

async function probeDim(ctx: Ctx, model: string): Promise<number | null> {
  const url = ctx.env.TABULARIUM_EMBED_URL || "http://localhost:11434/api/embed";

  try {
    const response = await ctx.fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, input: "dimension probe" }),
      signal: AbortSignal.timeout(60_000),
    });

    if (!response.ok) {
      return null;
    }

    const body = (await response.json()) as { embeddings?: number[][] };
    const width = body.embeddings?.[0]?.length ?? 0;

    return width > 0 ? width : null;
  } catch {
    return null;
  }
}

async function measureDim(ctx: Ctx, model: string): Promise<number> {
  const measured = await probeDim(ctx, model);

  if (measured !== null) {
    return measured;
  }

  const known = knownDim(model);

  if (known === null) {
    throw new Error(`could not measure the dimension of ${model}. Start Ollama and pull it, then run install again.`);
  }

  return known;
}

async function prepareEmbedder(ctx: Ctx, flags: string[]): Promise<EmbedSetting | null> {
  const requested = embedModelFlag(flags);
  const paths = resolvePaths(ctx);

  if (existsSync(paths.envFile)) {
    const kept = readEnvFile(paths.envFile).TABULARIUM_EMBED_MODEL || DEFAULT_EMBED_MODEL;

    if (requested !== undefined && requested !== kept) {
      throw new Error(`the embedding model is fixed after the first install (${kept}), so ${requested} cannot be used`);
    }

    ctx.out(`Keeping embedding model ${kept} (chosen at first install).`);
    await ensureModel(ctx, kept);

    return null;
  }

  const model = await chooseModel(ctx, requested);

  await ensureModel(ctx, model);

  const dim = await measureDim(ctx, model);

  ctx.out(`Embedding model ${model}, ${dim} dimensions.`);

  return { model, dim };
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

export async function install(ctx: Ctx, flags: string[] = []): Promise<number> {
  const problem = platformProblem(ctx) ?? nodeProblem(ctx);

  if (problem !== null) {
    ctx.err(`install: ${problem}`);

    return 1;
  }

  try {
    const embed = await prepareEmbedder(ctx, flags);

    await installVersion(ctx, ctx.version);
    pointCurrentAt(ctx, ctx.version);

    writeConfig(ctx, embed);
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
