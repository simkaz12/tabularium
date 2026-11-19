import { existsSync, readlinkSync } from "node:fs";
import { basename, join } from "node:path";

import type { Ctx } from "./context.ts";
import { doctor, waitHealthy } from "./doctor.ts";
import { installVersion, pointCurrentAt } from "./install.ts";
import { nodeProblem, platformProblem, resolvePaths } from "./paths.ts";
import { restartService } from "./service.ts";

const VERSION_FORMAT = /^\d+\.\d+\.\d+([-+][0-9A-Za-z.-]+)?$/;

function pinnedVersion(flags: string[]): string | null | undefined {
  const inline = flags.find((flag) => flag.startsWith("--version="));

  if (inline !== undefined) {
    return inline.slice("--version=".length);
  }

  const at = flags.indexOf("--version");

  return at === -1 ? undefined : (flags[at + 1] ?? null);
}

async function resolveTarget(ctx: Ctx, pinned: string | undefined): Promise<string> {
  if (pinned !== undefined) {
    return pinned;
  }

  const result = await ctx.run("npm", ["view", "tabumcp", "version"], ctx.env);
  const latest = result.stdout.trim();

  if (result.code !== 0 || latest === "") {
    throw new Error(`cannot read the latest version from the registry: ${result.stderr.trim() || `exit ${result.code}`}`);
  }

  return latest;
}

async function runPostUpdate(ctx: Ctx, target: string, previous: string): Promise<number> {
  const binary = join(resolvePaths(ctx).appDir, target, "node_modules", ".bin", "tabumcp");
  const result = await ctx.run(binary, ["post-update", "--previous", previous], ctx.env);

  if (result.code === 0) {
    return 0;
  }

  const detail = result.stderr.trim() || `exit ${result.code}`;

  ctx.err(`update: ${target} is active but its post-update step failed: ${detail}`);
  ctx.err("update: run: ~/.local/bin/tabumcp post-update");

  return 1;
}

export async function update(ctx: Ctx, flags: string[] = []): Promise<number> {
  const pinned = pinnedVersion(flags);

  if (pinned === null || (pinned !== undefined && !VERSION_FORMAT.test(pinned))) {
    ctx.err("usage: tabumcp update [--version <x.y.z>]");

    return 2;
  }

  const problem = platformProblem(ctx) ?? nodeProblem(ctx);

  if (problem !== null) {
    ctx.err(`update: ${problem}`);

    return 1;
  }

  const paths = resolvePaths(ctx);

  if (!existsSync(paths.currentLink)) {
    ctx.err("update: tabumcp is not installed, run: npx tabumcp install");

    return 1;
  }

  const previous = basename(readlinkSync(paths.currentLink));
  let target: string;

  try {
    target = await resolveTarget(ctx, pinned);
  } catch (error) {
    ctx.err(`update: ${(error as Error).message}`);

    return 1;
  }

  if (target === previous) {
    ctx.out(`already on ${target}`);

    return 0;
  }

  try {
    await installVersion(ctx, target);
    pointCurrentAt(ctx, target);
    await restartService(ctx);
  } catch (error) {
    ctx.err(`update: ${(error as Error).message}`);

    return rollback(ctx, previous);
  }

  const healthy = await waitHealthy(ctx);
  const code = healthy ? await doctor(ctx) : 1;

  if (code === 0) {
    return runPostUpdate(ctx, target, previous);
  }

  ctx.err(`update: doctor failed on ${target}, restoring ${previous}`);

  return rollback(ctx, previous);
}

async function rollback(ctx: Ctx, previous: string): Promise<number> {
  pointCurrentAt(ctx, previous);

  try {
    await restartService(ctx);
  } catch (error) {
    ctx.err(`update: restart after rollback failed: ${(error as Error).message}`);
  }

  return 1;
}
