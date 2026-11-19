import { randomBytes } from "node:crypto";
import { lstatSync, mkdirSync, readlinkSync, renameSync, rmSync, symlinkSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";

import type { Ctx } from "./context.ts";
import { resolvePaths } from "./paths.ts";

export type LinkState = "ours" | "absent" | "foreign";

export function linkTarget(ctx: Ctx): string {
  return join(resolvePaths(ctx).currentLink, "node_modules", ".bin", "tabumcp");
}

export function linkState(ctx: Ctx): LinkState {
  const paths = resolvePaths(ctx);

  try {
    lstatSync(paths.binLink);
  } catch {
    return "absent";
  }

  try {
    const pointsAt = resolve(dirname(paths.binLink), readlinkSync(paths.binLink));

    return pointsAt.startsWith(`${paths.appDir}/`) ? "ours" : "foreign";
  } catch {
    return "foreign";
  }
}

export function binDirOnPath(ctx: Ctx): boolean {
  const binDir = resolvePaths(ctx).binDir;

  return (ctx.env.PATH ?? "").split(delimiter).some((entry) => entry !== "" && resolve(entry) === binDir);
}

export function pathLine(): string {
  return 'export PATH="$HOME/.local/bin:$PATH"';
}

export function createBinLink(ctx: Ctx): "linked" | "foreign" {
  const paths = resolvePaths(ctx);

  if (linkState(ctx) === "foreign") {
    return "foreign";
  }

  const staging = `${paths.binLink}.${process.pid}.${randomBytes(6).toString("hex")}.next`;

  mkdirSync(paths.binDir, { recursive: true });
  try {
    symlinkSync(linkTarget(ctx), staging);
    renameSync(staging, paths.binLink);
  } catch (error) {
    rmSync(staging, { force: true });

    throw error;
  }

  return "linked";
}

export function removeBinLink(ctx: Ctx): void {
  if (linkState(ctx) === "ours") {
    rmSync(resolvePaths(ctx).binLink, { force: true });
  }
}

export function linkReport(ctx: Ctx): string {
  const paths = resolvePaths(ctx);
  const state = linkState(ctx);

  if (state === "absent") {
    return `${paths.binLink} missing, run: npx tabumcp install`;
  }

  if (state === "foreign") {
    return `${paths.binLink} exists and does not point into ${paths.appDir}`;
  }

  return binDirOnPath(ctx)
    ? `${paths.binLink}, on PATH`
    : `${paths.binLink}, not on PATH; add to your shell profile: ${pathLine()}`;
}
