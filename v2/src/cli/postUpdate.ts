import { readdirSync, readlinkSync, rmSync } from "node:fs";
import { basename, join } from "node:path";

import { refreshInstructions } from "./connect.ts";
import type { Ctx } from "./context.ts";
import { copySkill } from "./install.ts";
import { resolvePaths } from "./paths.ts";

function previousVersion(flags: string[]): string | null {
  const at = flags.indexOf("--previous");

  return at === -1 ? null : (flags[at + 1] ?? null);
}

function pruneVersions(ctx: Ctx, previous: string | null): void {
  const paths = resolvePaths(ctx);
  const keep = new Set([basename(readlinkSync(paths.currentLink)), ...(previous === null ? [] : [previous])]);

  for (const entry of readdirSync(paths.appDir, { withFileTypes: true })) {
    if (entry.isDirectory() && entry.name !== "current" && !keep.has(entry.name)) {
      rmSync(join(paths.appDir, entry.name), { recursive: true, force: true });
    }
  }
}

export async function postUpdate(ctx: Ctx, flags: string[] = []): Promise<number> {
  const previous = previousVersion(flags);

  copySkill(ctx);
  refreshInstructions(ctx);

  if (previous !== null) {
    pruneVersions(ctx, previous);
  }

  ctx.out(`post-update done on ${ctx.version}`);

  return 0;
}
