import { rmSync } from "node:fs";

import { removeBinLink } from "./binLink.ts";
import { connectedClients, disconnectClient } from "./connect.ts";
import type { Ctx } from "./context.ts";
import { resolvePaths } from "./paths.ts";
import { removeService } from "./service.ts";

export const CONNECTABLE_CLIENTS = ["claude", "codex"] as const;

export async function disconnectClients(ctx: Ctx, clients: readonly string[]): Promise<void> {
  const connected = connectedClients(ctx, (file) =>
    ctx.err(`uninstall: cannot read ${file}, leaving it and its client connection alone`),
  );

  for (const client of clients) {
    if (!connected.includes(client)) {
      continue;
    }

    try {
      await disconnectClient(ctx, client);
    } catch (error) {
      ctx.err(`uninstall: could not disconnect ${client}: ${(error as Error).message}`);
    }
  }
}

export interface UninstallOptions {
  purge: boolean;
  yes: boolean;
}

export async function uninstall(ctx: Ctx, options: UninstallOptions): Promise<number> {
  const paths = resolvePaths(ctx);

  try {
    await removeService(ctx);
  } catch (error) {
    ctx.err(`uninstall: ${(error as Error).message}`);

    return 1;
  }

  await disconnectClients(ctx, CONNECTABLE_CLIENTS);

  rmSync(paths.skillDir, { recursive: true, force: true });
  rmSync(paths.codexSkillDir, { recursive: true, force: true });
  removeBinLink(ctx);
  rmSync(paths.appDir, { recursive: true, force: true });
  ctx.out("tabumcp service, app and skill removed");

  if (!options.purge) {
    ctx.out(`kept config ${paths.configDir} and data ${paths.dataDir}, pass --purge to delete them`);

    return 0;
  }

  const approved = options.yes || (await ctx.confirm(`Delete ${paths.configDir} and ${paths.dataDir}, including the store?`));

  if (!approved) {
    ctx.out("purge cancelled, config and store kept");

    return 0;
  }

  rmSync(paths.configDir, { recursive: true, force: true });
  rmSync(paths.dataDir, { recursive: true, force: true });
  ctx.out("config and data deleted");

  return 0;
}
