import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";

import type { Ctx } from "./context.ts";
import { resolvePaths } from "./paths.ts";
import { restartService } from "./service.ts";

const KEY_NAME = "TYPESAFE_API_KEY";

function withoutKey(text: string): string {
  return text
    .split("\n")
    .filter((line) => !line.trim().startsWith(`${KEY_NAME}=`))
    .join("\n");
}

function withKey(text: string, key: string | null): string {
  const kept = withoutKey(text).replace(/\n*$/, "\n");

  return key === null ? kept : `${kept}${KEY_NAME}=${key}\n`;
}

async function applyKey(ctx: Ctx, key: string | null): Promise<number> {
  const file = resolvePaths(ctx).envFile;

  if (!existsSync(file)) {
    ctx.err(`key: no env file at ${file}, run: tabumcp install`);

    return 1;
  }

  writeFileSync(file, withKey(readFileSync(file, "utf8"), key));
  chmodSync(file, 0o600);

  try {
    await restartService(ctx);
  } catch (error) {
    ctx.err(`key: saved, but the service did not restart: ${(error as Error).message}`);

    return 1;
  }

  return 0;
}

export async function key(ctx: Ctx, flags: string[]): Promise<number> {
  const sub = flags[0];

  if (sub === "set") {
    const value = (await ctx.secret("TypeSafe API key (input hidden): ")).trim();

    if (value === "") {
      ctx.err("key set: no key entered, nothing changed. Run this in a terminal and type the key at the prompt.");

      return 1;
    }

    const code = await applyKey(ctx, value);

    if (code === 0) {
      ctx.out("TypeSafe key saved, service restarted. Jev checks, fold asks and the cleaner are now on.");
    }

    return code;
  }

  if (sub === "clear") {
    const code = await applyKey(ctx, null);

    if (code === 0) {
      ctx.out("TypeSafe key removed, service restarted. Jev checks and the cleaner are off.");
    }

    return code;
  }

  ctx.err("usage: tabumcp key set | clear");

  return 2;
}
