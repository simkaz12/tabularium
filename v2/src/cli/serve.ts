import { readFileSync } from "node:fs";

import type { Config } from "../config.ts";
import type { Ctx } from "./context.ts";
import { resolvePaths } from "./paths.ts";

export function parseEnvFile(text: string): Record<string, string> {
  const values: Record<string, string> = {};

  for (const line of text.split("\n")) {
    const trimmed = line.trim();

    if (trimmed === "" || trimmed.startsWith("#")) {
      continue;
    }

    const at = trimmed.indexOf("=");

    if (at <= 0) {
      continue;
    }

    const key = trimmed.slice(0, at).trim();
    const value = trimmed.slice(at + 1).trim().replace(/^(["'])(.*)\1$/, "$2");

    values[key] = value;
  }

  return values;
}

export function readEnvFile(file: string): Record<string, string> {
  try {
    return parseEnvFile(readFileSync(file, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return {};
    }

    throw error;
  }
}

export type StartFn = (config?: Config) => Promise<unknown>;

async function loadStart(): Promise<StartFn> {
  const module = await import("../main.ts");

  return module.start as StartFn;
}

export async function serve(ctx: Ctx, loader: () => Promise<StartFn> = loadStart): Promise<number> {
  const paths = resolvePaths(ctx);
  const fromFile = readEnvFile(paths.envFile);

  for (const [key, value] of Object.entries(fromFile)) {
    if (ctx.env[key] === undefined || ctx.env[key] === "") {
      ctx.env[key] = value;
    }
  }

  const { loadConfig } = await import("../config.ts");
  const start = await loader();

  await start(loadConfig(ctx.env));

  return 0;
}
