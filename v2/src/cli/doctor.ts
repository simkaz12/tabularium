import { existsSync, readFileSync } from "node:fs";

import { linkReport } from "./binLink.ts";
import type { Ctx } from "./context.ts";
import { EMBED_MODEL, mcpUrl, nodeProblem, platformProblem, portFrom, resolvePaths } from "./paths.ts";
import { readEnvFile } from "./serve.ts";
import { serviceActive } from "./service.ts";

export interface Check {
  name: string;
  ok: boolean;
  detail: string;
  informational?: boolean;
}

interface HealthBody {
  ok?: boolean;
  embedder?: { status?: string };
}

function mergedEnv(ctx: Ctx): NodeJS.ProcessEnv {
  return { ...readEnvFile(resolvePaths(ctx).envFile), ...ctx.env };
}

function readToken(ctx: Ctx): string | null {
  const file = resolvePaths(ctx).tokenFile;

  if (!existsSync(file)) {
    return null;
  }

  const token = readFileSync(file, "utf8").trim();

  return token === "" ? null : token;
}

export async function fetchHealth(ctx: Ctx): Promise<{ status: number; body: HealthBody | null } | null> {
  const token = readToken(ctx);
  const port = portFrom(mergedEnv(ctx));
  const headers: Record<string, string> = token === null ? {} : { Authorization: `Bearer ${token}` };

  try {
    const response = await ctx.fetch(`http://127.0.0.1:${port}/v2/health`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    const body = response.ok ? ((await response.json()) as HealthBody) : null;

    return { status: response.status, body };
  } catch {
    return null;
  }
}

export async function waitHealthy(ctx: Ctx): Promise<boolean> {
  for (let attempt = 0; attempt < ctx.healthAttempts; attempt += 1) {
    const health = await fetchHealth(ctx);

    if (health !== null && health.status === 200) {
      return true;
    }

    await ctx.sleep(500);
  }

  return false;
}

async function ollamaChecks(ctx: Ctx): Promise<Check[]> {
  const env = mergedEnv(ctx);
  const embedUrl = env.TABULARIUM_EMBED_URL || "http://localhost:11434/api/embed";
  const model = env.TABULARIUM_EMBED_MODEL || EMBED_MODEL;

  try {
    const response = await ctx.fetch(`${new URL(embedUrl).origin}/api/tags`, { signal: AbortSignal.timeout(5000) });

    if (!response.ok) {
      throw new Error(`answered ${response.status}`);
    }

    const tags = (await response.json()) as { models?: Array<{ name?: string }> };
    const present = (tags.models ?? []).some((entry) => entry.name === model || entry.name === `${model}:latest`);

    return [
      { name: "ollama", ok: true, detail: `reachable at ${new URL(embedUrl).origin}` },
      {
        name: "model",
        ok: present,
        detail: present ? `${model} present` : `${model} missing, run: ollama pull ${model}`,
      },
    ];
  } catch (error) {
    return [
      { name: "ollama", ok: false, detail: `not reachable at ${embedUrl}: ${(error as Error).message}` },
      { name: "model", ok: false, detail: `cannot check ${model} without Ollama` },
    ];
  }
}

function stampCheck(health: Awaited<ReturnType<typeof fetchHealth>>): Check {
  const status = health?.body?.embedder?.status;

  if (health === null || health.body === null) {
    return { name: "embedder build", ok: false, detail: "service did not answer" };
  }

  const ok = status === "match" || status === "missing" || status === "unchecked";

  return { name: "embedder build", ok, detail: status ?? "unknown" };
}

function keyHint(ctx: Ctx): string {
  return ctx.platform === "darwin"
    ? `a bad key shows as a warning in ${resolvePaths(ctx).logFile}`
    : "a bad key shows as a warning in: journalctl --user -u tabumcp";
}

export async function runChecks(ctx: Ctx): Promise<Check[]> {
  const checks: Check[] = [];
  const node = nodeProblem(ctx);
  const platform = platformProblem(ctx);

  checks.push({ name: "node", ok: node === null, detail: node ?? `v${ctx.nodeVersion}` });
  checks.push({ name: "platform", ok: platform === null, detail: platform ?? `${ctx.platform}/${ctx.arch}` });
  checks.push(...(await ollamaChecks(ctx)));

  const active = await serviceActive(ctx);
  const health = await fetchHealth(ctx);

  checks.push({ name: "service", ok: active, detail: active ? "active" : "not active" });
  checks.push({
    name: "health",
    ok: health?.status === 200,
    detail: health === null ? "no answer" : `status ${health.status}`,
  });
  checks.push(stampCheck(health));

  const key = Boolean(mergedEnv(ctx).TYPESAFE_API_KEY);

  checks.push({ name: "command link", ok: true, informational: true, detail: linkReport(ctx) });

  checks.push({
    name: "jev key",
    ok: true,
    informational: true,
    detail: key ? `set (not verified), ${keyHint(ctx)}` : "not set (optional, run: tabumcp key set)",
  });

  return checks;
}

export async function doctor(ctx: Ctx): Promise<number> {
  const checks = await runChecks(ctx);

  for (const check of checks) {
    const mark = check.ok ? "pass" : "FAIL";

    ctx.out(`${mark}  ${check.name}: ${check.detail}`);
  }

  ctx.out(`MCP URL: ${mcpUrl({ env: mergedEnv(ctx) })}`);
  ctx.out(`Token file: ${resolvePaths(ctx).tokenFile}`);

  return checks.every((check) => check.ok) ? 0 : 1;
}
