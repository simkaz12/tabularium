import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { parseArgs } from "node:util";

import type { Ctx } from "./context.ts";
import { portFrom, resolvePaths } from "./paths.ts";
import { readEnvFile } from "./serve.ts";

export const DEFAULT_PROFILE = "default";
const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

export interface ScopesDoc {
  profiles: string[];
  scopes: Record<string, string>;
  paths: Array<{ match: string; scope: string }>;
}

export interface ServiceReply {
  status: number;
  body: unknown;
}

export function readScopes(ctx: Ctx): ScopesDoc {
  const file = resolvePaths(ctx).scopesFile;

  if (!existsSync(file)) {
    return { profiles: [], scopes: {}, paths: [] };
  }

  const parsed = JSON.parse(readFileSync(file, "utf8")) as Partial<ScopesDoc>;

  return {
    profiles: (parsed.profiles ?? []).map((profile) => profile.trim()),
    scopes: Object.fromEntries(
      Object.entries(parsed.scopes ?? {}).map(([scope, owner]) => [scope.trim(), owner.trim()]),
    ),
    paths: (parsed.paths ?? []).map((rule) => ({ match: rule.match.trim(), scope: rule.scope.trim() })),
  };
}

export function writeScopes(ctx: Ctx, doc: ScopesDoc): void {
  const file = resolvePaths(ctx).scopesFile;

  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, `${JSON.stringify(doc, null, 2)}\n`);
}

export function readToken(ctx: Ctx): string | null {
  const file = resolvePaths(ctx).tokenFile;

  if (!existsSync(file)) {
    return null;
  }

  const token = readFileSync(file, "utf8").trim();

  return token === "" ? null : token;
}

export function servicePort(ctx: Ctx): number {
  return portFrom({ ...readEnvFile(resolvePaths(ctx).envFile), ...ctx.env });
}

export async function serviceCall(ctx: Ctx, method: string, path: string): Promise<ServiceReply> {
  const token = readToken(ctx);

  if (token === null) {
    throw new Error(`no token at ${resolvePaths(ctx).tokenFile}, run: tabumcp install`);
  }

  let response: Response;

  try {
    response = await ctx.fetch(`http://127.0.0.1:${servicePort(ctx)}${path}`, {
      method,
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(15000),
    });
  } catch (error) {
    throw new Error(`the tabumcp service did not answer: ${(error as Error).message}. Run: tabumcp doctor`);
  }

  const text = await response.text();
  let body: unknown = text;

  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }

  return { status: response.status, body };
}

function errorText(reply: ServiceReply): string {
  const body = reply.body as { error?: string } | string;

  return typeof body === "string" ? body : (body.error ?? JSON.stringify(body));
}

export async function reloadScopeMap(ctx: Ctx): Promise<boolean> {
  try {
    const reply = await serviceCall(ctx, "POST", "/v2/scope-map/reload");

    if (reply.status === 200) {
      return true;
    }

    ctx.err(`scope map saved, but the service refused it (${reply.status}): ${errorText(reply)}`);
  } catch (error) {
    ctx.err(`scope map saved, but not reloaded: ${(error as Error).message}`);
  }

  return false;
}

export function validName(name: string | undefined): name is string {
  return name !== undefined && NAME_PATTERN.test(name);
}

interface Counts {
  scopes: string[];
  records: number;
  proposals: number;
}

async function profileAdd(ctx: Ctx, name: string | undefined): Promise<number> {
  if (!validName(name) || name === DEFAULT_PROFILE) {
    ctx.err(`profile add: give a name of letters, digits, - or _ (not "${DEFAULT_PROFILE}")`);

    return 2;
  }

  const doc = readScopes(ctx);

  if (doc.profiles.includes(name)) {
    ctx.out(`profile ${name} already exists`);

    return 0;
  }

  doc.profiles.push(name);
  writeScopes(ctx, doc);

  if (!(await reloadScopeMap(ctx))) {
    return 1;
  }

  ctx.out(`profile ${name} added, shared scope global-${name}`);

  return 0;
}

function profileList(ctx: Ctx): number {
  const doc = readScopes(ctx);

  for (const profile of [DEFAULT_PROFILE, ...doc.profiles]) {
    const shared = profile === DEFAULT_PROFILE ? "global" : `global-${profile}`;
    const scopes = Object.entries(doc.scopes)
      .filter(([, owner]) => owner === profile)
      .map(([scope]) => scope);

    ctx.out(`${profile}: shared ${shared}, scopes ${scopes.length === 0 ? "none" : scopes.join(", ")}`);
  }

  return 0;
}

async function profileDelete(ctx: Ctx, name: string | undefined, yes: boolean): Promise<number> {
  if (name === DEFAULT_PROFILE || !validName(name)) {
    ctx.err(`profile delete: give the name of a profile other than "${DEFAULT_PROFILE}"`);

    return 2;
  }

  let counts: Counts;

  try {
    const reply = await serviceCall(ctx, "GET", `/v2/profiles/${encodeURIComponent(name)}/count`);

    if (reply.status !== 200) {
      ctx.err(`profile delete: ${errorText(reply)}`);

      return 1;
    }

    counts = reply.body as Counts;
  } catch (error) {
    ctx.err(`profile delete: ${(error as Error).message}`);

    return 1;
  }

  ctx.out(`profile ${name}: ${counts.records} records and ${counts.proposals} proposals in ${counts.scopes.join(", ")}`);

  if (!yes && !(await ctx.confirm(`Delete them permanently?`))) {
    ctx.out("nothing deleted");

    return 0;
  }

  try {
    const reply = await serviceCall(ctx, "DELETE", `/v2/profiles/${encodeURIComponent(name)}`);

    if (reply.status !== 200) {
      ctx.err(`profile delete: ${errorText(reply)}`);

      return 1;
    }
  } catch (error) {
    ctx.err(`profile delete: ${(error as Error).message}`);

    return 1;
  }

  const doc = readScopes(ctx);
  const dropped = Object.entries(doc.scopes).filter(([, owner]) => owner === name).map(([scope]) => scope);
  const deleted = dropped.filter((scope) => counts.scopes.includes(scope));

  doc.profiles = doc.profiles.filter((profile) => profile !== name);
  doc.paths = doc.paths.filter((rule) => !deleted.includes(rule.scope));

  for (const scope of dropped) {
    delete doc.scopes[scope];
  }

  writeScopes(ctx, doc);
  ctx.out(`deleted ${counts.records} records and ${counts.proposals} proposals, profile ${name} removed`);

  return (await reloadScopeMap(ctx)) ? 0 : 1;
}

export async function profile(ctx: Ctx, flags: string[]): Promise<number> {
  const [sub, ...rest] = flags;

  try {
    const { values, positionals } = parseArgs({
      args: rest,
      options: { yes: { type: "boolean", default: false } },
      allowPositionals: true,
    });

    if (sub === "add") {
      return await profileAdd(ctx, positionals[0]);
    }

    if (sub === "list") {
      return profileList(ctx);
    }

    if (sub === "delete") {
      return await profileDelete(ctx, positionals[0], values.yes);
    }
  } catch (error) {
    ctx.err(`profile ${sub}: ${(error as Error).message}`);

    return 1;
  }

  ctx.err("usage: tabumcp profile add <name> | list | delete <name> [--yes]");

  return 2;
}
