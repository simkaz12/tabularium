import { parseArgs } from "node:util";

import { isReservedScope } from "../scopeMap.ts";
import type { Ctx } from "./context.ts";
import { DEFAULT_PROFILE, readScopes, reloadScopeMap, validName, writeScopes } from "./profile.ts";

async function scopeAdd(ctx: Ctx, name: string | undefined, profileName: string | undefined, path: string[]): Promise<number> {
  if (!validName(name) || profileName === undefined) {
    ctx.err("scope add: usage: tabumcp scope add <name> --profile <profile> [--path <glob>]");

    return 2;
  }

  if (isReservedScope(name)) {
    ctx.err(`scope add: "${name}" is reserved for shared scopes (global, global-<profile>), pick another name`);

    return 1;
  }

  if (path.some((match) => match.trim() === "")) {
    ctx.err("scope add: --path needs a non-empty glob");

    return 2;
  }

  const doc = readScopes(ctx);

  if (profileName !== DEFAULT_PROFILE && !doc.profiles.includes(profileName)) {
    ctx.err(`scope add: unknown profile ${profileName}, run: tabumcp profile add ${profileName}`);

    return 1;
  }

  doc.scopes[name] = profileName;

  for (const match of path.map((glob) => glob.trim())) {
    doc.paths = doc.paths.filter((rule) => rule.match !== match);
    doc.paths.push({ match, scope: name });
  }

  writeScopes(ctx, doc);

  if (!(await reloadScopeMap(ctx))) {
    return 1;
  }

  ctx.out(`scope ${name} -> profile ${profileName}`);

  return 0;
}

async function scopeRemove(ctx: Ctx, name: string | undefined): Promise<number> {
  if (name === undefined) {
    ctx.err("scope remove: usage: tabumcp scope remove <name>");

    return 2;
  }

  const doc = readScopes(ctx);

  if (!(name in doc.scopes) && !doc.paths.some((rule) => rule.scope === name)) {
    ctx.err(`scope remove: no scope ${name} in the scope map`);

    return 1;
  }

  delete doc.scopes[name];
  doc.paths = doc.paths.filter((rule) => rule.scope !== name);
  writeScopes(ctx, doc);

  if (!(await reloadScopeMap(ctx))) {
    return 1;
  }

  ctx.out(`scope ${name} removed, it now follows its folder name in the default profile`);

  return 0;
}

function scopeList(ctx: Ctx): number {
  const doc = readScopes(ctx);
  const names = new Set([...Object.keys(doc.scopes), ...doc.paths.map((rule) => rule.scope)]);

  if (names.size === 0) {
    ctx.out("no scopes mapped, every folder uses its own name in the default profile");

    return 0;
  }

  for (const name of names) {
    const paths = doc.paths.filter((rule) => rule.scope === name).map((rule) => rule.match);

    ctx.out(`${name}: profile ${doc.scopes[name] ?? DEFAULT_PROFILE}${paths.length === 0 ? "" : `, paths ${paths.join(", ")}`}`);
  }

  return 0;
}

export async function scope(ctx: Ctx, flags: string[]): Promise<number> {
  const [sub, ...rest] = flags;

  try {
    const { values, positionals } = parseArgs({
      args: rest,
      options: { profile: { type: "string" }, path: { type: "string", multiple: true, default: [] } },
      allowPositionals: true,
    });

    if (sub === "add") {
      return await scopeAdd(ctx, positionals[0], values.profile, values.path);
    }

    if (sub === "remove") {
      return await scopeRemove(ctx, positionals[0]);
    }

    if (sub === "list") {
      return scopeList(ctx);
    }
  } catch (error) {
    ctx.err(`scope ${sub}: ${(error as Error).message}`);

    return 1;
  }

  ctx.err("usage: tabumcp scope add <name> --profile <p> [--path <glob>] | list | remove <name>");

  return 2;
}
