import { readFileSync } from "node:fs";
import { homedir } from "node:os";

export const DEFAULT_PROFILE = "default";
export const GLOBAL_SCOPE = "global";

interface PathRule {
  pattern: RegExp;
  scope: string;
}

interface ScopesFile {
  profiles?: unknown;
  scopes?: unknown;
  paths?: unknown;
}

export class ScopeMap {
  readonly profiles: string[];
  private readonly scopeProfiles: Map<string, string>;
  private readonly rules: PathRule[];

  constructor(profiles: string[], scopeProfiles: Map<string, string>, rules: PathRule[]) {
    this.profiles = profiles;
    this.scopeProfiles = scopeProfiles;
    this.rules = rules;
  }

  scopeFor(mainCheckoutPath: string, folderName: string): string {
    for (const rule of this.rules) {
      if (rule.pattern.test(mainCheckoutPath)) {
        return rule.scope;
      }
    }

    return folderName.trim();
  }

  profileOf(scope: string): string {
    const mapped = this.scopeProfiles.get(scope);

    if (mapped !== undefined) {
      return mapped;
    }

    const owner = this.profiles.find((profile) => scope === sharedScopeOfProfile(profile));

    return owner ?? DEFAULT_PROFILE;
  }

  sharedScopeOf(scope: string): string {
    return sharedScopeOfProfile(this.profileOf(scope));
  }

  scopesOf(profile: string): string[] {
    const scopes = [...this.scopeProfiles.entries()]
      .filter(([, owner]) => owner === profile)
      .map(([scope]) => scope);

    return [...scopes, sharedScopeOfProfile(profile)];
  }
}

export function sharedScopeOfProfile(profile: string): string {
  return profile === DEFAULT_PROFILE ? GLOBAL_SCOPE : `${GLOBAL_SCOPE}-${profile}`;
}

export function isReservedScope(name: string): boolean {
  return name === GLOBAL_SCOPE || name.startsWith(`${GLOBAL_SCOPE}-`);
}

export function emptyScopeMap(): ScopeMap {
  return new ScopeMap([], new Map(), []);
}

export function loadScopeMap(file: string): ScopeMap {
  let raw: string;

  try {
    raw = readFileSync(file, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return emptyScopeMap();
    }

    throw new Error(`cannot read scope map ${file}: ${(error as Error).message}`);
  }

  let parsed: ScopesFile;

  try {
    parsed = JSON.parse(raw) as ScopesFile;
  } catch (error) {
    throw new Error(`invalid JSON in scope map ${file}: ${(error as Error).message}`);
  }

  return buildScopeMap(parsed, file);
}

function buildScopeMap(parsed: ScopesFile, file: string): ScopeMap {
  const profiles = readStringList(parsed.profiles, "profiles", file);
  const scopeProfiles = readScopeProfiles(parsed.scopes, profiles, file);
  const rules = readPathRules(parsed.paths, file);

  return new ScopeMap(profiles, scopeProfiles, rules);
}

function readStringList(value: unknown, key: string, file: string): string[] {
  if (value === undefined) {
    return [];
  }

  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.trim() === "")) {
    throw new Error(`scope map ${file}: "${key}" must be a list of names`);
  }

  return (value as string[]).map((item) => item.trim());
}

function readScopeProfiles(value: unknown, profiles: string[], file: string): Map<string, string> {
  const result = new Map<string, string>();

  if (value === undefined) {
    return result;
  }

  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`scope map ${file}: "scopes" must be an object`);
  }

  for (const [rawScope, rawProfile] of Object.entries(value)) {
    const scope = rawScope.trim();
    const profile = typeof rawProfile === "string" ? rawProfile.trim() : rawProfile;

    assertMappable(scope, file);

    if (typeof profile !== "string" || (profile !== DEFAULT_PROFILE && !profiles.includes(profile))) {
      throw new Error(`scope map ${file}: scope "${scope}" names unknown profile ${JSON.stringify(rawProfile)}`);
    }

    result.set(scope, profile);
  }

  return result;
}

function assertMappable(scope: string, file: string): void {
  if (scope === "") {
    throw new Error(`scope map ${file}: a scope name cannot be empty`);
  }

  if (isReservedScope(scope)) {
    throw new Error(`scope map ${file}: "${scope}" is a reserved shared scope name and cannot be mapped`);
  }
}

function readPathRules(value: unknown, file: string): PathRule[] {
  if (value === undefined) {
    return [];
  }

  if (!Array.isArray(value)) {
    throw new Error(`scope map ${file}: "paths" must be a list`);
  }

  return value.map((entry) => {
    const rawMatch = (entry as { match?: unknown })?.match;
    const rawScope = (entry as { scope?: unknown })?.scope;

    if (typeof rawMatch !== "string" || typeof rawScope !== "string" || rawMatch.trim() === "" || rawScope.trim() === "") {
      throw new Error(`scope map ${file}: each path needs a "match" and a "scope"`);
    }

    const scope = rawScope.trim();

    assertMappable(scope, file);

    return { pattern: globToRegExp(expandHome(rawMatch.trim())), scope };
  });
}

function expandHome(pattern: string): string {
  if (pattern === "~") {
    return homedir();
  }

  return pattern.startsWith("~/") ? homedir() + pattern.slice(1) : pattern;
}

function globToRegExp(glob: string): RegExp {
  let source = "";

  for (let index = 0; index < glob.length; index += 1) {
    const char = glob[index];

    if (char === "*" && glob[index + 1] === "*") {
      source += ".*";
      index += 1;
    } else if (char === "*") {
      source += "[^/]*";
    } else if (char === "?") {
      source += "[^/]";
    } else {
      source += char.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }

  return new RegExp(`^${source}$`);
}
