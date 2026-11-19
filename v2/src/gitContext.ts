import { execFile } from "node:child_process";
import { statSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { promisify } from "node:util";

import { isReservedScope, type ScopeMap } from "./scopeMap.ts";

const run = promisify(execFile);
const GIT_TIMEOUT_MS = 5000;

export const CWD_REQUIRED = "cwd is required: pass the working folder of the session";

export class CwdError extends Error {}

export interface CwdContext {
  scope: string;
  projectId: string;
  branch: string;
  profile: string;
}

export async function resolveCwd(cwd: unknown, scopeMap: ScopeMap): Promise<CwdContext> {
  if (typeof cwd !== "string" || cwd.trim() === "") {
    throw new CwdError(CWD_REQUIRED);
  }

  const folder = resolve(cwd);

  if (!isDirectory(folder)) {
    throw new CwdError(`cwd ${folder} is not an existing folder: pass the working folder of the session`);
  }

  const mainCheckout = (await mainCheckoutOf(folder)) ?? folder;
  const branch = await currentBranch(folder);
  const scope = scopeMap.scopeFor(mainCheckout, basename(mainCheckout));

  if (scope === "" || isReservedScope(scope)) {
    throw new CwdError(
      `the folder name of ${mainCheckout} cannot be used as a scope (shared scope names global and global-<profile> are reserved). ` +
        `Map it with: tabumcp scope add <name> --path ${mainCheckout}`,
    );
  }

  return {
    scope,
    projectId: scope,
    branch,
    profile: scopeMap.profileOf(scope),
  };
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

async function mainCheckoutOf(folder: string): Promise<string | null> {
  const output = await git(folder, ["rev-parse", "--path-format=absolute", "--git-common-dir"]);

  return output === null || output === "" ? null : dirname(output);
}

async function currentBranch(folder: string): Promise<string> {
  return (await git(folder, ["branch", "--show-current"])) ?? "";
}

async function git(folder: string, args: string[]): Promise<string | null> {
  try {
    const { stdout } = await run("git", ["-C", folder, ...args], { timeout: GIT_TIMEOUT_MS });

    return stdout.trim();
  } catch {
    return null;
  }
}
