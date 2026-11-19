import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { CwdError, resolveCwd } from "./gitContext.ts";
import { loadScopeMap } from "./scopeMap.ts";

const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function tempDir(): string {
  const dir = realpathSync(mkdtempSync(join(tmpdir(), "gitctx-")));
  dirs.push(dir);

  return dir;
}

function git(cwd: string, ...args: string[]): void {
  execFileSync(
    "git",
    ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", ...args],
    { cwd, stdio: "ignore" },
  );
}

function makeRepo(name: string): string {
  const repo = join(tempDir(), name);
  mkdirSync(repo);
  git(repo, "init", "-b", "main");
  writeFileSync(join(repo, "a.txt"), "a");
  git(repo, "add", "a.txt");
  git(repo, "commit", "-m", "init");

  return repo;
}

function mapFor(content: object) {
  const file = join(tempDir(), "scopes.json");
  writeFileSync(file, JSON.stringify(content));

  return loadScopeMap(file);
}

const EMPTY = mapFor({});

describe("resolveCwd", () => {
  it("resolves a repo to its folder name and branch", async () => {
    const repo = makeRepo("alpha");
    git(repo, "checkout", "-b", "TASK-1");

    const context = await resolveCwd(repo, EMPTY);

    expect(context).toEqual({ scope: "alpha", projectId: "alpha", branch: "TASK-1", profile: "default" });
  });

  it("resolves a worktree to the main checkout scope with its own branch", async () => {
    const repo = makeRepo("alpha");
    const worktree = join(tempDir(), "wt");
    git(repo, "worktree", "add", "-b", "feature", worktree);

    const context = await resolveCwd(worktree, EMPTY);

    expect(context.scope).toBe("alpha");
    expect(context.branch).toBe("feature");
  });

  it("resolves a subfolder of a repo to the repo scope", async () => {
    const repo = makeRepo("alpha");
    mkdirSync(join(repo, "sub"));

    const context = await resolveCwd(join(repo, "sub"), EMPTY);

    expect(context.scope).toBe("alpha");
  });

  it("lets a paths match beat the folder name and carries the profile", async () => {
    const repo = makeRepo("acme-api");
    const map = mapFor({
      profiles: ["work"],
      scopes: { acme: "work" },
      paths: [{ match: join(dirname(repo), "acme-*"), scope: "acme" }],
    });

    const context = await resolveCwd(repo, map);

    expect(context.scope).toBe("acme");
    expect(context.profile).toBe("work");
  });

  it("uses the folder name and an empty branch outside git", async () => {
    const plain = join(tempDir(), "plain");
    mkdirSync(plain);

    const context = await resolveCwd(plain, EMPTY);

    expect(context).toEqual({ scope: "plain", projectId: "plain", branch: "", profile: "default" });
  });

  it("refuses a checkout whose folder name is a reserved shared scope and points at scope add", async () => {
    for (const name of ["global", "global-work"]) {
      const repo = makeRepo(name);
      const failure = await resolveCwd(repo, EMPTY).catch((error: unknown) => error);

      expect(failure).toBeInstanceOf(CwdError);
      expect((failure as Error).message).toContain(`tabumcp scope add <name> --path ${repo}`);
    }
  });

  it("accepts a reserved folder name once a path rule maps it", async () => {
    const repo = makeRepo("global-work");
    const map = mapFor({ paths: [{ match: repo, scope: "workrepo" }] });

    expect((await resolveCwd(repo, map)).scope).toBe("workrepo");
  });

  it("throws CwdError for a missing path", async () => {
    await expect(resolveCwd(join(tempDir(), "nope"), EMPTY)).rejects.toBeInstanceOf(CwdError);
  });

  it("throws CwdError for an absent cwd", async () => {
    await expect(resolveCwd(undefined, EMPTY)).rejects.toThrow("pass the working folder");
  });
});
