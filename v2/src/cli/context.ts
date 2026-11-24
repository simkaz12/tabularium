import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { dirname } from "node:path";
import { createInterface } from "node:readline/promises";
import { Writable } from "node:stream";
import { fileURLToPath } from "node:url";

export interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
}

export type Runner = (command: string, args: string[], env: NodeJS.ProcessEnv) => Promise<RunResult>;

export interface Ctx {
  platform: NodeJS.Platform;
  arch: string;
  home: string;
  env: NodeJS.ProcessEnv;
  nodeVersion: string;
  nodePath: string;
  uid: number;
  version: string;
  packageRoot: string;
  run: Runner;
  fetch: typeof fetch;
  out: (line: string) => void;
  err: (line: string) => void;
  interactive: boolean;
  confirm: (question: string) => Promise<boolean>;
  ask: (question: string) => Promise<string>;
  secret: (question: string) => Promise<string>;
  sleep: (ms: number) => Promise<void>;
  healthAttempts: number;
}

export const SUPPORTED_MESSAGE =
  "tabumcp supports Linux (x64, arm64) and macOS on Apple Silicon. On Windows, use WSL2: it counts as Linux.";

export const execRunner: Runner = (command, args, env) =>
  new Promise((resolve) => {
    execFile(command, args, { env, timeout: 600_000, maxBuffer: 16 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error === null) {
        resolve({ code: 0, stdout, stderr });

        return;
      }

      const code = typeof error.code === "number" ? error.code : 1;

      resolve({ code, stdout, stderr: stderr || `${command} failed (${error.code ?? error.signal ?? "error"})` });
    });
  });

async function promptConfirm(question: string): Promise<boolean> {
  if (!process.stdin.isTTY) {
    return false;
  }

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  try {
    const answer = await rl.question(`${question} [y/N] `);

    return /^y(es)?$/i.test(answer.trim());
  } finally {
    rl.close();
  }
}

async function promptAsk(question: string): Promise<string> {
  if (!process.stdin.isTTY) {
    return "";
  }

  const rl = createInterface({ input: process.stdin, output: process.stdout });

  try {
    return (await rl.question(question)).trim();
  } finally {
    rl.close();
  }
}

async function promptSecret(question: string): Promise<string> {
  if (!process.stdin.isTTY) {
    return "";
  }

  let silent = false;
  const muted = new Writable({
    write(chunk, encoding, done) {
      if (!silent) {
        process.stdout.write(chunk, encoding);
      }

      done();
    },
  });
  const rl = createInterface({ input: process.stdin, output: muted, terminal: true });

  process.stdout.write(question);
  silent = true;

  try {
    return (await rl.question("")).trim();
  } finally {
    silent = false;
    rl.close();
    process.stdout.write("\n");
  }
}

export function packageRootFrom(moduleUrl: string): string {
  return dirname(dirname(dirname(fileURLToPath(moduleUrl))));
}

export function defaultContext(moduleUrl: string, version: string): Ctx {
  return {
    platform: process.platform,
    arch: process.arch,
    home: homedir(),
    env: process.env,
    nodeVersion: process.versions.node,
    nodePath: process.execPath,
    uid: process.getuid?.() ?? 0,
    version,
    packageRoot: packageRootFrom(moduleUrl),
    run: execRunner,
    fetch,
    out: (line) => console.log(line),
    err: (line) => console.error(line),
    interactive: process.stdin.isTTY === true,
    confirm: promptConfirm,
    ask: promptAsk,
    secret: promptSecret,
    sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    healthAttempts: 20,
  };
}
