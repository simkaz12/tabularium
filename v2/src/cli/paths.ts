import { join } from "node:path";

import type { Ctx } from "./context.ts";
import { SUPPORTED_MESSAGE } from "./context.ts";

export const SERVICE_NAME = "tabumcp";
export const LAUNCHD_LABEL = "io.github.simkaz12.tabumcp";
export const DEFAULT_PORT = 8789;
export const EMBED_MODEL = "nomic-embed-text";
export const MIN_NODE_MAJOR = 22;

export interface CliPaths {
  configDir: string;
  envFile: string;
  tokenFile: string;
  scopesFile: string;
  dataDir: string;
  storeDir: string;
  appDir: string;
  currentLink: string;
  unitFile: string;
  plistFile: string;
  skillDir: string;
  claudeDir: string;
  claudeMd: string;
  codexSkillDir: string;
  codexConfig: string;
  codexAgentsMd: string;
  codexDir: string;
  logFile: string;
  binDir: string;
  binLink: string;
}

type PathCtx = Pick<Ctx, "home" | "env" | "platform">;

export function resolvePaths(ctx: PathCtx): CliPaths {
  const configDir = join(ctx.home, ".config", "tabumcp");
  const dataDir = dataDirFor(ctx);
  const appDir = join(dataDir, "app");

  return {
    configDir,
    envFile: join(configDir, "tabumcp.env"),
    tokenFile: join(configDir, "token"),
    scopesFile: join(configDir, "scopes.json"),
    dataDir,
    storeDir: join(dataDir, "store"),
    appDir,
    currentLink: join(appDir, "current"),
    unitFile: join(ctx.home, ".config", "systemd", "user", `${SERVICE_NAME}.service`),
    plistFile: join(ctx.home, "Library", "LaunchAgents", `${LAUNCHD_LABEL}.plist`),
    skillDir: join(ctx.home, ".claude", "skills", "tabumcp"),
    claudeDir: join(ctx.home, ".claude"),
    claudeMd: join(ctx.home, ".claude", "CLAUDE.md"),
    codexSkillDir: join(ctx.home, ".agents", "skills", "tabumcp"),
    codexConfig: join(ctx.home, ".codex", "config.toml"),
    codexAgentsMd: join(ctx.home, ".codex", "AGENTS.md"),
    codexDir: join(ctx.home, ".codex"),
    logFile: join(dataDir, "tabumcp.log"),
    binDir: join(ctx.home, ".local", "bin"),
    binLink: join(ctx.home, ".local", "bin", "tabumcp"),
  };
}

function dataDirFor(ctx: PathCtx): string {
  if (ctx.env.XDG_DATA_HOME) {
    return join(ctx.env.XDG_DATA_HOME, "tabumcp");
  }

  if (ctx.platform === "darwin") {
    return join(ctx.home, "Library", "Application Support", "tabumcp");
  }

  return join(ctx.home, ".local", "share", "tabumcp");
}

export function platformProblem(ctx: Pick<Ctx, "platform" | "arch">): string | null {
  const linux = ctx.platform === "linux" && (ctx.arch === "x64" || ctx.arch === "arm64");
  const mac = ctx.platform === "darwin" && ctx.arch === "arm64";

  if (linux || mac) {
    return null;
  }

  return `unsupported platform ${ctx.platform}/${ctx.arch}. ${SUPPORTED_MESSAGE}`;
}

export function nodeProblem(ctx: Pick<Ctx, "nodeVersion">): string | null {
  const major = Number(ctx.nodeVersion.split(".")[0]);

  if (Number.isInteger(major) && major >= MIN_NODE_MAJOR) {
    return null;
  }

  return `Node ${MIN_NODE_MAJOR} or newer is required, found ${ctx.nodeVersion}`;
}

export function mcpUrl(ctx: Pick<Ctx, "env">): string {
  return `http://127.0.0.1:${portFrom(ctx.env)}/mcp`;
}

export function portFrom(env: NodeJS.ProcessEnv): number {
  const port = Number(env.TABULARIUM_PORT);

  return Number.isInteger(port) && port > 0 ? port : DEFAULT_PORT;
}
