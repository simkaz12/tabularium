import { chmodSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

import type { Ctx, RunResult } from "./context.ts";
import { LAUNCHD_LABEL, SERVICE_NAME, resolvePaths } from "./paths.ts";

export function renderTemplate(template: string, values: Record<string, string>): string {
  return template.replace(/\{\{([A-Z_]+)\}\}/g, (match, key: string) => values[key] ?? match);
}

export function systemdEscape(value: string): string {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/%/g, "%%");
}

function xmlEscape(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function templateValues(ctx: Ctx): Record<string, string> {
  const paths = resolvePaths(ctx);
  const escape = ctx.platform === "darwin" ? xmlEscape : systemdEscape;

  return {
    HOME: escape(ctx.home),
    CURRENT: escape(paths.currentLink),
    CURRENT_EXEC: escape(paths.currentLink).replace(/\$/g, "$$$$"),
    NODE_DIR: escape(dirname(ctx.nodePath)),
    LOG: escape(paths.logFile),
  };
}

export function renderServiceFile(ctx: Ctx): string {
  const name = ctx.platform === "darwin" ? `${LAUNCHD_LABEL}.plist` : `${SERVICE_NAME}.service`;
  const template = readFileSync(join(ctx.packageRoot, "templates", name), "utf8");

  return renderTemplate(template, templateValues(ctx));
}

export function serviceFilePath(ctx: Ctx): string {
  const paths = resolvePaths(ctx);

  return ctx.platform === "darwin" ? paths.plistFile : paths.unitFile;
}

function guiTarget(ctx: Ctx): string {
  return `gui/${ctx.uid}`;
}

async function must(ctx: Ctx, command: string, args: string[]): Promise<RunResult> {
  const result = await ctx.run(command, args, ctx.env);

  if (result.code !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed: ${result.stderr.trim() || `exit ${result.code}`}`);
  }

  return result;
}

export async function installService(ctx: Ctx): Promise<string> {
  const file = serviceFilePath(ctx);

  mkdirSync(dirname(file), { recursive: true });
  mkdirSync(dirname(resolvePaths(ctx).logFile), { recursive: true });
  writeFileSync(file, renderServiceFile(ctx));
  chmodSync(file, 0o644);

  if (ctx.platform === "darwin") {
    await ctx.run("launchctl", ["bootout", `${guiTarget(ctx)}/${LAUNCHD_LABEL}`], ctx.env);
    await must(ctx, "launchctl", ["bootstrap", guiTarget(ctx), file]);

    return file;
  }

  await must(ctx, "systemctl", ["--user", "daemon-reload"]);
  await must(ctx, "systemctl", ["--user", "enable", "--now", SERVICE_NAME]);

  return file;
}

export async function restartService(ctx: Ctx): Promise<void> {
  if (ctx.platform === "darwin") {
    await must(ctx, "launchctl", ["kickstart", "-k", `${guiTarget(ctx)}/${LAUNCHD_LABEL}`]);

    return;
  }

  await must(ctx, "systemctl", ["--user", "restart", SERVICE_NAME]);
}

function manualStopCommand(ctx: Ctx): string {
  return ctx.platform === "darwin"
    ? `launchctl bootout ${guiTarget(ctx)}/${LAUNCHD_LABEL}`
    : `systemctl --user disable --now ${SERVICE_NAME}`;
}

export async function removeService(ctx: Ctx): Promise<void> {
  const file = serviceFilePath(ctx);

  if (ctx.platform === "darwin") {
    await ctx.run("launchctl", ["bootout", `${guiTarget(ctx)}/${LAUNCHD_LABEL}`], ctx.env);
  } else {
    await ctx.run("systemctl", ["--user", "disable", "--now", SERVICE_NAME], ctx.env);
  }

  if ((await serviceState(ctx)) !== "stopped") {
    throw new Error(
      `the ${SERVICE_NAME} service is running or its state could not be confirmed, nothing was removed. ` +
        `Stop it by hand with: ${manualStopCommand(ctx)}`,
    );
  }

  rmSync(file, { force: true });

  if (ctx.platform === "darwin") {
    return;
  }

  await ctx.run("systemctl", ["--user", "daemon-reload"], ctx.env);
}

export type ServiceState = "active" | "stopped" | "unknown";

const LAUNCHD_NOT_FOUND_CODE = 113;
const SYSTEMD_STOPPED_STATES = ["inactive", "failed"];

export async function serviceState(ctx: Ctx): Promise<ServiceState> {
  if (ctx.platform === "darwin") {
    const result = await ctx.run("launchctl", ["print", `${guiTarget(ctx)}/${LAUNCHD_LABEL}`], ctx.env);

    if (result.code === 0) {
      return "active";
    }

    const notFound = result.code === LAUNCHD_NOT_FOUND_CODE || /could not find service/i.test(result.stderr);

    return notFound ? "stopped" : "unknown";
  }

  const result = await ctx.run("systemctl", ["--user", "is-active", SERVICE_NAME], ctx.env);
  const state = result.stdout.trim();

  if (result.code === 0 && state === "active") {
    return "active";
  }

  return result.code !== 0 && SYSTEMD_STOPPED_STATES.includes(state) ? "stopped" : "unknown";
}

export async function serviceActive(ctx: Ctx): Promise<boolean> {
  return (await serviceState(ctx)) === "active";
}

export function serviceInstalled(ctx: Ctx): boolean {
  return existsSync(serviceFilePath(ctx));
}
