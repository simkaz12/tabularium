import { describe, expect, it } from "vitest";

import { makeHarness } from "./harness.ts";
import { removeService, renderServiceFile, serviceState } from "./service.ts";
import type { RunResult } from "./context.ts";

function statusStub(status: RunResult) {
  const calls: string[] = [];

  return {
    calls,
    run: async (command: string, args: string[]): Promise<RunResult> => {
      calls.push(`${command} ${args.join(" ")}`);

      return args[0] === "--user" && args[1] === "is-active" || command === "launchctl" && args[0] === "print"
        ? status
        : { code: 0, stdout: "", stderr: "" };
    },
  };
}

describe("serviceState", () => {
  it.each([
    ["inactive", { code: 3, stdout: "inactive\n", stderr: "" }, "stopped"],
    ["failed", { code: 3, stdout: "failed\n", stderr: "" }, "stopped"],
    ["active", { code: 0, stdout: "active\n", stderr: "" }, "active"],
    ["bus error", { code: 1, stdout: "", stderr: "Failed to connect to bus" }, "unknown"],
    ["activating", { code: 3, stdout: "activating\n", stderr: "" }, "unknown"],
    ["empty answer with exit 0", { code: 0, stdout: "", stderr: "" }, "unknown"],
  ])("systemd %s", async (_name, result, expected) => {
    const stub = statusStub(result);
    const h = makeHarness({ run: stub.run });

    expect(await serviceState(h.ctx)).toBe(expected);
  });

  it.each([
    ["loaded", { code: 0, stdout: "state = running", stderr: "" }, "active"],
    ["not found by code", { code: 113, stdout: "", stderr: "" }, "stopped"],
    ["not found by text", { code: 1, stdout: "", stderr: "Could not find service" }, "stopped"],
    ["other failure", { code: 1, stdout: "", stderr: "launchctl failed (ETIMEDOUT)" }, "unknown"],
  ])("launchd %s", async (_name, result, expected) => {
    const stub = statusStub(result);
    const h = makeHarness({ platform: "darwin", arch: "arm64", run: stub.run });

    expect(await serviceState(h.ctx)).toBe(expected);
  });
});

describe("removeService with an unconfirmed state", () => {
  it.each([
    ["linux", { code: 1, stdout: "", stderr: "Failed to connect to bus" }],
    ["darwin", { code: 1, stdout: "", stderr: "launchctl failed (ETIMEDOUT)" }],
  ] as const)("aborts on %s and removes nothing", async (platform, result) => {
    const stub = statusStub(result);
    const h = makeHarness({ platform, arch: platform === "darwin" ? "arm64" : "x64", run: stub.run });

    await expect(removeService(h.ctx)).rejects.toThrow("could not be confirmed");
    expect(stub.calls.some((call) => call.includes("daemon-reload"))).toBe(false);
  });
});

describe("renderServiceFile", () => {
  it("quotes ExecStart and Environment in the systemd unit when the path has spaces", () => {
    const h = makeHarness({ home: "/home/a b", nodePath: "/opt/my node/bin/node" });
    const unit = renderServiceFile(h.ctx);

    expect(unit).toContain('ExecStart="/home/a b/.local/share/tabumcp/app/current/node_modules/.bin/tabumcp" serve');
    expect(unit).toContain('Environment="PATH=/opt/my node/bin:/usr/local/bin:/usr/bin:/bin"');
  });

  it("escapes quotes, percent signs and dollars in ExecStart", () => {
    const h = makeHarness({ home: '/home/a"b%c$d' });
    const unit = renderServiceFile(h.ctx);

    expect(unit).toContain('ExecStart="/home/a\\"b%%c$$d/');
  });

  it("keeps each plist ProgramArguments entry its own string with a spaced path", () => {
    const h = makeHarness({ platform: "darwin", arch: "arm64", home: "/Users/a b & c" });
    const plist = renderServiceFile(h.ctx);

    expect(plist).toContain("<string>/Users/a b &amp; c/Library/Application Support/tabumcp/app/current/node_modules/.bin/tabumcp</string>\n    <string>serve</string>");
  });
});
