import { AuthenticationError, PermissionDeniedError } from "@typesafe-ai/sdk";
import { describe, expect, it } from "vitest";

import type { JevClient } from "../domain.ts";
import { isAuthError, withAuthWarning } from "./authWarning.ts";

function failing(error: unknown): JevClient {
  return {
    systemOne: async () => {
      throw error;
    },
  };
}

const REQUEST = { model: "m", state: {}, questions: {} } as never;

describe("withAuthWarning", () => {
  it("recognises 401 and 403 statuses and the SDK auth error classes", () => {
    expect(isAuthError({ status: 401 })).toBe(true);
    expect(isAuthError({ status: 403 })).toBe(true);
    expect(isAuthError(Object.create(AuthenticationError.prototype))).toBe(true);
    expect(isAuthError(Object.create(PermissionDeniedError.prototype))).toBe(true);
    expect(isAuthError({ status: 500 })).toBe(false);
    expect(isAuthError(new Error("timeout"))).toBe(false);
  });

  it("logs one warning naming the cause, rethrows, and stays quiet for an hour", async () => {
    const lines: string[] = [];
    let clock = 0;
    const client = withAuthWarning(failing({ status: 401 }), { now: () => clock, warn: (line) => lines.push(line) });

    await expect(client.systemOne(REQUEST)).rejects.toMatchObject({ status: 401 });
    await expect(client.systemOne(REQUEST)).rejects.toMatchObject({ status: 401 });

    expect(lines).toHaveLength(1);
    expect(lines[0]).toContain("TypeSafe rejected the API key");

    clock = 60 * 60 * 1000;
    await expect(client.systemOne(REQUEST)).rejects.toBeDefined();

    expect(lines).toHaveLength(2);
  });

  it("does not warn for other failures and passes successes through", async () => {
    const lines: string[] = [];
    const broken = withAuthWarning(failing(Object.assign(new Error("boom"), { status: 500 })), {
      warn: (line) => lines.push(line),
    });
    const fine = withAuthWarning({ systemOne: async () => ({ ok: true }) as never }, { warn: (line) => lines.push(line) });

    await expect(broken.systemOne(REQUEST)).rejects.toThrow("boom");
    await expect(fine.systemOne(REQUEST)).resolves.toEqual({ ok: true });

    expect(lines).toEqual([]);
  });
});
