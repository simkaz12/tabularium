import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HttpError } from "./domain.ts";
import { SettingsService, StoreCleanerState } from "./settings.ts";
import { openStore } from "./store.ts";

const dirs: string[] = [];

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function tmp(): string {
  const dir = mkdtempSync(join(tmpdir(), "tabularium-v2-settings-"));
  dirs.push(dir);

  return dir;
}

const DAY_MS = 24 * 60 * 60 * 1000;

describe("settings", () => {
  it("defaults the confidence floor to 0.7", async () => {
    const settings = new SettingsService(await openStore(tmp(), 8));

    expect(await settings.getSettings()).toEqual({ cleanerConfidenceFloor: 0.7 });
  });

  it("persists an update across reopening the store", async () => {
    const path = tmp();
    const settings = new SettingsService(await openStore(path, 8));

    expect(await settings.updateSettings({ cleanerConfidenceFloor: 0.8 })).toEqual({ cleanerConfidenceFloor: 0.8 });

    const reopened = new SettingsService(await openStore(path, 8));

    expect(await reopened.getSettings()).toEqual({ cleanerConfidenceFloor: 0.8 });

    await reopened.updateSettings({ cleanerConfidenceFloor: 0 });

    expect(await reopened.getSettings()).toEqual({ cleanerConfidenceFloor: 0 });
  });

  it("rejects out of range, non-number, unknown and empty patches", async () => {
    const settings = new SettingsService(await openStore(tmp(), 8));
    const bad: unknown[] = [
      { cleanerConfidenceFloor: 1.5 },
      { cleanerConfidenceFloor: -0.1 },
      { cleanerConfidenceFloor: "x" },
      { cleanerConfidenceFloor: null },
      { cleanerConfidenceFloor: 0.5, extra: 1 },
      { other: 1 },
      {},
      undefined,
      [],
      "x",
    ];

    for (const patch of bad) {
      await expect(settings.updateSettings(patch)).rejects.toMatchObject({ status: 400 });
      await expect(settings.updateSettings(patch)).rejects.toBeInstanceOf(HttpError);
    }

    expect(await settings.getSettings()).toEqual({ cleanerConfidenceFloor: 0.7 });
  });
});

describe("cleaner state", () => {
  const a = { id: "a", updatedAt: "2026-11-01T10:00:00.000Z" };
  const b = { id: "b", updatedAt: "2026-11-01T11:00:00.000Z" };

  it("remembers a pair for 90 days regardless of argument order", async () => {
    const clock = { value: "2026-11-01T12:00:00.000Z" };
    const state = new StoreCleanerState(await openStore(tmp(), 8), () => clock.value);

    expect(await state.isPairRemembered(a, b)).toBe(false);

    await state.rememberPair(a, b);

    expect(await state.isPairRemembered(a, b)).toBe(true);
    expect(await state.isPairRemembered(b, a)).toBe(true);

    clock.value = new Date(Date.parse("2026-11-01T12:00:00.000Z") + 89 * DAY_MS).toISOString();
    expect(await state.isPairRemembered(a, b)).toBe(true);

    clock.value = new Date(Date.parse("2026-11-01T12:00:00.000Z") + 91 * DAY_MS).toISOString();
    expect(await state.isPairRemembered(a, b)).toBe(false);
  });

  it("forgets a pair when either updatedAt changes", async () => {
    const state = new StoreCleanerState(await openStore(tmp(), 8), () => "2026-11-01T12:00:00.000Z");

    await state.rememberPair(a, b);

    expect(await state.isPairRemembered({ ...a, updatedAt: "2026-11-02T00:00:00.000Z" }, b)).toBe(false);
    expect(await state.isPairRemembered(a, { ...b, updatedAt: "2026-11-02T00:00:00.000Z" })).toBe(false);
    expect(await state.isPairRemembered(a, b)).toBe(true);
  });

  it("remembers a record until it expires or changes", async () => {
    const clock = { value: "2026-11-01T12:00:00.000Z" };
    const state = new StoreCleanerState(await openStore(tmp(), 8), () => clock.value);

    await state.rememberRecord(a);

    expect(await state.isRecordRemembered(a)).toBe(true);
    expect(await state.isRecordRemembered({ ...a, updatedAt: "2026-11-02T00:00:00.000Z" })).toBe(false);
    expect(await state.isRecordRemembered(b)).toBe(false);

    clock.value = new Date(Date.parse(clock.value) + 91 * DAY_MS).toISOString();
    expect(await state.isRecordRemembered(a)).toBe(false);
  });

  it("keeps lastRunAt across reopening the store", async () => {
    const path = tmp();
    const state = new StoreCleanerState(await openStore(path, 8));

    expect(await state.getLastRunAt()).toBeNull();

    await state.setLastRunAt("2026-11-01T03:00:00.000Z");
    await state.setLastRunAt("2026-11-08T03:00:00.000Z");

    const reopened = new StoreCleanerState(await openStore(path, 8));

    expect(await reopened.getLastRunAt()).toBe("2026-11-08T03:00:00.000Z");
  });

  it("keeps a separate last run per state key", async () => {
    const state = new StoreCleanerState(await openStore(tmp(), 8));

    await state.setLastRunAt("2026-11-08T03:00:00.000Z");
    await state.setLastRunAt("2026-11-09T04:00:00.000Z", "meta:lastExpiryAt");

    expect(await state.getLastRunAt()).toBe("2026-11-08T03:00:00.000Z");
    expect(await state.getLastRunAt("meta:lastExpiryAt")).toBe("2026-11-09T04:00:00.000Z");
    expect(await state.getLastRunAt("meta:other")).toBeNull();
  });
});
