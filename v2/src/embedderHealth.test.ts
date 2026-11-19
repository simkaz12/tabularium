import { afterEach, describe, expect, it, vi } from "vitest";

import { EmbedderHealth } from "./embedderBuild.ts";
import type { EmbedderCheck } from "./embedderBuild.ts";

const MATCH: EmbedderCheck = {
  status: "match",
  stamp: { model: "m", digest: "d", ollamaVersion: "1", stampedAt: "2026-01-01T00:00:00.000Z" },
};

describe("EmbedderHealth", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("recovers on its own once the embedder becomes reachable", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const results: EmbedderCheck[] = [
      { status: "unreachable", error: "fetch failed" },
      { status: "unreachable", error: "fetch failed" },
      MATCH,
    ];
    const run = vi.fn(async () => results.shift() ?? MATCH);
    const health = new EmbedderHealth();

    await health.refresh(run, 30_000);
    expect(health.get().status).toBe("unreachable");

    await vi.advanceTimersByTimeAsync(30_000);
    expect(health.get().status).toBe("unreachable");

    await vi.advanceTimersByTimeAsync(30_000);
    expect(health.get().status).toBe("match");

    await vi.advanceTimersByTimeAsync(120_000);
    expect(run).toHaveBeenCalledTimes(3);
  });

  it("does not retry when the first check is reachable", async () => {
    vi.useFakeTimers();

    const run = vi.fn(async () => MATCH);
    const health = new EmbedderHealth();

    await health.refresh(run, 30_000);
    await vi.advanceTimersByTimeAsync(120_000);

    expect(run).toHaveBeenCalledTimes(1);
  });

  it("warns once while the embedder stays unreachable", async () => {
    vi.useFakeTimers();

    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const health = new EmbedderHealth();

    await health.refresh(async () => ({ status: "unreachable", error: "down" }), 1000);
    await vi.advanceTimersByTimeAsync(5000);
    health.stop();

    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("unrefs its timer so it never keeps the process alive, and stop cancels it", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const run = vi.fn(async (): Promise<EmbedderCheck> => ({ status: "unreachable", error: "down" }));
    const health = new EmbedderHealth();

    await health.refresh(run, 1000);
    expect((health as unknown as { timer: { hasRef(): boolean } }).timer.hasRef()).toBe(false);

    health.stop();
    await vi.advanceTimersByTimeAsync(10_000);

    expect(run).toHaveBeenCalledTimes(1);
  });

  it("does not schedule a retry when stop lands while the probe is in flight", async () => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);

    let finish: (check: EmbedderCheck) => void = () => undefined;
    const run = vi.fn(
      () =>
        new Promise<EmbedderCheck>((resolve) => {
          finish = resolve;
        }),
    );
    const health = new EmbedderHealth();
    const pending = health.refresh(run, 1000);

    health.stop();
    finish({ status: "unreachable", error: "down" });
    await pending;
    await vi.advanceTimersByTimeAsync(10_000);

    expect(run).toHaveBeenCalledTimes(1);
    expect(health.get().status).toBe("unchecked");
  });
});
