import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { loadConfig } from "./config.ts";

describe("containment floor config", () => {
  it("defaults to 0.95", () => {
    expect(loadConfig({}).containmentFloor).toBe(0.95);
  });

  it("reads the environment override", () => {
    expect(loadConfig({ TABULARIUM_CONTAINMENT_FLOOR: "0.97" }).containmentFloor).toBe(0.97);
  });

  it("matches the value the service unit sets", () => {
    const unit = readFileSync(new URL("../deploy/tabularium-v2.service", import.meta.url), "utf8");

    expect(unit).toContain("Environment=TABULARIUM_CONTAINMENT_FLOOR=0.95");
  });
});

describe("network config", () => {
  it("binds to loopback by default", () => {
    expect(loadConfig({}).host).toBe("127.0.0.1");
  });

  it("reads the host override", () => {
    expect(loadConfig({ TABULARIUM_HOST: "192.0.2.10" }).host).toBe("192.0.2.10");
  });
});
