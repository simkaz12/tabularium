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

describe("mode and public defaults", () => {
  it("defaults to standalone on loopback with the XDG store", () => {
    const config = loadConfig({ XDG_DATA_HOME: "/data/xdg" });

    expect(config.mode).toBe("standalone");
    expect(config.host).toBe("127.0.0.1");
    expect(config.storePath).toBe("/data/xdg/tabumcp/store");
    expect(config.allowedHosts).toEqual(["127.0.0.1:8789", "localhost:8789"]);
    expect(config.tokenFile).toMatch(/\.config\/tabumcp\/token$/);
    expect(config.scopesFile).toMatch(/\.config\/tabumcp\/scopes\.json$/);
  });

  it("falls back to the home data folder without XDG_DATA_HOME", () => {
    expect(loadConfig({}).storePath).toMatch(/tabumcp[/\\]store$/);
  });

  it("turns fold asks on with a key in standalone and off in t3", () => {
    expect(loadConfig({ TYPESAFE_API_KEY: "k" }).foldAsk).toBe(true);
    expect(loadConfig({}).foldAsk).toBe(false);
    expect(loadConfig({ TYPESAFE_API_KEY: "k", TABUMCP_MODE: "t3" }).foldAsk).toBe(false);
  });

  it("lets an explicit fold ask setting win", () => {
    expect(loadConfig({ TYPESAFE_API_KEY: "k", TABULARIUM_FOLD_ASK: "off" }).foldAsk).toBe(false);
    expect(loadConfig({ TABULARIUM_FOLD_ASK: "on" }).foldAsk).toBe(true);
  });

  it("leaves token and scopes files null in t3 unless set", () => {
    const bare = loadConfig({ TABUMCP_MODE: "t3" });
    const set = loadConfig({ TABUMCP_MODE: "t3", TABUMCP_TOKEN_FILE: "/t/token", TABUMCP_SCOPES_FILE: "/t/s.json" });

    expect(bare.tokenFile).toBeNull();
    expect(bare.scopesFile).toBeNull();
    expect(set.tokenFile).toBe("/t/token");
    expect(set.scopesFile).toBe("/t/s.json");
  });

  it("defaults topics to the generic list in standalone and the owner list in t3", () => {
    expect(loadConfig({}).topicsPath).toMatch(/v2[/\\]topics\.default\.json$/);
    expect(loadConfig({ TABUMCP_MODE: "t3" }).topicsPath).toMatch(/v2[/\\]topics\.json$/);
  });

  it("lets the topics path variable win in both modes", () => {
    expect(loadConfig({ TABULARIUM_TOPICS_PATH: "/x/t.json" }).topicsPath).toBe("/x/t.json");
    expect(loadConfig({ TABUMCP_MODE: "t3", TABULARIUM_TOPICS_PATH: "/x/t.json" }).topicsPath).toBe("/x/t.json");
  });

  it("rejects an unknown mode", () => {
    expect(() => loadConfig({ TABUMCP_MODE: "bogus" })).toThrow("TABUMCP_MODE");
  });

  it("gives the service unit its old config plus mode t3", () => {
    const unit = readFileSync(new URL("../deploy/tabularium-v2.service", import.meta.url), "utf8");
    const env: NodeJS.ProcessEnv = {};

    for (const match of unit.matchAll(/^Environment=([A-Z0-9_]+)=(.*)$/gm)) {
      env[match[1]!] = match[2]!;
    }

    const config = loadConfig(env);

    expect(config.mode).toBe("t3");
    expect(config.host).toBe("127.0.0.1");
    expect(config.port).toBe(8789);
    expect(config.storePath).toBe("/var/lib/tabularium-v2/store");
    expect(config.allowedHosts).toEqual(["127.0.0.1:8789"]);
    expect(config.foldAsk).toBe(true);
    expect(config.containment).toBe(true);
    expect(config.tokenFile).toBeNull();
    expect(config.scopesFile).toBeNull();
    expect(config.topicsPath).toMatch(/topics\.json$/);
  });
});
