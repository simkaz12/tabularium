import { afterEach, describe, expect, it } from "vitest";

import { checkEmbedderBuild, readEmbedderStamp } from "../embedderBuild.ts";
import type { EmbedderBuild } from "../embedderBuild.ts";
import { HashEmbedder } from "../embed.ts";
import { selectWhere } from "../store.ts";
import { makeHarness, seedLegacy, seedRecord, TEST_DIM } from "../testing.ts";
import type { Harness } from "../testing.ts";
import { dryRun, parseCliArgs, reembed, reembedAndStamp } from "./reembed.ts";

const harnesses: Harness[] = [];

afterEach(() => {
  for (const harness of harnesses.splice(0)) {
    harness.cleanup();
  }
});

async function harness(): Promise<Harness> {
  const made = await makeHarness();
  harnesses.push(made);

  return made;
}

class ShiftedEmbedder extends HashEmbedder {
  override async embed(text: string): Promise<number[]> {
    return super.embed(`${text} shifted`);
  }
}

const BUILD: EmbedderBuild = { model: "nomic-embed-text", digest: "sha256:abc", ollamaVersion: "0.40.12" };

function ollamaFetch(build: EmbedderBuild): typeof fetch {
  return (async (url: string | URL | Request) => {
    const path = new URL(String(url)).pathname;
    const body =
      path === "/api/version"
        ? { version: build.ollamaVersion }
        : { models: [{ name: `${build.model}:latest`, digest: build.digest }] };

    return new Response(JSON.stringify(body), { status: 200 });
  }) as unknown as typeof fetch;
}

async function seedBoth(h: Harness) {
  await seedRecord(h, { id: "r-live", lifecycle: "live" });
  await seedRecord(h, { id: "r-retired", lifecycle: "retired", retiredReason: "stale" });
  await seedRecord(h, { id: "r-sup", lifecycle: "superseded", supersededBy: "r-live", reinforcedCount: 3 });
  await seedRecord(h, { id: "r-defect", kind: "defect", where: "app/X.php", trigger: "run it", defectStatus: "open" });
  await seedLegacy(h, { id: "l-live", lifecycle: "live", readCount: 4 });
  await seedLegacy(h, { id: "l-sup", lifecycle: "superseded", supersededBy: "r-live", reviewed: true });
}

describe("reembed", () => {
  it("rewrites every vector in both tables and every lifecycle, leaving other fields equal", async () => {
    const h = await harness();
    await seedBoth(h);
    const before = {
      records: await selectWhere<Record<string, unknown>>(h.store.records, undefined),
      legacy: await selectWhere<Record<string, unknown>>(h.store.legacy, undefined),
    };
    const progress: Array<Record<string, unknown>> = [];

    const report = await reembed(h.store, new ShiftedEmbedder(TEST_DIM), {
      batchSize: 2,
      onProgress: (event) => progress.push(event),
    });

    expect(report).toMatchObject({ records: 4, legacy: 2, vectorIndexRebuilt: [] });
    expect(progress.filter((event) => event.table === "records")).toHaveLength(2);

    for (const name of ["records", "legacy"] as const) {
      const after = await selectWhere<Record<string, unknown>>(h.store[name], undefined);

      expect(after).toHaveLength(before[name].length);

      for (const row of before[name]) {
        const next = after.find((candidate) => candidate.id === row.id) as Record<string, unknown>;
        const { vector: oldVector, ...oldRest } = row;
        const { vector: newVector, ...newRest } = next;

        expect(newRest).toEqual(oldRest);
        expect(newVector).not.toEqual(oldVector);
        expect(newVector).toHaveLength(TEST_DIM);
      }
    }
  });

  it("is idempotent: a second run leaves the vectors as the first wrote them", async () => {
    const h = await harness();
    await seedBoth(h);
    const embedder = new ShiftedEmbedder(TEST_DIM);

    await reembed(h.store, embedder);
    const first = await selectWhere<{ id: string; vector: number[] }>(h.store.records, undefined);
    await reembed(h.store, embedder);
    const second = await selectWhere<{ id: string; vector: number[] }>(h.store.records, undefined);

    expect(second.map((row) => [row.id, row.vector]).sort()).toEqual(first.map((row) => [row.id, row.vector]).sort());
  });

  it("writes the stamp after a re-embed", async () => {
    const h = await harness();
    await seedBoth(h);

    await reembedAndStamp(h.store, new ShiftedEmbedder(TEST_DIM), async () => BUILD, {
      now: () => "2026-11-18T10:00:00.000Z",
    });

    expect(await readEmbedderStamp(h.store)).toEqual({ ...BUILD, stampedAt: "2026-11-18T10:00:00.000Z" });
  });

  it("writes no stamp when the Ollama build changes mid-run", async () => {
    const h = await harness();
    await seedBoth(h);
    const builds = [BUILD, { ...BUILD, ollamaVersion: "0.41.0" }];

    await expect(
      reembedAndStamp(h.store, new ShiftedEmbedder(TEST_DIM), async () => builds.shift() as EmbedderBuild),
    ).rejects.toThrow(/changed during the re-embed/);
    expect(await readEmbedderStamp(h.store)).toBeNull();
  });

  it("dry run counts rows and times a sample without writing", async () => {
    const h = await harness();
    await seedBoth(h);
    const before = await selectWhere<{ id: string; vector: number[] }>(h.store.records, undefined);

    const report = await dryRun(h.store, new ShiftedEmbedder(TEST_DIM));

    expect(report).toMatchObject({ records: 4, legacy: 2, sampled: 6 });
    expect(await selectWhere(h.store.records, undefined)).toEqual(before);
    expect(await readEmbedderStamp(h.store)).toBeNull();
  });

  it("parses cli arguments", () => {
    expect(parseCliArgs(["node", "x"])).toEqual({ dryRun: false, batchSize: 50 });
    expect(parseCliArgs(["node", "x", "--dry-run", "--batch", "10"])).toEqual({ dryRun: true, batchSize: 10 });
    expect(() => parseCliArgs(["node", "x", "--bogus"])).toThrow(/usage/);
    expect(() => parseCliArgs(["node", "x", "--batch", "0"])).toThrow(/usage/);
  });
});

describe("startup embedder check", () => {
  const URL = "http://ollama.test/api/embed";

  it("flags a missing stamp", async () => {
    const h = await harness();

    const check = await checkEmbedderBuild(h.store, URL, BUILD.model, ollamaFetch(BUILD));

    expect(check).toEqual({ status: "missing", current: BUILD });
  });

  it("reports a match when the stamp equals the running build", async () => {
    const h = await harness();
    await reembedAndStamp(h.store, h.embedder, async () => BUILD, { now: () => "2026-11-18T10:00:00.000Z" });

    const check = await checkEmbedderBuild(h.store, URL, BUILD.model, ollamaFetch(BUILD));

    expect(check.status).toBe("match");
  });

  it.each([
    ["version", { ollamaVersion: "0.41.0" }, "ollamaVersion"],
    ["digest", { digest: "sha256:def" }, "digest"],
  ])("flags a mismatch on %s", async (_name, change, field) => {
    const h = await harness();
    await reembedAndStamp(h.store, h.embedder, async () => BUILD);

    const check = await checkEmbedderBuild(h.store, URL, BUILD.model, ollamaFetch({ ...BUILD, ...change }));

    expect(check).toMatchObject({ status: "mismatch", differs: [field] });
  });

  it("reports unreachable instead of throwing when Ollama is down", async () => {
    const h = await harness();
    const down = (async () => {
      throw new Error("connect refused");
    }) as unknown as typeof fetch;

    expect(await checkEmbedderBuild(h.store, URL, BUILD.model, down)).toEqual({
      status: "unreachable",
      error: "connect refused",
    });
  });

  it("keeps the check result on the service health holder", async () => {
    const h = await harness();
    await h.services.embedderHealth.refresh(() => checkEmbedderBuild(h.store, URL, BUILD.model, ollamaFetch(BUILD)));

    expect(h.services.embedderHealth.get().status).toBe("missing");
  });
});
