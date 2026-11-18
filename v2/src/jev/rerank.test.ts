import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { makeHarness, seedLegacy, seedRecord, stubJev } from "../testing.ts";
import type { Harness } from "../testing.ts";
import type { JevRequest } from "../domain.ts";
import { createJevReranker } from "./rerank.ts";

let harness: Harness | null = null;

afterEach(() => {
  harness?.cleanup();
  harness = null;
});

const scoreAnswer = (score: number) => ({ type: "score", score, confidence: 0.9, legend: {}, probabilities: {} });

describe("jev rerank", () => {
  it("reorders by jev score", async () => {
    const stub = stubJev({
      rerank: (_q, req: JevRequest) =>
        scoreAnswer((req.state as any).record.headline.includes("best") ? 3 : 1),
    });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    const weak = await seedRecord(harness, { headline: "repl readline crashes", claim: "repl readline error", why: "a" });
    const best = await seedRecord(harness, { headline: "best repl answer", claim: "repl readline fix", why: "b" });

    const { results } = await harness.services.recall.recall({ q: "repl readline error" });

    expect(results.map((entry) => entry.id)).toEqual([best.id, weak.id]);
    expect(results[0]?.score).toBe(3);
    expect(stub.calls.length).toBe(2);
    expect(stub.calls[0]?.state).toMatchObject({ query: "repl readline error" });
  });

  it("sends why to the reranker only when the record has one", async () => {
    const stub = stubJev({ rerank: () => scoreAnswer(2) });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    await seedRecord(harness, { headline: "repl readline plain", claim: "repl readline error", why: null });
    await seedLegacy(harness, {
      headline: "repl readline archived",
      claim: "repl readline error",
      why: "from the archive",
    });

    await harness.services.recall.recall({ q: "repl readline error" });

    const records = stub.calls.map((call) => (call.state as any).record);
    const plain = records.find((record) => record.headline === "repl readline plain");
    const archived = records.find((record) => record.headline === "repl readline archived");
    expect(plain).not.toHaveProperty("why");
    expect(archived.why).toBe("from the archive");
  });

  it("keeps v2 first when scores tie", async () => {
    const stub = stubJev({ rerank: () => scoreAnswer(2) });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    const content = { headline: "repl readline", claim: "repl readline error", why: "same" };
    await seedLegacy(harness, content);
    const current = await seedRecord(harness, content);

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(results.map((entry) => entry.source)).toEqual(["v2", "legacy"]);
    expect(results[0]?.id).toBe(current.id);
  });

  it("lets a higher scored legacy record beat v2", async () => {
    const stub = stubJev({
      rerank: (_q, req: JevRequest) => scoreAnswer((req.state as any).record.headline.startsWith("legacy") ? 3 : 0),
    });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    await seedRecord(harness, { headline: "repl readline", claim: "repl readline error", why: "v2" });
    const legacy = await seedLegacy(harness, { headline: "legacy repl readline", claim: "repl readline", why: "old" });

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(results[0]?.id).toBe(legacy.id);
  });

  it("falls back to RRF order when jev fails", async () => {
    const stub = stubJev();
    stub.failWith = new Error("down");
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });
    const content = { headline: "repl readline", claim: "repl readline error", why: "same" };
    await seedLegacy(harness, content);
    const current = await seedRecord(harness, content);

    const { results } = await harness.services.recall.recall({ q: "repl readline" });

    expect(results.map((entry) => entry.source)).toEqual(["v2", "legacy"]);
    expect(results[0]?.id).toBe(current.id);
    expect(await harness.store.readLog.countRows()).toBe(2);
  });

  it("reranks only the top 20 candidates", async () => {
    const stub = stubJev({ rerank: () => scoreAnswer(1) });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });

    for (let index = 0; index < 25; index += 1) {
      await seedRecord(harness, { headline: `repl readline ${index}`, claim: "repl readline error", why: "w" });
    }

    const { results } = await harness.services.recall.recall({ q: "repl readline", limit: 25 });

    expect(results.length).toBe(20);
    expect(stub.calls.length).toBe(20);
  });

  it("keeps scored order among scored slots when one call fails", async () => {
    let mode: "fail" | "partial" = "fail";
    const stub = stubJev({
      rerank: (_q, req: JevRequest) => {
        const headline = (req.state as any).record.headline as string;

        if (mode === "fail" || headline.includes("unscored")) {
          throw new Error("rerank call failed");
        }

        return scoreAnswer(Number(headline.match(/rank(\d)/)?.[1]));
      },
    });
    harness = await makeHarness({ proposals: { jev: stub, model: "jev-test" } });

    for (const headline of ["repl readline rank1", "repl readline rank2", "repl readline unscored", "repl readline rank3"]) {
      await seedRecord(harness, { headline, claim: "repl readline error", why: headline });
    }

    const hybrid = (await harness.services.recall.recall({ q: "repl readline error" })).results.map((entry) => entry.id);
    mode = "partial";
    const partial = (await harness.services.recall.recall({ q: "repl readline error" })).results;

    const byId = new Map(partial.map((entry) => [entry.id, entry]));
    const unscoredId = partial.find((entry) => entry.headline.includes("unscored"))?.id as string;
    const scoredSlots = hybrid.map((id, index) => (id === unscoredId ? -1 : index)).filter((index) => index >= 0);
    const scoredIds = scoredSlots.map((index) => partial[index]?.id as string);
    const scores = scoredIds.map((id) => byId.get(id)?.score as number);

    expect(partial.length).toBe(4);
    expect(hybrid.indexOf(unscoredId)).toBe(partial.findIndex((entry) => entry.id === unscoredId));
    expect(scores).toEqual([...scores].sort((a, b) => b - a));
    expect(scores).toEqual([3, 2, 1]);
  });

  it("returns null so recall keeps hybrid order when every rerank call fails", async () => {
    const stub = stubJev();
    stub.failWith = new Error("down");
    const errors: unknown[] = [];
    const rerank = createJevReranker({ client: stub, model: "jev-test", onError: (error) => errors.push(error) });
    const candidates = ["a", "b", "c"].map((id) => ({ id, source: "v2", kind: "fact", fields: {} }) as any);

    expect(await rerank("q", candidates)).toBeNull();
    expect(errors.length).toBe(3);
  });
});
