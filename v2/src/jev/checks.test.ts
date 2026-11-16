import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { askContradiction, askRelevance, suggestionFrom } from "./checks.ts";
import { createJevClient } from "../main.ts";
import { loadConfig } from "../config.ts";
import { choiceAnswer, defectBody, makeHarness, proposalBody, seedLegacy, seedRecord, stubJev } from "../testing.ts";
import type { Harness } from "../testing.ts";

let harness: Harness | null = null;

afterEach(() => {
  harness?.cleanup();
  harness = null;
});

async function setup(stub: ReturnType<typeof stubJev> | null, topicsPath: string | null = "/nonexistent/topics.json") {
  harness = await makeHarness({ proposals: { jev: stub, model: "jev-test", topicsPath } });

  return harness;
}

const decisionBody = () =>
  proposalBody({
    kind: "decision",
    fields: { claim: "use postgres", rejected: "mongo" },
  });

describe("propose checks", () => {
  it("sends the per-record questions in one request and stores raw answers", async () => {
    const stub = stubJev();
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(stub.calls.length).toBe(1);
    expect(Object.keys(stub.calls[0]?.questions ?? {}).sort()).toEqual(
      [
        "audience",
        "breadth",
        "claim_standing",
        "excerpt_grounds",
        "headline_faithful",
        "kind",
        "perishable",
        "purpose",
        "reader",
        "scope",
        "worth",
      ].sort(),
    );
    expect(stub.calls[0]?.model).toBe("jev-test");
    expect(proposal.checks.status).toBe("ok");
    expect(proposal.checks.answers).toHaveProperty("kind");
    expect(Object.keys(proposal.checks.wordingHashes ?? {})).toContain("worth");
    expect(proposal.flags).toEqual([]);
    expect(proposal.purpose).toBe("pitfall");
    expect(proposal.topic).toBeNull();
  });

  it("asks the topic question over the topics file and stores the pick", async () => {
    const dir = mkdtempSync(join(tmpdir(), "topics-"));
    const path = join(dir, "topics.json");
    writeFileSync(path, JSON.stringify([{ name: "test harness", description: "tests" }, "other"]));
    const stub = stubJev({ topic: () => choiceAnswer("test harness") });
    const h = await setup(stub, path);

    const proposal = await h.services.proposals.create(proposalBody());
    rmSync(dir, { recursive: true, force: true });

    const topicQuestion = stub.calls[0]?.questions.topic as { criteria: Record<string, unknown> };
    expect(Object.keys(topicQuestion.criteria)).toEqual(["test harness", "other", "unclear"]);
    expect(proposal.topic).toBe("test harness");
  });

  it("flags a decision whose rejected alternative is not real", async () => {
    const stub = stubJev({ rejected_names_alternative: () => ({ type: "noul", noul: 0.2 }) });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(decisionBody());

    expect(proposal.flags).toContain("shape_low:rejected_names_alternative");
  });

  it("flags a decision with no rejected field without asking about it", async () => {
    const stub = stubJev();
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(
      proposalBody({ kind: "decision", fields: { claim: "use postgres" } }),
    );

    expect(Object.keys(stub.calls[0]?.questions ?? {})).not.toContain("rejected_names_alternative");
    expect(proposal.flags).toContain("rejected_missing");
  });

  it("flags kind, worth, perishable, shape and low confidence", async () => {
    const stub = stubJev({
      kind: () => choiceAnswer("neither", 0.95),
      worth: () => choiceAnswer("derivable", 0.95),
      perishable: () => choiceAnswer("status", 0.4),
      headline_faithful: () => ({ type: "noul", noul: 0.3 }),
      claim_standing: () => ({ type: "noul", noul: 0.49 }),
    });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toEqual(
      expect.arrayContaining([
        "kind_differs",
        "worth_not_keep",
        "perishable_not_standing",
        "shape_low:headline_faithful",
        "shape_low:claim_standing",
        "low_confidence:perishable",
      ]),
    );
    expect(proposal.flags).not.toContain("low_confidence:kind");
  });

  it("does not flag exactly 0.5 confidence or 0.5 noul", async () => {
    const stub = stubJev({
      kind: (_q, req) => choiceAnswer((req.state as any).declared_kind, 0.5),
      headline_faithful: () => ({ type: "noul", noul: 0.5 }),
    });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toEqual([]);
  });

  it("does not ask perishable on a defect", async () => {
    const stub = stubJev();
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(defectBody());

    const asked = Object.keys(stub.calls[0]?.questions ?? {});
    expect(asked).not.toContain("perishable");
    expect(asked).toContain("reader");
    expect((stub.calls[0]?.state as any).where).toBe("app/Jobs/SyncReport.php");
    expect(proposal.flags).toEqual([]);
  });

  it("copies reader and audience from the answers and lets a body audience win", async () => {
    const stub = stubJev({
      reader: () => choiceAnswer("trap"),
      audience: () => choiceAnswer("navigator"),
    });
    const h = await setup(stub);

    const inferred = await h.services.proposals.create(proposalBody());
    const declared = await h.services.proposals.create(proposalBody({ audience: "reviewer" }));

    expect(inferred).toMatchObject({ reader: "trap", audience: "navigator" });
    expect(declared.audience).toBe("reviewer");
  });

  it("flags a confident reader none as a demotion and names the reason", async () => {
    const stub = stubJev({ reader: () => choiceAnswer("none", 0.8) });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toContain("demote:reader=none");
  });

  it("flags demotions on neither, a non-keep worth and a non-standing perishable at the floor", async () => {
    const stub = stubJev({
      kind: () => choiceAnswer("neither", 0.5),
      worth: () => choiceAnswer("derivable", 0.9),
      perishable: () => choiceAnswer("sequencing", 0.9),
    });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toEqual(
      expect.arrayContaining(["demote:kind=neither", "demote:worth=derivable", "demote:perishable=sequencing"]),
    );
  });

  it("flags no demotion below the confidence floor", async () => {
    const stub = stubJev({ reader: () => choiceAnswer("none", 0.4), worth: () => choiceAnswer("derivable", 0.3) });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags.filter((flag) => flag.startsWith("demote:"))).toEqual([]);
    expect(proposal.flags).toEqual(expect.arrayContaining(["low_confidence:reader", "low_confidence:worth"]));
  });

  it("flags an unclear worth without demoting and without worth_not_keep", async () => {
    const stub = stubJev({ worth: () => choiceAnswer("unclear", 0.9) });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toContain("unclear:worth");
    expect(proposal.flags).not.toContain("worth_not_keep");
    expect(proposal.flags.filter((flag) => flag.startsWith("demote:"))).toEqual([]);
  });

  it("raises no low_confidence flag for purpose, breadth, scope, topic or audience", async () => {
    const stub = stubJev({
      purpose: () => choiceAnswer("pitfall", 0.3),
      breadth: () => choiceAnswer("project_specific", 0.2),
      audience: () => choiceAnswer("builder", 0.1),
    });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags.filter((flag) => flag.startsWith("low_confidence"))).toEqual([]);
    expect(proposal.purpose).toBe("pitfall");
  });

  it("does not flag kind_differs on an unclear kind and stores a null audience for unclear", async () => {
    const stub = stubJev({
      kind: () => choiceAnswer("unclear", 0.9),
      audience: () => choiceAnswer("unclear", 0.9),
      purpose: () => choiceAnswer("unclear", 0.9),
    });
    const h = await setup(stub);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).not.toContain("kind_differs");
    expect(proposal.flags).toEqual(
      expect.arrayContaining(["unclear:kind", "unclear:audience", "unclear:purpose"]),
    );
    expect(proposal.audience).toBeNull();
    expect(proposal.purpose).toBeNull();
  });

  it("gives an unclear relation no suggestion and flags it", async () => {
    const stub = stubJev({ relation: () => choiceAnswer("unclear", 0.9) });
    const h = await setup(stub);
    const near = await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline error" });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.suggestion).toBeNull();
    expect(proposal.flags).toContain(`unclear:relation:${near.id}`);
  });

  it("asks relation per neighbour in parallel and preselects supersede on a replacement", async () => {
    const stub = stubJev({
      relation: (_q, req) =>
        (req.state as any).existing.includes("old claim") ? choiceAnswer("replacement", 0.9) : choiceAnswer("distinct"),
      reason: () => choiceAnswer("correction"),
    });
    const h = await setup(stub);
    const old = await seedRecord(h, {
      headline: "repl readline crashes",
      claim: "old claim about repl readline",
      why: "x",
    });
    await seedRecord(h, { headline: "repl note", claim: "repl readline history", why: "y" });
    await seedLegacy(h, { headline: "repl readline legacy", claim: "repl readline error", why: "z" });

    const proposal = await h.services.proposals.create(proposalBody());

    const relationCalls = stub.calls.filter((call) => "relation" in call.questions);
    expect(relationCalls.length).toBe(proposal.neighbours.length);
    expect(proposal.neighbours.length).toBeGreaterThan(1);
    const replaced = proposal.neighbours.find((entry) => entry.id === old.id);
    expect(replaced?.relation).toMatchObject({ pick: "replacement", reasonClass: "correction" });
    expect(proposal.suggestion).toEqual({
      action: "supersede",
      recordId: old.id,
      source: "v2",
      reasonClass: "correction",
    });
    expect(proposal.flags).toContain(`replacement:${old.id}`);
  });

  it("flags a low-confidence reason only when the relation is a replacement", async () => {
    const stub = stubJev({
      relation: () => choiceAnswer("replacement", 0.9),
      reason: () => choiceAnswer("changed", 0.4),
    });
    const h = await setup(stub);
    const old = await seedRecord(h, {
      headline: "repl readline crashes",
      claim: "repl readline error",
      why: "x",
    });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags).toContain(`low_confidence:reason:${old.id}`);
    expect(proposal.flags).not.toContain(`low_confidence:relation:${old.id}`);
  });

  it("does not flag a low-confidence reason for a distinct relation", async () => {
    const stub = stubJev({
      relation: () => choiceAnswer("distinct", 0.9),
      reason: () => choiceAnswer("changed", 0.4),
    });
    const h = await setup(stub);
    await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline error", why: "x" });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.flags.filter((flag) => flag.startsWith("low_confidence:reason"))).toEqual([]);
  });

  it("preselects reinforce on a restatement and flags it", async () => {
    const stub = stubJev({ relation: () => choiceAnswer("restatement", 0.9) });
    const h = await setup(stub);
    const twin = await seedRecord(h, {
      headline: "repl readline crashes on the history file",
      claim: "repl aborts with a readline error when the history file is unwritable",
      why: "the test harness depends on repl for fixtures",
    });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.suggestion).toMatchObject({ action: "reinforce", recordId: twin.id });
    expect(proposal.flags).toContain(`restatement:${twin.id}`);
  });

  it("lands both pair probabilities on the neighbour and sends one pair request each", async () => {
    const stub = stubJev({
      new_covers_old: () => ({
        type: "choice",
        choice: "yes",
        confidence: 0.93,
        probabilities: { yes: 0.93, no: 0.07 },
      }),
      same_subject: () => ({
        type: "choice",
        choice: "no",
        confidence: 0.6,
        probabilities: { yes: 0.4, no: 0.6 },
      }),
    });
    const h = await setup(stub);
    const near = await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline error" });

    const proposal = await h.services.proposals.create(proposalBody());

    const pairCalls = stub.calls.filter((call) => "new_covers_old" in call.questions);
    expect(pairCalls.length).toBe(proposal.neighbours.length);
    expect(Object.keys(pairCalls[0]?.questions ?? {}).sort()).toEqual(["new_covers_old", "same_subject"]);
    expect(Object.keys(pairCalls[0]?.state as object).sort()).toEqual(["existing", "incoming"]);
    expect(proposal.neighbours.find((entry) => entry.id === near.id)?.pair).toEqual({
      newCoversOld: 0.93,
      sameSubject: 0.4,
    });
    expect(proposal.checks.pairs).toHaveProperty(near.id);
    expect(proposal.checks.wordingHashes).toHaveProperty("new_covers_old");
    expect(proposal.flags.filter((flag) => flag.startsWith("pair_unavailable"))).toEqual([]);
  });

  it("builds the pair request existing text from the stored row with its rejected part, capped", async () => {
    const stub = stubJev();
    const h = await setup(stub);
    const near = await seedRecord(h, {
      kind: "decision",
      headline: "repl readline crashes",
      claim: "repl readline error",
      rejected: "a custom history path",
    });
    const long = await seedRecord(h, {
      kind: "decision",
      headline: "repl readline crashes badly",
      claim: `repl readline error ${"x".repeat(2000)}`,
      rejected: "never reached",
    });

    await h.services.proposals.create(proposalBody());

    const pairStates = stub.calls
      .filter((call) => "new_covers_old" in call.questions)
      .map((call) => (call.state as { existing: string }).existing);
    const relationStates = stub.calls
      .filter((call) => "relation" in call.questions)
      .map((call) => (call.state as { existing: string }).existing);

    expect(pairStates).toContain("repl readline crashes\nrepl readline error\nRejected: a custom history path");
    expect(relationStates).toContain("repl readline crashes\nrepl readline error");
    expect(pairStates.every((text) => text.length <= 1500)).toBe(true);
    expect(pairStates.some((text) => text.startsWith("repl readline crashes badly\n") && text.length === 1500)).toBe(
      true,
    );
    expect(near.id).not.toBe(long.id);
  });

  it("flags a rejected pair request, leaves pair null and still saves", async () => {
    const stub = stubJev();
    const original = stub.systemOne.bind(stub);
    stub.systemOne = async (request, options) => {
      if ("new_covers_old" in request.questions) {
        throw new Error("pair down");
      }

      return original(request, options);
    };
    const h = await setup(stub);
    const near = await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline error" });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.flags).toContain(`pair_unavailable:${near.id}`);
    expect(proposal.neighbours.find((entry) => entry.id === near.id)?.pair).toBeNull();
    expect(proposal.checks.status).toBe("ok");
  });

  it("still writes the proposal when jev fails", async () => {
    const stub = stubJev();
    stub.failWith = new Error("jev down");
    const h = await setup(stub);
    await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline", why: "x" });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.status).toBe("approved");
    expect(proposal.checks.status).toBe("unavailable");
    expect(proposal.checks.reason).toContain("jev down");
    expect(proposal.flags).toEqual(["jev_unavailable"]);
    expect(proposal.neighbours.length).toBe(1);
    expect(proposal.neighbours[0]?.relation).toBeNull();
    expect((await h.services.proposals.list("approved", null)).length).toBe(1);
  });

  it("still writes the proposal when no client is configured", async () => {
    const h = await setup(null);

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.checks.status).toBe("unavailable");
    expect(proposal.checks.reason).toContain("absent");
  });

  it("keeps the proposal and marks one relation when only that request fails", async () => {
    const stub = stubJev();
    const original = stub.systemOne.bind(stub);
    stub.systemOne = async (request, options) => {
      if ("relation" in request.questions) {
        throw new Error("relation down");
      }

      return original(request, options);
    };
    const h = await setup(stub);
    const near = await seedRecord(h, { headline: "repl readline crashes", claim: "repl readline", why: "x" });

    const proposal = await h.services.proposals.create(proposalBody());

    expect(proposal.checks.status).toBe("ok");
    expect(proposal.flags).toContain(`relation_unavailable:${near.id}`);
  });

  it("skips checks for cleaner proposals", async () => {
    const stub = stubJev();
    const h = await setup(stub);
    const stale = await seedRecord(h, { purpose: "pitfall", topic: "tooling" });

    const proposal = await h.services.proposals.createInternal(
      proposalBody({
        origin: "cleaner",
        target: { recordId: stale.id, source: "v2", action: "retire", reason: "stale" },
      }),
    );

    expect(stub.calls.length).toBe(0);
    expect(proposal.checks.status).toBe("skipped");
    expect(proposal.topic).toBe("tooling");
  });
});

describe("cleaner questions", () => {
  it("maps a relevance score answer to the grade with the most probability", async () => {
    const stub = stubJev({
      relevance: () => ({
        type: "score",
        score: 0.4,
        confidence: 0.8,
        probabilities: { "0": 0.7, "1": 0.2, "2": 0.05, "3": 0.05 },
      }),
    });

    const verdict = await askRelevance(stub, "jev-test", "some text");

    expect(verdict.pick).toBe("shelve-clear");
    expect(verdict.confidence).toBeCloseTo((4 * 0.7 - 1) / 3);
  });

  it("collapses correction and changed to replace", async () => {
    const stub = stubJev({ contradiction: () => choiceAnswer("changed") });

    const verdict = await askContradiction(stub, "jev-test", "old", "new");

    expect(verdict).toMatchObject({ pick: "replace", reasonClass: "changed" });
  });
});

describe("suggestionFrom", () => {
  const neighbour = (id: string, pick: string, confidence: number) => ({
    id,
    source: "v2" as const,
    headline: id,
    similarity: 0.9,
    relation: { pick, confidence, probabilities: {}, reasonClass: pick === "replacement" ? ("changed" as const) : null },
  });

  it("picks the highest confidence restatement or replacement, not the first", () => {
    const suggestion = suggestionFrom([
      neighbour("low", "restatement", 0.6),
      neighbour("distinct", "distinct", 0.99),
      neighbour("high", "replacement", 0.9),
    ]);

    expect(suggestion).toMatchObject({ action: "supersede", recordId: "high", reasonClass: "changed" });
  });

  it("returns null with no restatement or replacement", () => {
    expect(suggestionFrom([neighbour("a", "distinct", 0.9)])).toBeNull();
  });
});

describe("jev deadline", () => {
  it("returns jev_unavailable within the timeout with one request and no retries", async () => {
    let requests = 0;
    const sockets = new Set<import("node:net").Socket>();
    const server = createServer(() => {
      requests += 1;
    });
    server.on("connection", (socket) => {
      sockets.add(socket);
      socket.on("close", () => sockets.delete(socket));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as import("node:net").AddressInfo;
    const previous = process.env.TYPESAFE_BASE_URL;
    process.env.TYPESAFE_BASE_URL = `http://127.0.0.1:${port}`;

    try {
      const client = createJevClient(loadConfig({ TYPESAFE_API_KEY: "test-key" }));
      const checked = await makeHarness({ proposals: { jev: client, model: "jev-test", timeoutMs: 300 } });
      harness = checked;
      const started = Date.now();

      const proposal = await checked.services.proposals.create(proposalBody());
      const elapsed = Date.now() - started;

      expect(proposal.flags).toContain("jev_unavailable");
      expect(elapsed).toBeLessThan(600);
      expect(requests).toBe(1);
    } finally {
      if (previous === undefined) {
        delete process.env.TYPESAFE_BASE_URL;
      } else {
        process.env.TYPESAFE_BASE_URL = previous;
      }

      for (const socket of sockets) {
        socket.destroy();
      }

      await new Promise<void>((resolve) => server.close(() => resolve()));
    }
  });
});
