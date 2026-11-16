import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { HashEmbedder } from "../src/embed.ts";
import { choiceAnswer, stubJev } from "../src/testing.ts";
import { openEval } from "./engine.ts";
import type { EvalHarness } from "./engine.ts";
import { outputsIn, runPersona, scopeOf } from "./pmv2.ts";
import type { AskEndRow, HitRow, PmNote, PmQuestion, WriteRow } from "./pmv2.ts";
import { withProposalOptions } from "./save.ts";

const DIM = 64;
const PERSONA = "39";
const OTHER = "55";

let harnesses: EvalHarness[] = [];
let scratch: string[] = [];

afterEach(() => {
  for (const harness of harnesses) {
    rmSync(harness.dir, { recursive: true, force: true });
  }

  for (const dir of scratch) {
    rmSync(dir, { recursive: true, force: true });
  }

  harnesses = [];
  scratch = [];
});

function yesNo(probability: number) {
  return () => ({
    type: "choice",
    choice: probability >= 0.5 ? "yes" : "no",
    confidence: Math.max(probability, 1 - probability),
    probabilities: { yes: probability, no: 1 - probability },
  });
}

function relation(_question: unknown, request: { state: unknown }) {
  const state = request.state as { existing?: string; incoming?: unknown };
  const matches =
    state.existing?.includes("sourdough starter") === true &&
    JSON.stringify(state.incoming).includes("sourdough starter");

  return choiceAnswer(matches ? "follow_up" : "distinct", 0.9);
}

function note(index: number, personaId: string, headline: string, claim: string): PmNote {
  return { note_id: `${personaId}-n${index}`, persona_id: personaId, chunk_id: `${personaId}-c0`, headline, claim };
}

const NOTES: PmNote[] = [
  note(0, PERSONA, "User keeps a sourdough starter", "The user feeds a sourdough starter every morning before work."),
  note(1, PERSONA, "User keeps a sourdough starter", "The user feeds a sourdough starter every morning before work, in the fridge."),
  note(2, PERSONA, "User plays chess on weekends", "The user plays chess at a club on Saturday afternoons."),
  note(3, PERSONA, "User asked to forget the discrimination detail", "Please forget the detail about my experiencing subtle discrimination."),
  note(4, OTHER, "Other persona likes sailing", "The other persona sails a small dinghy on the lake."),
];

const QUESTIONS: PmQuestion[] = [
  { question_id: `${PERSONA}-1`, persona_id: PERSONA, query: "What does the user do with the sourdough starter?" },
  { question_id: `${PERSONA}-2`, persona_id: PERSONA, query: "Where does the user play chess?" },
];

async function harnessFor(): Promise<EvalHarness> {
  const embedder = new HashEmbedder(DIM);
  const jev = stubJev({ relation, new_covers_old: yesNo(0.2), same_subject: yesNo(0.9) });
  const opened = await openEval({ embedder, jev });

  harnesses.push(opened);

  return withProposalOptions(opened, { foldAsk: true });
}

function readRows<Row>(path: string): Row[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as Row);
}

describe("v2 pmv2 lane", () => {
  it("writes a row per note, stores an expired ask, and keeps hits in the persona scope", async () => {
    const harness = await harnessFor();
    const dir = mkdtempSync(join(tmpdir(), "pmv2-test-"));

    scratch.push(dir);

    const out = outputsIn(dir);
    const result = await runPersona(harness, PERSONA, NOTES.filter((n) => n.persona_id === PERSONA), QUESTIONS, out);
    const writes = readRows<WriteRow>(out.writes);

    expect(writes.map((row) => row.note_id)).toEqual(["39-n0", "39-n1", "39-n2", "39-n3"]);
    expect(writes[1]?.status).toBe("asked");
    expect(writes[1]?.record_id).toBeNull();

    const asks = readRows<AskEndRow>(out.asks);

    expect(asks).toHaveLength(1);
    expect(asks[0]?.note_id).toBe("39-n1");
    expect(asks[0]?.status).toBe("approved");
    expect(asks[0]?.ask_answer).toBe("unanswered");
    expect(asks[0]?.record_id).not.toBeNull();
    expect(result.asks).toHaveLength(1);

    const heldId = asks[0]?.record_id as string;
    const found = await harness.seed.recall.recall({
      q: "The user feeds a sourdough starter every morning before work, in the fridge.",
      scope: scopeOf(PERSONA),
      limit: 10,
    });

    expect(found.results.map((hit) => hit.id)).toContain(heldId);
  });

  it("keeps hits inside the persona scope for both arms", async () => {
    const harness = await harnessFor();
    const dir = mkdtempSync(join(tmpdir(), "pmv2-test-"));

    scratch.push(dir);

    const out = outputsIn(dir);

    await runPersona(harness, OTHER, NOTES.filter((n) => n.persona_id === OTHER), [], out);
    await runPersona(harness, PERSONA, NOTES.filter((n) => n.persona_id === PERSONA), QUESTIONS, out);

    for (const arm of ["v2_rerank", "v2_plain"] as const) {
      const hits = readRows<HitRow>(out.hits[arm]);

      expect(hits.length).toBeGreaterThan(0);
      expect(hits.every((hit) => hit.arm === arm)).toBe(true);
      expect(hits.some((hit) => hit.text.includes("sailing") || hit.text.includes("dinghy"))).toBe(false);
      expect(hits.filter((hit) => hit.question_id === QUESTIONS[0]?.question_id)[0]?.rank).toBe(1);
    }
  });
});
